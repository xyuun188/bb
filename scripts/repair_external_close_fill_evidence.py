#!/usr/bin/env python3
"""Backfill missing OKX close-fill evidence into exact reconciliation decisions.

The repair is deliberately limited to decision JSON evidence.  It never
changes execution state, positions, settlement amounts, or orders.  Dry-run is
the default; ``--apply`` requires the fingerprint returned by that dry-run.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shlex
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings
from core.remote_ssh import connect_remote_ssh, run_remote_text
from core.safe_output import safe_print
from db.session import get_read_session_ctx, get_session_ctx
from models.decision import AIDecision
from models.trade import Order
from services.exchange_close_fill_evidence import normalize_external_close_fill_evidence

REMOTE_APP_DIR = "/data/bb/app"
IMMUTABLE_KEYS = (
    "order_id",
    "inst_id",
    "contracts",
    "contract_size",
    "base_quantity",
    "avg_price",
    "trade_ids",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decision-id",
        type=int,
        action="append",
        required=True,
        help="Exact external reconciliation decision id; repeat for multiple rows.",
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--expected-fingerprint", default="")
    parser.add_argument("--online", action="store_true")
    return parser


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(_json_safe(value), ensure_ascii=True, sort_keys=True, default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def _decision_mode(decision: AIDecision) -> str:
    return "paper" if bool(decision.is_paper) else "live"


def _close_fill_payload(decision: AIDecision) -> dict[str, Any]:
    raw = decision.raw_llm_response if isinstance(decision.raw_llm_response, dict) else {}
    close_fill = raw.get("close_fill")
    return close_fill if isinstance(close_fill, dict) else {}


def _order_fact(order: Order) -> dict[str, Any]:
    raw = order.okx_raw_fills if isinstance(order.okx_raw_fills, dict) else {}
    payload = normalize_external_close_fill_evidence(raw)
    payload["order_id"] = str(order.exchange_order_id or payload.get("order_id") or "").strip()
    payload["source"] = payload.get("source") or "okx_fills_history"
    return payload


def _validate_exact_match(
    existing: dict[str, Any],
    fact: dict[str, Any],
    *,
    decision_id: int,
) -> None:
    for key in IMMUTABLE_KEYS:
        old = existing.get(key)
        new = fact.get(key)
        if old in (None, "", [], {}):
            continue
        if key == "trade_ids":
            if sorted(str(value) for value in old) != sorted(str(value) for value in new or []):
                raise RuntimeError(
                    f"Decision {decision_id} close_fill.{key} conflicts with the exact order fact"
                )
            continue
        try:
            same = abs(float(old) - float(new)) <= max(abs(float(old)), abs(float(new)), 1.0) * 1e-9
        except (TypeError, ValueError):
            same = str(old).strip() == str(new).strip()
        if not same:
            raise RuntimeError(
                f"Decision {decision_id} close_fill.{key} conflicts with the exact order fact"
            )


async def _load_plan(session: Any, decision_ids: list[int]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for decision_id in sorted({int(value) for value in decision_ids}):
        decision = await session.get(AIDecision, decision_id)
        if decision is None:
            raise RuntimeError(f"Decision {decision_id} does not exist")
        raw = decision.raw_llm_response if isinstance(decision.raw_llm_response, dict) else {}
        close_fill = _close_fill_payload(decision)
        if raw.get("system_sync") is not True or raw.get("source") != "okx_position_reconcile":
            raise RuntimeError(f"Decision {decision_id} is not an OKX reconciliation decision")
        order_id = str(close_fill.get("order_id") or close_fill.get("ordId") or "").strip()
        if not order_id:
            raise RuntimeError(f"Decision {decision_id} has no exact OKX close order id")
        order_result = await session.execute(
            select(Order)
            .where(
                Order.execution_mode == _decision_mode(decision),
                Order.exchange_order_id == order_id,
            )
            .order_by(Order.id.desc())
            .limit(1)
        )
        order = order_result.scalar_one_or_none()
        if order is None:
            raise RuntimeError(f"No exact local order exists for decision {decision_id}: {order_id}")
        fact = _order_fact(order)
        if (
            not fact.get("order_id")
            or fact.get("contract_size_verified") is not True
            or fact.get("contract_size_source") != "okx_public_instruments"
            or not fact.get("contracts")
            or not fact.get("base_quantity")
            or not fact.get("avg_price")
            or not fact.get("trade_ids")
            or fact.get("fee_abs") is None
        ):
            raise RuntimeError(f"Order {order_id} does not contain a complete authoritative OKX fill fact")
        _validate_exact_match(close_fill, fact, decision_id=decision_id)
        merged = dict(close_fill)
        for key, value in fact.items():
            if key in IMMUTABLE_KEYS or key in {
                "contract_size_verified",
                "contract_size_source",
                "fee_abs",
                "fills_history_confirmed",
            }:
                if merged.get(key) in (None, "", [], {}):
                    merged[key] = value
        rows.append(
            {
                "decision_id": decision_id,
                "order_id": order_id,
                "order_id_db": int(order.id),
                "before_close_fill": _json_safe(close_fill),
                "after_close_fill": _json_safe(merged),
                "decision": decision,
            }
        )
    state = [
        {
            key: value
            for key, value in row.items()
            if key != "decision"
        }
        for row in rows
    ]
    return {"rows": rows, "state": state, "fingerprint": _fingerprint(state)}


def _backup(state: list[dict[str, Any]]) -> Path:
    directory = settings.data_dir / "codex_backups" / "external_close_fill_evidence"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (
        "before-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + ".json"
    )
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    session_factory = get_session_ctx if args.apply else get_read_session_ctx
    async with session_factory() as session:
        plan = await _load_plan(session, args.decision_id)
        output: dict[str, Any] = {
            "apply": bool(args.apply),
            "input_fingerprint": plan["fingerprint"],
            "rows": plan["state"],
        }
        if not args.apply:
            return output
        if args.expected_fingerprint != plan["fingerprint"]:
            raise RuntimeError("Evidence changed after dry-run; refusing stale repair plan")
        backup_path = _backup(plan["state"])
        changed: list[int] = []
        for row in plan["rows"]:
            decision = row["decision"]
            raw = dict(decision.raw_llm_response or {})
            close_fill = dict(raw.get("close_fill") or {})
            close_fill.update(row["after_close_fill"])
            raw["close_fill"] = close_fill
            raw["external_close_fill_evidence_backfill"] = {
                "source": "exact_local_okx_order_fact",
                "order_id": row["order_id"],
                "order_id_db": row["order_id_db"],
                "at": datetime.now(UTC).isoformat(),
            }
            decision.raw_llm_response = raw
            changed.append(int(decision.id))
        await session.flush()
        output.update(
            {
                "backup_path": str(backup_path),
                "applied_decision_ids": changed,
                "verified": changed == sorted({int(value) for value in args.decision_id}),
            }
        )
        return output


def _run_online(args: argparse.Namespace) -> dict[str, Any]:
    remote_args = [
        ".venv/bin/python",
        "scripts/repair_external_close_fill_evidence.py",
    ]
    for decision_id in sorted({int(value) for value in args.decision_id}):
        remote_args.extend(("--decision-id", str(decision_id)))
    if args.apply:
        remote_args.extend(("--apply", "--expected-fingerprint", args.expected_fingerprint))
    app_script = "\n".join(
        (
            f"cd {REMOTE_APP_DIR}",
            "export DATABASE_URL='postgresql+asyncpg://bb@/bb_trading?host=/var/run/postgresql'",
            "exec " + " ".join(shlex.quote(value) for value in remote_args),
        )
    )
    ssh = connect_remote_ssh(ROOT, timeout=20)
    try:
        output = run_remote_text(
            ssh,
            "runuser -u bb -- /bin/bash -lc " + shlex.quote(app_script),
            timeout=120,
            max_output_chars=60_000,
        )
    finally:
        ssh.close()
    return json.loads(output)


def main() -> None:
    args = _parser().parse_args()
    result = _run_online(args) if args.online else asyncio.run(_run(args))
    safe_print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if args.apply and result.get("verified") is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
