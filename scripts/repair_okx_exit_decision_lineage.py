#!/usr/bin/env python3
"""Relink one OKX close fill to its original exit decision with audit evidence."""

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

from sqlalchemy import String, cast, select

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings
from core.remote_ssh import connect_remote_ssh, run_remote_text
from core.safe_output import safe_print
from db.session import get_read_session_ctx, get_session_ctx
from models.decision import AIDecision
from models.trade import Order, Position
from services.exchange_exit_decision_lineage import (
    apply_exit_decision_lineage,
    decision_exit_exchange_order_ids,
    load_exit_decision_lineage,
)

REMOTE_APP_DIR = "/data/bb/app"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--exchange-order-id", default="")
    target.add_argument("--decision-id", type=int)
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


def _decision_snapshot(decision: AIDecision) -> dict[str, Any]:
    return {
        "id": int(decision.id),
        "model_name": decision.model_name,
        "symbol": decision.symbol,
        "action": decision.action,
        "is_paper": bool(decision.is_paper),
        "was_executed": bool(decision.was_executed),
        "execution_reason": decision.execution_reason,
        "executed_at": _json_safe(decision.executed_at),
        "execution_price": decision.execution_price,
        "outcome": decision.outcome,
        "outcome_pnl_pct": decision.outcome_pnl_pct,
        "raw_llm_response": _json_safe(decision.raw_llm_response),
    }


def _fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _select_settlement_positions(
    candidates: list[Position],
    close_order_id: str,
) -> list[Position]:
    exact = [
        position
        for position in candidates
        if str(position.close_exchange_order_id or "").strip() == close_order_id
    ]
    if exact:
        return exact
    if len(candidates) == 1:
        return candidates
    if candidates:
        raise RuntimeError(
            "More than one closed position contains the OKX close order id and none is an "
            "exact lifecycle slice; refusing to aggregate duplicate settlement evidence"
        )
    return []


def _backup(payload: dict[str, Any]) -> Path:
    directory = settings.data_dir / "codex_backups" / "okx_exit_decision_lineage"
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / f"exit-lineage-before-{timestamp}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


async def _order_id_from_args(session: Any, args: argparse.Namespace) -> str:
    if args.exchange_order_id:
        return str(args.exchange_order_id).strip()
    decision = await session.get(AIDecision, int(args.decision_id))
    if decision is None:
        raise RuntimeError(f"Decision {args.decision_id} does not exist")
    order_ids = sorted(decision_exit_exchange_order_ids(decision))
    if len(order_ids) != 1:
        raise RuntimeError(
            f"Decision {args.decision_id} must expose exactly one exit exchange order id; "
            f"found {order_ids}"
        )
    return order_ids[0]


async def _load_plan(session: Any, args: argparse.Namespace) -> dict[str, Any]:
    close_order_id = await _order_id_from_args(session, args)
    order_result = await session.execute(
        select(Order).where(Order.exchange_order_id == close_order_id).limit(1)
    )
    order = order_result.scalar_one_or_none()
    if order is None:
        raise RuntimeError(f"No local order exists for OKX close order {close_order_id}")
    close_action = "close_short" if str(order.side or "").lower() == "buy" else "close_long"
    resolution = await load_exit_decision_lineage(
        session,
        model_name=order.model_name,
        symbol=order.symbol,
        action=close_action,
        is_paper=order.execution_mode != "live",
        execution_mode=order.execution_mode,
        close_order_id=close_order_id,
    )
    authoritative = resolution.authoritative
    matched = [
        await session.get(AIDecision, decision_id)
        for decision_id in resolution.matched_decision_ids
    ]
    decisions = [decision for decision in matched if decision is not None]
    position_result = await session.execute(
        select(Position).where(
            Position.execution_mode == order.execution_mode,
            Position.symbol == order.symbol,
            Position.is_open.is_(False),
            Position.close_exchange_order_id.is_not(None),
            cast(Position.close_exchange_order_id, String).contains(close_order_id),
        )
    )
    candidate_positions = [
        position
        for position in position_result.scalars().all()
        if close_order_id
        in {
            item.strip()
            for item in str(position.close_exchange_order_id or "").split(",")
            if item.strip()
        }
    ]
    positions = _select_settlement_positions(candidate_positions, close_order_id)
    state = {
        "close_order_id": close_order_id,
        "order_id": int(order.id),
        "linked_decision_id": order.decision_id,
        "authoritative_decision_id": int(authoritative.id) if authoritative else None,
        "matched_decision_ids": list(resolution.matched_decision_ids),
        "superseded_decision_ids": [int(row.id) for row in resolution.superseded],
        "decisions": [_decision_snapshot(row) for row in decisions],
        "positions": [
            {
                "id": int(position.id),
                "quantity": position.quantity,
                "entry_price": position.entry_price,
                "realized_pnl": position.realized_pnl,
                "closed_at": _json_safe(position.closed_at),
                "close_exchange_order_id": position.close_exchange_order_id,
            }
            for position in positions
        ],
    }
    return {
        "state": state,
        "fingerprint": _fingerprint(state),
        "repair_ready": bool(
            authoritative is not None
            and (
                int(order.decision_id or 0) != int(authoritative.id)
                or bool(resolution.superseded)
            )
        ),
        "order": order,
        "positions": positions,
        "resolution": resolution,
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    session_factory = get_session_ctx if args.apply else get_read_session_ctx
    async with session_factory() as session:
        plan = await _load_plan(session, args)
        output = {
            "apply": bool(args.apply),
            "repair_ready": plan["repair_ready"],
            "input_fingerprint": plan["fingerprint"],
            "before": plan["state"],
        }
        if not args.apply:
            return output
        if not args.expected_fingerprint:
            raise RuntimeError("--apply requires --expected-fingerprint from a fresh dry-run")
        if args.expected_fingerprint != plan["fingerprint"]:
            raise RuntimeError("Exit lineage changed after dry-run; refusing stale repair plan")
        if not plan["repair_ready"]:
            raise RuntimeError("Exit lineage does not have one repairable original decision")
        backup_path = _backup(plan["state"])
        positions = plan["positions"]
        order = plan["order"]
        realized_pnl = sum(float(position.realized_pnl or 0.0) for position in positions)
        entry_notional = sum(
            abs(float(position.entry_price or 0.0) * float(position.quantity or 0.0))
            for position in positions
        )
        closed_at = next(
            (
                position.closed_at
                for position in positions
                if isinstance(position.closed_at, datetime)
            ),
            order.filled_at or datetime.now(UTC),
        )
        close_fill = dict(order.okx_raw_fills or {})
        close_fill["order_id"] = plan["state"]["close_order_id"]
        result = apply_exit_decision_lineage(
            plan["resolution"],
            close_order_id=plan["state"]["close_order_id"],
            close_fill=_json_safe(close_fill),
            reconcile_origin="external_okx_sync",
            exit_price=float(order.price or 0.0),
            realized_pnl=realized_pnl,
            closed_at=closed_at,
            entry_notional=entry_notional,
        )
        await session.flush()
        output.update(
            {
                "backup_path": str(backup_path),
                "applied": result,
                "verified": bool(
                    result
                    and int(order.decision_id or 0)
                    == int(result["authoritative_decision_id"])
                    and all(not row.was_executed for row in plan["resolution"].superseded)
                ),
            }
        )
        return output


def _run_online(args: argparse.Namespace) -> dict[str, Any]:
    remote_args = [
        ".venv/bin/python",
        "scripts/repair_okx_exit_decision_lineage.py",
    ]
    if args.exchange_order_id:
        remote_args.extend(("--exchange-order-id", args.exchange_order_id))
    else:
        remote_args.extend(("--decision-id", str(args.decision_id)))
    if args.apply:
        remote_args.append("--apply")
        remote_args.extend(("--expected-fingerprint", args.expected_fingerprint))
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
            timeout=180,
            check=True,
        )
    finally:
        ssh.close()
    safe_print(output)
    return json.loads(output)


def main() -> None:
    args = _parser().parse_args()
    result = _run_online(args) if args.online else asyncio.run(_run(args))
    if not args.online:
        safe_print(json.dumps(result, ensure_ascii=False, default=str))
    if args.apply and result.get("verified") is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
