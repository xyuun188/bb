"""Read-only audit of the canonical trade-outcome consumer chain."""

from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.remote_ssh import connect_remote_ssh, run_remote_text
from core.safe_output import safe_print
from db.session import get_read_session_ctx
from models.learning import ExpertMemory
from services.authoritative_trade_outcome import (
    AUTHORITATIVE_TRADE_OUTCOME_VERSION,
    load_authoritative_trade_outcomes,
)
from services.training_data_quality import annotate_training_payload

LINKAGE_RECOVERY_GAPS = {
    "missing_position_history_entry_orders",
    "missing_loaded_entry_order_facts",
    "missing_position_history_close_orders",
    "missing_loaded_close_order_facts",
    "missing_exact_entry_order_decision_link",
    "missing_exact_entry_order_decision_payload",
    "missing_local_position_strategy_lineage",
    "missing_planned_stop_loss_lineage",
    "missing_planned_take_profit_lineage",
}
EXCHANGE_SPEC_RECOVERY_GAPS = {
    "missing_contract_ct_val",
    "missing_contract_ct_mult",
    "missing_contract_lot_size",
    "missing_fill_or_open_contracts",
}


def _position_ids(outcome: dict[str, Any]) -> set[int]:
    result: set[int] = set()
    for value in outcome.get("position_ids") or [outcome.get("position_id")]:
        try:
            position_id = int(value or 0)
        except (TypeError, ValueError):
            continue
        if position_id > 0:
            result.add(position_id)
    return result


def _outcome_summary(outcome: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_type": outcome.get("event_type"),
        "outcome_id": outcome.get("outcome_id"),
        "outcome_version": outcome.get("outcome_version"),
        "outcome_fingerprint": outcome.get("outcome_fingerprint"),
        "lifecycle_key": outcome.get("lifecycle_key"),
        "position_ids": sorted(_position_ids(outcome)),
        "decision_id": outcome.get("decision_id"),
        "entry_order_ids": outcome.get("entry_order_ids"),
        "close_order_ids": outcome.get("close_order_ids"),
        "symbol": outcome.get("symbol"),
        "side": outcome.get("side"),
        "realized_pnl": outcome.get("realized_pnl"),
        "authoritative_pnl_ratio_pct": outcome.get("authoritative_pnl_ratio_pct"),
        "outcome_complete": outcome.get("outcome_complete"),
        "outcome_evidence_gaps": outcome.get("outcome_evidence_gaps"),
        "stop_loss_slippage_pct": outcome.get("stop_loss_slippage_pct"),
        "trigger_to_first_fill_ms": outcome.get("trigger_to_first_fill_ms"),
        "attribution": outcome.get("attribution"),
        "counterfactual_evidence_count": len(outcome.get("counterfactual_evidence") or []),
        "counterfactual_production_weight": outcome.get(
            "counterfactual_production_weight"
        ),
        "consumer_provenance": outcome.get("consumer_provenance"),
        "learning_summary": outcome.get("learning_summary"),
    }


def _gap_summary(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    gap_counts: Counter[str] = Counter()
    gap_set_counts: Counter[tuple[str, ...]] = Counter()
    recovery_counts: Counter[str] = Counter()
    incomplete_samples: list[dict[str, Any]] = []
    for outcome in outcomes:
        gaps = tuple(sorted({str(item) for item in outcome.get("outcome_evidence_gaps") or [] if item}))
        gap_counts.update(gaps)
        gap_set_counts[gaps] += 1
        if not gaps:
            recovery_counts["complete"] += 1
            continue
        gap_set = set(gaps)
        if gap_set <= LINKAGE_RECOVERY_GAPS:
            recovery_class = "linkage_only_candidate"
        elif gap_set <= LINKAGE_RECOVERY_GAPS | EXCHANGE_SPEC_RECOVERY_GAPS:
            recovery_class = "linkage_and_exchange_spec_candidate"
        else:
            recovery_class = "official_fact_or_settlement_gap"
        recovery_counts[recovery_class] += 1
        if len(incomplete_samples) < 12:
            incomplete_samples.append(
                {
                    "outcome_id": outcome.get("outcome_id"),
                    "lifecycle_key": outcome.get("lifecycle_key"),
                    "symbol": outcome.get("symbol"),
                    "side": outcome.get("side"),
                    "recovery_class": recovery_class,
                    "evidence_gaps": list(gaps),
                }
            )
    return {
        "gap_counts": dict(gap_counts.most_common()),
        "gap_set_counts": [
            {"evidence_gaps": list(gaps), "count": count}
            for gaps, count in gap_set_counts.most_common(30)
        ],
        "recovery_class_counts": dict(recovery_counts),
        "incomplete_samples": incomplete_samples,
    }


async def audit(
    *,
    mode: str,
    position_id: int | None = None,
    summary_only: bool = False,
) -> dict[str, Any]:
    outcomes = await load_authoritative_trade_outcomes(mode=mode)
    annotated = annotate_training_payload(
        shadow_samples=[],
        trade_samples=outcomes,
        sequence_samples=[],
        text_sentiment_samples=[],
    )
    manifest = annotated["authoritative_outcome_manifest"]
    selected = [
        outcome
        for outcome in outcomes
        if position_id is None or position_id in _position_ids(outcome)
    ]
    selected_ids = {str(outcome.get("outcome_id") or "") for outcome in selected}
    memory_rows: list[Any] = []
    if position_id is not None:
        async with get_read_session_ctx() as session:
            memory_rows = list(
                (
                    await session.execute(
                        select(ExpertMemory).where(
                            ExpertMemory.source_position_id == int(position_id)
                        )
                    )
                ).scalars().all()
            )
    memory_bindings = [
        {
            "memory_id": int(row.id or 0),
            "expert_name": row.expert_name,
            "outcome_id": (row.extra or {}).get("outcome_id"),
            "outcome_version": (row.extra or {}).get("outcome_version"),
            "authority_level": (row.extra or {}).get("authority_level"),
            "production_evidence_eligible": (row.extra or {}).get(
                "production_evidence_eligible"
            ),
        }
        for row in memory_rows
    ]
    training_records = [
        row for row in manifest["records"] if str(row.get("outcome_id") or "") in selected_ids
    ]
    violations: list[str] = []
    lifecycle_keys = [str(item.get("lifecycle_key") or "") for item in outcomes]
    outcome_ids = [str(item.get("outcome_id") or "") for item in outcomes]
    if len(lifecycle_keys) != len(set(lifecycle_keys)):
        violations.append("duplicate_lifecycle_key")
    if len(outcome_ids) != len(set(outcome_ids)):
        violations.append("duplicate_outcome_id")
    if any(item.get("outcome_version") != AUTHORITATIVE_TRADE_OUTCOME_VERSION for item in outcomes):
        violations.append("mixed_outcome_version")
    if any(float(item.get("counterfactual_production_weight") or 0.0) != 0.0 for item in outcomes):
        violations.append("shadow_counterfactual_has_production_weight")
    if position_id is not None and not selected:
        violations.append("requested_position_outcome_missing")
    gap_summary = _gap_summary(outcomes)
    return {
        "status": "ok" if not violations else "blocked",
        "mode": mode,
        "contract_version": AUTHORITATIVE_TRADE_OUTCOME_VERSION,
        "outcome_count": len(outcomes),
        "complete_count": sum(item.get("outcome_complete") is True for item in outcomes),
        "incomplete_count": sum(item.get("outcome_complete") is not True for item in outcomes),
        "evidence_gap_summary": gap_summary,
        "violations": violations,
        "manifest": {
            key: value for key, value in manifest.items() if key != "records"
        },
        "selected_outcomes": (
            [] if summary_only else [_outcome_summary(item) for item in selected]
        ),
        "selected_training_records": [] if summary_only else training_records,
        "selected_expert_memory_bindings": [] if summary_only else memory_bindings,
        "summary_only": bool(summary_only),
    }


def _online_report(
    *,
    mode: str,
    position_id: int | None,
    summary_only: bool,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    remote_args = [
        ".venv/bin/python",
        "scripts/audit_authoritative_trade_outcomes.py",
        "--mode",
        mode,
    ]
    if position_id is not None:
        remote_args.extend(("--position-id", str(position_id)))
    if summary_only:
        remote_args.append("--summary-only")
    app_script = "\n".join(
        (
            "cd /data/bb/app",
            "export DATABASE_URL='postgresql+asyncpg://bb@/bb_trading?host=/var/run/postgresql'",
            "exec " + " ".join(shlex.quote(value) for value in remote_args),
        )
    )
    ssh = connect_remote_ssh(root, timeout=20)
    try:
        output = run_remote_text(
            ssh,
            "runuser -u bb -- /bin/bash -lc " + shlex.quote(app_script),
            timeout=180,
            check=False,
        )
    finally:
        ssh.close()
    safe_print(output)
    try:
        report = json.loads(output)
    except json.JSONDecodeError as exc:
        raise SystemExit("online outcome audit did not return JSON") from exc
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("paper", "live"), default="paper")
    parser.add_argument("--position-id", type=int)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if args.online:
        if _online_report(
            mode=args.mode,
            position_id=args.position_id,
            summary_only=bool(args.summary_only),
        ).get("status") != "ok":
            raise SystemExit(1)
        return
    report = asyncio.run(
        audit(
            mode=args.mode,
            position_id=args.position_id,
            summary_only=bool(args.summary_only),
        )
    )
    safe_print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
