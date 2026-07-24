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
from models.trade import Order
from services.authoritative_trade_outcome import (
    AUTHORITATIVE_TRADE_OUTCOME_VERSION,
    load_authoritative_trade_outcomes,
)
from services.okx_execution_slippage import OKX_FILL_MARK_SLIPPAGE_VERSION
from services.profit_training_contract import PROFIT_TRAINING_TARGET
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
        "execution_mode": outcome.get("execution_mode"),
        "strategy_entry_kind": outcome.get("strategy_entry_kind"),
        "strategy_training_role": outcome.get("strategy_training_role"),
        "strategy_selection_reason": outcome.get("strategy_selection_reason"),
        "paper_training_evidence": outcome.get("paper_training_evidence"),
        "paper_exploration_evidence": outcome.get("paper_exploration_evidence"),
        "entry_order_ids": outcome.get("entry_order_ids"),
        "close_order_ids": outcome.get("close_order_ids"),
        "symbol": outcome.get("symbol"),
        "side": outcome.get("side"),
        "realized_pnl": outcome.get("realized_pnl"),
        PROFIT_TRAINING_TARGET: outcome.get(PROFIT_TRAINING_TARGET),
        "outcome_complete": outcome.get("outcome_complete"),
        "outcome_evidence_gaps": outcome.get("outcome_evidence_gaps"),
        "slippage": outcome.get("slippage"),
        "trigger_to_first_fill_ms": outcome.get("trigger_to_first_fill_ms"),
        "attribution": outcome.get("attribution"),
        "counterfactual_evidence_count": len(outcome.get("counterfactual_evidence") or []),
        "counterfactual_production_weight": outcome.get(
            "counterfactual_production_weight"
        ),
        "consumer_provenance": outcome.get("consumer_provenance"),
        "learning_summary": outcome.get("learning_summary"),
    }


def _strategy_entry_kind_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(
        str(row.get("strategy_entry_kind") or "unclassified").strip()
        or "unclassified"
        for row in rows
    )
    return dict(counts.most_common())


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


def _compact_gap_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "gap_counts": dict(summary.get("gap_counts") or {}),
        "recovery_class_counts": dict(
            summary.get("recovery_class_counts") or {}
        ),
    }


def _slippage_integrity_summary(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    incomplete = [
        outcome
        for outcome in outcomes
        if "missing_authoritative_slippage"
        in set(outcome.get("outcome_evidence_gaps") or [])
    ]
    failed_orders: set[tuple[str, str]] = set()
    failed_order_reasons: set[tuple[str, str, str]] = set()
    reason_counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    sampled_reason_keys: set[str] = set()
    for outcome in incomplete:
        failures = outcome.get("execution_slippage_failures") or {}
        normalized_failures: dict[str, dict[str, list[str]]] = {}
        for side in ("entry", "close"):
            side_failures = failures.get(side) or {}
            normalized_side: dict[str, list[str]] = {}
            for order_id, values in side_failures.items():
                reasons = [str(value) for value in values or [] if str(value)]
                normalized_side[str(order_id)] = reasons
                failed_orders.add((side, str(order_id)))
                for reason in reasons:
                    key = (side, str(order_id), reason)
                    if key in failed_order_reasons:
                        continue
                    failed_order_reasons.add(key)
                    reason_counts[f"{side}:{reason}"] += 1
            normalized_failures[side] = normalized_side
        sample_reason_keys = {
            f"{side}:{reason}"
            for side, side_failures in normalized_failures.items()
            for reasons in side_failures.values()
            for reason in reasons
        }
        if not sample_reason_keys:
            sample_reason_keys = {"position_history_order_ids_missing"}
        if len(samples) < 12 and not sample_reason_keys <= sampled_reason_keys:
            sampled_reason_keys.update(sample_reason_keys)
            samples.append(
                {
                    "outcome_id": outcome.get("outcome_id"),
                    "position_ids": sorted(_position_ids(outcome)),
                    "entry_order_ids": list(outcome.get("entry_order_ids") or []),
                    "close_order_ids": list(outcome.get("close_order_ids") or []),
                    "entry_complete": (
                        outcome.get("entry_execution_slippage_complete") is True
                    ),
                    "close_complete": (
                        outcome.get("close_execution_slippage_complete") is True
                    ),
                    "failures": normalized_failures,
                }
            )
    return {
        "missing_outcome_count": len(incomplete),
        "entry_incomplete_outcome_count": sum(
            outcome.get("entry_execution_slippage_complete") is not True
            for outcome in incomplete
        ),
        "close_incomplete_outcome_count": sum(
            outcome.get("close_execution_slippage_complete") is not True
            for outcome in incomplete
        ),
        "unique_failed_order_count": len(failed_orders),
        "failed_order_reason_counts": dict(reason_counts.most_common()),
        "samples": samples,
    }


def _slippage_storage_summary(orders: list[Any]) -> dict[str, Any]:
    invalid_rows: list[dict[str, Any]] = []
    classification_counts: Counter[str] = Counter()
    for order in orders:
        raw = getattr(order, "okx_raw_fills", None)
        raw = raw if isinstance(raw, dict) else {}
        slippage = raw.get("execution_slippage")
        slippage = slippage if isinstance(slippage, dict) else {}
        if not slippage or slippage.get("version") == OKX_FILL_MARK_SLIPPAGE_VERSION:
            continue
        if raw.get("fills_history_confirmed") is True:
            origin = "fills_history"
        elif raw.get("order_detail_confirmed") is True:
            origin = "order_detail"
        elif raw.get("execution_result_confirmed") is True:
            origin = "execution_result"
        else:
            origin = "unconfirmed"
        rows = raw.get("rows")
        rows_available = isinstance(rows, list) and bool(rows)
        public_contract_size = bool(
            raw.get("contract_size_verified") is True
            and str(raw.get("contract_size_source") or "").strip()
            == "okx_public_instruments"
        )
        classification = ":".join(
            (
                origin,
                "rows_available" if rows_available else "rows_missing",
                "public_spec" if public_contract_size else "public_spec_missing",
            )
        )
        classification_counts[classification] += 1
        if len(invalid_rows) < 12:
            invalid_rows.append(
                {
                    "order_id": str(getattr(order, "exchange_order_id", "") or ""),
                    "origin": origin,
                    "rows_available": rows_available,
                    "row_count": len(rows) if isinstance(rows, list) else 0,
                    "public_contract_size": public_contract_size,
                    "stored_version": slippage.get("version"),
                    "stored_complete": slippage.get("complete"),
                }
            )
    return {
        "required_version": OKX_FILL_MARK_SLIPPAGE_VERSION,
        "invalid_version_order_count": sum(classification_counts.values()),
        "classification_counts": dict(classification_counts.most_common()),
        "samples": invalid_rows,
    }


async def audit(
    *,
    mode: str,
    position_id: int | None = None,
    summary_only: bool = False,
) -> dict[str, Any]:
    outcomes = await load_authoritative_trade_outcomes(mode=mode)
    async with get_read_session_ctx() as session:
        order_rows = list(
            (
                await session.execute(
                    select(Order).where(Order.execution_mode == mode)
                )
            ).scalars().all()
        )
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
    trainable = annotated["trade_samples"]
    if any(int(item.get("entry_decision_count") or 0) > 1 for item in trainable):
        violations.append("multiple_entry_decision_outcome_entered_training")
    if any(item.get("gross_return_price_consistent") is not True for item in trainable):
        violations.append("gross_return_price_mismatch_entered_training")
    if any(item.get("strategy_entry_supervision_eligible") is False for item in trainable):
        violations.append("research_only_outcome_entered_strategy_training")
    if position_id is not None and not selected:
        violations.append("requested_position_outcome_missing")
    gap_summary = _gap_summary(outcomes)
    slippage_integrity = _slippage_integrity_summary(outcomes)
    slippage_integrity["storage"] = _slippage_storage_summary(order_rows)
    if summary_only:
        gap_summary = _compact_gap_summary(gap_summary)
    return {
        "status": "ok" if not violations else "blocked",
        "mode": mode,
        "contract_version": AUTHORITATIVE_TRADE_OUTCOME_VERSION,
        "outcome_count": len(outcomes),
        "complete_count": sum(item.get("outcome_complete") is True for item in outcomes),
        "incomplete_count": sum(item.get("outcome_complete") is not True for item in outcomes),
        "training_integrity": {
            "trainable_count": len(trainable),
            "all_outcome_entry_kind_counts": _strategy_entry_kind_counts(outcomes),
            "trainable_entry_kind_counts": _strategy_entry_kind_counts(trainable),
            "loss_tolerant_paper_training_count": sum(
                item.get("strategy_entry_kind") == "loss_tolerant_paper_training"
                for item in trainable
            ),
            "bounded_paper_exploration_count": sum(
                item.get("strategy_entry_kind") == "bounded_risk_paper_exploration"
                for item in trainable
            ),
            "normal_strategy_trade_count": sum(
                item.get("strategy_entry_kind") == "normal_strategy_trade"
                for item in trainable
            ),
            "contract_notional_corrected_count": sum(
                item.get("contract_notional_corrected") is True for item in trainable
            ),
            "multiple_entry_decision_trainable_count": sum(
                int(item.get("entry_decision_count") or 0) > 1 for item in trainable
            ),
            "gross_return_price_mismatch_trainable_count": sum(
                item.get("gross_return_price_consistent") is not True
                for item in trainable
            ),
            "research_only_trainable_count": sum(
                item.get("strategy_entry_supervision_eligible") is False
                for item in trainable
            ),
        },
        "evidence_gap_summary": gap_summary,
        "slippage_integrity": slippage_integrity,
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
