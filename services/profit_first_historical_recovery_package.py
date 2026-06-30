"""Read-only historical recovery package builder for Profit-First blockers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from services.profit_first_position_ladder import ProfitFirstPositionLadderPolicy
from services.profit_first_trade_plan import build_profit_first_trade_plan

RECOVERY_PACKAGE_VERSION = "profit-first-historical-recovery-package-v1"
LEGACY_EXIT_FAILURE_REASON = (
    "legacy_exit_missing_original_profit_first_plan_reference_before_profit_first_v3"
)


@dataclass(frozen=True, slots=True)
class HistoricalRecoveryInput:
    entry_decisions: list[Any]
    exit_decisions: list[Any]
    orders: list[Any]
    blocking_actions: list[dict[str, Any]]


def build_historical_recovery_package(
    payload: HistoricalRecoveryInput,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a dry-run package of proposed repair/quarantine actions."""

    generated_at = (now or datetime.now(UTC)).isoformat()
    entry_plans = [
        _entry_recovery_plan(row, generated_at=generated_at)
        for row in payload.entry_decisions
    ]
    exit_plans = [
        _exit_recovery_plan(row, generated_at=generated_at)
        for row in payload.exit_decisions
    ]
    okx_reviews = [
        _okx_order_review(order, generated_at=generated_at)
        for order in payload.orders
    ]
    ranking_reviews = _ranking_reviews(payload.blocking_actions, generated_at=generated_at)
    all_items = [*entry_plans, *exit_plans, *okx_reviews, *ranking_reviews]
    return {
        "report_type": "profit_first_historical_recovery_package",
        "version": RECOVERY_PACKAGE_VERSION,
        "generated_at": generated_at,
        "status": "ready" if all_items else "empty",
        "dry_run": True,
        "read_only": True,
        "audit_only": True,
        "mutates_database": False,
        "starts_trading_service": False,
        "submits_orders": False,
        "changes_model_routing": False,
        "changes_live_sizing": False,
        "live_mutation": False,
        "resume_allowed_by_this_package": False,
        "summary": {
            "item_count": len(all_items),
            "entry_decision_count": len(entry_plans),
            "exit_decision_count": len(exit_plans),
            "okx_order_review_count": len(okx_reviews),
            "ranking_review_count": len(ranking_reviews),
            "proposed_raw_patch_count": sum(
                1 for item in all_items if bool(item.get("proposed_raw_patch"))
            ),
            "operator_approval_required_count": sum(
                1 for item in all_items if bool(item.get("operator_approval_required"))
            ),
        },
        "items": all_items,
        "apply_policy": {
            "apply_supported_by_this_script": False,
            "requires_separate_operator_approved_apply_step": True,
            "requires_backup": True,
            "requires_explicit_decision_or_order_allowlist": True,
            "default_training_policy": "exclude_until_manual_trust",
            "do_not_use_as_resume_permission": True,
        },
        "validation_after_apply": [
            "Run scripts/verify_profit_first_online_readiness.py --json-indent 2.",
            "Require trade contract current-window Profit-First missing counts to be zero.",
            "Require recovery_repair_plan.blocking_actions to be empty before any resume.",
            "Keep online new market analysis and new entries paused until go/no-go clears.",
        ],
    }


def target_ids_from_blocking_actions(
    blocking_actions: list[dict[str, Any]],
) -> dict[str, list[Any]]:
    """Extract decision/order ids from recovery repair-plan blocking actions."""

    entry_decision_ids: list[int] = []
    exit_decision_ids: list[int] = []
    order_ids: list[int] = []
    exchange_order_ids: list[str] = []
    for action in blocking_actions:
        row = _safe_dict(action)
        target = _safe_dict(row.get("target"))
        code = str(row.get("code") or "")
        decision_ids = _unique_ints(_safe_list(target.get("decision_ids")))
        if code in {"missing_profit_first_trade_plan", "missing_profit_first_position_ladder"}:
            entry_decision_ids.extend(decision_ids)
        elif code == "missing_profit_first_exit_plan_reference":
            exit_decision_ids.extend(decision_ids)
        else:
            entry_decision_ids.extend([])
        order_ids.extend(_unique_ints(_safe_list(target.get("order_ids"))))
        exchange_order_ids.extend(
            str(item).strip()
            for item in _safe_list(target.get("exchange_order_ids"))
            if str(item).strip()
        )
    return {
        "entry_decision_ids": _dedupe(entry_decision_ids),
        "exit_decision_ids": _dedupe(exit_decision_ids),
        "order_ids": _dedupe(order_ids),
        "exchange_order_ids": _dedupe(exchange_order_ids),
    }


def _entry_recovery_plan(row: Any, *, generated_at: str) -> dict[str, Any]:
    raw = _safe_dict(_row_get(row, "raw_llm_response"))
    existing_plan = _safe_dict(raw.get("profit_first_trade_plan"))
    sizing = dict(_safe_dict(raw.get("profit_risk_sizing")))
    existing_ladder = _safe_dict(sizing.get("profit_first_position_ladder"))
    derived_plan = build_profit_first_trade_plan(
        row,
        analysis_type=_row_get(row, "analysis_type"),
    ).to_dict()
    ladder = ProfitFirstPositionLadderPolicy().apply(
        lane=str(derived_plan.get("decision_lane") or "shadow_only"),
        current_size_pct=_safe_float(
            _row_get(row, "position_size_pct"),
            _safe_float(sizing.get("position_size_pct"), 0.0),
        ),
        low_payoff_quality=bool(sizing.get("low_payoff_quality")),
        high_risk_review=_safe_dict(raw.get("high_risk_review")),
    ).to_dict()
    patch: dict[str, Any] = {
        "profit_first_historical_recovery": _recovery_marker(
            generated_at=generated_at,
            kind="entry_trade_plan_and_ladder",
        ),
    }
    if not existing_plan:
        patch["profit_first_trade_plan"] = derived_plan
        patch["profit_first_exit_plan"] = _exit_plan_from_trade_plan(derived_plan)
        patch["profit_first_entry_exit_binding"] = {
            "exit_plan_id": derived_plan.get("exit_plan_id") or "",
            "required_for_real_entry": True,
            "exit_decisions_must_reference_plan": True,
            "source": "profit_first_historical_recovery_package",
            "training_policy": "exclude_until_manual_trust",
        }
    if not existing_ladder:
        sizing["profit_first_position_ladder"] = ladder
        patch["profit_risk_sizing"] = sizing
    complete = bool(derived_plan.get("is_complete_for_real_trade"))
    return {
        "item_type": "entry_decision_recovery",
        "decision_id": _safe_int(_row_get(row, "id")),
        "symbol": _row_get(row, "symbol"),
        "action": _row_get(row, "action"),
        "operator_approval_required": True,
        "recommended_resolution": (
            "operator_approved_backfill_then_quarantine_from_training_until_trusted"
            if complete
            else "quarantine_from_promotion_training_and_keep_resume_blocked_until_window_rolls_or_manual_review"
        ),
        "existing_state": {
            "has_profit_first_trade_plan": bool(existing_plan),
            "has_profit_first_position_ladder": bool(existing_ladder),
            "was_executed": bool(_row_get(row, "was_executed")),
            "analysis_type": _row_get(row, "analysis_type"),
        },
        "derived_state": {
            "plan_complete": complete,
            "missing_required_fields": _safe_list(derived_plan.get("missing_required_fields")),
            "decision_lane": derived_plan.get("decision_lane"),
            "exit_plan_id": derived_plan.get("exit_plan_id"),
            "ladder_lane": ladder.get("lane"),
            "ladder_adjusted_size_pct": ladder.get("adjusted_size_pct"),
        },
        "proposed_raw_patch": patch,
        "training_policy": "exclude_until_manual_trust",
        "apply_notes": [
            "Do not apply unless the raw decision payload and OKX-backed order facts are reviewed.",
            "If applied, write a backup of ai_decisions.raw_llm_response before mutation.",
            "Do not count this historical backfill as clean training until manually trusted.",
        ],
    }


def _exit_recovery_plan(row: Any, *, generated_at: str) -> dict[str, Any]:
    raw = _safe_dict(_row_get(row, "raw_llm_response"))
    reference = _safe_dict(raw.get("profit_first_exit_reference"))
    close_evidence = dict(_safe_dict(raw.get("close_evidence")))
    has_reference = bool(
        str(reference.get("exit_plan_id") or close_evidence.get("profit_first_exit_plan_id") or "").strip()
    )
    plan_failure_reason = str(
        reference.get("plan_failure_reason")
        or close_evidence.get("profit_first_plan_failure_reason")
        or raw.get("profit_first_plan_failure_reason")
        or raw.get("plan_failure_reason")
        or ""
    ).strip()
    close_evidence.setdefault("profit_first_plan_failure_reason", LEGACY_EXIT_FAILURE_REASON)
    patch = {
        "profit_first_exit_reference": {
            "exit_plan_id": reference.get("exit_plan_id") or "",
            "source": "profit_first_historical_recovery_legacy_missing_reference",
            "missing_original_exit_plan_reference": not has_reference,
            "plan_failure_reason": plan_failure_reason or LEGACY_EXIT_FAILURE_REASON,
            "training_policy": "exclude_until_manual_trust",
        },
        "close_evidence": close_evidence,
        "profit_first_historical_recovery": _recovery_marker(
            generated_at=generated_at,
            kind="exit_legacy_reference_marker",
        ),
    }
    return {
        "item_type": "exit_decision_recovery",
        "decision_id": _safe_int(_row_get(row, "id")),
        "symbol": _row_get(row, "symbol"),
        "action": _row_get(row, "action"),
        "operator_approval_required": True,
        "recommended_resolution": (
            "match_original_exit_plan_if_possible_otherwise_apply_legacy_failure_marker_and_quarantine"
        ),
        "existing_state": {
            "has_exit_plan_reference": has_reference,
            "has_plan_failure_reason": bool(plan_failure_reason),
            "was_executed": bool(_row_get(row, "was_executed")),
            "analysis_type": _row_get(row, "analysis_type"),
        },
        "proposed_raw_patch": patch,
        "training_policy": "exclude_until_manual_trust",
        "apply_notes": [
            "Prefer matching the original entry exit_plan_id before using the legacy marker.",
            "If no original plan can be proven, keep the sample quarantined from training.",
            "Re-run trade_execution_contract after any approved change.",
        ],
    }


def _okx_order_review(order: Any, *, generated_at: str) -> dict[str, Any]:
    raw_fills = _safe_dict(_row_get(order, "okx_raw_fills"))
    return {
        "item_type": "okx_order_quantity_review",
        "order_id": _safe_int(_row_get(order, "id")),
        "decision_id": _safe_int(_row_get(order, "decision_id")),
        "exchange_order_id": _row_get(order, "exchange_order_id"),
        "symbol": _row_get(order, "symbol"),
        "side": _row_get(order, "side"),
        "operator_approval_required": True,
        "recommended_resolution": "run_exact_okx_order_review_then_allowlisted_repair_or_quarantine",
        "existing_state": {
            "quantity": _safe_float(_row_get(order, "quantity")),
            "price": _safe_float(_row_get(order, "price")),
            "status": _row_get(order, "status"),
            "okx_inst_id": _row_get(order, "okx_inst_id"),
            "okx_fill_contracts": _row_get(order, "okx_fill_contracts"),
            "okx_sync_status": _row_get(order, "okx_sync_status"),
            "okx_raw_contract_size": raw_fills.get("contract_size"),
            "okx_raw_base_quantity": raw_fills.get("base_quantity"),
        },
        "proposed_raw_patch": {},
        "training_policy": "exclude_until_okx_backed_and_operator_trusted",
        "suggested_commands": [
            "python scripts/run_phase3_okx_fact_sync.py --mode paper --json-indent 2",
            (
                "python scripts/repair_okx_history_position_reconciliation.py "
                f"--exchange-order-id {_row_get(order, 'exchange_order_id')}"
            ),
        ],
        "recovery_marker": _recovery_marker(
            generated_at=generated_at,
            kind="okx_order_quantity_review",
        ),
    }


def _ranking_reviews(
    blocking_actions: list[dict[str, Any]],
    *,
    generated_at: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for action in blocking_actions:
        row = _safe_dict(action)
        if str(row.get("category") or "") != "ranking":
            continue
        rows.append(
            {
                "item_type": "ranking_disable_review",
                "code": row.get("code"),
                "operator_approval_required": True,
                "recommended_resolution": "keep_shadow_only_or_operator_disable_before_resume",
                "target": _safe_dict(row.get("target")),
                "proposed_raw_patch": {},
                "training_policy": "do_not_promote_or_increase_budget_until_clean_realized_pnl",
                "recovery_marker": _recovery_marker(
                    generated_at=generated_at,
                    kind="ranking_shadow_disable_review",
                ),
            }
        )
    return rows


def _exit_plan_from_trade_plan(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "exit_plan_id": plan.get("exit_plan_id") or "",
        "stop_loss_pct": plan.get("stop_loss_pct"),
        "take_profit_pct": plan.get("take_profit_pct"),
        "trailing_profit_trigger_pct": plan.get("trailing_profit_trigger_pct"),
        "profit_drawdown_exit_pct": plan.get("profit_drawdown_exit_pct"),
        "partial_exit_plan": plan.get("partial_exit_plan") or [],
        "full_exit_plan": plan.get("full_exit_plan") or {},
        "do_not_close_conditions": plan.get("do_not_close_conditions") or [],
        "max_hold_minutes": plan.get("max_hold_minutes"),
        "invalidation_price": plan.get("invalidation_price"),
        "generated_from_historical_recovery": True,
        "training_policy": "exclude_until_manual_trust",
    }


def _recovery_marker(*, generated_at: str, kind: str) -> dict[str, Any]:
    return {
        "version": RECOVERY_PACKAGE_VERSION,
        "kind": kind,
        "generated_at": generated_at,
        "dry_run_source": "profit_first_historical_recovery_package",
        "operator_approval_required": True,
        "training_policy": "exclude_until_manual_trust",
    }


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _unique_ints(values: list[Any]) -> list[int]:
    return _dedupe(_safe_int(value) for value in values if _safe_int(value) > 0)


def _dedupe(values: Any) -> list[Any]:
    result: list[Any] = []
    seen: set[Any] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
