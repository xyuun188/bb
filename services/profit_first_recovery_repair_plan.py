"""Read-only Profit-First recovery repair/quarantine planning."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any


def build_profit_first_recovery_repair_plan(
    recovery_blockers: dict[str, Any] | None,
) -> dict[str, Any]:
    """Convert recovery blockers into explicit operator-reviewed actions.

    The result is deliberately a dry-run plan.  It never edits historical facts,
    strategy state, sizing, routing, or OKX state.
    """

    blockers = _safe_dict(recovery_blockers)
    items = [_safe_dict(item) for item in _safe_list(blockers.get("items"))]
    actions = [_action_for_item(item) for item in items if item]
    status_counts = Counter(str(action.get("status") or "pending_review") for action in actions)
    category_counts = Counter(str(action.get("category") or "unknown") for action in actions)
    blocking_actions = [
        action
        for action in actions
        if str(action.get("resume_gate_effect") or "") == "blocks_resume_until_resolved"
    ]
    return {
        "report_type": "profit_first_recovery_repair_plan",
        "status": "clear" if not blocking_actions else "blocked",
        "generated_at": datetime.now(UTC).isoformat(),
        "dry_run": True,
        "read_only": True,
        "audit_only": True,
        "mutates_database": False,
        "starts_trading_service": False,
        "submits_orders": False,
        "changes_model_routing": False,
        "changes_live_sizing": False,
        "live_mutation": False,
        "can_start_trading_service": False,
        "can_submit_orders": False,
        "can_change_model_routing": False,
        "can_increase_live_size": False,
        "resume_allowed_by_this_plan": False,
        "input_summary": {
            "recovery_status": blockers.get("status") or "missing",
            "resume_clear": bool(blockers.get("resume_clear")),
            "blocking_item_count": _safe_int(blockers.get("blocking_item_count")),
            "warning_item_count": _safe_int(blockers.get("warning_item_count")),
            "item_count": len(items),
        },
        "summary": {
            "action_count": len(actions),
            "blocking_action_count": len(blocking_actions),
            "operator_approval_required_count": sum(
                1 for action in actions if bool(action.get("operator_approval_required"))
            ),
            "status_counts": dict(status_counts),
            "category_counts": dict(category_counts),
        },
        "blocking_actions": blocking_actions,
        "actions": actions,
        "next_validation": [
            "Run this repair plan again after each approved repair/quarantine.",
            "Run scripts/run_phase3_go_no_go_report.py --stdout-only and require status != blocked.",
            "Run scripts/verify_profit_first_online_readiness.py and require resume_allowed_by_this_check=true before restoring analysis or entries.",
        ],
        "policy": {
            "historical_backfill_must_be_operator_approved": True,
            "untrusted_or_synthetic_repairs_stay_out_of_training": True,
            "okx_quantity_differences_require_exact_order_id_review": True,
            "ranking_disable_actions_do_not_auto_mutate_live_routes": True,
            "do_not_restore_new_entries_until_go_no_go_clears": True,
        },
    }


def _action_for_item(item: dict[str, Any]) -> dict[str, Any]:
    category = str(item.get("category") or "unknown")
    code = str(item.get("code") or "unknown")
    if category == "trade_contract":
        return _trade_contract_action(item, code)
    if category == "ranking":
        return _ranking_action(item, code)
    if category == "okx_reconciliation":
        return _okx_action(item, code)
    return _base_action(
        item,
        action_type="manual_review",
        status="pending_review",
        recommended_action="Review and classify this blocker before resume.",
        approval=True,
    )


def _trade_contract_action(item: dict[str, Any], code: str) -> dict[str, Any]:
    samples = [_safe_dict(sample) for sample in _safe_list(item.get("samples"))]
    decision_ids = _unique_ints(sample.get("decision_id") for sample in samples)
    if code == "missing_profit_first_trade_plan":
        return _base_action(
            item,
            action_type="historical_trade_plan_backfill_or_quarantine",
            status="approval_required",
            recommended_action=(
                "For each executed legacy entry, either derive and persist a legacy ProfitFirstTradePlan "
                "with provenance, or quarantine the entry from promotion/training and keep resume blocked "
                "until the current window no longer contains missing plans."
            ),
            approval=True,
            target_decision_ids=decision_ids,
            validation=[
                "trade_execution_contract.current_summary.profit_first_plan_missing_count == 0",
                "No derived plan may be used as clean training unless OKX-backed and operator trusted.",
            ],
            suggested_commands=[
                _decision_review_command(decision_id) for decision_id in decision_ids[:8]
            ],
        )
    if code == "missing_profit_first_position_ladder":
        return _base_action(
            item,
            action_type="historical_position_ladder_backfill_or_quarantine",
            status="approval_required",
            recommended_action=(
                "For each affected executed entry, derive the position ladder from the persisted "
                "ProfitFirstTradePlan and original sizing evidence, or quarantine the entry from "
                "budget promotion if evidence is incomplete."
            ),
            approval=True,
            target_decision_ids=decision_ids,
            validation=[
                "trade_execution_contract.current_summary.profit_first_position_ladder_missing_count == 0",
                "Derived ladder must preserve original size and record provenance.",
            ],
            suggested_commands=[
                _decision_review_command(decision_id) for decision_id in decision_ids[:8]
            ],
        )
    if code == "missing_profit_first_exit_plan_reference":
        return _base_action(
            item,
            action_type="exit_reference_repair_or_legacy_failure_marker",
            status="approval_required",
            recommended_action=(
                "Match each exit to the original entry exit_plan_id. If no trustworthy entry plan exists, "
                "mark the exit as legacy missing-reference with a plan_failure_reason and quarantine it "
                "from promotion/training."
            ),
            approval=True,
            target_decision_ids=decision_ids,
            validation=[
                "trade_execution_contract.current_summary.exit_plan_reference_missing_count == 0",
                "Every non-plan exit has a concrete plan_failure_reason.",
            ],
            suggested_commands=[
                _decision_review_command(decision_id) for decision_id in decision_ids[:8]
            ],
        )
    if code == "missing_profit_first_exit_plan_failure_reason":
        return _base_action(
            item,
            action_type="exit_failure_reason_backfill",
            status="approval_required",
            recommended_action=(
                "Backfill a concrete Profit-First plan failure reason for exits outside the original plan, "
                "or quarantine the sample until reviewed."
            ),
            approval=True,
            target_decision_ids=decision_ids,
            validation=[
                "trade_execution_contract.current_summary.exit_plan_failure_reason_missing_count == 0",
            ],
            suggested_commands=[
                _decision_review_command(decision_id) for decision_id in decision_ids[:8]
            ],
        )
    return _base_action(
        item,
        action_type="trade_contract_manual_review",
        status="pending_review",
        recommended_action=(
            "Review the trade-contract blocker and resolve it before restoring market analysis or entries."
        ),
        approval=True,
        target_decision_ids=decision_ids,
    )


def _ranking_action(item: dict[str, Any], code: str) -> dict[str, Any]:
    severity = str(item.get("severity") or "warning")
    blocking = severity == "blocking"
    action_type = "ranking_shadow_disable_review" if blocking else "ranking_demotion_review"
    return _base_action(
        item,
        action_type=action_type,
        status="approval_required" if blocking else "shadow_only_observe",
        recommended_action=(
            "Keep this model/strategy/lane combination shadow-only or operator-disable it before resume."
            if blocking
            else "Do not increase budget for this model/strategy/lane combination until clean realized-PnL evidence recovers."
        ),
        approval=blocking,
        validation=[
            "profit_first_ranking.summary.disable_count == 0 before resume"
            if blocking
            else "profit_first_ranking warning is acceptable only if no live budget is increased",
            "No routing, weight, or sizing mutation happens from this dry-run plan.",
        ],
        suggested_commands=[],
    )


def _okx_action(item: dict[str, Any], code: str) -> dict[str, Any]:
    exchange_order_id = str(item.get("exchange_order_id") or "").strip()
    local_order_id = _safe_int(item.get("local_order_id"))
    if code == "local_order_quantity_differs_from_okx_fill":
        return _base_action(
            item,
            action_type="okx_exact_order_quantity_repair_review",
            status="approval_required",
            recommended_action=(
                "Review the exact local order and OKX ordId quantity conversion. If OKX ctVal/fill facts "
                "prove the local quantity cache is wrong, run an allowlisted repair with backup; otherwise "
                "quarantine the sample and keep it out of training."
            ),
            approval=True,
            target_order_ids=[local_order_id] if local_order_id > 0 else [],
            target_exchange_order_ids=[exchange_order_id] if exchange_order_id else [],
            validation=[
                "phase3_paper_resume_observation okx_authoritative_sync issue_count == 0",
                "OKX/local quantity comparison is clean after fresh pull, not from stale cache.",
            ],
            suggested_commands=[
                command
                for command in (
                    (
                        "python scripts/run_phase3_okx_fact_sync.py --mode paper --json-indent 2"
                    ),
                    (
                        "python scripts/repair_okx_history_position_reconciliation.py "
                        f"--exchange-order-id {exchange_order_id}"
                        if exchange_order_id
                        else ""
                    ),
                )
                if command
            ],
        )
    return _base_action(
        item,
        action_type="okx_reconciliation_manual_review",
        status="approval_required",
        recommended_action=(
            "Resolve or quarantine the OKX/local reconciliation blocker before resume."
        ),
        approval=True,
        target_order_ids=[local_order_id] if local_order_id > 0 else [],
        target_exchange_order_ids=[exchange_order_id] if exchange_order_id else [],
    )


def _base_action(
    item: dict[str, Any],
    *,
    action_type: str,
    status: str,
    recommended_action: str,
    approval: bool,
    target_decision_ids: list[int] | None = None,
    target_order_ids: list[int] | None = None,
    target_exchange_order_ids: list[str] | None = None,
    validation: list[str] | None = None,
    suggested_commands: list[str] | None = None,
) -> dict[str, Any]:
    category = str(item.get("category") or "unknown")
    severity = str(item.get("severity") or "warning")
    return {
        "category": category,
        "code": item.get("code") or "unknown",
        "severity": severity,
        "action_type": action_type,
        "status": status,
        "operator_approval_required": bool(approval),
        "resume_gate_effect": (
            "blocks_resume_until_resolved" if severity == "blocking" else "warning_no_budget_increase"
        ),
        "training_policy": _training_policy(category, action_type),
        "target": _target_payload(
            item,
            decision_ids=target_decision_ids or [],
            order_ids=target_order_ids or [],
            exchange_order_ids=target_exchange_order_ids or [],
        ),
        "recommended_action": recommended_action,
        "validation": validation or ["Re-run Profit-First online readiness after resolution."],
        "suggested_commands": suggested_commands or [],
        "source_item": _compact_source_item(item),
    }


def _target_payload(
    item: dict[str, Any],
    *,
    decision_ids: list[int],
    order_ids: list[int],
    exchange_order_ids: list[str],
) -> dict[str, Any]:
    return {
        "decision_ids": decision_ids,
        "order_ids": order_ids,
        "exchange_order_ids": exchange_order_ids,
        "symbol": item.get("symbol"),
        "side": item.get("side"),
        "model_name": item.get("model_name"),
        "strategy_profile_id": item.get("strategy_profile_id"),
        "decision_lane": item.get("decision_lane"),
    }


def _training_policy(category: str, action_type: str) -> str:
    if category == "okx_reconciliation":
        return "exclude_until_okx_backed_and_operator_trusted"
    if category == "ranking":
        return "do_not_promote_or_increase_budget_until_clean_realized_pnl"
    if "quarantine" in action_type or "legacy" in action_type:
        return "exclude_from_clean_training_view_until_manual_trust"
    return "exclude_until_manual_review"


def _compact_source_item(item: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "category",
        "severity",
        "code",
        "count",
        "message",
        "symbol",
        "side",
        "local_order_id",
        "local_position_id",
        "exchange_order_id",
        "classification",
        "model_name",
        "strategy_profile_id",
        "decision_lane",
        "realized_net_pnl",
        "ranking_reasons",
        "required_resolution",
        "samples",
    }
    return {key: item.get(key) for key in allowed if key in item}


def _decision_review_command(decision_id: int) -> str:
    return (
        "python - <<'PY'\n"
        "import asyncio, json\n"
        "from db.session import get_read_session_ctx\n"
        "from models.decision import AIDecision\n"
        f"DECISION_ID = {decision_id}\n"
        "async def main():\n"
        "    async with get_read_session_ctx() as session:\n"
        "        row = await session.get(AIDecision, DECISION_ID)\n"
        "        print(json.dumps({'id': getattr(row, 'id', None), 'symbol': getattr(row, 'symbol', None), 'action': getattr(row, 'action', None), 'raw_llm_response': getattr(row, 'raw_llm_response', None)}, ensure_ascii=False, indent=2))\n"
        "asyncio.run(main())\n"
        "PY"
    )


def _unique_ints(values: Any) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        number = _safe_int(value)
        if number <= 0 or number in seen:
            continue
        seen.add(number)
        result.append(number)
    return result


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
