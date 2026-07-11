"""Read-only Profit-First recovery blocker diagnosis."""

from __future__ import annotations

from collections import Counter
from typing import Any


def build_profit_first_recovery_blockers(
    *,
    trade_contract: dict[str, Any] | None,
    ranking: dict[str, Any] | None,
    observation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Turn resume blockers into a concrete, read-only cleanup checklist."""

    trade_contract = _safe_dict(trade_contract)
    ranking = _safe_dict(ranking)
    observation = _safe_dict(observation)

    contract_items = _contract_items(trade_contract)
    ranking_items = _ranking_items(ranking)
    okx_items = _okx_items(observation)
    items = [*contract_items, *ranking_items, *okx_items]
    category_counts = Counter(str(item.get("category") or "unknown") for item in items)
    severity_counts = Counter(str(item.get("severity") or "warning") for item in items)
    can_resume = not any(str(item.get("severity")) == "blocking" for item in items)
    return {
        "report_type": "profit_first_recovery_blockers",
        "status": "ready" if can_resume else "blocked",
        "read_only": True,
        "audit_only": True,
        "starts_trading_service": False,
        "submits_orders": False,
        "changes_model_routing": False,
        "changes_live_sizing": False,
        "live_mutation": False,
        "can_start_trading_service": False,
        "can_submit_orders": False,
        "can_change_model_routing": False,
        "can_increase_live_size": False,
        "resume_clear": can_resume,
        "blocking_item_count": int(severity_counts.get("blocking", 0)),
        "warning_item_count": int(severity_counts.get("warning", 0)),
        "category_counts": dict(category_counts),
        "items": items[:80],
        "summary": {
            "contract_blocker_count": len(
                [item for item in contract_items if item.get("severity") == "blocking"]
            ),
            "ranking_blocker_count": len(
                [item for item in ranking_items if item.get("severity") == "blocking"]
            ),
            "okx_blocker_count": len(
                [item for item in okx_items if item.get("severity") == "blocking"]
            ),
            "contract_item_count": len(contract_items),
            "ranking_item_count": len(ranking_items),
            "okx_item_count": len(okx_items),
        },
        "policy": {
            "does_not_repair_history": True,
            "does_not_mutate_strategy_state": True,
            "does_not_disable_routes": True,
            "recovery_requires_operator_approved_repair_or_quarantine": True,
        },
    }


def _contract_items(trade_contract: dict[str, Any]) -> list[dict[str, Any]]:
    summary = _safe_dict(trade_contract.get("current_summary")) or _safe_dict(
        trade_contract.get("summary")
    )
    violations = _safe_list(trade_contract.get("current_violations")) or _safe_list(
        trade_contract.get("violations")
    )
    quarantined_violations = _safe_list(
        trade_contract.get("current_historical_recovery_quarantined_violations")
    ) or _safe_list(trade_contract.get("historical_recovery_quarantined_violations"))
    items: list[dict[str, Any]] = []
    for reason, count_key, message in (
        (
            "missing_profit_first_trade_plan",
            "profit_first_plan_missing_count",
            "Executed entries have no persisted ProfitFirstTradePlan.",
        ),
        (
            "missing_profit_first_position_ladder",
            "profit_first_position_ladder_missing_count",
            "Executed entries have no persisted position-ladder decision.",
        ),
        (
            "missing_profit_first_exit_plan_reference",
            "exit_plan_reference_missing_count",
            "Executed exits did not reference the original Profit-First exit plan.",
        ),
        (
            "missing_profit_first_exit_plan_failure_reason",
            "exit_plan_failure_reason_missing_count",
            "Exits outside the original plan have no failure reason.",
        ),
        (
            "profit_first_probe_loss_brake_bypassed",
            "probe_loss_brake_bypassed_count",
            "Probe-loss brake was bypassed.",
        ),
    ):
        count = _safe_int(summary.get(count_key))
        quarantine_key = f"historical_recovery_quarantined_{count_key}"
        quarantined_count = _safe_int(summary.get(quarantine_key))
        unresolved_count = _safe_int(summary.get(f"{count_key}_unresolved"), count)
        if unresolved_count <= 0:
            if quarantined_count > 0:
                items.append(
                    {
                        "category": "trade_contract",
                        "severity": "warning",
                        "code": f"{reason}_historical_quarantined",
                        "count": quarantined_count,
                        "message": (
                            f"{message} These rows are already quarantined by "
                            "Profit-First historical recovery and stay excluded from "
                            "clean training, promotion, and budget increase."
                        ),
                        "samples": _samples_for_reason(quarantined_violations, reason),
                        "required_resolution": (
                            "keep_quarantined_until_manual_trust_or_window_rolls"
                        ),
                    }
                )
            continue
        items.append(
            {
                "category": "trade_contract",
                "severity": "blocking",
                "code": reason,
                "count": unresolved_count,
                "historical_recovery_quarantined_count": quarantined_count,
                "message": message,
                "samples": _samples_for_reason(violations, reason),
                "required_resolution": (
                    "operator_approved_fact_repair_or_quarantine_before_resume"
                ),
            }
        )
    return items


def _samples_for_reason(violations: list[Any], reason: str) -> list[dict[str, Any]]:
    return [
        {
            "decision_id": _safe_dict(item).get("decision_id"),
            "symbol": _safe_dict(item).get("symbol"),
            "action": _safe_dict(item).get("action"),
            "reason": _safe_dict(item).get("reason"),
        }
        for item in violations
        if _safe_dict(item).get("reason") == reason
    ][:10]


def _ranking_items(ranking: dict[str, Any]) -> list[dict[str, Any]]:
    blockers = [_safe_dict(item) for item in _safe_list(ranking.get("blockers"))]
    summary = _safe_dict(ranking.get("summary"))
    items: list[dict[str, Any]] = []
    for item in blockers:
        severity = str(item.get("severity") or "warning")
        if severity not in {"blocking", "warning"}:
            severity = "warning"
        evidence = _safe_dict(item.get("evidence"))
        if str(item.get("code") or "").startswith("strategy_"):
            lane_contained = bool(
                evidence.get("model_name")
                or evidence.get("strategy_profile_id")
                or evidence.get("symbol")
                or evidence.get("side")
                or evidence.get("decision_lane")
            )
            if severity == "blocking" and lane_contained:
                severity = "warning"
            items.append(
                {
                    "category": "ranking",
                    "severity": severity,
                    "code": item.get("code"),
                    "message": item.get("message")
                    or "Model/strategy/lane combination needs ranking action.",
                    "model_name": evidence.get("model_name"),
                    "strategy_profile_id": evidence.get("strategy_profile_id"),
                    "symbol": evidence.get("symbol"),
                    "side": evidence.get("side"),
                    "decision_lane": evidence.get("decision_lane"),
                    "realized_net_pnl": evidence.get("realized_net_pnl"),
                    "ranking_reasons": evidence.get("ranking_reasons") or [],
                    "lane_scoped_containment": lane_contained,
                    "required_resolution": (
                        "keep_affected_lane_disabled_until_clean_samples"
                        if lane_contained
                        else "operator_review_unscoped_ranking_blocker"
                    ),
                }
            )
    blocking_disable_items = [
        item
        for item in items
        if item.get("severity") == "blocking" and str(item.get("code") or "").startswith("strategy_")
    ]
    disable_count = _safe_int(summary.get("disable_count"))
    if disable_count > len(blocking_disable_items):
        items.append(
            {
                "category": "ranking",
                "severity": "warning",
                "code": "strategy_disable_summary",
                "count": disable_count,
                "message": (
                    "Ranking summary reports disabled model/strategy/lane combinations, "
                    "but detailed blockers were truncated or unavailable."
                ),
                "lane_scoped_containment": True,
                "required_resolution": "keep_affected_lanes_disabled_until_details_refresh",
            }
        )
    warning_demote_items = [
        item
        for item in items
        if item.get("severity") == "warning" and str(item.get("code") or "").startswith("strategy_")
    ]
    demote_count = _safe_int(summary.get("demote_count"))
    if demote_count > len(warning_demote_items):
        items.append(
            {
                "category": "ranking",
                "severity": "warning",
                "code": "strategy_demote_summary",
                "count": demote_count,
                "message": (
                    "Ranking summary reports demoted model/strategy/lane combinations; "
                    "no budget increase is allowed until clean realized-PnL evidence recovers."
                ),
                "required_resolution": "do_not_increase_budget_until_clean_samples",
            }
        )
    return items


def _okx_items(observation: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for blocker in _safe_list(observation.get("blockers")):
        blocker = _safe_dict(blocker)
        code = str(blocker.get("code") or "")
        if "okx_authoritative_sync" not in code:
            continue
        evidence = _safe_dict(blocker.get("evidence"))
        for issue in _safe_list(evidence.get("issues")):
            issue = _safe_dict(issue)
            items.append(
                {
                    "category": "okx_reconciliation",
                    "severity": "blocking",
                    "code": issue.get("kind") or code,
                    "message": issue.get("reason") or blocker.get("message"),
                    "symbol": issue.get("symbol"),
                    "local_order_id": issue.get("local_order_id"),
                    "local_position_id": issue.get("local_position_id"),
                    "exchange_order_id": issue.get("exchange_order_id"),
                    "classification": issue.get("classification"),
                    "required_resolution": "repair_or_quarantine_okx_difference_before_resume",
                }
            )
        if not _safe_list(evidence.get("issues")):
            items.append(
                {
                    "category": "okx_reconciliation",
                    "severity": "blocking",
                    "code": code,
                    "message": blocker.get("message"),
                    "required_resolution": "restore_clean_okx_authoritative_sync_before_resume",
                }
            )
    return items


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
