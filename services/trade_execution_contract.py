"""Read-only audit of the dynamic return execution contract."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from math import isfinite
from typing import Any

ENTRY_ACTIONS = {"long", "short", "open_long", "open_short", "buy", "sell"}
EXIT_ACTIONS = {"close_long", "close_short", "exit_long", "exit_short"}
FILLED_STATUSES = {"filled", "closed"}
OBSOLETE_POLICY_FIELDS = {
    "entry_evidence",
    "entry_evidence_probe",
    "profit_first_trade_plan",
    "profit_first_exit_plan",
    "profit_first_position_ladder",
    "quant_profit_probe",
    "tradeable_probe",
    "probe_fraction",
    "full_position_release",
}
PROVENANCE_FIELDS = (
    "source",
    "observation_window",
    "sample_count",
    "generated_at",
    "strategy_version",
    "fallback_reason",
)


class TradeExecutionContractService:
    def __init__(self, session_context_factory: Any | None = None) -> None:
        self._session_context_factory = session_context_factory

    async def report(
        self,
        *,
        hours: int = 24,
        limit: int = 500,
        since: datetime | None = None,
    ) -> dict[str, Any]:
        from sqlalchemy import or_, select

        from db.session import get_read_session_ctx
        from models.decision import AIDecision
        from models.trade import Order, Position

        capped_hours = max(1, min(int(hours or 24), 168))
        capped_limit = max(1, min(int(limit or 500), 5000))
        since_utc = _as_utc(since) or datetime.now(UTC) - timedelta(hours=capped_hours)
        since_naive = since_utc.replace(tzinfo=None)
        session_factory = self._session_context_factory or get_read_session_ctx
        async with session_factory() as session:
            decisions = list(
                (
                    await session.execute(
                        select(AIDecision)
                        .where(AIDecision.created_at >= since_naive)
                        .order_by(AIDecision.id.desc())
                        .limit(capped_limit)
                    )
                )
                .scalars()
                .all()
            )
            orders = list(
                (
                    await session.execute(
                        select(Order)
                        .where(Order.created_at >= since_naive)
                        .order_by(Order.id.desc())
                        .limit(capped_limit)
                    )
                )
                .scalars()
                .all()
            )
            positions = list(
                (
                    await session.execute(
                        select(Position)
                        .where(
                            or_(
                                Position.created_at >= since_naive,
                                Position.closed_at >= since_naive,
                            )
                        )
                        .order_by(Position.id.desc())
                        .limit(capped_limit)
                    )
                )
                .scalars()
                .all()
            )
        return summarize_trade_execution_contract(
            decisions,
            orders=orders,
            positions=positions,
        )


def summarize_trade_execution_contract(
    decisions: Sequence[Any],
    *,
    orders: Sequence[Any] | None = None,
    positions: Sequence[Any] | None = None,
) -> dict[str, Any]:
    orders_by_decision = _orders_by_decision(orders or [])
    violations: list[dict[str, Any]] = []
    entry_rows: list[dict[str, Any]] = []
    exit_rows: list[dict[str, Any]] = []
    executed_entry_count = 0
    executed_exit_count = 0

    for decision in decisions:
        action = _action(decision)
        if action not in ENTRY_ACTIONS | EXIT_ACTIONS:
            continue
        raw = _safe_dict(_row_get(decision, "raw_llm_response"))
        decision_orders = orders_by_decision.get(_safe_int(_row_get(decision, "id")), [])
        executed = _was_executed(decision, decision_orders)
        obsolete = sorted(_obsolete_fields(raw))
        if obsolete:
            violations.append(
                _violation(decision, "obsolete_policy_payload_present", {"fields": obsolete})
            )

        if action in ENTRY_ACTIONS:
            row, reasons = _entry_contract_row(decision, raw, decision_orders, executed)
            entry_rows.append(row)
            if executed:
                executed_entry_count += 1
                violations.extend(_violation(decision, reason, row) for reason in reasons)
        else:
            row, reasons = _exit_contract_row(decision, raw, decision_orders, executed)
            exit_rows.append(row)
            if executed:
                executed_exit_count += 1
                violations.extend(_violation(decision, reason, row) for reason in reasons)

    reason_counts = Counter(str(row["reason"]) for row in violations)
    realized_values = [
        _safe_float(_row_get(position, "realized_pnl"), 0.0)
        for position in positions or []
        if _row_get(position, "closed_at") is not None
    ]
    summary = {
        "decision_count": len(decisions),
        "executed_entry_count": executed_entry_count,
        "executed_exit_count": executed_exit_count,
        "entry_contract_ready_count": sum(
            1 for row in entry_rows if row["executed"] and row["contract_complete"]
        ),
        "exit_contract_ready_count": sum(
            1 for row in exit_rows if row["executed"] and row["contract_complete"]
        ),
        "contract_violation_count": len(violations),
        "obsolete_policy_payload_count": reason_counts["obsolete_policy_payload_present"],
        "closed_position_count": len(realized_values),
        "realized_net_pnl_usdt": round(sum(realized_values), 8),
        "negative_realized_position_count": sum(value < 0 for value in realized_values),
    }
    return {
        "audit_only": True,
        "live_entry_mutation": False,
        "live_exit_mutation": False,
        "can_bypass_risk_controls": False,
        "summary": summary,
        "violation_reason_counts": dict(reason_counts),
        "entry_contracts": entry_rows[:50],
        "exit_contracts": exit_rows[:50],
        "violations": violations[:100],
        "policy": {
            "optimization_target": "realized_fee_after_return",
            "entry_requires_positive_fee_after_return": True,
            "entry_requires_positive_return_lcb": True,
            "entry_requires_live_execution_cost": True,
            "entry_requires_dynamic_risk_budget": True,
            "entry_requires_complete_provenance": True,
            "exit_requires_position_economics": True,
            "exit_requires_dynamic_close_fraction": True,
            "filled_order_link_required": True,
            "obsolete_policy_payload_forbidden": sorted(OBSOLETE_POLICY_FIELDS),
        },
    }


def _entry_contract_row(
    decision: Any,
    raw: dict[str, Any],
    orders: list[Any],
    executed: bool,
) -> tuple[dict[str, Any], list[str]]:
    policy = _safe_dict(raw.get("production_return_policy"))
    opportunity = _safe_dict(raw.get("opportunity_score"))
    cost = _safe_dict(opportunity.get("execution_cost"))
    sizing = _safe_dict(raw.get("profit_risk_sizing"))
    reasons: list[str] = []
    if not policy or policy.get("eligible") is not True:
        reasons.append("production_return_policy_missing_or_ineligible")
    if _safe_float(policy.get("expected_net_return_pct"), 0.0) <= 0:
        reasons.append("fee_after_expected_return_not_positive")
    if _safe_float(policy.get("return_lcb_pct"), 0.0) <= 0:
        reasons.append("fee_after_return_lcb_not_positive")
    if _safe_int(policy.get("production_source_count")) <= 0:
        reasons.append("production_return_distribution_missing")
    if _safe_float(policy.get("position_size_pct"), 0.0) <= 0:
        reasons.append("dynamic_position_budget_zero")
    if not _provenance_complete(policy.get("policy_provenance")):
        reasons.append("production_return_provenance_incomplete")
    if opportunity.get("production_eligible") is not True:
        reasons.append("opportunity_return_distribution_ineligible")
    if not _provenance_complete(opportunity.get("policy_provenance")):
        reasons.append("opportunity_return_provenance_incomplete")
    if cost.get("production_eligible") is not True or _safe_float(cost.get("total_pct")) <= 0:
        reasons.append("live_execution_cost_incomplete")
    if not _provenance_complete(cost.get("policy_provenance")):
        reasons.append("execution_cost_provenance_incomplete")
    if sizing.get("production_eligible") is not True:
        reasons.append("dynamic_risk_budget_ineligible")
    if not _provenance_complete(sizing.get("policy_provenance")):
        reasons.append("dynamic_risk_budget_provenance_incomplete")
    if executed and not _has_filled_order(orders):
        reasons.append("executed_entry_without_filled_order")
    row = {
        "decision_id": _row_get(decision, "id"),
        "symbol": _row_get(decision, "symbol"),
        "action": _action(decision),
        "executed": executed,
        "filled_order_count": sum(_order_status(order) in FILLED_STATUSES for order in orders),
        "contract_complete": not reasons,
        "expected_net_return_pct": _safe_float(policy.get("expected_net_return_pct")),
        "return_lcb_pct": _safe_float(policy.get("return_lcb_pct")),
        "execution_cost_pct": _safe_float(policy.get("execution_cost_pct")),
        "position_size_pct": _safe_float(policy.get("position_size_pct")),
        "production_source_count": _safe_int(policy.get("production_source_count")),
        "reasons": reasons,
    }
    return row, reasons


def _exit_contract_row(
    decision: Any,
    raw: dict[str, Any],
    orders: list[Any],
    executed: bool,
) -> tuple[dict[str, Any], list[str]]:
    policy = _safe_dict(raw.get("dynamic_exit_policy"))
    reasons: list[str] = []
    if not policy or policy.get("eligible") is not True:
        reasons.append("dynamic_exit_policy_missing_or_ineligible")
    if _safe_float(policy.get("close_fraction"), 0.0) <= 0:
        reasons.append("dynamic_exit_fraction_zero")
    if _safe_int(_safe_dict(policy.get("policy_provenance")).get("sample_count")) <= 0:
        reasons.append("exit_position_economics_missing")
    if not _provenance_complete(policy.get("policy_provenance")):
        reasons.append("dynamic_exit_provenance_incomplete")
    if executed and not _has_filled_order(orders):
        reasons.append("executed_exit_without_filled_order")
    row = {
        "decision_id": _row_get(decision, "id"),
        "symbol": _row_get(decision, "symbol"),
        "action": _action(decision),
        "executed": executed,
        "filled_order_count": sum(_order_status(order) in FILLED_STATUSES for order in orders),
        "contract_complete": not reasons,
        "close_fraction": _safe_float(policy.get("close_fraction")),
        "hard_risk": bool(policy.get("hard_risk")),
        "fee_after_unrealized_pnl_usdt": _safe_float(
            policy.get("fee_after_unrealized_pnl_usdt")
        ),
        "reasons": reasons,
    }
    return row, reasons


def _obsolete_fields(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) in OBSOLETE_POLICY_FIELDS:
                found.add(str(key))
            found.update(_obsolete_fields(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_obsolete_fields(child))
    return found


def _provenance_complete(value: Any) -> bool:
    provenance = _safe_dict(value)
    if any(key not in provenance for key in PROVENANCE_FIELDS):
        return False
    return bool(
        str(provenance.get("source") or "").strip()
        and str(provenance.get("observation_window") or "").strip()
        and _safe_int(provenance.get("sample_count")) > 0
        and str(provenance.get("generated_at") or "").strip()
        and str(provenance.get("strategy_version") or "").strip()
        and not str(provenance.get("fallback_reason") or "").strip()
    )


def _orders_by_decision(orders: Sequence[Any]) -> dict[int, list[Any]]:
    grouped: dict[int, list[Any]] = {}
    for order in orders:
        decision_id = _safe_int(_row_get(order, "decision_id"))
        if decision_id:
            grouped.setdefault(decision_id, []).append(order)
    return grouped


def _was_executed(decision: Any, orders: list[Any]) -> bool:
    return bool(_row_get(decision, "was_executed")) or _has_filled_order(orders)


def _has_filled_order(orders: Sequence[Any]) -> bool:
    return any(_order_status(order) in FILLED_STATUSES for order in orders)


def _order_status(order: Any) -> str:
    value = _row_get(order, "status")
    return str(getattr(value, "value", value) or "").lower()


def _action(row: Any) -> str:
    value = _row_get(row, "action")
    return str(getattr(value, "value", value) or "").lower()


def _violation(decision: Any, reason: str, details: dict[str, Any]) -> dict[str, Any]:
    return {
        "reason": reason,
        "decision_id": _row_get(decision, "id"),
        "symbol": _row_get(decision, "symbol"),
        "action": _action(decision),
        "details": details,
    }


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if isfinite(result) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_utc(value: Any) -> datetime | None:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
