"""Read-only audit of the dynamic return execution contract."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from math import isclose, isfinite
from types import SimpleNamespace
from typing import Any

from services.okx_native_facts import OKX_PROTECTION_EXECUTION_VERSION
from services.paper_bootstrap_canary import (
    PAPER_BOOTSTRAP_CANARY_VERSION,
    PAPER_BOOTSTRAP_MIN_FILL_DRIFT_RESERVE_FRACTION,
    PAPER_BOOTSTRAP_SIZING_VERSION,
)

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
                SimpleNamespace(**dict(row))
                for row in (
                    await session.execute(
                        select(
                            AIDecision.id,
                            AIDecision.symbol,
                            AIDecision.action,
                            AIDecision.was_executed,
                            AIDecision.decision_learning_snapshot.label(
                                "raw_llm_response"
                            ),
                        )
                        .where(
                            AIDecision.created_at >= since_naive,
                            AIDecision.decision_learning_snapshot_version >= 1,
                        )
                        .order_by(AIDecision.id.desc())
                        .limit(capped_limit)
                    )
                )
                .mappings()
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
    filled_notional = sum(
        abs(_safe_float(_row_get(order, "quantity")) * _safe_float(_row_get(order, "price")))
        for order in orders
        if _order_status(order) in FILLED_STATUSES
    )
    contract, reasons = validate_entry_execution_contract(
        raw,
        filled_notional_usdt=filled_notional,
        executed=executed,
        filled_order_present=_has_filled_order(orders),
    )
    row = {
        "decision_id": _row_get(decision, "id"),
        "symbol": _row_get(decision, "symbol"),
        "action": _action(decision),
        "executed": executed,
        "filled_order_count": sum(
            _order_status(order) in FILLED_STATUSES for order in orders
        ),
        **contract,
        "reasons": reasons,
    }
    return row, reasons


def entry_contract_lifecycle(raw: dict[str, Any]) -> str:
    """Classify an entry contract before validating lifecycle-specific invariants."""

    canary = _safe_dict(raw.get("paper_bootstrap_canary"))
    sizing = _safe_dict(raw.get("profit_risk_sizing"))
    opportunity = _safe_dict(raw.get("opportunity_score"))
    if (
        any(
            key in canary
            for key in (
                "version",
                "authorized",
                "requested",
                "selected_observation",
            )
        )
        or sizing.get("contract_lifecycle") == "paper_bootstrap_canary"
        or opportunity.get("contract_lifecycle") == "paper_bootstrap_canary"
    ):
        return "paper_bootstrap_canary"
    return "production_return"


def entry_opportunity_evidence_score(raw: dict[str, Any]) -> float | None:
    """Return a finite lifecycle-specific score for self-check coverage."""

    if entry_contract_lifecycle(raw) == "paper_bootstrap_canary":
        canary = _safe_dict(raw.get("paper_bootstrap_canary"))
        if _paper_canary_observation_reasons(canary):
            return None
        return _finite_value(
            _safe_dict(canary.get("selected_observation")).get(
                "objective_expected_return_pct"
            )
        )
    return _finite_value(_safe_dict(raw.get("opportunity_score")).get("score"))


def validate_entry_execution_contract(
    raw: dict[str, Any],
    *,
    filled_notional_usdt: float = 0.0,
    executed: bool = False,
    filled_order_present: bool | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Validate the persisted contract for its declared execution lifecycle."""

    if entry_contract_lifecycle(raw) == "paper_bootstrap_canary":
        return validate_paper_canary_entry_contract(
            raw,
            filled_notional_usdt=filled_notional_usdt,
            executed=executed,
            filled_order_present=filled_order_present,
        )
    return validate_production_entry_contract(
        raw,
        filled_notional_usdt=filled_notional_usdt,
        executed=executed,
        filled_order_present=filled_order_present,
    )


def validate_paper_canary_entry_contract(
    raw: dict[str, Any],
    *,
    filled_notional_usdt: float = 0.0,
    executed: bool = False,
    filled_order_present: bool | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Validate a paper-only canary without applying production-return rules."""

    canary = _safe_dict(raw.get("paper_bootstrap_canary"))
    opportunity = _safe_dict(raw.get("opportunity_score"))
    observation = _safe_dict(canary.get("selected_observation"))
    sizing = _safe_dict(raw.get("profit_risk_sizing"))
    reasons = _paper_canary_observation_reasons(canary)
    bounded_fill_drift = _bounded_canary_fill_drift(
        canary=canary,
        opportunity=opportunity,
        sizing=sizing,
    )
    bounded_fill_drift_accepted = bounded_fill_drift["accepted"] is True

    if canary.get("runtime_authorized") is not True:
        reasons.append("paper_canary_runtime_guard_not_authorized")
    runtime_guard = _safe_dict(canary.get("runtime_guard"))
    if not runtime_guard or _safe_list(runtime_guard.get("blocking_reasons")):
        reasons.append("paper_canary_runtime_guard_incomplete")

    if sizing.get("contract_version") != PAPER_BOOTSTRAP_SIZING_VERSION:
        reasons.append("paper_canary_sizing_version_invalid")
    if sizing.get("contract_lifecycle") != "paper_bootstrap_canary":
        reasons.append("paper_canary_sizing_lifecycle_mismatch")
    if sizing.get("execution_scope") != "paper_only":
        reasons.append("paper_canary_sizing_scope_invalid")
    if sizing.get("production_permission") is not False:
        reasons.append("paper_canary_sizing_production_permission_invalid")
    if (
        sizing.get("production_eligible") is not True
        and not bounded_fill_drift_accepted
    ):
        reasons.append("paper_canary_risk_contract_ineligible")
    sizing_provenance = _safe_dict(sizing.get("policy_provenance"))
    if (
        (
            not _provenance_complete(sizing_provenance)
            and not (
                bounded_fill_drift_accepted
                and _provenance_core_complete(sizing_provenance)
                and set(
                    filter(
                        None,
                        str(sizing_provenance.get("fallback_reason") or "").split(","),
                    )
                )
                == set(bounded_fill_drift["reasons"])
            )
        )
        or sizing_provenance.get("strategy_version") != PAPER_BOOTSTRAP_SIZING_VERSION
        or not str(sizing_provenance.get("contract_fingerprint") or "").strip()
    ):
        reasons.append("paper_canary_sizing_provenance_incomplete")

    risk_budget = _safe_float(sizing.get("risk_budget_usdt"))
    portfolio_budget = _safe_float(sizing.get("portfolio_risk_budget_usdt"))
    planned_loss = _safe_float(sizing.get("planned_stressed_loss_usdt"))
    stress_fraction = _safe_float(sizing.get("stressed_loss_fraction"))
    target_notional = _safe_float(sizing.get("target_notional_usdt"))
    final_notional = _safe_float(sizing.get("final_notional_usdt"))
    final_margin = _safe_float(sizing.get("final_margin_usdt"))
    available_margin = _safe_float(sizing.get("available_margin_usdt"))
    position_size = _safe_float(sizing.get("position_size_pct"))
    if (
        risk_budget <= 0
        or portfolio_budget <= 0
        or risk_budget > portfolio_budget + 1e-8
        or planned_loss <= 0
        or (
            planned_loss > risk_budget + 1e-8
            and not bounded_fill_drift_accepted
        )
    ):
        reasons.append("paper_canary_risk_budget_invalid")
    if stress_fraction <= 0 or not isclose(
        planned_loss,
        final_notional * stress_fraction,
        rel_tol=1e-9,
        abs_tol=1e-8,
    ):
        reasons.append("paper_canary_stressed_loss_algebra_mismatch")
    if final_notional <= 0 or (
        final_notional > target_notional + 1e-8
        and not bounded_fill_drift_accepted
    ):
        reasons.append("paper_canary_notional_invalid")
    effective_position_size = (
        final_margin / available_margin
        if final_margin > 0 and available_margin > 0
        else 0.0
    )
    if (
        final_margin <= 0
        or available_margin <= 0
        or (
            not bounded_fill_drift_accepted
            and (
                position_size <= 0
                or not isclose(
                    position_size,
                    effective_position_size,
                    rel_tol=1e-7,
                    abs_tol=1e-8,
                )
            )
        )
    ):
        reasons.append("paper_canary_sizing_identity_incomplete")

    portfolio = _safe_dict(sizing.get("portfolio_risk_snapshot"))
    if (
        portfolio.get("scope") != "paper_bootstrap_canary_positions_only"
        or _safe_float(portfolio.get("current_stressed_loss_usdt")) < 0
    ):
        reasons.append("paper_canary_portfolio_snapshot_incomplete")
    availability = _safe_dict(sizing.get("entry_instrument_availability"))
    leverage_tier = _safe_dict(sizing.get("leverage_tier_selection"))
    if availability.get("available") is not True:
        reasons.append("paper_canary_instrument_availability_unconfirmed")
    if leverage_tier.get("production_eligible") is not True:
        reasons.append("paper_canary_leverage_tier_ineligible")

    if opportunity.get("contract_lifecycle") == "paper_bootstrap_canary":
        annotated_score = _finite_value(opportunity.get("score"))
        objective_score = _finite_value(observation.get("objective_expected_return_pct"))
        if (
            opportunity.get("score_kind")
            != "paper_canary_objective_expected_return"
            or annotated_score is None
            or objective_score is None
            or not isclose(annotated_score, objective_score, abs_tol=1e-8)
            or opportunity.get("production_eligible") is not False
            or opportunity.get("production_permission") is not False
            or opportunity.get("observation_only") is not True
            or opportunity.get("execution_scope") != "paper_only"
        ):
            reasons.append("paper_canary_opportunity_annotation_invalid")

    filled_notional = max(_safe_float(filled_notional_usdt), 0.0)
    if executed and filled_notional > 0 and not isclose(
        final_notional,
        filled_notional,
        rel_tol=1e-9,
        abs_tol=1e-8,
    ):
        reasons.append("filled_order_notional_differs_from_risk_contract")
    if executed and filled_order_present is not True:
        reasons.append("executed_entry_without_filled_order")

    reasons = list(dict.fromkeys(reasons))
    contract = {
        "contract_lifecycle": "paper_bootstrap_canary",
        "contract_complete": not reasons,
        "execution_scope": canary.get("execution_scope"),
        "production_permission": False,
        "observation_only": True,
        "artifact_version": canary.get("artifact_version"),
        "selected_side": canary.get("selected_side"),
        "opportunity_score": _finite_value(
            observation.get("objective_expected_return_pct")
        ),
        "observed_net_return_pct": _finite_value(
            observation.get("observed_net_return_pct")
        ),
        "risk_budget_usdt": risk_budget,
        "planned_stressed_loss_usdt": planned_loss,
        "final_notional_usdt": final_notional,
        "filled_order_notional_usdt": filled_notional,
        "production_source_count": 0,
        "bounded_fill_drift_accepted": bounded_fill_drift_accepted,
        "fill_drift_evidence": bounded_fill_drift,
        "effective_position_size_pct": effective_position_size,
    }
    return contract, reasons


def _bounded_canary_fill_drift(
    *,
    canary: dict[str, Any],
    opportunity: dict[str, Any],
    sizing: dict[str, Any],
) -> dict[str, Any]:
    """Validate a confirmed canary fill against its reserved risk ceiling."""

    rejected = {
        "accepted": False,
        "reasons": [],
        "reserve_fraction": 0.0,
        "notional_excess_fraction": 0.0,
        "risk_excess_fraction": 0.0,
    }
    reconciliations = [
        _safe_dict(item) for item in _safe_list(sizing.get("execution_reconciliations"))
    ]
    pre_submit = next(
        (
            item
            for item in reversed(reconciliations)
            if item.get("source") == "okx_pre_submit_order_shape"
            and item.get("eligible") is True
            and not _safe_list(item.get("reasons"))
        ),
        None,
    )
    confirmed_fill = next(
        (
            item
            for item in reversed(reconciliations)
            if item.get("source") == "okx_confirmed_entry_fill"
        ),
        None,
    )
    if not pre_submit or not confirmed_fill:
        return rejected

    fill_reasons = {
        str(reason) for reason in _safe_list(confirmed_fill.get("reasons")) if reason
    }
    allowed_reasons = {
        "execution_notional_exceeds_authoritative_target",
        "execution_stressed_loss_exceeds_risk_budget",
    }
    confirmed_fill_eligible = confirmed_fill.get("eligible") is True
    if (
        not fill_reasons.issubset(allowed_reasons)
        or (confirmed_fill_eligible and fill_reasons)
        or (not confirmed_fill_eligible and not fill_reasons)
    ):
        return rejected

    target_notional = _safe_float(sizing.get("target_notional_usdt"))
    settled_notional = _safe_float(confirmed_fill.get("final_notional_usdt"))
    final_notional = _safe_float(sizing.get("final_notional_usdt"))
    risk_budget = _safe_float(sizing.get("risk_budget_usdt"))
    planned_loss = _safe_float(sizing.get("planned_stressed_loss_usdt"))
    declared_reserve_fraction = _safe_float(
        sizing.get("estimated_fill_drift_reserve_fraction")
    )
    declared_fill_ceiling = _safe_float(sizing.get("fill_notional_ceiling_usdt"))
    observation_cost_pct = _safe_float(
        _safe_dict(canary.get("selected_observation")).get(
            "current_execution_cost_pct"
        )
    )
    opportunity_cost_pct = _safe_float(
        _safe_dict(opportunity.get("execution_cost")).get("total_pct")
    )
    observed_reserve_fraction = max(
        max(observation_cost_pct, opportunity_cost_pct) / 100.0,
        PAPER_BOOTSTRAP_MIN_FILL_DRIFT_RESERVE_FRACTION,
    )
    explicit_reserve_contract = bool(
        declared_reserve_fraction > 0
        and declared_fill_ceiling > 0
        and isclose(
            target_notional * (1.0 + declared_reserve_fraction),
            declared_fill_ceiling,
            rel_tol=1e-7,
            abs_tol=1e-8,
        )
        and declared_reserve_fraction + 1e-8 >= observed_reserve_fraction
    )
    reserve_fraction = (
        declared_reserve_fraction
        if explicit_reserve_contract
        else observed_reserve_fraction
    )
    fill_ceiling = (
        declared_fill_ceiling
        if explicit_reserve_contract
        else target_notional * (1.0 + reserve_fraction)
    )
    if (
        target_notional <= 0
        or settled_notional <= 0
        or risk_budget <= 0
        or planned_loss <= 0
        or reserve_fraction <= 0
        or not isclose(final_notional, settled_notional, rel_tol=1e-9, abs_tol=1e-8)
    ):
        return rejected
    notional_excess = max(settled_notional / target_notional - 1.0, 0.0)
    risk_excess = max(planned_loss / risk_budget - 1.0, 0.0)
    accepted = bool(
        notional_excess <= reserve_fraction + 1e-8
        and settled_notional <= fill_ceiling + 1e-8
        and (
            planned_loss <= risk_budget + 1e-8
            if explicit_reserve_contract
            else risk_excess <= reserve_fraction + 1e-8
        )
    )
    return {
        "accepted": accepted,
        "reasons": sorted(fill_reasons),
        "reserve_fraction": reserve_fraction,
        "notional_excess_fraction": notional_excess,
        "risk_excess_fraction": risk_excess,
        "fill_notional_ceiling_usdt": fill_ceiling,
        "explicit_reserve_contract": explicit_reserve_contract,
        "pre_submit_notional_usdt": _safe_float(
            pre_submit.get("final_notional_usdt")
        ),
        "settled_notional_usdt": settled_notional,
        "source": (
            "persisted_canary_fill_reserve_and_okx_reconciliations"
            if explicit_reserve_contract
            else "legacy_cost_bound_and_okx_reconciliations"
        ),
    }


def _paper_canary_observation_reasons(canary: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if canary.get("version") != PAPER_BOOTSTRAP_CANARY_VERSION:
        reasons.append("paper_canary_version_invalid")
    if canary.get("authorized") is not True or canary.get("requested") is not True:
        reasons.append("paper_canary_not_authorized")
    if canary.get("execution_scope") != "paper_only":
        reasons.append("paper_canary_scope_invalid")
    if canary.get("production_permission") is not False:
        reasons.append("paper_canary_production_permission_invalid")
    if canary.get("artifact_lifecycle") != "canary":
        reasons.append("paper_canary_artifact_lifecycle_invalid")

    observation = _safe_dict(canary.get("selected_observation"))
    side = str(canary.get("selected_side") or "").lower()
    observation_side = str(observation.get("side") or "").lower()
    required_finite = (
        "raw_expected_return_pct",
        "objective_expected_return_pct",
        "lower_quantile_return_pct",
        "dispersion_pct",
        "observed_net_return_pct",
    )
    if (
        side not in {"long", "short"}
        or observation_side != side
        or any(_finite_value(observation.get(key)) is None for key in required_finite)
        or _safe_int(observation.get("horizon_minutes")) <= 0
        or _safe_int(observation.get("distribution_member_count")) <= 0
        or not str(observation.get("source_authority") or "").strip()
    ):
        reasons.append("paper_canary_selected_observation_incomplete")
    direction_gap = _finite_value(canary.get("direction_score_gap"))
    confidence = _finite_value(canary.get("confidence"))
    if (
        direction_gap is None
        or direction_gap < 0
        or confidence is None
        or not 0 <= confidence <= 1
    ):
        reasons.append("paper_canary_direction_evidence_incomplete")
    provenance = _safe_dict(canary.get("policy_provenance"))
    if (
        not _provenance_complete(provenance)
        or provenance.get("strategy_version") != PAPER_BOOTSTRAP_CANARY_VERSION
    ):
        reasons.append("paper_canary_provenance_incomplete")
    return reasons


def validate_production_entry_contract(
    raw: dict[str, Any],
    *,
    filled_notional_usdt: float = 0.0,
    executed: bool = False,
    filled_order_present: bool | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Validate the persisted production-entry contract without mutating it."""

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
    risk_budget = _safe_float(sizing.get("risk_budget_usdt"))
    planned_loss = _safe_float(sizing.get("planned_stressed_loss_usdt"))
    stress_fraction = _safe_float(sizing.get("stressed_loss_fraction"))
    target_notional = _safe_float(sizing.get("target_notional_usdt"))
    final_notional = _safe_float(sizing.get("final_notional_usdt"))
    if risk_budget <= 0 or planned_loss <= 0 or planned_loss > risk_budget + 1e-8:
        reasons.append("dynamic_risk_budget_algebra_invalid")
    if stress_fraction <= 0 or not isclose(
        planned_loss,
        final_notional * stress_fraction,
        rel_tol=1e-9,
        abs_tol=1e-8,
    ):
        reasons.append("dynamic_stressed_loss_algebra_invalid")
    if final_notional <= 0 or final_notional > target_notional + 1e-8:
        reasons.append("dynamic_notional_target_invalid")
    filled_notional = max(_safe_float(filled_notional_usdt), 0.0)
    if executed and filled_notional > 0 and not isclose(
        final_notional,
        filled_notional,
        rel_tol=1e-9,
        abs_tol=1e-8,
    ):
        reasons.append("filled_order_notional_differs_from_risk_contract")
    if executed and filled_order_present is not True:
        reasons.append("executed_entry_without_filled_order")
    contract = {
        "contract_lifecycle": "production_return",
        "contract_complete": not reasons,
        "expected_net_return_pct": _safe_float(policy.get("expected_net_return_pct")),
        "return_lcb_pct": _safe_float(policy.get("return_lcb_pct")),
        "execution_cost_pct": _safe_float(policy.get("execution_cost_pct")),
        "position_size_pct": _safe_float(policy.get("position_size_pct")),
        "risk_budget_usdt": risk_budget,
        "planned_stressed_loss_usdt": planned_loss,
        "final_notional_usdt": final_notional,
        "filled_order_notional_usdt": filled_notional,
        "production_source_count": _safe_int(policy.get("production_source_count")),
    }
    return contract, reasons


def _exit_contract_row(
    decision: Any,
    raw: dict[str, Any],
    orders: list[Any],
    executed: bool,
) -> tuple[dict[str, Any], list[str]]:
    protection_contract = _system_protection_exit_contract(raw, orders)
    if protection_contract is not None:
        reasons = list(protection_contract.pop("reasons"))
        if executed and not _has_filled_order(orders):
            reasons.append("executed_exit_without_filled_order")
        row = {
            "decision_id": _row_get(decision, "id"),
            "symbol": _row_get(decision, "symbol"),
            "action": _action(decision),
            "executed": executed,
            "filled_order_count": sum(
                _order_status(order) in FILLED_STATUSES for order in orders
            ),
            "contract_complete": not reasons,
            **protection_contract,
            "reasons": reasons,
        }
        return row, reasons

    external_reconcile_contract = _external_okx_reconcile_exit_contract(raw, orders)
    if external_reconcile_contract is not None:
        reasons = list(external_reconcile_contract.pop("reasons"))
        if executed and not _has_filled_order(orders):
            reasons.append("executed_exit_without_filled_order")
        row = {
            "decision_id": _row_get(decision, "id"),
            "symbol": _row_get(decision, "symbol"),
            "action": _action(decision),
            "executed": executed,
            "filled_order_count": sum(
                _order_status(order) in FILLED_STATUSES for order in orders
            ),
            "contract_complete": not reasons,
            **external_reconcile_contract,
            "reasons": reasons,
        }
        return row, reasons

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


def classify_exit_execution_contract(
    decision: Any,
    orders: Sequence[Any],
) -> dict[str, Any]:
    """Classify an exit through the same contract owner used by the global audit."""

    raw = _safe_dict(_row_get(decision, "raw_llm_response"))
    executed = _was_executed(decision, list(orders))
    row, _ = _exit_contract_row(decision, raw, list(orders), executed)
    if not row.get("contract_kind"):
        row["contract_kind"] = "dynamic_exit"
    return row


def _system_protection_exit_contract(
    raw: dict[str, Any],
    orders: list[Any],
) -> dict[str, Any] | None:
    close_fill = _safe_dict(raw.get("close_fill"))
    identified = bool(
        raw.get("system_sync") is True
        and raw.get("source") == "okx_position_reconcile"
        and (
            raw.get("reconcile_origin") == "system_protection"
            or close_fill.get("reconcile_origin") == "system_protection"
        )
    )
    if not identified:
        return None

    lifecycles = [
        _safe_dict(_safe_dict(_row_get(order, "okx_raw_fills")).get("protection_execution"))
        for order in orders
        if _safe_dict(_safe_dict(_row_get(order, "okx_raw_fills")).get("protection_execution"))
    ]
    lifecycle = lifecycles[0] if len(lifecycles) == 1 else {}
    generated_order_id = str(lifecycle.get("generated_order_id") or "").strip()
    filled_order_ids = {
        str(_row_get(order, "exchange_order_id") or "").strip()
        for order in orders
        if _order_status(order) in FILLED_STATUSES
        and str(_row_get(order, "exchange_order_id") or "").strip()
    }
    reasons: list[str] = []
    if len(lifecycles) != 1:
        reasons.append("exchange_protection_lifecycle_not_unique")
    if lifecycle.get("version") != OKX_PROTECTION_EXECUTION_VERSION:
        reasons.append("exchange_protection_version_mismatch")
    if lifecycle.get("source_authority") != "okx_algo_history_plus_fills_history":
        reasons.append("exchange_protection_source_not_authoritative")
    if lifecycle.get("lifecycle_complete") is not True:
        reasons.append("exchange_protection_lifecycle_incomplete")
    if str(lifecycle.get("actual_side") or "").lower() not in {"sl", "tp"}:
        reasons.append("exchange_protection_trigger_side_missing")
    if _safe_float(lifecycle.get("contracts"), 0.0) <= 0:
        reasons.append("exchange_protection_fill_contracts_missing")
    if not generated_order_id or generated_order_id not in filled_order_ids:
        reasons.append("exchange_protection_generated_order_mismatch")
    return {
        "contract_kind": "okx_exchange_protection",
        "close_fraction": None,
        "close_contracts": _safe_float(lifecycle.get("contracts"), 0.0),
        "hard_risk": str(lifecycle.get("actual_side") or "").lower() == "sl",
        "fee_after_unrealized_pnl_usdt": None,
        "protection_execution_version": lifecycle.get("version"),
        "protection_algo_id": lifecycle.get("algo_id"),
        "protection_actual_side": lifecycle.get("actual_side"),
        "protection_trigger_to_first_fill_ms": lifecycle.get(
            "trigger_to_first_fill_ms"
        ),
        "reasons": reasons,
    }


def _external_okx_reconcile_exit_contract(
    raw: dict[str, Any],
    orders: list[Any],
) -> dict[str, Any] | None:
    close_fill = _safe_dict(raw.get("close_fill"))
    identified = bool(
        raw.get("system_sync") is True
        and raw.get("source") == "okx_position_reconcile"
        and (
            raw.get("reconcile_origin") == "external_okx_sync"
            or close_fill.get("reconcile_origin") == "external_okx_sync"
        )
    )
    if not identified:
        return None

    authoritative_orders = []
    for order in orders:
        fact = _safe_dict(_row_get(order, "okx_raw_fills"))
        exchange_order_id = str(_row_get(order, "exchange_order_id") or "").strip()
        fact_order_id = str(fact.get("order_id") or "").strip()
        complete = bool(
            _order_status(order) in FILLED_STATUSES
            and fact.get("fills_history_confirmed") is True
            and exchange_order_id
            and fact_order_id == exchange_order_id
            and str(fact.get("inst_id") or "").strip()
            and _safe_float(fact.get("contracts"), 0.0) > 0
            and fact.get("contract_size_verified") is True
            and _safe_float(fact.get("base_quantity"), 0.0) > 0
            and _safe_float(fact.get("avg_price"), 0.0) > 0
            and fact.get("fee_abs") is not None
        )
        if complete:
            authoritative_orders.append(order)
    reasons: list[str] = []
    if len(authoritative_orders) != 1:
        reasons.append("external_okx_close_fill_lifecycle_not_unique")
    authoritative = authoritative_orders[0] if len(authoritative_orders) == 1 else None
    fact = _safe_dict(_row_get(authoritative, "okx_raw_fills")) if authoritative else {}
    return {
        "contract_kind": "okx_external_reconciliation",
        "close_fraction": _safe_float(_safe_dict(raw.get("close_fill")).get("close_fraction")),
        "close_contracts": _safe_float(fact.get("contracts"), 0.0),
        "hard_risk": None,
        "fee_after_unrealized_pnl_usdt": None,
        "authoritative_close_order_id": (
            str(_row_get(authoritative, "exchange_order_id") or "")
            if authoritative is not None
            else None
        ),
        "reasons": reasons,
    }


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
    return bool(
        _provenance_core_complete(provenance)
        and not str(provenance.get("fallback_reason") or "").strip()
    )


def _provenance_core_complete(value: Any) -> bool:
    provenance = _safe_dict(value)
    if any(key not in provenance for key in PROVENANCE_FIELDS):
        return False
    return bool(
        str(provenance.get("source") or "").strip()
        and str(provenance.get("observation_window") or "").strip()
        and _safe_int(provenance.get("sample_count")) > 0
        and str(provenance.get("generated_at") or "").strip()
        and str(provenance.get("strategy_version") or "").strip()
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


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _finite_value(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


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
