"""Persist a complete, attributable trade recommendation for paper decisions."""

from __future__ import annotations

import copy
import math
from datetime import UTC, datetime
from typing import Any

from ai_brain.base_model import Action, DecisionOutput

TRADE_RECOMMENDATION_CONTRACT_VERSION = "2026-07-22.complete-trade-recommendation.v1"
ENTRY_ACTIONS = {Action.LONG, Action.SHORT}


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _positive(*values: Any) -> float | None:
    for value in values:
        number = _finite(value)
        if number is not None and number > 0:
            return number
    return None


def _first_number(*values: Any) -> float | None:
    for value in values:
        number = _finite(value)
        if number is not None:
            return number
    return None


def _side(decision: DecisionOutput) -> str:
    if decision.action in {Action.LONG, Action.CLOSE_LONG}:
        return "long"
    if decision.action in {Action.SHORT, Action.CLOSE_SHORT}:
        return "short"
    return "hold"


def _decision_kind(decision: DecisionOutput) -> str:
    if decision.is_entry:
        return "entry"
    if decision.is_exit:
        return "exit"
    return "hold"


def _opportunity_sources(
    raw: dict[str, Any],
    side: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    opportunity = _dict(raw.get("opportunity_score"))
    candidate = _dict(raw.get("entry_candidate_evidence"))
    side_candidate = _dict(candidate.get(side))
    training = _dict(raw.get("paper_training"))
    canary = _dict(_dict(raw.get("paper_bootstrap_canary")).get("selected_observation"))
    exploration = _dict(raw.get("paper_exploration"))
    return opportunity, side_candidate, training, canary, exploration


def _return_plan(raw: dict[str, Any], side: str) -> dict[str, Any]:
    opportunity, side_candidate, training, canary, exploration = _opportunity_sources(
        raw,
        side,
    )
    distribution = _dict(opportunity.get("return_distribution_contract"))
    expected = _first_number(
        opportunity.get("expected_net_return_pct"),
        distribution.get("objective_expected_return_pct"),
        side_candidate.get("expected_net_return_pct"),
        training.get("expected_net_return_pct"),
        canary.get("observed_net_return_pct"),
        exploration.get("expected_net_return_pct"),
    )
    lower = _first_number(
        opportunity.get("return_lcb_pct"),
        distribution.get("lower_quantile_return_pct"),
        side_candidate.get("return_lcb_pct"),
        training.get("return_lcb_pct"),
        canary.get("lower_quantile_net_return_pct"),
        exploration.get("return_lcb_pct"),
    )
    uncertainty = _first_number(
        opportunity.get("return_uncertainty_pct"),
        distribution.get("return_uncertainty_pct"),
        distribution.get("uncertainty_penalty_pct"),
        canary.get("dispersion_pct"),
    )
    upper = _first_number(
        distribution.get("upper_quantile_return_pct"),
        expected + abs(uncertainty)
        if expected is not None and uncertainty is not None
        else None,
        expected,
    )
    return {
        "unit": "percentage_points_after_cost",
        "expected_pct": expected,
        "lower_bound_pct": lower,
        "upper_bound_pct": upper,
        "includes_fee": True,
        "includes_slippage": True,
        "includes_funding": True,
        "source": (
            "authoritative_opportunity_return_distribution"
            if _finite(opportunity.get("expected_net_return_pct")) is not None
            else "paper_training_return_contract"
            if _finite(training.get("expected_net_return_pct")) is not None
            else "paper_canary_empirical_distribution"
            if _finite(canary.get("observed_net_return_pct")) is not None
            else "paper_exploration_return_contract"
            if _finite(exploration.get("expected_net_return_pct")) is not None
            else "missing"
        ),
    }


def _holding_plan(raw: dict[str, Any], side: str) -> dict[str, Any]:
    opportunity, side_candidate, training, canary, exploration = _opportunity_sources(
        raw,
        side,
    )
    distribution = _dict(opportunity.get("return_distribution_contract"))
    target = _positive(
        distribution.get("horizon_minutes"),
        side_candidate.get("horizon_minutes"),
        training.get("prediction_horizon_minutes"),
        training.get("horizon_minutes"),
        canary.get("horizon_minutes"),
        exploration.get("prediction_horizon_minutes"),
        exploration.get("horizon_minutes"),
    )
    max_minutes = _positive(
        raw.get("max_holding_minutes"),
        _dict(raw.get("dynamic_exit_policy")).get("max_holding_minutes"),
        target,
    )
    valid_for_seconds = _positive(
        training.get("valid_for_seconds"),
        exploration.get("valid_for_seconds"),
        _dict(raw.get("entry_permission_policy")).get("valid_for_seconds"),
        min(target * 60.0, 300.0) if target is not None else None,
    )
    return {
        "target_minutes": target,
        "maximum_minutes": max_minutes,
        "entry_valid_for_seconds": valid_for_seconds,
        "source": (
            "model_return_horizon"
            if target is not None
            else "missing"
        ),
    }


def _entry_plan(decision: DecisionOutput) -> dict[str, Any]:
    snapshot = _dict(decision.feature_snapshot)
    side = _side(decision)
    bid = _positive(snapshot.get("bid"), snapshot.get("best_bid"))
    ask = _positive(snapshot.get("ask"), snapshot.get("best_ask"))
    reference = _positive(
        ask if side == "long" else bid,
        snapshot.get("current_price"),
        snapshot.get("last_price"),
        snapshot.get("mark_price"),
        snapshot.get("close"),
        bid,
        ask,
    )
    quote_values = [value for value in (bid, ask, reference) if value is not None]
    return {
        "price_reference": reference,
        "minimum_price": min(quote_values) if quote_values else None,
        "maximum_price": max(quote_values) if quote_values else None,
        "source": "decision_time_executable_quote" if quote_values else "missing",
    }


def _loss_plan(decision: DecisionOutput, raw: dict[str, Any]) -> dict[str, Any]:
    side = _side(decision)
    opportunity, side_candidate, training, canary, exploration = _opportunity_sources(
        raw,
        side,
    )
    sizing = _dict(raw.get("profit_risk_sizing"))
    stop_fraction = _positive(
        decision.stop_loss_pct,
        sizing.get("stressed_loss_fraction"),
    )
    return_plan = _return_plan(raw, side)
    lower_return = _finite(return_plan.get("lower_bound_pct"))
    normal = _positive(
        sizing.get("expected_loss_pct"),
        opportunity.get("expected_loss_pct"),
        side_candidate.get("expected_loss_pct"),
        training.get("expected_loss_pct"),
        canary.get("dispersion_pct"),
        exploration.get("expected_loss_pct"),
        abs(min(lower_return, 0.0)) if lower_return is not None else None,
        (stop_fraction or 0.0) * 100.0,
    )
    extreme = max(
        value
        for value in (
            normal or 0.0,
            (stop_fraction or 0.0) * 100.0,
            _positive(opportunity.get("return_uncertainty_pct")) or 0.0,
        )
    )
    return {
        "unit": "percentage_points",
        "normal_loss_pct": normal,
        "extreme_loss_pct": extreme if extreme > 0 else None,
        "planned_max_loss_usdt": _positive(
            sizing.get("planned_stressed_loss_usdt"),
            sizing.get("risk_budget_usdt"),
        ),
        "source": "dynamic_risk_budget_and_return_distribution",
    }


def _exit_plan(decision: DecisionOutput, raw: dict[str, Any]) -> dict[str, Any]:
    holding = _holding_plan(raw, _side(decision))
    stop = _positive(decision.stop_loss_pct)
    take = _positive(decision.take_profit_pct)
    close_evidence = _dict(raw.get("close_evidence"))
    close_fraction = _positive(
        close_evidence.get("close_fraction"),
        close_evidence.get("position_size_pct"),
    )
    return {
        "stop_loss_fraction": stop,
        "take_profit_fraction": take,
        "invalidation": {
            "type": "hard_stop_loss",
            "threshold_fraction": stop,
        },
        "continue_holding": {
            "type": "within_horizon_without_invalidation",
            "maximum_minutes": holding.get("maximum_minutes"),
        },
        "partial_close": {
            "type": "dynamic_profit_or_risk_reduction",
            "close_fraction": close_fraction,
            "enabled": bool(close_fraction is not None and close_fraction < 1.0),
        },
        "full_close": {
            "type": "stop_loss_take_profit_max_horizon_or_dynamic_exit",
            "maximum_minutes": holding.get("maximum_minutes"),
        },
    }


def _model_recommendations(raw: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for opinion in _list(raw.get("opinions")):
        item = _dict(opinion)
        if not item:
            continue
        rows.append(
            {
                "model_name": str(item.get("model_name") or ""),
                "role": str(item.get("role") or ""),
                "action": str(item.get("action") or "hold"),
                "confidence": _finite(item.get("confidence")),
                "suggested_position_fraction": _finite(item.get("position_size_pct")),
                "suggested_leverage": _finite(item.get("suggested_leverage")),
                "stop_loss_fraction": _finite(item.get("stop_loss_pct")),
                "take_profit_fraction": _finite(item.get("take_profit_pct")),
                "effective_weight": _finite(item.get("effective_weight")),
                "trade_plan": copy.deepcopy(_dict(item.get("trade_plan"))),
                "reasoning": str(item.get("reasoning") or "")[:500],
            }
        )
    return rows


def _evidence_summary(
    recommendations: list[dict[str, Any]],
    action: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    support: list[dict[str, Any]] = []
    oppose: list[dict[str, Any]] = []
    for item in recommendations:
        summary = {
            "model_name": item.get("model_name"),
            "role": item.get("role"),
            "action": item.get("action"),
            "confidence": item.get("confidence"),
            "effective_weight": item.get("effective_weight"),
            "reasoning": item.get("reasoning"),
        }
        (support if item.get("action") == action else oppose).append(summary)
    return support[:12], oppose[:12]


def _recommendation(decision: DecisionOutput, raw: dict[str, Any]) -> dict[str, Any]:
    side = _side(decision)
    action = decision.action.value
    recommendations = _model_recommendations(raw)
    support, oppose = _evidence_summary(recommendations, action)
    holding = _holding_plan(raw, side)
    return {
        "symbol": decision.symbol,
        "decision": _decision_kind(decision),
        "action": action,
        "side": side,
        "confidence": _finite(decision.confidence),
        "entry": {
            **_entry_plan(decision),
            "valid_for_seconds": holding.get("entry_valid_for_seconds"),
        },
        "holding": holding,
        "return_after_cost": _return_plan(raw, side),
        "loss_range": _loss_plan(decision, raw),
        "suggested_leverage": _finite(decision.suggested_leverage),
        "suggested_position_fraction": _finite(decision.position_size_pct),
        "exit": _exit_plan(decision, raw),
        "supporting_evidence": support,
        "opposing_evidence": oppose,
        "reasoning": str(decision.reasoning or "")[:1000],
    }


def _entry_plan_reasons(plan: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not str(plan.get("symbol") or "").strip():
        reasons.append("trade_plan_symbol_missing")
    if plan.get("side") not in {"long", "short"}:
        reasons.append("trade_plan_side_missing")
    entry = _dict(plan.get("entry"))
    if _positive(entry.get("price_reference")) is None:
        reasons.append("trade_plan_entry_price_missing")
    if _positive(entry.get("minimum_price")) is None or _positive(
        entry.get("maximum_price")
    ) is None:
        reasons.append("trade_plan_entry_range_missing")
    if _positive(entry.get("valid_for_seconds")) is None:
        reasons.append("trade_plan_validity_missing")
    holding = _dict(plan.get("holding"))
    if _positive(holding.get("target_minutes")) is None:
        reasons.append("trade_plan_holding_horizon_missing")
    if _positive(holding.get("maximum_minutes")) is None:
        reasons.append("trade_plan_maximum_holding_missing")
    returns = _dict(plan.get("return_after_cost"))
    if any(
        _finite(returns.get(key)) is None
        for key in ("expected_pct", "lower_bound_pct", "upper_bound_pct")
    ):
        reasons.append("trade_plan_fee_after_return_range_missing")
    losses = _dict(plan.get("loss_range"))
    if _positive(losses.get("normal_loss_pct")) is None:
        reasons.append("trade_plan_normal_loss_missing")
    if _positive(losses.get("extreme_loss_pct")) is None:
        reasons.append("trade_plan_extreme_loss_missing")
    if _positive(plan.get("suggested_leverage")) is None:
        reasons.append("trade_plan_leverage_missing")
    if _positive(plan.get("suggested_position_fraction")) is None:
        reasons.append("trade_plan_position_size_missing")
    exit_plan = _dict(plan.get("exit"))
    if _positive(exit_plan.get("stop_loss_fraction")) is None:
        reasons.append("trade_plan_stop_loss_missing")
    if _positive(exit_plan.get("take_profit_fraction")) is None:
        reasons.append("trade_plan_take_profit_missing")
    if not _dict(exit_plan.get("invalidation")) or not _dict(
        exit_plan.get("full_close")
    ):
        reasons.append("trade_plan_exit_conditions_missing")
    return reasons


def attach_initial_trade_recommendation(
    decision: DecisionOutput,
    *,
    analysis_type: str,
    execution_mode: str,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    raw = _dict(decision.raw_response)
    existing = _dict(raw.get("trade_recommendation_contract"))
    recommendation = _recommendation(decision, raw)
    reasons = (
        _entry_plan_reasons(recommendation)
        if decision.action in ENTRY_ACTIONS
        else []
    )
    contract = {
        **existing,
        "version": TRADE_RECOMMENDATION_CONTRACT_VERSION,
        "generated_at": (generated_at or datetime.now(UTC)).isoformat(),
        "analysis_type": str(analysis_type or "market"),
        "execution_mode": str(execution_mode or "paper"),
        "model_recommendations": _model_recommendations(raw),
        "unified_recommendation": recommendation,
        "unified_recommendation_complete": not reasons,
        "unified_recommendation_reasons": reasons,
        "current_recommendation": recommendation,
        "current_recommendation_complete": not reasons,
        "current_recommendation_reasons": reasons,
        "risk_adjustment": _dict(existing.get("risk_adjustment")) or {
            "status": "pending",
            "complete": False,
            "reasons": ["risk_adjustment_pending"],
        },
        "execution": _dict(existing.get("execution")) or {
            "status": "pending",
            "exchange_confirmed": False,
        },
    }
    raw["trade_recommendation_contract"] = contract
    decision.raw_response = raw
    return contract


def attach_risk_adjusted_trade_recommendation(
    decision: DecisionOutput,
    *,
    status: str,
    reason: str = "",
) -> dict[str, Any]:
    raw = _dict(decision.raw_response)
    existing = _dict(raw.get("trade_recommendation_contract"))
    if existing.get("version") != TRADE_RECOMMENDATION_CONTRACT_VERSION:
        existing = attach_initial_trade_recommendation(
            decision,
            analysis_type=str(raw.get("analysis_type") or "market"),
            execution_mode="paper",
        )
        raw = _dict(decision.raw_response)
    before = _dict(existing.get("unified_recommendation"))
    after = _recommendation(decision, raw)
    reasons = _entry_plan_reasons(after) if decision.action in ENTRY_ACTIONS else []
    adjustments = []
    for field in (
        "suggested_leverage",
        "suggested_position_fraction",
    ):
        if before.get(field) != after.get(field):
            adjustments.append(
                {
                    "field": field,
                    "before": before.get(field),
                    "after": after.get(field),
                }
            )
    for field in ("stop_loss_fraction", "take_profit_fraction"):
        before_value = _dict(before.get("exit")).get(field)
        after_value = _dict(after.get("exit")).get(field)
        if before_value != after_value:
            adjustments.append(
                {
                    "field": field,
                    "before": before_value,
                    "after": after_value,
                }
            )
    existing["risk_adjustment"] = {
        "status": str(status or "unknown"),
        "complete": not reasons,
        "reasons": reasons,
        "reason": str(reason or ""),
        "adjustments": adjustments,
        "adjusted_recommendation": after,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    existing["current_recommendation"] = after
    existing["current_recommendation_complete"] = not reasons
    existing["current_recommendation_reasons"] = reasons
    raw["trade_recommendation_contract"] = existing
    decision.raw_response = raw
    return existing


def paper_trade_recommendation_reasons(decision: DecisionOutput) -> list[str]:
    if not decision.is_entry:
        return []
    raw = _dict(decision.raw_response)
    contract = _dict(raw.get("trade_recommendation_contract"))
    if contract.get("version") != TRADE_RECOMMENDATION_CONTRACT_VERSION:
        return ["trade_recommendation_contract_missing"]
    risk = _dict(contract.get("risk_adjustment"))
    plan = _dict(risk.get("adjusted_recommendation"))
    reasons = list(dict.fromkeys(str(item) for item in _list(risk.get("reasons"))))
    if str(risk.get("status") or "") not in {"prepared", "approved", "passed"}:
        reasons.append("trade_plan_risk_adjustment_not_approved")
    if risk.get("complete") is not True:
        reasons.extend(_entry_plan_reasons(plan))
    return list(dict.fromkeys(reasons or _entry_plan_reasons(plan)))


def paper_trade_recommendation_reason_text(reasons: list[str]) -> str:
    """Return one user-facing paper-entry rejection reason."""

    labels = {
        "trade_recommendation_contract_missing": "完整交易方案尚未生成",
        "trade_plan_symbol_missing": "交易币种缺失",
        "trade_plan_side_missing": "交易方向缺失",
        "trade_plan_entry_price_missing": "入场参考价格缺失",
        "trade_plan_entry_range_missing": "入场价格范围缺失",
        "trade_plan_validity_missing": "入场建议有效时间缺失",
        "trade_plan_holding_horizon_missing": "预计持仓周期缺失",
        "trade_plan_maximum_holding_missing": "最长持仓时间缺失",
        "trade_plan_fee_after_return_range_missing": "扣除费用后的预期收益范围缺失",
        "trade_plan_normal_loss_missing": "正常亏损范围缺失",
        "trade_plan_extreme_loss_missing": "极端亏损范围缺失",
        "trade_plan_leverage_missing": "杠杆建议缺失",
        "trade_plan_position_size_missing": "仓位建议缺失",
        "trade_plan_stop_loss_missing": "止损方案缺失",
        "trade_plan_take_profit_missing": "止盈方案缺失",
        "trade_plan_exit_conditions_missing": "退出条件缺失",
        "trade_plan_risk_adjustment_not_approved": "风险调整后的交易方案尚未确认",
    }
    details = [labels.get(reason, reason) for reason in reasons]
    return "模拟盘完整交易方案未通过，未提交订单：" + "、".join(details)


def attach_trade_execution_result(
    decision: DecisionOutput,
    execution_result: Any | None,
    *,
    source: str,
    exchange_confirmed: bool,
    exit_progress: bool = False,
) -> dict[str, Any]:
    raw = _dict(decision.raw_response)
    contract = _dict(raw.get("trade_recommendation_contract"))
    if contract.get("version") != TRADE_RECOMMENDATION_CONTRACT_VERSION:
        contract = attach_initial_trade_recommendation(
            decision,
            analysis_type=str(raw.get("analysis_type") or "market"),
            execution_mode="paper",
        )
        raw = _dict(decision.raw_response)
    status = getattr(execution_result, "status", None)
    contract["execution"] = {
        "status": getattr(status, "value", status) or str(source or "unknown"),
        "source": str(source or ""),
        "exchange_confirmed": bool(exchange_confirmed),
        "exit_progress": bool(exit_progress),
        "order_id": getattr(execution_result, "order_id", None),
        "exchange_order_id": getattr(execution_result, "exchange_order_id", None),
        "filled_quantity": _finite(getattr(execution_result, "quantity", None)),
        "filled_price": _finite(getattr(execution_result, "price", None)),
        "fee_usdt": _finite(getattr(execution_result, "fee", None)),
        "realized_pnl_usdt": _finite(getattr(execution_result, "pnl", None)),
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    raw["trade_recommendation_contract"] = contract
    decision.raw_response = raw
    return contract


def trade_recommendation_snapshot(raw_response: Any) -> dict[str, Any]:
    raw = _dict(raw_response)
    return copy.deepcopy(_dict(raw.get("trade_recommendation_contract")))
