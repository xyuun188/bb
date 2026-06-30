"""Profit-First v3 canonical trade-plan helpers.

This module is intentionally read-only: it normalizes existing decision payloads
into one audit contract, but it does not submit orders or mutate live sizing.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

PLAN_VERSION = "profit-first-v3.1"

ENTRY_ACTIONS = {"long", "short", "buy", "sell", "open_long", "open_short"}
EXIT_ACTIONS = {"close_long", "close_short", "sell_long", "buy_short"}
NO_ENTRY_CATEGORIES = {
    "profit_insufficient",
    "evidence_insufficient",
    "risk_gate_blocked",
    "model_disagreement",
    "budget_insufficient",
    "position_capacity_occupied",
    "same_side_crowded",
    "okx_unavailable_or_rejected",
    "market_data_incomplete",
    "phase3_model_unavailable",
    "shadow_only_missing_plan_fields",
    "recent_realized_edge_negative",
}
LOSING_EXIT_CATEGORIES = {
    "entry_wrong_direction",
    "entry_late",
    "stop_too_tight",
    "position_too_small_fee_drag",
    "hold_too_short",
    "trend_reversal",
    "model_false_positive",
    "server_profit_overestimated",
    "timeseries_false_signal",
    "sentiment_false_signal",
    "okx_slippage_or_execution",
    "exit_too_early",
    "exit_too_late",
    "capital_release_forced_loss",
    "unknown_requires_review",
}


@dataclass(slots=True)
class ProfitFirstExitPlan:
    exit_plan_id: str
    stop_loss_pct: float | None
    take_profit_pct: float | None
    trailing_profit_trigger_pct: float | None
    profit_drawdown_exit_pct: float | None
    partial_exit_plan: list[dict[str, Any]] = field(default_factory=list)
    full_exit_plan: dict[str, Any] = field(default_factory=dict)
    do_not_close_conditions: list[str] = field(default_factory=list)
    max_hold_minutes: float | None = None
    invalidation_price: float | None = None
    generated_from_defaults: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProfitFirstTradePlan:
    plan_version: str
    symbol: str
    side: str
    action: str
    analysis_type: str
    strategy_profile_id: str
    model_sources: list[str]
    independent_source_count: int
    expected_gross_return_pct: float | None
    expected_fee_pct: float | None
    expected_slippage_pct: float | None
    expected_net_return_pct: float | None
    expected_profit_usdt: float | None
    profit_quality_ratio: float | None
    reward_risk_ratio: float | None
    loss_probability: float | None
    expected_loss_usdt: float | None
    tail_loss_probability: float | None
    tail_loss_usdt: float | None
    max_stop_loss_usdt: float | None
    portfolio_side_pressure: float | None
    same_symbol_side_edge: float | None
    recent_realized_edge: float | None
    expected_hold_minutes: float | None
    max_hold_minutes: float | None
    entry_price_reference: float | None
    invalidation_price: float | None
    decision_lane: str
    position_size_pct: float | None
    leverage: float | None
    promotion_reasons: list[str]
    block_or_downgrade_reasons: list[str]
    exit_plan_id: str
    stop_loss_pct: float | None
    take_profit_pct: float | None
    trailing_profit_trigger_pct: float | None
    profit_drawdown_exit_pct: float | None
    partial_exit_plan: list[dict[str, Any]]
    full_exit_plan: dict[str, Any]
    do_not_close_conditions: list[str]
    profit_first_score: float
    missing_required_fields: list[str]
    shadow_only_reason: str
    is_complete_for_real_trade: bool
    model_contributions: list[dict[str, Any]] = field(default_factory=list)
    source_field_map: dict[str, str] = field(default_factory=dict)
    no_entry_reason: str = ""
    losing_exit_attribution: str = ""
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_profit_first_trade_plan(
    decision: Any,
    *,
    analysis_type: str | None = None,
    now: datetime | None = None,
) -> ProfitFirstTradePlan:
    """Build a Profit-First plan from a DecisionOutput or ORM-like row."""

    raw = _decision_raw(decision)
    action = _action_value(_row_get(decision, "action"))
    side = _entry_side(action) or _side_from_raw(raw)
    symbol = str(_row_get(decision, "symbol") or raw.get("symbol") or "").strip()
    resolved_analysis_type = (
        str(analysis_type or raw.get("analysis_type") or _row_get(decision, "analysis_type") or "")
        .lower()
        .strip()
        or ("position" if action in EXIT_ACTIONS else "market")
    )
    strategy_profile_id = _strategy_profile_id(raw)
    position_size_pct = _first_float(
        _row_get(decision, "position_size_pct"),
        _safe_dict(raw.get("profit_risk_sizing")).get("position_size_pct"),
    )
    leverage = _first_float(_row_get(decision, "suggested_leverage"), raw.get("leverage"))
    stop_loss_pct = _first_float(
        _row_get(decision, "stop_loss_pct"),
        _safe_dict(raw.get("exit_plan")).get("stop_loss_pct"),
        _safe_dict(raw.get("profit_first_exit_plan")).get("stop_loss_pct"),
    )
    take_profit_pct = _first_float(
        _row_get(decision, "take_profit_pct"),
        _safe_dict(raw.get("exit_plan")).get("take_profit_pct"),
        _safe_dict(raw.get("profit_first_exit_plan")).get("take_profit_pct"),
    )
    metrics = _entry_metrics(raw, side)
    opportunity = _safe_dict(raw.get("opportunity_score"))
    sizing = _safe_dict(raw.get("profit_risk_sizing"))
    execution_cost = _safe_dict(opportunity.get("execution_cost"))

    expected_fee_pct = _first_float(
        opportunity.get("fee_pct"),
        execution_cost.get("fee_pct"),
        _breakdown_component(opportunity, "fee"),
    )
    expected_slippage_pct = _first_float(
        opportunity.get("slippage_pct"),
        execution_cost.get("slippage_pct"),
        _breakdown_component(opportunity, "slippage"),
    )
    expected_net = metrics["expected_net_return_pct"]
    expected_gross = _first_float(
        opportunity.get("expected_gross_return_pct"),
        opportunity.get("expected_return_pct"),
    )
    if expected_gross is None and expected_net is not None:
        expected_gross = expected_net + abs(expected_fee_pct or 0.0) + abs(
            expected_slippage_pct or 0.0
        )
    notional = _first_float(
        sizing.get("final_notional_usdt"),
        sizing.get("target_min_notional_usdt"),
        _safe_dict(raw.get("execution_parameters")).get("notional_usdt"),
    )
    expected_profit_usdt = _first_float(sizing.get("expected_profit_usdt"))
    if expected_profit_usdt is None and notional is not None and expected_net is not None:
        expected_profit_usdt = notional * max(expected_net, 0.0) / 100.0

    expected_loss_pct = _first_float(opportunity.get("expected_loss_pct"))
    loss_probability = metrics["loss_probability"]
    if expected_loss_pct is None and stop_loss_pct is not None and loss_probability is not None:
        expected_loss_pct = abs(stop_loss_pct) * 100.0 * max(loss_probability, 0.0)
    planned_stop_loss_usdt = _first_float(sizing.get("planned_stop_loss_usdt"))
    max_stop_loss_usdt = _first_float(
        sizing.get("max_stop_loss_usdt"),
        opportunity.get("max_entry_stop_loss_usdt"),
        planned_stop_loss_usdt,
    )
    expected_loss_usdt = _first_float(sizing.get("expected_loss_usdt"))
    if expected_loss_usdt is None:
        if notional is not None and expected_loss_pct is not None:
            expected_loss_usdt = notional * expected_loss_pct / 100.0
        elif max_stop_loss_usdt is not None and loss_probability is not None:
            expected_loss_usdt = max_stop_loss_usdt * max(loss_probability, 0.0)

    tail_loss_probability = _first_float(
        opportunity.get("tail_loss_probability"),
        metrics["tail_risk_score"],
    )
    tail_loss_usdt = _first_float(
        sizing.get("tail_loss_usdt"),
        sizing.get("capped_stop_loss_usdt"),
        planned_stop_loss_usdt,
        max_stop_loss_usdt,
    )
    reward_risk_ratio = _first_float(opportunity.get("reward_risk_ratio"))
    if reward_risk_ratio is None and stop_loss_pct and take_profit_pct:
        reward_risk_ratio = abs(take_profit_pct) / max(abs(stop_loss_pct), 1e-9)

    entry_price_reference = _entry_price_reference(decision, raw)
    max_hold_minutes = _first_float(
        raw.get("max_hold_minutes"),
        _safe_dict(raw.get("exit_plan")).get("max_hold_minutes"),
        _safe_dict(raw.get("profit_first_exit_plan")).get("max_hold_minutes"),
        180.0,
    )
    expected_hold_minutes = _first_float(
        raw.get("expected_hold_minutes"),
        opportunity.get("expected_hold_minutes"),
        _safe_dict(raw.get("timing")).get("expected_hold_minutes"),
        min(max_hold_minutes or 180.0, 60.0),
    )
    invalidation_price = _first_float(
        raw.get("invalidation_price"),
        _safe_dict(raw.get("exit_plan")).get("invalidation_price"),
        _safe_dict(raw.get("profit_first_exit_plan")).get("invalidation_price"),
    )
    if invalidation_price is None and entry_price_reference and stop_loss_pct and side:
        invalidation_price = (
            entry_price_reference * (1.0 - abs(stop_loss_pct))
            if side == "long"
            else entry_price_reference * (1.0 + abs(stop_loss_pct))
        )

    exit_plan = _build_exit_plan(
        raw=raw,
        symbol=symbol,
        side=side,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        max_hold_minutes=max_hold_minutes,
        invalidation_price=invalidation_price,
    )
    sources, source_field_map, model_contributions = extract_model_sources(raw, decision, side)
    missing = _missing_required_fields(
        {
            "symbol": symbol,
            "side": side,
            "analysis_type": resolved_analysis_type,
            "strategy_profile_id": strategy_profile_id,
            "model_sources": sources,
            "expected_net_return_pct": expected_net,
            "expected_fee_pct": expected_fee_pct,
            "expected_slippage_pct": expected_slippage_pct,
            "expected_profit_usdt": expected_profit_usdt,
            "profit_quality_ratio": metrics["profit_quality_ratio"],
            "reward_risk_ratio": reward_risk_ratio,
            "loss_probability": loss_probability,
            "expected_loss_usdt": expected_loss_usdt,
            "tail_loss_probability": tail_loss_probability,
            "tail_loss_usdt": tail_loss_usdt,
            "max_stop_loss_usdt": max_stop_loss_usdt,
            "expected_hold_minutes": expected_hold_minutes,
            "max_hold_minutes": max_hold_minutes,
            "entry_price_reference": entry_price_reference,
            "invalidation_price": invalidation_price,
            "position_size_pct": position_size_pct,
            "leverage": leverage,
            "exit_plan_id": exit_plan.exit_plan_id,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "trailing_profit_trigger_pct": exit_plan.trailing_profit_trigger_pct,
            "profit_drawdown_exit_pct": exit_plan.profit_drawdown_exit_pct,
            "partial_exit_plan": exit_plan.partial_exit_plan,
            "full_exit_plan": exit_plan.full_exit_plan,
            "do_not_close_conditions": exit_plan.do_not_close_conditions,
        },
        is_entry=action in {"long", "short"},
    )
    score = profit_first_score(
        expected_net_return_pct=expected_net,
        expected_profit_usdt=expected_profit_usdt,
        profit_quality_ratio=metrics["profit_quality_ratio"],
        reward_risk_ratio=reward_risk_ratio,
        loss_probability=loss_probability,
        tail_loss_probability=tail_loss_probability,
        independent_source_count=len(sources),
        recent_realized_edge=_recent_realized_edge(opportunity),
    )
    lane, promotion_reasons, downgrade_reasons = classify_decision_lane(
        expected_net_return_pct=expected_net,
        expected_profit_usdt=expected_profit_usdt,
        profit_quality_ratio=metrics["profit_quality_ratio"],
        reward_risk_ratio=reward_risk_ratio,
        loss_probability=loss_probability,
        tail_loss_probability=tail_loss_probability,
        independent_source_count=len(sources),
        missing_required_fields=missing,
        high_risk_review=_safe_dict(raw.get("high_risk_review")),
        recent_realized_edge=_recent_realized_edge(opportunity),
        is_entry=action in {"long", "short"},
    )
    if expected_net is not None and expected_net <= 0:
        downgrade_reasons.append("expected_net_return_not_positive")
    shadow_reason = ""
    if lane == "shadow_only":
        shadow_reason = downgrade_reasons[0] if downgrade_reasons else "shadow_only"
    complete = bool(action in {"long", "short"} and lane != "shadow_only" and not missing)
    no_entry_reason = ""
    if lane == "shadow_only" and action in {"long", "short", "hold"}:
        no_entry_reason = normalize_no_entry_reason(raw, plan_missing_fields=missing)

    generated_at = (now or datetime.now(UTC)).isoformat()
    return ProfitFirstTradePlan(
        plan_version=PLAN_VERSION,
        symbol=symbol,
        side=side,
        action=action,
        analysis_type=resolved_analysis_type,
        strategy_profile_id=strategy_profile_id,
        model_sources=sources,
        independent_source_count=len(sources),
        expected_gross_return_pct=_round_or_none(expected_gross),
        expected_fee_pct=_round_or_none(expected_fee_pct),
        expected_slippage_pct=_round_or_none(expected_slippage_pct),
        expected_net_return_pct=_round_or_none(expected_net),
        expected_profit_usdt=_round_or_none(expected_profit_usdt),
        profit_quality_ratio=_round_or_none(metrics["profit_quality_ratio"]),
        reward_risk_ratio=_round_or_none(reward_risk_ratio),
        loss_probability=_round_or_none(loss_probability),
        expected_loss_usdt=_round_or_none(expected_loss_usdt),
        tail_loss_probability=_round_or_none(tail_loss_probability),
        tail_loss_usdt=_round_or_none(tail_loss_usdt),
        max_stop_loss_usdt=_round_or_none(max_stop_loss_usdt),
        portfolio_side_pressure=_round_or_none(_portfolio_side_pressure(opportunity, raw)),
        same_symbol_side_edge=_round_or_none(_same_symbol_side_edge(opportunity)),
        recent_realized_edge=_round_or_none(_recent_realized_edge(opportunity)),
        expected_hold_minutes=_round_or_none(expected_hold_minutes),
        max_hold_minutes=_round_or_none(max_hold_minutes),
        entry_price_reference=_round_or_none(entry_price_reference),
        invalidation_price=_round_or_none(invalidation_price),
        decision_lane=lane,
        position_size_pct=_round_or_none(position_size_pct),
        leverage=_round_or_none(leverage),
        promotion_reasons=promotion_reasons,
        block_or_downgrade_reasons=list(dict.fromkeys(downgrade_reasons)),
        exit_plan_id=exit_plan.exit_plan_id,
        stop_loss_pct=_round_or_none(stop_loss_pct),
        take_profit_pct=_round_or_none(take_profit_pct),
        trailing_profit_trigger_pct=_round_or_none(exit_plan.trailing_profit_trigger_pct),
        profit_drawdown_exit_pct=_round_or_none(exit_plan.profit_drawdown_exit_pct),
        partial_exit_plan=exit_plan.partial_exit_plan,
        full_exit_plan=exit_plan.full_exit_plan,
        do_not_close_conditions=exit_plan.do_not_close_conditions,
        profit_first_score=round(score, 6),
        missing_required_fields=missing,
        shadow_only_reason=shadow_reason,
        is_complete_for_real_trade=complete,
        model_contributions=model_contributions,
        source_field_map=source_field_map,
        no_entry_reason=no_entry_reason,
        generated_at=generated_at,
    )


def attach_profit_first_trade_plan(
    decision: Any,
    *,
    analysis_type: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Attach the canonical plan to a decision raw payload and return it."""

    raw = _decision_raw(decision)
    plan = build_profit_first_trade_plan(decision, analysis_type=analysis_type, now=now)
    plan_payload = plan.to_dict()
    raw["profit_first_trade_plan"] = plan_payload
    raw["profit_first_exit_plan"] = {
        "exit_plan_id": plan.exit_plan_id,
        "stop_loss_pct": plan.stop_loss_pct,
        "take_profit_pct": plan.take_profit_pct,
        "trailing_profit_trigger_pct": plan.trailing_profit_trigger_pct,
        "profit_drawdown_exit_pct": plan.profit_drawdown_exit_pct,
        "partial_exit_plan": plan.partial_exit_plan,
        "full_exit_plan": plan.full_exit_plan,
        "do_not_close_conditions": plan.do_not_close_conditions,
        "max_hold_minutes": plan.max_hold_minutes,
        "invalidation_price": plan.invalidation_price,
        "generated_from_trade_plan": True,
    }
    raw["profit_first_entry_exit_binding"] = {
        "exit_plan_id": plan.exit_plan_id,
        "required_for_real_entry": True,
        "exit_decisions_must_reference_plan": True,
        "source": "attach_profit_first_trade_plan",
    }
    _set_decision_raw(decision, raw)
    return raw


def classify_decision_lane(
    *,
    expected_net_return_pct: float | None,
    expected_profit_usdt: float | None,
    profit_quality_ratio: float | None,
    reward_risk_ratio: float | None,
    loss_probability: float | None,
    tail_loss_probability: float | None,
    independent_source_count: int,
    missing_required_fields: list[str] | None = None,
    high_risk_review: dict[str, Any] | None = None,
    recent_realized_edge: float | None = None,
    is_entry: bool = True,
) -> tuple[str, list[str], list[str]]:
    """Return lane, promotion reasons, and downgrade reasons."""

    promotion: list[str] = []
    downgrade: list[str] = []
    missing = list(missing_required_fields or [])
    if not is_entry:
        return "shadow_only", [], ["not_an_entry_candidate"]
    if missing:
        return "shadow_only", [], ["missing_required_fields:" + ",".join(missing[:8])]
    expected_net = expected_net_return_pct if expected_net_return_pct is not None else -999.0
    quality = profit_quality_ratio if profit_quality_ratio is not None else 0.0
    rr = reward_risk_ratio if reward_risk_ratio is not None else 0.0
    loss_probability_value = loss_probability if loss_probability is not None else 1.0
    tail = tail_loss_probability if tail_loss_probability is not None else 1.0
    realized_edge = recent_realized_edge if recent_realized_edge is not None else 0.0

    if expected_net <= 0:
        return "shadow_only", [], ["expected_net_return_not_positive"]
    if independent_source_count < 2:
        return "shadow_only", [], ["independent_source_count_below_tiny_probe"]
    if loss_probability_value > 0.65:
        return "shadow_only", [], ["loss_probability_too_high"]
    if tail > 0.98:
        return "shadow_only", [], ["tail_loss_probability_too_high"]
    if realized_edge < -0.08:
        return "shadow_only", [], ["recent_realized_edge_negative"]

    lane = "tiny_probe"
    promotion.append("positive_expected_net")
    promotion.append("minimum_independent_sources_met")

    if (
        expected_net >= 0.35
        and quality >= 0.45
        and rr >= 0.8
        and loss_probability_value <= 0.52
        and tail <= 0.88
        and independent_source_count >= 3
    ):
        lane = "validated_probe"
        promotion.append("validated_profit_quality")
    if (
        expected_net >= 0.75
        and (expected_profit_usdt or 0.0) > 0
        and quality >= 0.90
        and rr >= 1.10
        and loss_probability_value <= 0.45
        and tail <= 0.80
        and independent_source_count >= 4
        and realized_edge >= 0.0
    ):
        lane = "meaningful_entry"
        promotion.append("meaningful_size_quality_met")
    if (
        expected_net >= 1.20
        and quality >= 1.40
        and rr >= 1.50
        and loss_probability_value <= 0.38
        and tail <= 0.70
        and independent_source_count >= 5
        and bool(_safe_dict(high_risk_review).get("approved"))
        and bool(_safe_dict(high_risk_review).get("profit_first_allow_high_conviction"))
    ):
        lane = "high_conviction"
        promotion.append("high_risk_review_approved")
    elif expected_net >= 1.20 and independent_source_count >= 5:
        downgrade.append("high_conviction_disabled_until_observation_gates_pass")
    return lane, list(dict.fromkeys(promotion)), list(dict.fromkeys(downgrade))


def profit_first_score(
    *,
    expected_net_return_pct: float | None,
    expected_profit_usdt: float | None,
    profit_quality_ratio: float | None,
    reward_risk_ratio: float | None,
    loss_probability: float | None,
    tail_loss_probability: float | None,
    independent_source_count: int,
    recent_realized_edge: float | None,
) -> float:
    expected_net = expected_net_return_pct or 0.0
    expected_profit = min(max((expected_profit_usdt or 0.0) / 5.0, -1.0), 2.0)
    quality = profit_quality_ratio or 0.0
    rr = reward_risk_ratio or 0.0
    loss = loss_probability if loss_probability is not None else 1.0
    tail = tail_loss_probability if tail_loss_probability is not None else 1.0
    source_bonus = min(max(independent_source_count, 0), 6) * 0.12
    realized = recent_realized_edge or 0.0
    return (
        expected_net * 1.8
        + expected_profit * 0.45
        + quality * 0.85
        + rr * 0.30
        + source_bonus
        + realized * 0.75
        - loss * 1.15
        - tail * 0.55
    )


def normalize_no_entry_reason(
    raw: dict[str, Any] | None,
    *,
    execution_reason: str | None = None,
    plan_missing_fields: list[str] | None = None,
) -> str:
    """Map arbitrary no-entry evidence into the canonical taxonomy."""

    payload = _safe_dict(raw)
    text = " ".join(
        str(item or "")
        for item in (
            execution_reason,
            payload.get("reason"),
            payload.get("skip_reason"),
            payload.get("rejection_reason"),
            _safe_dict(payload.get("opportunity_score")).get("dynamic_score_reason"),
            _safe_dict(payload.get("entry_filters")).get("reason"),
        )
    ).lower()
    opportunity = _safe_dict(payload.get("opportunity_score"))
    plan = _safe_dict(payload.get("profit_first_trade_plan"))
    missing = plan_missing_fields or _safe_list(plan.get("missing_required_fields"))
    if missing:
        return "shadow_only_missing_plan_fields"
    if "profit_first_probe_loss_brake" in text or "probe-loss brake" in text:
        return "recent_realized_edge_negative"
    if "entry_evidence_shadow_only" in text or "entry_evidence_wait" in text:
        return "evidence_insufficient"
    if "entry_capacity" in text:
        return "position_capacity_occupied"
    if "high_risk_review" in text or "entry_opportunity_gate" in text:
        return "risk_gate_blocked"
    if _safe_float_or_none(opportunity.get("expected_net_return_pct")) is not None:
        if _safe_float_or_none(opportunity.get("expected_net_return_pct")) <= 0:
            return "profit_insufficient"
    if "okx" in text or "exchange" in text or "rejected" in text or "511" in text:
        return "okx_unavailable_or_rejected"
    if "crowd" in text or "same_side" in text or "单边" in text or "拥挤" in text:
        return "same_side_crowded"
    if "capacity" in text or "max open" in text or "持仓" in text or "占用" in text:
        return "position_capacity_occupied"
    if "budget" in text or "balance" in text or "margin" in text or "资金" in text:
        return "budget_insufficient"
    if "disagreement" in text or "conflict" in text or "分歧" in text:
        return "model_disagreement"
    if "market data" in text or "ticker" in text or "kline" in text or "行情" in text:
        return "market_data_incomplete"
    if "phase3" in text or "model unavailable" in text or "model server" in text:
        return "phase3_model_unavailable"
    if "risk" in text or "tail" in text or "风控" in text or "风险" in text:
        return "risk_gate_blocked"
    if (
        _safe_float_or_none(opportunity.get("recent_realized_edge")) is not None
        and _safe_float_or_none(opportunity.get("recent_realized_edge")) < 0
    ):
        return "recent_realized_edge_negative"
    evidence = _safe_dict(opportunity.get("evidence_score"))
    if evidence.get("hard_block") or str(evidence.get("tier") or "") in {
        "blocked",
        "weak_conflict_probe",
        "degraded_missing_probe",
    }:
        return "evidence_insufficient"
    if "profit" in text or "收益" in text or "期望" in text or "net" in text:
        return "profit_insufficient"
    return "evidence_insufficient"


def normalize_losing_exit_attribution(
    position_or_record: Any,
    *,
    entry_raw: dict[str, Any] | None = None,
    close_raw: dict[str, Any] | None = None,
    shadow: dict[str, Any] | None = None,
) -> str:
    """Classify a losing close into the canonical Profit-First taxonomy."""

    pnl = _first_float(_row_get(position_or_record, "realized_pnl"))
    if pnl is not None and pnl >= 0:
        return ""
    entry_payload = _safe_dict(entry_raw or _row_get(position_or_record, "entry_raw"))
    close_payload = _safe_dict(close_raw or _row_get(position_or_record, "close_raw"))
    side = str(_row_get(position_or_record, "side") or "").lower()
    opened = _parse_datetime(_row_get(position_or_record, "created_at") or _row_get(position_or_record, "entry_at"))
    closed = _parse_datetime(_row_get(position_or_record, "closed_at"))
    hold_minutes = _first_float(_row_get(position_or_record, "hold_minutes"))
    if hold_minutes is None and opened and closed:
        hold_minutes = max((closed - opened).total_seconds() / 60.0, 0.0)
    text = " ".join(
        str(item or "")
        for item in (
            _row_get(position_or_record, "main_reason"),
            _row_get(position_or_record, "reason"),
            close_payload.get("exit_intent"),
            _safe_dict(close_payload.get("close_evidence")).get("reason"),
            close_payload.get("execution_reason"),
        )
    ).lower()
    shadow_side = str(_safe_dict(shadow or _row_get(position_or_record, "shadow")).get("best_action") or "")
    if shadow_side in {"long", "short"} and side in {"long", "short"} and shadow_side != side:
        return "entry_wrong_direction"
    if "okx" in text or "slippage" in text or "滑点" in text or "execution" in text:
        return "okx_slippage_or_execution"
    if "capital" in text or "release" in text or "释放" in text or "rotation" in text:
        return "capital_release_forced_loss"
    if "trend reversal" in text or "反转" in text:
        return "trend_reversal"
    if "too late" in text or "late exit" in text:
        return "exit_too_late"
    if "too early" in text or "early exit" in text:
        return "exit_too_early"
    opportunity = _safe_dict(entry_payload.get("opportunity_score"))
    sizing = _safe_dict(entry_payload.get("profit_risk_sizing"))
    size = _first_float(
        _row_get(position_or_record, "position_size_pct"),
        sizing.get("position_size_pct"),
        _safe_dict(entry_payload.get("profit_first_trade_plan")).get("position_size_pct"),
    )
    notional = _first_float(
        sizing.get("final_notional_usdt"),
        _row_get(position_or_record, "notional_usdt"),
    )
    if (size is not None and size <= 0.015) or (notional is not None and notional < 15.0):
        return "position_too_small_fee_drag"
    if hold_minutes is not None and hold_minutes <= 5.0:
        return "hold_too_short"
    if "stop" in text or "止损" in text:
        return "stop_too_tight" if hold_minutes is None or hold_minutes <= 30 else "trend_reversal"
    if bool(opportunity.get("local_profit_aligned")):
        return "server_profit_overestimated"
    if bool(opportunity.get("timeseries_aligned")):
        return "timeseries_false_signal"
    evidence = _safe_dict(opportunity.get("evidence_score"))
    for component in _safe_list(evidence.get("components")):
        if _safe_dict(component).get("source") == "sentiment" and _safe_dict(component).get(
            "status"
        ) == "aligned":
            return "sentiment_false_signal"
    if _safe_float_or_none(opportunity.get("expected_net_return_pct")) is not None:
        if _safe_float_or_none(opportunity.get("expected_net_return_pct")) > 0:
            return "model_false_positive"
    return "unknown_requires_review"


def summarize_probe_loop_health(
    closed_positions: list[Any],
    *,
    now: datetime | None = None,
    window_hours: float = 8.0,
) -> dict[str, Any]:
    """Summarize whether tiny/probe trades are stuck in an all-loss loop."""

    cutoff = (now or datetime.now(UTC)) - timedelta(hours=max(float(window_hours), 0.1))
    probe_rows: list[Any] = []
    for row in closed_positions:
        closed_at = _parse_datetime(_row_get(row, "closed_at"))
        if closed_at and closed_at < cutoff:
            continue
        if _is_probe_row(row):
            probe_rows.append(row)
    pnl_values = [_first_float(_row_get(row, "realized_pnl"), 0.0) or 0.0 for row in probe_rows]
    losses = [pnl for pnl in pnl_values if pnl < 0]
    wins = [pnl for pnl in pnl_values if pnl > 0]
    fast_losses = [
        row
        for row in probe_rows
        if (_first_float(_row_get(row, "realized_pnl"), 0.0) or 0.0) < 0
        and (_hold_minutes(row) or 9999.0) <= 15.0
    ]
    brake_active = bool(len(probe_rows) >= 2 and len(losses) == len(probe_rows))
    return {
        "window_hours": round(float(window_hours), 3),
        "probe_closed_count": len(probe_rows),
        "probe_win_count": len(wins),
        "probe_loss_count": len(losses),
        "probe_total_realized_pnl": round(sum(pnl_values), 6),
        "probe_profit_factor": round(sum(wins) / abs(sum(losses)), 6)
        if losses
        else (999.0 if wins else 0.0),
        "fast_probe_loss_count": len(fast_losses),
        "all_recent_probes_losing": brake_active,
        "recommended_action": "shadow_new_tiny_probes_until_validated_upgrade"
        if brake_active
        else "probe_loop_observe",
    }


def summarize_model_strategy_realized_pnl(closed_positions: list[Any]) -> dict[str, Any]:
    """Build a small realized-PnL leaderboard skeleton by key dimensions."""

    buckets: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in closed_positions:
        raw = _safe_dict(_row_get(row, "entry_raw") or _row_get(row, "raw_llm_response"))
        plan = _safe_dict(raw.get("profit_first_trade_plan"))
        key = (
            str(_row_get(row, "model_name") or "unknown"),
            str(plan.get("strategy_profile_id") or _strategy_profile_id(raw) or "unknown"),
            str(_row_get(row, "symbol") or "unknown"),
            str(_row_get(row, "side") or "unknown"),
            str(plan.get("decision_lane") or "unknown"),
        )
        bucket = buckets.setdefault(
            key,
            {
                "model_name": key[0],
                "strategy_profile_id": key[1],
                "symbol": key[2],
                "side": key[3],
                "decision_lane": key[4],
                "count": 0,
                "wins": 0,
                "losses": 0,
                "realized_net_pnl": 0.0,
                "profit": 0.0,
                "loss": 0.0,
            },
        )
        pnl = _first_float(_row_get(row, "realized_pnl"), 0.0) or 0.0
        bucket["count"] += 1
        bucket["realized_net_pnl"] += pnl
        if pnl >= 0:
            bucket["wins"] += 1
            bucket["profit"] += pnl
        else:
            bucket["losses"] += 1
            bucket["loss"] += abs(pnl)
    rows = []
    for bucket in buckets.values():
        count = max(int(bucket["count"]), 1)
        loss = float(bucket["loss"])
        rows.append(
            {
                **bucket,
                "realized_net_pnl": round(float(bucket["realized_net_pnl"]), 6),
                "profit": round(float(bucket["profit"]), 6),
                "loss": round(loss, 6),
                "win_rate": round(int(bucket["wins"]) / count, 6),
                "profit_factor": round(float(bucket["profit"]) / loss, 6)
                if loss > 0
                else (999.0 if bucket["profit"] > 0 else 0.0),
            }
        )
    rows.sort(key=lambda item: item["realized_net_pnl"], reverse=True)
    return {"rows": rows, "count": len(rows)}


def extract_model_sources(
    raw: dict[str, Any],
    decision: Any,
    side: str,
) -> tuple[list[str], dict[str, str], list[dict[str, Any]]]:
    opportunity = _safe_dict(raw.get("opportunity_score"))
    evidence = _safe_dict(opportunity.get("evidence_score"))
    sources: list[str] = []
    source_field_map: dict[str, str] = {}

    def add(source: str, field_path: str) -> None:
        if source and source not in sources:
            sources.append(source)
            source_field_map[source] = field_path

    if str(_row_get(decision, "model_name") or "").strip():
        add("decision_llm", "decision.model_name")
    if opportunity.get("ml_aligned") or raw.get("ml_signal"):
        add("local_ml", "opportunity_score.ml_aligned")
    if opportunity.get("local_profit_aligned") or opportunity.get("server_profit_expected_return_pct"):
        add("server_profit", "opportunity_score.server_profit_expected_return_pct")
    if opportunity.get("timeseries_aligned") or opportunity.get("timeseries_expected_return_pct"):
        add("timeseries", "opportunity_score.timeseries_expected_return_pct")
    if opportunity.get("expert_aligned"):
        add("expert_alignment", "opportunity_score.expert_aligned")
    for component in _safe_list(evidence.get("components")):
        row = _safe_dict(component)
        if row.get("status") != "aligned":
            continue
        source = str(row.get("source") or "")
        if source == "sentiment":
            add("sentiment", "opportunity_score.evidence_score.components.sentiment")
        elif source == "shadow_memory":
            add("shadow_memory", "opportunity_score.evidence_score.components.shadow_memory")
    if _safe_dict(raw.get("high_risk_review")).get("approved"):
        add("high_risk_review", "high_risk_review.approved")
    contributions = [
        {
            "source": source,
            "field_path": source_field_map[source],
            "side": side,
            "valid": True,
            "independent": True,
        }
        for source in sources
    ]
    return sources, source_field_map, contributions


def _build_exit_plan(
    *,
    raw: dict[str, Any],
    symbol: str,
    side: str,
    stop_loss_pct: float | None,
    take_profit_pct: float | None,
    max_hold_minutes: float | None,
    invalidation_price: float | None,
) -> ProfitFirstExitPlan:
    direct = _safe_dict(raw.get("profit_first_exit_plan")) or _safe_dict(raw.get("exit_plan"))
    trailing = _first_float(
        direct.get("trailing_profit_trigger_pct"),
        raw.get("trailing_profit_trigger_pct"),
        (take_profit_pct * 0.6) if take_profit_pct is not None else None,
    )
    drawdown = _first_float(
        direct.get("profit_drawdown_exit_pct"),
        raw.get("profit_drawdown_exit_pct"),
        max(abs(stop_loss_pct or 0.0) * 0.5, abs(take_profit_pct or 0.0) * 0.35)
        if stop_loss_pct is not None or take_profit_pct is not None
        else None,
    )
    partial = _safe_list(direct.get("partial_exit_plan"))
    generated_from_defaults = False
    if not partial and take_profit_pct is not None and abs(take_profit_pct) > 0:
        partial = [
            {
                "trigger_return_pct": round(max(take_profit_pct * 0.55, take_profit_pct / 2), 6),
                "fraction": 0.5,
                "reason": "lock partial profit before full target",
            }
        ]
        generated_from_defaults = True
    full = _safe_dict(direct.get("full_exit_plan"))
    if not full and take_profit_pct is not None and abs(take_profit_pct) > 0:
        full = {
            "take_profit_pct": round(take_profit_pct, 6),
            "max_hold_minutes": round(max_hold_minutes or 0.0, 6),
            "reason": "full exit when target, invalidation, or max hold rule is met",
        }
        generated_from_defaults = True
    do_not_close = [
        str(item)
        for item in _safe_list(direct.get("do_not_close_conditions"))
        if str(item or "").strip()
    ]
    if not do_not_close:
        do_not_close = [
            "do_not_close_small_loss_without_hard_risk_or_better_replacement",
            "do_not_close_profit_trend_before_drawdown_rule",
        ]
        generated_from_defaults = True
    exit_plan_id = str(direct.get("exit_plan_id") or "").strip()
    if (
        not exit_plan_id
        and symbol
        and side
        and stop_loss_pct is not None
        and abs(stop_loss_pct) > 0
        and take_profit_pct is not None
        and abs(take_profit_pct) > 0
    ):
        signature = hashlib.sha1(
            f"{PLAN_VERSION}:{symbol}:{side}:{stop_loss_pct}:{take_profit_pct}:{max_hold_minutes}".encode(
                "utf-8"
            )
        ).hexdigest()[:16]
        exit_plan_id = f"pfep-{signature}"
        generated_from_defaults = True
    return ProfitFirstExitPlan(
        exit_plan_id=exit_plan_id,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        trailing_profit_trigger_pct=trailing,
        profit_drawdown_exit_pct=drawdown,
        partial_exit_plan=partial,
        full_exit_plan=full,
        do_not_close_conditions=do_not_close,
        max_hold_minutes=max_hold_minutes,
        invalidation_price=invalidation_price,
        generated_from_defaults=generated_from_defaults,
    )


def _missing_required_fields(values: dict[str, Any], *, is_entry: bool) -> list[str]:
    if not is_entry:
        return []
    missing: list[str] = []
    for key, value in values.items():
        if key in {
            "position_size_pct",
            "leverage",
            "stop_loss_pct",
            "take_profit_pct",
            "trailing_profit_trigger_pct",
            "profit_drawdown_exit_pct",
        }:
            parsed = _safe_float_or_none(value)
            if parsed is None or parsed <= 0:
                missing.append(key)
            continue
        if key == "model_sources":
            if not value:
                missing.append(key)
            continue
        if key in {"partial_exit_plan", "full_exit_plan", "do_not_close_conditions"}:
            if not value:
                missing.append(key)
            continue
        if value is None:
            missing.append(key)
        elif isinstance(value, str) and not value.strip():
            missing.append(key)
    return missing


def _entry_metrics(raw: dict[str, Any], side: str) -> dict[str, float | None]:
    opportunity = _safe_dict(raw.get("opportunity_score"))
    side_evidence = _selected_side_evidence(raw, side)
    return {
        "expected_net_return_pct": _first_float(
            side_evidence.get("expected_net_return_pct"),
            opportunity.get("expected_net_return_pct"),
        ),
        "profit_quality_ratio": _first_float(
            side_evidence.get("profit_quality_ratio"),
            opportunity.get("profit_quality_ratio"),
        ),
        "loss_probability": _first_float(
            side_evidence.get("loss_probability"),
            opportunity.get("server_profit_loss_probability"),
            opportunity.get("loss_probability"),
        ),
        "tail_risk_score": _first_float(
            side_evidence.get("tail_loss_probability"),
            opportunity.get("tail_loss_probability"),
            side_evidence.get("tail_risk_score"),
            opportunity.get("tail_risk_score"),
        ),
    }


def _selected_side_evidence(raw: dict[str, Any], side: str) -> dict[str, Any]:
    evidence = _safe_dict(raw.get("entry_candidate_evidence"))
    side_evidence = _safe_dict(evidence.get(side))
    if side_evidence:
        return side_evidence
    if str(evidence.get("side") or "").lower() == side:
        return evidence
    return {}


def _strategy_profile_id(raw: dict[str, Any]) -> str:
    candidates = (
        raw.get("strategy_profile_id"),
        _safe_dict(raw.get("strategy_learning_context")).get("strategy_profile_id"),
        _safe_dict(raw.get("strategy_mode")).get("strategy_profile_id"),
        _safe_dict(_safe_dict(raw.get("profit_risk_sizing")).get("strategy_learning_sizing")).get(
            "profile_id"
        ),
    )
    for value in candidates:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _breakdown_component(opportunity: dict[str, Any], key: str) -> float | None:
    breakdown = _safe_dict(opportunity.get("expected_net_breakdown"))
    for item in _safe_list(breakdown.get("components")):
        row = _safe_dict(item)
        if row.get("key") == key:
            value = _first_float(row.get("raw_return_pct"), row.get("contribution_pct"))
            return abs(value) if value is not None else None
    return None


def _entry_price_reference(decision: Any, raw: dict[str, Any]) -> float | None:
    feature = _safe_dict(_row_get(decision, "feature_snapshot"))
    return _first_float(
        raw.get("entry_price_reference"),
        raw.get("current_price"),
        _safe_dict(raw.get("market_data")).get("price"),
        feature.get("close"),
        feature.get("price"),
        feature.get("last_price"),
    )


def _portfolio_side_pressure(opportunity: dict[str, Any], raw: dict[str, Any]) -> float | None:
    exposure = _safe_dict(opportunity.get("position_exposure")) or _safe_dict(
        raw.get("position_exposure")
    )
    return _first_float(
        opportunity.get("portfolio_side_pressure"),
        exposure.get("same_side_ratio"),
        exposure.get("side_pressure"),
    )


def _same_symbol_side_edge(opportunity: dict[str, Any]) -> float | None:
    profile = _safe_dict(opportunity.get("symbol_side_profile")) or _safe_dict(
        opportunity.get("symbol_profile")
    )
    return _first_float(
        opportunity.get("same_symbol_side_edge"),
        profile.get("pnl"),
        profile.get("today_pnl"),
    )


def _recent_realized_edge(opportunity: dict[str, Any]) -> float | None:
    return _first_float(
        opportunity.get("recent_realized_edge"),
        opportunity.get("side_realized_pnl_usdt"),
        opportunity.get("symbol_realized_pnl_usdt"),
    )


def _is_probe_row(row: Any) -> bool:
    raw = _safe_dict(_row_get(row, "entry_raw") or _row_get(row, "raw_llm_response"))
    plan = _safe_dict(raw.get("profit_first_trade_plan"))
    lane = str(_row_get(row, "decision_lane") or plan.get("decision_lane") or "").lower()
    if lane in {"tiny_probe", "validated_probe"}:
        return True
    sizing = _safe_dict(raw.get("profit_risk_sizing"))
    quality_tier = str(sizing.get("quality_tier") or "").lower()
    if "probe" in quality_tier:
        return True
    size = _first_float(
        _row_get(row, "position_size_pct"),
        sizing.get("position_size_pct"),
        plan.get("position_size_pct"),
    )
    return bool(size is not None and size <= 0.02)


def _hold_minutes(row: Any) -> float | None:
    direct = _first_float(_row_get(row, "hold_minutes"))
    if direct is not None:
        return direct
    opened = _parse_datetime(_row_get(row, "created_at") or _row_get(row, "entry_at"))
    closed = _parse_datetime(_row_get(row, "closed_at"))
    if opened and closed:
        return max((closed - opened).total_seconds() / 60.0, 0.0)
    return None


def _decision_raw(decision: Any) -> dict[str, Any]:
    raw = _row_get(decision, "raw_response")
    if not isinstance(raw, dict):
        raw = _row_get(decision, "raw_llm_response")
    return dict(raw) if isinstance(raw, dict) else {}


def _set_decision_raw(decision: Any, raw: dict[str, Any]) -> None:
    if hasattr(decision, "raw_response"):
        setattr(decision, "raw_response", raw)
    elif hasattr(decision, "raw_llm_response"):
        setattr(decision, "raw_llm_response", raw)


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _action_value(action: Any) -> str:
    value = getattr(action, "value", action)
    text = str(value or "").lower().strip()
    if text in {"buy"}:
        return "long"
    if text in {"sell"}:
        return "short"
    if text in {"open_long"}:
        return "long"
    if text in {"open_short"}:
        return "short"
    return text


def _entry_side(action: str) -> str:
    if action in {"long", "buy", "open_long"}:
        return "long"
    if action in {"short", "sell", "open_short"}:
        return "short"
    return ""


def _side_from_raw(raw: dict[str, Any]) -> str:
    opportunity = _safe_dict(raw.get("opportunity_score"))
    side = str(opportunity.get("side") or raw.get("side") or "").lower().strip()
    return side if side in {"long", "short"} else ""


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _first_float(*values: Any) -> float | None:
    for value in values:
        parsed = _safe_float_or_none(value)
        if parsed is not None:
            return parsed
    return None


def _safe_float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def _round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 8)


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
