"""Adapters between BB runtime inputs and the executor-neutral strategy contract."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from ai_brain.base_model import Action, DecisionOutput
from core.strategy_contracts import (
    ExecutionMode,
    PositionSide,
    StrategyAction,
    StrategyContext,
    StrategyContractError,
    StrategyDecision,
    assert_decision_matches_context,
)
from data_feed.feature_vector import FeatureVector

STRATEGY_ADAPTER_VERSION = "bb.strategy-contract-adapter.v1"


def context_from_feature_vector(
    features: FeatureVector,
    *,
    execution_mode: ExecutionMode | str,
    strategy: Mapping[str, Any],
    parameters: Mapping[str, Any],
    position_snapshot: Mapping[str, Any] | None = None,
    account_constraints: Mapping[str, Any] | None = None,
    execution_assumptions: Mapping[str, Any] | None = None,
    decision_time: datetime | None = None,
) -> StrategyContext:
    """Adapt one real-time feature snapshot without reading exchange state."""

    feature_snapshot = features.to_dict()
    observed_at = decision_time or features.timestamp
    market_snapshot = {
        "symbol": features.symbol,
        "observed_at": _aware_utc(observed_at, assume_utc=False).isoformat(),
        "current_price": _finite(features.current_price),
        "bid": _finite(features.bid),
        "ask": _finite(features.ask),
        "mark_price": _finite(features.mark_price),
        "index_price": _finite(features.index_price),
        "spread_pct": _finite(features.spread_pct),
        "price_source": str(features.price_source or ""),
        "market_fact": dict(features.market_fact or {}),
    }
    return _build_context(
        symbol=features.symbol,
        market_snapshot=market_snapshot,
        feature_snapshot=feature_snapshot,
        position_snapshot=position_snapshot,
        account_constraints=account_constraints,
        decision_time=observed_at,
        execution_mode=execution_mode,
        strategy=strategy,
        parameters=parameters,
        execution_assumptions=execution_assumptions,
    )


def context_from_historical_bar(
    *,
    symbol: str,
    timestamp: datetime,
    timeframe: str,
    bar: Mapping[str, Any],
    feature_snapshot: Mapping[str, Any] | None,
    strategy: Mapping[str, Any],
    parameters: Mapping[str, Any],
    position_snapshot: Mapping[str, Any] | None = None,
    account_constraints: Mapping[str, Any] | None = None,
    execution_assumptions: Mapping[str, Any] | None = None,
) -> StrategyContext:
    """Adapt a completed historical bar and explicitly mark its availability time."""

    opened_at = _aware_utc(timestamp, assume_utc=True)
    decision_time = opened_at + timeframe_duration(timeframe)
    normalized_bar = {
        key: _finite(bar.get(key)) for key in ("open", "high", "low", "close", "volume")
    }
    normalized_bar.update(
        {
            "symbol": symbol,
            "observed_at": decision_time.isoformat(),
            "bar_opened_at": opened_at.isoformat(),
            "bar_closed_at": decision_time.isoformat(),
            "timeframe": timeframe,
            "price_source": str(bar.get("price_source") or "historical_ohlcv"),
        }
    )
    return _build_context(
        symbol=symbol,
        market_snapshot=normalized_bar,
        feature_snapshot=feature_snapshot,
        position_snapshot=position_snapshot,
        account_constraints=account_constraints,
        decision_time=decision_time,
        execution_mode=ExecutionMode.BACKTEST,
        strategy=strategy,
        parameters=parameters,
        execution_assumptions=execution_assumptions,
    )


def timeframe_duration(value: str) -> timedelta:
    text = str(value or "").strip().lower()
    match = re.fullmatch(r"([1-9][0-9]*)([mhdw])", text)
    if not match:
        raise StrategyContractError("timeframe must use a positive m/h/d/w interval")
    amount = int(match.group(1))
    unit = match.group(2)
    seconds = {
        "m": 60,
        "h": 60 * 60,
        "d": 24 * 60 * 60,
        "w": 7 * 24 * 60 * 60,
    }[unit]
    return timedelta(seconds=amount * seconds)


def decision_from_ai_output(
    decision: DecisionOutput,
    context: StrategyContext,
    *,
    reason_codes: list[str] | tuple[str, ...] | None = None,
) -> StrategyDecision:
    """Create the standard decision view without changing the AI decision."""

    if decision.symbol != context.symbol:
        raise StrategyContractError("AI decision symbol does not match strategy context")
    action, side, target_exposure = _standard_action(decision, context)
    raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
    codes = reason_codes or _reason_codes(raw, action, side)
    protection_hints = {
        "stop_loss_pct": _finite(decision.stop_loss_pct),
        "take_profit_pct": _finite(decision.take_profit_pct),
        "suggested_leverage": _finite(decision.suggested_leverage, default=1.0),
        "suggested_holding_minutes": _finite(decision.suggested_holding_minutes),
        "maximum_holding_minutes": _finite(decision.maximum_holding_minutes),
        "suggested_close_fraction": _finite(decision.suggested_close_fraction),
    }
    standard = StrategyDecision(
        symbol=decision.symbol,
        action=action,
        side=side,
        target_exposure=target_exposure,
        confidence=decision.confidence,
        reason_codes=codes,
        protection_hints=protection_hints,
        strategy_version=context.strategy_version,
        parameter_version=context.parameter_version,
        decision_time=context.decision_time,
        strategy_input_sha256=context.strategy_input_sha256,
        source=str(decision.model_name or "ai_decision"),
    )
    assert_decision_matches_context(context, standard)
    return standard


def ai_output_from_decision(
    decision: StrategyDecision,
    context: StrategyContext,
    *,
    model_name: str | None = None,
    reasoning: str | None = None,
) -> DecisionOutput:
    """Build a legacy executor view for compatibility with existing BB policies."""

    assert_decision_matches_context(context, decision)
    action = _legacy_action(decision)
    hints = dict(decision.protection_hints)
    current_exposure = _current_exposure(context.position_snapshot)
    close_fraction = 0.0
    if decision.action == StrategyAction.EXIT:
        close_fraction = 1.0
    elif decision.action == StrategyAction.REDUCE and current_exposure > 0:
        close_fraction = min(
            max((current_exposure - decision.target_exposure) / current_exposure, 0.0),
            1.0,
        )
    raw_response = {
        "strategy_contract": {
            "adapter_version": STRATEGY_ADAPTER_VERSION,
            "context_sha256": context.context_sha256,
            "strategy_input_sha256": context.strategy_input_sha256,
            "decision_sha256": decision.decision_sha256,
            "reason_codes": list(decision.reason_codes),
        }
    }
    return DecisionOutput(
        model_name=str(model_name or decision.source),
        symbol=decision.symbol,
        action=action,
        confidence=decision.confidence,
        reasoning=reasoning or ",".join(decision.reason_codes),
        position_size_pct=(
            decision.target_exposure if decision.action == StrategyAction.ENTER else 0.0
        ),
        suggested_leverage=_finite(hints.get("suggested_leverage"), default=1.0),
        stop_loss_pct=_finite(hints.get("stop_loss_pct")),
        take_profit_pct=_finite(hints.get("take_profit_pct")),
        suggested_holding_minutes=_finite(hints.get("suggested_holding_minutes")),
        maximum_holding_minutes=_finite(hints.get("maximum_holding_minutes")),
        suggested_close_fraction=close_fraction,
        timestamp=context.decision_time,
        raw_response=raw_response,
        feature_snapshot=dict(context.feature_snapshot),
    )


def _build_context(
    *,
    symbol: str,
    market_snapshot: Mapping[str, Any],
    feature_snapshot: Mapping[str, Any] | None,
    position_snapshot: Mapping[str, Any] | None,
    account_constraints: Mapping[str, Any] | None,
    decision_time: datetime,
    execution_mode: ExecutionMode | str,
    strategy: Mapping[str, Any],
    parameters: Mapping[str, Any],
    execution_assumptions: Mapping[str, Any] | None,
) -> StrategyContext:
    return StrategyContext(
        symbol=symbol,
        market_snapshot=market_snapshot,
        feature_snapshot=feature_snapshot or {},
        position_snapshot=position_snapshot or {},
        account_constraints=account_constraints or {},
        decision_time=_aware_utc(decision_time, assume_utc=False),
        execution_mode=execution_mode,
        strategy_id=str(strategy.get("strategy_id") or ""),
        strategy_version=str(strategy.get("strategy_version") or ""),
        parameter_set_id=str(parameters.get("parameter_set_id") or ""),
        parameter_version=str(parameters.get("parameter_version") or ""),
        parameter_values=dict(parameters.get("values") or {}),
        execution_assumptions=execution_assumptions or {},
    )


def _standard_action(
    decision: DecisionOutput,
    context: StrategyContext,
) -> tuple[StrategyAction, PositionSide, float]:
    action = (
        decision.action
        if isinstance(decision.action, Action)
        else Action.from_string(str(decision.action))
    )
    current_side = _current_side(context.position_snapshot)
    current_exposure = _current_exposure(context.position_snapshot)
    if action == Action.LONG:
        return StrategyAction.ENTER, PositionSide.LONG, _finite(decision.position_size_pct)
    if action == Action.SHORT:
        return StrategyAction.ENTER, PositionSide.SHORT, _finite(decision.position_size_pct)
    if action in {Action.CLOSE_LONG, Action.CLOSE_SHORT}:
        side = PositionSide.LONG if action == Action.CLOSE_LONG else PositionSide.SHORT
        fraction = _finite(decision.suggested_close_fraction)
        if 0 < fraction < 1 and current_exposure > 0:
            return StrategyAction.REDUCE, side, current_exposure * (1 - fraction)
        return StrategyAction.EXIT, side, 0.0
    return StrategyAction.HOLD, current_side, current_exposure


def _legacy_action(decision: StrategyDecision) -> Action:
    if decision.action == StrategyAction.ENTER:
        return Action.LONG if decision.side == PositionSide.LONG else Action.SHORT
    if decision.action in {StrategyAction.EXIT, StrategyAction.REDUCE}:
        return Action.CLOSE_LONG if decision.side == PositionSide.LONG else Action.CLOSE_SHORT
    return Action.HOLD


def _reason_codes(
    raw: Mapping[str, Any],
    action: StrategyAction,
    side: PositionSide,
) -> tuple[str, ...]:
    recorded = raw.get("reason_codes")
    if isinstance(recorded, (list, tuple)):
        codes = tuple(str(item).strip() for item in recorded if str(item).strip())
        if codes:
            return codes
    suffix = action.value.upper()
    if side != PositionSide.NONE:
        suffix = f"{suffix}_{side.value.upper()}"
    return (f"SIGNAL_{suffix}",)


def _current_side(snapshot: Mapping[str, Any]) -> PositionSide:
    raw = str(snapshot.get("side") or "none").strip().lower()
    return PositionSide(raw) if raw in {"long", "short"} else PositionSide.NONE


def _current_exposure(snapshot: Mapping[str, Any]) -> float:
    return _finite(snapshot.get("exposure", snapshot.get("exposure_pct", 0.0)))


def _finite(value: Any, *, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(number):
        raise StrategyContractError("strategy adapter received a non-finite number")
    return number


def _aware_utc(value: datetime, *, assume_utc: bool) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        if not assume_utc:
            raise StrategyContractError("real-time decision timestamp must be timezone-aware")
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
