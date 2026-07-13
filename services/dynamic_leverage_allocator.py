"""Dynamic entry leverage allocation.

The allocator keeps OKX-facing leverage executable as an integer while using
only continuous fee-after-return, cost, volatility, exposure, and account-risk
inputs to decide the target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import floor
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lower: float, upper: float) -> float:
    if upper < lower:
        return lower
    return min(max(value, lower), upper)


@dataclass(frozen=True, slots=True)
class DynamicLeverageInput:
    symbol: str
    requested_leverage: float
    system_max_leverage: float
    balance: float
    position_size_pct: float
    stress_stop_loss_pct: float
    max_loss_usdt: float
    expected_net_return_pct: float
    profit_quality_ratio: float
    loss_probability: float
    tail_risk_score: float
    aligned_source_count: int
    atr_pct: float = 0.0
    execution_cost: dict[str, Any] = field(default_factory=dict)
    portfolio_exposure_pct: float = 0.0


@dataclass(frozen=True, slots=True)
class DynamicLeverageDecision:
    requested_leverage: float
    theoretical_leverage: float
    final_integer_leverage: int
    rounding_policy: str
    system_max_leverage: float
    risk_budget_leverage: float
    volatility_leverage: float
    liquidity_leverage: float
    signal_quality_leverage: float
    history_leverage: float
    portfolio_leverage: float
    limiting_factor: str
    reasons: list[str]
    adjustments: list[dict[str, Any]]
    policy_provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "dynamic_leverage_allocator_v1",
            "requested_leverage": round(self.requested_leverage, 6),
            "theoretical_leverage": round(self.theoretical_leverage, 6),
            "final_integer_leverage": self.final_integer_leverage,
            "rounding_policy": self.rounding_policy,
            "system_max_leverage": round(self.system_max_leverage, 6),
            "risk_budget_leverage": round(self.risk_budget_leverage, 6),
            "volatility_leverage": round(self.volatility_leverage, 6),
            "liquidity_leverage": round(self.liquidity_leverage, 6),
            "signal_quality_leverage": round(self.signal_quality_leverage, 6),
            "history_leverage": round(self.history_leverage, 6),
            "portfolio_leverage": round(self.portfolio_leverage, 6),
            "limiting_factor": self.limiting_factor,
            "reasons": list(self.reasons),
            "adjustments": list(self.adjustments),
            "policy_provenance": dict(self.policy_provenance),
        }


class DynamicLeverageAllocator:
    """Allocate integer leverage from symbol/signal/risk context."""

    def allocate(self, data: DynamicLeverageInput) -> DynamicLeverageDecision:
        system_max = max(floor(_safe_float(data.system_max_leverage, 1.0)), 1)
        requested = _clamp(_safe_float(data.requested_leverage, 1.0), 1.0, float(system_max))
        reasons: list[str] = []
        adjustments: list[dict[str, Any]] = []
        cost = data.execution_cost if isinstance(data.execution_cost, dict) else {}
        missing_inputs: list[str] = []
        if int(data.aligned_source_count) <= 0:
            missing_inputs.append("authoritative_return_samples_missing")
        if _safe_float(data.expected_net_return_pct, 0.0) <= 0:
            missing_inputs.append("positive_fee_after_return_missing")
        if cost.get("production_eligible") is not True:
            missing_inputs.append("live_execution_cost_incomplete")
        if missing_inputs:
            fallback_reason = ",".join(missing_inputs)
            return DynamicLeverageDecision(
                requested_leverage=requested,
                theoretical_leverage=1.0,
                final_integer_leverage=1,
                rounding_policy="floor_to_exchange_integer",
                system_max_leverage=float(system_max),
                risk_budget_leverage=1.0,
                volatility_leverage=1.0,
                liquidity_leverage=1.0,
                signal_quality_leverage=1.0,
                history_leverage=1.0,
                portfolio_leverage=1.0,
                limiting_factor="production_inputs",
                reasons=missing_inputs,
                adjustments=[{"factor": "fail_closed", "reasons": missing_inputs}],
                policy_provenance={
                    "source": "current_decision_return_cost_volatility_exposure_and_account_budget",
                    "observation_window": "current_decision_with_active_account_state",
                    "sample_count": max(int(data.aligned_source_count), 0),
                    "generated_at": datetime.now(UTC).isoformat(),
                    "strategy_version": "2026-07-12.dynamic-leverage.v2",
                    "fallback_reason": fallback_reason,
                    "production_eligible": False,
                },
            )

        signal_quality = self._signal_quality_leverage(data, float(system_max), adjustments)
        risk_budget = self._risk_budget_leverage(data, float(system_max), adjustments)
        volatility = self._volatility_leverage(data, float(system_max), adjustments)
        liquidity = self._liquidity_leverage(data, float(system_max), adjustments)
        history = self._history_leverage(data, float(system_max), adjustments)
        portfolio = self._portfolio_leverage(data, float(system_max), adjustments)

        candidates = {
            "system_max": float(system_max),
            "risk_budget": risk_budget,
            "volatility": volatility,
            "liquidity": liquidity,
            "signal_quality": signal_quality,
            "history": history,
            "portfolio": portfolio,
        }
        limiting_factor, candidate_limit = min(candidates.items(), key=lambda item: item[1])

        theoretical = min(signal_quality, candidate_limit)
        theoretical = _clamp(theoretical, 1.0, float(system_max))
        if candidate_limit < requested:
            reasons.append(f"limited_by_{limiting_factor}")
        else:
            reasons.append("derived_from_return_quality_and_risk_budget")

        rounding_policy = self._rounding_policy(data)
        final_integer = floor(theoretical)

        integer_cap = max(1, floor(min(candidate_limit, float(system_max))))
        final_integer = max(1, min(final_integer, integer_cap, system_max))

        return DynamicLeverageDecision(
            requested_leverage=requested,
            theoretical_leverage=theoretical,
            final_integer_leverage=final_integer,
            rounding_policy=rounding_policy,
            system_max_leverage=float(system_max),
            risk_budget_leverage=risk_budget,
            volatility_leverage=volatility,
            liquidity_leverage=liquidity,
            signal_quality_leverage=signal_quality,
            history_leverage=history,
            portfolio_leverage=portfolio,
            limiting_factor=limiting_factor,
            reasons=reasons,
            adjustments=adjustments,
            policy_provenance={
                "source": "current_decision_return_cost_volatility_exposure_and_account_budget",
                "observation_window": "current_decision_with_active_account_state",
                "sample_count": max(int(data.aligned_source_count), 0),
                "generated_at": datetime.now(UTC).isoformat(),
                "strategy_version": "2026-07-12.dynamic-leverage.v2",
                "fallback_reason": "",
                "production_eligible": True,
            },
        )

    def _signal_quality_leverage(
        self,
        data: DynamicLeverageInput,
        system_max: float,
        adjustments: list[dict[str, Any]],
    ) -> float:
        positive_edge = max(data.expected_net_return_pct, 0.0)
        expected_component = positive_edge / (positive_edge + 1.0)
        quality = max(data.profit_quality_ratio, 0.0)
        quality_component = quality / (quality + 1.0)
        survival_component = (1.0 - _clamp(data.loss_probability, 0.0, 1.0)) * (
            1.0 - _clamp(data.tail_risk_score, 0.0, 1.0)
        )
        components = [
            expected_component,
            quality_component,
            survival_component,
        ]
        quality_index = 1.0
        for component in components:
            quality_index *= max(component, 0.0)
        quality_index = quality_index ** (1.0 / len(components))
        leverage = 1.0 + (system_max - 1.0) * _clamp(quality_index, 0.0, 1.0)
        adjustments.append(
            {
                "factor": "continuous_signal_quality",
                "quality_index": round(quality_index, 6),
                "components": [round(component, 6) for component in components],
            }
        )
        return _clamp(leverage, 1.0, system_max)

    def _risk_budget_leverage(
        self,
        data: DynamicLeverageInput,
        system_max: float,
        adjustments: list[dict[str, Any]],
    ) -> float:
        denominator = data.balance * data.position_size_pct * data.stress_stop_loss_pct
        if denominator <= 0 or data.max_loss_usdt <= 0:
            adjustments.append({"factor": "risk_budget_missing", "leverage": 1.0})
            return 1.0
        leverage = data.max_loss_usdt / denominator
        adjustments.append({"factor": "risk_budget", "leverage": round(leverage, 6)})
        return _clamp(leverage, 1.0, system_max)

    def _volatility_leverage(
        self,
        data: DynamicLeverageInput,
        system_max: float,
        adjustments: list[dict[str, Any]],
    ) -> float:
        atr_pct = max(data.atr_pct, 0.0)
        if atr_pct <= 0:
            adjustments.append({"factor": "volatility_missing", "leverage": 1.0})
            return 1.0
        risk_distance = max(data.stress_stop_loss_pct, 0.0)
        if risk_distance <= 0:
            adjustments.append({"factor": "risk_distance_missing", "leverage": 1.0})
            return 1.0
        volatility_share = risk_distance / (risk_distance + atr_pct)
        leverage = 1.0 + (system_max - 1.0) * volatility_share
        adjustments.append(
            {
                "factor": "atr_volatility",
                "atr_pct": round(atr_pct, 8),
                "leverage": round(leverage, 6),
            }
        )
        return _clamp(leverage, 1.0, system_max)

    def _liquidity_leverage(
        self,
        data: DynamicLeverageInput,
        system_max: float,
        adjustments: list[dict[str, Any]],
    ) -> float:
        cost = data.execution_cost if isinstance(data.execution_cost, dict) else {}
        slippage = max(
            _safe_float(cost.get("slippage_pct"), 0.0),
            _safe_float(cost.get("estimated_slippage_pct"), 0.0),
        )
        spread = max(
            _safe_float(cost.get("spread_pct"), 0.0),
            _safe_float(cost.get("bid_ask_spread_pct"), 0.0),
        )
        penalty = max(
            _safe_float(cost.get("liquidity_penalty_pct"), 0.0),
            _safe_float(cost.get("imbalance_penalty_pct"), 0.0),
        )
        cost_pressure = slippage + spread + penalty
        if cost_pressure <= 0:
            adjustments.append({"factor": "liquidity_cost_missing", "leverage": 1.0})
            return 1.0
        positive_edge = max(data.expected_net_return_pct, 0.0)
        cost_share = positive_edge / max(positive_edge + cost_pressure, 1e-9)
        leverage = 1.0 + (system_max - 1.0) * cost_share
        adjustments.append(
            {
                "factor": "liquidity_cost",
                "cost_pressure_pct": round(cost_pressure, 8),
                "leverage": round(leverage, 6),
            }
        )
        return _clamp(leverage, 1.0, system_max)

    def _history_leverage(
        self,
        data: DynamicLeverageInput,
        system_max: float,
        adjustments: list[dict[str, Any]],
    ) -> float:
        return_quality = max(data.profit_quality_ratio, 0.0)
        history_share = return_quality / (return_quality + 1.0)
        leverage = 1.0 + (system_max - 1.0) * history_share
        adjustments.append(
            {
                "factor": "symbol_history",
                "return_quality": round(return_quality, 6),
                "history_share": round(history_share, 6),
                "leverage": round(leverage, 6),
            }
        )
        return _clamp(leverage, 1.0, system_max)

    def _portfolio_leverage(
        self,
        data: DynamicLeverageInput,
        system_max: float,
        adjustments: list[dict[str, Any]],
    ) -> float:
        available_exposure = 1.0 - _clamp(data.portfolio_exposure_pct, 0.0, 1.0)
        leverage = 1.0 + (system_max - 1.0) * available_exposure
        adjustments.append(
            {
                "factor": "portfolio_pressure",
                "portfolio_exposure_pct": round(data.portfolio_exposure_pct, 6),
                "available_exposure": round(available_exposure, 6),
                "leverage": round(leverage, 6),
            }
        )
        return _clamp(leverage, 1.0, system_max)

    def _rounding_policy(self, data: DynamicLeverageInput) -> str:
        return "floor_to_exchange_integer"
