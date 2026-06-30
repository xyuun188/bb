"""Dynamic entry leverage allocation.

The allocator keeps OKX-facing leverage executable as an integer while using
continuous risk and signal inputs to decide the target.  It deliberately avoids
evidence-tier hard caps such as "exploration is always 3x"; tiers and warnings
are only risk factors in the calculation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    score: float
    min_score_required: float
    confidence: float
    aligned_source_count: int
    evidence_tier: str = ""
    evidence_effective_score: float = 0.0
    low_payoff_quality: bool = False
    weak_history: bool = False
    negative_local_expected: bool = False
    symbol_profit_tier: str = "neutral"
    quality_tier: str = "base"
    high_quality_entry: bool = False
    atr_pct: float = 0.0
    execution_cost: dict[str, Any] = field(default_factory=dict)
    open_positions_count: int = 0
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
        }


class DynamicLeverageAllocator:
    """Allocate integer leverage from symbol/signal/risk context."""

    def allocate(self, data: DynamicLeverageInput) -> DynamicLeverageDecision:
        system_max = max(floor(_safe_float(data.system_max_leverage, 1.0)), 1)
        requested = _clamp(_safe_float(data.requested_leverage, 1.0), 1.0, float(system_max))
        reasons: list[str] = []
        adjustments: list[dict[str, Any]] = []

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

        risk_tempered = (
            data.low_payoff_quality
            or data.negative_local_expected
            or (data.weak_history and not data.high_quality_entry)
            or data.tail_risk_score >= 0.82
            or data.loss_probability >= 0.58
        )
        if risk_tempered:
            theoretical = min(requested, signal_quality, candidate_limit)
        else:
            theoretical = min(max(requested, signal_quality), candidate_limit)
        theoretical = _clamp(theoretical, 1.0, float(system_max))
        if risk_tempered:
            reasons.append("tempered_by_risk_flags")
        if candidate_limit < requested:
            reasons.append(f"limited_by_{limiting_factor}")
        elif signal_quality > requested + 0.25:
            reasons.append("lifted_by_signal_quality")
        else:
            reasons.append("kept_near_requested_leverage")

        rounding_policy = self._rounding_policy(data)
        if rounding_policy == "floor_for_risk":
            final_integer = floor(theoretical)
        else:
            final_integer = int(round(theoretical))

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
        )

    def _signal_quality_leverage(
        self,
        data: DynamicLeverageInput,
        system_max: float,
        adjustments: list[dict[str, Any]],
    ) -> float:
        score_ratio = data.score / max(data.min_score_required, 1e-9)
        expected_component = _clamp(max(data.expected_net_return_pct, 0.0) / 2.4, 0.0, 1.0)
        quality_component = _clamp(data.profit_quality_ratio / 1.5, 0.0, 1.0)
        score_component = _clamp((score_ratio - 0.75) / 2.5, 0.0, 1.0)
        probability_component = _clamp((0.68 - data.loss_probability) / 0.38, 0.0, 1.0)
        tail_component = _clamp((0.92 - data.tail_risk_score) / 0.62, 0.0, 1.0)
        alignment_component = _clamp(data.aligned_source_count / 4.0, 0.0, 1.0)
        confidence_component = _clamp((data.confidence - 0.55) / 0.35, 0.0, 1.0)

        quality_index = (
            expected_component * 0.22
            + quality_component * 0.20
            + score_component * 0.18
            + probability_component * 0.16
            + tail_component * 0.12
            + alignment_component * 0.08
            + confidence_component * 0.04
        )
        if data.low_payoff_quality:
            quality_index *= 0.52
            adjustments.append({"factor": "low_payoff_quality", "multiplier": 0.52})
        if data.negative_local_expected:
            quality_index *= 0.68
            adjustments.append({"factor": "negative_local_expected", "multiplier": 0.68})
        if data.weak_history and not data.high_quality_entry:
            quality_index *= 0.78
            adjustments.append({"factor": "weak_history", "multiplier": 0.78})
        if data.evidence_tier in {"weak_conflict_probe", "degraded_missing_probe", "blocked"}:
            quality_index *= 0.62
            adjustments.append({"factor": f"evidence_tier:{data.evidence_tier}", "multiplier": 0.62})

        leverage = 1.0 + (system_max - 1.0) * _clamp(quality_index, 0.0, 1.0)
        if data.quality_tier in {"elite", "winner_add", "high_profit"}:
            leverage *= 1.08
            adjustments.append({"factor": f"quality_tier:{data.quality_tier}", "multiplier": 1.08})
        elif data.quality_tier in {"strong_probe", "quality_override"}:
            leverage *= 1.04
            adjustments.append({"factor": f"quality_tier:{data.quality_tier}", "multiplier": 1.04})
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
            adjustments.append({"factor": "volatility_missing", "leverage": system_max})
            return system_max
        leverage = 0.075 / max(atr_pct, 0.0025)
        if data.tail_risk_score > 0:
            leverage *= _clamp(1.15 - data.tail_risk_score * 0.35, 0.60, 1.10)
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
            _safe_float(cost.get("max_slippage_pct"), 0.0),
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
            adjustments.append({"factor": "liquidity_cost_missing", "leverage": system_max})
            return system_max
        leverage = system_max / (1.0 + cost_pressure * 18.0)
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
        multiplier = 1.0
        if data.symbol_profit_tier in {"symbol_winner", "side_winner"}:
            multiplier += 0.12 if data.high_quality_entry else 0.04
        elif data.symbol_profit_tier == "side_loser":
            multiplier -= 0.28 if not data.high_quality_entry else 0.10
        if data.weak_history:
            multiplier -= 0.18 if not data.high_quality_entry else 0.06
        leverage = system_max * _clamp(multiplier, 0.45, 1.15)
        adjustments.append(
            {
                "factor": "symbol_history",
                "symbol_profit_tier": data.symbol_profit_tier,
                "weak_history": data.weak_history,
                "multiplier": round(multiplier, 6),
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
        position_pressure = max(data.open_positions_count - 4, 0) * 0.18
        exposure_pressure = _clamp(data.portfolio_exposure_pct, 0.0, 1.5) * 0.35
        leverage = system_max / (1.0 + position_pressure + exposure_pressure)
        adjustments.append(
            {
                "factor": "portfolio_pressure",
                "open_positions_count": data.open_positions_count,
                "portfolio_exposure_pct": round(data.portfolio_exposure_pct, 6),
                "leverage": round(leverage, 6),
            }
        )
        return _clamp(leverage, 1.0, system_max)

    def _rounding_policy(self, data: DynamicLeverageInput) -> str:
        if (
            data.low_payoff_quality
            or data.negative_local_expected
            or (data.weak_history and not data.high_quality_entry)
            or data.tail_risk_score >= 0.82
            or data.loss_probability >= 0.58
        ):
            return "floor_for_risk"
        return "nearest_integer"
