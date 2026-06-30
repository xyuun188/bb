"""Entry profit-risk sizing policy.

This module owns the entry-side size/leverage caps that depend on expected
profit, loss budget, evidence tier, ATR stress stops, and existing same-side
winners. TradingService wires the dependencies; this policy owns the decision
math.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import DecisionOutput
from config.settings import settings
from services.dynamic_leverage_allocator import DynamicLeverageAllocator, DynamicLeverageInput
from services.entry_direction_metrics import selected_entry_metrics
from services.entry_priority import MIN_ENTRY_OPPORTUNITY_SCORE
from services.entry_sizing import apply_evidence_sizing_policy
from services.profit_first_position_ladder import ProfitFirstPositionLadderPolicy
from services.profit_first_trade_plan import classify_decision_lane
from services.trading_params import DEFAULT_TRADING_PARAMS

EntryProfitRiskSizingEvaluator = Callable[
    [DecisionOutput, str, list[dict[str, Any]]],
    Awaitable[None],
]
EntryBalanceProvider = Callable[[str, DecisionOutput | None], Awaitable[float | None]]

_ENTRY_RISK_SIZING_PARAMS = DEFAULT_TRADING_PARAMS.entry_risk_sizing
ENTRY_MIN_NET_PROFIT_QUALITY_RATIO = _ENTRY_RISK_SIZING_PARAMS.min_net_profit_quality_ratio
ENTRY_WEAK_HISTORY_MAX_SIZE = _ENTRY_RISK_SIZING_PARAMS.weak_history_max_size
ENTRY_WEAK_HISTORY_MAX_LEVERAGE = _ENTRY_RISK_SIZING_PARAMS.weak_history_max_leverage
ENTRY_WEAK_HISTORY_STRONG_ALIGNED_MAX_SIZE = (
    _ENTRY_RISK_SIZING_PARAMS.weak_history_strong_aligned_max_size
)
ENTRY_WEAK_HISTORY_STRONG_ALIGNED_MAX_LEVERAGE = (
    _ENTRY_RISK_SIZING_PARAMS.weak_history_strong_aligned_max_leverage
)
ENTRY_NEGATIVE_LOCAL_EXPECTED_MAX_SIZE = _ENTRY_RISK_SIZING_PARAMS.negative_local_expected_max_size
ENTRY_NEGATIVE_LOCAL_EXPECTED_MAX_LEVERAGE = (
    _ENTRY_RISK_SIZING_PARAMS.negative_local_expected_max_leverage
)
ENTRY_LOW_QUALITY_MAX_SIZE = _ENTRY_RISK_SIZING_PARAMS.low_quality_max_size
ENTRY_LOW_QUALITY_MAX_LEVERAGE = _ENTRY_RISK_SIZING_PARAMS.low_quality_max_leverage
ENTRY_SYMBOL_LOSER_SIZE_MULTIPLIER = _ENTRY_RISK_SIZING_PARAMS.symbol_loser_size_multiplier
ENTRY_HIGH_QUALITY_MIN_NOTIONAL_BALANCE_RATIO = (
    _ENTRY_RISK_SIZING_PARAMS.high_quality_min_notional_balance_ratio
)
ENTRY_NORMAL_MIN_NOTIONAL_BALANCE_RATIO = (
    _ENTRY_RISK_SIZING_PARAMS.normal_min_notional_balance_ratio
)
ENTRY_NOTIONAL_FLOOR_MAX_SIZE_PCT = _ENTRY_RISK_SIZING_PARAMS.notional_floor_max_size_pct
ENTRY_HIGH_PROFIT_MIN_NOTIONAL_BALANCE_RATIO = (
    _ENTRY_RISK_SIZING_PARAMS.high_profit_min_notional_balance_ratio
)
ENTRY_HIGH_PROFIT_MIN_LEVERAGE = _ENTRY_RISK_SIZING_PARAMS.high_profit_min_leverage
ENTRY_HIGH_PROFIT_ELITE_MIN_LEVERAGE = _ENTRY_RISK_SIZING_PARAMS.high_profit_elite_min_leverage
ENTRY_GOOD_PROBE_MIN_NOTIONAL_BALANCE_RATIO = (
    _ENTRY_RISK_SIZING_PARAMS.good_probe_min_notional_balance_ratio
)
ENTRY_WINNER_ADD_MIN_NOTIONAL_BALANCE_RATIO = (
    _ENTRY_RISK_SIZING_PARAMS.winner_add_min_notional_balance_ratio
)
ENTRY_STRONG_PROBE_MIN_NOTIONAL_BALANCE_RATIO = (
    _ENTRY_RISK_SIZING_PARAMS.strong_probe_min_notional_balance_ratio
)
ENTRY_ELITE_MIN_NOTIONAL_BALANCE_RATIO = _ENTRY_RISK_SIZING_PARAMS.elite_min_notional_balance_ratio
ENTRY_MEANINGFUL_SIZE_MAX_TAIL_RISK = _ENTRY_RISK_SIZING_PARAMS.meaningful_size_max_tail_risk
ENTRY_MEANINGFUL_SIZE_MIN_PROFIT_USDT = _ENTRY_RISK_SIZING_PARAMS.meaningful_size_min_profit_usdt
ENTRY_MEANINGFUL_SIZE_MIN_PROFIT_RATIO = _ENTRY_RISK_SIZING_PARAMS.meaningful_size_min_profit_ratio
PORTFOLIO_ROSTER_FILL_MAX_LOSS_PROBABILITY = (
    _ENTRY_RISK_SIZING_PARAMS.portfolio_roster_fill_max_loss_probability
)
PORTFOLIO_ROSTER_FILL_MIN_NET_PCT = _ENTRY_RISK_SIZING_PARAMS.portfolio_roster_fill_min_net_pct
PORTFOLIO_ROSTER_FILL_MIN_PROFIT_QUALITY_RATIO = (
    _ENTRY_RISK_SIZING_PARAMS.portfolio_roster_fill_min_profit_quality_ratio
)
PORTFOLIO_ROSTER_FILL_NOTIONAL_BALANCE_RATIO = (
    _ENTRY_RISK_SIZING_PARAMS.portfolio_roster_fill_notional_balance_ratio
)
ENTRY_PNL_STRUCTURE_MIN_EXPECTED_PROFIT_USDT = (
    _ENTRY_RISK_SIZING_PARAMS.pnl_structure_min_expected_profit_usdt
)
ENTRY_PNL_STRUCTURE_LOW_QUALITY_MAX_LOSS_MULTIPLE = (
    _ENTRY_RISK_SIZING_PARAMS.pnl_structure_low_quality_max_loss_multiple
)
ENTRY_PNL_STRUCTURE_NORMAL_MAX_LOSS_MULTIPLE = (
    _ENTRY_RISK_SIZING_PARAMS.pnl_structure_normal_max_loss_multiple
)
ENTRY_PNL_STRUCTURE_HIGH_QUALITY_MAX_LOSS_MULTIPLE = (
    _ENTRY_RISK_SIZING_PARAMS.pnl_structure_high_quality_max_loss_multiple
)
ENTRY_BALANCED_PROBE_MAX_LOSS_USDT = _ENTRY_RISK_SIZING_PARAMS.balanced_probe_max_loss_usdt
ENTRY_STRONG_PROBE_MAX_LOSS_USDT = _ENTRY_RISK_SIZING_PARAMS.strong_probe_max_loss_usdt
ENTRY_QUALITY_RISK_BASE_CAP_PCT = _ENTRY_RISK_SIZING_PARAMS.quality_risk_base_cap_pct
ENTRY_QUALITY_RISK_MAX_CAP_PCT = _ENTRY_RISK_SIZING_PARAMS.quality_risk_max_cap_pct
ENTRY_QUALITY_RISK_ELITE_CAP_PCT = _ENTRY_RISK_SIZING_PARAMS.quality_risk_elite_cap_pct
ENTRY_RECOVERY_PROBE_BASE_CAP_PCT = _ENTRY_RISK_SIZING_PARAMS.recovery_probe_base_cap_pct
ENTRY_RECOVERY_PROBE_MAX_CAP_PCT = _ENTRY_RISK_SIZING_PARAMS.recovery_probe_max_cap_pct
ENTRY_STRATEGY_SIZING_MIN_MULTIPLIER = _ENTRY_RISK_SIZING_PARAMS.strategy_sizing_min_multiplier
ENTRY_STRATEGY_SIZING_MAX_MULTIPLIER = _ENTRY_RISK_SIZING_PARAMS.strategy_sizing_max_multiplier
ENTRY_RELEASE_PROBE_FRACTION_FLOOR = _ENTRY_RISK_SIZING_PARAMS.release_probe_fraction_floor
ENTRY_RELEASE_PROBE_DEFAULT_CAP_PCT = _ENTRY_RISK_SIZING_PARAMS.release_probe_default_cap_pct
ENTRY_RELEASE_PROBE_MIN_CAP_PCT = _ENTRY_RISK_SIZING_PARAMS.release_probe_min_cap_pct
ENTRY_RELEASE_PROBE_MAX_CAP_PCT = _ENTRY_RISK_SIZING_PARAMS.release_probe_max_cap_pct
ENTRY_RECOVERY_MULTIPLIER_CAP = _ENTRY_RISK_SIZING_PARAMS.recovery_multiplier_cap
ENTRY_RECOVERY_PROBE_FRACTION_FLOOR = _ENTRY_RISK_SIZING_PARAMS.recovery_probe_fraction_floor
ENTRY_RECOVERY_PROBE_DEFAULT_CAP_PCT = _ENTRY_RISK_SIZING_PARAMS.recovery_probe_default_cap_pct
ENTRY_RECOVERY_PROBE_MIN_CAP_PCT = _ENTRY_RISK_SIZING_PARAMS.recovery_probe_min_cap_pct
ENTRY_RECOVERY_PROBE_MAX_LEARNING_CAP_PCT = (
    _ENTRY_RISK_SIZING_PARAMS.recovery_learning_probe_max_cap_pct
)
ENTRY_STRATEGY_PROBE_FRACTION_MAX = _ENTRY_RISK_SIZING_PARAMS.strategy_probe_fraction_max
ENTRY_STRATEGY_PROBE_CAP_MAX_PCT = _ENTRY_RISK_SIZING_PARAMS.strategy_probe_cap_max_pct
ENTRY_ADAPTIVE_RECOVERY_MIN_PROFIT_QUALITY = (
    _ENTRY_RISK_SIZING_PARAMS.adaptive_recovery_min_profit_quality
)
ENTRY_ADAPTIVE_RECOVERY_MAX_LOSS_PROBABILITY = (
    _ENTRY_RISK_SIZING_PARAMS.adaptive_recovery_max_loss_probability
)
ENTRY_ADAPTIVE_RECOVERY_MAX_TAIL_RISK = _ENTRY_RISK_SIZING_PARAMS.adaptive_recovery_max_tail_risk


def _settings_max_leverage() -> float:
    return float(settings.max_leverage or 1.0)


@dataclass(slots=True)
class EntryProfitRiskSizingPolicy:
    """Apply entry sizing caps that depend on profit/risk context."""

    evaluator: EntryProfitRiskSizingEvaluator | None = None
    allocated_order_balance: EntryBalanceProvider | None = None
    entry_low_payoff_quality: Any | None = None
    entry_stop_loss_budget: Any | None = None
    entry_stress_stop: Any | None = None
    entry_existing_winner_context: Any | None = None
    profit_first_position_ladder: Any | None = None
    dynamic_leverage_allocator: Any | None = None
    max_leverage_provider: Callable[[], float] = _settings_max_leverage
    probe_max_loss_usdt: float = ENTRY_BALANCED_PROBE_MAX_LOSS_USDT
    strong_probe_max_loss_usdt: float = ENTRY_STRONG_PROBE_MAX_LOSS_USDT

    @staticmethod
    def _safe_dict(value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _safe_list(value: Any) -> list[Any]:
        return value if isinstance(value, list) else []

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _strategy_learning_sizing(cls, raw: dict[str, Any]) -> dict[str, Any]:
        strategy_mode = cls._safe_dict(raw.get("strategy_mode"))
        sizing = cls._safe_dict(strategy_mode.get("strategy_learning_sizing"))
        direct_sizing = cls._safe_dict(raw.get("strategy_learning_sizing"))
        if direct_sizing:
            sizing = {**direct_sizing, **sizing}
        context = cls._safe_dict(raw.get("strategy_learning_context"))
        context_sizing = cls._safe_dict(context.get("strategy_learning_sizing"))
        if context_sizing:
            sizing = {**sizing, **context_sizing}
        learning = cls._safe_dict(context.get("strategy_learning"))
        mode_learning = cls._safe_dict(strategy_mode.get("strategy_learning"))
        if (
            context.get("strategy_learning_release_pressure_active")
            or strategy_mode.get("strategy_learning_release_pressure_active")
            or learning.get("release_pressure_active")
            or mode_learning.get("release_pressure_active")
        ):
            sizing = {
                **sizing,
                "release_pressure_active": True,
                "reason": (
                    context.get("strategy_learning_release_pressure_reason")
                    or strategy_mode.get("strategy_learning_release_pressure_reason")
                    or learning.get("release_pressure_reason")
                    or mode_learning.get("release_pressure_reason")
                    or sizing.get("reason")
                    or "strategy learning release pressure is active"
                ),
                "position_size_multiplier": min(
                    max(
                        cls._safe_float(sizing.get("position_size_multiplier"), 1.0),
                        ENTRY_STRATEGY_SIZING_MIN_MULTIPLIER,
                    ),
                    ENTRY_STRATEGY_SIZING_MAX_MULTIPLIER,
                ),
                "probe_fraction": max(
                    cls._safe_float(sizing.get("probe_fraction"), 0.0),
                    ENTRY_RELEASE_PROBE_FRACTION_FLOOR,
                ),
                "max_probe_size_pct": min(
                    max(
                        cls._safe_float(
                            sizing.get("max_probe_size_pct"),
                            ENTRY_RELEASE_PROBE_DEFAULT_CAP_PCT,
                        ),
                        ENTRY_RELEASE_PROBE_MIN_CAP_PCT,
                    ),
                    ENTRY_RELEASE_PROBE_MAX_CAP_PCT,
                ),
            }
        if (
            context.get("strategy_learning_entry_pause")
            or strategy_mode.get("strategy_learning_entry_pause")
            or learning.get("entry_pause")
            or mode_learning.get("entry_pause")
        ):
            sizing = {
                **sizing,
                "entry_paused": False,
                "strategy_learning_pause_is_hard_gate": False,
                "recovery_probe_allowed": True,
                "health_guard_active": True,
                "reason": (
                    context.get("strategy_learning_entry_pause_reason")
                    or strategy_mode.get("strategy_learning_entry_pause_reason")
                    or learning.get("entry_pause_reason")
                    or mode_learning.get("entry_pause_reason")
                    or "策略学习护栏提示：已转为小仓恢复探针，不作为新开仓硬拦截。"
                ),
                "position_size_multiplier": min(
                    cls._safe_float(sizing.get("position_size_multiplier"), 1.0),
                    ENTRY_RECOVERY_MULTIPLIER_CAP,
                ),
                "probe_fraction": max(
                    cls._safe_float(sizing.get("probe_fraction"), 0.0),
                    ENTRY_RECOVERY_PROBE_FRACTION_FLOOR,
                ),
                "max_probe_size_pct": min(
                    max(
                        cls._safe_float(
                            sizing.get("max_probe_size_pct"),
                            ENTRY_RECOVERY_PROBE_DEFAULT_CAP_PCT,
                        ),
                        ENTRY_RECOVERY_PROBE_MIN_CAP_PCT,
                    ),
                    ENTRY_RECOVERY_PROBE_MAX_LEARNING_CAP_PCT,
                ),
            }
        if (
            context.get("strategy_learning_recovery_probe_allowed")
            or strategy_mode.get("strategy_learning_recovery_probe_allowed")
            or learning.get("recovery_probe_allowed")
            or mode_learning.get("recovery_probe_allowed")
        ):
            sizing = {
                **sizing,
                "health_guard_active": True,
                "recovery_probe_allowed": True,
                "reason": (
                    context.get("strategy_learning_recovery_probe_reason")
                    or strategy_mode.get("strategy_learning_recovery_probe_reason")
                    or learning.get("recovery_probe_reason")
                    or mode_learning.get("recovery_probe_reason")
                    or sizing.get("reason")
                    or "strategy learning recovery probe is active"
                ),
                "position_size_multiplier": min(
                    cls._safe_float(sizing.get("position_size_multiplier"), 1.0),
                    ENTRY_RECOVERY_MULTIPLIER_CAP,
                ),
                "probe_fraction": max(
                    cls._safe_float(sizing.get("probe_fraction"), 0.0),
                    ENTRY_RECOVERY_PROBE_FRACTION_FLOOR,
                ),
                "max_probe_size_pct": min(
                    max(
                        cls._safe_float(
                            sizing.get("max_probe_size_pct"),
                            ENTRY_RECOVERY_PROBE_DEFAULT_CAP_PCT,
                        ),
                        ENTRY_RECOVERY_PROBE_MIN_CAP_PCT,
                    ),
                    ENTRY_RECOVERY_PROBE_MAX_LEARNING_CAP_PCT,
                ),
            }
        runtime = cls._safe_dict(learning.get("runtime"))
        if runtime:
            sizing = {**runtime, **sizing}
        if not sizing and strategy_mode:
            sizing = {
                "position_size_multiplier": strategy_mode.get("position_size_multiplier"),
                "probe_fraction": strategy_mode.get("probe_fraction"),
                "max_probe_size_pct": strategy_mode.get("max_probe_size_pct"),
                "side_overrides": strategy_mode.get("side_quality"),
            }
        return sizing

    @classmethod
    def _apply_strategy_learning_sizing(
        cls,
        *,
        current_size: float,
        action_side: str,
        sizing: dict[str, Any],
        quality_override: bool = False,
        recovery_quality_cap_pct: float = 0.0,
    ) -> dict[str, Any]:
        if current_size <= 0 or not sizing:
            return {"applied": False}
        if sizing.get("entry_paused"):
            sizing = {
                **sizing,
                "entry_paused": False,
                "strategy_learning_pause_is_hard_gate": False,
                "recovery_probe_allowed": True,
                "health_guard_active": True,
                "position_size_multiplier": min(
                    cls._safe_float(sizing.get("position_size_multiplier"), 1.0),
                    ENTRY_RECOVERY_MULTIPLIER_CAP,
                ),
                "probe_fraction": max(
                    cls._safe_float(sizing.get("probe_fraction"), 0.0),
                    ENTRY_RECOVERY_PROBE_FRACTION_FLOOR,
                ),
                "max_probe_size_pct": min(
                    max(
                        cls._safe_float(
                            sizing.get("max_probe_size_pct"),
                            ENTRY_RECOVERY_PROBE_DEFAULT_CAP_PCT,
                        ),
                        ENTRY_RECOVERY_PROBE_MIN_CAP_PCT,
                    ),
                    ENTRY_RECOVERY_PROBE_MAX_LEARNING_CAP_PCT,
                ),
                "reason": str(
                    sizing.get("reason")
                    or "策略学习护栏提示：已转为小仓恢复探针，不作为新开仓硬拦截。"
                )[:240],
            }
        global_multiplier = min(
            max(
                cls._safe_float(sizing.get("position_size_multiplier"), 1.0),
                ENTRY_STRATEGY_SIZING_MIN_MULTIPLIER,
            ),
            ENTRY_STRATEGY_SIZING_MAX_MULTIPLIER,
        )
        side_overrides = cls._safe_dict(sizing.get("side_overrides"))
        side_row = cls._safe_dict(side_overrides.get(action_side))
        side_multiplier = min(
            max(
                cls._safe_float(side_row.get("size_multiplier"), 1.0),
                ENTRY_STRATEGY_SIZING_MIN_MULTIPLIER,
            ),
            ENTRY_STRATEGY_SIZING_MAX_MULTIPLIER,
        )
        probe_fraction = min(
            max(cls._safe_float(sizing.get("probe_fraction"), 0.0), 0.0),
            ENTRY_STRATEGY_PROBE_FRACTION_MAX,
        )
        max_probe_size = min(
            max(cls._safe_float(sizing.get("max_probe_size_pct"), 0.0), 0.0),
            ENTRY_STRATEGY_PROBE_CAP_MAX_PCT,
        )
        recovery_probe_active = bool(
            sizing.get("recovery_probe_allowed")
            or sizing.get("health_guard_active")
            or sizing.get("execution_guard_active")
            or sizing.get("entry_paused")
        )
        adaptive_recovery_lift = bool(
            recovery_probe_active
            and recovery_quality_cap_pct > max(max_probe_size, ENTRY_RECOVERY_PROBE_BASE_CAP_PCT)
        )
        if adaptive_recovery_lift:
            max_probe_size = min(
                max(recovery_quality_cap_pct, max_probe_size),
                ENTRY_RECOVERY_PROBE_MAX_CAP_PCT,
            )
        effective_global_multiplier = global_multiplier
        effective_side_multiplier = side_multiplier
        if quality_override:
            effective_global_multiplier = max(effective_global_multiplier, 1.0)
            effective_side_multiplier = max(effective_side_multiplier, 1.0)
        adjusted = min(
            max(current_size * effective_global_multiplier * effective_side_multiplier, 0.0),
            1.0,
        )
        cap = max_probe_size if max_probe_size > 0 and probe_fraction > 0 else 0.0
        if cap > 0 and not quality_override:
            adjusted = min(adjusted, cap)
        size_changed = abs(adjusted - current_size) > 1e-9
        policy_active = bool(
            sizing.get("profile_id")
            or sizing.get("release_pressure_active")
            or sizing.get("recovery_probe_allowed")
            or sizing.get("health_guard_active")
            or sizing.get("execution_guard_active")
            or sizing.get("entry_paused")
            or probe_fraction > 0
            or max_probe_size > 0
            or abs(global_multiplier - 1.0) > 1e-9
            or abs(side_multiplier - 1.0) > 1e-9
            or quality_override
            or adaptive_recovery_lift
        )
        applied = bool(size_changed or policy_active)
        return {
            "applied": applied,
            "size_changed": size_changed,
            "profile_id": sizing.get("profile_id"),
            "action_side": action_side,
            "original_position_size_pct": round(current_size, 6),
            "position_size_pct": round(adjusted, 6),
            "position_size_multiplier": round(global_multiplier, 6),
            "side_size_multiplier": round(side_multiplier, 6),
            "effective_position_size_multiplier": round(effective_global_multiplier, 6),
            "effective_side_size_multiplier": round(effective_side_multiplier, 6),
            "probe_fraction": round(probe_fraction, 6),
            "max_probe_size_pct": round(max_probe_size, 6),
            "probe_cap_applied": bool(cap > 0 and adjusted <= cap + 1e-12 and not quality_override),
            "quality_override": bool(quality_override),
            "adaptive_recovery_lift_applied": adaptive_recovery_lift,
            "adaptive_recovery_cap_pct": round(recovery_quality_cap_pct, 6),
            "entry_paused": bool(sizing.get("entry_paused", False)),
            "strategy_learning_pause_is_hard_gate": bool(
                sizing.get("strategy_learning_pause_is_hard_gate", False)
            ),
            "release_pressure_active": bool(sizing.get("release_pressure_active")),
            "health_guard_active": bool(sizing.get("health_guard_active")),
            "recovery_probe_allowed": bool(sizing.get("recovery_probe_allowed")),
            "execution_guard_active": bool(sizing.get("execution_guard_active")),
            "reason": str(sizing.get("reason") or side_row.get("reason") or "")[:240],
        }

    @classmethod
    def _adaptive_quality_loss_cap_pct(
        cls,
        *,
        expected_net_return_pct: float,
        profit_quality_ratio: float,
        loss_probability: float,
        tail_risk_score: float,
        quality_tier: str,
    ) -> float:
        """Return the equity-risk cap for high-quality entries.

        The cap is intentionally adaptive: better net edge, stronger profit
        quality, lower loss probability, and lower tail risk increase the
        single-trade budget; weak or uncertain signals stay near the base cap.
        """

        expected_component = min(
            max(expected_net_return_pct, 0.0) / 100.0 * 0.55,
            _ENTRY_RISK_SIZING_PARAMS.adaptive_quality_expected_component_cap,
        )
        quality_component = min(
            max(profit_quality_ratio - 1.0, 0.0) * 0.004,
            _ENTRY_RISK_SIZING_PARAMS.adaptive_quality_profit_quality_component_cap,
        )
        probability_component = min(
            max(0.45 - loss_probability, 0.0) * 0.020,
            _ENTRY_RISK_SIZING_PARAMS.adaptive_quality_probability_component_cap,
        )
        tail_risk_discount = min(
            max(tail_risk_score - 0.55, 0.0) * 0.012,
            _ENTRY_RISK_SIZING_PARAMS.adaptive_quality_tail_discount_cap,
        )
        raw_cap = (
            ENTRY_QUALITY_RISK_BASE_CAP_PCT
            + expected_component
            + quality_component
            + probability_component
            - tail_risk_discount
        )
        hard_cap = (
            ENTRY_QUALITY_RISK_ELITE_CAP_PCT
            if quality_tier in {"elite", "winner_add", "high_profit"}
            else ENTRY_QUALITY_RISK_MAX_CAP_PCT
        )
        return min(max(raw_cap, ENTRY_QUALITY_RISK_BASE_CAP_PCT), hard_cap)

    @classmethod
    def _adaptive_recovery_probe_cap_pct(
        cls,
        *,
        expected_net_return_pct: float,
        profit_quality_ratio: float,
        loss_probability: float,
        tail_risk_score: float,
        score: float,
        min_score_required: float,
        aligned_source_count: int,
    ) -> float:
        """Return a quality-driven cap for recovery probes.

        Recovery mode exists to keep the system trading while model health or
        execution feedback is being rebuilt. It should not permanently flatten
        every good signal into dust-size orders; strong positive expectancy and
        multiple independent confirmations earn a larger but still bounded cap.
        """

        if (
            expected_net_return_pct <= 0.0
            or profit_quality_ratio < ENTRY_ADAPTIVE_RECOVERY_MIN_PROFIT_QUALITY
            or loss_probability > ENTRY_ADAPTIVE_RECOVERY_MAX_LOSS_PROBABILITY
            or tail_risk_score >= ENTRY_ADAPTIVE_RECOVERY_MAX_TAIL_RISK
            or aligned_source_count <= 0
        ):
            return 0.0
        score_ratio = score / max(min_score_required, 1e-12)
        expected_component = min(
            max(expected_net_return_pct, 0.0) * 0.010,
            _ENTRY_RISK_SIZING_PARAMS.adaptive_recovery_expected_component_cap,
        )
        quality_component = min(
            max(profit_quality_ratio - ENTRY_ADAPTIVE_RECOVERY_MIN_PROFIT_QUALITY, 0.0) * 0.016,
            _ENTRY_RISK_SIZING_PARAMS.adaptive_recovery_profit_quality_component_cap,
        )
        score_component = min(
            max(score_ratio - 1.0, 0.0) * 0.005,
            _ENTRY_RISK_SIZING_PARAMS.adaptive_recovery_score_component_cap,
        )
        alignment_component = min(
            aligned_source_count * 0.004,
            _ENTRY_RISK_SIZING_PARAMS.adaptive_recovery_alignment_component_cap,
        )
        loss_discount = (
            max(
                loss_probability - _ENTRY_RISK_SIZING_PARAMS.adaptive_recovery_loss_discount_anchor,
                0.0,
            )
            * 0.018
        )
        tail_discount = (
            max(
                tail_risk_score - _ENTRY_RISK_SIZING_PARAMS.adaptive_recovery_tail_discount_anchor,
                0.0,
            )
            * 0.014
        )
        cap = (
            ENTRY_RECOVERY_PROBE_BASE_CAP_PCT
            + expected_component
            + quality_component
            + score_component
            + alignment_component
            - loss_discount
            - tail_discount
        )
        return min(
            max(cap, ENTRY_RECOVERY_PROBE_BASE_CAP_PCT),
            ENTRY_RECOVERY_PROBE_MAX_CAP_PCT,
        )

    @classmethod
    def _profit_first_lane_for_sizing(
        cls,
        *,
        raw: dict[str, Any],
        decision: DecisionOutput,
        expected_net_return_pct: float,
        expected_profit_usdt: float,
        profit_quality_ratio: float,
        reward_risk_ratio: float,
        loss_probability: float,
        tail_risk_score: float,
        aligned_source_count: int,
    ) -> dict[str, Any]:
        persisted_plan = cls._safe_dict(raw.get("profit_first_trade_plan"))
        existing_lane = str(persisted_plan.get("decision_lane") or "").lower().strip()
        ignored_persisted_plan: dict[str, Any] = {}
        if existing_lane in {
            "shadow_only",
            "tiny_probe",
            "validated_probe",
            "meaningful_entry",
            "high_conviction",
        }:
            missing_fields = [
                str(item)
                for item in cls._safe_list(persisted_plan.get("missing_required_fields"))
                if str(item or "").strip()
            ]
            persisted_complete = bool(persisted_plan.get("is_complete_for_real_trade"))
            persisted_size = cls._safe_float(persisted_plan.get("position_size_pct"), 0.0)
            if existing_lane != "shadow_only" and persisted_complete and persisted_size > 0:
                return {
                    "lane": existing_lane,
                    "promotion_reasons": persisted_plan.get("promotion_reasons") or [],
                    "downgrade_reasons": persisted_plan.get("block_or_downgrade_reasons") or [],
                    "independent_source_count": int(
                        cls._safe_float(persisted_plan.get("independent_source_count"), 0.0)
                    ),
                    "source": "persisted_profit_first_plan",
                }
            ignored_persisted_plan = {
                "lane": existing_lane,
                "is_complete_for_real_trade": persisted_complete,
                "missing_required_fields": missing_fields[:8],
                "position_size_pct": round(persisted_size, 6),
                "reason": "stale_or_incomplete_profit_first_plan_reclassified_after_sizing",
            }

        opportunity = cls._safe_dict(raw.get("opportunity_score"))
        evidence = cls._safe_dict(opportunity.get("evidence_score"))
        component_sources = {
            str(component.get("source") or "").strip()
            for component in cls._safe_list(evidence.get("components"))
            if isinstance(component, dict)
            and str(component.get("status") or "").lower() == "aligned"
            and str(component.get("source") or "").strip()
        }
        independent_source_count = max(1 + int(aligned_source_count or 0), 1 + len(component_sources))
        if cls._safe_dict(raw.get("high_risk_review")).get("approved") is True:
            independent_source_count += 1
        lane, promotion, downgrade = classify_decision_lane(
            expected_net_return_pct=expected_net_return_pct,
            expected_profit_usdt=expected_profit_usdt,
            profit_quality_ratio=profit_quality_ratio,
            reward_risk_ratio=reward_risk_ratio,
            loss_probability=loss_probability,
            tail_loss_probability=tail_risk_score,
            independent_source_count=independent_source_count,
            missing_required_fields=[],
            high_risk_review=cls._safe_dict(raw.get("high_risk_review")),
            recent_realized_edge=cls._safe_float(
                opportunity.get(
                    "recent_realized_edge",
                    opportunity.get("side_realized_pnl_usdt", 0.0),
                ),
                0.0,
            ),
            is_entry=decision.is_entry,
        )
        return {
            "lane": lane,
            "promotion_reasons": promotion,
            "downgrade_reasons": downgrade,
            "independent_source_count": independent_source_count,
            "source": "sizing_profit_first_classifier",
            "ignored_persisted_plan": ignored_persisted_plan,
        }

    @classmethod
    def _portfolio_leverage_context(cls, open_positions: list[dict] | None) -> dict[str, Any]:
        positions = [position for position in open_positions or [] if isinstance(position, dict)]
        exposure = 0.0
        for position in positions:
            exposure += max(
                cls._safe_float(
                    position.get("position_size_pct")
                    or position.get("margin_pct")
                    or position.get("size_pct")
                    or position.get("margin_ratio")
                    or position.get("notional_pct"),
                    0.0,
                ),
                0.0,
            )
        return {
            "open_positions_count": len(positions),
            "portfolio_exposure_pct": exposure,
        }

    @classmethod
    def _execution_cost_context(cls, raw: dict[str, Any], opportunity: dict[str, Any]) -> dict[str, Any]:
        contexts = (
            opportunity.get("execution_cost"),
            opportunity.get("cost_snapshot"),
            raw.get("execution_cost"),
            raw.get("market_cost"),
        )
        for context in contexts:
            if isinstance(context, dict):
                return context
        return {}

    def _missing_dependencies(self) -> list[str]:
        required = {
            "allocated_order_balance": self.allocated_order_balance,
            "entry_low_payoff_quality": self.entry_low_payoff_quality,
            "entry_stop_loss_budget": self.entry_stop_loss_budget,
            "entry_stress_stop": self.entry_stress_stop,
            "entry_existing_winner_context": self.entry_existing_winner_context,
        }
        return [name for name, value in required.items() if value is None]

    async def apply(
        self,
        decision: DecisionOutput,
        model_mode: str,
        open_positions: list[dict[str, Any]] | None = None,
    ) -> None:
        """Apply profit-risk sizing to an entry decision."""

        if self.evaluator is not None:
            await self.evaluator(decision, model_mode, open_positions or [])
            return
        missing = self._missing_dependencies()
        if missing:
            raise RuntimeError(
                "EntryProfitRiskSizingPolicy requires dependencies: " + ", ".join(missing)
            )
        await self._apply_sizing(decision, model_mode, open_positions or [])

    async def _apply_sizing(
        self,
        decision: DecisionOutput,
        model_mode: str,
        open_positions: list[dict] | None = None,
    ) -> None:
        """Cap entry size by planned USDT loss at stop, especially during drawdown recovery."""
        if not decision.is_entry:
            return
        raw: dict[str, Any] = self._safe_dict(decision.raw_response)
        opportunity: dict[str, Any] = self._safe_dict(raw.get("opportunity_score"))
        risk_mode = str(opportunity.get("risk_mode") or "normal")
        configured_max_loss = self._safe_float(opportunity.get("max_entry_stop_loss_usdt"), 0.0)

        stop_loss_pct = max(float(decision.stop_loss_pct or 0.0), 0.0)
        leverage = max(float(decision.suggested_leverage or 1.0), 1.0)
        current_size = max(float(decision.position_size_pct or 0.0), 0.0)
        if stop_loss_pct <= 0 or current_size <= 0:
            return
        weak_history = bool(opportunity.get("weak_history_requires_stronger_edge"))
        local_expected = self._safe_float(opportunity.get("server_profit_expected_return_pct"), 0.0)
        local_aligned = bool(opportunity.get("local_profit_aligned"))
        ml_aligned = bool(opportunity.get("ml_aligned"))
        profit_quality_ratio = self._safe_float(opportunity.get("profit_quality_ratio"), 0.0)
        min_profit_quality_ratio = self._safe_float(
            opportunity.get("min_profit_quality_ratio_required"),
            ENTRY_MIN_NET_PROFIT_QUALITY_RATIO,
        )
        expected_net = self._safe_float(opportunity.get("expected_net_return_pct"), 0.0)
        tail_risk = self._safe_float(opportunity.get("tail_risk_score"), 0.0)
        selected_metrics = selected_entry_metrics(decision)
        if selected_metrics.has_selected_side:
            expected_net = selected_metrics.expected_net_return_pct
            profit_quality_ratio = selected_metrics.profit_quality_ratio
            tail_risk = selected_metrics.tail_risk_score
        score = self._safe_float(opportunity.get("score"), 0.0)
        min_score_required = self._safe_float(
            opportunity.get("min_score_required"),
            MIN_ENTRY_OPPORTUNITY_SCORE,
        )
        raw_expected_return = self._safe_float(
            opportunity.get("raw_expected_return_pct", opportunity.get("expected_return_pct")),
            0.0,
        )
        expected_loss_pct = self._safe_float(opportunity.get("expected_loss_pct"), 0.0)
        small_win_big_loss_penalty = self._safe_float(
            opportunity.get("small_win_big_loss_penalty"),
            0.0,
        )
        loss_probability = self._safe_float(opportunity.get("server_profit_loss_probability"), 1.0)
        if selected_metrics.has_selected_side:
            loss_probability = selected_metrics.loss_probability
        contribution_adjustment = self._safe_dict(opportunity.get("model_contribution_adjustment"))
        hard_contribution_caution = bool(contribution_adjustment.get("hard_caution"))
        quant_probe = self._safe_dict(raw.get("quant_profit_probe"))
        evidence_probe = self._safe_dict(raw.get("evidence_profit_probe"))
        quant_probe_triggered = bool(quant_probe.get("triggered"))
        evidence_probe_triggered = bool(evidence_probe.get("triggered"))
        strong_probe = bool(
            quant_probe.get("strong_probe") or evidence_probe.get("high_profit_potential")
        )
        roster_fill_relief = self._safe_dict(opportunity.get("portfolio_roster_fill_relief"))
        roster = self._safe_dict(opportunity.get("portfolio_roster"))
        symbol_profit_tier = str(opportunity.get("symbol_profit_tier") or "neutral")
        roster_fill_candidate = bool(
            roster_fill_relief.get("applied")
            or quant_probe.get("roster_fill_probe")
            or (
                roster.get("underfilled")
                and (quant_probe_triggered or evidence_probe_triggered)
                and not strong_probe
            )
        )
        if quant_probe_triggered:
            loss_probability = self._safe_float(
                quant_probe.get("loss_probability"),
                loss_probability,
            )
        if evidence_probe_triggered:
            loss_probability = self._safe_float(
                evidence_probe.get("loss_probability"),
                loss_probability,
            )
        high_quality_boost = self._safe_dict(raw.get("profit_quality_position_boost"))
        high_quality_entry = bool(
            high_quality_boost.get("allow")
            or (
                expected_net > 0
                and profit_quality_ratio >= max(min_profit_quality_ratio, 0.85)
                and tail_risk < 0.88
                and (local_aligned or ml_aligned or bool(opportunity.get("timeseries_aligned")))
            )
        )
        caps: list[str] = []
        evidence_score = self._safe_dict(opportunity.get("evidence_score"))
        evidence_sizing = apply_evidence_sizing_policy(
            evidence_score=evidence_score,
            current_size=current_size,
            leverage=leverage,
        )
        current_size = evidence_sizing.position_size_pct
        leverage = evidence_sizing.leverage
        decision.position_size_pct = current_size
        decision.suggested_leverage = leverage
        caps.extend(evidence_sizing.caps)
        evidence_effective_score = evidence_sizing.effective_score
        if weak_history:
            weak_history_max_size = (
                ENTRY_WEAK_HISTORY_STRONG_ALIGNED_MAX_SIZE
                if high_quality_entry
                else ENTRY_WEAK_HISTORY_MAX_SIZE
            )
            if current_size > weak_history_max_size:
                current_size = weak_history_max_size
                decision.position_size_pct = current_size
                caps.append(
                    "近期该币种/方向真实盈亏偏弱，但当前盈利证据同向，仓位按中小仓验证"
                    if high_quality_entry
                    else "近期该币种/方向真实盈亏偏弱，仓位降为小仓验证"
                )
        negative_local_expected = bool(local_expected < 0 and not (local_aligned or ml_aligned))
        if negative_local_expected:
            if current_size > ENTRY_NEGATIVE_LOCAL_EXPECTED_MAX_SIZE:
                current_size = ENTRY_NEGATIVE_LOCAL_EXPECTED_MAX_SIZE
                decision.position_size_pct = current_size
                caps.append("服务器盈利模型反向或预期为负，仅允许极小仓")
        strategy_sizing = self._strategy_learning_sizing(raw)
        timeseries_aligned = bool(opportunity.get("timeseries_aligned"))
        legacy_aligned_source_count = sum(
            1
            for aligned in (
                local_aligned,
                ml_aligned,
                timeseries_aligned,
                bool(opportunity.get("expert_aligned")),
            )
            if aligned
        )
        evidence_aligned_source_count = len(
            {
                str(source)
                for source in self._safe_list(evidence_score.get("aligned_support_sources"))
                if str(source or "").strip()
            }
        )
        aligned_source_count = max(legacy_aligned_source_count, evidence_aligned_source_count)
        roster_fill_quality = bool(
            roster_fill_candidate
            and expected_net >= PORTFOLIO_ROSTER_FILL_MIN_NET_PCT
            and profit_quality_ratio >= PORTFOLIO_ROSTER_FILL_MIN_PROFIT_QUALITY_RATIO
            and loss_probability <= PORTFOLIO_ROSTER_FILL_MAX_LOSS_PROBABILITY
            and tail_risk <= 0.88
            and (aligned_source_count >= 3 or bool(roster_fill_relief.get("applied")))
            and (
                local_aligned
                or ml_aligned
                or timeseries_aligned
                or quant_probe_triggered
                or evidence_aligned_source_count >= 3
                or bool(roster_fill_relief.get("applied"))
            )
        )
        low_payoff_reasons = self.entry_low_payoff_quality.reasons(
            score=score,
            min_score_required=min_score_required,
            expected_net_return_pct=expected_net,
            profit_quality_ratio=profit_quality_ratio,
            raw_expected_return_pct=raw_expected_return,
            small_win_big_loss_penalty=small_win_big_loss_penalty,
            hard_contribution_caution=hard_contribution_caution,
            evidence_score=evidence_score,
            evidence_effective_score=evidence_effective_score,
        )
        low_payoff_relief: dict[str, Any] = {"applied": False}
        if roster_fill_quality and low_payoff_reasons:
            roster_fill_soft_reasons = {
                "score_below_required",
                "expected_net_below_min",
                "profit_quality_below_min",
                "hard_contribution_caution",
                "evidence_low_payoff_quality",
            }
            remaining_low_payoff_reasons = [
                reason
                for reason in low_payoff_reasons
                if reason not in roster_fill_soft_reasons
            ]
            relieved_low_payoff_reasons = [
                reason for reason in low_payoff_reasons if reason in roster_fill_soft_reasons
            ]
            if relieved_low_payoff_reasons:
                low_payoff_relief = {
                    "applied": True,
                    "reason": "portfolio_roster_fill_quality",
                    "relieved_reasons": relieved_low_payoff_reasons,
                    "remaining_reasons": remaining_low_payoff_reasons,
                    "expected_net_return_pct": round(expected_net, 6),
                    "profit_quality_ratio": round(profit_quality_ratio, 6),
                    "loss_probability": round(loss_probability, 6),
                    "tail_risk_score": round(tail_risk, 6),
                    "aligned_source_count": aligned_source_count,
                }
                low_payoff_reasons = remaining_low_payoff_reasons
        low_payoff_quality = bool(low_payoff_reasons)
        if low_payoff_quality:
            high_quality_entry = False
            if current_size > ENTRY_LOW_QUALITY_MAX_SIZE:
                current_size = ENTRY_LOW_QUALITY_MAX_SIZE
                decision.position_size_pct = current_size
                caps.append("收益质量不足或存在小盈大亏风险，仓位降为小仓验证")
        if symbol_profit_tier == "side_loser" and not high_quality_entry:
            loser_size_cap = max(
                ENTRY_LOW_QUALITY_MAX_SIZE, current_size * ENTRY_SYMBOL_LOSER_SIZE_MULTIPLIER
            )
            if current_size > loser_size_cap:
                current_size = loser_size_cap
                decision.position_size_pct = current_size
                caps.append("该币种同方向近期真实亏损，未达到高质量解锁前缩小仓位")
        quality_override_reasons: list[str] = []
        strong_positive_strategy_signal = bool(
            not low_payoff_quality
            and aligned_source_count >= 2
            and score >= max(min_score_required, 1.0)
            and expected_net >= 0.90
            and profit_quality_ratio >= 0.80
            and loss_probability <= 0.46
            and tail_risk <= ENTRY_MEANINGFUL_SIZE_MAX_TAIL_RISK
        )
        if strong_positive_strategy_signal:
            quality_override_reasons.append("strong_positive_strategy_signal")
        high_quality_strategy_signal = bool(
            high_quality_entry
            and not low_payoff_quality
            and aligned_source_count >= 2
            and expected_net >= 0.70
            and profit_quality_ratio >= 0.75
            and loss_probability <= 0.48
            and tail_risk <= ENTRY_MEANINGFUL_SIZE_MAX_TAIL_RISK
        )
        if high_quality_strategy_signal:
            quality_override_reasons.append("high_quality_entry")
        strategy_quality_override = bool(
            strong_positive_strategy_signal
            or high_quality_strategy_signal
            or (
                high_quality_entry
                and not low_payoff_quality
                and (local_aligned or ml_aligned or timeseries_aligned)
                and (
                    expected_net >= 0.90
                    or profit_quality_ratio >= max(min_profit_quality_ratio, 1.05)
                )
                and loss_probability <= 0.46
                and tail_risk <= ENTRY_MEANINGFUL_SIZE_MAX_TAIL_RISK
            )
        )
        if strategy_quality_override and not quality_override_reasons:
            quality_override_reasons.append("quality_profit_risk_override")
        recovery_quality_cap_pct = self._adaptive_recovery_probe_cap_pct(
            expected_net_return_pct=expected_net,
            profit_quality_ratio=profit_quality_ratio,
            loss_probability=loss_probability,
            tail_risk_score=tail_risk,
            score=score,
            min_score_required=min_score_required,
            aligned_source_count=aligned_source_count,
        )
        strategy_sizing_applied = self._apply_strategy_learning_sizing(
            current_size=current_size,
            action_side="long" if str(decision.action.value) == "long" else "short",
            sizing=strategy_sizing,
            quality_override=strategy_quality_override,
            recovery_quality_cap_pct=recovery_quality_cap_pct,
        )
        if strategy_sizing_applied.get("applied"):
            current_size = self._safe_float(
                strategy_sizing_applied.get("position_size_pct"), current_size
            )
            decision.position_size_pct = current_size
            caps.append("strategy learning bounded sizing applied")
        balance = await self.allocated_order_balance(model_mode, decision)
        if balance <= 0:
            return
        stop_loss_budget = self.entry_stop_loss_budget.resolve(
            risk_mode=risk_mode,
            configured_max_loss_usdt=configured_max_loss,
            balance=balance,
            high_quality_entry=high_quality_entry,
            low_payoff_quality=low_payoff_quality,
        )
        max_loss = stop_loss_budget.max_loss_usdt
        risk_budget_boost = stop_loss_budget.risk_budget_boost
        probe_budget_guard: dict[str, Any] = {"applied": False}
        if (quant_probe_triggered or evidence_probe_triggered) and not high_quality_entry:
            probe_budget = (
                self.strong_probe_max_loss_usdt if strong_probe else self.probe_max_loss_usdt
            )
            if 0.0 < probe_budget < max_loss:
                probe_budget_guard = {
                    "applied": True,
                    "strong_probe": bool(strong_probe),
                    "previous_max_stop_loss_usdt": round(max_loss, 6),
                    "max_stop_loss_usdt": round(probe_budget, 6),
                    "reason": (
                        "\u63a2\u9488\u6863\u4f7f\u7528\u72ec\u7acb\u7684\u5c0f\u98ce\u9669\u9884\u7b97"
                        "\uff0c\u5355\u7b14\u6700\u5927\u4e8f\u635f\u9650\u5728\u63a2\u9488\u9884\u7b97\u5185"
                        "\uff0c\u4ee5\u4fdd\u6301\u6210\u4ea4\u91cf\u7684\u540c\u65f6\u63a7\u4f4f\u5355\u6b21\u635f\u5931\u3002"
                    ),
                }
                max_loss = probe_budget
        pnl_structure_guard: dict[str, Any] = {
            "applied": False,
            "expected_net_return_pct": round(expected_net, 6),
            "profit_quality_ratio": round(profit_quality_ratio, 6),
        }
        snapshot = decision.feature_snapshot if isinstance(decision.feature_snapshot, dict) else {}
        atr_14 = self._safe_float(snapshot.get("atr_14"), 0.0)
        current_price_for_atr = self._safe_float(
            snapshot.get("current_price", snapshot.get("close", 0.0)),
            0.0,
        )
        atr_pct = (
            atr_14 / current_price_for_atr if atr_14 > 0 and current_price_for_atr > 0 else 0.0
        )
        stress_stop_loss_pct = self.entry_stress_stop.stress_stop_loss_pct(
            declared_stop_loss_pct=stop_loss_pct,
            expected_loss_pct=expected_loss_pct,
            tail_risk_score=tail_risk,
            raw_expected_return_pct=raw_expected_return,
            low_payoff_quality=low_payoff_quality,
            atr_pct=atr_pct,
        )

        original_size_before_floor = current_size
        original_notional = balance * current_size * leverage
        target_min_notional = 0.0
        notional_floor_ratio = 0.0
        notional_floor_reason = ""
        quality_tier = "probe" if (quant_probe_triggered or evidence_probe_triggered) else "base"
        meaningful_size_reason = ""
        existing_winner = self.entry_existing_winner_context.context(
            decision,
            open_positions or [],
        )
        has_existing_winner = bool(existing_winner.get("has_winner"))
        strong_probe_quality = bool(
            strong_probe
            and expected_net >= _ENTRY_RISK_SIZING_PARAMS.strong_probe_min_expected_net
            and profit_quality_ratio >= _ENTRY_RISK_SIZING_PARAMS.strong_probe_min_profit_quality
            and loss_probability <= _ENTRY_RISK_SIZING_PARAMS.strong_probe_max_loss_probability
            and (
                score >= _ENTRY_RISK_SIZING_PARAMS.strong_probe_min_score_floor
                or evidence_probe_triggered
            )
        )
        elite_quality = bool(
            expected_net >= _ENTRY_RISK_SIZING_PARAMS.elite_min_expected_net
            and profit_quality_ratio >= _ENTRY_RISK_SIZING_PARAMS.elite_min_profit_quality
            and loss_probability <= _ENTRY_RISK_SIZING_PARAMS.elite_max_loss_probability
            and tail_risk <= ENTRY_MEANINGFUL_SIZE_MAX_TAIL_RISK
            and (local_aligned or ml_aligned or timeseries_aligned)
        )
        high_profit_quality = bool(
            expected_net >= _ENTRY_RISK_SIZING_PARAMS.high_profit_min_expected_net
            and profit_quality_ratio >= _ENTRY_RISK_SIZING_PARAMS.high_profit_min_profit_quality
            and loss_probability <= _ENTRY_RISK_SIZING_PARAMS.high_profit_max_loss_probability
            and tail_risk <= _ENTRY_RISK_SIZING_PARAMS.high_profit_max_tail_risk
            and score
            >= max(
                self._safe_float(
                    opportunity.get("min_score_required"), MIN_ENTRY_OPPORTUNITY_SCORE
                ),
                _ENTRY_RISK_SIZING_PARAMS.high_profit_min_score_floor,
            )
            and (local_aligned or ml_aligned or timeseries_aligned)
        )
        meaningful_quality_override = bool(
            strategy_quality_override
            and expected_net >= _ENTRY_RISK_SIZING_PARAMS.quality_override_min_expected_net
            and profit_quality_ratio
            >= _ENTRY_RISK_SIZING_PARAMS.quality_override_min_profit_quality
            and loss_probability <= _ENTRY_RISK_SIZING_PARAMS.quality_override_max_loss_probability
            and tail_risk <= ENTRY_MEANINGFUL_SIZE_MAX_TAIL_RISK
        )
        good_probe_quality = bool(
            (quant_probe_triggered or evidence_probe_triggered)
            and expected_net >= _ENTRY_RISK_SIZING_PARAMS.good_probe_min_expected_net
            and profit_quality_ratio >= _ENTRY_RISK_SIZING_PARAMS.good_probe_min_profit_quality
            and loss_probability <= _ENTRY_RISK_SIZING_PARAMS.good_probe_max_loss_probability
            and tail_risk <= ENTRY_MEANINGFUL_SIZE_MAX_TAIL_RISK
        )
        if has_existing_winner and (
            strong_probe_quality
            or elite_quality
            or (
                expected_net >= _ENTRY_RISK_SIZING_PARAMS.same_side_winner_min_expected_net
                and profit_quality_ratio
                >= _ENTRY_RISK_SIZING_PARAMS.same_side_winner_min_profit_quality
                and loss_probability
                <= _ENTRY_RISK_SIZING_PARAMS.same_side_winner_max_loss_probability
            )
        ):
            current_profit = self._safe_float(existing_winner.get("unrealized_pnl"), 0.0)
            profit_ratio = self._safe_float(existing_winner.get("pnl_ratio"), 0.0)
            if (
                current_profit >= ENTRY_MEANINGFUL_SIZE_MIN_PROFIT_USDT
                or profit_ratio >= ENTRY_MEANINGFUL_SIZE_MIN_PROFIT_RATIO
            ):
                quality_tier = "winner_add"
                notional_floor_ratio = ENTRY_WINNER_ADD_MIN_NOTIONAL_BALANCE_RATIO
                meaningful_size_reason = (
                    "已有同币种同方向持仓浮盈，且新信号仍然同向；允许用中等仓位加到赢家上，"
                    "把正确判断转成更有意义的利润。"
                )
        if quality_tier != "winner_add" and elite_quality:
            quality_tier = "elite"
            notional_floor_ratio = ENTRY_ELITE_MIN_NOTIONAL_BALANCE_RATIO
            meaningful_size_reason = (
                "净收益、盈亏质量和亏损概率同时达到精选标准，使用精选仓位地板。"
            )
        if (
            high_profit_quality
            and not low_payoff_quality
            and quality_tier in {"base", "probe", "good_probe", "strong_probe", "elite"}
        ):
            quality_tier = "high_profit"
            notional_floor_ratio = max(
                notional_floor_ratio, ENTRY_HIGH_PROFIT_MIN_NOTIONAL_BALANCE_RATIO
            )
            meaningful_size_reason = (
                "盈利可能性较大：预期净收益、盈亏质量、亏损概率和尾部风险同时达标，"
                "允许适当提高交易数量和杠杆，把高质量机会转成更大的实际收益。"
            )
        elif (
            quality_tier not in {"winner_add", "elite"}
            and strong_probe_quality
            and not low_payoff_quality
        ):
            quality_tier = "strong_probe"
            notional_floor_ratio = ENTRY_STRONG_PROBE_MIN_NOTIONAL_BALANCE_RATIO
            meaningful_size_reason = (
                "强量化探针通过净收益、盈亏质量、亏损概率和机会分联合校验，升级为有效仓位。"
            )
        elif quality_tier == "probe" and good_probe_quality and not low_payoff_quality:
            quality_tier = "good_probe"
            notional_floor_ratio = ENTRY_GOOD_PROBE_MIN_NOTIONAL_BALANCE_RATIO
            meaningful_size_reason = (
                "普通量化探针质量达标，不再只做极小验证仓，抬到可产生有效收益的基础仓位。"
            )
        elif (
            quality_tier in {"probe", "base"}
            and meaningful_quality_override
            and not low_payoff_quality
        ):
            quality_tier = "quality_override"
            notional_floor_ratio = max(
                notional_floor_ratio,
                ENTRY_HIGH_QUALITY_MIN_NOTIONAL_BALANCE_RATIO,
            )
            meaningful_size_reason = (
                "策略学习处于释放或恢复压力，但当前净收益、盈亏质量、亏损概率和多源同向证据达标；"
                "按质量驱动仓位执行，不再被旧的小仓探针档长期压住。"
            )
        elif quality_tier in {"probe", "base"} and roster_fill_quality and not low_payoff_quality:
            quality_tier = "roster_fill"
            notional_floor_ratio = PORTFOLIO_ROSTER_FILL_NOTIONAL_BALANCE_RATIO
            meaningful_size_reason = (
                "当前组合低于目标持仓组数，且该信号仍为正期望；使用小而不碎的补齐仓位，"
                "让组合逐步接近 10 个独立持仓组。"
            )

        if notional_floor_ratio > 0:
            target_min_notional = balance * notional_floor_ratio
            notional_floor_reason = meaningful_size_reason
            high_quality_entry = high_quality_entry or quality_tier in {
                "elite",
                "strong_probe",
                "winner_add",
                "high_profit",
                "quality_override",
            }
            if symbol_profit_tier in {"side_winner", "symbol_winner"} and quality_tier in {
                "base",
                "probe",
                "good_probe",
                "roster_fill",
            }:
                target_min_notional *= _ENTRY_RISK_SIZING_PARAMS.winner_notional_lift_multiplier
                notional_floor_reason = (
                    f"{notional_floor_reason} 该币种近期真实盈利，名义本金地板小幅上调。"
                    if notional_floor_reason
                    else "该币种近期真实盈利，名义本金地板小幅上调。"
                )

        if high_quality_entry and not low_payoff_quality:
            quality_reference = max(
                min_profit_quality_ratio,
                _ENTRY_RISK_SIZING_PARAMS.strong_probe_min_profit_quality,
            )
            quality_multiplier = min(
                max(
                    profit_quality_ratio / max(quality_reference, 1e-12),
                    _ENTRY_RISK_SIZING_PARAMS.high_quality_floor_min_multiplier,
                ),
                _ENTRY_RISK_SIZING_PARAMS.high_quality_floor_max_multiplier,
            )
            default_floor_ratio = ENTRY_HIGH_QUALITY_MIN_NOTIONAL_BALANCE_RATIO * quality_multiplier
            if default_floor_ratio > notional_floor_ratio:
                notional_floor_ratio = default_floor_ratio
                target_min_notional = balance * notional_floor_ratio
                notional_floor_reason = (
                    "高质量盈利证据同向，按当前可用资金和信号质量动态抬高名义本金，避免无意义小盈"
                )
        elif (
            expected_net > 0
            and profit_quality_ratio >= _ENTRY_RISK_SIZING_PARAMS.normal_positive_min_profit_quality
            and (local_aligned or ml_aligned or timeseries_aligned)
        ):
            quality_multiplier = min(
                max(
                    profit_quality_ratio
                    / _ENTRY_RISK_SIZING_PARAMS.normal_positive_min_profit_quality,
                    _ENTRY_RISK_SIZING_PARAMS.normal_positive_floor_min_multiplier,
                ),
                _ENTRY_RISK_SIZING_PARAMS.normal_positive_floor_max_multiplier,
            )
            notional_floor_ratio = ENTRY_NORMAL_MIN_NOTIONAL_BALANCE_RATIO * quality_multiplier
            target_min_notional = balance * notional_floor_ratio
            notional_floor_reason = (
                "普通正收益信号获得本地模型或时序模型支持，按当前可用资金动态设置基础名义本金"
            )

        intended_notional_for_profit = max(original_notional, target_min_notional)
        expected_profit_usdt = intended_notional_for_profit * max(expected_net, 0.0) / 100.0
        if expected_net > 0 and intended_notional_for_profit > 0:
            max_loss_multiple = (
                ENTRY_PNL_STRUCTURE_LOW_QUALITY_MAX_LOSS_MULTIPLE
                if low_payoff_quality or symbol_profit_tier == "side_loser"
                else (
                    ENTRY_PNL_STRUCTURE_HIGH_QUALITY_MAX_LOSS_MULTIPLE
                    if high_quality_entry
                    or quality_tier
                    in {"elite", "winner_add", "high_profit", "strong_probe", "quality_override"}
                    else ENTRY_PNL_STRUCTURE_NORMAL_MAX_LOSS_MULTIPLE
                )
            )
            structure_max_loss = max(
                ENTRY_PNL_STRUCTURE_MIN_EXPECTED_PROFIT_USDT,
                expected_profit_usdt * max_loss_multiple,
            )
            if structure_max_loss < max_loss:
                previous_max_loss = max_loss
                max_loss = max(structure_max_loss, 1.0)
                pnl_structure_guard = {
                    "applied": True,
                    "previous_max_stop_loss_usdt": round(previous_max_loss, 6),
                    "max_stop_loss_usdt": round(max_loss, 6),
                    "expected_profit_usdt": round(expected_profit_usdt, 6),
                    "expected_net_return_pct": round(expected_net, 6),
                    "max_loss_multiple": round(max_loss_multiple, 6),
                    "quality_tier": quality_tier,
                    "symbol_profit_tier": symbol_profit_tier,
                    "reason": "按预期净收益动态压缩单笔止损预算，避免小盈大亏结构。",
                }
            elif (
                high_quality_entry
                and not low_payoff_quality
                and target_min_notional > 0
                and quality_tier in {"elite", "winner_add", "high_profit", "strong_probe"}
            ):
                required_floor_loss = target_min_notional * stress_stop_loss_pct
                adaptive_cap_pct = self._adaptive_quality_loss_cap_pct(
                    expected_net_return_pct=expected_net,
                    profit_quality_ratio=profit_quality_ratio,
                    loss_probability=loss_probability,
                    tail_risk_score=tail_risk,
                    quality_tier=quality_tier,
                )
                adaptive_cap_usdt = balance * adaptive_cap_pct
                target_quality_budget = min(
                    max(required_floor_loss, structure_max_loss),
                    adaptive_cap_usdt,
                )
                if target_quality_budget > max_loss:
                    previous_max_loss = max_loss
                    max_loss = target_quality_budget
                    pnl_structure_guard = {
                        "applied": True,
                        "previous_max_stop_loss_usdt": round(previous_max_loss, 6),
                        "max_stop_loss_usdt": round(max_loss, 6),
                        "expected_profit_usdt": round(expected_profit_usdt, 6),
                        "expected_net_return_pct": round(expected_net, 6),
                        "max_loss_multiple": round(max_loss_multiple, 6),
                        "quality_tier": quality_tier,
                        "symbol_profit_tier": symbol_profit_tier,
                        "adaptive_cap_pct_of_equity": round(adaptive_cap_pct, 6),
                        "adaptive_cap_usdt": round(adaptive_cap_usdt, 6),
                        "required_floor_loss_usdt": round(required_floor_loss, 6),
                        "reason": (
                            "高质量正期望信号按收益质量动态放宽单笔止损预算，"
                            "避免策略学习或固定权益上限把有效机会压成无意义小仓。"
                        ),
                    }

        notional_floor_blocked = ""
        if target_min_notional > 0:
            if low_payoff_quality:
                notional_floor_blocked = "收益质量不足或小盈大亏风险偏高，不抬高仓位"
            elif tail_risk >= 0.88:
                notional_floor_blocked = "尾部风险偏高，不抬高仓位"
            elif weak_history and not high_quality_entry:
                notional_floor_blocked = "该币种/方向近期真实盈亏偏弱，普通信号不抬高仓位"
            elif local_expected < 0 and not (local_aligned or ml_aligned):
                notional_floor_blocked = (
                    "服务器盈利模型预期为负且未获得本地模型同向支持，不抬高仓位"
                )
            else:
                risk_max_size = max_loss / max(balance * leverage * stress_stop_loss_pct, 1e-12)
                floor_size = target_min_notional / max(balance * leverage, 1e-12)
                raised_size = min(
                    max(current_size, floor_size),
                    max(risk_max_size, 0.001),
                    ENTRY_NOTIONAL_FLOOR_MAX_SIZE_PCT,
                )
                if raised_size > current_size:
                    current_size = raised_size
                    decision.position_size_pct = current_size

        profit_first_lane = self._profit_first_lane_for_sizing(
            raw=raw,
            decision=decision,
            expected_net_return_pct=expected_net,
            expected_profit_usdt=expected_profit_usdt,
            profit_quality_ratio=profit_quality_ratio,
            reward_risk_ratio=self._safe_float(opportunity.get("reward_risk_ratio"), 0.0),
            loss_probability=loss_probability,
            tail_risk_score=tail_risk,
            aligned_source_count=aligned_source_count,
        )
        position_ladder_policy = self.profit_first_position_ladder or ProfitFirstPositionLadderPolicy()
        position_ladder_decision = position_ladder_policy.apply(
            lane=str(profit_first_lane.get("lane") or ""),
            current_size_pct=current_size,
            low_payoff_quality=low_payoff_quality,
            high_risk_review=self._safe_dict(raw.get("high_risk_review")),
        )
        profit_first_position_ladder = {
            **position_ladder_decision.to_dict(),
            "classifier": profit_first_lane,
            "pre_stop_budget_size_pct": round(current_size, 6),
            "low_payoff_quality": bool(low_payoff_quality),
        }
        if abs(position_ladder_decision.adjusted_size_pct - current_size) > 1e-9:
            current_size = position_ladder_decision.adjusted_size_pct
            decision.position_size_pct = current_size
            caps.append("profit-first position ladder applied")

        portfolio_leverage_context = self._portfolio_leverage_context(open_positions or [])
        dynamic_leverage_policy = self.dynamic_leverage_allocator or DynamicLeverageAllocator()
        dynamic_leverage_decision = dynamic_leverage_policy.allocate(
            DynamicLeverageInput(
                symbol=decision.symbol,
                requested_leverage=leverage,
                system_max_leverage=self.max_leverage_provider(),
                balance=balance,
                position_size_pct=current_size,
                stress_stop_loss_pct=stress_stop_loss_pct,
                max_loss_usdt=max_loss,
                expected_net_return_pct=expected_net,
                profit_quality_ratio=profit_quality_ratio,
                loss_probability=loss_probability,
                tail_risk_score=tail_risk,
                score=score,
                min_score_required=min_score_required,
                confidence=float(decision.confidence or 0.0),
                aligned_source_count=aligned_source_count,
                evidence_tier=evidence_sizing.tier,
                evidence_effective_score=evidence_effective_score,
                low_payoff_quality=low_payoff_quality,
                weak_history=weak_history,
                negative_local_expected=negative_local_expected,
                symbol_profit_tier=symbol_profit_tier,
                quality_tier=quality_tier,
                high_quality_entry=high_quality_entry,
                atr_pct=atr_pct,
                execution_cost=self._execution_cost_context(raw, opportunity),
                open_positions_count=int(
                    portfolio_leverage_context.get("open_positions_count") or 0
                ),
                portfolio_exposure_pct=self._safe_float(
                    portfolio_leverage_context.get("portfolio_exposure_pct"), 0.0
                ),
            )
        )
        leverage = float(dynamic_leverage_decision.final_integer_leverage)
        decision.suggested_leverage = leverage
        dynamic_leverage_payload = dynamic_leverage_decision.to_dict()
        caps.append(
            "dynamic leverage allocator applied: "
            f"{dynamic_leverage_payload['final_integer_leverage']}x "
            f"({dynamic_leverage_payload['limiting_factor']})"
        )

        planned_loss = balance * current_size * leverage * stress_stop_loss_pct
        max_size_pct = max_loss / max(balance * leverage * stress_stop_loss_pct, 1e-12)
        max_size_pct = max(min(max_size_pct, current_size), 0.001)
        final_size_pct = max_size_pct if planned_loss > max_loss else current_size
        if planned_loss > max_loss:
            profit_first_position_ladder["capped_by_stop_loss_budget"] = True
            profit_first_position_ladder["post_stop_budget_size_pct"] = round(max_size_pct, 6)
        else:
            profit_first_position_ladder["capped_by_stop_loss_budget"] = False
            profit_first_position_ladder["post_stop_budget_size_pct"] = round(current_size, 6)
        raw["profit_risk_sizing"] = {
            "applied": planned_loss > max_loss,
            "risk_mode": risk_mode,
            "original_position_size_pct": round(original_size_before_floor, 6),
            "position_size_pct": round(final_size_pct, 6),
            "planned_stop_loss_usdt": round(planned_loss, 6),
            "max_stop_loss_usdt": round(max_loss, 6),
            "declared_stop_loss_pct": round(stop_loss_pct, 6),
            "stress_stop_loss_pct": round(stress_stop_loss_pct, 6),
            "atr_14": round(atr_14, 8),
            "atr_pct": round(atr_pct, 8),
            "low_payoff_quality": bool(low_payoff_quality),
            "low_payoff_reasons": low_payoff_reasons,
            "low_payoff_relief": low_payoff_relief,
            "quality_caps": caps,
            "evidence_score": evidence_score,
            "high_quality_entry": bool(high_quality_entry),
            "high_profit_quality": bool(high_profit_quality),
            "quality_tier": quality_tier,
            "meaningful_size_reason": meaningful_size_reason,
            "same_side_existing_winner": existing_winner,
            "risk_budget_boost": risk_budget_boost,
            "probe_budget_guard": probe_budget_guard,
            "profit_first_position_ladder": profit_first_position_ladder,
            "dynamic_leverage_decision": dynamic_leverage_payload,
            "pnl_structure_guard": pnl_structure_guard,
            "strategy_learning_sizing": strategy_sizing_applied,
            "strategy_quality_override": bool(strategy_quality_override),
            "strategy_quality_override_reasons": quality_override_reasons,
            "aligned_source_count": aligned_source_count,
            "notional_floor_applied": current_size > original_size_before_floor,
            "original_notional_usdt": round(original_notional, 6),
            "target_min_notional_usdt": round(target_min_notional, 6),
            "target_min_notional_balance_ratio": round(notional_floor_ratio, 6),
            "notional_floor_reason": notional_floor_reason,
            "notional_floor_blocked": notional_floor_blocked,
            "final_notional_usdt": round(balance * final_size_pct * leverage, 6),
            "expected_profit_usdt": round(
                balance * final_size_pct * leverage * max(expected_net, 0.0) / 100.0,
                6,
            ),
        }
        raw["dynamic_leverage_decision"] = dynamic_leverage_payload
        if planned_loss > max_loss:
            decision.position_size_pct = max_size_pct
            if strategy_sizing_applied.get("applied"):
                strategy_sizing_applied["position_size_pct"] = round(max_size_pct, 6)
                strategy_sizing_applied["capped_by_stop_loss_budget"] = True
            raw["profit_risk_sizing"]["capped_stop_loss_usdt"] = round(
                balance * max_size_pct * leverage * stress_stop_loss_pct,
                6,
            )
            raw["profit_risk_sizing"][
                "reason"
            ] = "position size capped by stress-stop budget to prevent small-win-big-loss structure"
        decision.raw_response = raw
