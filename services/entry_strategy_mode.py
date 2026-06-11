"""Entry strategy-mode context policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from services.entry_priority import MIN_ENTRY_OPPORTUNITY_SCORE
from services.entry_stop_loss_budget import (
    ENTRY_MAX_STOP_LOSS_CAP_USDT,
    ENTRY_MAX_STOP_LOSS_DEFENSIVE_USDT,
    ENTRY_MAX_STOP_LOSS_DRAWDOWN_USDT,
    ENTRY_MAX_STOP_LOSS_NORMAL_USDT,
    ENTRY_MAX_STOP_LOSS_PCT_OF_EQUITY,
)

DRAWDOWN_REDUCED_RISK_USDT = 100.0
DRAWDOWN_DEFENSIVE_RISK_USDT = 220.0
DRAWDOWN_LIGHT_RISK_USDT = 30.0
DRAWDOWN_HARD_PAUSE_USDT = 80.0
PORTFOLIO_MIN_POSITION_GROUPS_TARGET = 10
PORTFOLIO_ROSTER_FILL_MARKET_SYMBOL_MIN = 36


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


@dataclass(frozen=True, slots=True)
class EntryStrategyModeContextPolicy:
    """Resolve portfolio strategy posture for market-entry analysis."""

    target_position_groups: int = PORTFOLIO_MIN_POSITION_GROUPS_TARGET
    roster_fill_market_symbol_min: int = PORTFOLIO_ROSTER_FILL_MARKET_SYMBOL_MIN

    def build(
        self,
        *,
        market_regime: dict[str, Any] | None,
        daily_state: dict[str, Any],
        side_performance: dict[str, Any],
        symbol_side_performance: dict[str, Any],
        model_contribution_performance: dict[str, Any],
        position_exposure: dict[str, Any],
        position_group_count: int,
        account_equity: float,
        account_config: dict[str, Any],
    ) -> dict[str, Any]:
        market_regime = _safe_dict(market_regime)
        side_performance = _safe_dict(side_performance)
        account_config = _safe_dict(account_config)
        today_total = _safe_float(daily_state.get("today_total_pnl"), 0.0)
        high_water = _safe_float(daily_state.get("today_high_water_pnl"), today_total)
        loss_pause = 0.0
        configured_max_loss = _safe_float(account_config.get("max_loss_usdt"), 0.0)
        drawdown_line = max(
            DRAWDOWN_LIGHT_RISK_USDT,
            (
                min(configured_max_loss * 0.05, 150.0)
                if configured_max_loss > 0
                else DRAWDOWN_REDUCED_RISK_USDT
            ),
        )
        defensive_line = max(
            DRAWDOWN_HARD_PAUSE_USDT,
            (
                min(configured_max_loss * 0.12, 300.0)
                if configured_max_loss > 0
                else DRAWDOWN_DEFENSIVE_RISK_USDT
            ),
        )

        risk_mode = "normal"
        max_entry_stop_loss_usdt = ENTRY_MAX_STOP_LOSS_NORMAL_USDT
        min_opportunity_score = MIN_ENTRY_OPPORTUNITY_SCORE
        if today_total <= -defensive_line:
            risk_mode = "defensive_recovery"
            max_entry_stop_loss_usdt = ENTRY_MAX_STOP_LOSS_DEFENSIVE_USDT
            min_opportunity_score = 1.80
        elif today_total <= -drawdown_line:
            risk_mode = "drawdown_recovery"
            max_entry_stop_loss_usdt = ENTRY_MAX_STOP_LOSS_DRAWDOWN_USDT
            min_opportunity_score = 1.35

        if account_equity > 0:
            dynamic_stop_cap = min(
                max(account_equity * ENTRY_MAX_STOP_LOSS_PCT_OF_EQUITY, 6.0),
                ENTRY_MAX_STOP_LOSS_CAP_USDT,
            )
            max_entry_stop_loss_usdt = min(max_entry_stop_loss_usdt, dynamic_stop_cap)

        regime_mode = str(market_regime.get("mode") or "unknown")
        regime_conf = _safe_float(market_regime.get("confidence"), 0.0)
        avoid_long = bool(market_regime.get("avoid_long"))
        avoid_short = bool(market_regime.get("avoid_short"))

        long_pnl = _safe_float(_safe_dict(side_performance.get("long")).get("pnl"), 0.0)
        short_pnl = _safe_float(_safe_dict(side_performance.get("short")).get("pnl"), 0.0)
        side_quality = self._side_quality(side_performance)
        if short_pnl < 0 and abs(short_pnl) > max(abs(long_pnl), 1e-9) * 1.5:
            avoid_short = True
        if long_pnl < 0 and abs(long_pnl) > max(abs(short_pnl), 1e-9) * 1.5:
            avoid_long = True

        allow_long = True
        allow_short = True
        if today_total <= -DRAWDOWN_HARD_PAUSE_USDT:
            risk_mode = "hard_recovery"
            min_opportunity_score = max(min_opportunity_score, 2.10)
            max_entry_stop_loss_usdt = min(max_entry_stop_loss_usdt, 4.5)
        elif today_total <= -DRAWDOWN_LIGHT_RISK_USDT:
            min_opportunity_score = max(min_opportunity_score, 1.45)
            max_entry_stop_loss_usdt = min(max_entry_stop_loss_usdt, 7.0)

        roster_gap = max(self.target_position_groups - position_group_count, 0)
        roster_fill_active = roster_gap > 0
        if roster_fill_active and today_total > -DRAWDOWN_LIGHT_RISK_USDT:
            min_opportunity_score = min(min_opportunity_score, 0.65)

        soft_biases: list[str] = []
        if avoid_long:
            soft_biases.append("long")
        if avoid_short:
            soft_biases.append("short")
        direction_filter_reason = (
            "Global regime is advisory only; symbol-level signals decide direction."
            if soft_biases
            else "No strong global directional bias; use independent symbol signals."
        )

        strategy = "normal_capture"
        posture = "balanced"
        reason = f"Normal capture of high-quality symbol opportunities. {direction_filter_reason}"
        if loss_pause > 0 and today_total <= -loss_pause:
            strategy = "loss_recovery_selective"
            posture = "selective_recovery"
            reason = (
                "Daily loss pause is active; stop ordinary trial entries while allowing "
                "small, independently confirmed recovery opportunities."
            )
        elif today_total < 0:
            if regime_mode != "mixed" and regime_conf >= 0.35 and not (avoid_long and avoid_short):
                strategy = "recovery_attack"
                posture = "profit_first_expansion"
                reason = (
                    "Daily PnL is negative but hard pause is not active. "
                    "Use profit-first recovery: allow more independently confirmed entries, "
                    f"without forcing one global direction. {direction_filter_reason}"
                )
            else:
                strategy = "recovery_selective"
                posture = "selective_recovery"
                reason = (
                    "Daily PnL is negative and market direction is unclear. "
                    "Keep entries selective until per-symbol signal quality improves."
                )
        elif regime_mode == "mixed" or regime_conf < 0.35:
            strategy = "chop_wait"
            posture = "patient"
            reason = "Market direction is choppy; reduce low-quality entries and wait for clearer signals."

        if today_total <= -DRAWDOWN_HARD_PAUSE_USDT:
            strategy = "hard_recovery"
            posture = "tight_selective_reentry"
            reason = (
                "Daily loss is deep; use tight selective recovery. "
                "Only higher-quality, lower-tail-risk new entries are allowed."
            )
        elif today_total <= -DRAWDOWN_LIGHT_RISK_USDT:
            strategy = "drawdown_clamp"
            posture = "tight_selective"
            reason = (
                "Daily drawdown is active; allow only higher-quality opportunities "
                "and explicitly reduce single-trade tail risk."
            )
        elif roster_fill_active:
            strategy = "portfolio_roster_build"
            posture = "diversified_positive_expectancy"
            reason = (
                f"Current grouped positions {position_group_count}/{self.target_position_groups}; "
                "prioritize independent positive-expectancy opportunities while keeping "
                "negative-return, abnormal-price, margin, and hard-risk guards."
            )

        return {
            "strategy": strategy,
            "posture": posture,
            "reason": reason,
            "preferred_direction": "neutral",
            "allow_long": allow_long,
            "allow_short": allow_short,
            "blocked_directions": [],
            "soft_avoided_directions": soft_biases,
            "direction_filter_policy": "soft_bias_no_hard_direction_ban",
            "long_short_policy": "evaluate_both_sides_per_symbol",
            "today_total_pnl": round(today_total, 4),
            "today_high_water_pnl": round(high_water, 4),
            "loss_pause_usdt": round(loss_pause, 4),
            "market_regime": market_regime,
            "side_performance": side_performance,
            "side_quality": side_quality,
            "symbol_side_performance": symbol_side_performance,
            "model_contribution_performance": model_contribution_performance,
            "position_exposure": position_exposure,
            "portfolio_roster": {
                "target_position_groups": self.target_position_groups,
                "current_position_groups": position_group_count,
                "gap": roster_gap,
                "underfilled": roster_fill_active,
                "market_symbol_min": self.roster_fill_market_symbol_min,
                "policy": (
                    "When grouped positions are below target, increase scanning and small-size "
                    "execution bias for independent positive-expectancy opportunities; restore "
                    "ordinary thresholds once the target is reached."
                ),
            },
            "risk_mode": risk_mode,
            "drawdown_line_usdt": round(drawdown_line, 4),
            "defensive_line_usdt": round(defensive_line, 4),
            "min_opportunity_score": round(min_opportunity_score, 4),
            "dynamic_opportunity_score_enabled": True,
            "max_entry_stop_loss_usdt": round(max_entry_stop_loss_usdt, 4),
            "goal": "maximize_realized_net_profit",
            "execution_policy": (
                "Auto-select strategy; global regime is advisory only; rank entries by "
                "expected net return, tail risk, fees, and capital efficiency; do not "
                "optimize for win rate."
            ),
        }

    @staticmethod
    def _side_quality(side_performance: dict[str, Any]) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for side in ("long", "short"):
            bucket = _safe_dict(side_performance.get(side))
            count = int(_safe_float(bucket.get("count"), 0.0))
            wins = int(_safe_float(bucket.get("wins"), 0.0))
            losses = int(_safe_float(bucket.get("losses"), 0.0))
            pnl = _safe_float(bucket.get("pnl"), 0.0)
            avg_pnl = _safe_float(bucket.get("avg_pnl"), 0.0)
            win_rate = _safe_float(bucket.get("win_rate"), 0.0)
            state = "neutral"
            score_adjustment = 0.0
            min_score_delta = 0.0
            size_multiplier = 1.0
            reason = "no strong realized side feedback"
            if count >= 2 and pnl < 0 and (losses >= wins + 2 or win_rate <= 0.35):
                state = "degraded"
                score_adjustment = -0.25
                min_score_delta = 0.22
                size_multiplier = 0.65
                reason = "today realized side performance is weak; require stronger proof"
            elif count >= 3 and pnl > 0 and win_rate >= 0.55 and avg_pnl > 0:
                state = "working"
                score_adjustment = 0.08
                min_score_delta = -0.05
                size_multiplier = 1.05
                reason = (
                    "today realized side performance is positive; allow small confidence support"
                )
            result[side] = {
                "state": state,
                "count": count,
                "wins": wins,
                "losses": losses,
                "pnl": round(pnl, 6),
                "avg_pnl": round(avg_pnl, 6),
                "win_rate": round(win_rate, 6),
                "score_adjustment": round(score_adjustment, 6),
                "min_score_delta": round(min_score_delta, 6),
                "size_multiplier": round(size_multiplier, 6),
                "reason": reason,
            }
        return result
