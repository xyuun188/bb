"""Fast exit and profit drawdown planning policies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from services.exit_predictive_reversal import (
    PREDICTIVE_REVERSAL_EXIT_SCORE,
    PREDICTIVE_REVERSAL_FULL_EXIT_SCORE,
    PREDICTIVE_REVERSAL_MIN_PROFIT_MULTIPLE,
    PREDICTIVE_REVERSAL_REDUCE_FRACTION,
    ExitPredictiveReversalPolicy,
)
from services.trading_params import ESTIMATED_TAKER_FEE_PCT

PROFIT_DRAWDOWN_MIN_HOLD_MINUTES = 8.0
PROFIT_DRAWDOWN_MIN_PROFIT_RATIO = 0.006
PROFIT_DRAWDOWN_STRONG_PROFIT_RATIO = 0.016
PROFIT_DRAWDOWN_PARTIAL_RETRACE = 0.38
PROFIT_DRAWDOWN_FULL_RETRACE = 0.68
PROFIT_DRAWDOWN_PARTIAL_CLOSE_FRACTION = 0.35
PROFIT_DRAWDOWN_MIN_NET_USDT = 5.0
PROFIT_DRAWDOWN_MIN_FEE_MULTIPLE = 4.0
PROFIT_DRAWDOWN_MIN_SECONDS_BETWEEN_EXITS = 600.0
PROFIT_DRAWDOWN_VOLUME_CONFIRM_RATIO = 1.05
PROFIT_DRAWDOWN_ACCELERATED_HOLD_MINUTES = 8.0

FAST_RISK_1M_MOVE_PCT = 0.025
FAST_RISK_5M_MOVE_PCT = 0.04
FAST_RISK_MIN_HOLD_MINUTES = 4.0
FAST_RISK_MIN_LOSS_PCT = 0.008
FAST_RISK_REDUCE_LOSS_PCT = 0.012
FAST_RISK_FULL_LOSS_PCT = 0.018
FAST_RISK_NEAR_STOP_PROGRESS = 0.50
FAST_RISK_FULL_STOP_PROGRESS = 0.78
FAST_RISK_VOLUME_CONFIRM_RATIO = 1.05
FAST_RISK_REDUCE_POSITION_PCT = 0.50
FAST_RISK_FORCE_FULL_LOSS_USDT = 4.0
FAST_RISK_FORCE_FULL_PROGRESS = 0.50
FAST_RISK_MAX_FEATURE_POSITION_PRICE_GAP = 0.03
FAST_RISK_PRICE_24H_RANGE_TOLERANCE_PCT = 0.03

SecondsSinceProfitExit = Callable[[dict[str, Any]], float]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class ExitFastRiskPolicy:
    """Plan fast non-AI exits and profit drawdown exits."""

    predictive_reversal: ExitPredictiveReversalPolicy
    seconds_since_profit_exit: SecondsSinceProfitExit = lambda _state: 0.0

    @staticmethod
    def suspicious_feature_price_reason(
        *,
        side: str,
        feature_price: float,
        position_price: float,
        high_24h: float,
        low_24h: float,
        returns_1: float,
        returns_5: float,
    ) -> str | None:
        """Return a reason when feature-vector price is unsafe for fast exits."""

        if feature_price <= 0:
            return None

        if high_24h > 0 and low_24h > 0 and high_24h >= low_24h:
            range_floor = low_24h * (1.0 - FAST_RISK_PRICE_24H_RANGE_TOLERANCE_PCT)
            range_ceiling = high_24h * (1.0 + FAST_RISK_PRICE_24H_RANGE_TOLERANCE_PCT)
            if not range_floor <= feature_price <= range_ceiling:
                return "feature price is outside the 24h exchange range"

        if position_price <= 0:
            return None

        feature_position_gap = abs(feature_price - position_price) / max(position_price, 1e-12)
        if feature_position_gap < FAST_RISK_MAX_FEATURE_POSITION_PRICE_GAP:
            return None

        feature_price_implies_adverse = (side == "long" and feature_price < position_price) or (
            side == "short" and feature_price > position_price
        )
        if not feature_price_implies_adverse:
            return None

        short_returns_confirm_adverse = (side == "long" and returns_1 < 0 and returns_5 < 0) or (
            side == "short" and returns_1 > 0 and returns_5 > 0
        )
        if short_returns_confirm_adverse:
            return None

        return "feature price diverges from exchange position price without return confirmation"

    def profit_drawdown_exit_plan(
        self,
        *,
        side: str,
        current_price: float,
        entry_price: float,
        unrealized_pnl: float,
        peak_state: dict[str, Any],
        hold_minutes: float | None,
        volume_ratio: float,
        returns_1: float,
        returns_5: float,
        returns_20: float = 0.0,
        rsi_14: float = 50.0,
        bb_pct: float = 0.5,
        macd_diff: float = 0.0,
        adx_14: float = 0.0,
    ) -> dict[str, Any]:
        if entry_price <= 0 or current_price <= 0:
            return {"should_exit": False, "fraction": 0.0, "note": "price data is insufficient"}

        hold_minutes = float(hold_minutes or 0.0)
        peak_pnl = _safe_float(peak_state.get("peak_unrealized_pnl"), 0.0)
        peak_ratio = _safe_float(peak_state.get("peak_pnl_ratio"), 0.0)
        current_pnl = _safe_float(unrealized_pnl, 0.0)
        if peak_pnl <= 0 or current_pnl <= 0:
            return {"should_exit": False, "fraction": 0.0, "note": "no protectable profit peak yet"}

        notional = abs(peak_pnl / max(peak_ratio, 1e-9)) if peak_ratio > 0 else 0.0
        estimated_round_trip_fee = max(notional * ESTIMATED_TAKER_FEE_PCT * 2.0, 1e-9)
        min_net_profit = max(
            PROFIT_DRAWDOWN_MIN_NET_USDT,
            estimated_round_trip_fee * PROFIT_DRAWDOWN_MIN_FEE_MULTIPLE,
        )
        seconds_since_exit = self.seconds_since_profit_exit(peak_state)
        if 0 < seconds_since_exit < PROFIT_DRAWDOWN_MIN_SECONDS_BETWEEN_EXITS:
            return {
                "should_exit": False,
                "fraction": 0.0,
                "note": "刚做过一次利润保护，暂不连续碎片化部分平仓。",
                "seconds_since_last_profit_exit": seconds_since_exit,
                "peak_ratio": peak_ratio,
            }

        retrace_abs = max(peak_pnl - current_pnl, 0.0)
        retrace_ratio = retrace_abs / peak_pnl if peak_pnl > 0 else 0.0
        same_direction_pressure = (
            returns_1 < 0 and returns_5 < 0 if side == "long" else returns_1 > 0 and returns_5 > 0
        )
        reversal = self.predictive_reversal.evidence(
            side=side,
            returns_1=returns_1,
            returns_5=returns_5,
            returns_20=returns_20,
            volume_ratio=volume_ratio,
            rsi_14=rsi_14,
            bb_pct=bb_pct,
            macd_diff=macd_diff,
            adx_14=adx_14,
        )
        volume_confirms = volume_ratio <= 0 or volume_ratio >= PROFIT_DRAWDOWN_VOLUME_CONFIRM_RATIO
        strong_profit = peak_ratio >= PROFIT_DRAWDOWN_STRONG_PROFIT_RATIO
        has_buffer = peak_ratio >= PROFIT_DRAWDOWN_MIN_PROFIT_RATIO
        severe_retrace = bool(
            peak_pnl >= min_net_profit and retrace_ratio >= PROFIT_DRAWDOWN_FULL_RETRACE
        )
        profit_salvage_floor = max(
            estimated_round_trip_fee * 2.0,
            min_net_profit * 0.85,
        )
        predictive_exit = bool(
            reversal.get("score", 0.0) >= PREDICTIVE_REVERSAL_EXIT_SCORE
            and peak_pnl >= min_net_profit * PREDICTIVE_REVERSAL_MIN_PROFIT_MULTIPLE
            and current_pnl >= min_net_profit
        )

        if (
            hold_minutes < PROFIT_DRAWDOWN_MIN_HOLD_MINUTES
            and not severe_retrace
            and not predictive_exit
        ):
            return {
                "should_exit": False,
                "fraction": 0.0,
                "note": "持仓时间较短，但未出现明显利润回吐，本轮不因普通波动主动锁盈。",
                "peak_ratio": peak_ratio,
                "retrace_ratio": retrace_ratio,
                "hold_minutes": hold_minutes,
                "predictive_reversal": reversal,
            }

        if current_pnl < min_net_profit:
            if (severe_retrace or predictive_exit) and current_pnl >= profit_salvage_floor:
                return {
                    "should_exit": True,
                    "fraction": 1.0 if severe_retrace else PREDICTIVE_REVERSAL_REDUCE_FRACTION,
                    "note": (
                        "曾经达到可保护浮盈，且短周期已出现反向预警；当前仍覆盖手续费缓冲，"
                        "先把账面利润转为已实现利润，避免从盈利拖成亏损。"
                    ),
                    "peak_ratio": peak_ratio,
                    "retrace_ratio": retrace_ratio,
                    "peak_unrealized_pnl": peak_pnl,
                    "current_pnl": current_pnl,
                    "min_net_profit": min_net_profit,
                    "profit_salvage_floor": profit_salvage_floor,
                    "predictive_reversal": reversal,
                }
            return {
                "should_exit": False,
                "fraction": 0.0,
                "note": "当前剩余浮盈已经低于动态锁盈线，且不足以安全覆盖手续费缓冲；本轮不做碎片化小额平仓。",
                "peak_ratio": peak_ratio,
                "retrace_ratio": retrace_ratio,
                "current_pnl": current_pnl,
                "min_net_profit": min_net_profit,
                "profit_salvage_floor": profit_salvage_floor,
                "predictive_reversal": reversal,
            }

        if not has_buffer and not severe_retrace:
            return {
                "should_exit": False,
                "fraction": 0.0,
                "note": "浮盈峰值还没有达到动态保护线，不因为普通小回撤主动平仓。",
                "peak_ratio": peak_ratio,
                "retrace_ratio": retrace_ratio,
            }

        if retrace_ratio >= PROFIT_DRAWDOWN_FULL_RETRACE and (
            (same_direction_pressure and volume_confirms) or retrace_ratio >= 0.75 or severe_retrace
        ):
            return {
                "should_exit": True,
                "fraction": 1.0,
                "note": "浮盈已经明显回撤，利润保护优先于继续等待，执行全平锁定剩余利润。",
                "peak_ratio": peak_ratio,
                "retrace_ratio": retrace_ratio,
                "peak_unrealized_pnl": peak_pnl,
                "predictive_reversal": reversal,
            }

        if predictive_exit and retrace_ratio >= 0.10:
            full_predictive = bool(
                reversal.get("score", 0.0) >= PREDICTIVE_REVERSAL_FULL_EXIT_SCORE
                and (retrace_ratio >= PROFIT_DRAWDOWN_PARTIAL_RETRACE or same_direction_pressure)
            )
            return {
                "should_exit": True,
                "fraction": 1.0 if full_predictive else PREDICTIVE_REVERSAL_REDUCE_FRACTION,
                "note": (
                    "预判型锁盈触发：持仓仍有浮盈，但短周期动量、量能或技术结构已经转向不利；"
                    "先减仓/平仓保护利润，避免等到浮盈回吐后再被动止损。"
                ),
                "peak_ratio": peak_ratio,
                "retrace_ratio": retrace_ratio,
                "peak_unrealized_pnl": peak_pnl,
                "current_pnl": current_pnl,
                "predictive_reversal": reversal,
            }

        if retrace_ratio >= PROFIT_DRAWDOWN_PARTIAL_RETRACE and (
            (same_direction_pressure and volume_confirms)
            or hold_minutes >= PROFIT_DRAWDOWN_ACCELERATED_HOLD_MINUTES
            or strong_profit
        ):
            return {
                "should_exit": True,
                "fraction": PROFIT_DRAWDOWN_PARTIAL_CLOSE_FRACTION,
                "note": "浮盈开始明显回撤，先减仓锁定一部分利润，剩余仓位继续观察。",
                "peak_ratio": peak_ratio,
                "retrace_ratio": retrace_ratio,
                "peak_unrealized_pnl": peak_pnl,
                "predictive_reversal": reversal,
            }

        return {
            "should_exit": False,
            "fraction": 0.0,
            "note": "浮盈回撤还没有达到主动减仓线。",
            "peak_ratio": peak_ratio,
            "retrace_ratio": retrace_ratio,
            "peak_unrealized_pnl": peak_pnl,
            "predictive_reversal": reversal,
        }

    def fast_adverse_exit_plan(
        self,
        *,
        side: str,
        entry_price: float,
        current_price: float,
        stop_loss: float,
        returns_1: float,
        returns_5: float,
        hold_minutes: float | None,
        volume_ratio: float,
        current_unrealized_pnl: float = 0.0,
        hard_adverse_observed: bool = False,
        data_quality_suspicious: bool = False,
        predictive_reversal_score: float = 0.0,
    ) -> dict[str, Any]:
        """Decide whether a fast adverse move is real risk or normal noise."""

        if entry_price <= 0 or current_price <= 0:
            return {
                "should_exit": False,
                "fraction": 0.0,
                "adverse_pct": 0.0,
                "note": "price data is insufficient",
            }

        if data_quality_suspicious:
            return {
                "should_exit": False,
                "fraction": 0.0,
                "adverse_pct": 0.0,
                "risk_progress": 0.0,
                "hard_adverse_observed": hard_adverse_observed,
                "data_quality_suspicious": True,
                "note": "market data is suspicious, so fast risk will wait for a fresher review",
            }

        if side == "long":
            adverse_pct = max((entry_price - current_price) / entry_price, 0.0)
            same_direction_pressure = returns_1 < 0 and returns_5 < 0
            stop_distance_pct = (
                (entry_price - stop_loss) / entry_price if 0 < stop_loss < entry_price else 0.0
            )
        else:
            adverse_pct = max((current_price - entry_price) / entry_price, 0.0)
            same_direction_pressure = returns_1 > 0 and returns_5 > 0
            stop_distance_pct = (
                (stop_loss - entry_price) / entry_price if stop_loss > entry_price else 0.0
            )

        near_stop = bool(
            stop_distance_pct > 0
            and adverse_pct >= stop_distance_pct * FAST_RISK_NEAR_STOP_PROGRESS
        )
        full_stop_progress = bool(
            stop_distance_pct > 0
            and adverse_pct >= stop_distance_pct * FAST_RISK_FULL_STOP_PROGRESS
        )
        risk_progress = (
            adverse_pct / max(stop_distance_pct, 1e-12) if stop_distance_pct > 0 else 0.0
        )
        volume_confirmed = volume_ratio <= 0 or volume_ratio >= FAST_RISK_VOLUME_CONFIRM_RATIO
        old_enough = hold_minutes is None or hold_minutes >= FAST_RISK_MIN_HOLD_MINUTES
        loss_usdt = abs(min(_safe_float(current_unrealized_pnl, 0.0), 0.0))
        predictive_confirms = predictive_reversal_score >= PREDICTIVE_REVERSAL_EXIT_SCORE
        strong_adverse_momentum = bool(
            same_direction_pressure
            and (abs(returns_1) >= FAST_RISK_1M_MOVE_PCT or abs(returns_5) >= FAST_RISK_5M_MOVE_PCT)
        )

        if adverse_pct <= 0:
            return {
                "should_exit": False,
                "fraction": 0.0,
                "adverse_pct": adverse_pct,
                "risk_progress": risk_progress,
                "note": "当前价格仍未相对开仓价亏损，短线反向只记录观察。",
            }

        if adverse_pct < FAST_RISK_MIN_LOSS_PCT and not near_stop:
            return {
                "should_exit": False,
                "fraction": 0.0,
                "adverse_pct": adverse_pct,
                "risk_progress": risk_progress,
                "note": "亏损幅度还小，未接近止损，继续交给持仓复盘判断。",
            }

        if not old_enough and not full_stop_progress and adverse_pct < FAST_RISK_FULL_LOSS_PCT:
            return {
                "should_exit": False,
                "fraction": 0.0,
                "adverse_pct": adverse_pct,
                "risk_progress": risk_progress,
                "note": f"开仓不足 {FAST_RISK_MIN_HOLD_MINUTES:.0f} 分钟，暂不因普通短线波动平仓。",
            }

        force_full_by_loss = bool(
            loss_usdt >= FAST_RISK_FORCE_FULL_LOSS_USDT
            and adverse_pct >= FAST_RISK_FULL_LOSS_PCT
            and risk_progress >= FAST_RISK_FORCE_FULL_PROGRESS
        )

        if (
            full_stop_progress
            or force_full_by_loss
            or predictive_confirms
            or (
                strong_adverse_momentum
                and volume_confirmed
                and adverse_pct >= FAST_RISK_FULL_LOSS_PCT
            )
        ):
            return {
                "should_exit": True,
                "fraction": 1.0,
                "adverse_pct": adverse_pct,
                "risk_progress": risk_progress,
                "loss_usdt": loss_usdt,
                "hard_adverse_observed": hard_adverse_observed,
                "strong_adverse_momentum": strong_adverse_momentum,
                "volume_confirmed": volume_confirmed,
                "predictive_reversal_score": predictive_reversal_score,
                "note": "价格已接近止损、亏损金额明显扩大，或出现持续同向恶化，直接全平控制风险。",
            }

        if hard_adverse_observed:
            return {
                "should_exit": False,
                "fraction": 0.0,
                "adverse_pct": adverse_pct,
                "risk_progress": risk_progress,
                "loss_usdt": loss_usdt,
                "hard_adverse_observed": True,
                "strong_adverse_momentum": strong_adverse_momentum,
                "volume_confirmed": volume_confirmed,
                "predictive_reversal_score": predictive_reversal_score,
                "note": (
                    "hard adverse price move observed, but stop progress, momentum, "
                    "volume and predictive reversal evidence are not strong enough for "
                    "an automatic full close"
                ),
            }

        if near_stop or (
            old_enough
            and same_direction_pressure
            and volume_confirmed
            and adverse_pct >= max(FAST_RISK_REDUCE_LOSS_PCT * 0.8, 0.008)
        ):
            return {
                "should_exit": False,
                "fraction": 0.0,
                "adverse_pct": adverse_pct,
                "risk_progress": risk_progress,
                "loss_usdt": loss_usdt,
                "note": (
                    "亏损仓位尚未达到强制全平条件，普通快速风控不再做部分减仓；"
                    "继续交给止损、严重趋势失效或下一轮持仓复盘判断。"
                ),
            }

        return {
            "should_exit": False,
            "fraction": 0.0,
            "adverse_pct": adverse_pct,
            "risk_progress": risk_progress,
            "note": "短线反向尚未满足减仓或全平条件，继续观察。",
        }
