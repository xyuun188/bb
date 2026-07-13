"""Observation-only strategy context for authoritative return execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


@dataclass(frozen=True, slots=True)
class EntryStrategyModeContextPolicy:
    """Expose current state without granting entry, sizing, or direction permission."""

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
        side_performance_multiday: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del account_config
        market = _safe_dict(market_regime)
        multiday = _safe_dict(side_performance_multiday)
        today_total = _safe_float(daily_state.get("today_total_pnl"), 0.0)
        high_water = _safe_float(daily_state.get("today_high_water_pnl"), today_total)
        drawdown = max(high_water - today_total, 0.0)
        budget_reference = max(abs(account_equity), abs(high_water), 1e-12)
        drawdown_pressure = min(drawdown / budget_reference, 1.0)
        side_quality = self._side_quality(side_performance, multiday)
        generated_at = datetime.now(UTC).isoformat()
        provenance = {
            "source": "account_pnl_side_returns_and_current_portfolio_state",
            "observation_window": "current_day_plus_available_multiday_side_distribution",
            "sample_count": sum(int(item["sample_count"]) for item in side_quality.values()),
            "generated_at": generated_at,
            "strategy_version": "2026-07-12.observation-only-strategy-context.v1",
            "fallback_reason": "",
        }
        return {
            "strategy": "authoritative_return_capture",
            "posture": "dynamic_return_budget",
            "reason": "Entry permission and risk are generated from the current fee-after return distribution.",
            "long_short_policy": "compare_authoritative_fee_after_return_lcb",
            "today_total_pnl": round(today_total, 8),
            "today_high_water_pnl": round(high_water, 8),
            "drawdown_usdt": round(drawdown, 8),
            "drawdown_pressure": round(drawdown_pressure, 8),
            "market_regime": market,
            "side_performance": _safe_dict(side_performance),
            "side_performance_multiday": multiday,
            "side_quality": side_quality,
            "symbol_side_performance": _safe_dict(symbol_side_performance),
            "model_contribution_performance": _safe_dict(model_contribution_performance),
            "position_exposure": _safe_dict(position_exposure),
            "portfolio_roster": {
                "current_position_groups": int(position_group_count or 0),
                "target_position_groups": None,
                "underfilled": False,
                "policy": "observation_only_dynamic_capacity_owns_limits",
            },
            "risk_mode": "dynamic_return_budget",
            "dynamic_opportunity_score_enabled": True,
            "goal": "maximize_realized_fee_after_return",
            "execution_policy": "authoritative_fee_after_return_lcb_and_dynamic_risk_budget",
            "policy_provenance": provenance,
        }

    @staticmethod
    def _side_quality(
        side_performance: dict[str, Any],
        side_performance_multiday: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, Any]]:
        current = _safe_dict(side_performance)
        multiday = _safe_dict(side_performance_multiday)
        result: dict[str, dict[str, Any]] = {}
        for side in ("long", "short"):
            today = _safe_dict(current.get(side))
            history = _safe_dict(multiday.get(side))
            count = int(_safe_float(today.get("count"), 0.0))
            history_count = int(_safe_float(history.get("count"), 0.0))
            pnl = _safe_float(today.get("pnl"), 0.0)
            history_pnl = _safe_float(history.get("pnl"), 0.0)
            result[side] = {
                "sample_count": count + history_count,
                "fee_after_pnl_usdt": round(pnl + history_pnl, 8),
                "today_avg_pnl_usdt": round(_safe_float(today.get("avg_pnl"), 0.0), 8),
                "today_profit_factor": round(_safe_float(today.get("profit_factor"), 0.0), 8),
                "multiday_profit_factor": round(
                    _safe_float(history.get("profit_factor"), 0.0), 8
                ),
                "return_lcb_pct": history.get("return_lcb_pct", today.get("return_lcb_pct")),
                "production_permission": False,
                "reason": "side performance is observation-only",
            }
        return result
