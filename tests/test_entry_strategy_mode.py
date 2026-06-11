from __future__ import annotations

from typing import Any

import pytest

from services.entry_strategy_mode import EntryStrategyModeContextPolicy
from services.entry_symbol_universe import EntrySymbolUniversePolicy
from services.trading_service import TradingService


def _base_kwargs(**overrides: Any) -> dict[str, Any]:
    data = {
        "market_regime": {"mode": "uptrend_continuation", "confidence": 0.6},
        "daily_state": {"today_total_pnl": 8.0, "today_high_water_pnl": 12.0},
        "side_performance": {"long": {"pnl": 4.0}, "short": {"pnl": 1.0}},
        "symbol_side_performance": {"BTC/USDT|long": {"pnl": 2.0}},
        "model_contribution_performance": {"ml_profit_model": {"pnl": 1.0}},
        "position_exposure": {"dominant_side": "neutral"},
        "position_group_count": 10,
        "account_equity": 2_000.0,
        "account_config": {"max_loss_usdt": 1_000.0},
    }
    data.update(overrides)
    return data


def test_strategy_mode_builds_roster_when_portfolio_underfilled() -> None:
    result = EntryStrategyModeContextPolicy().build(**_base_kwargs(position_group_count=3))

    assert result["strategy"] == "portfolio_roster_build"
    assert result["posture"] == "diversified_positive_expectancy"
    assert result["min_opportunity_score"] == 0.65
    assert result["portfolio_roster"]["underfilled"] is True
    assert result["portfolio_roster"]["gap"] == 7


def test_strategy_mode_clamps_during_drawdown() -> None:
    result = EntryStrategyModeContextPolicy().build(
        **_base_kwargs(
            daily_state={"today_total_pnl": -40.0, "today_high_water_pnl": 10.0},
            position_group_count=10,
        )
    )

    assert result["strategy"] == "drawdown_clamp"
    assert result["risk_mode"] == "normal"
    assert result["min_opportunity_score"] >= 1.45
    assert result["max_entry_stop_loss_usdt"] <= 7.0


def test_strategy_mode_uses_hard_recovery_for_deep_loss() -> None:
    result = EntryStrategyModeContextPolicy().build(
        **_base_kwargs(
            daily_state={"today_total_pnl": -100.0, "today_high_water_pnl": 5.0},
            position_group_count=10,
        )
    )

    assert result["strategy"] == "hard_recovery"
    assert result["risk_mode"] == "hard_recovery"
    assert result["min_opportunity_score"] >= 2.10
    assert result["max_entry_stop_loss_usdt"] <= 4.5


def test_strategy_mode_keeps_global_direction_as_soft_bias() -> None:
    result = EntryStrategyModeContextPolicy().build(
        **_base_kwargs(
            daily_state={"today_total_pnl": -5.0, "today_high_water_pnl": 1.0},
            market_regime={
                "mode": "uptrend_continuation",
                "confidence": 0.6,
                "avoid_short": True,
            },
            side_performance={"long": {"pnl": 1.0}, "short": {"pnl": -5.0}},
        )
    )

    assert result["strategy"] == "recovery_attack"
    assert result["allow_long"] is True
    assert result["allow_short"] is True
    assert result["blocked_directions"] == []
    assert result["soft_avoided_directions"] == ["short"]


def test_strategy_mode_marks_realized_losing_side_as_degraded() -> None:
    result = EntryStrategyModeContextPolicy().build(
        **_base_kwargs(
            side_performance={
                "long": {
                    "count": 4,
                    "wins": 0,
                    "losses": 4,
                    "pnl": -18.0,
                    "avg_pnl": -4.5,
                    "win_rate": 0.0,
                },
                "short": {
                    "count": 4,
                    "wins": 3,
                    "losses": 1,
                    "pnl": 9.0,
                    "avg_pnl": 2.25,
                    "win_rate": 0.75,
                },
            },
        )
    )

    long_quality = result["side_quality"]["long"]
    short_quality = result["side_quality"]["short"]
    assert long_quality["state"] == "degraded"
    assert long_quality["score_adjustment"] < 0
    assert long_quality["min_score_delta"] > 0
    assert long_quality["size_multiplier"] < 1.0
    assert short_quality["state"] == "working"
    assert short_quality["score_adjustment"] > 0


def test_strategy_mode_waits_in_choppy_market() -> None:
    result = EntryStrategyModeContextPolicy().build(
        **_base_kwargs(
            market_regime={"mode": "mixed", "confidence": 0.2},
            position_group_count=10,
        )
    )

    assert result["strategy"] == "chop_wait"
    assert result["posture"] == "patient"


@pytest.mark.asyncio
async def test_trading_service_strategy_mode_context_delegates_to_policy() -> None:
    service = object.__new__(TradingService)

    class Daily:
        async def state(self, mode: str) -> dict[str, Any]:
            assert mode == "paper"
            return {"today_total_pnl": 5.0, "today_high_water_pnl": 6.0}

    class Exposure:
        def context(self, open_positions):
            return {"count": len(open_positions), "dominant_side": "neutral"}

    async def side_perf(mode: str) -> dict[str, Any]:
        assert mode == "paper"
        return {"long": {"pnl": 1.0}, "short": {"pnl": 0.0}}

    async def symbol_side_perf(mode: str) -> dict[str, Any]:
        assert mode == "paper"
        return {"BTC/USDT|long": {"pnl": 1.0}}

    async def contribution_perf(mode: str) -> dict[str, Any]:
        assert mode == "paper"
        return {"server_profit_model": {"pnl": 1.0}}

    async def balance(mode: str) -> float:
        assert mode == "paper"
        return 1_000.0

    service.daily_performance_service = Daily()
    service._today_side_performance = side_perf
    service._recent_symbol_side_performance = symbol_side_perf
    service._recent_model_contribution_performance = contribution_perf
    service.entry_position_exposure = Exposure()
    service.entry_symbol_universe = EntrySymbolUniversePolicy(lambda symbol: str(symbol or ""))
    service.allocated_order_balance = balance
    service.entry_strategy_mode_context = EntryStrategyModeContextPolicy(
        target_position_groups=3,
        roster_fill_market_symbol_min=12,
    )
    result = await service._strategy_mode_context(
        "paper",
        {"mode": "uptrend_continuation", "confidence": 0.6},
        open_positions=[{"symbol": "BTC/USDT"}],
    )

    assert result["strategy"] == "portfolio_roster_build"
    assert result["position_exposure"]["count"] == 1
    assert result["symbol_side_performance"] == {"BTC/USDT|long": {"pnl": 1.0}}
    assert result["model_contribution_performance"] == {"server_profit_model": {"pnl": 1.0}}
    assert result["portfolio_roster"]["market_symbol_min"] == 12
