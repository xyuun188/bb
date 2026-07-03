from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

import services.trading_service as trading_service
from services.entry_strategy_mode import EntryStrategyModeContextPolicy
from services.entry_symbol_universe import EntrySymbolUniversePolicy
from services.trading_service import TradingService


def _base_kwargs(**overrides: Any) -> dict[str, Any]:
    data = {
        "market_regime": {"mode": "uptrend_continuation", "confidence": 0.6},
        "daily_state": {"today_total_pnl": 8.0, "today_high_water_pnl": 12.0},
        "side_performance": {"long": {"pnl": 4.0}, "short": {"pnl": 1.0}},
        "side_performance_multiday": {},
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
    assert result["strategy_profile_id"] == "baseline_current"
    assert result["strategy_learning_sizing"]["profile_id"] == "baseline_current"
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
    service._multiday_side_performance = side_perf
    service._recent_symbol_side_performance = symbol_side_perf
    service._recent_model_contribution_performance = contribution_perf
    service.entry_position_exposure = Exposure()
    service.entry_symbol_universe = EntrySymbolUniversePolicy(lambda symbol: str(symbol or ""))
    service.allocated_order_balance = balance
    service.entry_strategy_mode_context = EntryStrategyModeContextPolicy(
        target_position_groups=3,
        roster_fill_market_symbol_min=12,
    )
    service.strategy_learning_service = None
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
    assert result["portfolio_roster"]["market_symbol_min_is_batch_size"] is False
    assert result["strategy_context_runtime"]["parallel_context_fetch"] is True


@pytest.mark.asyncio
async def test_trading_service_strategy_mode_context_fetches_inputs_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = object.__new__(TradingService)

    class Daily:
        async def state(self, mode: str) -> dict[str, Any]:
            assert mode == "paper"
            await asyncio.sleep(0.05)
            return {"today_total_pnl": 1.0}

    class Exposure:
        def context(self, open_positions):
            return {"count": len(open_positions), "dominant_side": "neutral"}

    async def side_perf(mode: str) -> dict[str, Any]:
        assert mode == "paper"
        await asyncio.sleep(0.05)
        return {"long": {"pnl": 1.0}}

    async def symbol_side_perf(mode: str) -> dict[str, Any]:
        assert mode == "paper"
        await asyncio.sleep(0.05)
        return {"ETH/USDT|long": {"pnl": 2.0}}

    async def contribution_perf(mode: str) -> dict[str, Any]:
        assert mode == "paper"
        await asyncio.sleep(0.05)
        return {"server_profit_model": {"pnl": 3.0}}

    async def balance(mode: str) -> float:
        assert mode == "paper"
        await asyncio.sleep(0.05)
        return 1_000.0

    monkeypatch.setattr(
        trading_service.TradingService,
        "_write_runtime_heartbeat",
        lambda _self: None,
    )
    monkeypatch.setattr(
        trading_service.TradingService,
        "strategy_learning_perf_timeout_seconds",
        lambda _self: 0.5,
    )
    monkeypatch.setattr(
        trading_service.TradingService,
        "strategy_learning_account_timeout_seconds",
        lambda _self: 0.5,
    )
    service.daily_performance_service = Daily()
    service._today_side_performance = side_perf
    service._multiday_side_performance = side_perf
    service._recent_symbol_side_performance = symbol_side_perf
    service._recent_model_contribution_performance = contribution_perf
    service.entry_position_exposure = Exposure()
    service.entry_symbol_universe = EntrySymbolUniversePolicy(lambda symbol: str(symbol or ""))
    service.allocated_order_balance = balance
    service.entry_strategy_mode_context = EntryStrategyModeContextPolicy()
    service.strategy_learning_service = None

    started_at = asyncio.get_running_loop().time()
    result = await service._strategy_mode_context(
        "paper",
        {"mode": "uptrend_continuation", "confidence": 0.6},
        open_positions=[{"symbol": "ETH/USDT"}],
    )
    elapsed = asyncio.get_running_loop().time() - started_at

    assert elapsed < 0.18
    assert result["account_equity"] == 1_000.0
    assert result["symbol_side_performance"] == {"ETH/USDT|long": {"pnl": 2.0}}
    assert result["model_contribution_performance"] == {"server_profit_model": {"pnl": 3.0}}
    assert result["strategy_context_runtime"]["parallel_context_fetch"] is True


@pytest.mark.asyncio
async def test_trading_service_strategy_mode_context_refreshes_empty_positions() -> None:
    service = object.__new__(TradingService)

    class Daily:
        async def state(self, mode: str) -> dict[str, Any]:
            return {"today_total_pnl": 5.0, "today_high_water_pnl": 6.0}

    class Exposure:
        def context(self, open_positions):
            return {"count": len(open_positions), "dominant_side": "long"}

    class Sync:
        async def get_open_positions_context(self):
            return [{"symbol": "BTC/USDT", "side": "long", "unrealized_pnl": -1.0}]

    async def side_perf(_mode: str) -> dict[str, Any]:
        return {"long": {"pnl": 1.0}, "short": {"pnl": 0.0}}

    async def symbol_side_perf(_mode: str) -> dict[str, Any]:
        return {}

    async def contribution_perf(_mode: str) -> dict[str, Any]:
        return {}

    async def balance(_mode: str) -> float:
        return 1_000.0

    service.daily_performance_service = Daily()
    service._today_side_performance = side_perf
    service._multiday_side_performance = side_perf
    service._recent_symbol_side_performance = symbol_side_perf
    service._recent_model_contribution_performance = contribution_perf
    service.entry_position_exposure = Exposure()
    service.entry_symbol_universe = EntrySymbolUniversePolicy(lambda symbol: str(symbol or ""))
    service.allocated_order_balance = balance
    service.entry_strategy_mode_context = EntryStrategyModeContextPolicy(
        target_position_groups=3,
        roster_fill_market_symbol_min=12,
    )
    service.okx_sync_service = Sync()
    service.strategy_learning_service = None

    result = await service._strategy_mode_context(
        "paper",
        {"mode": "uptrend_continuation", "confidence": 0.6},
        open_positions=[],
    )

    assert result["position_exposure"]["count"] == 1
    assert result["portfolio_roster"]["current_position_groups"] == 1


@pytest.mark.asyncio
async def test_strategy_mode_context_does_not_block_on_slow_strategy_learning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = object.__new__(TradingService)
    service._current_capacity_context = {}
    service.dynamic_capacity = None
    service.position_quality_scorer = None

    class Daily:
        async def state(self, _mode: str) -> dict[str, Any]:
            return {}

    class Exposure:
        def context(self, open_positions):
            return {"count": len(open_positions), "dominant_side": "neutral"}

    class SlowLearning:
        async def apply_to_strategy_context(self, **_kwargs: Any) -> dict[str, Any]:
            await asyncio.sleep(60)
            return {"strategy_profile_id": "slow-profile"}

    async def empty_perf(_mode: str) -> dict[str, Any]:
        return {}

    async def balance(_mode: str) -> float:
        return 1000.0

    monkeypatch.setattr(
        trading_service.TradingService,
        "_write_runtime_heartbeat",
        lambda _self: None,
    )
    monkeypatch.setattr(
        trading_service.TradingService,
        "strategy_learning_context_timeout_seconds",
        lambda _self: 0.01,
    )
    service.daily_performance_service = Daily()
    service._today_side_performance = empty_perf
    service._multiday_side_performance = empty_perf
    service._recent_symbol_side_performance = empty_perf
    service._recent_model_contribution_performance = empty_perf
    service.entry_position_exposure = Exposure()
    service.entry_symbol_universe = EntrySymbolUniversePolicy(lambda symbol: str(symbol or ""))
    service.allocated_order_balance = balance
    service.entry_strategy_mode_context = EntryStrategyModeContextPolicy()
    service.strategy_learning_service = SlowLearning()
    service._strategy_learning_context_cache = {}
    service._strategy_learning_context_refresh_tasks = {}

    started_at = asyncio.get_running_loop().time()
    result = await service._strategy_mode_context(
        "paper",
        {"mode": "range", "confidence": 0.4},
        open_positions=[],
    )
    elapsed = asyncio.get_running_loop().time() - started_at

    assert elapsed < 0.5
    assert result["strategy_learning_cache_status"] == "baseline_timeout"
    assert result["strategy_profile_id"] == "baseline_current"
    assert result["strategy_learning_sizing"]["profile_id"] == "baseline_current"
    assert "dynamic_position_capacity" in result
    tasks = service._strategy_learning_context_refresh_tasks
    assert tasks["paper"].done() is False
    tasks["paper"].cancel()


def test_strategy_learning_context_wait_timeout_is_shorter_for_market_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = object.__new__(TradingService)
    monkeypatch.setattr(
        trading_service.TradingService,
        "strategy_learning_context_timeout_seconds",
        lambda _self: 10.0,
    )

    market_timeout = service.strategy_learning_context_wait_timeout_seconds("market")
    position_timeout = service.strategy_learning_context_wait_timeout_seconds("position")

    assert 0.5 <= market_timeout <= 3.0
    assert market_timeout < position_timeout
    assert position_timeout == 10.0


@pytest.mark.asyncio
async def test_strategy_learning_refresh_records_market_scope_wait_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = object.__new__(TradingService)
    service._strategy_learning_context_refresh_tasks = {}
    service._strategy_learning_context_cache = {}
    service._json_safe_payload = lambda payload: dict(payload)  # type: ignore[method-assign]
    monkeypatch.setattr(
        trading_service.TradingService,
        "strategy_learning_context_timeout_seconds",
        lambda _self: 10.0,
    )

    class Learning:
        async def apply_to_strategy_context(self, **kwargs: Any) -> dict[str, Any]:
            return dict(kwargs["strategy_context"])

    task = service._start_strategy_learning_context_refresh(
        mode="paper",
        analysis_scope="market",
        strategy_learning=Learning(),
        context={"strategy_profile_id": "baseline_current"},
        open_positions=[],
    )

    result = await asyncio.wait_for(task, timeout=1.0)

    assert result["strategy_learning_runtime_timeout_seconds"] == pytest.approx(3.0)
    cached = service._strategy_learning_context_cache["paper"]["context"]
    assert cached["strategy_learning_runtime_timeout_seconds"] == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_strategy_mode_context_uses_cached_learning_context_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = object.__new__(TradingService)
    service._current_capacity_context = {}
    service.dynamic_capacity = None
    service.position_quality_scorer = None

    class Daily:
        async def state(self, _mode: str) -> dict[str, Any]:
            return {}

    class Exposure:
        def context(self, open_positions):
            return {"count": len(open_positions), "dominant_side": "neutral"}

    class SlowLearning:
        async def apply_to_strategy_context(self, **_kwargs: Any) -> dict[str, Any]:
            await asyncio.sleep(60)
            return {"strategy_profile_id": "slow-profile"}

    async def empty_perf(_mode: str) -> dict[str, Any]:
        return {}

    async def balance(_mode: str) -> float:
        return 1000.0

    monkeypatch.setattr(
        trading_service.TradingService,
        "_write_runtime_heartbeat",
        lambda _self: None,
    )
    monkeypatch.setattr(
        trading_service.TradingService,
        "strategy_learning_context_timeout_seconds",
        lambda _self: 0.01,
    )
    service.daily_performance_service = Daily()
    service._today_side_performance = empty_perf
    service._multiday_side_performance = empty_perf
    service._recent_symbol_side_performance = empty_perf
    service._recent_model_contribution_performance = empty_perf
    service.entry_position_exposure = Exposure()
    service.entry_symbol_universe = EntrySymbolUniversePolicy(lambda symbol: str(symbol or ""))
    service.allocated_order_balance = balance
    service.entry_strategy_mode_context = EntryStrategyModeContextPolicy()
    service.strategy_learning_service = SlowLearning()
    service._strategy_learning_context_cache = {
        "paper": {
            "created_at": datetime.now(UTC),
            "context": {
                "strategy_profile_id": "cached-profile",
                "min_opportunity_score": 0.42,
            },
        }
    }
    service._strategy_learning_context_refresh_tasks = {}

    result = await service._strategy_mode_context(
        "paper",
        {"mode": "range", "confidence": 0.4},
        open_positions=[],
    )

    assert result["strategy_profile_id"] == "cached-profile"
    assert result["strategy_learning_cache_status"] == "stale_timeout"
    tasks = service._strategy_learning_context_refresh_tasks
    assert tasks["paper"].done() is False
    tasks["paper"].cancel()
