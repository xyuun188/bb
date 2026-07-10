from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
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
    synced_profiles: dict[str, Any] = {}

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
    service.entry_market_hold_penalty = SimpleNamespace(
        sync_recent_loss_profiles=lambda profiles: synced_profiles.update(profiles)
    )
    service.entry_strategy_mode_context = EntryStrategyModeContextPolicy(
        target_position_groups=3,
        roster_fill_market_symbol_min=12,
    )
    service._refresh_dynamic_capacity = (  # type: ignore[method-assign]
        lambda **kwargs: {
            **kwargs["strategy_context"],
            "account_equity": kwargs["account_equity"],
        }
    )
    service._json_safe_payload = lambda value: value  # type: ignore[method-assign]
    service._strategy_context_performance_snapshot_cache = {
        "paper": {
            "created_at": datetime.now(UTC),
            "values": {
                "daily_perf": {"today_total_pnl": 5.0, "today_high_water_pnl": 6.0},
                "today_side_perf": {"long": {"pnl": 1.0}, "short": {"pnl": 0.0}},
                "multiday_side_perf": {"long": {"pnl": 1.0}, "short": {"pnl": 0.0}},
                "symbol_side_perf": {"BTC/USDT|long": {"pnl": 1.0}},
                "model_contribution_perf": {"server_profit_model": {"pnl": 1.0}},
            },
            "version": 1,
            "refresh_timings": {},
            "failed_labels": [],
        }
    }
    service.strategy_learning_service = None
    result = await service._strategy_mode_context(
        "paper",
        {"mode": "uptrend_continuation", "confidence": 0.6},
        open_positions=[{"symbol": "BTC/USDT"}],
    )

    assert result["strategy"] == "portfolio_roster_build"
    assert result["position_exposure"]["count"] == 1
    assert result["symbol_side_performance"] == {"BTC/USDT|long": {"pnl": 1.0}}
    assert synced_profiles == {"BTC/USDT|long": {"pnl": 1.0}}
    assert result["model_contribution_performance"] == {"server_profit_model": {"pnl": 1.0}}
    assert result["portfolio_roster"]["market_symbol_min"] == 12
    assert result["portfolio_roster"]["market_symbol_min_is_batch_size"] is False
    assert result["strategy_context_performance"]["status"] == "fresh"
    assert result["strategy_context_runtime"]["parallel_context_fetch"] is False


@pytest.mark.asyncio
async def test_trading_service_strategy_mode_context_builds_and_reuses_background_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = object.__new__(TradingService)
    active_fetches = 0
    peak_fetches = 0

    async def slow_value(value: Any) -> Any:
        nonlocal active_fetches, peak_fetches
        active_fetches += 1
        peak_fetches = max(peak_fetches, active_fetches)
        try:
            await asyncio.sleep(0.05)
            return value
        finally:
            active_fetches -= 1

    class Daily:
        async def state(self, mode: str) -> dict[str, Any]:
            assert mode == "paper"
            return await slow_value({"today_total_pnl": 1.0})

    class Exposure:
        def context(self, open_positions):
            return {"count": len(open_positions), "dominant_side": "neutral"}

    async def side_perf(mode: str) -> dict[str, Any]:
        assert mode == "paper"
        return await slow_value({"long": {"pnl": 1.0}})

    async def symbol_side_perf(mode: str) -> dict[str, Any]:
        assert mode == "paper"
        return await slow_value({"ETH/USDT|long": {"pnl": 2.0}})

    async def contribution_perf(mode: str) -> dict[str, Any]:
        assert mode == "paper"
        return await slow_value({"server_profit_model": {"pnl": 3.0}})

    async def balance(mode: str) -> float:
        assert mode == "paper"
        return await slow_value(1_000.0)

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
    service._strategy_context_position_performance = (  # type: ignore[method-assign]
        lambda _mode: slow_value(
            {
                "daily_perf": {"today_total_pnl": 1.0},
                "today_side_perf": {"long": {"pnl": 1.0}},
                "multiday_side_perf": {"long": {"pnl": 1.0}},
                "symbol_side_perf": {"ETH/USDT|long": {"pnl": 2.0}},
            }
        )
    )
    service.entry_position_exposure = Exposure()
    service.entry_symbol_universe = EntrySymbolUniversePolicy(lambda symbol: str(symbol or ""))
    service.allocated_order_balance = balance
    service.entry_strategy_mode_context = EntryStrategyModeContextPolicy()
    service.strategy_learning_service = None
    service._refresh_dynamic_capacity = (  # type: ignore[method-assign]
        lambda **kwargs: {
            **kwargs["strategy_context"],
            "account_equity": kwargs["account_equity"],
        }
    )
    service._json_safe_payload = lambda value: value  # type: ignore[method-assign]

    started_at = asyncio.get_running_loop().time()
    result = await service._strategy_mode_context(
        "paper",
        {"mode": "uptrend_continuation", "confidence": 0.6},
        open_positions=[{"symbol": "ETH/USDT"}],
    )
    elapsed = asyncio.get_running_loop().time() - started_at

    task = service._strategy_context_performance_refresh_tasks()["paper"]
    await task
    result = await service._strategy_mode_context(
        "paper",
        {"mode": "uptrend_continuation", "confidence": 0.6},
        open_positions=[{"symbol": "ETH/USDT"}],
    )

    assert peak_fetches == trading_service.STRATEGY_CONTEXT_IO_CONCURRENCY
    assert elapsed < 0.3
    assert result["account_equity"] == 1_000.0
    assert result["symbol_side_performance"] == {"ETH/USDT|long": {"pnl": 2.0}}
    assert result["model_contribution_performance"] == {"server_profit_model": {"pnl": 3.0}}
    assert result["strategy_context_performance"]["status"] == "fresh"
    assert result["strategy_context_runtime"]["parallel_context_fetch"] is False


@pytest.mark.asyncio
async def test_strategy_context_performance_snapshot_reuses_stale_value_while_refresh_runs() -> None:
    service = object.__new__(TradingService)
    service._json_safe_payload = lambda value: value  # type: ignore[method-assign]
    service._strategy_context_performance_snapshot_cache = {
        "paper": {
            "created_at": datetime.now(UTC) - timedelta(seconds=30),
            "values": {"daily_perf": {"today_total_pnl": 7.0}},
            "version": 4,
            "refresh_timings": {},
            "failed_labels": [],
        }
    }
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def slow_value(value: Any) -> Any:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return value

    service.daily_performance_service = SimpleNamespace(
        state=lambda _mode: slow_value({"today_total_pnl": 8.0})
    )
    service._today_side_performance = lambda _mode: slow_value({"long": {"pnl": 1.0}})
    service._multiday_side_performance = lambda _mode: slow_value({"long": {"pnl": 2.0}})
    service._recent_symbol_side_performance = lambda _mode: slow_value({"BTC/USDT|long": {}})
    service._recent_model_contribution_performance = lambda _mode: slow_value({"expert": {}})
    service._strategy_context_position_performance = (  # type: ignore[method-assign]
        lambda _mode: slow_value(
            {
                "daily_perf": {"today_total_pnl": 8.0},
                "today_side_perf": {"long": {"pnl": 1.0}},
                "multiday_side_perf": {"long": {"pnl": 2.0}},
                "symbol_side_perf": {"BTC/USDT|long": {}},
            }
        )
    )

    values, snapshot = service._recent_strategy_context_performance_snapshot("paper")
    task = service._start_strategy_context_performance_refresh("paper")
    same_task = service._start_strategy_context_performance_refresh("paper")

    assert values == {"daily_perf": {"today_total_pnl": 7.0}}
    assert snapshot["status"] == "stale"
    assert task is same_task
    await started.wait()
    assert calls <= trading_service.STRATEGY_CONTEXT_IO_CONCURRENCY

    release.set()
    await task

    refreshed_values, refreshed_snapshot = service._recent_strategy_context_performance_snapshot("paper")
    assert refreshed_snapshot["status"] == "fresh"
    assert refreshed_snapshot["version"] == 5
    assert refreshed_values is not None
    assert refreshed_values["daily_perf"]["today_total_pnl"] == 8.0


@pytest.mark.asyncio
async def test_strategy_context_performance_refresh_failure_preserves_last_valid_snapshot() -> None:
    service = object.__new__(TradingService)
    service._json_safe_payload = lambda value: value  # type: ignore[method-assign]
    previous_values = {"daily_perf": {"today_total_pnl": 7.0}}
    service._strategy_context_performance_snapshot_cache = {
        "paper": {
            "created_at": datetime.now(UTC) - timedelta(seconds=30),
            "values": previous_values,
            "version": 4,
            "refresh_timings": {},
            "failed_labels": [],
        }
    }

    async def fail(_mode: str) -> dict[str, Any]:
        raise RuntimeError("database unavailable")

    service.daily_performance_service = SimpleNamespace(state=fail)
    service._today_side_performance = fail
    service._multiday_side_performance = fail
    service._recent_symbol_side_performance = fail
    service._recent_model_contribution_performance = fail
    service._strategy_context_position_performance = fail  # type: ignore[method-assign]

    await service._start_strategy_context_performance_refresh("paper")

    values, snapshot = service._recent_strategy_context_performance_snapshot("paper")
    cached = service._strategy_context_performance_snapshot_cache["paper"]
    assert values == previous_values
    assert snapshot["status"] == "stale"
    assert cached["version"] == 4
    assert len(cached["failed_labels"]) == 5
    assert isinstance(cached["last_refresh_failed_at"], datetime)


@pytest.mark.asyncio
async def test_strategy_context_performance_refresh_times_out_waiting_for_shared_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = object.__new__(TradingService)
    gate = asyncio.Semaphore(1)
    await gate.acquire()
    service._strategy_context_io_semaphore = gate
    monkeypatch.setattr(
        trading_service,
        "STRATEGY_CONTEXT_PERFORMANCE_REFRESH_TIMEOUT_SECONDS",
        0.01,
    )

    async def unexpected_loader() -> dict[str, Any]:
        raise AssertionError("loader must not run before a shared-gate timeout")

    label, succeeded, value, timing = await service._refresh_strategy_context_performance_value(
        "daily_perf",
        unexpected_loader,
    )

    assert label == "daily_perf"
    assert succeeded is False
    assert value is None
    assert timing["status"] == "queue_timeout"
    gate.release()


@pytest.mark.asyncio
async def test_strategy_context_performance_warmup_waits_for_initial_snapshot() -> None:
    service = object.__new__(TradingService)
    completed = asyncio.Event()

    async def refresh() -> None:
        await asyncio.sleep(0)
        completed.set()

    service._start_strategy_context_performance_refresh = lambda _mode: asyncio.create_task(  # type: ignore[method-assign]
        refresh()
    )

    await service._prime_strategy_context_performance_snapshot("paper")

    assert completed.is_set()


@pytest.mark.asyncio
async def test_market_and_position_contexts_share_one_performance_refresh() -> None:
    service = object.__new__(TradingService)
    service._json_safe_payload = lambda value: value  # type: ignore[method-assign]
    service._strategy_context_performance_snapshot_cache = {
        "paper": {
            "created_at": datetime.now(UTC) - timedelta(seconds=30),
            "values": {"daily_perf": {"today_total_pnl": 4.0}},
            "version": 1,
            "refresh_timings": {},
            "failed_labels": [],
        }
    }
    service.entry_position_exposure = SimpleNamespace(context=lambda positions: {"count": len(positions)})
    service.entry_symbol_universe = SimpleNamespace(
        open_position_group_count=lambda positions: len(positions)
    )
    service.entry_strategy_mode_context = EntryStrategyModeContextPolicy()
    service._refresh_dynamic_capacity = (  # type: ignore[method-assign]
        lambda **kwargs: {**kwargs["strategy_context"], "account_equity": kwargs["account_equity"]}
    )

    async def account_equity(_mode: str) -> float:
        return 1_000.0

    service._strategy_context_account_equity = account_equity  # type: ignore[method-assign]
    service.strategy_learning_service = None
    release = asyncio.Event()
    started = asyncio.Event()
    calls = 0

    async def slow_value(value: Any) -> Any:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return value

    service.daily_performance_service = SimpleNamespace(
        state=lambda _mode: slow_value({"today_total_pnl": 5.0})
    )
    service._today_side_performance = lambda _mode: slow_value({"long": {"pnl": 1.0}})
    service._multiday_side_performance = lambda _mode: slow_value({"long": {"pnl": 1.0}})
    service._recent_symbol_side_performance = lambda _mode: slow_value({})
    service._recent_model_contribution_performance = lambda _mode: slow_value({})
    service._strategy_context_position_performance = (  # type: ignore[method-assign]
        lambda _mode: slow_value(
            {
                "daily_perf": {"today_total_pnl": 5.0},
                "today_side_perf": {"long": {"pnl": 1.0}},
                "multiday_side_perf": {"long": {"pnl": 1.0}},
                "symbol_side_perf": {},
            }
        )
    )

    async def build_context(scope: str) -> dict[str, Any]:
        token = trading_service._analysis_scope_context.set(scope)
        try:
            return await service._strategy_mode_context(
                "paper",
                {"mode": "range", "confidence": 0.4},
                open_positions=[{"symbol": "BTC/USDT"}],
            )
        finally:
            trading_service._analysis_scope_context.reset(token)

    market_context, position_context = await asyncio.gather(
        build_context("market"),
        build_context("position"),
    )
    task = service._strategy_context_performance_refresh_tasks()["paper"]

    assert market_context["strategy_context_performance"]["status"] == "stale"
    assert position_context["strategy_context_performance"]["status"] == "stale"
    await started.wait()
    assert calls <= trading_service.STRATEGY_CONTEXT_IO_CONCURRENCY

    release.set()
    await task
    assert calls == 2


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
