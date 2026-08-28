from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.decision import AIDecision
from models.trade import Position
from services import okx_position_history_store as position_history_store
from web_dashboard.api import dashboard


@pytest.mark.asyncio
async def test_closed_ledger_read_model_builds_once_across_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dashboard._clear_dashboard_heavy_cache("closed-position-ledger")
    builds = {"count": 0}

    async def fake_build(
        *_args: Any, **_kwargs: Any
    ) -> tuple[list[dict[str, Any]], int, int, int, str]:
        builds["count"] += 1
        assert _kwargs["page"] == 1
        assert _kwargs["page_size"] == 5000
        assert _kwargs["paginate"] is False
        rows = [{"row": 1}, {"row": 2}, {"row": 3}]
        return (rows, len(rows), 1, 1, "test")

    monkeypatch.setattr(dashboard, "_dashboard_closed_position_ledger_rows_uncached", fake_build)

    first = await dashboard._dashboard_closed_position_ledger_rows(
        object(),
        object(),
        mode="paper",
        page=1,
        page_size=2,
    )
    second = await dashboard._dashboard_closed_position_ledger_rows(
        object(),
        object(),
        mode="paper",
        page=2,
        page_size=2,
    )
    full = await dashboard._dashboard_closed_position_ledger_rows(
        object(),
        object(),
        mode="paper",
        paginate=False,
    )

    assert first[:4] == ([{"row": 1}, {"row": 2}], 3, 1, 2)
    assert second[:4] == ([{"row": 3}], 3, 2, 2)
    assert full[0] == [{"row": 1}, {"row": 2}, {"row": 3}]
    assert builds["count"] == 1


@pytest.mark.asyncio
async def test_closed_ledger_rebuilds_when_okx_history_watermark_is_newer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dashboard._clear_dashboard_heavy_cache("closed-position-ledger")
    cache_key = dashboard._dashboard_closed_ledger_cache_key("paper", None)
    cached_at = datetime.now(UTC) - timedelta(minutes=2)
    dashboard._dashboard_heavy_cache[cache_key] = (
        cached_at,
        ([{"build": "stale"}], 1, 1, 1, "okx_authoritative"),
    )
    monkeypatch.setattr(
        position_history_store,
        "load_okx_position_history_watermark",
        lambda _mode: cached_at + timedelta(minutes=1),
    )
    monkeypatch.setattr(
        dashboard,
        "_load_dashboard_closed_ledger_snapshot",
        lambda **_kwargs: None,
    )

    async def fake_build(*_args: Any, **_kwargs: Any):
        return ([{"build": "fresh"}], 1, 1, 1, "okx_authoritative")

    monkeypatch.setattr(dashboard, "_dashboard_closed_position_ledger_rows_uncached", fake_build)

    result = await dashboard._dashboard_closed_position_ledger_rows(
        object(),
        object(),
        mode="paper",
    )

    assert result[0] == [{"build": "fresh"}]
    dashboard._clear_dashboard_heavy_cache("closed-position-ledger")


@pytest.mark.asyncio
async def test_closed_ledger_stale_value_returns_while_background_refreshes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dashboard._clear_dashboard_heavy_cache("closed-position-ledger")
    cache_key = dashboard._dashboard_closed_ledger_cache_key("paper", None)
    stale_payload = ([{"build": 1}], 1, 1, 1, "test")
    dashboard._dashboard_heavy_cache[cache_key] = (
        datetime.now(UTC)
        - timedelta(seconds=dashboard._DASHBOARD_CLOSED_LEDGER_CACHE_TTL_SECONDS + 1),
        stale_payload,
    )
    release_refresh = asyncio.Event()

    async def fake_refresh(*_args: Any, **_kwargs: Any):
        await release_refresh.wait()
        return dashboard._dashboard_heavy_cache_set(
            cache_key,
            ([{"build": 2}], 1, 1, 1, "test"),
        )

    monkeypatch.setattr(dashboard, "_rebuild_dashboard_closed_ledger_cache", fake_refresh)

    stale = await dashboard._dashboard_closed_position_ledger_rows(
        object(),
        object(),
        mode="paper",
    )
    task = dashboard._dashboard_closed_ledger_refresh_tasks[cache_key]
    assert stale[0] == [{"build": 1}]
    assert not task.done()

    release_refresh.set()
    await task
    refreshed = await dashboard._dashboard_closed_position_ledger_rows(
        object(),
        object(),
        mode="paper",
    )
    assert refreshed[0] == [{"build": 2}]
    dashboard._clear_dashboard_heavy_cache("closed-position-ledger")


@pytest.mark.asyncio
async def test_closed_ledger_cold_memory_uses_persisted_snapshot_before_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dashboard._clear_dashboard_heavy_cache("closed-position-ledger")
    started: list[tuple[Any, ...]] = []
    persisted = ([{"build": "persisted"}], 1, 1, 1, "okx_authoritative")

    monkeypatch.setattr(
        dashboard,
        "_load_dashboard_closed_ledger_snapshot",
        lambda **_kwargs: (datetime.now(UTC) - timedelta(minutes=5), persisted),
    )
    monkeypatch.setattr(
        dashboard,
        "_start_dashboard_closed_ledger_refresh",
        lambda cache_key, **_kwargs: started.append(cache_key),
    )

    result = await dashboard._dashboard_closed_position_ledger_rows(
        object(),
        object(),
        mode="paper",
    )

    assert result[0] == [{"build": "persisted"}]
    assert started == [dashboard._dashboard_closed_ledger_cache_key("paper", None)]
    dashboard._clear_dashboard_heavy_cache("closed-position-ledger")


@pytest.mark.asyncio
async def test_dashboard_startup_warmup_primes_bounded_first_visit_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def prime_positions(_mode: str) -> None:
        calls.append("positions")

    async def prime_balance(_mode: str) -> None:
        calls.append("balance")

    async def prime_strategy_learning(_mode: str) -> None:
        calls.append("strategy_learning")

    async def prime_model_registry() -> dict[str, Any]:
        calls.append("model_registry")
        return {}

    async def prime_data_collection() -> None:
        calls.append("data_collection")

    async def forbidden_ledger(_mode: str) -> None:
        raise AssertionError("closed ledger must not run during dashboard startup")

    monkeypatch.setattr(dashboard, "_refresh_dashboard_okx_position_cache", prime_positions)
    monkeypatch.setattr(dashboard, "_refresh_dashboard_okx_balance_cache", prime_balance)
    monkeypatch.setattr(
        dashboard,
        "_warm_dashboard_strategy_learning_cache",
        prime_strategy_learning,
    )
    monkeypatch.setattr(
        dashboard,
        "get_model_training_registry_status",
        prime_model_registry,
    )
    monkeypatch.setattr(
        dashboard,
        "_warm_dashboard_data_collection_cache",
        prime_data_collection,
    )
    monkeypatch.setattr(
        dashboard,
        "_warm_dashboard_closed_position_ledger_cache",
        forbidden_ledger,
    )

    await dashboard.warm_dashboard_read_caches("paper")

    assert sorted(calls) == ["balance", "model_registry", "positions"]
    assert "strategy_learning" not in calls
    assert "data_collection" not in calls


@pytest.mark.asyncio
async def test_strategy_learning_request_snapshot_skips_repeated_watermark_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dashboard._clear_dashboard_heavy_cache("strategy-learning")


@pytest.mark.asyncio
async def test_strategy_learning_timeout_returns_persisted_stale_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    dashboard._clear_dashboard_heavy_cache("strategy-learning")
    monkeypatch.setattr(dashboard, "_STRATEGY_LEARNING_SNAPSHOT_DIR", tmp_path)

    async def watermark(**_kwargs: Any) -> tuple[str]:
        return ("v1",)

    class HealthyStrategyLearningStub:
        async def dashboard_payload(self, **_kwargs: Any) -> dict[str, Any]:
            return {
                "mode": "paper",
                "schedule": {
                    "scheduler_mode": "shadow_validation",
                    "reason": "healthy_snapshot",
                    "candidates": [],
                },
            }

    monkeypatch.setattr(dashboard, "_strategy_learning_watermark_for_request", watermark)
    monkeypatch.setattr(
        dashboard,
        "_trading_service",
        SimpleNamespace(strategy_learning_service=HealthyStrategyLearningStub()),
    )
    fresh = await dashboard.get_strategy_learning(mode="paper")
    assert fresh["status"] == "ok"
    dashboard._clear_dashboard_heavy_cache("strategy-learning")

    async def failing_watermark(**_kwargs: Any) -> tuple[str]:
        return ("v2",)

    class TimeoutStrategyLearningStub:
        async def dashboard_payload(self, **_kwargs: Any) -> dict[str, Any]:
            raise TimeoutError("slow strategy query")

    monkeypatch.setattr(dashboard, "_strategy_learning_watermark_for_request", failing_watermark)
    monkeypatch.setattr(
        dashboard,
        "_trading_service",
        SimpleNamespace(strategy_learning_service=TimeoutStrategyLearningStub()),
    )
    stale = await dashboard.get_strategy_learning(mode="paper")
    assert stale["status"] == "stale"
    assert stale["stale"] is True
    assert stale["fallback_source"] == "persisted_last_success"
    assert stale["schedule"]["scheduler_mode"] == "stale_snapshot"
    assert stale["schedule"]["reason"] == "strategy_learning_query_timeout"
    dashboard._clear_dashboard_heavy_cache("strategy-learning")
    watermark_calls = 0
    payload_calls = 0

    async def watermark(**_kwargs: Any) -> tuple[str]:
        nonlocal watermark_calls
        watermark_calls += 1
        return ("v1",)

    class StrategyLearningStub:
        async def dashboard_payload(self, **_kwargs: Any) -> dict[str, Any]:
            nonlocal payload_calls
            payload_calls += 1
            return {"mode": "paper", "schedule": {"status": "ready"}}

    monkeypatch.setattr(dashboard, "_strategy_learning_watermark_for_request", watermark)
    monkeypatch.setattr(
        dashboard,
        "_trading_service",
        SimpleNamespace(strategy_learning_service=StrategyLearningStub()),
    )

    first = await dashboard.get_strategy_learning(mode="paper")
    second = await dashboard.get_strategy_learning(mode="paper")

    assert first == second
    assert watermark_calls == 1
    assert payload_calls == 1
    dashboard._clear_dashboard_heavy_cache("strategy-learning")


def test_strategy_learning_summary_keeps_rendered_fields_without_duplicate_rows() -> None:
    payload = {
        "feedback": {"generated_at": "2026-07-28T00:00:00+00:00"},
        "schedule": {
            "leading_candidate": {"id": "side-long", "version": 2},
            "runtime": {
                "historical_prior_context_enabled": True,
                "execution_owners": ["live_ml_profit_contract"],
                "continuous_strategy_routing": {"oversized": [1] * 1000},
                "governed_profiles": [{"oversized": [1] * 1000}],
            },
            "candidates": [
                {
                    "id": "side-long",
                    "version": 2,
                    "rank": 1,
                    "params": {
                        "selector": {"scope": "side", "side": "long"},
                        "historical_return_distribution": {"return_lcb_pct": 0.2},
                        "unused": [1] * 1000,
                    },
                    "promotion": {
                        "historical_prior_context_eligible": True,
                        "rejection_reasons": [],
                    },
                    "backtest": {
                        "status": "ready",
                        "metrics": {"return_lcb_pct": 0.1},
                        "rows": [{"oversized": [1] * 1000}],
                    },
                    "shadow_validation": {
                        "status": "ready",
                        "metrics": {"return_lcb_pct": 0.05},
                        "rows": [{"oversized": [1] * 1000}],
                    },
                }
            ],
            "backtest": {"rows": [{"oversized": [1] * 1000}]},
            "shadow_validation": {"rows": [{"oversized": [1] * 1000}]},
            "continuous_strategy_routing": {"oversized": [1] * 1000},
            "scheduler_mode": "governed_dynamic_return",
            "governed_candidate_count": 1,
            "rejected_candidate_count": 0,
        },
    }

    summary = dashboard._strategy_learning_dashboard_summary(payload)
    schedule = summary["schedule"]
    candidate = schedule["candidates"][0]

    assert schedule["leading_candidate"] is candidate
    assert candidate["params"]["selector"]["side"] == "long"
    assert candidate["backtest"]["metrics"]["return_lcb_pct"] == 0.1
    assert "rows" not in candidate["backtest"]
    assert "continuous_strategy_routing" not in schedule
    assert "continuous_strategy_routing" not in schedule["runtime"]


def test_profit_attribution_summary_drops_unrendered_reasoning_and_empty_distributions() -> None:
    record = {
        "position_id": 7,
        "symbol": "BTC/USDT",
        "side": "long",
        "realized_pnl": 1.2,
        "notes": ["first", "second", "third"],
        "entry_decision": {
            "id": 9,
            "action": "long",
            "confidence": 0.8,
            "reasoning": "unused" * 1000,
        },
        "signals": {
            "ml": {
                "available": True,
                "side": "long",
                "return_distribution_contract": {"q10": -0.1, "q50": 0.2},
                "unused": [1] * 1000,
            }
        },
        "evidence_status": {
            "ai": {
                "available": True,
                "action": "long",
                "missing_reason": "",
                "unused": [1] * 1000,
            }
        },
        "decision_state": {
            "summary": {
                "final_stage": "local_sync",
                "final_status": "completed",
                "final_reason": "done",
                "unused": [1] * 1000,
            }
        },
        "close_decision": {"reasoning": "unused" * 1000},
    }

    compact = dashboard._profit_attribution_dashboard_record(record)

    assert compact["notes"] == ["first", "second"]
    assert "reasoning" not in compact["entry_decision"]
    assert "unused" not in compact["signals"]["ml"]
    assert compact["decision_state"]["summary"]["final_stage"] == "local_sync"
    assert "close_decision" not in compact


@pytest.mark.asyncio
async def test_profit_attribution_parameter_snapshot_short_circuits_watermark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dashboard._clear_dashboard_heavy_cache("profit-attribution")
    cache_key = ("profit-attribution", "paper", 24, 200)
    cached_payload = {"mode": "paper", "records": [{"position_id": 7}]}
    dashboard._dashboard_heavy_cache_set(cache_key, cached_payload)

    async def fail_watermark(*_args: Any, **_kwargs: Any) -> tuple[Any, ...]:
        raise AssertionError("fresh parameter snapshot must skip the database watermark")

    monkeypatch.setattr(dashboard, "_profit_attribution_watermark", fail_watermark)

    assert await dashboard.get_profit_attribution() == cached_payload

    dashboard._dashboard_heavy_cache[cache_key] = (
        datetime.now(UTC) - timedelta(seconds=31),
        cached_payload,
    )
    assert dashboard._dashboard_heavy_cache_get(cache_key, ttl_seconds=30.0) is None


@pytest.mark.asyncio
async def test_fresh_persisted_closed_ledger_does_not_refresh_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dashboard._clear_dashboard_heavy_cache("closed-position-ledger")
    persisted = ([{"build": "fresh"}], 1, 1, 1, "okx_authoritative")
    refresh_started = False

    monkeypatch.setattr(
        dashboard,
        "_load_dashboard_closed_ledger_snapshot",
        lambda **_kwargs: (datetime.now(UTC), persisted),
    )

    def record_refresh(*_args: Any, **_kwargs: Any) -> None:
        nonlocal refresh_started
        refresh_started = True

    monkeypatch.setattr(
        dashboard,
        "_start_dashboard_closed_ledger_refresh",
        record_refresh,
    )

    result = await dashboard._dashboard_closed_position_ledger_rows(
        object(),
        object(),
        mode="paper",
    )

    assert result[0] == [{"build": "fresh"}]
    assert refresh_started is False
    dashboard._clear_dashboard_heavy_cache("closed-position-ledger")


def test_analysis_payload_bounds_transcripts_and_nested_collections() -> None:
    payload = {
        "reasoning": "x" * 5000,
        "opinions": [{"reasoning": "y" * 5000} for _ in range(120)],
        "nested": {"rows": list(range(120))},
    }

    bounded = dashboard._bounded_dashboard_payload(payload)

    assert len(bounded["reasoning"]) < 1700
    assert bounded["reasoning"].endswith("...")
    assert len(bounded["opinions"]) == 80
    assert len(bounded["opinions"][0]["reasoning"]) < 1700
    assert len(bounded["nested"]["rows"]) == 80


def test_closed_ledger_snapshot_round_trip(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_path = tmp_path / "closed_position_ledger_paper.json"
    monkeypatch.setattr(
        dashboard,
        "_dashboard_closed_ledger_snapshot_path",
        lambda **_kwargs: snapshot_path,
    )
    payload = ([{"row": 1}], 1, 1, 1, "okx_authoritative")

    dashboard._persist_dashboard_closed_ledger_snapshot(
        payload,
        mode="paper",
        model_names=None,
    )
    loaded = dashboard._load_dashboard_closed_ledger_snapshot(
        mode="paper",
        model_names=None,
    )

    assert loaded is not None
    _generated_at, loaded_payload = loaded
    assert loaded_payload == payload


def test_closed_ledger_snapshot_rejects_older_okx_history_watermark(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_path = tmp_path / "closed_position_ledger_paper.json"
    monkeypatch.setattr(
        dashboard,
        "_dashboard_closed_ledger_snapshot_path",
        lambda **_kwargs: snapshot_path,
    )
    dashboard._persist_dashboard_closed_ledger_snapshot(
        ([{"row": "stale"}], 1, 1, 1, "okx_authoritative"),
        mode="paper",
        model_names=None,
    )
    generated_at = datetime.fromisoformat(
        json.loads(snapshot_path.read_text(encoding="utf-8"))["generated_at"]
    )
    monkeypatch.setattr(
        position_history_store,
        "load_okx_position_history_watermark",
        lambda _mode: generated_at + timedelta(seconds=1),
    )

    assert dashboard._load_dashboard_closed_ledger_snapshot(
        mode="paper",
        model_names=None,
    ) is None


def test_official_history_matches_only_instrument_scoped_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened_at = datetime(2026, 7, 28, 8, 0, tzinfo=UTC)
    closed_at = opened_at + timedelta(minutes=30)

    def order(inst_id: str, order_id: str, side: str, filled_at: datetime) -> Any:
        return SimpleNamespace(
            okx_inst_id=inst_id,
            symbol=inst_id.replace("-SWAP", "").replace("-", "/"),
            exchange_order_id=order_id,
            side=side,
            quantity=1.0,
            price=100.0,
            fee=0.01,
            filled_at=filled_at,
            created_at=filled_at,
            okx_sync_status="okx_confirmed",
            okx_fill_contracts=1.0,
            okx_fill_pnl=0.5 if side == "sell" else 0.0,
            okx_trade_ids=f"trade-{order_id}",
            okx_raw_fills={
                "base_quantity": 1.0,
                "contracts": 1.0,
                "contract_size": 1.0,
                "avg_price": 100.0,
                "fill_pnl": 0.5 if side == "sell" else 0.0,
                "fee_abs": 0.01,
            },
        )

    relevant_orders = [
        order("BTC-USDT-SWAP", "btc-entry", "buy", opened_at),
        order("BTC-USDT-SWAP", "btc-close", "sell", closed_at),
    ]
    unrelated_orders = [
        order("ETH-USDT-SWAP", f"eth-{index}", "buy", opened_at) for index in range(200)
    ]
    relevant_position = SimpleNamespace(
        id=1,
        okx_inst_id="BTC-USDT-SWAP",
        symbol="BTC/USDT",
        okx_pos_id="btc-pos",
        side="long",
        created_at=opened_at,
        closed_at=closed_at,
    )
    unrelated_positions = [
        SimpleNamespace(
            id=index + 2,
            okx_inst_id="ETH-USDT-SWAP",
            symbol="ETH/USDT",
            okx_pos_id=f"eth-pos-{index}",
            side="long",
            created_at=opened_at,
            closed_at=closed_at,
        )
        for index in range(200)
    ]
    row = {
        "instId": "BTC-USDT-SWAP",
        "posId": "btc-pos",
        "posSide": "long",
        "openAvgPx": "100",
        "closeAvgPx": "101",
        "openMaxPos": "1",
        "closeTotalPos": "1",
        "realizedPnl": "0.48",
        "pnl": "0.5",
        "fundingFee": "0",
        "type": "2",
        "cTime": str(int(opened_at.timestamp() * 1000)),
        "uTime": str(int(closed_at.timestamp() * 1000)),
        "_dashboard_entry_order_ids": ["btc-entry"],
        "_dashboard_close_order_ids": ["btc-close"],
    }
    match_calls = 0
    original_match = dashboard._dashboard_order_matches_position_history_window

    def counted_match(*args: Any, **kwargs: Any) -> bool:
        nonlocal match_calls
        match_calls += 1
        return original_match(*args, **kwargs)

    monkeypatch.setattr(
        dashboard,
        "_dashboard_order_matches_position_history_window",
        counted_match,
    )

    result = dashboard._dashboard_position_history_official_rows_as_groups(
        [row],
        [],
        mode="paper",
        order_rows=[*relevant_orders, *unrelated_orders],
        closed_rows=[relevant_position, *unrelated_positions],
    )

    assert len(result) == 1
    assert result[0]["position_ids"] == [1]
    assert {fill["order_id"] for fill in result[0]["linked_fills"]} == {
        "btc-entry",
        "btc-close",
    }
    assert match_calls == len(relevant_orders)


@pytest.mark.asyncio
async def test_profit_attribution_watermark_ignores_unrelated_new_decisions(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'profit-watermark.db').as_posix()}",
    )
    await init_db()
    now = datetime.now(UTC)
    since = now - timedelta(hours=24)
    try:
        async with get_session_ctx() as session:
            position = Position(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="BTC/USDT",
                side="long",
                quantity=0.1,
                entry_price=100.0,
                current_price=101.0,
                realized_pnl=0.1,
                is_open=False,
                closed_at=now - timedelta(minutes=10),
                created_at=now - timedelta(hours=1),
                updated_at=now - timedelta(minutes=9),
            )
            session.add(position)

        async with get_session_ctx() as session:
            before = await dashboard._profit_attribution_watermark(
                session,
                selected_mode="paper",
                since=since,
            )

        async with get_session_ctx() as session:
            session.add(
                AIDecision(
                    model_name="ensemble_trader",
                    symbol="ETH/USDT",
                    action="hold",
                    confidence=0.5,
                    is_paper=True,
                    created_at=now,
                )
            )

        async with get_session_ctx() as session:
            after_decision = await dashboard._profit_attribution_watermark(
                session,
                selected_mode="paper",
                since=since,
            )
            persisted_position = await session.get(Position, position.id)
            assert persisted_position is not None
            persisted_position.updated_at = now

        async with get_session_ctx() as session:
            after_position_update = await dashboard._profit_attribution_watermark(
                session,
                selected_mode="paper",
                since=since,
            )

        assert after_decision == before
        assert after_position_update != before
    finally:
        await close_db()
