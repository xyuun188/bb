from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.decision import AIDecision
from models.trade import Position
from web_dashboard.api import dashboard


@pytest.mark.asyncio
async def test_closed_ledger_read_model_builds_once_across_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dashboard._clear_dashboard_heavy_cache("closed-position-ledger")
    builds = {"count": 0}

    async def fake_build(*_args: Any, **_kwargs: Any) -> tuple[list[dict[str, Any]], int, int, int, str]:
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
async def test_dashboard_startup_warmup_only_primes_bounded_okx_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def prime_positions(_mode: str) -> None:
        calls.append("positions")

    async def prime_balance(_mode: str) -> None:
        calls.append("balance")

    async def forbidden_ledger(_mode: str) -> None:
        raise AssertionError("closed ledger must not run during dashboard startup")

    monkeypatch.setattr(dashboard, "_refresh_dashboard_okx_position_cache", prime_positions)
    monkeypatch.setattr(dashboard, "_refresh_dashboard_okx_balance_cache", prime_balance)
    monkeypatch.setattr(
        dashboard,
        "_warm_dashboard_closed_position_ledger_cache",
        forbidden_ledger,
    )

    await dashboard.warm_dashboard_read_caches("paper")

    assert sorted(calls) == ["balance", "positions"]


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
