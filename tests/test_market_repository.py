from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from config.settings import settings
from db.repositories.market_repo import MarketRepository
from db.session import close_db, get_session_ctx, init_db


@pytest.mark.asyncio
async def test_market_repository_bulk_kline_upsert_updates_and_inserts_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    await close_db()
    db_path = tmp_path / "market-repository.db"
    monkeypatch.setattr(settings, "database_url", f"sqlite+aiosqlite:///{db_path.as_posix()}")
    await init_db()
    first = datetime(2026, 7, 10, 0, 0, tzinfo=UTC)
    second = first + timedelta(minutes=1)

    try:
        async with get_session_ctx() as session:
            repo = MarketRepository(session)
            assert await repo.upsert_klines_bulk(
                "BTC/USDT",
                "1m",
                [
                    (first, {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0}),
                    (second, {"open": 101.0, "high": 103.0, "low": 100.0, "close": 102.0, "volume": 2.0}),
                ],
            ) == 2
        async with get_session_ctx() as session:
            repo = MarketRepository(session)
            assert await repo.upsert_klines_bulk(
                "BTC/USDT",
                "1m",
                [(first, {"open": 100.0, "high": 104.0, "low": 99.0, "close": 103.0, "volume": 3.0})],
            ) == 1
            rows = await repo.get_klines("BTC/USDT", "1m", limit=10)
    finally:
        await close_db()

    assert len(rows) == 2
    assert rows[0].close == 103.0
    assert rows[0].volume == 3.0


@pytest.mark.asyncio
async def test_market_repository_loads_multiple_timeframes_with_individual_limits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    await close_db()
    db_path = tmp_path / "market-repository-timeframes.db"
    monkeypatch.setattr(settings, "database_url", f"sqlite+aiosqlite:///{db_path.as_posix()}")
    await init_db()
    started_at = datetime(2026, 7, 10, 0, 0, tzinfo=UTC)

    def payload(price: float) -> dict[str, float]:
        return {
            "open": price,
            "high": price + 1.0,
            "low": price - 1.0,
            "close": price + 0.5,
            "volume": price,
        }

    try:
        async with get_session_ctx() as session:
            repo = MarketRepository(session)
            await repo.upsert_klines_bulk(
                "BTC/USDT",
                "1m",
                [
                    (started_at + timedelta(minutes=index), payload(100.0 + index))
                    for index in range(5)
                ],
            )
            await repo.upsert_klines_bulk(
                "BTC/USDT",
                "1h",
                [
                    (started_at + timedelta(hours=index), payload(200.0 + index))
                    for index in range(3)
                ],
            )
        async with get_session_ctx() as session:
            repo = MarketRepository(session)
            grouped = await repo.get_klines_for_timeframes(
                "BTC/USDT",
                {"1m": 2, "1h": 1},
            )
    finally:
        await close_db()

    assert [row.close for row in grouped["1m"]] == [103.5, 104.5]
    assert [row.close for row in grouped["1h"]] == [202.5]
