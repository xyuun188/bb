from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
import pytest

from backtest import data_replay


class _FakeOkxClient:
    async def fetch_ohlcv(self, _symbol: str, _timeframe: str, *, limit: int):
        return [
            [1_767_225_600_000 + index * 3_600_000, 100, 102, 99, 101, 1000]
            for index in range(limit)
        ]

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_okx_history_records_source_provenance(monkeypatch) -> None:
    monkeypatch.setattr(data_replay, "OKXRestClient", _FakeOkxClient)

    frame = await data_replay.load_historical_from_okx("BTC/USDT", "1h", limit=3)

    assert isinstance(frame, pd.DataFrame)
    assert frame.attrs["bb_data_source"] == "okx_rest_api"


@pytest.mark.asyncio
async def test_db_history_records_okx_fallback_provenance(monkeypatch) -> None:
    from db import session as db_session
    from db.repositories import market_repo

    @asynccontextmanager
    async def fake_session_ctx():
        yield object()

    async def no_klines(_self, _symbol: str, _timeframe: str, _limit: int):
        return []

    monkeypatch.setattr(db_session, "get_session_ctx", fake_session_ctx)
    monkeypatch.setattr(market_repo.MarketRepository, "get_klines", no_klines)
    monkeypatch.setattr(data_replay, "OKXRestClient", _FakeOkxClient)

    frame = await data_replay.load_historical_from_db("BTC/USDT", "1h", limit=3)

    assert frame.attrs["bb_data_source"] == "okx_rest_api_fallback_for_postgresql"
