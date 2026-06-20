from __future__ import annotations

import asyncio
from typing import Any

import pytest

from services import data_service as data_service_module
from services.data_service import DataService


def _service() -> DataService:
    service = object.__new__(DataService)
    service._sentiment_cache = {}
    service._headlines_cache = {}
    service._news_items_cache = {}
    service._indicator_snapshot_cache = {}
    service._indicator_snapshot_tasks = {}
    service._indicator_remote_refresh_semaphore = asyncio.Semaphore(
        max(1, int(data_service_module.INDICATOR_REMOTE_REFRESH_CONCURRENCY))
    )
    service._kline_fetch_tasks = {}
    service._kline_background_refresh_tasks = {}
    service._kline_refresh_scheduled_at = {}
    service._derivatives_cache = {}
    service._derivatives_refresh_tasks = {}
    service._ticker_persisted_at = {}
    service._ticker_persist_inflight = set()
    return service


@pytest.mark.asyncio
async def test_indicator_snapshot_persists_all_training_timeframes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    fetch_calls: list[tuple[str, int]] = []
    persisted: list[tuple[str, str, int]] = []

    class FakeRestClient:
        async def fetch_ohlcv(
            self,
            symbol: str,
            timeframe: str = "1h",
            limit: int = 100,
        ) -> list[list[float]]:
            fetch_calls.append((timeframe, limit))
            return [
                [1_700_000_000_000 + index * 60_000, 100.0, 101.0, 99.0, 100.5, 10.0]
                for index in range(limit)
            ]

    async def fake_persist(symbol: str, timeframe: str, klines: list[Any]) -> None:
        persisted.append((symbol, timeframe, len(klines)))

    service.rest_client = FakeRestClient()
    service._persist_klines = fake_persist  # type: ignore[method-assign]
    monkeypatch.setattr(data_service_module, "compute_all_indicators", lambda df: df)
    monkeypatch.setattr(
        data_service_module,
        "extract_latest_features",
        lambda df: {"close": float(df["close"].iloc[-1])},
    )
    monkeypatch.setattr(service, "_kline_anomaly_snapshot", lambda df: {"abnormal": 0})

    features = await service._get_indicator_snapshot("BTC/USDT")

    assert features["close"] == 100.5
    assert set(fetch_calls) == set(data_service_module.KLINE_PERSIST_TIMEFRAME_LIMITS.items())
    assert {timeframe for _symbol, timeframe, _count in persisted} == {
        "1m",
        "5m",
        "15m",
        "1h",
    }
    assert dict((timeframe, count) for _symbol, timeframe, count in persisted)["1h"] == 100


@pytest.mark.asyncio
async def test_feature_vector_source_timeouts_do_not_block_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    service._last_sentiment_update = None
    service._sentiment_refresh_task = None
    service.ws_client = type("Ws", (), {"latest_tickers": {}})()
    monkeypatch.setattr(data_service_module.settings, "sentiment_blocking_timeout_seconds", 0.01)

    async def noop_sentiment(_symbols):
        return None

    async def slow_source(_symbol):
        await asyncio.sleep(60)
        return {"last_price": 100.0}

    async def fast_source(_symbol):
        return {"current_price": 101.0, "close": 101.0}

    service.refresh_sentiment = noop_sentiment  # type: ignore[method-assign]
    service._get_ticker_snapshot = slow_source  # type: ignore[method-assign]
    service._get_indicator_snapshot = fast_source  # type: ignore[method-assign]
    service._get_derivatives_snapshot = slow_source  # type: ignore[method-assign]

    fv = await service.get_feature_vector("BTC/USDT")

    assert fv.symbol == "BTC/USDT"
    assert fv.current_price == 101.0


@pytest.mark.asyncio
async def test_indicator_snapshot_uses_minute_returns_and_hourly_trend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()

    class FakeRestClient:
        async def fetch_ohlcv(
            self,
            symbol: str,
            timeframe: str = "1h",
            limit: int = 100,
        ) -> list[list[float]]:
            base_price = {
                "1m": 10.0,
                "5m": 20.0,
                "15m": 30.0,
                "1h": 100.0,
            }[timeframe]
            return [
                [
                    1_700_000_000_000 + index * 60_000,
                    base_price + index * 0.01,
                    base_price + index * 0.01 + 0.1,
                    base_price + index * 0.01 - 0.1,
                    base_price + index * 0.01,
                    10.0,
                ]
                for index in range(limit)
            ]

    async def fake_persist(symbol: str, timeframe: str, klines: list[Any]) -> None:
        return None

    def fake_extract_latest_features(df: Any) -> dict[str, float]:
        latest_close = float(df["close"].iloc[-1])
        if latest_close < 15:
            return {
                "close": latest_close,
                "volume": 10.0,
                "returns_1": 0.011,
                "returns_5": 0.055,
                "returns_20": 0.12,
                "volatility_20": 0.033,
                "price_vs_sma20": -0.25,
                "price_vs_sma50": -0.35,
            }
        return {
            "close": latest_close,
            "volume": 10.0,
            "returns_1": -0.001,
            "returns_5": -0.002,
            "returns_20": -0.003,
            "volatility_20": 0.004,
            "price_vs_sma20": 0.21,
            "price_vs_sma50": 0.34,
        }

    service.rest_client = FakeRestClient()
    service._persist_klines = fake_persist  # type: ignore[method-assign]
    monkeypatch.setattr(data_service_module, "compute_all_indicators", lambda df: df)
    monkeypatch.setattr(
        data_service_module,
        "extract_latest_features",
        fake_extract_latest_features,
    )
    monkeypatch.setattr(service, "_kline_anomaly_snapshot", lambda df: {"abnormal": 0})

    features = await service._get_indicator_snapshot("BTC/USDT")

    assert features["short_returns_timeframe"] == "1m"
    assert features["technical_indicator_timeframe"] == "1h"
    assert features["returns_1"] == pytest.approx(0.011)
    assert features["returns_5"] == pytest.approx(0.055)
    assert features["returns_20"] == pytest.approx(0.12)
    assert features["volatility_20"] == pytest.approx(0.033)
    assert features["price_vs_sma20"] == pytest.approx(0.21)
    assert features["price_vs_sma50"] == pytest.approx(0.34)


@pytest.mark.asyncio
async def test_indicator_snapshot_uses_cached_klines_before_okx_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    fetch_calls = 0

    class FakeRestClient:
        async def fetch_ohlcv(
            self,
            symbol: str,
            timeframe: str = "1h",
            limit: int = 100,
        ) -> list[list[float]]:
            nonlocal fetch_calls
            fetch_calls += 1
            return []

    async def cached_klines(symbol: str, timeframe: str, limit: int) -> list[list[float]]:
        return [
            [1_700_000_000_000 + index * 60_000, 100.0, 101.0, 99.0, 100.5, 10.0]
            for index in range(limit)
        ]

    service.rest_client = FakeRestClient()
    service._load_recent_cached_klines = cached_klines  # type: ignore[method-assign]
    monkeypatch.setattr(data_service_module, "compute_all_indicators", lambda df: df)
    monkeypatch.setattr(
        data_service_module,
        "extract_latest_features",
        lambda df: {"close": float(df["close"].iloc[-1]), "returns_5": 0.01},
    )
    monkeypatch.setattr(service, "_kline_anomaly_snapshot", lambda df: {"abnormal": 0})
    monkeypatch.setattr(service, "_schedule_kline_background_refresh", lambda _symbol: None)

    features = await service._get_indicator_snapshot("BTC/USDT")

    assert features["close"] == 100.5
    assert fetch_calls == 0


@pytest.mark.asyncio
async def test_concurrent_kline_fetches_are_deduplicated() -> None:
    service = _service()
    fetch_calls = 0
    release = asyncio.Event()

    class FakeRestClient:
        async def fetch_ohlcv(
            self,
            symbol: str,
            timeframe: str = "1h",
            limit: int = 100,
        ) -> list[list[float]]:
            nonlocal fetch_calls
            fetch_calls += 1
            await release.wait()
            return [
                [1_700_000_000_000 + index * 60_000, 100.0, 101.0, 99.0, 100.5, 10.0]
                for index in range(limit)
            ]

    async def fake_persist(symbol: str, timeframe: str, klines: list[Any]) -> None:
        return None

    async def no_cache(symbol: str, timeframe: str, limit: int) -> list[list[float]]:
        return []

    service.rest_client = FakeRestClient()
    service._persist_klines = fake_persist  # type: ignore[method-assign]
    service._load_recent_cached_klines = no_cache  # type: ignore[method-assign]

    first = asyncio.create_task(service._fetch_and_persist_klines("BTC/USDT", "1m", 120))
    second = asyncio.create_task(service._fetch_and_persist_klines("BTC/USDT", "1m", 120))
    await asyncio.sleep(0)
    release.set()
    first_result, second_result = await asyncio.gather(first, second)

    assert fetch_calls == 1
    assert first_result[0] == "1m"
    assert second_result[0] == "1m"
    assert len(first_result[1]) == 120
    assert len(second_result[1]) == 120


@pytest.mark.asyncio
async def test_background_kline_refresh_does_not_block_foreground_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    release = asyncio.Event()
    fetch_calls: list[str] = []

    async def slow_background(symbol: str, timeframe: str, limit: int) -> list[Any]:
        fetch_calls.append(timeframe)
        await release.wait()
        return []

    async def fast_foreground(symbol: str, timeframe: str, limit: int) -> list[Any]:
        return [["foreground"]]

    monkeypatch.setattr(service, "_fetch_and_persist_klines_uncached", slow_background)
    service._schedule_kline_background_refresh("BTC/USDT")
    await asyncio.sleep(0)

    monkeypatch.setattr(service, "_fetch_and_persist_klines_uncached", fast_foreground)
    timeframe, rows = await service._fetch_and_persist_klines("BTC/USDT", "1m", 120)
    release.set()
    await asyncio.sleep(0)

    assert timeframe == "1m"
    assert rows == [["foreground"]]
    assert "1m" in fetch_calls


def test_feature_vector_keeps_market_feature_source_timeframes() -> None:
    from data_feed.feature_vector import build_feature_vector

    vector = build_feature_vector(
        "BTC/USDT",
        indicators={
            "returns_1": 0.01,
            "returns_5": 0.02,
            "volatility_20": 0.03,
            "short_returns_timeframe": "1m",
            "technical_indicator_timeframe": "1h",
        },
    )

    assert vector.short_returns_timeframe == "1m"
    assert vector.technical_indicator_timeframe == "1h"
    assert "short_returns=1m" in vector.to_llm_context()
    assert "trend=1h" in vector.to_llm_context()


def test_news_item_summary_keeps_safe_external_url() -> None:
    service = _service()

    item = service._news_item_summary(
        {
            "source": "unit-news",
            "title": "BTC ETF inflows rise",
            "summary": "BTC market update",
            "url": " https://news.example.invalid/article?id=1#quote ",
            "symbols_mentioned": ["BTC"],
        },
        "BTC",
        direct_match=True,
    )

    assert item["url"] == "https://news.example.invalid/article?id=1#quote"


def test_news_item_summary_drops_unsafe_external_url() -> None:
    service = _service()

    for url in (
        "javascript:alert(1)",
        "data:text/html,<script>alert(1)</script>",
        "http://user:password@example.invalid/article",
    ):
        item = service._news_item_summary(
            {
                "source": "unit-news",
                "title": "BTC market update",
                "summary": "BTC market update",
                "url": url,
                "symbols_mentioned": ["BTC"],
            },
            "BTC",
            direct_match=True,
        )

        assert item["url"] == ""


def test_sentiment_cache_never_exposes_unsafe_news_urls() -> None:
    service = _service()

    service._build_sentiment_cache(
        ["BTC/USDT"],
        [
            {
                "source": "unit-news",
                "title": "BTC direct story",
                "summary": "BTC direct story",
                "url": "javascript:alert(1)",
                "symbols_mentioned": ["BTC"],
                "impact_level": 5,
            },
            {
                "source": "safe-news",
                "title": "BTC safe story",
                "summary": "BTC safe story",
                "url": "https://news.example.invalid/btc?src=unit",
                "symbols_mentioned": ["BTC"],
                "impact_level": 4,
            },
        ],
        [],
    )

    urls = [item["url"] for item in service._sentiment_cache["BTC/USDT"]["news_items"]]
    assert "" in urls
    assert "javascript:alert(1)" not in urls
    assert "https://news.example.invalid/btc?src=unit" in urls
