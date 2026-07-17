from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from data_feed.feature_vector import build_feature_vector
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
    service._indicator_snapshot_build_semaphore = asyncio.Semaphore(
        max(1, int(data_service_module.INDICATOR_SNAPSHOT_BUILD_CONCURRENCY))
    )
    service._kline_fetch_tasks = {}
    service._kline_background_refresh_tasks = {}
    service._kline_refresh_scheduled_at = {}
    service._kline_coverage_refresh_task = None
    service._kline_coverage_symbols = []
    service._kline_coverage_index = 0
    service._derivatives_cache = {}
    service._derivatives_refresh_tasks = {}
    service._ticker_persisted_at = {}
    service._ticker_persist_inflight = set()
    service._ticker_persist_semaphore = asyncio.Semaphore(
        data_service_module.TICKER_PERSIST_CONCURRENCY
    )
    service._available_symbols_cache = []
    service._available_symbols_cache_updated_at = None
    service._available_symbols_refresh_task = None
    return service


@pytest.mark.asyncio
async def test_ticker_persistence_is_bounded_below_database_pool_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    active_sessions = 0
    peak_sessions = 0
    persisted_symbols: list[str] = []

    class FakeSessionContext:
        async def __aenter__(self) -> object:
            nonlocal active_sessions, peak_sessions
            active_sessions += 1
            peak_sessions = max(peak_sessions, active_sessions)
            await asyncio.sleep(0)
            return object()

        async def __aexit__(self, *_args: object) -> None:
            nonlocal active_sessions
            await asyncio.sleep(0.01)
            active_sessions -= 1

    class FakeMarketRepository:
        def __init__(self, _session: object) -> None:
            pass

        async def upsert_ticker(self, symbol: str, _payload: dict[str, Any]) -> None:
            persisted_symbols.append(symbol)
            await asyncio.sleep(0.01)

    monkeypatch.setattr(data_service_module, "get_session_ctx", FakeSessionContext)
    monkeypatch.setattr(data_service_module, "MarketRepository", FakeMarketRepository)

    await asyncio.gather(
        *(
            service._persist_ticker_snapshot(
                f"COIN{index}/USDT",
                {
                    "last_price": index + 1,
                    "bid": index + 0.9,
                    "ask": index + 1.1,
                    "timestamp": "2026-07-16T00:00:00Z",
                },
            )
            for index in range(40)
        )
    )

    assert peak_sessions == data_service_module.TICKER_PERSIST_CONCURRENCY
    assert peak_sessions < int(data_service_module.settings.database_pool_size)
    assert len(persisted_symbols) == 40


def test_indicator_snapshot_ignores_incomplete_latest_kline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    now = data_service_module.pd.Timestamp.now(tz="UTC")
    complete_start = now - data_service_module.pd.Timedelta(minutes=22)
    klines = [
        [
            (complete_start + data_service_module.pd.Timedelta(minutes=index)).timestamp() * 1000.0,
            100.0,
            101.0,
            99.0,
            100.0 + index,
            10.0,
        ]
        for index in range(21)
    ]
    latest_incomplete = now.floor("min").timestamp() * 1000.0
    klines.append([latest_incomplete, 100.0, 101.0, 99.0, 999.0, 99999.0])

    monkeypatch.setattr(data_service_module, "compute_all_indicators", lambda df: df)
    monkeypatch.setattr(
        data_service_module,
        "extract_latest_features",
        lambda df: {
            "close": float(df["close"].iloc[-1]),
            "volume": float(df["volume"].iloc[-1]),
        },
    )

    features, df = service._features_from_klines(klines, "1m")

    assert len(df) == 21
    assert features["close"] != pytest.approx(999.0)
    assert features["volume"] == pytest.approx(10.0)


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
    assert features["indicator_snapshot_available"] is True
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
async def test_feature_vector_fetches_sentiment_concurrently_with_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    service._last_sentiment_update = None
    service._sentiment_refresh_task = None
    service.ws_client = type("Ws", (), {"latest_tickers": {}})()
    monkeypatch.setattr(data_service_module.settings, "sentiment_blocking_timeout_seconds", 1.0)
    monkeypatch.setattr(data_service_module, "FEATURE_SNAPSHOT_TIMEOUT_SECONDS", 1.0)

    sentiment_started = asyncio.Event()
    sentiment_done = asyncio.Event()
    ticker_started_before_sentiment_done = False

    async def slow_sentiment(_symbols):
        sentiment_started.set()
        await asyncio.sleep(0.05)
        service._sentiment_cache["BTC/USDT"] = {"news_sentiment_avg": 0.25}
        sentiment_done.set()

    async def ticker_source(_symbol):
        nonlocal ticker_started_before_sentiment_done
        await sentiment_started.wait()
        ticker_started_before_sentiment_done = not sentiment_done.is_set()
        return {"last_price": 100.0, "bid": 99.0, "ask": 101.0}

    async def indicator_source(_symbol):
        return {"close": 100.0}

    async def derivatives_source(_symbol):
        return {}

    service.refresh_sentiment = slow_sentiment  # type: ignore[method-assign]
    service._get_ticker_snapshot = ticker_source  # type: ignore[method-assign]
    service._get_indicator_snapshot = indicator_source  # type: ignore[method-assign]
    service._get_derivatives_snapshot = derivatives_source  # type: ignore[method-assign]

    fv = await asyncio.wait_for(service.get_feature_vector("BTC/USDT"), timeout=1.0)

    assert fv.current_price == pytest.approx(100.0)
    assert ticker_started_before_sentiment_done is True


@pytest.mark.asyncio
async def test_feature_vector_can_skip_initial_sentiment_wait_for_market_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    service._last_sentiment_update = None
    service._sentiment_refresh_task = None
    service.ws_client = type("Ws", (), {"latest_tickers": {}})()
    monkeypatch.setattr(data_service_module.settings, "sentiment_blocking_timeout_seconds", 1.0)
    monkeypatch.setattr(data_service_module, "FEATURE_SNAPSHOT_TIMEOUT_SECONDS", 1.0)

    async def slow_sentiment(_symbols):
        await asyncio.sleep(60)

    async def ticker_source(_symbol):
        return {"last_price": 100.0, "bid": 99.0, "ask": 101.0}

    async def indicator_source(_symbol):
        return {"close": 100.0}

    async def derivatives_source(_symbol):
        return {}

    service.refresh_sentiment = slow_sentiment  # type: ignore[method-assign]
    service._get_ticker_snapshot = ticker_source  # type: ignore[method-assign]
    service._get_indicator_snapshot = indicator_source  # type: ignore[method-assign]
    service._get_derivatives_snapshot = derivatives_source  # type: ignore[method-assign]

    try:
        fv = await asyncio.wait_for(
            service.get_feature_vector("BTC/USDT", wait_for_sentiment=False),
            timeout=0.2,
        )
    finally:
        task = service._sentiment_refresh_task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    assert fv.current_price == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_market_feature_vector_does_not_build_indicators_from_cached_klines() -> None:
    service = _service()
    service.ws_client = type(
        "Ws",
        (),
        {
            "latest_tickers": {
                "BTC/USDT": {
                    "last_price": 100.0,
                    "bid": 99.0,
                    "ask": 101.0,
                    "timestamp": int(time.time() * 1000),
                    "source": "websocket",
                }
            }
        },
    )()
    service._last_sentiment_update = None
    service._sentiment_refresh_task = None
    build_attempted = False

    async def noop_sentiment(_symbols):
        return None

    async def cached_kline_build(_symbol):
        nonlocal build_attempted
        build_attempted = True
        return {"indicator_snapshot_available": True, "close": 100.0}

    service.refresh_sentiment = noop_sentiment  # type: ignore[method-assign]
    service._indicator_features_from_cached_klines = cached_kline_build  # type: ignore[method-assign]

    fv = await service.get_feature_vector(
        "BTC/USDT",
        wait_for_sentiment=False,
        block_on_remote_indicators=False,
        allow_cached_indicator_build=False,
        allow_indicator_background_refresh=False,
    )

    assert fv.current_price == pytest.approx(100.0)
    assert fv.indicator_snapshot_available is False
    assert build_attempted is False
    assert service._indicator_snapshot_tasks.get("BTC/USDT") is None


@pytest.mark.asyncio
async def test_normal_feature_vector_can_build_indicators_from_cached_klines() -> None:
    service = _service()
    service.ws_client = type(
        "Ws",
        (),
        {
            "latest_tickers": {
                "BTC/USDT": {
                    "last_price": 100.0,
                    "bid": 99.0,
                    "ask": 101.0,
                    "timestamp": int(time.time() * 1000),
                    "source": "websocket",
                }
            }
        },
    )()
    service._last_sentiment_update = None
    service._sentiment_refresh_task = None

    async def noop_sentiment(_symbols):
        return None

    async def cached_kline_build(_symbol):
        return {
            "indicator_snapshot_available": True,
            "technical_indicator_timeframe": "1h",
            "close": 100.0,
            "volume_ratio": 1.2,
        }

    service.refresh_sentiment = noop_sentiment  # type: ignore[method-assign]
    service._indicator_features_from_cached_klines = cached_kline_build  # type: ignore[method-assign]

    fv = await service.get_feature_vector(
        "BTC/USDT",
        wait_for_sentiment=False,
        block_on_remote_indicators=False,
    )

    assert fv.current_price == pytest.approx(100.0)
    assert fv.indicator_snapshot_available is True
    assert fv.technical_indicator_timeframe == "1h"
    assert fv.volume_ratio == pytest.approx(1.2)


@pytest.mark.asyncio
async def test_available_symbols_uses_fresh_cache_without_rest_call() -> None:
    service = _service()
    service._available_symbols_cache = [{"symbol": "BTC/USDT"}]
    service._available_symbols_cache_updated_at = data_service_module.datetime.now(
        data_service_module.UTC
    )

    class FailingRestClient:
        async def get_available_symbols(self) -> list[dict[str, Any]]:
            raise AssertionError("fresh available-symbol cache should not call REST")

    service.rest_client = FailingRestClient()

    symbols = await service.get_available_symbols()

    assert symbols == [{"symbol": "BTC/USDT"}]


@pytest.mark.asyncio
async def test_available_symbols_returns_stale_cache_while_refreshing_background() -> None:
    service = _service()
    service._available_symbols_cache = [{"symbol": "BTC/USDT"}]
    service._available_symbols_cache_updated_at = data_service_module.datetime.now(
        data_service_module.UTC
    ) - timedelta(seconds=data_service_module.AVAILABLE_SYMBOLS_CACHE_TTL_SECONDS + 1.0)
    release_refresh = asyncio.Event()
    rest_calls: list[str] = []

    class SlowRestClient:
        async def get_available_symbols(self) -> list[dict[str, Any]]:
            rest_calls.append("called")
            await release_refresh.wait()
            return [{"symbol": "ETH/USDT"}]

    service.rest_client = SlowRestClient()

    started_at = time.monotonic()
    symbols = await asyncio.wait_for(service.get_available_symbols(), timeout=0.05)
    elapsed = time.monotonic() - started_at

    assert symbols == [{"symbol": "BTC/USDT"}]
    assert elapsed < 0.05
    await asyncio.sleep(0)
    assert rest_calls == ["called"]

    release_refresh.set()
    for _ in range(20):
        await asyncio.sleep(0.01)
        if service._available_symbols_cache == [{"symbol": "ETH/USDT"}]:
            break

    assert service._available_symbols_cache == [{"symbol": "ETH/USDT"}]


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
                "volume_ratio": 1.35,
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
            "volume_ratio": 0.04,
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
    assert features["volume_ratio_timeframe"] == "1h"
    assert features["indicator_snapshot_available"] is True
    assert features["volume_ratio"] == pytest.approx(0.04)
    assert features["entry_activity_volume_ratio"] == pytest.approx(1.35)
    assert features["entry_activity_volume_timeframe"] == "1m"
    assert features["returns_1"] == pytest.approx(0.011)
    assert features["returns_5"] == pytest.approx(0.055)
    assert features["returns_20"] == pytest.approx(0.12)
    assert features["volatility_20"] == pytest.approx(0.033)
    assert features["price_vs_sma20"] == pytest.approx(0.21)
    assert features["price_vs_sma50"] == pytest.approx(0.34)
    assert features["sequence_timeframe"] == "1m"
    assert features["sequence_length"] == data_service_module.KLINE_FEATURE_SEQUENCE_LIMIT
    assert len(features["close_sequence"]) == data_service_module.KLINE_FEATURE_SEQUENCE_LIMIT
    assert len(features["volume_sequence"]) == data_service_module.KLINE_FEATURE_SEQUENCE_LIMIT
    assert features["close_sequence"][-1] == pytest.approx(11.19)


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
async def test_indicator_snapshot_nonblocking_returns_stale_cache_and_refreshes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    service._indicator_snapshot_cache["BTC/USDT"] = {
        "updated_at": data_service_module.datetime.now(data_service_module.UTC)
        - timedelta(seconds=999),
        "data": {
            "close": 100.5,
            "indicator_snapshot_available": True,
        },
    }
    scheduled: list[str] = []
    monkeypatch.setattr(
        service,
        "_schedule_indicator_snapshot_refresh",
        lambda symbol: scheduled.append(symbol),
    )

    features = await service._get_indicator_snapshot("BTC/USDT", block_on_remote=False)

    assert features["close"] == pytest.approx(100.5)
    assert features["indicator_snapshot_stale"] is True
    assert features["indicator_snapshot_refresh_in_background"] is True
    assert scheduled == ["BTC/USDT"]


@pytest.mark.asyncio
async def test_indicator_snapshot_nonblocking_does_not_wait_for_remote_without_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    remote_started = asyncio.Event()

    class FakeRestClient:
        async def fetch_ohlcv(
            self,
            symbol: str,
            timeframe: str = "1h",
            limit: int = 100,
        ) -> list[list[float]]:
            remote_started.set()
            await asyncio.sleep(60)
            return []

    async def no_cached_klines(
        symbol: str,
        timeframe: str,
        limit: int,
    ) -> list[list[float]]:
        return []

    service.rest_client = FakeRestClient()
    service._load_recent_cached_klines = no_cached_klines  # type: ignore[method-assign]
    monkeypatch.setattr(data_service_module, "compute_all_indicators", lambda df: df)
    monkeypatch.setattr(
        data_service_module,
        "extract_latest_features",
        lambda df: {"close": float(df["close"].iloc[-1])},
    )

    features = await asyncio.wait_for(
        service._get_indicator_snapshot("BTC/USDT", block_on_remote=False),
        timeout=0.2,
    )

    assert features["indicator_snapshot_available"] is False
    assert features["indicator_snapshot_refresh_in_background"] is True
    task = service._indicator_snapshot_tasks.get("BTC/USDT")
    assert task is not None
    await asyncio.wait_for(remote_started.wait(), timeout=0.2)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


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


@pytest.mark.asyncio
async def test_kline_coverage_refresh_rotates_symbols_and_timeframes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    service.ws_client = type(
        "Ws", (), {"_subscribe_symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"]}
    )()
    calls: list[tuple[str, str, int]] = []

    async def fake_fetch(symbol: str, timeframe: str, limit: int) -> tuple[str, list[Any]]:
        calls.append((symbol, timeframe, limit))
        return timeframe, []

    monkeypatch.setattr(data_service_module, "KLINE_COVERAGE_REFRESH_BATCH_SIZE", 2)
    monkeypatch.setattr(data_service_module, "KLINE_COVERAGE_REFRESH_SYMBOL_CAP", 3)
    service._fetch_and_persist_klines = fake_fetch  # type: ignore[method-assign]

    first = await service.refresh_kline_coverage_once()
    second = await service.refresh_kline_coverage_once()

    assert first["refreshed_symbols"] == ["BTC/USDT", "ETH/USDT"]
    assert second["refreshed_symbols"] == ["SOL/USDT", "BTC/USDT"]
    assert {(symbol, timeframe) for symbol, timeframe, _limit in calls} >= {
        ("BTC/USDT", "1m"),
        ("BTC/USDT", "5m"),
        ("ETH/USDT", "15m"),
        ("SOL/USDT", "1h"),
    }


def test_kline_coverage_targets_all_subscribed_symbols_even_above_config_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    service._kline_coverage_symbols = ["BTC/USDT", "ETH/USDT"]
    service.ws_client = type(
        "Ws",
        (),
        {
            "_subscribe_symbols": [
                "BTC/USDT",
                "ETH/USDT",
                "SOL/USDT",
                "ADA/USDT",
                "FIL/USDT",
            ]
        },
    )()

    monkeypatch.setattr(data_service_module, "KLINE_COVERAGE_REFRESH_SYMBOL_CAP", 2)

    assert service._kline_coverage_target_symbols()[:5] == [
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
        "ADA/USDT",
        "FIL/USDT",
    ]


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
            "volume_ratio_timeframe": "1h",
            "entry_activity_volume_ratio": 1.4,
            "entry_activity_volume_timeframe": "1m",
            "close_sequence": [100.0 + index for index in range(60)],
            "volume_sequence": [10.0 + index for index in range(60)],
            "sequence_timeframe": "1m",
        },
    )

    assert vector.short_returns_timeframe == "1m"
    assert vector.technical_indicator_timeframe == "1h"
    assert vector.volume_ratio_timeframe == "1h"
    assert vector.indicator_snapshot_available is True
    assert vector.entry_activity_volume_ratio == pytest.approx(1.4)
    assert vector.sequence_length == 60
    assert vector.close_sequence[-1] == pytest.approx(159.0)
    assert vector.volume_sequence[-1] == pytest.approx(69.0)
    assert vector.to_dict()["close_sequence"][-1] == pytest.approx(159.0)
    assert "short_returns=1m" in vector.to_llm_context()
    assert "trend=1h" in vector.to_llm_context()
    assert "activity_volume=1m" in vector.to_llm_context()


def test_feature_vector_keeps_fresh_ticker_when_indicator_close_diverges() -> None:
    from data_feed.feature_vector import build_feature_vector

    vector = build_feature_vector(
        "PROS/USDT",
        ticker={
            "last_price": 0.5666,
            "bid": 0.5665,
            "ask": 0.5667,
            "high_24h": 0.569,
            "low_24h": 0.5491,
            "source": "rest",
        },
        indicators={"close": 0.3902, "returns_1": 0.01},
    )

    assert vector.current_price == pytest.approx(0.5666)
    assert vector.close == pytest.approx(0.5666)
    assert vector.indicator_close_price == pytest.approx(0.3902)
    assert vector.indicator_price_gap_pct > 20
    assert vector.price_reconciliation_warning == (
        "ticker_current_price_kept_indicator_close_diverged"
    )


def test_feature_vector_drops_sequence_when_indicator_price_diverges() -> None:
    from data_feed.feature_vector import build_feature_vector

    vector = build_feature_vector(
        "PROS/USDT",
        ticker={"last_price": 0.5666, "bid": 0.5665, "ask": 0.5667},
        indicators={
            "close": 0.3902,
            "close_sequence": [0.39 + index * 0.0001 for index in range(60)],
            "volume_sequence": [100.0 for _ in range(60)],
            "sequence_timeframe": "1m",
        },
    )

    assert vector.close_sequence == []
    assert vector.volume_sequence == []
    assert vector.sequence_length == 0
    assert vector.sequence_quality_warning == "indicator_sequence_dropped_due_to_ticker_gap"


def test_feature_vector_keeps_okx_swap_volume_units_separate() -> None:
    from data_feed.feature_vector import build_feature_vector

    vector = build_feature_vector(
        "PEPE/USDT",
        ticker={
            "last_price": 0.000002355,
            "volume_24h": 5_357_584.8,
            "volume_24h_contracts": 5_357_584.8,
            "volume_24h_base": 53_575_848_000_000,
            "notional_24h_usdt": 126_171_122.04,
            "volume_24h_source": "quote",
        },
    )

    assert vector.volume_24h_contracts == pytest.approx(5_357_584.8)
    assert vector.volume_24h == pytest.approx(53_575_848_000_000)
    assert vector.volume_24h_base == pytest.approx(53_575_848_000_000)
    assert vector.notional_24h_usdt == pytest.approx(126_171_122.04)
    assert vector.volume_24h_source == "quote"


@pytest.mark.asyncio
async def test_ticker_snapshot_refreshes_stale_ws_cache_from_swap_rest() -> None:
    service = _service()
    stale_timestamp_ms = int(
        (time.time() - data_service_module.TICKER_CACHE_MAX_AGE_SECONDS - 60) * 1000
    )

    class FakeWsClient:
        latest_tickers = {
            "PROS/USDT": {
                "symbol": "PROS/USDT",
                "last_price": 0.3902,
                "bid": 0.3901,
                "ask": 0.3903,
                "timestamp": stale_timestamp_ms,
            }
        }

    class FakeRestClient:
        def __init__(self) -> None:
            self.symbols: list[str] = []

        async def fetch_ticker(self, symbol: str) -> dict[str, Any]:
            self.symbols.append(symbol)
            return {
                "last": 0.5666,
                "bid": 0.5665,
                "ask": 0.5667,
                "high": 0.569,
                "low": 0.5491,
                "baseVolume": 1234,
                "percentage": 1.2,
                "timestamp": int(time.time() * 1000),
                "info": {"instId": "PROS-USDT-SWAP"},
            }

    service.ws_client = FakeWsClient()
    rest_client = FakeRestClient()
    service.rest_client = rest_client

    snapshot = await service._get_ticker_snapshot("PROS/USDT")

    assert rest_client.symbols == ["PROS/USDT"]
    assert snapshot["last_price"] == pytest.approx(0.5666)
    assert snapshot["source"] == "rest"
    assert snapshot["inst_type"] == "SWAP"
    assert service.ws_client.latest_tickers["PROS/USDT"]["last_price"] == pytest.approx(0.5666)


@pytest.mark.asyncio
async def test_ticker_snapshot_can_defer_rest_for_market_batch_scan() -> None:
    service = _service()
    stale_timestamp_ms = int(
        (time.time() - data_service_module.TICKER_CACHE_MAX_AGE_SECONDS - 60) * 1000
    )

    class FakeWsClient:
        latest_tickers = {
            "PROS/USDT": {
                "symbol": "PROS/USDT",
                "last_price": 0.3902,
                "bid": 0.3901,
                "ask": 0.3903,
                "timestamp": stale_timestamp_ms,
            }
        }

    class FakeRestClient:
        def __init__(self) -> None:
            self.symbols: list[str] = []

        async def fetch_ticker(self, symbol: str) -> dict[str, Any]:
            self.symbols.append(symbol)
            return {
                "last": 0.5666,
                "bid": 0.5665,
                "ask": 0.5667,
                "timestamp": int(time.time() * 1000),
            }

    service.ws_client = FakeWsClient()
    rest_client = FakeRestClient()
    service.rest_client = rest_client

    snapshot = await service._get_ticker_snapshot("PROS/USDT", block_on_remote=False)

    assert rest_client.symbols == []
    assert snapshot["last_price"] == pytest.approx(0.3902)
    assert snapshot["stale"] is True
    assert snapshot["ticker_remote_refresh_deferred"] is True


@pytest.mark.asyncio
async def test_feature_vector_can_defer_ticker_rest_for_market_batch_scan() -> None:
    service = _service()
    stale_timestamp_ms = int(
        (time.time() - data_service_module.TICKER_CACHE_MAX_AGE_SECONDS - 60) * 1000
    )

    class FakeWsClient:
        latest_tickers = {
            "PROS/USDT": {
                "symbol": "PROS/USDT",
                "last_price": 0.3902,
                "bid": 0.3901,
                "ask": 0.3903,
                "timestamp": stale_timestamp_ms,
            }
        }

    class FakeRestClient:
        def __init__(self) -> None:
            self.symbols: list[str] = []

        async def fetch_ticker(self, symbol: str) -> dict[str, Any]:
            self.symbols.append(symbol)
            return {
                "last": 0.5666,
                "bid": 0.5665,
                "ask": 0.5667,
                "timestamp": int(time.time() * 1000),
            }

    async def noop_sentiment(_symbols):
        return None

    service.ws_client = FakeWsClient()
    rest_client = FakeRestClient()
    service.rest_client = rest_client
    service._last_sentiment_update = None
    service._sentiment_refresh_task = None
    service.refresh_sentiment = noop_sentiment  # type: ignore[method-assign]

    vector = await service.get_feature_vector(
        "PROS/USDT",
        wait_for_sentiment=False,
        block_on_remote_ticker=False,
        block_on_remote_indicators=False,
        allow_cached_indicator_build=False,
        allow_indicator_background_refresh=False,
    )

    assert rest_client.symbols == []
    assert vector.current_price == pytest.approx(0.3902)
    assert vector.price_source == "stale_websocket"


@pytest.mark.asyncio
async def test_ticker_snapshot_refreshes_fresh_but_inconsistent_ws_cache() -> None:
    service = _service()
    fresh_timestamp_ms = int(time.time() * 1000)

    class FakeWsClient:
        latest_tickers = {
            "PROS/USDT": {
                "symbol": "PROS/USDT",
                "last_price": 0.3902,
                "bid": 0.3901,
                "ask": 0.3903,
                "high_24h": 0.569,
                "low_24h": 0.5491,
                "timestamp": fresh_timestamp_ms,
                "source": "websocket",
            }
        }

    class FakeRestClient:
        def __init__(self) -> None:
            self.symbols: list[str] = []

        async def fetch_ticker(self, symbol: str) -> dict[str, Any]:
            self.symbols.append(symbol)
            return {
                "last": 0.5531,
                "bid": 0.5530,
                "ask": 0.5533,
                "high": 0.578,
                "low": 0.5491,
                "baseVolume": 1234,
                "percentage": 1.2,
                "timestamp": int(time.time() * 1000),
                "info": {"instId": "PROS-USDT-SWAP"},
            }

    service.ws_client = FakeWsClient()
    rest_client = FakeRestClient()
    service.rest_client = rest_client

    snapshot = await service._get_ticker_snapshot("PROS/USDT")

    assert rest_client.symbols == ["PROS/USDT"]
    assert snapshot["last_price"] == pytest.approx(0.5531)
    assert snapshot["source"] == "rest"
    assert service.ws_client.latest_tickers["PROS/USDT"]["last_price"] == pytest.approx(0.5531)


def test_last_trade_outside_current_book_is_not_cache_corruption() -> None:
    service = _service()

    issue = service._ticker_snapshot_consistency_issue(
        {
            "last_price": 0.5530,
            "bid": 0.5531,
            "ask": 0.5532,
            "high_24h": 0.56,
            "low_24h": 0.54,
        }
    )

    assert issue is None


@pytest.mark.asyncio
async def test_market_batch_does_not_schedule_derivatives_refresh() -> None:
    service = _service()
    scheduled: list[str] = []
    service._schedule_derivatives_background_refresh = scheduled.append  # type: ignore[method-assign]

    result = await service._get_derivatives_snapshot(
        "PROS/USDT",
        block_on_remote=False,
        allow_background_refresh=False,
    )

    assert result == {}
    assert scheduled == []


@pytest.mark.asyncio
async def test_blocking_derivatives_read_refreshes_stale_cache() -> None:
    service = _service()
    service._derivatives_update_interval = 20.0
    service._derivatives_cache["PROS/USDT"] = {
        "updated_at": datetime.now(UTC) - timedelta(seconds=30),
        "data": {"funding_rate": 0.001},
    }
    refresh_calls: list[str] = []

    async def refresh(symbol: str) -> dict[str, Any]:
        refresh_calls.append(symbol)
        return {"funding_rate": 0.002, "source": "fresh_rest"}

    service._refresh_derivatives_snapshot = refresh  # type: ignore[method-assign]

    result = await service._get_derivatives_snapshot(
        "PROS/USDT",
        block_on_remote=True,
    )

    assert refresh_calls == ["PROS/USDT"]
    assert result == {"funding_rate": 0.002, "source": "fresh_rest"}
    assert "derivatives_snapshot_stale" not in result


@pytest.mark.asyncio
async def test_feature_market_fact_proves_rest_ws_book_reference_and_native_path() -> None:
    service = _service()
    timestamp = int(time.time() * 1000)
    minute_open = timestamp - timestamp % 60_000
    spec = {
        "instId": "PROS-USDT-SWAP",
        "instType": "SWAP",
        "uly": "PROS-USDT",
        "instFamily": "PROS-USDT",
        "ctType": "linear",
        "ctVal": "1",
        "ctMult": "1",
        "ctValCcy": "PROS",
        "settleCcy": "USDT",
        "lotSz": "1",
        "minSz": "1",
        "tickSz": "0.0001",
        "state": "live",
    }

    class FakeWsClient:
        latest_tickers = {
            "PROS/USDT": {
                "symbol": "PROS/USDT",
                "inst_id": "PROS-USDT-SWAP",
                "inst_type": "SWAP",
                "last_price": 0.5531,
                "bid": 0.5530,
                "ask": 0.5532,
                "high_24h": 0.56,
                "low_24h": 0.54,
                "notional_24h_usdt": 1_000_000.0,
                "volume_24h_contracts": 2_000_000.0,
                "volume_24h_base": 2_000_000.0,
                "timestamp": timestamp,
                "source": "websocket",
                "source_endpoint": "okx_ws_public",
                "source_channel": "tickers",
            }
        }

    class FakeRestClient:
        async def fetch_ticker(self, _symbol: str) -> dict[str, Any]:
            return {
                "last": 0.5531,
                "bid": 0.5530,
                "ask": 0.5532,
                "high": 0.56,
                "low": 0.54,
                "baseVolume": 2_000_000.0,
                "quoteVolume": 1_106_200.0,
                "percentage": 1.0,
                "timestamp": timestamp,
                "info": {
                    "instId": "PROS-USDT-SWAP",
                    "ts": str(timestamp),
                    "vol24h": "2000000",
                    "volCcy24h": "2000000",
                },
            }

        async def fetch_instrument_spec(self, _symbol: str) -> dict[str, Any]:
            return spec

        async def fetch_ohlcv(self, _symbol: str, **_kwargs) -> list[list[float]]:
            return [[minute_open, 0.5528, 0.5534, 0.5527, 0.5531, 10_000.0]]

    service.ws_client = FakeWsClient()
    service.rest_client = FakeRestClient()
    ticker = await service._get_ticker_snapshot("PROS/USDT")
    derivatives = {
        "orderbook_bid_depth": 50_000.0,
        "orderbook_ask_depth": 49_000.0,
        "mark_price": 0.5531,
        "index_price": 0.5530,
        "orderbook_fact": {
            "inst_id": "PROS-USDT-SWAP",
            "inst_type": "SWAP",
            "source_timestamp_ms": timestamp,
            "bid": 0.5530,
            "ask": 0.5532,
            "bid_depth_usdt": 50_000.0,
            "ask_depth_usdt": 49_000.0,
        },
        "mark_price_fact": {
            "inst_id": "PROS-USDT-SWAP",
            "inst_type": "SWAP",
            "source_timestamp_ms": timestamp,
            "price": 0.5531,
        },
        "index_price_fact": {
            "inst_id": "PROS-USDT",
            "inst_type": "INDEX",
            "source_timestamp_ms": timestamp,
            "price": 0.5530,
        },
    }

    enriched = service._attach_market_source_consistency(
        "PROS/USDT", ticker, derivatives
    )
    vector = build_feature_vector(
        "PROS/USDT",
        ticker=enriched,
        derivatives=derivatives,
    )

    assert len(ticker["market_source_snapshots"]) == 2
    assert vector.market_fact["source_consistency"]["status"] == "clean"
    assert vector.market_fact["quality"]["status"] == "clean"


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
