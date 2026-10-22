from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from core.feature_availability import (
    COMPONENT_AVAILABILITY_VERSION,
    ObservedKlines,
    availability_fingerprint,
    sentiment_input_availability,
)
from data_feed import okx_rest_client
from data_feed.okx_rest_client import OKXRestClient
from services import data_service as data_service_module
from services.data_service import DataService

RECEIVED_AT = datetime(2026, 10, 22, 12, 0, 2, tzinfo=UTC)


def test_news_availability_accepts_standard_rss_publication_date() -> None:
    evidence = sentiment_input_availability(
        [
            {
                "published_at": "Wed, 21 Oct 2026 12:00:00 GMT",
                "received_at": RECEIVED_AT.isoformat(),
            }
        ],
        [],
        available_at=RECEIVED_AT + timedelta(seconds=1),
    )

    assert evidence["news_clock_errors"] == 0
    assert evidence["news_received_at"] == RECEIVED_AT.isoformat()


def _service() -> DataService:
    service = object.__new__(DataService)
    service._sentiment_cache = {}
    service._headlines_cache = {}
    service._news_items_cache = {}
    service._indicator_snapshot_cache = {}
    service._derivatives_cache = {}
    service._derivatives_refresh_tasks = {}
    service._derivatives_failed_at = {}
    service.ws_client = SimpleNamespace(latest_tickers={})
    return service


@pytest.fixture
def receipt_clock(monkeypatch: pytest.MonkeyPatch):
    class Clock(datetime):
        current = RECEIVED_AT

        @classmethod
        def now(cls, tz=None):
            return cls.current.astimezone(tz) if tz else cls.current.replace(tzinfo=None)

    monkeypatch.setattr(data_service_module, "datetime", Clock)
    monkeypatch.setattr(okx_rest_client, "datetime", Clock)
    return Clock


def _rows(timeframe: str, *, count: int = 40) -> list[list[float]]:
    seconds = data_service_module.TIMEFRAME_SECONDS[timeframe]
    end_ms = int(RECEIVED_AT.timestamp()) // seconds * seconds * 1000
    return [
        [
            end_ms - (count - index) * seconds * 1000,
            100.0 + index,
            102.0 + index,
            99.0 + index,
            101.0 + index,
            10.0 + index,
        ]
        for index in range(count)
    ]


@pytest.fixture
def simple_indicators(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data_service_module, "compute_all_indicators", lambda frame: frame)
    monkeypatch.setattr(
        data_service_module,
        "extract_latest_features",
        lambda frame: {
            "close": float(frame["close"].iloc[-1]),
            "volume": float(frame["volume"].iloc[-1]),
            "returns_1": float(frame["close"].iloc[-1] / frame["close"].iloc[-2] - 1),
        },
    )


@pytest.mark.asyncio
async def test_rest_klines_receipt_measures_completion_not_request_start(
    monkeypatch: pytest.MonkeyPatch, receipt_clock,
) -> None:
    client = OKXRestClient()
    completed_at = RECEIVED_AT + timedelta(seconds=3)
    response = _rows("1m")

    async def fetch(method, *_args, **_kwargs):
        assert method == "fetch_ohlcv"
        receipt_clock.current = completed_at
        return response

    monkeypatch.setattr(client, "_ccxt_call", fetch)
    rows = await client.fetch_ohlcv("BTC/USDT", timeframe="1m")

    assert isinstance(rows, ObservedKlines)
    assert rows == response
    assert rows.received_at == completed_at.isoformat()


@pytest.mark.asyncio
async def test_observed_kline_cache_preserves_receipt_and_detaches_rows(
    monkeypatch: pytest.MonkeyPatch, receipt_clock,
) -> None:
    service = _service()
    response = ObservedKlines(_rows("1m"), received_at=RECEIVED_AT.isoformat())

    async def fetch(*_args, **_kwargs):
        return response

    async def persist(*_args):
        return None

    def no_database():
        raise AssertionError("same-process observed rows must not need a database read")

    service.rest_client = SimpleNamespace(fetch_ohlcv=fetch)
    monkeypatch.setattr(service, "_persist_klines", persist)
    monkeypatch.setattr(data_service_module, "get_session_ctx", no_database)
    await service._fetch_remote_klines_with_cache(
        "BTC/USDT", "1m", 40, persist_before_return=True,
    )
    response[0][4] = 9999.0
    receipt_clock.current += timedelta(seconds=10)
    cached = await service._load_recent_cached_klines("BTC/USDT", "1m", 40)
    assert isinstance(cached, ObservedKlines)
    assert cached.received_at == RECEIVED_AT.isoformat()
    assert cached[0][4] == 101.0
    cached[0][4] = -999.0
    reloaded = await service._load_recent_cached_klines("BTC/USDT", "1m", 40)
    assert reloaded[0][4] == 101.0
    assert reloaded.received_at == RECEIVED_AT.isoformat()


@pytest.mark.asyncio
async def test_plain_legacy_response_does_not_acquire_an_observed_receipt(
    monkeypatch: pytest.MonkeyPatch, receipt_clock, simple_indicators,
) -> None:
    service = _service()

    async def fetch(*_args, **_kwargs):
        return _rows("1m")

    async def persist(*_args):
        return None

    service.rest_client = SimpleNamespace(fetch_ohlcv=fetch)
    monkeypatch.setattr(service, "_persist_klines", persist)
    rows = await service._fetch_remote_klines_with_cache(
        "BTC/USDT", "1m", 40, persist_before_return=True,
    )
    assert not isinstance(rows, ObservedKlines)
    assert not getattr(service, "_observed_klines_cache", {})
    features = service._indicator_features_from_timeframes({"1m": rows})
    evidence = features["technical_availability"]
    assert evidence["source_delivery_observed"] is False
    assert evidence["1m_received_at"] is None


@pytest.mark.asyncio
async def test_database_klines_remain_unobserved(
    monkeypatch: pytest.MonkeyPatch, receipt_clock, simple_indicators,
) -> None:
    service = _service()

    class SessionContext:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *_args):
            return None

    class Repository:
        def __init__(self, _session):
            pass

        async def get_klines_for_timeframes(self, _symbol, _limits):
            return {
                "1m": [
                    SimpleNamespace(
                        open_time=datetime.fromtimestamp(row[0] / 1000.0, UTC),
                        open=row[1], high=row[2], low=row[3], close=row[4], volume=row[5],
                    )
                    for row in _rows("1m")
                ]
            }

    monkeypatch.setattr(data_service_module, "get_session_ctx", SessionContext)
    monkeypatch.setattr(data_service_module, "MarketRepository", Repository)
    grouped = await service._load_recent_cached_klines_for_timeframes(
        "BTC/USDT", {"1m": 40},
    )
    assert not isinstance(grouped["1m"], ObservedKlines)
    evidence = service._indicator_features_from_timeframes(grouped)["technical_availability"]
    assert evidence["source_delivery_observed"] is False
    assert evidence["1m_received_at"] is None
    assert evidence["1m_closed_at"] == RECEIVED_AT.replace(second=0).isoformat()


@pytest.mark.asyncio
async def test_cached_indicator_builder_retains_original_delivery_evidence(
    monkeypatch: pytest.MonkeyPatch, receipt_clock, simple_indicators,
) -> None:
    service = _service()
    monkeypatch.setattr(
        data_service_module, "KLINE_PERSIST_TIMEFRAME_LIMITS", {"1m": 40, "1h": 40},
    )

    def no_database():
        raise AssertionError("observed cache must feed the indicator builder directly")

    monkeypatch.setattr(data_service_module, "get_session_ctx", no_database)
    for timeframe in ("1m", "1h"):
        service._remember_observed_klines(
            "BTC/USDT",
            timeframe,
            ObservedKlines(_rows(timeframe), received_at=RECEIVED_AT.isoformat()),
        )
    receipt_clock.current += timedelta(seconds=10)
    features = await service._build_indicator_features_from_cached_klines("BTC/USDT")
    evidence = features["technical_availability"]

    assert evidence["source_delivery_observed"] is True
    assert evidence["available_at"] == receipt_clock.current.isoformat()
    assert evidence["1m_received_at"] == RECEIVED_AT.isoformat()
    assert evidence["1h_received_at"] == RECEIVED_AT.isoformat()
    assert evidence["1m_input_fingerprint"] == availability_fingerprint(_rows("1m"))
    assert evidence["1h_input_fingerprint"] == availability_fingerprint(_rows("1h"))
    assert service._indicator_snapshot_cache["BTC/USDT"]["data"][
        "technical_availability"
    ] == evidence


@pytest.mark.asyncio
async def test_feature_vector_retains_detached_component_availability(
    monkeypatch: pytest.MonkeyPatch, receipt_clock, simple_indicators,
) -> None:
    service = _service()
    monkeypatch.setattr(data_service_module, "KLINE_PERSIST_TIMEFRAME_LIMITS", {"1m": 40})
    service._remember_observed_klines(
        "BTC/USDT", "1m",
        ObservedKlines(_rows("1m"), received_at=RECEIVED_AT.isoformat()),
    )
    service._build_sentiment_cache(["BTC/USDT"], [_article(0)], [])

    async def sentiment(*_args, **_kwargs):
        return None

    async def ticker(*_args, **_kwargs):
        return {"last_price": 140.0, "bid": 139.9, "ask": 140.1}

    async def indicators(symbol, **_kwargs):
        return await service._build_indicator_features_from_cached_klines(symbol)

    async def derivatives(*_args, **_kwargs):
        return {
            "open_interest_contracts": 1000.0,
            "open_interest_received_at": RECEIVED_AT.isoformat(),
            "open_interest_observed_at": RECEIVED_AT.isoformat(),
        }

    monkeypatch.setattr(service, "_ensure_sentiment_for_analysis", sentiment)
    monkeypatch.setattr(service, "_get_feature_ticker_snapshot", ticker)
    monkeypatch.setattr(service, "_get_feature_indicator_snapshot", indicators)
    monkeypatch.setattr(service, "_get_feature_derivatives_snapshot", derivatives)
    monkeypatch.setattr(
        service, "_attach_market_source_consistency",
        lambda _symbol, snapshot, _derivatives: snapshot,
    )
    vector = await service.get_feature_vector(
        "BTC/USDT", block_on_remote_ticker=False, block_on_remote_derivatives=False,
    )
    assert vector.technical_availability["source_delivery_observed"] is True
    assert vector.technical_availability["1m_received_at"] == RECEIVED_AT.isoformat()
    assert vector.sentiment_availability["news_input_count"] == 1
    assert vector.sentiment_availability["news_received_at"] == RECEIVED_AT.isoformat()
    assert vector.open_interest_contracts == 1000.0
    assert vector.open_interest_received_at == RECEIVED_AT.isoformat()
    assert vector.open_interest_observed_at == RECEIVED_AT.isoformat()
    vector.technical_availability["available_at"] = "mutated"
    vector.sentiment_availability["available_at"] = "mutated"
    vector.recent_news_items[0]["received_at"] = "mutated"
    assert service._indicator_snapshot_cache["BTC/USDT"]["data"][
        "technical_availability"
    ]["available_at"] == RECEIVED_AT.isoformat()
    assert service._sentiment_cache["BTC/USDT"]["sentiment_availability"][
        "available_at"
    ] == RECEIVED_AT.isoformat()
    assert service._news_items_cache["BTC/USDT"][0]["received_at"] == RECEIVED_AT.isoformat()


@pytest.mark.parametrize(
    "receipt",
    [
        None,
        "2026-10-22T12:00:02",
        (RECEIVED_AT + timedelta(minutes=1)).isoformat(),
    ],
)
def test_technical_missing_ambiguous_or_future_receipt_remains_unobserved(
    receipt_clock, simple_indicators, receipt,
) -> None:
    service = _service()
    features = service._indicator_features_from_timeframes({
        "1m": ObservedKlines(_rows("1m"), received_at=receipt),
    })
    evidence = features["technical_availability"]
    assert evidence["source_delivery_observed"] is False
    if receipt is None or receipt == "2026-10-22T12:00:02":
        assert evidence["1m_received_at"] is None
    else:
        assert evidence["1m_received_at"] == receipt


def test_technical_evidence_binds_all_used_closed_bar_inputs(
    receipt_clock, simple_indicators,
) -> None:
    service = _service()
    raw = {
        timeframe: ObservedKlines(_rows(timeframe), received_at=RECEIVED_AT.isoformat())
        for timeframe in ("1m", "1h")
    }
    raw["15m"] = _rows("15m")
    features = service._indicator_features_from_timeframes(raw)
    evidence = features["technical_availability"]

    assert evidence["version"] == COMPONENT_AVAILABILITY_VERSION
    assert evidence["timeframes"] == "1h,1m"
    assert evidence["source_delivery_observed"] is True
    for timeframe in ("1h", "1m"):
        assert evidence[f"{timeframe}_received_at"] == RECEIVED_AT.isoformat()
        assert evidence[f"{timeframe}_closed_at"] == RECEIVED_AT.replace(second=0).isoformat()
        assert evidence[f"{timeframe}_row_count"] == 40
        assert evidence[f"{timeframe}_input_fingerprint"] == availability_fingerprint(
            _rows(timeframe)
        )
    original = deepcopy(evidence)
    raw["1h"][0][2] += 1.0
    changed = service._indicator_features_from_timeframes(raw)["technical_availability"]
    assert changed["input_fingerprint"] != original["input_fingerprint"]
    assert changed["1h_input_fingerprint"] != original["1h_input_fingerprint"]
    assert changed["1m_input_fingerprint"] == original["1m_input_fingerprint"]
    assert evidence == original


def test_open_candle_never_matures_only_because_cached_rows_are_read_later(
    receipt_clock, simple_indicators,
) -> None:
    service = _service()
    rows = ObservedKlines(_rows("1m"), received_at=RECEIVED_AT.isoformat())
    rows.append([
        int(RECEIVED_AT.replace(second=0).timestamp() * 1000),
        999.0, 1000.0, 998.0, 999.5, 99999.0,
    ])
    receipt_clock.current += timedelta(hours=2)
    features = service._indicator_features_from_timeframes({"1m": rows})

    assert features["close"] == 140.0
    assert features["technical_availability"]["1m_row_count"] == 40
    assert features["technical_availability"]["1m_closed_at"] == (
        RECEIVED_AT.replace(second=0).isoformat()
    )
    assert features["technical_availability"]["1m_input_fingerprint"] == (
        availability_fingerprint(_rows("1m"))
    )


def test_observed_kline_cache_has_explicit_row_and_key_budgets(
    monkeypatch: pytest.MonkeyPatch, receipt_clock,
) -> None:
    service = _service()
    monkeypatch.setattr(data_service_module, "OBSERVED_KLINE_CACHE_MAX_KEYS", 2)
    monkeypatch.setattr(data_service_module, "OBSERVED_KLINE_CACHE_MAX_ROWS", 3)
    for symbol in ("BTC/USDT", "ETH/USDT", "SOL/USDT"):
        service._remember_observed_klines(
            symbol, "1m", ObservedKlines(_rows("1m"), received_at=RECEIVED_AT.isoformat()),
        )
    assert set(service._observed_klines_cache) == {
        ("ETH/USDT", "1m"), ("SOL/USDT", "1m"),
    }
    assert all(len(rows) == 3 for rows in service._observed_klines_cache.values())


def _article(index: int, *, direct: bool = True) -> dict:
    return {
        "title": f"{'BTC' if direct else 'Market'} story {index}",
        "summary": f"market summary {index}",
        "source": "unit-news",
        "symbols_mentioned": ["BTC"] if direct else [],
        "published_at": (RECEIVED_AT - timedelta(minutes=1)).isoformat(),
        "received_at": RECEIVED_AT.isoformat(),
        "sentiment_score": 0.25,
    }


def test_sentiment_evidence_covers_every_scored_item_beyond_display_limit(
    receipt_clock,
) -> None:
    service = _service()
    articles = [_article(index) for index in range(25)]
    posts = [
        {
            "symbols": ["BTC"], "title": "BTC social", "content": "BTC growth",
            "posted_at": (RECEIVED_AT - timedelta(seconds=1)).isoformat(),
            "received_at": RECEIVED_AT.isoformat(), "sentiment_score": 0.5,
        }
    ]
    articles[-1].pop("received_at")
    service._build_sentiment_cache(["BTC/USDT"], articles, posts)
    cached = service._sentiment_cache["BTC/USDT"]
    evidence = deepcopy(cached["sentiment_availability"])

    assert len(cached["news_items"]) == 20
    assert evidence["news_input_count"] == 25
    assert evidence["social_input_count"] == 1
    assert evidence["news_clock_errors"] == 1
    assert evidence["social_clock_errors"] == 0
    assert evidence["news_input_fingerprint"] == availability_fingerprint(articles)
    articles[0]["title"] = "mutated"
    posts[0]["content"] = "mutated"
    assert cached["news_items"][0]["title"] != "mutated"
    assert cached["sentiment_availability"] == evidence
    assert service._news_items_cache["BTC/USDT"][0]["received_at"] == RECEIVED_AT.isoformat()


def test_sentiment_fallback_evidence_includes_background_and_titleless_direct_scores(
    receipt_clock,
) -> None:
    service = _service()
    background = [_article(index, direct=False) for index in range(6)]
    direct_score = {**_article(10), "title": "", "summary": "BTC direct scored fact"}
    articles = [*background, direct_score]
    service._build_sentiment_cache(["BTC/USDT"], articles, [])
    cached = service._sentiment_cache["BTC/USDT"]
    evidence = cached["sentiment_availability"]

    assert evidence["news_input_count"] == 6
    assert evidence["news_input_fingerprint"] == availability_fingerprint(
        [direct_score, *background[:5]]
    )
    assert cached["news_sentiment"] > 0.0
    assert cached["market_news_item_count"] == 5
    assert all(title.startswith("[全市场] ") for title in service._headlines_cache["BTC/USDT"])


@pytest.mark.parametrize("receipt", [None, "2026-10-22T12:00:00"])
def test_sentiment_missing_or_ambiguous_receipt_is_never_filled(
    receipt_clock, receipt,
) -> None:
    service = _service()
    article = {**_article(0), "received_at": receipt}
    service._build_sentiment_cache(["BTC/USDT"], [article], [])
    evidence = service._sentiment_cache["BTC/USDT"]["sentiment_availability"]
    assert evidence["news_clock_errors"] == 1
    assert evidence["news_received_at"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("receipt", [None, RECEIVED_AT.isoformat()])
async def test_ticker_adapter_preserves_source_receipt_without_retiming(
    monkeypatch: pytest.MonkeyPatch, receipt_clock, receipt,
) -> None:
    service = _service()

    async def fetch(*_args, **_kwargs):
        receipt_clock.current += timedelta(minutes=2)
        return {
            "last": 100.0, "bid": 99.9, "ask": 100.1,
            "timestamp": int(RECEIVED_AT.timestamp() * 1000),
            "received_at": receipt,
            "info": {"instId": "BTC-USDT-SWAP"},
        }

    async def attach(_symbol, snapshot, **_kwargs):
        return snapshot

    async def path(*_args, **_kwargs):
        return []

    service.rest_client = SimpleNamespace(fetch_ticker=fetch)
    monkeypatch.setattr(service, "_attach_native_market_fact", attach)
    monkeypatch.setattr(service, "_fetch_native_consistency_bars", path)
    monkeypatch.setattr(service, "_on_ticker_update", lambda *_args: None)
    snapshot = await service._get_ticker_snapshot("BTC/USDT")
    assert snapshot["received_at"] == receipt


@pytest.mark.asyncio
@pytest.mark.parametrize("receipt", [None, RECEIVED_AT.isoformat()])
async def test_native_market_fact_never_synthesizes_a_missing_ticker_receipt(
    monkeypatch: pytest.MonkeyPatch, receipt_clock, receipt,
) -> None:
    service = _service()

    async def instrument_spec(*_args, **_kwargs):
        return {}

    monkeypatch.setattr(service, "_get_instrument_spec", instrument_spec)
    receipt_clock.current += timedelta(minutes=2)
    snapshot = await service._attach_native_market_fact(
        "BTC/USDT",
        {
            "symbol": "BTC/USDT",
            "last_price": 100.0,
            "timestamp": int(RECEIVED_AT.timestamp() * 1000),
            "received_at": receipt,
        },
        block_on_remote=False,
    )
    assert snapshot["received_at"] == receipt
    if receipt is None:
        assert not snapshot["market_fact"]["received_at"]
    else:
        assert snapshot["market_fact"]["received_at"] == receipt


@pytest.mark.asyncio
@pytest.mark.parametrize("source_clock", [None, int(RECEIVED_AT.timestamp() * 1000)])
async def test_open_interest_records_real_receipt_and_preserves_missing_source_clock(
    monkeypatch: pytest.MonkeyPatch, receipt_clock, source_clock,
) -> None:
    client = OKXRestClient()
    completed_at = RECEIVED_AT + timedelta(seconds=3)

    async def fetch(method, *_args, **_kwargs):
        assert method == "fetch_open_interest"
        receipt_clock.current = completed_at
        return {
            "openInterestAmount": 1000.0,
            "openInterestValue": 100_000.0,
            "info": {"ts": source_clock},
        }

    monkeypatch.setattr(client, "_ccxt_call", fetch)
    result = await client.fetch_open_interest("BTC/USDT")
    assert result["open_interest_contracts"] == 1000.0
    assert result["open_interest_value"] == 100_000.0
    assert result["open_interest_received_at"] == completed_at.isoformat()
    assert result["open_interest_observed_at"] == source_clock


@pytest.mark.asyncio
async def test_required_derivative_refresh_retains_original_optional_receipts(
    monkeypatch: pytest.MonkeyPatch, receipt_clock,
) -> None:
    service = _service()
    funding = {
        "funding_rate": 0.0001, "funding_data_available": True,
        "funding_interval_minutes": 480,
        "funding_rate_observed_at": (RECEIVED_AT - timedelta(seconds=2)).isoformat(),
        "funding_rate_received_at": RECEIVED_AT.isoformat(),
        "open_interest_contracts": 1000.0,
        "open_interest_value": 100_000.0,
        "open_interest_observed_at": (RECEIVED_AT - timedelta(seconds=2)).isoformat(),
        "open_interest_received_at": RECEIVED_AT.isoformat(),
    }
    service._derivatives_cache["BTC/USDT"] = {"data": funding}

    async def spec(*_args, **_kwargs):
        return {}

    async def refresh(*_args, **_kwargs):
        return {"orderbook_bid_depth": 100.0}

    service.rest_client = SimpleNamespace(fetch_derivatives_snapshot=refresh)
    monkeypatch.setattr(service, "_get_instrument_spec", spec)
    receipt_clock.current += timedelta(minutes=1)
    result = await service._refresh_derivatives_snapshot("BTC/USDT", required_only=True)
    for key, value in funding.items():
        assert result[key] == value
