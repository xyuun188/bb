from __future__ import annotations

from typing import Any

import pytest

from services import data_service as data_service_module
from services.data_service import DataService


def _service() -> DataService:
    service = object.__new__(DataService)
    service._sentiment_cache = {}
    service._headlines_cache = {}
    service._news_items_cache = {}
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
