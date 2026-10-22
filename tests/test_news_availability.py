from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

import data_feed.news_fetcher as news_module
from data_feed.news_fetcher import NewsFetcher

STARTED_AT = datetime(2026, 10, 22, 1, 0, tzinfo=UTC)
RECEIVED_AT = datetime(2026, 10, 22, 1, 1, tzinfo=UTC)


class Response:
    status_code = 200

    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.body_read = False

    def json(self) -> Any:
        self.body_read = True
        return self.payload

    @property
    def text(self) -> str:
        self.body_read = True
        return str(self.payload)


def patch_network(
    monkeypatch: pytest.MonkeyPatch, fetcher: NewsFetcher, response: Response
) -> None:
    class Clock(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> datetime:
            return RECEIVED_AT if response.body_read else STARTED_AT

    client = SimpleNamespace(get=AsyncMock(return_value=response))
    monkeypatch.setattr(fetcher, "_get_client", AsyncMock(return_value=client))
    monkeypatch.setattr(news_module, "datetime", Clock)
    monkeypatch.setattr(
        news_module,
        "settings",
        SimpleNamespace(
            cryptopanic_api_key="configured",
            coinmarketcal_api_key="configured",
            newsapi_api_key="configured",
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("adapter", "payload", "args"),
    [
        (
            "fetch_cryptopanic",
            {
                "results": [
                    {"title": "Bitcoin ETF approval", "published_at": "2026-10-21T12:00:00Z"}
                ]
            },
            (),
        ),
        (
            "fetch_rss",
            "<rss><channel><item><title>Bitcoin ETF approval</title>"
            "<pubDate>Wed, 21 Oct 2026 12:00:00 GMT</pubDate></item></channel></rss>",
            ("https://example.com/rss",),
        ),
        (
            "fetch_coinmarketcal",
            {
                "body": [
                    {
                        "title": "Bitcoin upgrade",
                        "created_date": "2026-10-21T12:00:00Z",
                        "date_event": "2026-11-01T12:00:00Z",
                    }
                ]
            },
            (),
        ),
        (
            "fetch_newsapi_crypto",
            {
                "articles": [
                    {"title": "Bitcoin ETF approval", "publishedAt": "2026-10-21T12:00:00Z"}
                ]
            },
            (),
        ),
    ],
)
async def test_news_receipt_is_captured_after_response_body_read(
    monkeypatch: pytest.MonkeyPatch, adapter: str, payload: Any, args: tuple
) -> None:
    fetcher = NewsFetcher()
    response = Response(payload)
    patch_network(monkeypatch, fetcher, response)

    articles = await getattr(fetcher, adapter)(*args)

    assert len(articles) == 1
    assert response.body_read
    assert articles[0]["received_at"] == RECEIVED_AT.isoformat()
    assert articles[0]["published_at"] != articles[0]["received_at"]


@pytest.mark.asyncio
@pytest.mark.parametrize("published_at", [None, "2026-10-21T12:00:00Z"])
async def test_scheduled_event_time_is_distinct_from_publication_and_receipt(
    monkeypatch: pytest.MonkeyPatch, published_at: str | None
) -> None:
    fetcher = NewsFetcher()
    response = Response(
        {
            "body": [
                {
                    "title": "Bitcoin scheduled upgrade",
                    "created_date": published_at,
                    "date_event": "2026-11-01T12:00:00Z",
                }
            ]
        }
    )
    patch_network(monkeypatch, fetcher, response)

    article = (await fetcher.fetch_coinmarketcal())[0]

    assert article["event_at"] == "2026-11-01T12:00:00Z"
    assert article["published_at"] == published_at
    assert article["received_at"] == RECEIVED_AT.isoformat()
    if published_at is None:
        assert fetcher._article_sort_key(article)[0] == 0


@pytest.mark.asyncio
async def test_news_cache_retains_original_receipt_and_missing_legacy_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetcher = NewsFetcher()
    observed = fetcher._normalize_article(
        {
            "title": "Bitcoin observed article",
            "published_at": "2026-10-21T12:00:00Z",
            "received_at": RECEIVED_AT.isoformat(),
        },
        "cryptopanic",
    )
    legacy = fetcher._normalize_article(
        {"title": "Bitcoin legacy article", "published_at": "2026-10-20T12:00:00Z"},
        "cryptopanic",
    )
    fetcher._articles = [observed, legacy]
    for adapter in ("fetch_cryptopanic", "fetch_coinmarketcal", "fetch_newsapi_crypto"):
        monkeypatch.setattr(fetcher, adapter, AsyncMock(return_value=[]))
    monkeypatch.setattr(news_module, "RSS_FEEDS", [])

    articles = await fetcher.fetch_all()
    articles[0]["received_at"] = "2099-01-01T00:00:00+00:00"
    cached = await fetcher.fetch_all()

    assert cached[0]["received_at"] == RECEIVED_AT.isoformat()
    assert "received_at" not in cached[1]
