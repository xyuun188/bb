from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

import data_feed.sentiment_scraper as sentiment_module
from data_feed.sentiment_scraper import SentimentScraper

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
    monkeypatch: pytest.MonkeyPatch, scraper: SentimentScraper, response: Response
) -> None:
    class Clock(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> datetime:
            return RECEIVED_AT if response.body_read else STARTED_AT

    client = SimpleNamespace(get=AsyncMock(return_value=response))
    monkeypatch.setattr(scraper, "_get_client", AsyncMock(return_value=client))
    monkeypatch.setattr(sentiment_module, "datetime", Clock)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("adapter", "payload", "args"),
    [
        (
            "fetch_reddit_mentions",
            {"data": {"children": [{"data": {"id": "1", "title": "Bitcoin launch"}}]}},
            ("Bitcoin",),
        ),
        (
            "fetch_reddit_rss_mentions",
            "<feed><entry><id>1</id><title>Bitcoin launch</title></entry></feed>",
            ("Bitcoin",),
        ),
        (
            "fetch_hacker_news_mentions",
            {"hits": [{"objectID": "1", "title": "Bitcoin launch"}]},
            (),
        ),
    ],
)
async def test_social_receipt_follows_body_read_without_fabricating_publication(
    monkeypatch: pytest.MonkeyPatch, adapter: str, payload: Any, args: tuple
) -> None:
    scraper = SentimentScraper()
    response = Response(payload)
    patch_network(monkeypatch, scraper, response)

    posts = await getattr(scraper, adapter)(*args, limit=1)

    assert len(posts) == 1
    assert response.body_read
    assert posts[0]["received_at"] == RECEIVED_AT.isoformat()
    assert posts[0]["posted_at"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("adapter", "payload", "args"),
    [
        (
            "fetch_reddit_mentions",
            {
                "data": {
                    "children": [
                        {
                            "data": {
                                "id": "1",
                                "title": "Bitcoin launch",
                                "created_utc": 1792584000,
                            }
                        }
                    ]
                }
            },
            ("Bitcoin",),
        ),
        (
            "fetch_reddit_rss_mentions",
            "<feed><entry><id>1</id><title>Bitcoin launch</title>"
            "<published>2026-10-21T12:00:00Z</published></entry></feed>",
            ("Bitcoin",),
        ),
        (
            "fetch_hacker_news_mentions",
            {
                "hits": [
                    {
                        "objectID": "1",
                        "title": "Bitcoin launch",
                        "created_at": "2026-10-21T12:00:00Z",
                    }
                ]
            },
            (),
        ),
    ],
)
async def test_social_source_publication_remains_distinct_from_receipt(
    monkeypatch: pytest.MonkeyPatch, adapter: str, payload: Any, args: tuple
) -> None:
    scraper = SentimentScraper()
    response = Response(payload)
    patch_network(monkeypatch, scraper, response)

    post = (await getattr(scraper, adapter)(*args, limit=1))[0]

    assert post["posted_at"] == datetime(2026, 10, 21, 12, 0, tzinfo=UTC)
    assert post["received_at"] == RECEIVED_AT.isoformat()


@pytest.mark.asyncio
async def test_reddit_fallback_receipt_comes_from_successful_rss_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scraper = SentimentScraper()
    failed_response = SimpleNamespace(status_code=429)
    rss_response = Response(
        "<feed><entry><id>1</id><title>Bitcoin launch</title></entry></feed>"
    )
    patch_network(monkeypatch, scraper, rss_response)
    client = SimpleNamespace(get=AsyncMock(side_effect=[failed_response, rss_response]))
    monkeypatch.setattr(scraper, "_get_client", AsyncMock(return_value=client))

    posts = await scraper.fetch_reddit_mentions("Bitcoin", limit=1)

    assert len(posts) == 1
    assert posts[0]["platform"] == "reddit_rss"
    assert posts[0]["received_at"] == RECEIVED_AT.isoformat()
    assert posts[0]["posted_at"] is None
