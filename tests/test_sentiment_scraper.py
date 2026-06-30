from __future__ import annotations

from typing import Any

import pytest

from data_feed.sentiment_scraper import SentimentScraper


class FakeHackerNewsResponse:
    status_code = 200

    def json(self) -> dict[str, Any]:
        return {
            "hits": [
                {
                    "objectID": "123",
                    "title": "Ethereum scaling launch gets developer attention",
                    "url": "https://example.com/eth-scaling",
                    "points": 12,
                    "num_comments": 5,
                    "created_at": "2026-06-29T01:02:03Z",
                }
            ]
        }


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    async def get(self, _url: str, **_kwargs: Any) -> FakeHackerNewsResponse:
        self.calls += 1
        return FakeHackerNewsResponse()


@pytest.mark.asyncio
async def test_hacker_news_mentions_adds_second_social_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scraper = SentimentScraper()

    async def fake_get_client() -> FakeClient:
        return FakeClient()

    monkeypatch.setattr(scraper, "_get_client", fake_get_client)

    posts = await scraper.fetch_hacker_news_mentions()

    assert posts
    assert posts[0]["platform"] == "hacker_news"
    assert posts[0]["symbols"] == ["ETH"]
    assert posts[0]["engagement_count"] == 17
