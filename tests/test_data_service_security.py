from __future__ import annotations

from services.data_service import DataService


def _service() -> DataService:
    service = object.__new__(DataService)
    service._sentiment_cache = {}
    service._headlines_cache = {}
    service._news_items_cache = {}
    return service


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
