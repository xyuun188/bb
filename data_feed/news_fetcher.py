"""
Multi-source news fetcher for cryptocurrency news.
Supports CryptoPanic API (free tier) and RSS feeds as fallback.
Deduplicates by URL and extracts mentioned symbols.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
import structlog
from defusedxml import ElementTree

from config.settings import settings
from core.safe_output import safe_error_text

logger = structlog.get_logger(__name__)

# Known crypto symbol patterns for extraction
CRYPTO_KEYWORDS = [
    "BTC",
    "Bitcoin",
    "ETH",
    "Ethereum",
    "SOL",
    "Solana",
    "BNB",
    "XRP",
    "DOGE",
    "Dogecoin",
    "ADA",
    "Cardano",
    "AVAX",
    "Avalanche",
    "DOT",
    "Polkadot",
    "MATIC",
    "Polygon",
    "LINK",
    "Chainlink",
    "UNI",
    "Uniswap",
    "USDT",
    "USDC",
    "DAI",
    "APT",
    "SUI",
    "ARB",
    "OP",
    "PEPE",
    "WIF",
]

SYMBOL_ALIASES: dict[str, list[str]] = {
    "BTC": ["BTC", "Bitcoin"],
    "ETH": ["ETH", "Ethereum"],
    "SOL": ["SOL", "Solana"],
    "BNB": ["BNB", "Binance Coin", "BNB Chain"],
    "XRP": ["XRP", "Ripple"],
    "DOGE": ["DOGE", "Dogecoin"],
    "ADA": ["ADA", "Cardano"],
    "AVAX": ["AVAX", "Avalanche"],
    "DOT": ["DOT", "Polkadot"],
    "LINK": ["LINK", "Chainlink"],
    "UNI": ["UNI", "Uniswap"],
    "SUI": ["SUI", "Sui"],
    "APT": ["APT", "Aptos"],
    "ARB": ["ARB", "Arbitrum"],
    "OP": ["OP", "Optimism"],
    "TON": ["TON", "Toncoin", "The Open Network"],
    "XLM": ["XLM", "Stellar"],
    "FIL": ["FIL", "Filecoin"],
    "ICP": ["ICP", "Internet Computer"],
    "AAVE": ["AAVE"],
    "WLFI": ["WLFI", "World Liberty Financial"],
}

# Free RSS feeds (no API key required)
RSS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
    "https://cryptoslate.com/feed/",
    "https://www.theblock.co/rss.xml",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://bitcoinmagazine.com/.rss/full/",
    "https://blockworks.co/feed",
    "https://coingape.com/feed/",
    "https://cryptobriefing.com/feed/",
    "https://www.newsbtc.com/feed/",
    "https://u.today/rss",
    "https://coinjournal.net/news/feed/",
    "https://www.okx.com/help/rss/announcements-en.xml",
    "https://www.binance.com/en/support/announcement/rss",
]

EVENT_RULES: list[tuple[str, tuple[str, ...], float, int]] = [
    ("security", ("hack", "exploit", "drain", "stolen", "breach", "key compromise"), -0.85, 5),
    (
        "regulatory",
        ("sec sues", "lawsuit", "ban", "charges", "settlement", "investigation"),
        -0.55,
        4,
    ),
    ("listing", ("listing", "will list", "new trading pairs", "launchpool", "launchpad"), 0.60, 4),
    ("delisting", ("delist", "remove trading", "suspend trading"), -0.75, 5),
    ("unlock", ("token unlock", "unlock schedule", "vesting"), -0.35, 3),
    ("partnership", ("partnership", "integration", "collaboration", "mainnet", "upgrade"), 0.35, 3),
    ("etf", ("etf", "approval", "spot etf"), 0.45, 4),
]

SOURCE_WEIGHTS = {
    "cryptopanic": 0.85,
    "cointelegraph.com": 0.75,
    "www.coindesk.com": 0.80,
    "coindesk.com": 0.80,
    "www.theblock.co": 0.80,
    "theblock.co": 0.80,
    "decrypt.co": 0.65,
    "cryptoslate.com": 0.65,
    "www.okx.com": 0.90,
    "www.binance.com": 0.95,
    "bitcoinmagazine.com": 0.65,
    "blockworks.co": 0.70,
    "coingape.com": 0.60,
    "cryptobriefing.com": 0.65,
    "www.newsbtc.com": 0.55,
    "u.today": 0.60,
    "coinjournal.net": 0.55,
    "okx_announcements": 0.95,
    "coinmarketcal": 0.80,
    "newsapi": 0.65,
}


class NewsFetcher:
    """Aggregates crypto news from multiple sources with deduplication."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._seen_urls: set[str] = set()
        self._articles: list[dict] = []
        self._max_cache = 500
        self._symbol_aliases: dict[str, set[str]] = {
            base: {base, *aliases} for base, aliases in SYMBOL_ALIASES.items()
        }
        for keyword in CRYPTO_KEYWORDS:
            base = keyword.upper()
            self._symbol_aliases.setdefault(base, {base}).add(keyword)

    def _remember_dedup(self, key: str) -> bool:
        if not key:
            return False
        if key in self._seen_urls:
            return False
        self._seen_urls.add(key)
        if len(self._seen_urls) > self._max_cache * 4:
            self._seen_urls = set(list(self._seen_urls)[-self._max_cache * 2 :])
        return True

    def set_tracked_symbols(self, symbols: list[str]) -> None:
        """Add requested trading pairs to symbol extraction coverage."""
        for symbol in symbols:
            base = str(symbol or "").split("/")[0].split("-")[0].upper()
            if base:
                self._symbol_aliases.setdefault(base, {base}).add(base)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(15.0),
                headers={"User-Agent": "AI-Trading-Bot/1.0"},
            )
        return self._client

    def _extract_symbols(self, text: str) -> list[str]:
        """Extract mentioned cryptocurrency symbols from text."""
        found = []
        raw_text = str(text or "")
        text_upper = raw_text.upper()
        for base, aliases in self._symbol_aliases.items():
            for alias in sorted(aliases, key=len, reverse=True):
                pattern = rf"(?<![A-Z0-9]){re.escape(alias.upper())}(?![A-Z0-9])"
                if not re.search(pattern, text_upper):
                    continue
                if base not in found:
                    found.append(base)
                break
        return found

    def _classify_event(self, text: str) -> dict[str, Any]:
        lower = str(text or "").lower()
        best: tuple[str, float, int] | None = None
        for event_type, terms, score, impact in EVENT_RULES:
            if any(term in lower for term in terms):
                if best is None or impact > best[2]:
                    best = (event_type, score, impact)
        if best is None:
            return {"event_type": "market_news", "event_score": 0.0, "impact_level": 1}
        return {"event_type": best[0], "event_score": best[1], "impact_level": best[2]}

    def _source_weight(self, source: str) -> float:
        return SOURCE_WEIGHTS.get(str(source or "").lower(), 0.55)

    def _normalize_article(self, item: dict[str, Any], source: str) -> dict[str, Any]:
        title = str(item.get("title") or "").strip()
        summary = str(item.get("summary") or item.get("body") or "").strip()
        event = self._classify_event(f"{title} {summary}")
        sentiment = self._lexicon_sentiment(f"{title} {summary}")
        if abs(float(event.get("event_score") or 0.0)) > abs(sentiment):
            sentiment = float(event["event_score"])
        published_at = item.get("published_at")
        return {
            **item,
            "source": source,
            "title": title,
            "summary": summary[:500],
            "symbols_mentioned": item.get("symbols_mentioned")
            or self._extract_symbols(f"{title} {summary}"),
            "sentiment_score": sentiment,
            "event_type": event["event_type"],
            "impact_level": event["impact_level"],
            "source_weight": self._source_weight(source),
            "published_at": published_at,
        }

    def _dedup_key(self, title: str) -> str:
        return hashlib.sha256(title.strip().lower().encode()).hexdigest()

    async def fetch_cryptopanic(self) -> list[dict]:
        """Fetch news from CryptoPanic free API.

        CryptoPanic free tier provides recent news without an API key,
        but with rate limits. If API key present, use authenticated endpoint.
        """
        articles: list[dict[str, Any]] = []
        try:
            client = await self._get_client()
            # Public API endpoint for recent posts
            url = "https://cryptopanic.com/api/v1/posts/"
            params: dict = {
                "filter": "important",
                "regions": "en",
            }
            token = (settings.cryptopanic_api_key or "").strip()
            if not token:
                return articles
            params["auth_token"] = token
            headers: dict[str, str] = {}
            # Note: CryptoPanic free tier has an auth_token from their web app
            # For production, get a proper API key at cryptopanic.com/developers

            resp = await client.get(url, params=params, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                for post in data.get("results", [])[:20]:
                    title = post.get("title", "")
                    if not title:
                        continue
                    dkey = self._dedup_key(title)
                    if not self._remember_dedup(dkey):
                        continue

                    currencies = post.get("currencies", [])
                    symbols = [c.get("code", "") for c in currencies] if currencies else []

                    articles.append(
                        self._normalize_article(
                            {
                                "title": title,
                                "summary": post.get("body", "")[:500] if post.get("body") else "",
                                "url": post.get("url", ""),
                                "symbols_mentioned": symbols or self._extract_symbols(title),
                                "published_at": post.get("published_at"),
                            },
                            "cryptopanic",
                        )
                    )
            else:
                logger.debug("cryptopanic fetch failed", status=resp.status_code)
        except Exception as e:
            logger.debug("cryptopanic error", error=safe_error_text(e))

        return articles

    async def fetch_rss(self, feed_url: str) -> list[dict]:
        """Fetch and parse an RSS feed."""
        articles: list[dict[str, Any]] = []
        try:
            client = await self._get_client()
            resp = await client.get(feed_url)
            if resp.status_code != 200:
                return articles

            root = ElementTree.fromstring(resp.text)
            for item in root.iter("item"):
                title = ""
                link = ""
                description = ""
                pub_date = None

                for child in item:
                    tag = child.tag.lower() if "}" not in child.tag else child.tag.split("}")[-1]
                    if tag == "title":
                        title = (child.text or "").strip()
                    elif tag == "link":
                        link = (child.text or "").strip()
                    elif tag in ("description", "content"):
                        description = (child.text or "").strip()[:500]
                    elif tag == "pubdate":
                        pub_date = (child.text or "").strip()

                if not title:
                    continue

                dkey = self._dedup_key(title)
                if not self._remember_dedup(dkey):
                    continue

                source_name = feed_url.split("//")[-1].split("/")[0]
                articles.append(
                    self._normalize_article(
                        {
                            "title": title,
                            "summary": description,
                            "url": link,
                            "symbols_mentioned": self._extract_symbols(f"{title} {description}"),
                            "published_at": pub_date,
                        },
                        source_name,
                    )
                )

        except Exception as e:
            logger.debug("rss fetch error", url=feed_url, error=safe_error_text(e))

        return articles

    async def fetch_okx_announcements(self) -> list[dict]:
        """Fetch public OKX announcements from RSS instead of OKX REST API."""
        return await self.fetch_rss("https://www.okx.com/help/rss/announcements-en.xml")

    async def fetch_coinmarketcal(self) -> list[dict]:
        """Fetch CoinMarketCal events when a free API key is configured."""
        token = (settings.coinmarketcal_api_key or "").strip()
        if not token:
            return []
        articles: list[dict] = []
        try:
            client = await self._get_client()
            since = datetime.now(UTC).date().isoformat()
            until = (datetime.now(UTC) + timedelta(days=14)).date().isoformat()
            resp = await client.get(
                "https://developers.coinmarketcal.com/v1/events",
                headers={"x-api-key": token, "Accept": "application/json"},
                params={"dateRangeStart": since, "dateRangeEnd": until, "max": 50},
            )
            if resp.status_code != 200:
                logger.debug("coinmarketcal fetch failed", status=resp.status_code)
                return articles
            data = resp.json()
            rows = data.get("body") if isinstance(data, dict) else data
            if not isinstance(rows, list):
                return articles
            for item in rows[:50]:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or item.get("name") or "").strip()
                coins = item.get("coins") or item.get("currencies") or []
                symbols = []
                if isinstance(coins, list):
                    for coin in coins:
                        if isinstance(coin, dict):
                            symbol = coin.get("symbol") or coin.get("code")
                            if symbol:
                                symbols.append(str(symbol).upper())
                summary = str(item.get("description") or item.get("proof") or "")[:500]
                if not title:
                    continue
                url = str(item.get("source") or item.get("link") or item.get("url") or "")
                key = url or self._dedup_key(f"coinmarketcal:{title}:{','.join(symbols)}")
                if not self._remember_dedup(key):
                    continue
                articles.append(
                    self._normalize_article(
                        {
                            "title": f"[Event] {title}",
                            "summary": summary,
                            "url": url,
                            "symbols_mentioned": symbols
                            or self._extract_symbols(f"{title} {summary}"),
                            "published_at": item.get("date_event") or item.get("created_date"),
                        },
                        "coinmarketcal",
                    )
                )
        except Exception as e:
            logger.debug("coinmarketcal error", error=safe_error_text(e))
        return articles

    async def fetch_newsapi_crypto(self) -> list[dict]:
        """Fetch broad crypto/macroeconomic news when a free NewsAPI key is configured."""
        token = (settings.newsapi_api_key or "").strip()
        if not token:
            return []
        articles: list[dict] = []
        try:
            client = await self._get_client()
            resp = await client.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": "(crypto OR bitcoin OR ethereum OR blockchain OR SEC ETF)",
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 50,
                    "apiKey": token,
                },
            )
            if resp.status_code != 200:
                logger.debug("newsapi fetch failed", status=resp.status_code)
                return articles
            data = resp.json()
            for item in data.get("articles", [])[:50]:
                title = str(item.get("title") or "").strip()
                if not title:
                    continue
                url = str(item.get("url") or "")
                if not self._remember_dedup(url or self._dedup_key(f"newsapi:{title}")):
                    continue
                source = item.get("source") if isinstance(item.get("source"), dict) else {}
                articles.append(
                    self._normalize_article(
                        {
                            "title": title,
                            "summary": str(item.get("description") or item.get("content") or "")[
                                :500
                            ],
                            "url": url,
                            "symbols_mentioned": self._extract_symbols(
                                f"{title} {item.get('description') or ''}"
                            ),
                            "published_at": item.get("publishedAt"),
                        },
                        f"newsapi:{source.get('name') or 'unknown'}",
                    )
                )
        except Exception as e:
            logger.debug("newsapi error", error=safe_error_text(e))
        return articles

    async def fetch_all(self) -> list[dict]:
        """Fetch from all sources concurrently. Returns deduped article list."""
        tasks = [
            self.fetch_cryptopanic(),
            self.fetch_coinmarketcal(),
            self.fetch_newsapi_crypto(),
        ] + [self.fetch_rss(url) for url in RSS_FEEDS]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_articles = []
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)

        self._articles = sorted(
            (all_articles + self._articles)[: self._max_cache],
            key=self._article_sort_key,
            reverse=True,
        )
        logger.info("news fetched", count=len(all_articles), cached=len(self._articles))
        return list(self._articles)

    def _article_sort_key(self, article: dict[str, Any]) -> tuple[float, float, float]:
        published = article.get("published_at")
        ts = 0.0
        if isinstance(published, datetime):
            ts = published.timestamp()
        elif published:
            try:
                ts = datetime.fromisoformat(str(published).replace("Z", "+00:00")).timestamp()
            except ValueError:
                try:
                    ts = parsedate_to_datetime(str(published)).timestamp()
                except Exception:
                    ts = 0.0
        impact = float(article.get("impact_level") or 1.0)
        source_weight = float(article.get("source_weight") or 0.0)
        return ts, impact, source_weight

    def _parse_millis_time(self, value: Any) -> str | None:
        if value in (None, ""):
            return None
        try:
            number = float(value)
            if number > 1_000_000_000_000:
                number /= 1000.0
            return datetime.fromtimestamp(number, tz=UTC).isoformat()
        except (TypeError, ValueError, OSError):
            return str(value)

    def _lexicon_sentiment(self, text: str) -> float:
        lower = str(text or "").lower()
        positive = (
            "surge",
            "rally",
            "gain",
            "bull",
            "breakout",
            "record",
            "approval",
            "partnership",
            "launch",
            "up",
        )
        negative = (
            "crash",
            "hack",
            "lawsuit",
            "bear",
            "dump",
            "plunge",
            "ban",
            "exploit",
            "down",
            "liquidation",
        )
        score = sum(1 for word in positive if word in lower) - sum(
            1 for word in negative if word in lower
        )
        if score == 0:
            return 0.0
        return max(min(score / 4.0, 1.0), -1.0)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
