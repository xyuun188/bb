"""
Social sentiment data collection.
Scrapes Reddit (r/CryptoCurrency, r/Bitcoin, etc.) for mention counts and sentiment.
Twitter/X requires paid API access; placeholder for future integration.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
import structlog
from defusedxml import ElementTree

from core.safe_output import safe_error_text

logger = structlog.get_logger(__name__)

# Subreddits to monitor
MONITORED_SUBREDDITS = [
    "CryptoCurrency",
    "Bitcoin",
    "ethereum",
    "solana",
    "CryptoMarkets",
]

# Symbols to track mentions for
DEFAULT_SYMBOL_ALIASES = {
    "BTC": ["BTC", "Bitcoin"],
    "ETH": ["ETH", "Ethereum"],
    "SOL": ["SOL", "Solana"],
}
HACKER_NEWS_QUERIES = (
    "bitcoin",
    "ethereum",
    "solana",
    "crypto",
    "stablecoin",
    "defi",
    "blockchain",
)


class SentimentScraper:
    """Gathers social media sentiment data.

    Uses Reddit's public JSON API (.json suffix) which requires no API key
    for read-only access. Rate limited to ~60 requests/minute.
    """

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._mention_cache: dict[str, list[dict]] = {}
        self._last_fetch: dict[str, datetime] = {}
        self._symbol_aliases: dict[str, list[str]] = dict(DEFAULT_SYMBOL_ALIASES)

    def set_tracked_symbols(self, symbols: list[str]) -> None:
        """Expand Reddit mention extraction for auto-scanned symbols."""
        for symbol in symbols:
            base = str(symbol or "").split("/")[0].split("-")[0].upper()
            if not base:
                continue
            aliases = set(self._symbol_aliases.get(base, []))
            aliases.add(base)
            self._symbol_aliases[base] = sorted(aliases)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(15.0),
                headers={"User-Agent": "AI-Trading-Bot/1.0 (research)"},
            )
        return self._client

    async def fetch_reddit_mentions(self, subreddit: str, limit: int = 25) -> list[dict]:
        """Fetch recent posts from a subreddit and extract crypto mentions."""
        posts = []
        try:
            client = await self._get_client()
            url = f"https://www.reddit.com/r/{subreddit}/new.json"
            resp = await client.get(url, params={"limit": limit})
            if resp.status_code != 200:
                logger.debug("reddit fetch failed", subreddit=subreddit, status=resp.status_code)
                return await self.fetch_reddit_rss_mentions(subreddit, limit=limit)

            data = resp.json()
            for child in data.get("data", {}).get("children", []):
                post_data = child.get("data", {})
                title = post_data.get("title", "")
                selftext = post_data.get("selftext", "")
                combined = f"{title} {selftext}"
                permalink = post_data.get("permalink", "")

                # Count mentions of tracked symbols
                mentions = []
                for base, aliases in self._symbol_aliases.items():
                    if any(self._contains_alias(combined, alias) for alias in aliases):
                        mentions.append(base)

                if mentions:
                    posts.append(
                        {
                            "platform": "reddit",
                            "subreddit": subreddit,
                            "post_id": post_data.get("id", ""),
                            "title": title,
                            "content": selftext[:500],
                            "symbols": mentions,
                            "score": post_data.get("score", 0),
                            "sentiment_score": self._lexicon_sentiment(combined),
                            "num_comments": post_data.get("num_comments", 0),
                            "engagement_count": post_data.get("score", 0)
                            + post_data.get("num_comments", 0),
                            "posted_at": datetime.fromtimestamp(
                                post_data.get("created_utc", 0), tz=UTC
                            ),
                            "url": f"https://reddit.com{permalink}",
                        }
                    )

        except Exception as e:
            logger.debug("reddit error", subreddit=subreddit, error=safe_error_text(e))

        return posts

    async def fetch_reddit_rss_mentions(self, subreddit: str, limit: int = 25) -> list[dict]:
        """Fallback to Reddit RSS when the JSON endpoint is blocked or rate-limited."""
        posts: list[dict] = []
        try:
            client = await self._get_client()
            url = f"https://www.reddit.com/r/{subreddit}/new/.rss"
            resp = await client.get(url, params={"limit": limit})
            if resp.status_code != 200:
                logger.debug(
                    "reddit rss fetch failed", subreddit=subreddit, status=resp.status_code
                )
                return posts

            root = ElementTree.fromstring(resp.text)
            entries = [
                item
                for item in root.iter()
                if item.tag.lower().endswith("entry") or item.tag.lower().endswith("item")
            ]
            for entry in entries[:limit]:
                title = ""
                content = ""
                link = ""
                posted_at = None
                post_id = ""
                for child in entry:
                    tag = child.tag.lower().split("}")[-1]
                    if tag == "title":
                        title = (child.text or "").strip()
                    elif tag in {"content", "summary", "description"}:
                        content = (child.text or "").strip()[:500]
                    elif tag == "link":
                        link = child.attrib.get("href") or (child.text or "").strip()
                    elif tag in {"updated", "published", "pubdate"}:
                        raw_date = (child.text or "").strip()
                        try:
                            posted_at = parsedate_to_datetime(raw_date)
                        except Exception:
                            try:
                                posted_at = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
                            except Exception:
                                posted_at = None
                    elif tag == "id":
                        post_id = (child.text or "").strip()

                combined = f"{title} {content}"
                mentions = []
                for base, aliases in self._symbol_aliases.items():
                    if any(self._contains_alias(combined, alias) for alias in aliases):
                        mentions.append(base)
                if not mentions:
                    continue
                posts.append(
                    {
                        "platform": "reddit_rss",
                        "subreddit": subreddit,
                        "post_id": post_id or link or f"{subreddit}:{hash(combined)}",
                        "title": title,
                        "content": content,
                        "symbols": mentions,
                        "score": 0,
                        "sentiment_score": self._lexicon_sentiment(combined),
                        "num_comments": 0,
                        "engagement_count": 0,
                        "posted_at": posted_at or datetime.now(UTC),
                        "url": link,
                    }
                )
        except Exception as e:
            logger.debug("reddit rss error", subreddit=subreddit, error=safe_error_text(e))
        return posts

    def _contains_alias(self, text: str, alias: str) -> bool:
        pattern = rf"(?<![A-Za-z0-9]){re.escape(str(alias))}(?![A-Za-z0-9])"
        return re.search(pattern, str(text or ""), flags=re.IGNORECASE) is not None

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

    async def fetch_all_reddit(self) -> list[dict]:
        """Fetch from all monitored subreddits concurrently."""
        import asyncio

        tasks = [self.fetch_reddit_mentions(sub) for sub in MONITORED_SUBREDDITS]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_posts = []
        for result in results:
            if isinstance(result, list):
                all_posts.extend(result)

        logger.info("reddit posts fetched", total=len(all_posts))
        return all_posts

    async def fetch_hacker_news_mentions(self, limit: int = 40) -> list[dict]:
        """Fetch public Hacker News discussions that mention tracked crypto symbols."""
        posts: list[dict] = []
        seen_ids: set[str] = set()
        try:
            client = await self._get_client()
            per_query_limit = min(max(int(limit), 1), 100)
            for query in HACKER_NEWS_QUERIES:
                if len(posts) >= limit:
                    break
                resp = await client.get(
                    "https://hn.algolia.com/api/v1/search_by_date",
                    params={
                        "query": query,
                        "tags": "story",
                        "hitsPerPage": per_query_limit,
                    },
                )
                if resp.status_code != 200:
                    logger.debug(
                        "hacker news fetch failed", query=query, status=resp.status_code
                    )
                    continue

                data = resp.json()
                for hit in data.get("hits", []):
                    title = str(hit.get("title") or hit.get("story_title") or "").strip()
                    url = str(hit.get("url") or hit.get("story_url") or "").strip()
                    combined = f"{title} {url}"
                    mentions = [
                        base
                        for base, aliases in self._symbol_aliases.items()
                        if any(self._contains_alias(combined, alias) for alias in aliases)
                    ]
                    if not mentions:
                        continue
                    created_at = None
                    raw_created = str(hit.get("created_at") or "").strip()
                    if raw_created:
                        try:
                            created_at = datetime.fromisoformat(
                                raw_created.replace("Z", "+00:00")
                            )
                        except ValueError:
                            created_at = None
                    object_id = str(hit.get("objectID") or "").strip()
                    post_id = f"hacker_news:{object_id or hash(combined)}"
                    if post_id in seen_ids:
                        continue
                    seen_ids.add(post_id)
                    points = int(hit.get("points") or 0)
                    comments = int(hit.get("num_comments") or 0)
                    posts.append(
                        {
                            "platform": "hacker_news",
                            "post_id": post_id,
                            "title": title,
                            "content": title,
                            "symbols": mentions,
                            "score": points,
                            "sentiment_score": self._lexicon_sentiment(combined),
                            "num_comments": comments,
                            "engagement_count": points + comments,
                            "posted_at": created_at or datetime.now(UTC),
                            "url": url or f"https://news.ycombinator.com/item?id={object_id}",
                        }
                    )
                    if len(posts) >= limit:
                        break
        except Exception as e:
            logger.debug("hacker news error", error=safe_error_text(e))
        return posts

    async def fetch_all_social(self) -> list[dict]:
        """Fetch public social/discussion sources without API keys."""
        import asyncio

        results = await asyncio.gather(
            self.fetch_all_reddit(),
            self.fetch_hacker_news_mentions(),
            return_exceptions=True,
        )
        all_posts: list[dict] = []
        for result in results:
            if isinstance(result, list):
                all_posts.extend(result)
        logger.info(
            "social posts fetched",
            total=len(all_posts),
            platforms=sorted({str(item.get("platform") or "") for item in all_posts}),
        )
        return all_posts

    async def get_mention_stats(self, symbol: str) -> dict[str, Any]:
        """Get mention count and average score for a symbol across recent posts."""
        posts = await self.fetch_all_social()
        symbol_posts = [
            p for p in posts if symbol.upper() in [s.upper() for s in p.get("symbols", [])]
        ]
        total_engagement = sum(p.get("engagement_count", 0) for p in symbol_posts)
        return {
            "symbol": symbol,
            "mention_count": len(symbol_posts),
            "total_engagement": total_engagement,
            "avg_score": (
                sum(p.get("score", 0) for p in symbol_posts) / len(symbol_posts)
                if symbol_posts
                else 0
            ),
        }

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
