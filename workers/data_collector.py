"""
Background data collection worker.
Periodically fetches news, sentiment, and stores to DB and Redis cache.
"""

from __future__ import annotations

import asyncio

import structlog

from data_feed.news_fetcher import NewsFetcher
from data_feed.sentiment_scraper import SentimentScraper
from db.repositories.market_repo import MarketRepository
from db.session import get_session_ctx

logger = structlog.get_logger(__name__)


class DataCollectorWorker:
    """Runs periodic data collection tasks in the background."""

    def __init__(self, redis_client=None) -> None:
        self.news_fetcher = NewsFetcher()
        self.sentiment_scraper = SentimentScraper()
        self.redis = redis_client
        self._running = False
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        """Start all background collection tasks."""
        self._running = True

        # News collection every 5 minutes
        self._tasks.append(asyncio.create_task(self._news_loop()))

        # Sentiment collection every 10 minutes
        self._tasks.append(asyncio.create_task(self._sentiment_loop()))

        logger.info("data collector started")

    async def stop(self) -> None:
        """Stop all background collection tasks."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.news_fetcher.close()
        await self.sentiment_scraper.close()
        logger.info("data collector stopped")

    async def _news_loop(self) -> None:
        """Fetch news every 5 minutes and store to DB."""
        while self._running:
            try:
                articles = await self.news_fetcher.fetch_all()
                if articles:
                    async with get_session_ctx() as session:
                        from models.news import NewsArticle

                        for article in articles:
                            existing = await session.execute(
                                # Check by URL for dedup
                                f"SELECT id FROM news_articles WHERE url = :url",
                                {"url": article["url"]},
                            )
                            if existing.first():
                                continue

                            news_entry = NewsArticle(
                                source=article["source"],
                                title=article["title"],
                                summary=article.get("summary", ""),
                                url=article["url"],
                                symbols_mentioned=article.get("symbols_mentioned", []),
                                published_at=article.get("published_at"),
                                sentiment_score=float(article.get("sentiment_score") or 0.0),
                            )
                            session.add(news_entry)

                    logger.debug("news stored", count=len(articles))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("news collection error", error=str(e))

            await asyncio.sleep(300)  # 5 minutes

    async def _sentiment_loop(self) -> None:
        """Fetch social sentiment every 10 minutes."""
        while self._running:
            try:
                posts = await self.sentiment_scraper.fetch_all_reddit()
                if posts:
                    async with get_session_ctx() as session:
                        from models.news import SocialPost

                        for post in posts:
                            existing = await session.execute(
                                f"SELECT id FROM social_posts WHERE post_id = :pid",
                                {"pid": post["post_id"]},
                            )
                            if existing.first():
                                continue

                            social_entry = SocialPost(
                                platform=post["platform"],
                                post_id=post["post_id"],
                                content=post.get("content", ""),
                                symbols=post.get("symbols", []),
                                sentiment_score=0.0,
                                engagement_count=post.get("engagement_count", 0),
                                posted_at=post.get("posted_at"),
                            )
                            session.add(social_entry)

                    logger.debug("social posts stored", count=len(posts))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("sentiment collection error", error=str(e))

            await asyncio.sleep(600)  # 10 minutes
