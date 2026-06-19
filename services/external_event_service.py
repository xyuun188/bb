"""Background persistence service for optional external event scraping."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import structlog
from sqlalchemy import select

from config.settings import settings
from core.safe_output import safe_error_text
from core.url_safety import normalize_external_http_url
from data_feed.external_event_scraper import ExternalEventScraper
from db.session import get_session_ctx
from models.news import NewsArticle

logger = structlog.get_logger(__name__)


class ExternalEventService:
    """Collect external event pages in the background and persist them as news."""

    def __init__(self, scraper: ExternalEventScraper | None = None) -> None:
        self.scraper = scraper or ExternalEventScraper()
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        if not settings.external_event_scraper_enabled:
            return
        if self._task and not self._task.done():
            return
        self.scraper.set_tracked_symbols(list(settings.symbols))
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("external event service started")

    async def stop(self) -> None:
        self._running = False
        if not self._task:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        logger.info("external event service stopped")

    async def collect_once(self) -> dict[str, int]:
        articles = await self.scraper.fetch_all()
        stored = await self._persist_articles(articles)
        return {
            "fetched": len(articles),
            "stored": stored,
            "skipped": max(len(articles) - stored, 0),
        }

    async def _loop(self) -> None:
        while self._running:
            try:
                result = await self.collect_once()
                logger.info("external event collection completed", **result)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("external event collection failed", error=safe_error_text(exc))
            await asyncio.sleep(max(int(settings.external_event_scraper_interval_seconds or 0), 60))

    async def _persist_articles(self, articles: list[dict[str, Any]]) -> int:
        stored = 0
        if not articles:
            return stored
        async with get_session_ctx() as session:
            for item in articles:
                title = str(item.get("title") or "").strip()
                if not title:
                    continue
                try:
                    url = normalize_external_http_url(
                        str(item.get("url") or ""),
                        field_name="external event article URL",
                        allow_empty=False,
                        max_length=500,
                    )
                except ValueError:
                    continue
                exists = await session.execute(
                    select(NewsArticle.id).where(NewsArticle.url == url).limit(1)
                )
                if exists.scalar_one_or_none() is not None:
                    continue
                session.add(
                    NewsArticle(
                        source=str(item.get("source") or "scrapling")[:50],
                        title=title,
                        summary=str(item.get("summary") or "")[:2000] or None,
                        url=url,
                        sentiment_score=self._safe_score(item.get("sentiment_score")),
                        symbols_mentioned=item.get("symbols_mentioned") or [],
                        published_at=self._parse_datetime(item.get("published_at")),
                    )
                )
                stored += 1
            await session.flush()
        return stored

    def _safe_score(self, value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(min(score, 1.0), -1.0)

    def _parse_datetime(self, value: Any) -> datetime | None:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=UTC)
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed = parsedate_to_datetime(text)
            except Exception:
                return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
