"""Background persistence service for optional external event scraping."""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import structlog
from dotenv import dotenv_values
from sqlalchemy import select

from config.settings import parse_external_event_scraper_sources_value, settings
from core.safe_output import safe_error_text
from core.url_safety import normalize_external_http_url
from data_feed.external_event_scraper import ExternalEventScraper
from db.session import get_session_ctx
from models.news import NewsArticle
from services.secure_runtime_config import load_secure_settings_into_runtime

logger = structlog.get_logger(__name__)

_SCRAPER_SETTING_KEYS = (
    "EXTERNAL_EVENT_SCRAPER_ENABLED",
    "EXTERNAL_EVENT_SCRAPER_INTERVAL_SECONDS",
    "EXTERNAL_EVENT_SCRAPER_TIMEOUT_SECONDS",
    "EXTERNAL_EVENT_SCRAPER_MAX_SOURCES",
    "EXTERNAL_EVENT_SCRAPER_MAX_ITEMS_PER_SOURCE",
    "EXTERNAL_EVENT_SCRAPER_SOURCES",
    "CRYPTOPANIC_API_KEY",
    "COINMARKETCAL_API_KEY",
    "NEWSAPI_API_KEY",
)


def _project_env_path() -> Path:
    return Path(settings.project_root) / ".env"


def _bool_env(value: Any, default: bool) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _int_env(value: Any, default: int, *, minimum: int, maximum: int) -> int:
    try:
        parsed = int(float(str(value).strip()))
    except (TypeError, ValueError):
        parsed = default
    return max(min(parsed, maximum), minimum)


def _float_env(value: Any, default: float, *, minimum: float, maximum: float) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        parsed = default
    return max(min(parsed, maximum), minimum)


def _sources_env(value: Any) -> list[dict[str, Any]]:
    return parse_external_event_scraper_sources_value(value)


def load_external_event_settings_from_env() -> dict[str, Any]:
    """Reload external event collection settings from .env into this process."""

    env_path = _project_env_path()
    values: dict[str, Any] = {}
    if env_path.exists():
        values.update({key: value for key, value in dotenv_values(env_path).items() if key})
    values.update(
        {key: os.environ.get(key) for key in _SCRAPER_SETTING_KEYS if os.environ.get(key)}
    )

    settings.external_event_scraper_enabled = _bool_env(
        values.get("EXTERNAL_EVENT_SCRAPER_ENABLED"),
        bool(settings.external_event_scraper_enabled),
    )
    settings.external_event_scraper_interval_seconds = _int_env(
        values.get("EXTERNAL_EVENT_SCRAPER_INTERVAL_SECONDS"),
        int(settings.external_event_scraper_interval_seconds),
        minimum=60,
        maximum=86400,
    )
    settings.external_event_scraper_timeout_seconds = _float_env(
        values.get("EXTERNAL_EVENT_SCRAPER_TIMEOUT_SECONDS"),
        float(settings.external_event_scraper_timeout_seconds),
        minimum=1.0,
        maximum=30.0,
    )
    settings.external_event_scraper_max_sources = _int_env(
        values.get("EXTERNAL_EVENT_SCRAPER_MAX_SOURCES"),
        int(settings.external_event_scraper_max_sources),
        minimum=1,
        maximum=20,
    )
    settings.external_event_scraper_max_items_per_source = _int_env(
        values.get("EXTERNAL_EVENT_SCRAPER_MAX_ITEMS_PER_SOURCE"),
        int(settings.external_event_scraper_max_items_per_source),
        minimum=1,
        maximum=50,
    )
    settings.external_event_scraper_sources = _sources_env(
        values.get("EXTERNAL_EVENT_SCRAPER_SOURCES")
    )
    for env_key, attr in (
        ("CRYPTOPANIC_API_KEY", "cryptopanic_api_key"),
        ("COINMARKETCAL_API_KEY", "coinmarketcal_api_key"),
        ("NEWSAPI_API_KEY", "newsapi_api_key"),
    ):
        value = str(values.get(env_key) or "").strip()
        if value:
            setattr(settings, attr, value)
    return {
        "enabled": bool(settings.external_event_scraper_enabled),
        "source_count": len(settings.external_event_scraper_sources),
        "interval_seconds": int(settings.external_event_scraper_interval_seconds),
    }


class ExternalEventService:
    """Collect external event pages in the background and persist them as news."""

    def __init__(self, scraper: ExternalEventScraper | None = None) -> None:
        self.scraper = scraper or ExternalEventScraper()
        self._task: asyncio.Task | None = None
        self._settings_task: asyncio.Task | None = None
        self._running = False
        self._env_mtime: float | None = None

    async def start(self) -> None:
        load_external_event_settings_from_env()
        self._ensure_settings_watcher()
        if not settings.external_event_scraper_enabled:
            return
        if self._task and not self._task.done():
            return
        self.scraper.set_tracked_symbols(list(settings.symbols))
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("external event service started")

    async def start_controller(self) -> None:
        """Start runtime config watcher; collection task follows current settings."""

        await self.start()

    async def stop(self) -> None:
        self._running = False
        if self._settings_task:
            self._settings_task.cancel()
            try:
                await self._settings_task
            except asyncio.CancelledError:
                pass
            self._settings_task = None
        if not self._task:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        logger.info("external event service stopped")

    def _ensure_settings_watcher(self) -> None:
        if self._settings_task and not self._settings_task.done():
            return
        self._settings_task = asyncio.create_task(self._settings_loop())

    async def _settings_loop(self) -> None:
        env_path = _project_env_path()
        while True:
            try:
                mtime = env_path.stat().st_mtime if env_path.exists() else None
                if mtime != self._env_mtime:
                    self._env_mtime = mtime
                    await self.reload_runtime_settings()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("external event settings reload failed", error=safe_error_text(exc))
            await asyncio.sleep(5)

    async def reload_runtime_settings(self) -> dict[str, Any]:
        before_enabled = bool(settings.external_event_scraper_enabled)
        loaded = load_external_event_settings_from_env()
        await load_secure_settings_into_runtime()
        after_enabled = bool(settings.external_event_scraper_enabled)
        if after_enabled:
            self.scraper = ExternalEventScraper()
            self.scraper.set_tracked_symbols(list(settings.symbols))
            if not self._task or self._task.done():
                self._running = True
                self._task = asyncio.create_task(self._loop())
        elif self._task and not self._task.done():
            self._running = False
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info(
            "external event runtime settings reloaded",
            was_enabled=before_enabled,
            enabled=after_enabled,
            source_count=loaded["source_count"],
        )
        return loaded

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
