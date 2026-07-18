"""
Data service — orchestrates data collection from all sources.
Provides a unified interface for the trading loop to get the latest data.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections import deque
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import structlog
from sqlalchemy import select

from config.settings import settings
from core.market_facts import (
    build_market_fact,
    build_market_source_consistency,
)
from core.market_facts import (
    verify_market_fact_path as verify_native_market_fact_path,
)
from core.safe_output import safe_error_text
from core.url_safety import normalize_external_http_url
from data_feed.feature_vector import FeatureVector, build_feature_vector
from data_feed.news_fetcher import NewsFetcher
from data_feed.okx_rest_client import OKXRestClient
from data_feed.okx_ticker_volume import okx_swap_volume_fields
from data_feed.okx_ws_client import OKXWebSocketClient
from data_feed.sentiment_scraper import SentimentScraper
from data_feed.technical_indicators import compute_all_indicators, extract_latest_features
from db.repositories.market_repo import MarketRepository
from db.session import get_session_ctx
from models.news import NewsArticle, SocialPost
from services.external_event_service import ExternalEventService
from services.trading_params import DEFAULT_TRADING_PARAMS

logger = structlog.get_logger(__name__)

ABNORMAL_WICK_LOOKBACK_HOURS = 72.0
ABNORMAL_WICK_MIN_RATIO = 1.50
INDICATOR_FEATURE_TIMEFRAME = "1h"
_MARKET_DATA_PARAMS = DEFAULT_TRADING_PARAMS.entry_market_data_quality
SHORT_RETURN_FEATURE_TIMEFRAME_PRIORITY = _MARKET_DATA_PARAMS.short_return_feature_timeframes
TREND_FEATURE_TIMEFRAME_PRIORITY = _MARKET_DATA_PARAMS.trend_feature_timeframes
SHORT_RETURN_FEATURE_KEYS = ("returns_1", "returns_5", "returns_20", "volatility_20")
MIN_INDICATOR_ROWS = _MARKET_DATA_PARAMS.min_indicator_rows
KLINE_CACHE_MAX_AGE_MULTIPLIER = _MARKET_DATA_PARAMS.kline_cache_max_age_multiplier
KLINE_CACHE_MIN_MAX_AGE_SECONDS = _MARKET_DATA_PARAMS.kline_cache_min_max_age_seconds
FEATURE_SNAPSHOT_TIMEOUT_SECONDS = _MARKET_DATA_PARAMS.feature_snapshot_timeout_seconds
KLINE_REMOTE_FETCH_TIMEOUT_SECONDS = _MARKET_DATA_PARAMS.kline_remote_fetch_timeout_seconds
INDICATOR_SNAPSHOT_CACHE_TTL_SECONDS = _MARKET_DATA_PARAMS.indicator_snapshot_cache_ttl_seconds
KLINE_BACKGROUND_REFRESH_MIN_INTERVAL_SECONDS = (
    _MARKET_DATA_PARAMS.kline_background_refresh_min_interval_seconds
)
KLINE_COVERAGE_REFRESH_INTERVAL_SECONDS = (
    _MARKET_DATA_PARAMS.kline_coverage_refresh_interval_seconds
)
KLINE_COVERAGE_REFRESH_BATCH_SIZE = _MARKET_DATA_PARAMS.kline_coverage_refresh_batch_size
KLINE_COVERAGE_REFRESH_SYMBOL_CAP = _MARKET_DATA_PARAMS.kline_coverage_refresh_symbol_cap
KLINE_COVERAGE_INITIAL_DELAY_SECONDS = _MARKET_DATA_PARAMS.kline_coverage_initial_delay_seconds
INDICATOR_REMOTE_REFRESH_CONCURRENCY = _MARKET_DATA_PARAMS.indicator_remote_refresh_concurrency
INDICATOR_SNAPSHOT_BUILD_CONCURRENCY = max(
    1,
    min(4, int(INDICATOR_REMOTE_REFRESH_CONCURRENCY)),
)
KLINE_PERSIST_CONCURRENCY = 2
TICKER_PERSIST_CONCURRENCY = max(
    1,
    min(4, max(int(settings.database_pool_size or 1) // 8, 1)),
)
DERIVATIVES_STALE_MAX_AGE_SECONDS = _MARKET_DATA_PARAMS.derivatives_stale_max_age_seconds
TIMEFRAME_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}
KLINE_PERSIST_TIMEFRAME_LIMITS: dict[str, int] = {
    "1m": 120,
    "5m": 120,
    "15m": 120,
    "1h": 100,
}
KLINE_FEATURE_SEQUENCE_LIMIT = 80
TICKER_PERSIST_THROTTLE_SECONDS = 30.0
TICKER_CACHE_MAX_AGE_SECONDS = max(
    10.0,
    float(_MARKET_DATA_PARAMS.indicator_snapshot_cache_ttl_seconds),
)
AVAILABLE_SYMBOLS_CACHE_TTL_SECONDS = 600.0
CANDIDATE_INDICATOR_PREWARM_TIMEOUT_SECONDS = max(
    float(FEATURE_SNAPSHOT_TIMEOUT_SECONDS) + 2.0,
    7.0,
)


class _PriorityBuildGate:
    """Bound indicator work while always serving queued market candidates first."""

    def __init__(self, limit: int) -> None:
        self._limit = max(1, int(limit))
        self._active = 0
        self._priority_waiters: deque[asyncio.Future[None]] = deque()
        self._normal_waiters: deque[asyncio.Future[None]] = deque()

    async def acquire(self, *, priority: bool) -> None:
        if (
            self._active < self._limit
            and not self._priority_waiters
            and not self._normal_waiters
        ):
            self._active += 1
            return

        future = asyncio.get_running_loop().create_future()
        waiters = self._priority_waiters if priority else self._normal_waiters
        waiters.append(future)
        try:
            await future
        except BaseException:
            if future.done() and not future.cancelled():
                self.release()
            else:
                future.cancel()
                self._wake_next()
            raise

    def release(self) -> None:
        if self._active <= 0:
            raise RuntimeError("indicator build gate released without acquisition")
        self._active -= 1
        self._wake_next()

    def _wake_next(self) -> None:
        while self._active < self._limit:
            future = self._next_waiter()
            if future is None:
                return
            self._active += 1
            future.set_result(None)

    def _next_waiter(self) -> asyncio.Future[None] | None:
        for waiters in (self._priority_waiters, self._normal_waiters):
            while waiters:
                future = waiters.popleft()
                if not future.done():
                    return future
        return None


class DataService:
    """Central data orchestration service.

    Responsibilities:
    1. Manage WebSocket connections for real-time market data
    2. Periodically fetch news and sentiment
    3. Build FeatureVectors for each symbol on demand
    4. Cache sentiment scores for quick access
    """

    def __init__(self) -> None:
        self.ws_client = OKXWebSocketClient()
        self.rest_client = OKXRestClient()
        self.news_fetcher = NewsFetcher()
        self.sentiment_scraper = SentimentScraper()
        self.external_event_service = ExternalEventService()

        # Caches
        self._sentiment_cache: dict[str, dict] = {}  # symbol -> {news_sent, social_sent, ...}
        self._headlines_cache: dict[str, list[str]] = {}
        self._news_items_cache: dict[str, list[dict[str, Any]]] = {}
        self._kline_cache: dict[str, pd.DataFrame] = {}  # symbol:tf -> DataFrame
        self._indicator_snapshot_cache: dict[str, dict[str, Any]] = {}
        self._indicator_snapshot_tasks: dict[str, asyncio.Task] = {}
        self._indicator_snapshot_priority_gate = _PriorityBuildGate(
            INDICATOR_SNAPSHOT_BUILD_CONCURRENCY
        )
        self._indicator_remote_refresh_semaphore = asyncio.Semaphore(
            max(1, int(INDICATOR_REMOTE_REFRESH_CONCURRENCY))
        )
        self._kline_fetch_tasks: dict[tuple[str, str], asyncio.Task] = {}
        self._kline_background_refresh_tasks: dict[tuple[str, str], asyncio.Task] = {}
        self._kline_persist_tasks: dict[tuple[str, str], asyncio.Task] = {}
        self._kline_refresh_scheduled_at: dict[tuple[str, str], datetime] = {}
        self._kline_persist_semaphore = asyncio.Semaphore(KLINE_PERSIST_CONCURRENCY)
        self._ticker_persist_semaphore = asyncio.Semaphore(TICKER_PERSIST_CONCURRENCY)
        self._kline_coverage_refresh_task: asyncio.Task | None = None
        self._kline_coverage_symbols: list[str] = []
        self._kline_coverage_index = 0
        self._derivatives_cache: dict[str, dict[str, Any]] = {}
        self._derivatives_refresh_tasks: dict[str, asyncio.Task] = {}
        self._last_sentiment_update: datetime | None = None
        self._sentiment_update_interval = 300  # seconds
        self._derivatives_update_interval = 20  # seconds
        self._sentiment_lock = asyncio.Lock()
        self._last_articles: list[dict] = []
        self._last_social_posts: list[dict] = []
        self._sentiment_refresh_task: asyncio.Task | None = None
        self._ticker_persisted_at: dict[str, datetime] = {}
        self._ticker_persist_inflight: set[str] = set()
        self._available_symbols_cache: list[dict[str, Any]] = []
        self._available_symbols_cache_updated_at: datetime | None = None
        self._available_symbols_refresh_task: asyncio.Task | None = None
        self._instrument_spec_cache: dict[str, dict[str, Any]] = {}
        self._instrument_spec_tasks: dict[str, asyncio.Task] = {}

        # Register ticker callback for real-time price updates
        self.ws_client.on_ticker(self._on_ticker_update)

    def _on_ticker_update(self, symbol: str, data: dict) -> None:
        """Callback invoked by WebSocket client on each ticker update."""
        try:
            normalized = self._normalize_symbols([symbol])[0]
            last_persisted = getattr(self, "_ticker_persisted_at", {}).get(normalized)
            now = datetime.now(UTC)
            inflight = getattr(self, "_ticker_persist_inflight", set())
            if normalized in inflight:
                return
            if (
                last_persisted
                and (now - last_persisted).total_seconds() < TICKER_PERSIST_THROTTLE_SECONDS
            ):
                return
            loop = asyncio.get_running_loop()
            inflight.add(normalized)
            loop.create_task(self._persist_ticker_snapshot(symbol, data))
        except RuntimeError:
            return
        except Exception as exc:
            logger.debug(
                "ticker callback persist scheduling failed",
                symbol=symbol,
                error=safe_error_text(exc),
            )

    async def _persist_ticker_snapshot(self, symbol: str, data: dict[str, Any]) -> None:
        try:
            normalized = self._normalize_symbols([symbol])[0]
            source = str(data.get("source") or "websocket")
            timestamp = data.get("timestamp")
            last_price = self._safe_float(data.get("last_price"), 0.0)
            volume_fields = okx_swap_volume_fields(data, last_price)
            volume_24h = self._safe_float(
                volume_fields.get("volume_24h_base") or data.get("volume_24h"),
                0.0,
            )
            payload = {
                "last_price": last_price,
                "bid": self._safe_float(data.get("bid"), 0.0),
                "ask": self._safe_float(data.get("ask"), 0.0),
                "high_24h": self._safe_float(data.get("high_24h"), 0.0),
                "low_24h": self._safe_float(data.get("low_24h"), 0.0),
                "volume_24h": volume_24h,
                "change_24h_pct": self._safe_float(data.get("change_24h_pct"), 0.0),
                "raw_data": json.dumps(
                    {
                        "symbol": normalized,
                        "timestamp": timestamp,
                        "source": source,
                        "inst_type": data.get("inst_type") or "SWAP",
                        "inst_id": data.get("inst_id"),
                        "source_endpoint": data.get("source_endpoint"),
                        "source_channel": data.get("source_channel"),
                        "source_sequence": data.get("source_sequence"),
                        "received_at": data.get("received_at"),
                        "contract_spec": data.get("contract_spec"),
                        "market_fact": data.get("market_fact"),
                        **volume_fields,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
            # WebSocket updates arrive for the full market at once. Keep payload
            # preparation outside the gate and bound only the short DB section so
            # cache persistence cannot consume the reconciliation/trading pool.
            async with self._ticker_persist_gate():
                async with get_session_ctx() as session:
                    repo = MarketRepository(session)
                    await repo.upsert_ticker(normalized, payload)
            self._ticker_persisted_at[normalized] = datetime.now(UTC)
        except Exception as exc:
            logger.debug(
                "persist ticker snapshot failed",
                symbol=symbol,
                error=safe_error_text(exc),
            )
        finally:
            getattr(self, "_ticker_persist_inflight", set()).discard(symbol)
            normalized = self._normalize_symbols([symbol])[0] if symbol else ""
            if normalized and normalized != symbol:
                getattr(self, "_ticker_persist_inflight", set()).discard(normalized)

    def _ticker_persist_gate(self) -> asyncio.Semaphore:
        gate = getattr(self, "_ticker_persist_semaphore", None)
        if isinstance(gate, asyncio.Semaphore):
            return gate
        gate = asyncio.Semaphore(TICKER_PERSIST_CONCURRENCY)
        self._ticker_persist_semaphore = gate
        return gate

    async def start(self) -> None:
        """Start all data feed connections."""
        # Fetch all available USDT pairs for auto mode WS subscription
        try:
            available = await self._refresh_available_symbols_cache()
            all_symbols = [s["symbol"] for s in available]
            self.ws_client._subscribe_symbols = all_symbols if all_symbols else settings.symbols
            self._kline_coverage_symbols = self._normalize_symbols(all_symbols)
            logger.info(
                "ws subscribing to all symbols", count=len(self.ws_client._subscribe_symbols)
            )
        except Exception as e:
            logger.warning(
                "fetch available symbols failed, using defaults",
                error=safe_error_text(e),
            )
            self.ws_client._subscribe_symbols = settings.symbols
            self._kline_coverage_symbols = self._normalize_symbols(settings.symbols)

        await self.ws_client.connect()
        asyncio.create_task(self.ws_client.listen())
        self._start_kline_coverage_refresh()
        await self.external_event_service.start_controller()
        logger.info("data service started")

    async def stop(self) -> None:
        """Stop all data feed connections."""
        await self._stop_kline_coverage_refresh()
        await self._stop_kline_persistence()
        await self.ws_client.close()
        await self.rest_client.close()
        await self.news_fetcher.close()
        await self.sentiment_scraper.close()
        await self.external_event_service.stop()
        logger.info("data service stopped")

    async def refresh_sentiment(self, symbols: list[str] | None = None) -> None:
        """Fetch latest news/social sentiment and cache it for requested symbols."""
        target_symbols = self._normalize_symbols(symbols or settings.symbols)
        now = datetime.now(UTC)
        if (
            self._last_sentiment_update
            and (now - self._last_sentiment_update).total_seconds()
            < self._sentiment_update_interval
            and all(symbol in self._sentiment_cache for symbol in target_symbols)
        ):
            return

        async with self._sentiment_lock:
            now = datetime.now(UTC)
            if (
                self._last_sentiment_update
                and (now - self._last_sentiment_update).total_seconds()
                < self._sentiment_update_interval
                and all(symbol in self._sentiment_cache for symbol in target_symbols)
            ):
                return

            symbols_to_refresh = self._normalize_symbols(list(settings.symbols) + target_symbols)
            if hasattr(self.news_fetcher, "set_tracked_symbols"):
                self.news_fetcher.set_tracked_symbols(symbols_to_refresh)
            if hasattr(self.sentiment_scraper, "set_tracked_symbols"):
                self.sentiment_scraper.set_tracked_symbols(symbols_to_refresh)

            fresh_enough = (
                self._last_sentiment_update
                and (now - self._last_sentiment_update).total_seconds()
                < self._sentiment_update_interval
                and (self._last_articles or self._last_social_posts)
            )
            if fresh_enough:
                self._build_sentiment_cache(
                    symbols_to_refresh, self._last_articles, self._last_social_posts
                )
                return

        try:
            # Fetch news
            articles = await self.news_fetcher.fetch_all()

            # Fetch public social/discussion posts.
            if hasattr(self.sentiment_scraper, "fetch_all_social"):
                social_posts = await self.sentiment_scraper.fetch_all_social()
            else:
                social_posts = await self.sentiment_scraper.fetch_all_reddit()

            self._last_articles = articles
            self._last_social_posts = social_posts
            self._build_sentiment_cache(symbols_to_refresh, articles, social_posts)
            await self._persist_sentiment_samples(articles, social_posts)

            self._last_sentiment_update = now
            logger.info("sentiment refreshed", symbols=len(self._sentiment_cache))

        except Exception as e:
            logger.error("sentiment refresh failed", error=safe_error_text(e))

    def _normalize_symbols(self, symbols: list[str]) -> list[str]:
        normalized: list[str] = []
        for symbol in symbols:
            value = str(symbol or "").strip()
            if not value:
                continue
            value = value.split(":")[0]
            if value.endswith("-SWAP"):
                value = value[:-5]
            if "/" not in value and "-" in value:
                parts = value.split("-")
                if len(parts) >= 2:
                    value = f"{parts[0]}/{parts[1]}"
            if "/" not in value:
                value = f"{value}/USDT"
            if value not in normalized:
                normalized.append(value)
        return normalized

    def _symbol_base(self, symbol: str) -> str:
        return self._normalize_symbols([symbol])[0].split("/")[0].upper()

    def _build_sentiment_cache(
        self,
        symbols: list[str],
        articles: list[dict],
        social_posts: list[dict],
    ) -> None:
        for symbol in self._normalize_symbols(symbols):
            base = self._symbol_base(symbol)
            symbol_articles = [
                a
                for a in articles
                if self._mentions_symbol(a, base, fields=("symbols_mentioned", "title", "summary"))
            ]
            symbol_posts = [
                p
                for p in social_posts
                if self._mentions_symbol(p, base, fields=("symbols", "title", "content"))
            ]

            news_items = [
                self._news_item_summary(a, base, direct_match=True) for a in symbol_articles
            ]
            headlines = [a.get("title", "") for a in symbol_articles if a.get("title")]
            if not headlines:
                headlines = [
                    f"[全市场] {a.get('title', '')}" for a in articles[:5] if a.get("title")
                ]
                news_items = [
                    self._news_item_summary(a, base, direct_match=False) for a in articles[:5]
                ]
            news_scores = [self._weighted_news_score(a) for a in symbol_articles]
            social_scores = [
                self._safe_score(
                    p.get("sentiment_score"), f"{p.get('title', '')} {p.get('content', '')}"
                )
                for p in symbol_posts
            ]
            news_items = sorted(
                [item for item in news_items if item.get("title")],
                key=lambda item: (
                    int(item.get("direct_match") is True),
                    int(item.get("impact_level") or 0),
                    float(item.get("source_weight") or 0.0),
                    str(item.get("published_at") or ""),
                ),
                reverse=True,
            )
            self._headlines_cache[symbol] = headlines[:20]
            self._news_items_cache[symbol] = news_items[:20]
            direct_items = [item for item in news_items if item.get("direct_match")]
            self._sentiment_cache[symbol] = {
                "news_sentiment": sum(news_scores) / len(news_scores) if news_scores else 0.0,
                "social_sentiment": (
                    sum(social_scores) / len(social_scores) if social_scores else 0.0
                ),
                "mention_count": len(symbol_posts),
                "article_count": len(symbol_articles) if symbol_articles else len(news_items),
                "direct_article_count": len(symbol_articles),
                "headline_count": len([a for a in symbol_articles if a.get("title")]),
                "sentiment_data_available": bool(symbol_articles or symbol_posts or news_items),
                "direct_sentiment_data_available": bool(symbol_articles or symbol_posts),
                "news_sources": (
                    sorted(
                        {str(a.get("source") or "") for a in symbol_articles if a.get("source")}
                    )[:5]
                    if symbol_articles
                    else sorted(
                        {str(item.get("source") or "") for item in news_items if item.get("source")}
                    )[:5]
                ),
                "direct_news_sources": sorted(
                    {str(a.get("source") or "") for a in symbol_articles if a.get("source")}
                )[:5],
                "market_news_item_count": len(
                    [item for item in news_items if not item.get("direct_match")]
                ),
                "direct_news_item_count": len(direct_items),
                "news_items": news_items[:20],
            }

    def _mentions_symbol(self, item: dict, base: str, fields: tuple[str, ...]) -> bool:
        for field in fields:
            value = item.get(field)
            if isinstance(value, list):
                if base in {str(v).upper() for v in value}:
                    return True
            elif value:
                pattern = rf"(?<![A-Z0-9]){re.escape(base)}(?![A-Z0-9])"
                if re.search(pattern, str(value or "").upper()):
                    return True
        return False

    def _weighted_news_score(self, item: dict) -> float:
        score = self._safe_score(
            item.get("sentiment_score"), f"{item.get('title', '')} {item.get('summary', '')}"
        )
        impact = max(min(self._safe_float(item.get("impact_level"), 1.0), 5.0), 1.0)
        source_weight = max(min(self._safe_float(item.get("source_weight"), 0.55), 1.0), 0.2)
        return max(min(score * (0.7 + impact * 0.08) * source_weight, 1.0), -1.0)

    def _news_item_summary(self, item: dict, base: str, *, direct_match: bool) -> dict[str, Any]:
        title = str(item.get("title") or "").strip()
        summary = str(item.get("summary") or "").strip()
        symbols = item.get("symbols_mentioned") or []
        if not isinstance(symbols, list):
            symbols = [symbols]
        impact = int(max(min(self._safe_float(item.get("impact_level"), 1.0), 5.0), 1.0))
        match_reason = (
            f"标题/摘要或来源标签直接提到 {base}"
            if direct_match
            else "全市场新闻，仅作宏观背景，不给该币种直接加分"
        )
        return {
            "source": str(item.get("source") or "unknown")[:80],
            "title": title[:240],
            "summary": summary[:360],
            "url": self._safe_external_url(item.get("url")),
            "published_at": str(item.get("published_at") or ""),
            "symbols": [str(s).upper() for s in symbols if s][:12],
            "sentiment_score": round(self._weighted_news_score(item), 4),
            "event_type": str(item.get("event_type") or "market_news"),
            "impact_level": impact,
            "source_weight": round(self._safe_float(item.get("source_weight"), 0.55), 3),
            "direct_match": direct_match,
            "match_reason": match_reason,
        }

    def _safe_external_url(self, value: Any) -> str:
        try:
            return normalize_external_http_url(
                str(value or ""),
                field_name="news source URL",
                max_length=500,
            )
        except ValueError:
            return ""

    def _safe_score(self, value: Any, text: str) -> float:
        try:
            if value is not None:
                return max(min(float(value), 1.0), -1.0)
        except (TypeError, ValueError):
            pass
        return self._lexicon_sentiment(text)

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _parse_datetime(self, value: Any) -> datetime | None:
        if isinstance(value, datetime):
            return value
        if not value:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                from email.utils import parsedate_to_datetime

                return parsedate_to_datetime(text)
            except Exception:
                return None

    async def _persist_sentiment_samples(
        self, articles: list[dict], social_posts: list[dict]
    ) -> None:
        try:
            async with get_session_ctx() as session:
                for item in articles[:500]:
                    title = str(item.get("title") or "").strip()
                    url = self._safe_external_url(item.get("url")) or f"local-news:{hash(title)}"
                    if not title:
                        continue
                    exists = await session.execute(
                        select(NewsArticle.id).where(NewsArticle.url == url).limit(1)
                    )
                    if exists.scalar_one_or_none() is not None:
                        continue
                    session.add(
                        NewsArticle(
                            source=str(item.get("source") or "unknown")[:50],
                            title=title,
                            summary=str(item.get("summary") or "")[:2000] or None,
                            url=url,
                            sentiment_score=self._safe_score(
                                item.get("sentiment_score"), f"{title} {item.get('summary', '')}"
                            ),
                            symbols_mentioned=item.get("symbols_mentioned") or [],
                            published_at=self._parse_datetime(item.get("published_at")),
                        )
                    )
                for item in social_posts[:500]:
                    post_id = str(item.get("post_id") or item.get("url") or hash(str(item)))[:100]
                    content = str(item.get("content") or item.get("title") or "").strip()
                    if not content:
                        continue
                    exists = await session.execute(
                        select(SocialPost.id).where(SocialPost.post_id == post_id).limit(1)
                    )
                    if exists.scalar_one_or_none() is not None:
                        continue
                    session.add(
                        SocialPost(
                            platform=str(item.get("platform") or "unknown")[:20],
                            post_id=post_id,
                            content=content[:2000],
                            sentiment_score=self._safe_score(item.get("sentiment_score"), content),
                            engagement_count=int(
                                self._safe_float(
                                    item.get("engagement_count") or item.get("score"), 0.0
                                )
                            ),
                            symbols=item.get("symbols") or [],
                            posted_at=self._parse_datetime(item.get("posted_at")),
                        )
                    )
                await session.flush()
        except Exception as exc:
            logger.debug("persist sentiment samples failed", error=safe_error_text(exc))

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

    async def get_feature_vector(
        self,
        symbol: str,
        *,
        wait_for_sentiment: bool = True,
        block_on_remote_ticker: bool = True,
        block_on_remote_indicators: bool = True,
        block_on_remote_derivatives: bool = True,
        allow_cached_indicator_build: bool = True,
        allow_indicator_background_refresh: bool = True,
        allow_derivatives_background_refresh: bool = True,
        prioritize_indicator_build: bool = False,
    ) -> FeatureVector:
        """Build a complete FeatureVector for a symbol from all available data."""
        sentiment_task = asyncio.create_task(
            self._ensure_sentiment_for_analysis(
                symbol,
                wait_for_initial=wait_for_sentiment,
            )
        )

        async def bounded_snapshot(name: str, coro) -> dict[str, Any]:
            timeout = max(float(FEATURE_SNAPSHOT_TIMEOUT_SECONDS), 0.5)
            try:
                result = await asyncio.wait_for(coro, timeout=timeout)
                return result if isinstance(result, dict) else {}
            except TimeoutError:
                logger.warning(
                    "feature snapshot source timed out",
                    symbol=symbol,
                    source=name,
                    timeout_seconds=timeout,
                )
                return {}
            except Exception as exc:
                logger.debug(
                    "feature snapshot source failed",
                    symbol=symbol,
                    source=name,
                    error=safe_error_text(exc),
                )
                return {}

        ticker_task = asyncio.create_task(
            bounded_snapshot(
                "ticker",
                self._get_feature_ticker_snapshot(
                    symbol,
                    block_on_remote=block_on_remote_ticker,
                ),
            )
        )
        indicators_task = asyncio.create_task(
            bounded_snapshot(
                "indicators",
                self._get_feature_indicator_snapshot(
                    symbol,
                    block_on_remote=block_on_remote_indicators,
                    allow_cached_build=allow_cached_indicator_build,
                    allow_background_refresh=allow_indicator_background_refresh,
                    prioritize_build=prioritize_indicator_build,
                ),
            )
        )
        derivatives_task = asyncio.create_task(
            bounded_snapshot(
                "derivatives",
                self._get_feature_derivatives_snapshot(
                    symbol,
                    block_on_remote=block_on_remote_derivatives,
                    allow_background_refresh=allow_derivatives_background_refresh,
                ),
            )
        )
        gather_results: tuple[Any, Any, Any, Any] = await asyncio.gather(
            ticker_task,
            indicators_task,
            derivatives_task,
            sentiment_task,
            return_exceptions=True,
        )
        ticker_result, indicators_result, derivatives_result, _sentiment_result = gather_results
        ticker = ticker_result if isinstance(ticker_result, dict) else {}
        indicators = indicators_result if isinstance(indicators_result, dict) else {}
        derivatives = derivatives_result if isinstance(derivatives_result, dict) else {}
        ticker = self._attach_market_source_consistency(symbol, ticker, derivatives)

        # Get sentiment
        sentiment = self._sentiment_cache.get(symbol, {})

        # Get headlines
        headlines = self._headlines_cache.get(symbol, [])
        news_items = self._news_items_cache.get(symbol, [])
        if news_items:
            sentiment = dict(sentiment)
            sentiment["news_items"] = news_items

        return build_feature_vector(
            symbol=symbol,
            ticker=ticker,
            indicators=indicators,
            sentiment_data=sentiment,
            headlines=headlines,
            derivatives=derivatives,
        )

    async def prewarm_indicator_snapshots(
        self,
        symbols: list[str],
        *,
        timeout_seconds: float = CANDIDATE_INDICATOR_PREWARM_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        """Populate real indicator snapshots for the final market shortlist."""

        normalized_symbols = list(dict.fromkeys(self._normalize_symbols(symbols)))
        if not normalized_symbols:
            return {
                "status": "skipped",
                "requested_count": 0,
                "available_count": 0,
                "unavailable_symbols": [],
            }

        per_symbol_timeout = max(float(timeout_seconds or 0.0), 0.5)

        async def prewarm(symbol: str) -> tuple[str, dict[str, Any]]:
            try:
                snapshot = await asyncio.wait_for(
                    self._get_indicator_snapshot(
                        symbol,
                        block_on_remote=True,
                        allow_cached_build=True,
                        allow_background_refresh=False,
                        prioritize_build=True,
                    ),
                    timeout=per_symbol_timeout,
                )
                return symbol, dict(snapshot or {})
            except TimeoutError:
                logger.warning(
                    "candidate indicator prewarm timed out",
                    symbol=symbol,
                    timeout_seconds=per_symbol_timeout,
                )
                return symbol, self._unavailable_indicator_snapshot(
                    "candidate_indicator_prewarm_timeout"
                )
            except Exception as exc:
                logger.warning(
                    "candidate indicator prewarm failed",
                    symbol=symbol,
                    error=safe_error_text(exc),
                )
                return symbol, self._unavailable_indicator_snapshot(
                    "candidate_indicator_prewarm_failed"
                )

        results = await asyncio.gather(*(prewarm(symbol) for symbol in normalized_symbols))
        unavailable = [
            symbol
            for symbol, snapshot in results
            if not bool(snapshot.get("indicator_snapshot_available"))
        ]
        return {
            "status": "ok" if not unavailable else "partial",
            "requested_count": len(normalized_symbols),
            "available_count": len(normalized_symbols) - len(unavailable),
            "unavailable_count": len(unavailable),
            "unavailable_symbols": unavailable,
            "timeout_seconds": round(per_symbol_timeout, 3),
            "priority_queue_enabled": True,
        }

    async def _ensure_sentiment_for_analysis(
        self,
        symbol: str,
        *,
        wait_for_initial: bool = True,
    ) -> None:
        """Refresh sentiment without blocking every trading decision."""
        now = datetime.now(UTC)
        cached = symbol in self._sentiment_cache
        stale = (
            self._last_sentiment_update is None
            or (now - self._last_sentiment_update).total_seconds()
            >= self._sentiment_update_interval
        )
        if cached:
            if stale:
                self._start_background_sentiment_refresh([symbol])
            return

        self._start_background_sentiment_refresh([symbol])
        task = self._sentiment_refresh_task
        if task is None:
            return
        timeout = max(float(settings.sentiment_blocking_timeout_seconds or 0.0), 0.0)
        if not wait_for_initial or timeout <= 0:
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except TimeoutError:
            logger.info(
                "sentiment refresh still running; continue with neutral cache", symbol=symbol
            )
        except Exception as e:
            logger.debug(
                "sentiment refresh unavailable for analysis",
                symbol=symbol,
                error=safe_error_text(e),
            )

    def _start_background_sentiment_refresh(self, symbols: list[str]) -> None:
        if self._sentiment_refresh_task and not self._sentiment_refresh_task.done():
            return
        self._sentiment_refresh_task = asyncio.create_task(self.refresh_sentiment(symbols))

    async def _get_ticker_snapshot(
        self,
        symbol: str,
        *,
        block_on_remote: bool = True,
    ) -> dict[str, Any]:
        normalized = self._normalize_symbols([symbol])[0]
        ticker = dict(
            self.ws_client.latest_tickers.get(symbol, {})
            or self.ws_client.latest_tickers.get(normalized, {})
            or {}
        )
        ticker_consistency_issue = self._ticker_snapshot_consistency_issue(ticker)
        ticker_fresh = bool(ticker and self._is_fresh_ticker_snapshot(ticker))
        ticker_source = str(ticker.get("source") or "websocket").lower()
        websocket_candidate = bool(ticker_fresh and ticker_source == "websocket")
        if ticker_fresh and not ticker_consistency_issue:
            bid = self._safe_float(ticker.get("bid"), 0.0)
            ask = self._safe_float(ticker.get("ask"), 0.0)
            mid = (bid + ask) / 2 if bid and ask else 0.0
            if mid and "spread_pct" not in ticker:
                ticker["spread_pct"] = (ask - bid) / mid * 100 if ask >= bid else 0.0
            ticker["source"] = ticker.get("source") or "websocket"
            ticker["inst_type"] = ticker.get("inst_type") or "SWAP"
            if not block_on_remote:
                return await self._attach_native_market_fact(
                    normalized,
                    ticker,
                    block_on_remote=False,
                )
        if ticker and ticker_consistency_issue:
            logger.warning(
                "ticker cache inconsistent; refreshing from OKX REST",
                symbol=normalized,
                issue=ticker_consistency_issue,
                age_seconds=self._ticker_snapshot_age_seconds(ticker),
            )
        elif ticker:
            logger.info(
                "ticker cache stale; refreshing from OKX REST",
                symbol=normalized,
                age_seconds=self._ticker_snapshot_age_seconds(ticker),
            )
        if not block_on_remote:
            if ticker:
                ticker["source"] = ticker.get("source") or "stale_websocket"
                ticker["stale"] = True
                ticker["ticker_remote_refresh_deferred"] = True
                ticker["age_seconds"] = self._ticker_snapshot_age_seconds(ticker)
                if ticker_consistency_issue:
                    ticker["market_data_quality_issue"] = ticker_consistency_issue
                return await self._attach_native_market_fact(
                    normalized,
                    ticker,
                    block_on_remote=False,
                )
            return {
                "ticker_snapshot_available": False,
                "ticker_remote_refresh_deferred": True,
            }
        try:
            raw_ticker = await self.rest_client.fetch_ticker(normalized)
            ticker_info = raw_ticker.get("info") or {}
            bid = self._safe_float(raw_ticker.get("bid") or ticker_info.get("bidPx"), 0.0)
            ask = self._safe_float(raw_ticker.get("ask") or ticker_info.get("askPx"), 0.0)
            mid = (bid + ask) / 2 if bid and ask else 0.0
            last_price = self._safe_float(raw_ticker.get("last"), 0.0)
            volume_fields = okx_swap_volume_fields(raw_ticker, last_price)
            snapshot = {
                "symbol": normalized,
                "inst_id": ticker_info.get("instId"),
                "last_price": last_price,
                "bid": bid,
                "ask": ask,
                "high_24h": raw_ticker.get("high", 0),
                "low_24h": raw_ticker.get("low", 0),
                "volume_24h": volume_fields["volume_24h_base"]
                or self._safe_float(raw_ticker.get("baseVolume"), 0.0),
                **volume_fields,
                "change_24h_pct": raw_ticker.get("percentage", 0),
                "spread_pct": ((ask - bid) / mid * 100) if mid and ask and bid else 0,
                "timestamp": self._ticker_timestamp_from_raw(raw_ticker),
                "source_timestamp_ms": self._ticker_timestamp_from_raw(raw_ticker),
                "source_sequence": ticker_info.get("seqId"),
                "received_at": datetime.now(UTC).isoformat(),
                "source": "rest",
                "source_endpoint": "okx_rest_market_ticker",
                "source_channel": "tickers",
                "inst_type": "SWAP",
                "info": ticker_info,
            }
            rest_snapshot = await self._attach_native_market_fact(
                normalized,
                snapshot,
                block_on_remote=True,
            )
            source_snapshots = [rest_snapshot]
            selected = rest_snapshot
            if websocket_candidate:
                websocket_snapshot = dict(ticker)
                if ticker_consistency_issue:
                    websocket_snapshot["market_data_quality_issue"] = (
                        ticker_consistency_issue
                    )
                websocket_snapshot = await self._attach_native_market_fact(
                    normalized,
                    websocket_snapshot,
                    block_on_remote=True,
                )
                if ticker_consistency_issue:
                    source_snapshots.append(websocket_snapshot)
                else:
                    selected = websocket_snapshot
                    source_snapshots = [websocket_snapshot, rest_snapshot]

            selected = dict(selected)
            selected["market_source_snapshots"] = [
                self._market_source_snapshot_payload(item) for item in source_snapshots
            ]
            selected["native_consistency_bars_1m"] = (
                await self._fetch_native_consistency_bars(normalized, source_snapshots)
            )
            if selected.get("source") == "rest":
                self.ws_client.latest_tickers[normalized] = dict(selected)
                self._on_ticker_update(normalized, selected)
            return selected
        except Exception as e:
            logger.debug("failed to fetch ticker", symbol=symbol, error=safe_error_text(e))
            if ticker:
                ticker["source"] = ticker.get("source") or "stale_websocket"
                ticker["stale"] = True
                ticker["age_seconds"] = self._ticker_snapshot_age_seconds(ticker)
                if ticker_consistency_issue:
                    ticker["market_data_quality_issue"] = ticker_consistency_issue
                return await self._attach_native_market_fact(
                    normalized,
                    ticker,
                    block_on_remote=False,
                )
            return {}

    @staticmethod
    def _market_source_snapshot_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
        payload = dict(snapshot)
        for key in (
            "market_fact",
            "market_source_snapshots",
            "native_consistency_bars_1m",
            "market_source_consistency",
        ):
            payload.pop(key, None)
        return payload

    async def _fetch_native_consistency_bars(
        self,
        symbol: str,
        source_snapshots: list[dict[str, Any]],
    ) -> list[Any]:
        timestamps = [
            self._ticker_timestamp_from_raw(snapshot)
            for snapshot in source_snapshots
            if self._ticker_timestamp_from_raw(snapshot) > 0
        ]
        elapsed_minutes = (
            max((max(timestamps) - min(timestamps)) // 60_000, 0) + 1
            if timestamps
            else 1
        )
        limit = min(elapsed_minutes + 2, 300)
        fetcher = getattr(self.rest_client, "fetch_ohlcv", None)
        if not callable(fetcher):
            return []
        try:
            rows = await fetcher(symbol, timeframe="1m", limit=limit)
        except Exception as exc:
            logger.warning(
                "OKX native 1m consistency path unavailable",
                symbol=symbol,
                error=safe_error_text(exc),
            )
            return []
        return rows if isinstance(rows, list) else []

    async def _attach_native_market_fact(
        self,
        symbol: str,
        snapshot: dict[str, Any],
        *,
        block_on_remote: bool,
    ) -> dict[str, Any]:
        enriched = dict(snapshot)
        spec = await self._get_instrument_spec(symbol, block_on_remote=block_on_remote)
        if spec:
            enriched["contract_spec"] = spec
            enriched["uly"] = spec.get("uly")
        info = enriched.get("info") if isinstance(enriched.get("info"), dict) else {}
        enriched["inst_id"] = (
            enriched.get("inst_id")
            or info.get("instId")
            or spec.get("instId")
        )
        enriched["received_at"] = enriched.get("received_at") or datetime.now(UTC).isoformat()
        enriched["market_fact"] = build_market_fact(
            symbol,
            enriched,
            contract_spec=spec,
            received_at=enriched["received_at"],
        )
        return enriched

    async def _get_instrument_spec(
        self,
        symbol: str,
        *,
        block_on_remote: bool,
    ) -> dict[str, Any]:
        normalized = self._normalize_symbols([symbol])[0]
        cache = getattr(self, "_instrument_spec_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            self._instrument_spec_cache = cache
        tasks = getattr(self, "_instrument_spec_tasks", None)
        if not isinstance(tasks, dict):
            tasks = {}
            self._instrument_spec_tasks = tasks
        cached = cache.get(normalized)
        if isinstance(cached, dict) and cached:
            return dict(cached)
        task = tasks.get(normalized)
        if task and not task.done():
            if not block_on_remote:
                return {}
            result = await asyncio.shield(task)
            return dict(result or {})
        if not block_on_remote:
            return {}

        async def fetch() -> dict[str, Any]:
            try:
                result = await self.rest_client.fetch_instrument_spec(normalized)
            except Exception as exc:
                logger.warning(
                    "OKX native instrument identity unavailable",
                    symbol=normalized,
                    error=safe_error_text(exc),
                )
                return {}
            value = dict(result) if isinstance(result, dict) else {}
            if value:
                cache[normalized] = value
            return value

        task = asyncio.create_task(fetch())
        tasks[normalized] = task
        try:
            return dict(await asyncio.shield(task) or {})
        finally:
            if tasks.get(normalized) is task:
                tasks.pop(normalized, None)

    async def get_latest_market_fact(self, symbol: str) -> dict[str, Any]:
        """Return one executable, native-instrument-bound OKX market fact."""

        ticker_result, derivatives_result = await asyncio.gather(
            self._get_ticker_snapshot(symbol, block_on_remote=True),
            self._get_derivatives_snapshot(symbol, block_on_remote=True),
            return_exceptions=True,
        )
        ticker = ticker_result if isinstance(ticker_result, dict) else {}
        derivatives = derivatives_result if isinstance(derivatives_result, dict) else {}
        ticker = self._attach_market_source_consistency(symbol, ticker, derivatives)
        merged = {**ticker, **derivatives}
        merged["stale"] = bool(
            ticker.get("stale") or derivatives.get("derivatives_snapshot_stale")
        )
        return build_market_fact(
            symbol,
            merged,
            contract_spec=ticker.get("contract_spec"),
            received_at=ticker.get("received_at"),
        )

    def _attach_market_source_consistency(
        self,
        symbol: str,
        ticker: dict[str, Any],
        derivatives: dict[str, Any],
    ) -> dict[str, Any]:
        if not ticker:
            return {}
        enriched = {**ticker, **derivatives}
        raw_sources = ticker.get("market_source_snapshots")
        source_snapshots = (
            [dict(item) for item in raw_sources if isinstance(item, dict)]
            if isinstance(raw_sources, list)
            else [dict(ticker)]
        )
        facts = [
            build_market_fact(
                symbol,
                {**snapshot, **derivatives},
                contract_spec=snapshot.get("contract_spec") or ticker.get("contract_spec"),
                received_at=snapshot.get("received_at"),
            )
            for snapshot in source_snapshots
        ]
        primary_fact = facts[0] if facts else {}
        enriched["market_source_consistency"] = build_market_source_consistency(
            primary_fact,
            facts[1:],
            orderbook_fact=derivatives.get("orderbook_fact"),
            mark_price_fact=derivatives.get("mark_price_fact"),
            index_price_fact=derivatives.get("index_price_fact"),
            bars=ticker.get("native_consistency_bars_1m") or [],
        )
        return enriched

    async def verify_market_fact_path(
        self,
        entry_fact: dict[str, Any],
        result_fact: dict[str, Any],
    ) -> dict[str, Any]:
        entry_ms = int(entry_fact.get("source_timestamp_ms") or 0)
        result_ms = int(result_fact.get("source_timestamp_ms") or 0)
        elapsed_minutes = (
            max((result_ms - entry_ms) // 60_000, 0) + 1
            if entry_ms > 0 and result_ms >= entry_ms
            else 1
        )
        # OKX's public candle API caps a single response; this is an exchange boundary,
        # while the requested window is derived from the actual fact timestamps.
        limit = min(elapsed_minutes + 2, 300)
        try:
            rows = await self.rest_client.fetch_ohlcv(
                str(entry_fact.get("symbol") or ""),
                timeframe="1m",
                limit=limit,
            )
        except Exception as exc:
            logger.warning(
                "OKX native 1m path unavailable for shadow outcome",
                symbol=entry_fact.get("symbol"),
                error=safe_error_text(exc),
            )
            rows = []
        return verify_native_market_fact_path(entry_fact, result_fact, rows)

    @staticmethod
    def _ticker_timestamp_from_raw(raw_ticker: dict[str, Any]) -> int:
        info = raw_ticker.get("info") if isinstance(raw_ticker, dict) else {}
        value = raw_ticker.get("timestamp") or (info or {}).get("ts")
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    def _ticker_snapshot_age_seconds(self, ticker: dict[str, Any]) -> float | None:
        timestamp = ticker.get("timestamp") if isinstance(ticker, dict) else None
        try:
            ts = float(timestamp or 0.0)
        except (TypeError, ValueError):
            return None
        if ts <= 0:
            return None
        if ts > 10_000_000_000:
            ts /= 1000.0
        return max(datetime.now(UTC).timestamp() - ts, 0.0)

    def _is_fresh_ticker_snapshot(self, ticker: dict[str, Any]) -> bool:
        price = self._safe_float(ticker.get("last_price"), 0.0)
        if price <= 0:
            return False
        age = self._ticker_snapshot_age_seconds(ticker)
        return age is not None and age <= TICKER_CACHE_MAX_AGE_SECONDS

    def _ticker_snapshot_consistency_issue(self, ticker: dict[str, Any]) -> str | None:
        if not ticker:
            return None
        last_price = self._safe_float(ticker.get("last_price"), 0.0)
        bid = self._safe_float(ticker.get("bid"), 0.0)
        ask = self._safe_float(ticker.get("ask"), 0.0)
        high_24h = self._safe_float(ticker.get("high_24h"), 0.0)
        low_24h = self._safe_float(ticker.get("low_24h"), 0.0)
        if bid > 0 and ask > 0:
            if bid > ask:
                return "crossed_bid_ask"
            # Last trade and top-of-book updates are not atomic. Ordinary quote
            # movement can leave the last trade outside the current spread.
        if last_price <= 0 or high_24h <= 0 or low_24h <= 0 or high_24h < low_24h:
            return None
        if low_24h <= last_price <= high_24h:
            return None
        return "last_price_outside_24h_range"

    def _start_kline_coverage_refresh(self) -> None:
        if self._kline_coverage_refresh_task and not self._kline_coverage_refresh_task.done():
            return
        self._kline_coverage_refresh_task = asyncio.create_task(self._kline_coverage_refresh_loop())

    async def _stop_kline_coverage_refresh(self) -> None:
        task = self._kline_coverage_refresh_task
        self._kline_coverage_refresh_task = None
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return

    async def _kline_coverage_refresh_loop(self) -> None:
        await asyncio.sleep(max(float(KLINE_COVERAGE_INITIAL_DELAY_SECONDS), 0.0))
        while True:
            try:
                await self.refresh_kline_coverage_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("kline coverage refresh failed", error=safe_error_text(exc))
            await asyncio.sleep(max(float(KLINE_COVERAGE_REFRESH_INTERVAL_SECONDS), 5.0))

    async def refresh_kline_coverage_once(self) -> dict[str, Any]:
        symbols = self._kline_coverage_target_symbols()
        if not symbols:
            return {"refreshed_symbols": [], "timeframes": []}
        batch_size = max(int(KLINE_COVERAGE_REFRESH_BATCH_SIZE), 1)
        start = self._kline_coverage_index % len(symbols)
        batch = [symbols[(start + offset) % len(symbols)] for offset in range(batch_size)]
        self._kline_coverage_index = (start + batch_size) % len(symbols)
        tasks = [
            self._fetch_and_persist_klines(symbol, timeframe, limit)
            for symbol in batch
            for timeframe, limit in KLINE_PERSIST_TIMEFRAME_LIMITS.items()
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return {
            "refreshed_symbols": batch,
            "timeframes": list(KLINE_PERSIST_TIMEFRAME_LIMITS),
        }

    def _kline_coverage_target_symbols(self) -> list[str]:
        configured = self._normalize_symbols(self._kline_coverage_symbols or [])
        subscribed = self._normalize_symbols(
            list(getattr(self.ws_client, "_subscribe_symbols", []) or [])
        )
        fallback = self._normalize_symbols(settings.symbols)
        merged = list(dict.fromkeys([*configured, *subscribed, *fallback]))
        target_limit = max(
            int(KLINE_COVERAGE_REFRESH_SYMBOL_CAP),
            len(configured),
            len(subscribed),
            1,
        )
        return merged[:target_limit]

    async def _get_feature_indicator_snapshot(
        self,
        symbol: str,
        *,
        block_on_remote: bool = True,
        allow_cached_build: bool = True,
        allow_background_refresh: bool = True,
        prioritize_build: bool = False,
    ) -> dict[str, Any]:
        getter = self._get_indicator_snapshot
        if prioritize_build:
            try:
                return await getter(
                    symbol,
                    block_on_remote=block_on_remote,
                    allow_cached_build=allow_cached_build,
                    allow_background_refresh=allow_background_refresh,
                    prioritize_build=True,
                )
            except TypeError as exc:
                if "prioritize_build" not in safe_error_text(exc):
                    raise
        try:
            return await getter(
                symbol,
                block_on_remote=block_on_remote,
                allow_cached_build=allow_cached_build,
                allow_background_refresh=allow_background_refresh,
            )
        except TypeError as exc:
            error_text = safe_error_text(exc)
            if "allow_background_refresh" in error_text:
                try:
                    return await getter(
                        symbol,
                        block_on_remote=block_on_remote,
                        allow_cached_build=allow_cached_build,
                    )
                except TypeError as nested_exc:
                    nested_text = safe_error_text(nested_exc)
                    if "allow_cached_build" in nested_text:
                        try:
                            return await getter(symbol, block_on_remote=block_on_remote)
                        except TypeError as final_exc:
                            if "block_on_remote" not in safe_error_text(final_exc):
                                raise
                            return await getter(symbol)
                    if "block_on_remote" not in nested_text:
                        raise
                    return await getter(symbol)
            if "allow_cached_build" in error_text:
                try:
                    return await getter(symbol, block_on_remote=block_on_remote)
                except TypeError as nested_exc:
                    if "block_on_remote" not in safe_error_text(nested_exc):
                        raise
                    return await getter(symbol)
            if "block_on_remote" not in error_text:
                raise
            return await getter(symbol)

    async def _get_feature_derivatives_snapshot(
        self,
        symbol: str,
        *,
        block_on_remote: bool = True,
        allow_background_refresh: bool = True,
    ) -> dict[str, Any]:
        getter = self._get_derivatives_snapshot
        try:
            return await getter(
                symbol,
                block_on_remote=block_on_remote,
                allow_background_refresh=allow_background_refresh,
            )
        except TypeError as exc:
            error_text = safe_error_text(exc)
            if "allow_background_refresh" in error_text:
                return await getter(symbol, block_on_remote=block_on_remote)
            if "block_on_remote" not in error_text:
                raise
            return await getter(symbol)

    async def _get_indicator_snapshot(
        self,
        symbol: str,
        *,
        block_on_remote: bool = True,
        allow_cached_build: bool = True,
        allow_background_refresh: bool = True,
        prioritize_build: bool = False,
    ) -> dict[str, Any]:
        normalized = self._normalize_symbols([symbol])[0]
        cache = self._indicator_snapshot_cache_map()
        cached = cache.get(normalized)
        if self._is_fresh_indicator_cache(cached):
            if allow_background_refresh:
                self._schedule_kline_background_refresh(normalized)
            return dict(cached.get("data") or {})
        if isinstance(cached, dict) and not block_on_remote:
            if allow_background_refresh:
                self._schedule_indicator_snapshot_refresh(normalized)
            data = dict(cached.get("data") or {})
            if data:
                data["indicator_snapshot_stale"] = True
                data["indicator_snapshot_refresh_in_background"] = bool(
                    allow_background_refresh
                )
                return data

        tasks = self._indicator_snapshot_task_map()
        existing_task = tasks.get(normalized)
        if existing_task and not existing_task.done():
            if block_on_remote:
                result = await asyncio.shield(existing_task)
                return dict(result or {})
            return {
                "indicator_snapshot_available": False,
                "indicator_snapshot_quality": "unavailable",
                "indicator_snapshot_reason": "indicator_snapshot_build_in_progress",
                "indicator_snapshot_refresh_in_background": True,
            }

        if not block_on_remote:
            if allow_cached_build:
                cached_features = await self._indicator_features_from_cached_klines(normalized)
                if cached_features:
                    self._store_indicator_snapshot_cache(normalized, cached_features)
                    if allow_background_refresh:
                        self._schedule_kline_background_refresh(normalized)
                    return cached_features
            if allow_background_refresh:
                self._schedule_indicator_snapshot_refresh(normalized)
            return self._unavailable_indicator_snapshot(
                "indicator_snapshot_cache_miss",
                refresh_in_background=bool(allow_background_refresh),
                background_refresh_deferred=not allow_background_refresh,
            )

        if prioritize_build:
            task = asyncio.create_task(
                self._build_indicator_snapshot(normalized, priority=True)
            )
        else:
            task = asyncio.create_task(self._build_indicator_snapshot(normalized))
        tasks[normalized] = task

        def cleanup(_task: asyncio.Task, refresh_symbol: str = normalized) -> None:
            if tasks.get(refresh_symbol) is _task:
                tasks.pop(refresh_symbol, None)

        task.add_done_callback(cleanup)
        try:
            result = await asyncio.shield(task)
            return dict(result or {})
        finally:
            if task.done() and tasks.get(normalized) is task:
                tasks.pop(normalized, None)

    def _indicator_snapshot_cache_map(self) -> dict[str, dict[str, Any]]:
        cache = getattr(self, "_indicator_snapshot_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            self._indicator_snapshot_cache = cache
        return cache

    def _indicator_snapshot_task_map(self) -> dict[str, asyncio.Task]:
        tasks = getattr(self, "_indicator_snapshot_tasks", None)
        if not isinstance(tasks, dict):
            tasks = {}
            self._indicator_snapshot_tasks = tasks
        return tasks

    def _indicator_remote_refresh_gate(self) -> asyncio.Semaphore:
        gate = getattr(self, "_indicator_remote_refresh_semaphore", None)
        if not isinstance(gate, asyncio.Semaphore):
            gate = asyncio.Semaphore(max(1, int(INDICATOR_REMOTE_REFRESH_CONCURRENCY)))
            self._indicator_remote_refresh_semaphore = gate
        return gate

    def _indicator_snapshot_build_gate(self) -> _PriorityBuildGate:
        gate = getattr(self, "_indicator_snapshot_priority_gate", None)
        if not isinstance(gate, _PriorityBuildGate):
            gate = _PriorityBuildGate(INDICATOR_SNAPSHOT_BUILD_CONCURRENCY)
            self._indicator_snapshot_priority_gate = gate
        return gate

    @staticmethod
    def _unavailable_indicator_snapshot(
        reason: str,
        *,
        refresh_in_background: bool = False,
        background_refresh_deferred: bool = False,
    ) -> dict[str, Any]:
        return {
            "indicator_snapshot_available": False,
            "indicator_snapshot_quality": "unavailable",
            "indicator_snapshot_reason": str(reason or "indicator_snapshot_unavailable"),
            "indicator_snapshot_refresh_in_background": bool(refresh_in_background),
            "indicator_snapshot_background_refresh_deferred": bool(
                background_refresh_deferred
            ),
        }

    def _kline_fetch_task_map(self) -> dict[tuple[str, str], asyncio.Task]:
        tasks = getattr(self, "_kline_fetch_tasks", None)
        if not isinstance(tasks, dict):
            tasks = {}
            self._kline_fetch_tasks = tasks
        return tasks

    def _kline_persist_gate(self) -> asyncio.Semaphore:
        gate = getattr(self, "_kline_persist_semaphore", None)
        if isinstance(gate, asyncio.Semaphore):
            return gate
        gate = asyncio.Semaphore(KLINE_PERSIST_CONCURRENCY)
        self._kline_persist_semaphore = gate
        return gate

    def _kline_persist_task_map(self) -> dict[tuple[str, str], asyncio.Task]:
        tasks = getattr(self, "_kline_persist_tasks", None)
        if not isinstance(tasks, dict):
            tasks = {}
            self._kline_persist_tasks = tasks
        return tasks

    def _schedule_kline_persistence(
        self,
        symbol: str,
        timeframe: str,
        klines: list,
    ) -> None:
        if not klines:
            return
        normalized = self._normalize_symbols([symbol])[0]
        key = (normalized, timeframe)
        tasks = self._kline_persist_task_map()
        existing = tasks.get(key)
        if existing and not existing.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._persist_klines(normalized, timeframe, klines))
        tasks[key] = task

        def cleanup(_task: asyncio.Task, persist_key: tuple[str, str] = key) -> None:
            if tasks.get(persist_key) is _task:
                tasks.pop(persist_key, None)

        task.add_done_callback(cleanup)

    async def _stop_kline_persistence(self) -> None:
        tasks = list(self._kline_persist_task_map().values())
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._kline_persist_task_map().clear()

    def _kline_refresh_schedule_map(self) -> dict[tuple[str, str], datetime]:
        scheduled = getattr(self, "_kline_refresh_scheduled_at", None)
        if not isinstance(scheduled, dict):
            scheduled = {}
            self._kline_refresh_scheduled_at = scheduled
        return scheduled

    def _kline_background_refresh_task_map(self) -> dict[tuple[str, str], asyncio.Task]:
        tasks = getattr(self, "_kline_background_refresh_tasks", None)
        if not isinstance(tasks, dict):
            tasks = {}
            self._kline_background_refresh_tasks = tasks
        return tasks

    def _derivatives_refresh_task_map(self) -> dict[str, asyncio.Task]:
        tasks = getattr(self, "_derivatives_refresh_tasks", None)
        if not isinstance(tasks, dict):
            tasks = {}
            self._derivatives_refresh_tasks = tasks
        return tasks

    def _is_fresh_indicator_cache(self, cached: Any) -> bool:
        if not isinstance(cached, dict):
            return False
        updated_at = cached.get("updated_at")
        if not isinstance(updated_at, datetime):
            return False
        return (
            datetime.now(UTC) - updated_at
        ).total_seconds() <= INDICATOR_SNAPSHOT_CACHE_TTL_SECONDS

    def _store_indicator_snapshot_cache(self, symbol: str, data: dict[str, Any]) -> None:
        if not data:
            return
        normalized = self._normalize_symbols([symbol])[0]
        self._indicator_snapshot_cache_map()[normalized] = {
            "updated_at": datetime.now(UTC),
            "data": dict(data),
        }

    def _schedule_kline_background_refresh(self, symbol: str) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        normalized = self._normalize_symbols([symbol])[0]
        scheduled = self._kline_refresh_schedule_map()
        background_tasks = self._kline_background_refresh_task_map()
        now = datetime.now(UTC)
        for timeframe, limit in KLINE_PERSIST_TIMEFRAME_LIMITS.items():
            key = (normalized, timeframe)
            existing = background_tasks.get(key)
            if existing and not existing.done():
                continue
            last_scheduled = scheduled.get(key)
            if (
                last_scheduled
                and (now - last_scheduled).total_seconds()
                < KLINE_BACKGROUND_REFRESH_MIN_INTERVAL_SECONDS
            ):
                continue
            scheduled[key] = now
            task = loop.create_task(
                self._refresh_klines_in_background(normalized, timeframe, limit)
            )
            background_tasks[key] = task

            def cleanup(_task: asyncio.Task, refresh_key: tuple[str, str] = key) -> None:
                if background_tasks.get(refresh_key) is _task:
                    background_tasks.pop(refresh_key, None)

            task.add_done_callback(cleanup)

    def _schedule_indicator_snapshot_refresh(self, symbol: str) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        normalized = self._normalize_symbols([symbol])[0]
        tasks = self._indicator_snapshot_task_map()
        existing = tasks.get(normalized)
        if existing and not existing.done():
            return
        task = loop.create_task(self._build_indicator_snapshot(normalized))
        tasks[normalized] = task

        def cleanup(_task: asyncio.Task, refresh_symbol: str = normalized) -> None:
            if tasks.get(refresh_symbol) is _task:
                tasks.pop(refresh_symbol, None)

        task.add_done_callback(cleanup)

    async def _refresh_klines_in_background(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
    ) -> None:
        try:
            await self._fetch_and_persist_klines_uncached(symbol, timeframe, limit)
        except Exception as exc:
            logger.debug(
                "background kline refresh failed",
                symbol=symbol,
                timeframe=timeframe,
                error=safe_error_text(exc),
            )

    async def _build_indicator_snapshot(
        self,
        symbol: str,
        *,
        priority: bool = False,
    ) -> dict[str, Any]:
        gate = self._indicator_snapshot_build_gate()
        await gate.acquire(priority=priority)
        try:
            return await self._build_indicator_snapshot_uncached(symbol)
        finally:
            gate.release()

    async def _build_indicator_snapshot_uncached(self, symbol: str) -> dict[str, Any]:
        try:
            cached_features = await self._indicator_features_from_cached_klines(symbol)
            if cached_features:
                self._store_indicator_snapshot_cache(symbol, cached_features)
                self._schedule_kline_background_refresh(symbol)
                return cached_features

            gate = self._indicator_remote_refresh_gate()
            async with gate:
                kline_results = await asyncio.gather(
                    *(
                        self._fetch_indicator_klines(symbol, timeframe, limit)
                        for timeframe, limit in KLINE_PERSIST_TIMEFRAME_LIMITS.items()
                    ),
                    return_exceptions=True,
                )
            klines_by_timeframe = {
                timeframe: rows
                for item in kline_results
                if isinstance(item, tuple)
                for timeframe, rows in (item,)
                if rows
            }
            features = self._indicator_features_from_timeframes(klines_by_timeframe)
            if features:
                self._store_indicator_snapshot_cache(symbol, features)
            return features
        except Exception as e:
            logger.debug("failed to compute indicators", symbol=symbol, error=safe_error_text(e))
            return {}

    async def _get_feature_ticker_snapshot(
        self,
        symbol: str,
        *,
        block_on_remote: bool = True,
    ) -> dict[str, Any]:
        getter = self._get_ticker_snapshot
        try:
            return await getter(symbol, block_on_remote=block_on_remote)
        except TypeError as exc:
            if "block_on_remote" not in safe_error_text(exc):
                raise
            return await getter(symbol)

    async def _indicator_features_from_cached_klines(self, symbol: str) -> dict[str, Any]:
        cached_klines_by_timeframe = await self._load_recent_cached_klines_for_timeframes(
            symbol,
            KLINE_PERSIST_TIMEFRAME_LIMITS,
        )
        return self._indicator_features_from_timeframes(cached_klines_by_timeframe)

    def _indicator_features_from_timeframes(
        self,
        klines_by_timeframe: dict[str, list],
    ) -> dict[str, Any]:
        if not klines_by_timeframe:
            return {}

        trend_timeframe, trend_features, trend_df = self._select_indicator_features(
            klines_by_timeframe,
            TREND_FEATURE_TIMEFRAME_PRIORITY,
        )
        short_timeframe, short_features, short_df = self._select_indicator_features(
            klines_by_timeframe,
            SHORT_RETURN_FEATURE_TIMEFRAME_PRIORITY,
        )
        if not trend_features and not short_features:
            return {}

        features: dict[str, Any] = dict(trend_features or short_features)
        if short_features:
            for key in SHORT_RETURN_FEATURE_KEYS:
                if key in short_features:
                    features[key] = short_features[key]
            for key in ("close", "volume"):
                if key in short_features:
                    features[key] = short_features[key]
            if "volume_ratio" in short_features:
                features["entry_activity_volume_ratio"] = short_features["volume_ratio"]
                features["entry_activity_volume_timeframe"] = short_timeframe
            features["short_returns_timeframe"] = short_timeframe
        if trend_features:
            features["technical_indicator_timeframe"] = trend_timeframe
            features["volume_ratio_timeframe"] = trend_timeframe
        elif short_features:
            features["volume_ratio_timeframe"] = short_timeframe
        features["indicator_snapshot_available"] = True
        features["indicator_snapshot_quality"] = "full"
        features["indicator_snapshot_reason"] = "recent_complete_klines"
        sequence_df = short_df if not short_df.empty else trend_df
        sequence_timeframe = short_timeframe or trend_timeframe
        features.update(self._kline_sequence_snapshot(sequence_df, sequence_timeframe))
        anomaly_df = trend_df if not trend_df.empty else short_df
        features.update(self._kline_anomaly_snapshot(anomaly_df))
        return features

    def _kline_sequence_snapshot(self, df: pd.DataFrame, timeframe: str) -> dict[str, Any]:
        if df.empty:
            return {}
        close_sequence = self._numeric_sequence_from_series(
            df.get("close"),
            limit=KLINE_FEATURE_SEQUENCE_LIMIT,
        )
        if not close_sequence:
            return {}
        volume_sequence = self._numeric_sequence_from_series(
            df.get("volume"),
            limit=KLINE_FEATURE_SEQUENCE_LIMIT,
        )
        return {
            "close_sequence": close_sequence,
            "volume_sequence": volume_sequence,
            "sequence_timeframe": timeframe,
            "sequence_length": len(close_sequence),
        }

    @staticmethod
    def _numeric_sequence_from_series(value: Any, *, limit: int) -> list[float]:
        if value is None:
            return []
        try:
            raw_values = list(value.tail(limit))
        except AttributeError:
            try:
                raw_values = list(value)[-limit:]
            except TypeError:
                return []
        out: list[float] = []
        for item in raw_values:
            try:
                number = float(item)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                out.append(number)
        return out

    def _select_indicator_features(
        self,
        klines_by_timeframe: dict[str, list],
        priority: tuple[str, ...],
    ) -> tuple[str, dict[str, Any], pd.DataFrame]:
        for timeframe in priority:
            klines = klines_by_timeframe.get(timeframe) or []
            features, df = self._features_from_klines(klines, timeframe)
            if features:
                return timeframe, features, df
        return "", {}, pd.DataFrame()

    def _features_from_klines(
        self,
        klines: list,
        timeframe: str = "",
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        if len(klines) < MIN_INDICATOR_ROWS:
            return {}, pd.DataFrame()
        df = pd.DataFrame(
            klines,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.dropna(subset=["timestamp", "close"]).drop_duplicates("timestamp")
        df = df.sort_values("timestamp").tail(max(len(klines), MIN_INDICATOR_ROWS))
        df = self._drop_incomplete_latest_kline(df, timeframe)
        if len(df) < MIN_INDICATOR_ROWS:
            return {}, df
        computed = compute_all_indicators(df)
        return extract_latest_features(computed), computed

    @staticmethod
    def _drop_incomplete_latest_kline(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        seconds = TIMEFRAME_SECONDS.get(str(timeframe or ""))
        if not seconds or df.empty:
            return df
        latest = df.iloc[-1].get("timestamp")
        if not isinstance(latest, pd.Timestamp):
            return df
        if latest.tzinfo is None:
            latest = latest.tz_localize("UTC")
        now = pd.Timestamp.now(tz="UTC")
        if latest + pd.Timedelta(seconds=seconds) <= now:
            return df
        if len(df) <= 1:
            return df.iloc[0:0].copy()
        return df.iloc[:-1].copy()

    async def _fetch_and_persist_klines(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
    ) -> tuple[str, list]:
        """Fetch one timeframe and persist it for local model training."""
        normalized = self._normalize_symbols([symbol])[0]
        key = (normalized, timeframe)
        tasks = self._kline_fetch_task_map()
        existing_task = tasks.get(key)
        if existing_task and not existing_task.done():
            rows = await existing_task
            return timeframe, rows if isinstance(rows, list) else []

        task = asyncio.create_task(
            self._fetch_and_persist_klines_uncached(normalized, timeframe, limit)
        )
        tasks[key] = task
        try:
            rows = await task
            return timeframe, rows if isinstance(rows, list) else []
        finally:
            if tasks.get(key) is task:
                tasks.pop(key, None)

    async def _fetch_and_persist_klines_uncached(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
    ) -> list:
        return await self._fetch_remote_klines_with_cache(
            symbol,
            timeframe,
            limit,
            persist_before_return=True,
        )

    async def _fetch_indicator_klines(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
    ) -> tuple[str, list]:
        rows = await self._fetch_remote_klines_with_cache(
            symbol,
            timeframe,
            limit,
            persist_before_return=False,
        )
        return timeframe, rows

    async def _fetch_remote_klines_with_cache(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        *,
        persist_before_return: bool,
    ) -> list:
        try:
            timeout = max(float(KLINE_REMOTE_FETCH_TIMEOUT_SECONDS), 0.5)
            klines = await asyncio.wait_for(
                self.rest_client.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    limit=limit,
                ),
                timeout=timeout,
            )
            if klines:
                if persist_before_return:
                    await self._persist_klines(symbol, timeframe, klines)
                else:
                    self._schedule_kline_persistence(symbol, timeframe, klines)
                return klines
            cached = await self._load_recent_cached_klines(symbol, timeframe, limit)
            return cached
        except Exception as exc:
            logger.debug(
                "failed to fetch kline timeframe",
                symbol=symbol,
                timeframe=timeframe,
                error=safe_error_text(exc),
            )
            cached = await self._load_recent_cached_klines(symbol, timeframe, limit)
            return cached

    async def _load_recent_cached_klines(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
    ) -> list[list[float]]:
        grouped = await self._load_recent_cached_klines_for_timeframes(
            symbol,
            {timeframe: limit},
        )
        return list(grouped.get(timeframe) or [])

    async def _load_recent_cached_klines_for_timeframes(
        self,
        symbol: str,
        timeframe_limits: dict[str, int],
    ) -> dict[str, list[list[float]]]:
        try:
            normalized = self._normalize_symbols([symbol])[0]
            async with get_session_ctx() as session:
                repo = MarketRepository(session)
                grouped_rows = await repo.get_klines_for_timeframes(
                    normalized,
                    timeframe_limits,
                )
            now = datetime.now(UTC)
            grouped: dict[str, list[list[float]]] = {}
            for timeframe, rows in grouped_rows.items():
                if not rows:
                    continue
                latest = rows[-1].open_time
                if latest.tzinfo is None:
                    latest = latest.replace(tzinfo=UTC)
                max_age_seconds = max(
                    TIMEFRAME_SECONDS.get(timeframe, 60) * KLINE_CACHE_MAX_AGE_MULTIPLIER,
                    KLINE_CACHE_MIN_MAX_AGE_SECONDS,
                )
                if (now - latest).total_seconds() > max_age_seconds:
                    continue
                grouped[timeframe] = [
                    [
                        row.open_time.timestamp() * 1000.0,
                        row.open,
                        row.high,
                        row.low,
                        row.close,
                        row.volume,
                    ]
                    for row in rows
                ]
            return grouped
        except Exception as exc:
            logger.debug(
                "load cached kline timeframes failed",
                symbol=symbol,
                timeframes=list(timeframe_limits),
                error=safe_error_text(exc),
            )
            return {}

    def _kline_anomaly_snapshot(self, df: pd.DataFrame) -> dict[str, float]:
        """Detect repeat extreme wicks that can make stop losses fill far away."""
        try:
            if df.empty:
                return {}
            now = pd.Timestamp.now(tz="UTC")
            lookback_start = now - pd.Timedelta(hours=ABNORMAL_WICK_LOOKBACK_HOURS)
            recent = df[df["timestamp"] >= lookback_start].copy()
            if recent.empty:
                recent = df.tail(int(ABNORMAL_WICK_LOOKBACK_HOURS)).copy()
            max_body_top = recent[["open", "close"]].max(axis=1).astype(float).clip(lower=1e-12)
            min_body_bottom = recent[["open", "close"]].min(axis=1).astype(float).clip(lower=1e-12)
            high = recent["high"].astype(float)
            low = recent["low"].astype(float).clip(lower=1e-12)
            upper_ratio = high / max_body_top
            lower_ratio = min_body_bottom / low
            max_ratio = pd.concat([upper_ratio, lower_ratio], axis=1).max(axis=1)
            abnormal = recent[max_ratio >= ABNORMAL_WICK_MIN_RATIO]
            if abnormal.empty:
                return {
                    "abnormal_wick_count_72h": 0,
                    "abnormal_wick_max_pct": 0.0,
                    "abnormal_wick_recent_hours": 9999.0,
                }
            abnormal_ratios = max_ratio.loc[abnormal.index]
            latest_ts = abnormal["timestamp"].max()
            recent_hours = (
                (now - latest_ts).total_seconds() / 3600.0 if latest_ts is not None else 9999.0
            )
            if not math.isfinite(recent_hours):
                recent_hours = 9999.0
            return {
                "abnormal_wick_count_72h": int(len(abnormal)),
                "abnormal_wick_max_pct": round(
                    max(float(abnormal_ratios.max() - 1.0), 0.0) * 100.0, 6
                ),
                "abnormal_wick_recent_hours": round(max(recent_hours, 0.0), 4),
            }
        except Exception as exc:
            logger.debug(
                "failed to compute kline anomaly snapshot",
                error=safe_error_text(exc),
            )
            return {}

    async def _persist_klines(self, symbol: str, timeframe: str, klines: list) -> None:
        # Kline persistence is cache maintenance.  It must never fan out enough
        # long transactions to consume the trading and reconciliation connection pool.
        async with self._kline_persist_gate():
            try:
                async with get_session_ctx() as session:
                    repo = MarketRepository(session)
                    normalized = self._normalize_symbols([symbol])[0]
                    rows_to_persist: list[tuple[datetime, dict[str, float]]] = []
                    for row in klines[-300:]:
                        try:
                            ts, open_, high, low, close, volume = row[:6]
                            open_time = datetime.fromtimestamp(float(ts) / 1000.0, tz=UTC)
                            rows_to_persist.append(
                                (
                                    open_time,
                                    {
                                    "open": self._safe_float(open_),
                                    "high": self._safe_float(high),
                                    "low": self._safe_float(low),
                                    "close": self._safe_float(close),
                                    "volume": self._safe_float(volume),
                                    },
                                )
                            )
                        except Exception as exc:
                            logger.debug(
                                "skip invalid kline row",
                                symbol=normalized,
                                timeframe=timeframe,
                                error=safe_error_text(exc),
                            )
                            continue
                    await repo.upsert_klines_bulk(normalized, timeframe, rows_to_persist)
                    await repo.clean_old_klines(normalized, timeframe, keep=2000)
            except Exception as exc:
                logger.debug("persist klines failed", symbol=symbol, error=safe_error_text(exc))

    async def _get_derivatives_snapshot(
        self,
        symbol: str,
        *,
        block_on_remote: bool = True,
        allow_background_refresh: bool = True,
    ) -> dict[str, Any]:
        now = datetime.now(UTC)
        normalized = self._normalize_symbols([symbol])[0]
        cache = getattr(self, "_derivatives_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            self._derivatives_cache = cache
        cached = cache.get(normalized)
        if cached:
            updated_at = cached.get("updated_at")
            if (
                isinstance(updated_at, datetime)
                and (now - updated_at).total_seconds() < self._derivatives_update_interval
            ):
                return dict(cached.get("data") or {})
            if (
                isinstance(updated_at, datetime)
                and (now - updated_at).total_seconds() <= DERIVATIVES_STALE_MAX_AGE_SECONDS
                and not block_on_remote
            ):
                if allow_background_refresh:
                    self._schedule_derivatives_background_refresh(normalized)
                data = dict(cached.get("data") or {})
                if data:
                    data["derivatives_snapshot_stale"] = True
                    data["derivatives_refresh_in_background"] = True
                return data
            if not block_on_remote:
                if allow_background_refresh:
                    self._schedule_derivatives_background_refresh(normalized)
                data = dict(cached.get("data") or {})
                if data:
                    data["derivatives_snapshot_stale"] = True
                    data["derivatives_refresh_in_background"] = True
                    return data

        tasks = self._derivatives_refresh_task_map()
        existing_task = tasks.get(normalized)
        if existing_task and not existing_task.done():
            if block_on_remote:
                result = await asyncio.shield(existing_task)
                return dict(result or {})
            return {"derivatives_refresh_in_background": True}
        if not block_on_remote:
            if allow_background_refresh:
                self._schedule_derivatives_background_refresh(normalized)
                return {"derivatives_refresh_in_background": True}
            return {}
        task = asyncio.create_task(self._refresh_derivatives_snapshot(normalized))
        tasks[normalized] = task
        try:
            result = await asyncio.shield(task)
            return dict(result or {})
        finally:
            if tasks.get(normalized) is task:
                tasks.pop(normalized, None)

    def _schedule_derivatives_background_refresh(self, symbol: str) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        normalized = self._normalize_symbols([symbol])[0]
        tasks = self._derivatives_refresh_task_map()
        existing_task = tasks.get(normalized)
        if existing_task and not existing_task.done():
            return
        task = loop.create_task(self._refresh_derivatives_snapshot(normalized))
        tasks[normalized] = task

        def cleanup(_task: asyncio.Task) -> None:
            if tasks.get(normalized) is _task:
                tasks.pop(normalized, None)

        task.add_done_callback(cleanup)

    async def _refresh_derivatives_snapshot(self, symbol: str) -> dict[str, Any]:
        normalized = self._normalize_symbols([symbol])[0]
        now = datetime.now(UTC)
        tasks = self._derivatives_refresh_task_map()
        current_task = asyncio.current_task()
        existing_task = tasks.get(normalized)
        if existing_task and existing_task is not current_task and not existing_task.done():
            result = await existing_task
            return dict(result or {})
        try:
            contract_spec = await self._get_instrument_spec(
                normalized,
                block_on_remote=True,
            )
            data = await asyncio.wait_for(
                self.rest_client.fetch_derivatives_snapshot(
                    normalized,
                    contract_spec=contract_spec,
                ),
                timeout=max(float(FEATURE_SNAPSHOT_TIMEOUT_SECONDS), 0.5),
            )
        except Exception as e:
            logger.debug(
                "failed to fetch derivatives snapshot",
                symbol=normalized,
                error=safe_error_text(e),
            )
            data = {}

        self._derivatives_cache[normalized] = {
            "updated_at": now,
            "data": dict(data or {}),
        }
        return dict(data or {})

    async def get_all_feature_vectors(self) -> dict[str, FeatureVector]:
        """Get FeatureVectors for all configured symbols."""
        tasks = [self.get_feature_vector(s) for s in settings.symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        fvs = {}
        for symbol, result in zip(settings.symbols, results, strict=False):
            if isinstance(result, FeatureVector):
                fvs[symbol] = result
            else:
                logger.error(
                    "feature vector build failed",
                    symbol=symbol,
                    error=safe_error_text(result),
                )
        return fvs

    async def get_available_symbols(self) -> list[dict[str, Any]]:
        """Return all available USDT trading pairs from OKX."""
        cached = self._available_symbols_cache_if_usable()
        if cached:
            return cached
        stale_cached = self._available_symbols_cache_if_any()
        if stale_cached:
            self._schedule_available_symbols_refresh()
            return stale_cached
        return await self._refresh_available_symbols_cache()

    def _available_symbols_cache_if_usable(self) -> list[dict[str, Any]]:
        cached = getattr(self, "_available_symbols_cache", None)
        updated_at = getattr(self, "_available_symbols_cache_updated_at", None)
        if not cached or not isinstance(updated_at, datetime):
            return []
        age_seconds = (datetime.now(UTC) - updated_at).total_seconds()
        if age_seconds > AVAILABLE_SYMBOLS_CACHE_TTL_SECONDS:
            return []
        return [dict(item) for item in cached if isinstance(item, dict)]

    async def _refresh_available_symbols_cache(self) -> list[dict[str, Any]]:
        existing = getattr(self, "_available_symbols_refresh_task", None)
        current_task = asyncio.current_task()
        if existing is not None and existing is not current_task and not existing.done():
            result = await existing
            return [dict(item) for item in result if isinstance(item, dict)]

        task = asyncio.create_task(self.rest_client.get_available_symbols())
        self._available_symbols_refresh_task = task
        try:
            symbols = await task
            normalized = [dict(item) for item in symbols if isinstance(item, dict)]
            if normalized:
                self._available_symbols_cache = normalized
                self._available_symbols_cache_updated_at = datetime.now(UTC)
            return normalized or self._available_symbols_cache_if_any()
        finally:
            if getattr(self, "_available_symbols_refresh_task", None) is task:
                self._available_symbols_refresh_task = None

    def _available_symbols_cache_if_any(self) -> list[dict[str, Any]]:
        cached = getattr(self, "_available_symbols_cache", None)
        if not cached:
            return []
        return [dict(item) for item in cached if isinstance(item, dict)]

    def _schedule_available_symbols_refresh(self) -> None:
        existing = getattr(self, "_available_symbols_refresh_task", None)
        if existing is not None and not existing.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._available_symbols_refresh_task = loop.create_task(
            self._refresh_available_symbols_cache()
        )

    def get_market_state(self) -> dict[str, Any]:
        """Get a snapshot of current market state for the dashboard."""
        return {
            "tickers": {
                sym: {
                    "price": d.get("last_price", 0),
                    "change_24h": d.get("change_24h_pct", 0),
                    "volume_24h": d.get("volume_24h", 0),
                    "volume_24h_contracts": d.get("volume_24h_contracts", 0),
                    "volume_24h_base": d.get("volume_24h_base", 0),
                    "volume_24h_quote": d.get("volume_24h_quote", 0),
                    "notional_24h_usdt": d.get("notional_24h_usdt", 0),
                    "volume_24h_source": d.get("volume_24h_source", ""),
                    "bid": d.get("bid", 0),
                    "ask": d.get("ask", 0),
                }
                for sym, d in self.ws_client.latest_tickers.items()
            },
            "ws_stats": self.ws_client.get_stats(),
        }
