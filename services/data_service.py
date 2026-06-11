"""
Data service — orchestrates data collection from all sources.
Provides a unified interface for the trading loop to get the latest data.
"""

from __future__ import annotations

import asyncio
import math
import re
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import structlog
from sqlalchemy import select

from config.settings import settings
from core.safe_output import safe_error_text
from core.url_safety import normalize_external_http_url
from data_feed.feature_vector import FeatureVector, build_feature_vector
from data_feed.news_fetcher import NewsFetcher
from data_feed.okx_rest_client import OKXRestClient
from data_feed.okx_ws_client import OKXWebSocketClient
from data_feed.sentiment_scraper import SentimentScraper
from data_feed.technical_indicators import compute_all_indicators, extract_latest_features
from db.repositories.market_repo import MarketRepository
from db.session import get_session_ctx
from models.news import NewsArticle, SocialPost

logger = structlog.get_logger(__name__)

ABNORMAL_WICK_LOOKBACK_HOURS = 72.0
ABNORMAL_WICK_MIN_RATIO = 1.50


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

        # Caches
        self._sentiment_cache: dict[str, dict] = {}  # symbol -> {news_sent, social_sent, ...}
        self._headlines_cache: dict[str, list[str]] = {}
        self._news_items_cache: dict[str, list[dict[str, Any]]] = {}
        self._kline_cache: dict[str, pd.DataFrame] = {}  # symbol:tf -> DataFrame
        self._derivatives_cache: dict[str, dict[str, Any]] = {}
        self._last_sentiment_update: datetime | None = None
        self._sentiment_update_interval = 300  # seconds
        self._derivatives_update_interval = 20  # seconds
        self._sentiment_lock = asyncio.Lock()
        self._last_articles: list[dict] = []
        self._last_social_posts: list[dict] = []
        self._sentiment_refresh_task: asyncio.Task | None = None

        # Register ticker callback for real-time price updates
        self.ws_client.on_ticker(self._on_ticker_update)

    def _on_ticker_update(self, symbol: str, data: dict) -> None:
        """Callback invoked by WebSocket client on each ticker update."""
        # Update price in all dependent services
        pass  # Data is stored in ws_client.latest_tickers, accessed on demand

    async def start(self) -> None:
        """Start all data feed connections."""
        # Fetch all available USDT pairs for auto mode WS subscription
        try:
            available = await self.rest_client.get_available_symbols()
            all_symbols = [s["symbol"] for s in available]
            self.ws_client._subscribe_symbols = all_symbols if all_symbols else settings.symbols
            logger.info(
                "ws subscribing to all symbols", count=len(self.ws_client._subscribe_symbols)
            )
        except Exception as e:
            logger.warning(
                "fetch available symbols failed, using defaults",
                error=safe_error_text(e),
            )
            self.ws_client._subscribe_symbols = settings.symbols

        await self.ws_client.connect()
        asyncio.create_task(self.ws_client.listen())
        logger.info("data service started")

    async def stop(self) -> None:
        """Stop all data feed connections."""
        await self.ws_client.close()
        await self.rest_client.close()
        await self.news_fetcher.close()
        await self.sentiment_scraper.close()
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

            # Fetch social posts
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

    async def get_feature_vector(self, symbol: str) -> FeatureVector:
        """Build a complete FeatureVector for a symbol from all available data."""
        await self._ensure_sentiment_for_analysis(symbol)

        ticker_task = asyncio.create_task(self._get_ticker_snapshot(symbol))
        indicators_task = asyncio.create_task(self._get_indicator_snapshot(symbol))
        derivatives_task = asyncio.create_task(self._get_derivatives_snapshot(symbol))
        gather_results: tuple[Any, Any, Any] = await asyncio.gather(
            ticker_task,
            indicators_task,
            derivatives_task,
            return_exceptions=True,
        )
        ticker_result, indicators_result, derivatives_result = gather_results
        ticker = ticker_result if isinstance(ticker_result, dict) else {}
        indicators = indicators_result if isinstance(indicators_result, dict) else {}
        derivatives = derivatives_result if isinstance(derivatives_result, dict) else {}

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

    async def _ensure_sentiment_for_analysis(self, symbol: str) -> None:
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
        try:
            await asyncio.wait_for(
                asyncio.shield(task),
                timeout=max(float(settings.sentiment_blocking_timeout_seconds or 0.0), 0.0),
            )
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

    async def _get_ticker_snapshot(self, symbol: str) -> dict[str, Any]:
        ticker = dict(self.ws_client.latest_tickers.get(symbol, {}) or {})
        if ticker:
            bid = self._safe_float(ticker.get("bid"), 0.0)
            ask = self._safe_float(ticker.get("ask"), 0.0)
            mid = (bid + ask) / 2 if bid and ask else 0.0
            if mid and "spread_pct" not in ticker:
                ticker["spread_pct"] = (ask - bid) / mid * 100 if ask >= bid else 0.0
            return ticker
        try:
            raw_ticker = await self.rest_client.fetch_ticker(symbol)
            ticker_info = raw_ticker.get("info") or {}
            bid = self._safe_float(raw_ticker.get("bid") or ticker_info.get("bidPx"), 0.0)
            ask = self._safe_float(raw_ticker.get("ask") or ticker_info.get("askPx"), 0.0)
            mid = (bid + ask) / 2 if bid and ask else 0.0
            return {
                "symbol": symbol,
                "last_price": raw_ticker.get("last", 0),
                "bid": bid,
                "ask": ask,
                "high_24h": raw_ticker.get("high", 0),
                "low_24h": raw_ticker.get("low", 0),
                "volume_24h": raw_ticker.get("baseVolume", 0),
                "change_24h_pct": raw_ticker.get("percentage", 0),
                "spread_pct": ((ask - bid) / mid * 100) if mid and ask and bid else 0,
            }
        except Exception as e:
            logger.debug("failed to fetch ticker", symbol=symbol, error=safe_error_text(e))
            return {}

    async def _get_indicator_snapshot(self, symbol: str) -> dict[str, float]:
        try:
            klines = await self.rest_client.fetch_ohlcv(symbol, timeframe="1h", limit=100)
            if not klines:
                return {}
            await self._persist_klines(symbol, "1h", klines)
            df = pd.DataFrame(
                klines,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = compute_all_indicators(df)
            features = extract_latest_features(df)
            features.update(self._kline_anomaly_snapshot(df))
            return features
        except Exception as e:
            logger.debug("failed to compute indicators", symbol=symbol, error=safe_error_text(e))
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
        try:
            async with get_session_ctx() as session:
                repo = MarketRepository(session)
                normalized = self._normalize_symbols([symbol])[0]
                for row in klines[-300:]:
                    try:
                        ts, open_, high, low, close, volume = row[:6]
                        open_time = datetime.fromtimestamp(float(ts) / 1000.0, tz=UTC)
                        await repo.upsert_kline(
                            normalized,
                            timeframe,
                            open_time,
                            {
                                "open": self._safe_float(open_),
                                "high": self._safe_float(high),
                                "low": self._safe_float(low),
                                "close": self._safe_float(close),
                                "volume": self._safe_float(volume),
                            },
                        )
                    except Exception as exc:
                        logger.debug(
                            "skip invalid kline row",
                            symbol=normalized,
                            timeframe=timeframe,
                            error=safe_error_text(exc),
                        )
                        continue
                await repo.clean_old_klines(normalized, timeframe, keep=2000)
        except Exception as exc:
            logger.debug("persist klines failed", symbol=symbol, error=safe_error_text(exc))

    async def _get_derivatives_snapshot(self, symbol: str) -> dict[str, Any]:
        now = datetime.now(UTC)
        cached = self._derivatives_cache.get(symbol)
        if cached:
            updated_at = cached.get("updated_at")
            if (
                isinstance(updated_at, datetime)
                and (now - updated_at).total_seconds() < self._derivatives_update_interval
            ):
                return dict(cached.get("data") or {})

        try:
            data = await self.rest_client.fetch_derivatives_snapshot(symbol)
        except Exception as e:
            logger.debug(
                "failed to fetch derivatives snapshot",
                symbol=symbol,
                error=safe_error_text(e),
            )
            data = {}

        self._derivatives_cache[symbol] = {
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
        return await self.rest_client.get_available_symbols()

    def get_market_state(self) -> dict[str, Any]:
        """Get a snapshot of current market state for the dashboard."""
        return {
            "tickers": {
                sym: {
                    "price": d.get("last_price", 0),
                    "change_24h": d.get("change_24h_pct", 0),
                    "volume_24h": d.get("volume_24h", 0),
                    "bid": d.get("bid", 0),
                    "ask": d.get("ask", 0),
                }
                for sym, d in self.ws_client.latest_tickers.items()
            },
            "ws_stats": self.ws_client.get_stats(),
        }
