"""Dashboard API for data collection sources and training-sample visibility."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from config.settings import settings
from core.safe_output import safe_error_text
from data_feed.external_event_scraper import (
    SCRAPLING_SOURCE_PREFIX,
    _normalize_source,
    configured_external_event_sources,
)
from data_feed.news_fetcher import RSS_FEEDS
from db.session import get_session_ctx
from models.market_data import Kline, Ticker
from models.news import NewsArticle, SocialPost
from services.training_data_quality import assess_text_sentiment_sample
from web_dashboard.api import dashboard as _dash
from web_dashboard.api.text_sanitize import sanitize_payload

router = APIRouter()

TRAINING_SAMPLE_LIMIT = 240
EXPECTED_KLINE_TIMEFRAMES = ("1m", "5m", "15m", "1h")


class ExternalEventSourcePayload(BaseModel):
    name: str | None = None
    url: str
    symbols: list[str] = Field(default_factory=list)
    weight: float | None = None


class DataCollectionSettingsRequest(BaseModel):
    external_event_scraper_enabled: bool | None = None
    external_event_scraper_interval_seconds: int | None = None
    external_event_scraper_timeout_seconds: float | None = None
    external_event_scraper_max_sources: int | None = None
    external_event_scraper_max_items_per_source: int | None = None
    external_event_scraper_sources: list[ExternalEventSourcePayload] | None = None


def _scrapling_installed() -> bool:
    return importlib.util.find_spec("scrapling") is not None


def _iso(value: Any) -> str | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC).isoformat()
    return value.astimezone(UTC).isoformat()


def _age_minutes(value: Any) -> float | None:
    if not isinstance(value, datetime):
        return None
    dt = value if value.tzinfo else value.replace(tzinfo=UTC)
    return round(max((datetime.now(UTC) - dt.astimezone(UTC)).total_seconds(), 0.0) / 60, 1)


def _source_payload(source: Any) -> dict[str, Any]:
    return {
        "name": source.name,
        "url": source.url,
        "symbols": list(source.symbols),
        "weight": source.weight,
    }


def _safe_source_payload(raw: dict[str, Any]) -> dict[str, Any]:
    source = _normalize_source(raw)
    return _source_payload(source)


async def _source_breakdown() -> dict[str, Any]:
    async with get_session_ctx() as session:
        news_total_row = (
            await session.execute(
                select(
                    func.count(NewsArticle.id),
                    func.max(func.coalesce(NewsArticle.published_at, NewsArticle.fetched_at)),
                )
            )
        ).one()
        news_rows = list(
            (
                await session.execute(
                    select(
                        NewsArticle.source,
                        func.count(NewsArticle.id),
                        func.max(func.coalesce(NewsArticle.published_at, NewsArticle.fetched_at)),
                    )
                    .group_by(NewsArticle.source)
                    .order_by(func.count(NewsArticle.id).desc())
                    .limit(40)
                )
            ).all()
        )
        social_total_row = (
            await session.execute(select(func.count(SocialPost.id), func.max(SocialPost.posted_at)))
        ).one()
        social_rows = list(
            (
                await session.execute(
                    select(
                        SocialPost.platform,
                        func.count(SocialPost.id),
                        func.max(SocialPost.posted_at),
                    )
                    .group_by(SocialPost.platform)
                    .order_by(func.count(SocialPost.id).desc())
                    .limit(20)
                )
            ).all()
        )
        kline_rows = list(
            (
                await session.execute(
                    select(
                        Kline.timeframe,
                        func.count(Kline.id),
                        func.count(func.distinct(Kline.symbol)),
                        func.max(Kline.open_time),
                    )
                    .where(Kline.timeframe.in_(EXPECTED_KLINE_TIMEFRAMES))
                    .group_by(Kline.timeframe)
                    .order_by(Kline.timeframe.asc())
                )
            ).all()
        )
        ticker_row = (
            await session.execute(
                select(
                    func.count(Ticker.id),
                    func.max(func.coalesce(Ticker.updated_at, Ticker.created_at)),
                )
            )
        ).one()

    return {
        "news": {
            "total": int(news_total_row[0] or 0),
            "latest_at": _iso(news_total_row[1]),
            "age_minutes": _age_minutes(news_total_row[1]),
            "sources": [
                {
                    "name": str(source or "unknown"),
                    "count": int(count or 0),
                    "latest_at": _iso(latest),
                    "age_minutes": _age_minutes(latest),
                    "external_event": str(source or "").startswith(SCRAPLING_SOURCE_PREFIX),
                }
                for source, count, latest in news_rows
            ],
        },
        "social": {
            "total": int(social_total_row[0] or 0),
            "latest_at": _iso(social_total_row[1]),
            "age_minutes": _age_minutes(social_total_row[1]),
            "platforms": [
                {
                    "name": str(platform or "unknown"),
                    "count": int(count or 0),
                    "latest_at": _iso(latest),
                    "age_minutes": _age_minutes(latest),
                }
                for platform, count, latest in social_rows
            ],
        },
        "market": {
            "ticker_count": int(ticker_row[0] or 0),
            "ticker_latest_at": _iso(ticker_row[1]),
            "ticker_age_minutes": _age_minutes(ticker_row[1]),
            "klines": [
                {
                    "timeframe": str(timeframe),
                    "rows": int(count or 0),
                    "symbols": int(symbols or 0),
                    "latest_at": _iso(latest),
                    "age_minutes": _age_minutes(latest),
                }
                for timeframe, count, symbols, latest in kline_rows
            ],
        },
    }


async def _training_sample_quality() -> dict[str, Any]:
    async with get_session_ctx() as session:
        news_rows = list(
            (
                await session.execute(
                    select(NewsArticle)
                    .order_by(NewsArticle.id.desc())
                    .limit(TRAINING_SAMPLE_LIMIT // 2)
                )
            )
            .scalars()
            .all()
        )
        social_rows = list(
            (
                await session.execute(
                    select(SocialPost)
                    .order_by(SocialPost.id.desc())
                    .limit(TRAINING_SAMPLE_LIMIT // 2)
                )
            )
            .scalars()
            .all()
        )

    assessments = []
    for row in news_rows:
        text = " ".join(part for part in (row.title, row.summary) if part)
        assessments.append(
            assess_text_sentiment_sample(
                {
                    "source": "news",
                    "platform": row.source,
                    "text": text,
                    "sentiment_score": row.sentiment_score,
                }
            )
        )
    for row in social_rows:
        assessments.append(
            assess_text_sentiment_sample(
                {
                    "source": "social",
                    "platform": row.platform,
                    "text": row.content,
                    "sentiment_score": row.sentiment_score,
                }
            )
        )
    status_counts = Counter(item.status for item in assessments)
    reason_counts: Counter[str] = Counter()
    effective_weight = 0.0
    for item in assessments:
        effective_weight += item.weight
        reason_counts.update(item.reasons)
    total = len(assessments)
    return {
        "sampled": total,
        "included": int(status_counts.get("included", 0)),
        "downweighted": int(status_counts.get("downweighted", 0)),
        "excluded": int(status_counts.get("excluded", 0)),
        "effective_weight": round(effective_weight, 4),
        "effective_ratio": round(effective_weight / total, 4) if total else 0.0,
        "top_reasons": [
            {"reason": reason, "count": count} for reason, count in reason_counts.most_common(8)
        ],
    }


async def _local_ai_training_status() -> dict[str, Any]:
    local_ai_tools = _dash._dashboard_local_ai_tools_client()
    if local_ai_tools is None:
        return {"available": False, "status": "client_not_ready"}
    try:
        status = await asyncio.wait_for(local_ai_tools.status(), timeout=3.5)
    except TimeoutError:
        return {"available": False, "status": "timeout"}
    except Exception as exc:
        return {"available": False, "status": "error", "error": safe_error_text(exc, limit=180)}
    if not isinstance(status, dict):
        return {"available": False, "status": "invalid_status"}
    shadow_count = int(status.get("shadow_sample_count") or 0)
    trade_count = int(status.get("trade_sample_count") or 0)
    text_count = int(status.get("text_sentiment_sample_count") or 0)
    raw_status = str(status.get("status") or "unknown")
    visible_status = raw_status
    if raw_status == "unknown" and bool(status.get("available")):
        visible_status = "learning_only" if shadow_count or trade_count or text_count else "ready"
    return {
        "available": bool(status.get("available")),
        "status": visible_status,
        "raw_status": raw_status,
        "shadow_sample_count": shadow_count,
        "trade_sample_count": trade_count,
        "sequence_sample_count": int(status.get("sequence_sample_count") or 0),
        "text_sentiment_sample_count": text_count,
        "completed_shadow_sample_count": int(status.get("completed_shadow_sample_count") or 0),
        "completed_trade_sample_count": int(status.get("completed_trade_sample_count") or 0),
        "quality_report": (
            status.get("quality_report") if isinstance(status.get("quality_report"), dict) else {}
        ),
        "models": status.get("models") if isinstance(status.get("models"), dict) else {},
    }


def _configured_source_cards() -> list[dict[str, Any]]:
    sources = configured_external_event_sources()
    return [_source_payload(source) for source in sources]


def _collection_sources_summary() -> list[dict[str, Any]]:
    return [
        {
            "key": "rss",
            "name": "新闻 RSS",
            "enabled": True,
            "status": "active",
            "detail": f"{len(RSS_FEEDS)} 个公开 RSS 源，默认采集。",
        },
        {
            "key": "okx_announcements",
            "name": "OKX 公告",
            "enabled": True,
            "status": "active",
            "detail": "OKX 官方公告 API，默认采集。",
        },
        {
            "key": "reddit",
            "name": "Reddit 舆情",
            "enabled": True,
            "status": "active",
            "detail": "Reddit JSON/RSS，默认采集。",
        },
        {
            "key": "cryptopanic",
            "name": "CryptoPanic",
            "enabled": bool(settings.cryptopanic_api_key),
            "status": "active" if settings.cryptopanic_api_key else "not_configured",
            "detail": "需要 CRYPTOPANIC_API_KEY。",
        },
        {
            "key": "coinmarketcal",
            "name": "CoinMarketCal",
            "enabled": bool(settings.coinmarketcal_api_key),
            "status": "active" if settings.coinmarketcal_api_key else "not_configured",
            "detail": "需要 COINMARKETCAL_API_KEY。",
        },
        {
            "key": "newsapi",
            "name": "NewsAPI",
            "enabled": bool(settings.newsapi_api_key),
            "status": "active" if settings.newsapi_api_key else "not_configured",
            "detail": "需要 NEWSAPI_API_KEY。",
        },
        {
            "key": "scrapling",
            "name": "Scrapling 外部事件",
            "enabled": bool(settings.external_event_scraper_enabled),
            "status": (
                "active"
                if settings.external_event_scraper_enabled and _scrapling_installed()
                else "missing_dependency" if settings.external_event_scraper_enabled else "disabled"
            ),
            "detail": "用于交易所公告、项目博客、事件网页增强；不进入交易热路径。",
        },
    ]


@router.get("/data-collection/status")
async def get_data_collection_status() -> dict[str, Any]:
    source_stats, quality, local_ai_status = await asyncio.gather(
        _source_breakdown(),
        _training_sample_quality(),
        _local_ai_training_status(),
    )
    scrapling_installed = _scrapling_installed()
    payload = {
        "checked_at": datetime.now(UTC).isoformat(),
        "config": {
            "external_event_scraper_enabled": bool(settings.external_event_scraper_enabled),
            "external_event_scraper_dependency_installed": scrapling_installed,
            "external_event_scraper_runtime_active": bool(
                settings.external_event_scraper_enabled and scrapling_installed
            ),
            "external_event_scraper_interval_seconds": int(
                settings.external_event_scraper_interval_seconds
            ),
            "external_event_scraper_timeout_seconds": float(
                settings.external_event_scraper_timeout_seconds
            ),
            "external_event_scraper_max_sources": int(settings.external_event_scraper_max_sources),
            "external_event_scraper_max_items_per_source": int(
                settings.external_event_scraper_max_items_per_source
            ),
            "external_event_scraper_sources": _configured_source_cards(),
            "external_event_scraper_uses_default_sources": not bool(
                settings.external_event_scraper_sources
            ),
        },
        "sources": _collection_sources_summary(),
        "stats": source_stats,
        "training": {
            "text_sentiment_quality_sample": quality,
            "local_ai_tools": local_ai_status,
        },
    }
    return sanitize_payload(payload)


async def _sync_runtime_external_event_service(enabled: bool) -> dict[str, Any]:
    data_service = getattr(_dash, "_data_service", None)
    service = getattr(data_service, "external_event_service", None) if data_service else None
    if service is None:
        return {
            "attached": False,
            "message": "配置已保存；Dashboard 与交易主循环分离运行时，需要重启交易服务后完全生效。",
        }
    if enabled:
        await service.start()
        return {"attached": True, "message": "已尝试启动当前进程的数据采集后台任务。"}
    await service.stop()
    return {"attached": True, "message": "已停止当前进程的数据采集后台任务。"}


@router.post("/data-collection/settings")
async def update_data_collection_settings(req: DataCollectionSettingsRequest) -> dict[str, Any]:
    updates: dict[str, str] = {}

    if req.external_event_scraper_enabled is not None:
        settings.external_event_scraper_enabled = bool(req.external_event_scraper_enabled)
        updates["EXTERNAL_EVENT_SCRAPER_ENABLED"] = (
            "true" if settings.external_event_scraper_enabled else "false"
        )

    if req.external_event_scraper_interval_seconds is not None:
        interval = int(req.external_event_scraper_interval_seconds)
        if interval < 60 or interval > 86400:
            raise HTTPException(status_code=400, detail="采集间隔必须在 60 秒到 86400 秒之间。")
        settings.external_event_scraper_interval_seconds = interval
        updates["EXTERNAL_EVENT_SCRAPER_INTERVAL_SECONDS"] = str(interval)

    if req.external_event_scraper_timeout_seconds is not None:
        timeout = float(req.external_event_scraper_timeout_seconds)
        if timeout < 1 or timeout > 30:
            raise HTTPException(status_code=400, detail="单源超时必须在 1 秒到 30 秒之间。")
        settings.external_event_scraper_timeout_seconds = timeout
        updates["EXTERNAL_EVENT_SCRAPER_TIMEOUT_SECONDS"] = str(timeout)

    if req.external_event_scraper_max_sources is not None:
        max_sources = int(req.external_event_scraper_max_sources)
        if max_sources < 1 or max_sources > 20:
            raise HTTPException(status_code=400, detail="每轮源数量必须在 1 到 20 之间。")
        settings.external_event_scraper_max_sources = max_sources
        updates["EXTERNAL_EVENT_SCRAPER_MAX_SOURCES"] = str(max_sources)

    if req.external_event_scraper_max_items_per_source is not None:
        max_items = int(req.external_event_scraper_max_items_per_source)
        if max_items < 1 or max_items > 50:
            raise HTTPException(status_code=400, detail="每源条数必须在 1 到 50 之间。")
        settings.external_event_scraper_max_items_per_source = max_items
        updates["EXTERNAL_EVENT_SCRAPER_MAX_ITEMS_PER_SOURCE"] = str(max_items)

    if req.external_event_scraper_sources is not None:
        normalized_sources = []
        for raw_source in req.external_event_scraper_sources:
            try:
                normalized_sources.append(_safe_source_payload(raw_source.model_dump()))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=safe_error_text(exc)) from exc
        settings.external_event_scraper_sources = normalized_sources
        updates["EXTERNAL_EVENT_SCRAPER_SOURCES"] = json.dumps(
            normalized_sources,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    if updates:
        settings.update_env_file(updates)

    runtime = await _sync_runtime_external_event_service(settings.external_event_scraper_enabled)
    payload = await get_data_collection_status()
    payload["status"] = "ok"
    payload["message"] = "数据采集配置已保存。"
    payload["runtime_sync"] = runtime
    return sanitize_payload(payload)
