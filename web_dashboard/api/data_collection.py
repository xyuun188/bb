"""Dashboard API for data collection sources and training-sample visibility."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from collections import Counter
from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from config.settings import settings
from core.safe_output import safe_error_text
from core.secret_utils import is_masked_secret, mask_secret
from data_feed.external_event_scraper import (
    RECOMMENDED_EXTERNAL_EVENT_SOURCES,
    SCRAPLING_SOURCE_PREFIX,
    _normalize_source,
    configured_external_event_source_diagnostics,
)
from data_feed.news_fetcher import RSS_FEEDS
from db.session import get_session_ctx
from models.market_data import Kline, Ticker
from models.news import NewsArticle, SocialPost
from models.learning import ShadowBacktest, TradeReflection
from models.trade import Position
from services.secure_runtime_config import set_runtime_secret, strip_secret_env_updates
from services.trading_params import DEFAULT_TRADING_PARAMS
from services.training_data_quality import assess_text_sentiment_sample
from services.vector_memory import get_vector_memory_service
from web_dashboard.api import dashboard as _dash
from web_dashboard.api.security import require_dashboard_write_access
from web_dashboard.api.text_sanitize import sanitize_payload

router = APIRouter()
logger = structlog.get_logger(__name__)

TRAINING_SAMPLE_LIMIT = 240
GOVERNANCE_SNAPSHOT_SAMPLE_LIMIT = 500
STATUS_SECTION_TIMEOUT_SECONDS = 6.0
EXPECTED_KLINE_TIMEFRAMES = ("1m", "5m", "15m", "1h")
_LOCAL_ML_TRAINING_PARAMS = DEFAULT_TRADING_PARAMS.local_ml_training


def _status_error_payload(section: str, exc: BaseException) -> dict[str, Any]:
    return {
        "status": "error",
        "section": section,
        "error": safe_error_text(exc, limit=180),
    }


def _safe_status_section(
    result: Any,
    *,
    section: str,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    if isinstance(result, BaseException):
        logger.warning(
            "data collection status section failed",
            section=section,
            error=safe_error_text(result),
        )
        payload = dict(fallback)
        payload.update(_status_error_payload(section, result))
        return payload
    if isinstance(result, dict):
        return result
    payload = dict(fallback)
    payload.update(
        {
            "status": "error",
            "section": section,
            "error": f"{section} returned non-object status",
        }
    )
    return payload


def _visible_local_ai_training_status(
    raw_status: str,
    *,
    available: bool,
    shadow_count: int,
    trade_count: int,
    text_count: int,
) -> str:
    normalized = str(raw_status or "unknown").lower()
    if normalized == "unknown" and available:
        return "learning_only" if shadow_count or trade_count or text_count else "ready"
    return normalized


def _governance_quality_report(
    assessments: list[Any],
    *,
    total_trainable_count: int,
    quarantined_count: int,
    sample_limit: int,
) -> dict[str, Any]:
    """Build a bounded training-governance summary for status pages."""

    sampled = len(assessments)
    status_counts: Counter[str] = Counter(
        str(getattr(item, "status", "unknown") or "unknown") for item in assessments
    )
    reason_counts: Counter[str] = Counter(
        reason for item in assessments for reason in tuple(getattr(item, "reasons", ()) or ())
    )
    effective_weight = sum(float(getattr(item, "weight", 0.0) or 0.0) for item in assessments)
    effective_ratio = effective_weight / sampled if sampled else 0.0
    sampled_excluded_count = int(status_counts.get("excluded", 0))
    downweighted_count = int(status_counts.get("downweighted", 0))
    if quarantined_count:
        status = "quarantined"
    elif sampled_excluded_count:
        status = "error"
    elif downweighted_count:
        status = "downweighted"
    else:
        status = "clean" if total_trainable_count else "empty"
    return {
        "status": status,
        "summary": (
            f"训练视图 {total_trainable_count} 条，抽样 {sampled} 条，"
            f"有效权重 {effective_ratio * 100:.1f}%"
        ),
        "sampled_count": sampled,
        "sample_limit": int(sample_limit),
        "trainable_sample_count": int(total_trainable_count),
        "included_sample_count": int(status_counts.get("included", 0)),
        "downweighted_sample_count": downweighted_count,
        "excluded_sample_count": sampled_excluded_count + int(quarantined_count),
        "quarantined_sample_count": int(quarantined_count),
        "effective_weight_ratio": round(effective_ratio, 4),
        "top_reasons": [
            {"reason": reason, "count": count} for reason, count in reason_counts.most_common(8)
        ],
        "raw_records_preserved": True,
        "requires_artifact_refresh": bool(
            quarantined_count or sampled_excluded_count or downweighted_count
        ),
        "refresh_targets": ["local_ml_signal", "local_ai_tools", "vector_memory_reindex"],
    }


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
    cryptopanic_api_key: str | None = None
    coinmarketcal_api_key: str | None = None
    newsapi_api_key: str | None = None


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
        "category": getattr(source, "category", "project"),
        "description": getattr(source, "description", ""),
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
                _kline_coverage_row(timeframe, kline_rows)
                for timeframe in EXPECTED_KLINE_TIMEFRAMES
            ],
        },
    }


def _kline_coverage_row(
    timeframe: str,
    kline_rows: list[tuple[Any, Any, Any, Any]],
) -> dict[str, Any]:
    for row_timeframe, count, symbols, latest in kline_rows:
        if str(row_timeframe) == timeframe:
            return {
                "timeframe": timeframe,
                "rows": int(count or 0),
                "symbols": int(symbols or 0),
                "latest_at": _iso(latest),
                "age_minutes": _age_minutes(latest),
                "missing": False,
            }
    return {
        "timeframe": timeframe,
        "rows": 0,
        "symbols": 0,
        "latest_at": None,
        "age_minutes": None,
        "missing": True,
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
    source_counts: Counter[str] = Counter()
    trainable_source_counts: Counter[str] = Counter()
    for row in news_rows:
        text = " ".join(part for part in (row.title, row.summary) if part)
        sample = {
            "source": "news",
            "platform": row.source,
            "text": text,
            "sentiment_score": row.sentiment_score,
        }
        assessment = assess_text_sentiment_sample(sample)
        source = str(row.source or "news")
        source_counts[source] += 1
        if not assessment.exclude_from_training:
            trainable_source_counts[source] += 1
        assessments.append(assessment)
    for row in social_rows:
        sample = {
            "source": "social",
            "platform": row.platform,
            "text": row.content,
            "sentiment_score": row.sentiment_score,
        }
        assessment = assess_text_sentiment_sample(sample)
        source = str(row.platform or "social")
        source_counts[source] += 1
        if not assessment.exclude_from_training:
            trainable_source_counts[source] += 1
        assessments.append(assessment)
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
        "sources": dict(source_counts),
        "trainable_sources": dict(trainable_source_counts),
        "top_sources": [
            {
                "source": source,
                "count": count,
                "trainable": int(trainable_source_counts.get(source, 0)),
            }
            for source, count in source_counts.most_common(12)
        ],
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
    visible_status = _visible_local_ai_training_status(
        raw_status,
        available=bool(status.get("available")),
        shadow_count=shadow_count,
        trade_count=trade_count,
        text_count=text_count,
    )
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
        "governance_report": (
            status.get("governance_report")
            if isinstance(status.get("governance_report"), dict)
            else {}
        ),
        "models": status.get("models") if isinstance(status.get("models"), dict) else {},
    }


async def _training_governance_snapshot() -> dict[str, Any]:
    try:
        sample_limit = max(int(GOVERNANCE_SNAPSHOT_SAMPLE_LIMIT), 1)
        async with get_session_ctx() as session:
            completed_result = await session.execute(
                select(func.count(ShadowBacktest.id)).where(
                    ShadowBacktest.status == "completed",
                    ShadowBacktest.long_return_pct.is_not(None),
                    ShadowBacktest.short_return_pct.is_not(None),
                )
            )
            quarantined_result = await session.execute(
                select(func.count(ShadowBacktest.id)).where(ShadowBacktest.status == "quarantined")
            )
            trade_reflection_result = await session.execute(select(func.count(TradeReflection.id)))
            closed_position_result = await session.execute(
                select(func.count(Position.id)).where(
                    Position.is_open.is_(False),
                    Position.closed_at.is_not(None),
                )
            )

        completed_count = int(completed_result.scalar() or 0)
        quarantined_count = int(quarantined_result.scalar() or 0)
        trade_count = int(trade_reflection_result.scalar() or 0) + int(
            closed_position_result.scalar() or 0
        )
        if quarantined_count:
            status = "quarantined"
            summary = f"已隔离 {quarantined_count} 条训练样本；原始记录保留。"
        elif completed_count:
            status = "clean"
            summary = "状态页使用轻量治理快照；深度样本质量评估在训练/刷新任务中执行。"
        else:
            status = "empty"
            summary = "暂无可训练影子样本。"
        shadow_report = {
            "status": status,
            "summary": summary,
            "sampled": 0,
            "sample_limit": sample_limit,
            "total_trainable_count": completed_count,
            "quarantined_count": quarantined_count,
            "cleanup_mode": "quarantine_not_delete",
            "raw_records_preserved": True,
            "quarantine_applied": bool(quarantined_count),
            "requires_artifact_refresh": bool(quarantined_count),
            "refresh_targets": [
                "local_ml_signal",
                "local_ai_tools",
                "vector_memory_reindex",
            ],
            "deep_quality_evaluation": "deferred_to_training_refresh",
        }
        local_ai_report = dict(shadow_report)
        local_ai_report["trade_sample_count"] = trade_count
        return {
            "status": "ok",
            "local_ai_tools": local_ai_report,
            "local_ai_quality_report": shadow_report,
            "local_ml_signal": shadow_report,
            "local_ml_quality_report": shadow_report,
            "local_ml_trainable_shadow_sample_count": completed_count,
            "training_quarantine": {
                "status": "not_run",
                "message": "状态页只读取轻量治理快照；点击清洗刷新或等待自动训练时执行深度评估、隔离与重训。",
            },
            "cleanup_effective": True,
            "artifact_refresh_targets": [
                "local_ml_signal",
                "local_ai_tools",
                "vector_memory_reindex",
            ],
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": safe_error_text(exc, limit=180),
            "cleanup_effective": False,
        }


async def _train_local_ai_tools_from_dashboard() -> dict[str, Any]:
    local_ai_tools = _dash._dashboard_local_ai_tools_client()
    if local_ai_tools is None:
        return {"trained": False, "reason": "client_not_ready"}
    if not getattr(local_ai_tools, "enabled", lambda: False)():
        return {"trained": False, "reason": "disabled"}
    try:
        from scripts.train_local_ai_tools_models import (
            _completed_shadow_sample_count,
            _completed_trade_sample_count,
            _load_closed_position_samples,
            _load_sequence_samples,
            _load_shadow_samples,
            _load_text_sentiment_samples,
            _load_trade_reflection_samples,
            _merge_trade_samples,
        )
        from services.training_data_quality import annotate_training_payload

        shadow_samples = await _load_shadow_samples(
            _LOCAL_ML_TRAINING_PARAMS.training_shadow_sample_limit
        )
        trade_reflection_samples = await _load_trade_reflection_samples(
            _LOCAL_ML_TRAINING_PARAMS.training_trade_sample_limit
        )
        closed_position_samples = await _load_closed_position_samples(
            _LOCAL_ML_TRAINING_PARAMS.training_trade_sample_limit
        )
        trade_samples = _merge_trade_samples(trade_reflection_samples, closed_position_samples)
        sequence_samples = await _load_sequence_samples(
            _LOCAL_ML_TRAINING_PARAMS.training_sequence_sample_limit
        )
        text_sentiment_samples = await _load_text_sentiment_samples(
            _LOCAL_ML_TRAINING_PARAMS.training_text_sample_limit
        )
        payload = annotate_training_payload(
            shadow_samples=shadow_samples,
            trade_samples=trade_samples,
            sequence_samples=sequence_samples,
            text_sentiment_samples=text_sentiment_samples,
        )
        trainer = getattr(local_ai_tools, "train", None)
        if not callable(trainer):
            return {"trained": False, "reason": "train_method_missing"}
        result = await trainer(
            payload["shadow_samples"],
            payload["trade_samples"],
            payload["sequence_samples"],
            payload["text_sentiment_samples"],
            source="dashboard_training_governance_refresh",
            completed_shadow_sample_count=await _completed_shadow_sample_count(),
            completed_trade_sample_count=await _completed_trade_sample_count(),
            quality_report=payload["quality_report"],
            governance_report=payload["governance_report"],
        )
        result.setdefault("quality_report", payload["quality_report"])
        result.setdefault("governance_report", payload["governance_report"])
        return result
    except Exception as exc:
        return {
            "trained": False,
            "reason": "error",
            "error": safe_error_text(exc, limit=180),
        }


def _configured_source_cards() -> list[dict[str, Any]]:
    return configured_external_event_source_diagnostics()


def _recommended_source_cards() -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for raw_source in RECOMMENDED_EXTERNAL_EVENT_SOURCES:
        try:
            cards.append(_safe_source_payload(raw_source))
        except ValueError as exc:
            logger.warning(
                "recommended external event source rejected",
                error=safe_error_text(exc),
            )
    return cards


def _collection_sources_summary() -> list[dict[str, Any]]:
    scrapling_installed = _scrapling_installed()
    scrapling_sources = _configured_source_cards()
    valid_scrapling_sources = [
        source for source in scrapling_sources if source.get("valid") and source.get("enabled")
    ]
    invalid_scrapling_sources = [source for source in scrapling_sources if not source.get("valid")]
    if not settings.external_event_scraper_enabled:
        scrapling_status = "disabled"
        scrapling_detail = "用于交易所公告、项目博客、事件网页增强；默认关闭。"
    elif not scrapling_installed:
        scrapling_status = "missing_dependency"
        scrapling_detail = "Scrapling 依赖未安装，无法采集外部网页。"
    elif not valid_scrapling_sources:
        scrapling_status = "invalid_config"
        scrapling_detail = "已启用，但没有有效 HTTPS 公网采集源；请在外部事件采集设置中修复。"
    elif invalid_scrapling_sources:
        scrapling_status = "degraded"
        scrapling_detail = (
            f"有效源 {len(valid_scrapling_sources)} 个，"
            f"无效源 {len(invalid_scrapling_sources)} 个；请修复无效源。"
        )
    else:
        scrapling_status = "active"
        scrapling_detail = f"有效源 {len(valid_scrapling_sources)} 个，后台热加载采集中。"
    return [
        {
            "key": "rss",
            "name": "新闻 RSS",
            "group": "system",
            "enabled": True,
            "status": "active",
            "detail": f"{len(RSS_FEEDS)} 个公开 RSS 源，默认采集。",
        },
        {
            "key": "okx_announcements",
            "name": "OKX 公告",
            "group": "system",
            "enabled": True,
            "status": "active",
            "detail": "OKX 官方公告 API，默认采集。",
        },
        {
            "key": "reddit",
            "name": "Reddit 舆情",
            "group": "system",
            "enabled": True,
            "status": "active",
            "detail": "Reddit JSON/RSS，默认采集。",
        },
        {
            "key": "cryptopanic",
            "name": "CryptoPanic",
            "group": "api",
            "enabled": bool(settings.cryptopanic_api_key),
            "status": "active" if settings.cryptopanic_api_key else "not_configured",
            "detail": "外部新闻聚合 API，可在系统设置 → 外部事件采集中配置。",
        },
        {
            "key": "coinmarketcal",
            "name": "CoinMarketCal",
            "group": "api",
            "enabled": bool(settings.coinmarketcal_api_key),
            "status": "active" if settings.coinmarketcal_api_key else "not_configured",
            "detail": "事件日历 API，可在系统设置 → 外部事件采集中配置。",
        },
        {
            "key": "newsapi",
            "name": "NewsAPI",
            "group": "api",
            "enabled": bool(settings.newsapi_api_key),
            "status": "active" if settings.newsapi_api_key else "not_configured",
            "detail": "宏观/新闻补充 API，可在系统设置 → 外部事件采集中配置。",
        },
        {
            "key": "scrapling",
            "name": "Scrapling 外部事件",
            "group": "scrapling",
            "enabled": bool(settings.external_event_scraper_enabled),
            "status": scrapling_status,
            "detail": scrapling_detail,
        },
    ]


@router.get("/data-collection/status")
async def get_data_collection_status() -> dict[str, Any]:
    source_stats_result, quality_result, local_ai_status_result, governance_result = (
        await asyncio.gather(
            _source_breakdown(),
            _training_sample_quality(),
            _local_ai_training_status(),
            asyncio.wait_for(
                _training_governance_snapshot(),
                timeout=STATUS_SECTION_TIMEOUT_SECONDS,
            ),
            return_exceptions=True,
        )
    )
    source_stats = _safe_status_section(
        source_stats_result,
        section="source_breakdown",
        fallback={"news": {}, "social": {}, "market": {}},
    )
    quality = _safe_status_section(
        quality_result,
        section="training_sample_quality",
        fallback={"sampled": 0, "included": 0, "top_sources": [], "top_reasons": []},
    )
    local_ai_status = _safe_status_section(
        local_ai_status_result,
        section="local_ai_training_status",
        fallback={"available": False, "status": "error"},
    )
    governance = _safe_status_section(
        governance_result,
        section="training_governance",
        fallback={"cleanup_effective": False},
    )
    scrapling_installed = _scrapling_installed()
    configured_source_cards = _configured_source_cards()
    valid_scrapling_sources = [
        source
        for source in configured_source_cards
        if source.get("valid") and source.get("enabled")
    ]
    invalid_scrapling_sources = [
        source for source in configured_source_cards if not source.get("valid")
    ]
    payload = {
        "checked_at": datetime.now(UTC).isoformat(),
        "config": {
            "external_event_scraper_enabled": bool(settings.external_event_scraper_enabled),
            "external_event_scraper_dependency_installed": scrapling_installed,
            "external_event_scraper_runtime_active": bool(
                settings.external_event_scraper_enabled
                and scrapling_installed
                and valid_scrapling_sources
            ),
            "external_event_scraper_valid_source_count": len(valid_scrapling_sources),
            "external_event_scraper_invalid_source_count": len(invalid_scrapling_sources),
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
            "external_event_scraper_sources": configured_source_cards,
            "recommended_external_event_sources": _recommended_source_cards(),
            "external_event_scraper_uses_default_sources": not bool(
                settings.external_event_scraper_sources
            ),
            "api_channels": {
                "cryptopanic": {
                    "label": "CryptoPanic",
                    "configured": bool(settings.cryptopanic_api_key),
                    "api_key": mask_secret(settings.cryptopanic_api_key),
                },
                "coinmarketcal": {
                    "label": "CoinMarketCal",
                    "configured": bool(settings.coinmarketcal_api_key),
                    "api_key": mask_secret(settings.coinmarketcal_api_key),
                },
                "newsapi": {
                    "label": "NewsAPI",
                    "configured": bool(settings.newsapi_api_key),
                    "api_key": mask_secret(settings.newsapi_api_key),
                },
            },
        },
        "sources": _collection_sources_summary(),
        "stats": source_stats,
        "training": {
            "text_sentiment_quality_sample": quality,
            "local_ai_tools": local_ai_status,
            "governance": governance,
        },
    }
    return sanitize_payload(payload)


@router.post("/data-collection/training-governance/refresh")
async def refresh_training_governance(
    _access: None = Depends(require_dashboard_write_access),
) -> dict[str, Any]:
    quarantine_result: dict[str, Any]
    try:
        from services.shadow_training_quarantine import quarantine_dirty_shadow_samples

        quarantine_result = await quarantine_dirty_shadow_samples(
            batch_size=_LOCAL_ML_TRAINING_PARAMS.auto_quarantine_batch_size,
            max_batches=_LOCAL_ML_TRAINING_PARAMS.auto_quarantine_max_batches,
        )
    except Exception as exc:
        quarantine_result = _status_error_payload("training_quarantine", exc)

    ml_signal_service = _dash._dashboard_ml_signal_service()
    local_ai_result: dict[str, Any] = {"trained": False, "reason": "service_not_ready"}
    ml_result: dict[str, Any] = {"trained": False, "reason": "service_not_ready"}
    vector_result: dict[str, Any] = {"status": "skipped", "reason": "disabled_or_unavailable"}

    if ml_signal_service is not None:
        trainer = getattr(ml_signal_service, "maybe_auto_train", None)
        if callable(trainer):
            ml_result = await trainer(force=True)

    trading_service = getattr(_dash, "_trading_service", None)
    if trading_service is not None:
        trainer = getattr(trading_service, "_maybe_train_local_ai_tools", None)
        if callable(trainer):
            local_ai_result = await trainer(force=True)
    else:
        local_ai_result = await _train_local_ai_tools_from_dashboard()

    try:
        vector_result = await get_vector_memory_service().reindex_recent()
    except Exception as exc:
        vector_result = {"status": "error", "error": safe_error_text(exc, limit=180)}

    payload = await get_data_collection_status()
    payload["status"] = "ok"
    payload["message"] = "训练数据治理刷新已执行：按清洗视图重训本地模型并刷新向量索引。"
    payload["refresh_result"] = {
        "training_quarantine": quarantine_result,
        "local_ml_signal": ml_result,
        "local_ai_tools": local_ai_result,
        "vector_memory": vector_result,
    }
    return sanitize_payload(payload)


async def _sync_runtime_external_event_service(enabled: bool) -> dict[str, Any]:
    data_service = getattr(_dash, "_data_service", None)
    service = getattr(data_service, "external_event_service", None) if data_service else None
    if service is None:
        return {
            "attached": False,
            "message": "配置已保存；交易主循环会在数秒内自动热加载采集配置。",
        }
    reload_runtime_settings = getattr(service, "reload_runtime_settings", None)
    if callable(reload_runtime_settings):
        await reload_runtime_settings()
    if enabled:
        await service.start()
        return {"attached": True, "message": "已热加载并启动当前进程的数据采集后台任务。"}
    await service.stop()
    return {"attached": True, "message": "已热加载并停止当前进程的数据采集后台任务。"}


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

    for field_name, env_key, secure_key in (
        ("cryptopanic_api_key", "CRYPTOPANIC_API_KEY", "data_collection.cryptopanic_api_key"),
        (
            "coinmarketcal_api_key",
            "COINMARKETCAL_API_KEY",
            "data_collection.coinmarketcal_api_key",
        ),
        ("newsapi_api_key", "NEWSAPI_API_KEY", "data_collection.newsapi_api_key"),
    ):
        raw_value = getattr(req, field_name)
        if raw_value is None:
            continue
        value = raw_value.strip()
        if not value or is_masked_secret(value):
            continue
        setattr(settings, field_name, value)
        updates[env_key] = value
        await set_runtime_secret(secure_key, value)

    if updates:
        settings.update_env_file(strip_secret_env_updates(updates))

    runtime = await _sync_runtime_external_event_service(settings.external_event_scraper_enabled)
    payload = await get_data_collection_status()
    payload["status"] = "ok"
    payload["message"] = "数据采集配置已保存。"
    payload["runtime_sync"] = runtime
    return sanitize_payload(payload)
