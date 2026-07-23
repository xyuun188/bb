"""Dashboard API for data collection sources and training-sample visibility."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from collections import Counter
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from config.settings import settings
from core.safe_output import safe_error_text
from core.secret_utils import is_masked_secret, mask_secret
from data_feed.external_event_scraper import (
    EXTERNAL_EVENT_MAX_SOURCES_LIMIT,
    RECOMMENDED_EXTERNAL_EVENT_SOURCES,
    SCRAPLING_SOURCE_PREFIX,
    _normalize_source,
    configured_external_event_source_diagnostics,
)
from data_feed.news_fetcher import RSS_FEEDS
from db.session import get_session_ctx
from models.learning import ShadowBacktest
from models.market_data import Kline, Ticker
from models.news import NewsArticle, SocialPost
from services.crypto_feature_coverage import CryptoFeatureCoverageService
from services.ml_signal_service import (
    AUTO_TRAIN_LEASE_STALE_SECONDS,
    AUTO_TRAIN_RETRY_INTERVAL_SECONDS,
    MODEL_TRAINING_STATE_STORE,
)
from services.model_training_state import LOCAL_AI_TOOL_MODEL_IDS
from services.okx_training_gate import okx_training_refresh_gate
from services.secure_runtime_config import set_runtime_secret, strip_secret_env_updates
from services.trading_params import DEFAULT_TRADING_PARAMS
from services.training_data_quality import assess_text_sentiment_sample
from services.training_epoch import load_training_epoch_start
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


def _safe_int_count(value: Any, default: int = 0) -> int:
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError):
        return default
    return max(parsed, 0)


def _safe_feature_coverage_status(result: Any) -> dict[str, Any]:
    if isinstance(result, BaseException):
        logger.warning(
            "crypto feature coverage status failed",
            error=safe_error_text(result),
        )
        payload = _status_error_payload("crypto_feature_coverage", result)
    elif isinstance(result, dict):
        payload = dict(result)
    else:
        payload = {
            "status": "error",
            "section": "crypto_feature_coverage",
            "error": "crypto_feature_coverage returned non-object status",
        }
    payload["audit_only"] = True
    payload["live_signal_mutation"] = False
    payload["can_missing_features_drive_live_entry"] = False
    payload["feature_defaults_are_neutral"] = True
    payload["phase3_policy"] = "missing_or_stale_features_are_neutral_blocked"
    payload["cold_start_safe"] = bool(
        payload.get("waiting_for_decision_samples")
        or payload.get("decision_sample_count") in {0, "0", None}
    )
    payload["display_message"] = (
        "三期冷启动期间缺失/过期特征会展示为待补齐，但默认中性阻断，不能直接驱动开仓。"
    )
    policy = payload.get("feature_contribution_policy")
    if not isinstance(policy, dict):
        policy = {}
    policy["missing_feature_policy"] = "neutral_blocked"
    policy["stale_feature_policy"] = "neutral_blocked"
    policy["low_confidence_event_policy"] = "shadow_only"
    payload["feature_contribution_policy"] = policy
    features = payload.get("features") if isinstance(payload.get("features"), list) else []
    for feature in features:
        if isinstance(feature, dict) and str(feature.get("status") or "") in {
            "missing",
            "stale",
            "low_confidence",
        }:
            feature["live_entry_influence"] = "blocked"
    return payload


def _skipped_feature_coverage_status() -> dict[str, Any]:
    return {
        "status": "skipped",
        "section": "crypto_feature_coverage",
        "reason": "skipped_by_caller",
        "audit_only": True,
        "live_signal_mutation": False,
        "can_missing_features_drive_live_entry": False,
        "feature_defaults_are_neutral": True,
        "feature_contribution_policy": {
            "missing_feature_policy": "neutral_blocked",
            "stale_feature_policy": "neutral_blocked",
            "low_confidence_event_policy": "shadow_only",
        },
    }


async def _run_status_section(
    factory: Callable[[], Awaitable[dict[str, Any]]],
    *,
    timeout: float | None = None,  # noqa: ASYNC109
) -> dict[str, Any] | Exception:
    try:
        result = factory()
        if timeout is not None:
            return await asyncio.wait_for(result, timeout=timeout)
        return await result
    except Exception as exc:
        return exc


def _visible_local_ai_training_status(
    raw_status: str,
    *,
    available: bool,
    model_bundle_available: bool = False,
    shadow_count: int,
    trade_count: int,
    text_count: int,
) -> str:
    normalized = str(raw_status or "unknown").lower()
    if available and model_bundle_available and normalized in {"unknown", "learning_only"}:
        return "ready"
    if available and normalized in {
        "connected_trading_disabled",
    }:
        return "learning_only"
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
    # Read the remote artifact cursor once and publish database counts separately.
    status, db_completed_shadow_count, db_completed_trade_count = await asyncio.gather(
        _raw_local_ai_tools_status(),
        _completed_training_shadow_count(),
        _completed_training_trade_count(),
    )
    if not isinstance(status, dict):
        return {"available": False, "status": "invalid_status"}
    artifact_shadow_count = _safe_int_count(status.get("shadow_sample_count"))
    artifact_trade_count = _safe_int_count(status.get("trade_sample_count"))
    text_count = _safe_int_count(status.get("text_sentiment_sample_count"))
    raw_status = str(status.get("status") or "unknown")
    service_available = bool(status.get("service_available", status.get("available")))
    visible_status = _visible_local_ai_training_status(
        raw_status,
        available=service_available,
        model_bundle_available=bool(status.get("model_bundle_available")),
        shadow_count=db_completed_shadow_count,
        trade_count=db_completed_trade_count,
        text_count=text_count,
    )
    governance_report = (
        status.get("governance_report")
        if isinstance(status.get("governance_report"), dict)
        else {}
    )
    training_shadow_count = db_completed_shadow_count
    training_trade_count = db_completed_trade_count
    epoch_started_at = load_training_epoch_start()
    return {
        "available": service_available,
        "status": visible_status,
        "raw_status": raw_status,
        "training_policy": "current_training_epoch_only",
        "training_epoch_started_at": epoch_started_at.isoformat(),
        "pre_epoch_data_training_allowed": False,
        "model_bundle_available": bool(status.get("model_bundle_available")),
        "service_available": service_available,
        "trained_at": status.get("trained_at"),
        "training_mode": status.get("training_mode"),
        "model_stage": status.get("model_stage"),
        "promotion_flow": status.get("promotion_flow"),
        "live_ml_ready": status.get("live_ml_ready") is True,
        "promotion_recommendation": (
            status.get("promotion_recommendation")
            if isinstance(status.get("promotion_recommendation"), dict)
            else {}
        ),
        "shadow_sample_count": training_shadow_count,
        "training_shadow_sample_count": training_shadow_count,
        "artifact_training_shadow_sample_count": artifact_shadow_count,
        "trade_sample_count": training_trade_count,
        "training_trade_sample_count": training_trade_count,
        "artifact_training_trade_sample_count": artifact_trade_count,
        "sequence_sample_count": _safe_int_count(status.get("sequence_sample_count")),
        "text_sentiment_sample_count": text_count,
        "completed_shadow_sample_count": training_shadow_count,
        "completed_trade_sample_count": training_trade_count,
        "quality_report": (
            status.get("quality_report") if isinstance(status.get("quality_report"), dict) else {}
        ),
        "governance_report": governance_report,
        "models": status.get("models") if isinstance(status.get("models"), dict) else {},
    }


async def _raw_local_ai_tools_status() -> dict[str, Any]:
    local_ai_tools = _dash._dashboard_local_ai_tools_client()
    if local_ai_tools is None:
        return {"available": False, "status": "client_not_ready"}
    try:
        status = await local_ai_tools.status()
    except TimeoutError:
        return {"available": False, "status": "timeout"}
    except Exception as exc:
        return {"available": False, "status": "error", "error": safe_error_text(exc, limit=180)}
    if not isinstance(status, dict):
        return {"available": False, "status": "invalid_status"}
    return status


async def _completed_training_shadow_count() -> int:
    from scripts.train_local_ai_tools_models import _completed_shadow_sample_count

    return int(await _completed_shadow_sample_count())


async def _completed_training_trade_count() -> int:
    from scripts.train_local_ai_tools_models import _completed_trade_sample_count

    return int(await _completed_trade_sample_count())


async def _training_governance_snapshot() -> dict[str, Any]:
    try:
        sample_limit = max(int(GOVERNANCE_SNAPSHOT_SAMPLE_LIMIT), 1)
        epoch_started_at = load_training_epoch_start()
        trade_count = await _completed_training_trade_count()
        async with get_session_ctx() as session:
            epoch_filter = (ShadowBacktest.created_at >= epoch_started_at,)
            completed_result = await session.execute(
                select(func.count(ShadowBacktest.id)).where(
                    *epoch_filter,
                    ShadowBacktest.status == "completed",
                    ShadowBacktest.long_return_pct.is_not(None),
                    ShadowBacktest.short_return_pct.is_not(None),
                )
            )
            quarantined_result = await session.execute(
                select(func.count(ShadowBacktest.id)).where(
                    *epoch_filter,
                    ShadowBacktest.status == "quarantined",
                )
            )
            shadow_window_filter = (
                *epoch_filter,
                ShadowBacktest.status.in_(("completed", "quarantined")),
                ShadowBacktest.long_return_pct.is_not(None),
                ShadowBacktest.short_return_pct.is_not(None),
            )
            shadow_span_result = await session.execute(
                select(
                    func.count(ShadowBacktest.id),
                    func.min(ShadowBacktest.due_at),
                    func.max(ShadowBacktest.due_at),
                ).where(*shadow_window_filter)
            )
            shadow_action_rows = list(
                (
                    await session.execute(
                        select(ShadowBacktest.decision_action, func.count(ShadowBacktest.id))
                        .where(*shadow_window_filter)
                        .group_by(ShadowBacktest.decision_action)
                    )
                ).all()
            )
            trainable_action_rows = list(
                (
                    await session.execute(
                        select(ShadowBacktest.decision_action, func.count(ShadowBacktest.id))
                        .where(
                            *epoch_filter,
                            ShadowBacktest.status == "completed",
                            ShadowBacktest.long_return_pct.is_not(None),
                            ShadowBacktest.short_return_pct.is_not(None),
                        )
                        .group_by(ShadowBacktest.decision_action)
                    )
                ).all()
            )
            shadow_symbol_rows = list(
                (
                    await session.execute(
                        select(ShadowBacktest.symbol, func.count(ShadowBacktest.id))
                        .where(*shadow_window_filter)
                        .group_by(ShadowBacktest.symbol)
                    )
                ).all()
            )

        completed_count = int(completed_result.scalar() or 0)
        quarantined_count = int(quarantined_result.scalar() or 0)
        sample_total_count, oldest_sample_at, latest_sample_at = shadow_span_result.one()
        sample_total_count = int(sample_total_count or 0)
        action_counts = {
            str(action or "unknown").lower(): int(count or 0)
            for action, count in shadow_action_rows
        }
        trainable_action_counts = {
            str(action or "unknown").lower(): int(count or 0)
            for action, count in trainable_action_rows
        }
        symbol_counts = {
            str(symbol or "unknown"): int(count or 0) for symbol, count in shadow_symbol_rows
        }
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
            "training_policy": "current_training_epoch_only",
            "training_epoch_started_at": epoch_started_at.isoformat(),
            "pre_epoch_data_training_allowed": False,
            "sampled": 0,
            "sample_limit": sample_limit,
            "sample_total_count": sample_total_count,
            "raw_sample_count": sample_total_count,
            "current_epoch_trainable_sample_count": completed_count,
            "trainable_sample_source": "current_epoch_completed_shadow_backtests",
            "trainable_sample_count": completed_count,
            "quarantined_sample_count": quarantined_count,
            "historical_audit_sample_count": 0,
            "total_trainable_count": completed_count,
            "quarantined_count": quarantined_count,
            "action_counts": action_counts,
            "trainable_action_counts": trainable_action_counts,
            "symbol_count": len(symbol_counts),
            "symbol_counts": symbol_counts,
            "time_span": {
                "oldest_sample_at": _iso(oldest_sample_at),
                "latest_sample_at": _iso(latest_sample_at),
            },
            "latest_sample_at": _iso(latest_sample_at),
            "data_freshness_minutes": _age_minutes(latest_sample_at),
            "cleanup_mode": "current_training_epoch_only",
            "raw_records_preserved": True,
            "raw_records_preserved_for_audit_only": True,
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
            "training_policy": "current_training_epoch_only",
            "training_epoch_started_at": epoch_started_at.isoformat(),
            "pre_epoch_data_training_allowed": False,
            "raw_records_preserved": True,
            "local_ai_tools": local_ai_report,
            "local_ai_quality_report": shadow_report,
            "local_ml_signal": shadow_report,
            "local_ml_quality_report": shadow_report,
            "local_ml_trainable_shadow_sample_count": completed_count,
            "current_epoch_trainable_shadow_sample_count": completed_count,
            "quarantined_shadow_sample_count": quarantined_count,
            "historical_audit_shadow_sample_count": 0,
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
    lease_attempt = MODEL_TRAINING_STATE_STORE.try_acquire_lease(
        scheduler_id="local_ai_tools_auto_train",
        stale_after_seconds=AUTO_TRAIN_LEASE_STALE_SECONDS,
    )
    if not lease_attempt.acquired or lease_attempt.lease is None:
        return {
            "trained": False,
            "reason": lease_attempt.reason,
            "recovered_stale_lease": lease_attempt.recovered_stale_lease,
        }
    lease = lease_attempt.lease
    now = datetime.now(UTC)
    try:
        MODEL_TRAINING_STATE_STORE.heartbeat(
            scheduler_id="local_ai_tools_auto_train",
            model_ids=LOCAL_AI_TOOL_MODEL_IDS,
            interval_seconds=_LOCAL_ML_TRAINING_PARAMS.auto_train_check_interval_seconds,
        )
        MODEL_TRAINING_STATE_STORE.record_check(
            scheduler_id="local_ai_tools_auto_train",
            model_ids=LOCAL_AI_TOOL_MODEL_IDS,
            run_id=lease.run_id,
            force=True,
        )
        MODEL_TRAINING_STATE_STORE.start_run(
            scheduler_id="local_ai_tools_auto_train",
            model_ids=LOCAL_AI_TOOL_MODEL_IDS,
            run_id=lease.run_id,
            trigger_reason="dashboard_manual_refresh",
            timeout_seconds=AUTO_TRAIN_LEASE_STALE_SECONDS,
        )
    except Exception:
        lease.release()
        raise
    try:
        result = await _train_local_ai_tools_from_dashboard_process()
        failed = str(result.get("reason") or "") in {
            "error",
            "load_samples_error",
            "timeout",
        }
        delay = (
            AUTO_TRAIN_RETRY_INTERVAL_SECONDS
            if failed
            else _LOCAL_ML_TRAINING_PARAMS.auto_train_check_interval_seconds
        )
        MODEL_TRAINING_STATE_STORE.finish_check(
            scheduler_id="local_ai_tools_auto_train",
            model_ids=LOCAL_AI_TOOL_MODEL_IDS,
            run_id=lease.run_id,
            result=result,
            next_check_at=datetime.now(UTC) + timedelta(seconds=delay),
        )
        return result
    except asyncio.CancelledError:
        MODEL_TRAINING_STATE_STORE.record_exception(
            scheduler_id="local_ai_tools_auto_train",
            model_ids=LOCAL_AI_TOOL_MODEL_IDS,
            run_id=lease.run_id,
            error="training_cancelled",
            next_check_at=now + timedelta(seconds=AUTO_TRAIN_RETRY_INTERVAL_SECONDS),
        )
        raise
    except Exception as exc:
        error = safe_error_text(exc, limit=180)
        MODEL_TRAINING_STATE_STORE.record_exception(
            scheduler_id="local_ai_tools_auto_train",
            model_ids=LOCAL_AI_TOOL_MODEL_IDS,
            run_id=lease.run_id,
            error=error,
            next_check_at=now + timedelta(seconds=AUTO_TRAIN_RETRY_INTERVAL_SECONDS),
        )
        return {"trained": False, "reason": "error", "error": error}
    finally:
        lease.release()


async def _train_local_ai_tools_from_dashboard_process() -> dict[str, Any]:
    local_ai_tools = _dash._dashboard_local_ai_tools_client()
    if local_ai_tools is None:
        return {"trained": False, "reason": "client_not_ready"}
    if not getattr(local_ai_tools, "enabled", lambda: False)():
        return {"trained": False, "reason": "disabled"}
    try:
        from scripts.train_local_ai_tools_models import (
            _completed_shadow_sample_count,
            _completed_trade_sample_count,
            _load_sequence_samples,
            _load_shadow_samples,
            _load_text_sentiment_samples,
            _load_trade_samples,
        )
        from services.model_promotion_policy import (
            build_phase3_promotion_recommendation,
            build_return_objective_report,
            load_latest_paper_observation_report,
        )
        from services.training_data_quality import annotate_training_payload

        shadow_samples = await _load_shadow_samples()
        trade_samples = await _load_trade_samples()
        sequence_samples = await _load_sequence_samples()
        text_sentiment_samples = await _load_text_sentiment_samples()
        payload = annotate_training_payload(
            shadow_samples=shadow_samples,
            trade_samples=trade_samples,
            sequence_samples=sequence_samples,
            text_sentiment_samples=text_sentiment_samples,
        )
        trainer = getattr(local_ai_tools, "train", None)
        if not callable(trainer):
            return {"trained": False, "reason": "train_method_missing"}
        completed_shadow_count = await _completed_shadow_sample_count()
        completed_trade_count = await _completed_trade_sample_count()
        raw_trade_sample_count = len(trade_samples)
        trainable_trade_sample_count = len(payload["trade_samples"])
        quarantined_trade_sample_count = max(
            raw_trade_sample_count - trainable_trade_sample_count,
            0,
        )
        paper_observation_report = load_latest_paper_observation_report()
        return_objective_report = build_return_objective_report(
            trade_samples=payload["trade_samples"],
            shadow_samples=payload["shadow_samples"],
        )
        promotion_recommendation = build_phase3_promotion_recommendation(
            training_mode="shadow",
            quality_report=payload["quality_report"],
            governance_report=payload["governance_report"],
            paper_observation_report=paper_observation_report,
            completed_shadow_sample_count=completed_shadow_count,
            completed_trade_sample_count=completed_trade_count,
            return_objective_report=return_objective_report,
        )
        result = await trainer(
            payload["shadow_samples"],
            payload["trade_samples"],
            payload["sequence_samples"],
            payload["text_sentiment_samples"],
            source="dashboard_training_governance_refresh",
            completed_shadow_sample_count=completed_shadow_count,
            completed_trade_sample_count=completed_trade_count,
            raw_trade_sample_count=raw_trade_sample_count,
            trainable_trade_sample_count=trainable_trade_sample_count,
            quarantined_trade_sample_count=quarantined_trade_sample_count,
            trade_sample_cursor_policy="clean_training_view_only",
            quality_report=payload["quality_report"],
            governance_report=payload["governance_report"],
            training_mode="shadow",
            paper_observation_report=paper_observation_report,
            promotion_recommendation=promotion_recommendation,
            persist_artifact=True,
            confirm_phase3_rebuild=True,
        )
        result.setdefault("quality_report", payload["quality_report"])
        result.setdefault("governance_report", payload["governance_report"])
        result.setdefault("return_objective_report", return_objective_report)
        result.setdefault("promotion_recommendation", promotion_recommendation)
        result.setdefault("paper_observation_report", paper_observation_report)
        result.setdefault("raw_trade_sample_count", raw_trade_sample_count)
        result.setdefault("trainable_trade_sample_count", trainable_trade_sample_count)
        result.setdefault("quarantined_trade_sample_count", quarantined_trade_sample_count)
        result.setdefault("trade_sample_cursor_policy", "clean_training_view_only")
        result.setdefault("persist_artifact_requested", True)
        result.setdefault("confirm_phase3_rebuild", True)
        return result
    except Exception as exc:
        return {
            "trained": False,
            "reason": "error",
            "error": safe_error_text(exc, limit=180),
        }


def _training_refresh_okx_gate() -> dict[str, Any]:
    return okx_training_refresh_gate()


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
async def get_data_collection_status(
    include_feature_coverage: bool = True,
) -> dict[str, Any]:
    # asyncpg does not allow concurrent operations on the same connection.
    # Keep status sections serial so dashboard audits report real data
    # problems instead of connection scheduling noise from nested probes.
    source_stats_result = await _run_status_section(_source_breakdown)
    quality_result = await _run_status_section(_training_sample_quality)
    local_ai_status_result = await _run_status_section(_local_ai_training_status)
    governance_result = await _run_status_section(
        _training_governance_snapshot,
        timeout=STATUS_SECTION_TIMEOUT_SECONDS,
    )
    feature_coverage_result: dict[str, Any] | Exception
    if include_feature_coverage:
        feature_coverage_result = await _run_status_section(
            lambda: CryptoFeatureCoverageService().report(hours=24, limit=1000),
            timeout=STATUS_SECTION_TIMEOUT_SECONDS,
        )
    else:
        feature_coverage_result = _skipped_feature_coverage_status()
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
    feature_coverage = _safe_feature_coverage_status(feature_coverage_result)
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
        "feature_coverage": feature_coverage,
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
    okx_gate = _training_refresh_okx_gate()
    if not bool(okx_gate.get("allowed")):
        payload = await get_data_collection_status()
        payload["status"] = "blocked"
        payload["message"] = (
            "Training refresh is blocked until the latest OKX daily reconciliation "
            "report allows clean-view training refresh."
        )
        payload["refresh_blocked"] = True
        payload["okx_daily_reconciliation_gate"] = okx_gate
        payload["refresh_result"] = {
            "training_quarantine": {"status": "skipped", "reason": okx_gate["reason"]},
            "local_ml_signal": {"trained": False, "reason": okx_gate["reason"]},
            "local_ai_tools": {"trained": False, "reason": okx_gate["reason"]},
            "vector_memory_clear": {"status": "skipped", "reason": okx_gate["reason"]},
            "vector_memory": {"status": "skipped", "reason": okx_gate["reason"]},
        }
        return sanitize_payload(payload)

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
    vector_clear_result: dict[str, Any] = {
        "status": "skipped",
        "reason": "disabled_or_unavailable",
    }
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
        vector_service = get_vector_memory_service()
        vector_clear_result = await vector_service.clear_index(
            reason="phase3_training_governance_refresh"
        )
        vector_result = await vector_service.reindex_recent()
    except Exception as exc:
        vector_clear_result = {"status": "error", "error": safe_error_text(exc, limit=180)}
        vector_result = {"status": "error", "error": safe_error_text(exc, limit=180)}

    payload = await get_data_collection_status()
    payload["status"] = "ok"
    payload["message"] = "训练数据治理刷新已执行：按清洗视图重训本地模型并刷新向量索引。"
    payload["refresh_blocked"] = False
    payload["okx_daily_reconciliation_gate"] = okx_gate
    payload["refresh_result"] = {
        "training_quarantine": quarantine_result,
        "local_ml_signal": ml_result,
        "local_ai_tools": local_ai_result,
        "vector_memory_clear": vector_clear_result,
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
        if max_sources < 1 or max_sources > EXTERNAL_EVENT_MAX_SOURCES_LIMIT:
            raise HTTPException(
                status_code=400,
                detail=(
                    "external event max sources must be between 1 and "
                    f"{EXTERNAL_EVENT_MAX_SOURCES_LIMIT}."
                ),
            )
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
