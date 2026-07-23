"""Train the server-side local quant tools from local trading history."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import httpx
from sqlalchemy import func, select

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import settings
from core.safe_output import safe_error_text, safe_print, safe_response_error_text
from core.url_safety import normalize_http_base_url
from db.session import get_read_session_ctx, get_session_ctx
from models.learning import ShadowBacktest
from models.market_data import Kline
from models.news import NewsArticle, SocialPost
from services.authoritative_trade_outcome import load_authoritative_trade_outcomes
from services.execution_cost_model import round_trip_fee_pct
from services.model_promotion_policy import (
    build_phase3_promotion_recommendation,
    build_return_objective_report,
    load_latest_paper_observation_report,
)
from services.okx_training_gate import okx_training_refresh_gate
from services.shadow_training_quarantine import quarantine_dirty_shadow_samples
from services.trading_params import DEFAULT_TRADING_PARAMS
from services.training_data_quality import (
    COMPACT_SEQUENCE_SERIES_FORMAT,
    annotate_training_payload,
    artifact_bound_governance_report,
)
from services.training_epoch import load_training_epoch_start

_AUTH_FAILURE_STATUS_CODES = {401, 403}
_ERROR_EXCERPT_LIMIT = 700
_LOCAL_ML_TRAINING_PARAMS = DEFAULT_TRADING_PARAMS.local_ml_training
_LOCAL_AI_TOOLS_FEATURE_KEYS = {
    "change_24h_pct",
    "spread_pct",
    "rsi_14",
    "rsi_7",
    "macd",
    "macd_signal",
    "macd_diff",
    "stoch_k",
    "adx_14",
    "bb_width",
    "bb_pct",
    "atr_14",
    "atr_pct",
    "current_price",
    "close",
    "volume_ratio",
    "returns_1",
    "returns_5",
    "returns_20",
    "volatility_20",
    "price_vs_sma20",
    "price_vs_sma50",
    "funding_rate",
    "funding_interval_minutes",
    "funding_interval_hours",
    "round_trip_fee_pct",
    "volume_24h",
    "open_interest_value",
    "orderbook_imbalance",
    "orderbook_bid_depth",
    "orderbook_ask_depth",
    "news_sentiment_avg",
    "social_sentiment_avg",
    "social_mention_count",
    "news_article_count",
    "decision_confidence",
    "horizon_minutes",
    "symbol",
}
_LOCAL_AI_TOOLS_SEQUENCE_KEYS = {
    "close_sequence",
    "volume_sequence",
    "recent_closes",
    "recent_volumes",
}
_LOCAL_AI_TOOLS_TEXT_KEYS = {
    "recent_headlines",
    "headlines",
}
_LOCAL_AI_TOOLS_MAX_SEQUENCE_LENGTH = 80
_LOCAL_AI_TOOLS_MAX_TEXT_ITEMS = 12
_LOCAL_AI_TOOLS_MAX_TEXT_CHARS = 220
_LOCAL_AI_TOOLS_SHADOW_READ_PAGE_SIZE = 500
_TRAINING_LOCK_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "local_ai_tools_training.lock"
)
_LOCAL_AI_TOOLS_SHADOW_TOOL_KEYS = {
    "available",
    "status",
    "model",
    "route_mode",
    "fallback_reason",
    "best_side",
    "side",
    "direction",
    "expected_move_pct",
    "loss_probability",
    "profit_quality_score",
    "confidence",
    "specialist_inference_active",
    "specialist_primary_model",
    "specialist_challenger_model",
    "specialist_artifacts_ready",
    "timesfm_shadow_expected_return_pct",
    "timesfm_shadow_expected_move_pct",
    "timesfm_shadow_side",
    "timesfm_shadow_confidence",
    "timesfm_shadow_horizon_step",
    "chronos_shadow_expected_return_pct",
    "chronos_shadow_expected_move_pct",
    "chronos_shadow_side",
    "chronos_shadow_confidence",
    "chronos_shadow_horizon_step",
}
_LOCAL_AI_TOOLS_SHADOW_FEATURE_SNAPSHOT_KEYS = tuple(
    sorted(
        _LOCAL_AI_TOOLS_FEATURE_KEYS - {"symbol", "decision_confidence", "horizon_minutes"}
    )
)
_LOCAL_AI_TOOLS_SHADOW_FEATURE_COLUMN_PREFIX = "local_tools_feature__"
_LOCAL_AI_TOOLS_SHADOW_PROFESSIONAL_KEYS = {
    "kind",
    "primary_model",
    "challenger_model",
    "artifacts_ready",
    "actual_inference",
    "baseline_response",
    "baseline_model",
    "activation_blocker",
    "promotion_flow",
    "live_mutation",
}
def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_utc(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _snapshot(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _compact_numeric(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _compact_sequence(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    out: list[float] = []
    for item in value[-_LOCAL_AI_TOOLS_MAX_SEQUENCE_LENGTH:]:
        number = _compact_numeric(item)
        if number is not None:
            out.append(number)
    return out


def _compact_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value[-_LOCAL_AI_TOOLS_MAX_TEXT_ITEMS:]:
        text = str(item or "").strip()
        if text:
            out.append(text[:_LOCAL_AI_TOOLS_MAX_TEXT_CHARS])
    return out


def _compact_local_ai_tools_features(features: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in _LOCAL_AI_TOOLS_FEATURE_KEYS:
        if key not in features:
            continue
        if key == "symbol":
            symbol = str(features.get(key) or "").strip()
            if symbol:
                compact[key] = symbol[:40]
            continue
        number = _compact_numeric(features.get(key))
        if number is not None:
            compact[key] = number
    for key in _LOCAL_AI_TOOLS_SEQUENCE_KEYS:
        sequence = _compact_sequence(features.get(key))
        if sequence:
            compact[key] = sequence
    for key in _LOCAL_AI_TOOLS_TEXT_KEYS:
        texts = _compact_text_list(features.get(key))
        if texts:
            compact[key] = texts
    shadow = _compact_local_ai_tools_shadow(features.get("local_ai_tools_shadow"))
    if shadow:
        compact["local_ai_tools_shadow"] = shadow
    for contract_key in ("training_market_fact_contract", "training_label_contract"):
        contract = features.get(contract_key)
        if isinstance(contract, dict) and contract:
            compact[contract_key] = dict(contract)
    return compact


def _compact_shadow_scalar(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    number = _compact_numeric(value)
    if number is not None:
        return number
    if isinstance(value, str):
        return value.strip()[:160]
    return None


def _compact_professional_shadow(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    compact: dict[str, Any] = {}
    for key in _LOCAL_AI_TOOLS_SHADOW_PROFESSIONAL_KEYS:
        if key not in value:
            continue
        item = _compact_shadow_scalar(value.get(key))
        if item is not None:
            compact[key] = item

    def compact_shadow_result(result: Any) -> dict[str, Any]:
        if not isinstance(result, dict):
            return {}
        compact_result = {}
        for key in (
            "model",
            "actual_inference",
            "expected_return_pct",
            "expected_move_pct",
            "best_side",
            "direction",
            "confidence",
            "horizon_step",
            "sequence_length",
            "prediction_count",
        ):
            item = _compact_shadow_scalar(result.get(key))
            if item is not None:
                compact_result[key] = item
        return compact_result

    result = value.get("shadow_result")
    if isinstance(result, dict):
        compact_result = compact_shadow_result(result)
        if compact_result:
            compact["shadow_result"] = compact_result
    for key in ("primary_shadow_result", "challenger_shadow_result"):
        compact_result = compact_shadow_result(value.get(key))
        if compact_result:
            compact[key] = compact_result
    return compact


def _compact_local_ai_tools_shadow(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    compact: dict[str, Any] = {}
    status = _compact_shadow_scalar(value.get("status"))
    if status:
        compact["status"] = status
    for tool_name in (
        "profit_prediction",
        "time_series_prediction",
        "sentiment_analysis",
        "exit_advice",
    ):
        tool = value.get(tool_name)
        if not isinstance(tool, dict):
            continue
        item = {}
        for key in _LOCAL_AI_TOOLS_SHADOW_TOOL_KEYS:
            if key not in tool:
                continue
            scalar = _compact_shadow_scalar(tool.get(key))
            if scalar is not None:
                item[key] = scalar
        professional = _compact_professional_shadow(tool.get("professional_model_shadow"))
        if professional:
            item["professional_model_shadow"] = professional
        if item:
            compact[tool_name] = item
    return compact


def _normalize_base_url(raw_base_url: str) -> str:
    """Validate the configured local AI tools API base URL."""
    if not str(raw_base_url or "").strip():
        raise RuntimeError(
            "LOCAL_AI_TOOLS_API_BASE is empty; configure local_ai_tools_api_base "
            "or pass --base-url before training local AI tools."
        )
    try:
        return normalize_http_base_url(
            raw_base_url,
            field_name="LOCAL_AI_TOOLS_API_BASE",
        )
    except ValueError as exc:
        raise RuntimeError(safe_error_text(exc)) from exc


def _build_auth_headers(api_key: str | None = None) -> dict[str, str]:
    key = str(settings.local_ai_tools_api_key if api_key is None else api_key or "").strip()
    if not key:
        return {}
    return {"Authorization": f"Bearer {key}"}


def _response_error_excerpt(response: httpx.Response) -> str:
    return safe_response_error_text(response, limit=_ERROR_EXCERPT_LIMIT)


def _raise_for_training_response(response: httpx.Response) -> None:
    if response.is_success:
        return

    detail = _response_error_excerpt(response)
    if response.status_code in _AUTH_FAILURE_STATUS_CODES:
        message = (
            f"Local AI tools training request was rejected with HTTP {response.status_code}. "
            "Check that LOCAL_AI_TOOLS_API_KEY in /data/trade_ai/local_ai_tools.env "
            "matches local_ai_tools_api_key on this app side. The key itself is never printed."
        )
    else:
        message = f"Local AI tools training request failed with HTTP {response.status_code}."

    if detail:
        message = f"{message} Service response: {detail}"
    raise RuntimeError(message)


async def _post_training_payload(
    base_url: str,
    payload: dict[str, Any],
    *,
    request_timeout: float,
    auth_token: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    normalized_base_url = _normalize_base_url(base_url)
    headers = _build_auth_headers(auth_token)
    timeout = httpx.Timeout(
        connect=request_timeout,
        read=None,
        write=None,
        pool=request_timeout,
    )
    try:
        async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
            response = await client.post(
                f"{normalized_base_url}/train",
                json=payload,
                headers=headers,
            )
    except httpx.RequestError as exc:
        raise RuntimeError(
            f"Local AI tools training request could not reach the service: {safe_error_text(exc)}"
        ) from exc

    _raise_for_training_response(response)
    try:
        parsed = response.json()
    except ValueError as exc:
        raise RuntimeError("Local AI tools training response was not valid JSON.") from exc
    return dict(parsed) if isinstance(parsed, Mapping) else {"value": parsed}


def _shadow_sample_columns() -> tuple[Any, ...]:
    return (
        ShadowBacktest.id,
        ShadowBacktest.decision_id,
        ShadowBacktest.symbol,
        ShadowBacktest.analysis_type,
        ShadowBacktest.decision_action,
        ShadowBacktest.decision_confidence,
        ShadowBacktest.due_at,
        ShadowBacktest.horizon_minutes,
        ShadowBacktest.label_version,
        ShadowBacktest.long_return_pct,
        ShadowBacktest.short_return_pct,
        ShadowBacktest.best_action,
        ShadowBacktest.missed_opportunity,
        ShadowBacktest.training_feature_snapshot,
    )


def _shadow_sample_from_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    features = _snapshot(mapping.get("training_feature_snapshot"))
    due_at = mapping.get("due_at")
    return {
        "id": int(mapping.get("id") or 0),
        "decision_id": int(mapping.get("decision_id") or 0) or None,
        "label_version": str(mapping.get("label_version") or ""),
        "symbol": str(mapping.get("symbol") or ""),
        "analysis_type": str(mapping.get("analysis_type") or ""),
        "decision_action": str(mapping.get("decision_action") or ""),
        "decision_confidence": _as_float(mapping.get("decision_confidence")),
        "horizon_minutes": int(mapping.get("horizon_minutes") or 10),
        "features": features,
        "long_return_pct": _as_float(mapping.get("long_return_pct")),
        "short_return_pct": _as_float(mapping.get("short_return_pct")),
        "label_timestamp": due_at.isoformat() if isinstance(due_at, datetime) else None,
        "best_action": mapping.get("best_action"),
        "missed_opportunity": bool(mapping.get("missed_opportunity")),
    }


async def _load_shadow_samples() -> list[dict[str, Any]]:
    before_id: int | None = None
    samples: list[dict[str, Any]] = []
    epoch_start = load_training_epoch_start()
    while True:
        page_limit = _LOCAL_AI_TOOLS_SHADOW_READ_PAGE_SIZE
        async with get_read_session_ctx() as session:
            stmt = (
                select(*_shadow_sample_columns())
                .where(
                    ShadowBacktest.status == "completed",
                    ShadowBacktest.created_at >= epoch_start,
                    ShadowBacktest.long_return_pct.is_not(None),
                    ShadowBacktest.short_return_pct.is_not(None),
                )
                .order_by(ShadowBacktest.id.desc())
                .limit(page_limit)
            )
            if before_id is not None:
                stmt = stmt.where(ShadowBacktest.id < before_id)
            rows = [_shadow_sample_from_mapping(row) for row in (await session.execute(stmt)).mappings().all()]
        if not rows:
            break
        before_id = int(rows[-1].get("id") or 0) or before_id
        for row in rows:
            features = _snapshot(row.get("features"))
            if not features:
                continue
            features.setdefault("symbol", row.get("symbol"))
            features.setdefault("decision_confidence", _as_float(row.get("decision_confidence")))
            features.setdefault("horizon_minutes", int(row.get("horizon_minutes") or 10))
            fee_pct, _fee_source = round_trip_fee_pct(features)
            if fee_pct > 0:
                features["round_trip_fee_pct"] = fee_pct
            compact_features = _compact_local_ai_tools_features(features)
            if not compact_features:
                continue
            samples.append(
                {
                    "id": int(row.get("id") or 0),
                    "decision_id": int(row.get("decision_id") or 0) or None,
                    "label_version": str(row.get("label_version") or ""),
                    "symbol": row.get("symbol"),
                    "analysis_type": row.get("analysis_type"),
                    "decision_action": row.get("decision_action"),
                    "decision_confidence": _as_float(row.get("decision_confidence")),
                    "horizon_minutes": int(row.get("horizon_minutes") or 10),
                    "features": compact_features,
                    "model_shadow_action": str(
                        features.get("model_shadow_action") or ""
                    ).lower(),
                    "long_return_pct": _as_float(row.get("long_return_pct")),
                    "short_return_pct": _as_float(row.get("short_return_pct")),
                    "label_timestamp": row.get("label_timestamp"),
                    "best_action": row.get("best_action"),
                    "missed_opportunity": bool(row.get("missed_opportunity")),
                }
            )
        if len(rows) < page_limit:
            break
    samples.reverse()
    return samples


async def _load_trade_samples() -> list[dict[str, Any]]:
    """Load the only trainable realized-trade source."""

    return await load_authoritative_trade_outcomes(since=load_training_epoch_start())


async def _completed_shadow_sample_count() -> int:
    epoch_start = load_training_epoch_start()
    async with get_session_ctx() as session:
        result = await session.execute(
            select(func.count(ShadowBacktest.id)).where(
                ShadowBacktest.status == "completed",
                ShadowBacktest.created_at >= epoch_start,
                ShadowBacktest.long_return_pct.is_not(None),
                ShadowBacktest.short_return_pct.is_not(None),
            )
        )
        return int(result.scalar() or 0)


async def _completed_trade_sample_count() -> int:
    """Return the cumulative clean trade sample cursor for local AI training.

    Closed trade facts are preserved as raw audit history, so there is no durable
    `quarantined` row status to count. The training cursor must therefore be
    computed from the same clean view that is sent to the model server.
    """

    trade_samples = await _load_trade_samples()
    payload = annotate_training_payload(
        shadow_samples=[],
        trade_samples=trade_samples,
        sequence_samples=[],
        text_sentiment_samples=[],
    )
    return len(payload["trade_samples"])


async def _load_sequence_samples() -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    current_key: tuple[str, str] | None = None
    closes: list[float] = []
    volumes: list[float] = []
    first_open_time: datetime | None = None
    last_open_time: datetime | None = None

    def flush_series() -> None:
        nonlocal closes, volumes, first_open_time, last_open_time
        if current_key is None:
            return
        observation_count = max(len(closes) - 31, 0)
        if observation_count > 0:
            symbol, timeframe = current_key
            samples.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "sequence_format": COMPACT_SEQUENCE_SERIES_FORMAT,
                    "first_open_time": (
                        first_open_time.isoformat() if first_open_time else None
                    ),
                    "last_open_time": (
                        last_open_time.isoformat() if last_open_time else None
                    ),
                    "close_sequence": closes,
                    "volume_sequence": volumes,
                    "observation_count": observation_count,
                    "label_name": "gross_market_move_pct",
                    "label_version": "2026-07-12.observation-only.v1",
                    "production_eligible": False,
                }
            )
        closes = []
        volumes = []
        first_open_time = None
        last_open_time = None

    async with get_read_session_ctx() as session:
        epoch_start = load_training_epoch_start()
        stmt = (
            select(
                Kline.symbol,
                Kline.timeframe,
                Kline.open_time,
                Kline.close,
                Kline.volume,
            )
            .where(Kline.timeframe.in_(("1m", "5m", "15m", "1h")))
            .where(Kline.open_time >= epoch_start)
            .order_by(Kline.symbol.asc(), Kline.timeframe.asc(), Kline.open_time.asc())
        )
        result = await session.stream(stmt)
        async for row in result.mappings():
            key = (str(row.get("symbol") or ""), str(row.get("timeframe") or ""))
            if current_key is not None and key != current_key:
                flush_series()
            if key != current_key:
                current_key = key
            open_time = row.get("open_time")
            if isinstance(open_time, datetime):
                first_open_time = first_open_time or open_time
                last_open_time = open_time
            closes.append(_as_float(row.get("close")))
            volumes.append(_as_float(row.get("volume")))
    flush_series()
    return samples


def _lock_file(handle: TextIO) -> None:
    """Acquire the training process lock without a polling threshold."""

    if os.name == "nt":
        msvcrt = importlib.import_module("msvcrt")
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return
    fcntl = importlib.import_module("fcntl")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _try_acquire_training_lock() -> TextIO | None:
    """Prevent manual and scheduled rebuilds from expanding the same data twice."""

    _TRAINING_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    handle = open(_TRAINING_LOCK_PATH, "a+", encoding="utf-8")
    try:
        _lock_file(handle)
    except OSError:
        handle.close()
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def _symbols_from_json(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if isinstance(value, dict):
        values = value.get("symbols") or value.get("items") or value.get("mentioned") or []
        if isinstance(values, list):
            return [str(item) for item in values if item]
        return [str(k) for k, v in value.items() if v]
    return []


async def _load_text_sentiment_samples() -> list[dict[str, Any]]:
    epoch_start = load_training_epoch_start()
    async with get_session_ctx() as session:
        news_stmt = select(NewsArticle).order_by(
            NewsArticle.published_at.desc().nullslast(), NewsArticle.id.desc()
        )
        social_stmt = select(SocialPost).order_by(
            SocialPost.posted_at.desc().nullslast(), SocialPost.id.desc()
        )
        news_stmt = news_stmt.where(NewsArticle.published_at >= epoch_start)
        social_stmt = social_stmt.where(SocialPost.posted_at >= epoch_start)
        news_result = await session.execute(news_stmt)
        social_result = await session.execute(social_stmt)
        news_rows = list(news_result.scalars().all())
        social_rows = list(social_result.scalars().all())

    samples: list[dict[str, Any]] = []
    for news_row in news_rows:
        text = " ".join(part for part in [news_row.title, news_row.summary] if part)
        if not text.strip():
            continue
        samples.append(
            {
                "source": "news",
                "platform": news_row.source,
                "text": text[:1200],
                "sentiment_score": _as_float(news_row.sentiment_score),
                "symbols": _symbols_from_json(news_row.symbols_mentioned),
                "created_at": news_row.published_at.isoformat() if news_row.published_at else None,
            }
        )
    for social_row in social_rows:
        text = str(social_row.content or "").strip()
        if not text:
            continue
        samples.append(
            {
                "source": "social",
                "platform": social_row.platform,
                "text": text[:1200],
                "sentiment_score": _as_float(social_row.sentiment_score),
                "engagement_count": int(social_row.engagement_count or 0),
                "symbols": _symbols_from_json(social_row.symbols),
                "created_at": social_row.posted_at.isoformat() if social_row.posted_at else None,
            }
        )
    return samples


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Train server-side local AI quant tools")
    parser.add_argument("--base-url", default=settings.local_ai_tools_api_base)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--skip-quarantine", action="store_true")
    parser.add_argument(
        "--training-mode",
        choices=("shadow", "formal", "walk_forward"),
        default="shadow",
        help="Phase-3 model-factory mode; default is shadow and never mutates live routing.",
    )
    parser.add_argument(
        "--persist-artifact",
        action="store_true",
        help="Allow the remote local_ai_tools service to write its model bundle.",
    )
    parser.add_argument(
        "--confirm-phase3-rebuild",
        action="store_true",
        help="Required together with --persist-artifact for a Phase 3 bundle rebuild.",
    )
    args = parser.parse_args()
    if args.persist_artifact and not args.confirm_phase3_rebuild:
        raise SystemExit("--persist-artifact requires --confirm-phase3-rebuild")
    okx_gate = okx_training_refresh_gate()
    if not bool(okx_gate.get("allowed")):
        raise SystemExit(
            "OKX daily reconciliation blocks local AI tools training refresh: "
            f"{okx_gate.get('reason')}"
        )

    quarantine_result = {
        "skipped": True,
        "reason": "skip_quarantine flag enabled",
    }
    if not args.persist_artifact:
        quarantine_result = {
            "skipped": True,
            "reason": "phase3_preflight_no_quarantine_writes",
        }
    elif not args.skip_quarantine:
        quarantine_result = await quarantine_dirty_shadow_samples()

    shadow_samples = await _load_shadow_samples()
    trade_samples = await _load_trade_samples()
    sequence_samples = await _load_sequence_samples()
    text_sentiment_samples = await _load_text_sentiment_samples()
    training_payload = annotate_training_payload(
        shadow_samples=shadow_samples,
        trade_samples=trade_samples,
        sequence_samples=sequence_samples,
        text_sentiment_samples=text_sentiment_samples,
    )
    training_payload["governance_report"] = artifact_bound_governance_report(
        training_payload["quality_report"],
        persist_artifact=bool(args.persist_artifact),
    )
    label_consistency = training_payload["quality_report"].get(
        "training_label_consistency", {}
    )
    if label_consistency.get("promotion_blocked"):
        raise SystemExit(
            "Training labels failed algebra consistency checks: "
            + json.dumps(label_consistency, ensure_ascii=False)
        )
    completed_shadow_count = await _completed_shadow_sample_count()
    completed_trade_count = await _completed_trade_sample_count()
    raw_trade_sample_count = len(trade_samples)
    trainable_trade_sample_count = len(training_payload["trade_samples"])
    quarantined_trade_sample_count = max(raw_trade_sample_count - trainable_trade_sample_count, 0)
    paper_observation_report = load_latest_paper_observation_report()
    return_objective_report = build_return_objective_report(
        trade_samples=training_payload["trade_samples"],
        shadow_samples=training_payload["shadow_samples"],
    )

    payload = {
        "source": "local_trading_system",
        "shadow_samples": training_payload["shadow_samples"],
        "trade_samples": training_payload["trade_samples"],
        "sequence_samples": training_payload["sequence_samples"],
        "text_sentiment_samples": training_payload["text_sentiment_samples"],
        "completed_shadow_sample_count": completed_shadow_count,
        "completed_trade_sample_count": completed_trade_count,
        "raw_trade_sample_count": raw_trade_sample_count,
        "trainable_trade_sample_count": trainable_trade_sample_count,
        "quarantined_trade_sample_count": quarantined_trade_sample_count,
        "trade_sample_cursor_policy": "clean_training_view_only",
        "training_quarantine": quarantine_result,
        "quality_report": training_payload["quality_report"],
        "governance_report": training_payload["governance_report"],
        "training_mode": args.training_mode,
        "persist_artifact": bool(args.persist_artifact),
        "confirm_phase3_rebuild": bool(args.confirm_phase3_rebuild),
        "okx_daily_reconciliation_gate": okx_gate,
        "paper_observation_report": paper_observation_report,
        "return_objective_report": return_objective_report,
        "profit_supervision_report": training_payload["quality_report"].get(
            "profit_supervision",
            {},
        ),
    }
    payload["promotion_recommendation"] = build_phase3_promotion_recommendation(
        training_mode=args.training_mode,
        quality_report=training_payload["quality_report"],
        governance_report=training_payload["governance_report"],
        paper_observation_report=paper_observation_report,
        completed_shadow_sample_count=completed_shadow_count,
        completed_trade_sample_count=completed_trade_count,
        return_objective_report=return_objective_report,
    )
    result = await _post_training_payload(
        args.base_url,
        payload,
        request_timeout=args.timeout,
    )
    safe_print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    training_lock = _try_acquire_training_lock()
    if training_lock is None:
        safe_print(
            json.dumps(
                {
                    "trained": False,
                    "reason": "local_ai_tools_training_already_running",
                    "training_process_isolated": True,
                },
                ensure_ascii=False,
            )
        )
    else:
        try:
            asyncio.run(_main())
        finally:
            training_lock.close()
