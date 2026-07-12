"""Train the server-side local quant tools from local trading history."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import func, select

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import settings
from core.safe_output import safe_error_text, safe_print, safe_response_error_text
from core.url_safety import normalize_http_base_url
from db.session import get_read_session_ctx, get_session_ctx
from models.decision import AIDecision
from models.learning import ShadowBacktest, TradeReflection
from models.market_data import Kline
from models.news import NewsArticle, SocialPost
from models.trade import OkxPositionHistory, Order, Position
from services.model_promotion_policy import (
    build_phase3_promotion_recommendation,
    build_profit_first_promotion_report,
    load_latest_paper_observation_report,
)
from services.okx_training_facts import build_okx_history_training_sample
from services.okx_training_gate import okx_training_refresh_gate
from services.phase3_boundary import PHASE3_CLEAN_START_UTC
from services.shadow_training_quarantine import quarantine_dirty_shadow_samples
from services.trading_params import DEFAULT_TRADING_PARAMS
from services.training_data_quality import (
    annotate_training_payload,
    artifact_bound_governance_report,
)

_AUTH_FAILURE_STATUS_CODES = {401, 403}
_ERROR_EXCERPT_LIMIT = 700
_LOCAL_ML_TRAINING_PARAMS = DEFAULT_TRADING_PARAMS.local_ml_training
_LOCAL_AI_ROUND_TRIP_COST_PCT = (
    DEFAULT_TRADING_PARAMS.execution_cost.local_ml_round_trip_cost_pct
)
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
_LOCAL_AI_TOOLS_SHADOW_TOOL_KEYS = {
    "available",
    "status",
    "model",
    "route_mode",
    "fallback_reason",
    "best_side",
    "side",
    "direction",
    "expected_return_pct",
    "expected_move_pct",
    "adjusted_expected_return_pct",
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
_TRAINING_REPAIR_SOURCE_MARKERS = ("repair", "correction", "backfill")
_TRAINING_REPAIR_SOURCES = {
    "missing_closed_position_repair",
    "okx_native_full_close_fill_correction",
    "okx_order_pair_repair",
    "okx_orphan_position_quarantine",
    "okx_position_link_repair",
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


def _trade_reflection_repair_source(reflection: TradeReflection) -> str:
    reflection_source = _text(getattr(reflection, "source", None)).lower()
    lessons = _snapshot(getattr(reflection, "expert_lessons", None))
    lesson_source = _text(lessons.get("source")).lower()
    for candidate in (lesson_source, reflection_source):
        if candidate in _TRAINING_REPAIR_SOURCES:
            return candidate
    for candidate in (lesson_source, reflection_source):
        if candidate and any(token in candidate for token in _TRAINING_REPAIR_SOURCE_MARKERS):
            return candidate
    return ""


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
    try:
        async with httpx.AsyncClient(timeout=request_timeout, transport=transport) as client:
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
        ShadowBacktest.symbol,
        ShadowBacktest.analysis_type,
        ShadowBacktest.decision_action,
        ShadowBacktest.decision_confidence,
        ShadowBacktest.due_at,
        ShadowBacktest.horizon_minutes,
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


async def _load_shadow_samples(limit: int) -> list[dict[str, Any]]:
    remaining = max(int(limit), 1)
    before_id: int | None = None
    samples: list[dict[str, Any]] = []
    while remaining > 0:
        page_limit = min(_LOCAL_AI_TOOLS_SHADOW_READ_PAGE_SIZE, remaining)
        async with get_read_session_ctx() as session:
            stmt = (
                select(*_shadow_sample_columns())
                .where(
                    ShadowBacktest.status == "completed",
                    ShadowBacktest.created_at >= PHASE3_CLEAN_START_UTC,
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
        remaining -= len(rows)
        before_id = int(rows[-1].get("id") or 0) or before_id
        for row in rows:
            features = _snapshot(row.get("features"))
            if not features:
                continue
            features.setdefault("symbol", row.get("symbol"))
            features.setdefault("decision_confidence", _as_float(row.get("decision_confidence")))
            features.setdefault("horizon_minutes", int(row.get("horizon_minutes") or 10))
            compact_features = _compact_local_ai_tools_features(features)
            if not compact_features:
                continue
            samples.append(
                {
                    "id": int(row.get("id") or 0),
                    "symbol": row.get("symbol"),
                    "analysis_type": row.get("analysis_type"),
                    "decision_action": row.get("decision_action"),
                    "decision_confidence": _as_float(row.get("decision_confidence")),
                    "horizon_minutes": int(row.get("horizon_minutes") or 10),
                    "features": compact_features,
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


async def _load_trade_reflection_samples(limit: int | None) -> list[dict[str, Any]]:
    async with get_session_ctx() as session:
        stmt = select(TradeReflection).order_by(TradeReflection.id.desc())
        if limit is not None:
            stmt = stmt.limit(max(int(limit), 1))
        result = await session.execute(stmt)
        rows = list(result.scalars().all())

    samples: list[dict[str, Any]] = []
    for row in rows:
        repair_source = _trade_reflection_repair_source(row)
        samples.append(
            {
                "source": "trade_reflection",
                "id": int(row.id or 0),
                "position_id": int(row.position_id or 0),
                "model_name": row.model_name,
                "execution_mode": row.execution_mode,
                "symbol": row.symbol,
                "side": row.side,
                "entry_price": _as_float(row.entry_price),
                "exit_price": _as_float(row.exit_price),
                "quantity": _as_float(row.quantity),
                "realized_pnl": _as_float(row.realized_pnl),
                "fee_estimate": _as_float(row.fee_estimate),
                "hold_minutes": _as_float(row.hold_minutes),
                "outcome": row.outcome,
                "reflection_source": _text(row.source),
                "trade_fact_repair_source": repair_source,
            }
        )
    samples.reverse()
    return samples


async def _decision_raw_by_position_id(position_ids: set[int]) -> dict[int, dict[str, Any]]:
    if not position_ids:
        return {}
    async with get_session_ctx() as session:
        position_result = await session.execute(
            select(Position).where(Position.id.in_(sorted(position_ids)))
        )
        positions = list(position_result.scalars().all())
        symbols = {str(row.symbol or "").strip() for row in positions if str(row.symbol or "").strip()}
        orders: list[Order] = []
        if symbols:
            order_result = await session.execute(
                select(Order)
                .where(Order.symbol.in_(sorted(symbols)))
                .order_by(Order.filled_at.desc(), Order.created_at.desc(), Order.id.desc())
            )
            orders = list(order_result.scalars().all())
        decision_ids = {
            int(getattr(order, "decision_id", 0) or 0)
            for order in orders
            if int(getattr(order, "decision_id", 0) or 0) > 0
        }
        decisions_by_id: dict[int, Any] = {}
        if decision_ids:
            decision_result = await session.execute(
                select(AIDecision).where(AIDecision.id.in_(sorted(decision_ids)))
            )
            decisions_by_id = {
                int(decision.id): decision for decision in decision_result.scalars().all()
            }
    order_list = list(orders)
    raw_by_position_id: dict[int, dict[str, Any]] = {}
    for position in positions:
        decision = _match_entry_decision_for_training(position, order_list, decisions_by_id)
        if decision is None or not isinstance(getattr(decision, "raw_llm_response", None), dict):
            continue
        raw_by_position_id[int(position.id)] = dict(decision.raw_llm_response)
    return raw_by_position_id


def _match_entry_decision_for_training(
    position: Position,
    orders: list[Order],
    decisions_by_id: dict[int, Any],
) -> Any | None:
    position_symbol = str(getattr(position, "symbol", "") or "").strip()
    position_side = str(getattr(position, "side", "") or "").lower().strip()
    position_created = _as_utc(getattr(position, "created_at", None))
    entry_exchange_order_id = str(getattr(position, "entry_exchange_order_id", "") or "").strip()
    best: Any | None = None
    best_delta: float | None = None
    for order in orders:
        if position_symbol and str(getattr(order, "symbol", "") or "").strip() != position_symbol:
            continue
        decision = decisions_by_id.get(int(getattr(order, "decision_id", 0) or 0))
        if decision is None:
            continue
        action = str(getattr(decision, "action", "") or "").lower()
        if action not in {"long", "short"} or action != position_side:
            continue
        order_exchange_id = str(getattr(order, "exchange_order_id", "") or "").strip()
        if entry_exchange_order_id and order_exchange_id == entry_exchange_order_id:
            return decision
        order_time = _as_utc(getattr(order, "filled_at", None) or getattr(order, "created_at", None))
        if position_created is not None and order_time is not None:
            delta = abs((position_created - order_time).total_seconds())
            if delta > 15 * 60:
                continue
        else:
            delta = 0.0
        if best_delta is None or delta < best_delta:
            best = decision
            best_delta = delta
    return best


async def _load_authoritative_trade_samples(limit: int | None) -> list[dict[str, Any]]:
    """Load one training sample per mirrored OKX positions-history lifecycle."""

    async with get_session_ctx() as session:
        stmt = (
            select(OkxPositionHistory)
            .order_by(
                OkxPositionHistory.updated_at_okx.desc().nullslast(),
                OkxPositionHistory.id.desc(),
            )
        )
        if limit is not None:
            stmt = stmt.limit(max(int(limit), 1))
        records = list((await session.execute(stmt)).scalars().all())
        position_ids = {
            int(value)
            for record in records
            for value in (record.position_ids or [])
            if str(value or "").isdigit() and int(value) > 0
        }
        order_ids = {
            str(value or "").strip()
            for record in records
            for value in [
                *(record.entry_order_ids or []),
                *(record.close_order_ids or []),
                *(record.linked_order_ids or []),
            ]
            if str(value or "").strip()
        }
        positions_by_id: dict[int, Position] = {}
        if position_ids:
            position_rows = await session.execute(
                select(Position).where(Position.id.in_(sorted(position_ids)))
            )
            positions_by_id = {
                int(position.id): position for position in position_rows.scalars().all()
            }
        orders_by_exchange_id: dict[str, Order] = {}
        decision_raw_by_order_id: dict[str, dict[str, Any]] = {}
        if order_ids:
            order_rows = await session.execute(
                select(Order).where(Order.exchange_order_id.in_(sorted(order_ids)))
            )
            loaded_orders = list(order_rows.scalars().all())
            orders_by_exchange_id = {
                str(order.exchange_order_id): order
                for order in loaded_orders
                if str(order.exchange_order_id or "").strip()
            }
            decision_ids = {
                int(order.decision_id or 0)
                for order in loaded_orders
                if int(order.decision_id or 0) > 0
            }
            decisions_by_id: dict[int, AIDecision] = {}
            if decision_ids:
                decision_rows = await session.execute(
                    select(AIDecision).where(AIDecision.id.in_(sorted(decision_ids)))
                )
                decisions_by_id = {
                    int(decision.id): decision for decision in decision_rows.scalars().all()
                }
            for order in loaded_orders:
                exchange_id = str(order.exchange_order_id or "").strip()
                decision = decisions_by_id.get(int(order.decision_id or 0))
                raw = getattr(decision, "raw_llm_response", None)
                if exchange_id and isinstance(raw, dict) and raw:
                    decision_raw_by_order_id[exchange_id] = dict(raw)

    decision_raw_by_position_id = await _decision_raw_by_position_id(position_ids)
    samples = [
        build_okx_history_training_sample(
            record,
            positions_by_id=positions_by_id,
            orders_by_exchange_id=orders_by_exchange_id,
            decision_raw_by_position_id=decision_raw_by_position_id,
            decision_raw_by_order_id=decision_raw_by_order_id,
        )
        for record in records
    ]
    samples.reverse()
    return samples


def _deep_merge_trade_sample_dict(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if key not in merged or merged.get(key) in (None, "", [], {}):
            merged[key] = value
            continue
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_trade_sample_dict(existing, value)
    return merged


def _merge_trade_samples(
    reflection_samples: list[dict[str, Any]],
    authoritative_samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach reflection features to official lifecycle samples without duplicating labels."""

    merged = [
        dict(sample)
        for sample in authoritative_samples
        if str(sample.get("source") or "").strip() == "okx_position_history"
        and str(sample.get("lifecycle_key") or "").strip()
    ]
    index_by_position_id: dict[int, int] = {}
    for index, sample in enumerate(merged):
        position_ids = sample.get("position_ids") or [sample.get("position_id")]
        for value in position_ids:
            try:
                position_id = int(value or 0)
            except (TypeError, ValueError):
                continue
            if position_id > 0:
                index_by_position_id[position_id] = index
    for reflection in reflection_samples:
        repair_source = str(reflection.get("trade_fact_repair_source") or "").strip().lower()
        reflection_source = str(reflection.get("reflection_source") or "").strip().lower()
        if any(
            candidate in _TRAINING_REPAIR_SOURCES
            or any(token in candidate for token in _TRAINING_REPAIR_SOURCE_MARKERS)
            for candidate in (repair_source, reflection_source)
            if candidate
        ):
            continue
        position_id = int(reflection.get("position_id") or 0)
        index = index_by_position_id.get(position_id)
        if index is None:
            continue
        official = merged[index]
        combined = _deep_merge_trade_sample_dict(official, reflection)
        combined["source"] = "okx_position_history"
        combined["realized_pnl"] = official.get("realized_pnl")
        combined["fee_estimate"] = official.get("fee_estimate")
        combined["funding_fee"] = official.get("funding_fee")
        combined["pnl_source"] = official.get("pnl_source")
        combined["trade_fact_trusted"] = official.get("trade_fact_trusted")
        combined["trade_fact_trust_reason"] = official.get("trade_fact_trust_reason")
        merged[index] = combined
    return merged


async def _completed_shadow_sample_count() -> int:
    async with get_session_ctx() as session:
        result = await session.execute(
            select(func.count(ShadowBacktest.id)).where(
                ShadowBacktest.status == "completed",
                ShadowBacktest.created_at >= PHASE3_CLEAN_START_UTC,
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

    reflection_samples = await _load_trade_reflection_samples(None)
    authoritative_samples = await _load_authoritative_trade_samples(None)
    payload = annotate_training_payload(
        shadow_samples=[],
        trade_samples=_merge_trade_samples(reflection_samples, authoritative_samples),
        sequence_samples=[],
        text_sentiment_samples=[],
    )
    return len(payload["trade_samples"])


async def _load_sequence_samples(limit: int) -> list[dict[str, Any]]:
    row_limit = max(int(limit), 1)
    async with get_session_ctx() as session:
        result = await session.execute(
            select(Kline)
            .where(Kline.timeframe.in_(("1m", "5m", "15m", "1h")))
            .order_by(Kline.symbol.asc(), Kline.timeframe.asc(), Kline.open_time.desc())
            .limit(row_limit)
        )
        rows = list(result.scalars().all())

    grouped: dict[tuple[str, str], list[Kline]] = {}
    for row in rows:
        grouped.setdefault((row.symbol, row.timeframe), []).append(row)

    samples: list[dict[str, Any]] = []
    for (symbol, timeframe), items in grouped.items():
        ordered = sorted(items, key=lambda r: r.open_time)
        if len(ordered) < 32:
            continue
        closes = [_as_float(r.close) for r in ordered]
        volumes = [_as_float(r.volume) for r in ordered]
        for idx in range(30, len(ordered) - 1):
            start = max(0, idx - 59)
            base = closes[start : idx + 1]
            if len(base) < 30 or base[-1] <= 0:
                continue
            future = closes[idx + 1]
            move_pct = (future - base[-1]) / base[-1] * 100.0
            feature_timestamp = ordered[idx].open_time
            label_timestamp = ordered[idx + 1].open_time
            samples.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "open_time": feature_timestamp.isoformat() if feature_timestamp else None,
                    "feature_timestamp": (
                        feature_timestamp.isoformat() if feature_timestamp else None
                    ),
                    "label_timestamp": label_timestamp.isoformat() if label_timestamp else None,
                    "close_sequence": base,
                    "volume_sequence": volumes[start : idx + 1],
                    "future_return_pct": move_pct,
                    "long_return_pct": move_pct - _LOCAL_AI_ROUND_TRIP_COST_PCT,
                    "short_return_pct": -move_pct - _LOCAL_AI_ROUND_TRIP_COST_PCT,
                    "label_name": "net_return_after_cost_pct",
                    "label_version": "2026-07-12.v1",
                }
            )
    return samples[-max(int(limit), 1) :]


def _symbols_from_json(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if isinstance(value, dict):
        values = value.get("symbols") or value.get("items") or value.get("mentioned") or []
        if isinstance(values, list):
            return [str(item) for item in values if item]
        return [str(k) for k, v in value.items() if v]
    return []


async def _load_text_sentiment_samples(limit: int) -> list[dict[str, Any]]:
    row_limit = max(int(limit), 1)
    async with get_session_ctx() as session:
        news_result = await session.execute(
            select(NewsArticle)
            .order_by(NewsArticle.published_at.desc().nullslast(), NewsArticle.id.desc())
            .limit(row_limit)
        )
        social_result = await session.execute(
            select(SocialPost)
            .order_by(SocialPost.posted_at.desc().nullslast(), SocialPost.id.desc())
            .limit(row_limit)
        )
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
    return samples[-row_limit:]


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Train server-side local AI quant tools")
    parser.add_argument("--base-url", default=settings.local_ai_tools_api_base)
    parser.add_argument(
        "--shadow-limit",
        type=int,
        default=_LOCAL_ML_TRAINING_PARAMS.training_shadow_sample_limit,
    )
    parser.add_argument(
        "--trade-limit",
        type=int,
        default=_LOCAL_ML_TRAINING_PARAMS.training_trade_sample_limit,
    )
    parser.add_argument(
        "--sequence-limit",
        type=int,
        default=_LOCAL_ML_TRAINING_PARAMS.training_sequence_sample_limit,
    )
    parser.add_argument(
        "--text-limit",
        type=int,
        default=_LOCAL_ML_TRAINING_PARAMS.training_text_sample_limit,
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--skip-quarantine", action="store_true")
    parser.add_argument(
        "--training-mode",
        choices=("shadow", "formal", "walk_forward"),
        default="shadow",
        help="Phase-3 model-factory mode; default is shadow and never mutates live routing.",
    )
    parser.add_argument(
        "--model-stage",
        choices=("shadow", "canary", "live", "degraded", "retired"),
        default="shadow",
        help="Lifecycle stage recorded in the local_ai_tools metadata.",
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
        quarantine_result = await quarantine_dirty_shadow_samples(
            batch_size=min(args.shadow_limit, 1000),
            max_batches=max((int(args.shadow_limit) + 999) // 1000, 1),
        )

    shadow_samples = await _load_shadow_samples(args.shadow_limit)
    trade_reflection_samples = await _load_trade_reflection_samples(args.trade_limit)
    authoritative_samples = await _load_authoritative_trade_samples(args.trade_limit)
    trade_samples = _merge_trade_samples(trade_reflection_samples, authoritative_samples)
    sequence_samples = await _load_sequence_samples(args.sequence_limit)
    text_sentiment_samples = await _load_text_sentiment_samples(args.text_limit)
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
    profit_first_report = build_profit_first_promotion_report(
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
        "model_stage": args.model_stage,
        "persist_artifact": bool(args.persist_artifact),
        "confirm_phase3_rebuild": bool(args.confirm_phase3_rebuild),
        "okx_daily_reconciliation_gate": okx_gate,
        "evaluation_policy": {
            "promotion_flow": "shadow_to_canary_to_live",
            "live_mutation": False,
            "requires_walk_forward": args.training_mode != "walk_forward",
            "requires_paper_observation": True,
            "phase": "phase3_model_factory",
        },
        "paper_observation_report": paper_observation_report,
        "profit_first_report": profit_first_report,
    }
    payload["promotion_recommendation"] = build_phase3_promotion_recommendation(
        training_mode=args.training_mode,
        model_stage=args.model_stage,
        quality_report=training_payload["quality_report"],
        governance_report=training_payload["governance_report"],
        evaluation_policy=payload["evaluation_policy"],
        paper_observation_report=paper_observation_report,
        completed_shadow_sample_count=completed_shadow_count,
        completed_trade_sample_count=completed_trade_count,
        profit_first_report=profit_first_report,
    )
    result = await _post_training_payload(
        args.base_url,
        payload,
        request_timeout=args.timeout,
    )
    safe_print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
