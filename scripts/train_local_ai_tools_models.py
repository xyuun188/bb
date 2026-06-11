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
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import settings
from core.safe_output import safe_error_text, safe_print, safe_response_error_text
from core.url_safety import normalize_http_base_url
from db.session import get_session_ctx
from models.learning import ShadowBacktest, TradeReflection
from models.market_data import Kline
from models.news import NewsArticle, SocialPost
from models.trade import Position

_AUTH_FAILURE_STATUS_CODES = {401, 403}
_ERROR_EXCERPT_LIMIT = 700


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


async def _load_shadow_samples(limit: int) -> list[dict[str, Any]]:
    async with get_session_ctx() as session:
        result = await session.execute(
            select(ShadowBacktest)
            .where(
                ShadowBacktest.status == "completed",
                ShadowBacktest.long_return_pct.is_not(None),
                ShadowBacktest.short_return_pct.is_not(None),
            )
            .order_by(ShadowBacktest.id.desc())
            .limit(max(int(limit), 1))
        )
        rows = list(result.scalars().all())

    samples: list[dict[str, Any]] = []
    for row in rows:
        features = _snapshot(row.feature_snapshot)
        if not features:
            continue
        features.setdefault("symbol", row.symbol)
        features.setdefault("decision_confidence", _as_float(row.decision_confidence))
        features.setdefault("horizon_minutes", int(row.horizon_minutes or 10))
        samples.append(
            {
                "id": int(row.id or 0),
                "symbol": row.symbol,
                "analysis_type": row.analysis_type,
                "decision_action": row.decision_action,
                "decision_confidence": _as_float(row.decision_confidence),
                "horizon_minutes": int(row.horizon_minutes or 10),
                "features": features,
                "long_return_pct": _as_float(row.long_return_pct),
                "short_return_pct": _as_float(row.short_return_pct),
                "best_action": row.best_action,
                "missed_opportunity": bool(row.missed_opportunity),
            }
        )
    samples.reverse()
    return samples


async def _load_trade_reflection_samples(limit: int) -> list[dict[str, Any]]:
    async with get_session_ctx() as session:
        result = await session.execute(
            select(TradeReflection).order_by(TradeReflection.id.desc()).limit(max(int(limit), 1))
        )
        rows = list(result.scalars().all())

    samples: list[dict[str, Any]] = []
    for row in rows:
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
            }
        )
    samples.reverse()
    return samples


async def _load_closed_position_samples(limit: int) -> list[dict[str, Any]]:
    async with get_session_ctx() as session:
        result = await session.execute(
            select(Position)
            .where(Position.is_open.is_(False), Position.closed_at.is_not(None))
            .order_by(Position.closed_at.desc(), Position.id.desc())
            .limit(max(int(limit), 1))
        )
        rows = list(result.scalars().all())

    samples: list[dict[str, Any]] = []
    for row in rows:
        opened = _as_utc(row.created_at)
        closed = _as_utc(row.closed_at)
        hold_minutes = 0.0
        if opened and closed:
            hold_minutes = max((closed - opened).total_seconds() / 60.0, 0.0)
        samples.append(
            {
                "source": "closed_position",
                "id": int(row.id or 0),
                "position_id": int(row.id or 0),
                "model_name": row.model_name,
                "execution_mode": row.execution_mode,
                "symbol": row.symbol,
                "side": row.side,
                "entry_price": _as_float(row.entry_price),
                "exit_price": _as_float(row.current_price),
                "quantity": _as_float(row.quantity),
                "realized_pnl": _as_float(row.realized_pnl),
                "hold_minutes": hold_minutes,
                "outcome": (
                    "profit"
                    if _as_float(row.realized_pnl) > 0
                    else "loss" if _as_float(row.realized_pnl) < 0 else "flat"
                ),
            }
        )
    samples.reverse()
    return samples


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
            samples.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "open_time": (
                        ordered[idx].open_time.isoformat() if ordered[idx].open_time else None
                    ),
                    "close_sequence": base,
                    "volume_sequence": volumes[start : idx + 1],
                    "future_return_pct": move_pct,
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
    parser.add_argument("--shadow-limit", type=int, default=20000)
    parser.add_argument("--trade-limit", type=int, default=8000)
    parser.add_argument("--sequence-limit", type=int, default=12000)
    parser.add_argument("--text-limit", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    shadow_samples = await _load_shadow_samples(args.shadow_limit)
    trade_samples = await _load_trade_reflection_samples(args.trade_limit)
    trade_samples.extend(await _load_closed_position_samples(args.trade_limit))
    sequence_samples = await _load_sequence_samples(args.sequence_limit)
    text_sentiment_samples = await _load_text_sentiment_samples(args.text_limit)

    payload = {
        "source": "local_trading_system",
        "shadow_samples": shadow_samples,
        "trade_samples": trade_samples,
        "sequence_samples": sequence_samples,
        "text_sentiment_samples": text_sentiment_samples,
    }
    result = await _post_training_payload(
        args.base_url,
        payload,
        request_timeout=args.timeout,
    )
    safe_print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
