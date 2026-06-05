"""Train the server-side local quant tools from local trading history."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import settings
from db.session import get_session_ctx
from models.learning import ShadowBacktest, TradeReflection
from models.market_data import Kline
from models.news import NewsArticle, SocialPost
from models.trade import Position


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
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


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
        samples.append({
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
        })
    samples.reverse()
    return samples


async def _load_trade_reflection_samples(limit: int) -> list[dict[str, Any]]:
    async with get_session_ctx() as session:
        result = await session.execute(
            select(TradeReflection)
            .order_by(TradeReflection.id.desc())
            .limit(max(int(limit), 1))
        )
        rows = list(result.scalars().all())

    samples: list[dict[str, Any]] = []
    for row in rows:
        samples.append({
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
        })
    samples.reverse()
    return samples


async def _load_closed_position_samples(limit: int) -> list[dict[str, Any]]:
    async with get_session_ctx() as session:
        result = await session.execute(
            select(Position)
            .where(Position.is_open == False, Position.closed_at.is_not(None))
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
        samples.append({
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
            "outcome": "profit" if _as_float(row.realized_pnl) > 0 else "loss" if _as_float(row.realized_pnl) < 0 else "flat",
        })
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
            base = closes[start: idx + 1]
            if len(base) < 30 or base[-1] <= 0:
                continue
            future = closes[idx + 1]
            move_pct = (future - base[-1]) / base[-1] * 100.0
            samples.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "open_time": ordered[idx].open_time.isoformat() if ordered[idx].open_time else None,
                "close_sequence": base,
                "volume_sequence": volumes[start: idx + 1],
                "future_return_pct": move_pct,
            })
    return samples[-max(int(limit), 1):]


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
    for row in news_rows:
        text = " ".join(part for part in [row.title, row.summary] if part)
        if not text.strip():
            continue
        samples.append({
            "source": "news",
            "platform": row.source,
            "text": text[:1200],
            "sentiment_score": _as_float(row.sentiment_score),
            "symbols": _symbols_from_json(row.symbols_mentioned),
            "created_at": row.published_at.isoformat() if row.published_at else None,
        })
    for row in social_rows:
        text = str(row.content or "").strip()
        if not text:
            continue
        samples.append({
            "source": "social",
            "platform": row.platform,
            "text": text[:1200],
            "sentiment_score": _as_float(row.sentiment_score),
            "engagement_count": int(row.engagement_count or 0),
            "symbols": _symbols_from_json(row.symbols),
            "created_at": row.posted_at.isoformat() if row.posted_at else None,
        })
    return samples[-row_limit:]


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Train server-side local AI quant tools")
    parser.add_argument("--base-url", default=settings.local_ai_tools_api_base.rstrip("/"))
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
    headers = {}
    if settings.local_ai_tools_api_key:
        headers["Authorization"] = f"Bearer {settings.local_ai_tools_api_key}"
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        response = await client.post(f"{args.base_url}/train", json=payload, headers=headers)
        response.raise_for_status()
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
