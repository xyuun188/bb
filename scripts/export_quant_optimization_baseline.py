from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.safe_output import safe_print  # noqa: E402
from db.session import get_read_session_ctx  # noqa: E402
from models.decision import AIDecision  # noqa: E402
from models.learning import ShadowBacktest, TradeReflection  # noqa: E402
from models.market_data import Kline, Ticker  # noqa: E402
from models.news import NewsArticle, SocialPost  # noqa: E402
from models.trade import Order, Position  # noqa: E402
from scripts.train_local_ai_tools_models import (  # noqa: E402
    _load_closed_position_samples,
    _load_sequence_samples,
    _load_shadow_samples,
    _load_text_sentiment_samples,
    _load_trade_reflection_samples,
    _merge_trade_samples,
)
from services.trading_params import DEFAULT_TRADING_PARAMS  # noqa: E402
from services.training_data_quality import annotate_training_payload  # noqa: E402

DEFAULT_HOURS = 24
_LOCAL_ML_TRAINING_PARAMS = DEFAULT_TRADING_PARAMS.local_ml_training
DEFAULT_QUALITY_SHADOW_LIMIT = min(
    3_000,
    _LOCAL_ML_TRAINING_PARAMS.training_shadow_sample_limit,
)
DEFAULT_QUALITY_TRADE_REFLECTION_LIMIT = min(
    1_200,
    _LOCAL_ML_TRAINING_PARAMS.training_trade_sample_limit,
)
DEFAULT_QUALITY_CLOSED_POSITION_LIMIT = min(
    1_200,
    _LOCAL_ML_TRAINING_PARAMS.training_trade_sample_limit,
)
DEFAULT_QUALITY_SEQUENCE_LIMIT = min(
    3_000,
    _LOCAL_ML_TRAINING_PARAMS.training_sequence_sample_limit,
)
DEFAULT_QUALITY_TEXT_LIMIT = min(
    1_500,
    _LOCAL_ML_TRAINING_PARAMS.training_text_sample_limit,
)
DEEP_QUALITY_SHADOW_LIMIT = _LOCAL_ML_TRAINING_PARAMS.training_shadow_sample_limit
DEEP_QUALITY_TRADE_REFLECTION_LIMIT = _LOCAL_ML_TRAINING_PARAMS.training_trade_sample_limit
DEEP_QUALITY_CLOSED_POSITION_LIMIT = _LOCAL_ML_TRAINING_PARAMS.training_trade_sample_limit
DEEP_QUALITY_SEQUENCE_LIMIT = _LOCAL_ML_TRAINING_PARAMS.training_sequence_sample_limit
DEEP_QUALITY_TEXT_LIMIT = _LOCAL_ML_TRAINING_PARAMS.training_text_sample_limit


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        result = float(value)
        return result if result == result else default
    except (TypeError, ValueError):
        return default


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _nested_float(payload: dict[str, Any], *paths: tuple[str, ...]) -> float | None:
    for path in paths:
        current: Any = payload
        for key in path:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if current is not None:
            value = _as_float(current, default=float("nan"))
            if value == value:
                return value
    return None


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)

    def pct(ratio: float) -> float:
        index = min(max(int((len(ordered) - 1) * ratio), 0), len(ordered) - 1)
        return round(ordered[index], 6)

    return {
        "count": len(values),
        "min": round(ordered[0], 6),
        "p25": pct(0.25),
        "median": pct(0.5),
        "p75": pct(0.75),
        "max": round(ordered[-1], 6),
        "avg": round(statistics.fmean(values), 6),
    }


def _extract_expected_net(decision: AIDecision) -> float | None:
    raw = _safe_dict(decision.raw_llm_response)
    return _nested_float(
        raw,
        ("opportunity_score", "expected_net_return_pct"),
        ("entry_candidate_evidence", "long", "expected_net_return_pct"),
        ("entry_candidate_evidence", "short", "expected_net_return_pct"),
        ("expected_net_breakdown", "expected_net_return_pct"),
    )


async def _group_counts(model: Any, column: Any, since: datetime | None = None) -> dict[str, int]:
    async with get_read_session_ctx() as session:
        stmt = select(column, func.count()).select_from(model).group_by(column)
        if since is not None:
            stmt = stmt.where(model.created_at >= since)
        result = await session.execute(stmt)
        return {str(key or "unknown"): int(count or 0) for key, count in result.all()}


async def _recent_decision_metrics(since: datetime, limit: int) -> dict[str, Any]:
    async with get_read_session_ctx() as session:
        rows = list(
            (
                await session.execute(
                    select(AIDecision)
                    .where(AIDecision.created_at >= since)
                    .order_by(AIDecision.created_at.desc(), AIDecision.id.desc())
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
    expected_net_values = [
        value for row in rows if (value := _extract_expected_net(row)) is not None
    ]
    action_counts = Counter(str(row.action or "unknown") for row in rows)
    analysis_counts = Counter(str(row.analysis_type or "unknown") for row in rows)
    executed_count = sum(1 for row in rows if bool(row.was_executed))
    pending_reason_count = sum(
        1
        for row in rows
        if "已进入执行队列" in str(row.execution_reason or "")
        or "正在提交 OKX" in str(row.execution_reason or "")
    )
    return {
        "sampled_recent_decision_count": len(rows),
        "action_counts": dict(action_counts),
        "analysis_type_counts": dict(analysis_counts),
        "executed_count": executed_count,
        "not_executed_count": max(len(rows) - executed_count, 0),
        "pending_or_queue_reason_count": pending_reason_count,
        "expected_net_distribution": _distribution(expected_net_values),
    }


async def _order_metrics(since: datetime) -> dict[str, Any]:
    status_counts = await _group_counts(Order, Order.status, since)
    async with get_read_session_ctx() as session:
        rows = list(
            (
                await session.execute(
                    select(Order)
                    .where(Order.created_at >= since)
                    .order_by(Order.created_at.desc(), Order.id.desc())
                    .limit(3000)
                )
            )
            .scalars()
            .all()
        )
    return {
        "status_counts": status_counts,
        "total_recent_orders": len(rows),
        "filled_count": sum(1 for row in rows if str(row.status).lower() == "filled"),
        "rejected_count": sum(1 for row in rows if str(row.status).lower() == "rejected"),
        "zero_quantity_count": sum(1 for row in rows if _as_float(row.quantity) <= 0),
    }


async def _position_metrics(since: datetime) -> dict[str, Any]:
    async with get_read_session_ctx() as session:
        rows = list(
            (
                await session.execute(
                    select(Position)
                    .where(Position.created_at >= since)
                    .order_by(Position.created_at.desc(), Position.id.desc())
                    .limit(3000)
                )
            )
            .scalars()
            .all()
        )
    notional_values = [abs(_as_float(row.quantity) * _as_float(row.entry_price)) for row in rows]
    fast_loss_count = 0
    for row in rows:
        if bool(row.is_open) or not row.closed_at or not row.created_at:
            continue
        hold_minutes = max((row.closed_at - row.created_at).total_seconds() / 60.0, 0.0)
        if hold_minutes < 10 and _as_float(row.realized_pnl) < 0:
            fast_loss_count += 1
    return {
        "total_recent_positions": len(rows),
        "open_count": sum(1 for row in rows if bool(row.is_open)),
        "closed_count": sum(1 for row in rows if not bool(row.is_open)),
        "side_counts": dict(Counter(str(row.side or "unknown") for row in rows)),
        "notional_distribution": _distribution(notional_values),
        "small_notional_under_10_count": sum(1 for value in notional_values if 0 < value < 10),
        "fast_loss_close_under_10m_count": fast_loss_count,
        "realized_pnl_distribution": _distribution([_as_float(row.realized_pnl) for row in rows]),
    }


async def _training_source_metrics(
    *,
    shadow_limit: int,
    trade_reflection_limit: int,
    closed_position_limit: int,
    sequence_limit: int,
    text_sentiment_limit: int,
) -> dict[str, Any]:
    async with get_read_session_ctx() as session:
        shadow_total = int(
            (
                await session.execute(
                    select(func.count(ShadowBacktest.id)).where(
                        ShadowBacktest.status == "completed"
                    )
                )
            ).scalar()
            or 0
        )
        shadow_action_rows = (
            await session.execute(
                select(ShadowBacktest.decision_action, func.count())
                .where(ShadowBacktest.status == "completed")
                .group_by(ShadowBacktest.decision_action)
            )
        ).all()
        trade_outcome_rows = (
            await session.execute(
                select(TradeReflection.outcome, func.count()).group_by(TradeReflection.outcome)
            )
        ).all()
        kline_rows = (
            await session.execute(select(Kline.timeframe, func.count()).group_by(Kline.timeframe))
        ).all()
        ticker_count = int((await session.execute(select(func.count(Ticker.id)))).scalar() or 0)
        news_count = int((await session.execute(select(func.count(NewsArticle.id)))).scalar() or 0)
        social_count = int((await session.execute(select(func.count(SocialPost.id)))).scalar() or 0)
    shadow_samples = await _load_shadow_samples(max(int(shadow_limit), 0))
    trade_reflection_samples = await _load_trade_reflection_samples(
        max(int(trade_reflection_limit), 0)
    )
    closed_position_samples = await _load_closed_position_samples(
        max(int(closed_position_limit), 0)
    )
    trade_samples = _merge_trade_samples(trade_reflection_samples, closed_position_samples)
    sequence_samples = await _load_sequence_samples(max(int(sequence_limit), 0))
    text_sentiment_samples = await _load_text_sentiment_samples(max(int(text_sentiment_limit), 0))
    payload = annotate_training_payload(
        shadow_samples=shadow_samples,
        trade_samples=trade_samples,
        sequence_samples=sequence_samples,
        text_sentiment_samples=text_sentiment_samples,
    )
    return {
        "completed_shadow_total": shadow_total,
        "shadow_action_counts": {
            str(key or "unknown"): int(count or 0) for key, count in shadow_action_rows
        },
        "trade_reflection_outcome_counts": {
            str(key or "unknown"): int(count or 0) for key, count in trade_outcome_rows
        },
        "kline_timeframe_counts": {
            str(key or "unknown"): int(count or 0) for key, count in kline_rows
        },
        "ticker_count": ticker_count,
        "news_count": news_count,
        "social_count": social_count,
        "quality_sample_limits": {
            "shadow": shadow_limit,
            "trade_reflection": trade_reflection_limit,
            "closed_position": closed_position_limit,
            "sequence": sequence_limit,
            "text_sentiment": text_sentiment_limit,
        },
        "quality_report": payload["quality_report"],
        "trainable_shadow_sample_count": len(payload["shadow_samples"]),
        "trainable_trade_sample_count": len(payload["trade_samples"]),
        "trainable_sequence_sample_count": len(payload["sequence_samples"]),
        "trainable_text_sentiment_sample_count": len(payload["text_sentiment_samples"]),
    }


async def build_baseline(
    hours: int,
    decision_limit: int,
    *,
    quality_shadow_limit: int,
    quality_trade_reflection_limit: int,
    quality_closed_position_limit: int,
    quality_sequence_limit: int,
    quality_text_limit: int,
) -> dict[str, Any]:
    now = datetime.now(UTC)
    since = now - timedelta(hours=max(int(hours), 1))
    return {
        "generated_at": now.isoformat(),
        "window_hours": max(int(hours), 1),
        "since": since.isoformat(),
        "decision_metrics": await _recent_decision_metrics(since, decision_limit),
        "order_metrics": await _order_metrics(since),
        "position_metrics": await _position_metrics(since),
        "training_source_metrics": await _training_source_metrics(
            shadow_limit=quality_shadow_limit,
            trade_reflection_limit=quality_trade_reflection_limit,
            closed_position_limit=quality_closed_position_limit,
            sequence_limit=quality_sequence_limit,
            text_sentiment_limit=quality_text_limit,
        ),
    }


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Export second-batch quant optimization baseline")
    parser.add_argument("--hours", type=int, default=DEFAULT_HOURS)
    parser.add_argument("--decision-limit", type=int, default=5000)
    parser.add_argument("--quality-shadow-limit", type=int, default=DEFAULT_QUALITY_SHADOW_LIMIT)
    parser.add_argument(
        "--quality-trade-reflection-limit",
        type=int,
        default=DEFAULT_QUALITY_TRADE_REFLECTION_LIMIT,
    )
    parser.add_argument(
        "--quality-closed-position-limit",
        type=int,
        default=DEFAULT_QUALITY_CLOSED_POSITION_LIMIT,
    )
    parser.add_argument(
        "--quality-sequence-limit", type=int, default=DEFAULT_QUALITY_SEQUENCE_LIMIT
    )
    parser.add_argument("--quality-text-limit", type=int, default=DEFAULT_QUALITY_TEXT_LIMIT)
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Use the full training quality window; slower and intended for scheduled jobs.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.deep:
        args.quality_shadow_limit = DEEP_QUALITY_SHADOW_LIMIT
        args.quality_trade_reflection_limit = DEEP_QUALITY_TRADE_REFLECTION_LIMIT
        args.quality_closed_position_limit = DEEP_QUALITY_CLOSED_POSITION_LIMIT
        args.quality_sequence_limit = DEEP_QUALITY_SEQUENCE_LIMIT
        args.quality_text_limit = DEEP_QUALITY_TEXT_LIMIT

    baseline = await build_baseline(
        args.hours,
        args.decision_limit,
        quality_shadow_limit=args.quality_shadow_limit,
        quality_trade_reflection_limit=args.quality_trade_reflection_limit,
        quality_closed_position_limit=args.quality_closed_position_limit,
        quality_sequence_limit=args.quality_sequence_limit,
        quality_text_limit=args.quality_text_limit,
    )
    text = json.dumps(baseline, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        summary = {
            "output": str(args.output),
            "window_hours": baseline.get("window_hours"),
            "sampled_recent_decision_count": baseline.get("decision_metrics", {}).get(
                "sampled_recent_decision_count"
            ),
            "total_recent_orders": baseline.get("order_metrics", {}).get("total_recent_orders"),
            "quality_totals": baseline.get("training_source_metrics", {})
            .get("quality_report", {})
            .get("totals", {}),
        }
        safe_print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    else:
        safe_print(text)


if __name__ == "__main__":
    asyncio.run(_main())
