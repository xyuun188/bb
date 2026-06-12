"""
Dashboard API endpoints — system status, market data, account balance.
"""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, overload

import structlog
from fastapi import APIRouter, Depends

from config.settings import (
    DECISION_MAKER_NAME,
    ENSEMBLE_TRADER_NAME,
    FIXED_AI_MODEL_SLOTS,
    settings,
)
from core.safe_output import safe_error_text
from core.trading_mode import mode_manager
from services.manual_close_marker import MANUAL_CLOSE_LABEL, is_manual_close_order
from services.server_monitor_status import get_server_monitor_status_sync
from web_dashboard.api.security import require_destructive_dashboard_confirmation
from web_dashboard.api.text_sanitize import sanitize_payload, sanitize_text

router = APIRouter()
logger = structlog.get_logger(__name__)
BEIJING_TZ = timezone(timedelta(hours=8))


# In-memory reference to the trading service (set by main loop)
_trading_service = None
_data_service = None
_competition_service = None
_EXCHANGE_MARK_CACHE_TTL_SECONDS = 5.0
_EXCHANGE_OPEN_SYMBOL_CACHE_TTL_SECONDS = 5.0
_PUBLIC_TICKER_CACHE_TTL_SECONDS = 10.0
_exchange_mark_cache: dict[str, tuple[datetime, dict[tuple[str, str], dict[str, float]]]] = {}
_exchange_open_symbol_cache: dict[str, tuple[datetime, set[str]]] = {}
_public_ticker_cache: dict[str, tuple[datetime, dict[str, dict]]] = {}


def _log_dashboard_fallback(event: str, exc: Exception, **fields: Any) -> None:
    """Log a recoverable dashboard fallback without breaking the endpoint."""
    logger.debug(event, error=safe_error_text(exc), **fields)


def _as_utc_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


@overload
def _safe_float(value: Any, default: None) -> float | None: ...


@overload
def _safe_float(value: Any, default: float = 0.0) -> float: ...


def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_float_value(value: Any, default: float = 0.0) -> float:
    result = _safe_float(value, default)
    return default if result is None else result


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _execution_reason_is_unusable(reason: str | None) -> bool:
    text = str(reason or "").strip()
    if not text:
        return False
    return any(
        marker in text
        for marker in (
            "原始说明已损坏",
            "无法准确还原",
            "鍘嗗彶璁板綍",
            "鎹熷潖",
        )
    )


def _recover_execution_reason_from_raw_decision(decision) -> str | None:
    raw = _safe_dict(decision.raw_llm_response)
    action = str(getattr(decision, "action", "") or "")
    if action not in {"close_long", "close_short"}:
        return None
    close_evidence = _safe_dict(raw.get("close_evidence"))
    action_plan = str(close_evidence.get("action_plan") or "").lower()
    plan_label = (
        "全平" if action_plan == "full_close" else "减仓" if action_plan == "reduce" else "平仓"
    )
    close_reason = str(
        close_evidence.get("reason") or getattr(decision, "reasoning", "") or ""
    ).strip()
    pnl = _safe_float(close_evidence.get("position_unrealized_pnl"), 0.0) or 0.0
    if close_reason:
        return (
            f"平仓裁决已生成但本轮没有确认到 OKX 平仓订单结果：AI 建议{plan_label}，"
            f"当时估算浮动盈亏 {pnl:.4f} USDT。裁决依据：{close_reason}"
            "系统会继续以 OKX 实际仓位和执行记录为准同步；如果仓位仍存在，下一轮持仓复盘会重新评估并提交平仓。"
        )
    return (
        "平仓裁决已生成但本轮没有确认到 OKX 平仓订单结果。"
        "系统会继续以 OKX 实际仓位和执行记录为准同步；如果仓位仍存在，下一轮持仓复盘会重新评估并提交平仓。"
    )


def _display_execution_reason(decision, order=None) -> str | None:
    reason = getattr(decision, "execution_reason", None) or _fallback_execution_reason(
        decision, order
    )
    sanitized = sanitize_text(reason)
    if _execution_reason_is_unusable(str(sanitized or "")):
        recovered = _recover_execution_reason_from_raw_decision(decision)
        if recovered:
            return recovered
    return sanitized


def _side_from_action(action: str | None) -> str:
    value = str(action or "").lower()
    if "short" in value:
        return "short"
    if "long" in value:
        return "long"
    return "hold"


def _side_label(side: str | None) -> str:
    value = str(side or "").lower()
    if value == "long":
        return "做多"
    if value == "short":
        return "做空"
    return "观望"


def _model_side_from_payload(payload: dict | None, side_key: str = "best_side") -> str:
    if not isinstance(payload, dict):
        return ""
    value = str(payload.get(side_key) or payload.get("side") or "").lower()
    if not value:
        direction = str(
            payload.get("direction")
            or payload.get("forecast_direction")
            or payload.get("trend")
            or ""
        ).lower()
        if direction == "up":
            value = "long"
        elif direction == "down":
            value = "short"
    if not value:
        label = str(payload.get("label") or payload.get("sentiment") or "").lower()
        score = _safe_float(payload.get("score"), payload.get("sentiment_score", 0.0)) or 0.0
        if label in {"positive", "bullish"} or score > 0:
            value = "long"
        elif label in {"negative", "bearish"} or score < 0:
            value = "short"
    if value in {"long", "short"}:
        return value
    return ""


def _extract_primary_ml(raw: dict[str, Any]) -> dict[str, Any]:
    ml = _safe_dict(raw.get("ml_signal"))
    predictions = _safe_list(ml.get("predictions"))
    primary = _safe_dict(predictions[0] if predictions else {})
    best_side = _model_side_from_payload(primary)
    return {
        "available": bool(ml.get("available")) if ml else False,
        "side": best_side,
        "side_label": _side_label(best_side),
        "expected_return_pct": _safe_float(
            primary.get("best_expected_return_pct"), ml.get("expected_return_pct", 0.0)
        )
        or 0.0,
        "profit_edge_pct": _safe_float(
            primary.get("profit_edge_pct"), ml.get("profit_edge_pct", 0.0)
        )
        or 0.0,
        "win_rate": _safe_float(primary.get("best_win_rate"), 0.0) or 0.0,
        "influence_enabled": bool(ml.get("influence_enabled", True)),
        "summary": ml.get("suggestion") or ml.get("note") or "",
    }


def _extract_local_tools(raw: dict[str, Any]) -> dict[str, Any]:
    tools = _safe_dict(raw.get("local_ai_tools"))
    profit = _safe_dict(tools.get("profit_prediction"))
    ts = _safe_dict(
        tools.get("time_series_prediction")
        or tools.get("timeseries_prediction")
        or tools.get("sequence_prediction")
    )
    sentiment = _safe_dict(tools.get("sentiment_analysis") or tools.get("sentiment_prediction"))
    profit_side = _model_side_from_payload(profit)
    ts_side = _model_side_from_payload(ts)
    sentiment_side = _model_side_from_payload(sentiment)
    return {
        "profit": {
            "available": bool(profit.get("available") or profit.get("trained")),
            "side": profit_side,
            "side_label": _side_label(profit_side),
            "expected_return_pct": _safe_float(
                profit.get("expected_return_pct"),
                profit.get(f"{profit_side}_expected_return_pct", 0.0),
            )
            or 0.0,
            "profit_quality_score": _safe_float(profit.get("profit_quality_score"), 0.0) or 0.0,
            "loss_probability": _safe_float(profit.get(f"{profit_side}_loss_probability"), 0.0)
            or 0.0,
            "model": profit.get("model") or "",
        },
        "timeseries": {
            "available": bool(ts.get("available") or ts.get("trained")),
            "side": ts_side,
            "side_label": _side_label(ts_side),
            "expected_return_pct": _safe_float(
                ts.get("expected_return_pct"),
                ts.get("expected_move_pct", ts.get(f"{ts_side}_expected_return_pct", 0.0)),
            )
            or 0.0,
            "horizon_minutes": ts.get("horizon_minutes") or ts.get("primary_horizon_minutes"),
            "model": ts.get("model") or "",
        },
        "sentiment": {
            "available": bool(sentiment.get("available") or sentiment.get("trained") or sentiment),
            "side": sentiment_side,
            "side_label": _side_label(sentiment_side),
            "score": _safe_float(sentiment.get("score"), sentiment.get("sentiment_score", 0.0))
            or 0.0,
            "expected_return_pct": _safe_float(
                sentiment.get("expected_return_pct"),
                sentiment.get("expected_return_from_sentiment_pct", 0.0),
            )
            or 0.0,
            "summary": sentiment.get("summary") or sentiment.get("reason") or "",
            "model": sentiment.get("model") or "",
        },
    }


def _build_decision_attribution(
    decision: Any, raw: dict[str, Any], experts: list[dict[str, Any]]
) -> dict[str, Any]:
    raw = _safe_dict(raw)
    action = str(getattr(decision, "action", "") or "").lower()
    side = _side_from_action(action)
    executed = bool(getattr(decision, "was_executed", False))
    opportunity = _safe_dict(raw.get("opportunity_score"))
    decision_maker = _safe_dict(raw.get("decision_maker"))
    close_evidence = _safe_dict(raw.get("close_evidence"))
    high_risk_review = _safe_dict(raw.get("high_risk_review"))
    ml = _extract_primary_ml(raw)
    local = _extract_local_tools(raw)

    support = 0
    oppose = 0
    hold = 0
    for item in experts or []:
        expert_side = _side_from_action(item.get("action"))
        if side in {"long", "short"} and expert_side == side:
            support += 1
        elif expert_side in {"long", "short"} and expert_side != side:
            oppose += 1
        else:
            hold += 1

    if action == "hold":
        final_reason = (
            getattr(decision, "execution_reason", None)
            or getattr(decision, "reasoning", None)
            or "最终选择观望。"
        )
    elif action in {"close_long", "close_short"}:
        final_reason = (
            close_evidence.get("reason")
            or close_evidence.get("block_reason")
            or getattr(decision, "execution_reason", None)
            or getattr(decision, "reasoning", None)
            or "持仓复盘给出平仓/减仓信号。"
        )
    elif executed:
        final_reason = (
            getattr(decision, "execution_reason", None)
            or "通过机会评分、风控检查和执行前检查，已提交订单。"
        )
    else:
        final_reason = (
            getattr(decision, "execution_reason", None)
            or opportunity.get("selection_reason")
            or decision_maker.get("reasoning")
            or getattr(decision, "reasoning", None)
            or "未达到执行条件。"
        )

    return {
        "action": action,
        "side": side,
        "side_label": _side_label(side),
        "executed": executed,
        "ai_experts": {
            "support_count": support,
            "oppose_count": oppose,
            "hold_count": hold,
            "summary": f"{support} 个专家同向，{oppose} 个反向，{hold} 个观望/非方向。",
        },
        "local_ml": ml,
        "server_profit": local.get("profit", {}),
        "timeseries": local.get("timeseries", {}),
        "sentiment": local.get("sentiment", {}),
        "opportunity_score": opportunity,
        "high_risk_review": high_risk_review,
        "decision_maker": {
            "status": decision_maker.get("status"),
            "action": decision_maker.get("action"),
            "confidence": decision_maker.get("confidence"),
            "reasoning": decision_maker.get("reasoning")
            or decision_maker.get("reason")
            or decision_maker.get("guard_reason"),
        },
        "close_evidence": close_evidence,
        "final_reason": final_reason,
    }


async def _get_today_ai_decision_count(mode: str) -> int:
    """Count AI decisions for the current Beijing calendar day."""
    from sqlalchemy import func, select

    from db.session import get_session_ctx
    from models.decision import AIDecision

    selected_mode = "live" if mode == "live" else "paper"
    now_local = datetime.now(BEIJING_TZ)
    start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    end_local = start_local + timedelta(days=1)
    start_utc = start_local.astimezone(UTC).replace(tzinfo=None)
    end_utc = end_local.astimezone(UTC).replace(tzinfo=None)

    async with get_session_ctx() as session:
        result = await session.execute(
            select(func.count(AIDecision.id)).where(
                AIDecision.is_paper.is_(selected_mode != "live"),
                AIDecision.created_at >= start_utc,
                AIDecision.created_at < end_utc,
            )
        )
        return int(result.scalar_one() or 0)


def _execution_risk_floor(allocated: float, max_loss_pct: float, max_loss_usdt: float) -> float:
    if allocated <= 0:
        return 0.0
    loss_limit = max_loss_usdt if max_loss_usdt > 0 else allocated * max_loss_pct
    return max(allocated - loss_limit, 0.0)


def _cooldown_pause_reason_from_summary(pnl_summary: dict, cfg: dict, source: str) -> str | None:
    max_loss_usdt = _safe_float(cfg.get("max_loss_usdt"), 0.0) or 0.0
    cooldown_loss_pct = _safe_float(cfg.get("cooldown_loss_pct"), 0.0) or 0.0
    trigger_loss = max_loss_usdt * cooldown_loss_pct
    if trigger_loss <= 0:
        return None

    today_risk_pnl = _safe_float(pnl_summary.get("today_risk_pnl"), 0.0) or 0.0
    if today_risk_pnl > -trigger_loss:
        return None

    today_equity = (
        _safe_float(
            pnl_summary.get("today_equity_pnl"),
            pnl_summary.get("today_total_pnl", 0.0),
        )
        or 0.0
    )
    today_realized = _safe_float(pnl_summary.get("today_realized_pnl"), 0.0) or 0.0
    unrealized = _safe_float(pnl_summary.get("unrealized_pnl"), 0.0) or 0.0
    return (
        "冷静期触发比例已达到，暂停分析新的交易对。"
        f"{source} 今日权益盈亏 {today_equity:.2f} USDT，"
        f"其中今日已平仓盈亏 {today_realized:.2f} USDT，"
        f"当前持仓浮动盈亏 {unrealized:.2f} USDT；触发线为最高亏损金额 "
        f"{max_loss_usdt:.2f} USDT 的 {cooldown_loss_pct * 100:.0f}%"
        f"（{trigger_loss:.2f} USDT）。已有持仓仍会继续复盘、止盈止损和平仓。"
    )


async def _get_execution_pnl_summary(mode: str) -> dict:
    """Return allocation-scoped PnL and budget usage for the execution account."""
    from sqlalchemy import case, func, select

    from db.session import get_session_ctx
    from models.trade import Position

    selected_mode = "live" if mode == "live" else "paper"
    cfg = settings.get_execution_account_config(selected_mode)
    allocated = float(cfg.get("allocated_balance") or 0.0)
    if _trading_service:
        try:
            snapshot = await _dashboard_okx_balance_snapshot_for_mode(selected_mode)
            allocated = float(
                (snapshot or {}).get("allocatable")
                or (snapshot or {}).get("equity")
                or (snapshot or {}).get("total")
                or allocated
                or 0.0
            )
        except Exception as exc:
            _log_dashboard_fallback(
                "dashboard balance snapshot fallback",
                exc,
                mode=selected_mode,
            )
    realized_profit = 0.0
    realized_loss = 0.0
    today_realized_profit = 0.0
    today_realized_loss = 0.0
    unrealized_pnl = 0.0
    used_margin = 0.0
    open_count = 0
    position_rows = []
    now_local = datetime.now(timezone(timedelta(hours=8)))
    start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    start_utc = start_local.astimezone(UTC)

    try:
        exchange_marks = await _get_exchange_position_mark_map(selected_mode)
    except Exception as exc:
        _log_dashboard_fallback(
            "exchange mark snapshot fallback",
            exc,
            mode=selected_mode,
        )
        exchange_marks = {}
    try:
        exchange_symbols = await _get_exchange_open_position_symbols(selected_mode)
    except Exception as exc:
        _log_dashboard_fallback(
            "exchange open symbol fallback",
            exc,
            mode=selected_mode,
        )
        exchange_symbols = None

    try:
        async with get_session_ctx() as session:
            filters = (
                Position.execution_mode == selected_mode,
                Position.model_name == ENSEMBLE_TRADER_NAME,
            )
            all_result = await session.execute(
                select(Position)
                .where(*filters)
                .order_by(Position.created_at.asc(), Position.id.asc())
            )
            position_rows = list(all_result.scalars().all())
            realized_result = await session.execute(
                select(
                    func.coalesce(
                        func.sum(
                            case(
                                (Position.realized_pnl >= 0, Position.realized_pnl),
                                else_=0.0,
                            )
                        ),
                        0.0,
                    ),
                    func.coalesce(
                        func.sum(
                            case(
                                (Position.realized_pnl < 0, -Position.realized_pnl),
                                else_=0.0,
                            )
                        ),
                        0.0,
                    ),
                ).where(*filters, Position.is_open.is_(False))
            )
            realized_profit, realized_loss = [
                float(value or 0.0) for value in realized_result.one()
            ]
            today_result = await session.execute(
                select(
                    func.coalesce(
                        func.sum(
                            case(
                                (Position.realized_pnl >= 0, Position.realized_pnl),
                                else_=0.0,
                            )
                        ),
                        0.0,
                    ),
                    func.coalesce(
                        func.sum(
                            case(
                                (Position.realized_pnl < 0, -Position.realized_pnl),
                                else_=0.0,
                            )
                        ),
                        0.0,
                    ),
                ).where(
                    *filters,
                    Position.is_open.is_(False),
                    Position.closed_at >= start_utc,
                )
            )
            today_realized_profit, today_realized_loss = [
                float(value or 0.0) for value in today_result.one()
            ]
            for pos in position_rows:
                if pos.is_open:
                    if (
                        exchange_symbols
                        and _normalize_dashboard_symbol(pos.symbol) not in exchange_symbols
                    ):
                        continue
                    open_count += 1
                    latest_unrealized = float(pos.unrealized_pnl or 0.0)
                    snapshot = exchange_marks.get(
                        (
                            _normalize_dashboard_symbol(pos.symbol),
                            str(pos.side or "").lower(),
                        )
                    )
                    if snapshot:
                        latest_price = float(snapshot.get("mark_price") or 0.0)
                        snapshot_upl = _safe_float(snapshot.get("upl"), None)
                        if snapshot_upl is not None:
                            latest_unrealized = snapshot_upl
                        elif latest_price > 0:
                            entry_price = _safe_float(snapshot.get("entry_price"), 0.0) or float(
                                pos.entry_price or 0.0
                            )
                            quantity = _safe_float(snapshot.get("contracts"), 0.0) or float(
                                pos.quantity or 0.0
                            )
                            if pos.side == "short":
                                latest_unrealized = (entry_price - latest_price) * quantity
                            else:
                                latest_unrealized = (latest_price - entry_price) * quantity
                    unrealized_pnl += latest_unrealized
                    snapshot_margin = _safe_float((snapshot or {}).get("margin_used"), 0.0)
                    if snapshot_margin > 0:
                        used_margin += snapshot_margin
                    else:
                        leverage = max(float(pos.leverage or 1.0), 1.0)
                        used_margin += (
                            float(pos.quantity or 0.0) * float(pos.entry_price or 0.0)
                        ) / leverage
    except Exception as exc:
        _log_dashboard_fallback(
            "execution pnl database summary fallback",
            exc,
            mode=selected_mode,
        )

    realized_pnl = realized_profit - realized_loss
    today_realized_pnl = today_realized_profit - today_realized_loss
    total_pnl = realized_pnl + unrealized_pnl
    today_total_pnl = today_realized_pnl + unrealized_pnl
    today_risk_pnl = today_realized_pnl + min(unrealized_pnl, 0.0)
    equity_baseline = {}
    try:
        from services.equity_baseline import apply_daily_equity_baseline

        async with get_session_ctx() as session:
            equity_baseline = await apply_daily_equity_baseline(
                session,
                mode=selected_mode,
                model_name=ENSEMBLE_TRADER_NAME,
                allocated=allocated,
                positions=position_rows,
                realized_pnl=realized_pnl,
                unrealized_pnl=unrealized_pnl,
                total_pnl=total_pnl,
            )
        today_total_pnl = float(equity_baseline.get("today_equity_pnl") or 0.0)
        today_risk_pnl = today_total_pnl
    except Exception as exc:
        _log_dashboard_fallback(
            "daily equity baseline fallback",
            exc,
            mode=selected_mode,
        )
        equity_baseline = {}
    pnl_adjusted_budget = max(allocated + total_pnl, 0.0)
    remaining_allocation = max(pnl_adjusted_budget - used_margin, 0.0)
    return {
        "allocated_balance": allocated,
        "realized_profit": realized_profit,
        "realized_loss": realized_loss,
        "realized_pnl": realized_pnl,
        "today_realized_profit": today_realized_profit,
        "today_realized_loss": today_realized_loss,
        "today_realized_pnl": today_realized_pnl,
        "today_closed_realized_profit": today_realized_profit,
        "today_closed_realized_loss": today_realized_loss,
        "today_closed_realized_pnl": today_realized_pnl,
        "today_equity_pnl": today_total_pnl,
        "today_equity_baseline": _safe_float(equity_baseline.get("today_equity_baseline"), None),
        "today_equity_baseline_total_pnl": _safe_float(
            equity_baseline.get("today_equity_baseline_total_pnl"),
            None,
        ),
        "today_equity_baseline_at": equity_baseline.get("today_equity_baseline_at"),
        "today_equity_baseline_source": equity_baseline.get("today_equity_baseline_source"),
        "today_snapshot_date": equity_baseline.get("today_snapshot_date"),
        "today_total_pnl": today_total_pnl,
        "today_risk_pnl": today_risk_pnl,
        "unrealized_pnl": unrealized_pnl,
        "total_pnl": total_pnl,
        "cumulative_realized_pnl": realized_pnl,
        "cumulative_unrealized_pnl": unrealized_pnl,
        "cumulative_total_pnl": total_pnl,
        "used_margin": used_margin,
        "remaining_allocation": remaining_allocation,
        "open_positions": open_count,
    }


def _clamp_confidence(value: float | None) -> float:
    if value is None:
        return 0.0
    return max(min(float(value), 1.0), 0.0)


def _analysis_display_confidence(
    action: str, trade_confidence: float, experts: list[dict], raw: dict
) -> float:
    """Confidence shown on collaboration records.

    Trade confidence intentionally stays 0 for hold decisions so the executor
    cannot mistake an observation for an order. Analysis records need a
    separate confidence that reflects how strongly the experts supported the
    final collaboration outcome.
    """
    trade_confidence = _clamp_confidence(trade_confidence)
    if action != "hold" or trade_confidence > 0:
        return trade_confidence

    weighted_total = 0.0
    weight_sum = 0.0
    plain_confidences: list[float] = []
    for expert in experts:
        confidence = _safe_float(expert.get("confidence"), None)
        if confidence is None:
            continue
        confidence = _clamp_confidence(confidence)
        plain_confidences.append(confidence)
        weight = _safe_float(expert.get("weight"), 0.0) or 0.0
        if weight > 0:
            weighted_total += confidence * weight
            weight_sum += weight

    if weight_sum > 0:
        return _clamp_confidence(weighted_total / weight_sum)
    if plain_confidences:
        return _clamp_confidence(sum(plain_confidences) / len(plain_confidences))

    weighted_score = abs(_safe_float(raw.get("weighted_score"), 0.0) or 0.0)
    return _clamp_confidence(weighted_score)


def _build_execution_account_status(
    mode: str,
    paper_summary: dict | None = None,
    okx_account: dict | None = None,
    pnl_summary: dict | None = None,
) -> dict:
    """Build the single execution-account payload shown on the dashboard."""
    mode = "live" if mode == "live" else "paper"
    cfg = settings.get_execution_account_config(mode)
    pnl_summary = pnl_summary or {}
    legacy_allocated = float(cfg.get("allocated_balance") or 0.0)
    max_loss_pct = float(cfg.get("max_loss_pct") or 0.0)
    okx_error = str(okx_account.get("error")) if okx_account and okx_account.get("error") else None
    okx_available = (
        None if okx_error else (_safe_float(okx_account.get("free"), None) if okx_account else None)
    )
    okx_used = (
        None if okx_error else (_safe_float(okx_account.get("used"), 0.0) if okx_account else None)
    )
    okx_total = (
        None
        if okx_error
        else (_safe_float(okx_account.get("total"), okx_available) if okx_account else None)
    )
    okx_cash = (
        None
        if okx_error
        else (_safe_float(okx_account.get("cash"), okx_total) if okx_account else None)
    )
    okx_equity = (
        None
        if okx_error
        else (_safe_float(okx_account.get("equity"), okx_total) if okx_account else None)
    )
    okx_allocatable = (
        None
        if okx_error
        else (
            _safe_float(okx_account.get("allocatable"), okx_equity or okx_total or okx_available)
            if okx_account
            else None
        )
    )
    account_equity = (
        _safe_float(
            okx_equity or okx_total or okx_allocatable or okx_available,
            legacy_allocated,
        )
        or 0.0
    )
    max_loss_usdt = (
        account_equity * max_loss_pct if account_equity > 0 and max_loss_pct > 0 else 0.0
    )
    risk_floor = _execution_risk_floor(account_equity, max_loss_pct, max_loss_usdt)
    allocated = account_equity
    pause_reason = None
    if _trading_service:
        pause_reason = getattr(_trading_service, "_new_pair_pause_reasons", {}).get(
            ENSEMBLE_TRADER_NAME
        )
    pause_reason = _translate_pause_reason(pause_reason)
    if okx_error and not pause_reason:
        source = "OKX 实盘账户" if mode == "live" else "OKX 模拟盘账户"
        pause_reason = f"{source} 余额同步失败，系统不会分析新的交易对。原因：{okx_error}"
    total_pnl_for_risk = _safe_float(pnl_summary.get("total_pnl"), 0.0) or 0.0
    if (
        account_equity > 0
        and max_loss_usdt > 0
        and total_pnl_for_risk <= -max_loss_usdt
        and not pause_reason
    ):
        source = "OKX 实盘账户" if mode == "live" else "OKX 模拟盘账户"
        pause_reason = (
            f"{source} AI 执行账户累计盈亏 {total_pnl_for_risk:.2f} USDT 已达到最高亏损限制 "
            f"{max_loss_pct * 100:.1f}%（{max_loss_usdt:.2f} USDT），系统不会分析新的交易对。"
        )
    if not pause_reason:
        source = "OKX 实盘账户" if mode == "live" else "OKX 模拟盘账户"
        pause_reason = _cooldown_pause_reason_from_summary(
            pnl_summary,
            {**cfg, "max_loss_usdt": max_loss_usdt},
            source,
        )

    payload = {
        **cfg,
        "model_name": ENSEMBLE_TRADER_NAME,
        "allocated_balance": account_equity,
        "account_equity": account_equity,
        "max_loss_usdt": max_loss_usdt,
        "risk_floor": risk_floor,
        "risk_paused": bool(pause_reason),
        "risk_pause_reason": pause_reason,
        "balance_error": okx_error,
        "okx_available_balance": okx_available,
        "okx_used_balance": okx_used,
        "okx_total_balance": okx_total,
        "okx_cash_balance": okx_cash,
        "okx_equity_balance": okx_equity,
        "max_allocatable_balance": okx_allocatable if okx_allocatable is not None else 0.0,
        "allocation_exceeds_balance": False,
        "positions": [],
        "open_positions": int(pnl_summary.get("open_positions") or 0),
        "realized_profit": _safe_float(pnl_summary.get("realized_profit"), 0.0),
        "realized_loss": _safe_float(pnl_summary.get("realized_loss"), 0.0),
        "realized_pnl": _safe_float(pnl_summary.get("realized_pnl"), 0.0),
        "today_realized_profit": _safe_float(pnl_summary.get("today_realized_profit"), 0.0),
        "today_realized_loss": _safe_float(pnl_summary.get("today_realized_loss"), 0.0),
        "today_realized_pnl": _safe_float(pnl_summary.get("today_realized_pnl"), 0.0),
        "today_closed_realized_profit": _safe_float(
            pnl_summary.get("today_closed_realized_profit"), 0.0
        ),
        "today_closed_realized_loss": _safe_float(
            pnl_summary.get("today_closed_realized_loss"), 0.0
        ),
        "today_closed_realized_pnl": _safe_float(pnl_summary.get("today_closed_realized_pnl"), 0.0),
        "today_equity_pnl": _safe_float(pnl_summary.get("today_equity_pnl"), 0.0),
        "today_equity_baseline": _safe_float(pnl_summary.get("today_equity_baseline"), None),
        "today_equity_baseline_total_pnl": _safe_float(
            pnl_summary.get("today_equity_baseline_total_pnl"),
            None,
        ),
        "today_equity_baseline_at": pnl_summary.get("today_equity_baseline_at"),
        "today_equity_baseline_source": pnl_summary.get("today_equity_baseline_source"),
        "today_snapshot_date": pnl_summary.get("today_snapshot_date"),
        "today_total_pnl": _safe_float(pnl_summary.get("today_total_pnl"), 0.0),
        "today_risk_pnl": _safe_float(pnl_summary.get("today_risk_pnl"), 0.0),
        "cumulative_profit": _safe_float(pnl_summary.get("realized_profit"), 0.0),
        "cumulative_loss": _safe_float(pnl_summary.get("realized_loss"), 0.0),
        "cumulative_realized_pnl": _safe_float(pnl_summary.get("cumulative_realized_pnl"), 0.0),
        "cumulative_unrealized_pnl": _safe_float(pnl_summary.get("cumulative_unrealized_pnl"), 0.0),
        "cumulative_total_pnl": _safe_float(pnl_summary.get("cumulative_total_pnl"), 0.0),
        "remaining_allocation": okx_available,
    }

    if mode == "paper":
        summary = paper_summary or {}
        local_available = _safe_float(summary.get("available_balance"), allocated)
        local_used_margin = _safe_float(pnl_summary.get("used_margin"), 0.0) or 0.0
        unrealized = _safe_float(pnl_summary.get("unrealized_pnl"), 0.0) or 0.0
        if okx_error:
            available = None
            used_margin = None
            wallet = None
            equity = None
        else:
            available = okx_available if okx_available is not None else local_available
            used_margin = okx_used if okx_used is not None else local_used_margin
            wallet = (
                okx_total
                if okx_total is not None
                else _safe_float(summary.get("wallet_balance"), (available or 0.0) + used_margin)
            )
            equity = (
                okx_total
                if okx_total is not None
                else _safe_float(summary.get("equity"), (wallet or 0.0) + unrealized)
            )
        total_pnl = _safe_float(pnl_summary.get("total_pnl"), 0.0)
        cumulative_total_pnl = _safe_float(pnl_summary.get("cumulative_total_pnl"), total_pnl)
        payload.update(
            {
                "available_balance": available,
                "current_balance": available,
                "wallet_balance": wallet,
                "equity": equity,
                "used_margin": used_margin,
                "position_margin_used": (
                    used_margin if used_margin is not None else local_used_margin
                ),
                "unrealized_pnl": unrealized,
                "total_pnl": cumulative_total_pnl,
                "cumulative_total_pnl": cumulative_total_pnl,
                "total_pnl_pct": (
                    ((cumulative_total_pnl or 0.0) / account_equity * 100)
                    if account_equity > 0
                    else _safe_float(summary.get("total_pnl_pct"), 0.0)
                ),
                "initial_balance": account_equity,
                "paper_execution_available_balance": _safe_float(
                    okx_available,
                    local_available,
                ),
                "paper_execution_used_margin": (
                    used_margin if used_margin is not None else local_used_margin
                ),
                "positions": summary.get("positions", []),
                "open_positions": int(pnl_summary.get("open_positions") or 0),
                "balance_source": "OKX 模拟盘账户" if okx_account else "模拟盘执行账户",
            }
        )
        return payload

    if okx_account:
        available = okx_available
        used_margin = (
            okx_used if okx_used is not None else _safe_float(pnl_summary.get("used_margin"), 0.0)
        )
        total = okx_total
        unrealized = _safe_float(pnl_summary.get("unrealized_pnl"), 0.0)
        total_pnl = _safe_float(pnl_summary.get("total_pnl"), 0.0)
        cumulative_total_pnl = _safe_float(pnl_summary.get("cumulative_total_pnl"), total_pnl)
        payload.update(
            {
                "available_balance": available,
                "current_balance": available,
                "wallet_balance": total,
                "equity": total,
                "used_margin": used_margin,
                "position_margin_used": used_margin,
                "unrealized_pnl": unrealized,
                "total_pnl": cumulative_total_pnl,
                "cumulative_total_pnl": cumulative_total_pnl,
                "total_pnl_pct": (
                    ((cumulative_total_pnl or 0.0) / account_equity * 100)
                    if account_equity > 0
                    else 0.0
                ),
                "initial_balance": account_equity,
                "balance_source": "OKX 实盘账户",
            }
        )
    else:
        payload.update(
            {
                "available_balance": None,
                "current_balance": None,
                "wallet_balance": None,
                "equity": None,
                "used_margin": None,
                "position_margin_used": None,
                "unrealized_pnl": None,
                "total_pnl": None,
                "total_pnl_pct": None,
                "initial_balance": account_equity,
                "balance_source": "OKX 实盘账户",
                "balance_error": "实盘账户未连接或余额查询失败",
            }
        )
    return payload


def _order_status_label(status: str | None) -> str:
    status_map = {
        "filled": "执行成功",
        "partial": "部分成交",
        "open": "等待成交",
        "pending": "等待提交",
        "cancelled": "已取消",
        "canceled": "已取消",
        "rejected": "执行失败",
    }
    return status_map.get(str(status or "").lower(), str(status or "未知"))


def _translate_pause_reason(reason: str | None) -> str | None:
    text = str(reason or "").strip()
    if not text:
        return None
    if "Execution account reached max loss limit" in text:
        total_match = re.search(r"total_pnl=([-0-9.]+)\s*USDT", text)
        max_match = re.search(r"max_loss=([-0-9.]+)\s*USDT", text)
        pct_match = re.search(r"\(([-0-9.]+)%\)", text)
        total = total_match.group(1) if total_match else "-"
        max_loss = max_match.group(1) if max_match else "-"
        pct = pct_match.group(1) if pct_match else "-"
        return (
            f"执行账户已达到最高亏损限制：当前累计盈亏 {total} USDT，"
            f"最高允许亏损 {max_loss} USDT（{pct}%）。暂停分析新的交易对。"
        )
    if "Risk circuit breaker is open" in text:
        detail = text.split("reason=", 1)[1] if "reason=" in text else "触发风险阈值"
        return f"风险熔断已开启，暂停分析新的交易对。原因：{detail}"
    if "OKX usable balance snapshot is unavailable" in text:
        return "未获取到 OKX 可用余额快照，暂停分析新的交易对。"
    if "OKX equity/balance is unavailable" in text:
        return "未获取到 OKX 账户权益或余额，暂停分析新的交易对。"
    if "OKX tradable balance is too low" in text:
        available = re.search(r"available=([-0-9.]+)\s*USDT", text)
        required = re.search(r"minimum_required=([-0-9.]+)\s*USDT", text)
        return (
            f"OKX 可交易余额过低：当前可用 {available.group(1) if available else '-'} USDT，"
            f"最低需要 {required.group(1) if required else '-'} USDT，暂停分析新的交易对。"
        )
    return text


def _fallback_execution_reason(decision, order=None) -> str | None:
    if decision.was_executed:
        return None

    action = decision.action
    if action == "hold":
        return "AI 选择观望，未提交订单。"

    if action not in ("long", "short", "close_long", "close_short"):
        return "未保存具体未执行原因。"

    snapshot = decision.feature_snapshot or {}
    confidence = float(decision.confidence or 0.0)
    volume_ratio = float(snapshot.get("volume_ratio") or 0.0)
    adx_14 = float(snapshot.get("adx_14") or 0.0)
    price_vs_sma20 = float(snapshot.get("price_vs_sma20") or 0.0)
    price_vs_sma50 = float(snapshot.get("price_vs_sma50") or 0.0)

    if confidence < settings.confidence_threshold:
        return (
            f"信心不足，未执行。当前置信度 {confidence:.2f}，"
            f"低于入场门槛 {settings.confidence_threshold:.2f}。"
        )

    if action in ("long", "short"):
        if volume_ratio < settings.min_entry_volume_ratio:
            return (
                f"成交量太低，未执行。当前成交量只有近 20 根K线平均成交量的 {volume_ratio:.2f} 倍，"
                f"低于最低要求 {settings.min_entry_volume_ratio:.2f} 倍。"
            )

        if adx_14 < settings.min_entry_adx:
            return (
                f"趋势强度不够，未执行。当前 ADX 为 {adx_14:.1f}，"
                f"低于最低要求 {settings.min_entry_adx:.1f}。"
            )

        if action == "long":
            if price_vs_sma20 <= 0 or price_vs_sma50 <= 0:
                return (
                    "做多条件不够完整，未执行。当前价格还没有同时站上 SMA20 和 SMA50，"
                    "趋势没有完全对齐。"
                )
        else:
            if price_vs_sma20 >= 0 or price_vs_sma50 >= 0:
                return (
                    "做空条件不够完整，未执行。当前价格还没有同时跌破 SMA20 和 SMA50，"
                    "趋势没有完全对齐。"
                )

        return (
            "未找到对应订单记录：页面当前没有查到这条开仓裁决关联的本地订单，"
            "不等于 OKX 已拒单。常见情况是：普通信号仍在等待本轮候选排序、排序后未通过后续检查、"
            "下单接口没有返回执行结果，或订单回写有延迟。请优先查看本条未执行原因和同时间附近的执行记录；"
            "强信号会显示“强信号即时执行”，普通信号会显示“等待排序”或“排序后进入执行”。"
        )

    if action in ("close_long", "close_short"):
        if order is not None:
            status_label = _order_status_label(getattr(order, "status", None))
            exchange_order_id = str(getattr(order, "exchange_order_id", "") or "").strip()
            if exchange_order_id:
                return (
                    f"系统已向 OKX 提交平仓单，OKX 订单号 {exchange_order_id}，"
                    f"但最终状态是「{status_label}」。如果数量为 0，说明 OKX 没有确认成交或仓位没有减少，"
                    "本地不会把仓位标记为已平仓。"
                )
            return (
                f"本地已生成平仓单，但没有拿到 OKX 订单号，当前本地订单状态为「{status_label}」。"
                "请以同时间附近的执行记录和 OKX 订单状态为准。"
            )
        return (
            "这条平仓裁决没有找到对应的本地平仓委托记录，所以系统没有把它视为已执行。"
            "注意：页面或 OKX 里同币种的开仓单/持仓记录不等于本次平仓单；"
            "close_short 需要找到买入平空单，close_long 需要找到卖出平多单。"
            "如果同时间附近确实有平仓委托，请以 OKX 订单状态和执行记录为准。"
        )

    return "未保存具体未执行原因。"


def _humanize_expert_failure(reason: str | None) -> str:
    """Convert provider/runtime errors into short UI-friendly Chinese text."""
    text = str(reason or "").strip()
    lower = text.lower()
    if not text:
        return "本轮没有返回结果。"
    if "timeout" in lower or "timed out" in lower or "readtimeout" in lower:
        return "模型调用超时，本轮没有返回结果。"
    if "json" in lower or "decode" in lower or "parse" in lower:
        return "模型返回格式不符合 JSON 要求，本轮结果被丢弃。"
    if (
        "invalid response" in lower
        or "empty" in lower
        or "not supported" in lower
        or "不被该代理支持" in text
    ):
        return "模型接口返回空结果，可能是当前代理不支持这个模型名。"
    if "401" in lower or "unauthorized" in lower or "invalid api key" in lower:
        return "API Key 无效或没有权限，本轮没有返回结果。"
    if "403" in lower or "forbidden" in lower or "permission" in lower:
        return "模型或接口权限不足，本轮没有返回结果。"
    if "429" in lower or "rate limit" in lower or "too many requests" in lower:
        return "接口限流，本轮没有返回结果。"
    if "connect" in lower or "connection" in lower or "network" in lower:
        return "模型接口连接失败，本轮没有返回结果。"
    return "模型调用失败，本轮没有返回结果。"


def _decision_type(action: str | None) -> tuple[str, str]:
    if action in ("long", "short"):
        return "entry", "开仓决策"
    if action in ("close_long", "close_short"):
        return "exit", "平仓决策"
    if action == "hold":
        return "hold", "观望决策"
    return "other", "其他决策"


def set_services(trading_svc, data_svc, competition_svc):
    """Called by main loop to inject service references for API access."""
    global _trading_service, _data_service, _competition_service
    _trading_service = trading_svc
    _data_service = data_svc
    _competition_service = competition_svc


def _dashboard_okx_executor_for_mode(mode: str) -> Any | None:
    if not _trading_service:
        return None
    getter = getattr(_trading_service, "okx_executor_for_dashboard", None)
    if not callable(getter):
        return None
    return getter("live" if mode == "live" else "paper")


async def _dashboard_okx_balance_snapshot_for_mode(mode: str) -> dict[str, Any] | None:
    if not _trading_service:
        return None
    getter = getattr(_trading_service, "get_okx_balance_snapshot_for_mode", None)
    if not callable(getter):
        return None
    return await getter("live" if mode == "live" else "paper")


def _trading_service_is_running() -> bool:
    if not _trading_service:
        return False
    is_running = getattr(_trading_service, "is_running", None)
    return bool(is_running()) if callable(is_running) else False


async def _completed_ml_shadow_sample_count() -> int:
    if not _trading_service or not getattr(_trading_service, "ml_signal_service", None):
        return 0
    counter = getattr(_trading_service.ml_signal_service, "completed_shadow_sample_count", None)
    if not callable(counter):
        return 0
    return int(await counter())


async def _completed_local_ai_shadow_backtest_total() -> int:
    if not _trading_service:
        return 0
    counter = getattr(_trading_service, "completed_shadow_backtest_total", None)
    if not callable(counter):
        return 0
    return int(await counter())


def _reset_trading_decision_runtime_state() -> None:
    if not _trading_service:
        return
    reset = getattr(_trading_service, "reset_decision_runtime_state", None)
    if callable(reset):
        reset()


def _normalize_dashboard_symbol(symbol: str | None) -> str:
    """Normalize exchange/CCXT symbols to the dashboard ticker key format."""
    if not symbol:
        return ""
    normalized = str(symbol).split(":")[0]
    if normalized.endswith("-SWAP"):
        normalized = normalized[:-5]
    if "/" not in normalized and "-" in normalized:
        parts = normalized.split("-")
        if len(parts) >= 2:
            normalized = f"{parts[0]}/{parts[1]}"
    return normalized


def _dashboard_symbol_query_variants(symbols: set[str]) -> set[str]:
    """Return historical symbol spellings used across orders, decisions, and shadows."""
    variants: set[str] = set()
    for symbol in symbols:
        raw = str(symbol or "").strip()
        normalized = _normalize_dashboard_symbol(raw)
        for value in (raw, normalized):
            if not value:
                continue
            variants.add(value)
            variants.add(value.upper())
            if "/" in value:
                dashed = value.replace("/", "-")
                variants.add(dashed)
                variants.add(f"{dashed}-SWAP")
            elif "-" in value:
                parts = [part for part in value.replace("-SWAP", "").split("-") if part]
                if len(parts) >= 2:
                    slash = f"{parts[0]}/{parts[1]}"
                    variants.add(slash)
                    variants.add(slash.upper())
                    variants.add(f"{parts[0]}-{parts[1]}")
                    variants.add(f"{parts[0]}-{parts[1]}-SWAP")
    return {item for item in variants if item}


def _is_live_position_open(position: dict) -> bool:
    raw_size = (
        position.get("contracts")
        or position.get("size")
        or position.get("positionAmt")
        or (position.get("info") or {}).get("pos")
        or (position.get("info") or {}).get("qty")
        or 0
    )
    try:
        return abs(float(raw_size)) > 0
    except (TypeError, ValueError):
        return bool(position.get("symbol"))


async def _get_open_position_symbols(mode: str | None = None) -> set[str]:
    """Return symbols that currently have open positions for the selected mode."""
    from sqlalchemy import select

    from db.session import get_session_ctx
    from models.trade import Position

    selected_mode = mode or mode_manager.mode.value
    symbols: set[str] = set()

    try:
        async with get_session_ctx() as session:
            result = await session.execute(
                select(Position.symbol)
                .where(
                    Position.execution_mode == selected_mode,
                    Position.model_name == ENSEMBLE_TRADER_NAME,
                    Position.is_open.is_(True),
                )
                .distinct()
            )
            symbols.update(_normalize_dashboard_symbol(row[0]) for row in result.all())
    except Exception as exc:
        _log_dashboard_fallback(
            "open position db symbol fallback",
            exc,
            mode=selected_mode,
        )

    if selected_mode == "paper" and _trading_service and _trading_service.paper_executor:
        try:
            positions = await _trading_service.paper_executor.get_positions()
            symbols.update(_normalize_dashboard_symbol(p.get("symbol")) for p in positions)
        except Exception as exc:
            _log_dashboard_fallback(
                "paper executor open symbol fallback",
                exc,
                mode=selected_mode,
            )

    if selected_mode == "live" and _dashboard_okx_executor_for_mode(selected_mode):
        exchange_symbols = await _get_exchange_open_position_symbols(selected_mode)
        if exchange_symbols:
            symbols.update(exchange_symbols)

    return {s for s in symbols if s}


async def _get_exchange_open_position_symbols(mode: str | None = None) -> set[str] | None:
    """Return actual exchange open-position symbols, or None when unavailable."""
    selected_mode = mode or mode_manager.mode.value
    now = datetime.now(UTC)
    cached = _exchange_open_symbol_cache.get(selected_mode)
    if cached:
        cached_at, cached_value = cached
        if (now - cached_at).total_seconds() <= _EXCHANGE_OPEN_SYMBOL_CACHE_TTL_SECONDS:
            return set(cached_value)

    executor = _dashboard_okx_executor_for_mode(selected_mode)

    if not executor:
        return None

    try:
        positions = await asyncio.wait_for(executor.get_positions(), timeout=1.2)
    except Exception as exc:
        _log_dashboard_fallback(
            "exchange open position symbols fallback",
            exc,
            mode=selected_mode,
            has_cached=bool(cached),
        )
        return set(cached[1]) if cached else None

    symbols: set[str] = set()
    for position in positions or []:
        if _is_live_position_open(position):
            symbols.add(_normalize_dashboard_symbol(position.get("symbol")))
    result = {s for s in symbols if s}
    _exchange_open_symbol_cache[selected_mode] = (now, result)
    return set(result)


async def _get_exchange_position_mark_map(
    mode: str | None = None,
) -> dict[tuple[str, str], dict[str, float]]:
    """Return exchange position mark-price snapshots keyed by (symbol, side)."""
    selected_mode = mode or mode_manager.mode.value
    now = datetime.now(UTC)
    cached = _exchange_mark_cache.get(selected_mode)
    if cached:
        cached_at, cached_value = cached
        if (now - cached_at).total_seconds() <= _EXCHANGE_MARK_CACHE_TTL_SECONDS:
            return cached_value

    executor = _dashboard_okx_executor_for_mode(selected_mode)

    if not executor:
        return {}

    try:
        positions = await executor.get_positions()
    except Exception as exc:
        _log_dashboard_fallback(
            "exchange mark map fallback",
            exc,
            mode=selected_mode,
            has_cached=bool(cached),
        )
        return cached[1] if cached else {}

    snapshots: dict[tuple[str, str], dict[str, float]] = {}
    for position in positions or []:
        if not _is_live_position_open(position):
            continue
        symbol = _normalize_dashboard_symbol(position.get("symbol"))
        side = str(position.get("side") or "").lower()
        if not symbol or side not in {"long", "short"}:
            continue
        try:
            mark_price = float(position.get("markPrice") or 0.0)
        except (TypeError, ValueError):
            mark_price = 0.0
        if mark_price <= 0:
            continue
        info = position.get("info") or {}
        try:
            last_price = float(info.get("last") or position.get("lastPrice") or 0.0)
        except (TypeError, ValueError):
            last_price = 0.0
        try:
            index_price = float(info.get("idxPx") or 0.0)
        except (TypeError, ValueError):
            index_price = 0.0
        try:
            upl = float(info.get("upl") or position.get("unrealizedPnl") or 0.0)
        except (TypeError, ValueError):
            upl = 0.0
        entry_price = _safe_float(position.get("entryPrice"), 0.0) or _safe_float(
            info.get("avgPx"), 0.0
        )
        contracts = _safe_float(position.get("contracts"), 0.0) or _safe_float(info.get("pos"), 0.0)
        margin_used = (
            _safe_float(position.get("initialMargin"), 0.0)
            or _safe_float(position.get("margin"), 0.0)
            or _safe_float(info.get("imr"), 0.0)
            or _safe_float(info.get("margin"), 0.0)
        )
        snapshots[(symbol, side)] = {
            "mark_price": mark_price,
            "last_price": last_price,
            "index_price": index_price,
            "upl": upl,
            "entry_price": entry_price,
            "contracts": contracts,
            "margin_used": margin_used,
        }
    _exchange_mark_cache[selected_mode] = (now, snapshots)
    return snapshots


async def _get_dashboard_okx_account_snapshot(selected_mode: str) -> dict[str, Any] | None:
    """Fetch the OKX balance snapshot used by dashboard summary with fallback logging."""
    if not _trading_service:
        return None

    executor = _dashboard_okx_executor_for_mode(selected_mode)
    if not executor:
        return None

    try:
        return await executor.get_balance_snapshot("USDT")
    except Exception as exc:
        _log_dashboard_fallback(
            "dashboard summary okx balance fallback",
            exc,
            mode=selected_mode,
        )
        return None


async def _get_display_open_position_symbols(mode: str | None = None) -> set[str]:
    """Return symbols that should be shown as currently held on the dashboard."""
    local_symbols = await _get_open_position_symbols(mode)
    exchange_symbols = await _get_exchange_open_position_symbols(mode)
    if not exchange_symbols:
        return local_symbols
    return local_symbols & exchange_symbols


async def _get_open_position_prices(mode: str | None = None) -> dict[str, float]:
    """Return last known prices from open persisted positions."""
    from db.repositories.trade_repo import TradeRepository
    from db.session import get_session_ctx

    selected_mode = mode or mode_manager.mode.value
    prices: dict[str, float] = {}

    try:
        async with get_session_ctx() as session:
            repo = TradeRepository(session)
            rows = await repo.get_position_records(execution_mode=selected_mode, limit=1000)
            for p in rows:
                if not p.is_open:
                    continue
                symbol = _normalize_dashboard_symbol(p.symbol)
                price = float(p.current_price or p.entry_price or 0)
                if symbol and price > 0:
                    prices[symbol] = price
    except Exception as exc:
        _log_dashboard_fallback(
            "open position price fallback",
            exc,
            mode=selected_mode,
        )

    return prices


async def _build_tickers_for_open_positions(
    open_symbols: set[str],
    market_tickers: dict,
    mode: str | None = None,
) -> dict:
    tickers: dict = {}
    public_tickers = await _get_public_ticker_map(open_symbols)

    # For currently held positions, prefer the same account-side OKX price
    # source used by simulated/live execution. Demo-account prices can differ
    # from OKX production public tickers, so this keeps the dashboard aligned
    # with the user's OKX position page.
    exchange_mark_map = await _get_exchange_position_mark_map(mode)
    for symbol in open_symbols:
        snapshots = [
            data
            for (snap_symbol, _side), data in exchange_mark_map.items()
            if snap_symbol == symbol
        ]
        if not snapshots:
            continue
        snapshot = snapshots[0]
        price = float(snapshot.get("last_price") or snapshot.get("mark_price") or 0.0)
        if price > 0:
            market_ticker = dict(market_tickers.get(symbol, {}) or {})
            public_ticker = dict(public_tickers.get(symbol, {}) or {})
            raw_change = market_ticker.get("change_24h")
            if raw_change is None:
                raw_change = market_ticker.get("change24h")
            if raw_change is None or _safe_float(raw_change, 0.0) == 0.0:
                raw_change = public_ticker.get("change_24h", raw_change)
            tickers[symbol] = {
                "price": price,
                "mark_price": snapshot.get("mark_price"),
                "index_price": snapshot.get("index_price"),
                "change_24h": _safe_float(raw_change, 0.0),
                "volume_24h": _safe_float(market_ticker.get("volume_24h"), 0.0)
                or _safe_float(public_ticker.get("volume_24h"), 0.0),
                "bid": _safe_float(market_ticker.get("bid"), 0.0)
                or _safe_float(public_ticker.get("bid"), 0.0),
                "ask": _safe_float(market_ticker.get("ask"), 0.0)
                or _safe_float(public_ticker.get("ask"), 0.0),
            }

    for symbol in open_symbols - set(tickers):
        if symbol in market_tickers:
            tickers[symbol] = _merge_market_and_public_ticker(
                dict(market_tickers[symbol]),
                dict(public_tickers.get(symbol, {}) or {}),
            )

    if open_symbols - set(tickers):
        position_prices = await _get_open_position_prices(mode)
        for symbol in open_symbols - set(tickers):
            price = position_prices.get(symbol, 0)
            if price > 0:
                market_ticker = dict(market_tickers.get(symbol, {}) or {})
                public_ticker = dict(public_tickers.get(symbol, {}) or {})
                raw_change = market_ticker.get("change_24h")
                if raw_change is None:
                    raw_change = market_ticker.get("change24h")
                if raw_change is None or _safe_float(raw_change, 0.0) == 0.0:
                    raw_change = public_ticker.get("change_24h", raw_change)
                tickers[symbol] = {
                    "price": price,
                    "change_24h": _safe_float(raw_change, 0.0),
                    "volume_24h": _safe_float(market_ticker.get("volume_24h"), 0.0)
                    or _safe_float(public_ticker.get("volume_24h"), 0.0),
                    "bid": _safe_float(market_ticker.get("bid"), 0.0)
                    or _safe_float(public_ticker.get("bid"), 0.0),
                    "ask": _safe_float(market_ticker.get("ask"), 0.0)
                    or _safe_float(public_ticker.get("ask"), 0.0),
                }

    return tickers


def _merge_market_and_public_ticker(market_ticker: dict, public_ticker: dict) -> dict:
    merged = {**public_ticker, **market_ticker}
    market_change = market_ticker.get("change_24h")
    if market_change is None:
        market_change = market_ticker.get("change24h")
    if market_change is None or _safe_float(market_change, 0.0) == 0.0:
        merged["change_24h"] = public_ticker.get("change_24h", market_change or 0.0)
    for key in ("volume_24h", "bid", "ask"):
        if _safe_float(market_ticker.get(key), 0.0) == 0.0:
            merged[key] = public_ticker.get(key, market_ticker.get(key, 0.0))
    return merged


async def _get_public_ticker_map(symbols: set[str]) -> dict[str, dict]:
    requested = sorted(s for s in symbols if s)
    if not requested or not _data_service or not getattr(_data_service, "rest_client", None):
        return {}

    cache_key = ",".join(requested)
    now = datetime.now(UTC)
    cached = _public_ticker_cache.get(cache_key)
    if cached:
        cached_at, cached_value = cached
        if (now - cached_at).total_seconds() <= _PUBLIC_TICKER_CACHE_TTL_SECONDS:
            return cached_value

    try:
        raw_tickers = await _data_service.rest_client.fetch_tickers(requested)
    except Exception as exc:
        _log_dashboard_fallback(
            "public ticker fallback",
            exc,
            symbol_count=len(requested),
            has_cached=bool(cached),
        )
        return cached[1] if cached else {}

    parsed: dict[str, dict] = {}
    for raw_key, ticker in (raw_tickers or {}).items():
        if not isinstance(ticker, dict):
            continue
        info = ticker.get("info") or {}
        symbol = _normalize_dashboard_symbol(ticker.get("symbol") or info.get("instId") or raw_key)
        if symbol not in symbols:
            continue
        price = _safe_float(
            ticker.get("last") or ticker.get("close") or info.get("last") or info.get("lastPx"),
            0.0,
        )
        parsed[symbol] = {
            "price": price,
            "change_24h": _okx_display_change_pct(ticker),
            "volume_24h": _safe_float(
                ticker.get("baseVolume") or info.get("volCcy24h") or info.get("vol24h"),
                0.0,
            ),
            "bid": _safe_float(ticker.get("bid") or info.get("bidPx"), 0.0),
            "ask": _safe_float(ticker.get("ask") or info.get("askPx"), 0.0),
        }

    _public_ticker_cache[cache_key] = (now, parsed)
    return parsed


def _okx_display_change_pct(ticker: dict) -> float:
    """Match OKX web UI's displayed day-change baseline when available."""
    try:
        price = float(ticker.get("last") or ticker.get("close") or 0)
        info = ticker.get("info") or {}
        baseline = float(info.get("sodUtc8") or ticker.get("open") or 0)
        if price > 0 and baseline > 0:
            return (price - baseline) / baseline * 100
    except (TypeError, ValueError):
        pass
    try:
        return float(ticker.get("percentage") or 0)
    except (TypeError, ValueError):
        return 0.0


@router.get("/status")
async def get_status(mode: str | None = None):
    """System status overview. Optional mode filter for cumulative counts."""
    trading_stats = {}
    if _trading_service:
        trading_stats = _trading_service.get_stats(mode_filter=mode)

    return {
        "status": "running" if _trading_service_is_running() else "stopped",
        "mode": mode_manager.mode.value,
        "paused": mode_manager.is_paused,
        "scan_mode": mode_manager.scan_mode,
        "live_model": mode_manager.live_model_name,
        "timestamp": datetime.now(UTC).isoformat(),
        **trading_stats,
    }


@router.get("/ml-signal/status")
async def get_ml_signal_status():
    if not _trading_service or not getattr(_trading_service, "ml_signal_service", None):
        return {"available": False, "status": "service_not_ready"}
    status = _trading_service.ml_signal_service.status()
    if not isinstance(status, dict):
        return status

    training_count = int(status.get("sample_count") or status.get("trained_sample_count") or 0)
    status.setdefault("training_shadow_sample_count", training_count)
    status.setdefault("training_shadow_sample_limit", 20000)
    status.setdefault(
        "training_sample_note",
        "sample_count is the latest training window, not the all-time total.",
    )

    try:
        completed_total = await _completed_ml_shadow_sample_count()
        status.setdefault("completed_shadow_sample_count", completed_total)
        status.setdefault("total_shadow_sample_count", completed_total)

        auto_last = (
            status.get("auto_train_last_result")
            if isinstance(status.get("auto_train_last_result"), dict)
            else {}
        )
        auto_new = auto_last.get("new_sample_count") if isinstance(auto_last, dict) else None
        if auto_new is None:
            auto_new = max(int(completed_total) - training_count, 0)
        status.setdefault("new_shadow_sample_count", int(auto_new or 0))
    except Exception as exc:
        _log_dashboard_fallback("ml signal sample count fallback", exc)
        status.setdefault("completed_shadow_sample_count", training_count)
        status.setdefault("total_shadow_sample_count", training_count)
        status.setdefault("new_shadow_sample_count", 0)

    return status


@router.get("/local-ai-tools/status")
async def get_local_ai_tools_status():
    if not _trading_service or not getattr(_trading_service, "local_ai_tools", None):
        return {"available": False, "status": "service_not_ready"}
    status = await _trading_service.local_ai_tools.status()
    if isinstance(status, dict):
        try:
            completed_total = await _completed_local_ai_shadow_backtest_total()
            status.setdefault("completed_shadow_sample_count", completed_total)
            status.setdefault("training_shadow_sample_count", status.get("shadow_sample_count"))
            status.setdefault("training_shadow_sample_limit", 20000)
        except Exception as exc:
            _log_dashboard_fallback("local ai tools shadow count fallback", exc)
    return status


@router.get("/server-monitor/status")
async def get_server_monitor_status():
    return await asyncio.to_thread(get_server_monitor_status_sync)


@router.get("/dashboard/summary")
async def get_dashboard_summary():
    """Aggregate dashboard data."""
    market_state = {}
    if _data_service:
        market_state = _data_service.get_market_state()

    # The main dashboard ticker panel should reflect current holdings only.
    open_symbols = await _get_display_open_position_symbols()
    all_tickers = market_state.get("tickers", {})
    market_state["tickers"] = await _build_tickers_for_open_positions(
        open_symbols, all_tickers, mode_manager.mode.value
    )
    market_state["position_symbols"] = sorted(open_symbols)

    account_summaries = []
    if _trading_service and _trading_service.paper_executor:
        all_summaries = await _trading_service.paper_executor.get_all_summaries()
        account_summaries = [
            item for item in all_summaries if item.get("model_name") == ENSEMBLE_TRADER_NAME
        ]
        if not account_summaries:
            account_summaries = [
                await _trading_service.paper_executor.get_account_summary(ENSEMBLE_TRADER_NAME)
            ]
        exchange_symbols = await _get_exchange_open_position_symbols("paper")
        exchange_mark_map = await _get_exchange_position_mark_map("paper")
        exchange_total_upl = round(
            sum(item.get("upl", 0.0) for item in exchange_mark_map.values()), 8
        )
        if exchange_symbols:
            for account in account_summaries:
                filtered_positions = [
                    p
                    for p in account.get("positions", [])
                    if _normalize_dashboard_symbol(p.get("symbol")) in exchange_symbols
                ]
                display_unrealized = 0.0
                display_used_margin = 0.0
                for position in filtered_positions:
                    side = str(position.get("side") or "").lower()
                    snapshot = exchange_mark_map.get(
                        (_normalize_dashboard_symbol(position.get("symbol")), side)
                    )
                    if snapshot:
                        mark_price = float(snapshot.get("mark_price") or 0.0)
                        entry_price = _safe_float(snapshot.get("entry_price"), 0.0) or float(
                            position.get("entry_price") or 0.0
                        )
                        quantity = _safe_float(snapshot.get("contracts"), 0.0) or float(
                            position.get("quantity") or 0.0
                        )
                        snapshot_upl = _safe_float(snapshot.get("upl"), None)
                        if entry_price > 0:
                            position["entry_price"] = entry_price
                        if quantity > 0:
                            position["quantity"] = quantity
                        if snapshot_upl is not None:
                            position["unrealized_pnl"] = snapshot_upl
                        elif mark_price > 0 and entry_price > 0 and quantity > 0:
                            position["current_price"] = mark_price
                            if side == "short":
                                position["unrealized_pnl"] = (entry_price - mark_price) * quantity
                            else:
                                position["unrealized_pnl"] = (mark_price - entry_price) * quantity
                    display_unrealized += float(position.get("unrealized_pnl") or 0.0)
                    snapshot_margin = _safe_float((snapshot or {}).get("margin_used"), 0.0)
                    if snapshot_margin > 0:
                        display_used_margin += snapshot_margin
                    else:
                        leverage = max(float(position.get("leverage") or 1.0), 1.0)
                        display_used_margin += (
                            float(position.get("quantity") or 0.0)
                            * float(position.get("entry_price") or 0.0)
                        ) / leverage

                account["positions"] = filtered_positions
                account["open_positions"] = len(filtered_positions)
                account["used_margin"] = round(display_used_margin, 8)
                account["position_margin_used"] = round(display_used_margin, 8)
                account["wallet_balance"] = (
                    float(account.get("current_balance") or account.get("balance") or 0.0)
                    + display_used_margin
                )
                account["unrealized_pnl"] = round(display_unrealized, 8)
                account["display_unrealized_pnl"] = round(display_unrealized, 8)
                account["equity"] = account["wallet_balance"] + account["unrealized_pnl"]
                initial_balance = float(account.get("initial_balance") or 0.0)
                account["total_pnl"] = account["equity"] - initial_balance
                account["total_pnl_pct"] = (
                    ((account["total_pnl"] / initial_balance) * 100) if initial_balance > 0 else 0.0
                )
            if len(account_summaries) == 1 and exchange_mark_map:
                account_summaries[0]["display_unrealized_pnl"] = exchange_total_upl

    okx_account = await _get_dashboard_okx_account_snapshot(mode_manager.mode.value)

    paper_summary = account_summaries[0] if account_summaries else None
    pnl_summary = await _get_execution_pnl_summary(mode_manager.mode.value)
    execution_account = _build_execution_account_status(
        mode_manager.mode.value,
        paper_summary=paper_summary,
        okx_account=okx_account,
        pnl_summary=pnl_summary,
    )
    account_summaries = [execution_account]

    trading_stats = {}
    if _trading_service:
        trading_stats = _trading_service.get_stats(mode_filter=mode_manager.mode.value)
    today_decisions_total = await _get_today_ai_decision_count(mode_manager.mode.value)

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "mode": mode_manager.mode.value,
        "paused": mode_manager.is_paused,
        "scan_mode": mode_manager.scan_mode,
        "market": market_state,
        "model_rankings": [],
        "execution_account": execution_account,
        "accounts": account_summaries,
        "okx_account": okx_account,
        **trading_stats,
        "today_decisions_total": today_decisions_total,
        "today_decisions_timezone": "Asia/Shanghai",
    }


@router.get("/dashboard/market")
async def get_market_data():
    """Current market prices and stats."""
    if _data_service:
        return _data_service.get_market_state()
    return {"tickers": {}, "ws_stats": {}}


@router.get("/dashboard/positions")
async def get_positions(
    mode: str | None = None,
    page: int = 1,
    page_size: int = 20,
    open_only: bool = False,
    closed_only: bool = False,
):
    """All positions (open + closed) — 持仓记录. Optional mode filter (paper/live)."""
    from sqlalchemy import select

    from db.repositories.trade_repo import TradeRepository
    from db.session import get_session_ctx
    from models.decision import AIDecision
    from models.trade import Order

    page = max(int(page or 1), 1)
    page_size = max(1, min(int(page_size or 20), 100))
    positions = []
    exchange_mark_map = {} if closed_only else await _get_exchange_position_mark_map(mode)
    exchange_symbols = (
        {symbol for symbol, _side in exchange_mark_map.keys()} if exchange_mark_map else None
    )
    market_tickers: dict[str, dict[str, Any]] = {}
    if _data_service:
        market_tickers = (_data_service.get_market_state() or {}).get("tickers", {}) or {}
    async with get_session_ctx() as session:
        repo = TradeRepository(session)
        is_open_filter = True if open_only else False if closed_only else None
        total = await repo.count_positions(execution_mode=mode, is_open=is_open_filter)
        open_count = await repo.count_positions(execution_mode=mode, is_open=True)
        # When showing current open positions, exchange reconciliation happens
        # after DB rows are loaded. Load all open rows first, filter against OKX,
        # then paginate the display list; otherwise page 1 can incorrectly cap
        # the total at 20.
        should_paginate_after_exchange_filter = bool(
            open_only and not closed_only and exchange_mark_map
        )
        should_paginate_after_display_filter = should_paginate_after_exchange_filter or bool(
            closed_only
        )
        total_pages = max(1, (total + page_size - 1) // page_size) if total else 1
        page = min(page, total_pages)
        offset = 0 if should_paginate_after_display_filter else (page - 1) * page_size
        limit = (
            min(max(total, page_size), 5000) if should_paginate_after_display_filter else page_size
        )
        rows = await repo.get_position_records(
            execution_mode=mode,
            limit=limit,
            offset=offset,
            is_open=is_open_filter,
        )
        open_sibling_rows = []
        if closed_only:
            open_sibling_rows = await repo.get_position_records(
                execution_mode=mode,
                limit=5000,
                offset=0,
                is_open=True,
            )
        sibling_positions = list(rows) + list(open_sibling_rows)
        close_order_status_by_position_id: dict[int, tuple[str, str, str]] = {}
        if closed_only and rows:
            closed_rows = [p for p in rows if not p.is_open and p.closed_at]
            close_symbols = sorted(
                {_normalize_dashboard_symbol(p.symbol) for p in closed_rows if p.symbol}
            )
            if close_symbols:
                min_closed = min((p.closed_at for p in closed_rows if p.closed_at), default=None)
                max_closed = max((p.closed_at for p in closed_rows if p.closed_at), default=None)
                order_stmt = select(Order).where(
                    Order.symbol.in_(close_symbols),
                    Order.status == "filled",
                )
                if mode:
                    order_stmt = order_stmt.where(Order.execution_mode == mode)
                if min_closed:
                    order_stmt = order_stmt.where(
                        Order.created_at >= min_closed - timedelta(seconds=15)
                    )
                if max_closed:
                    order_stmt = order_stmt.where(
                        Order.created_at <= max_closed + timedelta(seconds=15)
                    )
                order_result = await session.execute(order_stmt)
                close_orders = list(order_result.scalars().all())
                decision_ids = {o.decision_id for o in close_orders if o.decision_id}
                decision_meta: dict[int, dict[str, Any]] = {}
                if decision_ids:
                    decision_result = await session.execute(
                        select(AIDecision).where(AIDecision.id.in_(decision_ids))
                    )
                    for decision in decision_result.scalars().all():
                        raw = _safe_dict(decision.raw_llm_response)
                        decision_meta[decision.id] = {
                            "action": decision.action,
                            "position_size_pct": decision.position_size_pct,
                            "raw": raw,
                        }

                def status_from_order_decision(order) -> tuple[str | None, str | None]:
                    if is_manual_close_order(order):
                        return "manual", MANUAL_CLOSE_LABEL
                    meta = _safe_dict(decision_meta.get(order.decision_id or -1))
                    raw = _safe_dict(meta.get("raw"))
                    close_evidence = _safe_dict(raw.get("close_evidence"))
                    pct = (
                        _safe_float(
                            meta.get("position_size_pct")
                            or close_evidence.get("position_size_pct")
                            or raw.get("close_fraction"),
                            0.0,
                        )
                        or 0.0
                    )
                    if pct > 0:
                        if pct < 0.999:
                            return "partial", "部分平仓"
                        return "full", "全部平仓"
                    action_plan = str(close_evidence.get("action_plan") or "").lower()
                    if action_plan in {"reduce", "partial", "partial_close"}:
                        return "partial", "部分平仓"
                    if action_plan in {"full_close", "close", "all"}:
                        return "full", "全部平仓"
                    return None, None

                for pos in closed_rows:
                    close_side = "buy" if str(pos.side or "").lower() == "short" else "sell"
                    closed_at = pos.closed_at
                    closed_at_cmp = (
                        closed_at.replace(tzinfo=None)
                        if closed_at and closed_at.tzinfo
                        else closed_at
                    )
                    candidates = []
                    for order in close_orders:
                        if _normalize_dashboard_symbol(order.symbol) != _normalize_dashboard_symbol(
                            pos.symbol
                        ):
                            continue
                        if str(order.side or "").lower() != close_side:
                            continue
                        order_time = order.created_at
                        order_time_cmp = (
                            order_time.replace(tzinfo=None)
                            if order_time and order_time.tzinfo
                            else order_time
                        )
                        if not closed_at_cmp or not order_time_cmp:
                            continue
                        delta = abs((order_time_cmp - closed_at_cmp).total_seconds())
                        if delta > 8:
                            continue
                        price_delta = abs(
                            (_safe_float(order.price, 0.0) or 0.0)
                            - (_safe_float(pos.current_price, 0.0) or 0.0)
                        )
                        candidates.append((delta, price_delta, order))
                    if not candidates:
                        continue
                    matched_order = sorted(candidates, key=lambda item: (item[0], item[1]))[0][2]
                    status, label = status_from_order_decision(matched_order)
                    if status and label:
                        close_order_status_by_position_id[pos.id] = (status, label, "order")

        def close_status_for(pos) -> tuple[str, str]:
            if pos.is_open:
                return "open", "持有中"

            order_status = close_order_status_by_position_id.get(pos.id)
            if order_status:
                return order_status[0], order_status[1]

            # A system partial close is stored as a separate closed row for the
            # closed quantity, while the original older position keeps the
            # remaining quantity. Detect that shape without changing DB schema.
            closed_at = pos.closed_at
            created_at = pos.created_at
            if closed_at and closed_at.tzinfo is None:
                closed_at = closed_at.replace(tzinfo=UTC)
            if created_at and created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=UTC)
            same_time_close = False
            if closed_at and created_at:
                same_time_close = abs((closed_at - created_at).total_seconds()) <= 5
            if not same_time_close:
                return "full", "全部平仓"

            symbol_key = _normalize_dashboard_symbol(pos.symbol)
            side_key = str(pos.side or "").lower()
            entry = float(pos.entry_price or 0.0)
            tolerance = max(abs(entry) * 1e-6, 1e-8)
            for sibling in sibling_positions:
                if sibling.id == pos.id:
                    continue
                if (
                    sibling.model_name != pos.model_name
                    or sibling.execution_mode != pos.execution_mode
                ):
                    continue
                if _normalize_dashboard_symbol(sibling.symbol) != symbol_key:
                    continue
                if str(sibling.side or "").lower() != side_key:
                    continue
                sibling_entry = float(sibling.entry_price or 0.0)
                if entry > 0 and abs(sibling_entry - entry) > tolerance:
                    continue
                sibling_created_at = sibling.created_at
                if sibling_created_at and sibling_created_at.tzinfo is None:
                    sibling_created_at = sibling_created_at.replace(tzinfo=UTC)
                if sibling_created_at and created_at and sibling_created_at < created_at:
                    return "partial", "部分平仓"
            return "full", "全部平仓"

        for p in rows:
            current_price = p.current_price
            unrealized_pnl = p.unrealized_pnl
            entry_price = p.entry_price
            quantity = p.quantity
            pnl_source = "local_db"
            change_24h = 0.0
            if p.is_open:
                snapshot = exchange_mark_map.get(
                    (_normalize_dashboard_symbol(p.symbol), str(p.side or "").lower())
                )
                market_ticker = market_tickers.get(_normalize_dashboard_symbol(p.symbol), {})
                change_24h = float(
                    market_ticker.get("change_24h") or market_ticker.get("change24h") or 0.0
                )
                if snapshot:
                    latest_price = float(snapshot.get("mark_price") or 0.0)
                    if latest_price > 0:
                        current_price = latest_price
                    exchange_entry = _safe_float(snapshot.get("entry_price"), 0.0)
                    exchange_qty = _safe_float(snapshot.get("contracts"), 0.0)
                    exchange_upl = _safe_float(snapshot.get("upl"), None)
                    if exchange_entry > 0:
                        entry_price = exchange_entry
                    if exchange_qty > 0:
                        quantity = exchange_qty
                    if exchange_upl is not None:
                        unrealized_pnl = exchange_upl
                        pnl_source = "okx_position"
                    elif latest_price > 0:
                        if p.side == "short":
                            unrealized_pnl = (entry_price - latest_price) * quantity
                        else:
                            unrealized_pnl = (latest_price - entry_price) * quantity
                        pnl_source = "okx_mark_recomputed"
                elif market_ticker:
                    latest_price = float(
                        market_ticker.get("price") or market_ticker.get("last_price") or 0.0
                    )
                    if latest_price > 0:
                        current_price = latest_price
                        if p.side == "short":
                            unrealized_pnl = (entry_price - latest_price) * quantity
                        else:
                            unrealized_pnl = (latest_price - entry_price) * quantity
                        pnl_source = "market_ticker_recomputed"

            exchange_synced = (
                True
                if exchange_symbols is None or not p.is_open
                else _normalize_dashboard_symbol(p.symbol) in exchange_symbols
            )
            display_is_open = bool(p.is_open and exchange_synced)
            if open_only and not display_is_open:
                continue
            close_status, close_status_label = close_status_for(p)
            close_status_source = (
                "order" if p.id in close_order_status_by_position_id else "position"
            )

            positions.append(
                {
                    "id": p.id,
                    "model_name": p.model_name,
                    "mode": p.execution_mode,
                    "symbol": p.symbol,
                    "side": p.side,
                    "quantity": quantity,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "change_24h": change_24h,
                    "unrealized_pnl": unrealized_pnl,
                    "pnl_source": pnl_source,
                    "local_quantity": p.quantity,
                    "local_entry_price": p.entry_price,
                    "local_unrealized_pnl": p.unrealized_pnl,
                    "realized_pnl": p.realized_pnl,
                    "leverage": p.leverage,
                    "stop_loss": p.stop_loss_price,
                    "take_profit": p.take_profit_price,
                    "is_open": display_is_open,
                    "db_is_open": p.is_open,
                    "exchange_synced": exchange_synced,
                    "close_status": close_status,
                    "close_status_label": close_status_label,
                    "close_status_source": close_status_source,
                    "position_status": close_status_label,
                    "opened_at": p.created_at.isoformat() if p.created_at else None,
                    "closed_at": p.closed_at.isoformat() if p.closed_at else None,
                }
            )

    if closed_only:
        grouped_positions: list[dict] = []
        grouped_index: dict[tuple, dict] = {}
        for item in positions:
            closed_at_text = str(item.get("closed_at") or "")
            closed_second = closed_at_text.split(".")[0] if closed_at_text else ""
            close_price = round(_safe_float(item.get("current_price"), 0.0) or 0.0, 8)
            key = (
                item.get("model_name"),
                item.get("mode"),
                _normalize_dashboard_symbol(str(item.get("symbol") or "")),
                str(item.get("side") or "").lower(),
                closed_second,
                close_price,
            )
            quantity = _safe_float(item.get("quantity"), 0.0) or 0.0
            entry_price = _safe_float(item.get("entry_price"), 0.0) or 0.0
            realized_pnl = _safe_float(item.get("realized_pnl"), 0.0) or 0.0
            if key not in grouped_index:
                item = dict(item)
                item["position_ids"] = [item.get("id")]
                item["split_count"] = 1
                item["_entry_notional_for_avg"] = entry_price * quantity
                grouped_index[key] = item
                grouped_positions.append(item)
                continue

            group = grouped_index[key]
            group_qty = (_safe_float(group.get("quantity"), 0.0) or 0.0) + quantity
            group["_entry_notional_for_avg"] = (
                _safe_float(group.get("_entry_notional_for_avg"), 0.0) or 0.0
            ) + entry_price * quantity
            group["quantity"] = group_qty
            if group_qty:
                group["entry_price"] = group["_entry_notional_for_avg"] / group_qty
            group["realized_pnl"] = (
                _safe_float(group.get("realized_pnl"), 0.0) or 0.0
            ) + realized_pnl
            group["split_count"] = int(group.get("split_count") or 1) + 1
            group.setdefault("position_ids", []).append(item.get("id"))
            group["id"] = min(
                [pid for pid in group.get("position_ids", []) if pid is not None]
                or [group.get("id")]
            )
            order_based_status = (
                group.get("close_status_source") == "order"
                or item.get("close_status_source") == "order"
            )
            if order_based_status:
                group["close_status_source"] = "order"
                if group.get("close_status") == "manual" or item.get("close_status") == "manual":
                    group["close_status"] = "manual"
                elif (
                    group.get("close_status") == "partial" or item.get("close_status") == "partial"
                ):
                    group["close_status"] = "partial"
                elif group.get("close_status") == "full" or item.get("close_status") == "full":
                    group["close_status"] = "full"
            else:
                group["close_status"] = (
                    "partial"
                    if group.get("close_status") == "partial"
                    or item.get("close_status") == "partial"
                    else group.get("close_status")
                )
            group["close_status_label"] = (
                "部分平仓"
                if group.get("close_status") == "partial"
                else group.get("close_status_label")
            )
            if group.get("close_status") == "manual":
                group["close_status_label"] = MANUAL_CLOSE_LABEL
            if group.get("close_status") == "full":
                group["close_status_label"] = "全部平仓"
            group["position_status"] = group["close_status_label"]
            if (
                int(group.get("split_count") or 1) > 1
                and group.get("close_status_source") != "order"
            ):
                group["close_status"] = "partial"
                group["close_status_label"] = "部分平仓"
                group["position_status"] = "部分平仓"

        for item in grouped_positions:
            item.pop("_entry_notional_for_avg", None)
        positions = grouped_positions

    display_total = (
        len(positions) if (open_only and exchange_symbols is not None) or closed_only else total
    )
    display_open_count = (
        len(positions) if open_only and exchange_symbols is not None else open_count
    )
    display_total_pages = (
        max(1, (display_total + page_size - 1) // page_size) if display_total else 1
    )
    page = min(page, display_total_pages)
    if should_paginate_after_display_filter:
        start = (page - 1) * page_size
        positions = positions[start : start + page_size]
    return {
        "positions": positions,
        "count": display_open_count,
        "total": display_total,
        "page": page,
        "page_size": page_size,
        "total_pages": display_total_pages,
    }


@router.get("/decisions")
async def get_decisions(
    limit: int = 200,
    page: int = 1,
    page_size: int | None = None,
    model_name: str | None = None,
    action: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    was_executed: bool | None = None,
    is_paper: bool | None = None,
):
    """Get AI decisions from DB with optional filters.

    - start_date / end_date: ISO datetime strings for time range filter
    - model_name: filter by AI model name
    - action: filter by decision direction/action
    - was_executed: true=executed only, false=pending only, omit=all
    - is_paper: true=paper trading, false=live trading, omit=all
    """
    from datetime import datetime as dt

    from sqlalchemy import select

    from db.repositories.decision_repo import DecisionRepository
    from db.session import get_session_ctx
    from models.trade import Order

    effective_page_size = page_size if page_size is not None else limit
    effective_page_size = max(1, min(int(effective_page_size or 50), 500))
    page = max(int(page or 1), 1)
    offset = (page - 1) * effective_page_size
    start_dt = dt.fromisoformat(start_date) if start_date else None
    end_dt = dt.fromisoformat(end_date) if end_date else None
    allowed_actions = {"long", "short", "close_long", "close_short", "hold"}
    action_filter = (action or "").strip().lower() or None
    if action_filter not in allowed_actions:
        action_filter = None

    async with get_session_ctx() as session:
        repo = DecisionRepository(session)
        rows = await repo.get_recent_decisions(
            model_name=model_name,
            action=action_filter,
            limit=effective_page_size,
            offset=offset,
            start_date=start_dt,
            end_date=end_dt,
            was_executed=was_executed,
            is_paper=is_paper,
        )
        total = await repo.count_decisions(
            model_name=model_name,
            action=action_filter,
            start_date=start_dt,
            end_date=end_dt,
            was_executed=was_executed,
            is_paper=is_paper,
        )
        order_map = {}
        decision_ids = [d.id for d in rows]
        if decision_ids:
            order_result = await session.execute(
                select(Order)
                .where(Order.decision_id.in_(decision_ids))
                .order_by(Order.created_at.desc())
            )
            for order in order_result.scalars().all():
                if order.decision_id not in order_map:
                    order_map[order.decision_id] = order

    decisions = []
    for d in rows:
        decision_type, decision_type_label = _decision_type(d.action)
        raw = d.raw_llm_response if isinstance(d.raw_llm_response, dict) else {}
        decisions.append(
            sanitize_payload(
                {
                    "id": d.id,
                    "model_name": d.model_name,
                    "symbol": _normalize_dashboard_symbol(d.symbol),
                    "action": d.action,
                    "decision_type": decision_type,
                    "decision_type_label": decision_type_label,
                    "confidence": d.confidence,
                    "reasoning": sanitize_text(d.reasoning),
                    "position_size_pct": d.position_size_pct,
                    "was_executed": d.was_executed,
                    "execution_reason": _display_execution_reason(d, order_map.get(d.id)),
                    "executed_at": d.executed_at.isoformat() if d.executed_at else None,
                    "execution_price": d.execution_price,
                    "created_at": d.created_at.isoformat() if d.created_at else None,
                    "outcome": d.outcome,
                    "is_paper": d.is_paper,
                    "opportunity_score": (
                        raw.get("opportunity_score")
                        if isinstance(raw.get("opportunity_score"), dict)
                        else None
                    ),
                }
            )
        )

    total_pages = max(1, (total + effective_page_size - 1) // effective_page_size) if total else 1
    return {
        "decisions": decisions,
        "count": len(decisions),
        "total": total,
        "page": page,
        "page_size": effective_page_size,
        "total_pages": total_pages,
    }


def _opening_funnel_reason_bucket(reason: str | None) -> str:
    text = str(reason or "").lower()
    if not text:
        return "unknown"
    if any(
        token in text
        for token in (
            "动态证据",
            "证据评分",
            "候选评分未达",
            "机会评分",
            "entry evidence",
            "entry_evidence",
            "evidence",
            "探索下限",
            "硬拦",
            "弱冲突",
        )
    ):
        return "evidence_gate"
    if any(
        token in text
        for token in (
            "risk",
            "风控",
            "熔断",
            "circuit",
            "confidence",
            "置信",
            "adx",
            "成交量",
            "volume",
            "趋势",
            "仓位",
            "capacity",
            "position limit",
            "止盈",
            "止损",
            "minimum",
            "too low",
            "余额",
            "balance",
        )
    ):
        return "risk_or_precheck"
    if any(token in text for token in ("等待", "排队", "candidate", "queue", "staged", "本轮")):
        return "waiting_queue"
    if any(
        token in text
        for token in (
            "okx",
            "订单",
            "order",
            "下单",
            "执行",
            "executor",
            "timeout",
            "超时",
            "接口",
            "api",
        )
    ):
        return "execution_or_exchange"
    if any(token in text for token in ("成本", "token", "budget", "预算")):
        return "ai_budget"
    return "other"


def _opening_funnel_is_repair_cleanup(decision) -> bool:
    """Rows created only to document a local repair must not pollute the entry funnel."""
    reason = str(getattr(decision, "execution_reason", "") or "")
    if "已清理修复前误记" in reason:
        return True
    if "未确认成交" in reason and "撤销" in reason:
        return True
    return False


@router.get("/opening-funnel")
async def get_opening_funnel(
    mode: str | None = None,
    hours: int = 24,
    limit: int = 500,
):
    """Diagnose where new entries are filtered out before becoming positions."""
    from sqlalchemy import select

    from db.session import get_session_ctx
    from models.decision import AIDecision
    from models.trade import Order

    selected_mode = "live" if mode == "live" else "paper"
    is_paper = selected_mode == "paper"
    capped_hours = max(1, min(int(hours or 24), 24 * 30))
    capped_limit = max(50, min(int(limit or 500), 2000))
    since = datetime.now(UTC) - timedelta(hours=capped_hours)

    async with get_session_ctx() as session:
        stmt = (
            select(AIDecision)
            .where(
                AIDecision.model_name == ENSEMBLE_TRADER_NAME,
                AIDecision.is_paper == is_paper,
                AIDecision.created_at >= since,
            )
            .order_by(AIDecision.created_at.desc())
            .limit(capped_limit)
        )
        result = await session.execute(stmt)
        rows = list(result.scalars().all())
        decision_ids = [row.id for row in rows]

        order_map: dict[int, Order] = {}
        if decision_ids:
            order_result = await session.execute(
                select(Order)
                .where(Order.decision_id.in_(decision_ids))
                .order_by(Order.created_at.desc())
            )
            for order in order_result.scalars().all():
                if order.decision_id is not None and order.decision_id not in order_map:
                    order_map[order.decision_id] = order

    market_rows = []
    repair_cleanup_rows = 0
    for row in rows:
        raw = row.raw_llm_response if isinstance(row.raw_llm_response, dict) else {}
        analysis_type = str(row.analysis_type or raw.get("analysis_type") or "").lower()
        if analysis_type == "position" or str(row.action or "").lower() in {
            "close_long",
            "close_short",
        }:
            continue
        if _opening_funnel_is_repair_cleanup(row):
            repair_cleanup_rows += 1
            continue
        market_rows.append(row)

    action_counts = {"hold": 0, "long": 0, "short": 0, "other": 0}
    reason_buckets = {
        "evidence_gate": 0,
        "risk_or_precheck": 0,
        "waiting_queue": 0,
        "execution_or_exchange": 0,
        "ai_budget": 0,
        "other": 0,
        "unknown": 0,
    }
    symbol_counts: dict[str, dict[str, int]] = {}
    recent_blocked: list[dict[str, Any]] = []

    entry_signals = 0
    orders_created = 0
    executed_entries = 0
    no_order_after_signal = 0
    confidence_total = 0.0
    confidence_count = 0

    for row in market_rows:
        action = str(row.action or "").lower()
        action_key = action if action in {"hold", "long", "short"} else "other"
        action_counts[action_key] += 1
        symbol = _normalize_dashboard_symbol(row.symbol)
        symbol_state = symbol_counts.setdefault(
            symbol or "-", {"scans": 0, "signals": 0, "executed": 0}
        )
        symbol_state["scans"] += 1
        try:
            confidence_total += float(row.confidence or 0.0)
            confidence_count += 1
        except (TypeError, ValueError):
            pass

        if action not in {"long", "short"}:
            continue

        entry_signals += 1
        symbol_state["signals"] += 1
        matched_order = order_map.get(row.id)
        if matched_order is not None:
            orders_created += 1
        if row.was_executed:
            executed_entries += 1
            symbol_state["executed"] += 1
            continue
        if matched_order is None:
            no_order_after_signal += 1

        reason = _display_execution_reason(row, matched_order)
        bucket = _opening_funnel_reason_bucket(reason)
        reason_buckets[bucket] += 1
        if len(recent_blocked) < 20:
            recent_blocked.append(
                sanitize_payload(
                    {
                        "id": row.id,
                        "created_at": row.created_at.isoformat() if row.created_at else None,
                        "symbol": symbol,
                        "action": action,
                        "confidence": row.confidence,
                        "reason_bucket": bucket,
                        "reason": reason or "未保存具体未执行原因。",
                        "has_order": matched_order is not None,
                    }
                )
            )

    total_scans = len(market_rows)
    hold_count = action_counts["hold"]
    signal_rate = (entry_signals / total_scans) if total_scans else 0.0
    order_rate = (orders_created / entry_signals) if entry_signals else 0.0
    execution_rate = (executed_entries / entry_signals) if entry_signals else 0.0
    overall_open_rate = (executed_entries / total_scans) if total_scans else 0.0

    bottleneck = "no_data"
    bottleneck_label = "暂无足够数据"
    if total_scans:
        if signal_rate < 0.08:
            bottleneck = "ai_hold"
            bottleneck_label = "AI 主要选择观望"
        elif entry_signals and order_rate < 0.5:
            top_bucket = max(reason_buckets.items(), key=lambda item: item[1])[0]
            bottleneck = top_bucket
            label_map = {
                "evidence_gate": "动态证据评分/机会评分拦截",
                "risk_or_precheck": "风控或入场预检拦截",
                "waiting_queue": "候选排队后未进入执行",
                "execution_or_exchange": "执行层或交易所接口问题",
                "ai_budget": "AI 成本预算仍在拦截",
                "other": "未执行原因分散",
                "unknown": "缺少未执行原因",
            }
            bottleneck_label = label_map.get(top_bucket, "开仓信号未形成订单")
        elif execution_rate < 0.75:
            bottleneck = "execution"
            bottleneck_label = "订单生成后执行/成交不足"
        else:
            bottleneck = "healthy_selective"
            bottleneck_label = "漏斗正常，开仓少主要来自选择性交易"

    top_symbols = sorted(
        (
            {
                "symbol": symbol,
                "scans": counts["scans"],
                "signals": counts["signals"],
                "executed": counts["executed"],
                "signal_rate": counts["signals"] / counts["scans"] if counts["scans"] else 0.0,
                "execution_rate": (
                    counts["executed"] / counts["signals"] if counts["signals"] else 0.0
                ),
            }
            for symbol, counts in symbol_counts.items()
        ),
        key=lambda item: (item["signals"], item["scans"]),
        reverse=True,
    )[:12]

    return sanitize_payload(
        {
            "mode": selected_mode,
            "window_hours": capped_hours,
            "sample_limit": capped_limit,
            "sampled_decisions": len(rows),
            "repair_cleanup_rows": repair_cleanup_rows,
            "market_scans": total_scans,
            "average_confidence": (
                (confidence_total / confidence_count) if confidence_count else 0.0
            ),
            "stages": {
                "market_scans": total_scans,
                "ai_entry_signals": entry_signals,
                "orders_created": orders_created,
                "executed_entries": executed_entries,
            },
            "rates": {
                "signal_rate": signal_rate,
                "order_rate": order_rate,
                "execution_rate": execution_rate,
                "overall_open_rate": overall_open_rate,
            },
            "action_counts": action_counts,
            "reason_buckets": reason_buckets,
            "hold_count": hold_count,
            "no_order_after_signal": no_order_after_signal,
            "bottleneck": bottleneck,
            "bottleneck_label": bottleneck_label,
            "top_symbols": top_symbols,
            "recent_blocked": recent_blocked,
            "generated_at": datetime.now(UTC).isoformat(),
        }
    )


@router.get("/analysis-records")
async def get_analysis_records(
    limit: int = 50,
    page: int = 1,
    page_size: int | None = None,
    decision_id: int | None = None,
    analysis_type: str | None = None,
    include_detail: bool = True,
    symbol: str | None = None,
    expert_name: str | None = None,
    is_paper: bool | None = None,
):
    """Return one collaboration record per ensemble decision."""
    from sqlalchemy import func, select

    from db.repositories.decision_repo import DecisionRepository
    from db.session import get_session_ctx
    from models.decision import AIDecision
    from models.trade import Position

    effective_page_size = page_size if page_size is not None else limit
    effective_page_size = max(1, min(int(effective_page_size or 50), 200))
    page = max(int(page or 1), 1)
    offset = (page - 1) * effective_page_size
    normalized_analysis_type = str(analysis_type or "").lower()
    if normalized_analysis_type not in {"market", "position"}:
        normalized_analysis_type = ""
    selected_mode = (
        "paper" if is_paper is True else "live" if is_paper is False else mode_manager.mode.value
    )
    current_position_symbols: set[str] = set()
    if normalized_analysis_type == "position" and decision_id is None:
        current_position_symbols = await _get_display_open_position_symbols(selected_mode)

    async with get_session_ctx() as session:
        repo = DecisionRepository(session)
        needs_server_filter = bool(normalized_analysis_type or expert_name)
        db_type_filter = bool(normalized_analysis_type and not expert_name and decision_id is None)
        total_without_server_filter = 0
        if decision_id is not None:
            decision_stmt = select(AIDecision).where(
                AIDecision.id == decision_id,
                AIDecision.model_name == ENSEMBLE_TRADER_NAME,
            )
            if is_paper is not None:
                decision_stmt = decision_stmt.where(AIDecision.is_paper == is_paper)
            decision_result = await session.execute(decision_stmt)
            all_rows = list(decision_result.scalars().all())
        elif db_type_filter:
            filters = [
                AIDecision.model_name == ENSEMBLE_TRADER_NAME,
                AIDecision.analysis_type == normalized_analysis_type,
            ]
            if symbol:
                filters.append(AIDecision.symbol == symbol)
            elif normalized_analysis_type == "position":
                if current_position_symbols:
                    filters.append(AIDecision.symbol.in_(current_position_symbols))
                else:
                    filters.append(AIDecision.id == -1)
            if is_paper is not None:
                filters.append(AIDecision.is_paper == is_paper)
            decision_stmt = (
                select(AIDecision)
                .where(*filters)
                .order_by(AIDecision.created_at.desc())
                .offset(offset)
                .limit(effective_page_size)
            )
            decision_result = await session.execute(decision_stmt)
            all_rows = list(decision_result.scalars().all())
            count_result = await session.execute(select(func.count(AIDecision.id)).where(*filters))
            total_without_server_filter = int(count_result.scalar() or 0)
        else:
            all_rows = await repo.get_recent_decisions(
                model_name=ENSEMBLE_TRADER_NAME,
                symbol=symbol,
                limit=5000 if needs_server_filter else effective_page_size,
                offset=0 if needs_server_filter else offset,
                is_paper=is_paper,
            )
        if decision_id is not None:
            total_without_server_filter = len(all_rows)
        elif db_type_filter:
            pass
        elif not needs_server_filter:
            total_without_server_filter = await repo.count_decisions(
                model_name=ENSEMBLE_TRADER_NAME,
                is_paper=is_paper,
            )
        position_stmt = select(
            Position.execution_mode,
            Position.symbol,
        ).where(
            Position.model_name == ENSEMBLE_TRADER_NAME,
            Position.is_open.is_(True),
        )
        if is_paper is not None:
            position_stmt = position_stmt.where(
                Position.execution_mode == ("paper" if is_paper else "live")
            )
        position_result = await session.execute(position_stmt)
        open_position_keys = {
            (row.execution_mode, _normalize_dashboard_symbol(row.symbol))
            for row in position_result.all()
            if _normalize_dashboard_symbol(row.symbol)
        }

    def infer_analysis_type(decision, raw: dict) -> tuple[str, str]:
        explicit = str(raw.get("analysis_type") or "").lower()
        if explicit in {"position", "position_review", "holding", "holdings"}:
            return "position", "持仓分析"
        if explicit in {"market", "market_scan", "symbol_scan"}:
            return "market", "市场分析"

        if raw.get("position_review_policy") or raw.get("position_review"):
            return "position", "持仓分析"

        if decision.action in {"close_long", "close_short"}:
            return "position", "持仓分析"

        return "market", "市场分析"

    def infer_position_lifecycle(decision, analysis_type: str) -> tuple[str | None, str | None]:
        if analysis_type != "position":
            return None, None

        decision_mode = "paper" if decision.is_paper else "live"
        symbol_key = _normalize_dashboard_symbol(decision.symbol)
        if (decision_mode, symbol_key) in open_position_keys:
            return "holding", "持仓中"
        return "closed", "已平仓"

    records: list[dict[str, Any]] = []
    filtered_count = 0
    for d in all_rows:
        raw = _safe_dict(d.raw_llm_response)
        if not raw:
            continue
        opinions = raw.get("opinions") or []
        if not isinstance(opinions, list):
            continue
        model_timings = raw.get("model_timings") or []
        if not isinstance(model_timings, list):
            model_timings = []
        timings_by_name = {
            str(item.get("name")): item
            for item in model_timings
            if isinstance(item, dict) and item.get("name")
        }

        experts = []
        for opinion in opinions:
            if not isinstance(opinion, dict):
                continue
            name = opinion.get("model_name") or ""
            experts.append(
                {
                    "expert_name": name,
                    "expert_label": opinion.get("label") or name,
                    "role": opinion.get("role") or "",
                    "action": opinion.get("action") or "hold",
                    "confidence": opinion.get("confidence") or 0.0,
                    "weight": opinion.get("weight") or 0.0,
                    "reasoning": opinion.get("reasoning") or "",
                    "cross_check_for": opinion.get("cross_check_for"),
                    "timeout_fallback": bool(opinion.get("timeout_fallback")),
                    "latency": timings_by_name.get(str(name)),
                }
            )

        if expert_name and not any(e["expert_name"] == expert_name for e in experts):
            continue

        cross_validations = raw.get("cross_validations") or []
        consultation = raw.get("consultation")
        conflict_resolution = raw.get("conflict_resolution") or {}
        attempted_experts = raw.get("attempted_experts") or []
        failure_rows = raw.get("expert_failures") or []
        if not isinstance(attempted_experts, list):
            attempted_experts = []
        if not isinstance(failure_rows, list):
            failure_rows = []
        failures_by_name = {
            str(item.get("expert_name")): _humanize_expert_failure(item.get("reason"))
            for item in failure_rows
            if isinstance(item, dict) and item.get("expert_name")
        }
        cross_requested = sum(1 for e in experts if e.get("cross_check_for"))
        divergent = sum(1 for v in cross_validations if v.get("consistency") == "divergent")
        aligned = sum(1 for v in cross_validations if v.get("consistency") == "aligned")
        major_conflicts = sum(1 for v in cross_validations if v.get("major_conflict"))
        unavailable_validations = sum(
            1 for v in cross_validations if v.get("validation_status") == "target_missing"
        )
        completed_validations = sum(
            1 for v in cross_validations if v.get("validation_status", "completed") == "completed"
        )

        expected_experts = [
            {
                "expert_name": slot.get("name", ""),
                "expert_label": slot.get("label", slot.get("name", "")),
                "role": slot.get("role", ""),
            }
            for slot in FIXED_AI_MODEL_SLOTS
            if slot.get("name") != DECISION_MAKER_NAME
        ]
        returned_names = {e["expert_name"] for e in experts}
        attempted_names = {str(name) for name in attempted_experts}
        fast_scan_payload = _safe_dict(raw.get("position_fast_scan"))
        fast_scan_skipped_llm = bool(fast_scan_payload.get("skipped_llm"))
        attempted_expert_count = (
            0 if fast_scan_skipped_llm else (len(attempted_names) or len(expected_experts))
        )
        missing_experts = [
            {
                **e,
                "latency": timings_by_name.get(e["expert_name"]),
                "reason": (
                    "本轮是持仓快速扫描记录，没有调用慢专家；只有出现强平仓、强加仓或高风险信号时才会插队进入专家深度复盘。"
                    if fast_scan_skipped_llm
                    else failures_by_name.get(e["expert_name"])
                    or (
                        "未发起调用，可能是该专家未启用或未配置 API Key。"
                        if attempted_names and e["expert_name"] not in attempted_names
                        else "本轮未返回结果，可能是模型调用失败、超时或返回格式不符合 JSON 要求。"
                    )
                ),
            }
            for e in expected_experts
            if e["expert_name"] not in returned_names
        ]
        trade_confidence = _clamp_confidence(_safe_float(d.confidence, 0.0))
        display_confidence = _analysis_display_confidence(d.action, trade_confidence, experts, raw)
        analysis_type, analysis_type_label = infer_analysis_type(d, raw)
        position_lifecycle_status, position_lifecycle_label = infer_position_lifecycle(
            d, analysis_type
        )
        if normalized_analysis_type and analysis_type != normalized_analysis_type:
            continue
        if (
            normalized_analysis_type == "position"
            and decision_id is None
            and not symbol
            and _normalize_dashboard_symbol(d.symbol) not in current_position_symbols
        ):
            continue
        if expert_name and not any(e["expert_name"] == expert_name for e in experts):
            continue

        filtered_count += 1
        if needs_server_filter and not db_type_filter:
            if filtered_count <= offset:
                continue
            if len(records) >= effective_page_size:
                continue

        detail_payload = (
            {
                "experts": experts,
                "missing_experts": missing_experts,
                "cross_validations": cross_validations,
                "consultation": consultation,
                "decision_maker": (
                    raw.get("decision_maker")
                    if isinstance(raw.get("decision_maker"), dict)
                    else None
                ),
                "decision_attribution": _build_decision_attribution(d, raw, experts),
                "ml_signal": (
                    raw.get("ml_signal") if isinstance(raw.get("ml_signal"), dict) else None
                ),
                "local_ai_tools": (
                    raw.get("local_ai_tools")
                    if isinstance(raw.get("local_ai_tools"), dict)
                    else None
                ),
                "agent_skills": (
                    raw.get("agent_skills") if isinstance(raw.get("agent_skills"), dict) else None
                ),
                "news_context": (
                    raw.get("news_context") if isinstance(raw.get("news_context"), dict) else None
                ),
                "opportunity_score": (
                    raw.get("opportunity_score")
                    if isinstance(raw.get("opportunity_score"), dict)
                    else None
                ),
                "position_review_policy": (
                    raw.get("position_review_policy")
                    if isinstance(raw.get("position_review_policy"), dict)
                    else {}
                ),
                "position_fast_scan": fast_scan_payload,
                "close_evidence": _safe_dict(raw.get("close_evidence")),
                "add_evidence": _safe_dict(raw.get("add_evidence")),
                "timing": _safe_dict(raw.get("timing")),
                "timing_breakdown": (
                    raw.get("timing_breakdown")
                    if isinstance(raw.get("timing_breakdown"), list)
                    else []
                ),
                "model_timings": model_timings,
                "latency_summary": (
                    raw.get("latency_summary")
                    if isinstance(raw.get("latency_summary"), dict)
                    else {}
                ),
                "conflict_resolution": (
                    conflict_resolution if isinstance(conflict_resolution, dict) else {}
                ),
            }
            if include_detail
            else {}
        )

        record = {
            "id": str(d.id),
            "decision_id": d.id,
            "analysis_type": d.analysis_type or analysis_type,
            "analysis_type_label": (
                "持仓分析" if (d.analysis_type or analysis_type) == "position" else "市场分析"
            ),
            "position_lifecycle_status": position_lifecycle_status,
            "position_lifecycle_label": position_lifecycle_label,
            "created_at": d.created_at.isoformat() if d.created_at else None,
            "symbol": _normalize_dashboard_symbol(d.symbol),
            "expert_count": len(experts),
            "expected_expert_count": len(expected_experts),
            "attempted_expert_count": attempted_expert_count,
            "attempted_experts": attempted_experts,
            "position_fast_scan": fast_scan_payload,
            "cross_requested": cross_requested,
            "cross_summary": {
                "total": len(cross_validations),
                "completed": completed_validations,
                "unavailable": unavailable_validations,
                "aligned": aligned,
                "divergent": divergent,
                "major_conflicts": major_conflicts,
            },
            "consultation_status": (
                consultation.get("status") if isinstance(consultation, dict) else None
            ),
            "validation_adjustment": raw.get("validation_adjustment"),
            "final_action": d.action,
            "final_confidence": display_confidence,
            "trade_confidence": trade_confidence,
            "confidence_note": (
                "观望/不下单时，信心度显示专家加权平均分析信心；内部下单信心仍为 0，避免误触发交易。"
                if d.action == "hold" and trade_confidence == 0.0
                else "信心度来自最终可执行裁决。"
            ),
            "final_reasoning": sanitize_text(d.reasoning),
            "position_size_pct": d.position_size_pct,
            "weighted_score": raw.get("weighted_score"),
            "disagreement": raw.get("disagreement"),
            "was_executed": d.was_executed,
            "execution_reason": _display_execution_reason(d),
            "is_paper": d.is_paper,
            "flow_summary": (
                "持仓快速扫描：未调用 5 个专家；发现强平仓、强加仓或高风险信号时才进入专家深度复盘。"
                if fast_scan_skipped_llm
                else (
                    f"{len(experts)}/{len(expected_experts)} 个专家返回，"
                    f"{cross_requested} 个交叉验证请求，"
                    f"{completed_validations} 次完成，"
                    f"{unavailable_validations} 次无法验证，"
                    f"{major_conflicts} 个重大矛盾"
                )
            ),
        }
        record.update(detail_payload)
        records.append(sanitize_payload(record))

    total = (
        total_without_server_filter
        if db_type_filter
        else (filtered_count if needs_server_filter else total_without_server_filter)
    )
    total_pages = max(1, (total + effective_page_size - 1) // effective_page_size) if total else 1

    return {
        "records": records,
        "count": len(records),
        "total": total,
        "page": page,
        "page_size": effective_page_size,
        "total_pages": total_pages,
    }


@router.get("/strategy-learning")
async def get_strategy_learning(
    mode: str | None = None,
    hours: int = 168,
    limit: int = 3000,
):
    """Return the active strategy-learning feedback and scheduler state."""
    from services.strategy_learning import StrategyLearningService

    selected_mode = "live" if str(mode or "").lower() == "live" else "paper"
    service = getattr(_trading_service, "strategy_learning_service", None)
    if service is None:
        service = StrategyLearningService()
    payload = await service.dashboard_payload(
        mode=selected_mode,
        hours=max(1, min(int(hours or 168), 24 * 90)),
        limit=max(100, min(int(limit or 3000), 20000)),
        max_open_positions=int(settings.max_open_positions_per_model or 20),
    )
    return sanitize_payload(payload)


@router.post("/strategy-learning/profiles/{profile_id}/disabled")
async def set_strategy_learning_profile_disabled(
    profile_id: str,
    disabled: bool = True,
    reason: str | None = None,
):
    """Disable or re-enable one strategy profile."""
    from services.strategy_learning import StrategyLearningService

    service = getattr(_trading_service, "strategy_learning_service", None)
    if service is None:
        service = StrategyLearningService()
    state = service.set_profile_disabled(
        profile_id,
        disabled=bool(disabled),
        reason=reason or "dashboard_manual_control",
    )
    return sanitize_payload({"profile_id": profile_id, "disabled": bool(disabled), "state": state})


@router.post("/strategy-learning/profiles/{profile_id}/activate")
async def activate_strategy_learning_profile(profile_id: str):
    """Manually select a strategy profile until rollback or another selection."""
    from services.strategy_learning import StrategyLearningService

    service = getattr(_trading_service, "strategy_learning_service", None)
    if service is None:
        service = StrategyLearningService()
    state = service.set_manual_active_profile(profile_id)
    return sanitize_payload({"profile_id": profile_id, "state": state})


@router.post("/strategy-learning/rollback")
async def rollback_strategy_learning_profile():
    """Rollback the scheduler to the baseline strategy profile."""
    from services.strategy_learning import StrategyLearningService

    service = getattr(_trading_service, "strategy_learning_service", None)
    if service is None:
        service = StrategyLearningService()
    state = service.rollback_to_baseline()
    return sanitize_payload({"profile_id": "baseline_current", "state": state})


@router.get("/profit-attribution")
async def get_profit_attribution(
    mode: str | None = None,
    hours: int = 24,
    limit: int = 200,
):
    """Explain why recent closed trades made or lost money."""
    from sqlalchemy import or_, select

    from db.session import get_session_ctx
    from models.decision import AIDecision
    from models.learning import ShadowBacktest
    from models.trade import Order, Position
    from services.profit_attribution import (
        build_profit_attribution,
        match_entry_decisions_for_positions,
    )

    selected_mode = "live" if str(mode or "").lower() == "live" else "paper"
    is_paper = selected_mode == "paper"
    capped_hours = max(1, min(int(hours or 24), 720))
    max_rows = max(20, min(int(limit or 200), 1000))
    since = datetime.now(UTC) - timedelta(hours=capped_hours)

    async with get_session_ctx() as session:
        position_result = await session.execute(
            select(Position)
            .where(
                Position.model_name == ENSEMBLE_TRADER_NAME,
                Position.execution_mode == selected_mode,
                Position.is_open.is_(False),
                Position.closed_at.is_not(None),
                Position.closed_at >= since,
            )
            .order_by(Position.closed_at.desc(), Position.created_at.desc())
            .limit(max_rows)
        )
        positions = list(position_result.scalars().all())
        if not positions:
            return {
                "mode": selected_mode,
                "window_hours": capped_hours,
                "summary": {
                    "trade_count": 0,
                    "total_closed_pnl": 0.0,
                    "win_count": 0,
                    "loss_count": 0,
                    "win_rate": 0.0,
                    "avg_win": 0.0,
                    "avg_loss": 0.0,
                    "profit_factor": 0.0,
                },
                "buckets": [],
                "records": [],
                "message": "最近窗口内暂无已平仓记录。",
            }

        symbols = {p.symbol for p in positions if p.symbol}
        symbol_variants = _dashboard_symbol_query_variants(symbols)
        position_times = [
            normalized
            for p in positions
            if p.created_at and (normalized := _as_utc_datetime(p.created_at)) is not None
        ]
        earliest_position_time = min(position_times, default=since)
        order_since = min(since, earliest_position_time or since) - timedelta(hours=2)
        order_limit = max(max_rows * 20, 2000)
        order_result = await session.execute(
            select(Order)
            .where(
                Order.model_name == ENSEMBLE_TRADER_NAME,
                Order.execution_mode == selected_mode,
                Order.symbol.in_(symbol_variants) if symbol_variants else Order.id == -1,
                Order.created_at >= order_since,
            )
            .order_by(Order.filled_at.desc().nullslast(), Order.created_at.desc())
            .limit(order_limit)
        )
        orders = list(order_result.scalars().all())

        decision_ids = {int(o.decision_id) for o in orders if o.decision_id}
        decisions_by_id = {}
        if decision_ids:
            decision_result = await session.execute(
                select(AIDecision)
                .where(AIDecision.id.in_(decision_ids))
                .order_by(AIDecision.created_at.desc())
            )
            decisions_by_id.update(
                {int(row.id): row for row in decision_result.scalars().all() if row.id is not None}
            )
        if symbol_variants:
            fallback_decision_result = await session.execute(
                select(AIDecision)
                .where(
                    AIDecision.model_name == ENSEMBLE_TRADER_NAME,
                    AIDecision.symbol.in_(symbol_variants),
                    AIDecision.is_paper.is_(is_paper),
                    AIDecision.created_at >= order_since,
                )
                .order_by(AIDecision.created_at.desc())
                .limit(max(max_rows * 12, 1200))
            )
            decisions_by_id.update(
                {
                    int(row.id): row
                    for row in fallback_decision_result.scalars().all()
                    if row.id is not None and int(row.id) not in decisions_by_id
                }
            )
        decisions = list(decisions_by_id.values())

        entry_decisions = match_entry_decisions_for_positions(positions, orders, decisions)
        shadow_decision_ids = {int(d.id) for d in entry_decisions if d.id is not None}
        shadows_by_id = {}
        if shadow_decision_ids:
            shadow_result = await session.execute(
                select(ShadowBacktest)
                .where(ShadowBacktest.decision_id.in_(shadow_decision_ids))
                .order_by(ShadowBacktest.created_at.desc())
                .limit(max(len(shadow_decision_ids) * 4, max_rows * 10))
            )
            shadows_by_id.update(
                {int(row.id): row for row in shadow_result.scalars().all() if row.id is not None}
            )
        shadow_time_conditions = []
        for decision in entry_decisions:
            decision_time = _as_utc_datetime(decision.created_at)
            if not decision.symbol or not decision_time:
                continue
            decision_symbol_variants = _dashboard_symbol_query_variants({decision.symbol})
            shadow_time_conditions.append(
                (ShadowBacktest.symbol.in_(decision_symbol_variants))
                & (ShadowBacktest.execution_mode == selected_mode)
                & (ShadowBacktest.created_at >= decision_time - timedelta(minutes=30))
                & (ShadowBacktest.created_at <= decision_time + timedelta(minutes=30))
            )
        for position in positions:
            opened_at = _as_utc_datetime(position.created_at)
            if not position.symbol or not opened_at:
                continue
            position_symbol_variants = _dashboard_symbol_query_variants({position.symbol})
            shadow_time_conditions.append(
                (ShadowBacktest.symbol.in_(position_symbol_variants))
                & (ShadowBacktest.execution_mode == selected_mode)
                & (ShadowBacktest.created_at >= opened_at - timedelta(minutes=45))
                & (ShadowBacktest.created_at <= opened_at + timedelta(minutes=45))
            )
        if shadow_time_conditions:
            fallback_shadow_result = await session.execute(
                select(ShadowBacktest)
                .where(or_(*shadow_time_conditions))
                .order_by(ShadowBacktest.created_at.desc())
            )
            shadows_by_id.update(
                {
                    int(row.id): row
                    for row in fallback_shadow_result.scalars().all()
                    if row.id is not None and int(row.id) not in shadows_by_id
                }
            )
        shadows = list(shadows_by_id.values())

    payload = build_profit_attribution(positions, orders, decisions, shadows)
    return sanitize_payload(
        {
            "mode": selected_mode,
            "window_hours": capped_hours,
            "sample_limit": max_rows,
            "since": since.isoformat(),
            **payload,
            "message": "按已平仓真实盈亏，结合 AI 决策、订单、影子复盘和本地模型证据做交易级归因。",
        }
    )


@router.get("/model-contribution/stats")
async def get_model_contribution_stats(
    mode: str | None = None,
    days: int = 7,
    limit: int = 2000,
):
    """Estimate which model signals helped or hurt realized PnL."""
    from sqlalchemy import select

    from db.session import get_session_ctx
    from models.decision import AIDecision
    from models.trade import Order, Position

    selected_mode = "live" if str(mode or "").lower() == "live" else "paper"
    since = datetime.now(UTC) - timedelta(days=max(1, min(int(days or 7), 90)))
    max_rows = max(50, min(int(limit or 2000), 10000))

    buckets: dict[str, dict[str, Any]] = {
        "local_ml_aligned": {"label": "本地 ML 同向"},
        "server_profit_aligned": {"label": "服务器盈利模型同向"},
        "timeseries_aligned": {"label": "时序预测同向"},
        "sentiment_aligned": {"label": "情绪预测同向"},
        "ai_only_ml_opposed": {"label": "AI 支持但 ML 反对"},
        "high_risk_review_approved": {"label": "高风险复核通过"},
    }
    stats: dict[str, dict[str, Any]] = {
        key: {
            **meta,
            "count": 0,
            "wins": 0,
            "losses": 0,
            "pnl": 0.0,
            "profit": 0.0,
            "loss": 0.0,
        }
        for key, meta in buckets.items()
    }

    def add(key: str, pnl: float) -> None:
        bucket = stats.get(key)
        if not bucket:
            return
        bucket["count"] += 1
        bucket["pnl"] += pnl
        if pnl >= 0:
            bucket["wins"] += 1
            bucket["profit"] += pnl
        else:
            bucket["losses"] += 1
            bucket["loss"] += abs(pnl)

    async with get_session_ctx() as session:
        position_result = await session.execute(
            select(Position)
            .where(
                Position.model_name == ENSEMBLE_TRADER_NAME,
                Position.execution_mode == selected_mode,
                Position.is_open.is_(False),
                Position.closed_at.is_not(None),
                Position.closed_at >= since,
            )
            .order_by(Position.closed_at.desc())
            .limit(max_rows)
        )
        positions = list(position_result.scalars().all())
        if not positions:
            return {
                "mode": selected_mode,
                "days": days,
                "total_positions": 0,
                "stats": list(stats.values()),
                "summary": "暂无已平仓样本，等新交易完成后会自动统计。",
            }

        symbols = {p.symbol for p in positions if p.symbol}
        order_result = await session.execute(
            select(Order)
            .where(
                Order.model_name == ENSEMBLE_TRADER_NAME,
                Order.execution_mode == selected_mode,
                Order.status == "filled",
                Order.decision_id.is_not(None),
                Order.symbol.in_(symbols) if symbols else Order.id == -1,
            )
            .order_by(Order.filled_at.desc(), Order.created_at.desc())
            .limit(max_rows * 3)
        )
        orders = list(order_result.scalars().all())
        decision_ids = [o.decision_id for o in orders if o.decision_id]
        decisions: dict[int, AIDecision] = {}
        if decision_ids:
            decision_result = await session.execute(
                select(AIDecision).where(AIDecision.id.in_(decision_ids))
            )
            decisions = {d.id: d for d in decision_result.scalars().all()}

    def aware(value):
        if value and value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value

    for pos in positions:
        pos_created = aware(pos.created_at)
        candidates = []
        for order in orders:
            if order.symbol != pos.symbol:
                continue
            if order.decision_id not in decisions:
                continue
            decision = decisions[order.decision_id]
            action_side = _side_from_action(decision.action)
            if action_side != str(pos.side or "").lower():
                continue
            order_time = aware(order.filled_at or order.created_at)
            if pos_created and order_time and abs((order_time - pos_created).total_seconds()) > 180:
                continue
            candidates.append(
                (
                    (
                        abs(((order_time or pos_created) - pos_created).total_seconds())
                        if pos_created and order_time
                        else 0
                    ),
                    decision,
                )
            )
        if not candidates:
            continue
        _, decision = sorted(candidates, key=lambda item: item[0])[0]
        raw = _safe_dict(decision.raw_llm_response)
        side = _side_from_action(decision.action)
        pnl = float(pos.realized_pnl or 0.0)
        ml = _extract_primary_ml(raw)
        local = _extract_local_tools(raw)
        if ml.get("side") == side:
            add("local_ml_aligned", pnl)
        elif (
            side in {"long", "short"}
            and ml.get("side") in {"long", "short"}
            and ml.get("side") != side
        ):
            add("ai_only_ml_opposed", pnl)
        if _safe_dict(local.get("profit")).get("side") == side:
            add("server_profit_aligned", pnl)
        if _safe_dict(local.get("timeseries")).get("side") == side:
            add("timeseries_aligned", pnl)
        if _safe_dict(local.get("sentiment")).get("side") == side:
            add("sentiment_aligned", pnl)
        review = _safe_dict(raw.get("high_risk_review"))
        if review.get("triggered") and review.get("approved") is True:
            add("high_risk_review_approved", pnl)

    result = []
    for item in stats.values():
        count = max(int(item["count"]), 0)
        item["pnl"] = round(float(item["pnl"]), 6)
        item["profit"] = round(float(item["profit"]), 6)
        item["loss"] = round(float(item["loss"]), 6)
        item["avg_pnl"] = round(item["pnl"] / count, 6) if count else 0.0
        item["win_rate"] = round(item["wins"] / count, 4) if count else 0.0
        item["profit_factor"] = (
            round(item["profit"] / item["loss"], 4)
            if item["loss"] > 0
            else (999.0 if item["profit"] > 0 else 0.0)
        )
        result.append(item)

    return {
        "mode": selected_mode,
        "days": days,
        "total_positions": len(positions),
        "stats": result,
        "summary": "按已平仓真实盈亏回看各模型同向信号贡献，用于后续自动降权/加权。",
    }


@router.get("/expert-memories")
async def get_expert_memories(
    limit: int = 10,
    page_size: int = 10,
    memory_page: int = 1,
    reflection_page: int = 1,
    expert_name: str | None = None,
    symbol: str | None = None,
):
    """Return long-term expert memories and recent trade reflections."""
    from db.repositories.memory_repo import MemoryRepository
    from db.session import get_session_ctx

    size = max(min(int(page_size or limit or 10), 100), 1)
    memory_page = max(int(memory_page or 1), 1)
    reflection_page = max(int(reflection_page or 1), 1)
    memory_offset = (memory_page - 1) * size
    reflection_offset = (reflection_page - 1) * size

    async with get_session_ctx() as session:
        repo = MemoryRepository(session)
        memory_total = await repo.count_memories(
            expert_name=expert_name,
            symbol=symbol,
        )
        reflection_total = await repo.count_reflections(symbol=symbol)
        memories = await repo.list_memories(
            expert_name=expert_name,
            symbol=symbol,
            limit=size,
            offset=memory_offset,
        )
        reflections = await repo.list_reflections(
            symbol=symbol,
            limit=size,
            offset=reflection_offset,
        )

    memory_rows = [
        {
            "id": m.id,
            "expert_name": m.expert_name,
            "expert_label": m.expert_label,
            "symbol": m.symbol,
            "side": m.side,
            "memory_type": m.memory_type,
            "market_pattern": sanitize_text(m.market_pattern),
            "lesson": sanitize_text(m.lesson),
            "recommended_action": m.recommended_action,
            "confidence_adjustment": m.confidence_adjustment,
            "position_size_multiplier": m.position_size_multiplier,
            "evidence_count": m.evidence_count,
            "hit_count": m.hit_count,
            "success_count": m.success_count,
            "failure_count": m.failure_count,
            "confidence_score": m.confidence_score,
            "last_used_at": m.last_used_at.isoformat() if m.last_used_at else None,
            "created_at": m.created_at.isoformat() if m.created_at else None,
            "updated_at": m.updated_at.isoformat() if m.updated_at else None,
        }
        for m in memories
    ]
    reflection_rows = [
        {
            "id": r.id,
            "position_id": r.position_id,
            "symbol": r.symbol,
            "side": r.side,
            "entry_price": r.entry_price,
            "exit_price": r.exit_price,
            "quantity": r.quantity,
            "realized_pnl": r.realized_pnl,
            "fee_estimate": r.fee_estimate,
            "hold_minutes": r.hold_minutes,
            "closed_at": r.closed_at.isoformat() if r.closed_at else None,
            "outcome": r.outcome,
            "mistake_summary": sanitize_text(r.mistake_summary),
            "improvement_summary": sanitize_text(r.improvement_summary),
            "source": r.source,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in reflections
    ]
    return {
        "memories": memory_rows,
        "reflections": reflection_rows,
        "count": memory_total,
        "reflection_count": reflection_total,
        "pagination": {
            "page_size": size,
            "memory_page": memory_page,
            "memory_total": memory_total,
            "memory_total_pages": max((memory_total + size - 1) // size, 1),
            "reflection_page": reflection_page,
            "reflection_total": reflection_total,
            "reflection_total_pages": max((reflection_total + size - 1) // size, 1),
        },
        "daily_target": _daily_target_payload(),
    }


@router.get("/shadow-backtests")
async def get_shadow_backtests(
    limit: int = 10,
    page_size: int = 10,
    page: int = 1,
    status: str | None = None,
    symbol: str | None = None,
):
    """Return shadow backtest records shown in the dashboard."""
    from db.repositories.memory_repo import MemoryRepository
    from db.session import get_session_ctx

    size = max(min(int(page_size or limit or 10), 100), 1)
    page = max(int(page or 1), 1)
    offset = (page - 1) * size
    status_filter: str | None = str(status or "").strip().lower()
    if status_filter not in {"pending", "completed"}:
        status_filter = None

    async with get_session_ctx() as session:
        repo = MemoryRepository(session)
        total = await repo.count_shadow_backtests(status=status_filter, symbol=symbol)
        pending_total = await repo.count_shadow_backtests(status="pending", symbol=symbol)
        completed_total = await repo.count_shadow_backtests(status="completed", symbol=symbol)
        rows = await repo.list_shadow_backtests(
            status=status_filter,
            symbol=symbol,
            limit=size,
            offset=offset,
        )

    records: list[dict[str, Any]] = []
    for row in rows:
        long_return = _safe_float(row.long_return_pct, None)
        short_return = _safe_float(row.short_return_pct, None)
        best_action = row.best_action or None
        raw = _safe_dict(row.raw_llm_response)
        decision_maker = _safe_dict(raw.get("decision_maker"))
        decision_note = ""
        if str(row.decision_action or "").lower() == "hold":
            decision_note = str(decision_maker.get("reason") or "").strip()
            if not decision_note:
                weighted_score = _safe_float(raw.get("weighted_score"), 0.0) or 0.0
                if abs(weighted_score) < 1e-9 and _safe_float(row.decision_confidence, 0.0) <= 0:
                    decision_note = "当时没有形成可执行开仓信号，系统记录为观望样本。"
                else:
                    decision_note = "当时最终裁决为观望，用于复盘是否错过机会。"
        records.append(
            {
                "id": row.id,
                "decision_id": row.decision_id,
                "model_name": row.model_name,
                "execution_mode": row.execution_mode,
                "symbol": row.symbol,
                "analysis_type": row.analysis_type,
                "decision_action": row.decision_action,
                "decision_action_label": _action_label_text(row.decision_action),
                "decision_confidence": row.decision_confidence,
                "decision_note": decision_note,
                "decision_maker_status": decision_maker.get("status") or "",
                "entry_price": row.entry_price,
                "status": row.status,
                "status_label": "已完成" if row.status == "completed" else "等待复盘",
                "due_at": row.due_at.isoformat() if row.due_at else None,
                "horizon_minutes": row.horizon_minutes,
                "actual_price": row.actual_price,
                "long_return_pct": long_return,
                "short_return_pct": short_return,
                "best_action": best_action,
                "best_action_label": _action_label_text(best_action),
                "missed_opportunity": bool(row.missed_opportunity),
                "conclusion": _shadow_backtest_conclusion(
                    row.decision_action,
                    best_action,
                    bool(row.missed_opportunity),
                    long_return,
                    short_return,
                ),
                "note": row.note,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            }
        )

    return {
        "records": records,
        "count": total,
        "pending_count": pending_total,
        "completed_count": completed_total,
        "pagination": {
            "page": page,
            "page_size": size,
            "total": total,
            "total_pages": max((total + size - 1) // size, 1),
        },
    }


def _action_label_text(action: str | None) -> str:
    return {
        "long": "做多",
        "short": "做空",
        "close_long": "平多",
        "close_short": "平空",
        "hold": "观望",
        None: "-",
        "": "-",
    }.get(action, str(action or "-"))


def _shadow_backtest_conclusion(
    decision_action: str | None,
    best_action: str | None,
    missed_opportunity: bool,
    long_return_pct: float | None,
    short_return_pct: float | None,
) -> str:
    if missed_opportunity and best_action in {"long", "short"}:
        return f"观望错过{_action_label_text(best_action)}机会"
    if not best_action:
        return "等待结果"
    if best_action == "hold":
        return "观望较合理"
    if decision_action == best_action:
        return "方向有效"
    if decision_action in {"long", "short"} and best_action in {"long", "short"}:
        return f"方向偏差，实际更适合{_action_label_text(best_action)}"
    best_return = max(long_return_pct or 0.0, short_return_pct or 0.0)
    if best_return > 0:
        return f"实际更适合{_action_label_text(best_action)}"
    return "结果中性"


def _daily_target_payload() -> dict:
    cny_per_usdt = max(float(settings.cny_per_usdt_assumption or 7.2), 0.0001)
    return {
        "enabled": False,
        "target_currency": "USDT",
        "target_usdt": 0.0,
        "target_cny": 0.0,
        "cny_per_usdt_assumption": cny_per_usdt,
        "note": "每日目标已禁用，不参与交易判断。",
    }


@router.delete(
    "/decisions",
    dependencies=[Depends(require_destructive_dashboard_confirmation)],
)
async def clear_decisions():
    """Delete all AI decision records."""
    from db.repositories.decision_repo import DecisionRepository
    from db.session import get_session_ctx

    async with get_session_ctx() as session:
        repo = DecisionRepository(session)
        count = await repo.delete_all()
        await session.commit()

    _reset_trading_decision_runtime_state()

    return {"status": "ok", "message": f"Deleted {count} decision records", "deleted": count}


@router.get("/dashboard/account")
async def get_account_balance():
    """Account balances (virtual for paper, real for live)."""
    summaries = []
    if _trading_service and _trading_service.paper_executor:
        summaries = await _trading_service.paper_executor.get_all_summaries()

    live_balance = 0.0
    if _trading_service and _trading_service.okx_executor:
        try:
            live_balance = await _trading_service.okx_executor.get_balance()
        except Exception as exc:
            _log_dashboard_fallback("live balance fallback", exc)

    return {
        "virtual_accounts": summaries,
        "live_balance": live_balance,
    }


@router.get("/dashboard/pnl-history")
async def get_pnl_history(mode: str | None = None):
    """PnL equity curve history for each model."""
    selected_mode = "live" if mode == "live" else mode_manager.mode.value
    db_history = await _build_execution_pnl_history_from_db(selected_mode)
    if db_history:
        return {"history": await _format_pnl_history(db_history)}

    if _trading_service:
        history = _trading_service.get_pnl_history()
        if not history:
            try:
                await _trading_service.record_equity_snapshot()
                history = _trading_service.get_pnl_history()
            except Exception as exc:
                _log_dashboard_fallback(
                    "pnl history snapshot fallback",
                    exc,
                    mode=selected_mode,
                )
                history = {}

        if not history and _trading_service.paper_executor and _trading_service.models:
            now = datetime.now(UTC).isoformat()
            history = {}
            for model_name in [ENSEMBLE_TRADER_NAME]:
                try:
                    summary = await _trading_service.paper_executor.get_account_summary(model_name)
                    equity = summary.get("equity", summary.get("balance", 0))
                    history[model_name] = [{"time": now, "equity": round(equity, 2)}]
                except Exception as exc:
                    _log_dashboard_fallback(
                        "paper pnl summary fallback",
                        exc,
                        mode=selected_mode,
                        model_name=model_name,
                    )

        # Convert to frontend-friendly format: {model: {pnl_curve: [...], labels: [...]}}
        return {"history": await _format_pnl_history(history)}

    return {"history": await _format_pnl_history(db_history)}


@router.get("/dashboard/daily-pnl")
async def get_daily_pnl_records(mode: str | None = None, days: int = 30):
    """Daily execution PnL grouped by Beijing calendar day."""
    from sqlalchemy import select

    from db.session import get_session_ctx
    from models.trade import Position

    selected_mode = "live" if mode == "live" else "paper"
    days = min(max(int(days or 30), 1), 180)
    today_local = datetime.now(BEIJING_TZ).date()
    start_day = today_local - timedelta(days=days - 1)
    start_local = datetime.combine(start_day, datetime.min.time(), tzinfo=BEIJING_TZ)
    start_utc = start_local.astimezone(UTC)

    records: dict[str, dict[str, Any]] = {
        (start_day + timedelta(days=offset)).isoformat(): {
            "date": (start_day + timedelta(days=offset)).isoformat(),
            "timezone": "Asia/Shanghai",
            "realized_profit": 0.0,
            "realized_loss": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "total_pnl": 0.0,
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "symbols": [],
            "symbol_pnl": {},
            "cumulative_realized_pnl": 0.0,
            "cumulative_total_pnl": 0.0,
        }
        for offset in range(days)
    }
    cumulative_before = 0.0
    open_unrealized = 0.0

    try:
        exchange_marks = await _get_exchange_position_mark_map(selected_mode)
    except Exception as exc:
        _log_dashboard_fallback(
            "daily pnl exchange mark fallback",
            exc,
            mode=selected_mode,
        )
        exchange_marks = {}
    try:
        exchange_symbols = await _get_exchange_open_position_symbols(selected_mode)
    except Exception as exc:
        _log_dashboard_fallback(
            "daily pnl exchange open symbol fallback",
            exc,
            mode=selected_mode,
        )
        exchange_symbols = None

    async with get_session_ctx() as session:
        result = await session.execute(
            select(Position)
            .where(
                Position.model_name == ENSEMBLE_TRADER_NAME,
                Position.execution_mode == selected_mode,
            )
            .order_by(Position.closed_at.asc(), Position.created_at.asc())
        )
        positions = list(result.scalars().all())

    for pos in positions:
        pnl = float(pos.realized_pnl or 0.0)
        closed_at = _as_utc_datetime(pos.closed_at)
        if not pos.is_open and closed_at and closed_at < start_utc:
            cumulative_before += pnl
            continue
        if not pos.is_open and closed_at:
            day = closed_at.astimezone(BEIJING_TZ).date().isoformat()
            row = records.get(day)
            if not row:
                continue
            if pnl >= 0:
                row["realized_profit"] += pnl
                row["win_count"] += 1
            else:
                row["realized_loss"] += abs(pnl)
                row["loss_count"] += 1
            row["realized_pnl"] += pnl
            row["total_pnl"] += pnl
            row["trade_count"] += 1
            symbol = str(pos.symbol or "")
            if symbol and symbol not in row["symbols"]:
                row["symbols"].append(symbol)
            if symbol:
                symbol_row = row["symbol_pnl"].setdefault(
                    symbol,
                    {
                        "symbol": symbol,
                        "realized_profit": 0.0,
                        "realized_loss": 0.0,
                        "realized_pnl": 0.0,
                        "trade_count": 0,
                        "win_count": 0,
                        "loss_count": 0,
                    },
                )
                if pnl >= 0:
                    symbol_row["realized_profit"] += pnl
                    symbol_row["win_count"] += 1
                else:
                    symbol_row["realized_loss"] += abs(pnl)
                    symbol_row["loss_count"] += 1
                symbol_row["realized_pnl"] += pnl
                symbol_row["trade_count"] += 1
            continue

        if pos.is_open:
            normalized = _normalize_dashboard_symbol(pos.symbol)
            side = str(pos.side or "").lower()
            if exchange_symbols and normalized not in exchange_symbols:
                continue
            latest_unrealized = float(pos.unrealized_pnl or 0.0)
            snapshot = exchange_marks.get((normalized, side))
            if snapshot:
                mark_price = float(snapshot.get("mark_price") or 0.0)
                snapshot_upl = _safe_float(snapshot.get("upl"), None)
                if snapshot_upl is not None:
                    latest_unrealized = snapshot_upl
                else:
                    entry_price = _safe_float(snapshot.get("entry_price"), 0.0) or float(
                        pos.entry_price or 0.0
                    )
                    quantity = _safe_float(snapshot.get("contracts"), 0.0) or float(
                        pos.quantity or 0.0
                    )
                    if mark_price > 0 and entry_price > 0 and quantity > 0:
                        latest_unrealized = (
                            (entry_price - mark_price) * quantity
                            if side == "short"
                            else (mark_price - entry_price) * quantity
                        )
            open_unrealized += latest_unrealized

    cumulative = cumulative_before
    for date_key in sorted(records):
        row = records[date_key]
        cumulative += row["realized_pnl"]
        row["realized_profit"] = round(row["realized_profit"], 8)
        row["realized_loss"] = round(row["realized_loss"], 8)
        row["realized_pnl"] = round(row["realized_pnl"], 8)
        if date_key == today_local.isoformat():
            row["unrealized_pnl"] = round(open_unrealized, 8)
            row["total_pnl"] = round(row["realized_pnl"] + open_unrealized, 8)
            row["cumulative_total_pnl"] = round(cumulative + open_unrealized, 8)
        else:
            row["total_pnl"] = round(row["realized_pnl"], 8)
            row["cumulative_total_pnl"] = round(cumulative, 8)
        row["cumulative_realized_pnl"] = round(cumulative, 8)
        row["symbols"] = sorted(row["symbols"])
        row["symbol_pnl"] = [
            {
                **symbol_row,
                "realized_profit": round(symbol_row["realized_profit"], 8),
                "realized_loss": round(symbol_row["realized_loss"], 8),
                "realized_pnl": round(symbol_row["realized_pnl"], 8),
            }
            for symbol_row in sorted(
                row["symbol_pnl"].values(),
                key=lambda item: abs(item["realized_pnl"]),
                reverse=True,
            )
        ]

    return {
        "mode": selected_mode,
        "timezone": "Asia/Shanghai",
        "start_date": start_day.isoformat(),
        "end_date": today_local.isoformat(),
        "records": [records[key] for key in sorted(records.keys(), reverse=True)],
    }


async def _build_execution_pnl_history_from_db(mode: str) -> dict[str, list[dict]]:
    """Rebuild the execution account curve from closed/open position records.

    The in-memory snapshot list is lost on process restart. The dashboard still
    needs a useful curve, so we reconstruct one from realized PnL events and add
    the current unrealized PnL as the latest point.
    """
    from sqlalchemy import select

    from db.session import get_session_ctx
    from models.trade import Position

    selected_mode = "live" if mode == "live" else "paper"
    cfg = settings.get_execution_account_config(selected_mode)
    allocated = _safe_float(cfg.get("allocated_balance"), 0.0) or 0.0
    if _trading_service:
        try:
            snapshot = await _dashboard_okx_balance_snapshot_for_mode(selected_mode)
            allocated = (
                _safe_float(
                    (snapshot or {}).get("allocatable")
                    or (snapshot or {}).get("equity")
                    or (snapshot or {}).get("total"),
                    allocated,
                )
                or allocated
            )
        except Exception as exc:
            _log_dashboard_fallback(
                "pnl history allocation snapshot fallback",
                exc,
                mode=selected_mode,
            )
    if allocated <= 0:
        allocated = settings.get_initial_balance(ENSEMBLE_TRADER_NAME)

    try:
        async with get_session_ctx() as session:
            result = await session.execute(
                select(Position)
                .where(
                    Position.model_name == ENSEMBLE_TRADER_NAME,
                    Position.execution_mode == selected_mode,
                )
                .order_by(Position.created_at.asc())
            )
            positions = list(result.scalars().all())
    except Exception as exc:
        _log_dashboard_fallback(
            "pnl history database fallback",
            exc,
            mode=selected_mode,
        )
        return {}

    if not positions or allocated <= 0:
        return {}

    def normalize_dt(dt: datetime | None) -> datetime:
        if not dt:
            return datetime.now(UTC)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)

    def iso(dt: datetime | None) -> str:
        return normalize_dt(dt).isoformat()

    event_times = [
        normalize_dt(dt) for p in positions for dt in (p.created_at, p.closed_at) if dt is not None
    ]
    first_time = min(event_times) if event_times else datetime.now(UTC)
    snapshots: list[dict] = [{"time": iso(first_time), "equity": round(allocated, 8)}]

    cumulative_realized = 0.0
    closed_positions = sorted(
        [p for p in positions if not p.is_open and p.closed_at],
        key=lambda p: p.closed_at or datetime.max,
    )
    for pos in closed_positions:
        cumulative_realized += _safe_float(pos.realized_pnl, 0.0) or 0.0
        snapshots.append(
            {
                "time": iso(pos.closed_at),
                "equity": round(allocated + cumulative_realized, 8),
            }
        )

    open_unrealized = sum(_safe_float(p.unrealized_pnl, 0.0) or 0.0 for p in positions if p.is_open)
    current_equity = round(allocated + cumulative_realized + open_unrealized, 8)
    if len(snapshots) < 2 or abs(float(snapshots[-1]["equity"]) - current_equity) > 1e-8:
        snapshots.append(
            {
                "time": datetime.now(UTC).isoformat(),
                "equity": current_equity,
            }
        )

    return {ENSEMBLE_TRADER_NAME: snapshots[-500:]}


async def _format_pnl_history(history: dict[str, list[dict]]) -> dict:
    result = {}
    for model_name, snapshots in (history or {}).items():
        if not snapshots:
            continue
        initial = _safe_float(snapshots[0].get("equity"), 0.0) or 0.0
        if _trading_service and _trading_service.paper_executor:
            try:
                summary = await _trading_service.paper_executor.get_account_summary(model_name)
                initial = _safe_float(summary.get("initial_balance"), initial) or initial
            except Exception as exc:
                _log_dashboard_fallback(
                    "pnl history initial balance fallback",
                    exc,
                    model_name=model_name,
                )
        result[model_name] = {
            "pnl_curve": [
                (
                    round(
                        ((_safe_float(s.get("equity"), initial) or initial) - initial)
                        / initial
                        * 100,
                        6,
                    )
                    if initial > 0
                    else 0
                )
                for s in snapshots
            ],
            "labels": [s.get("time") for s in snapshots],
        }
    return result
