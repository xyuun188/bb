"""Profit attribution helpers for closed trades."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

from models.decision import AIDecision
from models.learning import ShadowBacktest
from models.trade import Order, Position
from services.decision_state import (
    DecisionStage,
    DecisionStageStatus,
    STAGE_LABELS,
    STATUS_LABELS,
    decision_state_from_raw,
    summarize_decision_stages,
)
from web_dashboard.api.text_sanitize import sanitize_text


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _side_from_action(action: str | None) -> str:
    value = str(action or "").lower()
    if "short" in value:
        return "short"
    if "long" in value:
        return "long"
    return ""


def _action_label(action: str | None) -> str:
    labels = {
        "long": "做多",
        "short": "做空",
        "close_long": "平多",
        "close_short": "平空",
        "hold": "观望",
    }
    return labels.get(str(action or "").lower(), str(action or "-"))


def _side_label(side: str | None) -> str:
    labels = {"long": "做多", "short": "做空"}
    return labels.get(str(side or "").lower(), str(side or "-"))


def _order_time(order: Order) -> datetime | None:
    return _aware(order.filled_at or order.created_at)


def _position_close_time(position: Position) -> datetime | None:
    return _aware(position.closed_at)


def _position_open_time(position: Position) -> datetime | None:
    return _aware(position.created_at)


def _raw(decision: AIDecision | None) -> dict[str, Any]:
    return decision.raw_llm_response if decision and isinstance(decision.raw_llm_response, dict) else {}


def _payload_side(payload: dict[str, Any] | None, side_key: str = "best_side") -> str:
    if not isinstance(payload, dict):
        return ""
    value = str(payload.get(side_key) or payload.get("side") or "").lower()
    if value in {"long", "short"}:
        return value
    direction = str(payload.get("direction") or payload.get("forecast_direction") or "").lower()
    if direction == "up":
        return "long"
    if direction == "down":
        return "short"
    label = str(payload.get("label") or payload.get("sentiment") or "").lower()
    score = _safe_float(payload.get("score", payload.get("sentiment_score", 0.0)), 0.0)
    if label in {"positive", "bullish"} or score > 0:
        return "long"
    if label in {"negative", "bearish"} or score < 0:
        return "short"
    return ""


def _first_tool_payload(raw: dict[str, Any], *keys: str) -> dict[str, Any]:
    """Return the first model payload across legacy and current tool containers."""
    containers = (
        raw.get("local_ai_tools"),
        raw.get("server_quant_tools"),
        raw.get("quant_tools"),
        raw.get("local_tools"),
        raw.get("server_tools"),
        raw,
    )
    for container in containers:
        if not isinstance(container, dict):
            continue
        for key in keys:
            value = container.get(key)
            if isinstance(value, dict):
                return value
    return {}


def _signal_available(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict) or not payload:
        return False
    for key in ("available", "enabled", "ok"):
        if key in payload and payload.get(key) is False:
            return False
    return True


def _expected_return_pct(payload: dict[str, Any], side: str = "") -> float:
    side = str(side or "").lower()
    keys: list[str] = []
    if side in {"long", "short"}:
        keys.extend([
            f"expected_{side}_return_pct",
            f"{side}_expected_return_pct",
            f"adjusted_{side}_return_pct",
            f"{side}_return_pct",
        ])
    keys.extend([
        "expected_return_pct",
        "best_expected_return_pct",
        "expected_move_pct",
        "forecast_return_pct",
        "return_pct",
        "expected_profit_pct",
    ])
    for key in keys:
        if key in payload:
            return _safe_float(payload.get(key), 0.0)
    return 0.0


def extract_signal_sides(raw: dict[str, Any]) -> dict[str, Any]:
    ml = raw.get("ml_signal") if isinstance(raw.get("ml_signal"), dict) else {}
    predictions = ml.get("predictions") if isinstance(ml.get("predictions"), list) else []
    primary_ml = predictions[0] if predictions and isinstance(predictions[0], dict) else ml

    profit = _first_tool_payload(
        raw,
        "profit_prediction",
        "profit_model",
        "server_profit",
        "server_profit_model",
        "profit",
    )
    timeseries = _first_tool_payload(
        raw,
        "time_series_prediction",
        "timeseries_prediction",
        "sequence_prediction",
        "timeseries",
        "time_series",
    )
    sentiment = _first_tool_payload(
        raw,
        "sentiment_analysis",
        "sentiment_prediction",
        "sentiment_model",
        "sentiment",
    )
    profit_side = _payload_side(profit)
    timeseries_side = _payload_side(timeseries)
    sentiment_side = _payload_side(sentiment)

    return {
        "ml": {
            "available": bool(ml),
            "side": _payload_side(primary_ml),
            "expected_return_pct": _safe_float(
                primary_ml.get("best_expected_return_pct", ml.get("expected_return_pct", 0.0)),
                0.0,
            ),
        },
        "server_profit": {
            "available": _signal_available(profit),
            "side": profit_side,
            "expected_return_pct": _expected_return_pct(profit, profit_side),
        },
        "timeseries": {
            "available": _signal_available(timeseries),
            "side": timeseries_side,
            "expected_return_pct": _expected_return_pct(timeseries, timeseries_side),
        },
        "sentiment": {
            "available": _signal_available(sentiment),
            "side": sentiment_side,
            "expected_return_pct": _expected_return_pct(sentiment, sentiment_side),
            "score": _safe_float(sentiment.get("score", sentiment.get("sentiment_score", 0.0)), 0.0),
        },
    }


@dataclass(slots=True)
class MatchedDecision:
    decision: AIDecision | None
    order: Order | None
    confidence: str


def _match_order(
    position: Position,
    orders: Iterable[Order],
    decisions_by_id: dict[int, AIDecision],
    *,
    want_exit: bool,
) -> MatchedDecision:
    target_time = _position_close_time(position) if want_exit else _position_open_time(position)
    max_gap_seconds = 45 * 60 if want_exit else 30 * 60
    candidates: list[tuple[float, Order, AIDecision | None]] = []
    side = str(position.side or "").lower()
    for order in orders:
        if order.symbol != position.symbol:
            continue
        decision = decisions_by_id.get(int(order.decision_id or 0))
        action = str(decision.action if decision else "")
        if want_exit:
            expected_action = "close_long" if side == "long" else "close_short"
            if decision and action != expected_action:
                continue
        else:
            if decision and _side_from_action(action) != side:
                continue
        order_time = _order_time(order)
        if not target_time or not order_time:
            gap = 0.0
        else:
            gap = abs((order_time - target_time).total_seconds())
            if gap > max_gap_seconds:
                continue
        candidates.append((gap, order, decision))
    if not candidates:
        return MatchedDecision(None, None, "low")
    gap, order, decision = sorted(candidates, key=lambda item: item[0])[0]
    confidence = "high" if decision and gap <= 180 else "medium" if decision else "low"
    return MatchedDecision(decision, order, confidence)


def _match_nearest_decision(
    position: Position,
    decisions: Iterable[AIDecision],
    *,
    want_exit: bool,
) -> MatchedDecision:
    target_time = _position_close_time(position) if want_exit else _position_open_time(position)
    side = str(position.side or "").lower()
    max_gap_seconds = 45 * 60 if want_exit else 30 * 60
    candidates: list[tuple[float, AIDecision]] = []
    for decision in decisions:
        if decision.symbol != position.symbol:
            continue
        action = str(decision.action or "").lower()
        if want_exit:
            expected = "close_long" if side == "long" else "close_short"
            if action != expected:
                continue
        else:
            if _side_from_action(action) != side:
                continue
        decision_time = _aware(decision.executed_at or decision.created_at)
        if not target_time or not decision_time:
            gap = 0.0
        else:
            gap = abs((decision_time - target_time).total_seconds())
            if gap > max_gap_seconds:
                continue
        candidates.append((gap, decision))
    if not candidates:
        return MatchedDecision(None, None, "low")
    gap, decision = sorted(candidates, key=lambda item: item[0])[0]
    return MatchedDecision(decision, None, "medium" if gap <= 300 else "low")


def _best_shadow(
    decision: AIDecision | None,
    shadows: Iterable[ShadowBacktest],
) -> ShadowBacktest | None:
    if not decision:
        return None
    matches = [
        row
        for row in shadows
        if row.decision_id == decision.id
        or (
            row.symbol == decision.symbol
            and _aware(row.created_at)
            and _aware(decision.created_at)
            and abs((_aware(row.created_at) - _aware(decision.created_at)).total_seconds()) <= 600
        )
    ]
    if not matches:
        return None
    completed = [row for row in matches if row.status == "completed"]
    rows = completed or matches
    return sorted(rows, key=lambda row: (row.horizon_minutes or 999, row.id), reverse=True)[0]


def _shadow_best_action(row: ShadowBacktest | None) -> str:
    if not row:
        return ""
    value = str(row.best_action or "").lower()
    if value in {"long", "short", "hold"}:
        return value
    long_ret = _safe_float(row.long_return_pct, 0.0)
    short_ret = _safe_float(row.short_return_pct, 0.0)
    if max(long_ret, short_ret) <= 0:
        return "hold"
    return "long" if long_ret >= short_ret else "short"


def _classify_record(
    position: Position,
    entry_decision: AIDecision | None,
    close_decision: AIDecision | None,
    shadow: ShadowBacktest | None,
) -> tuple[str, str, str, list[str]]:
    pnl = _safe_float(position.realized_pnl, 0.0)
    side = str(position.side or "").lower()
    raw_entry = _raw(entry_decision)
    raw_close = _raw(close_decision)
    signals = extract_signal_sides(raw_entry)
    shadow_action = _shadow_best_action(shadow)
    hold_minutes = 0.0
    opened = _position_open_time(position)
    closed = _position_close_time(position)
    if opened and closed:
        hold_minutes = max((closed - opened).total_seconds() / 60.0, 0.0)

    notes: list[str] = []
    if shadow_action in {"long", "short"} and shadow_action != side:
        notes.append(f"影子复盘更优方向是{_side_label(shadow_action)}")
    for key, label in (
        ("ml", "本地ML"),
        ("server_profit", "服务器盈利模型"),
        ("timeseries", "时序预测"),
        ("sentiment", "情绪模型"),
    ):
        signal_side = signals.get(key, {}).get("side")
        if signal_side in {"long", "short"} and signal_side != side:
            notes.append(f"{label}当时偏向{_side_label(signal_side)}")

    close_evidence = raw_close.get("close_evidence") if isinstance(raw_close.get("close_evidence"), dict) else {}
    close_reason = str(
        raw_close.get("execution_reason")
        or close_evidence.get("reason")
        or getattr(close_decision, "execution_reason", "")
        or getattr(close_decision, "reasoning", "")
        or ""
    )
    close_reason = str(sanitize_text(close_reason) or "")
    if close_reason:
        lowered = close_reason.lower()
        if "止损" in close_reason or "stop" in lowered:
            notes.append("平仓来自止损或快速风控")
        if "锁盈" in close_reason or "止盈" in close_reason or "profit" in lowered:
            notes.append("平仓来自锁盈/止盈")

    if pnl < 0:
        if shadow_action in {"long", "short"} and shadow_action != side:
            return "ai_direction_error", "AI方向判断偏差", "high", notes
        if notes:
            return "model_conflict_ignored", "模型分歧未充分消化", "medium", notes
        if hold_minutes <= 5:
            return "entry_quality", "入场后快速不利", "medium", notes
        if "止损" in close_reason or "stop" in close_reason.lower():
            return "stop_loss_or_fast_risk", "止损/快速风控亏损", "medium", notes
        return "loss_unclassified", "亏损原因待复盘", "low", notes

    if pnl > 0:
        if hold_minutes <= 5 and pnl < 1.0:
            return "early_small_profit", "小盈快跑", "medium", notes
        if close_reason and ("锁盈" in close_reason or "止盈" in close_reason):
            return "profit_locked", "利润保护生效", "medium", notes
        return "profitable_exit", "盈利兑现", "medium", notes

    return "flat_or_fee_churn", "盈亏接近 0", "medium", notes


def build_profit_attribution(
    positions: list[Position],
    orders: list[Order],
    decisions: list[AIDecision],
    shadows: list[ShadowBacktest],
) -> dict[str, Any]:
    decisions_by_id = {int(d.id): d for d in decisions if d.id is not None}
    records: list[dict[str, Any]] = []
    buckets: dict[str, dict[str, Any]] = {}

    total_pnl = 0.0
    wins = 0
    losses = 0
    profit = 0.0
    loss = 0.0

    for position in positions:
        pnl = _safe_float(position.realized_pnl, 0.0)
        total_pnl += pnl
        if pnl > 0:
            wins += 1
            profit += pnl
        elif pnl < 0:
            losses += 1
            loss += abs(pnl)

        entry_match = _match_order(position, orders, decisions_by_id, want_exit=False)
        if not entry_match.decision:
            entry_match = _match_nearest_decision(position, decisions, want_exit=False)
        close_match = _match_order(position, orders, decisions_by_id, want_exit=True)
        if not close_match.decision:
            close_match = _match_nearest_decision(position, decisions, want_exit=True)

        shadow = _best_shadow(entry_match.decision, shadows)
        bucket_key, bucket_label, confidence, notes = _classify_record(
            position,
            entry_match.decision,
            close_match.decision,
            shadow,
        )
        bucket = buckets.setdefault(bucket_key, {
            "key": bucket_key,
            "label": bucket_label,
            "count": 0,
            "pnl": 0.0,
            "profit": 0.0,
            "loss": 0.0,
        })
        bucket["count"] += 1
        bucket["pnl"] += pnl
        if pnl >= 0:
            bucket["profit"] += pnl
        else:
            bucket["loss"] += abs(pnl)

        raw_entry = _raw(entry_match.decision)
        raw_close = _raw(close_match.decision)
        signals = extract_signal_sides(raw_entry)
        opened = _position_open_time(position)
        closed = _position_close_time(position)
        hold_minutes = (
            max((closed - opened).total_seconds() / 60.0, 0.0)
            if opened and closed
            else 0.0
        )

        records.append({
            "position_id": position.id,
            "symbol": position.symbol,
            "side": position.side,
            "side_label": _side_label(position.side),
            "entry_at": opened.isoformat() if opened else None,
            "closed_at": closed.isoformat() if closed else None,
            "hold_minutes": round(hold_minutes, 2),
            "entry_price": _safe_float(position.entry_price, 0.0),
            "exit_price": _safe_float(position.current_price, 0.0),
            "quantity": _safe_float(position.quantity, 0.0),
            "realized_pnl": round(pnl, 6),
            "bucket": bucket_key,
            "main_reason": bucket_label,
            "attribution_confidence": confidence,
            "notes": notes,
            "entry_decision": _decision_payload(entry_match.decision, entry_match.confidence),
            "close_decision": _decision_payload(close_match.decision, close_match.confidence),
            "signals": signals,
            "shadow": _shadow_payload(shadow),
            "decision_state": _decision_state_or_closed_fallback(raw_entry, position, entry_match.decision),
            "close_state": _decision_state_or_closed_fallback(raw_close, position, close_match.decision),
        })

    trade_count = len(positions)
    avg_win = profit / wins if wins else 0.0
    avg_loss = loss / losses if losses else 0.0
    summary = {
        "trade_count": trade_count,
        "total_closed_pnl": round(total_pnl, 6),
        "win_count": wins,
        "loss_count": losses,
        "win_rate": round(wins / trade_count, 4) if trade_count else 0.0,
        "avg_win": round(avg_win, 6),
        "avg_loss": round(avg_loss, 6),
        "profit_factor": round(profit / loss, 4) if loss > 0 else (999.0 if profit > 0 else 0.0),
        "small_win_count": sum(1 for row in records if 0 < row["realized_pnl"] < 1.0),
        "large_loss_count": sum(1 for row in records if row["realized_pnl"] <= -5.0),
        "early_exit_count": sum(1 for row in records if row["bucket"] == "early_small_profit"),
        "direction_error_count": sum(1 for row in records if row["bucket"] == "ai_direction_error"),
        "execution_issue_count": sum(1 for row in records if row["bucket"] in {"stop_loss_or_fast_risk", "flat_or_fee_churn"}),
    }

    bucket_rows = []
    for item in buckets.values():
        count = max(int(item["count"]), 1)
        bucket_rows.append({
            **item,
            "pnl": round(float(item["pnl"]), 6),
            "profit": round(float(item["profit"]), 6),
            "loss": round(float(item["loss"]), 6),
            "avg_pnl": round(float(item["pnl"]) / count, 6),
        })
    bucket_rows.sort(key=lambda item: abs(float(item["pnl"])), reverse=True)

    return {
        "summary": summary,
        "buckets": bucket_rows,
        "records": records,
    }


def _decision_payload(decision: AIDecision | None, confidence: str) -> dict[str, Any] | None:
    if decision is None:
        return None
    return {
        "id": decision.id,
        "action": decision.action,
        "action_label": _action_label(decision.action),
        "confidence": _safe_float(decision.confidence, 0.0),
        "reasoning": sanitize_text(decision.reasoning or ""),
        "execution_reason": sanitize_text(decision.execution_reason or ""),
        "was_executed": bool(decision.was_executed),
        "matched_confidence": confidence,
        "created_at": _aware(decision.created_at).isoformat() if _aware(decision.created_at) else None,
    }


def _shadow_payload(row: ShadowBacktest | None) -> dict[str, Any] | None:
    if row is None:
        return None
    best = _shadow_best_action(row)
    return {
        "id": row.id,
        "decision_id": row.decision_id,
        "horizon_minutes": row.horizon_minutes,
        "status": row.status,
        "best_action": best,
        "best_action_label": _action_label(best),
        "long_return_pct": _safe_float(row.long_return_pct, 0.0),
        "short_return_pct": _safe_float(row.short_return_pct, 0.0),
        "missed_opportunity": bool(row.missed_opportunity),
        "note": sanitize_text(row.note or ""),
    }


def _decision_state_or_closed_fallback(
    raw_response: dict[str, Any],
    position: Position,
    decision: AIDecision | None,
) -> dict[str, Any]:
    machine = decision_state_from_raw(raw_response)
    summary = machine.get("summary") if isinstance(machine.get("summary"), dict) else {}
    if summary.get("final_stage"):
        return machine
    if position.is_open or not position.closed_at:
        return machine

    at = (_position_close_time(position) or datetime.now(timezone.utc)).isoformat()
    reason = (
        "历史记录已完成平仓；旧版本未写入逐阶段状态机，"
        "系统根据已平仓持仓和本地订单记录推断为本地同步完成。"
    )
    stages = []
    if decision is not None:
        stages.append({
            "stage": DecisionStage.AI_ANALYSIS,
            "stage_label": STAGE_LABELS[DecisionStage.AI_ANALYSIS],
            "status": DecisionStageStatus.COMPLETED,
            "status_label": STATUS_LABELS[DecisionStageStatus.COMPLETED],
            "reason": "历史 AI 决策记录存在，但旧版本未保存完整状态机。",
            "at": (_aware(decision.created_at) or _position_open_time(position) or _position_close_time(position)).isoformat(),
            "inferred": True,
        })
    stages.append({
        "stage": DecisionStage.LOCAL_SYNC,
        "stage_label": STAGE_LABELS[DecisionStage.LOCAL_SYNC],
        "status": DecisionStageStatus.COMPLETED,
        "status_label": STATUS_LABELS[DecisionStageStatus.COMPLETED],
        "reason": reason,
        "at": at,
        "inferred": True,
    })
    return {
        "stages": stages,
        "summary": {
            **summarize_decision_stages(stages),
            "inferred": True,
            "final_reason": reason,
        },
        "inferred": True,
    }


# Clean UTF-8 overrides.  Earlier helper definitions are kept above only to
# avoid a high-risk full-file rewrite while this module is being refactored.
def _action_label(action: str | None) -> str:
    labels = {
        "long": "做多",
        "short": "做空",
        "close_long": "平多",
        "close_short": "平空",
        "hold": "观望",
    }
    return labels.get(str(action or "").lower(), str(action or "-"))


def _side_label(side: str | None) -> str:
    labels = {"long": "做多", "short": "做空"}
    return labels.get(str(side or "").lower(), str(side or "-"))


def _classify_record(
    position: Position,
    entry_decision: AIDecision | None,
    close_decision: AIDecision | None,
    shadow: ShadowBacktest | None,
) -> tuple[str, str, str, list[str]]:
    pnl = _safe_float(position.realized_pnl, 0.0)
    side = str(position.side or "").lower()
    raw_entry = _raw(entry_decision)
    raw_close = _raw(close_decision)
    signals = extract_signal_sides(raw_entry)
    shadow_action = _shadow_best_action(shadow)
    hold_minutes = 0.0
    opened = _position_open_time(position)
    closed = _position_close_time(position)
    if opened and closed:
        hold_minutes = max((closed - opened).total_seconds() / 60.0, 0.0)

    notes: list[str] = []
    if shadow_action in {"long", "short"} and shadow_action != side:
        notes.append(f"影子复盘显示更优方向是{_side_label(shadow_action)}")
    for key, label in (
        ("ml", "本地ML"),
        ("server_profit", "服务器盈利模型"),
        ("timeseries", "时序预测"),
        ("sentiment", "情绪模型"),
    ):
        signal_side = signals.get(key, {}).get("side")
        if signal_side in {"long", "short"} and signal_side != side:
            notes.append(f"{label}当时偏向{_side_label(signal_side)}")

    close_evidence = raw_close.get("close_evidence") if isinstance(raw_close.get("close_evidence"), dict) else {}
    close_reason = str(
        raw_close.get("execution_reason")
        or close_evidence.get("reason")
        or getattr(close_decision, "execution_reason", "")
        or getattr(close_decision, "reasoning", "")
        or ""
    )
    close_reason = str(sanitize_text(close_reason) or "")
    lowered_close_reason = close_reason.lower()
    if close_reason:
        if "止损" in close_reason or "stop" in lowered_close_reason:
            notes.append("平仓来自止损或快速风控")
        if "锁盈" in close_reason or "止盈" in close_reason or "profit" in lowered_close_reason:
            notes.append("平仓来自锁盈/止盈")

    if pnl < 0:
        if shadow_action in {"long", "short"} and shadow_action != side:
            return "ai_direction_error", "AI方向判断偏差", "high", notes
        if notes:
            return "model_conflict_ignored", "模型分歧未充分消化", "medium", notes
        if hold_minutes <= 5:
            return "entry_quality", "入场后快速不利", "medium", notes
        if "止损" in close_reason or "stop" in lowered_close_reason:
            return "stop_loss_or_fast_risk", "止损/快速风控亏损", "medium", notes
        return "loss_unclassified", "亏损原因待复盘", "low", notes

    if pnl > 0:
        if hold_minutes <= 5 and pnl < 1.0:
            return "early_small_profit", "小盈快跑", "medium", notes
        if close_reason and ("锁盈" in close_reason or "止盈" in close_reason):
            return "profit_locked", "利润保护生效", "medium", notes
        return "profitable_exit", "盈利兑现", "medium", notes

    return "flat_or_fee_churn", "盈亏接近 0", "medium", notes


def _decision_state_or_closed_fallback(
    raw_response: dict[str, Any],
    position: Position,
    decision: AIDecision | None,
) -> dict[str, Any]:
    machine = decision_state_from_raw(raw_response)
    summary = machine.get("summary") if isinstance(machine.get("summary"), dict) else {}
    if summary.get("final_stage"):
        return machine
    if position.is_open or not position.closed_at:
        return machine

    at = (_position_close_time(position) or datetime.now(timezone.utc)).isoformat()
    reason = (
        "历史记录已完成平仓；旧版本未写入完整逐阶段状态机，"
        "系统根据已平仓持仓和本地订单记录推断为本地同步完成。"
    )
    stages = []
    if decision is not None:
        stages.append({
            "stage": DecisionStage.AI_ANALYSIS,
            "stage_label": STAGE_LABELS[DecisionStage.AI_ANALYSIS],
            "status": DecisionStageStatus.COMPLETED,
            "status_label": STATUS_LABELS[DecisionStageStatus.COMPLETED],
            "reason": "历史 AI 决策记录存在，但旧版本未保存完整状态机。",
            "at": (_aware(decision.created_at) or _position_open_time(position) or _position_close_time(position)).isoformat(),
            "inferred": True,
        })
    stages.append({
        "stage": DecisionStage.LOCAL_SYNC,
        "stage_label": STAGE_LABELS[DecisionStage.LOCAL_SYNC],
        "status": DecisionStageStatus.COMPLETED,
        "status_label": STATUS_LABELS[DecisionStageStatus.COMPLETED],
        "reason": reason,
        "at": at,
        "inferred": True,
    })
    return {
        "stages": stages,
        "summary": {
            **summarize_decision_stages(stages),
            "inferred": True,
            "final_reason": reason,
        },
        "inferred": True,
    }
