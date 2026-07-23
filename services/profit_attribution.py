"""Profit attribution helpers for closed trades."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from models.decision import AIDecision
from models.learning import ShadowBacktest
from models.trade import Order, Position
from services.decision_state import (
    STAGE_LABELS,
    STATUS_LABELS,
    DecisionStage,
    DecisionStageStatus,
    decision_state_from_raw,
    summarize_decision_stages,
)
from services.entry_signal_extraction import extract_entry_signal_sides
from services.trade_fact_trust import filter_trusted_closed_positions
from web_dashboard.api.text_sanitize import sanitize_text

ACTION_LABELS = {
    "long": "做多",
    "short": "做空",
    "close_long": "平多",
    "close_short": "平空",
    "hold": "观望",
}
SIDE_LABELS = {"long": "做多", "short": "做空"}
SIGNAL_LABELS = (
    ("ml", "本地 ML"),
    ("server_profit", "服务器盈利模型"),
    ("timeseries", "时序预测"),
    ("sentiment", "情绪模型"),
)


@dataclass(slots=True)
class MatchedDecision:
    decision: AIDecision | None
    order: Order | None
    confidence: str


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _symbol_key(symbol: str | None) -> str:
    """Normalize historical OKX/CCXT symbol variants for attribution matching."""
    value = str(symbol or "").strip().upper().split(":")[0].replace("_", "-")
    if not value:
        return ""
    if value.endswith("-SWAP"):
        value = value.removesuffix("-SWAP")
    if "/" not in value and "-" in value:
        parts = [part for part in value.split("-") if part]
        if len(parts) >= 2:
            value = f"{parts[0]}/{parts[1]}"
    return value


def _action_label(action: str | None) -> str:
    return ACTION_LABELS.get(str(action or "").lower(), str(action or "-"))


def _side_label(side: str | None) -> str:
    return SIDE_LABELS.get(str(side or "").lower(), str(side or "-"))


def _side_from_action(action: str | None) -> str:
    value = str(action or "").lower()
    if "short" in value:
        return "short"
    if "long" in value:
        return "long"
    return ""


def _order_time(order: Order) -> datetime | None:
    return _aware(order.filled_at or order.created_at)


def _position_open_time(position: Position) -> datetime | None:
    return _aware(position.created_at)


def _position_close_time(position: Position) -> datetime | None:
    return _aware(position.closed_at)


def _raw(decision: AIDecision | None) -> dict[str, Any]:
    if decision and isinstance(decision.raw_llm_response, dict):
        return decision.raw_llm_response
    return {}


def extract_signal_sides(raw: dict[str, Any]) -> dict[str, Any]:
    return extract_entry_signal_sides(raw)


def _match_order(
    position: Position,
    orders: Iterable[Order],
    decisions_by_id: dict[int, AIDecision],
    *,
    want_exit: bool,
) -> MatchedDecision:
    target_time = _position_close_time(position) if want_exit else _position_open_time(position)
    max_gap_seconds = 45 * 60 if want_exit else 30 * 60
    side = str(position.side or "").lower()
    position_symbol = _symbol_key(position.symbol)
    candidates: list[tuple[float, Order, AIDecision | None]] = []

    for order in orders:
        if _symbol_key(order.symbol) != position_symbol:
            continue
        decision = decisions_by_id.get(int(order.decision_id or 0))
        action = str(decision.action if decision else "").lower()
        if want_exit:
            expected_action = "close_long" if side == "long" else "close_short"
            if decision and action != expected_action:
                continue
        elif decision and _side_from_action(action) != side:
            continue

        order_time = _order_time(order)
        if target_time and order_time:
            gap = abs((order_time - target_time).total_seconds())
            if gap > max_gap_seconds:
                continue
        else:
            gap = 0.0
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
    max_gap_seconds = 45 * 60 if want_exit else 30 * 60
    side = str(position.side or "").lower()
    position_symbol = _symbol_key(position.symbol)
    candidates: list[tuple[float, AIDecision]] = []

    for decision in decisions:
        if _symbol_key(decision.symbol) != position_symbol:
            continue
        action = str(decision.action or "").lower()
        if want_exit:
            expected = "close_long" if side == "long" else "close_short"
            if action != expected:
                continue
        elif _side_from_action(action) != side:
            continue

        decision_time = _aware(decision.executed_at or decision.created_at)
        if target_time and decision_time:
            gap = abs((decision_time - target_time).total_seconds())
            if gap > max_gap_seconds:
                continue
        else:
            gap = 0.0
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
    decision_time = _aware(decision.created_at)
    decision_symbol = _symbol_key(decision.symbol)
    matches: list[ShadowBacktest] = []
    for row in shadows:
        if row.decision_id == decision.id:
            matches.append(row)
            continue
        row_time = _aware(row.created_at)
        if (
            _symbol_key(row.symbol) == decision_symbol
            and decision_time is not None
            and row_time is not None
            and abs((row_time - decision_time).total_seconds()) <= 1800
        ):
            matches.append(row)
    if not matches:
        return None
    completed = [row for row in matches if row.status == "completed"]
    rows = completed or matches
    return sorted(rows, key=lambda row: (row.horizon_minutes or 0, row.id or 0), reverse=True)[0]


def _best_shadow_for_position(
    position: Position,
    decision: AIDecision | None,
    shadows: Iterable[ShadowBacktest],
) -> ShadowBacktest | None:
    matched = _best_shadow(decision, shadows)
    if matched is not None:
        return matched

    opened = _position_open_time(position)
    if not opened:
        return None

    side = str(position.side or "").lower()
    execution_mode = str(position.execution_mode or "").lower()
    position_symbol = _symbol_key(position.symbol)
    candidates: list[tuple[float, ShadowBacktest]] = []
    for row in shadows:
        if _symbol_key(row.symbol) != position_symbol:
            continue
        if execution_mode and str(row.execution_mode or "").lower() != execution_mode:
            continue
        row_side = _side_from_action(row.decision_action)
        if row_side in {"long", "short"} and side in {"long", "short"} and row_side != side:
            continue
        row_time = _aware(row.created_at)
        if not row_time:
            continue
        gap = abs((row_time - opened).total_seconds())
        if gap <= 45 * 60:
            candidates.append((gap, row))

    if not candidates:
        return None
    completed = [(gap, row) for gap, row in candidates if row.status == "completed"]
    rows = completed or candidates
    return sorted(
        rows,
        key=lambda item: (item[0], -(item[1].horizon_minutes or 0), -(item[1].id or 0)),
    )[0][1]


def _shadow_best_action(row: ShadowBacktest | None) -> str:
    if row is None:
        return ""
    value = str(row.best_action or "").lower()
    if value in {"long", "short", "hold"}:
        return value
    long_return = _safe_float(row.long_return_pct, 0.0)
    short_return = _safe_float(row.short_return_pct, 0.0)
    if max(long_return, short_return) <= 0:
        return "hold"
    return "long" if long_return >= short_return else "short"


def _decision_payload(decision: AIDecision | None, confidence: str) -> dict[str, Any] | None:
    if decision is None:
        return None
    raw = _raw(decision)
    opportunity = _safe_dict(raw.get("opportunity_score"))
    evidence_score = _safe_dict(opportunity.get("evidence_score"))
    created_at = _aware(decision.created_at)
    return {
        "id": decision.id,
        "action": decision.action,
        "action_label": _action_label(decision.action),
        "confidence": _safe_float(decision.confidence, 0.0),
        "reasoning": sanitize_text(decision.reasoning or ""),
        "execution_reason": sanitize_text(decision.execution_reason or ""),
        "was_executed": bool(decision.was_executed),
        "matched_confidence": confidence,
        "created_at": created_at.isoformat() if created_at else None,
        "opportunity_score": opportunity,
        "evidence_score": evidence_score,
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


def _signal_status(
    signals: dict[str, Any],
    key: str,
    label: str,
    *,
    entry_decision: AIDecision | None,
) -> dict[str, Any]:
    signal = _safe_dict(signals.get(key))
    has_side = str(signal.get("side") or "").lower() in {"long", "short"}
    has_expected = "expected_return_pct" in signal
    has_score = key == "sentiment" and "score" in signal
    available = bool(signal.get("available")) and (has_side or has_expected or has_score)
    if entry_decision is None:
        missing_reason = f"未匹配到开仓 AI 决策，无法读取{label}证据"
    else:
        missing_reason = f"开仓 AI 决策未保存{label} 证据"
    return {
        "available": available,
        "label": label,
        "side": signal.get("side") or "",
        "expected_return_pct": _safe_float(signal.get("expected_return_pct"), 0.0),
        "production_eligible": signal.get("production_eligible"),
        "status": "matched" if available else "missing",
        "missing_reason": "" if available else missing_reason,
    }


def _evidence_status(
    entry_decision: AIDecision | None,
    signals: dict[str, Any],
    shadow: ShadowBacktest | None,
) -> dict[str, Any]:
    ai_available = entry_decision is not None
    shadow_available = shadow is not None
    shadow_action = _shadow_best_action(shadow)
    return {
        "ai": {
            "available": ai_available,
            "label": "AI 决策",
            "action": entry_decision.action if entry_decision is not None else "",
            "action_label": _action_label(entry_decision.action if entry_decision else None),
            "confidence": _safe_float(getattr(entry_decision, "confidence", None), 0.0),
            "status": "matched" if ai_available else "missing",
            "missing_reason": (
                ""
                if ai_available
                else "未匹配到开仓 AI 决策；已检查订单 decision_id、币种和开仓时间窗"
            ),
        },
        "ml": _signal_status(signals, "ml", "本地 ML", entry_decision=entry_decision),
        "server_profit": _signal_status(
            signals,
            "server_profit",
            "服务器盈利模型",
            entry_decision=entry_decision,
        ),
        "timeseries": _signal_status(
            signals,
            "timeseries",
            "时序预测",
            entry_decision=entry_decision,
        ),
        "sentiment": _signal_status(
            signals,
            "sentiment",
            "情绪模型",
            entry_decision=entry_decision,
        ),
        "shadow": {
            "available": shadow_available,
            "label": "影子复盘",
            "status": shadow.status if shadow is not None else "missing",
            "best_action": shadow_action,
            "best_action_label": _action_label(shadow_action),
            "horizon_minutes": shadow.horizon_minutes if shadow is not None else None,
            "missing_reason": (
                ""
                if shadow_available
                else "未匹配到影子复盘样本；已检查 decision_id、币种和开仓时间窗"
            ),
        },
    }


def _close_reason(close_decision: AIDecision | None) -> str:
    raw_close = _raw(close_decision)
    close_evidence = _safe_dict(raw_close.get("close_evidence"))
    reason = str(
        raw_close.get("execution_reason")
        or close_evidence.get("reason")
        or getattr(close_decision, "execution_reason", "")
        or getattr(close_decision, "reasoning", "")
        or ""
    )
    return str(sanitize_text(reason) or "")


def _classify_record(
    position: Position,
    entry_decision: AIDecision | None,
    close_decision: AIDecision | None,
    shadow: ShadowBacktest | None,
) -> tuple[str, str, str, list[str]]:
    pnl = _safe_float(position.realized_pnl, 0.0)
    side = str(position.side or "").lower()
    signals = extract_signal_sides(_raw(entry_decision))
    shadow_action = _shadow_best_action(shadow)
    notes: list[str] = []
    if shadow_action in {"long", "short"} and shadow_action != side:
        notes.append(f"影子复盘显示更优方向是{_side_label(shadow_action)}")
    for key, label in SIGNAL_LABELS:
        signal_side = _safe_dict(signals.get(key)).get("side")
        if signal_side in {"long", "short"} and signal_side != side:
            notes.append(f"{label} 当时偏向{_side_label(signal_side)}")

    close_raw = _raw(close_decision)
    dynamic_exit = _safe_dict(close_raw.get("dynamic_exit_policy"))
    if not dynamic_exit:
        dynamic_exit = _safe_dict(
            _safe_dict(close_raw.get("close_evidence")).get("dynamic_exit_policy")
        )
    governed_exit = bool(dynamic_exit.get("eligible") is True)
    planned_stop_crossed = bool(dynamic_exit.get("planned_stop_crossed"))
    profit_retrace = _safe_float(dynamic_exit.get("profit_retrace_ratio"), 0.0)
    if governed_exit and planned_stop_crossed:
        notes.append("governed planned stop crossed")
    elif governed_exit and profit_retrace > 0.0:
        notes.append("governed fee-after profit retrace")

    if pnl < 0:
        if shadow_action in {"long", "short"} and shadow_action != side:
            return "ai_direction_error", "AI 方向判断偏差", "high", notes
        if notes:
            return "model_conflict_ignored", "模型分歧未充分消化", "medium", notes
        if governed_exit and planned_stop_crossed:
            return "stop_loss_or_fast_risk", "止损/快速风控亏损", "medium", notes
        return "loss_unclassified", "亏损原因待复盘", "low", notes

    if pnl > 0:
        if governed_exit and profit_retrace > 0.0:
            return "profit_locked", "利润保护生效", "medium", notes
        return "profitable_exit", "盈利兑现", "medium", notes

    return "flat_or_fee_churn", "盈亏接近 0", "medium", notes


def _decision_state_or_closed_fallback(
    raw_response: dict[str, Any],
    position: Position,
    decision: AIDecision | None,
) -> dict[str, Any]:
    machine = decision_state_from_raw(raw_response)
    summary = _safe_dict(machine.get("summary"))
    if summary.get("final_stage") or position.is_open or not position.closed_at:
        return machine

    at = (_position_close_time(position) or datetime.now(UTC)).isoformat()
    reason = (
        "历史记录已完成平仓；旧版本未写入完整状态机，"
        "系统根据已平仓持仓和本地订单记录推断为本地同步完成。"
    )
    stages: list[dict[str, Any]] = []
    if decision is not None:
        decision_at = (
            _aware(decision.created_at)
            or _position_open_time(position)
            or _position_close_time(position)
        )
        stages.append(
            {
                "stage": DecisionStage.AI_ANALYSIS,
                "stage_label": STAGE_LABELS[DecisionStage.AI_ANALYSIS],
                "status": DecisionStageStatus.COMPLETED,
                "status_label": STATUS_LABELS[DecisionStageStatus.COMPLETED],
                "reason": "历史 AI 决策记录存在，但旧版本未保存完整状态机。",
                "at": decision_at.isoformat() if decision_at else at,
                "inferred": True,
            }
        )
    stages.append(
        {
            "stage": DecisionStage.LOCAL_SYNC,
            "stage_label": STAGE_LABELS[DecisionStage.LOCAL_SYNC],
            "status": DecisionStageStatus.COMPLETED,
            "status_label": STATUS_LABELS[DecisionStageStatus.COMPLETED],
            "reason": reason,
            "at": at,
            "inferred": True,
        }
    )
    return {
        "stages": stages,
        "summary": {
            **summarize_decision_stages(stages),
            "inferred": True,
            "final_reason": reason,
        },
        "inferred": True,
    }


def build_profit_attribution(
    positions: list[Position],
    orders: list[Order],
    decisions: list[AIDecision],
    shadows: list[ShadowBacktest],
) -> dict[str, Any]:
    positions, trade_fact_quarantine = filter_trusted_closed_positions(list(positions))
    decisions_by_id = {
        int(decision.id): decision for decision in decisions if decision.id is not None
    }
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

        shadow = _best_shadow_for_position(position, entry_match.decision, shadows)
        bucket_key, bucket_label, confidence, notes = _classify_record(
            position,
            entry_match.decision,
            close_match.decision,
            shadow,
        )
        bucket = buckets.setdefault(
            bucket_key,
            {
                "key": bucket_key,
                "label": bucket_label,
                "count": 0,
                "pnl": 0.0,
                "profit": 0.0,
                "loss": 0.0,
            },
        )
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
            max((closed - opened).total_seconds() / 60.0, 0.0) if opened and closed else 0.0
        )
        records.append(
            {
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
                "evidence_status": _evidence_status(entry_match.decision, signals, shadow),
                "decision_state": _decision_state_or_closed_fallback(
                    raw_entry,
                    position,
                    entry_match.decision,
                ),
                "close_state": _decision_state_or_closed_fallback(
                    raw_close,
                    position,
                    close_match.decision,
                ),
            }
        )

    trade_count = len(positions)
    evidence_coverage = {
        key: sum(
            1
            for row in records
            if _safe_dict(_safe_dict(row.get("evidence_status")).get(key)).get("available")
        )
        for key in ("ai", "ml", "server_profit", "shadow")
    }
    evidence_coverage["all_core"] = sum(
        1
        for row in records
        if all(
            _safe_dict(_safe_dict(row.get("evidence_status")).get(key)).get("available")
            for key in ("ai", "ml", "shadow")
        )
    )
    summary = {
        "trade_count": trade_count,
        "total_closed_pnl": round(total_pnl, 6),
        "win_count": wins,
        "loss_count": losses,
        "win_rate": round(wins / trade_count, 4) if trade_count else 0.0,
        "avg_win": round(profit / wins, 6) if wins else 0.0,
        "avg_loss": round(loss / losses, 6) if losses else 0.0,
        "profit_factor": round(profit / loss, 4) if loss > 0 else None,
        "direction_error_count": sum(1 for row in records if row["bucket"] == "ai_direction_error"),
        "execution_issue_count": sum(
            1 for row in records if row["bucket"] in {"stop_loss_or_fast_risk", "flat_or_fee_churn"}
        ),
        "evidence_coverage": evidence_coverage,
    }
    bucket_rows = []
    for item in buckets.values():
        count = max(int(item["count"]), 1)
        bucket_rows.append(
            {
                **item,
                "pnl": round(float(item["pnl"]), 6),
                "profit": round(float(item["profit"]), 6),
                "loss": round(float(item["loss"]), 6),
                "avg_pnl": round(float(item["pnl"]) / count, 6),
            }
        )
    bucket_rows.sort(key=lambda item: abs(float(item["pnl"])), reverse=True)

    return {
        "summary": summary,
        "buckets": bucket_rows,
        "records": records,
        "trade_fact_quarantine": trade_fact_quarantine,
    }


def match_entry_decisions_for_positions(
    positions: list[Position],
    orders: list[Order],
    decisions: list[AIDecision],
) -> list[AIDecision]:
    """Return entry decisions used by the attribution builder."""
    positions, _trade_fact_quarantine = filter_trusted_closed_positions(list(positions))
    decisions_by_id = {
        int(decision.id): decision for decision in decisions if decision.id is not None
    }
    matched: dict[int, AIDecision] = {}
    for position in positions:
        entry_match = _match_order(position, orders, decisions_by_id, want_exit=False)
        if not entry_match.decision:
            entry_match = _match_nearest_decision(position, decisions, want_exit=False)
        decision = entry_match.decision
        if decision and decision.id is not None:
            matched[int(decision.id)] = decision
    return list(matched.values())
