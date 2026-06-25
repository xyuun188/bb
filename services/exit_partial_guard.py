"""Partial-exit guard for losing positions.

Ordinary partial closes on an aggregate losing position can shrink exposure
before the later stop/take-profit logic has a clean chance to work.  This policy
keeps that rule outside the main TradingService while preserving hard-risk
exit bypasses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import DecisionOutput
from services.exit_intent import (
    COOLDOWN_BYPASS_INTENTS,
    classify_exit_intent,
    is_low_quality_release_without_hard_risk,
)
from services.exit_position_matcher import ExitPositionMatcher


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class ExitPartialGuardPolicy:
    """Block ordinary partial exits while the aggregate symbol-side is losing."""

    matcher: ExitPositionMatcher

    def guard_reason(
        self,
        model_name: str,
        decision: DecisionOutput,
        open_positions: list[dict[str, Any]] | None,
    ) -> str | None:
        if not decision.is_exit:
            return None

        raw = _safe_dict(decision.raw_response)
        close_evidence = _safe_dict(raw.get("close_evidence"))
        fast_trigger = str(raw.get("fast_risk_trigger") or "")
        exit_intent = classify_exit_intent(decision)
        raw = _safe_dict(decision.raw_response)
        close_evidence = _safe_dict(raw.get("close_evidence"))
        close_fraction = _safe_float(
            (
                raw.get("close_fraction")
                if raw.get("close_fraction") is not None
                else decision.position_size_pct
            ),
            1.0,
        )
        action_plan = str(
            raw.get("action_plan")
            or close_evidence.get("action_plan")
            or raw.get("exit_action_plan")
            or ""
        ).lower()
        partial_intent = (0.0 < close_fraction < 0.999) or action_plan == "reduce"
        if not partial_intent:
            return None

        exit_quality = _safe_dict(raw.get("exit_quality"))
        invalidation = _safe_dict(exit_quality.get("invalidation"))
        forced_hard_exit = bool(
            not is_low_quality_release_without_hard_risk(raw)
            and (
                fast_trigger in {"stop_loss", "take_profit", "hard_adverse_move"}
                or (
                    fast_trigger in {"near_stop_progress", "fast_adverse_move"}
                    and close_fraction >= 0.999
                )
                or raw.get("forced_exit")
                or close_evidence.get("hard_risk")
                or close_evidence.get("forced_exit")
                or invalidation.get("severe")
                or decision.model_name == "risk_engine"
                or exit_intent in COOLDOWN_BYPASS_INTENTS
            )
        )
        if forced_hard_exit:
            return None

        matches = self.matcher.matching_positions(
            open_positions,
            model_name,
            decision,
            require_model_name=False,
        )
        if not matches:
            return None

        target_side = self.matcher.target_side(decision)
        latest_price = _safe_float(
            (decision.feature_snapshot or {}).get("current_price")
            or (decision.feature_snapshot or {}).get("close"),
            0.0,
        )
        total_qty = 0.0
        entry_value = 0.0
        estimated_gross = 0.0
        reported_unrealized = 0.0
        reported_available = False
        for pos in matches:
            pos_info = _safe_dict(pos.get("info"))
            qty = abs(
                _safe_float(pos.get("quantity") or pos.get("contracts") or pos.get("sz"), 0.0)
            )
            contract_size = _safe_float(
                pos.get("contract_size") or pos.get("contractSize") or pos_info.get("ctVal"),
                1.0,
            )
            qty_for_pnl = qty * (contract_size if contract_size > 0 else 1.0)
            entry = _safe_float(
                pos.get("entry_price") or pos.get("entryPrice") or pos.get("avgPx"),
                0.0,
            )
            current = _safe_float(
                pos.get("current_price") or pos.get("markPrice") or pos.get("lastPrice") or entry,
                entry,
            )
            if latest_price <= 0 and current > 0:
                latest_price = current
            if qty_for_pnl <= 0 or entry <= 0:
                continue
            total_qty += qty_for_pnl
            entry_value += entry * qty_for_pnl
            mark = latest_price if latest_price > 0 else current
            if mark > 0:
                estimated_gross += (
                    (mark - entry) * qty_for_pnl
                    if target_side == "long"
                    else (entry - mark) * qty_for_pnl
                )
            unrealized = _safe_float(
                pos.get("unrealized_pnl")
                or pos.get("unrealizedPnl")
                or pos.get("upl")
                or pos_info.get("upl")
                or pos_info.get("unrealizedPnl"),
                0.0,
            )
            if abs(unrealized) > 1e-12:
                reported_available = True
                reported_unrealized += unrealized

        if total_qty <= 0:
            return None

        aggregate_entry = entry_value / max(total_qty, 1e-12)
        aggregate_pnl = reported_unrealized if reported_available else estimated_gross
        if aggregate_pnl >= -1e-9:
            return None

        raw["loss_partial_exit_guard"] = {
            "applied": True,
            "symbol": self.matcher.normalize_symbol(decision.symbol),
            "side": target_side,
            "fragments": len(matches),
            "close_fraction": round(close_fraction, 6),
            "fast_risk_trigger": fast_trigger,
            "aggregate_entry_price": round(aggregate_entry, 8),
            "latest_price": round(latest_price, 8),
            "aggregate_unrealized_pnl": round(aggregate_pnl, 6),
            "reason": "亏损仓位的普通部分平仓已禁用，避免切碎仓位后错过后续止盈。",
        }
        decision.raw_response = raw
        side_label = "做多" if target_side == "long" else "做空"
        return (
            f"亏损部分平仓保护：{decision.symbol} {side_label} 当前按整体持仓估算仍浮亏 "
            f"{aggregate_pnl:.4f}U，本次只计划平 {close_fraction:.0%}。"
            "系统已禁用普通亏损部分平仓，避免在回撤中反复切碎仓位，导致后续真正触发止盈时剩余仓位太小。"
            "若触发硬止损、真实止盈、接近/超过计划止损、严重趋势失效或风控强制全平，仍会允许执行。"
        )
