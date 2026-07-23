"""Turn authoritative fee-after trade memories into advisory decision feedback."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from core.training_contracts import is_authoritative_expert_memory_extra
from services.return_objective import RETURN_OBJECTIVE_NAME, RETURN_OBJECTIVE_VERSION

SIDES = ("long", "short")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_extra(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def canonical_memory_outcome(memory: dict[str, Any]) -> dict[str, Any] | None:
    extra = _safe_extra(memory.get("extra"))
    if not is_authoritative_expert_memory_extra(extra):
        return None
    if extra.get("objective") != RETURN_OBJECTIVE_NAME:
        return None
    if extra.get("objective_version") != RETURN_OBJECTIVE_VERSION:
        return None

    aggregation = _safe_dict(extra.get("outcome_aggregation"))
    if aggregation:
        count = _safe_int(aggregation.get("count"), 0)
        position_ids = {
            _safe_int(value, 0)
            for value in aggregation.get("source_position_ids", [])
            if _safe_int(value, 0) > 0
        }
        outcome_ids = {
            str(value)
            for value in aggregation.get("source_outcome_ids", [])
            if str(value or "").strip()
        }
        if count <= 0 or not position_ids or not outcome_ids:
            return None
        return {
            "count": count,
            "position_ids": position_ids,
            "total_return_pct": _safe_float(aggregation.get("total_net_return_pct")),
            "return_lcb_pct": _safe_float(
                aggregation.get("return_lcb_pct"),
                _safe_float(aggregation.get("avg_net_return_pct")),
            ),
            "total_pnl_usdt": _safe_float(
                aggregation.get("total_realized_net_pnl_usdt")
            ),
            "gross_profit_usdt": _safe_float(aggregation.get("gross_profit_usdt")),
            "gross_loss_usdt": _safe_float(aggregation.get("gross_loss_usdt")),
            "worst_return_pct": _safe_float(
                aggregation.get("worst_net_return_pct"),
                min(_safe_float(aggregation.get("avg_net_return_pct")), 0.0),
            ),
            "outcome_ids": outcome_ids,
        }

    position_id = _safe_int(extra.get("source_position_id"), 0)
    outcome_id = str(extra.get("outcome_id") or "").strip()
    if position_id <= 0 or not outcome_id or "net_return_after_all_cost_pct" not in extra:
        return None
    net_return = _safe_float(extra.get("net_return_after_all_cost_pct"))
    pnl = _safe_float(extra.get("realized_pnl"))
    return {
        "count": 1,
        "position_ids": {position_id},
        "total_return_pct": net_return,
        "return_lcb_pct": net_return,
        "total_pnl_usdt": pnl,
        "gross_profit_usdt": max(pnl, 0.0),
        "gross_loss_usdt": max(-pnl, 0.0),
        "worst_return_pct": min(net_return, 0.0),
        "outcome_ids": {outcome_id},
    }


@dataclass(frozen=True, slots=True)
class MemoryFeedbackPolicy:
    """Keep memory observable while denying it independent production permission."""

    def build(self, memories: list[dict[str, Any]]) -> dict[str, Any]:
        side_rows = {side: [] for side in SIDES}
        for memory in memories:
            if not isinstance(memory, dict):
                continue
            side = str(memory.get("side") or "").lower()
            if side in side_rows:
                side_rows[side].append(memory)

        by_side = {
            side: self._side_feedback(side, side_memories)
            for side, side_memories in side_rows.items()
        }
        preferred = "neutral"
        return {
            "enabled": any(
                item["canonical_outcome_count"] > 0 for item in by_side.values()
            ),
            "preferred_side_by_memory": preferred,
            "by_side": by_side,
            "decision_habit": self._decision_habit(by_side, preferred),
            "policy": (
                "Only complete authoritative OKX fee-after outcomes are accepted; memory never "
                "grants production permission."
            ),
        }

    def _side_feedback(self, side: str, memories: list[dict[str, Any]]) -> dict[str, Any]:
        positive = 0
        risk = 0
        top_reasons: list[str] = []
        outcomes: list[dict[str, Any]] = []
        seen_positions: set[int] = set()

        for memory in memories:
            outcome = canonical_memory_outcome(memory)
            if outcome is None:
                continue
            position_ids = set(outcome["position_ids"])
            if position_ids.intersection(seen_positions):
                continue
            outcomes.append(outcome)
            seen_positions.update(position_ids)
            if outcome["total_pnl_usdt"] > 0:
                positive += outcome["count"]
            elif outcome["total_pnl_usdt"] < 0:
                risk += outcome["count"]
            if len(top_reasons) < 3:
                lesson = str(memory.get("lesson") or memory.get("market_pattern") or "").strip()
                if lesson:
                    top_reasons.append(lesson[:96])

        count = sum(int(item["count"]) for item in outcomes)
        total_return = sum(float(item["total_return_pct"]) for item in outcomes)
        total_pnl = sum(float(item["total_pnl_usdt"]) for item in outcomes)
        gross_profit = sum(float(item["gross_profit_usdt"]) for item in outcomes)
        gross_loss = sum(float(item["gross_loss_usdt"]) for item in outcomes)
        avg_return = total_return / count if count else 0.0
        return_lcb = (
            sum(float(item["return_lcb_pct"]) * int(item["count"]) for item in outcomes)
            / count
            if count
            else 0.0
        )
        worst_return = min(
            [float(item["worst_return_pct"]) for item in outcomes],
            default=0.0,
        )
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
        pnl_activity = gross_profit + gross_loss
        pnl_efficiency = total_pnl / pnl_activity if pnl_activity > 0 else 0.0
        downside_scale = max(abs(worst_return), abs(avg_return), 1e-9)
        return_quality = return_lcb / downside_scale if count else 0.0
        utility = (return_quality + pnl_efficiency) / 2.0
        return {
            "side": side,
            "authoritative_memory_count": len(outcomes),
            "positive_evidence_count": positive,
            "risk_evidence_count": risk,
            "net_evidence_count": positive - risk,
            "canonical_outcome_count": count,
            "canonical_position_count": len(seen_positions),
            "cost_complete": bool(count),
            "total_realized_net_pnl_usdt": round(total_pnl, 6),
            "avg_net_return_after_all_cost_pct": round(avg_return, 6),
            "return_lcb_pct": round(return_lcb, 6),
            "worst_net_return_pct": round(worst_return, 6),
            "profit_factor": round(profit_factor, 6) if profit_factor is not None else None,
            "utility": round(utility, 6),
            "score_adjustment": 0.0,
            "candidate_score_bonus": 0.0,
            "action_bias": "fee_after_observation_only",
            "expected_return_hint_pct": 0.0,
            "missed_return_evidence_count": 0,
            "missed_avg_return_pct": 0.0,
            "top_reasons": top_reasons,
        }

    @staticmethod
    def _decision_habit(
        by_side: dict[str, dict[str, Any]],
        preferred_side: str,
    ) -> dict[str, Any]:
        side_habits: dict[str, dict[str, Any]] = {}
        for side in SIDES:
            item = _safe_dict(by_side.get(side))
            side_habits[side] = {
                "stance": "fee_after_observation_only",
                "proactive_level": 0.0,
                "expected_return_hint_pct": 0.0,
                "score_adjustment": 0.0,
                "return_lcb_pct": item.get("return_lcb_pct"),
                "canonical_outcome_count": item.get("canonical_outcome_count", 0),
                "cost_complete": bool(item.get("cost_complete")),
                "reason": "memory is observation-only and cannot change production actions",
            }
        return {
            "posture": "observation_only",
            "preferred_side": "neutral",
            "conservative_sides": [],
            "by_side": side_habits,
            "rule": (
                "Memory cannot grant or modify trades, direction, sizing, leverage, exits, or routing."
            ),
        }
