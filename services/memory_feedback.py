"""Turn authoritative fee-after trade memories into advisory decision feedback."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from services.return_objective import RETURN_OBJECTIVE_NAME, RETURN_OBJECTIVE_VERSION
from services.trading_params import DEFAULT_TRADING_PARAMS

SIDES = ("long", "short")
ENTRY_RISK_SIZING_PARAMS = DEFAULT_TRADING_PARAMS.entry_risk_sizing


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
    source = str(extra.get("source") or "").lower()
    if source.startswith("shadow") or str(memory.get("memory_type") or "").startswith("shadow_"):
        return None
    if extra.get("cost_complete") is not True:
        return None
    if extra.get("production_evidence_eligible") is not True:
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
        if count <= 0 or not position_ids:
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
        }

    position_id = _safe_int(extra.get("source_position_id"), 0)
    if position_id <= 0 or "net_return_after_cost_pct" not in extra:
        return None
    net_return = _safe_float(extra.get("net_return_after_cost_pct"))
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
        preferred = self._preferred_side(by_side)
        return {
            "enabled": bool(memories),
            "preferred_side_by_memory": preferred,
            "by_side": by_side,
            "decision_habit": self._decision_habit(by_side, preferred),
            "policy": (
                "Shadow and cost-incomplete memories are observation-only. Only authoritative "
                "fee-after outcomes may tighten risk; memory never grants a production probe."
            ),
        }

    def _side_feedback(self, side: str, memories: list[dict[str, Any]]) -> dict[str, Any]:
        shadow = 0
        trade = 0
        positive = 0
        risk = 0
        missed = 0
        top_reasons: list[str] = []
        outcomes: list[dict[str, Any]] = []
        seen_positions: set[int] = set()

        for memory in memories:
            memory_type = str(memory.get("memory_type") or "lesson")
            evidence = max(_safe_int(memory.get("evidence_count"), 1), 1)
            if memory_type.startswith("shadow_"):
                shadow += evidence
                if memory_type == "shadow_missed_opportunity":
                    missed += evidence
            else:
                trade += evidence
            outcome = canonical_memory_outcome(memory)
            if outcome is not None:
                position_ids = set(outcome["position_ids"])
                if not position_ids.intersection(seen_positions):
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
        credibility = math.sqrt(count) / (math.sqrt(count) + 1.0) if count else 0.0
        score_adjustment = min(math.tanh(utility) * credibility, 0.0)
        risk_dominant = bool(count and return_lcb < 0.0)

        return {
            "side": side,
            "memory_count": len(memories),
            "shadow_evidence_count": shadow,
            "trade_evidence_count": trade,
            "missed_opportunity_count": missed,
            "positive_evidence_count": positive,
            "risk_evidence_count": risk,
            "net_evidence_count": positive - risk,
            "canonical_outcome_count": count,
            "canonical_position_count": len(seen_positions),
            "cost_complete": bool(count),
            "total_realized_net_pnl_usdt": round(total_pnl, 6),
            "avg_net_return_after_cost_pct": round(avg_return, 6),
            "return_lcb_pct": round(return_lcb, 6),
            "worst_net_return_pct": round(worst_return, 6),
            "profit_factor": round(profit_factor, 6) if profit_factor is not None else None,
            "utility": round(utility, 6),
            "score_adjustment": round(score_adjustment, 6),
            "candidate_score_bonus": 0.0,
            "allow_probe": False,
            "action_bias": (
                "require_stronger_confirmation" if risk_dominant else "fee_after_observation_only"
            ),
            "max_probe_size_pct": 0.0,
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
            strict = bool(item.get("canonical_outcome_count") and _safe_float(item.get("return_lcb_pct")) < 0)
            side_habits[side] = {
                "stance": "strict_confirm" if strict else "fee_after_observation_only",
                "proactive_level": 0.0,
                "probe_budget_pct": 0.0,
                "expected_return_hint_pct": 0.0,
                "score_adjustment": _safe_float(item.get("score_adjustment")),
                "return_lcb_pct": item.get("return_lcb_pct"),
                "canonical_outcome_count": item.get("canonical_outcome_count", 0),
                "cost_complete": bool(item.get("cost_complete")),
                "reason": (
                    "authoritative fee-after return lower bound is negative"
                    if strict
                    else "memory is observation-only and cannot authorize production probes"
                ),
            }
        conservative = [
            side for side, habit in side_habits.items() if habit["stance"] == "strict_confirm"
        ]
        return {
            "posture": "defensive_selective" if conservative else "neutral",
            "preferred_side": preferred_side,
            "active_probe_sides": [],
            "conservative_sides": conservative,
            "by_side": side_habits,
            "rule": (
                "Memory cannot grant probes. Negative authoritative fee-after lower bounds "
                "tighten risk; current dynamic strategy must independently authorize entries."
            ),
        }

    @staticmethod
    def _preferred_side(by_side: dict[str, dict[str, Any]]) -> str:
        long_item = _safe_dict(by_side.get("long"))
        short_item = _safe_dict(by_side.get("short"))
        long_count = _safe_int(long_item.get("canonical_outcome_count"), 0)
        short_count = _safe_int(short_item.get("canonical_outcome_count"), 0)
        if not long_count and not short_count:
            return "neutral"
        long_lcb = _safe_float(long_item.get("return_lcb_pct")) if long_count else -math.inf
        short_lcb = _safe_float(short_item.get("return_lcb_pct")) if short_count else -math.inf
        if long_lcb == short_lcb:
            return "neutral"
        return "long" if long_lcb > short_lcb else "short"
