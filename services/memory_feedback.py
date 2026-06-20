"""Turn review memories into structured decision feedback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from services.trading_params import DEFAULT_TRADING_PARAMS

SIDES = ("long", "short")
ENTRY_RISK_SIZING_PARAMS = DEFAULT_TRADING_PARAMS.entry_risk_sizing
POSITIVE_MEMORY_TYPES = {
    "profit_pattern",
    "shadow_good_signal",
    "shadow_missed_opportunity",
}
RISK_MEMORY_TYPES = {
    "loss_lesson",
    "flat_lesson",
    "shadow_bad_signal",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


@dataclass(frozen=True, slots=True)
class MemoryFeedbackPolicy:
    """Aggregate shadow backtests, trade reflections, and expert memory.

    The result is intentionally advisory. It may bias the model toward a small
    probe when repeated missed opportunities are present, but it never creates
    permission to bypass hard risk checks.
    """

    max_candidate_bonus: float = 0.24
    max_candidate_penalty: float = -0.32

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
        habit = self._decision_habit(by_side, preferred)
        return {
            "enabled": bool(memories),
            "preferred_side_by_memory": preferred,
            "by_side": by_side,
            "decision_habit": habit,
            "policy": (
                "Review feedback is advisory evidence from shadow backtests, trade reflections, "
                "and expert long-term memory. Use it to choose small probes or stricter "
                "confirmation; never use it to bypass hard risk."
            ),
        }

    def _side_feedback(self, side: str, memories: list[dict[str, Any]]) -> dict[str, Any]:
        missed = 0
        positive = 0
        risk = 0
        shadow = 0
        trade = 0
        contribution = 0.0
        top_reasons: list[str] = []

        for memory in memories:
            memory_type = str(memory.get("memory_type") or "lesson")
            evidence = max(_safe_int(memory.get("evidence_count"), 1), 1)
            capped_evidence = min(evidence, 10)
            confidence = _clamp(_safe_float(memory.get("confidence_score"), 0.5), 0.10, 0.95)
            adjustment = _safe_float(memory.get("confidence_adjustment"), 0.0)
            weight = confidence * (1.0 + capped_evidence * 0.08)

            if memory_type.startswith("shadow_"):
                shadow += evidence
            else:
                trade += evidence

            if memory_type == "shadow_missed_opportunity":
                missed += evidence
                positive += evidence
                contribution += max(adjustment, 0.035) * weight * 1.20
            elif memory_type in POSITIVE_MEMORY_TYPES:
                positive += evidence
                contribution += max(adjustment, 0.020) * weight
            elif memory_type in RISK_MEMORY_TYPES:
                risk += evidence
                contribution += min(adjustment, -0.035) * weight * 1.15
            elif adjustment:
                if adjustment > 0:
                    positive += evidence
                else:
                    risk += evidence
                contribution += adjustment * weight

            if len(top_reasons) < 3:
                lesson = str(memory.get("lesson") or memory.get("market_pattern") or "").strip()
                if lesson:
                    top_reasons.append(lesson[:96])

        net_evidence = positive - risk
        score_adjustment = _clamp(contribution, -0.25, 0.18)
        candidate_bonus = _clamp(
            score_adjustment * 1.35, self.max_candidate_penalty, self.max_candidate_bonus
        )
        allow_probe = bool(missed >= 2 and positive >= max(risk, 1) and score_adjustment >= 0.035)
        risk_dominant = bool(risk >= positive + 2 and score_adjustment < 0)
        if risk_dominant:
            action_bias = "require_stronger_confirmation"
            max_probe_size_pct = 0.0
        elif allow_probe:
            action_bias = "prefer_small_probe_when_current_ev_positive"
            max_probe_size_pct = (
                ENTRY_RISK_SIZING_PARAMS.memory_feedback_strong_probe_size_pct
                if missed >= 6 and score_adjustment >= 0.08
                else ENTRY_RISK_SIZING_PARAMS.memory_feedback_normal_probe_size_pct
            )
        elif score_adjustment > 0.02:
            action_bias = "slightly_improve_entry_confidence"
            max_probe_size_pct = ENTRY_RISK_SIZING_PARAMS.memory_feedback_light_probe_size_pct
        else:
            action_bias = "neutral"
            max_probe_size_pct = 0.0

        return {
            "side": side,
            "memory_count": len(memories),
            "shadow_evidence_count": shadow,
            "trade_evidence_count": trade,
            "missed_opportunity_count": missed,
            "positive_evidence_count": positive,
            "risk_evidence_count": risk,
            "net_evidence_count": net_evidence,
            "score_adjustment": round(score_adjustment, 6),
            "candidate_score_bonus": round(candidate_bonus, 6),
            "allow_probe": allow_probe,
            "action_bias": action_bias,
            "max_probe_size_pct": round(max_probe_size_pct, 6),
            "top_reasons": top_reasons,
        }

    def _decision_habit(
        self,
        by_side: dict[str, dict[str, Any]],
        preferred_side: str,
    ) -> dict[str, Any]:
        """Summarize review memory as an explicit LLM behavior contract."""

        side_habits = {side: self._side_habit(_safe_dict(by_side.get(side))) for side in SIDES}
        active_probe_sides = [
            side for side, habit in side_habits.items() if habit.get("stance") == "probe_when_ev_ok"
        ]
        conservative_sides = [
            side for side, habit in side_habits.items() if habit.get("stance") == "strict_confirm"
        ]
        if active_probe_sides:
            posture = "selective_probe"
        elif conservative_sides:
            posture = "defensive_selective"
        else:
            posture = "neutral"
        return {
            "posture": posture,
            "preferred_side": preferred_side,
            "active_probe_sides": active_probe_sides,
            "conservative_sides": conservative_sides,
            "by_side": side_habits,
            "rule": (
                "Act slightly earlier only for sides with repeated missed opportunities, "
                "positive current EV and controlled tail risk; tighten confirmation and size "
                "for sides dominated by realized losses."
            ),
        }

    @staticmethod
    def _side_habit(item: dict[str, Any]) -> dict[str, Any]:
        missed = _safe_int(item.get("missed_opportunity_count"), 0)
        positive = _safe_int(item.get("positive_evidence_count"), 0)
        risk = _safe_int(item.get("risk_evidence_count"), 0)
        score_adjustment = _safe_float(item.get("score_adjustment"), 0.0)
        candidate_bonus = _safe_float(item.get("candidate_score_bonus"), 0.0)
        max_probe_size = _safe_float(item.get("max_probe_size_pct"), 0.0)
        if risk >= positive + 2 and score_adjustment < 0:
            return {
                "stance": "strict_confirm",
                "proactive_level": 0.0,
                "probe_budget_pct": 0.0,
                "min_expected_net_pct": 0.35,
                "max_loss_probability": 0.42,
                "max_tail_risk": 0.82,
                "score_adjustment": round(candidate_bonus, 6),
                "reason": "realized or shadow losses dominate this side",
            }
        if missed >= 2 and positive >= max(risk, 1) and score_adjustment >= 0.035:
            proactive_level = _clamp(0.25 + missed * 0.055 + score_adjustment, 0.30, 0.85)
            return {
                "stance": "probe_when_ev_ok",
                "proactive_level": round(proactive_level, 6),
                "probe_budget_pct": round(max(max_probe_size, 0.012), 6),
                "min_expected_net_pct": 0.12,
                "max_loss_probability": 0.58,
                "max_tail_risk": 0.98,
                "score_adjustment": round(candidate_bonus, 6),
                "reason": "shadow or trade reviews show repeated missed opportunities",
            }
        if score_adjustment > 0.02:
            return {
                "stance": "slightly_support",
                "proactive_level": round(_clamp(score_adjustment * 2.5, 0.05, 0.30), 6),
                "probe_budget_pct": round(max_probe_size, 6),
                "min_expected_net_pct": 0.20,
                "max_loss_probability": 0.54,
                "max_tail_risk": 0.92,
                "score_adjustment": round(candidate_bonus, 6),
                "reason": "positive review evidence is present but not repeated enough",
            }
        return {
            "stance": "neutral",
            "proactive_level": 0.0,
            "probe_budget_pct": 0.0,
            "min_expected_net_pct": 0.25,
            "max_loss_probability": 0.50,
            "max_tail_risk": 0.90,
            "score_adjustment": round(candidate_bonus, 6),
            "reason": "no strong review habit adjustment",
        }

    @staticmethod
    def _preferred_side(by_side: dict[str, dict[str, Any]]) -> str:
        long_score = _safe_float(by_side.get("long", {}).get("score_adjustment"), 0.0)
        short_score = _safe_float(by_side.get("short", {}).get("score_adjustment"), 0.0)
        if long_score > short_score + 0.025:
            return "long"
        if short_score > long_score + 0.025:
            return "short"
        return "neutral"
