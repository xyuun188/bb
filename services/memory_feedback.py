"""Turn review memories into structured decision feedback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SIDES = ("long", "short")
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
        return {
            "enabled": bool(memories),
            "preferred_side_by_memory": preferred,
            "by_side": by_side,
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
            max_probe_size_pct = 0.025 if missed >= 6 and score_adjustment >= 0.08 else 0.015
        elif score_adjustment > 0.02:
            action_bias = "slightly_improve_entry_confidence"
            max_probe_size_pct = 0.012
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

    @staticmethod
    def _preferred_side(by_side: dict[str, dict[str, Any]]) -> str:
        long_score = _safe_float(by_side.get("long", {}).get("score_adjustment"), 0.0)
        short_score = _safe_float(by_side.get("short", {}).get("score_adjustment"), 0.0)
        if long_score > short_score + 0.025:
            return "long"
        if short_score > long_score + 0.025:
            return "short"
        return "neutral"
