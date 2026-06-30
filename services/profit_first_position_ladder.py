"""Profit-First v3 lane-based position ladder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


LANE_ORDER = {
    "shadow_only": 0,
    "tiny_probe": 1,
    "validated_probe": 2,
    "meaningful_entry": 3,
    "high_conviction": 4,
}


@dataclass(frozen=True, slots=True)
class ProfitFirstPositionLadderDecision:
    lane: str
    target_min_pct: float
    target_max_pct: float
    original_size_pct: float
    adjusted_size_pct: float
    capped_by_low_payoff: bool = False
    capped_by_high_conviction_gate: bool = False
    raised_to_lane_floor: bool = False
    capped_to_lane_ceiling: bool = False
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "profit-first-position-ladder-v1",
            "lane": self.lane,
            "target_min_pct": round(self.target_min_pct, 6),
            "target_max_pct": round(self.target_max_pct, 6),
            "original_size_pct": round(self.original_size_pct, 6),
            "adjusted_size_pct": round(self.adjusted_size_pct, 6),
            "capped_by_low_payoff": self.capped_by_low_payoff,
            "capped_by_high_conviction_gate": self.capped_by_high_conviction_gate,
            "raised_to_lane_floor": self.raised_to_lane_floor,
            "capped_to_lane_ceiling": self.capped_to_lane_ceiling,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class ProfitFirstPositionLadderPolicy:
    """Map Profit-First lanes to auditable position-size ranges."""

    tiny_probe_range: tuple[float, float] = (0.01, 0.02)
    validated_probe_range: tuple[float, float] = (0.03, 0.05)
    meaningful_entry_range: tuple[float, float] = (0.05, 0.08)
    high_conviction_range: tuple[float, float] = (0.08, 0.12)
    low_payoff_max_pct: float = 0.02
    high_conviction_enabled: bool = False

    def apply(
        self,
        *,
        lane: str,
        current_size_pct: float,
        low_payoff_quality: bool,
        high_risk_review: dict[str, Any] | None = None,
    ) -> ProfitFirstPositionLadderDecision:
        normalized_lane = self._normalize_lane(lane)
        target_min, target_max = self._target_range(normalized_lane)
        original = max(float(current_size_pct or 0.0), 0.0)
        adjusted = original
        reasons: list[str] = []
        capped_by_low_payoff = False
        capped_by_high_conviction_gate = False

        if normalized_lane == "shadow_only":
            adjusted = 0.0
            reasons.append("shadow_only_lane_has_no_real_position")
        else:
            if normalized_lane == "high_conviction" and not self._high_conviction_allowed(
                high_risk_review
            ):
                normalized_lane = "meaningful_entry"
                target_min, target_max = self.meaningful_entry_range
                capped_by_high_conviction_gate = True
                reasons.append("high_conviction_requires_enabled_gate_and_high_risk_review")
            if adjusted < target_min:
                adjusted = target_min
                reasons.append("raised_to_profit_first_lane_floor")
            if adjusted > target_max:
                adjusted = target_max
                reasons.append("capped_to_profit_first_lane_ceiling")

        if low_payoff_quality and adjusted > self.low_payoff_max_pct:
            adjusted = min(adjusted, self.low_payoff_max_pct)
            capped_by_low_payoff = True
            reasons.append("low_payoff_cannot_receive_meaningful_size")

        return ProfitFirstPositionLadderDecision(
            lane=normalized_lane,
            target_min_pct=target_min,
            target_max_pct=target_max,
            original_size_pct=original,
            adjusted_size_pct=max(adjusted, 0.0),
            capped_by_low_payoff=capped_by_low_payoff,
            capped_by_high_conviction_gate=capped_by_high_conviction_gate,
            raised_to_lane_floor=adjusted > original,
            capped_to_lane_ceiling=adjusted < original,
            reasons=tuple(dict.fromkeys(reasons)),
        )

    def _high_conviction_allowed(self, review: dict[str, Any] | None) -> bool:
        row = review if isinstance(review, dict) else {}
        return bool(
            self.high_conviction_enabled
            and row.get("approved") is True
            and row.get("profit_first_allow_high_conviction") is True
        )

    def _target_range(self, lane: str) -> tuple[float, float]:
        if lane == "tiny_probe":
            return self.tiny_probe_range
        if lane == "validated_probe":
            return self.validated_probe_range
        if lane == "meaningful_entry":
            return self.meaningful_entry_range
        if lane == "high_conviction":
            return self.high_conviction_range
        return (0.0, 0.0)

    @staticmethod
    def _normalize_lane(lane: str) -> str:
        text = str(lane or "").lower().strip()
        return text if text in LANE_ORDER else "shadow_only"
