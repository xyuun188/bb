"""Predictive reversal scoring for exit decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PREDICTIVE_REVERSAL_REVIEW_SCORE = 38.0
PREDICTIVE_REVERSAL_EXIT_SCORE = 64.0
PREDICTIVE_REVERSAL_FULL_EXIT_SCORE = 82.0
PREDICTIVE_REVERSAL_MIN_PROFIT_MULTIPLE = 1.0
PREDICTIVE_REVERSAL_REDUCE_FRACTION = 0.60


@dataclass(frozen=True, slots=True)
class ExitPredictiveReversalPolicy:
    """Score whether the next short window is turning against the held side."""

    review_score: float = PREDICTIVE_REVERSAL_REVIEW_SCORE
    exit_score: float = PREDICTIVE_REVERSAL_EXIT_SCORE
    full_exit_score: float = PREDICTIVE_REVERSAL_FULL_EXIT_SCORE

    def evidence(
        self,
        *,
        side: str,
        returns_1: float,
        returns_5: float,
        returns_20: float,
        volume_ratio: float,
        rsi_14: float,
        bb_pct: float,
        macd_diff: float,
        adx_14: float,
    ) -> dict[str, Any]:
        side = str(side or "").lower()
        if side not in {"long", "short"}:
            return {"score": 0.0, "level": "none", "reasons": []}

        reasons: list[str] = []
        score = 0.0
        adverse_1 = returns_1 <= -0.0025 if side == "long" else returns_1 >= 0.0025
        adverse_5 = returns_5 <= -0.0060 if side == "long" else returns_5 >= 0.0060
        adverse_20 = returns_20 <= -0.0100 if side == "long" else returns_20 >= 0.0100
        strong_adverse_1 = returns_1 <= -0.0060 if side == "long" else returns_1 >= 0.0060
        strong_adverse_5 = returns_5 <= -0.0140 if side == "long" else returns_5 >= 0.0140

        if adverse_1:
            score += 18.0
            reasons.append("1m_against")
        if adverse_5:
            score += 22.0
            reasons.append("5m_against")
        if adverse_20:
            score += 12.0
            reasons.append("20m_against")
        if strong_adverse_1 or strong_adverse_5:
            score += 12.0
            reasons.append("strong_short_window_against")
        if volume_ratio >= 1.15 and (adverse_1 or adverse_5):
            score += 12.0
            reasons.append("volume_confirms_reversal")
        if side == "long":
            if bb_pct >= 0.86 and rsi_14 >= 66 and (adverse_1 or adverse_5):
                score += 12.0
                reasons.append("long_overheated_reversal")
            if macd_diff < 0:
                score += 8.0
                reasons.append("macd_against_long")
        else:
            if bb_pct <= 0.14 and rsi_14 <= 34 and (adverse_1 or adverse_5):
                score += 12.0
                reasons.append("short_oversold_rebound")
            if macd_diff > 0:
                score += 8.0
                reasons.append("macd_against_short")
        if adx_14 < 16 and (adverse_1 or adverse_5):
            score += 6.0
            reasons.append("trend_strength_weak")

        if score >= self.full_exit_score:
            level = "full_exit"
        elif score >= self.exit_score:
            level = "exit"
        elif score >= self.review_score:
            level = "review"
        else:
            level = "none"

        return {
            "score": round(score, 4),
            "level": level,
            "reasons": reasons,
            "returns_1": round(returns_1, 6),
            "returns_5": round(returns_5, 6),
            "returns_20": round(returns_20, 6),
            "volume_ratio": round(volume_ratio, 4),
            "rsi_14": round(rsi_14, 4),
            "bb_pct": round(bb_pct, 4),
            "macd_diff": round(macd_diff, 8),
            "adx_14": round(adx_14, 4),
        }
