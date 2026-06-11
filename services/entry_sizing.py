"""Entry sizing policy helpers.

The functions here translate lower-level evidence scores into concrete size and
leverage caps.  TradingService remains the orchestrator that owns balances and
exchange state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from services.entry_evidence import (
    ENTRY_EVIDENCE_SCORE_SMALL,
    ENTRY_EVIDENCE_SCORE_WEAK_PROBE,
)
from services.entry_signal_extraction import safe_float


@dataclass(slots=True)
class EvidenceSizingResult:
    position_size_pct: float
    leverage: float
    caps: list[str] = field(default_factory=list)
    tier: str = ""
    effective_score: float = 100.0


def apply_evidence_sizing_policy(
    *,
    evidence_score: dict[str, Any],
    current_size: float,
    leverage: float,
) -> EvidenceSizingResult:
    """Apply evidence score tier caps to the current entry size/leverage."""
    caps: list[str] = []
    if not evidence_score:
        return EvidenceSizingResult(
            position_size_pct=current_size,
            leverage=leverage,
            caps=caps,
        )

    tier = str(evidence_score.get("tier") or "")
    multiplier = safe_float(evidence_score.get("size_multiplier"), 1.0)
    effective_score = safe_float(evidence_score.get("effective_score"), 100.0)
    max_size = safe_float(evidence_score.get("max_size_pct"), 0.0)

    if multiplier <= 0:
        current_size = 0.0
        caps.append("动态证据评分低于可交易下限，仓位归零等待下一轮重新分析")
    elif multiplier < 0.999:
        original_size = current_size
        current_size = min(current_size * multiplier, current_size)
        caps.append(
            f"动态证据评分 {effective_score:.1f}，按 {tier or 'unknown'} 档把仓位从 "
            f"{original_size:.4f} 调整为 {current_size:.4f}"
        )

    if max_size > 0 and current_size > max_size:
        current_size = max_size
        caps.append(f"动态证据评分限制最大仓位为 {max_size:.4f}")

    leverage_cap = {
        "weak_conflict_probe": 2.0,
        "exploration": 3.0,
        "small": 4.0,
        "medium": 6.0,
    }.get(tier)
    if leverage_cap and leverage > leverage_cap:
        leverage = leverage_cap
        caps.append(f"动态证据评分为 {tier} 档，杠杆上限降到 {leverage_cap:.1f}x")

    return EvidenceSizingResult(
        position_size_pct=current_size,
        leverage=leverage,
        caps=caps,
        tier=tier,
        effective_score=effective_score,
    )


def evidence_is_low_payoff_quality(evidence_score: dict[str, Any], effective_score: float) -> bool:
    if not evidence_score:
        return False
    return effective_score < ENTRY_EVIDENCE_SCORE_SMALL or str(evidence_score.get("tier")) in {
        "weak_conflict_probe",
        "blocked",
    }


def evidence_is_tradeable_probe(evidence_score: dict[str, Any], effective_score: float) -> bool:
    """Return True when low evidence may still trade only as a controlled probe."""

    return bool(evidence_score) and effective_score >= ENTRY_EVIDENCE_SCORE_WEAK_PROBE
