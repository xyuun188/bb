"""Entry sizing policy helpers.

The functions here translate lower-level evidence scores into concrete size
context.  Leverage is allocated later by the dynamic leverage allocator so an
evidence tier cannot permanently lock a symbol to a fixed 2x/3x/4x bucket.
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
    """Apply evidence score size context without fixed leverage buckets."""
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
        "degraded_missing_probe",
        "blocked",
    }


def evidence_is_tradeable_probe(evidence_score: dict[str, Any], effective_score: float) -> bool:
    """Return True when low evidence may still trade only as a controlled probe."""

    if not evidence_score or effective_score < ENTRY_EVIDENCE_SCORE_WEAK_PROBE:
        return False
    return str(evidence_score.get("tier") or "") not in {
        "weak_conflict_probe",
        "degraded_missing_probe",
        "blocked",
    }
