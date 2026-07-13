"""Execution messaging for return-ranked entry candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import DecisionOutput


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class EntryExecutionPriorityPolicy:
    """Never bypass same-round authoritative return ranking."""

    def immediate_execution_reason(self, decision: DecisionOutput) -> None:
        del decision
        return None

    def wait_sort_reason(
        self,
        decision: DecisionOutput,
        *,
        rank: int | None = None,
        candidate_count: int | None = None,
    ) -> str:
        raw = _safe_dict(decision.raw_response)
        opportunity = _safe_dict(raw.get("opportunity_score"))
        return_lcb = _safe_float(opportunity.get("return_lcb_pct"), 0.0)
        expected_loss = _safe_float(opportunity.get("expected_loss_pct"), 0.0)
        rank_text = (
            f"{rank}/{candidate_count}"
            if rank is not None and candidate_count is not None
            else "pending"
        )
        return (
            f"费后收益候选排名 {rank_text}；收益下界 {return_lcb:.4f}%，"
            f"预期下行 {expected_loss:.4f}%。"
        )
