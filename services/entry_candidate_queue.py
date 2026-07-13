"""Authoritative fee-after return ranking for entry candidates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import DecisionOutput

EntryCandidate = tuple[str, str, DecisionOutput, Any, int | None]
ScoreCandidate = Callable[[DecisionOutput, dict[str, Any] | None], float]
WaitSortReason = Callable[..., str]


@dataclass(frozen=True, slots=True)
class RankedEntryCandidate:
    candidate: EntryCandidate
    rank: int
    candidate_count: int
    score: float
    wait_reason: str


@dataclass(frozen=True, slots=True)
class EntryCandidateQueuePolicy:
    """Sort only by the authoritative opportunity score."""

    score_candidate: ScoreCandidate
    wait_sort_reason: WaitSortReason

    def ranked(
        self,
        candidates: list[EntryCandidate],
        strategy_context: dict[str, Any] | None,
    ) -> list[RankedEntryCandidate]:
        scored = [
            (
                float(self.score_candidate(candidate[2], strategy_context)),
                candidate,
            )
            for candidate in candidates
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        count = len(scored)
        ranked: list[RankedEntryCandidate] = []
        for rank, (score, candidate) in enumerate(scored, start=1):
            decision = candidate[2]
            raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
            opportunity = (
                raw.get("opportunity_score")
                if isinstance(raw.get("opportunity_score"), dict)
                else {}
            )
            opportunity["authoritative_queue"] = {
                "rank": rank,
                "candidate_count": count,
                "score": round(score, 8),
                "policy": "fee_after_return_lcb_minus_expected_downside_only",
            }
            raw["opportunity_score"] = opportunity
            decision.raw_response = raw
            ranked.append(
                RankedEntryCandidate(
                    candidate=candidate,
                    rank=rank,
                    candidate_count=count,
                    score=score,
                    wait_reason=self.wait_sort_reason(
                        decision,
                        rank=rank,
                        candidate_count=count,
                    ),
                )
            )
        return ranked
