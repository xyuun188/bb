"""Entry candidate queue ranking for round-end execution."""

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
    """An entry candidate with its current round ranking metadata."""

    candidate: EntryCandidate
    rank: int
    candidate_count: int
    score: float
    wait_reason: str


@dataclass(frozen=True, slots=True)
class EntryCandidateQueuePolicy:
    """Sort queued entry candidates by opportunity score."""

    score_candidate: ScoreCandidate
    wait_sort_reason: WaitSortReason

    def ranked(
        self,
        candidates: list[EntryCandidate],
        strategy_context: dict[str, Any] | None,
    ) -> list[RankedEntryCandidate]:
        scored_candidates = [
            (self.score_candidate(candidate[2], strategy_context), candidate)
            for candidate in candidates
        ]
        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        candidate_count = len(scored_candidates)
        ranked: list[RankedEntryCandidate] = []
        for rank, (score, candidate) in enumerate(scored_candidates, start=1):
            decision = candidate[2]
            wait_reason = self.wait_sort_reason(
                decision,
                rank=rank,
                candidate_count=candidate_count,
            )
            ranked.append(
                RankedEntryCandidate(
                    candidate=candidate,
                    rank=rank,
                    candidate_count=candidate_count,
                    score=float(score),
                    wait_reason=wait_reason,
                )
            )
        return ranked
