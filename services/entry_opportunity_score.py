"""Entry opportunity scoring boundary."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import DecisionOutput

EntryOpportunityScoreEvaluator = Callable[[DecisionOutput, dict[str, Any] | None], float]


@dataclass(frozen=True, slots=True)
class EntryOpportunityScorePolicy:
    """Score an entry candidate using the configured opportunity model."""

    evaluator: EntryOpportunityScoreEvaluator

    def score_candidate(
        self,
        decision: DecisionOutput,
        strategy: dict[str, Any] | None = None,
    ) -> float:
        """Return the opportunity score for an entry candidate."""

        return self.evaluator(decision, strategy)
