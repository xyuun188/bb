"""Observation-only record for batched expert diversity."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from ai_brain.base_model import DecisionOutput

if TYPE_CHECKING:
    from data_feed.feature_vector import FeatureVector


@dataclass(frozen=True, slots=True)
class ObjectiveEvidence:
    side: str | None
    score: float
    reasons: tuple[str, ...] = ()
    hard_risk: bool = False
    hard_risk_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExpertDiversityReview:
    should_retry: bool
    reason: str
    low_information_consensus: bool
    objective_evidence: ObjectiveEvidence
    target_experts: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["objective_evidence"] = self.objective_evidence.to_dict()
        return payload


def review_batch_expert_consensus(
    features: FeatureVector,
    context: dict[str, Any],
    decisions: dict[str, DecisionOutput],
) -> ExpertDiversityReview:
    """Record expert consensus without triggering more model calls."""

    del features, decisions
    evidence = context.get("entry_candidate_evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    side = str(evidence.get("preferred_side_by_evidence") or "").lower()
    side_payload = evidence.get(side) if side in {"long", "short"} else None
    side_payload = side_payload if isinstance(side_payload, dict) else {}
    try:
        return_lcb = float(side_payload.get("return_lcb_pct") or 0.0)
    except (TypeError, ValueError):
        return_lcb = 0.0
    objective = ObjectiveEvidence(
        side=side if side in {"long", "short"} else None,
        score=return_lcb,
        reasons=("authoritative_fee_after_return_observation",) if side_payload else (),
    )
    return ExpertDiversityReview(
        should_retry=False,
        reason=(
            "expert opinions are observation-only; additional retries cannot grant "
            "production execution, size, leverage or threshold changes"
        ),
        low_information_consensus=False,
        objective_evidence=objective,
        target_experts=(),
    )
