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
    expert_count: int = 0
    distinct_action_count: int = 0
    confidence_span: float = 0.0

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

    del features
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
    valid = [decision for decision in decisions.values() if isinstance(decision, DecisionOutput)]
    actions = {decision.action.value for decision in valid}
    confidences = [float(decision.confidence or 0.0) for decision in valid]
    confidence_span = max(confidences) - min(confidences) if confidences else 0.0
    low_information_consensus = bool(
        len(valid) >= 3
        and len(actions) <= 1
        and confidence_span <= 0.05
    )
    return ExpertDiversityReview(
        should_retry=False,
        reason=(
            "low-information expert consensus recorded for audit; no retry can grant "
            "production execution, size, leverage or threshold changes"
            if low_information_consensus
            else "expert opinions are observation-only; additional retries cannot grant "
            "production execution, size, leverage or threshold changes"
        ),
        low_information_consensus=low_information_consensus,
        objective_evidence=objective,
        target_experts=(),
        expert_count=len(valid),
        distinct_action_count=len(actions),
        confidence_span=round(confidence_span, 6),
    )
