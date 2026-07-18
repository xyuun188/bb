from types import SimpleNamespace

from ai_brain.base_model import Action, DecisionOutput
from ai_brain.expert_diversity_policy import review_batch_expert_consensus


def test_expert_diversity_never_triggers_additional_model_calls() -> None:
    review = review_batch_expert_consensus(
        SimpleNamespace(),
        {
            "entry_candidate_evidence": {
                "preferred_side_by_evidence": "long",
                "long": {"production_eligible": True, "return_lcb_pct": 0.4},
            }
        },
        {},
    )

    assert review.should_retry is False
    assert review.target_experts == ()
    assert review.objective_evidence.side == "long"
    assert review.objective_evidence.score == 0.4


def test_missing_return_evidence_stays_observation_only() -> None:
    review = review_batch_expert_consensus(SimpleNamespace(), {}, {})

    assert review.should_retry is False
    assert review.objective_evidence.side is None


def test_low_information_consensus_is_recorded_without_retry() -> None:
    decisions = {
        name: DecisionOutput(
            model_name=name,
            symbol="BTC/USDT",
            action=Action.HOLD,
            confidence=0.5,
            reasoning=f"{name} neutral",
        )
        for name in ("trend_expert", "momentum_expert", "risk_expert")
    }

    review = review_batch_expert_consensus(SimpleNamespace(), {}, decisions)

    assert review.low_information_consensus is True
    assert review.expert_count == 3
    assert review.distinct_action_count == 1
    assert review.confidence_span == 0.0
    assert review.should_retry is False
