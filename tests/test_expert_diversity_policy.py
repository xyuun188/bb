from types import SimpleNamespace

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
