import pytest

from services.memory_feedback import MemoryFeedbackPolicy


def test_memory_feedback_turns_missed_opportunities_into_probe_bias() -> None:
    feedback = MemoryFeedbackPolicy().build(
        [
            {
                "side": "short",
                "memory_type": "shadow_missed_opportunity",
                "confidence_adjustment": 0.04,
                "confidence_score": 0.8,
                "evidence_count": 4,
                "lesson": "Short opportunity was missed after hold.",
            },
            {
                "side": "short",
                "memory_type": "profit_pattern",
                "confidence_adjustment": 0.03,
                "confidence_score": 0.7,
                "evidence_count": 2,
                "lesson": "Similar short pattern was profitable.",
            },
        ]
    )

    short = feedback["by_side"]["short"]
    assert feedback["preferred_side_by_memory"] == "short"
    assert short["allow_probe"] is True
    assert short["action_bias"] == "prefer_small_probe_when_current_ev_positive"
    assert short["candidate_score_bonus"] > 0
    assert short["missed_opportunity_count"] == 4
    habit = feedback["decision_habit"]
    assert habit["posture"] == "selective_probe"
    assert habit["active_probe_sides"] == ["short"]
    assert habit["by_side"]["short"]["stance"] == "probe_when_ev_ok"
    assert habit["by_side"]["short"]["probe_budget_pct"] > 0


def test_memory_feedback_keeps_losing_side_conservative() -> None:
    feedback = MemoryFeedbackPolicy().build(
        [
            {
                "side": "long",
                "memory_type": "loss_lesson",
                "confidence_adjustment": -0.12,
                "confidence_score": 0.85,
                "evidence_count": 4,
                "lesson": "Long entries lost in this setup.",
            },
            {
                "side": "long",
                "memory_type": "shadow_bad_signal",
                "confidence_adjustment": -0.06,
                "confidence_score": 0.75,
                "evidence_count": 2,
                "lesson": "Shadow review marked long as bad.",
            },
        ]
    )

    long_side = feedback["by_side"]["long"]
    assert long_side["allow_probe"] is False
    assert long_side["action_bias"] == "require_stronger_confirmation"
    assert long_side["candidate_score_bonus"] < 0
    assert long_side["risk_evidence_count"] == 6
    habit = feedback["decision_habit"]
    assert habit["posture"] == "defensive_selective"
    assert habit["conservative_sides"] == ["long"]
    assert habit["by_side"]["long"]["stance"] == "strict_confirm"
    assert habit["by_side"]["long"]["max_loss_probability"] < 0.5


def test_memory_feedback_extracts_missed_shadow_returns_as_ev_hint() -> None:
    feedback = MemoryFeedbackPolicy().build(
        [
            {
                "side": "long",
                "memory_type": "shadow_missed_opportunity",
                "confidence_adjustment": 0.05,
                "confidence_score": 0.70,
                "evidence_count": 3,
                "extra": {
                    "best_action": "long",
                    "horizon_minutes": 10,
                    "long_return_pct": 0.72,
                    "short_return_pct": -0.20,
                },
            },
            {
                "side": "long",
                "memory_type": "shadow_missed_opportunity",
                "confidence_adjustment": 0.04,
                "confidence_score": 0.64,
                "evidence_count": 2,
                "extra": {
                    "best_action": "long",
                    "horizon_minutes": 30,
                    "long_return_pct": 0.42,
                    "short_return_pct": -0.10,
                },
            },
        ]
    )

    long_side = feedback["by_side"]["long"]
    assert long_side["missed_return_evidence_count"] == 5
    assert long_side["missed_avg_return_pct"] == pytest.approx(0.6)
    assert 0.0 < long_side["expected_return_hint_pct"] <= long_side["missed_avg_return_pct"]
    habit = feedback["decision_habit"]["by_side"]["long"]
    assert habit["expected_return_hint_pct"] == long_side["expected_return_hint_pct"]
