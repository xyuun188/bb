import pytest

from services.authoritative_trade_outcome import (
    AUTHORITATIVE_TRADE_OUTCOME_AUTHORITY,
    AUTHORITATIVE_TRADE_OUTCOME_VERSION,
)
from services.memory_feedback import MemoryFeedbackPolicy
from services.return_objective import RETURN_OBJECTIVE_NAME, RETURN_OBJECTIVE_VERSION


def _trade_memory(
    *,
    side: str,
    position_id: int,
    net_return_pct: float,
    pnl_usdt: float,
    cost_complete: bool = True,
) -> dict:
    return {
        "side": side,
        "memory_type": "profit_pattern" if pnl_usdt > 0 else "loss_lesson",
        "evidence_count": 1,
        "extra": {
            "source": "authoritative_trade_outcome",
            "source_position_id": position_id,
            "outcome_id": f"ato:{position_id}",
            "outcome_fingerprint": f"fingerprint:{position_id}",
            "outcome_version": AUTHORITATIVE_TRADE_OUTCOME_VERSION,
            "authority_level": AUTHORITATIVE_TRADE_OUTCOME_AUTHORITY,
            "net_return_after_all_cost_pct": net_return_pct,
            "realized_pnl": pnl_usdt,
            "objective": RETURN_OBJECTIVE_NAME,
            "objective_version": RETURN_OBJECTIVE_VERSION,
            "cost_complete": cost_complete,
            "production_evidence_eligible": cost_complete,
        },
    }


def test_non_authoritative_memory_is_rejected() -> None:
    feedback = MemoryFeedbackPolicy().build(
        [
            {
                "side": "short",
                "memory_type": "invalid_memory",
                "extra": {
                    "source": "forged_source",
                    "cost_complete": True,
                    "production_evidence_eligible": True,
                },
            }
        ]
    )

    short = feedback["by_side"]["short"]
    assert feedback["enabled"] is False
    assert short["authoritative_memory_count"] == 0
    assert short["canonical_outcome_count"] == 0
    assert short["candidate_score_bonus"] == 0.0
    assert short["score_adjustment"] == 0.0
    assert feedback["preferred_side_by_memory"] == "neutral"
    assert feedback["decision_habit"]["posture"] == "observation_only"


def test_authoritative_fee_after_loss_ignores_non_authoritative_rows() -> None:
    memories = [
        {
            "side": "short",
            "memory_type": "invalid_memory",
            "extra": {"source": "forged_source", "production_evidence_eligible": True},
        },
        _trade_memory(
            side="short",
            position_id=101,
            net_return_pct=-61.38,
            pnl_usdt=-21.2562,
        ),
    ]

    feedback = MemoryFeedbackPolicy().build(memories)
    short = feedback["by_side"]["short"]

    assert short["canonical_outcome_count"] == 1
    assert short["return_lcb_pct"] == pytest.approx(-61.38)
    assert short["total_realized_net_pnl_usdt"] == pytest.approx(-21.2562)
    assert short["score_adjustment"] == 0.0
    assert short["action_bias"] == "fee_after_observation_only"
    assert feedback["decision_habit"]["by_side"]["short"]["stance"] == (
        "fee_after_observation_only"
    )


def test_positive_authoritative_memory_is_observation_not_probe_permission() -> None:
    feedback = MemoryFeedbackPolicy().build(
        [
            _trade_memory(
                side="long",
                position_id=201,
                net_return_pct=4.0,
                pnl_usdt=12.0,
            )
        ]
    )

    long_side = feedback["by_side"]["long"]
    assert feedback["preferred_side_by_memory"] == "neutral"
    assert long_side["canonical_outcome_count"] == 1
    assert long_side["score_adjustment"] == 0.0
    assert long_side["action_bias"] == "fee_after_observation_only"


def test_cost_incomplete_trade_memory_is_observation_only() -> None:
    feedback = MemoryFeedbackPolicy().build(
        [
            _trade_memory(
                side="long",
                position_id=301,
                net_return_pct=8.0,
                pnl_usdt=20.0,
                cost_complete=False,
            )
        ]
    )

    long_side = feedback["by_side"]["long"]
    assert long_side["canonical_outcome_count"] == 0
    assert feedback["preferred_side_by_memory"] == "neutral"
