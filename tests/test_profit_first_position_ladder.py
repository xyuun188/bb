from __future__ import annotations

import pytest

from services.profit_first_position_ladder import ProfitFirstPositionLadderPolicy


def test_meaningful_entry_raises_to_lane_floor() -> None:
    decision = ProfitFirstPositionLadderPolicy().apply(
        lane="meaningful_entry",
        current_size_pct=0.02,
        low_payoff_quality=False,
    )

    assert decision.adjusted_size_pct == pytest.approx(0.05)
    assert decision.raised_to_lane_floor is True
    assert decision.target_min_pct == pytest.approx(0.05)
    assert decision.target_max_pct == pytest.approx(0.08)


def test_low_payoff_cannot_receive_meaningful_size() -> None:
    decision = ProfitFirstPositionLadderPolicy().apply(
        lane="meaningful_entry",
        current_size_pct=0.06,
        low_payoff_quality=True,
    )

    assert decision.adjusted_size_pct == pytest.approx(0.02)
    assert decision.capped_by_low_payoff is True
    assert "low_payoff_cannot_receive_meaningful_size" in decision.reasons


def test_high_conviction_stays_disabled_until_gate_and_review_pass() -> None:
    policy = ProfitFirstPositionLadderPolicy()
    decision = policy.apply(
        lane="high_conviction",
        current_size_pct=0.10,
        low_payoff_quality=False,
        high_risk_review={"approved": True, "profit_first_allow_high_conviction": True},
    )

    assert decision.lane == "meaningful_entry"
    assert decision.adjusted_size_pct == pytest.approx(0.08)
    assert decision.capped_by_high_conviction_gate is True


def test_high_conviction_can_use_upper_range_when_gate_enabled() -> None:
    policy = ProfitFirstPositionLadderPolicy(high_conviction_enabled=True)
    decision = policy.apply(
        lane="high_conviction",
        current_size_pct=0.10,
        low_payoff_quality=False,
        high_risk_review={"approved": True, "profit_first_allow_high_conviction": True},
    )

    assert decision.lane == "high_conviction"
    assert decision.adjusted_size_pct == pytest.approx(0.10)
    assert decision.target_min_pct == pytest.approx(0.08)
    assert decision.target_max_pct == pytest.approx(0.12)
