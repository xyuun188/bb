from __future__ import annotations

import pytest

from services.entry_stress_stop import (
    ENTRY_LOW_QUALITY_STRESS_STOP_MIN_PCT,
    ENTRY_STRESS_STOP_MAX_PCT,
    EntryStressStopPolicy,
)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, 0.02),
        ({"declared_stop_loss_pct": 0.03}, 0.03),
        ({"expected_loss_pct": 4.2}, 0.042),
        ({"tail_risk_score": 0.70}, 0.0525),
        ({"atr_pct": 0.03}, 0.048),
        ({"raw_expected_return_pct": -6.0}, 0.039),
        ({"low_payoff_quality": True}, ENTRY_LOW_QUALITY_STRESS_STOP_MIN_PCT),
        (
            {"declared_stop_loss_pct": 0.03, "tail_risk_score": 2.0},
            ENTRY_STRESS_STOP_MAX_PCT,
        ),
    ],
)
def test_entry_stress_stop_policy_matches_sizing_formula(kwargs: dict, expected: float) -> None:
    payload = {
        "declared_stop_loss_pct": 0.02,
        "expected_loss_pct": 0.0,
        "tail_risk_score": 0.0,
        "raw_expected_return_pct": 0.0,
        "low_payoff_quality": False,
    }
    payload.update(kwargs)

    assert EntryStressStopPolicy().stress_stop_loss_pct(**payload) == pytest.approx(expected)


def test_entry_stress_stop_never_goes_below_declared_stop() -> None:
    assert EntryStressStopPolicy().stress_stop_loss_pct(
        declared_stop_loss_pct=0.09,
        expected_loss_pct=0.0,
        tail_risk_score=0.0,
        raw_expected_return_pct=0.0,
        low_payoff_quality=False,
    ) == pytest.approx(ENTRY_STRESS_STOP_MAX_PCT)
