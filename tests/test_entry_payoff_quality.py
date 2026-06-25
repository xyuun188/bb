from __future__ import annotations

import pytest

from services.entry_payoff_quality import EntryLowPayoffQualityPolicy


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"score": 0.80, "min_score_required": 0.95}, True),
        ({"expected_net_return_pct": 0.20}, True),
        ({"profit_quality_ratio": 0.40}, True),
        ({"raw_expected_return_pct": -0.01}, True),
        ({"small_win_big_loss_penalty": 0.65}, True),
        ({"hard_contribution_caution": True}, True),
        ({"evidence_score": {"tier": "small"}, "evidence_effective_score": 59.0}, True),
        ({}, False),
    ],
)
def test_entry_low_payoff_quality_policy_flags_defensive_sizing_cases(
    overrides: dict,
    expected: bool,
) -> None:
    payload = {
        "score": 1.20,
        "min_score_required": 0.95,
        "expected_net_return_pct": 0.80,
        "profit_quality_ratio": 0.90,
        "raw_expected_return_pct": 0.10,
        "small_win_big_loss_penalty": 0.0,
        "hard_contribution_caution": False,
        "evidence_score": {},
        "evidence_effective_score": 100.0,
    }
    payload.update(overrides)

    assert EntryLowPayoffQualityPolicy().is_low_payoff(**payload) is expected


def test_entry_low_payoff_quality_policy_returns_stable_reason_codes() -> None:
    payload = {
        "score": 0.80,
        "min_score_required": 0.95,
        "expected_net_return_pct": 0.20,
        "profit_quality_ratio": 0.40,
        "raw_expected_return_pct": -0.01,
        "small_win_big_loss_penalty": 0.65,
        "hard_contribution_caution": True,
        "evidence_score": {"tier": "small"},
        "evidence_effective_score": 59.0,
    }

    assert EntryLowPayoffQualityPolicy().reasons(**payload) == [
        "score_below_required",
        "expected_net_below_min",
        "profit_quality_below_min",
        "raw_expected_return_negative",
        "small_win_big_loss_penalty_high",
        "hard_contribution_caution",
        "evidence_low_payoff_quality",
    ]
