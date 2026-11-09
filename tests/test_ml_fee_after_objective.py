from __future__ import annotations

import math

import pytest

from services.ml_signal_service import _fee_after_prediction_objective


def test_objective_uses_q90_q10_dispersion_when_present() -> None:
    result = _fee_after_prediction_objective(
        {
            "expected_net_return_pct": 0.60,
            "return_q10": -0.70,
            "return_q90": 1.90,
            "tail_loss_probability": 0.10,
        },
        tail_loss_scale_pct=2.0,
    )

    # sigma = (1.90 - -0.70) / 2.563103131 = 1.014...
    assert result["uncertainty_penalty_pct"] == pytest.approx(1.014, abs=1e-3)
    assert result["tail_loss_penalty_pct"] == pytest.approx(0.2)
    assert result["objective_net_return_pct"] == pytest.approx(-0.614, abs=1e-3)


def test_objective_falls_back_to_mean_to_q10_sigma_without_q90() -> None:
    result = _fee_after_prediction_objective(
        {
            "expected_net_return_pct": 0.60,
            "return_q10": -0.70,
            "tail_loss_probability": 0.10,
        },
        tail_loss_scale_pct=2.0,
    )

    # sigma = (0.60 - -0.70) / 1.281551566 = 1.014...
    assert result["uncertainty_penalty_pct"] == pytest.approx(1.014, abs=1e-3)
    assert result["objective_net_return_pct"] == pytest.approx(-0.614, abs=1e-3)


def test_objective_is_nan_when_required_inputs_are_missing() -> None:
    result = _fee_after_prediction_objective(
        {"expected_net_return_pct": None, "return_q10": -0.7},
        tail_loss_scale_pct=2.0,
    )

    assert math.isnan(result["objective_net_return_pct"])
