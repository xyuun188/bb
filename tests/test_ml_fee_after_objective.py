from __future__ import annotations

import math

import pytest

from services.ml_signal_service import _fee_after_prediction_objective, _select_fee_after_side


def test_objective_uses_explicit_model_dispersion_not_quantile_width() -> None:
    result = _fee_after_prediction_objective(
        {
            "expected_net_return_pct": 0.60,
            "return_q10": -0.70,
            "return_q90": 1.90,
            "dispersion_pct": 0.15,
            "tail_loss_probability": 0.10,
        },
        tail_loss_scale_pct=2.0,
    )

    assert result["uncertainty_penalty_pct"] == pytest.approx(0.15)
    assert result["tail_loss_penalty_pct"] == pytest.approx(0.2)
    assert result["objective_net_return_pct"] == pytest.approx(0.25)


def test_objective_does_not_require_upper_quantile_when_model_dispersion_is_known() -> None:
    result = _fee_after_prediction_objective(
        {
            "expected_net_return_pct": 0.60,
            "return_q10": -0.70,
            "dispersion_pct": 0.15,
            "tail_loss_probability": 0.10,
        },
        tail_loss_scale_pct=2.0,
    )

    assert result["uncertainty_penalty_pct"] == pytest.approx(0.15)
    assert result["objective_net_return_pct"] == pytest.approx(0.25)


def test_objective_does_not_invent_model_dispersion_from_tail_quantiles() -> None:
    result = _fee_after_prediction_objective(
        {
            "expected_net_return_pct": 0.60,
            "return_q10": -0.70,
            "return_q90": 1.90,
            "tail_loss_probability": 0.10,
        },
        tail_loss_scale_pct=2.0,
    )
    assert math.isnan(result["objective_net_return_pct"])


def test_objective_is_nan_when_required_inputs_are_missing() -> None:
    result = _fee_after_prediction_objective(
        {"expected_net_return_pct": None, "return_q10": -0.7},
        tail_loss_scale_pct=2.0,
    )

    assert math.isnan(result["objective_net_return_pct"])


@pytest.mark.parametrize("ready", [False, True])
def test_incomplete_objectives_preserve_the_existing_paper_direction_fallback(ready: bool) -> None:
    incomplete = {
        side: _fee_after_prediction_objective({}, tail_loss_scale_pct=0.0)
        for side in ("long", "short")
    }

    assert _select_fee_after_side(
        incomplete,
        net_predictions={
            "long": {"expected_net_return_pct": 0.6},
            "short": {"expected_net_return_pct": 0.1},
        },
        long_ready=ready, short_ready=ready,
    ) == "long"
