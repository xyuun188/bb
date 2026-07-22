from __future__ import annotations

import pytest

from services.profit_supervision import PROFIT_SUPERVISION_VERSION
from services.return_objective import (
    COST_MODEL_VERSION,
    RETURN_DISTRIBUTION_CONTRACT_VERSION,
    RETURN_LABEL_VERSION,
    RETURN_OBJECTIVE_VERSION,
    combine_production_return_distribution,
    risk_adjusted_expected_return,
    standardized_return_distribution,
)


def _distribution(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "side": "long",
        "horizon_minutes": 30,
        "raw_expected_return_pct": 0.8,
        "median_return_pct": 0.7,
        "lower_quantile_return_pct": 0.4,
        "upper_quantile_return_pct": 1.1,
        "dispersion_pct": 0.2,
        "tail_loss_probability": 0.1,
        "tail_loss_scale_pct": 0.5,
        "distribution_member_count": 32,
        "return_semantics": "gross_market_opportunity_before_execution",
        "source_authority": "random_forest_tree_empirical_distribution",
        "objective_version": RETURN_OBJECTIVE_VERSION,
        "label_version": RETURN_LABEL_VERSION,
        "cost_model_version": COST_MODEL_VERSION,
        "profit_supervision_version": PROFIT_SUPERVISION_VERSION,
    }
    values.update(overrides)
    return standardized_return_distribution(**values)


def test_standardized_return_distribution_keeps_raw_and_objective_separate() -> None:
    contract = _distribution()

    assert contract["version"] == RETURN_DISTRIBUTION_CONTRACT_VERSION
    assert contract["production_eligible"] is True
    assert contract["raw_expected_return_pct"] == 0.8
    assert contract["lower_quantile_return_pct"] == 0.4
    assert contract["dispersion_pct"] == 0.2
    assert contract["uncertainty_penalty_pct"] == pytest.approx(0.4)
    assert contract["tail_loss_penalty_pct"] == pytest.approx(0.05)
    assert contract["objective_expected_return_pct"] == pytest.approx(0.35)


def test_lower_quantile_above_raw_expected_is_blocked_not_clamped() -> None:
    contract = _distribution(
        raw_expected_return_pct=0.46,
        median_return_pct=0.48,
        lower_quantile_return_pct=0.496,
    )

    assert contract["production_eligible"] is False
    assert "lower_quantile_above_raw_expected" in contract["blockers"]
    assert contract["lower_quantile_return_pct"] == 0.496
    assert contract["raw_expected_return_pct"] == 0.46
    assert contract["objective_expected_return_pct"] is None

    with pytest.raises(ValueError, match="lower_quantile_above_raw_expected"):
        risk_adjusted_expected_return(
            expected_return_pct=0.46,
            lower_quantile_return_pct=0.496,
            tail_loss_probability=0.1,
            tail_loss_scale_pct=0.5,
        )


@pytest.mark.parametrize("expected", (-1.2, -0.1, 0.0, 0.4, 2.0))
@pytest.mark.parametrize("distance", (0.0, 0.01, 0.2, 1.0))
@pytest.mark.parametrize("dispersion", (0.0, 0.01, 0.3))
def test_return_distribution_quantile_and_uncertainty_properties(
    expected: float,
    distance: float,
    dispersion: float,
) -> None:
    lower = expected - distance
    contract = _distribution(
        raw_expected_return_pct=expected,
        median_return_pct=expected,
        lower_quantile_return_pct=lower,
        upper_quantile_return_pct=expected + distance,
        dispersion_pct=dispersion,
        distribution_member_count=1,
    )

    assert contract["production_eligible"] is True
    assert contract["lower_quantile_return_pct"] <= contract["raw_expected_return_pct"]
    assert contract["uncertainty_penalty_pct"] >= 0.0
    if distance > 0.0 or dispersion > 0.0:
        assert contract["uncertainty_penalty_pct"] > 0.0


@pytest.mark.parametrize(
    ("overrides", "blocker"),
    [
        ({"dispersion_pct": -0.01}, "return_dispersion_negative"),
        ({"tail_loss_probability": 1.01}, "tail_loss_probability_out_of_bounds"),
        ({"distribution_member_count": 0}, "distribution_members_missing"),
        ({"horizon_minutes": 0}, "distribution_horizon_missing"),
    ],
)
def test_invalid_distribution_facts_fail_closed(
    overrides: dict[str, object],
    blocker: str,
) -> None:
    contract = _distribution(**overrides)

    assert contract["production_eligible"] is False
    assert blocker in contract["blockers"]


def test_production_distribution_deducts_live_cost_once_and_keeps_transform_audit() -> None:
    contract = combine_production_return_distribution(
        side="long",
        model_contracts=[
            _distribution(raw_expected_return_pct=0.8),
            _distribution(
                raw_expected_return_pct=0.6,
                median_return_pct=0.55,
                lower_quantile_return_pct=0.3,
                upper_quantile_return_pct=0.9,
            ),
        ],
        live_execution_cost_pct=0.1,
        live_slippage_pct=0.01,
        counterfactual_cost_distributions=[
            {
                "expected_pct": 0.08,
                "upper_tail_pct": 0.11,
                "uncertainty_pct": 0.02,
                "distribution_ready": True,
                "source_authority": "shadow_counterfactual_live_microstructure",
            }
        ],
        actual_trade_calibrations=[
            {
                "source_authority": "okx_position_history",
                "side": "long",
                "net_return_after_cost_pct": {
                    "count": 12,
                    "expected": 0.5,
                    "lower_hinge": 0.4,
                },
                "slippage_pct": {
                    "count": 12,
                    "expected": 0.02,
                    "upper_hinge": 0.03,
                },
            }
        ],
        profit_supervision_version=PROFIT_SUPERVISION_VERSION,
        source_authority="test_production_combiner",
    )

    assert contract["production_eligible"] is True
    assert contract["gross_market_distribution"]["raw_expected_return_pct"] == (
        pytest.approx(0.7)
    )
    assert contract["raw_expected_return_pct"] == pytest.approx(0.58)
    assert contract["transformations"]["live_execution_cost_pct"] == 0.1
    assert contract["transformations"][
        "authoritative_slippage_tail_excess_pct"
    ] == pytest.approx(0.02)
    assert contract["transformations"]["cost_deduction_count"] == 1
    assert contract["lower_quantile_return_pct"] <= contract[
        "raw_expected_return_pct"
    ]


def test_production_distribution_blocks_mismatched_model_contract_versions() -> None:
    mismatched = _distribution()
    mismatched["cost_model_version"] = "obsolete-cost-model"

    contract = combine_production_return_distribution(
        side="long",
        model_contracts=[_distribution(), mismatched],
        live_execution_cost_pct=0.1,
        live_slippage_pct=0.01,
        counterfactual_cost_distributions=[],
        actual_trade_calibrations=[],
        profit_supervision_version=PROFIT_SUPERVISION_VERSION,
        source_authority="test_production_combiner",
    )

    assert contract["production_eligible"] is False
    assert "return_distribution_cost_model_version_mismatch" in contract[
        "blockers"
    ]
    assert "model_distribution_cost_model_version_mismatch" in contract[
        "blockers"
    ]


def test_paper_model_weights_change_central_return_without_weakening_tail_guard() -> None:
    contract = combine_production_return_distribution(
        side="long",
        model_contracts=[
            _distribution(raw_expected_return_pct=1.0),
            _distribution(
                raw_expected_return_pct=-0.2,
                median_return_pct=-0.2,
                lower_quantile_return_pct=-0.4,
                upper_quantile_return_pct=0.0,
            ),
        ],
        model_weights=[1.4, 0.1],
        live_execution_cost_pct=0.1,
        live_slippage_pct=0.01,
        counterfactual_cost_distributions=[
            {
                "expected_pct": 0.08,
                "upper_tail_pct": 0.11,
                "uncertainty_pct": 0.02,
                "distribution_ready": True,
                "source_authority": "shadow_counterfactual_live_microstructure",
            }
        ],
        actual_trade_calibrations=[
            {
                "source_authority": "okx_position_history",
                "side": "long",
                "net_return_after_cost_pct": {
                    "count": 12,
                    "expected": 0.5,
                    "lower_hinge": 0.4,
                },
                "slippage_pct": {
                    "count": 12,
                    "expected": 0.02,
                    "upper_hinge": 0.03,
                },
            }
        ],
        profit_supervision_version=PROFIT_SUPERVISION_VERSION,
        source_authority="test_paper_weighted_combiner",
    )

    expected = (1.0 * 1.4 - 0.2 * 0.1) / 1.5
    assert contract["gross_market_distribution"]["raw_expected_return_pct"] == (
        pytest.approx(expected)
    )
    assert contract["gross_market_distribution"]["lower_quantile_return_pct"] == -0.4
    assert contract["model_weighting"]["normalized_eligible_weights"] == pytest.approx(
        [1.4 / 1.5, 0.1 / 1.5]
    )
