from copy import deepcopy
from types import SimpleNamespace

import pytest

from services.entry_direction_competition import EntryDirectionCompetitionPolicy
from services.profit_supervision import PROFIT_SUPERVISION_VERSION
from services.return_objective import (
    COST_MODEL_VERSION,
    RETURN_DISTRIBUTION_CONTRACT_VERSION,
    RETURN_LABEL_NAME,
    RETURN_LABEL_VERSION,
    RETURN_OBJECTIVE_NAME,
    RETURN_OBJECTIVE_VERSION,
    standardized_return_distribution,
)


def _distribution(side: str, expected: float, *, horizon_minutes: int = 30) -> dict:
    return standardized_return_distribution(
        side=side,
        horizon_minutes=horizon_minutes,
        raw_expected_return_pct=expected,
        median_return_pct=expected,
        lower_quantile_return_pct=expected - 0.1,
        upper_quantile_return_pct=expected + 0.1,
        dispersion_pct=0.1,
        tail_loss_probability=0.2 if side == "long" else 0.3,
        tail_loss_scale_pct=0.1,
        distribution_member_count=32,
        return_semantics="gross_market_opportunity_before_execution",
        source_authority="test_tree_empirical_distribution",
        cost_model_version=COST_MODEL_VERSION,
        profit_supervision_version=PROFIT_SUPERVISION_VERSION,
    )


def _governed_payload(long_return: float, short_return: float) -> dict:
    return {
        "available": True,
        "route_mode": "live",
        "live_ml_ready": True,
        "objective_name": RETURN_OBJECTIVE_NAME,
        "objective_version": RETURN_OBJECTIVE_VERSION,
        "label_name": RETURN_LABEL_NAME,
        "label_version": RETURN_LABEL_VERSION,
        "training_cost_policy": "separated_market_opportunity_and_execution_cost_tasks",
        "profit_supervision_version": PROFIT_SUPERVISION_VERSION,
        "return_semantics": "gross_market_opportunity_before_execution",
        "prediction_quality": {
            "production_eligible": True,
            "anomalous": False,
        },
        "best_side": "long" if long_return >= short_return else "short",
        "return_distribution_contract_version": RETURN_DISTRIBUTION_CONTRACT_VERSION,
        "return_distribution_contract": {
            "version": RETURN_DISTRIBUTION_CONTRACT_VERSION,
            "long": _distribution("long", long_return),
            "short": _distribution("short", short_return),
        },
        "long_market_expected_return_pct": long_return,
        "short_market_expected_return_pct": short_return,
    }


def _governed_ml(long_return: float, short_return: float) -> dict:
    return {
        **_governed_payload(long_return, short_return),
        "live_ml_ready": True,
        "influence_enabled": True,
        "influence_policy": {
            "long": {"enabled": True},
            "short": {"enabled": True},
        },
        "predictions": [
            {
                "best_side": "long" if long_return >= short_return else "short",
                "return_distribution_contract_version": (
                    RETURN_DISTRIBUTION_CONTRACT_VERSION
                ),
                "return_distribution_contract": {
                    "version": RETURN_DISTRIBUTION_CONTRACT_VERSION,
                    "long": _distribution("long", long_return),
                    "short": _distribution("short", short_return),
                },
                "long_market_expected_return_pct": long_return,
                "short_market_expected_return_pct": short_return,
            }
        ],
    }


def _context(*, ml=None, tools=None, feature=None, market=None, strategy=None) -> dict:
    return EntryDirectionCompetitionPolicy().context(
        feature or SimpleNamespace(),
        ml,
        tools,
        market,
        strategy,
    )


def test_only_governed_return_models_choose_observed_side() -> None:
    context = _context(
        ml=_governed_ml(0.8, -0.2),
        tools={"profit_prediction": _governed_payload(0.6, -0.1)},
    )

    assert context["preferred_side"] == "long"
    assert context["long"]["raw_expected_return_pct"] == 0.7
    assert context["long"]["objective_expected_return_pct"] == pytest.approx(0.58)
    assert context["short"]["raw_expected_return_pct"] == pytest.approx(-0.15)
    assert context["short"]["objective_expected_return_pct"] == pytest.approx(-0.28)
    assert context["production_source_count"] == 4
    assert context["production_permission"] is False
    assert context["policy"] == "governed_gross_market_observation_only_no_fixed_gap"


def test_missing_governance_cannot_enter_direction_scores() -> None:
    context = _context(
        ml={
            "predictions": [
                {
                    "long_expected_return_pct": 1000.0,
                    "short_expected_return_pct": -1000.0,
                }
            ]
        },
        tools={"profit_prediction": {"long_expected_return_pct": 1000.0}},
    )

    assert context["preferred_side"] == "neutral"
    assert context["production_source_count"] == 0
    assert context["policy_provenance"]["fallback_reason"]


def test_shadow_model_cannot_enter_direction_scores() -> None:
    payload = _governed_payload(1000.0, -1000.0)
    payload["route_mode"] = "shadow"

    context = _context(tools={"profit_prediction": payload})

    assert context["preferred_side"] == "neutral"
    assert context["production_source_count"] == 0


def test_shadow_model_still_exposes_direction_for_paper_training() -> None:
    payload = _governed_ml(-0.4, -0.1)
    payload.update(
        {
            "route_mode": "shadow_observation",
            "live_influence": False,
            "live_ml_ready": False,
            "influence_enabled": False,
            "promotion_ready": False,
        }
    )

    context = _context(ml=payload)

    assert context["preferred_side"] == "neutral"
    assert context["production_source_count"] == 0
    assert context["training_preferred_side"] == "short"
    assert context["training_short"]["observation_count"] == 1
    assert context["training_short"]["horizon_minutes"] == 30
    assert context["training_short"]["horizon_source_count"] == 1
    assert context["training_permission"] is False


def test_diagnostic_win_rate_cannot_change_direction_scores() -> None:
    first_ml = _governed_ml(0.5, 0.2)
    second_ml = deepcopy(first_ml)
    first_ml["predictions"][0]["long_win_rate"] = 0.99
    second_ml["predictions"][0]["long_win_rate"] = 0.01

    first = _context(ml=first_ml)
    second = _context(ml=second_ml)

    assert first["long"]["score"] == second["long"]["score"]
    assert first["preferred_side"] == second["preferred_side"]


def test_features_regime_and_strategy_weights_are_observation_excluded() -> None:
    ml = _governed_ml(0.4, 0.6)
    first = _context(
        ml=ml,
        feature=SimpleNamespace(adx_14=99.0, returns_5=10.0),
        market={"mode": "uptrend"},
        strategy={"side_weights": {"long": 999.0}, "blocked_directions": ["short"]},
    )
    second = _context(
        ml=ml,
        feature=SimpleNamespace(adx_14=0.0, returns_5=-10.0),
        market={"mode": "downtrend"},
        strategy={"side_weights": {"short": 999.0}, "blocked_directions": ["long"]},
    )

    assert first["preferred_side"] == "short"
    assert first["long"]["score"] == second["long"]["score"]
    assert first["short"]["score"] == second["short"]["score"]


def test_paper_continuous_weights_can_change_direction_without_affecting_live() -> None:
    ml = _governed_ml(0.6, 0.0)
    tools = {"profit_prediction": _governed_payload(0.0, 0.5)}
    equal = _context(ml=ml, tools=tools)
    weighted = _context(
        ml=ml,
        tools=tools,
        strategy={
            "execution_mode": "paper",
            "continuous_model_weights": {
                "applied": True,
                "quant_source_weights": {
                    "local_ml": {"effective_multiplier": 0.1},
                    "server_profit": {"effective_multiplier": 1.4},
                },
            },
        },
    )
    live = _context(
        ml=ml,
        tools=tools,
        strategy={
            "execution_mode": "live",
            "continuous_model_weights": {
                "applied": True,
                "quant_source_weights": {
                    "local_ml": {"effective_multiplier": 0.1},
                    "server_profit": {"effective_multiplier": 1.4},
                },
            },
        },
    )

    assert equal["preferred_side"] == "long"
    assert weighted["preferred_side"] == "short"
    assert weighted["continuous_model_weighting"]["applied"] is True
    assert live == equal


def test_mismatched_horizons_cannot_enter_direction_aggregation() -> None:
    payload = _governed_payload(0.6, -0.1)
    for side in ("long", "short"):
        payload["return_distribution_contract"][side]["horizon_minutes"] = 60

    context = _context(
        ml=_governed_ml(0.8, -0.2),
        tools={"profit_prediction": payload},
    )

    assert context["preferred_side"] == "neutral"
    assert context["production_source_count"] == 0
    assert "direction_competition_horizon_minutes_mismatch" in context[
        "aggregate_blockers"
    ]
