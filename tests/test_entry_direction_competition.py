from copy import deepcopy
from types import SimpleNamespace

import pytest

from services.entry_direction_competition import EntryDirectionCompetitionPolicy
from services.return_objective import RETURN_OBJECTIVE_NAME, RETURN_OBJECTIVE_VERSION


def _governed_payload(long_return: float, short_return: float) -> dict:
    return {
        "available": True,
        "route_mode": "live",
        "live_influence": True,
        "promotion_ready": True,
        "objective_name": RETURN_OBJECTIVE_NAME,
        "objective_version": RETURN_OBJECTIVE_VERSION,
        "prediction_quality": {
            "production_eligible": True,
            "anomalous": False,
        },
        "long_expected_return_pct": long_return,
        "short_expected_return_pct": short_return,
    }


def _governed_ml(long_return: float, short_return: float) -> dict:
    return {
        **_governed_payload(long_return, short_return),
        "allow_live_position_influence": True,
        "influence_enabled": True,
        "influence_policy": {
            "long": {"enabled": True},
            "short": {"enabled": True},
        },
        "predictions": [
            {
                "long_expected_return_pct": long_return,
                "short_expected_return_pct": short_return,
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
    assert context["long"]["expected_return_pct"] == 0.7
    assert context["short"]["expected_return_pct"] == pytest.approx(-0.15)
    assert context["production_source_count"] == 4
    assert context["production_permission"] is False
    assert context["policy"] == "production_governed_expected_returns_only_no_fixed_gap"


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
