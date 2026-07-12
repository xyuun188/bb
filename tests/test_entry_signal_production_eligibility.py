from services.entry_signal_extraction import (
    first_tool_payload,
    signal_available,
    signal_production_eligibility,
    signal_production_eligible,
)
from services.return_objective import RETURN_OBJECTIVE_NAME, RETURN_OBJECTIVE_VERSION


def test_shadow_signal_remains_observable_but_cannot_influence_production() -> None:
    payload = {
        "available": True,
        "route_mode": "shadow_candidate",
        "live_mutation": False,
        "best_side": "short",
        "expected_return_pct": 92.0,
    }

    assert signal_available(payload) is True
    assert signal_production_eligible(payload) is False
    assert signal_production_eligibility(payload)["reason"] == "non_production_route_mode"


def test_wrapper_governance_cannot_be_removed_by_payload_unwrapping() -> None:
    payload = {
        "ok": True,
        "route_mode": "live",
        "evaluation_policy": {"live_mutation": False},
        "data": {"prediction": {"best_side": "long", "expected_return_pct": 1.2}},
    }

    result = signal_production_eligibility(payload)

    assert result == {
        "eligible": False,
        "reason": "evaluation_policy_blocks_live_mutation",
    }

    extracted = first_tool_payload({"local_ai_tools": {"profit_prediction": payload}}, "profit_prediction")
    assert signal_production_eligibility(extracted)["reason"] == (
        "evaluation_policy_blocks_live_mutation"
    )


def test_explicit_unpromoted_signal_cannot_use_live_route_label() -> None:
    payload = {
        "available": True,
        "route_mode": "live",
        "live_mutation": True,
        "promotion_ready": False,
    }

    assert signal_production_eligibility(payload)["reason"] == "promotion_not_ready"


def test_live_fee_after_objective_signal_is_production_eligible() -> None:
    payload = {
        "available": True,
        "route_mode": "live",
        "live_mutation": True,
        "artifact_objective": RETURN_OBJECTIVE_NAME,
        "artifact_objective_version": RETURN_OBJECTIVE_VERSION,
        "prediction_quality": {
            "production_eligible": True,
            "anomalous": False,
        },
    }

    assert signal_production_eligibility(payload) == {
        "eligible": True,
        "reason": "governance_allows_live_influence",
    }


def test_dynamic_prediction_quality_block_overrides_live_route() -> None:
    payload = {
        "available": True,
        "route_mode": "live",
        "live_mutation": True,
        "prediction_quality": {
            "production_eligible": False,
            "anomalous": True,
            "reason": "outside_dynamic_rolling_forecast_interval",
        },
    }

    assert signal_production_eligibility(payload)["reason"] == (
        "outside_dynamic_rolling_forecast_interval"
    )
