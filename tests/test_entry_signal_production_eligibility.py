from services.entry_signal_extraction import (
    first_tool_payload,
    signal_available,
    signal_production_eligibility,
    signal_production_eligible,
    signal_runtime_recovery_eligibility,
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
        "promotion_ready": True,
        "artifact_objective": RETURN_OBJECTIVE_NAME,
        "artifact_objective_version": RETURN_OBJECTIVE_VERSION,
        "artifact_persisted": True,
        "training_cost_policy": "per_sample_live_spread_fee_and_funding_complete",
        "label_name": "net_return_after_cost_pct",
        "label_version": RETURN_OBJECTIVE_VERSION,
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


def test_runtime_recovery_keeps_prediction_quality_and_objective_contracts() -> None:
    payload = {
        "available": True,
        "trained": True,
        "route_mode": "shadow_candidate",
        "live_mutation": False,
        "promotion_ready": False,
        "artifact_objective": RETURN_OBJECTIVE_NAME,
        "artifact_objective_version": RETURN_OBJECTIVE_VERSION,
        "artifact_persisted": True,
        "training_cost_policy": "per_sample_live_spread_fee_and_funding_complete",
        "label_name": "net_return_after_cost_pct",
        "label_version": RETURN_OBJECTIVE_VERSION,
        "prediction_quality": {
            "production_eligible": True,
            "anomalous": False,
        },
    }

    assert signal_runtime_recovery_eligibility(payload) == {
        "eligible": True,
        "reason": "trained_shadow_return_contract_intact",
    }

    anomalous = dict(payload)
    anomalous["prediction_quality"] = {
        "production_eligible": False,
        "anomalous": True,
        "reason": "outside_dynamic_rolling_forecast_interval",
    }
    assert signal_runtime_recovery_eligibility(anomalous)["reason"] == (
        "outside_dynamic_rolling_forecast_interval"
    )

    wrong_objective = dict(payload)
    wrong_objective["artifact_objective_version"] = "legacy-win-rate-objective"
    assert signal_runtime_recovery_eligibility(wrong_objective)["reason"] == (
        "artifact_objective_version_mismatch"
    )

    missing_artifact = dict(payload)
    missing_artifact["artifact_persisted"] = False
    assert signal_runtime_recovery_eligibility(missing_artifact)["reason"] == (
        "runtime_recovery_contract_incomplete"
    )


def test_signal_without_governance_metadata_is_observation_only() -> None:
    payload = {
        "available": True,
        "best_side": "short",
        "expected_return_pct": 99.0,
    }

    result = signal_production_eligibility(payload)

    assert result["eligible"] is False
    assert result["reason"] == "production_governance_incomplete"
    assert "governance_metadata" in result["missing_governance"]


def test_each_required_live_governance_field_is_fail_closed() -> None:
    complete = {
        "available": True,
        "route_mode": "live",
        "live_mutation": True,
        "promotion_ready": True,
        "artifact_objective": RETURN_OBJECTIVE_NAME,
        "artifact_objective_version": RETURN_OBJECTIVE_VERSION,
        "prediction_quality": {
            "production_eligible": True,
            "anomalous": False,
        },
    }
    removals = {
        "route_mode": "live_route",
        "live_mutation": "live_influence",
        "promotion_ready": "promotion_ready",
        "artifact_objective": "return_objective",
        "artifact_objective_version": "return_objective_version",
        "prediction_quality": "prediction_quality",
    }

    for key, missing_name in removals.items():
        payload = dict(complete)
        payload.pop(key)
        result = signal_production_eligibility(payload)
        assert result["eligible"] is False
        assert missing_name in result["missing_governance"]
