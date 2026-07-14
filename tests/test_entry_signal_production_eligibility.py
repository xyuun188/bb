from services.entry_signal_extraction import (
    first_tool_payload,
    signal_available,
    signal_production_eligibility,
    signal_production_eligible,
)
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


def _distribution(side: str = "long") -> dict:
    return standardized_return_distribution(
        side=side,
        horizon_minutes=30,
        raw_expected_return_pct=0.8,
        median_return_pct=0.7,
        lower_quantile_return_pct=0.4,
        upper_quantile_return_pct=1.1,
        dispersion_pct=0.2,
        tail_loss_probability=0.1,
        tail_loss_scale_pct=0.5,
        distribution_member_count=32,
        return_semantics="gross_market_opportunity_before_execution",
        source_authority="test_tree_empirical_distribution",
        cost_model_version=COST_MODEL_VERSION,
        profit_supervision_version=PROFIT_SUPERVISION_VERSION,
    )


def _contract_fields() -> dict:
    return {
        "best_side": "long",
        "return_distribution_contract_version": RETURN_DISTRIBUTION_CONTRACT_VERSION,
        "return_distribution_contract": {
            "version": RETURN_DISTRIBUTION_CONTRACT_VERSION,
            "long": _distribution("long"),
            "short": _distribution("short"),
        },
    }


def test_shadow_signal_remains_observable_but_cannot_influence_production() -> None:
    payload = {
        **_contract_fields(),
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


def test_live_separated_supervision_signal_is_production_eligible() -> None:
    payload = {
        **_contract_fields(),
        "available": True,
        "route_mode": "live",
        "live_mutation": True,
        "promotion_ready": True,
        "artifact_objective": RETURN_OBJECTIVE_NAME,
        "artifact_objective_version": RETURN_OBJECTIVE_VERSION,
        "artifact_persisted": True,
        "training_cost_policy": "separated_market_opportunity_and_execution_cost_tasks",
        "label_name": RETURN_LABEL_NAME,
        "label_version": RETURN_LABEL_VERSION,
        "profit_supervision_version": PROFIT_SUPERVISION_VERSION,
        "return_semantics": "gross_market_opportunity_before_execution",
        "prediction_quality": {
            "production_eligible": True,
            "anomalous": False,
        },
    }

    result = signal_production_eligibility(payload)
    assert result["eligible"] is True
    assert result["reason"] == (
        "governance_and_return_distribution_allow_live_influence"
    )
    assert result["side"] == "long"


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


def test_complete_training_metadata_cannot_recover_shadow_route_for_production() -> None:
    payload = {
        "available": True,
        "trained": True,
        "route_mode": "shadow_candidate",
        "live_mutation": False,
        "promotion_ready": False,
        "artifact_objective": RETURN_OBJECTIVE_NAME,
        "artifact_objective_version": RETURN_OBJECTIVE_VERSION,
        "artifact_persisted": True,
        "training_cost_policy": "separated_market_opportunity_and_execution_cost_tasks",
        "label_name": RETURN_LABEL_NAME,
        "label_version": RETURN_LABEL_VERSION,
        "profit_supervision_version": PROFIT_SUPERVISION_VERSION,
        "return_semantics": "gross_market_opportunity_before_execution",
        "prediction_quality": {
            "production_eligible": True,
            "anomalous": False,
        },
    }

    assert signal_production_eligibility(payload) == {
        "eligible": False,
        "reason": "non_production_route_mode",
        "route_mode": "shadow_candidate",
    }


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
        **_contract_fields(),
        "available": True,
        "route_mode": "live",
        "live_mutation": True,
        "promotion_ready": True,
        "artifact_objective": RETURN_OBJECTIVE_NAME,
        "artifact_objective_version": RETURN_OBJECTIVE_VERSION,
        "label_name": RETURN_LABEL_NAME,
        "label_version": RETURN_LABEL_VERSION,
        "training_cost_policy": "separated_market_opportunity_and_execution_cost_tasks",
        "profit_supervision_version": PROFIT_SUPERVISION_VERSION,
        "return_semantics": "gross_market_opportunity_before_execution",
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
        "label_name": "return_label",
        "label_version": "return_label_version",
        "training_cost_policy": "separated_cost_policy",
        "profit_supervision_version": "profit_supervision",
        "return_semantics": "gross_market_return_semantics",
        "prediction_quality": "prediction_quality",
    }

    for key, missing_name in removals.items():
        payload = dict(complete)
        payload.pop(key)
        result = signal_production_eligibility(payload)
        assert result["eligible"] is False
        assert missing_name in result["missing_governance"]


def test_live_governance_without_standard_distribution_is_observation_only() -> None:
    payload = {
        "available": True,
        "route_mode": "live",
        "live_mutation": True,
        "promotion_ready": True,
        "best_side": "long",
        "artifact_objective": RETURN_OBJECTIVE_NAME,
        "artifact_objective_version": RETURN_OBJECTIVE_VERSION,
        "label_name": RETURN_LABEL_NAME,
        "label_version": RETURN_LABEL_VERSION,
        "training_cost_policy": "separated_market_opportunity_and_execution_cost_tasks",
        "profit_supervision_version": PROFIT_SUPERVISION_VERSION,
        "return_semantics": "gross_market_opportunity_before_execution",
        "prediction_quality": {
            "production_eligible": True,
            "anomalous": False,
        },
    }

    result = signal_production_eligibility(payload)

    assert result["eligible"] is False
    assert result["reason"] == "return_distribution_contract_missing"
