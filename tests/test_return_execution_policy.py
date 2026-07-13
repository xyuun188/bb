from copy import deepcopy

import pytest

from ai_brain.base_model import Action, DecisionOutput
from services.return_execution_policy import apply_production_entry_policy


def _decision() -> DecisionOutput:
    provenance = {
        "source": "test_live_distribution",
        "observation_window": "current_test_round",
        "sample_count": 2,
        "generated_at": "2026-07-12T00:00:00+00:00",
        "strategy_version": "test-v1",
        "fallback_reason": "",
    }
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.LONG,
        confidence=0.8,
        reasoning="test",
        position_size_pct=0.02,
        suggested_leverage=2.0,
        stop_loss_pct=0.02,
        take_profit_pct=0.08,
        raw_response={
            "opportunity_score": {
                "production_eligible": True,
                "policy_provenance": provenance,
                "expected_net_return_pct": 0.8,
                "expected_loss_pct": 0.1,
                "execution_cost": {
                    "total_pct": 0.05,
                    "spread_source": "bid_ask",
                    "production_eligible": True,
                    "policy_provenance": provenance,
                },
                "expected_net_breakdown": {
                    "components": [
                        {
                            "key": "server_profit",
                            "production_eligible": True,
                            "raw_return_pct": 1.2,
                        },
                        {
                            "key": "timeseries",
                            "production_eligible": True,
                            "raw_return_pct": 1.1,
                        },
                    ]
                },
            },
            "profit_risk_sizing": {
                "production_eligible": True,
                "account_balance_usdt": 1000.0,
                "final_notional_usdt": 40.0,
                "max_stop_loss_usdt": 10.0,
                "stress_stop_loss_pct": 0.02,
                "policy_provenance": provenance,
            },
        },
    )


def test_return_policy_derives_size_from_return_lcb_and_account_budget() -> None:
    decision = _decision()

    result = apply_production_entry_policy(decision)

    assert result.eligible is True
    assert result.return_lcb_pct == pytest.approx(0.7)
    assert result.position_size_pct == pytest.approx(0.21875)
    assert decision.position_size_pct == pytest.approx(result.position_size_pct)
    assert result.policy_provenance["sample_count"] == 2
    assert "legacy_score_gate_enabled" not in result.policy_provenance


def test_return_policy_rejects_shadow_only_or_missing_production_observations() -> None:
    decision = _decision()
    components = decision.raw_response["opportunity_score"]["expected_net_breakdown"]["components"]
    for component in components:
        component["production_eligible"] = False

    result = apply_production_entry_policy(decision)

    assert result.eligible is False
    assert "production_return_observations_missing" in result.reason
    assert decision.position_size_pct == 0.0


def test_return_policy_rejects_missing_live_spread_instead_of_using_cost_fallback() -> None:
    decision = _decision()
    decision.raw_response["opportunity_score"]["execution_cost"]["spread_source"] = "missing"

    result = apply_production_entry_policy(decision)

    assert result.eligible is False
    assert "live_spread_observation_missing" in result.reason


def test_return_policy_rejects_observation_only_execution_cost() -> None:
    decision = _decision()
    decision.raw_response["opportunity_score"]["execution_cost"][
        "production_eligible"
    ] = False

    result = apply_production_entry_policy(decision)

    assert result.eligible is False
    assert "execution_cost_distribution_missing" in result.reason


def test_obsolete_evidence_payload_cannot_change_production_adjudication() -> None:
    first = _decision()
    second = deepcopy(first)
    first.raw_response["opportunity_score"]["evidence_score"] = {
        "tier": "blocked",
        "shadow_only": True,
    }
    second.raw_response["opportunity_score"]["evidence_score"] = {
        "tier": "elite",
        "tradeable_probe": True,
    }

    first_result = apply_production_entry_policy(first)
    second_result = apply_production_entry_policy(second)

    assert first_result == second_result
