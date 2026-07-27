from __future__ import annotations

from services.entry_direction_support import assess_directional_entry_support
from services.paper_exploration import paper_exploration_contract_reasons
from tests.legacy_paper_contract_fixtures import build_legacy_paper_exploration_contract


def _contract() -> dict:
    direction_support = assess_directional_entry_support(
        {
            "long": {
                "evidence": [
                    {
                        "source": "local_ml",
                        "raw_expected_return_pct": 0.30,
                        "objective_expected_return_pct": 0.10,
                        "horizon_minutes": 30,
                    }
                ]
            }
        },
        [
            {
                "model_name": "trend_expert",
                "action": "long",
                "reasoning": "trend supports long",
                "effective_weight": 0.2,
                "source_group": "llm:expert",
            },
            {
                "model_name": "momentum_expert",
                "action": "long",
                "reasoning": "momentum supports long",
                "effective_weight": 0.2,
                "source_group": "llm:expert",
            },
            {
                "model_name": "risk_expert",
                "action": "hold",
                "reasoning": "no hard risk",
                "effective_weight": 0.1,
                "source_group": "llm:expert",
            },
        ],
        "long",
    )
    selected = {
        "eligible": True,
        "intervention_scope": "bounded_return_uncertainty",
        "expected_net_return_pct": 0.30,
        "objective_net_return_pct": -0.10,
        "return_lcb_pct": -0.10,
        "lcb_gap_ratio": 1.0 / 3.0,
        "loss_probability": 0.30,
        "tail_risk_score": 0.20,
        "return_source_count": 3,
        "historical_evidence_count": 0,
        "validated_route_evidence_count": 0,
        "reliable_evidence_count": 0,
        "exploration_maturity_source": "cold_start",
        "exploration_maturity_evidence": {},
        "exploration_allocation_multiplier": 1.0,
        "prediction_horizon_minutes": 30.0,
        "valid_for_seconds": 1800.0,
        "feature_opportunity_score": 8.0,
        "information_value_score": 2.4,
        "policy_provenance": {},
    }
    return build_legacy_paper_exploration_contract(
        {
            "paper_exploration": {
                "preferred_side": "long",
                "selected": selected,
                "reason": "bounded_paper_exploration_side_selected",
            }
        },
        symbol="BTC/USDT",
        independent_direction_support=direction_support,
    )


def test_historical_exploration_contract_is_read_only_and_tamper_evident() -> None:
    import services.paper_exploration as module

    contract = _contract()
    assert paper_exploration_contract_reasons(contract) == []
    assert not hasattr(module, "build_paper_exploration_contract")
    assert not hasattr(module, "assess_paper_exploration_entry")
    assert not hasattr(module, "select_paper_exploration_side")

    tampered = {**contract, "daily_sample_quota": 10}
    reasons = paper_exploration_contract_reasons(tampered)
    assert "paper_exploration_sample_quota_forbidden" in reasons
    assert "paper_exploration_contract_fingerprint_mismatch" in reasons
