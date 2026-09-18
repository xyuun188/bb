from ai_brain.base_model import Action, DecisionOutput
from services.entry_direction_metrics import selected_entry_metrics
from services.normal_paper_trade import build_normal_paper_trade_contract
from tests.normal_paper_test_fixtures import paper_quality_permissions


def _decision(raw_response: dict) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.LONG,
        confidence=0.7,
        reasoning="test",
        raw_response=raw_response,
    )


def _quality_observation_contract() -> dict:
    permission = paper_quality_permissions()["local_ml"]
    permission.update(
        {
            "paper_execution_permission": False,
            "paper_execution_reason": "average_fee_after_return_not_positive",
            "paper_execution_blockers": ["average_fee_after_return_not_positive"],
            "paper_execution_evidence": {"sample_count": 0},
        }
    )
    return build_normal_paper_trade_contract(
        symbol="BTC/USDT",
        side="long",
        selection_reason="paper_quality_observation",
        direction_support={
            "eligible": True,
            "selected_side": "long",
            "prediction_horizon_minutes": 30.0,
            "expected_net_return_pct": 0.0241367,
            "objective_net_return_pct": -0.686895,
            "loss_probability": 0.4851,
            "quant_evidence_families": ["local_ml"],
            "quant_quality_permissions": {"local_ml": permission},
            "paper_quality_observation_only": True,
            "paper_quality_observation_reasons": [
                "average_fee_after_return_not_positive"
            ],
            "strong_expert_opposition": False,
        },
    )


def test_selected_entry_metrics_uses_selected_side_for_normal_entry() -> None:
    metrics = selected_entry_metrics(
        _decision(
            {
                "opportunity_score": {
                    "expected_net_return_pct": -0.2,
                    "profit_quality_ratio": -0.3,
                    "server_profit_expected_return_pct": -0.1,
                    "server_profit_loss_probability": 0.62,
                    "tail_risk_score": 0.8,
                },
                "entry_candidate_evidence": {
                    "long": {
                        "expected_net_return_pct": 0.9,
                        "profit_quality_ratio": 1.1,
                        "server_profit_expected_return_pct": 0.4,
                        "loss_probability": 0.42,
                        "tail_risk_score": 0.35,
                    }
                },
            }
        )
    )

    assert metrics.source == "entry_candidate_evidence"
    assert metrics.expected_net_return_pct == 0.9
    assert metrics.profit_quality_ratio == 1.1
    assert metrics.loss_probability == 0.42


def test_selected_entry_metrics_ignore_deleted_probe_payload() -> None:
    metrics = selected_entry_metrics(
        _decision(
            {
                "opportunity_score": {
                    "expected_net_return_pct": 0.17,
                    "profit_quality_ratio": 0.22,
                    "server_profit_expected_return_pct": -0.3,
                    "server_profit_loss_probability": 0.57,
                    "tail_risk_score": 0.44,
                },
                "entry_candidate_evidence": {
                    "long": {
                        "expected_net_return_pct": 1.08,
                        "profit_quality_ratio": 1.4,
                        "server_profit_expected_return_pct": 0.5,
                        "loss_probability": 0.41,
                        "tail_risk_score": 0.20,
                    }
                },
                "evidence_profit_probe": {
                    "triggered": True,
                    "ai_original_action": "hold",
                    "side": "long",
                },
                "opinions": [{"model_name": "trend_expert", "action": "hold", "confidence": 0.72}],
            }
        )
    )

    assert metrics.source == "entry_candidate_evidence"
    assert metrics.expected_net_return_pct == 1.08
    assert metrics.profit_quality_ratio == 1.4
    assert metrics.server_profit_expected_return_pct == 0.5
    assert metrics.loss_probability == 0.41
    assert metrics.tail_risk_score == 0.20


def test_selected_entry_metrics_are_unchanged_by_legacy_probe_support_flags() -> None:
    metrics = selected_entry_metrics(
        _decision(
            {
                "opportunity_score": {
                    "expected_net_return_pct": 0.17,
                    "profit_quality_ratio": 0.22,
                    "server_profit_expected_return_pct": -0.3,
                    "server_profit_loss_probability": 0.57,
                    "tail_risk_score": 0.44,
                },
                "entry_candidate_evidence": {
                    "long": {
                        "expected_net_return_pct": 1.08,
                        "profit_quality_ratio": 1.4,
                        "server_profit_expected_return_pct": 0.5,
                        "loss_probability": 0.41,
                        "tail_risk_score": 0.20,
                    }
                },
                "evidence_profit_probe": {
                    "triggered": True,
                    "ai_original_action": "hold",
                    "side": "long",
                },
                "opinions": [
                    {
                        "model_name": "trend_expert",
                        "action": "long",
                        "confidence": 0.72,
                        "independent_expert_retry": True,
                    }
                ],
            }
        )
    )

    assert metrics.source == "entry_candidate_evidence"
    assert metrics.expected_net_return_pct == 1.08
    assert metrics.profit_quality_ratio == 1.4
    assert metrics.server_profit_expected_return_pct == 0.5
    assert metrics.loss_probability == 0.41
    assert metrics.tail_risk_score == 0.20


def test_selected_entry_metrics_prefers_valid_v10_contract_in_paper_mode() -> None:
    metrics = selected_entry_metrics(
        _decision(
            {
                "opportunity_score": {
                    "expected_net_return_pct": -0.1278,
                    "return_lcb_pct": -0.9,
                    "server_profit_loss_probability": 0.0,
                },
                "entry_candidate_evidence": {
                    "long": {
                        "expected_net_return_pct": -0.1278,
                        "return_lcb_pct": -0.9,
                        "loss_probability": 0.0,
                    }
                },
                "normal_paper_trade": _quality_observation_contract(),
            }
        ),
        "paper",
    )

    assert metrics.source == "normal_paper_trade_contract"
    assert metrics.expected_net_return_pct == 0.0241367
    assert metrics.objective_net_return_pct == -0.686895
    assert metrics.loss_probability == 0.4851
    assert metrics.quality_observation is True


def test_selected_entry_metrics_does_not_use_paper_contract_for_live() -> None:
    metrics = selected_entry_metrics(
        _decision(
            {
                "entry_candidate_evidence": {
                    "long": {
                        "expected_net_return_pct": -0.1278,
                        "loss_probability": 0.7,
                    }
                },
                "normal_paper_trade": _quality_observation_contract(),
            }
        ),
        "live",
    )

    assert metrics.source == "entry_candidate_evidence"
    assert metrics.expected_net_return_pct == -0.1278
    assert metrics.loss_probability == 0.7
    assert metrics.quality_observation is False


def test_selected_entry_metrics_rejects_tampered_paper_contract() -> None:
    contract = _quality_observation_contract()
    contract["expected_net_return_pct"] = 9.0
    metrics = selected_entry_metrics(
        _decision(
            {
                "entry_candidate_evidence": {
                    "long": {
                        "expected_net_return_pct": -0.1278,
                        "loss_probability": 0.7,
                    }
                },
                "normal_paper_trade": contract,
            }
        ),
        "paper",
    )

    assert metrics.source == "entry_candidate_evidence"
    assert metrics.expected_net_return_pct == -0.1278
    assert metrics.loss_probability == 0.7
