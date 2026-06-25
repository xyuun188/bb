from ai_brain.base_model import Action, DecisionOutput
from services.entry_direction_metrics import selected_entry_metrics


def _decision(raw_response: dict) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.LONG,
        confidence=0.7,
        reasoning="test",
        raw_response=raw_response,
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


def test_selected_entry_metrics_conserves_original_hold_probe_without_support() -> None:
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

    assert metrics.source == "entry_candidate_evidence:original_hold_probe_conservative"
    assert metrics.expected_net_return_pct == 0.17
    assert metrics.profit_quality_ratio == 0.22
    assert metrics.server_profit_expected_return_pct == -0.3
    assert metrics.loss_probability == 0.57
    assert metrics.tail_risk_score == 0.44


def test_selected_entry_metrics_keeps_supported_original_hold_probe_side_metrics() -> None:
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
