from __future__ import annotations

from types import SimpleNamespace

from services.profit_first_brain_training import ProfitFirstBrainTrainingService


def _raw(lane: str = "validated_probe") -> dict:
    return {
        "profit_first_trade_plan": {
            "plan_version": "profit-first-v3.1",
            "symbol": "BTC/USDT",
            "side": "long",
            "action": "long",
            "strategy_profile_id": "balanced_probe",
            "decision_lane": lane,
            "profit_first_score": 2.4,
            "expected_net_return_pct": 0.8,
            "loss_probability": 0.38,
            "tail_loss_probability": 0.62,
            "position_size_pct": 0.04,
            "exit_plan_id": "pfep-btc",
            "model_sources": ["decision_llm", "server_profit"],
            "model_contributions": [
                {"source": "decision_llm", "field_path": "decision.model_name"},
                {
                    "source": "server_profit",
                    "field_path": "opportunity_score.expected_net_return_pct",
                },
            ],
        }
    }


def test_profit_first_brain_training_builds_dataset_and_recommendations() -> None:
    decisions = [
        SimpleNamespace(
            id=1,
            symbol="BTC/USDT",
            action="long",
            was_executed=True,
            raw_llm_response=_raw(),
        ),
        SimpleNamespace(
            id=2,
            symbol="ETH/USDT",
            action="long",
            was_executed=False,
            raw_llm_response={
                "profit_first_trade_plan": {
                    "decision_lane": "shadow_only",
                    "no_entry_reason": "profit_insufficient",
                    "missing_required_fields": [],
                }
            },
        ),
    ]
    positions = [
        SimpleNamespace(
            id=10,
            model_name="ensemble_trader",
            symbol="BTC/USDT",
            side="long",
            realized_pnl=2.0,
            entry_raw=_raw(),
        ),
        SimpleNamespace(
            id=11,
            model_name="ensemble_trader",
            symbol="ETH/USDT",
            side="short",
            realized_pnl=-1.0,
            entry_raw=_raw("tiny_probe"),
        ),
    ]

    result = ProfitFirstBrainTrainingService(min_canary_samples=1).build_dataset(
        decisions=decisions,
        closed_positions=positions,
    )

    assert result["audit_only"] is True
    assert result["live_mutation"] is False
    assert result["dataset"]["entry_plan_count"] == 2
    assert result["dataset"]["no_entry_count"] == 1
    assert result["dataset"]["losing_exit_count"] == 1
    assert result["dataset"]["model_contribution_count"] == 4
    assert result["leaderboard"]["count"] == 2
    assert result["recommendations"]["source_weights"][0]["source"] in {
        "decision_llm",
        "server_profit",
    }
    assert result["recommendations"]["brain_output_coverage"] == {
        "source_weights": True,
        "strategy_weights": True,
        "lane_threshold_recommendations": True,
        "size_promotion_demotion": True,
        "no_entry_threshold_recommendations": True,
        "exit_policy_adjustments": True,
        "shadow_canary_live_decisions": True,
    }
    assert result["recommendations"]["requires_operator_resume_gate"] is True


def test_profit_first_brain_training_maps_skips_and_loss_causes_to_adjustments() -> None:
    decisions = [
        SimpleNamespace(
            id=20,
            symbol="BTC/USDT",
            action="long",
            was_executed=False,
            raw_llm_response={
                "skip_kind": "profit_first_probe_loss_brake",
                "profit_first_trade_plan": {
                    "decision_lane": "tiny_probe",
                    "missing_required_fields": [],
                },
            },
        ),
        SimpleNamespace(
            id=21,
            symbol="ETH/USDT",
            action="long",
            was_executed=False,
            raw_llm_response={
                "skip_kind": "entry_evidence_wait",
                "shadow_outcome": {"shadow_return_pct": 0.7},
                "profit_first_trade_plan": {
                    "decision_lane": "shadow_only",
                    "missing_required_fields": [],
                },
            },
        ),
    ]
    positions = [
        SimpleNamespace(
            id=30,
            model_name="ensemble_trader",
            symbol="BTC/USDT",
            side="long",
            realized_pnl=-0.2,
            position_size_pct=0.01,
            entry_raw=_raw("tiny_probe"),
        )
    ]

    result = ProfitFirstBrainTrainingService().build_dataset(
        decisions=decisions,
        closed_positions=positions,
    )

    no_entry = result["recommendations"]["no_entry_governance"]
    assert {row["value"] for row in no_entry["reason_counts"]} >= {
        "recent_realized_edge_negative",
        "evidence_insufficient",
    }
    assert no_entry["missed_positive_shadow_count"] == 1

    exit_adjustments = result["recommendations"]["exit_policy_adjustments"]
    assert exit_adjustments[0]["attribution"] == "position_too_small_fee_drag"
    assert "tiny probes" in exit_adjustments[0]["recommendation"]

    size_actions = result["recommendations"]["size_promotion_demotion"]
    assert any(
        row.get("recommendation") == "do_not_continue_tiny_size_when_fee_drag_losses_repeat"
        for row in size_actions
    )
