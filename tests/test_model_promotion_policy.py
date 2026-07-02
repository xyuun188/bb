import json

from services.model_promotion_policy import (
    build_profit_first_promotion_report,
    build_phase3_promotion_recommendation,
    load_latest_paper_observation_report,
)


def _healthy_paper_observation() -> dict[str, object]:
    return {
        "status": "healthy",
        "paper_active": True,
        "can_use_for_promotion": True,
        "starts_trading_service": False,
        "submits_orders": False,
        "changes_model_routing": False,
        "blockers": [],
        "warnings": [],
        "checked_at": "2026-06-27T10:00:00+00:00",
    }


def test_phase3_promotion_policy_keeps_shadow_when_samples_or_quality_are_weak() -> None:
    recommendation = build_phase3_promotion_recommendation(
        training_mode="shadow",
        model_stage="shadow",
        quality_report={
            "totals": {
                "total": 50,
                "excluded": 8,
                "effective_weight_ratio": 0.4,
            }
        },
        governance_report={
            "trainable_sample_count": 42,
            "contamination_risk": "high",
        },
        evaluation_policy={"live_mutation": False, "requires_walk_forward": True},
        paper_observation_report=_healthy_paper_observation(),
        completed_shadow_sample_count=50,
        completed_trade_sample_count=5,
    )

    assert recommendation["recommended_stage"] == "degraded"
    assert recommendation["canary_ready"] is False
    assert recommendation["live_ready"] is False
    assert "high_contamination_risk" in recommendation["canary_blocking_reasons"]
    assert "walk_forward_required" in recommendation["live_blocking_reasons"]
    assert recommendation["live_mutation"] is False


def test_phase3_promotion_policy_allows_live_ready_only_after_walk_forward_gate() -> None:
    recommendation = build_phase3_promotion_recommendation(
        training_mode="walk_forward",
        model_stage="live",
        quality_report={
            "totals": {
                "total": 500,
                "excluded": 0,
                "effective_weight_ratio": 0.92,
            }
        },
        governance_report={
            "trainable_sample_count": 500,
            "contamination_risk": "low",
        },
        evaluation_policy={"live_mutation": True, "requires_walk_forward": True},
        paper_observation_report=_healthy_paper_observation(),
        completed_shadow_sample_count=500,
        completed_trade_sample_count=80,
    )

    assert recommendation["recommended_stage"] == "live"
    assert recommendation["canary_ready"] is True
    assert recommendation["live_ready"] is True
    assert recommendation["canary_blocking_reasons"] == []
    assert recommendation["live_blocking_reasons"] == []
    assert recommendation["live_mutation"] is False


def test_phase3_promotion_policy_blocks_when_specialist_shadow_floor_not_met() -> None:
    recommendation = build_phase3_promotion_recommendation(
        training_mode="walk_forward",
        model_stage="live",
        quality_report={
            "totals": {
                "total": 500,
                "excluded": 0,
                "effective_weight_ratio": 0.92,
            },
            "specialist_shadow_models": {
                "time_series_prediction": {
                    "actual_inference_count": 12,
                    "direction_count": 12,
                    "direction_hit_rate": 0.75,
                }
            },
        },
        governance_report={
            "trainable_sample_count": 500,
            "contamination_risk": "low",
        },
        evaluation_policy={"live_mutation": True, "requires_walk_forward": True},
        paper_observation_report=_healthy_paper_observation(),
        completed_shadow_sample_count=500,
        completed_trade_sample_count=80,
    )

    assert recommendation["recommended_stage"] == "shadow"
    assert recommendation["canary_ready"] is False
    assert recommendation["live_ready"] is False
    assert (
        "time_series_prediction_specialist_shadow_sample_floor_not_met"
        in recommendation["canary_blocking_reasons"]
    )
    assert recommendation["specialist_shadow_gate"]["time_series_prediction"][
        "actual_inference_count"
    ] == 12


def test_phase3_promotion_policy_uses_model_key_for_clean_sample_floor() -> None:
    model_key = "time_series_prediction:chronos-2-shadow-primary"
    recommendation = build_phase3_promotion_recommendation(
        training_mode="walk_forward",
        model_stage="live",
        quality_report={
            "totals": {
                "total": 500,
                "excluded": 0,
                "effective_weight_ratio": 0.92,
            },
            "specialist_shadow_models": {
                model_key: {
                    "tool": "time_series_prediction",
                    "model": "chronos-2-shadow-primary",
                    "model_key": model_key,
                    "actual_inference_count": 0,
                    "direction_count": 0,
                    "direction_hit_rate": 0.0,
                    "sequence_too_short_count": 0,
                    "legacy_mixed_shadow_count": 12,
                    "legacy_quarantined_count": 12,
                    "legacy_sequence_too_short_count": 12,
                    "promotion_blockers": [
                        "specialist_shadow_sample_floor_not_met",
                    ],
                }
            },
        },
        governance_report={
            "trainable_sample_count": 500,
            "contamination_risk": "low",
        },
        evaluation_policy={"live_mutation": True, "requires_walk_forward": True},
        paper_observation_report=_healthy_paper_observation(),
        completed_shadow_sample_count=500,
        completed_trade_sample_count=80,
    )

    assert recommendation["recommended_stage"] == "shadow"
    assert recommendation["canary_ready"] is False
    assert recommendation["specialist_shadow_gate"][model_key]["model"] == "chronos-2-shadow-primary"
    assert recommendation["specialist_shadow_gate"][model_key]["actual_inference_count"] == 0
    assert recommendation["specialist_shadow_gate"][model_key]["sequence_too_short_count"] == 0
    assert recommendation["specialist_shadow_gate"][model_key]["legacy_quarantined_count"] == 12
    assert (
        recommendation["specialist_shadow_gate"][model_key]["legacy_sequence_too_short_count"]
        == 12
    )
    assert (
        f"{model_key}_specialist_shadow_sample_floor_not_met"
        in recommendation["canary_blocking_reasons"]
    )
    assert (
        f"{model_key}_legacy_mixed_shadow_result_not_promotable"
        not in recommendation["canary_blocking_reasons"]
    )


def test_phase3_promotion_policy_blocks_specialist_tail_losses() -> None:
    model_key = "time_series_prediction:chronos-2-shadow-primary"
    recommendation = build_phase3_promotion_recommendation(
        training_mode="walk_forward",
        model_stage="live",
        quality_report={
            "totals": {
                "total": 500,
                "excluded": 0,
                "effective_weight_ratio": 0.92,
            },
            "specialist_shadow_models": {
                model_key: {
                    "tool": "time_series_prediction",
                    "model": "chronos-2-shadow-primary",
                    "model_key": model_key,
                    "actual_inference_count": 34,
                    "direction_count": 34,
                    "direction_hit_rate": 0.0,
                    "avg_realized_return_pct": -0.25,
                    "worst_realized_return_pct": -0.25,
                    "false_signal_count": 34,
                    "tail_loss_count": 34,
                    "tail_loss_symbols": [{"symbol": "ACT/USDT", "count": 34}],
                    "worst_samples": [
                        {
                            "symbol": "ACT/USDT",
                            "predicted_side": "long",
                            "actual_best_side": "short",
                            "actual_return_pct": -0.25,
                        }
                    ],
                    "promotion_blockers": [
                        "direction_hit_rate_below_floor",
                        "avg_realized_return_below_floor",
                        "false_signal_loss_exceeds_floor",
                    ],
                }
            },
        },
        governance_report={
            "trainable_sample_count": 500,
            "contamination_risk": "low",
        },
        evaluation_policy={"live_mutation": True, "requires_walk_forward": True},
        paper_observation_report=_healthy_paper_observation(),
        completed_shadow_sample_count=500,
        completed_trade_sample_count=80,
    )

    assert recommendation["recommended_stage"] == "shadow"
    assert recommendation["canary_ready"] is False
    assert recommendation["specialist_shadow_gate"][model_key]["tail_loss_count"] == 34
    assert recommendation["specialist_shadow_gate"][model_key]["worst_samples"][0][
        "symbol"
    ] == "ACT/USDT"
    assert f"{model_key}_avg_realized_return_below_floor" in recommendation[
        "canary_blocking_reasons"
    ]
    assert f"{model_key}_false_signal_loss_exceeds_floor" in recommendation[
        "canary_blocking_reasons"
    ]


def test_phase3_promotion_policy_blocks_canary_until_paper_observation_is_healthy() -> None:
    recommendation = build_phase3_promotion_recommendation(
        training_mode="walk_forward",
        model_stage="live",
        quality_report={
            "totals": {
                "total": 500,
                "excluded": 0,
                "effective_weight_ratio": 0.92,
            }
        },
        governance_report={
            "trainable_sample_count": 500,
            "contamination_risk": "low",
        },
        evaluation_policy={"live_mutation": True, "requires_walk_forward": True},
        paper_observation_report={
            "status": "waiting_for_resume",
            "paper_active": False,
            "can_use_for_promotion": False,
            "starts_trading_service": False,
            "submits_orders": False,
            "changes_model_routing": False,
        },
        completed_shadow_sample_count=500,
        completed_trade_sample_count=80,
    )

    assert recommendation["recommended_stage"] == "shadow"
    assert recommendation["canary_ready"] is False
    assert "paper_observation_not_healthy:waiting_for_resume" in recommendation[
        "canary_blocking_reasons"
    ]
    assert recommendation["paper_observation_gate"]["required"] is True
    assert recommendation["paper_observation_gate"]["can_use_for_promotion"] is False


def test_phase3_promotion_policy_blocks_unsafe_paper_observation_contract() -> None:
    recommendation = build_phase3_promotion_recommendation(
        training_mode="walk_forward",
        model_stage="live",
        quality_report={
            "totals": {
                "total": 500,
                "excluded": 0,
                "effective_weight_ratio": 0.92,
            }
        },
        governance_report={
            "trainable_sample_count": 500,
            "contamination_risk": "low",
        },
        evaluation_policy={"live_mutation": True, "requires_walk_forward": True},
        paper_observation_report={
            **_healthy_paper_observation(),
            "starts_trading_service": True,
        },
        completed_shadow_sample_count=500,
        completed_trade_sample_count=80,
    )

    assert recommendation["canary_ready"] is False
    assert "paper_observation_unsafe_starts_trading" in recommendation[
        "canary_blocking_reasons"
    ]


def test_load_latest_paper_observation_report_reads_local_data_dir(
    tmp_path,
) -> None:
    report_dir = tmp_path / "data" / "phase3_paper_resume_observation_reports"
    report_dir.mkdir(parents=True)
    report = _healthy_paper_observation()
    (report_dir / "latest.json").write_text(json.dumps(report), encoding="utf-8")

    loaded = load_latest_paper_observation_report(root=tmp_path)

    assert loaded["available"] is True
    assert loaded["status"] == "healthy"
    assert loaded["can_use_for_promotion"] is True
    assert loaded["report_path"].endswith("latest.json")


def test_phase3_promotion_policy_blocks_canary_when_profit_first_net_pnl_is_non_positive() -> None:
    profit_first_report = {
        "summary": {"promote_candidate_count": 1},
        "strategy_rankings": [
            {"recommended_stage": "canary", "realized_net_pnl": -1.0, "profit_factor": 0.92}
        ],
        "source_rankings": [{"recommended_stage": "promote"}],
        "runtime_feedback": {
            "profit_acceptance": {
                "window_closed_trade_count": 20,
                "net_pnl": -1.0,
                "profit_factor": 0.92,
            },
            "size_feedback": [
                {"sizing_bias": "quality_entries_can_expand_after_validation"}
            ],
        },
    }
    recommendation = build_phase3_promotion_recommendation(
        training_mode="shadow",
        model_stage="shadow",
        quality_report={
            "totals": {
                "total": 500,
                "excluded": 0,
                "effective_weight_ratio": 0.92,
            }
        },
        governance_report={
            "trainable_sample_count": 500,
            "contamination_risk": "low",
        },
        evaluation_policy={"live_mutation": False, "requires_walk_forward": True},
        paper_observation_report=_healthy_paper_observation(),
        completed_shadow_sample_count=500,
        completed_trade_sample_count=20,
        profit_first_report=profit_first_report,
    )

    assert recommendation["canary_ready"] is False
    assert "profit_first_net_pnl_non_positive" in recommendation["canary_blocking_reasons"]
    assert recommendation["runtime_permissions"]["canary_budget_permission"] == "shadow_only"


def test_phase3_promotion_policy_uses_profit_first_report_to_unlock_canary_permissions() -> None:
    profit_first_report = {
        "summary": {"promote_candidate_count": 1},
        "strategy_rankings": [
            {"recommended_stage": "canary", "realized_net_pnl": 3.4, "profit_factor": 1.4}
        ],
        "source_rankings": [{"recommended_stage": "promote"}],
        "runtime_feedback": {
            "profit_acceptance": {
                "window_closed_trade_count": 20,
                "net_pnl": 3.4,
                "profit_factor": 1.4,
            },
            "size_feedback": [
                {"sizing_bias": "quality_entries_can_expand_after_validation"}
            ],
            "missed_opportunity_feedback": {
                "entry_bias": "expand_quality_entries",
                "missed_positive_shadow_count": 2,
            },
        },
    }
    recommendation = build_phase3_promotion_recommendation(
        training_mode="shadow",
        model_stage="shadow",
        quality_report={
            "totals": {
                "total": 500,
                "excluded": 0,
                "effective_weight_ratio": 0.92,
            }
        },
        governance_report={
            "trainable_sample_count": 500,
            "contamination_risk": "low",
        },
        evaluation_policy={"live_mutation": False, "requires_walk_forward": True},
        paper_observation_report=_healthy_paper_observation(),
        completed_shadow_sample_count=500,
        completed_trade_sample_count=20,
        profit_first_report=profit_first_report,
    )

    assert recommendation["canary_ready"] is True
    assert recommendation["runtime_permissions"]["canary_budget_permission"] == (
        "operator_review_canary_expand"
    )
    assert recommendation["runtime_permissions"]["size_permission"] == (
        "operator_review_canary_expand"
    )


def test_build_profit_first_promotion_report_uses_training_samples_for_runtime_feedback() -> None:
    report = build_profit_first_promotion_report(
        shadow_samples=[
            {
                "symbol": "BTC/USDT",
                "decision_action": "hold",
                "missed_opportunity": True,
                "long_return_pct": 0.12,
                "short_return_pct": -0.03,
            }
        ],
        trade_samples=[
            {
                "id": 7,
                "symbol": "BTC/USDT",
                "side": "long",
                "realized_pnl": 2.5,
                "raw_llm_response": {
                    "profit_first_trade_plan": {
                        "decision_lane": "validated",
                        "model_contributions": [{"source": "local_ml"}],
                    }
                },
            }
        ],
    )

    assert report["evidence_source"] == "phase3_training_samples"
    assert report["summary"]["closed_position_count"] == 1
    assert report["runtime_feedback"]["missed_opportunity_feedback"][
        "missed_positive_shadow_count"
    ] == 1
