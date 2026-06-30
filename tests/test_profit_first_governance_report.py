from __future__ import annotations

from services.profit_first_governance_report import ProfitFirstGovernanceReportService


def _ranking_report() -> dict:
    return {
        "status": "ready",
        "ranking_ready": True,
        "audit_only": True,
        "read_only": True,
        "live_mutation": False,
        "summary": {
            "decision_count": 8,
            "closed_position_count": 3,
            "demote_count": 0,
            "disable_count": 0,
        },
        "policy": {"trade_fact_policy": "okx_confirmed_closed_positions_only"},
        "trade_fact_report": {"policy": "okx_confirmed_closed_positions_only"},
        "brain_recommendations": {
            "brain_output_coverage": {
                "source_weights": True,
                "strategy_weights": True,
                "lane_threshold_recommendations": True,
                "size_promotion_demotion": True,
                "no_entry_threshold_recommendations": True,
                "exit_policy_adjustments": True,
                "shadow_canary_live_decisions": True,
            },
            "no_entry_governance": {
                "window_policy": "rolling_24h_no_entry_governance",
                "sample_count": 8,
                "diagnosis": "system_over_conservative_review",
                "reason_counts": [{"value": "evidence_insufficient", "count": 5}],
                "recommendations": [
                    {
                        "reason": "evidence_insufficient",
                        "recommendation": (
                            "collect shadow samples and require stronger independent source alignment"
                        ),
                    }
                ],
            },
            "losing_exit_governance": {
                "sample_count": 2,
                "attribution_counts": [
                    {"value": "position_too_small_fee_drag", "count": 2}
                ],
                "exit_policy_adjustments": [
                    {
                        "attribution": "position_too_small_fee_drag",
                        "recommendation": (
                            "stop tiny probes in this regime unless expected net profit clears fee drag"
                        ),
                    }
                ],
            },
            "size_promotion_demotion": [
                {
                    "decision_lane": "tiny_probe",
                    "recommendation": (
                        "do_not_continue_tiny_size_when_fee_drag_losses_repeat"
                    ),
                }
            ],
        },
    }


def test_profit_first_governance_report_summarizes_no_entry_and_loss_actions() -> None:
    report = ProfitFirstGovernanceReportService().build_report(
        ranking_report=_ranking_report(),
        hours=24,
        limit=100,
    )

    assert report["report_type"] == "profit_first_governance"
    assert report["status"] == "ready"
    assert report["read_only"] is True
    assert report["live_mutation"] is False
    assert report["can_submit_orders"] is False
    assert report["summary"]["no_entry_diagnosis"] == "system_over_conservative_review"
    assert report["summary"]["losing_exit_sample_count"] == 2
    assert report["missing_brain_outputs"] == []
    assert "review_no_entry_thresholds_against_positive_shadow_outcomes" in (
        report["next_cycle_actions"]
    )
    assert "pause_tiny_probe_repeats_or_raise_quality_floor_for_fee_drag_regime" in (
        report["next_cycle_actions"]
    )
    assert report["policy"]["live_changes_require_go_no_go_and_operator_approval"] is True


def test_profit_first_governance_report_marks_missing_brain_outputs_incomplete() -> None:
    ranking = _ranking_report()
    ranking["brain_recommendations"]["brain_output_coverage"] = {
        "source_weights": True,
        "strategy_weights": True,
    }

    report = ProfitFirstGovernanceReportService().build_report(ranking_report=ranking)

    assert report["status"] == "incomplete"
    assert "exit_policy_adjustments" in report["missing_brain_outputs"]
