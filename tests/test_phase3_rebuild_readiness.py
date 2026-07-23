from __future__ import annotations

from services.phase3_rebuild_readiness import Phase3RebuildReadinessService


def test_phase3_rebuild_readiness_blocks_when_clean_training_inputs_are_weak() -> None:
    report = Phase3RebuildReadinessService().report(
        local_ai_tools={
            "training_shadow_sample_count": 120,
            "training_trade_sample_count": 4,
            "promotion_flow": "candidate_to_shadow_to_canary_to_active",
        },
        governance={"status": "clean", "contamination_risk": "low"},
        historical_trade_fact_audit={
            "status": "dirty",
            "trainable_closed_positions": 0,
            "training_policy": "clean_training_view_only",
        },
        artifact_retirement_audit={"status": "ready", "retired_or_untrusted_count": 0},
        runtime_probe={"status": "ok"},
    )

    assert report["status"] == "blocked"
    assert report["read_only"] is True
    assert report["writes_artifacts"] is False
    assert report["can_run_confirmed_rebuild"] is False
    assert report["can_persist_artifact"] is False
    assert "shadow_sample_floor_not_met" not in report["blockers"]
    assert "trade_sample_floor_not_met" not in report["blockers"]
    assert "no_clean_closed_trade_facts" in report["blockers"]
    assert report["sample_floor"]["distribution_requirement"]
    assert report["target_artifacts"]["ml_signal"]["can_persist_artifact"] is False
    assert report["next_action"] == "run_preflight_until_blockers_clear"


def test_phase3_rebuild_readiness_allows_confirmed_shadow_artifact_write_only() -> None:
    report = Phase3RebuildReadinessService().report(
        local_ai_tools={
            "training_shadow_sample_count": 500,
            "training_trade_sample_count": 80,
            "quality_report": {
                "totals": {
                    "total": 580,
                    "excluded": 0,
                    "effective_weight_ratio": 0.91,
                }
            },
            "promotion_flow": "candidate_to_shadow_to_canary_to_active",
        },
        governance={"status": "clean", "contamination_risk": "low"},
        historical_trade_fact_audit={
            "status": "clean",
            "trainable_closed_positions": 80,
            "training_policy": "clean_training_view_only",
        },
        artifact_retirement_audit={
            "status": "ready_with_retired_legacy",
            "retired_or_untrusted_count": 2,
            "retired_legacy_count": 2,
            "unresolved_artifact_count": 0,
        },
        runtime_probe={"status": "ok"},
        requested_persist_artifact=True,
        confirm_phase3_rebuild=True,
    )

    assert report["status"] == "ready_with_warnings"
    assert report["can_run_confirmed_rebuild"] is True
    assert report["can_persist_artifact"] is True
    assert report["target_artifacts"]["local_ai_tools"]["target_stage"] == "shadow"
    assert report["target_artifacts"]["local_ai_tools"]["can_persist_artifact"] is True
    assert "legacy_artifacts_preserved_read_only" in report["warnings"]
    assert "artifact_write_gate_open_for_confirmed_rebuild" in report["passed_checks"]
    assert report["next_action"] == "confirmed_rebuild_may_persist_shadow_artifacts"


def test_phase3_rebuild_readiness_requires_double_confirmation_for_write() -> None:
    report = Phase3RebuildReadinessService().report(
        local_ai_tools={
            "training_shadow_sample_count": 500,
            "training_trade_sample_count": 80,
            "promotion_flow": "candidate_to_shadow_to_canary_to_active",
        },
        governance={"status": "clean", "contamination_risk": "low"},
        historical_trade_fact_audit={"status": "clean", "trainable_closed_positions": 80},
        artifact_retirement_audit={"status": "ready", "retired_or_untrusted_count": 0},
        runtime_probe={"status": "ok"},
        requested_persist_artifact=True,
        confirm_phase3_rebuild=False,
    )

    assert report["status"] == "blocked"
    assert report["can_run_confirmed_rebuild"] is False
    assert report["can_persist_artifact"] is False
    assert "confirmed_rebuild_required_for_artifact_write" in report["blockers"]


def test_phase3_rebuild_readiness_blocks_untrusted_artifacts() -> None:
    report = Phase3RebuildReadinessService().report(
        local_ai_tools={
            "training_shadow_sample_count": 500,
            "training_trade_sample_count": 80,
            "promotion_flow": "candidate_to_shadow_to_canary_to_active",
        },
        governance={"status": "clean", "contamination_risk": "low"},
        historical_trade_fact_audit={"status": "clean", "trainable_closed_positions": 80},
        artifact_retirement_audit={
            "status": "retired_required",
            "retired_or_untrusted_count": 1,
            "retired_legacy_count": 0,
            "unresolved_artifact_count": 1,
        },
        runtime_probe={"status": "ok"},
    )

    assert report["status"] == "blocked"
    assert "unresolved_or_untrusted_artifacts_block_rebuild" in report["blockers"]


def test_phase3_rebuild_readiness_fails_closed_when_contamination_is_unverified() -> None:
    report = Phase3RebuildReadinessService().report(
        local_ai_tools={
            "training_shadow_sample_count": 500,
            "training_trade_sample_count": 80,
            "promotion_flow": "candidate_to_shadow_to_canary_to_active",
        },
        governance={"status": "quarantined", "contamination_risk": "unknown"},
        historical_trade_fact_audit={"status": "clean", "trainable_closed_positions": 80},
        artifact_retirement_audit={"status": "ready", "retired_or_untrusted_count": 0},
        runtime_probe={"status": "ok"},
    )

    assert report["status"] == "blocked"
    assert "contamination_risk_unverified" in report["blockers"]
