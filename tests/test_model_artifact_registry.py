from __future__ import annotations

import json
from dataclasses import replace

import pytest

from services.artifact_retirement_audit import (
    PHASE3_ARTIFACT_POLICY_ID,
    PHASE3_REQUIRED_PROMOTION_FLOW,
    ArtifactRetirementAuditService,
)
from services.ml_signal_service import MLSignalService
from services.ml_training_contract import DECISION_GROUP_PARTITION_VERSION
from services.model_artifact_registry import (
    ARTIFACT_REGISTRY_VERSION,
    ModelArtifactRegistry,
)
from services.profit_supervision import PROFIT_SUPERVISION_VERSION
from services.return_objective import (
    COST_MODEL_VERSION,
    RETURN_LABEL_NAME,
    RETURN_LABEL_VERSION,
    RETURN_OBJECTIVE_NAME,
    RETURN_OBJECTIVE_VERSION,
)
from services.training_epoch import CURRENT_TRAINING_EPOCH_POLICY

SOURCE_CODE_SHA256 = "b" * 64
SOURCE_CODE_VERSION = f"source-sha256:{SOURCE_CODE_SHA256}"


def _metadata() -> dict:
    return {
        "artifact_policy_id": PHASE3_ARTIFACT_POLICY_ID,
        "artifact_persisted": True,
        "phase": "phase3_model_factory",
        "training_policy": CURRENT_TRAINING_EPOCH_POLICY,
        "training_mode": "walk_forward",
        "model_stage": "candidate",
        "promotion_flow": PHASE3_REQUIRED_PROMOTION_FLOW,
        "sample_count": 128,
        "last_trained_completed_shadow_sample_count": 256,
        "quality_report": {"data_quality_version": "2026-07-11.v5"},
        "metrics": {"long_pr_auc": 0.61},
        "objective_name": RETURN_OBJECTIVE_NAME,
        "objective_version": RETURN_OBJECTIVE_VERSION,
        "label_name": RETURN_LABEL_NAME,
        "label_version": RETURN_LABEL_VERSION,
        "cost_model_version": COST_MODEL_VERSION,
        "profit_supervision_version": PROFIT_SUPERVISION_VERSION,
        "evaluation_group_policy": "chronological_disjoint_decision_groups",
        "train_decision_group_count": 64,
        "test_decision_group_count": 64,
        "decision_group_partition": {
            "version": DECISION_GROUP_PARTITION_VERSION,
            "ready": True,
            "reason": "ready",
            "sample_count": 256,
            "decision_group_count": 128,
            "candidate_training_decision_group_count": 64,
            "purged_training_decision_group_count": 0,
            "purged_training_sample_count": 0,
            "train_sample_count": 128,
            "train_decision_group_count": 64,
            "holdout_sample_count": 128,
            "holdout_decision_group_count": 64,
            "minimum_train_sample_count": 16,
            "minimum_train_decision_group_count": 2,
            "holdout_decision_start": "2026-07-14T02:00:00+00:00",
            "training_label_end": "2026-07-14T01:00:00+00:00",
            "decision_group_overlap_count": 0,
            "chronological_label_disjoint": True,
        },
        "training_data_sha256": "a" * 64,
        "source_code_sha256": SOURCE_CODE_SHA256,
        "walk_forward_report": {
            "status": "complete",
            "decision_group_disjoint": True,
            "chronological_label_disjoint": True,
            "model_refit_per_fold": True,
            "folds": [
                {
                    "fold": 1,
                    "decision_group_overlap_count": 0,
                    "sides": {
                        "long": {"promotion_math_ready": True},
                        "short": {"promotion_math_ready": True},
                    },
                },
                {
                    "fold": 2,
                    "decision_group_overlap_count": 0,
                    "sides": {
                        "long": {"promotion_math_ready": True},
                        "short": {"promotion_math_ready": True},
                    },
                },
            ],
            "sides": {
                "long": {
                    "promotion_math_ready": True,
                    "market_regime_stability": {"stable": True},
                },
                "short": {
                    "promotion_math_ready": True,
                    "market_regime_stability": {"stable": True},
                },
            },
            "stable": True,
        },
        "leave_one_symbol_out_report": {
            "long": {"stable": True},
            "short": {"stable": True},
        },
        "oos_return_evaluation": {
            "long": {
                "cvar_10_pct": 0.1,
                "max_drawdown_pct": 0.1,
                "promotion_math_ready": True,
            },
            "short": {
                "cvar_10_pct": 0.1,
                "max_drawdown_pct": 0.1,
                "promotion_math_ready": True,
            },
        },
    }


def _shadow_activation(*, reason: str = "actual_trade_calibration_not_ready") -> dict:
    return {
        "activation_stage": "shadow",
        "readiness_state": "degraded",
        "live_ml_ready": False,
        "blocking_reasons": [reason],
        "evidence_contract": "artifact_integrity_and_return_readiness",
    }


def _production_activation(
    stage: str,
    *,
    state: str = "ready",
    sides: list[str] | None = None,
) -> dict:
    enabled_sides = sides or ["long", "short"]
    readiness = {
        "state": state,
        "live_ml_ready": True,
        "live_enabled_sides": enabled_sides,
        "blocking_reasons": [],
    }
    return {
        "activation_stage": stage,
        "readiness_state": state,
        "live_ml_ready": True,
        "live_enabled_sides": enabled_sides,
        "blocking_reasons": [],
        "return_evidence_report": readiness,
    }


def test_registry_persists_versioned_hash_verified_artifact(tmp_path) -> None:
    registry = ModelArtifactRegistry(
        root=tmp_path / "model_artifacts",
        model_id="local_ml_profit_quality",
    )

    persisted = registry.persist_candidate_joblib(
        {"weights": [1, 2, 3]},
        _metadata(),
        parent_model_identity="sklearn RandomForest/Dummy classifier-regressor pipelines",
        code_version=SOURCE_CODE_VERSION,
    )
    resolved = registry.resolve_candidate()

    assert resolved is not None
    assert resolved.version == persisted.version
    assert resolved.sha256 == persisted.sha256
    assert resolved.model_path.parent.name == persisted.version
    assert resolved.manifest["artifact_registry_version"] == ARTIFACT_REGISTRY_VERSION
    assert resolved.manifest["training_data_version"] == "2026-07-11.v5"
    assert resolved.manifest["sample_cursor"] == 256
    assert resolved.manifest["parent_model_identity"] == (
        "sklearn RandomForest/Dummy classifier-regressor pipelines"
    )
    assert resolved.manifest["code_version"] == SOURCE_CODE_VERSION
    assert resolved.manifest["metadata_sha256"]
    pointer = json.loads(registry.candidate_path.read_text(encoding="utf-8"))
    assert pointer["version"] == persisted.version
    assert pointer["sha256"] == persisted.sha256
    assert pointer["pointer_role"] == "candidate"
    assert registry.resolve_current() is None

    current = registry.promote_candidate(_shadow_activation())

    assert current.version == persisted.version
    assert current.pointer_role == "current"
    assert current.activation_manifest is not None
    assert current.activation_manifest["activation_stage"] == "shadow"
    assert current.activation_manifest["live_ml_ready"] is False
    assert not registry.candidate_path.exists()


def test_registry_rejects_tampered_current_artifact(tmp_path) -> None:
    registry = ModelArtifactRegistry(
        root=tmp_path / "model_artifacts",
        model_id="local_ml_profit_quality",
    )
    persisted = registry.persist_candidate_joblib(
        {"weights": [1]},
        _metadata(),
        parent_model_identity="sklearn RandomForest/Dummy classifier-regressor pipelines",
        code_version=SOURCE_CODE_VERSION,
    )
    registry.promote_candidate(_shadow_activation())
    persisted.model_path.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="hash verification failed"):
        registry.resolve_current()
    assert registry.status()["available"] is False


@pytest.mark.asyncio
async def test_artifact_audit_recognizes_registry_version_as_phase3_compatible(
    tmp_path,
) -> None:
    registry = ModelArtifactRegistry(
        root=tmp_path / "model_artifacts",
        model_id="local_ml_profit_quality",
    )
    persisted = registry.persist_candidate_joblib(
        {"weights": [1]},
        _metadata(),
        parent_model_identity="sklearn RandomForest/Dummy classifier-regressor pipelines",
        code_version=SOURCE_CODE_VERSION,
    )

    report = await ArtifactRetirementAuditService(root=tmp_path).report()
    rows = {row["path"]: row for row in report["artifacts"]}

    assert rows[str(persisted.model_path.resolve())]["classification"] == "phase3_compatible"
    assert rows[str(persisted.model_path.resolve())]["can_influence_live"] is False


def test_registry_rejects_tampered_metadata_and_manifest(tmp_path) -> None:
    registry = ModelArtifactRegistry(
        root=tmp_path / "model_artifacts",
        model_id="local_ml_profit_quality",
    )
    persisted = registry.persist_candidate_joblib(
        {"weights": [1]},
        _metadata(),
        parent_model_identity="sklearn RandomForest/Dummy classifier-regressor pipelines",
        code_version=SOURCE_CODE_VERSION,
    )
    persisted.metadata_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="metadata hash verification failed"):
        registry.resolve_candidate()

    registry = ModelArtifactRegistry(
        root=tmp_path / "second" / "model_artifacts",
        model_id="local_ml_profit_quality",
    )
    persisted = registry.persist_candidate_joblib(
        {"weights": [1]},
        _metadata(),
        parent_model_identity="sklearn RandomForest/Dummy classifier-regressor pipelines",
        code_version=SOURCE_CODE_VERSION,
    )
    persisted.manifest_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest hash verification failed"):
        registry.resolve_candidate()


def test_candidate_does_not_replace_current_until_atomic_promotion(tmp_path) -> None:
    registry = ModelArtifactRegistry(
        root=tmp_path / "model_artifacts",
        model_id="local_ml_profit_quality",
    )
    first = registry.persist_candidate_joblib(
        {"weights": [1]},
        _metadata(),
        parent_model_identity="sklearn RandomForest/Dummy classifier-regressor pipelines",
        code_version=SOURCE_CODE_VERSION,
    )
    registry.promote_candidate(_shadow_activation(reason="first_shadow_activation"))
    second = registry.persist_candidate_joblib(
        {"weights": [2]},
        _metadata(),
        parent_model_identity="sklearn RandomForest/Dummy classifier-regressor pipelines",
        code_version=SOURCE_CODE_VERSION,
    )

    assert registry.resolve_current().version == first.version
    assert registry.resolve_candidate().version == second.version

    registry.promote_candidate(_shadow_activation(reason="second_shadow_activation"))

    assert registry.resolve_current().version == second.version
    assert registry.resolve_rollback().version == first.version
    registry.transition_current(_production_activation("canary"))
    registry.transition_current(_production_activation("active"))
    assert registry.resolve_current().version == second.version
    assert registry.resolve_current().activation_manifest["activation_stage"] == "active"
    assert registry.resolve_rollback().version == first.version
    assert registry.rollback_current().version == first.version
    assert registry.resolve_rollback().version == second.version


def test_current_without_partition_contract_is_replaced_without_rollback_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    registry = ModelArtifactRegistry(
        root=tmp_path / "model_artifacts",
        model_id="local_ml_profit_quality",
    )
    first = registry.persist_candidate_joblib(
        {"weights": [1]},
        _metadata(),
        parent_model_identity="sklearn RandomForest/Dummy classifier-regressor pipelines",
        code_version=SOURCE_CODE_VERSION,
    )
    first = registry.promote_candidate(_shadow_activation())
    second = registry.persist_candidate_joblib(
        {"weights": [2]},
        _metadata(),
        parent_model_identity="sklearn RandomForest/Dummy classifier-regressor pipelines",
        code_version=SOURCE_CODE_VERSION,
    )
    invalid_current = replace(
        first,
        manifest={**first.manifest, "decision_group_partition": None},
    )
    original_resolve_current = ModelArtifactRegistry.resolve_current
    calls = 0

    def resolve_current_once_as_invalid(self):
        nonlocal calls
        assert self is registry
        calls += 1
        return invalid_current if calls == 1 else original_resolve_current(self)

    monkeypatch.setattr(
        ModelArtifactRegistry,
        "resolve_current",
        resolve_current_once_as_invalid,
    )

    current = registry.promote_candidate(_shadow_activation())

    assert current.version == second.version
    assert registry.resolve_rollback() is None
    assert first.model_path.exists()


def test_rollback_rejects_artifact_without_current_partition_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    registry = ModelArtifactRegistry(
        root=tmp_path / "model_artifacts",
        model_id="local_ml_profit_quality",
    )
    artifact = registry.persist_candidate_joblib(
        {"weights": [1]},
        _metadata(),
        parent_model_identity="sklearn RandomForest/Dummy classifier-regressor pipelines",
        code_version=SOURCE_CODE_VERSION,
    )
    invalid = replace(
        artifact,
        manifest={**artifact.manifest, "decision_group_partition": None},
    )
    monkeypatch.setattr(
        ModelArtifactRegistry,
        "resolve_rollback",
        lambda self: invalid if self is registry else None,
    )

    with pytest.raises(ValueError, match="rollback artifact violates"):
        registry.rollback_current()


def test_rejected_candidate_is_preserved_as_challenger_without_replacing_champion(
    tmp_path,
) -> None:
    registry = ModelArtifactRegistry(
        root=tmp_path / "model_artifacts",
        model_id="local_ml_profit_quality",
    )
    first = registry.persist_candidate_joblib(
        {"weights": [1]},
        _metadata(),
        parent_model_identity="sklearn RandomForest/Dummy classifier-regressor pipelines",
        code_version=SOURCE_CODE_VERSION,
    )
    registry.promote_candidate(_shadow_activation())
    second_metadata = _metadata()
    second_metadata["last_trained_completed_shadow_sample_count"] = 512
    second = registry.persist_candidate_joblib(
        {"weights": [2]},
        second_metadata,
        parent_model_identity="sklearn RandomForest/Dummy classifier-regressor pipelines",
        code_version=SOURCE_CODE_VERSION,
    )

    challenger = registry.reject_candidate(
        {
            "accepted": False,
            "reason": "active_champion_retained",
            "blocking_reasons": ["candidate_primary_fee_after_metrics_not_improved"],
        }
    )

    assert first.version != second.version
    assert registry.resolve_current().version == first.version
    assert challenger.version == second.version
    assert challenger.pointer_role == "challenger"
    assert challenger.rejection_manifest["comparison_report"]["accepted"] is False
    assert registry.resolve_candidate() is None
    assert registry.status()["pointers"]["challenger"]["available"] is True
    cursor_metadata = MLSignalService(
        artifact_registry=registry
    )._training_cursor_metadata(registry.resolve_current().manifest)
    assert cursor_metadata["artifact_version"] == second.version
    assert cursor_metadata["last_trained_completed_shadow_sample_count"] == 512


def test_promotion_rejects_old_or_corrupt_current_pointer(tmp_path) -> None:
    registry = ModelArtifactRegistry(
        root=tmp_path / "model_artifacts",
        model_id="local_ml_profit_quality",
    )
    registry.persist_candidate_joblib(
        {"weights": [2]},
        _metadata(),
        parent_model_identity="sklearn RandomForest/Dummy classifier-regressor pipelines",
        code_version=SOURCE_CODE_VERSION,
    )
    unknown_pointer = {
        "artifact_registry_version": "unknown-version",
        "pointer_role": "current",
        "model_id": "local_ml_profit_quality",
    }
    registry.current_path.write_text(json.dumps(unknown_pointer), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported artifact registry pointer version"):
        registry.promote_candidate(_shadow_activation())

    assert json.loads(registry.current_path.read_text(encoding="utf-8")) == unknown_pointer
    assert registry.resolve_candidate() is not None

    corrupt_pointer = {
        "artifact_registry_version": ARTIFACT_REGISTRY_VERSION,
        "pointer_role": "current",
        "model_id": "local_ml_profit_quality",
    }
    registry.current_path.write_text(json.dumps(corrupt_pointer), encoding="utf-8")
    with pytest.raises(ValueError, match="version is required"):
        registry.promote_candidate(_shadow_activation())

    assert json.loads(registry.current_path.read_text(encoding="utf-8")) == corrupt_pointer
    assert registry.resolve_candidate() is not None


def test_registry_rejects_tampered_activation_manifest(tmp_path) -> None:
    registry = ModelArtifactRegistry(
        root=tmp_path / "model_artifacts",
        model_id="local_ml_profit_quality",
    )
    registry.persist_candidate_joblib(
        {"weights": [1]},
        _metadata(),
        parent_model_identity="sklearn RandomForest/Dummy classifier-regressor pipelines",
        code_version=SOURCE_CODE_VERSION,
    )
    current = registry.promote_candidate(_shadow_activation())
    pointer = json.loads(registry.current_path.read_text(encoding="utf-8"))
    activation_path = registry.model_root / pointer["activation_manifest_path"]
    activation_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="activation manifest hash verification failed"):
        registry.resolve_current()
    assert current.activation_manifest is not None


def test_active_activation_requires_bound_return_evidence(tmp_path) -> None:
    registry = ModelArtifactRegistry(
        root=tmp_path / "model_artifacts",
        model_id="local_ml_profit_quality",
    )
    registry.persist_candidate_joblib(
        {"weights": [1]},
        _metadata(),
        parent_model_identity="sklearn RandomForest/Dummy classifier-regressor pipelines",
        code_version=SOURCE_CODE_VERSION,
    )

    registry.promote_candidate(_shadow_activation())
    registry.transition_current(_production_activation("canary"))

    with pytest.raises(ValueError, match="requires a return evidence report"):
        registry.transition_current(
            {
                "activation_stage": "active",
                "readiness_state": "ready",
                "live_ml_ready": True,
                "blocking_reasons": [],
            }
        )

    current = registry.transition_current(_production_activation("active"))

    assert current.activation_manifest["return_evidence_report"]["state"] == "ready"
    assert current.activation_manifest["activation_stage"] == "active"
    assert registry.resolve_active().sha256 == current.sha256
    assert registry.active_path == registry.current_path
    assert registry.status()["activation_manifest"][
        "live_ml_ready"
    ] is True


def test_candidate_rejects_invalid_training_or_source_fingerprint(tmp_path) -> None:
    registry = ModelArtifactRegistry(
        root=tmp_path / "model_artifacts",
        model_id="local_ml_profit_quality",
    )
    invalid_training = _metadata()
    invalid_training["training_data_sha256"] = "not-a-hash"

    with pytest.raises(ValueError, match="training_data_sha256"):
        registry.persist_candidate_joblib(
            {"weights": [1]},
            invalid_training,
            parent_model_identity="sklearn RandomForest/Dummy classifier-regressor pipelines",
            code_version=SOURCE_CODE_VERSION,
        )

    with pytest.raises(ValueError, match="does not match source_code_sha256"):
        registry.persist_candidate_joblib(
            {"weights": [1]},
            _metadata(),
            parent_model_identity="sklearn RandomForest/Dummy classifier-regressor pipelines",
            code_version=f"source-sha256:{'c' * 64}",
        )


def test_candidate_rejects_old_partition_metadata_without_compatibility(tmp_path) -> None:
    registry = ModelArtifactRegistry(
        root=tmp_path / "model_artifacts",
        model_id="local_ml_profit_quality",
    )
    metadata = _metadata()
    metadata.pop("decision_group_partition")

    with pytest.raises(ValueError, match="decision_group_partition is required"):
        registry.persist_candidate_joblib(
            {"weights": [1]},
            metadata,
            parent_model_identity="sklearn RandomForest/Dummy classifier-regressor pipelines",
            code_version=SOURCE_CODE_VERSION,
        )


def test_active_activation_rejects_symbol_unstable_candidate(tmp_path) -> None:
    registry = ModelArtifactRegistry(
        root=tmp_path / "model_artifacts",
        model_id="local_ml_profit_quality",
    )
    metadata = _metadata()
    metadata["walk_forward_report"]["sides"]["short"][
        "promotion_math_ready"
    ] = False
    registry.persist_candidate_joblib(
        {"weights": [1]},
        metadata,
        parent_model_identity="sklearn RandomForest/Dummy classifier-regressor pipelines",
        code_version=SOURCE_CODE_VERSION,
    )
    registry.promote_candidate(_shadow_activation())
    registry.transition_current(
        _production_activation("canary", state="partial_ready", sides=["long"])
    )

    with pytest.raises(ValueError, match="stable walk-forward evidence"):
        registry.transition_current(_production_activation("active"))

    assert registry.resolve_current().activation_manifest["activation_stage"] == "canary"
    assert registry.resolve_candidate() is None


def test_canary_activation_validates_only_partial_ready_live_side(tmp_path) -> None:
    registry = ModelArtifactRegistry(
        root=tmp_path / "model_artifacts",
        model_id="local_ml_profit_quality",
    )
    metadata = _metadata()
    metadata["walk_forward_report"]["sides"]["short"][
        "promotion_math_ready"
    ] = False
    metadata["walk_forward_report"]["folds"][0]["sides"]["short"][
        "promotion_math_ready"
    ] = False
    metadata["leave_one_symbol_out_report"]["short"]["stable"] = False
    metadata["oos_return_evaluation"]["short"]["promotion_math_ready"] = False
    registry.persist_candidate_joblib(
        {"weights": [1]},
        metadata,
        parent_model_identity="sklearn RandomForest/Dummy classifier-regressor pipelines",
        code_version=SOURCE_CODE_VERSION,
    )
    readiness = {
        "state": "partial_ready",
        "live_ml_ready": True,
        "live_enabled_sides": ["long"],
        "blocking_reasons": [],
        "side_blocking_reasons": {
            "long": [],
            "short": [{"code": "short_walk_forward_return_stability_failed"}],
        },
    }

    registry.promote_candidate(_shadow_activation())
    current = registry.transition_current(
        {
            "activation_stage": "canary",
            "readiness_state": "partial_ready",
            "live_ml_ready": True,
            "live_enabled_sides": ["long"],
            "blocking_reasons": [],
            "return_evidence_report": readiness,
        }
    )

    assert current.activation_manifest["activation_stage"] == "canary"
    assert current.activation_manifest["live_enabled_sides"] == ["long"]
    assert current.activation_manifest["live_ml_ready"] is True


def test_paper_canary_activation_does_not_grant_production_permission(tmp_path) -> None:
    registry = ModelArtifactRegistry(
        root=tmp_path / "model_artifacts",
        model_id="local_ml_profit_quality",
    )
    registry.persist_candidate_joblib(
        {"weights": [1]},
        _metadata(),
        parent_model_identity="sklearn RandomForest/Dummy classifier-regressor pipelines",
        code_version=SOURCE_CODE_VERSION,
    )
    paper_report = {
        "state": "ready",
        "authorized": True,
        "execution_scope": "paper_only",
        "production_permission": False,
        "eligible_sides": ["long", "short"],
        "blocking_reasons": [],
    }
    registry.promote_candidate(_shadow_activation())
    current = registry.transition_current(
        {
            "activation_stage": "canary",
            "readiness_state": "paper_canary_ready",
            "live_ml_ready": False,
            "paper_canary_authorized": True,
            "blocking_reasons": [],
            "paper_canary_report": paper_report,
            "return_evidence_report": {},
        }
    )

    activation = current.activation_manifest
    assert activation["activation_stage"] == "canary"
    assert activation["paper_canary_authorized"] is True
    assert activation["live_ml_ready"] is False


def test_default_loader_does_not_fall_back_to_retired_legacy_artifact(tmp_path) -> None:
    legacy_root = tmp_path / "ml_signal"
    legacy_root.mkdir()
    (legacy_root / "winrate_model.joblib").write_bytes(b"retired")
    registry = ModelArtifactRegistry(
        root=tmp_path / "model_artifacts",
        model_id="local_ml_profit_quality",
    )

    status = MLSignalService(artifact_registry=registry).status()

    assert status["available"] is False
    assert status["artifact_registry"]["error"] == "current_artifact_not_registered"
