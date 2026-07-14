from __future__ import annotations

import hashlib
import json

import pytest

from services.artifact_retirement_audit import (
    PHASE3_ARTIFACT_POLICY_ID,
    PHASE3_REQUIRED_PROMOTION_FLOW,
    PHASE3_REQUIRED_TRAINING_POLICY,
    ArtifactRetirementAuditService,
)
from services.ml_signal_service import MLSignalService
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

SOURCE_CODE_SHA256 = "b" * 64
SOURCE_CODE_VERSION = f"source-sha256:{SOURCE_CODE_SHA256}"


def _file_sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_legacy_current_pointer(registry: ModelArtifactRegistry) -> dict:
    version = "legacy-version"
    version_root = registry.versions_root / version
    version_root.mkdir(parents=True)
    model_path = version_root / "model.joblib"
    metadata_path = version_root / "model_metadata.json"
    manifest_path = version_root / "manifest.json"
    model_path.write_bytes(b"legacy-model")
    artifact_hash = _file_sha256(model_path)
    metadata = {
        "artifact_registry_version": "2026-07-11.v1",
        "artifact_model_id": registry.model_id,
        "artifact_version": version,
        "artifact_sha256": artifact_hash,
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    metadata_hash = _file_sha256(metadata_path)
    manifest = {
        **metadata,
        "metadata_sha256": metadata_hash,
        "model_relative_path": "model.joblib",
        "metadata_relative_path": "model_metadata.json",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    pointer = {
        "artifact_registry_version": "2026-07-11.v1",
        "model_id": registry.model_id,
        "version": version,
        "manifest_path": str(manifest_path.relative_to(registry.model_root)),
        "sha256": artifact_hash,
        "metadata_sha256": metadata_hash,
        "manifest_sha256": _file_sha256(manifest_path),
    }
    registry.current_path.write_text(json.dumps(pointer), encoding="utf-8")
    return pointer


def _metadata() -> dict:
    return {
        "artifact_policy_id": PHASE3_ARTIFACT_POLICY_ID,
        "artifact_persisted": True,
        "phase": "phase3_model_factory",
        "training_policy": PHASE3_REQUIRED_TRAINING_POLICY,
        "training_mode": "walk_forward",
        "model_stage": "candidate",
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
        "training_data_sha256": "a" * 64,
        "source_code_sha256": SOURCE_CODE_SHA256,
        "walk_forward_report": {
            "status": "complete",
            "decision_group_disjoint": True,
            "chronological_label_disjoint": True,
            "model_refit_per_fold": True,
            "folds": [{"fold": 1, "decision_group_overlap_count": 0}],
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
        "evaluation_policy": {
            "promotion_flow": PHASE3_REQUIRED_PROMOTION_FLOW,
            "live_mutation": False,
        },
    }


def _shadow_activation(*, reason: str = "actual_trade_calibration_not_ready") -> dict:
    return {
        "activation_stage": "shadow",
        "readiness_state": "degraded",
        "production_influence_authorized": False,
        "blocking_reasons": [reason],
        "evidence_contract": "artifact_integrity_and_return_readiness",
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
    assert current.activation_manifest["production_influence_authorized"] is False
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
    assert registry.rollback_current().version == first.version
    assert registry.resolve_rollback().version == second.version


def test_promotion_retires_known_incompatible_current_pointer(tmp_path) -> None:
    registry = ModelArtifactRegistry(
        root=tmp_path / "model_artifacts",
        model_id="local_ml_profit_quality",
    )
    candidate = registry.persist_candidate_joblib(
        {"weights": [2]},
        _metadata(),
        parent_model_identity="sklearn RandomForest/Dummy classifier-regressor pipelines",
        code_version=SOURCE_CODE_VERSION,
    )
    legacy_pointer = _write_legacy_current_pointer(registry)

    current = registry.promote_candidate(_shadow_activation())

    assert current.version == candidate.version
    assert current.activation_manifest["registry_migration"]["reason"] == (
        "incompatible_artifact_registry_version"
    )
    assert current.activation_manifest["registry_migration"][
        "from_registry_version"
    ] == "2026-07-11.v1"
    retired = [
        path
        for path in registry.retired_pointers_root.glob("current-*.json")
        if not path.name.endswith(".retirement.json")
    ]
    assert len(retired) == 1
    assert json.loads(retired[0].read_text(encoding="utf-8")) == legacy_pointer
    audit = retired[0].with_suffix(".retirement.json")
    assert json.loads(audit.read_text(encoding="utf-8"))["pointer_sha256"]
    assert registry.resolve_rollback() is None


def test_promotion_does_not_retire_unknown_or_corrupt_current_pointer(tmp_path) -> None:
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

    with pytest.raises(ValueError, match="cannot be retired"):
        registry.promote_candidate(_shadow_activation())

    assert json.loads(registry.current_path.read_text(encoding="utf-8")) == unknown_pointer
    assert registry.resolve_candidate() is not None
    assert not registry.retired_pointers_root.exists()

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
    assert not registry.retired_pointers_root.exists()


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


def test_live_activation_requires_bound_return_evidence(tmp_path) -> None:
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

    with pytest.raises(ValueError, match="requires a return evidence report"):
        registry.promote_candidate(
            {
                "activation_stage": "live",
                "readiness_state": "ready",
                "production_influence_authorized": True,
                "blocking_reasons": [],
            }
        )

    readiness = {
        "state": "ready",
        "allow_live_position_influence": True,
        "blocking_reasons": [],
    }
    current = registry.promote_candidate(
        {
            "activation_stage": "live",
            "readiness_state": "ready",
            "production_influence_authorized": True,
            "blocking_reasons": [],
            "return_evidence_report": readiness,
        }
    )

    assert current.activation_manifest["return_evidence_report"] == readiness
    assert registry.status()["activation_manifest"][
        "production_influence_authorized"
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


def test_live_activation_rejects_symbol_unstable_candidate(tmp_path) -> None:
    registry = ModelArtifactRegistry(
        root=tmp_path / "model_artifacts",
        model_id="local_ml_profit_quality",
    )
    metadata = _metadata()
    metadata["walk_forward_report"] = {
        **metadata["walk_forward_report"],
        "stable": False,
    }
    registry.persist_candidate_joblib(
        {"weights": [1]},
        metadata,
        parent_model_identity="sklearn RandomForest/Dummy classifier-regressor pipelines",
        code_version=SOURCE_CODE_VERSION,
    )
    readiness = {
        "state": "ready",
        "allow_live_position_influence": True,
        "blocking_reasons": [],
    }

    with pytest.raises(ValueError, match="stable walk-forward evidence"):
        registry.promote_candidate(
            {
                "activation_stage": "live",
                "readiness_state": "ready",
                "production_influence_authorized": True,
                "blocking_reasons": [],
                "return_evidence_report": readiness,
            }
        )

    assert registry.resolve_current() is None
    assert registry.resolve_candidate() is not None


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
