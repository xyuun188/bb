from __future__ import annotations

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


def _metadata() -> dict:
    return {
        "artifact_policy_id": PHASE3_ARTIFACT_POLICY_ID,
        "artifact_persisted": True,
        "phase": "phase3_model_factory",
        "training_policy": PHASE3_REQUIRED_TRAINING_POLICY,
        "training_mode": "walk_forward",
        "model_stage": "shadow",
        "sample_count": 128,
        "last_trained_completed_shadow_sample_count": 256,
        "quality_report": {"data_quality_version": "2026-07-11.v5"},
        "metrics": {"long_pr_auc": 0.61},
        "evaluation_policy": {
            "promotion_flow": PHASE3_REQUIRED_PROMOTION_FLOW,
            "live_mutation": False,
        },
    }


def test_registry_persists_versioned_hash_verified_artifact(tmp_path) -> None:
    registry = ModelArtifactRegistry(
        root=tmp_path / "model_artifacts",
        model_id="local_ml_profit_quality",
    )

    persisted = registry.persist_joblib(
        {"weights": [1, 2, 3]},
        _metadata(),
        parent_model_identity="sklearn RandomForest/Dummy classifier-regressor pipelines",
        code_version="source-sha256:test",
    )
    resolved = registry.resolve_current()

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
    assert resolved.manifest["code_version"] == "source-sha256:test"
    assert resolved.manifest["metadata_sha256"]
    pointer = json.loads(registry.current_path.read_text(encoding="utf-8"))
    assert pointer["version"] == persisted.version
    assert pointer["sha256"] == persisted.sha256


def test_registry_rejects_tampered_current_artifact(tmp_path) -> None:
    registry = ModelArtifactRegistry(
        root=tmp_path / "model_artifacts",
        model_id="local_ml_profit_quality",
    )
    persisted = registry.persist_joblib(
        {"weights": [1]},
        _metadata(),
        parent_model_identity="sklearn RandomForest/Dummy classifier-regressor pipelines",
        code_version="source-sha256:test",
    )
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
    persisted = registry.persist_joblib(
        {"weights": [1]},
        _metadata(),
        parent_model_identity="sklearn RandomForest/Dummy classifier-regressor pipelines",
        code_version="source-sha256:test",
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
    persisted = registry.persist_joblib(
        {"weights": [1]},
        _metadata(),
        parent_model_identity="sklearn RandomForest/Dummy classifier-regressor pipelines",
        code_version="source-sha256:test",
    )
    persisted.metadata_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="metadata hash verification failed"):
        registry.resolve_current()

    registry = ModelArtifactRegistry(
        root=tmp_path / "second" / "model_artifacts",
        model_id="local_ml_profit_quality",
    )
    persisted = registry.persist_joblib(
        {"weights": [1]},
        _metadata(),
        parent_model_identity="sklearn RandomForest/Dummy classifier-regressor pipelines",
        code_version="source-sha256:test",
    )
    persisted.manifest_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest hash verification failed"):
        registry.resolve_current()


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
