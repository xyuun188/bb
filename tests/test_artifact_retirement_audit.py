from __future__ import annotations

import json

import pytest

from services.artifact_retirement_audit import (
    PHASE3_ARTIFACT_POLICY_ID,
    ArtifactRetirementAuditService,
)


@pytest.mark.asyncio
async def test_legacy_ml_signal_artifact_is_preserved_but_retired(tmp_path) -> None:
    ml_signal_dir = tmp_path / "ml_signal"
    ml_signal_dir.mkdir()
    artifact = ml_signal_dir / "winrate_model.joblib"
    metadata = ml_signal_dir / "winrate_model_metadata.json"
    artifact.write_bytes(b"legacy-model")
    metadata.write_text(
        json.dumps(
            {
                "trained_at": "2026-06-18T19:11:46+00:00",
                "sample_count": 260,
                "artifact_persisted": True,
                "quality_report": {"data_quality_version": "2026-06-19.v1"},
            }
        ),
        encoding="utf-8",
    )

    report = await ArtifactRetirementAuditService(root=tmp_path).report()

    assert artifact.exists()
    assert metadata.exists()
    assert report["status"] == "ready_with_retired_legacy"
    assert report["retired_legacy_count"] == 2
    assert report["unresolved_artifact_count"] == 0
    assert report["read_only"] is True
    assert report["can_delete_artifacts"] is False
    assert report["retired_or_untrusted_count"] == 2
    rows = {item["relative_path"]: item for item in report["artifacts"]}
    assert rows["ml_signal/winrate_model.joblib"]["classification"] == "retired_legacy"
    assert rows["ml_signal/winrate_model.joblib"]["preserved"] is True
    assert rows["ml_signal/winrate_model.joblib"]["can_influence_live"] is False
    assert "known_legacy_artifact_path" in rows["ml_signal/winrate_model.joblib"]["reasons"]
    assert "artifact_policy_id_not_phase3" in rows["ml_signal/winrate_model.joblib"]["reasons"]


@pytest.mark.asyncio
async def test_phase3_artifact_with_clean_manifest_is_compatible(tmp_path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    artifact = model_dir / "phase3_profit_model.joblib"
    metadata = model_dir / "phase3_profit_model_metadata.json"
    artifact.write_bytes(b"phase3-model")
    metadata.write_text(
        json.dumps(
            {
                "artifact_policy_id": PHASE3_ARTIFACT_POLICY_ID,
                "phase": "phase3_model_factory",
                "trade_sample_cursor_policy": "clean_training_view_only",
                "training_mode": "walk_forward",
                "model_stage": "canary",
                "artifact_persisted": True,
                "evaluation_policy": {
                "promotion_flow": "candidate_to_shadow_to_canary_to_active",
                    "live_mutation": False,
                },
                "quality_report": {"data_quality_version": "2026-06-27.phase3"},
            }
        ),
        encoding="utf-8",
    )

    report = await ArtifactRetirementAuditService(root=tmp_path).report()

    assert report["status"] == "ready"
    assert report["phase3_compatible_count"] == 2
    assert report["retired_or_untrusted_count"] == 0
    rows = {item["relative_path"]: item for item in report["artifacts"]}
    assert rows["models/phase3_profit_model.joblib"]["classification"] == "phase3_compatible"
    assert rows["models/phase3_profit_model.joblib"]["can_influence_live"] is True
    assert rows["models/phase3_profit_model.joblib"]["phase3_evidence"]["training_mode"] == "walk_forward"


@pytest.mark.asyncio
async def test_unknown_artifact_without_manifest_requires_retirement(tmp_path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    artifact = model_dir / "mystery_model.joblib"
    artifact.write_bytes(b"unknown")

    report = await ArtifactRetirementAuditService(root=tmp_path).report()

    assert artifact.exists()
    assert report["status"] == "retired_required"
    assert report["status_counts"] == {"missing_manifest": 1}
    sample = report["retired_or_untrusted_samples"][0]
    assert sample["relative_path"] == "models/mystery_model.joblib"
    assert sample["classification"] == "missing_manifest"
    assert "missing_phase3_metadata" in sample["reasons"]
    assert sample["can_delete"] is False


@pytest.mark.asyncio
async def test_unreferenced_registry_version_is_retired_without_hiding_unknown_artifacts(
    tmp_path,
) -> None:
    model_root = tmp_path / "model_artifacts" / "local_ml_profit_quality"
    active_version = "20260716T051245774293Z-f783c19b"
    orphan_version = "20260712T090034389927Z-3cd65b9c"
    active_root = model_root / "versions" / active_version
    orphan_root = model_root / "versions" / orphan_version
    active_root.mkdir(parents=True)
    orphan_root.mkdir(parents=True)
    active_manifest = active_root / "manifest.json"
    active_manifest.write_text(
        json.dumps(
            {
                "artifact_policy_id": PHASE3_ARTIFACT_POLICY_ID,
                "artifact_model_id": "local_ml_profit_quality",
                "artifact_version": active_version,
                "phase": "phase3_model_factory",
                "training_policy": "clean_training_view_only",
                "model_stage": "shadow",
                "artifact_persisted": True,
                "promotion_flow": "candidate_to_shadow_to_canary_to_active",
                "quality_report": {"data_quality_version": "2026-07-16.v1"},
            }
        ),
        encoding="utf-8",
    )
    (model_root / "current.json").write_text(
        json.dumps(
            {
                "model_id": "local_ml_profit_quality",
                "pointer_role": "current",
                "version": active_version,
                "manifest_path": f"versions/{active_version}/manifest.json",
            }
        ),
        encoding="utf-8",
    )
    orphan = orphan_root / "model.joblib"
    orphan.write_bytes(b"pre-registry-orphan")

    report = await ArtifactRetirementAuditService(root=tmp_path).report()

    assert orphan.exists()
    assert report["status"] == "ready_with_retired_legacy"
    assert report["unresolved_artifact_count"] == 0
    assert report["retired_unreferenced_count"] == 1
    rows = {item["relative_path"]: item for item in report["artifacts"]}
    orphan_row = rows[
        f"model_artifacts/local_ml_profit_quality/versions/{orphan_version}/model.joblib"
    ]
    assert orphan_row["classification"] == "retired_unreferenced"
    assert "unreferenced_registry_version" in orphan_row["reasons"]
    assert orphan_row["can_influence_live"] is False


@pytest.mark.asyncio
async def test_registry_orphan_stays_blocked_when_current_pointer_is_not_valid(tmp_path) -> None:
    model_root = tmp_path / "model_artifacts" / "local_ml_profit_quality"
    orphan_version = "20260712T090034389927Z-3cd65b9c"
    orphan_root = model_root / "versions" / orphan_version
    orphan_root.mkdir(parents=True)
    (orphan_root / "model.joblib").write_bytes(b"unknown")
    (model_root / "current.json").write_text(
        json.dumps(
            {
                "model_id": "local_ml_profit_quality",
                "pointer_role": "current",
                "version": "20260716T051245774293Z-f783c19b",
                "manifest_path": "versions/missing/manifest.json",
            }
        ),
        encoding="utf-8",
    )

    report = await ArtifactRetirementAuditService(root=tmp_path).report()

    assert report["status"] == "retired_required"
    assert report["unresolved_artifact_count"] == 1
    assert report["retired_unreferenced_count"] == 0
