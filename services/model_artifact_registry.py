"""Versioned, hash-verified registry for locally trained model artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.model_artifact_safety import dump_trusted_joblib

ARTIFACT_REGISTRY_VERSION = "2026-07-11.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"artifact registry JSON must be an object: {path}")
    return value


@dataclass(frozen=True)
class ResolvedModelArtifact:
    model_id: str
    version: str
    model_path: Path
    metadata_path: Path
    manifest_path: Path
    sha256: str
    manifest: dict[str, Any]


@dataclass(frozen=True)
class ModelArtifactRegistry:
    root: Path
    model_id: str

    @property
    def model_root(self) -> Path:
        return self.root / self.model_id

    @property
    def versions_root(self) -> Path:
        return self.model_root / "versions"

    @property
    def current_path(self) -> Path:
        return self.model_root / "current.json"

    def persist_joblib(
        self,
        bundle: dict[str, Any],
        metadata: dict[str, Any],
        *,
        parent_model_identity: str,
        code_version: str,
    ) -> ResolvedModelArtifact:
        parent_model_identity = parent_model_identity.strip()
        code_version = code_version.strip()
        quality_report = metadata.get("quality_report")
        quality_report = quality_report if isinstance(quality_report, dict) else {}
        training_data_version = str(quality_report.get("data_quality_version") or "").strip()
        sample_cursor = metadata.get("last_trained_completed_shadow_sample_count")
        if not parent_model_identity:
            raise ValueError("parent_model_identity is required")
        if not code_version:
            raise ValueError("code_version is required")
        if not training_data_version:
            raise ValueError("quality_report.data_quality_version is required")
        if sample_cursor is None:
            raise ValueError("last_trained_completed_shadow_sample_count is required")
        if not isinstance(metadata.get("metrics"), dict):
            raise ValueError("metrics are required")

        created_at = datetime.now(UTC)
        version = f"{created_at.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex[:8]}"
        version_root = self.versions_root / version
        model_path = version_root / "model.joblib"
        metadata_path = version_root / "model_metadata.json"
        manifest_path = version_root / "manifest.json"
        version_root.mkdir(parents=True, exist_ok=False)

        registry_metadata = {
            **metadata,
            "artifact_registry_version": ARTIFACT_REGISTRY_VERSION,
            "artifact_model_id": self.model_id,
            "artifact_version": version,
            "artifact_path": str(model_path),
            "artifact_manifest_path": str(manifest_path),
            "parent_model_identity": parent_model_identity,
            "training_data_version": training_data_version,
            "sample_cursor": sample_cursor,
            "code_version": code_version,
        }
        persisted_bundle = dict(bundle)
        persisted_bundle["metadata"] = registry_metadata
        dump_trusted_joblib(persisted_bundle, model_path, trusted_root=version_root)
        artifact_hash = _sha256(model_path)
        registry_metadata["artifact_sha256"] = artifact_hash
        registry_metadata["artifact_size_bytes"] = model_path.stat().st_size
        _write_json_atomic(metadata_path, registry_metadata)

        metadata_hash = _sha256(metadata_path)
        manifest = {
            **registry_metadata,
            "created_at": created_at.isoformat(),
            "metadata_sha256": metadata_hash,
            "model_relative_path": "model.joblib",
            "metadata_relative_path": "model_metadata.json",
        }
        _write_json_atomic(manifest_path, manifest)
        manifest_hash = _sha256(manifest_path)
        _write_json_atomic(
            self.current_path,
            {
                "artifact_registry_version": ARTIFACT_REGISTRY_VERSION,
                "model_id": self.model_id,
                "version": version,
                "manifest_path": str(manifest_path.relative_to(self.model_root)),
                "sha256": artifact_hash,
                "metadata_sha256": metadata_hash,
                "manifest_sha256": manifest_hash,
                "updated_at": created_at.isoformat(),
            },
        )
        return ResolvedModelArtifact(
            model_id=self.model_id,
            version=version,
            model_path=model_path,
            metadata_path=metadata_path,
            manifest_path=manifest_path,
            sha256=artifact_hash,
            manifest=manifest,
        )

    def resolve_current(self) -> ResolvedModelArtifact | None:
        if not self.current_path.exists():
            return None
        pointer = _read_json(self.current_path)
        if pointer.get("artifact_registry_version") != ARTIFACT_REGISTRY_VERSION:
            raise ValueError("unsupported artifact registry pointer version")
        if pointer.get("model_id") != self.model_id:
            raise ValueError("artifact registry pointer model identity mismatch")
        version = str(pointer.get("version") or "").strip()
        manifest_relative = str(pointer.get("manifest_path") or "").strip()
        if not version or not manifest_relative:
            raise ValueError("artifact registry pointer is incomplete")
        manifest_path = (self.model_root / manifest_relative).resolve(strict=True)
        version_root = (self.versions_root / version).resolve(strict=True)
        manifest_path.relative_to(version_root)
        expected_manifest_hash = str(pointer.get("manifest_sha256") or "")
        if not expected_manifest_hash or _sha256(manifest_path) != expected_manifest_hash:
            raise ValueError("artifact manifest hash verification failed")
        manifest = _read_json(manifest_path)
        if manifest.get("artifact_model_id") != self.model_id:
            raise ValueError("artifact manifest model identity mismatch")
        if manifest.get("artifact_version") != version:
            raise ValueError("artifact manifest version mismatch")
        model_path = (version_root / str(manifest.get("model_relative_path") or "")).resolve(
            strict=True
        )
        metadata_path = (
            version_root / str(manifest.get("metadata_relative_path") or "")
        ).resolve(strict=True)
        model_path.relative_to(version_root)
        metadata_path.relative_to(version_root)
        pointer_hash = str(pointer.get("sha256") or "")
        manifest_hash = str(manifest.get("artifact_sha256") or "")
        if not pointer_hash or pointer_hash != manifest_hash:
            raise ValueError("artifact hash evidence mismatch")
        actual_hash = _sha256(model_path)
        if actual_hash != pointer_hash:
            raise ValueError("artifact hash verification failed")
        pointer_metadata_hash = str(pointer.get("metadata_sha256") or "")
        manifest_metadata_hash = str(manifest.get("metadata_sha256") or "")
        if not pointer_metadata_hash or pointer_metadata_hash != manifest_metadata_hash:
            raise ValueError("artifact metadata hash evidence mismatch")
        if _sha256(metadata_path) != pointer_metadata_hash:
            raise ValueError("artifact metadata hash verification failed")
        metadata = _read_json(metadata_path)
        if metadata.get("artifact_model_id") != self.model_id:
            raise ValueError("artifact metadata model identity mismatch")
        if metadata.get("artifact_version") != version:
            raise ValueError("artifact metadata version mismatch")
        if metadata.get("artifact_sha256") != actual_hash:
            raise ValueError("artifact metadata model hash mismatch")
        return ResolvedModelArtifact(
            model_id=self.model_id,
            version=version,
            model_path=model_path,
            metadata_path=metadata_path,
            manifest_path=manifest_path,
            sha256=actual_hash,
            manifest=manifest,
        )

    def status(self) -> dict[str, Any]:
        try:
            current = self.resolve_current()
        except Exception as exc:
            return {
                "available": False,
                "model_id": self.model_id,
                "registry_version": ARTIFACT_REGISTRY_VERSION,
                "current_pointer": str(self.current_path),
                "error": str(exc),
            }
        if current is None:
            return {
                "available": False,
                "model_id": self.model_id,
                "registry_version": ARTIFACT_REGISTRY_VERSION,
                "current_pointer": str(self.current_path),
                "error": "current_artifact_not_registered",
            }
        return {
            "available": True,
            "model_id": self.model_id,
            "registry_version": ARTIFACT_REGISTRY_VERSION,
            "version": current.version,
            "model_path": str(current.model_path),
            "manifest_path": str(current.manifest_path),
            "sha256": current.sha256,
            "manifest": current.manifest,
        }
