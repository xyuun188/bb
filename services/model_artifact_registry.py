"""Atomic candidate, current, and rollback registry for local model artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.model_artifact_safety import dump_trusted_joblib, load_trusted_joblib

ARTIFACT_REGISTRY_VERSION = "2026-07-15.v2"
ARTIFACT_ACTIVATION_MANIFEST_VERSION = "2026-07-15.v1"
_WRITABLE_ACTIVATION_STAGES = frozenset({"shadow", "canary", "active"})
_READABLE_ACTIVATION_STAGES = frozenset(
    {*_WRITABLE_ACTIVATION_STAGES, "live"}
)
_ALLOWED_STAGE_TRANSITIONS = {
    "shadow": frozenset({"canary"}),
    "canary": frozenset({"active"}),
    # Existing manifests used `live` for the unified production artifact.
    "live": frozenset({"active"}),
}
_MIGRATABLE_REGISTRY_VERSIONS = frozenset({"2026-07-11.v1"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"immutable artifact manifest already exists: {path}")
    _write_json_atomic(path, payload)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"artifact registry JSON must be an object: {path}")
    return value


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError(f"{key} is required")
    return value


def _is_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _payload_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ResolvedModelArtifact:
    model_id: str
    version: str
    model_path: Path
    metadata_path: Path
    manifest_path: Path
    sha256: str
    manifest: dict[str, Any]
    pointer_role: str
    pointer_path: Path
    activation_manifest: dict[str, Any] | None = None
    rejection_manifest: dict[str, Any] | None = None


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
    def candidate_path(self) -> Path:
        return self.model_root / "candidate.json"

    @property
    def challenger_path(self) -> Path:
        return self.model_root / "challenger.json"

    @property
    def current_path(self) -> Path:
        return self.model_root / "current.json"

    @property
    def active_path(self) -> Path:
        """Canonical unified runtime pointer; shares the legacy current.json file."""

        return self.current_path

    @property
    def rollback_path(self) -> Path:
        return self.model_root / "rollback.json"

    @property
    def retired_pointers_root(self) -> Path:
        return self.model_root / "retired_pointers"

    def persist_candidate_joblib(
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
        self._validate_candidate_training_contract(metadata, code_version)

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
            "artifact_lifecycle": "candidate",
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
        _write_json_once(metadata_path, registry_metadata)

        metadata_hash = _sha256(metadata_path)
        manifest = {
            **registry_metadata,
            "created_at": created_at.isoformat(),
            "metadata_sha256": metadata_hash,
            "model_relative_path": "model.joblib",
            "metadata_relative_path": "model_metadata.json",
        }
        _write_json_once(manifest_path, manifest)
        manifest_hash = _sha256(manifest_path)
        pointer = {
            "artifact_registry_version": ARTIFACT_REGISTRY_VERSION,
            "pointer_role": "candidate",
            "model_id": self.model_id,
            "version": version,
            "manifest_path": str(manifest_path.relative_to(self.model_root)),
            "sha256": artifact_hash,
            "metadata_sha256": metadata_hash,
            "manifest_sha256": manifest_hash,
            "updated_at": created_at.isoformat(),
        }
        _write_json_atomic(self.candidate_path, pointer)
        return self.resolve_candidate(required=True)

    def resolve_candidate(self, *, required: bool = False) -> ResolvedModelArtifact | None:
        return self._resolve_pointer(
            self.candidate_path,
            expected_role="candidate",
            required=required,
        )

    def resolve_current(self) -> ResolvedModelArtifact | None:
        return self._resolve_pointer(
            self.current_path,
            expected_role="current",
            required=False,
        )

    def resolve_challenger(self) -> ResolvedModelArtifact | None:
        return self._resolve_pointer(
            self.challenger_path,
            expected_role="challenger",
            required=False,
        )

    def resolve_active(self) -> ResolvedModelArtifact | None:
        """Resolve the one artifact used by both paper and live execution."""

        return self.resolve_current()

    def resolve_rollback(self) -> ResolvedModelArtifact | None:
        return self._resolve_pointer(
            self.rollback_path,
            expected_role="rollback",
            required=False,
        )

    def promote_candidate(
        self,
        activation_evidence: dict[str, Any],
    ) -> ResolvedModelArtifact:
        if str(activation_evidence.get("activation_stage") or "") != "shadow":
            raise ValueError("candidate artifact must first activate as shadow")
        candidate = self.resolve_candidate(required=True)
        assert candidate is not None
        previous_pointer = _read_json(self.current_path) if self.current_path.exists() else None
        previous: ResolvedModelArtifact | None = None
        retirement: dict[str, Any] | None = None
        if previous_pointer is not None:
            previous_registry_version = str(
                previous_pointer.get("artifact_registry_version") or ""
            )
            if previous_registry_version != ARTIFACT_REGISTRY_VERSION:
                if previous_registry_version not in _MIGRATABLE_REGISTRY_VERSIONS:
                    raise ValueError(
                        "unsupported current registry version cannot be retired"
                    )
                self._validate_migratable_current_pointer(previous_pointer)
                retired_at = datetime.now(UTC)
                retirement_path = self.retired_pointers_root / (
                    f"current-{retired_at.strftime('%Y%m%dT%H%M%S%fZ')}-"
                    f"{uuid.uuid4().hex[:8]}.json"
                )
                retirement = {
                    "reason": "incompatible_artifact_registry_version",
                    "from_registry_version": previous_registry_version or None,
                    "to_registry_version": ARTIFACT_REGISTRY_VERSION,
                    "pointer_sha256": _payload_sha256(previous_pointer),
                    "retired_pointer_path": str(retirement_path),
                    "retired_at": retired_at.isoformat(),
                }
            else:
                previous = self.resolve_current()
                if previous is None:
                    raise ValueError("current artifact pointer disappeared during promotion")
        effective_evidence = dict(activation_evidence)
        if retirement is not None:
            effective_evidence["registry_migration"] = retirement
        activation = self._build_activation_manifest(candidate, effective_evidence)
        activation_root = candidate.manifest_path.parent / "activations"
        activation_path = activation_root / f"a-{uuid.uuid4().hex[:8]}.json"
        _write_json_once(activation_path, activation)
        activation_hash = _sha256(activation_path)

        if retirement is not None:
            retirement_path = Path(str(retirement["retired_pointer_path"]))
            retirement_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(self.current_path, retirement_path)
            _write_json_once(
                retirement_path.with_suffix(".retirement.json"),
                retirement,
            )
        elif previous_pointer is not None and previous is not None:
            _write_json_atomic(
                self.rollback_path,
                {
                    **previous_pointer,
                    "pointer_role": "rollback",
                    "updated_at": datetime.now(UTC).isoformat(),
                },
            )
            self.resolve_rollback()

        candidate_pointer = _read_json(self.candidate_path)
        _write_json_atomic(
            self.current_path,
            {
                **candidate_pointer,
                "pointer_role": "current",
                "activation_manifest_path": str(
                    activation_path.relative_to(self.model_root)
                ),
                "activation_manifest_sha256": activation_hash,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        current = self.resolve_current()
        if current is None:
            raise ValueError("promoted current artifact is unavailable")
        self.candidate_path.unlink(missing_ok=True)
        return current

    def transition_current(
        self,
        activation_evidence: dict[str, Any],
    ) -> ResolvedModelArtifact:
        """Advance the unified artifact without replacing its version rollback."""

        current = self.resolve_current()
        if current is None or not isinstance(current.activation_manifest, dict):
            raise ValueError("current artifact is not registered for transition")
        current_stage = str(current.activation_manifest.get("activation_stage") or "")
        target_stage = _required_text(activation_evidence, "activation_stage")
        allowed_targets = _ALLOWED_STAGE_TRANSITIONS.get(current_stage, frozenset())
        if target_stage not in allowed_targets:
            raise ValueError(
                f"artifact stage transition {current_stage or 'unregistered'} -> "
                f"{target_stage} is not allowed"
            )
        evidence = {
            **activation_evidence,
            "transition_from_stage": current_stage,
        }
        activation = self._build_activation_manifest(current, evidence)
        activation_root = current.manifest_path.parent / "activations"
        activation_path = activation_root / f"a-{uuid.uuid4().hex[:8]}.json"
        _write_json_once(activation_path, activation)
        activation_hash = _sha256(activation_path)
        current_pointer = _read_json(self.current_path)
        _write_json_atomic(
            self.current_path,
            {
                **current_pointer,
                "activation_manifest_path": str(
                    activation_path.relative_to(self.model_root)
                ),
                "activation_manifest_sha256": activation_hash,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        transitioned = self.resolve_current()
        if transitioned is None:
            raise ValueError("artifact stage transition did not preserve current artifact")
        return transitioned

    def reject_candidate(
        self,
        comparison_report: dict[str, Any],
    ) -> ResolvedModelArtifact:
        """Preserve an evaluated challenger without changing the champion pointer."""

        candidate = self.resolve_candidate(required=True)
        assert candidate is not None
        rejected_at = datetime.now(UTC)
        rejection = {
            "artifact_registry_version": ARTIFACT_REGISTRY_VERSION,
            "model_id": self.model_id,
            "version": candidate.version,
            "artifact_sha256": candidate.sha256,
            "comparison_report": comparison_report,
            "rejected_at": rejected_at.isoformat(),
        }
        rejection_path = candidate.manifest_path.parent / "rejections" / (
            f"r-{uuid.uuid4().hex[:8]}.json"
        )
        _write_json_once(rejection_path, rejection)
        rejection_hash = _sha256(rejection_path)
        candidate_pointer = _read_json(self.candidate_path)
        _write_json_atomic(
            self.challenger_path,
            {
                **candidate_pointer,
                "pointer_role": "challenger",
                "rejection_manifest_path": str(
                    rejection_path.relative_to(self.model_root)
                ),
                "rejection_manifest_sha256": rejection_hash,
                "updated_at": rejected_at.isoformat(),
            },
        )
        challenger = self.resolve_challenger()
        if challenger is None:
            raise ValueError("rejected challenger artifact is unavailable")
        self.candidate_path.unlink(missing_ok=True)
        return challenger

    def rollback_current(self) -> ResolvedModelArtifact:
        rollback = self.resolve_rollback()
        if rollback is None:
            raise ValueError("rollback artifact is not registered")
        current_pointer = _read_json(self.current_path) if self.current_path.exists() else None
        rollback_pointer = _read_json(self.rollback_path)
        _write_json_atomic(
            self.current_path,
            {
                **rollback_pointer,
                "pointer_role": "current",
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        restored = self.resolve_current()
        if restored is None:
            raise ValueError("rollback activation did not produce a current artifact")
        if current_pointer is not None:
            _write_json_atomic(
                self.rollback_path,
                {
                    **current_pointer,
                    "pointer_role": "rollback",
                    "updated_at": datetime.now(UTC).isoformat(),
                },
            )
            self.resolve_rollback()
        return restored

    @staticmethod
    def _validate_candidate_training_contract(
        metadata: dict[str, Any],
        code_version: str,
    ) -> None:
        for field in (
            "objective_name",
            "objective_version",
            "label_name",
            "label_version",
            "cost_model_version",
            "profit_supervision_version",
            "training_data_sha256",
            "source_code_sha256",
            "evaluation_group_policy",
            "model_stage",
        ):
            _required_text(metadata, field)
        if metadata.get("model_stage") != "candidate":
            raise ValueError("candidate artifact model_stage must be candidate")
        training_hash = metadata.get("training_data_sha256")
        source_hash = metadata.get("source_code_sha256")
        if not _is_sha256(training_hash):
            raise ValueError("training_data_sha256 must be a SHA-256 digest")
        if not _is_sha256(source_hash):
            raise ValueError("source_code_sha256 must be a SHA-256 digest")
        if code_version != f"source-sha256:{source_hash}":
            raise ValueError("code_version does not match source_code_sha256")
        for field in ("train_decision_group_count", "test_decision_group_count"):
            try:
                value = int(metadata.get(field) or 0)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field} must be an integer") from exc
            if value <= 0:
                raise ValueError(f"{field} must describe a non-empty decision partition")
        walk_forward = metadata.get("walk_forward_report")
        if not isinstance(walk_forward, dict) or (
            walk_forward.get("status") != "complete"
            or walk_forward.get("decision_group_disjoint") is not True
            or walk_forward.get("chronological_label_disjoint") is not True
            or walk_forward.get("model_refit_per_fold") is not True
            or not list(walk_forward.get("folds") or [])
        ):
            raise ValueError("walk_forward_report is incomplete")
        loso = metadata.get("leave_one_symbol_out_report")
        if not isinstance(loso, dict) or any(
            not isinstance(loso.get(side), dict) for side in ("long", "short")
        ):
            raise ValueError("leave_one_symbol_out_report is incomplete")
        oos = metadata.get("oos_return_evaluation")
        if not isinstance(oos, dict) or any(
            not isinstance(oos.get(side), dict)
            or "cvar_10_pct" not in oos[side]
            or "max_drawdown_pct" not in oos[side]
            for side in ("long", "short")
        ):
            raise ValueError("oos_return_evaluation is incomplete")

    def _validate_migratable_current_pointer(
        self,
        pointer: dict[str, Any],
    ) -> None:
        registry_version = _required_text(pointer, "artifact_registry_version")
        if registry_version not in _MIGRATABLE_REGISTRY_VERSIONS:
            raise ValueError("unsupported current registry version cannot be retired")
        if pointer.get("model_id") != self.model_id:
            raise ValueError("incompatible current pointer model identity mismatch")
        if pointer.get("pointer_role") not in (None, "current"):
            raise ValueError("incompatible current pointer role mismatch")
        version = _required_text(pointer, "version")
        version_root = (self.versions_root / version).resolve(strict=True)
        manifest_path = (
            self.model_root / _required_text(pointer, "manifest_path")
        ).resolve(strict=True)
        manifest_path.relative_to(version_root)
        manifest_hash = _required_text(pointer, "manifest_sha256")
        if not _is_sha256(manifest_hash) or _sha256(manifest_path) != manifest_hash:
            raise ValueError("incompatible current manifest hash verification failed")
        manifest = _read_json(manifest_path)
        if manifest.get("artifact_registry_version") != registry_version:
            raise ValueError("incompatible current manifest registry mismatch")
        if manifest.get("artifact_model_id") != self.model_id:
            raise ValueError("incompatible current manifest model identity mismatch")
        if manifest.get("artifact_version") != version:
            raise ValueError("incompatible current manifest version mismatch")
        model_path = (
            version_root / _required_text(manifest, "model_relative_path")
        ).resolve(strict=True)
        metadata_path = (
            version_root / _required_text(manifest, "metadata_relative_path")
        ).resolve(strict=True)
        model_path.relative_to(version_root)
        metadata_path.relative_to(version_root)
        artifact_hash = _required_text(pointer, "sha256")
        if (
            not _is_sha256(artifact_hash)
            or manifest.get("artifact_sha256") != artifact_hash
            or _sha256(model_path) != artifact_hash
        ):
            raise ValueError("incompatible current model hash verification failed")
        metadata_hash = _required_text(pointer, "metadata_sha256")
        if (
            not _is_sha256(metadata_hash)
            or manifest.get("metadata_sha256") != metadata_hash
            or _sha256(metadata_path) != metadata_hash
        ):
            raise ValueError("incompatible current metadata hash verification failed")
        metadata = _read_json(metadata_path)
        if (
            metadata.get("artifact_registry_version") != registry_version
            or metadata.get("artifact_model_id") != self.model_id
            or metadata.get("artifact_version") != version
            or metadata.get("artifact_sha256") != artifact_hash
        ):
            raise ValueError("incompatible current metadata identity mismatch")

    def _build_activation_manifest(
        self,
        candidate: ResolvedModelArtifact,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        stage = _required_text(evidence, "activation_stage")
        if stage not in _WRITABLE_ACTIVATION_STAGES:
            raise ValueError("activation_stage must be shadow, canary, or active")
        production_authorized = evidence.get("production_influence_authorized") is True
        paper_canary_authorized = evidence.get("paper_canary_authorized") is True
        readiness_state = str(evidence.get("readiness_state") or "").strip()
        blockers = evidence.get("blocking_reasons")
        blockers = blockers if isinstance(blockers, list) else []
        if stage == "shadow" and production_authorized:
            raise ValueError("shadow artifact cannot receive production influence")
        if paper_canary_authorized:
            if stage != "canary" or production_authorized:
                raise ValueError(
                    "paper canary activation must be canary without production authorization"
                )
            if readiness_state != "paper_canary_ready" or blockers:
                raise ValueError("paper canary activation requires clean readiness evidence")
            paper_report = evidence.get("paper_canary_report")
            if not isinstance(paper_report, dict) or (
                paper_report.get("authorized") is not True
                or paper_report.get("state") != "ready"
                or paper_report.get("execution_scope") != "paper_only"
                or paper_report.get("production_permission") is not False
                or list(paper_report.get("blocking_reasons") or [])
                or not list(paper_report.get("eligible_sides") or [])
            ):
                raise ValueError("paper canary readiness report is incomplete")
            walk_forward = candidate.manifest.get("walk_forward_report")
            if (
                not isinstance(walk_forward, dict)
                or walk_forward.get("status") != "complete"
                or walk_forward.get("decision_group_disjoint") is not True
                or walk_forward.get("chronological_label_disjoint") is not True
                or walk_forward.get("model_refit_per_fold") is not True
                or not list(walk_forward.get("folds") or [])
            ):
                raise ValueError("paper canary activation requires complete walk-forward evidence")
        elif stage in {"canary", "active"}:
            if not production_authorized:
                raise ValueError(f"{stage} activation requires production authorization")
            if readiness_state not in {"ready", "partial_ready"}:
                raise ValueError(f"{stage} activation requires ready artifact evidence")
            if blockers:
                raise ValueError(f"{stage} activation cannot contain blocking reasons")
            return_evidence = evidence.get("return_evidence_report")
            if not isinstance(return_evidence, dict):
                raise ValueError(f"{stage} activation requires a return evidence report")
            if (
                return_evidence.get("allow_live_position_influence") is not True
                or return_evidence.get("state") not in {"ready", "partial_ready"}
                or list(return_evidence.get("blocking_reasons") or [])
            ):
                raise ValueError(f"{stage} return evidence does not authorize production")
            live_enabled_sides = return_evidence.get("live_enabled_sides")
            if not isinstance(live_enabled_sides, list):
                raise ValueError(f"{stage} activation requires explicit live-enabled sides")
            live_sides = {
                str(side).strip().lower()
                for side in live_enabled_sides
                if str(side).strip()
            }
            if (
                not live_sides
                or not live_sides.issubset({"long", "short"})
                or len(live_sides) != len(live_enabled_sides)
            ):
                raise ValueError(f"{stage} activation contains invalid live-enabled sides")
            if return_evidence.get("state") == "ready" and live_sides != {"long", "short"}:
                raise ValueError(f"{stage} ready activation requires both live sides")
            if return_evidence.get("state") == "partial_ready" and len(live_sides) != 1:
                raise ValueError(f"{stage} partial-ready activation requires exactly one live side")
            if stage == "active" and return_evidence.get("state") != "ready":
                raise ValueError("active activation requires ready return evidence")
            evidence_sides = evidence.get("live_enabled_sides")
            if not isinstance(evidence_sides, list) or set(evidence_sides) != live_sides:
                raise ValueError(f"{stage} activation sides must match return evidence")
            walk_forward = candidate.manifest.get("walk_forward_report")
            loso = candidate.manifest.get("leave_one_symbol_out_report")
            oos = candidate.manifest.get("oos_return_evaluation")
            if (
                not isinstance(walk_forward, dict)
                or walk_forward.get("status") != "complete"
                or walk_forward.get("decision_group_disjoint") is not True
                or walk_forward.get("chronological_label_disjoint") is not True
                or walk_forward.get("model_refit_per_fold") is not True
                or not isinstance(walk_forward.get("sides"), dict)
                or not list(walk_forward.get("folds") or [])
            ):
                raise ValueError(f"{stage} activation requires complete walk-forward evidence")
            walk_sides = walk_forward["sides"]
            folds = walk_forward["folds"]
            if len(folds) < 2:
                raise ValueError(
                    f"{stage} activation requires multiple walk-forward windows"
                )
            if any(
                not isinstance(walk_sides.get(side), dict)
                or walk_sides[side].get("promotion_math_ready") is not True
                or not isinstance(
                    walk_sides[side].get("market_regime_stability"), dict
                )
                or walk_sides[side]["market_regime_stability"].get("stable")
                is not True
                or any(
                    not isinstance(fold, dict)
                    or not isinstance(fold.get("sides"), dict)
                    or not isinstance(fold["sides"].get(side), dict)
                    or fold["sides"][side].get("promotion_math_ready") is not True
                    for fold in folds
                )
                for side in live_sides
            ):
                raise ValueError(f"{stage} activation requires stable walk-forward evidence")
            if not isinstance(loso, dict) or any(
                not isinstance(loso.get(side), dict)
                or loso[side].get("stable") is not True
                for side in live_sides
            ):
                raise ValueError(f"{stage} activation requires stable symbol-removal evidence")
            if not isinstance(oos, dict) or any(
                not isinstance(oos.get(side), dict)
                or oos[side].get("promotion_math_ready") is not True
                for side in live_sides
            ):
                raise ValueError(f"{stage} activation requires complete OOS return evidence")
        return {
            **evidence,
            "activation_manifest_version": ARTIFACT_ACTIVATION_MANIFEST_VERSION,
            "artifact_registry_version": ARTIFACT_REGISTRY_VERSION,
            "artifact_model_id": self.model_id,
            "artifact_version": candidate.version,
            "artifact_sha256": candidate.sha256,
            "artifact_manifest_sha256": _sha256(candidate.manifest_path),
            "training_data_sha256": candidate.manifest.get("training_data_sha256"),
            "source_code_sha256": candidate.manifest.get("source_code_sha256"),
            "walk_forward_report_sha256": _payload_sha256(
                candidate.manifest.get("walk_forward_report") or {}
            ),
            "leave_one_symbol_out_report_sha256": _payload_sha256(
                candidate.manifest.get("leave_one_symbol_out_report") or {}
            ),
            "return_evidence_report_sha256": _payload_sha256(
                evidence.get("return_evidence_report") or {}
            ),
            "paper_canary_report_sha256": _payload_sha256(
                evidence.get("paper_canary_report") or {}
            ),
            "activation_stage": stage,
            "production_influence_authorized": production_authorized,
            "paper_canary_authorized": paper_canary_authorized,
            "blocking_reasons": blockers,
            "activated_at": datetime.now(UTC).isoformat(),
        }

    def _resolve_pointer(
        self,
        pointer_path: Path,
        *,
        expected_role: str,
        required: bool,
    ) -> ResolvedModelArtifact | None:
        if not pointer_path.exists():
            if required:
                raise ValueError(f"{expected_role} artifact is not registered")
            return None
        pointer = _read_json(pointer_path)
        if pointer.get("artifact_registry_version") != ARTIFACT_REGISTRY_VERSION:
            raise ValueError("unsupported artifact registry pointer version")
        if pointer.get("pointer_role") != expected_role:
            raise ValueError("artifact registry pointer role mismatch")
        if pointer.get("model_id") != self.model_id:
            raise ValueError("artifact registry pointer model identity mismatch")
        version = _required_text(pointer, "version")
        manifest_relative = _required_text(pointer, "manifest_path")
        version_root = (self.versions_root / version).resolve(strict=True)
        manifest_path = (self.model_root / manifest_relative).resolve(strict=True)
        manifest_path.relative_to(version_root)
        expected_manifest_hash = _required_text(pointer, "manifest_sha256")
        if _sha256(manifest_path) != expected_manifest_hash:
            raise ValueError("artifact manifest hash verification failed")
        manifest = _read_json(manifest_path)
        if manifest.get("artifact_registry_version") != ARTIFACT_REGISTRY_VERSION:
            raise ValueError("artifact manifest registry version mismatch")
        if manifest.get("artifact_model_id") != self.model_id:
            raise ValueError("artifact manifest model identity mismatch")
        if manifest.get("artifact_version") != version:
            raise ValueError("artifact manifest version mismatch")

        model_path = (version_root / _required_text(manifest, "model_relative_path")).resolve(
            strict=True
        )
        metadata_path = (
            version_root / _required_text(manifest, "metadata_relative_path")
        ).resolve(strict=True)
        model_path.relative_to(version_root)
        metadata_path.relative_to(version_root)
        pointer_hash = _required_text(pointer, "sha256")
        if pointer_hash != _required_text(manifest, "artifact_sha256"):
            raise ValueError("artifact hash evidence mismatch")
        actual_hash = _sha256(model_path)
        if actual_hash != pointer_hash:
            raise ValueError("artifact hash verification failed")
        pointer_metadata_hash = _required_text(pointer, "metadata_sha256")
        if pointer_metadata_hash != _required_text(manifest, "metadata_sha256"):
            raise ValueError("artifact metadata hash evidence mismatch")
        if _sha256(metadata_path) != pointer_metadata_hash:
            raise ValueError("artifact metadata hash verification failed")
        metadata = _read_json(metadata_path)
        self._validate_metadata_identity(metadata, manifest, version, actual_hash)
        bundle = load_trusted_joblib(
            model_path,
            trusted_root=version_root,
            expected_type=dict,
        )
        embedded_metadata = bundle.get("metadata")
        if not isinstance(embedded_metadata, dict):
            raise ValueError("artifact bundle metadata is missing")
        self._validate_embedded_identity(embedded_metadata, metadata)

        activation = None
        rejection = None
        if expected_role in {"current", "rollback"}:
            activation_relative = _required_text(pointer, "activation_manifest_path")
            activation_path = (self.model_root / activation_relative).resolve(strict=True)
            activation_path.relative_to(version_root)
            if _sha256(activation_path) != _required_text(
                pointer, "activation_manifest_sha256"
            ):
                raise ValueError("artifact activation manifest hash verification failed")
            activation = _read_json(activation_path)
            self._validate_activation_identity(
                activation,
                manifest,
                actual_hash,
                expected_manifest_hash,
            )
        elif expected_role == "challenger":
            rejection_relative = _required_text(pointer, "rejection_manifest_path")
            rejection_path = (self.model_root / rejection_relative).resolve(strict=True)
            rejection_path.relative_to(version_root)
            if _sha256(rejection_path) != _required_text(
                pointer, "rejection_manifest_sha256"
            ):
                raise ValueError("artifact rejection manifest hash verification failed")
            rejection = _read_json(rejection_path)
            if (
                rejection.get("artifact_registry_version") != ARTIFACT_REGISTRY_VERSION
                or rejection.get("model_id") != self.model_id
                or rejection.get("version") != version
                or rejection.get("artifact_sha256") != actual_hash
                or not isinstance(rejection.get("comparison_report"), dict)
            ):
                raise ValueError("artifact rejection manifest identity mismatch")
        return ResolvedModelArtifact(
            model_id=self.model_id,
            version=version,
            model_path=model_path,
            metadata_path=metadata_path,
            manifest_path=manifest_path,
            sha256=actual_hash,
            manifest=manifest,
            pointer_role=expected_role,
            pointer_path=pointer_path,
            activation_manifest=activation,
            rejection_manifest=rejection,
        )

    def _validate_metadata_identity(
        self,
        metadata: dict[str, Any],
        manifest: dict[str, Any],
        version: str,
        artifact_hash: str,
    ) -> None:
        if metadata.get("artifact_model_id") != self.model_id:
            raise ValueError("artifact metadata model identity mismatch")
        if metadata.get("artifact_version") != version:
            raise ValueError("artifact metadata version mismatch")
        if metadata.get("artifact_sha256") != artifact_hash:
            raise ValueError("artifact metadata model hash mismatch")
        identity_fields = (
            "artifact_registry_version",
            "artifact_model_id",
            "artifact_version",
            "parent_model_identity",
            "training_data_version",
            "sample_cursor",
            "code_version",
            "objective_name",
            "objective_version",
            "label_version",
            "profit_supervision_version",
            "training_data_sha256",
            "source_code_sha256",
            "evaluation_group_policy",
            "model_stage",
        )
        for field in identity_fields:
            if metadata.get(field) != manifest.get(field):
                raise ValueError(f"artifact metadata/manifest {field} mismatch")

    @staticmethod
    def _validate_embedded_identity(
        embedded: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        identity_fields = (
            "artifact_registry_version",
            "artifact_model_id",
            "artifact_version",
            "parent_model_identity",
            "training_data_version",
            "sample_cursor",
            "code_version",
            "objective_name",
            "objective_version",
            "label_version",
            "profit_supervision_version",
        )
        for field in identity_fields:
            if embedded.get(field) != metadata.get(field):
                raise ValueError(f"artifact bundle/metadata {field} mismatch")

    def _validate_activation_identity(
        self,
        activation: dict[str, Any],
        manifest: dict[str, Any],
        artifact_hash: str,
        manifest_hash: str,
    ) -> None:
        if (
            activation.get("activation_manifest_version")
            != ARTIFACT_ACTIVATION_MANIFEST_VERSION
        ):
            raise ValueError("artifact activation manifest version mismatch")
        if activation.get("artifact_registry_version") != ARTIFACT_REGISTRY_VERSION:
            raise ValueError("artifact activation registry version mismatch")
        if activation.get("artifact_model_id") != self.model_id:
            raise ValueError("artifact activation model identity mismatch")
        if activation.get("artifact_version") != manifest.get("artifact_version"):
            raise ValueError("artifact activation version mismatch")
        if activation.get("artifact_sha256") != artifact_hash:
            raise ValueError("artifact activation hash mismatch")
        if activation.get("artifact_manifest_sha256") != manifest_hash:
            raise ValueError("artifact activation manifest identity mismatch")
        if activation.get("training_data_sha256") != manifest.get(
            "training_data_sha256"
        ):
            raise ValueError("artifact activation training-data identity mismatch")
        if activation.get("source_code_sha256") != manifest.get("source_code_sha256"):
            raise ValueError("artifact activation source-code identity mismatch")
        if activation.get("walk_forward_report_sha256") != _payload_sha256(
            manifest.get("walk_forward_report") or {}
        ):
            raise ValueError("artifact activation walk-forward identity mismatch")
        if activation.get("leave_one_symbol_out_report_sha256") != _payload_sha256(
            manifest.get("leave_one_symbol_out_report") or {}
        ):
            raise ValueError("artifact activation symbol-removal identity mismatch")
        if activation.get("return_evidence_report_sha256") != _payload_sha256(
            activation.get("return_evidence_report") or {}
        ):
            raise ValueError("artifact activation return-evidence identity mismatch")
        if activation.get("paper_canary_report_sha256") is not None and (
            activation.get("paper_canary_report_sha256")
            != _payload_sha256(activation.get("paper_canary_report") or {})
        ):
            raise ValueError("artifact activation paper-canary identity mismatch")
        stage = activation.get("activation_stage")
        if stage not in _READABLE_ACTIVATION_STAGES:
            raise ValueError("artifact activation stage is invalid")
        production_authorized = activation.get("production_influence_authorized") is True
        paper_canary_authorized = activation.get("paper_canary_authorized") is True
        blockers = activation.get("blocking_reasons")
        blockers = blockers if isinstance(blockers, list) else []
        if stage == "shadow" and production_authorized:
            raise ValueError("shadow artifact has production authorization")
        if paper_canary_authorized:
            paper_report = activation.get("paper_canary_report")
            if (
                stage != "canary"
                or production_authorized
                or activation.get("readiness_state") != "paper_canary_ready"
                or blockers
                or not isinstance(paper_report, dict)
                or paper_report.get("authorized") is not True
                or paper_report.get("execution_scope") != "paper_only"
                or paper_report.get("production_permission") is not False
                or list(paper_report.get("blocking_reasons") or [])
            ):
                raise ValueError("paper canary artifact activation evidence is incomplete")
        elif stage in {"canary", "active", "live"} and (
            not production_authorized
            or activation.get("readiness_state") not in {"ready", "partial_ready"}
            or blockers
        ):
            raise ValueError("production artifact activation evidence is incomplete")
        if stage == "active" and activation.get("readiness_state") != "ready":
            raise ValueError("active artifact requires ready return evidence")
        if stage in {"canary", "active", "live"} and not paper_canary_authorized:
            return_evidence = activation.get("return_evidence_report")
            if not isinstance(return_evidence, dict) or (
                return_evidence.get("allow_live_position_influence") is not True
                or return_evidence.get("state") not in {"ready", "partial_ready"}
                or list(return_evidence.get("blocking_reasons") or [])
            ):
                raise ValueError("production artifact return evidence is incomplete")

    def status(self) -> dict[str, Any]:
        pointer_status = {
            role: self._pointer_status(role)
            for role in ("candidate", "challenger", "current", "rollback")
        }
        current = pointer_status["current"]
        return {
            "available": bool(current.get("available")),
            "model_id": self.model_id,
            "registry_version": ARTIFACT_REGISTRY_VERSION,
            "candidate_pointer": str(self.candidate_path),
            "challenger_pointer": str(self.challenger_path),
            "current_pointer": str(self.current_path),
            "active_pointer": str(self.active_path),
            "rollback_pointer": str(self.rollback_path),
            "pointers": pointer_status,
            **(
                {
                    key: current.get(key)
                    for key in (
                        "version",
                        "model_path",
                        "manifest_path",
                        "sha256",
                        "manifest",
                        "activation_manifest",
                    )
                }
                if current.get("available")
                else {"error": current.get("error")}
            ),
        }

    def _pointer_status(self, role: str) -> dict[str, Any]:
        resolver = {
            "candidate": self.resolve_candidate,
            "challenger": self.resolve_challenger,
            "current": self.resolve_current,
            "rollback": self.resolve_rollback,
        }[role]
        try:
            resolved = resolver()
        except Exception as exc:
            return {"available": False, "role": role, "error": str(exc)}
        if resolved is None:
            return {
                "available": False,
                "role": role,
                "error": f"{role}_artifact_not_registered",
            }
        return {
            "available": True,
            "role": role,
            "version": resolved.version,
            "model_path": str(resolved.model_path),
            "manifest_path": str(resolved.manifest_path),
            "sha256": resolved.sha256,
            "manifest": resolved.manifest,
            "activation_manifest": resolved.activation_manifest,
            "rejection_manifest": resolved.rejection_manifest,
        }
