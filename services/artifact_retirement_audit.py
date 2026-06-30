from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config.settings import settings
from core.safe_output import safe_error_text

PHASE3_ARTIFACT_POLICY_ID = "phase3_clean_training_artifact_v1"
PHASE3_REQUIRED_TRAINING_POLICY = "clean_training_view_only"
PHASE3_REQUIRED_PROMOTION_FLOW = "shadow_to_canary_to_live"

ARTIFACT_SUFFIXES = {".joblib", ".pkl", ".pickle", ".onnx", ".pt", ".safetensors", ".bin"}
METADATA_SUFFIXES = {".json"}

DEFAULT_SCAN_RELATIVE_PATHS = (
    "ml_signal",
    "local_ai_tools",
    "models",
)

LEGACY_RELATIVE_PATHS = {
    "ml_signal/winrate_model.joblib",
    "ml_signal/winrate_model_metadata.json",
}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _read_json(path: Path) -> tuple[dict[str, Any], str | None]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, "missing"
    except (OSError, json.JSONDecodeError) as exc:
        return {}, safe_error_text(exc, limit=160)
    return (parsed if isinstance(parsed, dict) else {}), None


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(root.resolve(strict=False)).as_posix()
    except ValueError:
        return path.as_posix()


def _metadata_candidates(artifact_path: Path) -> list[Path]:
    if artifact_path.suffix in METADATA_SUFFIXES:
        return [artifact_path]
    stem = artifact_path.with_suffix("")
    return [
        artifact_path.with_name(f"{stem.name}_metadata.json"),
        artifact_path.with_suffix(".metadata.json"),
        artifact_path.with_suffix(".json"),
    ]


def _metadata_for_artifact(artifact_path: Path) -> tuple[Path | None, dict[str, Any], str | None]:
    for candidate in _metadata_candidates(artifact_path):
        if not candidate.exists():
            continue
        metadata, error = _read_json(candidate)
        return candidate, metadata, error
    return None, {}, "missing_manifest"


def _phase3_evidence(metadata: dict[str, Any]) -> dict[str, Any]:
    evaluation_policy = _safe_dict(metadata.get("evaluation_policy"))
    governance_report = _safe_dict(metadata.get("governance_report"))
    quality_report = _safe_dict(metadata.get("quality_report"))
    return {
        "policy_id": metadata.get("artifact_policy_id") or metadata.get("policy_id"),
        "phase": metadata.get("phase") or evaluation_policy.get("phase"),
        "training_policy": metadata.get("trade_sample_cursor_policy")
        or metadata.get("training_policy")
        or governance_report.get("training_policy"),
        "training_mode": metadata.get("training_mode"),
        "model_stage": metadata.get("model_stage"),
        "promotion_flow": evaluation_policy.get("promotion_flow")
        or metadata.get("promotion_flow"),
        "live_mutation": bool(evaluation_policy.get("live_mutation")),
        "artifact_persisted": metadata.get("artifact_persisted"),
        "quality_version": quality_report.get("data_quality_version")
        or metadata.get("data_quality_version"),
    }


def _artifact_classification(
    *,
    relative_path: str,
    metadata: dict[str, Any],
    metadata_error: str | None,
) -> tuple[str, list[str], dict[str, Any]]:
    evidence = _phase3_evidence(metadata)
    reasons: list[str] = []
    normalized_relative_path = relative_path.replace("\\", "/")

    if normalized_relative_path in LEGACY_RELATIVE_PATHS:
        reasons.append("known_legacy_artifact_path")
    if metadata_error:
        reasons.append("missing_or_unreadable_phase3_manifest")
    if not metadata:
        reasons.append("missing_phase3_metadata")

    policy_ok = evidence["policy_id"] == PHASE3_ARTIFACT_POLICY_ID
    training_policy_ok = evidence["training_policy"] == PHASE3_REQUIRED_TRAINING_POLICY
    promotion_ok = evidence["promotion_flow"] == PHASE3_REQUIRED_PROMOTION_FLOW
    live_mutation_ok = evidence["live_mutation"] is False
    persisted_ok = evidence["artifact_persisted"] is True or normalized_relative_path.endswith(
        ".json"
    )

    if metadata and not policy_ok:
        reasons.append("artifact_policy_id_not_phase3")
    if metadata and not training_policy_ok:
        reasons.append("training_policy_not_clean_view")
    if metadata and not promotion_ok:
        reasons.append("promotion_flow_missing_or_untrusted")
    if metadata and not live_mutation_ok:
        reasons.append("metadata_allows_live_mutation")
    if metadata and not persisted_ok:
        reasons.append("artifact_persisted_not_confirmed")

    if reasons:
        if "known_legacy_artifact_path" in reasons:
            return "retired_legacy", reasons, evidence
        if "missing_phase3_metadata" in reasons or "missing_or_unreadable_phase3_manifest" in reasons:
            return "missing_manifest", reasons, evidence
        return "untrusted", reasons, evidence
    return "phase3_compatible", [], evidence


@dataclass(frozen=True)
class ArtifactRetirementAuditService:
    """Read-only audit that prevents legacy model artifacts from entering Phase 3."""

    root: Path | None = None
    scan_relative_paths: tuple[str, ...] = DEFAULT_SCAN_RELATIVE_PATHS

    def _root(self) -> Path:
        return self.root or settings.data_dir

    def _candidate_paths(self) -> list[Path]:
        root = self._root()
        candidates: list[Path] = []
        for relative in self.scan_relative_paths:
            base = root / relative
            if not base.exists():
                continue
            if base.is_file():
                candidates.append(base)
                continue
            for path in base.rglob("*"):
                if not path.is_file():
                    continue
                suffix = path.suffix.lower()
                if suffix in ARTIFACT_SUFFIXES or (
                    suffix in METADATA_SUFFIXES
                    and (
                        "metadata" in path.name.lower()
                        or "manifest" in path.name.lower()
                    )
                ):
                    candidates.append(path)
        return sorted({path.resolve(strict=False) for path in candidates})

    async def report(self) -> dict[str, Any]:
        started_at = datetime.now(UTC)
        root = self._root()
        artifacts: list[dict[str, Any]] = []
        status_counts: dict[str, int] = {}

        for path in self._candidate_paths():
            relative = _relative_path(path, root)
            metadata_path, metadata, metadata_error = _metadata_for_artifact(path)
            classification, reasons, evidence = _artifact_classification(
                relative_path=relative,
                metadata=metadata,
                metadata_error=metadata_error,
            )
            status_counts[classification] = status_counts.get(classification, 0) + 1
            artifacts.append(
                {
                    "path": str(path),
                    "relative_path": relative,
                    "size_bytes": path.stat().st_size,
                    "modified_at": datetime.fromtimestamp(
                        path.stat().st_mtime, tz=UTC
                    ).isoformat(),
                    "metadata_path": str(metadata_path) if metadata_path else None,
                    "classification": classification,
                    "reasons": reasons,
                    "phase3_evidence": evidence,
                    "preserved": True,
                    "can_delete": False,
                    "can_influence_live": classification == "phase3_compatible",
                }
            )

        retired_or_untrusted = [
            item
            for item in artifacts
            if item["classification"]
            in {"retired_legacy", "missing_manifest", "untrusted"}
        ]
        phase3_compatible = [
            item for item in artifacts if item["classification"] == "phase3_compatible"
        ]
        status = "retired_required" if retired_or_untrusted else "ready"
        return {
            "status": status,
            "audit_only": True,
            "read_only": True,
            "raw_artifacts_preserved": True,
            "can_delete_artifacts": False,
            "training_policy": PHASE3_REQUIRED_TRAINING_POLICY,
            "artifact_policy_id": PHASE3_ARTIFACT_POLICY_ID,
            "root": str(root),
            "scan_relative_paths": list(self.scan_relative_paths),
            "artifact_count": len(artifacts),
            "phase3_compatible_count": len(phase3_compatible),
            "retired_or_untrusted_count": len(retired_or_untrusted),
            "status_counts": status_counts,
            "artifacts": artifacts[:50],
            "retired_or_untrusted_samples": retired_or_untrusted[:20],
            "checked_at": _now_iso(),
            "duration_seconds": round((datetime.now(UTC) - started_at).total_seconds(), 6),
            "next_required_action": (
                "rebuild_phase3_artifacts_from_clean_training_view"
                if retired_or_untrusted
                else "keep_phase3_artifact_manifest_attached"
            ),
        }
