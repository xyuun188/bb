"""Verified candidate-manifest contract for the single local model.

The platform host must never infer that a model is ready from a model name or a
directory alone.  A manifest is produced by the model-host validation job and
contains immutable identity, file fingerprints, license/quantization metadata,
and measured resource evidence.  The sync process consumes this contract but
does not perform a large weight download or silently promote a candidate.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from core.model_topology import ModelTopology, activate_verified_candidate

MANIFEST_VERSION = "bb.model-candidate.v1"
VERIFIED_STATUS = "verified"
MIN_SHA256_LENGTH = 64
LEGACY_MODEL_MARKERS = ("14b", "deepseek", "finquant-expert")
REVISION_PATTERN = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{8,128}$|^[A-Za-z0-9._/-]{8,256}$")


def _clean_text(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text or any(char in text for char in ("\x00", "\r", "\n")):
        raise ValueError(f"candidate manifest field {field!r} is empty or contains control characters")
    return text


def _sha256(value: Any, *, field: str) -> str:
    digest = _clean_text(value, field=field).lower()
    if len(digest) != MIN_SHA256_LENGTH or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"candidate manifest field {field!r} is not a SHA-256 digest")
    return digest


@dataclass(frozen=True, slots=True)
class CandidateFile:
    path: str
    size_bytes: int
    sha256: str

    @classmethod
    def from_dict(cls, value: Any, *, field: str) -> CandidateFile:
        if not isinstance(value, dict):
            raise ValueError(f"{field} must be an object")
        path = _clean_text(value.get("path"), field=f"{field}.path")
        if path.startswith(("/", "\\")) or ".." in Path(path).parts:
            raise ValueError(f"{field}.path must be a relative safe path")
        size = value.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"{field}.size_bytes must be a non-negative integer")
        return cls(path=path, size_bytes=size, sha256=_sha256(value.get("sha256"), field=f"{field}.sha256"))

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "size_bytes": self.size_bytes, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class ModelCandidateManifest:
    manifest_version: str
    status: str
    model_id: str
    repo_id: str
    revision: str
    model_path: str
    tokenizer_path: str
    license: str
    quantization: str
    context_length: int
    config_sha256: str
    tokenizer_sha256: str
    weight_files: tuple[CandidateFile, ...]
    gpu_memory_peak_gib: float
    inference_p95_ms: float
    max_concurrency: int
    validated_at: str
    validator_version: str

    @classmethod
    def from_dict(cls, value: Any) -> ModelCandidateManifest:
        if not isinstance(value, dict):
            raise ValueError("candidate manifest must be a JSON object")
        version = _clean_text(value.get("manifest_version"), field="manifest_version")
        if version != MANIFEST_VERSION:
            raise ValueError(f"unsupported candidate manifest version: {version}")
        status = _clean_text(value.get("status"), field="status").lower()
        if status != VERIFIED_STATUS:
            raise ValueError(f"candidate manifest status must be {VERIFIED_STATUS!r}")
        model_id = _clean_text(value.get("model_id"), field="model_id")
        lowered = model_id.lower()
        if any(marker in lowered for marker in LEGACY_MODEL_MARKERS):
            raise ValueError("candidate manifest refers to a legacy 14B/reviewer model")
        revision = _clean_text(value.get("revision"), field="revision")
        if revision.lower() in {"unknown", "unverified", "pending"} or not REVISION_PATTERN.fullmatch(revision):
            raise ValueError("candidate manifest revision is not a fixed immutable identifier")
        context_length = value.get("context_length")
        if isinstance(context_length, bool) or not isinstance(context_length, int) or context_length < 256:
            raise ValueError("candidate manifest context_length must be an integer >= 256")
        max_concurrency = value.get("max_concurrency")
        if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int) or max_concurrency < 1:
            raise ValueError("candidate manifest max_concurrency must be a positive integer")
        peak = value.get("gpu_memory_peak_gib")
        p95 = value.get("inference_p95_ms")
        if isinstance(peak, bool) or not isinstance(peak, (int, float)) or peak <= 0:
            raise ValueError("candidate manifest gpu_memory_peak_gib must be positive")
        if isinstance(p95, bool) or not isinstance(p95, (int, float)) or p95 <= 0:
            raise ValueError("candidate manifest inference_p95_ms must be positive")
        files_raw = value.get("weight_files")
        if not isinstance(files_raw, list) or not files_raw:
            raise ValueError("candidate manifest weight_files must be non-empty")
        files = tuple(CandidateFile.from_dict(item, field=f"weight_files[{index}]") for index, item in enumerate(files_raw))
        if len({item.path for item in files}) != len(files):
            raise ValueError("candidate manifest weight_files contains duplicate paths")
        return cls(
            manifest_version=version,
            status=status,
            model_id=model_id,
            repo_id=_clean_text(value.get("repo_id"), field="repo_id"),
            revision=revision,
            model_path=_absolute_model_path(value.get("model_path"), field="model_path"),
            tokenizer_path=_absolute_model_path(value.get("tokenizer_path"), field="tokenizer_path"),
            license=_clean_text(value.get("license"), field="license"),
            quantization=_clean_text(value.get("quantization"), field="quantization"),
            context_length=context_length,
            config_sha256=_sha256(value.get("config_sha256"), field="config_sha256"),
            tokenizer_sha256=_sha256(value.get("tokenizer_sha256"), field="tokenizer_sha256"),
            weight_files=files,
            gpu_memory_peak_gib=float(peak),
            inference_p95_ms=float(p95),
            max_concurrency=max_concurrency,
            validated_at=_clean_text(value.get("validated_at"), field="validated_at"),
            validator_version=_clean_text(value.get("validator_version"), field="validator_version"),
        )

    @classmethod
    def load(cls, path: str | Path) -> ModelCandidateManifest:
        manifest_path = Path(path)
        if not manifest_path.is_file():
            raise ValueError(f"candidate manifest does not exist: {manifest_path}")
        try:
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read candidate manifest: {manifest_path}") from exc
        return cls.from_dict(value)

    def validate_evidence(self) -> tuple[str, ...]:
        errors: list[str] = []
        try:
            parsed = datetime.fromisoformat(self.validated_at.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                errors.append("validated_at_missing_timezone")
        except ValueError:
            errors.append("validated_at_invalid")
        if self.gpu_memory_peak_gib >= 38.0:
            errors.append("gpu_memory_peak_exceeds_a100_budget")
        if self.inference_p95_ms > 30_000:
            errors.append("inference_p95_exceeds_30_seconds")
        return tuple(errors)

    def to_topology(self, *, stage: str = "candidate_validated") -> ModelTopology:
        if stage not in {"candidate_validated", "paper"}:
            raise ValueError("verified candidate topology cannot enable live routing")
        evidence_errors = self.validate_evidence()
        if evidence_errors:
            raise ValueError("candidate evidence is invalid: " + ", ".join(evidence_errors))
        from core.model_topology import qwen27_candidate_topology

        return activate_verified_candidate(
            qwen27_candidate_topology(),
            model_id=self.model_id,
            repo_id=self.repo_id,
            revision=self.revision,
            path=self.model_path,
            stage=stage,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "status": self.status,
            "model_id": self.model_id,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "model_path": self.model_path,
            "tokenizer_path": self.tokenizer_path,
            "license": self.license,
            "quantization": self.quantization,
            "context_length": self.context_length,
            "config_sha256": self.config_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "weight_files": [item.to_dict() for item in self.weight_files],
            "gpu_memory_peak_gib": self.gpu_memory_peak_gib,
            "inference_p95_ms": self.inference_p95_ms,
            "max_concurrency": self.max_concurrency,
            "validated_at": self.validated_at,
            "validator_version": self.validator_version,
        }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_model_path(value: Any, *, field: str) -> str:
    path = _clean_text(value, field=field)
    windows_absolute = bool(re.match(r"^[A-Za-z]:[\\/]", path))
    posix_absolute = path.startswith("/")
    parts = [part for part in re.split(r"[\\/]", path) if part]
    if not (windows_absolute or posix_absolute) or any(part in {"..", "."} for part in parts):
        raise ValueError(f"candidate manifest field {field!r} must be an absolute safe path")
    # Remote production manifests live below /data; local validator tests and
    # staging may use an absolute Windows path before the manifest is copied.
    if posix_absolute and not path.startswith("/data/"):
        raise ValueError(f"candidate manifest field {field!r} must stay under /data")
    return path.rstrip("/")
