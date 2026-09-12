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
import math
import re
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from core.model_topology import (
    TARGET_SINGLE_MODEL_ID,
    TARGET_SINGLE_MODEL_REPO,
    ModelTopology,
    activate_verified_candidate,
)

MANIFEST_VERSION = "bb.model-candidate.v2"
VERIFIED_STATUS = "verified"
MIN_SHA256_LENGTH = 64
LEGACY_MODEL_MARKERS = ("14b", "deepseek", "finquant-expert")
REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
A100_MODEL_BUDGET_GIB = 34.0
TARGET_REPOSITORY_ID = TARGET_SINGLE_MODEL_REPO
TARGET_MODEL_TYPE = "qwen3_5"
TARGET_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
TARGET_MODEL_ID = TARGET_SINGLE_MODEL_ID
ALLOWED_MODEL_ROOTS = ("/data/", "/home/linux/")
MIN_TRANSFORMERS_VERSION = (5, 8, 0)
MIN_VLLM_VERSION = (0, 17, 1)
MAX_MANIFEST_AGE = timedelta(days=7)
MAX_MANIFEST_FUTURE_SKEW = timedelta(minutes=5)


@dataclass(frozen=True, slots=True)
class CandidateRuntime:
    engine: str
    engine_version: str
    transformers_version: str
    probe_sha256: str

    @classmethod
    def from_dict(cls, value: Any) -> CandidateRuntime:
        if not isinstance(value, dict):
            raise ValueError("candidate manifest runtime must be an object")
        engine = _clean_text(value.get("engine"), field="runtime.engine").lower()
        if engine not in {"transformers", "vllm", "sglang"}:
            raise ValueError("candidate manifest runtime.engine is unsupported")
        engine_version = _clean_text(value.get("engine_version"), field="runtime.engine_version")
        transformers_version = _clean_text(
            value.get("transformers_version"), field="runtime.transformers_version"
        )
        if engine_version.lower() in {"unknown", "unverified", "pending"}:
            raise ValueError("candidate manifest runtime.engine_version is not verified")
        if transformers_version.lower() in {"unknown", "unverified", "pending"}:
            raise ValueError("candidate manifest runtime.transformers_version is not verified")
        if version_triplet(transformers_version) < MIN_TRANSFORMERS_VERSION:
            raise ValueError("candidate Transformers runtime is too old for Qwen3.8-27B")
        if engine == "transformers" and version_triplet(engine_version) < MIN_TRANSFORMERS_VERSION:
            raise ValueError("candidate Transformers runtime is too old for Qwen3.8-27B")
        if engine == "vllm" and version_triplet(engine_version) < MIN_VLLM_VERSION:
            raise ValueError("candidate vLLM runtime is too old for Qwen3.8-27B")
        return cls(
            engine=engine,
            engine_version=engine_version,
            transformers_version=transformers_version,
            probe_sha256=_sha256(value.get("probe_sha256"), field="runtime.probe_sha256"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "engine": self.engine,
            "engine_version": self.engine_version,
            "transformers_version": self.transformers_version,
            "probe_sha256": self.probe_sha256,
        }


def _clean_text(value: Any, *, field: str) -> str:
    text = value.strip() if isinstance(value, str) else ""
    if not text or any(ord(char) < 32 or ord(char) == 127 for char in text):
        raise ValueError(f"candidate manifest field {field!r} is empty or contains control characters")
    return text


def version_triplet(value: str) -> tuple[int, int, int]:
    match = re.search(r"(?<!\d)(\d+)\.(\d+)\.(\d+)", value)
    if match is None:
        raise ValueError(f"candidate runtime version is not semantic: {value!r}")
    return tuple(int(part) for part in match.groups())


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
        if (
            PurePosixPath(path).is_absolute() or PureWindowsPath(path).drive
            or "\\" in path or any(part in {"", ".", ".."} for part in path.split("/"))
        ):
            raise ValueError(f"{field}.path must be a relative safe path")
        size = value.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"{field}.size_bytes must be a positive integer")
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
    architecture: str
    model_type: str
    language_model_only: bool
    text_inference_verified: bool
    runtime: CandidateRuntime
    context_length: int
    config_sha256: str
    tokenizer_sha256: str
    weight_files: tuple[CandidateFile, ...]
    gpu_memory_peak_gib: float
    inference_p95_ms: float
    max_concurrency: int
    storage_available_gib: float
    storage_required_free_gib: float
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
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]*", model_id):
            raise ValueError("candidate manifest model_id is not a safe identifier")
        lowered = model_id.lower()
        if any(marker in lowered for marker in LEGACY_MODEL_MARKERS):
            raise ValueError("candidate manifest refers to a legacy 14B/reviewer model")
        if model_id != TARGET_MODEL_ID:
            raise ValueError("candidate model_id must use the canonical Qwen3.8-27B service identity")
        repo_id = _clean_text(value.get("repo_id"), field="repo_id")
        if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9._-]+", repo_id):
            raise ValueError("candidate repo_id must identify an explicit repository")
        if any(marker in repo_id.lower() for marker in LEGACY_MODEL_MARKERS):
            raise ValueError("candidate repo_id refers to a legacy model")
        if repo_id != TARGET_REPOSITORY_ID:
            raise ValueError("candidate repo_id must be the approved Qwen3.8-27B repository")
        revision = _clean_text(value.get("revision"), field="revision")
        if revision.lower() in {"unknown", "unverified", "pending"} or not REVISION_PATTERN.fullmatch(revision):
            raise ValueError("candidate manifest revision is not a fixed immutable identifier")
        architecture = _clean_text(value.get("architecture"), field="architecture")
        if architecture != TARGET_ARCHITECTURE:
            raise ValueError("candidate architecture is not the approved Qwen3.8-27B architecture")
        model_type = _clean_text(value.get("model_type"), field="model_type")
        if model_type != TARGET_MODEL_TYPE:
            raise ValueError("candidate model_type is not the approved Qwen3.8-27B model type")
        language_model_only = value.get("language_model_only")
        if not isinstance(language_model_only, bool):
            raise ValueError("candidate manifest language_model_only must be boolean")
        text_inference_verified = value.get("text_inference_verified")
        if text_inference_verified is not True:
            raise ValueError("candidate text-only inference compatibility is not verified")
        runtime = CandidateRuntime.from_dict(value.get("runtime"))
        context_length = value.get("context_length")
        if (
            isinstance(context_length, bool)
            or not isinstance(context_length, int)
            or not 256 <= context_length <= 8192
        ):
            raise ValueError("candidate manifest context_length must be an integer from 256 to 8192")
        max_concurrency = value.get("max_concurrency")
        if max_concurrency != 1:
            raise ValueError("candidate manifest max_concurrency must remain one on the A100 host")
        peak = value.get("gpu_memory_peak_gib")
        p95 = value.get("inference_p95_ms")
        available_storage = value.get("storage_available_gib")
        required_storage = value.get("storage_required_free_gib")
        if isinstance(peak, bool) or not isinstance(peak, (int, float)) or not math.isfinite(peak) or peak <= 0:
            raise ValueError("candidate manifest gpu_memory_peak_gib must be positive")
        if isinstance(p95, bool) or not isinstance(p95, (int, float)) or not math.isfinite(p95) or p95 <= 0:
            raise ValueError("candidate manifest inference_p95_ms must be positive")
        if (
            isinstance(available_storage, bool)
            or not isinstance(available_storage, (int, float))
            or not math.isfinite(available_storage)
            or available_storage <= 0
        ):
            raise ValueError("candidate manifest storage_available_gib must be positive")
        if (
            isinstance(required_storage, bool)
            or not isinstance(required_storage, (int, float))
            or not math.isfinite(required_storage)
            or required_storage < 20
        ):
            raise ValueError("candidate manifest storage_required_free_gib must be at least 20")
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
            repo_id=repo_id,
            revision=revision,
            model_path=_absolute_model_path(value.get("model_path"), field="model_path"),
            tokenizer_path=_absolute_model_path(value.get("tokenizer_path"), field="tokenizer_path"),
            license=_clean_text(value.get("license"), field="license"),
            quantization=_clean_text(value.get("quantization"), field="quantization"),
            architecture=architecture,
            model_type=model_type,
            language_model_only=language_model_only,
            text_inference_verified=text_inference_verified,
            runtime=runtime,
            context_length=context_length,
            config_sha256=_sha256(value.get("config_sha256"), field="config_sha256"),
            tokenizer_sha256=_sha256(value.get("tokenizer_sha256"), field="tokenizer_sha256"),
            weight_files=files,
            gpu_memory_peak_gib=float(peak),
            inference_p95_ms=float(p95),
            max_concurrency=max_concurrency,
            storage_available_gib=float(available_storage),
            storage_required_free_gib=float(required_storage),
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
            else:
                now = datetime.now(UTC)
                if parsed > now + MAX_MANIFEST_FUTURE_SKEW:
                    errors.append("validated_at_in_future")
                elif now - parsed > MAX_MANIFEST_AGE:
                    errors.append("validated_at_stale")
        except ValueError:
            errors.append("validated_at_invalid")
        if not math.isfinite(self.gpu_memory_peak_gib) or self.gpu_memory_peak_gib > A100_MODEL_BUDGET_GIB:
            errors.append("gpu_memory_peak_exceeds_a100_budget")
        if not math.isfinite(self.inference_p95_ms) or self.inference_p95_ms > 30_000:
            errors.append("inference_p95_exceeds_30_seconds")
        if self.storage_available_gib < self.storage_required_free_gib:
            errors.append("storage_available_below_required_free_space")
        return tuple(errors)

    def to_topology(self, *, stage: str = "candidate_validated") -> ModelTopology:
        if stage not in {"candidate_validated", "paper"}:
            raise ValueError("verified candidate topology cannot enable live routing")
        evidence_errors = self.validate_evidence()
        if evidence_errors:
            raise ValueError("candidate evidence is invalid: " + ", ".join(evidence_errors))
        from core.model_topology import qwen27_candidate_topology

        topology = activate_verified_candidate(
            qwen27_candidate_topology(),
            model_id=self.model_id,
            repo_id=self.repo_id,
            revision=self.revision,
            path=self.model_path,
            stage=stage,
        )
        return replace(topology, models=(replace(
            topology.models[0], tokenizer=self.tokenizer_path,
            context_length=self.context_length, max_concurrency=self.max_concurrency,
            gpu_memory_budget_gib=A100_MODEL_BUDGET_GIB,
        ),))

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
            "architecture": self.architecture,
            "model_type": self.model_type,
            "language_model_only": self.language_model_only,
            "text_inference_verified": self.text_inference_verified,
            "runtime": self.runtime.to_dict(),
            "context_length": self.context_length,
            "config_sha256": self.config_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "weight_files": [item.to_dict() for item in self.weight_files],
            "gpu_memory_peak_gib": self.gpu_memory_peak_gib,
            "inference_p95_ms": self.inference_p95_ms,
            "max_concurrency": self.max_concurrency,
            "storage_available_gib": self.storage_available_gib,
            "storage_required_free_gib": self.storage_required_free_gib,
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
    # The model host has a small /data volume, so verified artifacts may reside
    # on the linux user's larger filesystem. Other absolute paths are refused.
    if posix_absolute and not path.startswith(ALLOWED_MODEL_ROOTS):
        raise ValueError(
            f"candidate manifest field {field!r} must stay under an approved model root"
        )
    return path.rstrip("/")
