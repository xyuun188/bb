"""Verified training-backend contract for the Qwen3.8-27B migration.

Inference readiness is not evidence that a model can be trained.  This module
binds a model candidate to an independently executed QLoRA probe so training
orchestration can fail before opening SSH or stopping any service.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from core.model_candidate_manifest import (
    ALLOWED_MODEL_ROOTS,
    MIN_TRANSFORMERS_VERSION,
    TARGET_ARCHITECTURE,
    TARGET_MODEL_TYPE,
    TARGET_REPOSITORY_ID,
    ModelCandidateManifest,
    version_triplet,
)

TRAINING_BACKEND_MANIFEST_VERSION = "bb.model-training-backend.v2"
TRAINING_BACKEND_ID = "qwen3_5_text_qlora"
TRAINING_LOADER_CLASS = "AutoModelForMultimodalLM"
TRAINING_QUANTIZATION = "bitsandbytes-nf4"
TRAINING_COMPUTE_DTYPE = "bfloat16"
MAX_TRAINING_BACKEND_AGE = timedelta(days=7)
MAX_TRAINING_BACKEND_FUTURE_SKEW = timedelta(minutes=5)
MAX_TRAINING_GPU_MEMORY_GIB = 34.0
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _text(value: Any, *, field: str) -> str:
    result = value.strip() if isinstance(value, str) else ""
    if not result or any(ord(character) < 32 or ord(character) == 127 for character in result):
        raise ValueError(f"training backend field {field!r} is empty or unsafe")
    return result


def _sha256(value: Any, *, field: str) -> str:
    result = _text(value, field=field).lower()
    if not _SHA256_PATTERN.fullmatch(result):
        raise ValueError(f"training backend field {field!r} is not a SHA-256 digest")
    return result


def _absolute_model_path(value: Any, *, field: str) -> str:
    result = _text(value, field=field).rstrip("/")
    if not result.startswith(ALLOWED_MODEL_ROOTS):
        raise ValueError(f"training backend field {field!r} must use an approved model root")
    if any(part in {"", ".", ".."} for part in result.split("/")[1:]):
        raise ValueError(f"training backend field {field!r} is not a safe absolute path")
    return result


@dataclass(frozen=True, slots=True)
class ModelTrainingBackendManifest:
    manifest_version: str
    status: str
    backend_id: str
    model_id: str
    repo_id: str
    revision: str
    architecture: str
    model_type: str
    training_model_path: str
    model_config_sha256: str
    tokenizer_sha256: str
    loader_class: str
    quantization: str
    compute_dtype: str
    torch_version: str
    transformers_version: str
    peft_version: str
    trl_version: str
    bitsandbytes_version: str
    trainer_sha256: str
    runtime_probe_sha256: str
    text_forward_backward_verified: bool
    dpo_step_verified: bool
    adapter_save_reload_verified: bool
    gpu_memory_peak_gib: float
    storage_available_gib: float
    storage_required_free_gib: float
    validated_at: str
    validator_version: str

    @classmethod
    def from_dict(cls, value: Any) -> ModelTrainingBackendManifest:
        if not isinstance(value, dict):
            raise ValueError("training backend manifest must be a JSON object")
        manifest_version = _text(value.get("manifest_version"), field="manifest_version")
        if manifest_version != TRAINING_BACKEND_MANIFEST_VERSION:
            raise ValueError(f"unsupported training backend manifest version: {manifest_version}")
        status = _text(value.get("status"), field="status").lower()
        if status != "verified":
            raise ValueError("training backend status must be 'verified'")
        backend_id = _text(value.get("backend_id"), field="backend_id")
        if backend_id != TRAINING_BACKEND_ID:
            raise ValueError("training backend does not identify the approved Qwen3.5 QLoRA path")
        revision = _text(value.get("revision"), field="revision")
        if not _REVISION_PATTERN.fullmatch(revision):
            raise ValueError("training backend revision is not immutable")
        loader_class = _text(value.get("loader_class"), field="loader_class")
        if loader_class != TRAINING_LOADER_CLASS:
            raise ValueError("training backend loader is not Qwen3.5 conditional-generation capable")
        quantization = _text(value.get("quantization"), field="quantization")
        if quantization != TRAINING_QUANTIZATION:
            raise ValueError("training backend quantization must be bitsandbytes NF4 QLoRA")
        compute_dtype = _text(value.get("compute_dtype"), field="compute_dtype")
        if compute_dtype != TRAINING_COMPUTE_DTYPE:
            raise ValueError("training backend compute dtype must be bfloat16 on A100")
        proofs = {
            "text_forward_backward_verified": value.get("text_forward_backward_verified"),
            "dpo_step_verified": value.get("dpo_step_verified"),
            "adapter_save_reload_verified": value.get("adapter_save_reload_verified"),
        }
        missing_proofs = [name for name, verified in proofs.items() if verified is not True]
        if missing_proofs:
            raise ValueError("training backend capability proof missing: " + ", ".join(missing_proofs))
        peak = value.get("gpu_memory_peak_gib")
        storage_available = value.get("storage_available_gib")
        storage_required = value.get("storage_required_free_gib")
        for field, number in (
            ("gpu_memory_peak_gib", peak),
            ("storage_available_gib", storage_available),
            ("storage_required_free_gib", storage_required),
        ):
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(number)
                or number <= 0
            ):
                raise ValueError(f"training backend {field} must be a positive finite number")
        versions = {}
        for field in (
            "torch_version",
            "transformers_version",
            "peft_version",
            "trl_version",
            "bitsandbytes_version",
        ):
            version = _text(value.get(field), field=field)
            if version.lower() in {"unknown", "pending", "unverified"}:
                raise ValueError(f"training backend {field} is not verified")
            versions[field] = version
        if version_triplet(versions["transformers_version"]) < MIN_TRANSFORMERS_VERSION:
            raise ValueError("training backend Transformers runtime is too old for Qwen3.8-27B")
        return cls(
            manifest_version=manifest_version,
            status=status,
            backend_id=backend_id,
            model_id=_text(value.get("model_id"), field="model_id"),
            repo_id=_text(value.get("repo_id"), field="repo_id"),
            revision=revision,
            architecture=_text(value.get("architecture"), field="architecture"),
            model_type=_text(value.get("model_type"), field="model_type"),
            training_model_path=_absolute_model_path(
                value.get("training_model_path"), field="training_model_path"
            ),
            model_config_sha256=_sha256(
                value.get("model_config_sha256"), field="model_config_sha256"
            ),
            tokenizer_sha256=_sha256(
                value.get("tokenizer_sha256"), field="tokenizer_sha256"
            ),
            loader_class=loader_class,
            quantization=quantization,
            compute_dtype=compute_dtype,
            trainer_sha256=_sha256(value.get("trainer_sha256"), field="trainer_sha256"),
            runtime_probe_sha256=_sha256(
                value.get("runtime_probe_sha256"), field="runtime_probe_sha256"
            ),
            text_forward_backward_verified=True,
            dpo_step_verified=True,
            adapter_save_reload_verified=True,
            gpu_memory_peak_gib=float(peak),
            storage_available_gib=float(storage_available),
            storage_required_free_gib=float(storage_required),
            validated_at=_text(value.get("validated_at"), field="validated_at"),
            validator_version=_text(value.get("validator_version"), field="validator_version"),
            **versions,
        )

    @classmethod
    def load(cls, path: str | Path) -> ModelTrainingBackendManifest:
        source = Path(path)
        if not source.is_file():
            raise ValueError(f"training backend manifest does not exist: {source}")
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read training backend manifest: {source}") from exc
        return cls.from_dict(payload)

    def validate_for_candidate(
        self,
        candidate: ModelCandidateManifest,
        *,
        trainer_sha256: str,
    ) -> tuple[str, ...]:
        errors: list[str] = []
        if self.model_id != candidate.model_id:
            errors.append("training_backend_model_id_mismatch")
        if self.repo_id != candidate.repo_id or self.repo_id != TARGET_REPOSITORY_ID:
            errors.append("training_backend_repo_mismatch")
        if self.revision != candidate.revision:
            errors.append("training_backend_revision_mismatch")
        if self.architecture != candidate.architecture or self.architecture != TARGET_ARCHITECTURE:
            errors.append("training_backend_architecture_mismatch")
        if self.model_type != candidate.model_type or self.model_type != TARGET_MODEL_TYPE:
            errors.append("training_backend_model_type_mismatch")
        if self.training_model_path != candidate.model_path:
            errors.append("training_backend_model_path_mismatch")
        if self.model_config_sha256 != candidate.config_sha256:
            errors.append("training_backend_model_config_hash_mismatch")
        if self.tokenizer_sha256 != candidate.tokenizer_sha256:
            errors.append("training_backend_tokenizer_hash_mismatch")
        if self.trainer_sha256 != trainer_sha256:
            errors.append("training_backend_trainer_hash_mismatch")
        try:
            validated_at = datetime.fromisoformat(self.validated_at.replace("Z", "+00:00"))
        except ValueError:
            errors.append("training_backend_validated_at_invalid")
        else:
            if validated_at.tzinfo is None:
                errors.append("training_backend_validated_at_missing_timezone")
            else:
                now = datetime.now(UTC)
                if validated_at > now + MAX_TRAINING_BACKEND_FUTURE_SKEW:
                    errors.append("training_backend_validated_at_in_future")
                elif now - validated_at > MAX_TRAINING_BACKEND_AGE:
                    errors.append("training_backend_evidence_stale")
        if self.gpu_memory_peak_gib > MAX_TRAINING_GPU_MEMORY_GIB:
            errors.append("training_backend_gpu_memory_exceeds_a100_budget")
        if self.storage_available_gib < self.storage_required_free_gib:
            errors.append("training_backend_storage_below_required_free_space")
        return tuple(errors)

    def to_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}
