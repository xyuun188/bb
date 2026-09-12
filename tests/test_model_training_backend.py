from __future__ import annotations

from datetime import UTC, datetime

import pytest

from core.model_candidate_manifest import ModelCandidateManifest
from core.model_training_backend import (
    TRAINING_LOADER_CLASS,
    ModelTrainingBackendManifest,
)


def _candidate() -> ModelCandidateManifest:
    return ModelCandidateManifest.from_dict(
        {
            "manifest_version": "bb.model-candidate.v2",
            "status": "verified",
            "model_id": "qwen3.8-27b",
            "repo_id": "Qwen/Qwen3.8-27B",
            "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            "model_path": "/home/linux/trade_models/qwen3.8-27b-awq",
            "tokenizer_path": "/home/linux/trade_models/qwen3.8-27b-awq",
            "license": "apache-2.0",
            "quantization": "awq-int4",
            "architecture": "Qwen3_5ForConditionalGeneration",
            "model_type": "qwen3_5",
            "language_model_only": False,
            "text_inference_verified": True,
            "runtime": {
                "engine": "vllm",
                "engine_version": "0.17.1",
                "transformers_version": "5.8.0",
                "probe_sha256": "e" * 64,
            },
            "context_length": 8192,
            "config_sha256": "a" * 64,
            "tokenizer_sha256": "b" * 64,
            "weight_files": [
                {"path": "model.safetensors", "size_bytes": 1, "sha256": "c" * 64}
            ],
            "gpu_memory_peak_gib": 32.0,
            "inference_p95_ms": 1800.0,
            "max_concurrency": 1,
            "storage_available_gib": 80.0,
            "storage_required_free_gib": 20.0,
            "validated_at": datetime.now(UTC).isoformat(),
            "validator_version": "test.v1",
        }
    )


def _backend(**overrides) -> dict:
    value = {
        "manifest_version": "bb.model-training-backend.v2",
        "status": "verified",
        "backend_id": "qwen3_5_text_qlora",
        "model_id": "qwen3.8-27b",
        "repo_id": "Qwen/Qwen3.8-27B",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "architecture": "Qwen3_5ForConditionalGeneration",
        "model_type": "qwen3_5",
        "training_model_path": "/home/linux/trade_models/qwen3.8-27b-awq",
        "model_config_sha256": "a" * 64,
        "tokenizer_sha256": "b" * 64,
        "loader_class": TRAINING_LOADER_CLASS,
        "quantization": "bitsandbytes-nf4",
        "compute_dtype": "bfloat16",
        "torch_version": "2.8.0",
        "transformers_version": "5.8.0",
        "peft_version": "0.18.0",
        "trl_version": "0.23.0",
        "bitsandbytes_version": "0.47.0",
        "trainer_sha256": "f" * 64,
        "runtime_probe_sha256": "d" * 64,
        "text_forward_backward_verified": True,
        "dpo_step_verified": True,
        "adapter_save_reload_verified": True,
        "gpu_memory_peak_gib": 32.0,
        "storage_available_gib": 80.0,
        "storage_required_free_gib": 20.0,
        "validated_at": datetime.now(UTC).isoformat(),
        "validator_version": "test.v1",
    }
    value.update(overrides)
    return value


def test_backend_accepts_official_multimodal_loader_for_text_qlora() -> None:
    backend = ModelTrainingBackendManifest.from_dict(_backend())

    assert TRAINING_LOADER_CLASS == "AutoModelForMultimodalLM"
    assert backend.validate_for_candidate(_candidate(), trainer_sha256="f" * 64) == ()


def test_backend_rejects_old_image_text_loader_and_identity_drift() -> None:
    with pytest.raises(ValueError, match="loader"):
        ModelTrainingBackendManifest.from_dict(
            _backend(loader_class="AutoModelForImageTextToText")
        )

    backend = ModelTrainingBackendManifest.from_dict(_backend(model_id="wrong-model"))
    assert "training_backend_model_id_mismatch" in backend.validate_for_candidate(
        _candidate(), trainer_sha256="f" * 64
    )
