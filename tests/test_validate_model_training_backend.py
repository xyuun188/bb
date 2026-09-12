from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from scripts.validate_model_training_backend import build_manifest


def _inputs(tmp_path):
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(
        json.dumps(
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
        ),
        encoding="utf-8",
    )
    trainer = tmp_path / "train.py"
    trainer.write_text("print('trainer')\n", encoding="utf-8")
    probe = tmp_path / "probe.json"
    probe.write_text(
        json.dumps(
            {
                "probe_version": "bb.model-training-capability-probe.v1",
                "status": "verified",
                "observed_at": datetime.now(UTC).isoformat(),
                "backend_id": "qwen3_5_text_qlora",
                "model_id": "qwen3.8-27b",
                "repo_id": "Qwen/Qwen3.8-27B",
                "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                "architecture": "Qwen3_5ForConditionalGeneration",
                "model_type": "qwen3_5",
                "training_model_path": "/home/linux/trade_models/qwen3.8-27b-awq",
                "model_config_sha256": "a" * 64,
                "tokenizer_sha256": "b" * 64,
                "loader_class": "AutoModelForMultimodalLM",
                "quantization": "bitsandbytes-nf4",
                "compute_dtype": "bfloat16",
                "torch_version": "2.8.0",
                "transformers_version": "5.8.0",
                "peft_version": "0.18.0",
                "trl_version": "0.23.0",
                "bitsandbytes_version": "0.47.0",
                "text_forward_backward_verified": True,
                "dpo_step_verified": True,
                "adapter_save_reload_verified": True,
                "gpu_memory_peak_gib": 32.0,
                "storage_available_gib": 80.0,
                "storage_required_free_gib": 20.0,
            }
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(
        candidate_manifest=candidate_path,
        trainer_file=trainer,
        probe_file=probe,
        output=tmp_path / "backend.json",
        validator_version="test-validator.v1",
    )


def test_backend_validator_binds_probe_candidate_and_trainer(tmp_path) -> None:
    args = _inputs(tmp_path)

    manifest = build_manifest(args)

    assert manifest["manifest_version"] == "bb.model-training-backend.v2"
    assert manifest["loader_class"] == "AutoModelForMultimodalLM"
    assert manifest["model_config_sha256"] == "a" * 64
    assert manifest["trainer_sha256"] != manifest["runtime_probe_sha256"]


def test_backend_validator_rejects_probe_from_another_model(tmp_path) -> None:
    args = _inputs(tmp_path)
    probe = json.loads(args.probe_file.read_text(encoding="utf-8"))
    probe["revision"] = "0" * 40
    probe["adapter_save_reload_verified"] = False
    args.probe_file.write_text(json.dumps(probe), encoding="utf-8")

    with pytest.raises(ValueError, match="adapter_save_reload_verified, revision"):
        build_manifest(args)
