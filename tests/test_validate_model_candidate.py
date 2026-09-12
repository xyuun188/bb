from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from scripts.validate_model_candidate import build_manifest


def _write_runtime_probe(args) -> None:
    payload = {
        "probe_version": "bb.model-runtime-probe.v1",
        "status": "verified",
        "observed_at": datetime.now(UTC).isoformat(),
        "model_id": args.model_id,
        "repo_id": args.repo_id,
        "revision": args.revision,
        "model_path": str(args.model_path.resolve()),
        "tokenizer_path": str((args.tokenizer_path or args.model_path).resolve()),
        "quantization": args.quantization,
        "architecture": "Qwen3_5ForConditionalGeneration",
        "model_type": "qwen3_5",
        "runtime_engine": args.runtime_engine,
        "runtime_engine_version": args.runtime_engine_version,
        "transformers_version": args.transformers_version,
        "context_length": args.context_length,
        "max_concurrency": args.max_concurrency,
        "gpu_memory_peak_gib": args.gpu_memory_peak_gib,
        "inference_p95_ms": args.inference_p95_ms,
        "text_inference_verified": True,
        "request_count": 20,
        "successful_request_count": 20,
        "oom_count": 0,
        "timeout_count": 0,
        "json_contract_failure_count": 0,
    }
    args.runtime_probe_file.write_text(json.dumps(payload), encoding="utf-8")


def test_candidate_validator_fingerprints_model_and_tokenizer(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        '{"model_type":"qwen3_5","architectures":["Qwen3_5ForConditionalGeneration"]}',
        encoding="utf-8",
    )
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")

    args = type(
        "Args",
        (),
        {
            "model_id": "qwen3.8-27b",
            "repo_id": "Qwen/Qwen3.8-27B",
            "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            "model_path": model,
            "tokenizer_path": None,
            "license": "apache-2.0",
            "quantization": "awq-int4",
            "context_length": 8192,
            "gpu_memory_peak_gib": 33.5,
            "inference_p95_ms": 1800,
            "max_concurrency": 1,
            "storage_required_free_gib": 20.0,
            "text_inference_verified": True,
            "runtime_engine": "vllm",
            "runtime_engine_version": "0.17.1",
            "transformers_version": "5.8.0",
            "runtime_probe_file": tmp_path / "runtime-probe.json",
            "validator_version": "bb-model-validator.v1",
        },
    )()
    _write_runtime_probe(args)

    manifest = build_manifest(args)

    assert manifest["status"] == "verified"
    assert manifest["language_model_only"] is False
    assert manifest["weight_files"][0]["path"] == "model.safetensors"
    assert len(manifest["config_sha256"]) == 64
    json.dumps(manifest)


def test_candidate_validator_rejects_legacy_model(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        '{"model_type":"qwen3_5","architectures":["Qwen3_5ForConditionalGeneration"]}',
        encoding="utf-8",
    )
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")

    args = type(
        "Args",
        (),
        {
            "model_id": "BB-FinQuant-Expert-14B",
            "repo_id": "legacy/repo",
            "revision": "sha256:" + "d" * 64,
            "model_path": model,
            "tokenizer_path": None,
            "license": "apache-2.0",
            "quantization": "awq-int4",
            "context_length": 8192,
            "gpu_memory_peak_gib": 33.5,
            "inference_p95_ms": 1800,
            "max_concurrency": 1,
            "storage_required_free_gib": 20.0,
            "text_inference_verified": True,
            "runtime_engine": "vllm",
            "runtime_engine_version": "0.17.1",
            "transformers_version": "5.8.0",
            "runtime_probe_file": tmp_path / "runtime-probe.json",
            "validator_version": "bb-model-validator.v1",
        },
    )()
    _write_runtime_probe(args)

    with pytest.raises(ValueError, match="legacy"):
        build_manifest(args)


def test_candidate_validator_rejects_probe_identity_or_measurement_drift(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        '{"model_type":"qwen3_5","architectures":["Qwen3_5ForConditionalGeneration"]}',
        encoding="utf-8",
    )
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    args = type(
        "Args",
        (),
        {
            "model_id": "qwen3.8-27b",
            "repo_id": "Qwen/Qwen3.8-27B",
            "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            "model_path": model,
            "tokenizer_path": None,
            "license": "apache-2.0",
            "quantization": "awq-int4",
            "context_length": 8192,
            "gpu_memory_peak_gib": 33.5,
            "inference_p95_ms": 1800,
            "max_concurrency": 1,
            "storage_required_free_gib": 20.0,
            "text_inference_verified": True,
            "runtime_engine": "vllm",
            "runtime_engine_version": "0.17.1",
            "transformers_version": "5.8.0",
            "runtime_probe_file": tmp_path / "runtime-probe.json",
            "validator_version": "bb-model-validator.v1",
        },
    )()
    _write_runtime_probe(args)
    probe = json.loads(args.runtime_probe_file.read_text(encoding="utf-8"))
    probe["model_id"] = "another-model"
    probe["gpu_memory_peak_gib"] = 10.0
    args.runtime_probe_file.write_text(json.dumps(probe), encoding="utf-8")

    with pytest.raises(ValueError, match="gpu_memory_peak_gib, model_id"):
        build_manifest(args)
