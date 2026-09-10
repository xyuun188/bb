from __future__ import annotations

import json

import pytest

from scripts.validate_model_candidate import build_manifest


def test_candidate_validator_fingerprints_model_and_tokenizer(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen"}', encoding="utf-8")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")

    args = type(
        "Args",
        (),
        {
            "model_id": "qwen3.8-27b-awq",
            "repo_id": "verified/qwen3.8-27b-awq",
            "revision": "sha256:" + "d" * 64,
            "model_path": model,
            "tokenizer_path": None,
            "license": "apache-2.0",
            "quantization": "awq-int4",
            "context_length": 8192,
            "gpu_memory_peak_gib": 33.5,
            "inference_p95_ms": 1800,
            "max_concurrency": 1,
            "validator_version": "bb-model-validator.v1",
        },
    )()

    manifest = build_manifest(args)

    assert manifest["status"] == "verified"
    assert manifest["weight_files"][0]["path"] == "model.safetensors"
    assert len(manifest["config_sha256"]) == 64
    json.dumps(manifest)


def test_candidate_validator_rejects_legacy_model(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
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
            "validator_version": "bb-model-validator.v1",
        },
    )()

    with pytest.raises(ValueError, match="legacy"):
        build_manifest(args)
