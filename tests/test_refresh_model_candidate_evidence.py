from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from core.model_candidate_manifest import MANIFEST_VERSION, ModelCandidateManifest
from scripts import refresh_model_candidate_evidence as refresh


def _payload(*, validated_at: datetime) -> dict:
    return {
        "manifest_version": MANIFEST_VERSION,
        "status": "verified",
        "model_id": "qwen3.8-27b",
        "repo_id": "Qwen/Qwen3.8-27B",
        "revision": "1" * 40,
        "model_path": "/data/BB/models/qwen3.8-27b",
        "tokenizer_path": "/data/BB/models/qwen3.8-27b",
        "license": "apache-2.0",
        "quantization": "bitsandbytes-nf4",
        "architecture": "Qwen3_5ForConditionalGeneration",
        "model_type": "qwen3_5",
        "language_model_only": False,
        "text_inference_verified": True,
        "runtime": {
            "engine": "transformers",
            "engine_version": "5.8.1",
            "transformers_version": "5.8.1",
            "probe_sha256": "a" * 64,
        },
        "context_length": 4096,
        "config_sha256": "b" * 64,
        "tokenizer_sha256": "c" * 64,
        "weight_files": [{"path": "model.safetensors", "size_bytes": 1, "sha256": "d" * 64}],
        "gpu_memory_peak_gib": 24.0,
        "inference_p95_ms": 1800.0,
        "max_concurrency": 1,
        "storage_available_gib": 34.0,
        "storage_required_free_gib": 20.0,
        "validated_at": validated_at.isoformat(),
        "validator_version": "bb-model-validator.v1",
    }


def _args(tmp_path: Path, *, force: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        candidate_manifest=tmp_path / "candidate.json",
        probe_output=tmp_path / "probe.json",
        endpoint="http://127.0.0.1:8000",
        max_age_hours=120.0,
        request_count=20,
        request_timeout_seconds=180.0,
        long_prompt_tokens=4096,
        force=force,
    )


def test_refresh_required_uses_pre_expiry_threshold() -> None:
    now = datetime.now(UTC)
    fresh = ModelCandidateManifest.from_dict(_payload(validated_at=now - timedelta(hours=10)))
    due = ModelCandidateManifest.from_dict(_payload(validated_at=now - timedelta(hours=121)))

    assert refresh.refresh_required(fresh, max_age_hours=120, now=now) is False
    assert refresh.refresh_required(due, max_age_hours=120, now=now) is True


def test_fresh_candidate_does_not_call_probe(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.candidate_manifest.write_text(
        json.dumps(_payload(validated_at=datetime.now(UTC))), encoding="utf-8"
    )

    result = refresh.refresh_candidate_evidence(
        args, probe_runner=lambda _args: pytest.fail("fresh evidence must not be probed")
    )

    assert result["status"] == "fresh"
    assert result["refreshed"] is False


def test_failed_probe_preserves_existing_candidate(tmp_path: Path) -> None:
    args = _args(tmp_path)
    original = _payload(validated_at=datetime.now(UTC) - timedelta(days=8))
    args.candidate_manifest.write_text(json.dumps(original), encoding="utf-8")

    with pytest.raises(RuntimeError, match="did not verify"):
        refresh.refresh_candidate_evidence(
            args,
            probe_runner=lambda _args: {"status": "failed"},
            manifest_builder=lambda _args: pytest.fail("failed probe must not validate"),
        )

    assert json.loads(args.candidate_manifest.read_text(encoding="utf-8")) == original
    assert not args.probe_output.exists()


def test_verified_refresh_atomically_updates_candidate_and_probe(tmp_path: Path) -> None:
    args = _args(tmp_path)
    original = _payload(validated_at=datetime.now(UTC) - timedelta(days=8))
    args.candidate_manifest.write_text(json.dumps(original), encoding="utf-8")
    new_time = datetime.now(UTC).isoformat()
    probe = {
        "status": "verified",
        "runtime_engine_version": "5.8.1",
        "transformers_version": "5.8.1",
        "gpu_memory_peak_gib": 23.5,
        "inference_p95_ms": 1750.0,
        "request_count": 20,
    }

    def build(_validator_args):
        return {**original, "validated_at": new_time}

    result = refresh.refresh_candidate_evidence(
        args, probe_runner=lambda _args: probe, manifest_builder=build
    )

    assert result["status"] == "refreshed"
    assert json.loads(args.candidate_manifest.read_text(encoding="utf-8"))["validated_at"] == new_time
    assert json.loads(args.probe_output.read_text(encoding="utf-8"))["status"] == "verified"
