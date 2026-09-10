from __future__ import annotations

import json

import pytest

from core.model_candidate_manifest import MANIFEST_VERSION, ModelCandidateManifest
from core.model_topology import target_topology_ready


def _manifest(**overrides):
    value = {
        "manifest_version": MANIFEST_VERSION,
        "status": "verified",
        "model_id": "qwen3.8-27b-awq",
        "repo_id": "verified/qwen3.8-27b-awq",
        "revision": "sha256:" + "d" * 64,
        "model_path": "/data/trade_models/verified/qwen3.8-27b-awq",
        "tokenizer_path": "/data/trade_models/verified/qwen3.8-27b-awq",
        "license": "apache-2.0",
        "quantization": "awq-int4",
        "context_length": 8192,
        "config_sha256": "a" * 64,
        "tokenizer_sha256": "b" * 64,
        "weight_files": [
            {"path": "model-00001.safetensors", "size_bytes": 123, "sha256": "c" * 64},
        ],
        "gpu_memory_peak_gib": 33.5,
        "inference_p95_ms": 1800,
        "max_concurrency": 1,
        "validated_at": "2026-09-10T08:00:00Z",
        "validator_version": "bb-model-validator.v1",
    }
    value.update(overrides)
    return value


def test_verified_manifest_builds_non_live_single_model_topology():
    manifest = ModelCandidateManifest.from_dict(_manifest())
    topology = manifest.to_topology(stage="paper")

    assert target_topology_ready(topology)
    assert topology.profile == "target_single_model"
    assert topology.models[0].model_id == "qwen3.8-27b-awq"
    assert topology.live_routing_enabled is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "candidate"),
        ("weight_files", []),
        ("config_sha256", "not-a-hash"),
        ("model_id", "deepseek-r1-14b-risk"),
    ],
)
def test_manifest_rejects_unverified_or_legacy_candidates(field, value):
    payload = _manifest(**{field: value})
    with pytest.raises(ValueError):
        ModelCandidateManifest.from_dict(payload)


def test_manifest_load_round_trip(tmp_path):
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(_manifest()), encoding="utf-8")

    loaded = ModelCandidateManifest.load(path)

    assert loaded.to_dict()["manifest_version"] == MANIFEST_VERSION
    assert loaded.weight_files[0].path == "model-00001.safetensors"


def test_manifest_rejects_incomplete_resource_evidence():
    manifest = ModelCandidateManifest.from_dict(
        _manifest(gpu_memory_peak_gib=39.0, inference_p95_ms=31_000)
    )

    with pytest.raises(ValueError, match="candidate evidence is invalid"):
        manifest.to_topology()
