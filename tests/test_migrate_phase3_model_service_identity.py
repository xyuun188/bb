import json

import pytest

from core.model_candidate_manifest import ModelCandidateManifest
from scripts import migrate_phase3_model_service_identity as migration
from scripts.migrate_phase3_model_service_identity import (
    RETIRED_MODEL_SERVICES,
    TARGET_SERVICE,
    main,
    render_target_migration,
    target_service_manifest,
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
                {
                    "path": "model.safetensors",
                    "size_bytes": 1,
                    "sha256": "c" * 64,
                }
            ],
            "gpu_memory_peak_gib": 33.5,
            "inference_p95_ms": 1800,
            "max_concurrency": 1,
            "storage_available_gib": 85.0,
            "storage_required_free_gib": 20.0,
            "validated_at": "2026-09-10T08:00:00Z",
            "validator_version": "bb-model-validator.v1",
        }
    )


def _write_candidate(tmp_path, candidate: ModelCandidateManifest | None = None):
    path = tmp_path / "candidate.json"
    payload = (candidate or _candidate()).to_dict()
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_target_manifest_contains_only_one_verified_local_model() -> None:
    candidate = _candidate()
    payload = target_service_manifest(candidate)

    assert payload["topology_profile"] == "target_single_model"
    assert payload["candidate_model_id"] == candidate.model_id
    assert payload["shadow_only"] is True
    assert payload["live_routing_enabled"] is False
    assert payload["can_start_trading"] is False
    assert len(payload["services"]) == 1
    assert payload["services"][0] == {
        "slot": "llm_decision_and_expert_carrier",
        "role": "decision_and_expert_carrier",
        "service_name": TARGET_SERVICE,
        "served_model_name": candidate.model_id,
        "model_dir": candidate.model_path,
        "tokenizer_dir": candidate.tokenizer_path,
        "port": 8000,
        "max_model_len": candidate.context_length,
        "max_num_seqs": candidate.max_concurrency,
        "shadow_only": True,
        "live_routing_enabled": False,
    }


def test_target_migration_is_transactional_and_conflicts_with_every_retired_service() -> None:
    candidate = _candidate()
    rendered = render_target_migration(candidate)

    assert "deploy_target" in rendered
    assert "target_model_candidate.json" in rendered
    assert "qwen38-probe.pid" in rendered
    assert candidate.model_id in rendered
    assert TARGET_SERVICE in rendered
    assert "rm -rf" not in rendered
    for service in RETIRED_MODEL_SERVICES:
        assert service in rendered


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--profile", "legacy_shadow"], "legacy_shadow deployment"),
        (["--sync-control-manifests"], "control-manifest sync is retired"),
        ([], "requires --candidate-manifest"),
    ],
)
def test_main_rejects_retired_or_incomplete_deployment_modes(argv, message) -> None:
    with pytest.raises(SystemExit):
        main(argv)


def test_plan_mode_validates_candidate_without_connecting_ssh(
    tmp_path, monkeypatch, capsys
) -> None:
    path = _write_candidate(tmp_path)

    def unexpected_connect(*args, **kwargs):
        raise AssertionError("plan mode must not connect to the model host")

    monkeypatch.setattr(migration, "connect_remote_ssh", unexpected_connect)

    assert main(["--candidate-manifest", str(path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["candidate_model_id"] == "qwen3.8-27b"
    assert output["live_routing_enabled"] is False


def test_main_rejects_unverified_candidate_before_connecting_ssh(
    tmp_path, monkeypatch
) -> None:
    payload = _candidate().to_dict()
    payload["status"] = "draft"
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    def unexpected_connect(*args, **kwargs):
        raise AssertionError("invalid candidate must not connect to the model host")

    monkeypatch.setattr(migration, "connect_remote_ssh", unexpected_connect)

    with pytest.raises(ValueError, match="status"):
        main(["--apply", "--candidate-manifest", str(path)])
