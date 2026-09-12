from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from core.remote_server_info import RemoteServerInfo
from scripts import run_phase3_model_server_readiness_audit as readiness_cli
from services.phase3_model_server_readiness import (
    Phase3ModelServerReadinessAuditService,
    evaluate_phase3_model_server_snapshot,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]




def _target_candidate() -> dict[str, Any]:
    return {
        "manifest_version": "bb.model-candidate.v2",
        "status": "verified",
        "model_id": "qwen3.8-27b",
        "repo_id": "Qwen/Qwen3.8-27B",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "model_path": "/home/linux/trade_models/qwen3.8-27b-awq",
        "tokenizer_path": "/home/linux/trade_models/qwen3.8-27b-awq",
        "license": "apache-2.0",
        "quantization": "awq",
        "architecture": "Qwen3_5ForConditionalGeneration",
        "model_type": "qwen3_5",
        "language_model_only": False,
        "text_inference_verified": True,
        "runtime": {
            "engine": "vllm",
            "engine_version": "0.17.1",
            "transformers_version": "5.8.0",
            "probe_sha256": "d" * 64,
        },
        "context_length": 8192,
        "config_sha256": "a" * 64,
        "tokenizer_sha256": "b" * 64,
        "weight_files": [
            {"path": "model.safetensors", "size_bytes": 1, "sha256": "c" * 64}
        ],
        "gpu_memory_peak_gib": 32.0,
        "inference_p95_ms": 950.0,
        "max_concurrency": 1,
        "storage_available_gib": 80.0,
        "storage_required_free_gib": 20.0,
        "validated_at": datetime.now(UTC).isoformat(),
        "validator_version": "test-validator.v1",
    }


def _target_ready_snapshot() -> dict[str, Any]:
    model_id = "qwen3.8-27b"
    service_name = "bb-phase3-llm-target.service"
    return {
        "topology_profile": "target_single_model",
        "target_candidate_manifest": {"present": True, "data": _target_candidate()},
        "service_manifest": {
            "present": True,
            "data": {
                "topology_profile": "target_single_model",
                "services": [
                    {
                        "slot": "llm_decision_and_expert_carrier",
                        "role": "decision_and_expert_carrier",
                        "service_name": service_name,
                        "served_model_name": model_id,
                        "port": 8000,
                        "shadow_only": True,
                        "live_routing_enabled": False,
                    }
                ],
            },
        },
        "services": [
            f"{service_name} loaded active running BB target single model",
        ],
        "port_probes": [
            {
                "port": 8000,
                "path": "/v1/models",
                "ok": True,
                "response": json.dumps({"data": [{"id": model_id}]}),
            }
        ],
        "gpu": ["0, NVIDIA A100-SXM4-40GB, 30000, 40960, 0, 36"],
        "gpu_processes": ["GPU-0, 1001, python, 32000"],
        "validation_manifest": {
            "present": True,
            "data": {"torch": {"cuda_available": True, "device_count": 1}},
        },
        "download_manifest": {"present": True, "data": {}},
    }


def test_target_single_model_readiness_accepts_verified_runtime_contract() -> None:
    report = evaluate_phase3_model_server_snapshot(_target_ready_snapshot())

    assert report["status"] == "ready"
    assert report["artifact_ready"] is True
    assert report["runtime_ready"] is True
    assert report["blockers"] == []
    assert report["target_model_topology"]["blockers"] == []
    assert report["target_model_topology"]["model_id"] == "qwen3.8-27b"


def test_target_readiness_surfaces_missing_candidate_in_final_topology_blockers() -> None:
    snapshot = _target_ready_snapshot()
    snapshot["target_candidate_manifest"] = {"present": False, "data": {}}

    report = evaluate_phase3_model_server_snapshot(snapshot)
    blocker_codes = {item["code"] for item in report["blockers"]}

    assert report["status"] == "blocked"
    assert "target_candidate_manifest_missing" in blocker_codes
    assert "target_candidate_manifest_missing" in report["target_model_topology"]["blockers"]


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("architecture", "Qwen2ForCausalLM", "target_candidate_manifest_invalid"),
        ("model_type", "qwen2", "target_candidate_manifest_invalid"),
        (
            "runtime",
            {
                "engine": "vllm",
                "engine_version": "unverified",
                "transformers_version": "5.8.0",
                "probe_sha256": "d" * 64,
            },
            "target_candidate_manifest_invalid",
        ),
    ],
)
def test_target_readiness_blocks_candidate_identity_and_runtime_mismatches(
    field: str, value: Any, expected: str
) -> None:
    snapshot = _target_ready_snapshot()
    snapshot["target_candidate_manifest"]["data"][field] = value

    report = evaluate_phase3_model_server_snapshot(snapshot)

    assert report["status"] == "blocked"
    assert any(expected in item["code"] or expected in str(item.get("evidence")) for item in report["blockers"])


def test_target_readiness_blocks_service_contract_and_legacy_services() -> None:
    snapshot = _target_ready_snapshot()
    service = snapshot["service_manifest"]["data"]["services"][0]
    service["service_name"] = "bb-phase3-llm-decision.service"
    snapshot["services"].append(
        "bb-phase3-llm-risk-review.service loaded active running retired reviewer"
    )

    report = evaluate_phase3_model_server_snapshot(snapshot)
    blocker_codes = {item["code"] for item in report["blockers"]}

    assert {
        "target_service_contract_mismatch",
        "legacy_model_services_active",
    } <= blocker_codes
    assert report["target_model_topology"]["blockers"]


def test_target_readiness_blocks_endpoint_and_gpu_evidence() -> None:
    snapshot = _target_ready_snapshot()
    snapshot["port_probes"][0]["response"] = '{"data":[{"id":"wrong-model"}]}'
    snapshot["gpu"] = []
    snapshot["validation_manifest"]["data"]["torch"]["device_count"] = 0

    report = evaluate_phase3_model_server_snapshot(snapshot)
    blocker_codes = {item["code"] for item in report["blockers"]}

    assert {"target_endpoint_not_ready", "gpu_count_below_target_contract"} <= blocker_codes


@pytest.mark.asyncio
async def test_target_readiness_service_uses_injected_probe() -> None:
    report = await Phase3ModelServerReadinessAuditService(
        remote_probe=lambda: _target_ready_snapshot()
    ).report()

    assert report["status"] == "ready"
    assert report["remote_probe_available"] is True


@pytest.mark.asyncio
async def test_target_readiness_service_awaits_async_loader_and_closes_ssh() -> None:
    calls: list[str] = []
    info = RemoteServerInfo(
        host="203.0.113.12",
        port=22,
        username="root",
        password="secret",
        source_path=Path("<test>"),
    )

    async def async_info_loader(_root: Path) -> RemoteServerInfo:
        calls.append("loader")
        return info

    class FakeSsh:
        def close(self) -> None:
            calls.append("close")

    def ssh_connector(_root: Path, **kwargs: Any) -> FakeSsh:
        calls.append(f"ssh:{kwargs['info'].host}")
        return FakeSsh()

    class FakeResult:
        status = 0
        stdout = json.dumps(_target_ready_snapshot())
        stderr = ""

    def command_executor(_ssh: FakeSsh, _command: str, **_kwargs: Any) -> FakeResult:
        calls.append("command")
        return FakeResult()

    report = await Phase3ModelServerReadinessAuditService(
        async_info_loader=async_info_loader,
        ssh_connector=ssh_connector,
        command_executor=command_executor,
    ).report()

    assert report["status"] == "ready"
    assert calls == ["loader", "ssh:203.0.113.12", "command", "close"]




def test_phase3_model_server_readiness_writes_dated_and_latest_report(tmp_path) -> None:
    report = {
        "status": "artifact_ready_service_pending",
        "checked_at": "2026-06-27T00:45:00+00:00",
        "read_only": True,
        "phase3_model_service_go_live_blocked": True,
    }

    artifacts = readiness_cli.write_report(report, tmp_path, indent=2)

    report_path = tmp_path / artifacts["report_path"].split("\\")[-1]
    latest_path = tmp_path / "latest.json"
    assert report_path.exists()
    assert latest_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["report_artifacts"] == artifacts
    assert latest_path.read_text(encoding="utf-8") == report_path.read_text(encoding="utf-8")


def test_phase3_model_server_readiness_preserves_verified_latest_on_config_error(
    tmp_path,
) -> None:
    latest_path = tmp_path / "latest.json"
    verified = {
        "status": "ready",
        "checked_at": "2026-06-28T00:57:21+00:00",
        "artifact_ready": True,
        "runtime_ready": True,
        "phase3_model_service_go_live_blocked": False,
    }
    latest_path.write_text(json.dumps(verified), encoding="utf-8")
    report = {
        "status": "unverified",
        "checked_at": "2026-06-28T19:23:41+00:00",
        "artifact_ready": False,
        "runtime_ready": False,
        "phase3_model_service_go_live_blocked": True,
        "error": "BB_SECURE_SETTINGS_KEY is required for encrypted settings",
        "blockers": [
            {
                "code": "model_server_config_error",
                "message": "Phase 3 model-server artifact/runtime readiness could not be verified.",
                "severity": "blocking",
            }
        ],
    }

    artifacts = readiness_cli.write_report(report, tmp_path, indent=2)

    report_path = Path(artifacts["report_path"])
    assert report_path.exists()
    assert artifacts["latest_preserved"] is True
    assert json.loads(latest_path.read_text(encoding="utf-8")) == verified
    dated_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert dated_payload["latest_preserved_reason"] == (
        "config_environment_error_did_not_overwrite_last_verified_latest"
    )


def test_phase3_model_server_readiness_cli_imports_online_runtime_bootstrap() -> None:
    source = readiness_cli.ROOT.joinpath(
        "scripts",
        "run_phase3_model_server_readiness_audit.py",
    ).read_text(encoding="utf-8")

    assert "from scripts.runtime_env_bootstrap import" in source
    assert "load_runtime_env_files(project_root=ROOT)" in source
    assert "drop_privileges_to_runtime_user_if_needed(project_root=ROOT)" in source
