from __future__ import annotations

import pytest

from pathlib import Path

from core.remote_server_info import RemoteServerInfo
from services.phase3_server_migration_audit import (
    PHASE3_RESOURCE_POLICY_ID,
    Phase3ServerMigrationAuditService,
    evaluate_phase3_server_snapshot,
    render_phase3_server_probe,
)


def _ready_snapshot() -> dict:
    return {
        "resource_release_marker": {
            "present": True,
            "data": {
                "policy_id": PHASE3_RESOURCE_POLICY_ID,
                "legacy_resources_stopped": True,
                "old_data_preserved": True,
                "phase3_root": "/data/BB",
            },
        },
        "migration_manifest": {
            "present": True,
            "data": {
                "whitelist_only": True,
                "items": [
                    {
                        "category": "clean_training_export_manifest",
                        "source": "platform_clean_training_view",
                        "path": "/data/exports/clean-training.jsonl",
                    }
                ],
            },
        },
        "forbidden_paths": [
            {"path": "/data/trade_models/Qwen/Qwen3.5-122B", "exists": True}
        ],
        "phase3_roots": [
            {"path": "/data/BB", "exists": True},
            {"path": "/data/BB/models", "exists": True},
        ],
        "forbidden_services": [
            {
                "name": "qwen3-32b-main.service",
                "unit_exists": False,
                "active": False,
                "enabled": False,
            }
        ],
        "legacy_processes": [],
        "approved_roots": [{"path": "/data/BB", "exists": True}],
    }


def test_phase3_server_migration_snapshot_ready() -> None:
    report = evaluate_phase3_server_snapshot(_ready_snapshot())

    assert report["status"] == "ready"
    assert report["phase3_go_live_blocked"] is False
    assert report["can_delete_remote_data"] is False
    assert report["migration_manifest"]["item_count"] == 1
    assert report["blockers"] == []
    assert report["legacy_data_path_count"] == 1
    assert report["warnings"][0]["code"] == "legacy_data_paths_preserved"


def test_phase3_server_migration_allows_phase3_vllm_under_data_bb() -> None:
    snapshot = _ready_snapshot()
    process = (
        "324221 /data/BB/envs/phase3-quant/bin/python -m "
        "vllm.entrypoints.openai.api_server --model "
        "/data/BB/models/llm_decision_maker/Qwen--Qwen3-32B-AWQ "
        "--served-model-name qwen3-32b-trade --port 8000"
    )
    snapshot["legacy_processes"] = [process]

    report = evaluate_phase3_server_snapshot(snapshot)

    assert report["status"] == "ready"
    assert report["phase3_go_live_blocked"] is False
    assert report["legacy_process_count"] == 0
    assert report["phase3_allowed_process_count"] == 1
    assert report["phase3_allowed_processes"] == [process]


def test_phase3_server_migration_blocks_legacy_processes_outside_data_bb() -> None:
    snapshot = _ready_snapshot()
    snapshot["candidate_model_processes"] = [
        "777 python -m vllm.entrypoints.openai.api_server "
        "--model /data/trade_models/Qwen/Qwen3-32B-AWQ --port 8000",
        "888 python -m vllm.entrypoints.openai.api_server "
        "--model /data/BB/models/old/Qwen3.5-122B --port 8001",
        "999 /usr/local/bin/open-webui serve --model-dir /data/BB/models",
    ]

    report = evaluate_phase3_server_snapshot(snapshot)
    codes = {item["code"] for item in report["blockers"]}

    assert report["status"] == "blocked"
    assert "legacy_processes_running" in codes
    assert report["legacy_process_count"] == 3
    assert report["phase3_allowed_process_count"] == 0


def test_phase3_server_migration_ignores_self_audit_probe_processes() -> None:
    snapshot = _ready_snapshot()
    process = (
        "2077059 bash -c python3 - <<'PY' import json import os import subprocess "
        'DOWNLOAD_MANIFEST_PATH = "/data/BB/manifests/phase3_model_download_manifest.json" '
        'SERVICE_MANIFEST_PATH = "/data/BB/manifests/phase3_model_service_manifest.json" '
        'MODEL_RUNTIME_PORTS = [8000, 8001, 8002] '
        '"Qwen3-32B|Qwen3.5-122B|trade_ai|trade_models|vllm"'
    )
    snapshot["candidate_model_processes"] = [process]

    report = evaluate_phase3_server_snapshot(snapshot)

    assert report["status"] == "ready"
    assert report["phase3_go_live_blocked"] is False
    assert report["legacy_process_count"] == 0
    assert report["ignored_probe_process_count"] == 1
    assert report["ignored_probe_processes"] == [process]


def test_phase3_server_migration_ignores_systemctl_grep_audit_probe_processes() -> None:
    snapshot = _ready_snapshot()
    process = (
        "418955 bash -lc systemctl list-unit-files --type=service --all "
        "| grep -Ei 'bb|qwen|deepseek|chronos|timesfm|vllm|ollama|open-webui'"
    )
    snapshot["candidate_model_processes"] = [process]

    report = evaluate_phase3_server_snapshot(snapshot)

    assert report["status"] == "ready"
    assert report["phase3_go_live_blocked"] is False
    assert report["legacy_process_count"] == 0
    assert report["ignored_probe_process_count"] == 1
    assert report["ignored_probe_processes"] == [process]


def test_phase3_server_migration_compacts_long_process_evidence() -> None:
    snapshot = _ready_snapshot()
    process = (
        "2077059 bash -c python3 - <<'PY' "
        + ("import json; " * 400)
        + '"Qwen3-32B|Qwen3.5-122B|trade_ai|trade_models|vllm"'
    )
    snapshot["candidate_model_processes"] = [process]

    report = evaluate_phase3_server_snapshot(snapshot)

    assert report["status"] == "ready"
    assert report["ignored_probe_process_count"] == 1
    ignored = report["ignored_probe_processes"][0]
    assert len(ignored) < 780
    assert "truncated process evidence" in ignored


def test_phase3_server_migration_snapshot_blocks_legacy_and_non_whitelist() -> None:
    snapshot = _ready_snapshot()
    snapshot["resource_release_marker"] = {"present": False, "data": {}}
    snapshot["migration_manifest"]["data"]["whitelist_only"] = False
    snapshot["migration_manifest"]["data"]["items"].append(
        {
            "category": "old_model_cache",
            "source": "old_model_server_disk",
            "path": "/data/trade_ai/models",
        }
    )
    snapshot["forbidden_services"][0]["active"] = True
    snapshot["legacy_processes"] = ["1234 python -m vllm --model Qwen3.5-122B"]

    report = evaluate_phase3_server_snapshot(snapshot)
    codes = {item["code"] for item in report["blockers"]}
    warning_codes = {item["code"] for item in report["warnings"]}

    assert report["status"] == "blocked"
    assert report["phase3_go_live_blocked"] is True
    assert "resource_release_marker_missing" in codes
    assert "legacy_data_paths_preserved" in warning_codes
    assert "legacy_services_present" in codes
    assert "legacy_processes_running" in codes
    assert "migration_not_whitelist_only" in codes
    assert "migration_category_not_approved" in codes
    assert "migration_source_not_approved" in codes


def test_phase3_server_probe_script_renders_valid_python() -> None:
    compile(render_phase3_server_probe(), "<phase3_server_probe>", "exec")


@pytest.mark.asyncio
async def test_phase3_server_migration_service_uses_injected_probe() -> None:
    service = Phase3ServerMigrationAuditService(remote_probe=_ready_snapshot)

    report = await service.report()

    assert report["status"] == "ready"
    assert report["remote_probe_available"] is True
    assert report["phase3_go_live_blocked"] is False


@pytest.mark.asyncio
async def test_phase3_server_migration_service_awaits_async_info_loader() -> None:
    calls: list[str] = []
    info = RemoteServerInfo(
        host="203.0.113.10",
        port=22,
        username="root",
        password="secret",
        source_path=Path("<test>"),
    )

    async def async_info_loader(_root):
        calls.append("async_info_loader")
        return info

    class FakeSsh:
        def close(self) -> None:
            calls.append("ssh_close")

    def ssh_connector(_root, **kwargs):
        calls.append(f"ssh:{kwargs['info'].host}")
        return FakeSsh()

    class FakeResult:
        status = 0
        stdout = '{"resource_release_marker":{"present":true,"data":{"policy_id":"phase3_stop_legacy_release_gpu_keep_old_data_2026_06_26","legacy_resources_stopped":true,"old_data_preserved":true,"phase3_root":"/data/BB"}},"migration_manifest":{"present":true,"data":{"whitelist_only":true,"items":[]}},"phase3_roots":[{"path":"/data/BB","exists":true}],"legacy_data_paths":[],"forbidden_services":[],"legacy_processes":[]}'
        stderr = ""

    def command_executor(_ssh, _command, **_kwargs):
        calls.append("command")
        return FakeResult()

    service = Phase3ServerMigrationAuditService(
        async_info_loader=async_info_loader,
        ssh_connector=ssh_connector,
        command_executor=command_executor,
    )

    report = await service.report()

    assert report["status"] == "ready"
    assert calls == ["async_info_loader", "ssh:203.0.113.10", "command", "ssh_close"]


@pytest.mark.asyncio
async def test_phase3_server_migration_service_preserves_sync_info_loader_injection() -> None:
    calls: list[str] = []
    info = RemoteServerInfo(
        host="203.0.113.11",
        port=22,
        username="root",
        password="secret",
        source_path=Path("<test>"),
    )

    def info_loader(_root):
        calls.append("sync_info_loader")
        return info

    class FakeSsh:
        def close(self) -> None:
            calls.append("ssh_close")

    def ssh_connector(_root, **kwargs):
        calls.append(f"ssh:{kwargs['info'].host}")
        return FakeSsh()

    class FakeResult:
        status = 0
        stdout = '{"resource_release_marker":{"present":true,"data":{"policy_id":"phase3_stop_legacy_release_gpu_keep_old_data_2026_06_26","legacy_resources_stopped":true,"old_data_preserved":true,"phase3_root":"/data/BB"}},"migration_manifest":{"present":true,"data":{"whitelist_only":true,"items":[]}},"phase3_roots":[{"path":"/data/BB","exists":true}],"legacy_data_paths":[],"forbidden_services":[],"legacy_processes":[]}'
        stderr = ""

    def command_executor(_ssh, _command, **_kwargs):
        calls.append("command")
        return FakeResult()

    service = Phase3ServerMigrationAuditService(
        info_loader=info_loader,
        ssh_connector=ssh_connector,
        command_executor=command_executor,
    )

    report = await service.report()

    assert report["status"] == "ready"
    assert calls == ["sync_info_loader", "ssh:203.0.113.11", "command", "ssh_close"]
