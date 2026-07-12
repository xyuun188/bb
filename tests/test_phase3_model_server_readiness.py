from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import services.phase3_model_server_readiness as readiness_module
from core.remote_server_info import RemoteServerInfo
from scripts import run_phase3_model_server_readiness_audit as readiness_cli
from services.phase3_model_server_readiness import (
    REQUIRED_ARTIFACT_SLOTS,
    Phase3ModelServerReadinessAuditService,
    evaluate_phase3_model_server_snapshot,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _model(slot: str, *, repo_id: str | None = None, status: str = "ok") -> dict[str, Any]:
    return {
        "slot": slot,
        "repo_id": repo_id or f"test/{slot}",
        "target": f"/data/BB/models/{slot}",
        "path": f"/data/BB/models/{slot}",
        "stage": "shadow_candidate_not_live" if slot.startswith("llm_") else "shadow_first",
        "status": status,
        "exists": True,
        "required_missing": [],
        "required_any_ok": True,
        "live_routing_enabled": False,
    }


def _artifact_ready_snapshot(*, runtime_ready: bool = False) -> dict[str, Any]:
    model_rows = [_model(slot) for slot in REQUIRED_ARTIFACT_SLOTS]
    service_rows = [
        {
            "slot": "llm_decision_maker",
            "role": "decision_maker_shadow",
            "service_name": "bb-phase3-llm-decision.service",
            "served_model_name": "qwen3-32b-trade",
            "port": 8000,
            "shadow_only": True,
            "live_routing_enabled": False,
        },
        {
            "slot": "llm_high_risk_review",
            "role": "high_risk_review_shadow",
            "service_name": "bb-phase3-llm-risk-review.service",
            "served_model_name": "deepseek-r1-14b-risk",
            "port": 8002,
            "shadow_only": True,
            "live_routing_enabled": False,
        },
        {
            "slot": "llm_expert_pool",
            "role": "expert_pool_shadow",
            "service_name": "bb-phase3-llm-expert.service",
            "served_model_name": "BB-FinQuant-Expert-14B",
            "port": 8003,
            "shadow_only": True,
            "live_routing_enabled": False,
        },
    ]
    return {
        "download_manifest": {
            "present": True,
            "data": {
                "created_at": "2026-06-27T00:00:00+00:00",
                "policy": {
                    "quant_server_only": True,
                    "llm_live_routing_enabled": False,
                },
                "models": model_rows,
            },
        },
        "validation_manifest": {
            "present": True,
            "data": {
                "checked_at": "2026-06-27T00:01:00+00:00",
                "torch": {
                    "cuda_available": True,
                    "device_count": 8,
                    "tiny_cuda_tensor_ok": True,
                },
                "models": model_rows,
            },
        },
        "service_manifest": {
            "present": runtime_ready,
            "data": {"services": service_rows} if runtime_ready else {},
        },
        "services": (
            [
                "bb-phase3-llm-decision.service loaded active running Phase 3 Qwen",
                "bb-phase3-llm-risk-review.service loaded active running Phase 3 DeepSeek",
                "bb-phase3-llm-expert.service loaded active running Phase 3 BB-FinQuant",
            ]
            if runtime_ready
            else []
        ),
        "gpu": [f"{index}, NVIDIA GeForce RTX 5090, 2, 32607, 0, 27" for index in range(8)],
        "gpu_processes": (["GPU-0, 1001, python, 12000"] if runtime_ready else []),
        "listening_ports": (
            [
                "LISTEN 0 4096 127.0.0.1:8000 0.0.0.0:* users:(('python',pid=1001,fd=3))",
                "LISTEN 0 4096 127.0.0.1:8002 0.0.0.0:* users:(('python',pid=1002,fd=3))",
                "LISTEN 0 4096 127.0.0.1:8003 0.0.0.0:* users:(('python',pid=1003,fd=3))",
            ]
            if runtime_ready
            else []
        ),
        "port_probes": [
            {
                "port": 8000,
                "path": "/v1/models",
                "ok": runtime_ready,
                "response": '{"data":[{"id":"qwen3-32b-trade"}]}',
            },
            {
                "port": 8002,
                "path": "/v1/models",
                "ok": runtime_ready,
                "response": '{"data":[{"id":"deepseek-r1-14b-risk"}]}',
            },
            {
                "port": 8003,
                "path": "/v1/models",
                "ok": runtime_ready,
                "response": '{"data":[{"id":"BB-FinQuant-Expert-14B"}]}',
            },
        ],
        "model_paths": ["d /data/BB/models/llm_decision_maker"],
        "manifest_files": ["/data/BB/manifests/phase3_model_validation.json"],
    }


def _old_one_gpu_takeover_snapshot(*, runtime_ready: bool = True) -> dict[str, Any]:
    model_rows = [
        _model("timeseries_primary", repo_id="google/timesfm-2.5-200m-pytorch"),
        _model("timeseries_challenger", repo_id="amazon/chronos-2"),
        _model("sentiment_primary", repo_id="ProsusAI/finbert"),
        _model("sentiment_challenger", repo_id="yiyanghkust/finbert-tone"),
        _model("timeseries_fallback", repo_id="ibm-granite/granite-timeseries-ttm-r2"),
    ]
    return {
        "download_manifest": {
            "present": True,
            "data": {"models": model_rows},
        },
        "validation_manifest": {
            "present": True,
            "data": {
                "checked_at": "2026-07-08T07:50:50Z",
                "torch": {},
                "models": model_rows,
            },
        },
        "service_manifest": {"present": False, "data": {}},
        "services": (
            [
                "qwen3-14b-trade.service loaded active running Qwen3 14B",
                "deepseek-r1-14b-risk.service loaded active running DeepSeek risk",
                "bb-finquant-expert-gateway.service loaded active running FinQuant verified gateway",
                "bb-phase3-quant-api.service loaded active running Phase 3 Quant API",
            ]
            if runtime_ready
            else []
        ),
        "gpu": ["0, NVIDIA A100-SXM4-40GB, 26720, 40960, 0, 34"],
        "gpu_processes": (
            [
                "GPU-a3, 2644133, /home/linux/anaconda3/envs/trade_vllm/bin/python, 13732",
                "GPU-a3, 2650039, /home/linux/anaconda3/envs/trade_vllm/bin/python, 11482",
                "GPU-a3, 3978899, /data/BB/envs/phase3-quant/bin/python, 1472",
            ]
            if runtime_ready
            else []
        ),
        "listening_ports": (
            [
                "LISTEN 0 2048 0.0.0.0:8000 0.0.0.0:*",
                "LISTEN 0 2048 0.0.0.0:8002 0.0.0.0:*",
                "LISTEN 0 5 127.0.0.1:8003 0.0.0.0:*",
                "LISTEN 0 2048 127.0.0.1:8101 0.0.0.0:*",
            ]
            if runtime_ready
            else []
        ),
        "port_probes": [
            {
                "port": 8000,
                "path": "/v1/models",
                "ok": runtime_ready,
                "response": '{"data":[{"id":"qwen3-14b-trade"}]}',
            },
            {
                "port": 8002,
                "path": "/v1/models",
                "ok": runtime_ready,
                "response": '{"data":[{"id":"deepseek-r1-14b-risk"}]}',
            },
            {
                "port": 8003,
                "path": "/v1/models",
                "ok": runtime_ready,
                "response": '{"data":[{"id":"BB-FinQuant-Expert-14B"}]}',
            },
            {
                "port": 8101,
                "path": "/health",
                "ok": runtime_ready,
                "response": '{"status":"ready","trained_models_available":true}',
            },
        ],
        "model_paths": [
            "d /data/BB/models/timeseries/google--timesfm-2.5-200m-pytorch",
            "f /data/BB/models/local_ai_tools/local_quant_models.joblib",
        ],
        "manifest_files": [
            "/data/BB/manifests/phase3_model_download_manifest.json",
            "/data/BB/manifests/phase3_model_validation.json",
        ],
    }


def test_phase3_model_server_artifacts_ready_but_services_pending() -> None:
    report = evaluate_phase3_model_server_snapshot(_artifact_ready_snapshot())
    warning_codes = {item["code"] for item in report["warnings"]}

    assert report["status"] == "artifact_ready_service_pending"
    assert report["artifact_ready"] is True
    assert report["runtime_ready"] is False
    assert report["phase3_model_service_go_live_blocked"] is True
    assert report["blockers"] == []
    assert "model_service_manifest_missing" in warning_codes
    assert "model_services_not_running" in warning_codes
    assert "model_endpoints_unavailable" in warning_codes
    assert "gpu_runtime_idle" in warning_codes


def test_phase3_model_server_ready_requires_services_and_endpoint() -> None:
    report = evaluate_phase3_model_server_snapshot(_artifact_ready_snapshot(runtime_ready=True))

    assert report["status"] == "ready"
    assert report["artifact_ready"] is True
    assert report["runtime_ready"] is True
    assert report["phase3_model_service_go_live_blocked"] is False
    assert report["active_model_service_count"] == 3
    assert report["active_endpoint_count"] == 3
    assert report["manifest_service_ready_count"] == 3


def test_phase3_model_server_accepts_old_one_gpu_timesfm_takeover_contract() -> None:
    report = evaluate_phase3_model_server_snapshot(_old_one_gpu_takeover_snapshot())
    blocker_codes = {item["code"] for item in report["blockers"]}
    warning_codes = {item["code"] for item in report["warnings"]}

    assert report["status"] == "ready"
    assert report["deployment_contract"] == "old_one_gpu_timesfm_takeover"
    assert report["artifact_ready"] is True
    assert report["runtime_ready"] is True
    assert report["phase3_model_service_go_live_blocked"] is False
    assert report["expected_gpu_count"] == 1
    assert report["gpu_count"] == 1
    assert report["required_slot_count"] == 3
    assert report["required_slot_ready_count"] == 3
    assert {row["slot"] for row in report["required_slots"]} == {
        "timeseries_primary",
        "timeseries_challenger",
        "sentiment_primary",
        "llm_expert_pool",
    }
    assert report["active_endpoint_count"] == 4
    assert blocker_codes == set()
    assert "old_model_server_takeover_active" in warning_codes
    assert "old_takeover_service_manifest_not_required" in warning_codes
    assert "cuda_unavailable" not in blocker_codes
    assert "required_model_slot_not_ready" not in blocker_codes


@pytest.mark.asyncio
async def test_phase3_model_server_readiness_falls_back_to_platform_bridge_when_secure_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback_info = RemoteServerInfo(
        host="model.example",
        port=22,
        username="bb",
        password="secret",
        source_path=Path("<test>"),
    )
    used: dict[str, Any] = {}

    async def missing_key_loader(_project_root: Path) -> RemoteServerInfo:
        raise readiness_module.ModelServerConfigError(
            "BB_SECURE_SETTINGS_KEY is required for encrypted settings"
        )

    def fake_platform_loader(project_root: Path) -> RemoteServerInfo:
        used["project_root"] = project_root
        return fallback_info

    class FakeSsh:
        def close(self) -> None:
            used["closed"] = True

    def fake_ssh_connector(_project_root: Path, **kwargs: Any) -> FakeSsh:
        used["ssh_info"] = kwargs.get("info")
        return FakeSsh()

    class FakeResult:
        status = 0
        stdout = json.dumps(_artifact_ready_snapshot(runtime_ready=True), ensure_ascii=False)
        stderr = ""

    def fake_command_executor(_ssh: FakeSsh, _command: str, **_kwargs: Any) -> FakeResult:
        used["executed"] = True
        return FakeResult()

    monkeypatch.setattr(
        "core.model_server_bridge.load_model_server_info_from_platform",
        fake_platform_loader,
    )

    report = await Phase3ModelServerReadinessAuditService(
        async_info_loader=missing_key_loader,
        ssh_connector=fake_ssh_connector,
        command_executor=fake_command_executor,
    ).report()

    assert used["project_root"] == PROJECT_ROOT
    assert used["ssh_info"] == fallback_info
    assert used["executed"] is True
    assert used["closed"] is True
    assert report["status"] == "ready"
    assert report["remote_probe_available"] is True


@pytest.mark.asyncio
async def test_phase3_model_server_readiness_falls_back_to_platform_bridge_when_local_info_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback_info = RemoteServerInfo(
        host="model.example",
        port=22,
        username="bb",
        password="secret",
        source_path=Path("<test>"),
    )
    used: dict[str, Any] = {}

    async def missing_local_info_loader(_project_root: Path) -> RemoteServerInfo:
        raise readiness_module.ModelServerConfigNotConfigured(
            "Could not find server info file. Expected an ignored local server info file in the project root."
        )

    def fake_platform_loader(project_root: Path) -> RemoteServerInfo:
        used["project_root"] = project_root
        return fallback_info

    class FakeSsh:
        def close(self) -> None:
            used["closed"] = True

    def fake_ssh_connector(_project_root: Path, **kwargs: Any) -> FakeSsh:
        used["ssh_info"] = kwargs.get("info")
        return FakeSsh()

    class FakeResult:
        status = 0
        stdout = json.dumps(_artifact_ready_snapshot(runtime_ready=True), ensure_ascii=False)
        stderr = ""

    monkeypatch.setattr(
        "core.model_server_bridge.load_model_server_info_from_platform",
        fake_platform_loader,
    )

    report = await Phase3ModelServerReadinessAuditService(
        async_info_loader=missing_local_info_loader,
        ssh_connector=fake_ssh_connector,
        command_executor=lambda *_args, **_kwargs: FakeResult(),
    ).report()

    assert used["project_root"] == PROJECT_ROOT
    assert used["ssh_info"] == fallback_info
    assert used["closed"] is True
    assert report["status"] == "ready"
    assert report["remote_probe_available"] is True


def test_phase3_model_server_blocks_duplicate_decision_and_expert_base_model() -> None:
    snapshot = _artifact_ready_snapshot(runtime_ready=True)
    duplicate_repo = "Qwen/Qwen3-14B-AWQ"
    for manifest_key in ("download_manifest", "validation_manifest"):
        for row in snapshot[manifest_key]["data"]["models"]:
            if row["slot"] in {"llm_decision_maker", "llm_expert_pool"}:
                row["repo_id"] = duplicate_repo

    report = evaluate_phase3_model_server_snapshot(snapshot)
    blocker_codes = {item["code"] for item in report["blockers"]}

    assert report["status"] == "blocked"
    assert report["artifact_ready"] is False
    assert report["runtime_ready"] is True
    assert report["phase3_model_service_go_live_blocked"] is True
    assert "llm_role_diversity_missing" in blocker_codes


def test_phase3_model_server_allows_distinct_decision_and_expert_models() -> None:
    snapshot = _artifact_ready_snapshot(runtime_ready=True)
    for manifest_key in ("download_manifest", "validation_manifest"):
        for row in snapshot[manifest_key]["data"]["models"]:
            if row["slot"] == "llm_decision_maker":
                row["repo_id"] = "Qwen/Qwen3-32B-AWQ"
            if row["slot"] == "llm_expert_pool":
                row["repo_id"] = "Qwen/Qwen3-14B-AWQ"

    report = evaluate_phase3_model_server_snapshot(snapshot)

    assert report["status"] == "ready"
    assert "llm_role_diversity_missing" not in {item["code"] for item in report["blockers"]}
    assert "finquant_expert_specialization_pending" in {item["code"] for item in report["warnings"]}


def test_phase3_model_server_blocks_legacy_expert_pool_runtime_name() -> None:
    snapshot = _artifact_ready_snapshot(runtime_ready=True)
    for service in snapshot["service_manifest"]["data"]["services"]:
        if service["slot"] == "llm_expert_pool":
            service["served_model_name"] = "qwen3-14b-expert-pool"
    for probe in snapshot["port_probes"]:
        if probe["port"] == 8003:
            probe["response"] = '{"data":[{"id":"qwen3-14b-expert-pool"}]}'

    report = evaluate_phase3_model_server_snapshot(snapshot)
    blocker_codes = {item["code"] for item in report["blockers"]}

    assert report["status"] == "blocked"
    assert report["artifact_ready"] is False
    assert report["phase3_model_service_go_live_blocked"] is True
    assert "finquant_expert_service_name_mismatch" in blocker_codes


def test_phase3_model_server_blocks_policy_candidate_mismatch() -> None:
    snapshot = _artifact_ready_snapshot(runtime_ready=True)
    snapshot["download_manifest"]["data"]["policy"]["llm_candidates"] = {
        "decision_maker": "Qwen/Qwen3-14B-AWQ",
        "expert_pool": "Qwen/Qwen3-14B-AWQ",
        "high_risk_review": "test/llm_high_risk_review",
    }
    for manifest_key in ("download_manifest", "validation_manifest"):
        for row in snapshot[manifest_key]["data"]["models"]:
            if row["slot"] == "llm_decision_maker":
                row["repo_id"] = "Qwen/Qwen3-32B-AWQ"
            if row["slot"] == "llm_expert_pool":
                row["repo_id"] = "Qwen/Qwen3-14B-AWQ"

    report = evaluate_phase3_model_server_snapshot(snapshot)
    blocker_codes = {item["code"] for item in report["blockers"]}

    assert report["status"] == "blocked"
    assert "llm_candidate_policy_mismatch" in blocker_codes


def test_phase3_model_server_policy_accepts_finquant_served_identity() -> None:
    snapshot = _artifact_ready_snapshot(runtime_ready=True)
    snapshot["download_manifest"]["data"]["policy"]["llm_candidates"] = {
        "decision_maker": "qwen3-32b-trade",
        "expert_pool": "BB-FinQuant-Expert-14B",
        "high_risk_review": "deepseek-r1-14b-risk",
    }
    for manifest_key in ("download_manifest", "validation_manifest"):
        for row in snapshot[manifest_key]["data"]["models"]:
            if row["slot"] == "llm_decision_maker":
                row["repo_id"] = "Qwen/Qwen3-32B-AWQ"
                row["served_model_name"] = "qwen3-32b-trade"
            if row["slot"] == "llm_expert_pool":
                row["repo_id"] = "Qwen/Qwen3-14B-AWQ"
                row["served_model_name"] = "BB-FinQuant-Expert-14B"
                row["specialization_required"] = True
                row["specialization_target"] = "BB-FinQuant-Expert-14B"
                row["specialization_status"] = "pending"
                row["base_model_carrier"] = "Qwen/Qwen3-14B-AWQ"
            if row["slot"] == "llm_high_risk_review":
                row["repo_id"] = "casperhansen/deepseek-r1-distill-qwen-14b-awq"
                row["served_model_name"] = "deepseek-r1-14b-risk"

    report = evaluate_phase3_model_server_snapshot(snapshot)
    blocker_codes = {item["code"] for item in report["blockers"]}
    warning_codes = {item["code"] for item in report["warnings"]}

    assert report["status"] == "ready"
    assert "llm_candidate_policy_mismatch" not in blocker_codes
    assert "finquant_expert_specialization_pending" in warning_codes


def test_phase3_model_server_policy_accepts_finquant_service_manifest_identity() -> None:
    snapshot = _artifact_ready_snapshot(runtime_ready=True)
    snapshot["download_manifest"]["data"]["policy"]["llm_candidates"] = {
        "decision_maker": "Qwen/Qwen3-32B-AWQ",
        "expert_pool": "BB-FinQuant-Expert-14B",
        "high_risk_review": "casperhansen/deepseek-r1-distill-qwen-14b-awq",
    }
    for manifest_key in ("download_manifest", "validation_manifest"):
        for row in snapshot[manifest_key]["data"]["models"]:
            if row["slot"] == "llm_decision_maker":
                row["repo_id"] = "Qwen/Qwen3-32B-AWQ"
            if row["slot"] == "llm_expert_pool":
                row["repo_id"] = "Qwen/Qwen3-14B-AWQ"
            if row["slot"] == "llm_high_risk_review":
                row["repo_id"] = "casperhansen/deepseek-r1-distill-qwen-14b-awq"

    report = evaluate_phase3_model_server_snapshot(snapshot)
    blocker_codes = {item["code"] for item in report["blockers"]}
    warning_codes = {item["code"] for item in report["warnings"]}

    assert report["status"] == "ready"
    assert "llm_candidate_policy_mismatch" not in blocker_codes
    assert "finquant_expert_specialization_pending" in warning_codes


def test_phase3_model_server_accepts_specialized_finquant_expert_evidence() -> None:
    snapshot = _artifact_ready_snapshot(runtime_ready=True)
    for manifest_key in ("download_manifest", "validation_manifest"):
        for row in snapshot[manifest_key]["data"]["models"]:
            if row["slot"] == "llm_decision_maker":
                row["repo_id"] = "Qwen/Qwen3-32B-AWQ"
            if row["slot"] == "llm_expert_pool":
                row["repo_id"] = "Qwen/Qwen3-14B-AWQ"
                row["specialization_evidence"] = {
                    "verification_status": "verified",
                    "identity_verified": True,
                    "legacy_read_only": False,
                    "objective_name": "maximize_expected_realized_net_return_after_cost",
                    "objective_version": "2026-07-12.v1",
                    "preference_contract_version": "bb_finquant_return_preference.v1",
                    "preference_selection_accuracy": 1.0,
                    "training_stages": [
                        "sft_format_domain",
                        "trl_dpo_return_preference",
                    ],
                    "adapter_version": "20260712T010203Z-aaaaaaaaaaaa",
                    "adapter_path": "/data/BB/models/finquant_lora/versions/v2",
                    "specialization_manifest": "/data/BB/models/finquant_lora/versions/v2/specialization_manifest.json",
                    "specialization_id": "BB-FinQuant-Expert-14B-v2",
                    "dataset_version": "bb-finquant-sft-v2-aaaaaaaaaaaa-bbbbbbbb",
                    "source_code_version": "commit-sha",
                    "base_model_repo": "Qwen/Qwen3-14B",
                    "trained_at": "2026-07-12T01:02:03+00:00",
                    "adapter_sha256": "a" * 64,
                    "manifest_sha256": "b" * 64,
                    "dataset_sha256": "c" * 64,
                    "dataset_lineage_sha256": "1" * 64,
                    "dataset_manifest_sha256": "d" * 64,
                    "source_script_sha256": "e" * 64,
                    "trainer_code_sha256": "2" * 64,
                    "base_model_config_sha256": "f" * 64,
                    "inference_base_model_config_sha256": "0" * 64,
                }

    report = evaluate_phase3_model_server_snapshot(snapshot)

    assert report["status"] == "ready"
    assert "finquant_expert_specialization_pending" not in {
        item["code"] for item in report["warnings"]
    }


def test_phase3_model_server_rejects_unverified_finquant_manifest_name_only() -> None:
    snapshot = _artifact_ready_snapshot(runtime_ready=True)
    for manifest_key in ("download_manifest", "validation_manifest"):
        for row in snapshot[manifest_key]["data"]["models"]:
            if row["slot"] == "llm_expert_pool":
                row["specialization_id"] = "BB-FinQuant-Expert-14B-v1"
                row["specialization_manifest"] = "/data/BB/models/finquant_lora/unverified.json"

    report = evaluate_phase3_model_server_snapshot(snapshot)

    assert "finquant_expert_specialization_pending" in {item["code"] for item in report["warnings"]}


def test_phase3_model_server_accepts_systemctl_aligned_active_columns() -> None:
    snapshot = _artifact_ready_snapshot(runtime_ready=True)
    snapshot["services"] = [
        "bb-phase3-llm-decision.service            loaded    active   running BB Phase 3 decision model",
        "bb-phase3-llm-risk-review.service         loaded    active   running BB Phase 3 risk model",
        "bb-phase3-llm-expert.service              loaded    active   running BB Phase 3 expert model",
    ]

    report = evaluate_phase3_model_server_snapshot(snapshot)

    assert report["status"] == "ready"
    assert report["runtime_ready"] is True
    assert report["active_model_service_count"] == 3
    assert report["manifest_service_ready_count"] == 3


def test_phase3_model_server_service_manifest_requires_declared_endpoint() -> None:
    snapshot = _artifact_ready_snapshot(runtime_ready=True)
    snapshot["port_probes"][0]["response"] = '{"data":[{"id":"wrong-model"}]}'

    report = evaluate_phase3_model_server_snapshot(snapshot)
    warning_codes = {item["code"] for item in report["warnings"]}

    assert report["status"] == "artifact_ready_service_pending"
    assert report["artifact_ready"] is True
    assert report["runtime_ready"] is False
    assert report["manifest_service_ready_count"] == 2
    assert "manifest_model_endpoints_not_ready" in warning_codes
    assert [item["slot"] for item in report["manifest_services"] if not item["endpoint_ready"]] == [
        "llm_decision_maker"
    ]


def test_phase3_model_server_blocks_cuda_and_missing_required_slot() -> None:
    snapshot = _artifact_ready_snapshot()
    snapshot["validation_manifest"]["data"]["torch"]["cuda_available"] = False
    snapshot["validation_manifest"]["data"]["models"] = [
        row
        for row in snapshot["validation_manifest"]["data"]["models"]
        if row["slot"] != "llm_high_risk_review"
    ]

    report = evaluate_phase3_model_server_snapshot(snapshot)
    blocker_codes = {item["code"] for item in report["blockers"]}

    assert report["status"] == "blocked"
    assert report["artifact_ready"] is False
    assert "cuda_unavailable" in blocker_codes
    assert "required_model_slot_not_ready" in blocker_codes


@pytest.mark.asyncio
async def test_phase3_model_server_service_uses_injected_probe() -> None:
    service = Phase3ModelServerReadinessAuditService(
        remote_probe=lambda: _artifact_ready_snapshot(runtime_ready=True)
    )

    report = await service.report()

    assert report["status"] == "ready"
    assert report["remote_probe_available"] is True


@pytest.mark.asyncio
async def test_phase3_model_server_service_awaits_async_info_loader() -> None:
    calls: list[str] = []
    info = RemoteServerInfo(
        host="203.0.113.12",
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
        stdout = json.dumps(_artifact_ready_snapshot(runtime_ready=True))
        stderr = ""

    def command_executor(_ssh, _command, **_kwargs):
        calls.append("command")
        return FakeResult()

    service = Phase3ModelServerReadinessAuditService(
        async_info_loader=async_info_loader,
        ssh_connector=ssh_connector,
        command_executor=command_executor,
    )

    report = await service.report()

    assert report["status"] == "ready"
    assert calls == ["async_info_loader", "ssh:203.0.113.12", "command", "ssh_close"]


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
