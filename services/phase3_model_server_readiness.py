"""Read-only Phase 3 model-server model/runtime readiness audit."""

from __future__ import annotations

import asyncio
import inspect
import json
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.remote_ssh import connect_remote_ssh, exec_remote_command
from core.safe_output import safe_error_text
from services.model_server_config import (
    ModelServerConfigError,
    ModelServerConfigNotConfigured,
    load_model_server_info_for_monitor,
    load_model_server_info_for_monitor_async,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE3_ROOT = "/data/BB"
DOWNLOAD_MANIFEST_PATH = "/data/BB/manifests/phase3_model_download_manifest.json"
VALIDATION_MANIFEST_PATH = "/data/BB/manifests/phase3_model_validation.json"
SERVICE_MANIFEST_PATH = "/data/BB/manifests/phase3_model_service_manifest.json"
PHASE3_MODEL_POLICY_ID = "phase3_quant_model_server_shadow_first_2026_06_27"
EXPECTED_GPU_COUNT = 8
OLD_TAKEOVER_CONTRACT_ID = "old_one_gpu_timesfm_takeover"
OLD_TAKEOVER_EXPECTED_GPU_COUNT = 1
OLD_TAKEOVER_QUANT_API_PORT = 8101
OLD_TAKEOVER_QUANT_API_SERVICE = "bb-phase3-quant-api.service"
FINQUANT_EXPERT_SLOT = "llm_expert_pool"
FINQUANT_EXPERT_SERVED_MODEL_NAME = "BB-FinQuant-Expert-14B"

REQUIRED_ARTIFACT_SLOTS = (
    "timeseries_primary",
    "timeseries_challenger",
    "sentiment_primary",
    "llm_decision_maker",
    "llm_expert_pool",
    "llm_high_risk_review",
)

SHADOW_FIRST_LLM_SLOTS = (
    "llm_decision_maker",
    "llm_expert_pool",
    "llm_high_risk_review",
)

OLD_TAKEOVER_REQUIRED_ARTIFACT_SLOTS = (
    "timeseries_primary",
    "timeseries_challenger",
    "sentiment_primary",
)

OLD_TAKEOVER_REQUIRED_SERVICES = (
    "qwen3-14b-trade.service",
    "deepseek-r1-14b-risk.service",
    "bb-finquant-expert-alias.service",
    OLD_TAKEOVER_QUANT_API_SERVICE,
)

OLD_TAKEOVER_REQUIRED_ENDPOINTS = (
    (8000, "qwen3-14b-trade"),
    (8002, "deepseek-r1-14b-risk"),
    (8003, FINQUANT_EXPERT_SERVED_MODEL_NAME),
    (OLD_TAKEOVER_QUANT_API_PORT, ""),
)

LLM_ROLE_DIVERSITY_REQUIRED_PAIRS = (
    ("llm_decision_maker", "llm_expert_pool"),
)

LLM_POLICY_CANDIDATE_SLOT_MAP = {
    "decision_maker": "llm_decision_maker",
    "expert_pool": "llm_expert_pool",
    "high_risk_review": "llm_high_risk_review",
}

LLM_SPECIALIZATION_KEYS = (
    "adapter_path",
    "lora_adapter",
    "specialization_manifest",
    "specialization_id",
    "fine_tune_id",
    "training_artifact",
)

MODEL_RUNTIME_PORTS = tuple(range(8000, 8011))
PROBED_RUNTIME_PORTS = tuple(dict.fromkeys((*MODEL_RUNTIME_PORTS, OLD_TAKEOVER_QUANT_API_PORT)))

RemoteProbe = Callable[[], dict[str, Any]]
InfoLoader = Callable[[Path], Any]
AsyncInfoLoader = Callable[[Path], Any]
SshConnector = Callable[..., Any]
CommandExecutor = Callable[..., Any]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _blocker(code: str, message: str, *, evidence: Any | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {
        "code": code,
        "severity": "blocking",
        "message": message,
    }
    if evidence is not None:
        item["evidence"] = evidence
    return item


def _warning(code: str, message: str, *, evidence: Any | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {
        "code": code,
        "severity": "warning",
        "message": message,
    }
    if evidence is not None:
        item["evidence"] = evidence
    return item


def _manifest_payload(snapshot: dict[str, Any], key: str) -> dict[str, Any]:
    wrapper = _safe_dict(snapshot.get(key))
    return _safe_dict(wrapper.get("data"))


def _manifest_present(snapshot: dict[str, Any], key: str) -> bool:
    return bool(_safe_dict(snapshot.get(key)).get("present"))


def _model_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = data.get("models")
    return [_safe_dict(item) for item in _safe_list(rows)]


def _slot_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        slot = str(row.get("slot") or "").strip()
        if slot and slot not in result:
            result[slot] = row
    return result


def _nested_validation(download_manifest: dict[str, Any]) -> dict[str, Any]:
    return _safe_dict(download_manifest.get("validation"))


def _required_missing(row: dict[str, Any]) -> list[Any]:
    return _safe_list(row.get("required_missing")) + _safe_list(row.get("incomplete_cache_files"))


def _slot_ok(download_row: dict[str, Any], validation_row: dict[str, Any]) -> bool:
    row = validation_row or download_row
    if not row:
        return False
    status = str(row.get("status") or download_row.get("status") or "").strip().lower()
    exists = bool(row.get("exists", True))
    required_any_ok = bool(row.get("required_any_ok", True))
    return bool(
        exists
        and status in {"ok", "ready", "validated"}
        and required_any_ok
        and not _required_missing(row)
        and not _required_missing(download_row)
    )


def _normalize_model_identity(value: Any) -> str:
    text = str(value or "").strip().lower().replace("\\", "/")
    text = text.replace("--", "/")
    return "".join(ch for ch in text if ch not in {" ", "\t", "\r", "\n"})


def _slot_model_identity(row: dict[str, Any]) -> str:
    served_model_name = _normalize_model_identity(row.get("served_model_name"))
    if served_model_name:
        return served_model_name
    specialization_target = _normalize_model_identity(row.get("specialization_target"))
    if specialization_target:
        return specialization_target
    repo_id = _normalize_model_identity(row.get("repo_id"))
    if repo_id:
        return repo_id
    path = str(row.get("path") or row.get("target") or "").strip().replace("\\", "/")
    if not path:
        return ""
    return _normalize_model_identity(path.rstrip("/").rsplit("/", 1)[-1])


def _slot_artifact_identity(row: dict[str, Any]) -> str:
    repo_id = _normalize_model_identity(row.get("repo_id"))
    if repo_id:
        return repo_id
    path = str(row.get("path") or row.get("target") or "").strip().replace("\\", "/")
    if not path:
        return ""
    return _normalize_model_identity(path.rstrip("/").rsplit("/", 1)[-1])


def _slot_candidate_identities(row: dict[str, Any]) -> set[str]:
    identities = {
        _normalize_model_identity(row.get("served_model_name")),
        _normalize_model_identity(row.get("specialization_target")),
        _normalize_model_identity(row.get("repo_id")),
        _normalize_model_identity(row.get("base_model_carrier")),
    }
    path = str(row.get("path") or row.get("target") or "").strip().replace("\\", "/")
    if path:
        identities.add(_normalize_model_identity(path.rstrip("/").rsplit("/", 1)[-1]))
    return {item for item in identities if item}


def _slot_specialization_evidence(row: dict[str, Any]) -> dict[str, str]:
    evidence: dict[str, str] = {}
    nested = row.get("specialization_evidence")
    if isinstance(nested, dict):
        for key, value in nested.items():
            text = str(value or "").strip()
            if text:
                evidence[str(key)] = text
    for key in LLM_SPECIALIZATION_KEYS:
        value = str(row.get(key) or "").strip()
        if value:
            evidence[key] = value
    return evidence


def _llm_role_diversity_blockers(
    reports_by_slot: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    for left_slot, right_slot in LLM_ROLE_DIVERSITY_REQUIRED_PAIRS:
        left = reports_by_slot.get(left_slot, {})
        right = reports_by_slot.get(right_slot, {})
        if not left or not right:
            continue
        left_identity = _slot_artifact_identity(left)
        right_identity = _slot_artifact_identity(right)
        if not left_identity or left_identity != right_identity:
            continue
        left_specialization = _slot_specialization_evidence(left)
        right_specialization = _slot_specialization_evidence(right)
        if left_specialization or right_specialization:
            continue
        blockers.append(
            _blocker(
                "llm_role_diversity_missing",
                (
                    "Decision maker and expert pool cannot use the same base LLM "
                    "without audited fine-tune/adapter specialization; duplicate "
                    "Qwen slots are allowed only for temporary shadow bootstrap, "
                    "not canary/live promotion."
                ),
                evidence={
                    "left_slot": left_slot,
                    "right_slot": right_slot,
                    "model_identity": left_identity,
                    "left": left,
                    "right": right,
                },
            )
        )
    return blockers


def _finquant_specialization_warnings(
    reports_by_slot: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    expert = reports_by_slot.get(FINQUANT_EXPERT_SLOT, {})
    if not expert:
        return []
    evidence = _slot_specialization_evidence(expert)
    if evidence:
        return []
    return [
        _warning(
            "finquant_expert_specialization_pending",
            (
                "Expert-pool LLM is still a base model without audited BB quant "
                "LoRA/fine-tune/RAG specialization evidence. It may run in shadow, "
                "but final promotion requires BB-FinQuant-Expert specialization."
            ),
            evidence=expert,
        )
    ]


def _finquant_service_manifest_blockers(
    manifest_service_reports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expert_services = [
        item
        for item in manifest_service_reports
        if str(item.get("slot") or "") == FINQUANT_EXPERT_SLOT
    ]
    if not expert_services:
        return [
            _blocker(
                "finquant_expert_service_missing",
                "GPU 2 expert-pool service must be declared as BB-FinQuant-Expert-14B.",
                evidence={"slot": FINQUANT_EXPERT_SLOT},
            )
        ]
    blockers: list[dict[str, Any]] = []
    for service in expert_services:
        served_model = str(service.get("served_model_name") or "").strip()
        if served_model == FINQUANT_EXPERT_SERVED_MODEL_NAME:
            continue
        blockers.append(
            _blocker(
                "finquant_expert_service_name_mismatch",
                (
                    "GPU 2 expert-pool runtime must be exposed as "
                    "BB-FinQuant-Expert-14B, even when the current carrier is "
                    "a Qwen3-14B base model waiting for audited specialization."
                ),
                evidence={
                    "expected_served_model_name": FINQUANT_EXPERT_SERVED_MODEL_NAME,
                    "actual_served_model_name": served_model,
                    "service": service,
                },
            )
        )
    return blockers


def _llm_policy_candidate_blockers(
    policy: dict[str, Any],
    reports_by_slot: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    llm_candidates = _safe_dict(policy.get("llm_candidates"))
    if not llm_candidates:
        return []
    blockers: list[dict[str, Any]] = []
    for policy_key, slot in LLM_POLICY_CANDIDATE_SLOT_MAP.items():
        expected = _normalize_model_identity(llm_candidates.get(policy_key))
        if not expected:
            continue
        report = reports_by_slot.get(slot, {})
        actual_identities = _slot_candidate_identities(report)
        if not actual_identities or expected in actual_identities:
            continue
        blockers.append(
            _blocker(
                "llm_candidate_policy_mismatch",
                (
                    "Phase 3 LLM policy candidate does not match the validated "
                    "model slot. Inventory and runtime contracts must use one "
                    "authoritative model identity."
                ),
                evidence={
                    "policy_key": policy_key,
                    "slot": slot,
                    "policy_candidate": llm_candidates.get(policy_key),
                    "accepted_identities": sorted(actual_identities),
                    "validated_repo_id": report.get("repo_id"),
                    "served_model_name": report.get("served_model_name"),
                    "specialization_target": report.get("specialization_target"),
                    "validated_path": report.get("path"),
                },
            )
        )
    return blockers


def _merge_slot_service_identity(
    slot_reports: list[dict[str, Any]],
    service_reports: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    reports_by_slot = {str(item.get("slot") or ""): dict(item) for item in slot_reports}
    for service in service_reports:
        slot = str(service.get("slot") or "").strip()
        if not slot:
            continue
        report = reports_by_slot.setdefault(slot, {"slot": slot})
        served_model = str(service.get("served_model_name") or "").strip()
        if served_model and not str(report.get("served_model_name") or "").strip():
            report["served_model_name"] = served_model
        if service.get("ready"):
            report["service_ready"] = True
    return reports_by_slot


def _gpu_rows(snapshot: dict[str, Any]) -> list[str]:
    return [str(item) for item in _safe_list(snapshot.get("gpu")) if str(item).strip()]


def _active_service_lines(snapshot: dict[str, Any]) -> list[str]:
    active: list[str] = []
    for line in _safe_list(snapshot.get("services")):
        text = str(line or "").strip()
        columns = text.split()
        if (
            len(columns) >= 4
            and columns[1] == "loaded"
            and columns[2] == "active"
            and columns[3] == "running"
        ):
            active.append(text)
    return active


def _active_endpoints(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _safe_dict(item)
        for item in _safe_list(snapshot.get("port_probes"))
        if bool(_safe_dict(item).get("ok"))
    ]


def _service_manifest_services(service_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [_safe_dict(item) for item in _safe_list(service_manifest.get("services"))]


def _service_name_active(service_name: str, active_services: list[str]) -> bool:
    name = str(service_name or "").strip()
    return bool(name and any(name in line for line in active_services))


def _endpoint_ready(service: dict[str, Any], active_endpoints: list[dict[str, Any]]) -> bool:
    try:
        expected_port = int(service.get("port"))
    except (TypeError, ValueError):
        return False
    expected_model = str(service.get("served_model_name") or "").strip().lower()
    for endpoint in active_endpoints:
        try:
            port = int(endpoint.get("port"))
        except (TypeError, ValueError):
            continue
        if port != expected_port:
            continue
        response = str(endpoint.get("response") or "").lower()
        if not expected_model or expected_model in response:
            return True
    return False


def _endpoint_model_ready(
    port: int,
    served_model_name: str,
    active_endpoints: list[dict[str, Any]],
) -> bool:
    expected_model = str(served_model_name or "").strip().lower()
    for endpoint in active_endpoints:
        try:
            endpoint_port = int(endpoint.get("port"))
        except (TypeError, ValueError):
            continue
        if endpoint_port != port:
            continue
        if not bool(endpoint.get("ok")):
            continue
        response = str(endpoint.get("response") or "").lower()
        return bool(not expected_model or expected_model in response)
    return False


def _endpoint_port_ready(port: int, active_endpoints: list[dict[str, Any]]) -> bool:
    return _endpoint_model_ready(port, "", active_endpoints)


def _service_fragment_active(fragment: str, active_services: list[str]) -> bool:
    expected = str(fragment or "").strip().lower()
    return bool(expected and any(expected in line.lower() for line in active_services))


def _old_takeover_runtime_checks(
    *,
    active_services: list[str],
    active_endpoints: list[dict[str, Any]],
) -> dict[str, Any]:
    service_checks = [
        {
            "service_name": service_name,
            "active": _service_fragment_active(service_name, active_services),
        }
        for service_name in OLD_TAKEOVER_REQUIRED_SERVICES
    ]
    endpoint_checks = [
        {
            "port": port,
            "served_model_name": served_model_name,
            "ready": _endpoint_model_ready(port, served_model_name, active_endpoints),
        }
        for port, served_model_name in OLD_TAKEOVER_REQUIRED_ENDPOINTS
    ]
    return {
        "contract_id": OLD_TAKEOVER_CONTRACT_ID,
        "required_gpu_count": OLD_TAKEOVER_EXPECTED_GPU_COUNT,
        "required_services": service_checks,
        "required_endpoints": endpoint_checks,
        "service_ready": all(bool(item.get("active")) for item in service_checks),
        "endpoint_ready": all(bool(item.get("ready")) for item in endpoint_checks),
    }


def _old_takeover_artifact_signal(
    download_by_slot: dict[str, dict[str, Any]],
    validation_by_slot: dict[str, dict[str, Any]],
) -> bool:
    row = validation_by_slot.get("timeseries_primary") or download_by_slot.get(
        "timeseries_primary",
        {},
    )
    identity = _slot_artifact_identity(row)
    return "timesfm" in identity


def _looks_like_old_one_gpu_takeover(
    *,
    observed_gpu_count: int,
    active_services: list[str],
    active_endpoints: list[dict[str, Any]],
    artifact_signal: bool,
) -> bool:
    old_gpu_shape = 0 < observed_gpu_count < EXPECTED_GPU_COUNT
    if not old_gpu_shape:
        return False
    signals = 0
    if artifact_signal:
        signals += 1
    if _service_fragment_active("qwen3-14b-trade.service", active_services) or _endpoint_model_ready(
        8000,
        "qwen3-14b-trade",
        active_endpoints,
    ):
        signals += 1
    if _service_fragment_active(
        "deepseek-r1-14b-risk.service",
        active_services,
    ) or _endpoint_model_ready(8002, "deepseek-r1-14b-risk", active_endpoints):
        signals += 1
    if _service_fragment_active(
        "bb-finquant-expert-alias.service",
        active_services,
    ) or _endpoint_model_ready(8003, FINQUANT_EXPERT_SERVED_MODEL_NAME, active_endpoints):
        signals += 2
    if _service_fragment_active(OLD_TAKEOVER_QUANT_API_SERVICE, active_services) or _endpoint_port_ready(
        OLD_TAKEOVER_QUANT_API_PORT,
        active_endpoints,
    ):
        signals += 1
    return signals >= 3


def evaluate_phase3_model_server_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Evaluate model artifacts and runtime service readiness without mutation."""

    download_manifest = _manifest_payload(snapshot, "download_manifest")
    validation_manifest = _manifest_payload(snapshot, "validation_manifest")
    nested_validation = _nested_validation(download_manifest)
    validation_source = validation_manifest or nested_validation
    service_manifest = _manifest_payload(snapshot, "service_manifest")
    manifest_services = _service_manifest_services(service_manifest)
    policy = _safe_dict(download_manifest.get("policy"))

    download_rows = _model_rows(download_manifest)
    validation_rows = _model_rows(validation_source)
    download_by_slot = _slot_map(download_rows)
    validation_by_slot = _slot_map(validation_rows)
    torch_info = _safe_dict(validation_source.get("torch"))
    active_services = _active_service_lines(snapshot)
    active_endpoints = _active_endpoints(snapshot)
    gpu_rows = _gpu_rows(snapshot)
    gpu_processes = [
        str(item).strip()
        for item in _safe_list(snapshot.get("gpu_processes"))
        if str(item).strip()
    ]
    validation_gpu_count = int(torch_info.get("device_count") or 0)
    observed_gpu_count = max(len(gpu_rows), validation_gpu_count)
    old_takeover_contract = _looks_like_old_one_gpu_takeover(
        observed_gpu_count=observed_gpu_count,
        active_services=active_services,
        active_endpoints=active_endpoints,
        artifact_signal=_old_takeover_artifact_signal(download_by_slot, validation_by_slot),
    )
    expected_gpu_count = (
        OLD_TAKEOVER_EXPECTED_GPU_COUNT if old_takeover_contract else EXPECTED_GPU_COUNT
    )
    required_artifact_slots = (
        OLD_TAKEOVER_REQUIRED_ARTIFACT_SLOTS
        if old_takeover_contract
        else REQUIRED_ARTIFACT_SLOTS
    )
    old_takeover_runtime = (
        _old_takeover_runtime_checks(
            active_services=active_services,
            active_endpoints=active_endpoints,
        )
        if old_takeover_contract
        else {}
    )

    slot_reports: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    download_present = _manifest_present(snapshot, "download_manifest")
    validation_present = _manifest_present(snapshot, "validation_manifest") or bool(
        nested_validation
    )
    service_present = _manifest_present(snapshot, "service_manifest")

    if not download_present:
        blockers.append(
            _blocker(
                "model_download_manifest_missing",
                "Phase 3 model download manifest is missing.",
                evidence=DOWNLOAD_MANIFEST_PATH,
            )
        )
    if not validation_present:
        blockers.append(
            _blocker(
                "model_validation_manifest_missing",
                "Phase 3 model validation manifest is missing.",
                evidence=VALIDATION_MANIFEST_PATH,
            )
        )

    if policy and policy.get("quant_server_only") is not True:
        blockers.append(
            _blocker(
                "model_server_not_quant_only",
                "New model server must be reserved for the Phase 3 crypto-quant plan.",
                evidence=policy,
            )
        )

    cuda_available = bool(torch_info.get("cuda_available"))
    tiny_cuda_ok = bool(torch_info.get("tiny_cuda_tensor_ok", cuda_available))
    old_takeover_gpu_observed = bool(
        old_takeover_contract and gpu_rows and observed_gpu_count >= OLD_TAKEOVER_EXPECTED_GPU_COUNT
    )
    if validation_present and not cuda_available:
        if old_takeover_gpu_observed:
            warnings.append(
                _warning(
                    "old_takeover_cuda_validation_manifest_incomplete",
                    (
                        "Old one-GPU takeover is using live nvidia-smi/GPU process "
                        "evidence because the lightweight validation manifest does "
                        "not include torch CUDA fields."
                    ),
                    evidence={
                        "torch": torch_info,
                        "gpu_rows": gpu_rows[:4],
                        "gpu_process_count": len(gpu_processes),
                    },
                )
            )
        else:
            blockers.append(
                _blocker(
                    "cuda_unavailable",
                    "Torch validation says CUDA is unavailable on the model server.",
                    evidence=torch_info,
                )
            )
    if validation_present and not tiny_cuda_ok:
        if old_takeover_gpu_observed:
            warnings.append(
                _warning(
                    "old_takeover_cuda_tensor_probe_manifest_incomplete",
                    (
                        "Old one-GPU takeover is using live GPU service/process "
                        "evidence because the lightweight validation manifest does "
                        "not include the tiny CUDA tensor probe result."
                    ),
                    evidence={
                        "torch": torch_info,
                        "gpu_process_count": len(gpu_processes),
                    },
                )
            )
        else:
            blockers.append(
                _blocker(
                    "cuda_tensor_probe_failed",
                    "Tiny CUDA tensor validation failed on the model server.",
                    evidence=torch_info.get("tiny_cuda_tensor_error") or torch_info,
                )
            )
    if observed_gpu_count < expected_gpu_count:
        blockers.append(
            _blocker(
                "gpu_count_below_phase3_plan",
                (
                    "Model-server GPU count is below the active deployment "
                    "contract requirement."
                ),
                evidence={
                    "deployment_contract": (
                        OLD_TAKEOVER_CONTRACT_ID
                        if old_takeover_contract
                        else "phase3_full_model_server"
                    ),
                    "expected": expected_gpu_count,
                    "observed": observed_gpu_count,
                },
            )
        )

    for slot in required_artifact_slots:
        download_row = download_by_slot.get(slot, {})
        validation_row = validation_by_slot.get(slot, {})
        validation_slot_missing = bool(validation_present and not validation_row)
        ok = _slot_ok(download_row, validation_row) and not validation_slot_missing
        live_routing_enabled = bool(
            validation_row.get("live_routing_enabled", download_row.get("live_routing_enabled"))
        )
        report = {
            "slot": slot,
            "ok": ok,
            "validation_slot_missing": validation_slot_missing,
            "repo_id": validation_row.get("repo_id") or download_row.get("repo_id") or "",
            "served_model_name": validation_row.get("served_model_name")
            or download_row.get("served_model_name")
            or "",
            "path": validation_row.get("path") or download_row.get("target") or "",
            "stage": validation_row.get("stage") or download_row.get("stage") or "",
            "status": validation_row.get("status") or download_row.get("status") or "",
            "specialization_required": bool(
                validation_row.get(
                    "specialization_required",
                    download_row.get("specialization_required", False),
                )
            ),
            "specialization_target": validation_row.get("specialization_target")
            or download_row.get("specialization_target")
            or "",
            "specialization_status": validation_row.get("specialization_status")
            or download_row.get("specialization_status")
            or "",
            "base_model_carrier": validation_row.get("base_model_carrier")
            or download_row.get("base_model_carrier")
            or "",
            "required_missing": _required_missing(validation_row) or _required_missing(download_row),
            "live_routing_enabled": live_routing_enabled,
            "specialization_evidence": _slot_specialization_evidence(validation_row)
            or _slot_specialization_evidence(download_row),
        }
        slot_reports.append(report)
        if not ok:
            blockers.append(
                _blocker(
                    "required_model_slot_not_ready",
                    f"Required Phase 3 model slot is not ready: {slot}",
                    evidence=report,
                )
            )
        if slot in SHADOW_FIRST_LLM_SLOTS and live_routing_enabled:
            blockers.append(
                _blocker(
                    "llm_live_routing_enabled_before_shadow_gate",
                    "LLM candidates must remain shadow/candidate until Phase 3 service and promotion gates pass.",
                    evidence=report,
                )
            )

    optional_bad = [
        {
            "slot": row.get("slot"),
            "status": row.get("status"),
            "path": row.get("path") or row.get("target"),
            "required_missing": _required_missing(row),
        }
        for row in validation_rows
        if str(row.get("slot") or "") not in required_artifact_slots and not _slot_ok(row, row)
    ]
    if optional_bad:
        warnings.append(
            _warning(
                "optional_model_slot_not_ready",
                "Optional challenger/fallback model slots have validation warnings.",
                evidence=optional_bad[:8],
            )
        )

    manifest_service_reports: list[dict[str, Any]] = []
    for service in manifest_services:
        service_name = str(service.get("service_name") or "").strip()
        report = {
            "slot": service.get("slot"),
            "role": service.get("role"),
            "service_name": service_name,
            "port": service.get("port"),
            "served_model_name": service.get("served_model_name"),
            "shadow_only": bool(service.get("shadow_only", True)),
            "live_routing_enabled": bool(service.get("live_routing_enabled")),
            "service_active": _service_name_active(service_name, active_services),
            "endpoint_ready": _endpoint_ready(service, active_endpoints),
        }
        report["ready"] = bool(report["service_active"] and report["endpoint_ready"])
        manifest_service_reports.append(report)
        if report["live_routing_enabled"]:
            blockers.append(
                _blocker(
                    "model_service_live_routing_enabled_before_shadow_gate",
                    "Phase 3 model services must remain shadow-only until promotion gates pass.",
                    evidence=report,
                )
            )
    if not service_present:
        if old_takeover_contract:
            warnings.append(
                _warning(
                    "old_takeover_service_manifest_not_required",
                    (
                        "Old one-GPU takeover uses live legacy Qwen/DeepSeek, "
                        "BB-FinQuant alias, and Phase 3 quant API service checks "
                        "instead of the full new-server service manifest."
                    ),
                    evidence={
                        "deployment_contract": OLD_TAKEOVER_CONTRACT_ID,
                        "service_manifest_path": SERVICE_MANIFEST_PATH,
                    },
                )
            )
        else:
            warnings.append(
                _warning(
                    "model_service_manifest_missing",
                    (
                        "Model service manifest is missing; runtime services "
                        "cannot be promoted from an audited contract."
                    ),
                    evidence=SERVICE_MANIFEST_PATH,
                )
            )
    elif not manifest_services:
        blockers.append(
            _blocker(
                "model_service_manifest_empty",
                "Model service manifest is present but declares no Phase 3 model services.",
                evidence=SERVICE_MANIFEST_PATH,
            )
        )
    if not active_services:
        warnings.append(
            _warning(
                "model_services_not_running",
                "No Phase 3 model-serving systemd services are active yet.",
            )
        )
    missing_manifest_services = [
        item for item in manifest_service_reports if not bool(item.get("service_active"))
    ]
    missing_manifest_endpoints = [
        item for item in manifest_service_reports if not bool(item.get("endpoint_ready"))
    ]
    if missing_manifest_services:
        warnings.append(
            _warning(
                "manifest_model_services_not_active",
                "Some Phase 3 model services declared in the service manifest are not active.",
                evidence=missing_manifest_services[:8],
            )
        )
    if missing_manifest_endpoints:
        warnings.append(
            _warning(
                "manifest_model_endpoints_not_ready",
                "Some Phase 3 model endpoints declared in the service manifest did not return their served model.",
                evidence=missing_manifest_endpoints[:8],
            )
        )
    if service_present and manifest_services:
        blockers.extend(_finquant_service_manifest_blockers(manifest_service_reports))
    slot_reports_by_slot = _merge_slot_service_identity(slot_reports, manifest_service_reports)
    if old_takeover_contract:
        old_missing_services = [
            item
            for item in _safe_list(old_takeover_runtime.get("required_services"))
            if not bool(_safe_dict(item).get("active"))
        ]
        old_missing_endpoints = [
            item
            for item in _safe_list(old_takeover_runtime.get("required_endpoints"))
            if not bool(_safe_dict(item).get("ready"))
        ]
        if old_missing_services:
            warnings.append(
                _warning(
                    "old_takeover_required_services_not_active",
                    "Some required old-server takeover services are not active.",
                    evidence=old_missing_services,
                )
            )
        if old_missing_endpoints:
            warnings.append(
                _warning(
                    "old_takeover_required_endpoints_not_ready",
                    "Some required old-server takeover endpoints are not ready.",
                    evidence=old_missing_endpoints,
                )
            )
        warnings.append(
            _warning(
                "old_model_server_takeover_active",
                (
                    "Temporary old one-GPU model-server takeover is active; keep "
                    "the full Phase 3 model-server contract for switching back "
                    "when the new server is repaired."
                ),
                evidence=old_takeover_runtime,
            )
        )
    else:
        blockers.extend(_llm_policy_candidate_blockers(policy, slot_reports_by_slot))
        blockers.extend(_llm_role_diversity_blockers(slot_reports_by_slot))
        warnings.extend(_finquant_specialization_warnings(slot_reports_by_slot))
    if not active_endpoints:
        warnings.append(
            _warning(
                "model_endpoints_unavailable",
                "No local model endpoint responded on the approved runtime port range.",
                evidence=list(PROBED_RUNTIME_PORTS),
            )
        )
    if not gpu_processes:
        warnings.append(
            _warning(
                "gpu_runtime_idle",
                "GPU compute is idle; artifacts are present but no model runtime is serving them.",
            )
        )

    artifact_ready = not blockers
    full_runtime_ready = bool(
        service_present
        and manifest_services
        and manifest_service_reports
        and all(bool(item.get("ready")) for item in manifest_service_reports)
    )
    old_takeover_runtime_ready = bool(
        old_takeover_contract
        and old_takeover_runtime.get("service_ready")
        and old_takeover_runtime.get("endpoint_ready")
        and observed_gpu_count >= expected_gpu_count
    )
    runtime_ready = bool(
        old_takeover_runtime_ready if old_takeover_contract else full_runtime_ready
    )
    service_go_live_blocked = bool(blockers or not runtime_ready)
    status = "blocked" if blockers else ("ready" if runtime_ready else "artifact_ready_service_pending")

    return {
        "status": status,
        "read_only": True,
        "audit_only": True,
        "can_mutate_remote": False,
        "can_start_services": False,
        "can_change_live_routing": False,
        "live_routing_enabled": False,
        "artifact_ready": artifact_ready,
        "runtime_ready": runtime_ready,
        "phase3_model_service_go_live_blocked": service_go_live_blocked,
        "policy_id": PHASE3_MODEL_POLICY_ID,
        "phase3_root": PHASE3_ROOT,
        "deployment_contract": (
            OLD_TAKEOVER_CONTRACT_ID if old_takeover_contract else "phase3_full_model_server"
        ),
        "expected_gpu_count": expected_gpu_count,
        "old_takeover_runtime": old_takeover_runtime,
        "download_manifest_path": DOWNLOAD_MANIFEST_PATH,
        "validation_manifest_path": VALIDATION_MANIFEST_PATH,
        "service_manifest_path": SERVICE_MANIFEST_PATH,
        "download_manifest": {
            "present": download_present,
            "model_count": len(download_rows),
            "policy": policy,
            "created_at": download_manifest.get("created_at"),
        },
        "validation_manifest": {
            "present": validation_present,
            "model_count": len(validation_rows),
            "checked_at": validation_source.get("checked_at"),
            "torch": torch_info,
        },
        "service_manifest": {
            "present": service_present,
            "service_count": len(manifest_services),
            "data": service_manifest,
        },
        "manifest_service_count": len(manifest_service_reports),
        "manifest_service_ready_count": sum(
            1 for item in manifest_service_reports if bool(item.get("ready"))
        ),
        "manifest_services": manifest_service_reports,
        "required_slots": slot_reports,
        "required_slot_count": len(required_artifact_slots),
        "required_slot_ready_count": sum(1 for item in slot_reports if item.get("ok")),
        "gpu_count": observed_gpu_count,
        "gpu_rows": gpu_rows[:16],
        "gpu_process_count": len(gpu_processes),
        "gpu_processes": gpu_processes[:40],
        "active_model_service_count": len(active_services),
        "active_model_services": active_services[:40],
        "active_endpoint_count": len(active_endpoints),
        "active_endpoints": active_endpoints[:16],
        "listening_ports": _safe_list(snapshot.get("listening_ports"))[:80],
        "model_paths": _safe_list(snapshot.get("model_paths"))[:120],
        "manifest_files": _safe_list(snapshot.get("manifest_files"))[:80],
        "blockers": blockers,
        "warnings": warnings,
        "checked_at": _now_iso(),
    }


def render_phase3_model_server_probe() -> str:
    """Render the read-only remote probe executed on the model server."""

    return textwrap.dedent(
        f"""
        import json
        import os
        import subprocess

        DOWNLOAD_MANIFEST_PATH = {json.dumps(DOWNLOAD_MANIFEST_PATH)}
        VALIDATION_MANIFEST_PATH = {json.dumps(VALIDATION_MANIFEST_PATH)}
        SERVICE_MANIFEST_PATH = {json.dumps(SERVICE_MANIFEST_PATH)}
        PROBED_RUNTIME_PORTS = {json.dumps(PROBED_RUNTIME_PORTS)}

        def run(command, timeout=8):
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                    check=False,
                )
                return {{
                    "code": result.returncode,
                    "stdout": result.stdout.strip()[:12000],
                    "stderr": result.stderr.strip()[:2000],
                }}
            except Exception as exc:
                return {{"code": 124, "stdout": "", "stderr": str(exc)[:500]}}

        def read_json(path):
            if not os.path.exists(path):
                return {{"present": False, "data": {{}}}}
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except Exception as exc:
                return {{"present": True, "data": {{}}, "error": str(exc)[:180]}}
            return {{"present": True, "data": compact_manifest(data)}}

        def compact_model(row):
            if not isinstance(row, dict):
                return {{}}
            keys = (
                "slot",
                "repo_id",
                "target",
                "path",
                "role",
                "stage",
                "status",
                "error",
                "exists",
                "file_count",
                "size_bytes",
                "required_missing",
                "required_any_ok",
                "required_tokenizer_any_ok",
                "incomplete_cache_files",
                "validation_note",
                "live_routing_enabled",
                "served_model_name",
                "specialization_required",
                "specialization_target",
                "specialization_status",
                "base_model_carrier",
                "specialization_evidence",
                "adapter_path",
                "lora_adapter",
                "specialization_manifest",
                "specialization_id",
                "fine_tune_id",
                "training_artifact",
            )
            return {{key: row.get(key) for key in keys if key in row}}

        def compact_manifest(data):
            if not isinstance(data, dict):
                return {{}}
            result = {{}}
            for key in (
                "schema_version",
                "created_at",
                "checked_at",
                "storage_root",
                "root",
                "policy",
                "package_install",
                "torch",
                "imports",
            ):
                if key in data:
                    result[key] = data.get(key)
            if isinstance(data.get("models"), list):
                result["models"] = [compact_model(item) for item in data.get("models", [])]
            if isinstance(data.get("services"), list):
                result["services"] = data.get("services", [])
            if isinstance(data.get("validation"), dict):
                result["validation"] = compact_manifest(data["validation"])
            return result

        def lines(text, limit):
            return [
                line for line in (text or "").splitlines()
                if line.strip()
            ][:limit]

        def port_probe(port):
            models = run(
                "curl -fsS --max-time 3 http://127.0.0.1:%s/v1/models" % port,
                timeout=5,
            )
            response = models["stdout"]
            path = "/v1/models"
            if not response:
                health = run(
                    "curl -fsS --max-time 3 http://127.0.0.1:%s/health" % port,
                    timeout=5,
                )
                response = health["stdout"]
                path = "/health"
            return {{
                "port": port,
                "path": path,
                "ok": bool(response.strip()),
                "response": response[:1200],
            }}

        payload = {{
            "download_manifest": read_json(DOWNLOAD_MANIFEST_PATH),
            "validation_manifest": read_json(VALIDATION_MANIFEST_PATH),
            "service_manifest": read_json(SERVICE_MANIFEST_PATH),
            "services": lines(run(
                "systemctl list-units --type=service --all --no-pager "
                "| grep -Ei 'bb|qwen|deepseek|glm|chronos|timesfm|vllm|ollama|model|local-ai|trade-ai|phase3|quant' || true",
                timeout=10,
            )["stdout"], 120),
            "unit_files": lines(run(
                "systemctl list-unit-files --type=service --no-pager "
                "| grep -Ei 'bb|qwen|deepseek|glm|chronos|timesfm|vllm|ollama|model|local-ai|trade-ai|phase3|quant' || true",
                timeout=10,
            )["stdout"], 120),
            "listening_ports": lines(run(
                "ss -ltnp 2>/dev/null | grep -E ':(8000|8001|8002|8003|8004|8005|8006|8007|8008|8009|8010|8101|18000|18001|18002|18003)\\\\b' || true",
                timeout=5,
            )["stdout"], 80),
            "gpu": lines(run(
                "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu "
                "--format=csv,noheader,nounits 2>/dev/null || true",
                timeout=8,
            )["stdout"], 16),
            "gpu_processes": lines(run(
                "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory "
                "--format=csv,noheader,nounits 2>/dev/null || true",
                timeout=8,
            )["stdout"], 80),
            "model_paths": lines(run(
                "find /data/BB/models -maxdepth 3 -mindepth 1 -printf '%y %p\\\\n' 2>/dev/null | sort | head -240 || true",
                timeout=10,
            )["stdout"], 240),
            "manifest_files": lines(run(
                "find /data/BB/manifests -maxdepth 2 -type f -printf '%p\\\\n' 2>/dev/null | sort || true",
                timeout=5,
            )["stdout"], 80),
            "port_probes": [port_probe(port) for port in PROBED_RUNTIME_PORTS],
        }}
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        """
    ).strip()


def _remote_command() -> str:
    script = render_phase3_model_server_probe()
    if "\nPY\n" in f"\n{script}\n":
        raise ValueError("Phase 3 model-server probe cannot contain a bare PY delimiter.")
    return f"python3 - <<'PY'\n{script}\nPY"


@dataclass(slots=True)
class Phase3ModelServerReadinessAuditService:
    """Read-only gate for Phase 3 quant model-server artifacts and runtime."""

    project_root: Path = PROJECT_ROOT
    remote_probe: RemoteProbe | None = None
    info_loader: InfoLoader = load_model_server_info_for_monitor
    async_info_loader: AsyncInfoLoader | None = load_model_server_info_for_monitor_async
    ssh_connector: SshConnector = connect_remote_ssh
    command_executor: CommandExecutor = exec_remote_command
    timeout_seconds: int = 24

    def __post_init__(self) -> None:
        if (
            self.info_loader is not load_model_server_info_for_monitor
            and self.async_info_loader is load_model_server_info_for_monitor_async
        ):
            self.async_info_loader = None

    async def report(self) -> dict[str, Any]:
        started_at = datetime.now(UTC)
        if self.remote_probe is not None:
            try:
                snapshot = await asyncio.to_thread(self.remote_probe)
            except Exception as exc:
                return self._unavailable_report(exc, started_at=started_at)
            return self._evaluated_report(snapshot, started_at=started_at)

        try:
            info = await self._load_remote_info()
            snapshot = await asyncio.wait_for(
                asyncio.to_thread(self._collect_remote_snapshot, info),
                timeout=max(int(self.timeout_seconds or 1), 1),
            )
        except Exception as exc:
            return self._unavailable_report(exc, started_at=started_at)
        return self._evaluated_report(snapshot, started_at=started_at)

    async def _load_remote_info(self) -> Any:
        loader = self.async_info_loader or self.info_loader
        try:
            result = loader(self.project_root)
            if inspect.isawaitable(result):
                return await result
            return result
        except ModelServerConfigError as exc:
            if not _should_fallback_to_platform_bridge(exc):
                raise
            from core.model_server_bridge import load_model_server_info_from_platform

            return await asyncio.to_thread(
                load_model_server_info_from_platform,
                self.project_root,
            )

    def _collect_remote_snapshot(self, info: Any) -> dict[str, Any]:
        ssh = self.ssh_connector(
            self.project_root,
            timeout=8,
            banner_timeout=8,
            auth_timeout=8,
            info=info,
        )
        try:
            result = self.command_executor(
                ssh,
                _remote_command(),
                timeout=max(int(self.timeout_seconds or 24), 5),
                max_output_chars=80_000,
            )
        finally:
            ssh.close()
        if result.status != 0:
            raise RuntimeError(
                safe_error_text(
                    result.stderr or result.stdout or "phase3 model-server probe failed",
                    fallback="phase3 model-server probe failed",
                )
            )
        try:
            payload = json.loads(str(result.stdout or "{}"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                safe_error_text(
                    result.stdout or result.stderr or "invalid phase3 model-server payload"
                )
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError("phase3 model-server probe payload was not an object")
        return payload

    def _evaluated_report(
        self,
        snapshot: dict[str, Any],
        *,
        started_at: datetime,
    ) -> dict[str, Any]:
        report = evaluate_phase3_model_server_snapshot(snapshot)
        report["remote_probe_available"] = True
        report["duration_seconds"] = round((datetime.now(UTC) - started_at).total_seconds(), 6)
        return report

    def _unavailable_report(self, exc: Exception, *, started_at: datetime) -> dict[str, Any]:
        status = "model_server_not_configured" if isinstance(
            exc, ModelServerConfigNotConfigured
        ) else "model_server_probe_unavailable"
        if isinstance(exc, ModelServerConfigError):
            status = "model_server_config_error"
        blocker = _blocker(
            status,
            "Phase 3 model-server artifact/runtime readiness could not be verified.",
            evidence=safe_error_text(exc, limit=180),
        )
        return {
            "status": "unverified",
            "read_only": True,
            "audit_only": True,
            "can_mutate_remote": False,
            "can_start_services": False,
            "can_change_live_routing": False,
            "live_routing_enabled": False,
            "artifact_ready": False,
            "runtime_ready": False,
            "phase3_model_service_go_live_blocked": True,
            "remote_probe_available": False,
            "error": safe_error_text(exc, limit=180),
            "policy_id": PHASE3_MODEL_POLICY_ID,
            "phase3_root": PHASE3_ROOT,
            "download_manifest_path": DOWNLOAD_MANIFEST_PATH,
            "validation_manifest_path": VALIDATION_MANIFEST_PATH,
            "service_manifest_path": SERVICE_MANIFEST_PATH,
            "download_manifest": {"present": False, "model_count": 0, "policy": {}},
            "validation_manifest": {"present": False, "model_count": 0, "torch": {}},
            "service_manifest": {"present": False, "service_count": 0, "data": {}},
            "manifest_service_count": 0,
            "manifest_service_ready_count": 0,
            "manifest_services": [],
            "required_slots": [],
            "required_slot_count": len(REQUIRED_ARTIFACT_SLOTS),
            "required_slot_ready_count": 0,
            "gpu_count": 0,
            "gpu_rows": [],
            "gpu_process_count": 0,
            "gpu_processes": [],
            "active_model_service_count": 0,
            "active_model_services": [],
            "active_endpoint_count": 0,
            "active_endpoints": [],
            "listening_ports": [],
            "model_paths": [],
            "manifest_files": [],
            "blockers": [blocker],
            "warnings": [],
            "checked_at": _now_iso(),
            "duration_seconds": round((datetime.now(UTC) - started_at).total_seconds(), 6),
        }


def _is_missing_secure_settings_key_error(exc: Exception) -> bool:
    return "BB_SECURE_SETTINGS_KEY" in str(exc or "")


def _should_fallback_to_platform_bridge(exc: Exception) -> bool:
    text = str(exc or "")
    return (
        _is_missing_secure_settings_key_error(exc)
        or "Could not find server info file" in text
        or "Could not find model server info file" in text
    )
