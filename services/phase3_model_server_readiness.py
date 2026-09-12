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

from core.model_candidate_manifest import ModelCandidateManifest
from core.model_topology import qwen27_candidate_topology
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
TARGET_CANDIDATE_MANIFEST_PATH = "/data/BB/manifests/target_model_candidate.json"
PHASE3_MODEL_POLICY_ID = "phase3_quant_model_server_shadow_first_2026_06_27"
MINIMUM_RUNTIME_GPU_COUNT = 1
TARGET_SERVICE_NAME = "bb-phase3-llm-target.service"
TARGET_SERVICE_SLOT = "llm_decision_and_expert_carrier"
TARGET_MODEL_PORT = 8000
TARGET_MODEL_ID = "qwen3.8-27b"
REQUIRED_ARTIFACT_SLOTS = ("target_single_model",)
RETIRED_SERVICE_MARKERS = (
    "bb-phase3-llm-decision.service",
    "bb-phase3-llm-expert.service",
    "bb-phase3-llm-risk-review.service",
    "qwen3-14b-trade.service",
    "deepseek-r1-14b-risk.service",
)

LLM_SPECIALIZATION_KEYS = (
    "adapter_path",
    "lora_adapter",
    "specialization_manifest",
    "specialization_id",
    "fine_tune_id",
    "training_artifact",
    "objective_name",
    "objective_version",
    "preference_contract_version",
    "preference_selection_accuracy",
    "training_stages",
)

MODEL_RUNTIME_PORTS = (TARGET_MODEL_PORT,)
PROBED_RUNTIME_PORTS = MODEL_RUNTIME_PORTS

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


def _target_topology_report(
    candidate: ModelCandidateManifest | None = None,
    *,
    service_ready: bool = False,
    endpoint_ready: bool = False,
) -> dict[str, Any]:
    """Expose the target contract without inventing an identity.

    A placeholder model id is deliberately not returned as a production
    identity.  This keeps dashboards and paper preflight from mistaking the
    unconfigured candidate topology for a deployed model.
    """

    topology = qwen27_candidate_topology()
    if candidate is None:
        model = topology.models[0]
        return {
            "profile": topology.profile,
            "local_model_count_target": topology.local_model_count_target,
            "model_id": "",
            "role": model.role,
            "port": model.port,
            "endpoint": model.endpoint,
            "repo_id": "",
            "revision": "",
            "path": "",
            "identity_complete": False,
            "context_length": model.context_length,
            "max_concurrency": model.max_concurrency,
            "gpu_memory_budget_gib": model.gpu_memory_budget_gib,
            "stage": "candidate_not_configured",
            "runtime": {},
            "service_ready": False,
            "endpoint_ready": False,
            "live_routing_enabled": False,
            "activation_blocked": True,
            "blockers": ["target_candidate_manifest_missing"],
            "cloud_reviewer_required_for_high_risk_entry": (
                topology.cloud_reviewer_required_for_high_risk_entry
            ),
        }
    target = candidate.to_topology(stage="paper")
    model = target.models[0]
    return {
        "profile": target.profile,
        "local_model_count_target": target.local_model_count_target,
        "model_id": candidate.model_id,
        "role": model.role,
        "port": model.port,
        "endpoint": model.endpoint,
        "repo_id": candidate.repo_id,
        "revision": candidate.revision,
        "path": candidate.model_path,
        "tokenizer_path": candidate.tokenizer_path,
        "architecture": candidate.architecture,
        "model_type": candidate.model_type,
        "runtime": candidate.runtime.to_dict(),
        "identity_complete": True,
        "context_length": candidate.context_length,
        "max_concurrency": candidate.max_concurrency,
        "gpu_memory_budget_gib": candidate.gpu_memory_peak_gib,
        "stage": "paper",
        "service_ready": service_ready,
        "endpoint_ready": endpoint_ready,
        "live_routing_enabled": False,
        "activation_blocked": not (service_ready and endpoint_ready),
        "blockers": [] if service_ready and endpoint_ready else ["target_runtime_not_ready"],
        "cloud_reviewer_required_for_high_risk_entry": (
            target.cloud_reviewer_required_for_high_risk_entry
        ),
    }


def _manifest_payload(snapshot: dict[str, Any], key: str) -> dict[str, Any]:
    wrapper = _safe_dict(snapshot.get(key))
    return _safe_dict(wrapper.get("data"))


def _manifest_present(snapshot: dict[str, Any], key: str) -> bool:
    return bool(_safe_dict(snapshot.get(key)).get("present"))


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


def _endpoint_models(endpoint: dict[str, Any]) -> set[str]:
    """Extract exact model ids from an OpenAI-compatible models response."""

    response = str(endpoint.get("response") or "")
    try:
        payload = json.loads(response)
    except (TypeError, ValueError):
        return set()
    rows = payload.get("data") if isinstance(payload, dict) else None
    return {
        str(row.get("id") or "").strip()
        for row in _safe_list(rows)
        if isinstance(row, dict) and str(row.get("id") or "").strip()
    }


def _target_endpoint_ready(
    active_endpoints: list[dict[str, Any]],
    *,
    model_id: str,
) -> bool:
    expected = str(model_id or "").strip()
    if not expected:
        return False
    return any(
        int(endpoint.get("port") or 0) == TARGET_MODEL_PORT
        and str(endpoint.get("path") or "") == "/v1/models"
        and expected in _endpoint_models(endpoint)
        for endpoint in active_endpoints
    )


def _target_profile_name(snapshot: dict[str, Any]) -> str:
    service_manifest = _manifest_payload(snapshot, "service_manifest")
    download_manifest = _manifest_payload(snapshot, "download_manifest")
    return str(
        snapshot.get("topology_profile")
        or service_manifest.get("topology_profile")
        or download_manifest.get("topology_profile")
        or ""
    ).strip().lower()


def _target_candidate(snapshot: dict[str, Any]) -> tuple[ModelCandidateManifest | None, list[str]]:
    wrapper = _safe_dict(snapshot.get("target_candidate_manifest"))
    if not bool(wrapper.get("present")):
        return None, ["target_candidate_manifest_missing"]
    data = _safe_dict(wrapper.get("data"))
    try:
        candidate = ModelCandidateManifest.from_dict(data)
    except (TypeError, ValueError) as exc:
        return None, [f"target_candidate_manifest_invalid:{safe_error_text(exc, limit=180)}"]
    evidence_errors = list(candidate.validate_evidence())
    if evidence_errors:
        return None, [f"target_candidate_evidence_invalid:{item}" for item in evidence_errors]
    return candidate, []


def _evaluate_target_model_server_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the single local Qwen3.8-27B runtime contract."""

    service_manifest = _manifest_payload(snapshot, "service_manifest")
    service_present = _manifest_present(snapshot, "service_manifest")
    manifest_services = _service_manifest_services(service_manifest)
    active_services = _active_service_lines(snapshot)
    active_endpoints = _active_endpoints(snapshot)
    gpu_rows = _gpu_rows(snapshot)
    gpu_processes = [
        str(item).strip() for item in _safe_list(snapshot.get("gpu_processes")) if str(item).strip()
    ]
    torch_info = _safe_dict(_manifest_payload(snapshot, "validation_manifest").get("torch"))
    candidate, candidate_errors = _target_candidate(snapshot)
    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    profile = _target_profile_name(snapshot)

    if profile != "target_single_model":
        blockers.append(
            _blocker(
                "target_topology_profile_missing_or_mismatched",
                "Target readiness requires an explicit target_single_model profile.",
                evidence={"profile": profile},
            )
        )
    for error in candidate_errors:
        error_code = str(error).split(":", 1)[0] or "target_candidate_manifest_invalid"
        blockers.append(
            _blocker(
                error_code,
                "The verified Qwen3.8-27B candidate manifest is missing or invalid.",
                evidence=error,
            )
        )

    topology = _target_topology_report(candidate) if candidate else _target_topology_report()
    expected_model_id = candidate.model_id if candidate else ""
    if not service_present:
        blockers.append(
            _blocker(
                "target_service_manifest_missing",
                "Target profile requires a service manifest for the single local carrier.",
                evidence=SERVICE_MANIFEST_PATH,
            )
        )
    elif str(service_manifest.get("topology_profile") or "").strip().lower() != "target_single_model":
        blockers.append(
            _blocker(
                "target_service_manifest_profile_mismatch",
                "The service manifest does not declare target_single_model.",
                evidence=service_manifest.get("topology_profile"),
            )
        )
    if len(manifest_services) != 1:
        blockers.append(
            _blocker(
                "target_service_manifest_count_invalid",
                "Target profile must declare exactly one local model service.",
                evidence={"service_count": len(manifest_services)},
            )
        )
    target_service = manifest_services[0] if len(manifest_services) == 1 else {}
    expected_service_fields = {
        "service_name": TARGET_SERVICE_NAME,
        "slot": TARGET_SERVICE_SLOT,
        "port": TARGET_MODEL_PORT,
    }
    for field, expected in expected_service_fields.items():
        actual = target_service.get(field)
        if actual != expected:
            blockers.append(
                _blocker(
                    "target_service_contract_mismatch",
                    f"Target service field {field} does not match the single-model contract.",
                    evidence={"field": field, "expected": expected, "actual": actual},
                )
            )
    if target_service.get("live_routing_enabled") is not False or target_service.get("shadow_only") is not True:
        blockers.append(
            _blocker(
                "target_service_live_policy_invalid",
                "The target service must remain shadow-only with live routing disabled.",
                evidence=target_service,
            )
        )
    if expected_model_id and str(target_service.get("served_model_name") or "") != expected_model_id:
        blockers.append(
            _blocker(
                "target_service_model_identity_mismatch",
                "Target service served_model_name must equal the verified candidate model id.",
                evidence={"expected": expected_model_id, "actual": target_service.get("served_model_name")},
            )
        )

    service_ready = _service_name_active(TARGET_SERVICE_NAME, active_services)
    endpoint_ready = _target_endpoint_ready(active_endpoints, model_id=expected_model_id)
    target_report = dict(topology)
    target_report["service_ready"] = service_ready
    target_report["endpoint_ready"] = endpoint_ready
    if not service_ready:
        blockers.append(
            _blocker(
                "target_service_inactive",
                "bb-phase3-llm-target.service is not active.",
                evidence=TARGET_SERVICE_NAME,
            )
        )
    if not endpoint_ready:
        blockers.append(
            _blocker(
                "target_endpoint_not_ready",
                "Port 8000 /v1/models did not return the verified candidate model id.",
                evidence={"port": TARGET_MODEL_PORT, "model_id": expected_model_id},
            )
        )
    legacy_active = [
        line for line in active_services
        if any(marker.lower() in line.lower() for marker in RETIRED_SERVICE_MARKERS)
    ]
    if legacy_active:
        blockers.append(
            _blocker(
                "legacy_model_services_active",
                "Retired 14B model services must be inactive under target_single_model.",
                evidence=legacy_active[:12],
            )
        )
    observed_gpu_count = max(len(gpu_rows), int(torch_info.get("device_count") or 0))
    if observed_gpu_count < MINIMUM_RUNTIME_GPU_COUNT:
        blockers.append(
            _blocker(
                "gpu_count_below_target_contract",
                "At least one GPU must be visible for the target carrier.",
                evidence={"observed": observed_gpu_count},
            )
        )
    if not gpu_processes:
        warnings.append(
            _warning(
                "gpu_runtime_process_evidence_missing",
                "No GPU process listing was returned; endpoint evidence is still required.",
            )
        )

    target_report["activation_blocked"] = bool(blockers or not (service_ready and endpoint_ready))
    # Keep the topology contract machine-readable.  Evidence can be a nested
    # object or a path string, but downstream gates need stable blocker codes
    # rather than Python's representation of arbitrary evidence.
    target_report["blockers"] = [
        str(item.get("code") or "unknown_target_blocker") for item in blockers
    ]
    runtime_ready = not blockers
    manifest_report = {
        "slot": target_service.get("slot") or TARGET_SERVICE_SLOT,
        "role": target_service.get("role") or "decision_and_expert_carrier",
        "service_name": TARGET_SERVICE_NAME,
        "port": TARGET_MODEL_PORT,
        "served_model_name": target_service.get("served_model_name") or expected_model_id,
        "shadow_only": target_service.get("shadow_only") is True,
        "live_routing_enabled": target_service.get("live_routing_enabled") is True,
        "service_active": service_ready,
        "endpoint_ready": endpoint_ready,
        "ready": service_ready and endpoint_ready and not blockers,
    }
    return {
        "status": "ready" if runtime_ready else "blocked",
        "read_only": True,
        "audit_only": True,
        "can_mutate_remote": False,
        "can_start_services": False,
        "can_change_live_routing": False,
        "live_routing_enabled": False,
        "topology_profile": "target_single_model",
        "artifact_ready": bool(candidate and not candidate_errors),
        "runtime_ready": runtime_ready,
        "phase3_model_service_go_live_blocked": not runtime_ready,
        "policy_id": "phase3_target_single_model.v1",
        "phase3_root": PHASE3_ROOT,
        "deployment_contract": "evidence_driven_target_single_model_runtime",
        "target_model_topology": target_report,
        "target_candidate_manifest": {
            "present": _manifest_present(snapshot, "target_candidate_manifest"),
            "path": TARGET_CANDIDATE_MANIFEST_PATH,
            "status": candidate.status if candidate else "invalid",
            "model_id": candidate.model_id if candidate else "",
            "repo_id": candidate.repo_id if candidate else "",
            "revision": candidate.revision if candidate else "",
            "architecture": candidate.architecture if candidate else "",
            "model_type": candidate.model_type if candidate else "",
            "runtime": candidate.runtime.to_dict() if candidate else {},
            "validated_at": candidate.validated_at if candidate else "",
            "errors": candidate_errors,
        },
        "expected_gpu_count": MINIMUM_RUNTIME_GPU_COUNT,
        "gpu_count": observed_gpu_count,
        "gpu_rows": gpu_rows[:16],
        "gpu_process_count": len(gpu_processes),
        "gpu_processes": gpu_processes[:40],
        "active_model_service_count": len(active_services),
        "active_model_services": active_services[:40],
        "active_endpoint_count": len(active_endpoints),
        "active_endpoints": active_endpoints[:16],
        "service_manifest_path": SERVICE_MANIFEST_PATH,
        "service_manifest": {"present": service_present, "service_count": len(manifest_services), "data": service_manifest},
        "manifest_service_count": 1 if target_service else 0,
        "manifest_service_ready_count": 1 if manifest_report["ready"] else 0,
        "manifest_services": [manifest_report] if target_service else [],
        "required_slots": [],
        "required_slot_count": 0,
        "required_slot_ready_count": 0,
        "download_manifest_path": DOWNLOAD_MANIFEST_PATH,
        "validation_manifest_path": VALIDATION_MANIFEST_PATH,
        "download_manifest": {"present": _manifest_present(snapshot, "download_manifest")},
        "validation_manifest": {"present": _manifest_present(snapshot, "validation_manifest"), "torch": torch_info},
        "listening_ports": _safe_list(snapshot.get("listening_ports"))[:80],
        "model_paths": _safe_list(snapshot.get("model_paths"))[:120],
        "manifest_files": _safe_list(snapshot.get("manifest_files"))[:80],
        "blockers": blockers,
        "warnings": warnings,
        "checked_at": _now_iso(),
    }


def evaluate_phase3_model_server_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Evaluate model artifacts and runtime service readiness without mutation."""
    # There is one executable topology.  A missing or unknown profile is a
    # blocker inside the same target evaluator; no legacy evaluator is used as
    # a fallback and no old endpoint can unlock readiness.
    return _evaluate_target_model_server_snapshot(snapshot)

def render_phase3_model_server_probe() -> str:
    """Render the read-only remote probe executed on the model server."""

    return textwrap.dedent(f"""
        import json
        import os
        import subprocess

        DOWNLOAD_MANIFEST_PATH = {json.dumps(DOWNLOAD_MANIFEST_PATH)}
        VALIDATION_MANIFEST_PATH = {json.dumps(VALIDATION_MANIFEST_PATH)}
        SERVICE_MANIFEST_PATH = {json.dumps(SERVICE_MANIFEST_PATH)}
        TARGET_CANDIDATE_MANIFEST_PATH = {json.dumps(TARGET_CANDIDATE_MANIFEST_PATH)}
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

        def read_candidate_json(path):
            if not os.path.exists(path):
                return {{"present": False, "data": {{}}}}
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except Exception as exc:
                return {{"present": True, "data": {{}}, "error": str(exc)[:180]}}
            if not isinstance(data, dict):
                return {{"present": True, "data": {{}}, "error": "candidate manifest is not an object"}}
            keys = (
                "manifest_version", "status", "model_id", "repo_id", "revision",
                "model_path", "tokenizer_path", "license", "quantization",
                "architecture", "model_type", "language_model_only",
                "text_inference_verified", "context_length", "config_sha256",
                "tokenizer_sha256", "gpu_memory_peak_gib", "inference_p95_ms",
                "max_concurrency", "storage_available_gib", "storage_required_free_gib",
                "validated_at", "validator_version", "runtime",
            )
            compact = {{key: data.get(key) for key in keys if key in data}}
            compact["weight_files"] = [
                {{key: item.get(key) for key in ("path", "size_bytes", "sha256") if key in item}}
                for item in data.get("weight_files", [])
                if isinstance(item, dict)
            ]
            return {{"present": True, "data": compact}}

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
            # The target single-model profile is the default contract.  An
            # explicit legacy audit remains available through the dedicated
            # status script, but readiness must not infer legacy production
            # semantics merely because old manifests omit a profile field.
            "topology_profile": os.environ.get(
                "BB_MODEL_TOPOLOGY_PROFILE", "target_single_model"
            ).strip().lower(),
            "download_manifest": read_json(DOWNLOAD_MANIFEST_PATH),
            "validation_manifest": read_json(VALIDATION_MANIFEST_PATH),
            "service_manifest": read_json(SERVICE_MANIFEST_PATH),
            "target_candidate_manifest": read_candidate_json(TARGET_CANDIDATE_MANIFEST_PATH),
            "services": lines(run(
                "systemctl list-units --type=service --all --no-pager "
                "| grep -Ei 'bb-phase3-llm-target|phase3|quant|local-ai|redis' || true",
                timeout=10,
            )["stdout"], 120),
            "unit_files": lines(run(
                "systemctl list-unit-files --type=service --no-pager "
                "| grep -Ei 'bb-phase3-llm-target|phase3|quant|local-ai|redis' || true",
                timeout=10,
            )["stdout"], 120),
            "listening_ports": lines(run(
                "ss -ltnp 2>/dev/null | grep -E ':(8000|8101|18000|18001)\\\\b' || true",
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
        """).strip()


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
        status = (
            "model_server_not_configured"
            if isinstance(exc, ModelServerConfigNotConfigured)
            else "model_server_probe_unavailable"
        )
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

