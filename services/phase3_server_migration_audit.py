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

from core.phase3_model_contract import PHASE3_APPROVED_RUNTIME_MODEL_PATHS
from core.remote_ssh import connect_remote_ssh, exec_remote_command
from core.safe_output import safe_error_text
from services.model_server_config import (
    ModelServerConfigError,
    ModelServerConfigNotConfigured,
    load_model_server_info_for_monitor,
    load_model_server_info_for_monitor_async,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESOURCE_RELEASE_MARKER_PATH = "/data/BB/manifests/phase3_resource_release_manifest.json"
MIGRATION_MANIFEST_PATH = "/data/BB/manifests/phase3_migration_whitelist.json"
PHASE3_RESOURCE_POLICY_ID = "phase3_stop_legacy_release_gpu_keep_old_data_2026_06_26"
PHASE3_ROOT = "/data/BB"
PROCESS_EVIDENCE_TEXT_LIMIT = 700

FORBIDDEN_LEGACY_SERVICE_NAMES = (
    "qwen3-122b.service",
    "qwen3-5-122b.service",
    "qwen3-32b-main.service",
    "qwen3-32b-review.service",
    "qwen3-14b.service",
    "deepseek-32b-main.service",
    "deepseek-14b-main.service",
    "deepseek-r1-32b-main.service",
)

LEGACY_DATA_PATHS = (
    "/data/trade_models/Qwen/Qwen3-32B-AWQ",
    "/data/trade_models/Qwen/Qwen3.5-122B",
    "/data/trade_models/Qwen/Qwen3_5_122B",
    "/data/trade_models/DeepSeek/DeepSeek-R1-Distill-Qwen-32B-AWQ",
    "/data/trade_models/DeepSeek/deepseek-r1-distill-qwen-32b-awq",
    "/data/trade_ai/models",
    "/data/trade_ai/bundles",
    "/data/trade_ai/experiments",
    "/data/trade_ai/legacy",
    "/data/trade_ai/logs/qwen3_32b_main.log",
    "/data/trade_ai/logs/qwen3_32b_review.log",
    "/data/trade_ai/logs/deepseek_32b_main.log",
)

PHASE3_REQUIRED_ROOTS = (
    "/data/BB",
    "/data/BB/models",
    "/data/BB/cache",
    "/data/BB/training",
    "/data/BB/runtime",
    "/data/BB/logs",
    "/data/BB/manifests",
)

PHASE3_ALLOWED_PROCESS_HINTS = (
    "/data/BB/envs/phase3-quant/bin/python",
    "/data/BB/models/",
    "/data/BB/runtime/",
    "/data/BB/services/",
    "/data/BB/scripts/",
)

AUDIT_PROBE_PROCESS_HINTS = (
    "python3 - <<'py'",
    'python3 - <<"py"',
    "python - <<'py'",
    'python - <<"py"',
    "systemctl list-unit-files",
    "systemctl list-units",
    "grep -ei",
    "grep -e",
    "scripts/run_phase3_go_no_go_report.py",
    "scripts/run_phase3_model_server_readiness_audit.py",
    "scripts/run_phase3_rebuild_preflight.py",
    "scripts/run_phase3_paper_resume_preflight.py",
    "scripts/run_phase3_paper_resume_observation.py",
)

LEGACY_PROCESS_HINTS_ALWAYS_BLOCK = (
    "qwen3.5-122b",
    "qwen3_5_122b",
    "qwen3-5-122b",
    "122b",
    "qwen3-32b",
    "deepseek-r1-distill-qwen-32b",
    "deepseek_32b",
    "qwen3_32b",
    "/data/trade_ai/",
    "open-webui",
    "text-generation-webui",
    "ollama",
    "finquant_expert_alias.py",
)

LEGACY_PROCESS_HINTS_BLOCK_OUTSIDE_PHASE3: tuple[str, ...] = ()

APPROVED_MIGRATION_CATEGORIES = (
    "platform_secure_settings_reference",
    "clean_training_export_manifest",
    "approved_phase3_deploy_manifest",
    "regenerated_runtime_secret",
    "operator_reset_evidence",
)

MIGRATION_SOURCES_ALLOWED = (
    "platform_secure_settings",
    "platform_clean_training_view",
    "current_repository",
    "operator_regenerated",
    "manual_reset_report",
)

RemoteProbe = Callable[[], dict[str, Any]]
InfoLoader = Callable[[Path], Any]
AsyncInfoLoader = Callable[[Path], Any]
SshConnector = Callable[..., Any]
CommandExecutor = Callable[..., Any]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _safe_bool(value: Any) -> bool:
    return bool(value) if value is not None else False


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


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


def _path_hit(row: Any) -> bool:
    data = _safe_dict(row)
    return bool(data.get("exists"))


def _service_hit(row: Any) -> bool:
    data = _safe_dict(row)
    return bool(data.get("active") or data.get("enabled"))


def _process_line(value: Any) -> str:
    return str(value or "").strip()


def _process_lower(value: Any) -> str:
    return _process_line(value).lower()


def _dedupe_processes(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _compact_process_evidence(value: Any, *, limit: int = PROCESS_EVIDENCE_TEXT_LIMIT) -> str:
    line = _process_line(value)
    if len(line) <= limit:
        return line
    return f"{line[:limit]}...[truncated process evidence at {limit} chars]"


def _is_phase3_allowed_process(value: Any) -> bool:
    line = _process_lower(value)
    if not line:
        return False
    if any(path.lower() in line for path in PHASE3_APPROVED_RUNTIME_MODEL_PATHS):
        return True
    if PHASE3_ROOT.lower() not in line:
        return False
    return any(hint.lower() in line for hint in PHASE3_ALLOWED_PROCESS_HINTS)


def _is_audit_probe_process(value: Any) -> bool:
    line = _process_lower(value)
    if not line:
        return False
    return any(hint.lower() in line for hint in AUDIT_PROBE_PROCESS_HINTS)


def _is_legacy_process(value: Any) -> bool:
    line = _process_lower(value)
    if not line:
        return False
    if _is_audit_probe_process(line):
        return False
    if any(hint in line for hint in LEGACY_PROCESS_HINTS_ALWAYS_BLOCK):
        return True
    if _is_phase3_allowed_process(line):
        return False
    return any(hint in line for hint in LEGACY_PROCESS_HINTS_BLOCK_OUTSIDE_PHASE3)


def _service_rows(snapshot: dict[str, Any], key: str) -> list[dict[str, Any]]:
    return [_safe_dict(item) for item in _safe_list(snapshot.get(key))]


def _manifest_items(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = manifest.get("items")
    if not isinstance(rows, list):
        rows = manifest.get("assets")
    return [_safe_dict(item) for item in _safe_list(rows)]


def _manifest_category(item: dict[str, Any]) -> str:
    return str(item.get("category") or item.get("type") or "").strip()


def _manifest_source(item: dict[str, Any]) -> str:
    return str(item.get("source") or item.get("source_type") or "").strip()


def evaluate_phase3_server_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the model-server reset/migration snapshot without mutating it."""

    marker = _safe_dict(snapshot.get("resource_release_marker") or snapshot.get("reset_marker"))
    marker_present = bool(marker.get("present"))
    marker_data = _safe_dict(marker.get("data"))
    marker_policy = str(marker_data.get("policy_id") or marker_data.get("policy") or "").strip()
    legacy_resources_stopped = _safe_bool(
        marker_data.get("legacy_resources_stopped")
        if "legacy_resources_stopped" in marker_data
        else marker_data.get("full_reset")
    )
    data_preserved = _safe_bool(
        marker_data.get("old_data_preserved")
        if "old_data_preserved" in marker_data
        else not _safe_bool(marker_data.get("old_data_deleted"))
    )
    phase3_root = str(marker_data.get("phase3_root") or PHASE3_ROOT).strip()

    legacy_paths = [
        _safe_dict(item)
        for item in _safe_list(snapshot.get("legacy_data_paths"))
        if _path_hit(item)
    ]
    if not legacy_paths:
        legacy_paths = [
            _safe_dict(item)
            for item in _safe_list(snapshot.get("forbidden_paths"))
            if _path_hit(item)
        ]
    phase3_roots = [_safe_dict(item) for item in _safe_list(snapshot.get("phase3_roots"))]
    missing_phase3_roots = [
        item.get("path") for item in phase3_roots if not _safe_bool(item.get("exists"))
    ]
    forbidden_services = [
        _safe_dict(item)
        for item in _safe_list(snapshot.get("forbidden_services"))
        if _service_hit(item)
    ]
    raw_processes = [_process_line(item) for item in _safe_list(snapshot.get("legacy_processes"))]
    raw_processes.extend(
        _process_line(item) for item in _safe_list(snapshot.get("candidate_model_processes"))
    )
    raw_processes = _dedupe_processes([item for item in raw_processes if item])
    legacy_processes = [
        item for item in raw_processes if _is_legacy_process(item)
    ]
    phase3_allowed_processes = [
        item
        for item in raw_processes
        if _is_phase3_allowed_process(item) and not _is_legacy_process(item)
    ]
    ignored_probe_processes = [item for item in raw_processes if _is_audit_probe_process(item)]
    legacy_process_evidence = [_compact_process_evidence(item) for item in legacy_processes]
    phase3_allowed_process_evidence = [
        _compact_process_evidence(item) for item in phase3_allowed_processes
    ]
    ignored_probe_process_evidence = [
        _compact_process_evidence(item) for item in ignored_probe_processes
    ]

    manifest = _safe_dict(snapshot.get("migration_manifest"))
    manifest_present = bool(manifest.get("present"))
    manifest_data = _safe_dict(manifest.get("data"))
    manifest_items = _manifest_items(manifest_data)
    disallowed_categories = sorted(
        {
            category
            for category in (_manifest_category(item) for item in manifest_items)
            if category and category not in APPROVED_MIGRATION_CATEGORIES
        }
    )
    disallowed_sources = sorted(
        {
            source
            for source in (_manifest_source(item) for item in manifest_items)
            if source and source not in MIGRATION_SOURCES_ALLOWED
        }
    )

    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    if not marker_present:
        blockers.append(
            _blocker(
                "resource_release_marker_missing",
                "Phase 3 resource-release evidence is missing on the new model server.",
                evidence=RESOURCE_RELEASE_MARKER_PATH,
            )
        )
    elif marker_policy not in {PHASE3_RESOURCE_POLICY_ID, "phase3_full_reset_2026_06_26"}:
        blockers.append(
            _blocker(
                "resource_release_marker_invalid",
                "Resource-release evidence exists but does not prove the required Phase 3 policy.",
                evidence={
                    "expected_policy_id": PHASE3_RESOURCE_POLICY_ID,
                    "actual_policy_id": marker_policy,
                    "legacy_resources_stopped": legacy_resources_stopped,
                    "old_data_preserved": data_preserved,
                },
            )
        )
    else:
        if not legacy_resources_stopped:
            blockers.append(
                _blocker(
                    "legacy_resources_not_released",
                    "Legacy services/processes/containers must be stopped before Phase 3 go-live.",
                    evidence=RESOURCE_RELEASE_MARKER_PATH,
                )
            )
        if phase3_root != PHASE3_ROOT:
            blockers.append(
                _blocker(
                    "phase3_root_not_data_bb",
                    "Phase 3 model/cache/training/runtime/log data must be rooted under /data/BB.",
                    evidence=phase3_root,
                )
            )
        if not data_preserved:
            warnings.append(
                _warning(
                    "old_data_not_marked_preserved",
                    "Current policy is to preserve old data in place and only release resource usage.",
                    evidence=RESOURCE_RELEASE_MARKER_PATH,
                )
            )

    if missing_phase3_roots:
        blockers.append(
            _blocker(
                "phase3_required_roots_missing",
                "Required /data/BB Phase 3 workspace directories are missing.",
                evidence=missing_phase3_roots,
            )
        )

    if legacy_paths:
        warnings.append(
            _warning(
                "legacy_data_paths_preserved",
                "Legacy data paths still exist by policy; they must remain isolated from Phase 3 runtime.",
                evidence=[item.get("path") for item in legacy_paths],
            )
        )
    if forbidden_services:
        blockers.append(
            _blocker(
                "legacy_services_present",
                "Legacy model services still exist or are active on the new model server.",
                evidence=[
                    {
                        "service": item.get("name"),
                        "active": item.get("active"),
                        "enabled": item.get("enabled"),
                    }
                    for item in forbidden_services
                ],
            )
        )
    if legacy_processes:
        blockers.append(
            _blocker(
                "legacy_processes_running",
                "Legacy model/download processes are still running.",
                evidence=legacy_process_evidence[:10],
            )
        )

    if not manifest_present:
        blockers.append(
            _blocker(
                "migration_manifest_missing",
                "Old-server migration must be controlled by a whitelist manifest, not an ad-hoc copy.",
                evidence=MIGRATION_MANIFEST_PATH,
            )
        )
    else:
        if not bool(manifest_data.get("whitelist_only", True)):
            blockers.append(
                _blocker(
                    "migration_not_whitelist_only",
                    "Migration manifest does not explicitly enforce whitelist-only migration.",
                )
            )
        if disallowed_categories:
            blockers.append(
                _blocker(
                    "migration_category_not_approved",
                    "Migration manifest contains categories outside the Phase 3 whitelist.",
                    evidence=disallowed_categories,
                )
            )
        if disallowed_sources:
            blockers.append(
                _blocker(
                    "migration_source_not_approved",
                    "Migration manifest contains non-approved data sources.",
                    evidence=disallowed_sources,
                )
            )
        if not manifest_items:
            warnings.append(
                _warning(
                    "migration_manifest_empty",
                    "Migration manifest is present but contains no explicit asset rows.",
                )
            )

    status = "ready" if not blockers else "blocked"
    return {
        "status": status,
        "read_only": True,
        "audit_only": True,
        "can_mutate_remote": False,
        "can_delete_remote_data": False,
        "phase3_go_live_blocked": bool(blockers),
        "deployment_contract": "evidence_driven_model_runtime",
        "policy_id": PHASE3_RESOURCE_POLICY_ID,
        "resource_release_marker_path": RESOURCE_RELEASE_MARKER_PATH,
        "reset_marker_path": RESOURCE_RELEASE_MARKER_PATH,
        "migration_manifest_path": MIGRATION_MANIFEST_PATH,
        "new_server_policy": {
            "full_reset_required": False,
            "legacy_resource_release_required": True,
            "keep_existing_model_server_data": True,
            "legacy_model_or_cache_allowed_as_isolated_data": True,
            "phase3_control_plane_root": PHASE3_ROOT,
            "approved_runtime_model_paths": list(PHASE3_APPROVED_RUNTIME_MODEL_PATHS),
            "target_usage": "verified_phase3_runtime_only",
        },
        "migration_policy": {
            "whitelist_only": True,
            "whole_disk_copy_allowed": False,
            "old_server_production_role_after_migration": "retired",
            "approved_categories": list(APPROVED_MIGRATION_CATEGORIES),
            "approved_sources": list(MIGRATION_SOURCES_ALLOWED),
        },
        "resource_release_marker": {
            "present": marker_present,
            "policy_id": marker_policy,
            "legacy_resources_stopped": legacy_resources_stopped,
            "old_data_preserved": data_preserved,
            "phase3_root": phase3_root,
            "data": marker_data,
        },
        "reset_marker": {
            "present": marker_present,
            "policy_id": marker_policy,
            "full_reset": False,
            "data": marker_data,
        },
        "migration_manifest": {
            "present": manifest_present,
            "whitelist_only": (
                bool(manifest_data.get("whitelist_only", True)) if manifest_present else False
            ),
            "item_count": len(manifest_items),
            "disallowed_categories": disallowed_categories,
            "disallowed_sources": disallowed_sources,
            "data": manifest_data,
        },
        "phase3_root": PHASE3_ROOT,
        "phase3_root_count": len(phase3_roots),
        "missing_phase3_roots": missing_phase3_roots,
        "legacy_data_path_count": len(legacy_paths),
        "forbidden_path_count": 0,
        "forbidden_service_count": len(forbidden_services),
        "legacy_process_count": len(legacy_processes),
        "phase3_allowed_process_count": len(phase3_allowed_processes),
        "ignored_probe_process_count": len(ignored_probe_processes),
        "legacy_data_paths": legacy_paths[:30],
        "forbidden_paths": [],
        "forbidden_services": forbidden_services[:30],
        "legacy_processes": legacy_process_evidence[:10],
        "phase3_allowed_processes": phase3_allowed_process_evidence[:10],
        "ignored_probe_processes": ignored_probe_process_evidence[:10],
        "approved_roots": _safe_list(snapshot.get("approved_roots")),
        "blockers": blockers,
        "warnings": warnings,
        "checked_at": _now_iso(),
    }


def render_phase3_server_probe() -> str:
    """Render the read-only remote probe executed on the model server."""

    return textwrap.dedent(f"""
        import json
        import os
        import subprocess

        RESOURCE_RELEASE_MARKER_PATH = {json.dumps(RESOURCE_RELEASE_MARKER_PATH)}
        MIGRATION_MANIFEST_PATH = {json.dumps(MIGRATION_MANIFEST_PATH)}
        FORBIDDEN_LEGACY_SERVICE_NAMES = {json.dumps(FORBIDDEN_LEGACY_SERVICE_NAMES)}
        LEGACY_DATA_PATHS = {json.dumps(LEGACY_DATA_PATHS)}
        PHASE3_REQUIRED_ROOTS = {json.dumps(PHASE3_REQUIRED_ROOTS)}
        LEGACY_PROCESS_PATTERN = (
            "Qwen3-32B|Qwen3.5-122B|Qwen3_5_122B|122B|"
            "DeepSeek-R1-Distill-Qwen-32B|deepseek_32b|qwen3_32b|"
            "trade_ai|trade_models|open-webui|text-generation-webui|ollama|vllm"
        )
        PHASE3_ROOT = {json.dumps(PHASE3_ROOT)}
        PHASE3_ALLOWED_PROCESS_HINTS = {json.dumps(PHASE3_ALLOWED_PROCESS_HINTS)}
        PHASE3_APPROVED_RUNTIME_MODEL_PATHS = {
            json.dumps(PHASE3_APPROVED_RUNTIME_MODEL_PATHS)
        }
        AUDIT_PROBE_PROCESS_HINTS = {json.dumps(AUDIT_PROBE_PROCESS_HINTS)}
        LEGACY_PROCESS_HINTS_ALWAYS_BLOCK = {json.dumps(LEGACY_PROCESS_HINTS_ALWAYS_BLOCK)}
        LEGACY_PROCESS_HINTS_BLOCK_OUTSIDE_PHASE3 = {
            json.dumps(LEGACY_PROCESS_HINTS_BLOCK_OUTSIDE_PHASE3)
        }
        PROCESS_EVIDENCE_TEXT_LIMIT = {json.dumps(PROCESS_EVIDENCE_TEXT_LIMIT)}

        def compact_process_evidence(line):
            text = str(line or "").strip()
            if len(text) <= PROCESS_EVIDENCE_TEXT_LIMIT:
                return text
            return (
                text[:PROCESS_EVIDENCE_TEXT_LIMIT]
                + "...[truncated process evidence at "
                + str(PROCESS_EVIDENCE_TEXT_LIMIT)
                + " chars]"
            )

        def is_phase3_allowed_process(line):
            lowered = str(line or "").strip().lower()
            if not lowered:
                return False
            if any(path.lower() in lowered for path in PHASE3_APPROVED_RUNTIME_MODEL_PATHS):
                return True
            if PHASE3_ROOT.lower() not in lowered:
                return False
            return any(hint.lower() in lowered for hint in PHASE3_ALLOWED_PROCESS_HINTS)

        def is_audit_probe_process(line):
            lowered = str(line or "").strip().lower()
            if not lowered:
                return False
            return any(hint.lower() in lowered for hint in AUDIT_PROBE_PROCESS_HINTS)

        def is_legacy_process(line):
            lowered = str(line or "").strip().lower()
            if not lowered:
                return False
            if is_audit_probe_process(lowered):
                return False
            if any(hint in lowered for hint in LEGACY_PROCESS_HINTS_ALWAYS_BLOCK):
                return True
            if is_phase3_allowed_process(lowered):
                return False
            return any(hint in lowered for hint in LEGACY_PROCESS_HINTS_BLOCK_OUTSIDE_PHASE3)

        def run(args, timeout=4):
            try:
                result = subprocess.run(
                    args,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                    check=False,
                )
                return result.returncode, result.stdout.strip(), result.stderr.strip()
            except Exception as exc:
                return 124, "", str(exc)

        def read_json(path):
            if not os.path.exists(path):
                return {{"present": False, "data": {{}}}}
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception as exc:
                return {{"present": True, "data": {{}}, "error": str(exc)[:180]}}
            return {{"present": True, "data": data if isinstance(data, dict) else {{}}}}

        def path_state(path):
            exists = os.path.exists(path)
            state = {{
                "path": path,
                "exists": exists,
                "kind": "missing",
                "sample_children": [],
            }}
            if not exists:
                return state
            if os.path.isdir(path):
                state["kind"] = "directory"
                try:
                    state["sample_children"] = sorted(os.listdir(path))[:20]
                except Exception as exc:
                    state["error"] = str(exc)[:180]
            elif os.path.isfile(path):
                state["kind"] = "file"
                try:
                    state["size_bytes"] = os.path.getsize(path)
                except OSError:
                    state["size_bytes"] = None
            else:
                state["kind"] = "other"
            return state

        def service_state(name):
            active_code, active_out, _ = run(["systemctl", "is-active", name])
            enabled_code, enabled_out, _ = run(["systemctl", "is-enabled", name])
            cat_code, _cat_out, _cat_err = run(["systemctl", "cat", name], timeout=3)
            return {{
                "name": name,
                "unit_exists": cat_code == 0,
                "active": active_code == 0 and active_out == "active",
                "active_state": active_out or "unknown",
                "enabled": enabled_code == 0 and enabled_out in {{"enabled", "static"}},
                "enabled_state": enabled_out or "unknown",
            }}

        _pg_code, pg_out, _pg_err = run(["pgrep", "-af", LEGACY_PROCESS_PATTERN], timeout=4)
        candidate_model_processes = [
            line.strip() for line in pg_out.splitlines()
            if line.strip() and "pgrep -af" not in line
        ][:20]
        legacy_processes = [
            compact_process_evidence(line) for line in candidate_model_processes
            if is_legacy_process(line)
        ][:20]
        phase3_allowed_processes = [
            compact_process_evidence(line) for line in candidate_model_processes
            if is_phase3_allowed_process(line) and not is_legacy_process(line)
        ][:20]
        ignored_probe_processes = [
            compact_process_evidence(line) for line in candidate_model_processes
            if is_audit_probe_process(line)
        ][:20]
        candidate_model_processes = [
            compact_process_evidence(line) for line in candidate_model_processes
        ]

        payload = {{
            "resource_release_marker": read_json(RESOURCE_RELEASE_MARKER_PATH),
            "migration_manifest": read_json(MIGRATION_MANIFEST_PATH),
            "legacy_data_paths": [path_state(path) for path in LEGACY_DATA_PATHS],
            "phase3_roots": [path_state(path) for path in PHASE3_REQUIRED_ROOTS],
            "forbidden_services": [
                service_state(name) for name in FORBIDDEN_LEGACY_SERVICE_NAMES
            ],
            "candidate_model_processes": candidate_model_processes,
            "legacy_processes": legacy_processes,
            "phase3_allowed_processes": phase3_allowed_processes,
            "ignored_probe_processes": ignored_probe_processes,
            "approved_roots": [
                path_state("/data/BB"),
                path_state("/data/trade_ai"),
                path_state("/data/trade_models"),
            ],
        }}
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        """).strip()


def _remote_command() -> str:
    script = render_phase3_server_probe()
    if "\nPY\n" in f"\n{script}\n":
        raise ValueError("Phase 3 server probe cannot contain a bare PY heredoc delimiter.")
    return f"python3 - <<'PY'\n{script}\nPY"


@dataclass(slots=True)
class Phase3ServerMigrationAuditService:
    """Read-only gate for Phase 3 model-server reset and migration readiness."""

    project_root: Path = PROJECT_ROOT
    remote_probe: RemoteProbe | None = None
    info_loader: InfoLoader = load_model_server_info_for_monitor
    async_info_loader: AsyncInfoLoader | None = load_model_server_info_for_monitor_async
    ssh_connector: SshConnector = connect_remote_ssh
    command_executor: CommandExecutor = exec_remote_command
    timeout_seconds: int = 18

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
        result = loader(self.project_root)
        if inspect.isawaitable(result):
            return await result
        return result

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
                timeout=max(int(self.timeout_seconds or 18), 5),
                max_output_chars=20_000,
            )
        finally:
            ssh.close()
        if result.status != 0:
            raise RuntimeError(
                safe_error_text(
                    result.stderr or result.stdout or "phase3 server probe failed",
                    fallback="phase3 server probe failed",
                )
            )
        try:
            payload = json.loads(str(result.stdout or "{}"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                safe_error_text(result.stdout or result.stderr or "invalid phase3 probe payload")
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError("phase3 server probe payload was not an object")
        return payload

    def _evaluated_report(
        self,
        snapshot: dict[str, Any],
        *,
        started_at: datetime,
    ) -> dict[str, Any]:
        report = evaluate_phase3_server_snapshot(snapshot)
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
            "Phase 3 server resource-release/migration readiness could not be verified.",
            evidence=safe_error_text(exc, limit=180),
        )
        return {
            "status": "unverified",
            "read_only": True,
            "audit_only": True,
            "can_mutate_remote": False,
            "can_delete_remote_data": False,
            "phase3_go_live_blocked": True,
            "remote_probe_available": False,
            "error": safe_error_text(exc, limit=180),
            "policy_id": PHASE3_RESOURCE_POLICY_ID,
            "resource_release_marker_path": RESOURCE_RELEASE_MARKER_PATH,
            "reset_marker_path": RESOURCE_RELEASE_MARKER_PATH,
            "migration_manifest_path": MIGRATION_MANIFEST_PATH,
            "new_server_policy": {
                "full_reset_required": False,
                "legacy_resource_release_required": True,
                "keep_existing_model_server_data": True,
                "legacy_model_or_cache_allowed_as_isolated_data": True,
                "phase3_data_root": PHASE3_ROOT,
                "target_usage": "full_capacity_for_phase3_quant_plan",
            },
            "migration_policy": {
                "whitelist_only": True,
                "whole_disk_copy_allowed": False,
                "old_server_production_role_after_migration": "retired",
                "approved_categories": list(APPROVED_MIGRATION_CATEGORIES),
                "approved_sources": list(MIGRATION_SOURCES_ALLOWED),
            },
            "blockers": [blocker],
            "warnings": [],
            "phase3_root": PHASE3_ROOT,
            "phase3_root_count": 0,
            "missing_phase3_roots": [],
            "legacy_data_path_count": 0,
            "legacy_data_paths": [],
            "forbidden_path_count": 0,
            "forbidden_service_count": 0,
            "legacy_process_count": 0,
            "phase3_allowed_process_count": 0,
            "ignored_probe_process_count": 0,
            "phase3_allowed_processes": [],
            "ignored_probe_processes": [],
            "checked_at": _now_iso(),
            "duration_seconds": round((datetime.now(UTC) - started_at).total_seconds(), 6),
        }
