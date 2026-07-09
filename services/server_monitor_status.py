"""Remote server and model monitor status service."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Any
from urllib.parse import urlparse

import httpx
import paramiko

from config.settings import settings
from core.remote_ssh import connect_remote_ssh, exec_remote_command
from core.safe_output import safe_error_text
from core.server_monitor_probe import (
    SERVER_MONITOR_REMOTE_COMMAND_TIMEOUT_SECONDS,
    display_provider_model_name,
    render_python_here_doc,
    render_server_monitor_probe,
)
from services.model_server_config import (
    ModelServerConfigError,
    ModelServerConfigNotConfigured,
    load_model_server_info_for_monitor,
    load_model_server_info_from_secure_settings,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVER_MONITOR_CACHE_TTL_SECONDS = 10.0
SERVER_MONITOR_STALE_CACHE_TTL_SECONDS = 60.0
PLATFORM_RUNTIME_PROBE_TIMEOUT_SECONDS = 3.5
LOCAL_AI_TOOLS_DEEP_PROBE_TIMEOUT_SECONDS = 15.0
LOCAL_AI_TOOLS_CHILD_ENDPOINT_TIMEOUT_SECONDS = {
    "profit_prediction": PLATFORM_RUNTIME_PROBE_TIMEOUT_SECONDS,
    "time_series_prediction": LOCAL_AI_TOOLS_DEEP_PROBE_TIMEOUT_SECONDS,
    "sentiment_analysis": 8.0,
    "exit_advice": 8.0,
}
PLATFORM_SERVICE_NAMES = (
    "bb-dashboard.service",
    "bb-paper-trading.service",
    "bb-model-tunnels.service",
    "postgresql.service",
    "redis-server.service",
    "redis.service",
)
ONLINE_PHASE3_QUANT_API_PLATFORM_BASE = "http://127.0.0.1:18001"
ONLINE_LOCAL_AI_TOOLS_PLATFORM_BASE = ONLINE_PHASE3_QUANT_API_PLATFORM_BASE
ONLINE_PHASE3_DEFAULT_AI_MODELS = (
    {
        "name": "phase3_decision_maker",
        "label": "Phase 3 decision maker",
        "api_base": "http://127.0.0.1:18000/v1",
        "model": "qwen3-32b-trade",
    },
    {
        "name": "phase3_high_risk_review",
        "label": "Phase 3 high-risk review",
        "api_base": "http://127.0.0.1:18002/v1",
        "model": "deepseek-r1-14b-risk",
    },
    {
        "name": "phase3_finquant_expert",
        "label": "Phase 3 FinQuant expert",
        "api_base": "http://127.0.0.1:18003/v1",
        "model": "BB-FinQuant-Expert-14B",
    },
)
ONLINE_PHASE3_TUNNEL_CONTRACTS = (
    {
        "name": "qwen3-32b-trade",
        "role": "decision_maker",
        "capability": "create_strategy",
        "local_port": 18_000,
        "api_base": "http://127.0.0.1:18000/v1",
        "model": "qwen3-32b-trade",
    },
    {
        "name": "phase3-quant-api",
        "role": "quant_tool",
        "capability": "quant_tool",
        "local_port": 18_001,
        "api_base": ONLINE_PHASE3_QUANT_API_PLATFORM_BASE,
        "model": "",
    },
    {
        "name": "deepseek-r1-14b-risk",
        "role": "risk_review",
        "capability": "risk_review",
        "local_port": 18_002,
        "api_base": "http://127.0.0.1:18002/v1",
        "model": "deepseek-r1-14b-risk",
    },
    {
        "name": "BB-FinQuant-Expert-14B",
        "role": "expert_pool",
        "capability": "finquant_expert",
        "local_port": 18_003,
        "api_base": "http://127.0.0.1:18003/v1",
        "model": "BB-FinQuant-Expert-14B",
    },
)
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}

InfoLoader = Callable[[Path], Any]
SshConnector = Callable[..., Any]
CommandExecutor = Callable[..., Any]
Clock = Callable[[], float]
ModelIdProvider = Callable[[], str]


def primary_provider_model_id() -> str:
    """Return the configured primary provider model id for monitor probes."""
    rows = [
        cfg for cfg in settings.get_fixed_ai_models(include_empty=True) if isinstance(cfg, dict)
    ]
    for cfg in rows:
        name = str(cfg.get("name") or "").strip()
        if name != "decision_maker" or cfg.get("enabled") is False:
            continue
        model = str(cfg.get("model") or "").strip()
        if model:
            return model
    for cfg in rows:
        if not isinstance(cfg, dict) or not cfg.get("enabled", True):
            continue
        model = str(cfg.get("model") or "").strip()
        if model:
            return model
    return str(settings.ai_model or "").strip()


def _host_from_api_base(value: Any) -> str:
    try:
        host = str(urlparse(str(value or "")).hostname or "").strip()
    except Exception:
        return ""
    return host


def _normalized_base_url(value: Any) -> str:
    return str(value or "").strip().rstrip("/")


def _redacted_info(info: Any) -> dict[str, Any]:
    if hasattr(info, "redacted"):
        try:
            value = info.redacted()
            if isinstance(value, dict):
                return value
        except Exception:
            return {}
    return {
        "host": getattr(info, "host", ""),
        "access_host": getattr(info, "access_host", "") or getattr(info, "host", ""),
        "port": getattr(info, "port", ""),
        "username": getattr(info, "username", ""),
        "source_path": str(getattr(info, "source_path", "") or ""),
    }


def _ssh_failure_payload(
    exc: Exception,
    *,
    checked_at: str,
    info: Any | None,
) -> dict[str, Any]:
    if isinstance(exc, paramiko.AuthenticationException):
        status = "ssh_auth_failed"
        message = (
            "SSH authentication failed for the configured model-server monitor. "
            "Update the SSH password in the server settings; the password is not printed."
        )
    else:
        status = "ssh_failed"
        message = safe_error_text(exc)
    return {
        "available": False,
        "remote_monitor_available": False,
        "status": status,
        "message": message,
        "checked_at": checked_at,
        "credential_source": _redacted_info(info) if info is not None else {},
    }


def _http_probe_error(status_code: int, data: Any) -> str:
    if 200 <= status_code < 300:
        return ""
    detail = ""
    if isinstance(data, dict):
        raw_detail = data.get("detail") or data.get("message") or data.get("error")
        if raw_detail:
            detail = safe_error_text(raw_detail, limit=180)
    if detail:
        return detail
    if status_code == 401:
        return "HTTP 401：本地量化工具 API Key 不匹配或未同步。"
    if status_code == 403:
        return "HTTP 403：本地量化工具拒绝访问。"
    if status_code == 404:
        return "HTTP 404：接口路径不存在。"
    if status_code >= 500:
        return f"HTTP {status_code}：服务端异常或模型正在启动。"
    if status_code > 0:
        return f"HTTP {status_code}"
    return ""


def _http_probe_category(status_code: int) -> str:
    if 200 <= status_code < 300:
        return "ok"
    if status_code == 401:
        return "auth_failed"
    if status_code == 403:
        return "auth_forbidden"
    if status_code == 404:
        return "not_found"
    if status_code >= 500:
        return "server_error"
    if status_code > 0:
        return "http_error"
    return "network_error"


def _platform_tunnel_contract(actual_base: str, expected_base: str) -> dict[str, Any]:
    actual = _normalized_base_url(actual_base)
    expected = _normalized_base_url(expected_base)
    host = _host_from_api_base(actual)
    if not actual:
        return {
            "ok": False,
            "actual": actual,
            "expected": expected,
            "status": "not_configured",
            "message": "未配置平台调用地址。",
        }
    if host and host not in LOOPBACK_HOSTS:
        return {
            "ok": True,
            "actual": actual,
            "expected": expected,
            "status": "external_or_dev_endpoint",
            "message": "当前不是线上平台 loopback 隧道地址，跳过 18001 契约检查。",
        }
    ok = actual == expected
    return {
        "ok": ok,
        "actual": actual,
        "expected": expected,
        "status": "ok" if ok else "wrong_loopback_port",
        "message": "平台本地量化工具必须通过 127.0.0.1:18001 隧道访问。" if not ok else "",
    }


def _model_access_host_from_platform_runtime(platform_runtime: dict[str, Any]) -> str:
    rows = platform_runtime.get("ai_models")
    if isinstance(rows, list):
        for item in rows:
            if not isinstance(item, dict):
                continue
            host = _host_from_api_base(item.get("api_base"))
            if host and host not in {"127.0.0.1", "localhost", "::1"}:
                return host
    local_tools = platform_runtime.get("local_ai_tools")
    if isinstance(local_tools, dict):
        host = _host_from_api_base(local_tools.get("api_base"))
        if host and host not in {"127.0.0.1", "localhost", "::1"}:
            return host
    return ""


class ServerMonitorStatusService:
    """Collect and cache remote server/model health status."""

    def __init__(
        self,
        *,
        root_dir: Path = PROJECT_ROOT,
        model_id_provider: ModelIdProvider = primary_provider_model_id,
        info_loader: InfoLoader = load_model_server_info_for_monitor,
        ssh_connector: SshConnector = connect_remote_ssh,
        command_executor: CommandExecutor = exec_remote_command,
        clock: Clock = monotonic,
        cache_ttl_seconds: float = SERVER_MONITOR_CACHE_TTL_SECONDS,
        stale_cache_ttl_seconds: float = SERVER_MONITOR_STALE_CACHE_TTL_SECONDS,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.model_id_provider = model_id_provider
        self.info_loader = info_loader
        self.ssh_connector = ssh_connector
        self.command_executor = command_executor
        self.clock = clock
        self.cache_ttl_seconds = cache_ttl_seconds
        self.stale_cache_ttl_seconds = stale_cache_ttl_seconds
        self._cache: tuple[float, dict[str, Any]] | None = None
        self._cache_lock = Lock()
        self._refresh_lock = Lock()

    def collect_sync(self, info_override: Any | None = None) -> dict[str, Any]:
        """Collect remote server/model health once without using the process cache."""
        primary_model_id = self.model_id_provider()
        primary_model_label = display_provider_model_name(primary_model_id)
        checked_at = datetime.now(UTC).isoformat()
        info: Any | None = None
        try:
            info = info_override if info_override is not None else self.info_loader(self.root_dir)
            command = render_python_here_doc(
                render_server_monitor_probe(
                    primary_model_id,
                    primary_model_label,
                    str(settings.local_ai_tools_api_key or ""),
                )
            )
            ssh = self.ssh_connector(
                self.root_dir,
                timeout=8,
                banner_timeout=8,
                auth_timeout=8,
                info=info,
            )
            try:
                result = self.command_executor(
                    ssh,
                    command,
                    timeout=SERVER_MONITOR_REMOTE_COMMAND_TIMEOUT_SECONDS,
                )
            finally:
                ssh.close()
            out = result.stdout.strip()
            err = result.stderr.strip()
            if result.status != 0:
                return {
                    "available": False,
                    "remote_monitor_available": False,
                    "status": "remote_command_failed",
                    "message": safe_error_text(
                        err or out or "remote server monitor command failed"
                    ),
                    "checked_at": checked_at,
                }
            try:
                payload = json.loads(out or "{}")
            except json.JSONDecodeError:
                return {
                    "available": False,
                    "remote_monitor_available": False,
                    "status": "remote_payload_invalid",
                    "message": safe_error_text(
                        out or err or "remote monitor returned invalid JSON"
                    ),
                    "checked_at": checked_at,
                }
            if not isinstance(payload, dict):
                return {
                    "available": False,
                    "remote_monitor_available": False,
                    "status": "remote_payload_invalid",
                    "message": "remote monitor returned a non-object JSON payload",
                    "checked_at": checked_at,
                }
            payload.update(
                {
                    "available": True,
                    "status": "ok",
                    "host": info.host,
                    "model_access_host": getattr(info, "access_host", "") or info.host,
                    "checked_at": checked_at,
                }
            )
            return payload
        except ModelServerConfigNotConfigured as exc:
            return {
                "available": False,
                "remote_monitor_available": False,
                "status": "model_server_not_configured",
                "message": safe_error_text(exc),
                "checked_at": checked_at,
            }
        except ModelServerConfigError as exc:
            return {
                "available": False,
                "remote_monitor_available": False,
                "status": "model_server_config_error",
                "message": safe_error_text(exc),
                "checked_at": checked_at,
            }
        except ModuleNotFoundError as exc:
            return {
                "available": False,
                "remote_monitor_available": False,
                "status": "paramiko_unavailable",
                "message": safe_error_text(f"SSH dependency unavailable: {exc}"),
                "checked_at": checked_at,
            }
        except TimeoutError as exc:
            return {
                "available": False,
                "remote_monitor_available": False,
                "status": "remote_command_timeout",
                "message": safe_error_text(
                    exc,
                    fallback="remote server monitor command timed out",
                ),
                "checked_at": checked_at,
            }
        except Exception as exc:
            return _ssh_failure_payload(exc, checked_at=checked_at, info=info)

    def clear_cache(self) -> None:
        """Clear the short-lived server monitor cache."""
        with self._cache_lock:
            self._cache = None

    def get_status_sync(self, info_override: Any | None = None) -> dict[str, Any]:
        """Return server monitor status with a short single-process cache."""
        now = self.clock()
        cached = self._read_cache(now, max_age_seconds=self.cache_ttl_seconds)
        if cached is not None:
            payload, age = cached
            return self._copy_payload(payload, cache_status="fresh", age_seconds=age)

        if not self._refresh_lock.acquire(blocking=False):
            stale = self._read_cache(now, max_age_seconds=self.stale_cache_ttl_seconds)
            if stale is not None:
                payload, age = stale
                return self._copy_payload(
                    payload,
                    cache_status="stale_refreshing",
                    age_seconds=age,
                )
            return {
                "available": False,
                "status": "server_monitor_refreshing",
                "message": "server monitor refresh already in progress",
                "checked_at": datetime.now(UTC).isoformat(),
                "cache": {
                    "status": "initial_refreshing",
                    "age_seconds": None,
                    "ttl_seconds": self.cache_ttl_seconds,
                },
            }

        try:
            now = self.clock()
            cached = self._read_cache(now, max_age_seconds=self.cache_ttl_seconds)
            if cached is not None:
                payload, age = cached
                return self._copy_payload(payload, cache_status="fresh", age_seconds=age)
            stale_before_refresh = self._read_cache(
                self.clock(),
                max_age_seconds=self.stale_cache_ttl_seconds,
            )
            payload = self.collect_sync(info_override=info_override)
            if not payload.get("available") and stale_before_refresh is not None:
                stale_payload, stale_age = stale_before_refresh
                return self._copy_payload(
                    stale_payload,
                    cache_status="stale_refresh_failed",
                    age_seconds=stale_age,
                    refresh_error=self._refresh_error(payload),
                )
            cached_at = self.clock()
            with self._cache_lock:
                self._cache = (cached_at, copy.deepcopy(payload))
            return self._copy_payload(
                payload,
                cache_status="refreshed",
                age_seconds=0.0,
            )
        finally:
            self._refresh_lock.release()

    def _copy_payload(
        self,
        payload: dict[str, Any],
        *,
        cache_status: str,
        age_seconds: float,
        refresh_error: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cloned = copy.deepcopy(payload)
        cloned["cache"] = {
            "status": cache_status,
            "age_seconds": round(max(age_seconds, 0.0), 1),
            "ttl_seconds": self.cache_ttl_seconds,
        }
        if refresh_error is not None:
            cloned["refresh_error"] = copy.deepcopy(refresh_error)
        return cloned

    def _read_cache(
        self,
        now: float,
        *,
        max_age_seconds: float,
    ) -> tuple[dict[str, Any], float] | None:
        with self._cache_lock:
            if self._cache is None:
                return None
            cached_at, cached_payload = self._cache
            age = now - cached_at
            if age > max_age_seconds:
                return None
            return copy.deepcopy(cached_payload), age

    def _refresh_error(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": str(payload.get("status") or "unknown"),
            "message": safe_error_text(payload.get("message") or payload.get("status") or ""),
            "checked_at": payload.get("checked_at"),
        }


_default_service = ServerMonitorStatusService()


def collect_server_monitor_sync(root_dir: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Collect remote server/model health once using default dependencies."""
    if Path(root_dir) == _default_service.root_dir:
        return _default_service.collect_sync()
    return ServerMonitorStatusService(root_dir=Path(root_dir)).collect_sync()


def clear_server_monitor_cache() -> None:
    """Clear the default server monitor cache."""
    _default_service.clear_cache()


def get_server_monitor_status_sync() -> dict[str, Any]:
    """Return cached server monitor status using default dependencies."""
    return _default_service.get_status_sync()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value or "").strip())
    except Exception:
        return default


def _parse_phase3_gpu_row(row: Any) -> dict[str, Any] | None:
    """Parse readiness-report GPU rows into the dashboard GPU contract."""

    parts = [part.strip() for part in str(row or "").split(",")]
    if len(parts) < 6:
        return None
    index = parts[0]
    name = parts[1]
    memory_used_mb = _safe_float(parts[2])
    memory_total_mb = _safe_float(parts[3])
    if memory_total_mb <= 0:
        return None
    utilization_pct = _safe_float(parts[4])
    temperature_c = _safe_float(parts[5])
    return {
        "index": index,
        "name": name,
        "memory_used_mb": memory_used_mb,
        "memory_total_mb": memory_total_mb,
        "memory_used_pct": round(memory_used_mb / memory_total_mb * 100, 1),
        "utilization_pct": utilization_pct,
        "temperature_c": temperature_c,
        "power_w": 0.0,
        "source": "phase3_model_server_readiness",
    }


def phase3_model_server_gpu_status_from_latest_report() -> dict[str, Any]:
    """Return the latest audited model-server GPU summary for UI consistency."""

    report_path = settings.data_dir / "phase3_model_server_readiness_reports" / "latest.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {
            "available": False,
            "source": "phase3_model_server_readiness",
            "report_path": str(report_path),
            "error": "phase3 model-server readiness report is missing",
            "gpus": [],
        }
    except Exception as exc:
        return {
            "available": False,
            "source": "phase3_model_server_readiness",
            "report_path": str(report_path),
            "error": safe_error_text(exc, limit=180),
            "gpus": [],
        }

    rows = report.get("gpu_rows") if isinstance(report, dict) else []
    gpus = [
        parsed
        for parsed in (_parse_phase3_gpu_row(row) for row in rows if str(row or "").strip())
        if parsed is not None
    ]
    return {
        "available": bool(gpus),
        "source": "phase3_model_server_readiness",
        "report_path": str(report_path),
        "checked_at": report.get("checked_at") if isinstance(report, dict) else None,
        "readiness_status": report.get("status") if isinstance(report, dict) else None,
        "runtime_ready": bool(report.get("runtime_ready")) if isinstance(report, dict) else False,
        "gpu_count": int(report.get("gpu_count") or len(gpus)) if isinstance(report, dict) else len(gpus),
        "gpus": gpus,
        "error": "" if gpus else "phase3 model-server readiness report has no GPU rows",
    }


def _with_phase3_model_server_gpu_status(payload: dict[str, Any]) -> dict[str, Any]:
    payload["phase3_model_server_gpu"] = phase3_model_server_gpu_status_from_latest_report()
    return payload


def _safe_run_platform_command(args: list[str], *, timeout: float = 2.0) -> tuple[int, str, str]:
    try:
        result = subprocess.run(  # noqa: S603 - args come from fixed platform probe allowlists.
            args,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except Exception as exc:
        return 124, "", safe_error_text(exc, limit=180)


def _platform_cpu_snapshot() -> dict[str, Any]:
    load_1m = load_5m = load_15m = 0.0
    if hasattr(os, "getloadavg"):
        try:
            load_1m, load_5m, load_15m = os.getloadavg()
        except OSError:
            pass
    usage_pct = 0.0
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/stat", encoding="utf-8") as handle:
                first = [int(item) for item in handle.readline().split()[1:]]
            time.sleep(0.15)
            with open("/proc/stat", encoding="utf-8") as handle:
                second = [int(item) for item in handle.readline().split()[1:]]
            idle_delta = (second[3] + second[4]) - (first[3] + first[4])
            total_delta = sum(second) - sum(first)
            usage_pct = round((1 - idle_delta / max(total_delta, 1)) * 100, 1)
        except Exception as exc:
            _ = exc
            usage_pct = 0.0
    return {
        "usage_pct": usage_pct,
        "load_1m": round(float(load_1m), 2),
        "load_5m": round(float(load_5m), 2),
        "load_15m": round(float(load_15m), 2),
        "cores": os.cpu_count() or 0,
    }


def _platform_memory_snapshot() -> dict[str, Any]:
    if sys.platform.startswith("linux"):
        try:
            values: dict[str, int] = {}
            with open("/proc/meminfo", encoding="utf-8") as handle:
                for line in handle:
                    key, value = line.split(":", 1)
                    values[key] = int(value.strip().split()[0])
            total_mb = values.get("MemTotal", 0) / 1024
            available_mb = values.get("MemAvailable", 0) / 1024
            used_mb = max(total_mb - available_mb, 0)
            return {
                "total_mb": round(total_mb, 1),
                "used_mb": round(used_mb, 1),
                "available_mb": round(available_mb, 1),
                "used_pct": round((used_mb / total_mb * 100) if total_mb else 0.0, 1),
            }
        except Exception as exc:
            _ = exc
    return {"total_mb": 0.0, "used_mb": 0.0, "available_mb": 0.0, "used_pct": 0.0}


def _platform_disk_snapshot(path: Path) -> dict[str, Any]:
    target = path if path.exists() else Path.cwd()
    usage = shutil.disk_usage(target)
    total_gb = usage.total / 1024 / 1024 / 1024
    used_gb = usage.used / 1024 / 1024 / 1024
    free_gb = usage.free / 1024 / 1024 / 1024
    return {
        "path": str(target),
        "total_gb": round(total_gb, 1),
        "used_gb": round(used_gb, 1),
        "free_gb": round(free_gb, 1),
        "used_pct": round((used_gb / total_gb * 100) if total_gb else 0.0, 1),
    }


def _platform_uptime_seconds() -> float | None:
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/uptime", encoding="utf-8") as handle:
                return round(float(handle.readline().split()[0]), 1)
        except Exception:
            return None
    return None


def _platform_service_status(name: str) -> dict[str, Any]:
    if not sys.platform.startswith("linux"):
        return {
            "name": name,
            "active": False,
            "status": "local_dev_unavailable",
            "pid": "",
            "elapsed": "",
        }
    code, out, err = _safe_run_platform_command(["systemctl", "is-active", name])
    _pid_code, pid_out, _pid_err = _safe_run_platform_command(
        ["systemctl", "show", name, "-p", "MainPID", "--value"]
    )
    pid = pid_out.strip()
    elapsed = ""
    if pid and pid != "0":
        _elapsed_code, elapsed_out, _elapsed_err = _safe_run_platform_command(
            ["ps", "-p", pid, "-o", "etime="]
        )
        elapsed = elapsed_out.strip()
    return {
        "name": name,
        "active": out.strip() == "active",
        "status": out.strip() or err.strip() or ("unknown" if code else "active"),
        "pid": pid if pid and pid != "0" else "",
        "elapsed": elapsed,
    }


def collect_platform_server_status() -> dict[str, Any]:
    """Collect platform-host status without exposing secrets or account values."""
    checked_at = datetime.now(UTC).isoformat()
    try:
        root_disk = _platform_disk_snapshot(PROJECT_ROOT)
        services = [_platform_service_status(name) for name in PLATFORM_SERVICE_NAMES]
        return {
            "available": True,
            "status": "ok",
            "checked_at": checked_at,
            "hostname": socket.gethostname(),
            "platform": sys.platform,
            "python": sys.version.split()[0],
            "project_root": str(PROJECT_ROOT),
            "uptime_seconds": _platform_uptime_seconds(),
            "cpu": _platform_cpu_snapshot(),
            "memory": _platform_memory_snapshot(),
            "disks": [root_disk],
            "services": services,
        }
    except Exception as exc:
        return {
            "available": False,
            "status": "platform_probe_failed",
            "checked_at": checked_at,
            "message": safe_error_text(exc, limit=240),
        }


async def _probe_platform_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    api_key: str = "",
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    started = monotonic()
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        request_kwargs: dict[str, Any] = {"headers": headers, "json": payload}
        if timeout_seconds is not None:
            request_kwargs["timeout"] = timeout_seconds
        try:
            response = await client.request(method, url, **request_kwargs)
        except TypeError:
            # Some tests use minimal fake clients without the httpx timeout kwarg.
            request_kwargs.pop("timeout", None)
            response = await client.request(method, url, **request_kwargs)
        latency_ms = round((monotonic() - started) * 1000, 1)
        data: Any = None
        try:
            data = response.json()
        except ValueError:
            data = None
        return {
            "ok": 200 <= response.status_code < 300,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
            "status_category": _http_probe_category(response.status_code),
            "error": _http_probe_error(response.status_code, data),
            "data": data if isinstance(data, dict) else None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "status_code": 0,
            "latency_ms": round((monotonic() - started) * 1000, 1),
            "status_category": "network_error",
            "error": safe_error_text(exc, limit=180),
            "data": None,
        }


def _model_ids_from_models_response(payload: dict[str, Any] | None) -> list[str]:
    data = payload.get("data") if isinstance(payload, dict) else None
    rows = data if isinstance(data, list) else []
    model_ids: list[str] = []
    for item in rows[:24]:
        if not isinstance(item, dict):
            continue
        value = str(item.get("id") or item.get("root") or "").strip()
        if value:
            model_ids.append(value)
    return model_ids


def _is_loopback_api_base(value: Any) -> bool:
    host = _host_from_api_base(value)
    return bool(not host or host in LOOPBACK_HOSTS)


def _decision_runtime_row(ai_rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in ai_rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("name") or "").strip() == "decision_maker":
            return row
    for row in ai_rows:
        if not isinstance(row, dict):
            continue
        model = str(row.get("model") or "").strip().lower()
        if model and ("qwen" in model or "deepseek" in model):
            return row
    return {}


def _platform_model_tunnel_summary(
    ai_rows: list[dict[str, Any]],
    local_tools: dict[str, Any],
) -> dict[str, Any]:
    rows_by_model = {
        str(row.get("model") or "").strip(): row
        for row in ai_rows
        if isinstance(row, dict) and str(row.get("model") or "").strip()
    }
    decision_row = _decision_runtime_row(ai_rows)
    decision_model = str(decision_row.get("model") or "").strip()
    decision_api_base = str(decision_row.get("api_base") or "").strip()
    decision_available = bool(decision_row.get("available"))
    decision_endpoint_ok = bool(decision_row.get("endpoint_ok"))
    decision_uses_external_route = bool(
        decision_api_base and not _is_loopback_api_base(decision_api_base)
    )
    tunnels: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    for spec in ONLINE_PHASE3_TUNNEL_CONTRACTS:
        model = str(spec.get("model") or "").strip()
        is_decision_tunnel = str(spec.get("role") or "") == "decision_maker"
        required = True
        if is_decision_tunnel and decision_uses_external_route:
            required = False
        if model:
            probe = rows_by_model.get(model, {})
            available = bool(probe.get("available"))
            endpoint_ok = bool(probe.get("endpoint_ok"))
            status = (
                "ok"
                if available
                else "model_mismatch"
                if endpoint_ok and not bool(probe.get("model_available"))
                else str(probe.get("status_category") or "unavailable")
            )
            row = {
                "name": spec["name"],
                "role": spec["role"],
                "capability": spec["capability"],
                "local_port": spec["local_port"],
                "required": required,
                "expected_api_base": spec["api_base"],
                "api_base": probe.get("api_base") or spec["api_base"],
                "model": model,
                "available": available,
                "endpoint_ok": endpoint_ok,
                "model_available": bool(probe.get("model_available")),
                "status": status,
                "status_code": probe.get("status_code"),
                "latency_ms": probe.get("latency_ms"),
                "error": str(probe.get("error") or ""),
            }
            if is_decision_tunnel and decision_uses_external_route:
                row["status"] = "standby" if endpoint_ok else "standby_unavailable"
                row["active_decision_model"] = decision_model
                row["active_decision_api_base"] = decision_api_base
        else:
            health = local_tools.get("health") if isinstance(local_tools.get("health"), dict) else {}
            status_probe = (
                local_tools.get("status") if isinstance(local_tools.get("status"), dict) else {}
            )
            tunnel_contract = (
                local_tools.get("tunnel_contract")
                if isinstance(local_tools.get("tunnel_contract"), dict)
                else {}
            )
            available = bool(local_tools.get("available"))
            endpoint_ok = bool(health.get("ok") or status_probe.get("ok"))
            raw_status = str(
                tunnel_contract.get("status")
                or health.get("status_category")
                or status_probe.get("status_category")
                or "unavailable"
            )
            status = "ok" if available else ("unavailable" if raw_status == "ok" else raw_status)
            row = {
                "name": spec["name"],
                "role": spec["role"],
                "capability": spec["capability"],
                "local_port": spec["local_port"],
                "required": required,
                "expected_api_base": spec["api_base"],
                "api_base": local_tools.get("api_base") or spec["api_base"],
                "model": "",
                "available": available,
                "endpoint_ok": endpoint_ok,
                "model_available": bool(local_tools.get("model_bundle_available")),
                "status": status,
                "status_code": health.get("status_code") or status_probe.get("status_code"),
                "latency_ms": health.get("latency_ms") or status_probe.get("latency_ms"),
                "error": str(
                    local_tools.get("config_issue")
                    or health.get("error")
                    or status_probe.get("error")
                    or ""
                ),
            }
        tunnels.append(row)
        if row["required"] and not row["available"]:
            unavailable.append(row)

    by_name = {str(row.get("name") or ""): row for row in tunnels}
    local_decision_available = bool(by_name.get("qwen3-32b-trade", {}).get("available"))
    can_call_decision = bool(decision_available or local_decision_available)
    can_call_expert = bool(by_name.get("BB-FinQuant-Expert-14B", {}).get("available"))
    can_call_quant_tool = bool(by_name.get("phase3-quant-api", {}).get("available"))
    blocker_codes = [
        f"tunnel_port_{int(row['local_port'])}_{str(row['status'] or 'unavailable')}"
        for row in unavailable
    ]
    if not can_call_decision:
        blocker_codes.append(
            "decision_route_unavailable"
            if decision_model
            else "decision_route_not_configured"
        )
    unavailable_all = [row for row in tunnels if not row["available"]]
    return {
        "expected_ports": [int(spec["local_port"]) for spec in ONLINE_PHASE3_TUNNEL_CONTRACTS],
        "ready": not unavailable and can_call_decision,
        "available_count": sum(1 for row in tunnels if row["available"]),
        "unavailable_count": len(unavailable_all),
        "blocking_unavailable_count": len(unavailable),
        "tunnels": tunnels,
        "unavailable_ports": [int(row["local_port"]) for row in unavailable],
        "blocking_unavailable_ports": [int(row["local_port"]) for row in unavailable],
        "blocker_codes": blocker_codes,
        "can_call_decision_maker": can_call_decision,
        "decision_route": {
            "model": decision_model,
            "api_base": decision_api_base,
            "available": decision_available,
            "endpoint_ok": decision_endpoint_ok,
            "external": decision_uses_external_route,
            "fallback_local_available": local_decision_available,
        },
        "can_call_expert": can_call_expert,
        "can_call_quant_tool": can_call_quant_tool,
        "can_create_strategy": bool(can_call_decision and can_call_expert and can_call_quant_tool),
    }


def _platform_runtime_to_model_runtime(platform_runtime: dict[str, Any]) -> dict[str, Any]:
    """Build a UI-compatible model runtime view from platform endpoint probes."""

    rows = platform_runtime.get("ai_models") if isinstance(platform_runtime, dict) else []
    if not isinstance(rows, list):
        rows = []
    endpoint_rows: list[dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        model = str(item.get("model") or "").strip()
        if not model:
            continue
        api_base = str(item.get("api_base") or "").strip().rstrip("/")
        endpoint = api_base
        if endpoint.endswith("/v1"):
            endpoint = endpoint[:-3]
        endpoint_rows.append(
            {
                "name": str(item.get("name") or ""),
                "label": str(item.get("label") or item.get("name") or model),
                "provider_model": model,
                "api_base": api_base,
                "endpoint": endpoint,
                "available": bool(item.get("available")),
                "endpoint_available": bool(item.get("endpoint_ok")),
                "model_available": bool(item.get("model_available")),
                "model_mismatch": bool(item.get("endpoint_ok") and not item.get("model_available")),
                "status": "ok"
                if item.get("available")
                else (
                    "model_mismatch"
                    if item.get("endpoint_ok") and not item.get("model_available")
                    else str(item.get("status_category") or "unavailable")
                ),
                "status_code": item.get("status_code"),
                "latency_ms": item.get("latency_ms"),
                "models": list(item.get("models") or []),
                "error": str(item.get("error") or ""),
                "source": "platform_runtime_probe",
                "health": {
                    "ok": bool(item.get("endpoint_ok")),
                    "status_code": item.get("status_code"),
                    "latency_ms": item.get("latency_ms"),
                    "status_category": item.get("status_category"),
                    "error": str(item.get("error") or ""),
                },
            }
        )
    local_tools = (
        platform_runtime.get("local_ai_tools")
        if isinstance(platform_runtime.get("local_ai_tools"), dict)
        else {}
    )
    primary = next(
        (row for row in endpoint_rows if row.get("name") == "decision_maker"),
        next(
            (row for row in endpoint_rows if row.get("provider_model") == "qwen3-32b-trade"),
            endpoint_rows[0] if endpoint_rows else {},
        ),
    )
    return {
        "vllm": dict(primary),
        "vllm_endpoints": endpoint_rows,
        "local_ai_tools": {
            "available": bool(local_tools.get("available")),
            "endpoint": str(local_tools.get("api_base") or "").replace("http://", "").replace(
                "https://", ""
            ),
            "service_role": local_tools.get("service_role") or "phase3_quant_api",
            "model_bundle_available": bool(local_tools.get("model_bundle_available")),
            "trained_models_available": bool(local_tools.get("trained_models_available")),
            "trained_at": local_tools.get("trained_at") or "",
            "training_mode": local_tools.get("training_mode"),
            "model_stage": local_tools.get("model_stage"),
            "models": local_tools.get("models") if isinstance(local_tools.get("models"), dict) else {},
            "shadow_sample_count": int(local_tools.get("shadow_sample_count") or 0),
            "trade_sample_count": int(local_tools.get("trade_sample_count") or 0),
            "completed_shadow_sample_count": int(
                local_tools.get("completed_shadow_sample_count") or 0
            ),
            "completed_trade_sample_count": int(local_tools.get("completed_trade_sample_count") or 0),
            "health": local_tools.get("health") if isinstance(local_tools.get("health"), dict) else {},
            "status_health": local_tools.get("status")
            if isinstance(local_tools.get("status"), dict)
            else {},
            "child_endpoints": local_tools.get("child_endpoints")
            if isinstance(local_tools.get("child_endpoints"), dict)
            else {},
            "source": "platform_runtime_probe",
        },
    }


def _platform_runtime_available(platform_runtime: dict[str, Any]) -> bool:
    if not isinstance(platform_runtime, dict):
        return False
    models = platform_runtime.get("ai_models") if isinstance(platform_runtime.get("ai_models"), list) else []
    has_model = any(isinstance(row, dict) and row.get("available") for row in models)
    local_tools = (
        platform_runtime.get("local_ai_tools")
        if isinstance(platform_runtime.get("local_ai_tools"), dict)
        else {}
    )
    return bool(has_model or local_tools.get("available") or local_tools.get("child_available"))


def _remote_monitor_unavailable_payload(
    *,
    status: str,
    message: str,
    checked_at: str,
    platform_server: dict[str, Any],
    platform_runtime: dict[str, Any],
) -> dict[str, Any]:
    platform_available = _platform_runtime_available(platform_runtime)
    return _with_phase3_model_server_gpu_status(
        {
            "available": platform_available,
            "status": "platform_runtime_ok_remote_monitor_unavailable"
            if platform_available
            else status,
            "remote_monitor_status": status,
            "message": message,
            "checked_at": checked_at,
            "platform_server": platform_server,
            "platform_runtime": platform_runtime,
            "model_runtime": _platform_runtime_to_model_runtime(platform_runtime),
            "remote_monitor_available": False,
            "monitor_source": "platform_runtime_probe",
        }
    )


def _local_ai_tools_probe_payload() -> dict[str, Any]:
    return {
        "symbol": "BTC/USDT",
        "features": {
            "symbol": "BTC/USDT",
            "current_price": 100.0,
            "close": 100.0,
            "returns_1": 0.02,
            "returns_5": 0.04,
            "returns_20": 0.08,
            "volume_ratio": 1.0,
            "volatility_20": 0.01,
            "spread_pct": 0.01,
            "adx_14": 20.0,
            "news_sentiment_avg": 0.0,
            "social_sentiment_avg": 0.0,
        },
        "open_positions": [
            {
                "symbol": "BTC/USDT",
                "side": "long",
                "entry_price": 100.0,
                "current_price": 100.2,
                "unrealized_pnl": 0.2,
                "unrealized_pnl_pct": 0.002,
            }
        ],
    }


def _local_ai_tools_api_key_for_platform_probe() -> str:
    return str(
        settings.local_ai_tools_api_key or os.environ.get("LOCAL_AI_TOOLS_API_KEY") or ""
    ).strip()


async def _probe_local_ai_tools_child_endpoints(
    client: httpx.AsyncClient,
    local_base: str,
    api_key: str,
) -> dict[str, dict[str, Any]]:
    specs = {
        "profit_prediction": "/profit/predict",
        "time_series_prediction": "/timeseries/deep/predict",
        "sentiment_analysis": "/sentiment/deep/analyze",
        "exit_advice": "/exit/advise",
    }
    payload = _local_ai_tools_probe_payload()

    async def probe(name: str, path: str) -> tuple[str, dict[str, Any]]:
        timeout_seconds = LOCAL_AI_TOOLS_CHILD_ENDPOINT_TIMEOUT_SECONDS.get(
            name,
            PLATFORM_RUNTIME_PROBE_TIMEOUT_SECONDS,
        )
        result = await _probe_platform_json(
            client,
            f"{local_base}{path}",
            api_key=api_key,
            method="POST",
            payload=payload,
            timeout_seconds=timeout_seconds,
        )
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        payload_available = bool(data.get("available", True)) if isinstance(data, dict) else True
        return name, {
            "available": bool(result.get("ok") and payload_available),
            "ok": bool(result.get("ok")),
            "path": path,
            "status_code": result.get("status_code"),
            "latency_ms": result.get("latency_ms"),
            "timeout_seconds": timeout_seconds,
            "status_category": result.get("status_category"),
            "error": result.get("error", ""),
        }

    results = await asyncio.gather(
        *(probe(name, path) for name, path in specs.items()),
        return_exceptions=True,
    )
    child_endpoints: dict[str, dict[str, Any]] = {}
    for result in results:
        if isinstance(result, Exception):
            continue
        name, item = result
        child_endpoints[name] = item
    return child_endpoints


async def collect_platform_runtime_status() -> dict[str, Any]:
    """Probe the endpoints the platform actually calls, without returning secrets."""
    ai_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    async with httpx.AsyncClient(timeout=PLATFORM_RUNTIME_PROBE_TIMEOUT_SECONDS) as client:
        fixed_model_rows = [
            cfg for cfg in settings.get_fixed_ai_models(include_empty=False) if isinstance(cfg, dict)
        ]
        default_models_enabled = not any(
            str(cfg.get("api_base") or "").strip() and str(cfg.get("model") or "").strip()
            for cfg in fixed_model_rows
        )
        model_configs = list(fixed_model_rows)
        if default_models_enabled:
            model_configs.extend(dict(item, default_phase3_tunnel=True) for item in ONLINE_PHASE3_DEFAULT_AI_MODELS)

        for cfg in model_configs:
            if not isinstance(cfg, dict) or cfg.get("enabled") is False:
                continue
            api_base = str(cfg.get("api_base") or "").strip().rstrip("/")
            model = str(cfg.get("model") or "").strip()
            if not api_base or not model:
                continue
            key = (api_base, model)
            if key in seen:
                continue
            seen.add(key)
            result = await _probe_platform_json(
                client,
                f"{api_base}/models",
                api_key=str(cfg.get("api_key") or settings.ai_api_key or ""),
            )
            models = _model_ids_from_models_response(result.get("data"))
            lowered = model.lower()
            model_available = bool(
                models
                and any(
                    lowered == item.lower() or lowered in item.lower() or item.lower() in lowered
                    for item in models
                )
            )
            ai_rows.append(
                {
                    "name": str(cfg.get("name") or ""),
                    "label": str(cfg.get("label") or cfg.get("name") or ""),
                    "api_base": api_base,
                    "model": model,
                    "available": bool(result.get("ok") and model_available),
                    "endpoint_ok": bool(result.get("ok")),
                    "model_available": model_available,
                    "status_code": result.get("status_code"),
                    "latency_ms": result.get("latency_ms"),
                    "models": models,
                    "error": result.get("error", ""),
                    "default_phase3_tunnel": bool(cfg.get("default_phase3_tunnel")),
                }
            )

        configured_local_base = str(settings.local_ai_tools_api_base or "").strip().rstrip("/")
        local_base = configured_local_base or ONLINE_PHASE3_QUANT_API_PLATFORM_BASE
        local_tools: dict[str, Any] = {
            "configured": bool(configured_local_base),
            "using_default_phase3_tunnel": not bool(configured_local_base),
        }
        if local_base:
            tunnel_contract = _platform_tunnel_contract(
                local_base,
                ONLINE_PHASE3_QUANT_API_PLATFORM_BASE,
            )
            api_key = _local_ai_tools_api_key_for_platform_probe()
            health = await _probe_platform_json(
                client,
                f"{local_base}/health",
                api_key=api_key,
            )
            status_probe = await _probe_platform_json(
                client,
                f"{local_base}/models/status",
                api_key=api_key,
            )
            metadata = health.get("data") if isinstance(health.get("data"), dict) else {}
            status_metadata = (
                status_probe.get("data") if isinstance(status_probe.get("data"), dict) else {}
            )
            is_phase3_quant_api = metadata.get("service") == "phase3_quant_api"
            service_available = bool(
                tunnel_contract.get("ok") and health.get("ok") and is_phase3_quant_api
            )
            model_bundle_available = bool(
                status_metadata.get("available") or metadata.get("trained_models_available")
            )
            child_endpoints = await _probe_local_ai_tools_child_endpoints(
                client,
                local_base,
                api_key,
            )
            child_available = any(
                bool(item.get("available") or item.get("ok"))
                for item in child_endpoints.values()
                if isinstance(item, dict)
            )
            platform_call_available = bool(
                tunnel_contract.get("ok") and (service_available or child_available)
            )
            local_tools.update(
                {
                    "api_base": local_base,
                    "configured_api_base": configured_local_base,
                    "expected_platform_api_base": ONLINE_PHASE3_QUANT_API_PLATFORM_BASE,
                    "service_role": "phase3_quant_api",
                    "legacy_local_ai_tools": False,
                    "tunnel_contract": tunnel_contract,
                    "available": platform_call_available,
                    "service_available": service_available,
                    "child_available": child_available,
                    "config_issue": (
                        "" if tunnel_contract.get("ok") else tunnel_contract.get("message", "")
                    ),
                    "health": {
                        "ok": bool(health.get("ok")),
                        "status_code": health.get("status_code"),
                        "latency_ms": health.get("latency_ms"),
                        "status_category": health.get("status_category"),
                        "error": health.get("error", ""),
                        "service": metadata.get("service"),
                        "root": metadata.get("root"),
                        "validation_all_ok": metadata.get("validation_all_ok"),
                        "downloaded_model_count": metadata.get("downloaded_model_count"),
                        "validated_model_count": metadata.get("validated_model_count"),
                    },
                    "status": {
                        "ok": bool(status_probe.get("ok")),
                        "status_code": status_probe.get("status_code"),
                        "latency_ms": status_probe.get("latency_ms"),
                        "status_category": status_probe.get("status_category"),
                        "error": status_probe.get("error", ""),
                    },
                    "model_bundle_available": model_bundle_available,
                    "trained_models_available": model_bundle_available,
                    "trained_at": status_metadata.get("trained_at")
                    or metadata.get("trained_at")
                    or "",
                    "training_mode": status_metadata.get("training_mode")
                    or metadata.get("training_mode"),
                    "model_stage": status_metadata.get("model_stage") or metadata.get("model_stage"),
                    "models": (
                        status_metadata.get("models")
                        if isinstance(status_metadata.get("models"), dict)
                        else metadata.get("model_status")
                        if isinstance(metadata, dict)
                        else {}
                    ),
                    "shadow_sample_count": int(
                        status_metadata.get("shadow_sample_count")
                        or metadata.get("shadow_sample_count")
                        or 0
                    ),
                    "trade_sample_count": int(
                        status_metadata.get("trade_sample_count")
                        or metadata.get("trade_sample_count")
                        or 0
                    ),
                    "sequence_sample_count": int(
                        status_metadata.get("sequence_sample_count")
                        or metadata.get("sequence_sample_count")
                        or 0
                    ),
                    "text_sentiment_sample_count": int(
                        status_metadata.get("text_sentiment_sample_count")
                        or metadata.get("text_sentiment_sample_count")
                        or 0
                    ),
                    "completed_shadow_sample_count": int(
                        status_metadata.get("completed_shadow_sample_count")
                        or metadata.get("completed_shadow_sample_count")
                        or 0
                    ),
                    "completed_trade_sample_count": int(
                        status_metadata.get("completed_trade_sample_count")
                        or metadata.get("completed_trade_sample_count")
                        or 0
                    ),
                    "child_endpoints": child_endpoints,
                }
            )
        # 加入 high_risk_review 独立模型（如 deepseek-r1-14b-risk）
        hr_base = str(getattr(settings, "high_risk_review_api_base", "") or "").strip().rstrip("/")
        hr_model = str(getattr(settings, "high_risk_review_model", "") or "").strip()
        if hr_base and hr_model:
            hr_key = (hr_base, hr_model)
            if hr_key not in seen:
                seen.add(hr_key)
                result = await _probe_platform_json(
                    client,
                    f"{hr_base}/models",
                    api_key=str(
                        getattr(settings, "high_risk_review_api_key", "")
                        or getattr(settings, "ai_api_key", "")
                        or ""
                    ),
                )
                models = _model_ids_from_models_response(result.get("data"))
                lowered = hr_model.lower()
                model_available = bool(
                    models
                    and any(
                        lowered == item.lower()
                        or lowered in item.lower()
                        or item.lower() in lowered
                        for item in models
                    )
                )
                ai_rows.append(
                    {
                        "name": "high_risk_review",
                        "label": f"高风险复核 ({hr_model})",
                        "api_base": hr_base,
                        "model": hr_model,
                        "available": bool(result.get("ok") and model_available),
                        "endpoint_ok": bool(result.get("ok")),
                        "model_available": model_available,
                        "status_code": result.get("status_code"),
                        "latency_ms": result.get("latency_ms"),
                        "models": models,
                        "error": result.get("error", ""),
                    }
                )
    return {
        "ai_models": ai_rows,
        "local_ai_tools": local_tools,
        "model_tunnels": _platform_model_tunnel_summary(ai_rows, local_tools),
        "checked_at": datetime.now(UTC).isoformat(),
    }


async def get_server_monitor_status_async() -> dict[str, Any]:
    """Return cached server monitor status using async DB loading on the main loop."""
    checked_at = datetime.now(UTC).isoformat()
    platform_server = await asyncio.to_thread(collect_platform_server_status)
    platform_runtime = await collect_platform_runtime_status()
    try:
        info = await load_model_server_info_from_secure_settings()
    except ModelServerConfigNotConfigured as exc:
        return _remote_monitor_unavailable_payload(
            status="model_server_not_configured",
            message=safe_error_text(exc),
            checked_at=checked_at,
            platform_server=platform_server,
            platform_runtime=platform_runtime,
        )
    except ModelServerConfigError as exc:
        return _remote_monitor_unavailable_payload(
            status="model_server_config_error",
            message=safe_error_text(exc),
            checked_at=checked_at,
            platform_server=platform_server,
            platform_runtime=platform_runtime,
        )
    payload = await asyncio.to_thread(_default_service.get_status_sync, info)
    payload["platform_runtime"] = platform_runtime
    payload["platform_server"] = platform_server
    payload["remote_monitor_available"] = bool(payload.get("available"))
    platform_access_host = _model_access_host_from_platform_runtime(platform_runtime)
    if platform_access_host:
        payload["model_access_host"] = platform_access_host
    return _with_phase3_model_server_gpu_status(payload)
