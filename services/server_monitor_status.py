"""Remote server and model monitor status service."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Any

from config.settings import settings
from core.remote_server_info import load_remote_server_info
from core.remote_ssh import connect_remote_ssh, exec_remote_command
from core.safe_output import safe_error_text
from core.server_monitor_probe import (
    SERVER_MONITOR_REMOTE_COMMAND_TIMEOUT_SECONDS,
    display_provider_model_name,
    render_python_here_doc,
    render_server_monitor_probe,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVER_MONITOR_CACHE_TTL_SECONDS = 10.0
SERVER_MONITOR_STALE_CACHE_TTL_SECONDS = 60.0

InfoLoader = Callable[[Path], Any]
SshConnector = Callable[..., Any]
CommandExecutor = Callable[..., Any]
Clock = Callable[[], float]
ModelIdProvider = Callable[[], str]


def primary_provider_model_id() -> str:
    """Return the configured primary provider model id for monitor probes."""
    for cfg in settings.get_fixed_ai_models(include_empty=True):
        if not isinstance(cfg, dict) or not cfg.get("enabled", True):
            continue
        model = str(cfg.get("model") or "").strip()
        if model:
            return model
    return str(settings.ai_model or "").strip()


class ServerMonitorStatusService:
    """Collect and cache remote server/model health status."""

    def __init__(
        self,
        *,
        root_dir: Path = PROJECT_ROOT,
        model_id_provider: ModelIdProvider = primary_provider_model_id,
        info_loader: InfoLoader = load_remote_server_info,
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

    def collect_sync(self) -> dict[str, Any]:
        """Collect remote server/model health once without using the process cache."""
        primary_model_id = self.model_id_provider()
        primary_model_label = display_provider_model_name(primary_model_id)
        checked_at = datetime.now(UTC).isoformat()
        try:
            info = self.info_loader(self.root_dir)
            command = render_python_here_doc(
                render_server_monitor_probe(primary_model_id, primary_model_label)
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
                    "status": "remote_payload_invalid",
                    "message": safe_error_text(
                        out or err or "remote monitor returned invalid JSON"
                    ),
                    "checked_at": checked_at,
                }
            if not isinstance(payload, dict):
                return {
                    "available": False,
                    "status": "remote_payload_invalid",
                    "message": "remote monitor returned a non-object JSON payload",
                    "checked_at": checked_at,
                }
            payload.update(
                {
                    "available": True,
                    "status": "ok",
                    "host": info.host,
                    "checked_at": checked_at,
                }
            )
            return payload
        except ModuleNotFoundError as exc:
            return {
                "available": False,
                "status": "paramiko_unavailable",
                "message": safe_error_text(f"SSH dependency unavailable: {exc}"),
                "checked_at": checked_at,
            }
        except TimeoutError as exc:
            return {
                "available": False,
                "status": "remote_command_timeout",
                "message": safe_error_text(
                    exc,
                    fallback="remote server monitor command timed out",
                ),
                "checked_at": checked_at,
            }
        except Exception as exc:
            return {
                "available": False,
                "status": "ssh_failed",
                "message": safe_error_text(exc),
                "checked_at": checked_at,
            }

    def clear_cache(self) -> None:
        """Clear the short-lived server monitor cache."""
        with self._cache_lock:
            self._cache = None

    def get_status_sync(self) -> dict[str, Any]:
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
            payload = self.collect_sync()
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
