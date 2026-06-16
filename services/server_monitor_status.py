"""Remote server and model monitor status service."""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Any
from urllib.parse import urlparse

import httpx

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


def _host_from_api_base(value: Any) -> str:
    try:
        host = str(urlparse(str(value or "")).hostname or "").strip()
    except Exception:
        return ""
    return host


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
                    "model_access_host": getattr(info, "access_host", "") or info.host,
                    "checked_at": checked_at,
                }
            )
            return payload
        except ModelServerConfigNotConfigured as exc:
            return {
                "available": False,
                "status": "model_server_not_configured",
                "message": safe_error_text(exc),
                "checked_at": checked_at,
            }
        except ModelServerConfigError as exc:
            return {
                "available": False,
                "status": "model_server_config_error",
                "message": safe_error_text(exc),
                "checked_at": checked_at,
            }
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


async def _probe_platform_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    api_key: str = "",
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = monotonic()
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        response = await client.request(method, url, headers=headers, json=payload)
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
            "data": data if isinstance(data, dict) else None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "status_code": 0,
            "latency_ms": round((monotonic() - started) * 1000, 1),
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
        result = await _probe_platform_json(
            client,
            f"{local_base}{path}",
            api_key=api_key,
            method="POST",
            payload=payload,
        )
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        payload_available = bool(data.get("available", True)) if isinstance(data, dict) else True
        return name, {
            "available": bool(result.get("ok") and payload_available),
            "ok": bool(result.get("ok")),
            "path": path,
            "status_code": result.get("status_code"),
            "latency_ms": result.get("latency_ms"),
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
    async with httpx.AsyncClient(timeout=3.5) as client:
        for cfg in settings.get_fixed_ai_models(include_empty=False):
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
                }
            )

        local_tools: dict[str, Any] = {"configured": bool(settings.local_ai_tools_api_base)}
        local_base = str(settings.local_ai_tools_api_base or "").strip().rstrip("/")
        if local_base:
            health = await _probe_platform_json(
                client,
                f"{local_base}/health",
                api_key=str(settings.local_ai_tools_api_key or ""),
            )
            status = await _probe_platform_json(
                client,
                f"{local_base}/models/status",
                api_key=str(settings.local_ai_tools_api_key or ""),
            )
            child_endpoints = await _probe_local_ai_tools_child_endpoints(
                client,
                local_base,
                str(settings.local_ai_tools_api_key or ""),
            )
            child_available = any(
                bool(item.get("available") or item.get("ok"))
                for item in child_endpoints.values()
                if isinstance(item, dict)
            )
            metadata = status.get("data") if isinstance(status.get("data"), dict) else {}
            local_tools.update(
                {
                    "api_base": local_base,
                    "available": bool(health.get("ok") or status.get("ok") or child_available),
                    "health": {
                        "ok": bool(health.get("ok")),
                        "status_code": health.get("status_code"),
                        "latency_ms": health.get("latency_ms"),
                        "error": health.get("error", ""),
                    },
                    "status": {
                        "ok": bool(status.get("ok")),
                        "status_code": status.get("status_code"),
                        "latency_ms": status.get("latency_ms"),
                        "error": status.get("error", ""),
                    },
                    "model_bundle_available": bool(metadata.get("available")),
                    "trained_at": metadata.get("trained_at"),
                    "models": metadata.get("models") if isinstance(metadata, dict) else {},
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
    return {"ai_models": ai_rows, "local_ai_tools": local_tools}


async def get_server_monitor_status_async() -> dict[str, Any]:
    """Return cached server monitor status using async DB loading on the main loop."""
    checked_at = datetime.now(UTC).isoformat()
    try:
        info = await load_model_server_info_from_secure_settings()
    except ModelServerConfigNotConfigured as exc:
        return {
            "available": False,
            "status": "model_server_not_configured",
            "message": safe_error_text(exc),
            "checked_at": checked_at,
        }
    except ModelServerConfigError as exc:
        return {
            "available": False,
            "status": "model_server_config_error",
            "message": safe_error_text(exc),
            "checked_at": checked_at,
        }
    payload = await asyncio.to_thread(_default_service.get_status_sync, info)
    platform_runtime = await collect_platform_runtime_status()
    payload["platform_runtime"] = platform_runtime
    platform_access_host = _model_access_host_from_platform_runtime(platform_runtime)
    if platform_access_host:
        payload["model_access_host"] = platform_access_host
    return payload
