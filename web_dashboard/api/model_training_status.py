"""Dashboard model-training registry and scheduler status endpoints."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter

from config.settings import settings
from core.safe_output import safe_error_text
from core.trading_mode import mode_manager
from services.cloud_reviewer_verification import load_cloud_reviewer_verification
from services.entry_high_risk_review import validate_cloud_reviewer_route
from services.model_contribution_performance import ModelContributionPerformanceService
from services.model_training_registry import build_model_training_registry
from services.model_training_state import ModelTrainingStateStore
from web_dashboard.api.text_sanitize import sanitize_payload

router = APIRouter()
logger = structlog.get_logger(__name__)

MODEL_TRAINING_STATE_STORE = ModelTrainingStateStore(
    Path(settings.data_dir) / "model_training_scheduler_state.json"
)
_REGISTRY_CACHE_TTL_SECONDS = 300.0
_CONTRIBUTION_TIMEOUT_SECONDS = 4.0
_FAST_LOCAL_STATUS_TIMEOUT_SECONDS = 9.0
_FAST_OBSERVABILITY_TIMEOUT_SECONDS = 1.5
_REGISTRY_SHUTDOWN_GRACE_SECONDS = 5.0
_REGISTRY_SNAPSHOT_PATH = Path(settings.data_dir) / "model_training_registry_snapshot.json"
_registry_cache: tuple[float, dict[str, Any]] | None = None
_registry_refresh_task: asyncio.Task[Any] | None = None
_registry_refresh_error: str | None = None
_registry_last_success_at: str | None = None


def load_model_training_report(relative_path: str) -> dict[str, Any]:
    try:
        payload = json.loads((settings.data_dir / relative_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _model_server_report_with_runtime_configuration() -> dict[str, Any]:
    report = load_model_training_report(
        "phase3_model_server_readiness_reports/latest.json"
    )
    cloud = report.get("cloud_reviewer")
    cloud = dict(cloud) if isinstance(cloud, dict) else {}
    valid, reason = validate_cloud_reviewer_route(
        str(settings.high_risk_review_api_base or ""),
        str(settings.high_risk_review_model or ""),
        str(getattr(settings, "high_risk_review_model_revision", "") or ""),
        str(settings.high_risk_review_api_key or ""),
    )
    verification = load_cloud_reviewer_verification(
        api_base=str(settings.high_risk_review_api_base or ""),
        model=str(settings.high_risk_review_model or ""),
        revision=str(getattr(settings, "high_risk_review_model_revision", "") or ""),
        api_key=str(settings.high_risk_review_api_key or ""),
    )
    connection_verified = bool(valid and verification.get("connection_verified"))
    runtime_available = bool(
        connection_verified and settings.high_risk_review_enabled
    )
    cloud.update(
        {
            "configured": valid,
            "enabled": bool(settings.high_risk_review_enabled),
            "model": str(settings.high_risk_review_model or "") or None,
            "revision": str(
                getattr(settings, "high_risk_review_model_revision", "") or ""
            )
            or None,
            "route_error": reason or None,
            "connection_verified": connection_verified,
            "identity_verified": connection_verified,
            "runtime_available": runtime_available,
            "verified_at": verification.get("verified_at"),
            "last_test_latency_ms": verification.get("latency_ms"),
            "identity_source": verification.get("identity_source"),
        }
    )
    report["cloud_reviewer"] = cloud
    return report


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _write_registry_snapshot(payload: dict[str, Any]) -> None:
    try:
        _REGISTRY_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = _REGISTRY_SNAPSHOT_PATH.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(_REGISTRY_SNAPSHOT_PATH)
    except OSError as exc:
        logger.warning(
            "model training registry snapshot write failed",
            error=safe_error_text(exc, limit=180),
        )


def _load_registry_snapshot() -> dict[str, Any] | None:
    try:
        payload = json.loads(_REGISTRY_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) and payload.get("models") else None


def _cached_registry(*, include_stale: bool = False) -> dict[str, Any] | None:
    global _registry_cache, _registry_last_success_at
    if _registry_cache is None:
        persisted = _load_registry_snapshot()
        if persisted is None:
            return None
        _registry_cache = (
            time.monotonic() - _REGISTRY_CACHE_TTL_SECONDS - 1.0,
            persisted,
        )
        _registry_last_success_at = str(
            persisted.get("registry_snapshot_generated_at") or ""
        ) or None
    stored_at, payload = _registry_cache
    age_seconds = max(0.0, time.monotonic() - stored_at)
    stale = age_seconds > _REGISTRY_CACHE_TTL_SECONDS
    if stale and not include_stale:
        return None
    result = dict(payload)
    result["cache"] = {
        "hit": True,
        "stale": stale,
        "age_seconds": round(age_seconds, 3),
        "refresh_in_background": stale,
        "last_success_at": _registry_last_success_at,
        "refresh_error": _registry_refresh_error,
    }
    return result


async def build_model_training_registry_status() -> dict[str, Any]:
    """Build the complete model lifecycle view outside the request path."""

    from web_dashboard.api.dashboard import _get_model_observability_snapshot_for_refresh

    observability = await _get_model_observability_snapshot_for_refresh()
    sections = observability.get("sections") if isinstance(observability, dict) else {}
    sections = sections if isinstance(sections, dict) else {}
    selected_mode = "live" if mode_manager.mode.value == "live" else "paper"
    contribution_status: dict[str, Any] = {
        "state": "ready",
        "timeout_seconds": _CONTRIBUTION_TIMEOUT_SECONDS,
    }
    try:
        contribution_performance = await asyncio.wait_for(
            ModelContributionPerformanceService().recent(selected_mode),
            timeout=_CONTRIBUTION_TIMEOUT_SECONDS,
        )
        if not isinstance(contribution_performance, dict):
            contribution_performance = {}
    except TimeoutError:
        contribution_performance = {}
        contribution_status["state"] = "timeout"
    except Exception as exc:
        contribution_performance = {}
        contribution_status.update(
            {
                "state": "degraded",
                "error": safe_error_text(exc, limit=240),
            }
        )

    from web_dashboard.api.dashboard import _compact_training_scheduler_state

    scheduler_state = _compact_training_scheduler_state(MODEL_TRAINING_STATE_STORE.read())
    registry = build_model_training_registry(
        local_ml_status=sections.get("local_ml") or {"status": "missing"},
        local_tools_status=sections.get("local_ai_tools") or {"status": "missing"},
        specialist_report=load_model_training_report(
            "phase3/specialist_shadow_evaluation_latest.json"
        ),
        model_server_report=_model_server_report_with_runtime_configuration(),
        contribution_performance=contribution_performance,
        scheduler_state=scheduler_state,
    )
    registry["contribution_performance_status"] = contribution_status
    registry["scheduler_state"] = scheduler_state
    registry["model_observability"] = observability
    return registry


async def _refresh_registry_cache() -> None:
    global _registry_cache, _registry_last_success_at, _registry_refresh_error, _registry_refresh_task
    try:
        payload = await build_model_training_registry_status()
        _registry_last_success_at = _utc_now_iso()
        _registry_refresh_error = None
        payload["registry_snapshot_generated_at"] = _registry_last_success_at
        _registry_cache = (time.monotonic(), payload)
        _write_registry_snapshot(payload)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        _registry_refresh_error = safe_error_text(exc, limit=240)
        logger.warning(
            "model training registry background refresh failed",
            error=_registry_refresh_error,
        )
    finally:
        _registry_refresh_task = None


async def _bounded_local_status(
    factory: Any,
    *,
    name: str,
    timeout_seconds: float = _FAST_LOCAL_STATUS_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    try:
        payload = await asyncio.wait_for(
            factory(),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        return {
            "status": "status_timeout",
            "degraded_reason": f"{name}_status_timeout",
        }
    except Exception as exc:
        return {
            "status": "status_error",
            "degraded_reason": f"{name}_status_error",
            "error": safe_error_text(exc, limit=180),
        }
    return payload if isinstance(payload, dict) else {"status": "status_error"}


async def _fast_local_registry_status() -> dict[str, Any]:
    """Build truthful local cards while the full observability report warms."""

    from web_dashboard.api.dashboard import (
        get_local_ai_tools_status,
        get_ml_signal_status,
        get_model_observability_snapshot,
    )

    # The observability endpoint already owns the local-model readers and their
    # single-flight/cache lifecycle.  Calling both readers again here made a
    # cold dashboard request fan out into duplicate probes and could turn a
    # temporary timeout into a misleading "unavailable" card.  Reuse the
    # sections returned by that endpoint and only probe a section that is truly
    # absent or still represented by a generic warming/error placeholder.
    observability = await _bounded_local_status(
        lambda: get_model_observability_snapshot(request=object()),
        name="model_observability",
        timeout_seconds=_FAST_OBSERVABILITY_TIMEOUT_SECONDS,
    )
    raw_sections = observability.get("sections") if isinstance(observability, dict) else {}
    sections = dict(raw_sections) if isinstance(raw_sections, dict) else {}

    def _usable_section(value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        status = str(value.get("status") or "").strip().lower()
        return status not in {"", "warming", "status_timeout", "timeout", "status_error", "error"}

    local_ml = sections.get("local_ml") if _usable_section(sections.get("local_ml")) else None
    local_tools = (
        sections.get("local_ai_tools")
        if _usable_section(sections.get("local_ai_tools"))
        else None
    )
    fallback_tasks: list[Awaitable[dict[str, Any]]] = []
    fallback_names: list[str] = []
    if local_ml is None:
        fallback_names.append("local_ml")
        fallback_tasks.append(_bounded_local_status(get_ml_signal_status, name="local_ml"))
    if local_tools is None:
        fallback_names.append("local_ai_tools")
        fallback_tasks.append(
            _bounded_local_status(get_local_ai_tools_status, name="local_ai_tools")
        )
    if fallback_tasks:
        fallback_values = await asyncio.gather(*fallback_tasks)
        for name, value in zip(fallback_names, fallback_values, strict=True):
            if name == "local_ml":
                local_ml = value
            else:
                local_tools = value
    local_ml = local_ml or {"status": "status_error", "degraded_reason": "local_ml_missing"}
    local_tools = local_tools or {
        "status": "status_error",
        "degraded_reason": "local_ai_tools_missing",
    }
    sections = observability.get("sections") if isinstance(observability, dict) else {}
    sections = dict(sections) if isinstance(sections, dict) else {}
    sections.update({"local_ml": local_ml, "local_ai_tools": local_tools})
    observability = dict(observability) if isinstance(observability, dict) else {}
    observability["sections"] = sections

    from web_dashboard.api.dashboard import _compact_training_scheduler_state

    scheduler_state = _compact_training_scheduler_state(MODEL_TRAINING_STATE_STORE.read())
    registry = build_model_training_registry(
        local_ml_status=local_ml,
        local_tools_status=local_tools,
        specialist_report=load_model_training_report(
            "phase3/specialist_shadow_evaluation_latest.json"
        ),
        model_server_report=_model_server_report_with_runtime_configuration(),
        scheduler_state=scheduler_state,
    )
    registry["scheduler_state"] = scheduler_state
    registry["model_observability"] = observability
    registry["cache"] = {"hit": False, "refresh_in_background": True}
    return registry


@router.get("/model-training/registry")
async def get_model_training_registry_status() -> dict[str, Any]:
    """Return the cached lifecycle view while a complete refresh runs."""

    global _registry_refresh_task
    cached = _cached_registry(include_stale=True)
    if cached is not None:
        if (
            cached.get("cache", {}).get("stale")
            and (_registry_refresh_task is None or _registry_refresh_task.done())
        ):
            _registry_refresh_task = asyncio.create_task(_refresh_registry_cache())
        return sanitize_payload(cached)

    if _registry_refresh_task is None or _registry_refresh_task.done():
        _registry_refresh_task = asyncio.create_task(_refresh_registry_cache())

    fast = await _fast_local_registry_status()
    fast["cache"] = {
        "hit": False,
        "stale": False,
        "refresh_in_background": True,
        "last_success_at": None,
        "refresh_error": None,
    }
    return sanitize_payload(fast)


@router.get("/model-training/scheduler")
async def get_model_training_scheduler_status() -> dict[str, Any]:
    from web_dashboard.api.dashboard import _compact_training_scheduler_state

    return sanitize_payload(
        _compact_training_scheduler_state(MODEL_TRAINING_STATE_STORE.read())
    )


async def shutdown_model_training_status_tasks() -> None:
    global _registry_refresh_task
    task = _registry_refresh_task
    _registry_refresh_task = None
    if task is not None and not task.done():
        # Let an in-flight read finish its connection handshake first.  A hard
        # cancellation during aiosqlite connect can create an orphaned driver
        # object after AsyncSession.close() has already run.
        try:
            await asyncio.wait_for(
                asyncio.shield(task), timeout=_REGISTRY_SHUTDOWN_GRACE_SECONDS
            )
        except TimeoutError:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        except asyncio.CancelledError:
            pass

    # A registry refresh owns the complete model-observability build, which in
    # turn may have spawned section refresh tasks that touch the async DB.  A
    # caller shutting down only this module must drain that child task tree too;
    # otherwise aiosqlite can be finalized after the event loop is closed.
    from web_dashboard.api.dashboard import shutdown_dashboard_observability_tasks

    await shutdown_dashboard_observability_tasks()
    # Registry/observability refreshes may have opened read-only SQLite
    # sessions before cancellation reached their callers.  Dispose the shared
    # engine here as part of module shutdown so worker connections are closed
    # before the event loop is torn down.
    from db.session import close_db

    await close_db()
