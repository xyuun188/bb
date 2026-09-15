"""Dashboard model-training registry and scheduler status endpoints."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter

from config.settings import settings
from core.safe_output import safe_error_text
from core.trading_mode import mode_manager
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
_registry_cache: tuple[float, dict[str, Any]] | None = None
_registry_refresh_task: asyncio.Task[Any] | None = None


def load_model_training_report(relative_path: str) -> dict[str, Any]:
    try:
        payload = json.loads((settings.data_dir / relative_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _cached_registry() -> dict[str, Any] | None:
    if _registry_cache is None:
        return None
    stored_at, payload = _registry_cache
    if time.monotonic() - stored_at > _REGISTRY_CACHE_TTL_SECONDS:
        return None
    return dict(payload)


async def build_model_training_registry_status() -> dict[str, Any]:
    """Build the complete model lifecycle view outside the request path."""

    from web_dashboard.api.dashboard import get_model_observability_snapshot

    observability = await get_model_observability_snapshot()
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

    scheduler_state = MODEL_TRAINING_STATE_STORE.read()
    registry = build_model_training_registry(
        local_ml_status=sections.get("local_ml") or {"status": "missing"},
        local_tools_status=sections.get("local_ai_tools") or {"status": "missing"},
        specialist_report=load_model_training_report(
            "phase3/specialist_shadow_evaluation_latest.json"
        ),
        model_server_report=load_model_training_report(
            "phase3_model_server_readiness_reports/latest.json"
        ),
        contribution_performance=contribution_performance,
        scheduler_state=scheduler_state,
    )
    registry["contribution_performance_status"] = contribution_status
    registry["scheduler_state"] = scheduler_state
    registry["model_observability"] = observability
    return registry


async def _refresh_registry_cache() -> None:
    global _registry_cache, _registry_refresh_task
    try:
        payload = await build_model_training_registry_status()
        _registry_cache = (time.monotonic(), payload)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning(
            "model training registry background refresh failed",
            error=safe_error_text(exc, limit=240),
        )
    finally:
        _registry_refresh_task = None


@router.get("/model-training/registry")
async def get_model_training_registry_status() -> dict[str, Any]:
    """Return the cached lifecycle view while a complete refresh runs."""

    cached = _cached_registry()
    if cached is not None:
        return sanitize_payload(cached)

    global _registry_refresh_task
    if _registry_refresh_task is None or _registry_refresh_task.done():
        _registry_refresh_task = asyncio.create_task(_refresh_registry_cache())

    from web_dashboard.api.dashboard import get_model_observability_snapshot

    observability = await get_model_observability_snapshot(request=object())
    sections = observability.get("sections") if isinstance(observability, dict) else {}
    sections = sections if isinstance(sections, dict) else {}
    registry = build_model_training_registry(
        local_ml_status=sections.get("local_ml") or {"status": "warming"},
        local_tools_status=sections.get("local_ai_tools") or {"status": "warming"},
    )
    registry["model_observability"] = observability
    registry["cache"] = {"hit": False, "refresh_in_background": True}
    return sanitize_payload(registry)


@router.get("/model-training/scheduler")
async def get_model_training_scheduler_status() -> dict[str, Any]:
    return sanitize_payload(MODEL_TRAINING_STATE_STORE.read())


async def shutdown_model_training_status_tasks() -> None:
    global _registry_refresh_task
    task = _registry_refresh_task
    _registry_refresh_task = None
    if task is None or task.done():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
