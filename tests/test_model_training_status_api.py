from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from web_dashboard.api import dashboard, model_training_status


def test_model_training_status_routes_have_one_owner() -> None:
    route_paths = {route.path for route in model_training_status.router.routes}

    assert "/model-training/registry" in route_paths
    assert "/model-training/scheduler" in route_paths
    assert "get_model_training_registry_status" not in dashboard.__dict__
    assert "get_model_training_scheduler_status" not in dashboard.__dict__
    assert "_build_model_training_registry_status" not in dashboard.__dict__


def test_local_model_diagnostic_budget_fits_dashboard_light_request_budget() -> None:
    from web_dashboard import app

    assert "/local-ai-tools/status" in app._LIGHT_API_PATH_MARKERS
    assert "/ml-signal/status" in app._LIGHT_API_PATH_MARKERS
    assert (
        dashboard._DASHBOARD_LOCAL_AI_STATUS_TIMEOUT_SECONDS
        + dashboard._DASHBOARD_LOCAL_AI_CURSOR_TIMEOUT_SECONDS
        < app.DASHBOARD_API_TIMEOUT_SECONDS["light"]
    )
    assert (
        model_training_status._FAST_OBSERVABILITY_TIMEOUT_SECONDS
        < app.DASHBOARD_API_TIMEOUT_SECONDS["light"]
    )


@pytest.mark.asyncio
async def test_model_training_scheduler_status_reads_owned_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Store:
        def read(self) -> dict[str, Any]:
            return {"status": "ok", "schedulers": {"platform": {"state": "idle"}}}

    monkeypatch.setattr(model_training_status, "MODEL_TRAINING_STATE_STORE", Store())

    payload = await model_training_status.get_model_training_scheduler_status()

    assert payload["status"] == "ok"
    assert payload["schedulers"]["platform"]["state"] == "idle"


@pytest.mark.asyncio
async def test_cold_registry_uses_live_local_status_instead_of_warming_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def local_ml() -> dict[str, Any]:
        return {"available": True, "status": "trained", "trained_at": "2026-09-15T00:00:00Z"}

    async def local_tools() -> dict[str, Any]:
        return {
            "available": True,
            "service_available": True,
            "model_bundle_available": True,
            "status": "canary",
            "models": {
                "profit": "profit-model",
                "loss_filter": "loss-model",
                "timeseries": "timeseries-model",
                "deep_timeseries": "sequence-model",
                "deep_sentiment": "sentiment-model",
                "exit": "exit-model",
            },
        }

    async def warming_observability(request: object = None) -> dict[str, Any]:
        del request
        return {"status": "warming", "sections": {}}

    monkeypatch.setattr(model_training_status, "_registry_cache", None)
    monkeypatch.setattr(model_training_status, "_registry_refresh_task", None)
    monkeypatch.setattr(dashboard, "get_ml_signal_status", local_ml)
    monkeypatch.setattr(dashboard, "get_local_ai_tools_status", local_tools)
    monkeypatch.setattr(dashboard, "get_model_observability_snapshot", warming_observability)

    payload = await model_training_status.get_model_training_registry_status()
    local_rows = [
        row for row in payload["models"] if str(row["model_id"]).startswith("local_ai_")
    ]

    assert local_rows
    assert all(row["runtime_available"] is True for row in local_rows)
    assert all(row["artifact_available"] is True for row in local_rows)
    assert all(row["lifecycle"] == "promotion_blocked" for row in local_rows)
    await model_training_status.shutdown_model_training_status_tasks()


@pytest.mark.asyncio
async def test_cold_registry_bounds_slow_observability_without_hiding_local_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def local_ml() -> dict[str, Any]:
        return {"available": True, "status": "trained"}

    async def local_tools() -> dict[str, Any]:
        return {
            "available": True,
            "service_available": True,
            "model_bundle_available": True,
            "status": "canary",
            "models": {"profit": "profit-model"},
        }

    async def slow_observability(request: object = None) -> dict[str, Any]:
        del request
        await asyncio.sleep(0.05)
        return {"status": "ready", "sections": {}}

    monkeypatch.setattr(model_training_status, "_registry_cache", None)
    monkeypatch.setattr(model_training_status, "_registry_refresh_task", None)
    monkeypatch.setattr(model_training_status, "_FAST_OBSERVABILITY_TIMEOUT_SECONDS", 0.001)
    monkeypatch.setattr(dashboard, "get_ml_signal_status", local_ml)
    monkeypatch.setattr(dashboard, "get_local_ai_tools_status", local_tools)
    monkeypatch.setattr(dashboard, "get_model_observability_snapshot", slow_observability)

    payload = await model_training_status.get_model_training_registry_status()

    assert payload["model_observability"]["status"] == "status_timeout"
    local_rows = [
        row for row in payload["models"] if str(row["model_id"]).startswith("local_ai_")
    ]
    assert local_rows
    assert all(row["runtime_available"] is True for row in local_rows)
    await model_training_status.shutdown_model_training_status_tasks()


@pytest.mark.asyncio
async def test_stale_registry_keeps_last_known_good_during_background_refresh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    snapshot = {
        "models": [
            {
                "model_id": "local_ml_profit_quality",
                "runtime_available": True,
                "artifact_available": True,
                "identity_verified": True,
                "lifecycle": "trained",
            }
        ],
        "registry_snapshot_generated_at": "2026-09-15T00:00:00+00:00",
    }
    snapshot_path = tmp_path / "registry.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    monkeypatch.setattr(model_training_status, "_REGISTRY_SNAPSHOT_PATH", snapshot_path)
    monkeypatch.setattr(
        model_training_status,
        "_registry_cache",
        (time.monotonic() - model_training_status._REGISTRY_CACHE_TTL_SECONDS - 1, snapshot),
    )
    monkeypatch.setattr(model_training_status, "_registry_refresh_task", None)

    payload = await model_training_status.get_model_training_registry_status()

    assert payload["models"][0]["runtime_available"] is True
    assert payload["models"][0]["lifecycle"] == "trained"
    assert payload["cache"]["stale"] is True
    assert payload["cache"]["refresh_in_background"] is True
    await model_training_status.shutdown_model_training_status_tasks()


@pytest.mark.asyncio
async def test_model_observability_serves_stale_snapshot_while_refreshing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale = {
        "status": "ok",
        "source": "dashboard.model_observability",
        "sections": {"local_ml": {"status": "trained"}},
        "registry": {"models": [{"model_id": "local_ml_profit_quality"}]},
    }
    monkeypatch.setattr(dashboard, "_dashboard_heavy_cache", {})
    monkeypatch.setattr(dashboard, "_dashboard_heavy_cache_locks", {})
    dashboard._dashboard_heavy_cache[("model-observability",)] = (
        datetime.now(UTC) - timedelta(seconds=dashboard._DASHBOARD_MODEL_OBSERVABILITY_TTL_SECONDS + 1),
        stale,
    )

    async def never_finishes() -> dict[str, Any]:
        await asyncio.sleep(60)
        return stale

    monkeypatch.setattr(dashboard, "_build_model_observability_snapshot", never_finishes)
    monkeypatch.setattr(dashboard, "_model_observability_refresh_task", None)

    payload = await dashboard.get_model_observability_snapshot(object())

    assert payload["status"] == "stale"
    assert payload["cache"]["stale"] is True
    assert payload["cache"]["refresh_in_background"] is True
    await dashboard.shutdown_dashboard_observability_tasks()


@pytest.mark.asyncio
async def test_cold_model_observability_keeps_local_models_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def local_ml() -> dict[str, Any]:
        return {"status": "trained", "available": True, "trained_at": "2026-09-15T00:00:00Z"}

    async def local_tools() -> dict[str, Any]:
        return {
            "status": "canary",
            "available": True,
            "service_available": True,
            "model_bundle_available": True,
            "models": {"profit": "profit-model"},
        }

    monkeypatch.setattr(dashboard, "_dashboard_heavy_cache", {})
    monkeypatch.setattr(dashboard, "_dashboard_heavy_cache_locks", {})
    monkeypatch.setattr(dashboard, "_model_observability_refresh_task", None)
    monkeypatch.setattr(dashboard, "get_ml_signal_status", local_ml)
    monkeypatch.setattr(dashboard, "get_local_ai_tools_status", local_tools)
    monkeypatch.setattr(
        dashboard,
        "_refresh_model_observability_cache",
        lambda: asyncio.sleep(60),
    )

    payload = await dashboard.get_model_observability_snapshot(object())

    assert payload["status"] == "warming"
    assert payload["sections"]["local_ml"]["available"] is True
    assert payload["sections"]["local_ai_tools"]["service_available"] is True
    assert payload["sections"]["analysis"]["status"] == "warming"
    assert payload["cache"]["refresh_in_background"] is True
    await dashboard.shutdown_dashboard_observability_tasks()


@pytest.mark.asyncio
async def test_model_observability_caches_slow_sections_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def local_ml() -> dict[str, Any]:
        calls.append("local_ml")
        return {"status": "trained", "available": True}

    async def local_tools() -> dict[str, Any]:
        calls.append("local_tools")
        return {"status": "canary", "available": True}

    async def analysis() -> dict[str, Any]:
        calls.append("analysis")
        return {"status": "ok", "round_id": "r1"}

    async def trade() -> dict[str, Any]:
        calls.append("trade")
        return {"status": "ok", "profit_attribution": {}}

    async def memory() -> dict[str, Any]:
        calls.append("memory")
        return {"status": "ok", "memory_count": 1}

    monkeypatch.setattr(dashboard, "get_ml_signal_status", local_ml)
    monkeypatch.setattr(dashboard, "get_local_ai_tools_status", local_tools)
    monkeypatch.setattr(dashboard, "_latest_analysis_observability", analysis)
    monkeypatch.setattr(dashboard, "_build_trade_observability_snapshot", trade)
    monkeypatch.setattr(dashboard, "_build_expert_memory_observability", memory)
    monkeypatch.setattr(dashboard, "MODEL_TRAINING_STATE_STORE", type("Store", (), {"read": lambda self: {}})())
    monkeypatch.setattr(dashboard, "load_model_training_report", lambda *_args, **_kwargs: {})
    dashboard._clear_dashboard_heavy_cache()

    def registry(**kwargs: Any) -> dict[str, Any]:
        return {"models": [], "summary": {}, **kwargs}

    monkeypatch.setattr(dashboard, "build_model_training_registry", registry)
    first = await dashboard._build_model_observability_snapshot()
    second = await dashboard._build_model_observability_snapshot()

    assert first["sections"]["local_ml"]["status"] == "trained"
    assert second["sections"]["local_ml"]["status"] == "trained"
    assert calls.count("trade") == 1
    assert calls.count("memory") == 1


@pytest.mark.asyncio
async def test_model_observability_does_not_cache_transient_section_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def local_ml() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"status": "status_timeout", "available": False}
        return {"status": "trained", "available": True}

    async def local_tools() -> dict[str, Any]:
        return {"status": "canary", "available": True}

    async def analysis() -> dict[str, Any]:
        return {"status": "ok"}

    async def trade() -> dict[str, Any]:
        return {"status": "ok", "profit_attribution": {}}

    async def memory() -> dict[str, Any]:
        return {"status": "ok"}

    monkeypatch.setattr(dashboard, "get_ml_signal_status", local_ml)
    monkeypatch.setattr(dashboard, "get_local_ai_tools_status", local_tools)
    monkeypatch.setattr(dashboard, "_latest_analysis_observability", analysis)
    monkeypatch.setattr(dashboard, "_build_trade_observability_snapshot", trade)
    monkeypatch.setattr(dashboard, "_build_expert_memory_observability", memory)
    monkeypatch.setattr(
        dashboard,
        "MODEL_TRAINING_STATE_STORE",
        type("Store", (), {"read": lambda self: {}})(),
    )
    monkeypatch.setattr(dashboard, "load_model_training_report", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        dashboard,
        "build_model_training_registry",
        lambda **kwargs: {"models": [], "summary": {}, **kwargs},
    )
    dashboard._clear_dashboard_heavy_cache()

    first = await dashboard._build_model_observability_snapshot()
    second = await dashboard._build_model_observability_snapshot()

    assert first["sections"]["local_ml"]["status"] == "status_timeout"
    assert second["sections"]["local_ml"]["status"] == "trained"
    assert calls == 2


@pytest.mark.asyncio
async def test_model_observability_returns_stale_section_without_waiting_for_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def local_ml() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.5)
        return {"status": "trained", "available": True}

    async def fast_section() -> dict[str, Any]:
        return {"status": "ok"}

    monkeypatch.setattr(dashboard, "get_ml_signal_status", local_ml)
    monkeypatch.setattr(dashboard, "get_local_ai_tools_status", fast_section)
    monkeypatch.setattr(dashboard, "_latest_analysis_observability", fast_section)
    monkeypatch.setattr(dashboard, "_build_trade_observability_snapshot", fast_section)
    monkeypatch.setattr(dashboard, "_build_expert_memory_observability", fast_section)
    monkeypatch.setattr(
        dashboard,
        "MODEL_TRAINING_STATE_STORE",
        type("Store", (), {"read": lambda self: {}})(),
    )
    monkeypatch.setattr(dashboard, "load_model_training_report", lambda *_a, **_k: {})
    monkeypatch.setattr(
        dashboard,
        "build_model_training_registry",
        lambda **kwargs: {"models": [], "summary": {}, **kwargs},
    )
    dashboard._clear_dashboard_heavy_cache()
    dashboard._dashboard_heavy_cache[("model-observability-local-ml",)] = (
        datetime.now(UTC)
        - timedelta(seconds=dashboard._DASHBOARD_MODEL_OBSERVABILITY_TTL_SECONDS + 1),
        {"status": "trained", "available": True},
    )

    started = time.perf_counter()
    payload = await dashboard._build_model_observability_snapshot()
    elapsed = time.perf_counter() - started

    assert elapsed < 0.3
    assert payload["sections"]["local_ml"]["status"] == "stale"
    assert payload["sections"]["local_ml"]["cache"]["refresh_in_background"] is True
    await asyncio.sleep(0)
    assert calls == 1
    await dashboard.shutdown_dashboard_observability_tasks()
