from __future__ import annotations

import httpx
import pytest

from config.settings import settings
from web_dashboard.api import vector_memory as vector_memory_module
from web_dashboard.app import create_app


@pytest.mark.asyncio
async def test_vector_memory_status_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "dashboard_admin_api_key", "")
    monkeypatch.setattr(settings, "vector_memory_enabled", False)

    app = create_app()
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/vector-memory/status")

    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is False
    assert body["status"] == "disabled"


@pytest.mark.asyncio
async def test_vector_memory_search_disabled_is_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "dashboard_admin_api_key", "")
    monkeypatch.setattr(settings, "vector_memory_enabled", False)

    app = create_app()
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/vector-memory/search",
            json={"query": "AI16Z 重复亏损开仓", "top_k": 3},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "disabled"
    assert body["hits"] == []


@pytest.mark.asyncio
async def test_vector_memory_settings_persists_and_reloads_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_updates: list[dict[str, str]] = []
    reset_calls = 0

    class FakeVectorMemoryService:
        async def reset_store(self) -> None:
            nonlocal reset_calls
            reset_calls += 1

        async def status(self) -> dict[str, object]:
            return {
                "enabled": settings.vector_memory_enabled,
                "status": "ready" if settings.vector_memory_enabled else "disabled",
                "configured_backend": settings.vector_memory_backend,
            }

    def capture_update_env_file(self: object, updates: dict[str, str]) -> None:
        captured_updates.append(updates)

    monkeypatch.setattr(settings, "dashboard_admin_api_key", "")
    monkeypatch.setattr(settings, "vector_memory_enabled", False)
    monkeypatch.setattr(settings, "vector_memory_backend", "auto")
    monkeypatch.setattr(settings.__class__, "update_env_file", capture_update_env_file)
    monkeypatch.setattr(
        vector_memory_module,
        "get_vector_memory_service",
        lambda: FakeVectorMemoryService(),
    )

    app = create_app()
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/vector-memory/settings",
            json={"enabled": True, "backend": "jsonl"},
        )

    assert response.status_code == 200
    assert response.json()["enabled"] is True
    assert captured_updates[-1] == {
        "VECTOR_MEMORY_ENABLED": "true",
        "VECTOR_MEMORY_BACKEND": "jsonl",
    }
    assert reset_calls == 1


@pytest.mark.asyncio
async def test_vector_memory_clear_endpoint_clears_training_epoch_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_calls = 0
    clear_reasons: list[str] = []

    class FakeVectorMemoryService:
        async def clear_index(self, *, reason: str) -> dict[str, object]:
            nonlocal clear_calls
            clear_calls += 1
            clear_reasons.append(reason)
            return {
                "enabled": True,
                "status": "cleared",
                "removed": 7,
                "document_count": 0,
                "training_epoch_policy": "exclude_documents_before_training_epoch",
            }

    monkeypatch.setattr(settings, "dashboard_admin_api_key", "")
    monkeypatch.setattr(
        vector_memory_module,
        "get_vector_memory_service",
        lambda: FakeVectorMemoryService(),
    )

    app = create_app()
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/api/vector-memory/clear")

    assert response.status_code == 200
    assert response.json()["status"] == "cleared"
    assert response.json()["removed"] == 7
    assert clear_calls == 1
    assert clear_reasons == ["phase3_dashboard_clear_old_index"]
