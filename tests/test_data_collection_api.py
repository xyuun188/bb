from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from config.settings import settings
from db.session import close_db, init_db
from web_dashboard.api import data_collection as data_collection_module
from web_dashboard.app import create_app


async def _use_temp_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    await close_db()
    db_path = tmp_path / "data-collection-api.db"
    monkeypatch.setattr(settings, "database_url", f"sqlite+aiosqlite:///{db_path.as_posix()}")
    await init_db()


@pytest.mark.asyncio
async def test_data_collection_status_exposes_sources_and_training(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    await _use_temp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(settings, "dashboard_admin_api_key", "")
    monkeypatch.setattr(settings, "external_event_scraper_enabled", False)
    monkeypatch.setattr(settings, "external_event_scraper_sources", [])

    try:
        app = create_app()
        transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/api/data-collection/status")
    finally:
        await close_db()

    assert response.status_code == 200
    body = response.json()
    assert body["config"]["external_event_scraper_enabled"] is False
    assert body["config"]["external_event_scraper_uses_default_sources"] is True
    assert any(source["key"] == "scrapling" for source in body["sources"])
    assert "news" in body["stats"]
    assert "text_sentiment_quality_sample" in body["training"]
    assert "local_ai_tools" in body["training"]


@pytest.mark.asyncio
async def test_data_collection_normalizes_unknown_local_ai_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLocalAIToolsClient:
        async def status(self) -> dict[str, Any]:
            return {
                "available": True,
                "status": "unknown",
                "shadow_sample_count": 12,
                "trade_sample_count": 3,
                "text_sentiment_sample_count": 8,
            }

    monkeypatch.setattr(
        data_collection_module._dash,
        "_dashboard_local_ai_tools_client",
        lambda: FakeLocalAIToolsClient(),
    )

    status = await data_collection_module._local_ai_training_status()

    assert status["status"] == "learning_only"
    assert status["raw_status"] == "unknown"
    assert status["available"] is True


@pytest.mark.asyncio
async def test_data_collection_settings_rejects_private_scrapling_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_updates: list[dict[str, Any]] = []
    monkeypatch.setattr(settings, "dashboard_admin_api_key", "")

    def capture_update_env_file(self: object, updates: dict[str, Any]) -> None:
        captured_updates.append(updates)

    monkeypatch.setattr(settings.__class__, "update_env_file", capture_update_env_file)
    app = create_app()
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/data-collection/settings",
            json={
                "external_event_scraper_sources": [
                    {"name": "bad", "url": "https://127.0.0.1/internal"}
                ]
            },
        )

    assert response.status_code == 400
    assert "public" in response.json()["detail"] or "globally routable" in response.json()["detail"]
    assert captured_updates == []


@pytest.mark.asyncio
async def test_data_collection_settings_persists_safe_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    await _use_temp_db(monkeypatch, tmp_path)
    captured_updates: list[dict[str, Any]] = []
    monkeypatch.setattr(settings, "dashboard_admin_api_key", "")
    monkeypatch.setattr(settings, "external_event_scraper_enabled", False)
    monkeypatch.setattr(settings, "external_event_scraper_sources", [])

    def capture_update_env_file(self: object, updates: dict[str, Any]) -> None:
        captured_updates.append(updates)

    monkeypatch.setattr(settings.__class__, "update_env_file", capture_update_env_file)

    try:
        app = create_app()
        transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post(
                "/api/data-collection/settings",
                json={
                    "external_event_scraper_enabled": True,
                    "external_event_scraper_interval_seconds": 600,
                    "external_event_scraper_timeout_seconds": 5,
                    "external_event_scraper_max_sources": 2,
                    "external_event_scraper_max_items_per_source": 4,
                    "external_event_scraper_sources": [
                        {
                            "name": "ethereum_blog",
                            "url": "https://blog.ethereum.org/",
                            "symbols": ["ETH"],
                            "weight": 0.72,
                        }
                    ],
                },
            )
    finally:
        await close_db()

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert captured_updates
    persisted = captured_updates[-1]
    assert persisted["EXTERNAL_EVENT_SCRAPER_ENABLED"] == "true"
    assert persisted["EXTERNAL_EVENT_SCRAPER_INTERVAL_SECONDS"] == "600"
    assert "ethereum_blog" in persisted["EXTERNAL_EVENT_SCRAPER_SOURCES"]
    assert "api_key" not in str(persisted).lower()
