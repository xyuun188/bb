from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from config.settings import settings
from db.session import close_db, init_db
from web_dashboard.api.security import hash_dashboard_password
from web_dashboard.app import create_app


async def _create_auth_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    password: str = "InitialPass123",
) -> httpx.AsyncClient:
    await close_db()
    db_path = tmp_path / "dashboard-auth.db"
    monkeypatch.setattr(settings, "database_url", f"sqlite+aiosqlite:///{db_path.as_posix()}")
    monkeypatch.setattr(settings, "dashboard_auth_enabled", True)
    monkeypatch.setattr(settings, "dashboard_auth_username", "admin")
    monkeypatch.setattr(settings, "dashboard_auth_email", "admin@example.test")
    monkeypatch.setattr(settings, "dashboard_auth_password_hash", hash_dashboard_password(password))
    monkeypatch.setattr(settings, "dashboard_session_secret", "unit-dashboard-session-secret")
    monkeypatch.setattr(settings, "dashboard_admin_api_key", "")
    monkeypatch.setattr(settings, "dashboard_host", "0.0.0.0")
    await init_db()
    app = create_app()
    transport = httpx.ASGITransport(app=app, client=("203.0.113.9", 12345))
    return httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
        follow_redirects=False,
    )


@pytest.mark.asyncio
async def test_dashboard_account_api_requires_login(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = await _create_auth_client(monkeypatch, tmp_path)
    try:
        response = await client.get("/api/auth/account")
    finally:
        await client.aclose()
        await close_db()

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_dashboard_account_lists_seeded_admin_after_login(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = await _create_auth_client(monkeypatch, tmp_path)
    try:
        login = await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "InitialPass123"},
        )
        response = await client.get("/api/auth/account")
    finally:
        await client.aclose()
        await close_db()

    assert login.status_code == 200
    assert response.status_code == 200
    body = response.json()
    assert body["current_user"]["username"] == "admin"
    assert body["current_user"]["masked_email"] == "a***n@example.test"
    assert [user["username"] for user in body["users"]] == ["admin"]


@pytest.mark.asyncio
async def test_dashboard_password_change_replaces_database_login(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = await _create_auth_client(monkeypatch, tmp_path)
    try:
        login = await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "InitialPass123"},
        )
        bad_change = await client.post(
            "/api/auth/account/password",
            json={"current_password": "wrong", "new_password": "ChangedPass123"},
        )
        good_change = await client.post(
            "/api/auth/account/password",
            json={"current_password": "InitialPass123", "new_password": "ChangedPass123"},
        )
        await client.post("/api/auth/logout")
        old_login = await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "InitialPass123"},
        )
        new_login = await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "ChangedPass123"},
        )
    finally:
        await client.aclose()
        await close_db()

    assert login.status_code == 200
    assert bad_change.status_code == 400
    assert good_change.status_code == 200
    assert old_login.status_code == 401
    assert new_login.status_code == 200


@pytest.mark.asyncio
async def test_dashboard_admin_can_create_reset_and_disable_user(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = await _create_auth_client(monkeypatch, tmp_path)
    try:
        await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "InitialPass123"},
        )
        created = await client.post(
            "/api/auth/users",
            json={
                "username": "ops",
                "email": "ops@example.test",
                "password": "OpsPass12345",
            },
        )
        reset = await client.put(
            "/api/auth/users/ops",
            json={"password": "OpsPassChanged123", "is_active": True},
        )
        await client.post("/api/auth/logout")
        old_login = await client.post(
            "/api/auth/login",
            json={"username": "ops", "password": "OpsPass12345"},
        )
        new_login = await client.post(
            "/api/auth/login",
            json={"username": "ops", "password": "OpsPassChanged123"},
        )
        await client.post("/api/auth/logout")
        await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "InitialPass123"},
        )
        current_delete = await client.delete("/api/auth/users/admin")
        current_deactivate = await client.post("/api/auth/users/admin/deactivate")
        disabled = await client.post("/api/auth/users/ops/deactivate")
        deleted = await client.delete("/api/auth/users/ops")
        await client.post("/api/auth/logout")
        disabled_login = await client.post(
            "/api/auth/login",
            json={"username": "ops", "password": "OpsPassChanged123"},
        )
    finally:
        await client.aclose()
        await close_db()

    assert created.status_code == 200
    assert reset.status_code == 200
    assert old_login.status_code == 401
    assert new_login.status_code == 200
    assert current_delete.status_code == 400
    assert current_deactivate.status_code == 400
    assert disabled.status_code == 200
    assert deleted.status_code == 200
    assert disabled_login.status_code == 401


@pytest.mark.asyncio
async def test_dashboard_account_rejects_disabled_current_user(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    admin_key = "unit-dashboard-write-key"
    client = await _create_auth_client(monkeypatch, tmp_path)
    monkeypatch.setattr(settings, "dashboard_admin_api_key", admin_key)
    try:
        await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "InitialPass123"},
        )
        created = await client.post(
            "/api/auth/users",
            json={
                "username": "ops",
                "email": "ops@example.test",
                "password": "OpsPass12345",
            },
        )
        await client.post("/api/auth/logout")
        login = await client.post(
            "/api/auth/login",
            json={"username": "ops", "password": "OpsPass12345"},
        )
        disabled = await client.post(
            "/api/auth/users/ops/deactivate",
            headers={"X-Dashboard-Admin-Key": admin_key},
        )
        response = await client.get("/api/auth/account")
        status_response = await client.get("/api/auth/status")
    finally:
        await client.aclose()
        await close_db()

    assert created.status_code == 200
    assert login.status_code == 200
    assert disabled.status_code == 200
    assert response.status_code == 401
    assert status_response.status_code == 401
    assert "重新登录" in response.json()["detail"]
