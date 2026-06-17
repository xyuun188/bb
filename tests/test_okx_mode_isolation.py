from __future__ import annotations

import httpx
import pytest

from config.settings import settings
from web_dashboard.api import settings_api as settings_api_module
from web_dashboard.app import create_app


@pytest.mark.asyncio
async def test_live_mode_switch_rejects_legacy_unified_okx_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured = "unit-" + "dashboard-write-token"
    monkeypatch.setattr(settings, "dashboard_admin_api_key", configured)
    monkeypatch.setattr(settings, "okx_api_key", "legacy-key")
    monkeypatch.setattr(settings, "okx_api_secret", "legacy-secret")
    monkeypatch.setattr(settings, "okx_passphrase", "legacy-pass")
    monkeypatch.setattr(settings, "okx_live_api_key", "")
    monkeypatch.setattr(settings, "okx_live_api_secret", "")
    monkeypatch.setattr(settings, "okx_live_passphrase", "")
    app = create_app()
    transport = httpx.ASGITransport(app=app, client=("203.0.113.9", 12345))

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/control/mode",
            headers={"Authorization": f"Bearer {configured}"},
            json={"mode": "live"},
        )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["mode"] == "live"
    assert detail["settings_tab"] == "okx"
    assert set(detail["missing_fields"]) == {"API Key", "API Secret", "Passphrase"}


@pytest.mark.asyncio
async def test_okx_settings_live_does_not_mask_legacy_unified_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "okx_api_key", "legacy-key")
    monkeypatch.setattr(settings, "okx_api_secret", "legacy-secret")
    monkeypatch.setattr(settings, "okx_passphrase", "legacy-pass")
    monkeypatch.setattr(settings, "okx_paper_api_key", "")
    monkeypatch.setattr(settings, "okx_paper_api_secret", "")
    monkeypatch.setattr(settings, "okx_paper_passphrase", "")
    monkeypatch.setattr(settings, "okx_live_api_key", "")
    monkeypatch.setattr(settings, "okx_live_api_secret", "")
    monkeypatch.setattr(settings, "okx_live_passphrase", "")

    response = await settings_api_module.get_okx_settings()

    assert response["paper"]["api_key"]
    assert response["paper"]["has_secret"] is True
    assert response["paper"]["has_passphrase"] is True
    assert response["live"]["api_key"] == ""
    assert response["live"]["has_secret"] is False
    assert response["live"]["has_passphrase"] is False


def test_live_okx_credentials_do_not_fallback_to_legacy_unified_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "okx_api_key", "legacy-key")
    monkeypatch.setattr(settings, "okx_api_secret", "legacy-secret")
    monkeypatch.setattr(settings, "okx_passphrase", "legacy-pass")
    monkeypatch.setattr(settings, "okx_live_api_key", "")
    monkeypatch.setattr(settings, "okx_live_api_secret", "")
    monkeypatch.setattr(settings, "okx_live_passphrase", "")

    credentials = settings.get_okx_credentials("live")

    assert credentials["api_key"] == ""
    assert credentials["api_secret"] == ""
    assert "passphrase" not in credentials
