from __future__ import annotations

import pytest

from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from services.secure_runtime_config import load_secure_settings_into_runtime
from services.secure_settings import SecureSettingsService


@pytest.mark.asyncio
async def test_secure_runtime_keeps_env_local_ai_tools_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("BB_SECURE_SETTINGS_KEY", "03" * 32)
    monkeypatch.setenv("LOCAL_AI_TOOLS_API_KEY", "runtime-tools-key")
    db_path = tmp_path / "secure.db"
    monkeypatch.setattr(settings, "database_url", f"sqlite+aiosqlite:///{db_path.as_posix()}")
    monkeypatch.setattr(settings, "local_ai_tools_api_key", "runtime-tools-key")
    await close_db()
    await init_db()
    try:
        async with get_session_ctx() as session:
            service = SecureSettingsService(session)
            await service.set_secret(
                "local_ai_tools.api_key",
                "stale-secure-tools-key",
                actor="test",
            )

        loaded = await load_secure_settings_into_runtime()
    finally:
        await close_db()

    assert settings.local_ai_tools_api_key == "runtime-tools-key"
    assert "local_ai_tools.api_key" in loaded["skipped"]
    assert "local_ai_tools.api_key" not in loaded["loaded_keys"]
