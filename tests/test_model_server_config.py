from __future__ import annotations

from pathlib import Path

import pytest

from config.settings import ENSEMBLE_TRADER_NAME, settings
from db.session import close_db, get_session_ctx, init_db
from models.trade import Position
from services import server_monitor_status
from services.model_server_config import (
    ModelServerConfigError,
    ModelServerConfigNotConfigured,
    get_model_server_settings_public,
    load_model_server_info_from_secure_settings,
    save_model_server_settings,
)
from services.server_monitor_status import ServerMonitorStatusService
from web_dashboard.api import dashboard


async def _use_temp_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    await close_db()
    db_path = tmp_path / "model-server-settings.db"
    monkeypatch.setattr(settings, "database_url", f"sqlite+aiosqlite:///{db_path.as_posix()}")
    await init_db()


@pytest.mark.asyncio
async def test_model_server_settings_are_encrypted_and_masked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BB_SECURE_SETTINGS_KEY", "01" * 32)
    await _use_temp_db(monkeypatch, tmp_path)
    try:
        secret_value = "ModelServer" + "Pass123"
        payload = await save_model_server_settings(
            host="203.0.113.17",
            port=2222,
            username="bbops",
            password=secret_value,
        )
        public = await get_model_server_settings_public()
        info = await load_model_server_info_from_secure_settings()
    finally:
        await close_db()

    assert payload.configured is True
    assert public.as_dict()["password_configured"] is True
    assert public.as_dict()["host"] == "203.0.113.17"
    assert secret_value not in str(public.as_dict())
    assert info.connection_kwargs() == {
        "host": "203.0.113.17",
        "port": 2222,
        "username": "bbops",
        "password": secret_value,
    }


@pytest.mark.asyncio
async def test_model_server_settings_update_reuses_existing_password(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BB_SECURE_SETTINGS_KEY", "02" * 32)
    await _use_temp_db(monkeypatch, tmp_path)
    try:
        secret_value = "OriginalModel" + "Pass123"
        await save_model_server_settings(
            host="203.0.113.17",
            port=2222,
            username="bbops",
            password=secret_value,
        )
        await save_model_server_settings(
            host="203.0.113.18",
            port=22,
            username="root",
            password="",
        )
        info = await load_model_server_info_from_secure_settings()
    finally:
        await close_db()

    assert info.host == "203.0.113.18"
    assert info.port == 22
    assert info.username == "root"
    assert info.password == secret_value


def test_server_monitor_reports_model_server_not_configured() -> None:
    def missing_info(_root: Path) -> object:
        raise ModelServerConfigNotConfigured("请在系统设置 > 模型服务器 中配置服务器连接信息。")

    service = ServerMonitorStatusService(info_loader=missing_info)

    result = service.collect_sync()

    assert result["available"] is False
    assert result["remote_monitor_available"] is False
    assert result["status"] == "model_server_not_configured"
    assert "系统设置" in result["message"]


@pytest.mark.asyncio
async def test_server_monitor_keeps_platform_runtime_when_remote_config_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failing_loader():
        raise ModelServerConfigError("BB_SECURE_SETTINGS_KEY is required")

    async def platform_runtime():
        return {
            "ai_models": [
                {
                    "model": "qwen3-14b-trade",
                    "api_base": "http://127.0.0.1:18000/v1",
                    "available": True,
                }
            ],
            "local_ai_tools": {
                "configured": True,
                "api_base": "http://127.0.0.1:18001",
                "available": True,
            },
        }

    monkeypatch.setattr(
        server_monitor_status,
        "load_model_server_info_from_secure_settings",
        failing_loader,
    )
    monkeypatch.setattr(server_monitor_status, "collect_platform_server_status", lambda: {})
    monkeypatch.setattr(server_monitor_status, "collect_platform_runtime_status", platform_runtime)

    result = await server_monitor_status.get_server_monitor_status_async()

    assert result["available"] is True
    assert result["remote_monitor_available"] is False
    assert result["status"] == "model_server_config_error"
    assert result["platform_runtime"]["ai_models"][0]["model"] == "qwen3-14b-trade"


@pytest.mark.asyncio
async def test_dashboard_ticker_fallback_reads_open_position_prices_from_db(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    await _use_temp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(dashboard, "_data_service", None)
    monkeypatch.setattr(dashboard, "_trading_service", None)
    try:
        async with get_session_ctx() as session:
            session.add(
                Position(
                    model_name=ENSEMBLE_TRADER_NAME,
                    execution_mode="paper",
                    symbol="AAVE/USDT",
                    side="long",
                    quantity=1.0,
                    entry_price=100.0,
                    current_price=101.5,
                    is_open=True,
                )
            )
        prices = await dashboard._get_open_position_prices("paper")
        tickers = await dashboard._build_tickers_for_open_positions(
            {"AAVE/USDT"},
            {},
            "paper",
        )
    finally:
        await close_db()

    assert prices == {"AAVE/USDT": 101.5}
    assert tickers["AAVE/USDT"]["price"] == 101.5


@pytest.mark.asyncio
async def test_local_ai_tools_status_uses_client_when_dashboard_is_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLocalAIToolsClient:
        async def status(self) -> dict:
            return {
                "available": True,
                "status": "ok",
                "models": {"profit": "trained"},
                "completed_shadow_sample_count": 123,
            }

    monkeypatch.setattr(dashboard, "_trading_service", None)
    monkeypatch.setattr(dashboard, "_local_ai_tools_status_client", FakeLocalAIToolsClient())

    result = await dashboard.get_local_ai_tools_status()

    assert result["available"] is True
    assert result["models"]["profit"] == "trained"
    assert result["completed_shadow_sample_count"] == 123


@pytest.mark.asyncio
async def test_ml_signal_status_uses_local_service_when_dashboard_is_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeMLSignalService:
        def status(self) -> dict:
            return {
                "available": True,
                "status": "learning_only",
                "sample_count": 42,
                "auto_train_enabled": True,
            }

        async def completed_shadow_sample_count(self) -> int:
            return 321

    monkeypatch.setattr(dashboard, "_trading_service", None)
    monkeypatch.setattr(dashboard, "_ml_signal_status_service", FakeMLSignalService())

    result = await dashboard.get_ml_signal_status()

    assert result["available"] is True
    assert result["status"] == "learning_only"
    assert result["auto_train_enabled"] is True
    assert result["training_shadow_sample_count"] == 42
    assert result["completed_shadow_sample_count"] == 321
    assert result["new_shadow_sample_count"] == 279
