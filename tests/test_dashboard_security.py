from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from config.settings import TradingMode, settings
from core.model_runtime import HIGH_RISK_REVIEW_TOKEN_CAP, HIGH_RISK_REVIEW_TOKEN_FLOOR
from web_dashboard.api import settings_api as settings_api_module
from web_dashboard.api.security import require_destructive_dashboard_confirmation
from web_dashboard.app import create_app


class _FakeHumanMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeOKXExecutor:
    error_text = ""
    raise_on_balance = False
    raise_on_initialize = False
    created_kwargs: list[dict[str, Any]] = []

    def __init__(self, mode: str = "paper", **kwargs: Any) -> None:
        self.mode = mode
        self.kwargs = kwargs
        self.created_kwargs.append(kwargs)

    async def initialize(self) -> None:
        if self.raise_on_initialize:
            raise RuntimeError(self.error_text)
        return None

    async def get_balance_snapshot(self, currency: str) -> dict[str, Any]:
        if self.raise_on_balance:
            raise RuntimeError(self.error_text)
        return {"error": self.error_text}

    async def shutdown(self) -> None:
        return None


class _FakeTradingService:
    def __init__(self) -> None:
        self._okx_paper = None
        self._okx_live = None


class _SuccessfulBalanceExecutor:
    created = 0

    def __init__(self, mode: str = "paper", **kwargs: Any) -> None:
        type(self).created += 1
        self.mode = mode
        self.kwargs = kwargs

    async def initialize(self) -> None:
        return None

    async def get_balance_snapshot(self, currency: str) -> dict[str, Any]:
        return {
            "free": 11.0,
            "used": 2.0,
            "total": 13.0,
            "cash": 13.0,
            "equity": 14.0,
            "allocatable": 14.0,
        }

    async def shutdown(self) -> None:
        return None


def _request_from(host: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "DELETE",
            "path": "/api/decisions",
            "headers": [],
            "client": (host, 12345),
            "server": ("testserver", 80),
            "scheme": "http",
        }
    )


@pytest.mark.asyncio
async def test_destructive_dashboard_action_requires_explicit_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "dashboard_admin_api_key", "")

    with pytest.raises(HTTPException) as exc_info:
        await require_destructive_dashboard_confirmation(
            _request_from("127.0.0.1"),
            x_dashboard_confirm=None,
        )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_destructive_dashboard_action_allows_local_confirmed_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "dashboard_admin_api_key", "")

    await require_destructive_dashboard_confirmation(
        _request_from("127.0.0.1"),
        x_dashboard_confirm="delete-records",
    )


@pytest.mark.asyncio
async def test_destructive_dashboard_action_rejects_remote_without_admin_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "dashboard_admin_api_key", "")

    with pytest.raises(HTTPException) as exc_info:
        await require_destructive_dashboard_confirmation(
            _request_from("203.0.113.9"),
            x_dashboard_confirm="delete-records",
        )

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_destructive_dashboard_action_allows_matching_admin_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured = "unit-" + "dashboard-admin-token"
    monkeypatch.setattr(settings, "dashboard_admin_" + "api_key", configured)

    await require_destructive_dashboard_confirmation(
        _request_from("203.0.113.9"),
        x_dashboard_confirm="delete-records",
        authorization=f"Bearer {configured}",
    )


@pytest.mark.asyncio
async def test_dashboard_middleware_rejects_remote_read_without_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "dashboard_admin_api_key", "")
    app = create_app()
    transport = httpx.ASGITransport(app=app, client=("203.0.113.9", 12345))

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/control/state")

    assert response.status_code == 401
    assert response.json()["detail"] == "Dashboard login required."


@pytest.mark.asyncio
async def test_dashboard_auth_status_accepts_admin_key_for_read_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured = "unit-" + "dashboard-read-token"
    monkeypatch.setattr(settings, "dashboard_admin_api_key", configured)
    monkeypatch.setattr(settings, "dashboard_auth_enabled", True)
    monkeypatch.setattr(settings, "dashboard_auth_username", "admin")
    monkeypatch.setattr(settings, "dashboard_host", "0.0.0.0")  # noqa: S104
    app = create_app()
    transport = httpx.ASGITransport(app=app, client=("203.0.113.9", 12345))

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get(
            "/api/auth/status",
            headers={"X-Dashboard-Admin-Key": configured},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["authenticated"] is True
    assert payload["auth_method"] == "admin_key"
    assert payload["username"] == "admin"


@pytest.mark.asyncio
async def test_dashboard_write_middleware_rejects_remote_write_without_admin_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "dashboard_admin_api_key", "")
    app = create_app()
    transport = httpx.ASGITransport(app=app, client=("203.0.113.9", 12345))

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/api/control/pause")

    assert response.status_code == 401
    assert response.json()["detail"] == "Dashboard login required."


@pytest.mark.asyncio
async def test_dashboard_write_middleware_rejects_removed_manual_scan_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured = "unit-" + "dashboard-write-token"
    monkeypatch.setattr(settings, "dashboard_admin_" + "api_key", configured)
    app = create_app()
    transport = httpx.ASGITransport(app=app, client=("203.0.113.9", 12345))

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/control/scan-mode",
            headers={"Authorization": f"Bearer {configured}"},
            json={"mode": "manual"},
        )

    assert response.status_code == 400
    assert "自动模式" in response.json()["detail"]


@pytest.mark.asyncio
async def test_dashboard_settings_rejects_credentialed_service_url_without_persisting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_updates: list[dict[str, Any]] = []
    monkeypatch.setattr(settings, "dashboard_admin_api_key", "")
    monkeypatch.setattr(settings, "local_ai_tools_api_base", "http://127.0.0.1:8001")

    def capture_update_env_file(self: object, updates: dict[str, Any]) -> None:
        captured_updates.append(updates)

    monkeypatch.setattr(settings.__class__, "update_env_file", capture_update_env_file)
    app = create_app()
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/settings/thresholds",
            json={
                "local_ai_tools_api_base": "http://user:password@127.0.0.1:8001",
            },
        )

    assert response.status_code == 400
    assert "must not include credentials" in response.json()["detail"]
    assert captured_updates == []
    assert settings.local_ai_tools_api_base == "http://127.0.0.1:8001"


@pytest.mark.asyncio
async def test_dashboard_settings_exposes_high_risk_review_token_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "dashboard_admin_api_key", "")
    app = create_app()
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/settings/thresholds")

    assert response.status_code == 200
    data = response.json()
    assert data["high_risk_review_token_floor"] == HIGH_RISK_REVIEW_TOKEN_FLOOR
    assert data["high_risk_review_token_cap"] == HIGH_RISK_REVIEW_TOKEN_CAP


@pytest.mark.asyncio
async def test_dashboard_settings_threshold_catalog_governs_manual_auto_and_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "dashboard_admin_api_key", "")
    app = create_app()
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/settings/threshold-catalog")

    assert response.status_code == 200
    data = response.json()
    assert data["policy"]["hard_risk_auto_relax"] is False
    assert data["policy"]["auto_tunable_not_rendered_as_manual_inputs"] is True
    assert data["policy"]["removed_fake_thresholds"] is True

    manual_keys = {item["key"] for item in data["manual_editable"]}
    auto_keys = {item["key"] for item in data["auto_tunable"]}
    hard_keys = {item["key"] for item in data["manual_hard_guards"]}
    removed_keys = {item["key"] for item in data["removed_or_deprecated"]}

    assert "decision_interval_seconds" in manual_keys
    assert "confidence_threshold" in manual_keys
    assert "min_entry_volume_ratio" not in manual_keys
    assert "min_entry_volume_ratio" in auto_keys
    assert "min_entry_adx" in auto_keys
    assert "max_leverage" in hard_keys
    assert "max_daily_loss_pct" in hard_keys
    assert "max_auto_trades_per_round" in removed_keys
    assert "daily_profit_target_usdt_cny" in removed_keys
    assert "fee.estimated_taker_fee_pct" in removed_keys
    assert "entry_opportunity_gate.selected_side_positive_net_hard_gate" in removed_keys

    confidence_item = next(
        item for item in data["manual_editable"] if item["key"] == "confidence_threshold"
    )
    assert (
        confidence_item["effective"] >= data["risk_references"]["min_entry_confidence_after_fees"]
    )
    assert "不会被手动调低" in confidence_item["effect"]


@pytest.mark.asyncio
async def test_dashboard_settings_updates_manual_hard_risk_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_updates: list[dict[str, Any]] = []
    monkeypatch.setattr(settings, "dashboard_admin_api_key", "")
    monkeypatch.setattr(settings, "max_position_pct", 0.25)
    monkeypatch.setattr(settings, "max_leverage", 20.0)
    monkeypatch.setattr(settings, "max_daily_loss_pct", 0.05)
    monkeypatch.setattr(settings, "hard_stop_loss_pct", 0.05)
    monkeypatch.setattr(settings, "max_open_positions_per_model", 20)
    monkeypatch.setattr(settings, "max_same_symbol_positions_per_side", 2)

    def capture_update_env_file(self: object, updates: dict[str, Any]) -> None:
        captured_updates.append(updates)

    monkeypatch.setattr(settings.__class__, "update_env_file", capture_update_env_file)
    app = create_app()
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/settings/thresholds",
            json={
                "max_position_pct": 0.12,
                "max_leverage": 7.5,
                "max_daily_loss_pct": 0.03,
                "hard_stop_loss_pct": 0.04,
                "max_open_positions_per_model": 12,
                "max_same_symbol_positions_per_side": 3,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["max_position_pct"] == 0.12
    assert data["max_leverage"] == 7.5
    assert data["max_daily_loss_pct"] == 0.03
    assert data["hard_stop_loss_pct"] == 0.04
    assert data["max_open_positions_per_model"] == 12
    assert data["max_same_symbol_positions_per_side"] == 3
    assert settings.max_position_pct == 0.12
    assert settings.max_leverage == 7.5
    assert settings.max_daily_loss_pct == 0.03
    assert settings.hard_stop_loss_pct == 0.04
    assert settings.max_open_positions_per_model == 12
    assert settings.max_same_symbol_positions_per_side == 3
    assert captured_updates
    assert captured_updates[-1]["MAX_POSITION_PCT"] == "0.12"
    assert captured_updates[-1]["MAX_LEVERAGE"] == "7.5"
    assert captured_updates[-1]["MAX_DAILY_LOSS_PCT"] == "0.03"
    assert captured_updates[-1]["HARD_STOP_LOSS_PCT"] == "0.04"
    assert captured_updates[-1]["MAX_OPEN_POSITIONS_PER_MODEL"] == "12"
    assert captured_updates[-1]["MAX_SAME_SYMBOL_POSITIONS_PER_SIDE"] == "3"


@pytest.mark.asyncio
async def test_execution_account_update_ignores_legacy_allocated_balance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_updates: list[dict[str, Any]] = []
    original_balances = dict(settings.execution_account_balances)

    async def fake_status(mode: str) -> dict[str, Any]:
        return settings.get_execution_account_config(mode)

    def capture_update_env_file(self: object, updates: dict[str, Any]) -> None:
        captured_updates.append(updates)

    monkeypatch.setattr(settings, "execution_account_balances", original_balances)
    monkeypatch.setattr(settings.__class__, "update_env_file", capture_update_env_file)
    monkeypatch.setattr(settings_api_module, "_execution_account_status", fake_status)

    response = await settings_api_module.update_execution_account_settings(
        settings_api_module.ExecutionAccountRequest(
            mode="paper",
            allocated_balance=321.5,
            max_loss_pct=0.25,
        )
    )

    assert response["status"] == "ok"
    assert settings.execution_account_balances == original_balances
    assert captured_updates
    assert "EXECUTION_ACCOUNT_BALANCES" not in captured_updates[-1]
    persisted = json.loads(captured_updates[-1]["EXECUTION_ACCOUNT_MAX_LOSS_PCT"])
    assert persisted["paper"] == 0.25


@pytest.mark.asyncio
async def test_dashboard_settings_rejects_high_risk_review_tokens_above_runtime_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_updates: list[dict[str, Any]] = []
    original_tokens = 420
    monkeypatch.setattr(settings, "dashboard_admin_api_key", "")
    monkeypatch.setattr(settings, "high_risk_review_max_tokens", original_tokens)

    def capture_update_env_file(self: object, updates: dict[str, Any]) -> None:
        captured_updates.append(updates)

    monkeypatch.setattr(settings.__class__, "update_env_file", capture_update_env_file)
    app = create_app()
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/settings/thresholds",
            json={"high_risk_review_max_tokens": HIGH_RISK_REVIEW_TOKEN_CAP + 1},
        )

    assert response.status_code == 400
    assert (
        f"between {HIGH_RISK_REVIEW_TOKEN_FLOOR} and {HIGH_RISK_REVIEW_TOKEN_CAP}"
        in response.json()["detail"]
    )
    assert captured_updates == []
    assert settings.high_risk_review_max_tokens == original_tokens


@pytest.mark.asyncio
async def test_dashboard_ai_model_api_base_is_normalized_before_persisting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_updates: list[dict[str, Any]] = []

    async def noop_sync() -> None:
        return None

    monkeypatch.setattr(settings, "dashboard_admin_api_key", "")
    monkeypatch.setattr(settings, "ai_models", [])

    def capture_update_env_file(self: object, updates: dict[str, Any]) -> None:
        captured_updates.append(updates)

    monkeypatch.setattr(settings.__class__, "update_env_file", capture_update_env_file)
    monkeypatch.setattr(settings_api_module, "_sync_models_to_running_services", noop_sync)
    app = create_app()
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.put(
            "/api/settings/ai-models/trend_expert",
            json={
                "name": "trend_expert",
                "api_base": " https://model.example.invalid/v1/ ",
                "api_key": "",
                "model": "qwen3-32b-trade",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["model"]["api_base"] == "https://model.example.invalid/v1"
    assert captured_updates
    assert "https://model.example.invalid/v1" in captured_updates[-1]["AI_MODELS"]


@pytest.mark.asyncio
async def test_dashboard_ai_model_connection_error_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaked_value = "abcdefghijklmnopqrstuvwxyz123456"

    class FakeChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        async def ainvoke(self, messages: list[Any]) -> object:
            raise RuntimeError(f"Authorization: Bearer {leaked_value} is invalid")

    monkeypatch.setattr(settings, "dashboard_admin_api_key", "")
    monkeypatch.setitem(
        sys.modules,
        "langchain_core.messages",
        SimpleNamespace(HumanMessage=_FakeHumanMessage),
    )
    monkeypatch.setitem(
        sys.modules,
        "langchain_openai",
        SimpleNamespace(ChatOpenAI=FakeChatOpenAI),
    )
    app = create_app()
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/settings/ai-models/test",
            json={
                "api_base": "https://model.example.invalid/v1",
                "api_key": "test-model-key",
                "model": "qwen3-32b-trade",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert leaked_value not in body["error"]
    assert "Authorization: ***" in body["error"]


@pytest.mark.asyncio
async def test_dashboard_ai_model_connection_success_response_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaked_value = "abcdefghijklmnopqrstuvwxyz123456"
    captured_kwargs: dict[str, Any] = {}
    captured_prompts: list[str] = []

    class FakeChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

        async def ainvoke(self, messages: list[Any]) -> object:
            captured_prompts.extend(str(getattr(message, "content", "")) for message in messages)
            return SimpleNamespace(content=f"Authorization: Bearer {leaked_value} accepted")

    monkeypatch.setattr(settings, "dashboard_admin_api_key", "")
    monkeypatch.setitem(
        sys.modules,
        "langchain_core.messages",
        SimpleNamespace(HumanMessage=_FakeHumanMessage),
    )
    monkeypatch.setitem(
        sys.modules,
        "langchain_openai",
        SimpleNamespace(ChatOpenAI=FakeChatOpenAI),
    )
    app = create_app()
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/settings/ai-models/test",
            json={
                "api_base": "https://model.example.invalid/v1/",
                "api_key": "test-model-key",
                "model": "Qwen/Qwen3-32B-AWQ",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert leaked_value not in body["message"]
    assert "Authorization: ***" in body["message"]
    assert captured_kwargs["base_url"] == "https://model.example.invalid/v1"
    assert captured_kwargs["max_tokens"] == 10
    assert captured_kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert any("/no_think" in prompt for prompt in captured_prompts)


@pytest.mark.asyncio
async def test_dashboard_okx_connection_error_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import executor.okx_executor as okx_executor_module

    leaked_value = "abcdefghijklmnopqrstuvwxyz123456"
    _FakeOKXExecutor.error_text = f"Authorization: Bearer {leaked_value} is invalid"
    _FakeOKXExecutor.raise_on_balance = False
    _FakeOKXExecutor.created_kwargs.clear()
    settings_api_module._OKX_BALANCE_CACHE.clear()
    monkeypatch.setattr(settings, "dashboard_admin_api_key", "")
    monkeypatch.setattr(settings, "okx_paper_api_key", "paper-key")
    monkeypatch.setattr(settings, "okx_paper_api_secret", "paper-secret")
    monkeypatch.setattr(settings, "okx_paper_passphrase", "paper-pass")
    monkeypatch.setattr(okx_executor_module, "OKXExecutor", _FakeOKXExecutor)
    app = create_app()
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/api/settings/okx/test", json={"mode": "paper"})

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert leaked_value not in body["error"]
    assert "Authorization: ***" in body["error"]
    assert _FakeOKXExecutor.created_kwargs[-1]["load_markets_on_initialize"] is False


@pytest.mark.asyncio
async def test_dashboard_okx_update_logs_redacted_executor_init_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import executor.okx_executor as okx_executor_module

    leaked_value = "abcdefghijklmnopqrstuvwxyz123456"
    warning_events: list[dict[str, Any]] = []
    trading_service = _FakeTradingService()
    _FakeOKXExecutor.created_kwargs.clear()

    class FakeLogger:
        def warning(self, event: str, **fields: Any) -> None:
            warning_events.append({"event": event, **fields})

    monkeypatch.setattr(
        _FakeOKXExecutor,
        "error_text",
        f"Authorization: Bearer {leaked_value} failed",
    )
    monkeypatch.setattr(_FakeOKXExecutor, "raise_on_initialize", True)
    monkeypatch.setattr(settings, "dashboard_admin_api_key", "")
    monkeypatch.setattr(settings, "trading_mode", TradingMode.PAPER)
    monkeypatch.setattr(settings_api_module._dash, "_trading_service", trading_service)
    monkeypatch.setattr(settings_api_module._dash, "_data_service", None)
    monkeypatch.setattr(settings_api_module, "logger", FakeLogger())
    monkeypatch.setattr(okx_executor_module, "OKXExecutor", _FakeOKXExecutor)

    response = await settings_api_module.update_okx_settings(
        settings_api_module.OKXSettingsRequest(mode="paper")
    )

    assert response["status"] == "ok"
    assert trading_service._okx_paper is None
    assert _FakeOKXExecutor.created_kwargs[-1]["load_markets_on_initialize"] is False
    assert warning_events == [
        {
            "event": "failed to initialize OKX executor after credential update",
            "mode": "paper",
            "error": "Authorization: *** failed",
        }
    ]
    assert leaked_value not in str(warning_events)


@pytest.mark.asyncio
async def test_dashboard_okx_balance_error_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import executor.okx_executor as okx_executor_module

    leaked_value = "abcdefghijklmnopqrstuvwxyz123456"
    _FakeOKXExecutor.error_text = f"Authorization: Bearer {leaked_value} is invalid"
    _FakeOKXExecutor.raise_on_balance = True
    _FakeOKXExecutor.created_kwargs.clear()
    settings_api_module._OKX_BALANCE_CACHE.clear()
    monkeypatch.setattr(settings, "dashboard_admin_api_key", "")
    monkeypatch.setattr(settings, "okx_paper_api_key", "paper-key")
    monkeypatch.setattr(settings, "okx_paper_api_secret", "paper-secret")
    monkeypatch.setattr(settings, "okx_paper_passphrase", "paper-pass")
    monkeypatch.setattr(settings, "okx_live_api_key", "live-key")
    monkeypatch.setattr(settings, "okx_live_api_secret", "live-secret")
    monkeypatch.setattr(settings, "okx_live_passphrase", "live-pass")
    monkeypatch.setattr(okx_executor_module, "OKXExecutor", _FakeOKXExecutor)
    app = create_app()
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/settings/okx/balance")

    assert response.status_code == 200
    body = response.json()
    assert leaked_value not in body["paper_error"]
    assert leaked_value not in body["live_error"]
    assert "Authorization: ***" in body["paper_error"]
    assert "Authorization: ***" in body["live_error"]


@pytest.mark.asyncio
async def test_settings_okx_snapshot_reuses_dashboard_cache_without_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings_api_module._OKX_BALANCE_CACHE.clear()
    settings_api_module._OKX_BALANCE_ERROR_CACHE.clear()
    monkeypatch.setattr(settings, "okx_paper_api_key", "paper-key")
    monkeypatch.setattr(settings, "okx_paper_api_secret", "paper-secret")
    monkeypatch.setattr(settings, "okx_paper_passphrase", "paper-pass")
    monkeypatch.setattr(
        settings_api_module._dash,
        "_dashboard_okx_balance_cache",
        {
            "paper": (
                settings_api_module.datetime.now(settings_api_module.UTC),
                {
                    "free": 11.0,
                    "used": 2.0,
                    "total": 13.0,
                    "cash": 13.0,
                    "equity": 14.0,
                    "allocatable": 14.0,
                },
            )
        },
    )

    async def fail_dashboard_fetch(mode: str) -> dict[str, Any]:
        raise AssertionError("settings must only read dashboard cache, not fetch through it")

    monkeypatch.setattr(
        settings_api_module._dash,
        "_get_dashboard_okx_account_snapshot",
        fail_dashboard_fetch,
    )

    snapshot = await settings_api_module._get_okx_usdt_snapshot("paper")

    assert snapshot["allocatable_balance"] == 14.0
    assert snapshot["balance_error"] is None


@pytest.mark.asyncio
async def test_settings_okx_snapshot_timeout_returns_chinese_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import executor.okx_executor as okx_executor_module

    settings_api_module._OKX_BALANCE_CACHE.clear()
    settings_api_module._OKX_BALANCE_ERROR_CACHE.clear()
    _FakeOKXExecutor.raise_on_balance = True
    _FakeOKXExecutor.error_text = "TimeoutError"
    monkeypatch.setattr(settings, "okx_paper_api_key", "paper-key")
    monkeypatch.setattr(settings, "okx_paper_api_secret", "paper-secret")
    monkeypatch.setattr(settings, "okx_paper_passphrase", "paper-pass")
    monkeypatch.setattr(settings_api_module._dash, "_dashboard_okx_balance_cache", {})
    monkeypatch.setattr(okx_executor_module, "OKXExecutor", _FakeOKXExecutor)

    snapshot = await settings_api_module._get_okx_usdt_snapshot("paper", force=True)

    assert snapshot["balance_error"] == "OKX 余额响应超时，已优先返回缓存数据"
    assert snapshot["error_cached"] is True


@pytest.mark.asyncio
async def test_settings_okx_snapshot_uses_trading_service_cache_before_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import executor.okx_executor as okx_executor_module

    class FailingExecutor:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("settings snapshot should not create a new OKX executor")

    settings_api_module._OKX_BALANCE_CACHE.clear()
    settings_api_module._OKX_BALANCE_ERROR_CACHE.clear()
    monkeypatch.setattr(settings, "okx_paper_api_key", "paper-key")
    monkeypatch.setattr(settings, "okx_paper_api_secret", "paper-secret")
    monkeypatch.setattr(settings, "okx_paper_passphrase", "paper-pass")
    monkeypatch.setattr(settings_api_module._dash, "_dashboard_okx_balance_cache", {})
    monkeypatch.setattr(
        settings_api_module._dash,
        "_trading_service",
        SimpleNamespace(
            peek_okx_balance_snapshot_for_mode=lambda mode, allow_stale=True: {
                "free": 21.0,
                "used": 4.0,
                "total": 25.0,
                "cash": 25.0,
                "equity": 26.0,
                "allocatable": 26.0,
                "error": "OKX balance snapshot request timed out",
                "stale": True,
            }
        ),
    )
    monkeypatch.setattr(okx_executor_module, "OKXExecutor", FailingExecutor)

    snapshot = await settings_api_module._get_okx_usdt_snapshot("paper")

    assert snapshot["allocatable_balance"] == 26.0
    assert snapshot["available_balance"] == 21.0
    assert snapshot["balance_error"] is None
    assert snapshot["stale"] is True


@pytest.mark.asyncio
async def test_dashboard_login_session_allows_remote_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from web_dashboard.api.security import hash_dashboard_password

    monkeypatch.setattr(settings, "dashboard_admin_api_key", "")
    monkeypatch.setattr(settings, "dashboard_auth_enabled", True)
    monkeypatch.setattr(settings, "dashboard_auth_username", "admin")
    monkeypatch.setattr(
        settings, "dashboard_auth_password_hash", hash_dashboard_password("unit-pass")
    )
    monkeypatch.setattr(settings, "dashboard_session_secret", "unit-session-secret")
    app = create_app()
    transport = httpx.ASGITransport(app=app, client=("203.0.113.9", 12345))

    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver", follow_redirects=False
    ) as client:
        login = await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "unit-pass"},
        )
        response = await client.get("/api/control/state")

    assert login.status_code == 200
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_dashboard_public_bind_requires_login_even_from_local_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "dashboard_admin_api_key", "")
    monkeypatch.setattr(settings, "dashboard_host", "0.0.0.0")  # noqa: S104
    app = create_app()
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/control/state")

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_dashboard_mode_switch_requires_configured_okx_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured = "unit-" + "dashboard-write-token"
    monkeypatch.setattr(settings, "dashboard_admin_api_key", configured)
    for field in (
        "okx_paper_api_key",
        "okx_paper_api_secret",
        "okx_paper_passphrase",
        "okx_live_api_key",
        "okx_live_api_secret",
        "okx_live_passphrase",
        "okx_api_key",
        "okx_api_secret",
        "okx_passphrase",
    ):
        monkeypatch.setattr(settings, field, "")
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
    assert "API Key" in detail["missing_fields"]


@pytest.mark.asyncio
async def test_dashboard_manual_open_trade_endpoint_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured = "unit-" + "dashboard-write-token"
    monkeypatch.setattr(settings, "dashboard_admin_api_key", configured)
    app = create_app()
    transport = httpx.ASGITransport(app=app, client=("203.0.113.9", 12345))

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/trade/manual",
            headers={"Authorization": f"Bearer {configured}"},
            json={"symbol": "BTC/USDT"},
        )

    assert response.status_code == 410
    assert "自动模式" in response.json()["detail"]
