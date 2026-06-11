from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest

from config.settings import settings
from services.local_ai_tools_client import LocalAIToolsClient


@pytest.fixture
def local_tools_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "local_ai_tools_enabled", True)
    monkeypatch.setattr(settings, "local_ai_tools_api_base", "http://local-ai-tools.test")
    monkeypatch.setattr(settings, "local_ai_tools_api_key", "")
    monkeypatch.setattr(settings, "local_ai_tools_timeout_seconds", 0.5)
    monkeypatch.setattr(settings, "local_ai_tools_circuit_breaker_failures", 2)
    monkeypatch.setattr(settings, "local_ai_tools_circuit_breaker_cooldown_seconds", 30.0)


@pytest.mark.asyncio
async def test_local_ai_tools_circuit_breaker_opens_after_total_failures(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = LocalAIToolsClient()
    calls: list[str] = []

    async def fail(
        path: str,
        payload: dict[str, Any],
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        calls.append(path)
        raise RuntimeError("local tools unavailable")

    monkeypatch.setattr(client, "_post", fail)

    first = await client.enrich_with_context({"symbol": "BTC/USDT"})
    second = await client.enrich_with_context({"symbol": "ETH/USDT"})
    third = await client.enrich_with_context({"symbol": "SOL/USDT"})

    assert first["status"] == "unavailable"
    assert first["failure_count"] == 1
    assert second["status"] == "unavailable"
    assert second["failure_count"] == 2
    assert second.get("circuit_open_until")
    assert third["status"] == "circuit_open"
    assert third["available"] is False
    assert len(calls) == 6


@pytest.mark.asyncio
async def test_local_ai_tools_circuit_breaker_recovers_after_cooldown(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = LocalAIToolsClient()
    client._failure_count = 2
    client._circuit_open_until = datetime.now(UTC) - timedelta(seconds=1)

    async def succeed(
        path: str,
        payload: dict[str, Any],
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        return {"available": True, "path": path, "best_side": "long"}

    monkeypatch.setattr(client, "_post", succeed)

    result = await client.enrich_with_context({"symbol": "BTC/USDT"})

    assert result["status"] == "completed"
    assert result["failure_count"] == 0
    assert result["profit_prediction"]["available"] is True
    assert result["time_series_prediction"]["side"] == "long"


def test_local_ai_tools_client_refreshes_runtime_settings(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = LocalAIToolsClient()

    monkeypatch.setattr(settings, "local_ai_tools_timeout_seconds", 1.25)
    monkeypatch.setattr(settings, "local_ai_tools_circuit_breaker_failures", 5)
    monkeypatch.setattr(settings, "local_ai_tools_circuit_breaker_cooldown_seconds", 90.0)

    assert client.enabled() is True
    assert client._timeout == 1.25
    assert client._failure_threshold == 5
    assert client._cooldown_seconds == 90.0


def test_local_ai_tools_client_clamps_runtime_settings(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = LocalAIToolsClient()

    monkeypatch.setattr(settings, "local_ai_tools_timeout_seconds", 999.0)
    monkeypatch.setattr(settings, "local_ai_tools_circuit_breaker_failures", 999)
    monkeypatch.setattr(settings, "local_ai_tools_circuit_breaker_cooldown_seconds", 99999.0)

    assert client.enabled() is True
    assert client._timeout == 15.0
    assert client._failure_threshold == 20
    assert client._cooldown_seconds == 3600.0

    monkeypatch.setattr(settings, "local_ai_tools_timeout_seconds", 0.01)
    monkeypatch.setattr(settings, "local_ai_tools_circuit_breaker_failures", -5)
    monkeypatch.setattr(settings, "local_ai_tools_circuit_breaker_cooldown_seconds", 0.01)

    assert client.enabled() is True
    assert client._timeout == 0.2
    assert client._failure_threshold == 1
    assert client._cooldown_seconds == 0.2


def test_local_ai_tools_exit_advice_uses_clean_chinese_labels(
    local_tools_settings: None,
) -> None:
    client = LocalAIToolsClient()

    hold = client._normalize_signal(
        "exit_advice",
        {"action": "hold", "reason": "no trained exit pressure"},
    )
    reduce = client._normalize_signal(
        "exit_advice",
        {
            "action": "reduce",
            "reason": "profit exists but historical giveback/loss pressure is elevated",
        },
    )
    unknown = client._normalize_signal(
        "exit_advice",
        {
            "recommendation": "unexpected_model_token",
            "note": "no matching open position was supplied",
        },
    )

    assert hold["action_label"] == "继续持有"
    assert hold["reason"] == "平仓建议模型未识别到明确的主动平仓压力，本轮倾向继续持有。"
    assert reduce["action_label"] == "减仓"
    assert reduce["reason"] == "当前已有浮盈，但历史回吐或亏损压力偏高，建议优先保护利润。"
    assert unknown["action_label"] == "继续观察"
    assert unknown["reason"] == "本轮没有传入与该币种匹配的当前持仓，平仓建议模型不参与。"


@pytest.mark.asyncio
async def test_local_ai_tools_train_returns_structured_failure(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = LocalAIToolsClient()

    async def fail(
        path: str,
        payload: dict[str, Any],
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        raise RuntimeError("service refused training request")

    monkeypatch.setattr(client, "_post", fail)

    result = await client.train([{"symbol": "BTC/USDT"}], [{"symbol": "BTC/USDT"}])

    assert result["trained"] is False
    assert result["reason"] == "request_failed"
    assert result["error"] == "service refused training request"
    assert result["failure_count"] == 1


@pytest.mark.asyncio
async def test_local_ai_tools_enrich_failure_fields_are_redacted(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaked_value = "abcdefghijklmnopqrstuvwxyz123456"
    client = LocalAIToolsClient()

    async def fail(
        path: str,
        payload: dict[str, Any],
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        raise RuntimeError(f"Authorization: Bearer {leaked_value} failed")

    monkeypatch.setattr(client, "_post", fail)

    result = await client.enrich_with_context({"symbol": "BTC/USDT"})

    assert result["status"] == "unavailable"
    assert leaked_value not in str(result)
    assert result["errors"]["profit_prediction"] == "Authorization: *** failed"
    assert result["profit_prediction"]["error"] == "Authorization: *** failed"
    assert client._last_failure == (
        "Authorization: *** failed; Authorization: *** failed; " "Authorization: *** failed"
    )


@pytest.mark.asyncio
async def test_local_ai_tools_status_failure_is_redacted(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaked_value = "abcdefghijklmnopqrstuvwxyz123456"
    client = LocalAIToolsClient()

    async def fail(path: str) -> dict[str, Any]:
        raise RuntimeError(f"Authorization: Bearer {leaked_value} failed")

    monkeypatch.setattr(client, "_get", fail)

    result = await client.status()

    assert result["status"] == "error"
    assert leaked_value not in str(result)
    assert result["error"] == "Authorization: *** failed"
    assert client._last_failure == "Authorization: *** failed"


@pytest.mark.asyncio
async def test_local_ai_tools_train_failure_is_redacted(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaked_value = "abcdefghijklmnopqrstuvwxyz123456"
    client = LocalAIToolsClient()

    async def fail(
        path: str,
        payload: dict[str, Any],
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        raise RuntimeError(f"Authorization: Bearer {leaked_value} failed")

    monkeypatch.setattr(client, "_post", fail)

    result = await client.train([{"symbol": "BTC/USDT"}], [{"symbol": "BTC/USDT"}])

    assert result["trained"] is False
    assert result["reason"] == "request_failed"
    assert leaked_value not in str(result)
    assert result["error"] == "Authorization: *** failed"
    assert client._last_failure == "Authorization: *** failed"


def test_local_ai_tools_client_rejects_credentials_in_base_url(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "local_ai_tools_api_base",
        "http://user:password@127.0.0.1:8001",
    )

    with pytest.raises(RuntimeError, match="must not include credentials"):
        LocalAIToolsClient()._api_base()


@pytest.mark.asyncio
async def test_local_ai_tools_public_payload_does_not_leak_credentials_in_base_url(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "local_ai_tools_api_base",
        "http://user:password@127.0.0.1:8001",
    )
    client = LocalAIToolsClient()

    async def succeed(
        path: str,
        payload: dict[str, Any],
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        return {"available": True, "path": path, "best_side": "long"}

    monkeypatch.setattr(client, "_post", succeed)

    result = await client.enrich_with_context({"symbol": "BTC/USDT"})

    assert result["status"] == "completed"
    assert result["api_base"] == "invalid_config"
    assert "password" not in str(result)
    assert "user:password" not in str(result)


def test_local_ai_tools_circuit_payload_does_not_leak_credentials_in_base_url(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "local_ai_tools_api_base",
        "http://user:password@127.0.0.1:8001",
    )
    client = LocalAIToolsClient()
    client._circuit_open_until = datetime.now(UTC) + timedelta(seconds=30)
    client._last_failure = "Authorization: *** failed"

    result = client._circuit_open_payload()

    assert result is not None
    assert result["api_base"] == "invalid_config"
    assert "password" not in str(result)
    assert "user:password" not in str(result)


def test_local_ai_tools_client_auth_failure_is_redacted(local_tools_settings: None) -> None:
    leaked_value = "abcdefghijklmnopqrstuvwxyz123456"
    response = httpx.Response(
        401,
        json={"detail": f"Authorization: Bearer {leaked_value} is invalid"},
        request=httpx.Request("POST", "http://local-ai-tools.test/train"),
    )

    with pytest.raises(RuntimeError) as exc_info:
        LocalAIToolsClient()._parse_response(response, "/train")

    message = str(exc_info.value)
    assert "HTTP 401" in message
    assert "LOCAL_AI_TOOLS_API_KEY" in message
    assert leaked_value not in message
    assert "Authorization: ***" in message
