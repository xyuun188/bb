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
    assert result["profit_prediction"]["status"] == "returned"
    assert result["profit_prediction"]["path"] == "/profit/predict"
    assert result["profit_prediction"]["duration_sec"] > 0
    assert result["time_series_prediction"]["side"] == "long"


@pytest.mark.asyncio
async def test_local_ai_tools_enrich_uses_configured_timeout_without_three_second_cap(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "local_ai_tools_timeout_seconds", 8.0)
    client = LocalAIToolsClient()
    timeouts: list[float | None] = []

    async def succeed(
        path: str,
        payload: dict[str, Any],
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        timeouts.append(request_timeout)
        return {"available": True, "path": path, "best_side": "long"}

    monkeypatch.setattr(client, "_post", succeed)

    result = await client.enrich_with_context({"symbol": "BTC/USDT"})

    assert result["status"] == "completed"
    assert timeouts == [8.0, 8.0, 8.0]
    assert result["profit_prediction"]["duration_sec"] > 0
    assert result["time_series_prediction"]["duration_sec"] > 0
    assert result["sentiment_analysis"]["duration_sec"] > 0


@pytest.mark.asyncio
async def test_local_ai_tools_readtimeout_does_not_open_circuit(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "local_ai_tools_timeout_seconds", 8.0)
    monkeypatch.setattr(settings, "local_ai_tools_circuit_breaker_failures", 2)
    client = LocalAIToolsClient()
    calls: list[str] = []

    async def timeout(
        path: str,
        payload: dict[str, Any],
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        calls.append(path)
        raise RuntimeError("local AI tools request could not reach the service: ReadTimeout")

    monkeypatch.setattr(client, "_post", timeout)

    first = await client.enrich_with_context({"symbol": "BTC/USDT"})
    second = await client.enrich_with_context({"symbol": "ETH/USDT"})
    third = await client.enrich_with_context({"symbol": "SOL/USDT"})

    assert first["status"] == "unavailable"
    assert second["status"] == "unavailable"
    assert third["status"] == "unavailable"
    assert third["profit_prediction"]["status"] == "error"
    assert third["profit_prediction"]["path"] == "/profit/predict"
    assert third["profit_prediction"]["duration_sec"] > 0
    assert "circuit_open_until" not in third
    assert len(calls) == 9


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


def test_local_ai_tools_normalizes_wrapped_prediction_payloads(
    local_tools_settings: None,
) -> None:
    client = LocalAIToolsClient()

    profit = client._normalize_signal(
        "profit_prediction",
        {
            "ok": True,
            "data": {
                "prediction": {
                    "predicted_side": "short",
                    "expected_short_return_pct": 0.42,
                    "expected_long_return_pct": -0.18,
                }
            },
        },
    )
    timeseries = client._normalize_signal(
        "time_series_prediction",
        {"status": "ok", "result": {"forecast_direction": "up", "expected_move_pct": 0.16}},
    )
    sentiment = client._normalize_signal(
        "sentiment_analysis",
        {"available": True, "payload": {"sentiment": "bearish", "sentiment_score": -0.31}},
    )

    assert profit["available"] is True
    assert profit["best_side"] == "short"
    assert profit["expected_return_pct"] == 0.42
    assert timeseries["side"] == "long"
    assert timeseries["expected_return_pct"] == 0.16
    assert sentiment["side"] == "short"
    assert sentiment["available"] is True


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

    async def fail(path: str, request_timeout: float | None = None) -> dict[str, Any]:
        raise RuntimeError(f"Authorization: Bearer {leaked_value} failed")

    monkeypatch.setattr(client, "_get", fail)

    result = await client.status()

    assert result["status"] == "error"
    assert leaked_value not in str(result)
    assert result["error"] == "Authorization: *** failed"
    assert client._last_failure == "Authorization: *** failed"


@pytest.mark.asyncio
async def test_local_ai_tools_status_uses_child_endpoint_health_when_bundle_missing(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = LocalAIToolsClient()
    get_calls: list[str] = []
    post_calls: list[str] = []

    async def get_status(path: str, request_timeout: float | None = None) -> dict[str, Any]:
        get_calls.append(path)
        assert request_timeout == 0.5
        return {"available": False, "message": "No trained local quant bundle found"}

    async def post_probe(
        path: str,
        payload: dict[str, Any],
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        post_calls.append(path)
        return {"available": True, "path": path, "best_side": "long"}

    monkeypatch.setattr(client, "_get", get_status)
    monkeypatch.setattr(client, "_post", post_probe)

    result = await client.status()

    assert get_calls == ["/models/status"]
    assert set(post_calls) == {
        "/profit/predict",
        "/timeseries/deep/predict",
        "/sentiment/deep/analyze",
        "/exit/advise",
    }
    assert result["available"] is True
    assert result["model_bundle_available"] is False
    assert result["service_available"] is True
    assert result["status"] == "heuristic_fallback_available"
    assert result["child_endpoints"]["profit_prediction"]["available"] is True


@pytest.mark.asyncio
async def test_local_ai_tools_status_uses_child_endpoint_health_when_status_fails(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = LocalAIToolsClient()

    async def fail_status(path: str, request_timeout: float | None = None) -> dict[str, Any]:
        assert request_timeout == 0.5
        raise RuntimeError("models status endpoint unavailable")

    async def post_probe(
        path: str,
        payload: dict[str, Any],
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        if path == "/profit/predict":
            return {"available": True, "best_side": "long"}
        raise RuntimeError(f"{path} unavailable")

    monkeypatch.setattr(client, "_get", fail_status)
    monkeypatch.setattr(client, "_post", post_probe)

    result = await client.status()

    assert result["available"] is True
    assert result["service_available"] is True
    assert result["model_bundle_available"] is False
    assert result["status"] == "heuristic_fallback_available"
    assert result["status_error"] == "models status endpoint unavailable"
    assert result["child_endpoints"]["profit_prediction"]["available"] is True
    assert result["failure_count"] == 0


@pytest.mark.asyncio
async def test_local_ai_tools_status_uses_short_cache(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = LocalAIToolsClient()
    get_calls: list[str] = []
    post_calls: list[str] = []

    async def get_status(path: str, request_timeout: float | None = None) -> dict[str, Any]:
        get_calls.append(path)
        assert request_timeout == 0.5
        return {"available": False, "message": "No trained local quant bundle found"}

    async def post_probe(
        path: str,
        payload: dict[str, Any],
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        post_calls.append(path)
        return {"available": True, "path": path, "best_side": "long"}

    monkeypatch.setattr(client, "_get", get_status)
    monkeypatch.setattr(client, "_post", post_probe)

    first = await client.status()
    first["child_endpoints"]["profit_prediction"]["available"] = False
    second = await client.status()

    assert len(get_calls) == 1
    assert set(post_calls) == {
        "/profit/predict",
        "/timeseries/deep/predict",
        "/sentiment/deep/analyze",
        "/exit/advise",
    }
    assert second["status_cache"]["hit"] is True
    assert second["child_endpoints"]["profit_prediction"]["available"] is True


def test_local_ai_tools_auth_headers_close_connections(
    local_tools_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "local_ai_tools_api_key", "  local-secret-token  ")
    headers = LocalAIToolsClient()._auth_headers()

    assert headers == {
        "Authorization": "Bearer local-secret-token",
        "Connection": "close",
    }


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
