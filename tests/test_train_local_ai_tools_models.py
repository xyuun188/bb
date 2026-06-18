from __future__ import annotations

import httpx
import pytest

from scripts.train_local_ai_tools_models import (
    _build_auth_headers,
    _merge_trade_samples,
    _normalize_base_url,
    _post_training_payload,
)


def test_local_ai_tools_training_headers_use_bearer_token() -> None:
    assert _build_auth_headers("  local-secret-token  ") == {
        "Authorization": "Bearer local-secret-token"
    }
    assert _build_auth_headers("") == {}


def test_local_ai_tools_training_base_url_validation() -> None:
    assert _normalize_base_url(" http://127.0.0.1:8001/ ") == "http://127.0.0.1:8001"

    with pytest.raises(RuntimeError, match="absolute http"):
        _normalize_base_url("127.0.0.1:8001")

    with pytest.raises(RuntimeError, match="credentials"):
        _normalize_base_url("http://user:password@127.0.0.1:8001")

    with pytest.raises(RuntimeError, match="LOCAL_AI_TOOLS_API_BASE is empty"):
        _normalize_base_url("")


def test_local_ai_tools_training_merges_trade_samples_without_duplicate_positions() -> None:
    reflection_samples = [
        {"source": "trade_reflection", "id": 11, "position_id": 7, "realized_pnl": -1.2},
        {"source": "trade_reflection", "id": 12, "position_id": 8, "realized_pnl": 2.4},
    ]
    closed_position_samples = [
        {"source": "closed_position", "id": 7, "position_id": 7, "realized_pnl": -1.2},
        {"source": "closed_position", "id": 9, "position_id": 9, "realized_pnl": 0.4},
    ]

    merged = _merge_trade_samples(reflection_samples, closed_position_samples)

    assert [item["source"] for item in merged] == [
        "trade_reflection",
        "trade_reflection",
        "closed_position",
    ]
    assert [item["position_id"] for item in merged] == [7, 8, 9]


@pytest.mark.asyncio
async def test_local_ai_tools_training_post_sends_auth_header() -> None:
    captured: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("authorization")
        return httpx.Response(200, json={"trained": True}, request=request)

    result = await _post_training_payload(
        "https://local-ai-tools.test/",
        {"shadow_samples": []},
        request_timeout=3.0,
        auth_token="test-local-tools-key",
        transport=httpx.MockTransport(handler),
    )

    assert result == {"trained": True}
    assert captured["url"] == "https://local-ai-tools.test/train"
    assert captured["authorization"] == "Bearer test-local-tools-key"


@pytest.mark.asyncio
async def test_local_ai_tools_training_auth_failure_is_actionable_and_redacted() -> None:
    leaked_token = "abcdefghijklmnopqrstuvwxyz123456"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"detail": f"Authorization: Bearer {leaked_token} is invalid"},
            request=request,
        )

    with pytest.raises(RuntimeError) as exc_info:
        await _post_training_payload(
            "http://127.0.0.1:8001",
            {"shadow_samples": []},
            request_timeout=3.0,
            auth_token=leaked_token,
            transport=httpx.MockTransport(handler),
        )

    message = str(exc_info.value)
    assert "HTTP 401" in message
    assert "LOCAL_AI_TOOLS_API_KEY" in message
    assert "/data/trade_ai/local_ai_tools.env" in message
    assert leaked_token not in message
    assert "Authorization: ***" in message
