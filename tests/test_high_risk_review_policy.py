from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from ai_brain.base_model import Action, DecisionOutput
from config.settings import settings
from core.model_runtime import HIGH_RISK_REVIEW_TOKEN_CAP
from services.entry_high_risk_review import EntryHighRiskReviewGatePolicy
from services.high_risk_review_service import HighRiskReviewService


@pytest.fixture
def high_risk_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "high_risk_review_enabled", True)
    monkeypatch.setattr(settings, "high_risk_review_api_base", "https://api.deepseek.com")
    monkeypatch.setattr(settings, "high_risk_review_api_key", "test-review-key")
    monkeypatch.setattr(settings, "high_risk_review_model", "deepseek-reasoner")
    monkeypatch.setattr(settings, "high_risk_review_timeout_seconds", 12.0)
    monkeypatch.setattr(settings, "high_risk_review_max_tokens", 420)
    monkeypatch.setattr(settings, "high_risk_review_circuit_breaker_failures", 1)
    monkeypatch.setattr(settings, "high_risk_review_circuit_breaker_cooldown_seconds", 60.0)


class CapturingReviewer(HighRiskReviewService):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []

    async def call_model(
        self,
        *,
        api_base: str,
        api_key: str,
        model: str,
        messages: list[dict[str, str]],
        use_json_mode: bool,
        max_tokens: int,
        request_timeout: float,
    ) -> tuple[dict[str, Any], str, dict[str, Any]]:
        self.calls.append(
            {
                "api_base": api_base,
                "api_key": api_key,
                "model": model,
                "messages": messages,
                "use_json_mode": use_json_mode,
                "max_tokens": max_tokens,
                "request_timeout": request_timeout,
            }
        )
        return (
            {},
            '{"approved": true, "confidence": 0.91, "reason": "证据尚可，允许小心执行"}',
            {"finish_reason": "stop", "usage": {"completion_tokens": 32}},
        )


class _CapturingAsyncClient:
    requests: list[dict[str, Any]] = []

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout

    async def __aenter__(self) -> _CapturingAsyncClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> httpx.Response:
        self.requests.append(
            {
                "timeout": self.timeout,
                "url": url,
                "headers": headers,
                "json": json,
            }
        )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": '{"approved": true, "confidence": 0.8, "reason": "ok"}'
                        },
                    }
                ],
                "usage": {"completion_tokens": 12},
            },
            request=httpx.Request("POST", url),
        )


class DeadlineReviewer(HighRiskReviewService):
    def __init__(self) -> None:
        super().__init__()
        self.now = 0.0
        self.calls: list[float] = []

    def _monotonic_seconds(self) -> float:
        return self.now

    async def call_model(
        self,
        *,
        api_base: str,
        api_key: str,
        model: str,
        messages: list[dict[str, str]],
        use_json_mode: bool,
        max_tokens: int,
        request_timeout: float,
    ) -> tuple[dict[str, Any], str, dict[str, Any]]:
        self.calls.append(request_timeout)
        self.now = 13.0
        return ({}, "", {"finish_reason": "length"})


@pytest.mark.asyncio
async def test_high_risk_review_service_uses_short_timeout_and_token_cap(
    high_risk_settings: None,
) -> None:
    reviewer = CapturingReviewer()
    request_kwargs = {
        "api_base": "https://api.deepseek.com",
        "api_" + "key": settings.high_risk_review_api_key,
        "model": "deepseek-reasoner",
    }

    result = await reviewer.review_trade(
        {"symbol": "BTC/USDT", "side": "long"},
        **request_kwargs,
    )

    assert result.approved is True
    assert len(reviewer.calls) == 1
    assert reviewer.calls[0]["api_base"] == "https://api.deepseek.com"
    assert reviewer.calls[0]["api_key"] == "test-review-key"
    assert reviewer.calls[0]["request_timeout"] == 12.0
    assert reviewer.calls[0]["max_tokens"] == 420
    assert reviewer.calls[0]["use_json_mode"] is True
    assert reviewer.failure_count == 0


@pytest.mark.asyncio
async def test_high_risk_review_runtime_caps_oversized_setting(
    high_risk_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "high_risk_review_max_tokens", HIGH_RISK_REVIEW_TOKEN_CAP + 400)
    reviewer = CapturingReviewer()

    await reviewer.review_trade(
        {"symbol": "BTC/USDT", "side": "long"},
        api_base="https://api.deepseek.com",
        api_key=settings.high_risk_review_api_key,
        model="deepseek-reasoner",
    )

    assert reviewer.calls[0]["max_tokens"] == HIGH_RISK_REVIEW_TOKEN_CAP


@pytest.mark.asyncio
async def test_high_risk_review_retries_share_total_timeout_budget(
    high_risk_settings: None,
) -> None:
    reviewer = DeadlineReviewer()

    with pytest.raises(TimeoutError, match="exceeded total timeout"):
        await reviewer.review_trade(
            {"symbol": "BTC/USDT", "side": "long"},
            api_base="https://api.deepseek.com",
            api_key=settings.high_risk_review_api_key,
            model="deepseek-reasoner",
        )

    assert len(reviewer.calls) == 1
    assert reviewer.calls[0] == 12.0


@pytest.mark.asyncio
async def test_high_risk_review_normalizes_english_reason_to_chinese(
    high_risk_settings: None,
) -> None:
    class EnglishReasonReviewer(HighRiskReviewService):
        async def call_model(
            self,
            *,
            api_base: str,
            api_key: str,
            model: str,
            messages: list[dict[str, str]],
            use_json_mode: bool,
            max_tokens: int,
            request_timeout: float,
        ) -> tuple[dict[str, Any], str, dict[str, Any]]:
            return (
                {},
                (
                    '{"approved": false, "confidence": 0.34, '
                    '"reason": "Expected net profit is poor, risk is asymmetric, or evidence conflicts."}'
                ),
                {"finish_reason": "stop"},
            )

    result = await EnglishReasonReviewer().review_trade(
        {"symbol": "BTC/USDT", "side": "long"},
        api_base="https://api.deepseek.com",
        api_key=settings.high_risk_review_api_key,
        model="deepseek-reasoner",
    )

    assert result.approved is False
    assert result.reason == "预期净收益偏弱、风险收益不对称、证据存在冲突。"


def test_high_risk_review_extracts_json_from_reasoning_and_list_content() -> None:
    reviewer = HighRiskReviewService()

    reasoning_content, metadata = reviewer.extract_content(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "",
                        "reasoning_content": '```json\n{"approved": false, "reason": "冲突"}\n```',
                    },
                }
            ],
            "usage": {"completion_tokens": 16},
        }
    )
    list_content, _ = reviewer.extract_content(
        {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"text": "prefix"},
                            {"text": '{"approved": true, "reason": "ok"}'},
                        ]
                    }
                }
            ]
        }
    )

    assert reasoning_content == '{"approved": false, "reason": "冲突"}'
    assert metadata["finish_reason"] == "stop"
    assert metadata["usage"] == {"completion_tokens": 16}
    assert list_content == '{"approved": true, "reason": "ok"}'


def test_high_risk_review_records_reasoning_strip_metadata() -> None:
    reviewer = HighRiskReviewService()

    content, metadata = reviewer.extract_content(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": '<think>hidden</think>{"approved": false, "reason": "risk"}'
                    },
                }
            ],
            "usage": {"completion_tokens": 128},
        }
    )

    assert content == '{"approved": false, "reason": "risk"}'
    assert metadata["raw_has_think_tag"] is True
    assert metadata["reasoning_stripped"] is True


def test_high_risk_review_auth_failure_is_redacted() -> None:
    leaked_value = "abcdefghijklmnopqrstuvwxyz123456"
    response = httpx.Response(
        401,
        json={"detail": f"Authorization: Bearer {leaked_value} is invalid"},
        request=httpx.Request("POST", "https://review.example.invalid/v1/chat/completions"),
    )

    with pytest.raises(RuntimeError) as exc_info:
        HighRiskReviewService()._parse_response(response)

    message = str(exc_info.value)
    assert "HTTP 401" in message
    assert "HIGH_RISK_REVIEW_API_KEY" in message
    assert leaked_value not in message
    assert "Authorization: ***" in message


def test_high_risk_review_circuit_payload_exposes_redacted_last_failure(
    high_risk_settings: None,
) -> None:
    leaked_value = "abcdefghijklmnopqrstuvwxyz123456"
    reviewer = HighRiskReviewService()

    reviewer.record_failure(f"Authorization: Bearer {leaked_value} failed")
    payload = reviewer.circuit_payload()

    assert payload is not None
    assert payload["status"] == "circuit_open"
    assert payload["last_failure"] == "Authorization: *** failed"
    assert leaked_value not in str(payload)
    assert "Authorization: ***" in str(payload)


def test_high_risk_review_rejects_invalid_json_response() -> None:
    response = httpx.Response(
        200,
        text="not-json",
        request=httpx.Request("POST", "https://review.example.invalid/v1/chat/completions"),
    )

    with pytest.raises(RuntimeError, match="invalid JSON"):
        HighRiskReviewService()._parse_response(response)


def test_high_risk_review_rejects_non_object_json_response() -> None:
    response = httpx.Response(
        200,
        json=["not", "an", "object"],
        request=httpx.Request("POST", "https://review.example.invalid/v1/chat/completions"),
    )

    with pytest.raises(RuntimeError, match="non-object JSON payload"):
        HighRiskReviewService()._parse_response(response)


@pytest.mark.asyncio
async def test_high_risk_review_call_model_enforces_runtime_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _CapturingAsyncClient.requests = []
    monkeypatch.setattr(httpx, "AsyncClient", _CapturingAsyncClient)

    _, content, _ = await HighRiskReviewService().call_model(
        api_base="https://review.example.invalid/v1",
        api_key="test-" + "review-key",
        model="qwen3-32b-awq",
        messages=[{"role": "user", "content": "return json only"}],
        use_json_mode=True,
        max_tokens=HIGH_RISK_REVIEW_TOKEN_CAP + 500,
        request_timeout=9.0,
    )

    assert content
    request = _CapturingAsyncClient.requests[0]
    body = request["json"]
    assert request["timeout"] == 9.0
    assert body["max_tokens"] == HIGH_RISK_REVIEW_TOKEN_CAP
    assert body["response_format"] == {"type": "json_object"}
    assert body["chat_template_kwargs"]["enable_thinking"] is False
    assert body["messages"][0]["content"].endswith("/no_think")


@pytest.mark.asyncio
async def test_high_risk_review_compacts_oversized_expected_net_breakdown(
    high_risk_settings: None,
) -> None:
    reviewer = CapturingReviewer()
    huge_note = "oversized-breakdown-marker" * 200
    components = [
        {
            "key": f"component_{index}",
            "available": True,
            "side": "long",
            "raw_return_pct": 0.35 + index,
            "weight": 0.1,
            "contribution_pct": 0.01,
            "large_debug_payload": huge_note,
        }
        for index in range(64)
    ]

    await reviewer.review_trade(
        {
            "symbol": "BTC/USDT",
            "side": "long",
            "confidence": 0.91,
            "position_size_pct": 0.12,
            "leverage": 12,
            "opportunity_score": {
                "expected_net_return_pct": 0.82,
                "profit_quality_ratio": 1.5,
                "expected_net_breakdown": {
                    "formula": "test",
                    "net_pct": 0.82,
                    "model_net_pct": 1.2,
                    "components": components,
                },
            },
        },
        api_base="https://api.deepseek.com",
        api_key=settings.high_risk_review_api_key,
        model="deepseek-reasoner",
    )

    user_content = reviewer.calls[0]["messages"][1]["content"]
    payload = json.loads(user_content)
    compact_components = (
        payload["opportunity_score"].get("expected_net_breakdown", {}).get("components", [])
    )
    assert len(user_content) <= 3000
    assert huge_note not in user_content
    assert len(compact_components) <= 8


@pytest.mark.asyncio
async def test_entry_high_risk_review_is_observation_only_and_never_calls_reviewer(
    high_risk_settings: None,
) -> None:
    class ReviewerMustNotRun:
        called = False

        async def review_trade(self, *_args: Any, **_kwargs: Any) -> None:
            self.called = True
            raise AssertionError("observation-only review must not call an external model")

    reviewer = ReviewerMustNotRun()
    decision = DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.LONG,
        confidence=0.8,
        reasoning="test",
        raw_response={
            "opinions": [
                {"action": "long"},
                {"action": "short"},
            ],
            "ml_signal": {"predictions": [{"best_side": "short"}]},
        },
    )

    result = await EntryHighRiskReviewGatePolicy(reviewer=reviewer).evaluate(
        decision,
        "paper",
        [],
    )

    assert result is None
    assert reviewer.called is False
    review = decision.raw_response["high_risk_review"]
    assert review["read_only"] is True
    assert review["production_permission"] is False
    assert review["expert_disagreement"] == 0.5
    assert review["ml_ai_direction_conflict"] is True


@pytest.mark.asyncio
async def test_entry_high_risk_review_does_not_annotate_non_entry() -> None:
    decision = DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.HOLD,
        confidence=0.8,
        reasoning="test",
        raw_response={},
    )

    await EntryHighRiskReviewGatePolicy().evaluate(decision, "paper", [])

    assert "high_risk_review" not in decision.raw_response
