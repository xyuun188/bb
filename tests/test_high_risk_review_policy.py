from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest

from ai_brain.base_model import Action, DecisionOutput
from config.settings import settings
from core.model_runtime import HIGH_RISK_REVIEW_TOKEN_CAP
from services.entry_high_risk_review import EntryHighRiskReviewGatePolicy
from services.high_risk_review_service import HighRiskReviewResult, HighRiskReviewService


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
    monkeypatch.setattr(settings, "ai_api_base", "https://primary-llm.example.invalid/v1")
    monkeypatch.setattr(settings, "ai_api_key", "local")


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


class FailingReviewer(HighRiskReviewService):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

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
        self.calls += 1
        raise RuntimeError("deepseek timeout")


class SuccessfulReviewer(HighRiskReviewService):
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
        return ({}, '{"approved": true, "confidence": 0.8, "reason": "恢复后通过"}', {})


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


class GateReviewer:
    def __init__(self, result: HighRiskReviewResult | None = None) -> None:
        self.result = result or HighRiskReviewResult(
            approved=True,
            confidence=0.9,
            reason="证据尚可，允许小心执行",
            attempts=[{"attempt": 1, "json_mode": True, "max_tokens": 420}],
        )
        self.calls: list[dict[str, Any]] = []
        self.failures: list[str] = []

    def api_key(self, api_base: str) -> str:
        return "test-review-key"

    def circuit_payload(self) -> dict[str, Any] | None:
        return None

    async def review_trade(
        self,
        prompt: dict[str, Any],
        *,
        api_base: str,
        api_key: str,
        model: str,
    ) -> HighRiskReviewResult:
        self.calls.append(
            {
                "prompt": prompt,
                "api_base": api_base,
                "api_key": api_key,
                "model": model,
            }
        )
        return self.result

    def record_failure(self, reason: str) -> None:
        self.failures.append(reason)


class LeakyGateReviewer(GateReviewer):
    def __init__(self, leaked_value: str) -> None:
        super().__init__()
        self.leaked_value = leaked_value

    async def review_trade(
        self,
        prompt: dict[str, Any],
        *,
        api_base: str,
        api_key: str,
        model: str,
    ) -> HighRiskReviewResult:
        self.calls.append(
            {
                "prompt": prompt,
                "api_base": api_base,
                "api_key": api_key,
                "model": model,
            }
        )
        raise RuntimeError(f"Authorization: Bearer {self.leaked_value} failed")


def _gate(review_service: Any | None = None) -> EntryHighRiskReviewGatePolicy:
    async def allocation_state(model_mode: str) -> dict[str, Any]:
        return {"today_risk_pnl": 0.0}

    return EntryHighRiskReviewGatePolicy(
        reviewer=review_service or HighRiskReviewService(),
        allocation_state_provider=allocation_state,
    )


def _raw_response(decision: DecisionOutput) -> dict[str, Any]:
    raw = decision.raw_response
    assert isinstance(raw, dict)
    return raw


def _high_risk_decision() -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.LONG,
        confidence=0.82,
        reasoning="unit test",
        position_size_pct=0.13,
        suggested_leverage=12.0,
        raw_response={
            "opportunity_score": {
                "expected_net_return_pct": 0.42,
                "profit_quality_ratio": 1.4,
            },
            "quant_profit_probe": {"loss_probability": 0.55},
            "opinions": [],
        },
    )


def _local_controlled_probe_decision() -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.LONG,
        confidence=0.76,
        reasoning="unit test",
        position_size_pct=0.012,
        suggested_leverage=3.0,
        raw_response={
            "opportunity_score": {
                "score": 0.98,
                "min_score_required": 1.2,
                "expected_net_return_pct": 1.12,
                "profit_quality_ratio": 1.45,
                "server_profit_loss_probability": 0.46,
                "tail_risk_score": 0.31,
                "evidence_score": {
                    "tier": "exploration",
                    "effective_score": 49.5,
                    "shadow_only": False,
                    "hard_block": False,
                },
            },
            "profit_risk_sizing": {
                "quality_tier": "probe",
                "low_payoff_quality": True,
                "low_payoff_reasons": ["evidence_low_payoff_quality"],
                "position_size_pct": 0.012,
                "leverage": 3.0,
            },
            "opinions": [
                {"action": "short"},
                {"action": "short"},
                {"action": "short"},
            ],
        },
    )


def _gate_with_today_loss(review_service: Any | None = None) -> EntryHighRiskReviewGatePolicy:
    async def allocation_state(model_mode: str) -> dict[str, Any]:
        return {"today_risk_pnl": -3.5}

    return EntryHighRiskReviewGatePolicy(
        reviewer=review_service or HighRiskReviewService(),
        allocation_state_provider=allocation_state,
    )


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
async def test_high_risk_review_gate_delegates_to_runtime_service(
    high_risk_settings: None,
) -> None:
    reviewer = GateReviewer()
    gate = _gate(reviewer)

    decision = _high_risk_decision()
    reason = await gate.evaluate(decision, "paper", [])

    assert reason is None
    assert len(reviewer.calls) == 1
    assert reviewer.calls[0]["api_base"] == "https://api.deepseek.com"
    assert reviewer.calls[0]["api_key"] == "test-review-key"
    assert reviewer.calls[0]["model"] == "deepseek-reasoner"
    assert reviewer.calls[0]["prompt"]["trigger_reasons"] == [
        "high_leverage:12.0x",
        "large_position:13.0%",
    ]
    review = _raw_response(decision)["high_risk_review"]
    assert review["status"] == "completed"
    assert review["approved"] is True


@pytest.mark.asyncio
async def test_high_risk_review_skips_online_reviewer_for_local_controlled_probe(
    high_risk_settings: None,
) -> None:
    reviewer = GateReviewer()
    gate = _gate_with_today_loss(reviewer)
    decision = _local_controlled_probe_decision()

    reason = await gate.evaluate(decision, "paper", [])

    assert reason is None
    assert reviewer.calls == []
    review = _raw_response(decision)["high_risk_review"]
    assert review["status"] == "skipped_local_controlled_probe"
    assert review["approved"] is True
    assert review["triggered"] is False
    assert review["low_payoff_quality"] is True
    assert "today_recovery_after_loss" in review["advisory_reasons"]
    assert "expert_disagreement:100%" in review["advisory_reasons"]
    assert "sizing:probe" in review["probe_sources"]
    assert review["expected_net_return_pct"] == 1.12
    assert review["position_size_pct"] == 0.012


@pytest.mark.asyncio
async def test_high_risk_review_local_probe_does_not_skip_large_or_high_leverage_entries(
    high_risk_settings: None,
) -> None:
    reviewer = GateReviewer()
    gate = _gate_with_today_loss(reviewer)
    decision = _local_controlled_probe_decision()
    decision.position_size_pct = 0.13
    decision.suggested_leverage = 12.0

    reason = await gate.evaluate(decision, "paper", [])

    assert reason is None
    assert len(reviewer.calls) == 1
    review = _raw_response(decision)["high_risk_review"]
    assert review["status"] == "completed"
    assert review["approved"] is True
    assert review["hard_review_required"] is True
    assert reviewer.calls[0]["prompt"]["trigger_reasons"] == [
        "high_leverage:12.0x",
        "large_position:13.0%",
        "expert_disagreement:100%",
        "today_recovery_after_loss",
    ]


@pytest.mark.asyncio
async def test_high_risk_review_local_probe_does_not_skip_weak_loss_profile(
    high_risk_settings: None,
) -> None:
    reviewer = GateReviewer()
    gate = _gate_with_today_loss(reviewer)
    decision = _local_controlled_probe_decision()
    raw = _raw_response(decision)
    raw["opportunity_score"]["server_profit_loss_probability"] = 0.76
    decision.raw_response = raw

    reason = await gate.evaluate(decision, "paper", [])

    assert reason is None
    assert len(reviewer.calls) == 1
    review = _raw_response(decision)["high_risk_review"]
    assert review["status"] == "completed"
    assert review["hard_review_required"] is True
    assert "expert_disagreement:100%" in reviewer.calls[0]["prompt"]["trigger_reasons"]
    assert "today_recovery_after_loss" in reviewer.calls[0]["prompt"]["trigger_reasons"]


@pytest.mark.asyncio
async def test_high_risk_review_gate_rejects_invalid_api_base_without_leaking_it(
    high_risk_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad_url = "http://user:password@review.example.invalid/v1"
    monkeypatch.setattr(settings, "high_risk_review_api_base", bad_url)
    reviewer = GateReviewer()
    gate = _gate(reviewer)

    decision = _high_risk_decision()
    reason = await gate.evaluate(decision, "paper", [])

    assert "高风险复核地址配置无效" in str(reason)
    assert reviewer.calls == []
    review = _raw_response(decision)["high_risk_review"]
    assert review["status"] == "skipped_blocked"
    assert review["api_base"] == "invalid"
    assert bad_url not in str(review)
    assert "user:password" not in str(reason)


@pytest.mark.asyncio
async def test_high_risk_review_gate_blocks_required_review_when_api_base_missing(
    high_risk_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "high_risk_review_api_base", "")
    reviewer = GateReviewer()
    gate = _gate(reviewer)

    decision = _high_risk_decision()
    reason = await gate.evaluate(decision, "paper", [])

    assert "高风险复核地址配置无效" in str(reason)
    assert "High-risk review API base is required" in str(reason)
    assert reviewer.calls == []
    review = _raw_response(decision)["high_risk_review"]
    assert review["status"] == "skipped_blocked"
    assert review["api_base"] == "invalid"
    assert review["approved"] is False


@pytest.mark.asyncio
async def test_high_risk_review_gate_failure_reason_is_redacted(
    high_risk_settings: None,
) -> None:
    leaked_value = "abcdefghijklmnopqrstuvwxyz123456"
    reviewer = LeakyGateReviewer(leaked_value)
    gate = _gate(reviewer)

    decision = _high_risk_decision()
    reason = await gate.evaluate(decision, "paper", [])

    review = _raw_response(decision)["high_risk_review"]
    assert review["status"] == "error_blocked"
    assert review["error"] == "Authorization: *** failed"
    assert leaked_value not in str(reason)
    assert leaked_value not in str(review)
    assert leaked_value not in str(reviewer.failures)
    assert "Authorization: ***" in str(reason)
    assert reviewer.failures == ["Authorization: *** failed"]


@pytest.mark.asyncio
async def test_high_risk_review_failure_opens_circuit_and_skips_next_call(
    high_risk_settings: None,
) -> None:
    reviewer = FailingReviewer()
    gate = _gate(reviewer)

    first = _high_risk_decision()
    first_reason = await gate.evaluate(first, "paper", [])

    assert "高风险复核调用失败" in str(first_reason)
    assert _raw_response(first)["high_risk_review"]["status"] == "error_blocked"
    assert reviewer.circuit_open_until is not None
    assert reviewer.calls == 1

    second = _high_risk_decision()
    second_reason = await gate.evaluate(second, "paper", [])

    assert "熔断冷却中" in str(second_reason)
    second_review = _raw_response(second)["high_risk_review"]
    assert second_review["status"] == "circuit_open"
    assert second_review["last_failure"] == "deepseek timeout"
    assert reviewer.calls == 1


@pytest.mark.asyncio
async def test_high_risk_review_circuit_recovers_after_cooldown(
    high_risk_settings: None,
) -> None:
    reviewer = SuccessfulReviewer()
    reviewer._failure_count = 1
    reviewer._circuit_open_until = datetime.now(UTC) - timedelta(seconds=1)
    gate = _gate(reviewer)

    decision = _high_risk_decision()
    reason = await gate.evaluate(decision, "paper", [])

    assert reason is None
    assert _raw_response(decision)["high_risk_review"]["status"] == "completed"
    assert reviewer.failure_count == 0


@pytest.mark.asyncio
async def test_high_risk_review_requires_explicit_key_for_deepseek_base(
    high_risk_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "high_risk_review_api_key", "")
    gate = _gate(HighRiskReviewService())

    decision = _high_risk_decision()
    reason = await gate.evaluate(decision, "paper", [])

    assert "未完整配置" in str(reason)
    review = _raw_response(decision)["high_risk_review"]
    assert review["status"] == "skipped_blocked"
    assert review["approved"] is False


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
async def test_high_risk_review_gate_sends_opportunity_summary_only(
    high_risk_settings: None,
) -> None:
    reviewer = GateReviewer()
    gate = _gate(reviewer)
    decision = _high_risk_decision()
    raw = _raw_response(decision)
    marker = "raw-breakdown-marker" * 200
    raw["opportunity_score"]["expected_net_breakdown"] = {
        "formula": "test",
        "net_pct": 0.8,
        "model_net_pct": 1.1,
        "components": [
            {
                "key": f"component_{index}",
                "available": True,
                "side": "long",
                "raw_return_pct": index,
                "weight": 0.1,
                "contribution_pct": 0.01,
                "large_debug_payload": marker,
            }
            for index in range(32)
        ],
    }
    decision.raw_response = raw

    reason = await gate.evaluate(decision, "paper", [])

    assert reason is None
    sent_prompt = reviewer.calls[0]["prompt"]
    sent_text = json.dumps(sent_prompt, ensure_ascii=False)
    components = sent_prompt["opportunity_score"]["expected_net_breakdown"]["components"]
    assert marker not in sent_text
    assert len(components) == 8
    assert all("large_debug_payload" not in component for component in components)
