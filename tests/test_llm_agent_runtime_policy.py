from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ai_brain.llm_agent import (
    LLMAgent,
    _backup_model_names,
    _extract_json,
    _provider_response_contract,
)
from core.exceptions import LLMResponseParseError
from data_feed.feature_vector import FeatureVector


def test_paper_parser_preserves_multidimensional_recommendation() -> None:
    agent = LLMAgent(
        name="trend_expert",
        api_config={"role": "trend_direction", "label": "trend"},
    )
    features = FeatureVector(symbol="BTC/USDT", current_price=100.0)
    parsed = {
        "action": "long",
        "confidence": 0.8,
        "reasoning": "趋势、收益和风险支持模拟开仓",
        "position_size_pct": 0.12,
        "suggested_leverage": 4.0,
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.06,
        "suggested_holding_minutes": 30.0,
        "maximum_holding_minutes": 90.0,
        "suggested_close_fraction": 0.4,
        "cross_check_for": None,
    }

    decision = agent._decision_from_parsed(
        dict(parsed),
        features,
        {"execution_mode": "paper"},
    )

    assert decision.position_size_pct == 0.12
    assert decision.suggested_leverage == 4.0
    assert decision.stop_loss_pct == 0.02
    assert decision.take_profit_pct == 0.06
    assert decision.suggested_holding_minutes == 30.0
    assert decision.maximum_holding_minutes == 90.0
    assert decision.suggested_close_fraction == 0.4


def test_live_parser_does_not_add_paper_holding_fields() -> None:
    agent = LLMAgent(
        name="trend_expert",
        api_config={"role": "trend_direction", "label": "trend"},
    )
    features = FeatureVector(symbol="BTC/USDT", current_price=100.0)

    decision = agent._decision_from_parsed(
        {
            "action": "long",
            "confidence": 0.8,
            "reasoning": "live compatibility",
            "suggested_holding_minutes": 30.0,
            "maximum_holding_minutes": 90.0,
            "suggested_close_fraction": 0.4,
        },
        features,
        {"execution_mode": "live"},
    )

    assert decision.suggested_holding_minutes == 0.0
    assert decision.maximum_holding_minutes == 0.0
    assert decision.suggested_close_fraction == 0.0


@pytest.mark.asyncio
async def test_keyless_loopback_model_uses_process_local_client_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("ai_brain.llm_agent.ChatOpenAI", FakeChatOpenAI)
    agent = LLMAgent(
        name="trend_expert",
        api_config={
            "api_base": "http://127.0.0.1:18003/v1",
            "api_key": "",
            "model": "BB-FinQuant-Expert-14B",
            "role": "trend_direction",
        },
    )

    await agent.initialize()

    assert captured["api_key"] == "local-loopback"
    assert captured["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert agent._api_key == ""


def test_finquant_loopback_does_not_try_unsupported_provider_model_names() -> None:
    assert _backup_model_names("BB-FinQuant-Expert-14B") == []


@pytest.mark.asyncio
async def test_backup_qwen3_model_gets_model_specific_no_think_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_calls: list[dict[str, Any]] = []

    class FakeChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.model = str(kwargs["model"])

        async def ainvoke(self, messages: list[Any]) -> SimpleNamespace:
            captured_calls.append(
                {
                    "model": self.model,
                    "kwargs": self.kwargs,
                    "messages": messages,
                }
            )
            if self.model == "plain-primary":
                return SimpleNamespace(content="not-json")
            return SimpleNamespace(
                content=(
                    '{"action":"hold","confidence":0.55,"reasoning":"backup ok",'
                    '"position_size_pct":0,"suggested_leverage":1}'
                )
            )

    monkeypatch.setattr("ai_brain.llm_agent.ChatOpenAI", FakeChatOpenAI)
    agent = LLMAgent(
        name="trend_expert",
        api_config={
            "api_base": "http://llm.test/v1",
            "api_key": "test-key",
            "model": "plain-primary",
            "role": "technical_trend",
        },
    )
    await agent.initialize()

    decision = await agent.decide(
        FeatureVector(symbol="BTC/USDT", current_price=100.0),
        {"expert_mode": True},
    )

    primary_calls = [call for call in captured_calls if call["model"] == "plain-primary"]
    qwen_calls = [call for call in captured_calls if call["model"] == "qwen3-max"]

    assert len(primary_calls) == 2
    assert qwen_calls
    assert all("/no_think" not in str(call["messages"][1].content) for call in primary_calls)
    assert str(qwen_calls[0]["messages"][1].content).endswith("/no_think")
    assert qwen_calls[0]["kwargs"]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert decision.raw_response
    assert decision.raw_response["provider_model"] == "qwen3-max"
    assert decision.raw_response["fallback_from"] == "plain-primary"


@pytest.mark.asyncio
async def test_fast_independent_expert_uses_short_json_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_calls: list[dict[str, Any]] = []

    class FakeChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        async def ainvoke(self, messages: list[Any]) -> SimpleNamespace:
            captured_calls.append({"kwargs": self.kwargs, "messages": messages})
            return SimpleNamespace(
                content=(
                    '{"action":"hold","confidence":0.55,"reasoning":"???????",'
                    '"position_size_pct":0,"suggested_leverage":1,'
                    '"stop_loss_pct":0.05,"take_profit_pct":0.1,"cross_check_for":null}'
                )
            )

    monkeypatch.setattr("ai_brain.llm_agent.ChatOpenAI", FakeChatOpenAI)
    agent = LLMAgent(
        name="risk_expert",
        api_config={
            "api_base": "http://llm.test/v1",
            "api_key": "test-key",
            "model": "deepseek-r1-14b-risk",
            "role": "risk_anomaly",
        },
    )
    await agent.initialize()

    decision = await agent.decide(
        FeatureVector(symbol="SOL/USDT", current_price=150.0),
        {
            "execution_mode": "paper",
            "expert_mode": True,
            "_force_fast_independent_expert": True,
        },
    )

    assert decision.action.value == "hold"
    assert captured_calls
    kwargs = captured_calls[-1]["kwargs"]
    assert kwargs["timeout"] <= 12.0
    assert kwargs["max_tokens"] <= 320
    assert kwargs["model_kwargs"]["response_format"] == {"type": "json_object"}
    assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    system_prompt = str(captured_calls[-1]["messages"][0].content)
    assert "PAPER_FAST_EXPERT_JSON_V1" in system_prompt
    assert "suggested_holding_minutes" in system_prompt
    assert "PAPER_MULTIDIMENSIONAL_PLAN_V1" not in system_prompt
    prompt_text = str(captured_calls[-1]["messages"][1].content)
    assert "FAST_EXPERT_JSON_V1" not in prompt_text
    assert "Return JSON only" in prompt_text
    assert len(prompt_text) < 1400


def test_final_decision_always_uses_structured_json_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: list[dict[str, Any]] = []

    class FakeChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.append(kwargs)

    monkeypatch.setattr("ai_brain.llm_agent.ChatOpenAI", FakeChatOpenAI)
    agent = LLMAgent(
        name="decision_maker",
        api_config={
            "api_base": "https://api.example.test/v1",
            "api_key": "test-key",
            "model": "deepseek-v4-pro",
            "role": "final_decision",
        },
    )

    agent._base_url = "https://api.example.test/v1"
    agent._api_key = "test-key"
    agent._create_llm("deepseek-v4-pro")

    kwargs = captured_kwargs[-1]
    assert kwargs["model_kwargs"]["response_format"] == {"type": "json_object"}
    assert kwargs["max_tokens"] <= 320
    assert kwargs["extra_body"]["thinking"] == {"type": "disabled"}


def test_provider_response_contract_distinguishes_reasoning_only_from_final_json() -> None:
    reasoning_only = _provider_response_contract(
        SimpleNamespace(
            content="",
            additional_kwargs={"reasoning_content": "internal reasoning"},
            response_metadata={"finish_reason": "length"},
            usage_metadata={"completion_tokens": 320},
        )
    )
    completed = _provider_response_contract(
        SimpleNamespace(
            content='{"action":"hold"}',
            additional_kwargs={"reasoning_content": "brief reasoning"},
            response_metadata={"finish_reason": "stop"},
            usage_metadata={
                "completion_tokens": 42,
                "completion_tokens_details": {"reasoning_tokens": 20},
            },
        )
    )

    assert reasoning_only["reasoning_only"] is True
    assert reasoning_only["truncated"] is True
    assert reasoning_only["has_final_content"] is False
    assert completed["reasoning_only"] is False
    assert completed["has_final_content"] is True
    assert completed["reasoning_tokens"] == 20


def test_invalid_final_json_is_reported_as_a_parse_error() -> None:
    with pytest.raises(LLMResponseParseError):
        _extract_json("not a JSON decision")
