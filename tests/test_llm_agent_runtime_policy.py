from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ai_brain.llm_agent import LLMAgent
from data_feed.feature_vector import FeatureVector


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
        {"expert_mode": True, "_force_fast_independent_expert": True},
    )

    assert decision.action.value == "hold"
    assert captured_calls
    kwargs = captured_calls[-1]["kwargs"]
    assert kwargs["timeout"] <= 12.0
    assert kwargs["max_tokens"] <= 220
    assert kwargs["model_kwargs"]["response_format"] == {"type": "json_object"}
    assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    prompt_text = str(captured_calls[-1]["messages"][1].content)
    assert "FAST_EXPERT_JSON_V1" not in prompt_text
    assert "Return JSON only" in prompt_text
    assert len(prompt_text) < 1400
