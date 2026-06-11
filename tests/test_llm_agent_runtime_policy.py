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
