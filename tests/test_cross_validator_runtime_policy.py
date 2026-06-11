from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ai_brain.cross_validator import CrossValidator, _is_local_qwen3_trade_model
from core.model_runtime import completion_token_limit


def test_consultation_messages_add_no_think_for_qwen3() -> None:
    messages = [
        SystemMessage(content="system"),
        HumanMessage(content='{"major_conflicts": []}'),
    ]

    controlled = CrossValidator._consultation_messages_for_model(messages, "qwen3-32b-trade")

    assert controlled is not messages
    assert controlled[0] is messages[0]
    assert controlled[1] is not messages[1]
    assert str(controlled[1].content).endswith("/no_think")
    assert str(messages[1].content) == '{"major_conflicts": []}'


def test_consultation_messages_keep_plain_models_unchanged() -> None:
    messages = [HumanMessage(content="plain")]

    assert (
        CrossValidator._consultation_messages_for_model(messages, "Qwen2.5-32B-Instruct")
        is messages
    )


def test_local_qwen3_trade_detection_excludes_review_alias() -> None:
    assert _is_local_qwen3_trade_model("qwen3-32b-trade")
    assert not _is_local_qwen3_trade_model("qwen3-32b-risk-review")


async def test_qwen3_consultation_uses_short_non_thinking_runtime_policy(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured["kwargs"] = kwargs

        async def ainvoke(self, messages: list[Any]) -> AIMessage:
            captured["messages"] = messages
            return AIMessage(content='{"recommended_action":"hold"}')

    monkeypatch.setattr("ai_brain.cross_validator.ChatOpenAI", FakeChatOpenAI)

    response, content = await CrossValidator()._invoke_consultation_model(
        [SystemMessage(content="system"), HumanMessage(content="payload")],
        {
            "api_base": "http://127.0.0.1:8000/v1",
            "api_key": "test-key",
            "model": "qwen3-32b-trade",
        },
    )

    assert isinstance(response, AIMessage)
    assert content == '{"recommended_action":"hold"}'
    assert captured["kwargs"]["max_completion_tokens"] == completion_token_limit(
        "consultation", 1400, floor=160
    )
    assert captured["kwargs"]["max_completion_tokens"] == 700
    assert captured["kwargs"]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert str(captured["messages"][1].content).endswith("/no_think")
