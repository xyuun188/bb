import asyncio
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import ai_brain.cross_validator as cross_validator_module
from ai_brain.cross_validator import (
    ConsultationQueueTimeoutError,
    CrossValidator,
    _is_local_qwen3_trade_model,
)
from ai_brain.llm_agent import shared_llm_capacity_slot
from config.settings import settings
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
    assert _is_local_qwen3_trade_model("BB-FinQuant-Expert-14B")
    assert not _is_local_qwen3_trade_model("qwen3-32b-risk-review")


def test_keyless_loopback_prefers_decision_fallback_before_slow_review(monkeypatch) -> None:
    monkeypatch.setattr(settings, "high_risk_review_enabled", True)
    monkeypatch.setattr(settings, "high_risk_review_api_base", "http://127.0.0.1:18002/v1")
    monkeypatch.setattr(settings, "high_risk_review_api_key", "review-key")
    monkeypatch.setattr(settings, "high_risk_review_model", "deepseek-r1-14b-risk")
    monkeypatch.setattr(
        CrossValidator,
        "_fixed_model_cfg",
        lambda self, name: {
            "name": name,
            "label": "decision",
            "api_base": "https://decision.example.com/v1",
            "api_key": "decision-key",
            "model": "decision-model",
        }
        if name == "decision_maker"
        else {},
    )

    candidates = CrossValidator()._consultation_candidates(
        {
            "api_base": "http://127.0.0.1:18003/v1",
            "api_key": "",
            "model": "BB-FinQuant-Expert-14B",
        }
    )

    assert candidates[0]["model"] == "BB-FinQuant-Expert-14B"
    assert candidates[0]["configuration_type"] == "keyless_loopback"
    assert candidates[1]["model"] == "decision-model"
    assert candidates[2]["model"] == "deepseek-r1-14b-risk"
    assert not any(candidate["source"] == "backup" for candidate in candidates)


def test_keyless_non_loopback_is_not_a_consultation_candidate(monkeypatch) -> None:
    monkeypatch.setattr(settings, "high_risk_review_enabled", False)
    monkeypatch.setattr(
        CrossValidator,
        "_fixed_model_cfg",
        lambda self, name: {},
    )

    candidates = CrossValidator()._consultation_candidates(
        {
            "api_base": "https://models.example.com/v1",
            "api_key": "",
            "model": "BB-FinQuant-Expert-14B",
        }
    )

    assert candidates == []


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
    assert captured["kwargs"]["model_kwargs"] == {
        "response_format": {"type": "json_object"}
    }
    assert captured["kwargs"]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert str(captured["messages"][1].content).endswith("/no_think")


async def test_consultation_request_timeout_is_capped(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured["kwargs"] = kwargs

        async def ainvoke(self, messages: list[Any]) -> AIMessage:
            return AIMessage(content='{"recommended_action":"hold"}')

    monkeypatch.setattr("ai_brain.cross_validator.ChatOpenAI", FakeChatOpenAI)

    await CrossValidator()._invoke_consultation_model(
        [SystemMessage(content="system"), HumanMessage(content="payload")],
        {
            "api_base": "http://127.0.0.1:8000/v1",
            "api_key": "test-key",
            "model": "qwen3-32b-trade",
        },
        request_timeout=60.0,
    )

    assert captured["kwargs"]["timeout"] == 12.0


async def test_keyless_loopback_uses_process_local_placeholder(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        async def ainvoke(self, messages: list[Any]) -> AIMessage:
            return AIMessage(content='{"resolution_status":"unresolved"}')

    monkeypatch.setattr("ai_brain.cross_validator.ChatOpenAI", FakeChatOpenAI)

    await CrossValidator()._invoke_consultation_model(
        [HumanMessage(content="payload")],
        {
            "api_base": "http://127.0.0.1:18003/v1",
            "api_key": "",
            "model": "BB-FinQuant-Expert-14B",
        },
    )

    assert captured["api_key"] == "local-loopback"


async def test_r1_failure_keeps_budget_for_completed_fallback(monkeypatch) -> None:
    validator = CrossValidator()
    candidates = [
        {
            "name": "high_risk_review",
            "label": "risk",
            "api_base": "http://127.0.0.1:18002/v1",
            "api_key": "review-key",
            "model": "deepseek-r1-14b-risk",
            "source": "high_risk_review",
            "retries": 1,
        },
        {
            "name": "decision_maker",
            "label": "fallback",
            "api_base": "http://127.0.0.1:18003/v1",
            "api_key": "",
            "model": "BB-FinQuant-Expert-14B",
            "source": "decision_maker",
            "retries": 1,
        },
    ]
    monkeypatch.setattr(validator, "_consultation_candidates", lambda trend_cfg: candidates)

    async def fake_invoke(
        messages,
        candidate,
        request_timeout=None,
        *,
        queue_timeout_seconds=None,
        runtime_metrics=None,
        capacity_context=None,
    ):
        del queue_timeout_seconds, capacity_context
        if runtime_metrics is not None:
            runtime_metrics.update(
                {
                    "queue_wait_seconds": 0.0,
                    "inference_duration_seconds": 0.0,
                    "consultation_concurrency": 2,
                }
            )
        if candidate["model"] == "deepseek-r1-14b-risk":
            raise TimeoutError("r1 timed out")
        await asyncio.sleep(0)
        return (
            AIMessage(content="ok"),
            '{"conflict_note":"已统一",'
            '"observation_summary":"后备完成",'
            '"resolution_status":"resolved",'
            '"resolved_action":"hold",'
            '"resolved_conflict_pairs":[["risk_expert","trend_expert"]]}',
        )

    monkeypatch.setattr(validator, "_invoke_consultation_model", fake_invoke)
    result = await validator.consult_if_needed(
        {},
        [
            {
                "expert_pair": ["trend_expert", "risk_expert"],
                "major_conflict": True,
                "conflict_note": "方向冲突",
            }
        ],
        timeout_seconds=1.0,
    )

    assert result is not None
    assert result["status"] == "completed"
    assert result["resolution_status"] == "resolved"
    assert result["resolved_action"] == "hold"
    assert result["fallback_used"] is True
    assert [item["status"] for item in result["consultation_attempts"]] == [
        "call_failed",
        "completed",
    ]


async def test_primary_consultation_uses_full_first_attempt_budget(monkeypatch) -> None:
    validator = CrossValidator()
    candidates = [
        {
            "name": "trend_expert",
            "label": "trend",
            "api_base": "http://127.0.0.1:18003/v1",
            "api_key": "",
            "model": "BB-FinQuant-Expert-14B",
            "source": "primary",
            "retries": 1,
        },
        {
            "name": "decision_maker",
            "label": "fallback",
            "api_base": "https://decision.example.com/v1",
            "api_key": "decision-key",
            "model": "decision-model",
            "source": "decision_maker",
            "retries": 1,
        },
    ]
    monkeypatch.setattr(validator, "_consultation_candidates", lambda trend_cfg: candidates)
    request_timeouts: list[float] = []

    async def fake_invoke(
        messages,
        candidate,
        request_timeout=None,
        *,
        queue_timeout_seconds=None,
        runtime_metrics=None,
        capacity_context=None,
    ):
        del messages, candidate, queue_timeout_seconds, capacity_context
        request_timeouts.append(float(request_timeout or 0.0))
        if runtime_metrics is not None:
            runtime_metrics.update(
                {
                    "queue_wait_seconds": 0.0,
                    "inference_duration_seconds": 0.0,
                    "consultation_concurrency": 2,
                }
            )
        return (
            AIMessage(content="ok"),
            '{"conflict_note":"reviewed",'
            '"observation_summary":"primary completed",'
            '"resolution_status":"resolved",'
            '"resolved_action":"hold",'
            '"resolved_conflict_pairs":[["risk_expert","trend_expert"]]}',
        )

    monkeypatch.setattr(validator, "_invoke_consultation_model", fake_invoke)
    result = await validator.consult_if_needed(
        {},
        [
            {
                "expert_pair": ["trend_expert", "risk_expert"],
                "major_conflict": True,
                "conflict_note": "direction conflict",
            }
        ],
        timeout_seconds=12.0,
    )

    assert result is not None
    assert result["status"] == "completed"
    assert request_timeouts == [pytest.approx(10.0, abs=0.1)]


async def test_consultation_allows_two_inferences_without_queue_serialization(
    monkeypatch,
) -> None:
    entered_two = asyncio.Event()
    release = asyncio.Event()
    active = 0
    maximum_active = 0

    class FakeChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def ainvoke(self, messages: list[Any]) -> AIMessage:
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            if active == 2:
                entered_two.set()
            try:
                await release.wait()
                return AIMessage(content='{"resolution_status":"unresolved"}')
            finally:
                active -= 1

    monkeypatch.setattr("ai_brain.cross_validator.ChatOpenAI", FakeChatOpenAI)
    metrics = [{}, {}]
    validator = CrossValidator()
    candidate = {
        "api_base": "http://127.0.0.1:18003/v1",
        "api_key": "",
        "model": "BB-FinQuant-Expert-14B",
    }

    tasks = [
        asyncio.create_task(
            validator._invoke_consultation_model(
                [HumanMessage(content="payload")],
                candidate,
                request_timeout=1.0,
                queue_timeout_seconds=0.2,
                runtime_metrics=metrics[index],
            )
        )
        for index in range(2)
    ]
    await asyncio.wait_for(entered_two.wait(), timeout=0.5)
    release.set()
    await asyncio.gather(*tasks)

    assert maximum_active == 2
    assert all(item["consultation_concurrency"] == 2 for item in metrics)
    assert all(item["queue_wait_seconds"] < 0.1 for item in metrics)


async def test_consultation_queue_timeout_includes_shared_expert_capacity(
    monkeypatch,
) -> None:
    invoked = False

    class FakeChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def ainvoke(self, messages: list[Any]) -> AIMessage:
            nonlocal invoked
            invoked = True
            return AIMessage(content="{}")

    shared_slots = [
        shared_llm_capacity_slot({"_analysis_budget_scope": "market_test"})
        for _ in range(2)
    ]
    for slot in shared_slots:
        await slot.__aenter__()
    monkeypatch.setattr("ai_brain.cross_validator.ChatOpenAI", FakeChatOpenAI)
    metrics: dict[str, Any] = {}

    try:
        with pytest.raises(ConsultationQueueTimeoutError):
            await CrossValidator()._invoke_consultation_model(
                [HumanMessage(content="payload")],
                {
                    "api_base": "http://127.0.0.1:18003/v1",
                    "api_key": "",
                    "model": "BB-FinQuant-Expert-14B",
                },
                request_timeout=1.0,
                queue_timeout_seconds=0.05,
                runtime_metrics=metrics,
                capacity_context={"_analysis_budget_scope": "position_test"},
            )
    finally:
        for slot in reversed(shared_slots):
            await slot.__aexit__(None, None, None)

    assert invoked is False
    assert metrics["queue_timeout_seconds"] == 0.05
    assert metrics["queue_wait_seconds"] >= 0.04
    assert metrics["inference_duration_seconds"] == 0.0


async def test_consultation_queue_timeout_is_audited_in_attempts(monkeypatch) -> None:
    validator = CrossValidator()
    shared_slots = [
        shared_llm_capacity_slot({"_analysis_budget_scope": "market_test"})
        for _ in range(2)
    ]
    for slot in shared_slots:
        await slot.__aenter__()

    class FakeChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def ainvoke(self, messages: list[Any]) -> AIMessage:
            raise AssertionError("a saturated consultation queue must not invoke the model")

    monkeypatch.setattr("ai_brain.cross_validator.ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(
        cross_validator_module,
        "_CONSULTATION_QUEUE_WAIT_CAP_SECONDS",
        0.05,
    )
    monkeypatch.setattr(
        validator,
        "_consultation_candidates",
        lambda trend_cfg: [
            {
                "name": "trend_expert",
                "label": "trend",
                "api_base": "http://127.0.0.1:18003/v1",
                "api_key": "",
                "model": "BB-FinQuant-Expert-14B",
                "source": "primary",
                "retries": 1,
            }
        ],
    )

    try:
        result = await validator.consult_if_needed(
            {},
            [
                {
                    "expert_pair": ["trend_expert", "risk_expert"],
                    "major_conflict": True,
                    "conflict_note": "direction conflict",
                }
            ],
            timeout_seconds=1.0,
        )
    finally:
        for slot in reversed(shared_slots):
            await slot.__aexit__(None, None, None)

    assert result is not None
    assert result["status"] == "failed"
    assert result["resolution_status"] == "unresolved"
    assert result["resolved_action"] == "unclear"
    assert result["consultation_attempts"][0]["status"] == "queue_timeout"
    assert result["consultation_attempts"][0]["queue_timeout_seconds"] == 0.05
    assert result["consultation_attempts"][0]["inference_duration_seconds"] == 0.0


def test_consultation_budget_is_bounded(monkeypatch) -> None:
    from ai_brain.cross_validator import _consultation_budget_seconds

    monkeypatch.setattr(settings, "ai_decision_maker_timeout_seconds", 120.0)
    assert _consultation_budget_seconds() == 18.0

    monkeypatch.setattr(settings, "ai_decision_maker_timeout_seconds", 2.0)
    assert _consultation_budget_seconds() == 6.0
