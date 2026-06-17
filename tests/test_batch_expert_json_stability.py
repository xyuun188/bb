from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ai_brain.base_model import AbstractAIModel, Action, DecisionOutput
from ai_brain.ensemble_coordinator import EnsembleCoordinator
from ai_brain.llm_agent import LLMAgent, _extract_json
from ai_brain.model_registry import ModelRegistry
from ai_brain.prompts import build_batch_experts_user_prompt
from config.settings import settings
from core.exceptions import LLMResponseParseError
from data_feed.feature_vector import FeatureVector
from tests.model_endpoint_fixtures import LOCAL_DEEPSEEK_TEST_BASE, LOCAL_QWEN_TEST_BASE


def test_batch_expert_prompt_uses_compact_json_contract() -> None:
    prompt = build_batch_experts_user_prompt(
        "symbol=BTC/USDT price=100",
        {"review_positions": True, "open_positions": [{"symbol": "BTC/USDT"}]},
    )

    assert "BATCH_EXPERT_JSON_V7" in prompt
    assert (
        "Required experts: trend_expert, momentum_expert, sentiment_expert, position_expert, risk_expert"
        in prompt
    )
    assert "do not copy one expert's opinion into all experts" in prompt
    assert "weak evidence=hold" in prompt
    assert '"action":"hold","confidence":0.50' not in prompt
    assert "daily_target" not in prompt
    assert "STRICT_COMPACT_BATCH_JSON_V3" not in prompt
    assert "Payload JSON, truncated to 1000 chars" in prompt


def test_batch_expert_prompt_can_scope_to_provider_group() -> None:
    prompt = build_batch_experts_user_prompt(
        "symbol=BTC/USDT price=100",
        {"review_positions": False},
        ["sentiment_expert", "position_expert", "risk_expert"],
    )

    assert "Required experts: sentiment_expert, position_expert, risk_expert" in prompt
    assert (
        '"experts":{"sentiment_expert":{...},"position_expert":{...},"risk_expert":{...}}' in prompt
    )
    assert "Do not include these omitted experts: trend_expert, momentum_expert." in prompt


def test_extract_json_repairs_missing_batch_tail() -> None:
    parsed = _extract_json(
        """
        ```json
        {"experts":{"trend_expert":{"action":"hold","confidence":0.5},
        "momentum_expert":{"action":"hold","confidence":0.4},
        ```
        """
    )

    assert parsed["experts"]["trend_expert"]["action"] == "hold"
    assert parsed["experts"]["momentum_expert"]["confidence"] == 0.4


def test_latency_summary_deduplicates_shared_batch_wall_time() -> None:
    coordinator = EnsembleCoordinator(ModelRegistry())
    summary = coordinator._latency_summary(
        [{"stage": "expert_initial", "duration_sec": 6.2}],
        [
            {
                "stage": "expert_initial",
                "name": "trend_expert",
                "started_at": "2026-06-14T01:00:00Z",
                "provider_model": "qwen3-14b-trade",
                "duration_kind": "shared_wall_time",
                "duration_sec": 2.0,
                "shared_batch_call": True,
            },
            {
                "stage": "expert_initial",
                "name": "momentum_expert",
                "started_at": "2026-06-14T01:00:00Z",
                "provider_model": "qwen3-14b-trade",
                "duration_kind": "shared_wall_time",
                "duration_sec": 2.0,
                "shared_batch_call": True,
            },
            {
                "stage": "expert_initial",
                "name": "sentiment_expert",
                "started_at": "2026-06-14T01:00:02Z",
                "provider_model": "deepseek-r1-14b-risk",
                "duration_kind": "shared_wall_time",
                "duration_sec": 4.0,
                "shared_batch_call": True,
            },
            {
                "stage": "expert_initial",
                "name": "position_expert",
                "started_at": "2026-06-14T01:00:02Z",
                "provider_model": "deepseek-r1-14b-risk",
                "duration_kind": "shared_wall_time",
                "duration_sec": 4.0,
                "shared_batch_call": True,
            },
            {
                "stage": "expert_initial",
                "name": "risk_expert",
                "started_at": "2026-06-14T01:00:02Z",
                "provider_model": "deepseek-r1-14b-risk",
                "duration_kind": "shared_wall_time",
                "duration_sec": 4.0,
                "shared_batch_call": True,
            },
            {"stage": "decision_maker", "name": "decision_maker", "duration_sec": 1.5},
        ],
    )

    assert summary["stage_duration_sec"] == 6.2
    assert summary["raw_model_duration_sec"] == 17.5
    assert summary["model_duration_sec"] == 7.5
    assert summary["shared_batch_total_duration_sec"] == 6.0
    assert summary["shared_batch_duration_sec"] == 4.0
    assert summary["shared_batch_call_count"] == 2


@pytest.mark.asyncio
async def test_batch_expert_missing_provider_group_is_repaired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[Any]] = []
    captured_kwargs: list[dict[str, Any]] = []

    class FakeChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            captured_kwargs.append(kwargs)

        async def ainvoke(self, messages: list[Any]) -> SimpleNamespace:
            calls.append(messages)
            if len(calls) == 1:
                return SimpleNamespace(
                    content=(
                        '{"experts":{"sentiment_expert":{"action":"hold",'
                        '"confidence":0.41,"reasoning":"情绪中性先观望",'
                        '"position_size_pct":0,"suggested_leverage":1,'
                        '"stop_loss_pct":0.05,"take_profit_pct":0.1,'
                        '"cross_check_for":null}}}'
                    )
                )
            return SimpleNamespace(
                content=(
                    '{"experts":{'
                    '"position_expert":{"action":"hold","confidence":0.42,'
                    '"reasoning":"无持仓不操作","position_size_pct":0,'
                    '"suggested_leverage":1,"stop_loss_pct":0.05,'
                    '"take_profit_pct":0.1,"cross_check_for":null},'
                    '"risk_expert":{"action":"hold","confidence":0.43,'
                    '"reasoning":"无硬风险观望","position_size_pct":0,'
                    '"suggested_leverage":1,"stop_loss_pct":0.05,'
                    '"take_profit_pct":0.1,"cross_check_for":null}}}'
                )
            )

    monkeypatch.setattr("ai_brain.llm_agent.ChatOpenAI", FakeChatOpenAI)
    agent = LLMAgent(
        name="sentiment_expert",
        api_config={
            "api_base": LOCAL_QWEN_TEST_BASE,
            "api_key": "test-key",
            "model": "qwen3-14b-trade",
            "role": "short_timeseries",
        },
    )
    await agent.initialize()

    decisions = await agent.decide_batch_experts(
        FeatureVector(symbol="BTC/USDT"),
        {},
        ["sentiment_expert", "position_expert", "risk_expert"],
    )

    assert set(decisions) == {"sentiment_expert", "position_expert", "risk_expert"}
    assert len(calls) == 2
    json_kwargs = [kwargs for kwargs in captured_kwargs if kwargs.get("model_kwargs")]
    assert json_kwargs == []
    assert all(
        kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
        for kwargs in captured_kwargs
    )
    assert decisions["risk_expert"].raw_response["batch_repair_retry"] is True
    assert not decisions["risk_expert"].raw_response.get("batch_expert_fallback")


@pytest.mark.asyncio
async def test_batch_expert_missing_after_repair_raises_for_independent_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[Any]] = []

    class FakeChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        async def ainvoke(self, messages: list[Any]) -> SimpleNamespace:
            calls.append(messages)
            return SimpleNamespace(
                content=(
                    '{"experts":{"sentiment_expert":{"action":"hold",'
                    '"confidence":0.41,"reasoning":"情绪中性先观望",'
                    '"position_size_pct":0,"suggested_leverage":1,'
                    '"stop_loss_pct":0.05,"take_profit_pct":0.1,'
                    '"cross_check_for":null}}}'
                )
            )

    monkeypatch.setattr("ai_brain.llm_agent.ChatOpenAI", FakeChatOpenAI)
    agent = LLMAgent(
        name="sentiment_expert",
        api_config={
            "api_base": LOCAL_QWEN_TEST_BASE,
            "api_key": "test-key",
            "model": "qwen3-14b-trade",
            "role": "short_timeseries",
        },
    )
    await agent.initialize()

    with pytest.raises(LLMResponseParseError, match="incomplete after repair"):
        await agent.decide_batch_experts(
            FeatureVector(symbol="BTC/USDT"),
            {},
            ["sentiment_expert", "position_expert", "risk_expert"],
        )

    assert len(calls) == 2


@pytest.mark.asyncio
async def test_deepseek_r1_batch_json_fails_fast() -> None:
    agent = LLMAgent(
        name="risk_expert",
        api_config={"model": "deepseek-r1-14b-risk", "role": "risk_anomaly"},
    )
    agent._model_name = "deepseek-r1-14b-risk"
    agent._llm = object()

    with pytest.raises(LLMResponseParseError, match="batch expert JSON is disabled"):
        await agent.decide_batch_experts(
            FeatureVector(symbol="BTC/USDT"),
            {},
            ["risk_expert"],
        )


class _BatchFormatFailingExpert(AbstractAIModel):
    calls = 0
    individual_calls = 0

    def __init__(self, name: str) -> None:
        self.name = name
        self._llm = object()
        self._model_name = "qwen3-32b-trade"

    async def initialize(self) -> None:
        return None

    async def decide(
        self,
        features: FeatureVector,
        context: dict[str, Any],
    ) -> DecisionOutput:
        type(self).individual_calls += 1
        assert context.get("_force_independent_expert") is True
        assert context.get("_force_fast_independent_expert") is True
        assert context.get("_provider_independent_expert_mode") is True
        return DecisionOutput(
            model_name=self.name,
            symbol=features.symbol,
            action=Action.HOLD,
            confidence=0.52,
            reasoning="format failure independent retry ok",
            raw_response={"provider_model": self._model_name},
            feature_snapshot=features.to_dict(),
        )

    async def decide_batch_experts(
        self,
        features: FeatureVector,
        context: dict[str, Any],
        expert_names: list[str],
    ) -> dict[str, DecisionOutput]:
        type(self).calls += 1
        raise RuntimeError('Could not extract valid JSON from: {"experts":')

    def _local_expert_fallback(
        self,
        features: FeatureVector,
        context: dict[str, Any],
        error: str,
    ) -> DecisionOutput:
        return DecisionOutput(
            model_name=self.name,
            symbol=features.symbol,
            action=Action.HOLD,
            confidence=0.1,
            reasoning=f"local fallback: {error}",
            raw_response={"local_fallback_called": True},
            feature_snapshot=features.to_dict(),
        )

    async def shutdown(self) -> None:
        return None


class _BatchFailingIndividualSuccessExpert(AbstractAIModel):
    batch_calls = 0
    individual_calls = 0

    def __init__(self, name: str) -> None:
        self.name = name
        self._llm = object()
        self._model_name = "qwen3-14b-trade"

    async def initialize(self) -> None:
        return None

    async def decide(
        self,
        features: FeatureVector,
        context: dict[str, Any],
    ) -> DecisionOutput:
        type(self).individual_calls += 1
        assert context.get("_force_independent_expert") is True
        assert context.get("_force_fast_independent_expert") is True
        assert context.get("_provider_independent_expert_mode") is True
        return DecisionOutput(
            model_name=self.name,
            symbol=features.symbol,
            action=Action.HOLD,
            confidence=0.55,
            reasoning="independent retry ok",
            raw_response={"provider_model": self._model_name},
            feature_snapshot=features.to_dict(),
        )

    async def decide_batch_experts(
        self,
        features: FeatureVector,
        context: dict[str, Any],
        expert_names: list[str],
    ) -> dict[str, DecisionOutput]:
        type(self).batch_calls += 1
        raise RuntimeError('Could not extract valid JSON from: {"experts":')

    async def shutdown(self) -> None:
        return None


class _ProviderBatchExpert(AbstractAIModel):
    batch_calls: list[tuple[str, tuple[str, ...]]] = []
    individual_calls: list[tuple[str, str]] = []

    def __init__(
        self,
        name: str,
        *,
        base_url: str,
        model_name: str,
        fail_batch: bool = False,
        allow_individual: bool = False,
    ) -> None:
        self.name = name
        self._llm = object()
        self._base_url = base_url
        self._model_name = model_name
        self.fail_batch = fail_batch
        self.allow_individual = allow_individual

    async def initialize(self) -> None:
        return None

    async def decide(
        self,
        features: FeatureVector,
        context: dict[str, Any],
    ) -> DecisionOutput:
        if not self.allow_individual:
            raise AssertionError("provider batch tests should not call individual decide")
        assert context.get("_force_independent_expert") is True
        assert context.get("_force_fast_independent_expert") is True
        assert context.get("_provider_independent_expert_mode")
        type(self).individual_calls.append((self._model_name, self.name))
        return DecisionOutput(
            model_name=self.name,
            symbol=features.symbol,
            action=Action.HOLD,
            confidence=0.44,
            reasoning=f"{self._model_name} provider independent hold",
            raw_response={"provider_model": self._model_name},
            feature_snapshot=features.to_dict(),
        )

    async def decide_batch_experts(
        self,
        features: FeatureVector,
        context: dict[str, Any],
        expert_names: list[str],
    ) -> dict[str, DecisionOutput]:
        type(self).batch_calls.append((self._model_name, tuple(expert_names)))
        if self.fail_batch:
            raise RuntimeError('Could not extract valid JSON from: {"experts":')
        return {
            name: DecisionOutput(
                model_name=name,
                symbol=features.symbol,
                action=Action.HOLD,
                confidence=0.42,
                reasoning=f"{self._model_name} provider batch hold",
                raw_response={"provider_model": self._model_name, "batch_expert": True},
                feature_snapshot=features.to_dict(),
            )
            for name in expert_names
        }

    def _local_expert_fallback(
        self,
        features: FeatureVector,
        context: dict[str, Any],
        error: str,
    ) -> DecisionOutput:
        return DecisionOutput(
            model_name=self.name,
            symbol=features.symbol,
            action=Action.HOLD,
            confidence=0.1,
            reasoning=f"local fallback: {error}",
            raw_response={"provider_model": self._model_name, "local_fallback_called": True},
            feature_snapshot=features.to_dict(),
        )

    async def shutdown(self) -> None:
        return None


class _BatchTimeoutExpert(AbstractAIModel):
    batch_calls = 0
    individual_calls = 0

    def __init__(self, name: str) -> None:
        self.name = name
        self._llm = object()
        self._model_name = "qwen3-14b-trade"

    async def initialize(self) -> None:
        return None

    async def decide(
        self,
        features: FeatureVector,
        context: dict[str, Any],
    ) -> DecisionOutput:
        type(self).individual_calls += 1
        assert context.get("_force_independent_expert") is True
        assert context.get("_force_fast_independent_expert") is True
        assert context.get("_provider_independent_expert_mode") is True
        return DecisionOutput(
            model_name=self.name,
            symbol=features.symbol,
            action=Action.HOLD,
            confidence=0.51,
            reasoning="timeout independent retry ok",
            raw_response={"provider_model": self._model_name},
            feature_snapshot=features.to_dict(),
        )

    async def decide_batch_experts(
        self,
        features: FeatureVector,
        context: dict[str, Any],
        expert_names: list[str],
    ) -> dict[str, DecisionOutput]:
        type(self).batch_calls += 1
        raise TimeoutError()

    def _local_expert_fallback(
        self,
        features: FeatureVector,
        context: dict[str, Any],
        error: str,
    ) -> DecisionOutput:
        return DecisionOutput(
            model_name=self.name,
            symbol=features.symbol,
            action=Action.HOLD,
            confidence=0.1,
            reasoning=f"timeout local fallback: {error}",
            raw_response={"provider_model": self._model_name, "local_fallback_called": True},
            feature_snapshot=features.to_dict(),
        )

    async def shutdown(self) -> None:
        return None


@pytest.mark.asyncio
async def test_batch_format_failure_disables_batch_but_not_real_experts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "ai_batch_experts_enabled", True)
    monkeypatch.setattr(settings, "ai_batch_expert_circuit_breaker_seconds", 0.0)
    monkeypatch.setattr(
        settings,
        "ai_batch_expert_format_failure_circuit_breaker_seconds",
        60.0,
    )
    _BatchFormatFailingExpert.calls = 0
    _BatchFormatFailingExpert.individual_calls = 0
    registry = ModelRegistry()
    for name in (
        "trend_expert",
        "momentum_expert",
        "sentiment_expert",
        "position_expert",
        "risk_expert",
    ):
        registry.register(_BatchFormatFailingExpert(name))

    first_context: dict[str, Any] = {}
    first = await registry.decide_all(FeatureVector(symbol="BTC/USDT"), first_context)

    assert _BatchFormatFailingExpert.calls == 1
    assert _BatchFormatFailingExpert.individual_calls == 5
    assert set(first) == {
        "trend_expert",
        "momentum_expert",
        "sentiment_expert",
        "position_expert",
        "risk_expert",
    }
    assert {row["status"] for row in first_context["_model_timings"]} == {"completed"}

    second_context: dict[str, Any] = {}
    second = await registry.decide_all(FeatureVector(symbol="BTC/USDT"), second_context)

    assert _BatchFormatFailingExpert.calls == 1
    assert _BatchFormatFailingExpert.individual_calls == 10
    assert set(second) == set(first)
    assert {row["status"] for row in second_context["_model_timings"]} == {"completed"}


@pytest.mark.asyncio
async def test_batch_failure_retries_real_individual_experts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "ai_batch_experts_enabled", True)
    monkeypatch.setattr(settings, "ai_batch_expert_circuit_breaker_seconds", 0.0)
    monkeypatch.setattr(
        settings,
        "ai_batch_expert_format_failure_circuit_breaker_seconds",
        60.0,
    )
    _BatchFailingIndividualSuccessExpert.batch_calls = 0
    _BatchFailingIndividualSuccessExpert.individual_calls = 0
    registry = ModelRegistry()
    for name in (
        "trend_expert",
        "momentum_expert",
        "sentiment_expert",
        "position_expert",
        "risk_expert",
    ):
        registry.register(_BatchFailingIndividualSuccessExpert(name))

    context: dict[str, Any] = {}
    decisions = await registry.decide_all(FeatureVector(symbol="BTC/USDT"), context)

    assert set(decisions) == {
        "trend_expert",
        "momentum_expert",
        "sentiment_expert",
        "position_expert",
        "risk_expert",
    }
    assert _BatchFailingIndividualSuccessExpert.batch_calls == 1
    assert _BatchFailingIndividualSuccessExpert.individual_calls == 5
    assert {row["status"] for row in context["_model_timings"]} == {"completed"}
    assert all(row["stage"] == "expert_independent_provider" for row in context["_model_timings"])
    assert all(
        decision.raw_response.get("provider_independent_expert_mode")
        for decision in decisions.values()
    )

    second_context: dict[str, Any] = {}
    await registry.decide_all(FeatureVector(symbol="BTC/USDT"), second_context)

    assert _BatchFailingIndividualSuccessExpert.batch_calls == 1
    assert _BatchFailingIndividualSuccessExpert.individual_calls == 10
    assert {row["status"] for row in second_context["_model_timings"]} == {"completed"}
    assert all(
        row["stage"] == "expert_independent_provider" for row in second_context["_model_timings"]
    )


@pytest.mark.asyncio
async def test_batch_timeout_uses_bounded_independent_retry_before_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "ai_batch_experts_enabled", True)
    monkeypatch.setattr(settings, "ai_batch_expert_circuit_breaker_seconds", 60.0)
    _BatchTimeoutExpert.batch_calls = 0
    _BatchTimeoutExpert.individual_calls = 0
    registry = ModelRegistry()
    for name in (
        "trend_expert",
        "momentum_expert",
        "sentiment_expert",
        "position_expert",
        "risk_expert",
    ):
        registry.register(_BatchTimeoutExpert(name))

    context: dict[str, Any] = {}
    decisions = await registry.decide_all(FeatureVector(symbol="BTC/USDT"), context)

    assert set(decisions) == {
        "trend_expert",
        "momentum_expert",
        "sentiment_expert",
        "position_expert",
        "risk_expert",
    }
    assert _BatchTimeoutExpert.batch_calls == 1
    assert _BatchTimeoutExpert.individual_calls == 5
    assert {row["status"] for row in context["_model_timings"]} == {"completed"}
    assert all(float(row["duration_sec"]) > 0 for row in context["_model_timings"])
    assert all(
        decision.raw_response.get("provider_independent_expert_mode")
        for decision in decisions.values()
    )


@pytest.mark.asyncio
async def test_batch_timeout_activates_minimum_circuit_breaker_when_config_is_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "ai_batch_experts_enabled", True)
    monkeypatch.setattr(settings, "ai_batch_expert_circuit_breaker_seconds", 0.0)
    _BatchTimeoutExpert.batch_calls = 0
    _BatchTimeoutExpert.individual_calls = 0
    registry = ModelRegistry()
    for name in (
        "trend_expert",
        "momentum_expert",
        "sentiment_expert",
        "position_expert",
        "risk_expert",
    ):
        registry.register(_BatchTimeoutExpert(name))

    await registry.decide_all(FeatureVector(symbol="BTC/USDT"), {})
    second_context: dict[str, Any] = {}
    await registry.decide_all(FeatureVector(symbol="BTC/USDT"), second_context)

    assert _BatchTimeoutExpert.batch_calls == 1
    assert _BatchTimeoutExpert.individual_calls == 10
    assert {row["status"] for row in second_context["_model_timings"]} == {"completed"}


@pytest.mark.asyncio
async def test_independent_provider_retry_uses_configured_expert_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "ai_batch_experts_enabled", True)
    monkeypatch.setattr(settings, "ai_expert_timeout_seconds", 30.0)
    _ProviderBatchExpert.batch_calls = []
    _ProviderBatchExpert.individual_calls = []
    registry = ModelRegistry()
    for name in ("sentiment_expert", "position_expert", "risk_expert"):
        registry.register(
            _ProviderBatchExpert(
                name,
                base_url=LOCAL_DEEPSEEK_TEST_BASE,
                model_name="deepseek-r1-14b-risk",
                allow_individual=True,
            )
        )

    context: dict[str, Any] = {}
    await registry.decide_all(FeatureVector(symbol="BTC/USDT"), context)

    assert _ProviderBatchExpert.batch_calls == []
    assert _ProviderBatchExpert.individual_calls == [
        ("deepseek-r1-14b-risk", "sentiment_expert"),
        ("deepseek-r1-14b-risk", "position_expert"),
        ("deepseek-r1-14b-risk", "risk_expert"),
    ]
    timeout_by_name = {row["name"]: row["timeout_seconds"] for row in context["_model_timings"]}
    assert timeout_by_name == {
        "sentiment_expert": 60.0,
        "position_expert": 60.0,
        "risk_expert": 60.0,
    }
    assert all(timeout >= settings.ai_expert_timeout_seconds for timeout in timeout_by_name.values())


@pytest.mark.asyncio
async def test_batch_experts_are_grouped_by_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "ai_batch_experts_enabled", True)
    _ProviderBatchExpert.batch_calls = []
    _ProviderBatchExpert.individual_calls = []
    registry = ModelRegistry()
    for name in ("trend_expert", "momentum_expert"):
        registry.register(
            _ProviderBatchExpert(
                name,
                base_url=LOCAL_QWEN_TEST_BASE,
                model_name="qwen3-14b-trade",
            )
        )
    for name in ("sentiment_expert", "position_expert", "risk_expert"):
        registry.register(
            _ProviderBatchExpert(
                name,
                base_url=LOCAL_DEEPSEEK_TEST_BASE,
                model_name="deepseek-r1-14b-risk",
                allow_individual=True,
            )
        )

    context: dict[str, Any] = {}
    decisions = await registry.decide_all(FeatureVector(symbol="BTC/USDT"), context)

    assert set(decisions) == {
        "trend_expert",
        "momentum_expert",
        "sentiment_expert",
        "position_expert",
        "risk_expert",
    }
    assert _ProviderBatchExpert.batch_calls == [
        (
            "qwen3-14b-trade",
            (
                "trend_expert",
                "momentum_expert",
            ),
        ),
    ]
    assert _ProviderBatchExpert.individual_calls == [
        ("deepseek-r1-14b-risk", "sentiment_expert"),
        ("deepseek-r1-14b-risk", "position_expert"),
        ("deepseek-r1-14b-risk", "risk_expert"),
    ]
    timings_by_name = {row["name"]: row for row in context["_model_timings"]}
    assert timings_by_name["trend_expert"]["provider_model"] == "qwen3-14b-trade"
    assert timings_by_name["trend_expert"]["batch_model_count"] == 2
    assert timings_by_name["trend_expert"]["batch_provider_group_count"] == 2
    assert timings_by_name["risk_expert"]["provider_model"] == "deepseek-r1-14b-risk"
    assert timings_by_name["risk_expert"]["stage"] == "expert_independent_provider"
    assert timings_by_name["risk_expert"]["status"] == "completed"
    assert timings_by_name["risk_expert"]["batch_expert"] is False
    assert timings_by_name["risk_expert"]["shared_batch_call"] is False
    assert decisions["risk_expert"].raw_response["provider_independent_expert_mode"] is True
    assert not decisions["risk_expert"].raw_response.get("batch_failure_independent_retry")


@pytest.mark.asyncio
async def test_batch_expert_circuit_breaker_is_provider_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "ai_batch_experts_enabled", True)
    monkeypatch.setattr(settings, "ai_batch_expert_circuit_breaker_seconds", 0.0)
    monkeypatch.setattr(
        settings,
        "ai_batch_expert_format_failure_circuit_breaker_seconds",
        60.0,
    )
    _ProviderBatchExpert.batch_calls = []
    registry = ModelRegistry()
    for name in ("trend_expert", "momentum_expert"):
        registry.register(
            _ProviderBatchExpert(
                name,
                base_url=LOCAL_QWEN_TEST_BASE,
                model_name="qwen3-14b-trade",
            )
        )
    registry.register(
        _ProviderBatchExpert(
            "sentiment_expert",
            base_url=LOCAL_DEEPSEEK_TEST_BASE,
            model_name="qwen2.5-risk-14b",
            fail_batch=True,
        )
    )
    for name in ("position_expert", "risk_expert"):
        registry.register(
            _ProviderBatchExpert(
                name,
                base_url=LOCAL_DEEPSEEK_TEST_BASE,
                model_name="qwen2.5-risk-14b",
            )
        )

    first_context: dict[str, Any] = {}
    await registry.decide_all(FeatureVector(symbol="BTC/USDT"), first_context)
    second_context: dict[str, Any] = {}
    await registry.decide_all(FeatureVector(symbol="BTC/USDT"), second_context)

    assert _ProviderBatchExpert.batch_calls == [
        ("qwen3-14b-trade", ("trend_expert", "momentum_expert")),
        (
            "qwen2.5-risk-14b",
            ("sentiment_expert", "position_expert", "risk_expert"),
        ),
        ("qwen3-14b-trade", ("trend_expert", "momentum_expert")),
    ]
    second_timings = {row["name"]: row for row in second_context["_model_timings"]}
    assert second_timings["trend_expert"]["status"] == "completed"
    assert second_timings["sentiment_expert"]["status"] == "independent_provider_fallback"
    assert second_timings["sentiment_expert"]["independent_retry_status"] == (
        "independent_provider_failed"
    )
    assert second_timings["sentiment_expert"]["provider_model"] == "qwen2.5-risk-14b"
