from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ai_brain.base_model import AbstractAIModel, Action, DecisionOutput
from ai_brain.llm_agent import LLMAgent, _extract_json
from ai_brain.model_registry import ModelRegistry
from ai_brain.prompts import build_batch_experts_user_prompt
from config.settings import settings
from data_feed.feature_vector import FeatureVector


def test_batch_expert_prompt_uses_compact_json_contract() -> None:
    prompt = build_batch_experts_user_prompt(
        "symbol=BTC/USDT price=100",
        {"review_positions": True, "open_positions": [{"symbol": "BTC/USDT"}]},
    )

    assert "STRICT_COMPACT_BATCH_JSON_V4" in prompt
    assert (
        "Required experts for THIS call: trend_expert, momentum_expert, sentiment_expert, position_expert, risk_expert"
        in prompt
    )
    assert "Each expert must judge only its role" in prompt
    assert "usable positive EV may be small/probe" in prompt
    assert '"action":"hold","confidence":0.50' not in prompt
    assert "daily_target" not in prompt


def test_batch_expert_prompt_can_scope_to_provider_group() -> None:
    prompt = build_batch_experts_user_prompt(
        "symbol=BTC/USDT price=100",
        {"review_positions": False},
        ["sentiment_expert", "position_expert", "risk_expert"],
    )

    assert (
        "Required experts for THIS call: sentiment_expert, position_expert, risk_expert" in prompt
    )
    assert (
        '"experts":{"sentiment_expert":{...},"position_expert":{...},"risk_expert":{...}}' in prompt
    )
    assert "Do not include these omitted experts: trend_expert, momentum_expert." in prompt


def test_extract_json_repairs_missing_batch_tail() -> None:
    parsed = _extract_json("""
        ```json
        {"experts":{"trend_expert":{"action":"hold","confidence":0.5},
        "momentum_expert":{"action":"hold","confidence":0.4},
        ```
        """)

    assert parsed["experts"]["trend_expert"]["action"] == "hold"
    assert parsed["experts"]["momentum_expert"]["confidence"] == 0.4


@pytest.mark.asyncio
async def test_batch_expert_missing_provider_group_is_repaired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[Any]] = []

    class FakeChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

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
            "api_base": "http://127.0.0.1:8003/v1",
            "api_key": "test-key",
            "model": "deepseek-r1-14b-risk",
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
    assert decisions["risk_expert"].raw_response["batch_repair_retry"] is True
    assert not decisions["risk_expert"].raw_response.get("batch_expert_fallback")


class _BatchFormatFailingExpert(AbstractAIModel):
    calls = 0

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
        raise AssertionError("batch path should call decide_batch_experts")

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


class _ProviderBatchExpert(AbstractAIModel):
    batch_calls: list[tuple[str, tuple[str, ...]]] = []

    def __init__(
        self,
        name: str,
        *,
        base_url: str,
        model_name: str,
        fail_batch: bool = False,
    ) -> None:
        self.name = name
        self._llm = object()
        self._base_url = base_url
        self._model_name = model_name
        self.fail_batch = fail_batch

    async def initialize(self) -> None:
        return None

    async def decide(
        self,
        features: FeatureVector,
        context: dict[str, Any],
    ) -> DecisionOutput:
        raise AssertionError("provider batch tests should not call individual decide")

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


@pytest.mark.asyncio
async def test_batch_format_failure_activates_circuit_breaker(
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
    assert set(first) == {
        "trend_expert",
        "momentum_expert",
        "sentiment_expert",
        "position_expert",
        "risk_expert",
    }
    assert first_context["_model_timings"][0]["status"] == "batch_fallback"

    second_context: dict[str, Any] = {}
    second = await registry.decide_all(FeatureVector(symbol="BTC/USDT"), second_context)

    assert _BatchFormatFailingExpert.calls == 1
    assert set(second) == set(first)
    assert second_context["_model_timings"][0]["status"] == "circuit_breaker_fallback"


@pytest.mark.asyncio
async def test_batch_experts_are_grouped_by_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "ai_batch_experts_enabled", True)
    _ProviderBatchExpert.batch_calls = []
    registry = ModelRegistry()
    for name in ("trend_expert", "momentum_expert"):
        registry.register(
            _ProviderBatchExpert(
                name,
                base_url="http://127.0.0.1:8000/v1",
                model_name="qwen3-14b-trade",
            )
        )
    for name in ("sentiment_expert", "position_expert", "risk_expert"):
        registry.register(
            _ProviderBatchExpert(
                name,
                base_url="http://127.0.0.1:8003/v1",
                model_name="deepseek-r1-14b-risk",
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
        ("qwen3-14b-trade", ("trend_expert", "momentum_expert")),
        (
            "deepseek-r1-14b-risk",
            ("sentiment_expert", "position_expert", "risk_expert"),
        ),
    ]
    timings_by_name = {row["name"]: row for row in context["_model_timings"]}
    assert timings_by_name["trend_expert"]["provider_model"] == "qwen3-14b-trade"
    assert timings_by_name["trend_expert"]["batch_model_count"] == 2
    assert timings_by_name["risk_expert"]["provider_model"] == "deepseek-r1-14b-risk"
    assert timings_by_name["risk_expert"]["batch_model_count"] == 3
    assert timings_by_name["risk_expert"]["batch_provider_group_count"] == 2


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
                base_url="http://127.0.0.1:8000/v1",
                model_name="qwen3-14b-trade",
            )
        )
    registry.register(
        _ProviderBatchExpert(
            "sentiment_expert",
            base_url="http://127.0.0.1:8003/v1",
            model_name="deepseek-r1-14b-risk",
            fail_batch=True,
        )
    )
    for name in ("position_expert", "risk_expert"):
        registry.register(
            _ProviderBatchExpert(
                name,
                base_url="http://127.0.0.1:8003/v1",
                model_name="deepseek-r1-14b-risk",
            )
        )

    first_context: dict[str, Any] = {}
    await registry.decide_all(FeatureVector(symbol="BTC/USDT"), first_context)
    second_context: dict[str, Any] = {}
    await registry.decide_all(FeatureVector(symbol="BTC/USDT"), second_context)

    assert _ProviderBatchExpert.batch_calls == [
        ("qwen3-14b-trade", ("trend_expert", "momentum_expert")),
        (
            "deepseek-r1-14b-risk",
            ("sentiment_expert", "position_expert", "risk_expert"),
        ),
        ("qwen3-14b-trade", ("trend_expert", "momentum_expert")),
    ]
    second_timings = {row["name"]: row for row in second_context["_model_timings"]}
    assert second_timings["trend_expert"]["status"] == "completed"
    assert second_timings["sentiment_expert"]["status"] == "circuit_breaker_fallback"
    assert second_timings["sentiment_expert"]["provider_model"] == "deepseek-r1-14b-risk"
