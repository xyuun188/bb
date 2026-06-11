from __future__ import annotations

from typing import Any

import pytest

from ai_brain.base_model import AbstractAIModel, Action, DecisionOutput
from ai_brain.model_registry import ModelRegistry
from data_feed.feature_vector import FeatureVector


class _FailingModel(AbstractAIModel):
    name = "failing_expert"

    def __init__(self, error_text: str) -> None:
        self.error_text = error_text

    async def initialize(self) -> None:
        return None

    async def decide(
        self,
        features: FeatureVector,
        context: dict[str, Any],
    ) -> DecisionOutput:
        raise RuntimeError(self.error_text)

    async def shutdown(self) -> None:
        return None


class _LocalFallbackModel(AbstractAIModel):
    name = "trend_expert"

    async def initialize(self) -> None:
        return None

    async def decide(
        self,
        features: FeatureVector,
        context: dict[str, Any],
    ) -> DecisionOutput:
        raise AssertionError("fast prefilter should use the local fallback")

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
            confidence=0.11,
            reasoning=f"local fallback used: {error}",
            raw_response={"local_fallback_called": True},
            feature_snapshot=features.to_dict(),
        )

    async def shutdown(self) -> None:
        return None


def _secret_bearing_error() -> tuple[str, str, str]:
    token = "abcdefghi" + "jklmnopqrst" + "uvwxyz123456"
    hidden_value = "plain-credential-value"
    return token, hidden_value, f"Authorization: Bearer {token} failed password={hidden_value}"


@pytest.mark.asyncio
async def test_model_failure_context_is_redacted() -> None:
    token, hidden_value, error_text = _secret_bearing_error()
    registry = ModelRegistry()
    registry.register(_FailingModel(error_text))
    context: dict[str, Any] = {}

    decisions = await registry.decide_all(FeatureVector(symbol="BTC/USDT"), context)

    rendered = str(context)
    assert decisions == {}
    assert token not in rendered
    assert hidden_value not in rendered
    assert "Authorization: ***" in rendered
    assert "password=***" in rendered
    assert context["_model_failures"][0]["expert_name"] == "failing_expert"
    assert context["_model_timings"][0]["status"] == "failed"


@pytest.mark.asyncio
async def test_fast_prefilter_uses_local_fallback_callable() -> None:
    registry = ModelRegistry()
    registry.register(_LocalFallbackModel())
    context: dict[str, Any] = {"_skip_llm_experts_reason": "fast prefilter"}

    decisions = await registry.decide_all(FeatureVector(symbol="BTC/USDT"), context)

    decision = decisions["trend_expert"]
    assert decision.raw_response is not None
    assert decision.raw_response["local_fallback_called"] is True
    assert decision.raw_response["market_fast_prefilter"] is True
    assert decision.raw_response["provider_model"] == "local_fast_prefilter"
    assert context["_model_timings"][0]["status"] == "fast_prefilter"
