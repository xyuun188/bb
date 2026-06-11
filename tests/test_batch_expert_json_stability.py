from __future__ import annotations

from typing import Any

import pytest

from ai_brain.base_model import AbstractAIModel, Action, DecisionOutput
from ai_brain.llm_agent import _extract_json
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
    assert "Each expert must judge only its role" in prompt
    assert "usable positive EV may be small/probe" in prompt
    assert '"action":"hold","confidence":0.50' not in prompt
    assert "daily_target" not in prompt


def test_extract_json_repairs_missing_batch_tail() -> None:
    parsed = _extract_json("""
        ```json
        {"experts":{"trend_expert":{"action":"hold","confidence":0.5},
        "momentum_expert":{"action":"hold","confidence":0.4},
        ```
        """)

    assert parsed["experts"]["trend_expert"]["action"] == "hold"
    assert parsed["experts"]["momentum_expert"]["confidence"] == 0.4


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
