from __future__ import annotations

from typing import Any

import pytest

from ai_brain.base_model import AbstractAIModel, Action, DecisionOutput
from ai_brain.model_registry import ModelRegistry
from config.settings import settings
from data_feed.feature_vector import FeatureVector

EXPERT_NAMES = (
    "trend_expert",
    "momentum_expert",
    "sentiment_expert",
    "position_expert",
    "risk_expert",
)


class _ConsensusBatchExpert(AbstractAIModel):
    batch_calls = 0
    individual_calls: list[str] = []

    def __init__(self, name: str, independent_actions: dict[str, Action] | None = None) -> None:
        self.name = name
        self.independent_actions = independent_actions or {}
        self._llm = object()
        self._model_name = "qwen3-32b-trade"

    async def initialize(self) -> None:
        return None

    async def decide(
        self,
        features: FeatureVector,
        context: dict[str, Any],
    ) -> DecisionOutput:
        type(self).individual_calls.append(self.name)
        action = self.independent_actions.get(self.name, Action.HOLD)
        return DecisionOutput(
            model_name=self.name,
            symbol=features.symbol,
            action=action,
            confidence=0.72 if action != Action.HOLD else 0.52,
            reasoning=f"{self.name} independent retry",
            position_size_pct=0.03 if action.is_entry() else 0.0,
            suggested_leverage=2.0,
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
        return {
            name: DecisionOutput(
                model_name=name,
                symbol=features.symbol,
                action=Action.HOLD,
                confidence=0.50,
                reasoning="batch low information hold",
                raw_response={"provider_model": self._model_name, "batch_expert": True},
                feature_snapshot=features.to_dict(),
            )
            for name in expert_names
        }

    async def shutdown(self) -> None:
        return None


def _registry(independent_actions: dict[str, Action] | None = None) -> ModelRegistry:
    _ConsensusBatchExpert.batch_calls = 0
    _ConsensusBatchExpert.individual_calls = []
    registry = ModelRegistry()
    for name in EXPERT_NAMES:
        registry.register(_ConsensusBatchExpert(name, independent_actions))
    return registry


def _features(**kwargs: Any) -> FeatureVector:
    values: dict[str, Any] = {
        "symbol": "BTC/USDT",
        "returns_5": 0.006,
        "returns_20": 0.010,
        "price_vs_sma20": 0.012,
        "price_vs_sma50": 0.014,
        "adx_14": 22.0,
        "volume_ratio": 1.35,
    }
    values.update(kwargs)
    return FeatureVector(**values)


def _strong_ml_context() -> dict[str, Any]:
    return {
        "expert_mode": True,
        "ml_signal": {
            "available": True,
            "influence_enabled": True,
            "predictions": [
                {
                    "best_side": "long",
                    "long_expected_return_pct": 0.36,
                    "short_expected_return_pct": -0.04,
                }
            ],
        },
    }


@pytest.mark.asyncio
async def test_low_information_batch_consensus_retries_directional_experts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "ai_batch_experts_enabled", True)
    registry = _registry(
        {
            "trend_expert": Action.LONG,
            "momentum_expert": Action.LONG,
            "sentiment_expert": Action.HOLD,
        }
    )
    context = _strong_ml_context()

    decisions = await registry.decide_all(_features(), context)

    assert _ConsensusBatchExpert.batch_calls == 1
    assert set(_ConsensusBatchExpert.individual_calls) == {
        "trend_expert",
        "momentum_expert",
        "sentiment_expert",
    }
    assert decisions["trend_expert"].action == Action.LONG
    assert decisions["momentum_expert"].action == Action.LONG
    assert decisions["position_expert"].action == Action.HOLD
    assert decisions["trend_expert"].raw_response["independent_expert_retry"] is True
    assert "independent_expert_retry" not in decisions["position_expert"].raw_response

    policy = context["_expert_diversity_policy"]
    assert policy["should_retry"] is True
    assert policy["objective_evidence"]["side"] == "long"
    assert any(
        item.get("stage") == "expert_independent_retry" and item.get("status") == "completed"
        for item in context["_model_timings"]
    )


@pytest.mark.asyncio
async def test_low_information_batch_consensus_does_not_retry_without_objective_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "ai_batch_experts_enabled", True)
    registry = _registry({"trend_expert": Action.LONG})
    context = {"expert_mode": True}

    decisions = await registry.decide_all(FeatureVector(symbol="BTC/USDT"), context)

    assert _ConsensusBatchExpert.batch_calls == 1
    assert _ConsensusBatchExpert.individual_calls == []
    assert all(decision.action == Action.HOLD for decision in decisions.values())
    assert context["_expert_diversity_policy"]["should_retry"] is False


@pytest.mark.asyncio
async def test_hard_wick_risk_explains_batch_hold_consensus_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "ai_batch_experts_enabled", True)
    registry = _registry({"trend_expert": Action.LONG})
    context = _strong_ml_context()

    decisions = await registry.decide_all(
        _features(
            abnormal_wick_count_72h=1, abnormal_wick_max_pct=86.0, abnormal_wick_recent_hours=4.0
        ),
        context,
    )

    assert _ConsensusBatchExpert.individual_calls == []
    assert all(decision.action == Action.HOLD for decision in decisions.values())
    policy = context["_expert_diversity_policy"]
    assert policy["should_retry"] is False
    assert policy["objective_evidence"]["hard_risk"] is True
