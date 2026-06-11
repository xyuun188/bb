from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ai_brain.base_model import Action, DecisionOutput
from services.position_review_risk_assessment import PositionReviewRiskAssessmentPolicy


def _decision() -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.CLOSE_LONG,
        confidence=0.7,
        reasoning="test",
        position_size_pct=0.0,
        suggested_leverage=1.0,
    )


class FakeRiskEngine:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def assess(self, decision: DecisionOutput, **kwargs: Any) -> str:
        self.calls.append({"decision": decision, **kwargs})
        return "assessment"


@pytest.mark.asyncio
async def test_position_review_risk_assessment_filters_model_positions() -> None:
    risk_engine = FakeRiskEngine()
    policy = PositionReviewRiskAssessmentPolicy(risk_engine)
    decision = _decision()

    result = await policy.assess(
        decision=decision,
        model_name="ensemble_trader",
        open_positions=[
            {"model_name": "ensemble_trader", "symbol": "BTC/USDT"},
            {"model_name": "other_model", "symbol": "ETH/USDT"},
        ],
        feature_vector=SimpleNamespace(
            recent_headlines=["risk headline"],
            returns_1=-0.01,
            volume_ratio=1.4,
            adx_14=22.0,
        ),
        account_balance_provider=lambda model_name: _balance(model_name, 456.0),
    )

    assert result == "assessment"
    assert risk_engine.calls == [
        {
            "decision": decision,
            "current_positions": [{"model_name": "ensemble_trader", "symbol": "BTC/USDT"}],
            "account_balance": 456.0,
            "headlines": ["risk headline"],
            "sentiment_scores": [],
            "price_change_1m": -0.01,
            "volume_ratio": 1.4,
            "adx_14": 22.0,
        }
    ]


@pytest.mark.asyncio
async def test_position_review_risk_assessment_uses_feature_defaults() -> None:
    risk_engine = FakeRiskEngine()
    policy = PositionReviewRiskAssessmentPolicy(risk_engine)

    await policy.assess(
        decision=_decision(),
        model_name="ensemble_trader",
        open_positions=[],
        feature_vector=SimpleNamespace(),
        account_balance_provider=lambda model_name: _balance(model_name, 100.0),
    )

    call = risk_engine.calls[0]
    assert call["headlines"] == []
    assert call["price_change_1m"] == 0.0
    assert call["volume_ratio"] == 1.0
    assert call["adx_14"] is None


async def _balance(model_name: str, value: float) -> float:
    assert model_name == "ensemble_trader"
    return value
