from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ai_brain.base_model import Action, DecisionOutput
from services.manual_trade_risk_assessment import ManualTradeRiskAssessmentPolicy


def _decision() -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.LONG,
        confidence=0.8,
        reasoning="manual",
        position_size_pct=0.1,
        suggested_leverage=3.0,
    )


class FakeRiskEngine:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def assess(self, decision: DecisionOutput, **kwargs: Any) -> str:
        self.calls.append({"decision": decision, **kwargs})
        return "assessment"


@pytest.mark.asyncio
async def test_manual_trade_risk_assessment_filters_model_positions() -> None:
    risk_engine = FakeRiskEngine()
    policy = ManualTradeRiskAssessmentPolicy(risk_engine)
    decision = _decision()

    result = await policy.assess(
        decision=decision,
        model_name="ensemble_trader",
        open_positions=[
            {"model_name": "ensemble_trader", "symbol": "BTC/USDT"},
            {"model_name": "other_model", "symbol": "ETH/USDT"},
        ],
        feature_vector=SimpleNamespace(
            recent_headlines=["manual headline"],
            returns_1=0.012,
            volume_ratio=1.3,
            adx_14=18.0,
        ),
        account_balance_provider=lambda model_name: _balance(model_name, 321.0),
    )

    assert result == "assessment"
    assert risk_engine.calls == [
        {
            "decision": decision,
            "current_positions": [{"model_name": "ensemble_trader", "symbol": "BTC/USDT"}],
            "account_balance": 321.0,
        }
    ]


@pytest.mark.asyncio
async def test_manual_trade_risk_assessment_does_not_forward_strategy_features() -> None:
    risk_engine = FakeRiskEngine()
    policy = ManualTradeRiskAssessmentPolicy(risk_engine)

    await policy.assess(
        decision=_decision(),
        model_name="ensemble_trader",
        open_positions=[],
        feature_vector=SimpleNamespace(),
        account_balance_provider=lambda model_name: _balance(model_name, 100.0),
    )

    call = risk_engine.calls[0]
    assert set(call) == {"decision", "current_positions", "account_balance"}


async def _balance(model_name: str, value: float) -> float:
    assert model_name == "ensemble_trader"
    return value
