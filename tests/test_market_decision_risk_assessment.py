from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ai_brain.base_model import Action, DecisionOutput
from services.market_decision_risk_assessment import MarketDecisionRiskAssessmentPolicy


class _RiskEngine:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def assess(self, decision: DecisionOutput, **kwargs: Any) -> Any:
        self.calls.append({"decision": decision, **kwargs})
        return SimpleNamespace(approved=True, decision=decision, rejection_reason="")


@pytest.mark.asyncio
async def test_market_risk_adapter_uses_only_risk_engine_and_account_state() -> None:
    engine = _RiskEngine()
    decision = DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.LONG,
        confidence=0.1,
        reasoning="expert fields are observation-only",
        raw_response={"model_timings": [{"status": "failed"}]},
    )

    async def balance(_model_name: str) -> float:
        return 1234.5

    result = await MarketDecisionRiskAssessmentPolicy(engine, balance).assess(
        decision=decision,
        model_name="ensemble_trader",
        open_positions=[
            {"model_name": "ensemble_trader", "symbol": "BTC/USDT"},
            {"model_name": "other", "symbol": "ETH/USDT"},
        ],
        feature_vector=SimpleNamespace(returns_1=-0.01, volume_ratio=1.7, adx_14=22.0),
    )
    assert result.approved is True
    assert engine.calls[0]["current_positions"] == [
        {"model_name": "ensemble_trader", "symbol": "BTC/USDT"}
    ]
    assert engine.calls[0]["account_balance"] == 1234.5
