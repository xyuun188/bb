from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ai_brain.base_model import Action, DecisionOutput
from services.market_decision_risk_assessment import (
    MarketDecisionRiskAssessmentPolicy,
    expert_analysis_entry_block_reason,
)


def _completed_timings() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "status": "completed",
            "duration_sec": 1.2,
            "provider_model": "qwen3-32b-trade",
        }
        for name in (
            "trend_expert",
            "momentum_expert",
            "sentiment_expert",
            "position_expert",
            "risk_expert",
        )
    ]


def _decision(raw_response: dict[str, Any] | None = None) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.LONG,
        confidence=0.8,
        reasoning="entry",
        raw_response=(
            {"model_timings": _completed_timings()} if raw_response is None else raw_response
        ),
    )


class _RiskEngine:
    def __init__(self, assessment: Any) -> None:
        self.assessment = assessment
        self.calls: list[dict[str, Any]] = []

    def assess(self, decision: DecisionOutput, **kwargs: Any) -> Any:
        self.calls.append({"decision": decision, **kwargs})
        return self.assessment


@pytest.mark.asyncio
async def test_market_decision_risk_assessment_filters_model_positions_and_features() -> None:
    assessment = SimpleNamespace(approved=True, decision=None, rejection_reason="")
    risk_engine = _RiskEngine(assessment)
    balance_calls: list[str] = []
    false_positive_calls: list[Any] = []

    async def account_balance(model_name: str) -> float:
        balance_calls.append(model_name)
        return 1234.5

    async def false_positive(decision: DecisionOutput, reason: str | None, result: Any) -> bool:
        false_positive_calls.append((decision, reason, result))
        return False

    fv = SimpleNamespace(
        recent_headlines=["news"],
        returns_1=-0.01,
        volume_ratio=1.7,
        adx_14=22.0,
    )
    result = await MarketDecisionRiskAssessmentPolicy(
        risk_engine=risk_engine,
        account_balance_provider=account_balance,
        false_positive_checker=false_positive,
    ).assess(
        decision=_decision(),
        model_name="ensemble_trader",
        open_positions=[
            {"model_name": "ensemble_trader", "symbol": "BTC/USDT"},
            {"model_name": "other", "symbol": "ETH/USDT"},
        ],
        feature_vector=fv,
    )

    assert result is assessment
    assert balance_calls == ["ensemble_trader"]
    assert false_positive_calls == []
    call = risk_engine.calls[0]
    assert call["current_positions"] == [{"model_name": "ensemble_trader", "symbol": "BTC/USDT"}]
    assert call["account_balance"] == 1234.5
    assert call["headlines"] == ["news"]
    assert call["price_change_1m"] == -0.01
    assert call["volume_ratio"] == 1.7
    assert call["adx_14"] == 22.0


@pytest.mark.asyncio
async def test_market_decision_risk_assessment_rescues_false_positive_rejection() -> None:
    assessment = SimpleNamespace(
        approved=False,
        decision=None,
        rejection_reason="black swan",
    )
    risk_engine = _RiskEngine(assessment)
    decision = _decision()

    async def account_balance(model_name: str) -> float:
        return 1000.0

    async def false_positive(
        checked_decision: DecisionOutput,
        reason: str | None,
        result: Any,
    ) -> bool:
        assert checked_decision is decision
        assert reason == "black swan"
        assert result is assessment
        return True

    result = await MarketDecisionRiskAssessmentPolicy(
        risk_engine=risk_engine,
        account_balance_provider=account_balance,
        false_positive_checker=false_positive,
    ).assess(
        decision=decision,
        model_name="ensemble_trader",
        open_positions=[],
        feature_vector=SimpleNamespace(),
    )

    assert result is assessment
    assert result.approved is True
    assert result.decision is decision
    assert result.rejection_reason == ""


@pytest.mark.asyncio
async def test_market_decision_risk_assessment_keeps_real_rejection() -> None:
    assessment = SimpleNamespace(
        approved=False,
        decision=None,
        rejection_reason="risk blocked",
    )
    risk_engine = _RiskEngine(assessment)

    async def account_balance(model_name: str) -> float:
        return 1000.0

    async def false_positive(decision: DecisionOutput, reason: str | None, result: Any) -> bool:
        return False

    result = await MarketDecisionRiskAssessmentPolicy(
        risk_engine=risk_engine,
        account_balance_provider=account_balance,
        false_positive_checker=false_positive,
    ).assess(
        decision=_decision(),
        model_name="ensemble_trader",
        open_positions=[],
        feature_vector=SimpleNamespace(),
    )

    assert result.approved is False
    assert result.rejection_reason == "risk blocked"


@pytest.mark.asyncio
async def test_market_decision_risk_assessment_blocks_entry_when_experts_are_fallback() -> None:
    assessment = SimpleNamespace(approved=True, decision=None, rejection_reason="")
    risk_engine = _RiskEngine(assessment)
    false_positive_calls: list[Any] = []
    decision = _decision(
        {
            "model_timings": [
                {
                    "name": name,
                    "status": "circuit_breaker_fallback",
                    "duration_sec": 0.001,
                    "provider_model": "qwen3-32b-trade",
                    "reason": "recent JSON failure",
                }
                for name in (
                    "trend_expert",
                    "momentum_expert",
                    "sentiment_expert",
                    "position_expert",
                    "risk_expert",
                )
            ]
        }
    )

    async def account_balance(model_name: str) -> float:
        return 1000.0

    async def false_positive(
        checked_decision: DecisionOutput,
        reason: str | None,
        result: Any,
    ) -> bool:
        false_positive_calls.append((checked_decision, reason, result))
        return True

    result = await MarketDecisionRiskAssessmentPolicy(
        risk_engine=risk_engine,
        account_balance_provider=account_balance,
        false_positive_checker=false_positive,
    ).assess(
        decision=decision,
        model_name="ensemble_trader",
        open_positions=[],
        feature_vector=SimpleNamespace(),
    )

    assert result.approved is False
    assert result.decision is None
    assert "expert_integrity" in result.rejection_reason
    assert "circuit_breaker_fallback" in result.rejection_reason
    assert false_positive_calls == []


def test_expert_analysis_entry_block_reason_allows_completed_experts() -> None:
    assert expert_analysis_entry_block_reason(_decision()) is None


def test_expert_analysis_entry_block_reason_blocks_missing_timings() -> None:
    reason = expert_analysis_entry_block_reason(_decision({}))

    assert reason is not None
    assert "expert_integrity" in reason


def test_expert_analysis_entry_block_reason_allows_independent_provider_completed() -> None:
    raw = {
        "model_timings": [
            {
                "name": "trend_expert",
                "status": "completed",
                "provider_model": "qwen3-14b-trade",
                "provider_independent_expert_mode": True,
                "duration_sec": 2.1,
            },
            {
                "name": "momentum_expert",
                "status": "completed",
                "provider_model": "qwen3-14b-trade",
                "provider_independent_expert_mode": True,
                "duration_sec": 2.0,
            },
            {
                "name": "sentiment_expert",
                "status": "completed",
                "provider_model": "deepseek-r1-14b-risk",
                "provider_independent_expert_mode": True,
                "duration_sec": 8.4,
            },
            {
                "name": "position_expert",
                "status": "completed",
                "provider_model": "deepseek-r1-14b-risk",
                "provider_independent_expert_mode": True,
                "duration_sec": 8.8,
            },
            {
                "name": "risk_expert",
                "status": "completed",
                "provider_model": "deepseek-r1-14b-risk",
                "provider_independent_expert_mode": True,
                "duration_sec": 9.1,
            },
        ]
    }

    assert expert_analysis_entry_block_reason(_decision(raw)) is None


def test_expert_analysis_entry_block_reason_blocks_local_fallback_even_when_completed() -> None:
    timings = _completed_timings()
    timings[-1]["local_fallback"] = True

    reason = expert_analysis_entry_block_reason(_decision({"model_timings": timings}))

    assert reason is not None
    assert "expert_integrity" in reason
    assert "risk_expert" in reason


def test_expert_analysis_entry_block_reason_allows_balanced_probe_missing_non_core() -> None:
    raw = {
        "model_timings": [
            {"name": "trend_expert", "status": "completed", "provider_model": "qwen3-14b"},
            {"name": "momentum_expert", "status": "completed", "provider_model": "qwen3-14b"},
            {
                "name": "sentiment_expert",
                "status": "partial_batch_fallback",
                "provider_model": "qwen3-14b",
            },
            {
                "name": "position_expert",
                "status": "completed",
                "provider_model": "deepseek-r1-14b",
            },
            {"name": "risk_expert", "status": "completed", "provider_model": "deepseek-r1-14b"},
        ]
    }
    decision = _decision(raw)
    decision.position_size_pct = 0.05

    reason = expert_analysis_entry_block_reason(
        decision,
        strategy_mode_context={
            "expert_integrity_mode": "balanced_probe_allow_one_non_core_missing",
        },
    )

    assert reason is None
    assert decision.position_size_pct == 0.018
    assert decision.raw_response["expert_integrity_probe"]["applied"] is True
    assert decision.raw_response["expert_integrity_probe"]["missing_expert"] == "sentiment_expert"


def test_expert_analysis_entry_block_reason_keeps_core_expert_strict() -> None:
    raw = {
        "model_timings": [
            {"name": "trend_expert", "status": "timeout", "provider_model": "qwen3-14b"},
            {"name": "momentum_expert", "status": "completed", "provider_model": "qwen3-14b"},
            {"name": "sentiment_expert", "status": "completed", "provider_model": "qwen3-14b"},
            {
                "name": "position_expert",
                "status": "completed",
                "provider_model": "deepseek-r1-14b",
            },
            {"name": "risk_expert", "status": "completed", "provider_model": "deepseek-r1-14b"},
        ]
    }

    reason = expert_analysis_entry_block_reason(
        _decision(raw),
        strategy_mode_context={
            "expert_integrity_mode": "balanced_probe_allow_one_non_core_missing",
        },
    )

    assert reason is not None
    assert "trend_expert" in reason
    assert "expert_integrity" in reason
