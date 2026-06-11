from __future__ import annotations

from typing import Any

import pytest

from ai_brain.base_model import Action, DecisionOutput
from services.position_review_decision_service import (
    PositionReviewDecisionRequest,
    PositionReviewDecisionService,
)


def _decision() -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.HOLD,
        confidence=0.6,
        reasoning="review",
        raw_response={},
    )


def _request(model_name: str = "ensemble_trader") -> PositionReviewDecisionRequest:
    return PositionReviewDecisionRequest(
        model_name=model_name,
        symbol="BTC/USDT",
        normalized_symbol="BTC/USDT",
        feature_vector=object(),
        open_positions=[{"symbol": "BTC/USDT", "side": "long"}],
        trading_mode="paper",
        position_entry_pause_reason="",
        market_regime_context={"regime": "normal"},
        strategy_mode_context={"mode": "balanced"},
        portfolio_symbol_context={"active": True, "net_pnl": 3.0},
        position_profit_peak_context={"peak_pnl": 5.0},
    )


@pytest.mark.asyncio
async def test_position_review_decision_service_builds_ensemble_context_and_metadata() -> None:
    calls: list[tuple[str, Any]] = []
    decision = _decision()

    async def memory(symbol: str) -> dict[str, Any]:
        calls.append(("memory", symbol))
        return {"expert_memory": {"symbol": symbol}}

    def ml_predictor(_feature_vector: Any) -> dict[str, Any]:
        calls.append(("ml", None))
        return {"ready": True}

    async def local_tools(_feature_vector: Any, ml_signal: dict[str, Any], **kwargs: Any) -> dict:
        calls.append(("local", (ml_signal, kwargs)))
        return {"exit_advice": {"action": "hold"}}

    def position_skills(**kwargs: Any) -> list[str]:
        calls.append(("skills", kwargs))
        return ["skill"]

    def attach(decision_arg: DecisionOutput, **kwargs: Any) -> dict[str, Any]:
        calls.append(("attach", (decision_arg.symbol, kwargs)))
        return decision_arg.raw_response or {}

    async def ensemble_decider(_feature_vector: Any, context: dict[str, Any]) -> tuple[Any, Any]:
        calls.append(("ensemble", context))
        return decision, []

    service = PositionReviewDecisionService(
        default_model_name="ensemble_trader",
        expert_memory_context_provider=memory,
        ml_signal_predictor=ml_predictor,
        local_ai_tools_context_provider=local_tools,
        position_skills_provider=position_skills,
        agent_skills_attacher=attach,
        ensemble_decider=ensemble_decider,
        model_provider=lambda _name: None,
    )

    result = await service.decide(_request())

    assert result is not None
    assert result.decision is decision
    context = next(value for name, value in calls if name == "ensemble")
    assert context["review_positions"] is True
    assert context["expert_memory"] == {"symbol": "BTC/USDT"}
    assert context["ml_signal"] == {"ready": True}
    assert context["local_ai_tools"] == {"exit_advice": {"action": "hold"}}
    assert decision.raw_response["analysis_type"] == "position_review"
    assert decision.raw_response["portfolio_profit_protection"]["active"] is True
    attach_kwargs = next(value for name, value in calls if name == "attach")[1]
    assert attach_kwargs["phase"] == "position_review"
    assert attach_kwargs["skills"] == ["skill"]


@pytest.mark.asyncio
async def test_position_review_decision_service_uses_single_model_context() -> None:
    calls: list[dict[str, Any]] = []

    class Model:
        async def decide(self, _feature_vector: Any, context: dict[str, Any]) -> DecisionOutput:
            calls.append(context)
            return _decision()

    service = PositionReviewDecisionService(
        default_model_name="ensemble_trader",
        expert_memory_context_provider=lambda _symbol: _async_dict({}),
        ml_signal_predictor=lambda _feature_vector: {},
        local_ai_tools_context_provider=lambda *_args, **_kwargs: _async_dict({}),
        position_skills_provider=lambda **_kwargs: [],
        agent_skills_attacher=lambda *_args, **_kwargs: {},
        ensemble_decider=lambda *_args, **_kwargs: _async_tuple((None, [])),
        model_provider=lambda _name: Model(),
    )

    result = await service.decide(_request("manual_model"))

    assert result is not None
    assert calls[0]["review_positions"] is True
    assert "ml_signal" not in calls[0]
    assert calls[0]["portfolio_profit_protection"]["active"] is True


@pytest.mark.asyncio
async def test_position_review_decision_service_returns_none_when_model_missing() -> None:
    service = PositionReviewDecisionService(
        default_model_name="ensemble_trader",
        expert_memory_context_provider=lambda _symbol: _async_dict({}),
        ml_signal_predictor=lambda _feature_vector: {},
        local_ai_tools_context_provider=lambda *_args, **_kwargs: _async_dict({}),
        position_skills_provider=lambda **_kwargs: [],
        agent_skills_attacher=lambda *_args, **_kwargs: {},
        ensemble_decider=lambda *_args, **_kwargs: _async_tuple((None, [])),
        model_provider=lambda _name: None,
    )

    assert await service.decide(_request("missing_model")) is None


async def _async_dict(value: dict[str, Any]) -> dict[str, Any]:
    return value


async def _async_tuple(value: tuple[Any, Any]) -> tuple[Any, Any]:
    return value
