from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from ai_brain.base_model import Action, DecisionOutput
from services.trade_recommendation_contract import (
    attach_initial_trade_recommendation,
    attach_risk_adjusted_trade_recommendation,
    attach_trade_execution_result,
    paper_trade_recommendation_reasons,
    trade_recommendation_snapshot,
)


def _raw() -> dict:
    return {
        "analysis_type": "market",
        "opinions": [
            {
                "model_name": "trend_expert",
                "role": "trend_direction",
                "action": "long",
                "confidence": 0.8,
                "position_size_pct": 0.2,
                "suggested_leverage": 2.0,
                "stop_loss_pct": 0.01,
                "take_profit_pct": 0.02,
                "effective_weight": 1.0,
                "reasoning": "趋势和量价支持做多。",
            },
            {
                "model_name": "risk_expert",
                "role": "risk_anomaly",
                "action": "hold",
                "confidence": 0.6,
                "effective_weight": 0.5,
                "reasoning": "波动偏高，建议缩小仓位。",
            },
        ],
        "opportunity_score": {
            "expected_net_return_pct": 0.4,
            "return_lcb_pct": -0.1,
            "return_uncertainty_pct": 0.2,
            "expected_loss_pct": 0.3,
            "return_distribution_contract": {"horizon_minutes": 30},
        },
        "profit_risk_sizing": {
            "expected_loss_pct": 0.3,
            "planned_stressed_loss_usdt": 1.0,
            "risk_budget_usdt": 1.0,
            "stressed_loss_fraction": 0.01,
        },
    }


def _decision(*, action: Action = Action.LONG, raw: dict | None = None) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=action,
        confidence=0.8,
        reasoning="费后收益与风险可接受。",
        position_size_pct=0.2 if action.is_entry() else 0.0,
        suggested_leverage=2.0,
        stop_loss_pct=0.01 if action.is_entry() else 0.0,
        take_profit_pct=0.02 if action.is_entry() else 0.0,
        raw_response=deepcopy(raw if raw is not None else _raw()),
        feature_snapshot={"bid": 99.0, "ask": 101.0, "current_price": 100.0},
    )


def _prepare(decision: DecisionOutput, *, status: str = "approved") -> dict:
    attach_initial_trade_recommendation(
        decision,
        analysis_type="market",
        execution_mode="paper",
    )
    return attach_risk_adjusted_trade_recommendation(decision, status=status)


def test_complete_paper_entry_plan_passes() -> None:
    decision = _decision()

    contract = _prepare(decision)

    assert contract["unified_recommendation_complete"] is True
    assert contract["current_recommendation_complete"] is True
    assert contract["risk_adjustment"]["complete"] is True
    assert paper_trade_recommendation_reasons(decision) == []
    assert contract["unified_recommendation"]["entry"] == {
        "price_reference": 101.0,
        "minimum_price": 99.0,
        "maximum_price": 101.0,
        "source": "decision_time_executable_quote",
        "valid_for_seconds": 300.0,
    }


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("entry", "trade_plan_entry_price_missing"),
        ("holding", "trade_plan_holding_horizon_missing"),
        ("return", "trade_plan_fee_after_return_range_missing"),
        ("exit", "trade_plan_stop_loss_missing"),
    ],
)
def test_missing_required_plan_sections_block_entry(
    mutation: str,
    expected_reason: str,
) -> None:
    decision = _decision()
    if mutation == "entry":
        decision.feature_snapshot = {}
    elif mutation == "holding":
        decision.raw_response["opportunity_score"]["return_distribution_contract"] = {}
    elif mutation == "return":
        decision.raw_response["opportunity_score"].update(
            {
                "expected_net_return_pct": None,
                "return_lcb_pct": None,
                "return_uncertainty_pct": None,
            }
        )
    elif mutation == "exit":
        decision.stop_loss_pct = 0.0
        decision.raw_response["profit_risk_sizing"]["stressed_loss_fraction"] = 0.0

    _prepare(decision)

    assert expected_reason in paper_trade_recommendation_reasons(decision)


def test_positive_gross_but_negative_fee_after_return_can_remain_hold() -> None:
    raw = _raw()
    raw["opportunity_score"].update(
        {
            "expected_gross_return_pct": 0.8,
            "expected_net_return_pct": -0.1,
            "return_lcb_pct": -0.3,
        }
    )
    decision = _decision(action=Action.HOLD, raw=raw)

    contract = attach_initial_trade_recommendation(
        decision,
        analysis_type="market",
        execution_mode="paper",
    )

    assert contract["unified_recommendation"]["decision"] == "hold"
    assert contract["unified_recommendation"]["return_after_cost"]["expected_pct"] == -0.1
    assert paper_trade_recommendation_reasons(decision) == []


def test_risk_adjustment_keeps_before_and_after_plans_separate() -> None:
    decision = _decision()
    attach_initial_trade_recommendation(
        decision,
        analysis_type="market",
        execution_mode="paper",
    )
    decision.position_size_pct = 0.05
    decision.suggested_leverage = 1.0
    decision.stop_loss_pct = 0.015
    decision.take_profit_pct = 0.025

    contract = attach_risk_adjusted_trade_recommendation(decision, status="approved")

    before = contract["unified_recommendation"]
    after = contract["risk_adjustment"]["adjusted_recommendation"]
    assert before["suggested_position_fraction"] == 0.2
    assert after["suggested_position_fraction"] == 0.05
    assert before["suggested_leverage"] == 2.0
    assert after["suggested_leverage"] == 1.0
    assert contract["current_recommendation"] == after
    assert contract["current_recommendation_complete"] is True
    assert {item["field"] for item in contract["risk_adjustment"]["adjustments"]} == {
        "suggested_leverage",
        "suggested_position_fraction",
        "stop_loss_fraction",
        "take_profit_fraction",
    }


def test_execution_result_is_stored_as_a_separate_layer() -> None:
    decision = _decision()
    _prepare(decision)
    result = SimpleNamespace(
        status=SimpleNamespace(value="filled"),
        order_id="local-1",
        exchange_order_id="okx-1",
        quantity=0.01,
        price=100.5,
        fee=0.02,
        pnl=-0.02,
    )

    contract = attach_trade_execution_result(
        decision,
        result,
        source="exchange_confirmed",
        exchange_confirmed=True,
    )

    assert contract["unified_recommendation"]["entry"]["price_reference"] == 101.0
    assert contract["execution"]["filled_price"] == 100.5
    assert contract["execution"]["fee_usdt"] == 0.02
    assert contract["execution"]["exchange_confirmed"] is True


def test_exit_plan_has_hold_partial_and_full_exit_paths() -> None:
    raw = _raw()
    raw["close_evidence"] = {"close_fraction": 0.5}
    decision = _decision(raw=raw)

    _prepare(decision)
    plan = trade_recommendation_snapshot(decision.raw_response)["unified_recommendation"]

    assert plan["exit"]["continue_holding"]["maximum_minutes"] == 30.0
    assert plan["exit"]["partial_close"] == {
        "type": "dynamic_profit_or_risk_reduction",
        "close_fraction": 0.5,
        "enabled": True,
    }
    assert plan["exit"]["full_close"]["maximum_minutes"] == 30.0


@pytest.mark.asyncio
async def test_trading_service_attaches_contract_only_to_paper_decisions() -> None:
    from services.trading_service import TradingService

    captured: list[dict] = []

    class Persistence:
        @staticmethod
        def analysis_type(_decision: DecisionOutput, _raw: dict) -> str:
            return "market"

        async def log_decision(self, decision: DecisionOutput, _is_paper: bool) -> int:
            captured.append(deepcopy(decision.raw_response))
            return len(captured)

    async def record_learning_event(**_kwargs) -> None:
        return None

    service = object.__new__(TradingService)
    service.decision_persistence = Persistence()
    service._record_strategy_learning_event = record_learning_event
    paper = _decision()
    live = _decision()

    await service._log_decision(paper, is_paper=True)
    await service._log_decision(live, is_paper=False)

    assert "trade_recommendation_contract" in captured[0]
    assert captured[0]["trade_recommendation_contract"]["execution_mode"] == "paper"
    assert "trade_recommendation_contract" not in captured[1]
