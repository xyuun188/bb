from __future__ import annotations

from copy import deepcopy

import pytest

from ai_brain.base_model import Action, DecisionOutput
from risk_manager.engine import RiskEngine
from services.entry_profit_risk_sizing import EntryProfitRiskSizingPolicy
from services.paper_exploration import (
    PAPER_EXPLORATION_MAX_PORTFOLIO_RISK_FRACTION,
    PAPER_EXPLORATION_MAX_SINGLE_TRADE_RISK_FRACTION,
    assess_paper_exploration_entry,
    build_paper_exploration_contract,
    evaluate_paper_exploration_side,
    paper_exploration_contract_reasons,
    select_paper_exploration_side,
)
from services.trade_execution_contract import validate_entry_execution_contract
from services.trading_policies import EntryPolicy


def _provenance() -> dict:
    return {
        "source": "cost_complete_test_distribution",
        "observation_window": "current_test_candidate",
        "sample_count": 3,
        "generated_at": "2026-07-21T00:00:00+00:00",
        "strategy_version": "test.return.v1",
        "fallback_reason": "",
    }


def _side(
    *,
    expected_net: float = 0.30,
    return_lcb: float = -0.10,
    loss_probability: float = 0.30,
    tail_risk: float = 0.20,
) -> dict:
    return {
        "side": "long",
        "expected_net_return_pct": expected_net,
        "return_lcb_pct": return_lcb,
        "return_uncertainty_pct": 0.20,
        "expected_loss_pct": 0.20,
        "horizon_minutes": 30.0,
        "loss_probability": loss_probability,
        "tail_risk_score": tail_risk,
        "production_source_count": 3,
        "return_distribution_ready": True,
        "production_eligible": False,
        "execution_cost": {"production_eligible": True, "total_pct": 0.08},
        "policy_provenance": _provenance(),
    }


def _candidate_evidence() -> dict:
    side = _side()
    selection = select_paper_exploration_side(
        {"long": side, "short": {**side, "side": "short", "expected_net_return_pct": -0.1}},
        feature_opportunity_score=8.0,
    )
    return {
        "preferred_side_by_evidence": "neutral",
        "preferred_exploration_side": selection["preferred_side"],
        "feature_opportunity_score": 8.0,
        "long": side,
        "short": {**side, "side": "short", "expected_net_return_pct": -0.1},
        "paper_exploration": selection,
    }


def _decision() -> DecisionOutput:
    candidate_evidence = _candidate_evidence()
    contract = build_paper_exploration_contract(
        candidate_evidence,
        symbol="BTC/USDT",
    )
    calibration = {
        "source_authority": "okx_position_history",
        "net_return_after_cost_pct": {"count": 12, "expected": 0.2, "lower_hinge": -0.1},
        "slippage_pct": {"count": 12, "expected": 0.02, "upper_hinge": 0.03},
        "stop_loss_slippage_pct": {"count": 4, "expected": 0.03, "upper_hinge": 0.05},
    }
    components = [
        {
            "key": key,
            "production_eligible": True,
            "included_in_return_distribution": True,
            "actual_trade_calibration": calibration,
        }
        for key in ("local_ml", "local_ai_tools")
    ]
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.LONG,
        confidence=0.2,
        reasoning="bounded paper exploration test",
        position_size_pct=0.0,
        suggested_leverage=8.0,
        stop_loss_pct=0.02,
        take_profit_pct=0.08,
        feature_snapshot={
            "current_price": 100.0,
            "atr_14": 1.5,
            "volatility_20": 0.012,
            "abnormal_wick_max_pct": 1.2,
            "orderbook_ask_depth": 800.0,
            "orderbook_bid_depth": 700.0,
            "close_sequence": [100.0, 101.0, 99.5, 102.0],
        },
        raw_response={
            "entry_candidate_evidence": candidate_evidence,
            "paper_exploration": contract,
            "strategy_mode": {"drawdown_pressure": 0.1, "portfolio_correlation": {}},
            "exchange_risk_facts": {
                "production_eligible": True,
                "account_equity_usdt": 1000.0,
                "available_margin_usdt": 1000.0,
                "target_inst_id": "BTC-USDT-SWAP",
                "contract_specs": {"BTC-USDT-SWAP": {"ctVal": "1", "ctMult": "1"}},
                "leverage_tiers": [
                    {"tier": "1", "minSz": "0", "maxSz": "100", "maxLeverage": 20}
                ],
                "policy_provenance": _provenance(),
            },
            "opportunity_score": {
                "score": -0.10,
                "expected_net_return_pct": 0.30,
                "return_lcb_pct": -0.10,
                "return_uncertainty_pct": 0.20,
                "expected_loss_pct": 0.20,
                "server_profit_loss_probability": 0.30,
                "tail_risk_score": 0.20,
                "return_distribution_contract": {
                    "raw_expected_return_pct": 0.30,
                    "objective_expected_return_pct": -0.10,
                    "uncertainty_penalty_pct": 0.20,
                    "tail_loss_penalty_pct": 0.20,
                    "tail_loss_probability": 0.30,
                },
                "execution_cost": {
                    "production_eligible": True,
                    "total_pct": 0.08,
                    "slippage_pct": 0.02,
                    "spread_pct": 0.02,
                },
                "expected_net_breakdown": {"components": components},
            },
        },
    )


async def _balance(_mode: str, _decision: DecisionOutput | None) -> float:
    return 1000.0


class _NoopSizing:
    async def apply(self, *_args, **_kwargs) -> None:
        return None


def test_only_positive_mean_near_threshold_side_is_explorable() -> None:
    eligible = evaluate_paper_exploration_side(
        _side(),
        feature_opportunity_score=8.0,
    )
    negative = evaluate_paper_exploration_side(
        _side(expected_net=-0.01),
        feature_opportunity_score=8.0,
    )
    far_below = evaluate_paper_exploration_side(
        _side(expected_net=0.1, return_lcb=-0.2),
        feature_opportunity_score=8.0,
    )

    assert eligible["eligible"] is True
    assert eligible["information_value_score"] > 0
    assert negative["eligible"] is False
    assert "paper_exploration_expected_net_return_not_positive" in negative["reasons"]
    assert far_below["eligible"] is False
    assert "paper_exploration_not_close_to_profitable_threshold" in far_below["reasons"]


def test_exploration_contract_has_no_sample_quota_and_detects_tampering() -> None:
    contract = build_paper_exploration_contract(
        _candidate_evidence(),
        symbol="BTC/USDT",
    )

    assert paper_exploration_contract_reasons(contract) == []
    assert contract["trade_is_normal"] is True
    assert contract["sample_target"] is None
    assert contract["daily_sample_quota"] is None
    tampered = {**contract, "daily_sample_quota": 10}
    reasons = paper_exploration_contract_reasons(tampered)
    assert "paper_exploration_sample_quota_forbidden" in reasons
    assert "paper_exploration_contract_fingerprint_mismatch" in reasons


@pytest.mark.asyncio
async def test_exploration_uses_tiny_one_x_budget_and_full_execution_contract() -> None:
    decision = _decision()
    policy = EntryProfitRiskSizingPolicy(allocated_order_balance=_balance)

    await policy.apply(decision, "paper", [])

    sizing = decision.raw_response["profit_risk_sizing"]
    assert sizing["production_eligible"] is True
    assert sizing["contract_lifecycle"] == "paper_exploration"
    assert sizing["risk_budget_usdt"] <= (
        1000.0 * PAPER_EXPLORATION_MAX_SINGLE_TRADE_RISK_FRACTION + 1e-8
    )
    assert sizing["portfolio_risk_budget_usdt"] <= (
        1000.0 * PAPER_EXPLORATION_MAX_PORTFOLIO_RISK_FRACTION + 1e-8
    )
    assert decision.suggested_leverage == pytest.approx(1.0)
    assert sizing["final_leverage"] == pytest.approx(1.0)
    assert decision.position_size_pct > 0
    final_notional = sizing["final_notional_usdt"]
    decision.raw_response["opportunity_score"]["execution_cost"].update(
        {
            "order_size_complete": True,
            "order_notional_usdt": final_notional,
        }
    )
    decision.raw_response["pre_order_execution_facts"] = {
        "production_eligible": True,
        "input_fingerprint": "test-pre-order",
    }
    decision.raw_response["execution_cost_sizing_pass"] = {
        "order_size_complete": True,
        "final_notional_usdt": final_notional,
    }

    assessment = assess_paper_exploration_entry(decision, "paper")
    contract, reasons = validate_entry_execution_contract(decision.raw_response)
    assert assessment.eligible is True
    assert reasons == []
    assert contract["contract_lifecycle"] == "paper_exploration"
    assert RiskEngine._dynamic_risk_contract_reason(decision) is None

    gate = await EntryPolicy(entry_profit_risk_sizing=_NoopSizing()).evaluate(
        decision,
        decision.model_name,
        "paper",
        [],
    )
    assert gate.passed is True
    assert gate.data["intent"] == "paper_exploration_entry"
    assert gate.data["production_permission"] is False

    repriced_negative = deepcopy(decision)
    repriced_negative.raw_response["opportunity_score"]["return_distribution_contract"][
        "raw_expected_return_pct"
    ] = -0.01
    blocked = assess_paper_exploration_entry(repriced_negative, "paper")
    assert blocked.eligible is False
    assert (
        "paper_exploration_size_aware_expected_return_not_positive"
        in blocked.blocking_reasons
    )


@pytest.mark.asyncio
async def test_exploration_is_fail_closed_outside_paper_mode() -> None:
    decision = deepcopy(_decision())
    policy = EntryProfitRiskSizingPolicy(allocated_order_balance=_balance)

    await policy.apply(decision, "live", [])

    sizing = decision.raw_response["profit_risk_sizing"]
    assert sizing["production_eligible"] is False
    assert "paper_exploration_live_execution_forbidden" in sizing["reason"]
    assert decision.position_size_pct == 0.0
