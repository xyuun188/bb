from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ai_brain.base_model import Action, DecisionOutput
from services.entry_opportunity_score import EntryOpportunityScorePolicy
from services.entry_profit_risk_sizing import EntryProfitRiskSizingPolicy
from services.paper_training import (
    PAPER_TRAINING_ORDER_IDENTITY_VERSION,
    PAPER_TRAINING_POSITION_LIFECYCLE_VERSION,
    assess_paper_training_entry,
    assess_paper_training_position_horizon,
    attach_paper_training_order_identity,
    build_paper_training_contract,
    build_paper_training_position_lifecycle,
    paper_training_contract_reasons,
    paper_training_decision_id_from_client_order_id,
    paper_training_mode_enabled,
)
from services.strategy_learning import StrategyLearningService
from services.trade_execution_contract import validate_entry_execution_contract
from services.trading_policies import EntryPolicy


def _decision(*, expected_net: float = -0.8) -> DecisionOutput:
    contract = build_paper_training_contract(
        symbol="BTC/USDT",
        selected_side="long",
        signal_source="local_ml_observation",
        expected_net_return_pct=expected_net,
        return_lcb_pct=-1.2,
        feature_opportunity_score=5.0,
        horizon_minutes=10.0,
    )
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.LONG,
        confidence=0.1,
        reasoning="paper bootstrap training",
        position_size_pct=0.0,
        suggested_leverage=1.0,
        stop_loss_pct=0.0,
        take_profit_pct=0.0,
        feature_snapshot={
            "current_price": 100.0,
            "close": 100.0,
            "volatility_20": 0.02,
            "orderbook_ask_depth": 800.0,
            "orderbook_bid_depth": 700.0,
            "close_sequence": [100.0, 99.5, 100.5],
        },
        raw_response={
            "paper_training": contract,
            "paper_training_mode": "bootstrap",
            "opportunity_score": {
                "expected_net_return_pct": expected_net,
                "return_lcb_pct": -1.2,
                "return_distribution_contract": {
                    "raw_expected_return_pct": expected_net,
                    "objective_expected_return_pct": -1.2,
                },
                "execution_cost": {
                    "production_eligible": True,
                    "total_pct": 0.1,
                    "order_size_complete": False,
                    "order_notional_usdt": 0.0,
                },
            },
            "exchange_risk_facts": {
                "production_eligible": True,
                "account_equity_usdt": 1000.0,
                "available_margin_usdt": 1000.0,
                "target_inst_id": "BTC-USDT-SWAP",
                "contract_specs": {
                    "BTC-USDT-SWAP": {"ctVal": "1", "ctMult": "1"}
                },
                "leverage_tiers": [
                    {"tier": "1", "minSz": "0", "maxSz": "100", "maxLeverage": 20}
                ],
                "policy_provenance": {"source": "paper_test"},
            },
        },
    )


async def _balance(_mode: str, _decision: DecisionOutput | None) -> float:
    return 1000.0


def test_paper_training_contract_accepts_negative_expected_return() -> None:
    decision = _decision(expected_net=-5.0)

    assert paper_training_contract_reasons(
        decision.raw_response["paper_training"]
    ) == []
    assert decision.raw_response["paper_training"]["valid_for_seconds"] == 600.0
    assessment = assess_paper_training_entry(decision, "paper")
    assert assessment.eligible is True
    assert assessment.details["expected_net_return_pct"] == -5.0

    live = assess_paper_training_entry(decision, "live")
    assert live.eligible is False
    assert "paper_training_live_execution_forbidden" in live.blocking_reasons


def test_paper_training_order_identity_is_stable_and_paper_only() -> None:
    decision = _decision()

    identity = attach_paper_training_order_identity(decision, 104208, "paper")

    assert identity == {
        "version": PAPER_TRAINING_ORDER_IDENTITY_VERSION,
        "execution_scope": "paper_only",
        "production_permission": False,
        "decision_id": 104208,
        "client_order_id": "BBPT104208",
    }
    assert paper_training_decision_id_from_client_order_id("BBPT104208") == 104208
    assert paper_training_decision_id_from_client_order_id("BBPT-not-a-number") is None

    live_decision = _decision()
    assert attach_paper_training_order_identity(live_decision, 104208, "live") == {}
    assert "paper_training_order_identity" not in live_decision.raw_response


def test_paper_training_position_closes_at_model_prediction_horizon() -> None:
    decision = _decision(expected_net=-1.0)
    lifecycle = build_paper_training_position_lifecycle(
        SimpleNamespace(
            id=42,
            symbol=decision.symbol,
            action="long",
            raw_response=decision.raw_response,
            is_paper=True,
            was_executed=True,
            executed_at=datetime.now(UTC) - timedelta(minutes=11),
        )
    )

    position = {
        "symbol": decision.symbol,
        "side": "long",
        "execution_mode": "paper",
        "is_open": True,
        "quantity": 1.0,
        "paper_training_lifecycle": lifecycle,
    }
    assessment = assess_paper_training_position_horizon(position)

    assert lifecycle["version"] == PAPER_TRAINING_POSITION_LIFECYCLE_VERSION
    assert lifecycle["horizon_minutes"] == 10.0
    assert lifecycle["continuous_training_after_settlement"] is True
    assert assessment["authorized"] is True
    assert assessment["elapsed"] is True

    position["execution_mode"] = "live"
    assert assess_paper_training_position_horizon(position)["authorized"] is False


def test_paper_training_mode_only_runs_without_paper_champion() -> None:
    assert paper_training_mode_enabled(
        {
            "execution_mode": "paper",
            "paper_training_mode": "bootstrap",
            "paper_strategy_champion": {"active": False},
        }
    )
    assert not paper_training_mode_enabled(
        {
            "execution_mode": "paper",
            "paper_training_mode": "normal",
            "paper_strategy_champion": {"active": True},
        }
    )
    assert not paper_training_mode_enabled(
        {"execution_mode": "live", "paper_training_mode": "bootstrap"}
    )


@pytest.mark.asyncio
async def test_promoted_paper_champion_immediately_disables_fast_training() -> None:
    champion = {
        "active": True,
        "status": "champion",
        "paper_execution_permission": True,
        "live_execution_permission": False,
        "model_version": "paper-model-v2",
    }

    class Engine:
        def build_from_feedback(self, *_args, **_kwargs):
            return {"schedule": {"candidates": []}}

        def apply_to_context(
            self,
            strategy_context,
            _payload,
            *,
            paper_strategy_champion,
        ):
            return {
                **strategy_context,
                "paper_strategy_champion": paper_strategy_champion,
                "strategy_learning": {
                    "paper_strategy_champion": paper_strategy_champion
                },
            }

    class ChampionService:
        async def reconcile(self, **_kwargs):
            return champion

    service = StrategyLearningService(
        engine=Engine(),
        champion_service=ChampionService(),
    )
    service._feedback = AsyncMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(open_position_pressure={})
    )

    context = await service.apply_to_strategy_context(
        mode="paper",
        strategy_context={"paper_training_mode": "bootstrap"},
        open_positions=[],
    )

    assert context["paper_training_mode"] == "normal"
    assert context["strategy_learning"]["paper_training_mode"] == "normal"
    assert context["paper_strategy_champion"]["active"] is True
    assert paper_training_mode_enabled(context) is False


@pytest.mark.asyncio
async def test_paper_training_sizing_has_no_profit_or_virtual_risk_cap() -> None:
    decision = _decision(expected_net=-2.5)
    policy = EntryProfitRiskSizingPolicy(allocated_order_balance=_balance)

    await policy.apply(decision, "paper", [])

    sizing = decision.raw_response["profit_risk_sizing"]
    assert sizing["production_eligible"] is True
    assert sizing["contract_lifecycle"] == "paper_training"
    assert sizing["paper_training_no_profit_gate"] is True
    assert sizing["paper_training_no_virtual_risk_cap"] is True
    assert sizing["expected_net_return_pct"] == -2.5
    assert sizing["final_notional_usdt"] == pytest.approx(800.0)
    assert sizing["risk_budget_usdt"] == pytest.approx(16.0)
    assert decision.position_size_pct == pytest.approx(0.8)

    final_notional = sizing["final_notional_usdt"]
    decision.raw_response["opportunity_score"]["execution_cost"].update(
        {
            "order_size_complete": True,
            "order_notional_usdt": final_notional,
        }
    )
    decision.raw_response["pre_order_execution_facts"] = {
        "production_eligible": True,
        "input_fingerprint": "paper-training-test",
    }
    decision.raw_response["execution_cost_sizing_pass"] = {
        "order_size_complete": True,
        "final_notional_usdt": final_notional,
    }

    contract, reasons = validate_entry_execution_contract(decision.raw_response)
    assert reasons == []
    assert contract["contract_lifecycle"] == "paper_training"
    assert contract["loss_tolerant_for_training"] is True


@pytest.mark.asyncio
async def test_paper_training_sizing_uses_training_contract_for_complete_protection() -> None:
    decision = _decision(expected_net=-0.35)
    opportunity = decision.raw_response["opportunity_score"]
    opportunity.pop("expected_net_return_pct")
    opportunity.pop("return_lcb_pct")
    opportunity["return_distribution_contract"] = {}
    policy = EntryProfitRiskSizingPolicy(allocated_order_balance=_balance)

    await policy.apply(decision, "paper", [])

    sizing = decision.raw_response["profit_risk_sizing"]
    assert sizing["expected_net_return_pct"] == pytest.approx(-0.35)
    assert sizing["return_lcb_pct"] == pytest.approx(-1.2)
    assert sizing["dynamic_take_profit_fraction"] == pytest.approx(0.02)
    assert decision.stop_loss_pct == pytest.approx(0.02)
    assert decision.take_profit_pct == pytest.approx(0.02)
    assert sizing["audit_inputs"]["expected_net_return_pct"] == pytest.approx(-0.35)
    assert sizing["audit_inputs"]["dynamic_take_profit_fraction"] == pytest.approx(0.02)


@pytest.mark.asyncio
async def test_paper_training_records_final_size_aware_cost_pass() -> None:
    decision = _decision(expected_net=-2.5)

    def score(current: DecisionOutput, _strategy: dict | None) -> float:
        planned = float(
            (current.feature_snapshot or {}).get("planned_order_notional_usdt") or 0.0
        )
        raw = current.raw_response
        opportunity = raw["opportunity_score"]
        opportunity["score"] = -2.5
        opportunity["execution_cost"].update(
            {
                "order_size_complete": planned > 0,
                "order_notional_usdt": planned,
            }
        )
        return -2.5

    policy = EntryPolicy(
        entry_opportunity_score=EntryOpportunityScorePolicy(score),
        entry_profit_risk_sizing=EntryProfitRiskSizingPolicy(
            allocated_order_balance=_balance
        ),
    )

    await policy.prepare_dynamic_risk_contract(decision, "paper", [])

    sizing = decision.raw_response["profit_risk_sizing"]
    sizing_pass = decision.raw_response["execution_cost_sizing_pass"]
    assert sizing_pass["impact_basis_notional_usdt"] == pytest.approx(800.0)
    assert sizing_pass["final_notional_usdt"] == sizing["final_notional_usdt"]
    assert sizing_pass["order_size_complete"] is True


def test_paper_training_contract_tampering_fails_closed() -> None:
    decision = _decision()
    tampered = deepcopy(decision.raw_response["paper_training"])
    tampered["production_permission"] = True

    reasons = paper_training_contract_reasons(tampered)

    assert "paper_training_production_permission_invalid" in reasons
    assert "paper_training_contract_fingerprint_mismatch" in reasons
