from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from ai_brain.base_model import Action, DecisionOutput
from ai_brain.ensemble_coordinator import EnsembleCoordinator
from data_feed.feature_vector import FeatureVector
from risk_manager.engine import RiskEngine
from services.entry_profit_risk_sizing import reconcile_profit_risk_sizing
from services.execution_service import _return_entry_contract_result
from services.paper_bootstrap_canary import (
    PAPER_BOOTSTRAP_CANARY_VERSION,
    PAPER_BOOTSTRAP_SIZING_VERSION,
    PaperBootstrapCanaryPolicy,
    select_paper_bootstrap_candidate,
)


def _context() -> dict[str, Any]:
    def distribution(side: str, objective: float) -> dict[str, Any]:
        return {
            "side": side,
            "horizon_minutes": 10,
            "raw_expected_return_pct": objective + 0.2,
            "objective_expected_return_pct": objective,
            "lower_quantile_return_pct": objective - 0.1,
            "dispersion_pct": 0.1,
            "distribution_member_count": 128,
            "source_authority": "extra_trees_empirical_distribution",
        }

    return {
        "trading_mode": "paper",
        "entry_candidate_evidence": {
            "long": {
                "production_source_count": 0,
                "execution_cost": {"total_pct": 0.08},
            },
            "short": {
                "production_source_count": 0,
                "execution_cost": {"total_pct": 0.08},
            },
        },
        "ml_signal": {
            "paper_canary_authorized": True,
            "artifact_lifecycle": "canary",
            "model_version": "candidate-v1",
            "trained_sample_count": 1200,
            "paper_canary": {
                "state": "ready",
                "authorized": True,
                "execution_scope": "paper_only",
                "production_permission": False,
                "eligible_sides": ["long", "short"],
            },
            "predictions": [
                {
                    "return_distribution_contract": {
                        "long": distribution("long", -0.12),
                        "short": distribution("short", -0.35),
                    }
                }
            ],
        },
    }


def _decision() -> DecisionOutput:
    canary = select_paper_bootstrap_candidate(_context())
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.LONG,
        confidence=float(canary["confidence"]),
        reasoning="paper canary test",
        position_size_pct=0.0,
        suggested_leverage=1.0,
        stop_loss_pct=0.0,
        take_profit_pct=0.0,
        raw_response={
            "paper_bootstrap_canary": canary,
            "pre_order_execution_facts": {
                "production_eligible": True,
                "input_fingerprint": "book-v1",
            },
        },
        feature_snapshot={
            "current_price": 100.0,
            "atr_14": 1.0,
            "volatility_20": 0.01,
        },
    )


def test_negative_shadow_objective_can_request_paper_sample_without_production_permission() -> None:
    canary = select_paper_bootstrap_candidate(_context())

    assert canary["authorized"] is True
    assert canary["selected_side"] == "long"
    assert canary["selected_observation"]["objective_expected_return_pct"] < 0
    assert canary["production_permission"] is False
    assert canary["version"] == PAPER_BOOTSTRAP_CANARY_VERSION


def test_paper_canary_selection_is_forbidden_in_live_mode() -> None:
    context = _context()
    context["trading_mode"] = "live"

    canary = select_paper_bootstrap_candidate(context)

    assert canary["authorized"] is False
    assert canary["reason"] == "paper_execution_mode_required"


def test_ensemble_emits_paper_canary_entry_when_production_source_is_absent() -> None:
    coordinator = EnsembleCoordinator(SimpleNamespace(get=lambda _name: None))

    decision = coordinator.combine(
        FeatureVector(symbol="BTC/USDT", current_price=100.0, close=100.0),
        _context(),
        opinions={},
    )

    assert decision.action == Action.LONG
    assert decision.raw_response["paper_bootstrap_canary"]["authorized"] is True
    assert decision.raw_response["entry_permission_policy"]["execution_scope"] == "paper_only"


@pytest.mark.asyncio
async def test_paper_canary_builds_bounded_contract_that_passes_hard_risk() -> None:
    async def balance(_mode: str, _decision: DecisionOutput) -> float:
        return 500.0

    async def facts(
        _mode: str,
        _decision: DecisionOutput,
        _positions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "production_eligible": True,
            "account_equity_usdt": 1000.0,
            "available_margin_usdt": 500.0,
            "target_inst_id": "BTC-USDT-SWAP",
            "contract_specs": {
                "BTC-USDT-SWAP": {"ctVal": "0.01", "ctMult": "1"}
            },
            "leverage_tiers": [{"maxLeverage": 100, "maxNotional": 10000}],
            "entry_instrument_availability": {"available": True},
        }

    async def history() -> list[Any]:
        return []

    decision = _decision()
    decision.feature_snapshot["atr_14"] = 2.443918123
    policy = PaperBootstrapCanaryPolicy(
        allocated_order_balance=balance,
        exchange_risk_facts=facts,
        history_provider=history,
    )

    prepared = await policy.prepare(decision, "paper", [])
    assessed = policy.assess(decision, "paper")
    hard_risk = RiskEngine().assess(decision, [], account_balance=1000.0)

    assert prepared.eligible is True
    assert assessed.eligible is True
    assert hard_risk.approved is True
    sizing = decision.raw_response["profit_risk_sizing"]
    assert sizing["contract_lifecycle"] == "paper_bootstrap_canary"
    assert sizing["risk_budget_usdt"] == pytest.approx(0.5)
    assert sizing["portfolio_risk_budget_usdt"] == pytest.approx(1.0)
    assert sizing["leverage_tier_selection"]["production_eligible"] is True
    assert sizing["leverage_tier_selection"]["contract_spec"]["ctVal"] == "0.01"
    assert sizing["contract_version"] == PAPER_BOOTSTRAP_SIZING_VERSION
    assert sizing["portfolio_risk_snapshot"]["gross_notional_usdt"] == 0.0
    assert sizing["portfolio_risk_snapshot"]["scope"] == (
        "paper_bootstrap_canary_positions_only"
    )
    assert sizing["entry_instrument_availability"]["available"] is True
    assert decision.suggested_leverage == 1.0
    assert decision.stop_loss_pct > 0
    assert decision.take_profit_pct > decision.stop_loss_pct
    assert _return_entry_contract_result(decision, "paper").passed is True
    assert _return_entry_contract_result(decision, "live").passed is False

    reconciled = reconcile_profit_risk_sizing(
        decision,
        final_notional_usdt=sizing["final_notional_usdt"],
        final_leverage=1.0,
        source="test_exchange_precision",
    )
    assert reconciled["eligible"] is True
    assert decision.raw_response["profit_risk_sizing"]["policy_provenance"][
        "strategy_version"
    ] == PAPER_BOOTSTRAP_SIZING_VERSION


@pytest.mark.asyncio
async def test_non_canary_account_positions_do_not_consume_canary_position_slot() -> None:
    async def balance(_mode: str, _decision: DecisionOutput) -> float:
        return 500.0

    async def facts(
        _mode: str,
        _decision: DecisionOutput,
        _positions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "production_eligible": True,
            "account_equity_usdt": 1000.0,
            "available_margin_usdt": 500.0,
            "target_inst_id": "BTC-USDT-SWAP",
            "contract_specs": {
                "BTC-USDT-SWAP": {"ctVal": "0.01", "ctMult": "1"}
            },
            "leverage_tiers": [{"maxLeverage": 100, "maxNotional": 10000}],
            "entry_instrument_availability": {"available": True},
        }

    async def history() -> list[Any]:
        return []

    decision = _decision()
    policy = PaperBootstrapCanaryPolicy(
        allocated_order_balance=balance,
        exchange_risk_facts=facts,
        history_provider=history,
    )

    result = await policy.prepare(
        decision,
        "paper",
        [
            {"symbol": "ETH/USDT", "quantity": 1.0, "is_open": True},
            {"symbol": "SOL/USDT", "quantity": 1.0, "is_open": True},
        ],
    )

    assert result.eligible is True
    guard = decision.raw_response["paper_bootstrap_canary"]["runtime_guard"]
    assert guard["open_position_count"] == 0
    assert guard["account_open_position_count"] == 2


@pytest.mark.asyncio
async def test_unsettled_canary_entry_consumes_canary_position_slot() -> None:
    async def unused_balance(_mode: str, _decision: DecisionOutput) -> float:
        raise AssertionError("open canary must block before account calls")

    async def unused_facts(
        _mode: str,
        _decision: DecisionOutput,
        _positions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        raise AssertionError("open canary must block before exchange calls")

    now = datetime.now(UTC)

    async def history() -> list[Any]:
        return [
            SimpleNamespace(
                id=101,
                raw_llm_response={"paper_bootstrap_canary": {"authorized": True}},
                outcome=None,
                executed_at=now,
                created_at=now,
            )
        ]

    decision = _decision()
    policy = PaperBootstrapCanaryPolicy(
        allocated_order_balance=unused_balance,
        exchange_risk_facts=unused_facts,
        history_provider=history,
    )

    result = await policy.prepare(decision, "paper", [])

    assert result.eligible is False
    assert "paper_canary_open_position_limit_reached" in result.reason
    guard = decision.raw_response["paper_bootstrap_canary"]["runtime_guard"]
    assert guard["open_position_count"] == 1
    assert guard["account_open_position_count"] == 0


@pytest.mark.asyncio
async def test_paper_canary_opens_circuit_after_two_completed_losses() -> None:
    async def unused_balance(_mode: str, _decision: DecisionOutput) -> float:
        raise AssertionError("loss circuit must block before account calls")

    async def unused_facts(
        _mode: str,
        _decision: DecisionOutput,
        _positions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        raise AssertionError("loss circuit must block before exchange calls")

    now = datetime.now(UTC)

    async def history() -> list[Any]:
        raw = {"paper_bootstrap_canary": {"authorized": True}}
        return [
            SimpleNamespace(
                raw_llm_response=raw,
                outcome="loss",
                executed_at=now,
                created_at=now,
            ),
            SimpleNamespace(
                raw_llm_response=raw,
                outcome="loss",
                executed_at=now,
                created_at=now,
            ),
        ]

    decision = _decision()
    policy = PaperBootstrapCanaryPolicy(
        allocated_order_balance=unused_balance,
        exchange_risk_facts=unused_facts,
        history_provider=history,
    )

    result = await policy.prepare(decision, "paper", [])

    assert result.eligible is False
    assert "paper_canary_consecutive_loss_circuit_open" in result.reason
