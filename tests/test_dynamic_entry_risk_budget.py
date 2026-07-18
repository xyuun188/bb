from copy import deepcopy

import pytest

from ai_brain.base_model import Action, DecisionOutput
from services.entry_profit_risk_sizing import (
    EntryProfitRiskSizingPolicy,
    build_portfolio_correlation_context,
    reconcile_profit_risk_sizing,
    select_okx_leverage_tier,
)


def _decision() -> DecisionOutput:
    calibration = {
        "source_authority": "okx_position_history",
        "profile_source": "symbol_side",
        "net_return_after_cost_pct": {
            "count": 12,
            "expected": 0.7,
            "lower_hinge": 0.5,
        },
        "slippage_pct": {"count": 12, "expected": 0.02, "upper_hinge": 0.03},
        "stop_loss_slippage_pct": {
            "count": 4,
            "expected": 0.03,
            "upper_hinge": 0.05,
        },
    }
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.LONG,
        confidence=0.8,
        reasoning="dynamic sizing test",
        position_size_pct=0.07,
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
            "strategy_mode": {
                "drawdown_pressure": 0.1,
                "portfolio_correlation": {},
            },
            "exchange_risk_facts": {
                "production_eligible": True,
                "account_equity_usdt": 1000.0,
                "available_margin_usdt": 1000.0,
                "reported_max_leverage": 20.0,
                "target_inst_id": "BTC-USDT-SWAP",
                "contract_specs": {
                    "BTC-USDT-SWAP": {"ctVal": "1", "ctMult": "1"},
                },
                "leverage_tiers": [
                    {"tier": "1", "minSz": "0", "maxSz": "100", "maxLeverage": 20},
                    {
                        "tier": "2",
                        "minSz": "100.01",
                        "maxSz": "1000",
                        "maxLeverage": 10,
                    },
                ],
                "policy_provenance": {
                    "source": "okx_test_facts",
                    "observation_window": "current",
                    "sample_count": 1,
                    "generated_at": "2026-07-15T00:00:00+00:00",
                    "strategy_version": "test",
                    "fallback_reason": "",
                },
            },
            "opportunity_score": {
                "expected_net_return_pct": 0.9,
                "expected_loss_pct": 0.2,
                "server_profit_loss_probability": 0.25,
                "tail_risk_score": 0.2,
                "profit_quality_ratio": 1.6,
                "return_lcb_pct": 0.6,
                "return_distribution_contract": {
                    "raw_expected_return_pct": 0.9,
                    "objective_expected_return_pct": 0.6,
                    "uncertainty_penalty_pct": 0.1,
                    "tail_loss_penalty_pct": 0.2,
                    "tail_loss_probability": 0.25,
                },
                "execution_cost": {
                    "production_eligible": True,
                    "total_pct": 0.08,
                    "slippage_pct": 0.02,
                    "spread_pct": 0.02,
                },
                "expected_net_breakdown": {
                    "components": [
                        {
                            "key": "server_profit",
                            "production_eligible": True,
                            "included_in_return_distribution": True,
                            "raw_return_pct": 1.1,
                            "actual_trade_calibration": calibration,
                        }
                    ]
                },
            }
        },
    )


async def _balance(_mode: str, _decision: DecisionOutput | None) -> float:
    return 1000.0


def test_okx_leverage_tier_tracks_target_notional_across_contract_size_bounds() -> None:
    tiers = [
        {"tier": "1", "minSz": "0", "maxSz": "100", "maxLeverage": 20},
        {"tier": "2", "minSz": "100.01", "maxSz": "500", "maxLeverage": 8},
    ]
    contract_spec = {"ctVal": "1", "ctMult": "1"}

    first = select_okx_leverage_tier(
        tiers,
        target_notional_usdt=10_000.0,
        mark_price=100.0,
        contract_spec=contract_spec,
    )
    second = select_okx_leverage_tier(
        tiers,
        target_notional_usdt=10_100.0,
        mark_price=100.0,
        contract_spec=contract_spec,
    )

    assert first["production_eligible"] is True
    assert first["selected_tier"]["tier"] == "1"
    assert first["max_leverage"] == 20
    assert second["production_eligible"] is True
    assert second["selected_tier"]["tier"] == "2"
    assert second["max_leverage"] == 8
    assert second["projected_position_contracts"] == pytest.approx(101.0)


def test_okx_leverage_tier_includes_existing_same_side_position() -> None:
    selection = select_okx_leverage_tier(
        [
            {"tier": "1", "maxSz": "100", "maxLeverage": 20},
            {"tier": "2", "maxSz": "500", "maxLeverage": 8},
        ],
        target_notional_usdt=2_000.0,
        mark_price=100.0,
        contract_spec={"ctVal": "1", "ctMult": "1"},
        current_position_notional_usdt=9_000.0,
        current_position_contracts=90.0,
    )

    assert selection["selected_tier"]["tier"] == "2"
    assert selection["projected_position_contracts"] == pytest.approx(110.0)
    assert selection["max_leverage"] == 8


def test_multiple_okx_leverage_tiers_without_bounds_fail_closed() -> None:
    selection = select_okx_leverage_tier(
        [{"maxLeverage": 20}, {"maxLeverage": 8}],
        target_notional_usdt=2_000.0,
        mark_price=100.0,
        contract_spec={"ctVal": "1", "ctMult": "1"},
    )

    assert selection["production_eligible"] is False
    assert selection["reason"] == "okx_leverage_tier_bounds_missing"


@pytest.mark.asyncio
async def test_legacy_probe_and_evidence_payload_cannot_change_dynamic_sizing() -> None:
    first = _decision()
    second = deepcopy(first)
    second.position_size_pct = 0.9
    first.raw_response.update(
        {
            "quant_profit_probe": {"triggered": True, "strong_probe": True},
            "evidence_profit_probe": {"triggered": True},
        }
    )
    first.raw_response["opportunity_score"]["evidence_score"] = {
        "tier": "blocked",
        "shadow_only": True,
    }
    second.raw_response["opportunity_score"]["evidence_score"] = {
        "tier": "elite",
        "tradeable_probe": True,
    }
    policy = EntryProfitRiskSizingPolicy(allocated_order_balance=_balance)

    await policy.apply(first, "paper", [])
    await policy.apply(second, "paper", [])

    assert first.position_size_pct == pytest.approx(second.position_size_pct)
    assert first.suggested_leverage == second.suggested_leverage
    sizing = first.raw_response["profit_risk_sizing"]
    assert sizing["production_eligible"] is True
    assert sizing["risk_budget_usdt"] > 0
    assert sizing["target_notional_usdt"] == pytest.approx(
        sizing["risk_budget_usdt"] / sizing["stressed_loss_fraction"],
        rel=1e-7,
    )
    assert sizing["planned_stressed_loss_usdt"] <= sizing["risk_budget_usdt"]
    assert "final_position_size" not in sizing["audit_inputs"]
    provenance = sizing["policy_provenance"]
    assert "legacy_evidence_tier_enabled" not in provenance
    assert "legacy_probe_sizing_enabled" not in provenance
    assert {
        "source",
        "observation_window",
        "sample_count",
        "generated_at",
        "strategy_version",
        "fallback_reason",
    }.issubset(provenance)


@pytest.mark.asyncio
async def test_missing_production_cost_fails_closed_without_fixed_fallback() -> None:
    decision = _decision()
    decision.raw_response["opportunity_score"]["execution_cost"][
        "production_eligible"
    ] = False
    policy = EntryProfitRiskSizingPolicy(allocated_order_balance=_balance)

    await policy.apply(decision, "paper", [])

    assert decision.position_size_pct == 0.0
    assert decision.suggested_leverage == 1.0
    sizing = decision.raw_response["profit_risk_sizing"]
    assert sizing["production_eligible"] is False
    assert "production_execution_cost_missing" in sizing["reason"]


@pytest.mark.asyncio
async def test_missing_okx_leverage_tiers_fails_closed() -> None:
    decision = _decision()
    decision.raw_response["exchange_risk_facts"]["leverage_tiers"] = []
    policy = EntryProfitRiskSizingPolicy(allocated_order_balance=_balance)

    await policy.apply(decision, "paper", [])

    sizing = decision.raw_response["profit_risk_sizing"]
    assert sizing["production_eligible"] is False
    assert sizing["leverage_tier_selection"]["production_eligible"] is False
    assert "okx_leverage_tiers_missing" in sizing["reason"]
    assert decision.position_size_pct == 0.0


@pytest.mark.asyncio
async def test_risk_increase_cannot_increase_position() -> None:
    baseline = _decision()
    wider_stop = deepcopy(baseline)
    worse_tail = deepcopy(baseline)
    wider_stop.stop_loss_pct = 0.05
    worse_tail.raw_response["opportunity_score"]["return_distribution_contract"][
        "tail_loss_penalty_pct"
    ] = 0.6
    policy = EntryProfitRiskSizingPolicy(allocated_order_balance=_balance)

    await policy.apply(baseline, "paper", [])
    await policy.apply(wider_stop, "paper", [])
    await policy.apply(worse_tail, "paper", [])

    assert wider_stop.raw_response["profit_risk_sizing"]["final_notional_usdt"] <= baseline.raw_response[
        "profit_risk_sizing"
    ]["final_notional_usdt"]
    assert worse_tail.raw_response["profit_risk_sizing"]["risk_budget_usdt"] <= baseline.raw_response[
        "profit_risk_sizing"
    ]["risk_budget_usdt"]


@pytest.mark.asyncio
async def test_lower_return_lcb_cannot_increase_risk_budget() -> None:
    baseline = _decision()
    lower_lcb = deepcopy(baseline)
    lower_lcb.raw_response["opportunity_score"]["return_distribution_contract"][
        "objective_expected_return_pct"
    ] = 0.3
    policy = EntryProfitRiskSizingPolicy(allocated_order_balance=_balance)

    await policy.apply(baseline, "paper", [])
    await policy.apply(lower_lcb, "paper", [])

    assert lower_lcb.raw_response["profit_risk_sizing"]["risk_budget_usdt"] <= baseline.raw_response[
        "profit_risk_sizing"
    ]["risk_budget_usdt"]


@pytest.mark.asyncio
async def test_lower_available_margin_cannot_increase_final_notional() -> None:
    baseline = _decision()
    constrained = deepcopy(baseline)
    constrained.raw_response["exchange_risk_facts"]["available_margin_usdt"] = 100.0
    policy = EntryProfitRiskSizingPolicy(allocated_order_balance=_balance)

    await policy.apply(baseline, "paper", [])
    await policy.apply(constrained, "paper", [])

    baseline_sizing = baseline.raw_response["profit_risk_sizing"]
    constrained_sizing = constrained.raw_response["profit_risk_sizing"]
    assert constrained_sizing["production_eligible"] is True
    assert constrained_sizing["final_notional_usdt"] <= baseline_sizing["final_notional_usdt"]
    assert constrained_sizing["risk_budget_usdt"] == pytest.approx(
        baseline_sizing["risk_budget_usdt"]
    )


@pytest.mark.asyncio
async def test_portfolio_dependency_cannot_increase_risk_budget() -> None:
    baseline = _decision()
    pressured = _decision()
    pressured.raw_response["strategy_mode"]["portfolio_correlation"] = {
        "BTC/USDT|long": {"weighted_adverse_correlation": 0.8}
    }
    pressured.raw_response["exchange_risk_facts"]["contract_specs"] = {
        "BTC-USDT-SWAP": {"ctVal": "1", "ctMult": "1"},
        "ETH-USDT-SWAP": {"ctVal": "1", "ctMult": "1"}
    }
    open_positions = [
        {
            "symbol": "ETH/USDT",
            "side": "long",
            "contracts": 1.0,
            "current_price": 100.0,
            "stop_loss": 95.0,
            "leverage": 5.0,
            "margin": 20.0,
            "notional": 100.0,
            "is_open": True,
            "info": {
                "instId": "ETH-USDT-SWAP",
                "ctVal": "1",
                "ctMult": "1",
                "mgnMode": "cross",
            },
        }
    ]
    policy = EntryProfitRiskSizingPolicy(allocated_order_balance=_balance)

    await policy.apply(baseline, "paper", [])
    await policy.apply(pressured, "paper", open_positions)

    baseline_budget = baseline.raw_response["profit_risk_sizing"]["risk_budget_usdt"]
    pressured_sizing = pressured.raw_response["profit_risk_sizing"]
    assert pressured_sizing["production_eligible"] is True
    assert pressured_sizing["risk_budget_usdt"] <= baseline_budget
    assert pressured_sizing["budget_factors"]["portfolio_dependency_capacity"] < 1.0
    assert pressured_sizing["current_portfolio_stressed_loss_usdt"] == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_execution_reconciliation_rebuilds_every_notional_dependent_field() -> None:
    decision = _decision()
    policy = EntryProfitRiskSizingPolicy(allocated_order_balance=_balance)
    await policy.apply(decision, "paper", [])
    before = decision.raw_response["profit_risk_sizing"]
    reduced_notional = before["final_notional_usdt"] / 2.0
    leverage = decision.suggested_leverage

    result = reconcile_profit_risk_sizing(
        decision,
        final_notional_usdt=reduced_notional,
        final_leverage=leverage,
        source="test_exchange_precision",
        execution_facts={"contracts": 2.0, "price": reduced_notional / 2.0},
    )

    assert result["eligible"] is True
    sizing = decision.raw_response["profit_risk_sizing"]
    assert sizing["target_notional_usdt"] == before["target_notional_usdt"]
    assert sizing["final_notional_usdt"] == pytest.approx(reduced_notional)
    assert sizing["planned_stressed_loss_usdt"] == pytest.approx(
        reduced_notional * sizing["stressed_loss_fraction"]
    )
    assert sizing["expected_profit_usdt"] == pytest.approx(
        reduced_notional * sizing["expected_net_return_pct"] / 100.0
    )
    assert decision.position_size_pct == pytest.approx(
        reduced_notional / (sizing["available_margin_usdt"] * leverage)
    )
    assert sizing["execution_reconciliations"][-1]["source"] == "test_exchange_precision"
    assert sizing["policy_provenance"]["contract_fingerprint"]


@pytest.mark.asyncio
async def test_execution_leverage_change_updates_margin_without_changing_notional() -> None:
    decision = _decision()
    policy = EntryProfitRiskSizingPolicy(allocated_order_balance=_balance)
    await policy.apply(decision, "paper", [])
    before = decision.raw_response["profit_risk_sizing"]
    original_notional = before["final_notional_usdt"]
    lower_leverage = max(1.0, decision.suggested_leverage - 1.0)

    result = reconcile_profit_risk_sizing(
        decision,
        final_notional_usdt=original_notional,
        final_leverage=lower_leverage,
        source="test_okx_actual_leverage_change",
    )

    assert result["eligible"] is True
    sizing = decision.raw_response["profit_risk_sizing"]
    assert sizing["final_notional_usdt"] == pytest.approx(original_notional)
    assert sizing["final_margin_usdt"] == pytest.approx(original_notional / lower_leverage)
    assert sizing["planned_stressed_loss_usdt"] == pytest.approx(
        original_notional * sizing["stressed_loss_fraction"]
    )


@pytest.mark.asyncio
async def test_execution_reconciliation_rejects_notional_enlargement() -> None:
    decision = _decision()
    policy = EntryProfitRiskSizingPolicy(allocated_order_balance=_balance)
    await policy.apply(decision, "paper", [])
    sizing = decision.raw_response["profit_risk_sizing"]

    result = reconcile_profit_risk_sizing(
        decision,
        final_notional_usdt=sizing["target_notional_usdt"] * 2.0,
        final_leverage=decision.suggested_leverage,
        source="test_forbidden_enlargement",
    )

    assert result["eligible"] is False
    assert "execution_notional_exceeds_authoritative_target" in result["reasons"]
    assert decision.position_size_pct == 0.0


@pytest.mark.asyncio
async def test_confirmed_fill_can_use_reserved_ceiling_but_other_enlargement_cannot() -> None:
    decision = _decision()
    policy = EntryProfitRiskSizingPolicy(allocated_order_balance=_balance)
    await policy.apply(decision, "paper", [])
    sizing = decision.raw_response["profit_risk_sizing"]
    risk_ceiling = sizing["risk_budget_usdt"] / sizing["stressed_loss_fraction"]
    reserved_target = risk_ceiling / 1.002
    sizing["target_notional_usdt"] = reserved_target
    sizing["fill_notional_ceiling_usdt"] = risk_ceiling
    sizing["final_notional_usdt"] = reserved_target
    decision.raw_response["profit_risk_sizing"] = sizing
    confirmed_notional = reserved_target * 1.001

    confirmed = reconcile_profit_risk_sizing(
        decision,
        final_notional_usdt=confirmed_notional,
        final_leverage=decision.suggested_leverage,
        source="okx_confirmed_entry_fill",
    )

    assert confirmed["eligible"] is True

    sizing = decision.raw_response["profit_risk_sizing"]
    sizing["final_notional_usdt"] = reserved_target
    sizing["production_eligible"] = True
    decision.raw_response["profit_risk_sizing"] = sizing
    non_fill = reconcile_profit_risk_sizing(
        decision,
        final_notional_usdt=confirmed_notional,
        final_leverage=decision.suggested_leverage,
        source="test_non_fill_enlargement",
    )

    assert non_fill["eligible"] is False
    assert "execution_notional_exceeds_authoritative_target" in non_fill["reasons"]


def test_portfolio_correlation_is_side_aware() -> None:
    features = {
        "BTC/USDT": {"close_sequence": [100.0, 101.0, 102.0, 103.0]},
        "ETH/USDT": {"close_sequence": [50.0, 50.5, 51.0, 51.5]},
    }
    positions = [
        {
            "symbol": "ETH/USDT",
            "side": "long",
            "notional": 100.0,
            "is_open": True,
        }
    ]

    context = build_portfolio_correlation_context(features, positions)

    assert context["BTC/USDT|long"]["weighted_adverse_correlation"] > 0.0
    assert context["BTC/USDT|short"]["weighted_adverse_correlation"] == 0.0
