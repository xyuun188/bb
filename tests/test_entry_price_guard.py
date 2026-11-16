from __future__ import annotations

from typing import Any

import pytest

from ai_brain.base_model import Action, DecisionOutput
from services.entry_opportunity_score import EntryOpportunityScorePolicy
from services.entry_price_guard import EntryPriceGuardPolicy, paper_entry_execution_price_limit
from services.normal_paper_trade import NORMAL_PAPER_TRADE_VERSION
from services.production_trade_gate import PRODUCTION_TRADE_GATE_VERSION
from services.trading_policies import EntryPolicy


def _decision(*, return_lcb: float = 0.6, expected_net: float = 0.8) -> DecisionOutput:
    provenance = {
        "source": "test",
        "observation_window": "test",
        "sample_count": 5,
        "generated_at": "2026-07-12T00:00:00+00:00",
        "strategy_version": "test.v1",
        "fallback_reason": "",
    }
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.LONG,
        confidence=0.1,
        reasoning="dynamic return",
        feature_snapshot={"current_price": 100.0, "close": 100.0},
        raw_response={
            "entry_candidate_evidence": {
                "long": {
                    "production_eligible": True,
                    "expected_net_return_pct": expected_net,
                    "return_lcb_pct": return_lcb,
                    "production_source_count": 5,
                    "policy_provenance": provenance,
                }
            }
        },
    )


def _policy(*, latest: float, fresh: dict[str, Any] | None = None) -> EntryPriceGuardPolicy:
    async def fresh_feature(_symbol: str) -> Any:
        return (
            {"current_price": latest, "close": latest}
            if fresh is None
            else fresh
        )

    return EntryPriceGuardPolicy(
        fresh_feature_provider=fresh_feature,
        market_data_quality_reason_provider=lambda _snapshot, **_kwargs: None,
        decision_age_seconds_provider=lambda _decision: 12.0,
    )


@pytest.mark.asyncio
async def test_missing_latest_price_fails_closed() -> None:
    assert "fails closed" in await _policy(latest=0.0).guard_reason(_decision())


@pytest.mark.asyncio
async def test_adverse_move_must_fit_return_lcb() -> None:
    assert await _policy(latest=100.4).guard_reason(_decision(return_lcb=0.6)) is None
    reason = await _policy(latest=100.7).guard_reason(_decision(return_lcb=0.6))
    assert "exceeds" in reason


@pytest.mark.asyncio
async def test_fresh_snapshot_cannot_rebase_a_decision_past_its_return_budget() -> None:
    decision = _decision(return_lcb=0.2)
    reason = await _policy(
        latest=101.0,
        fresh={"current_price": 101.0, "close": 101.0},
    ).guard_reason(decision)
    assert "exceeds" in reason
    assert decision.feature_snapshot["current_price"] == 100.0


@pytest.mark.asyncio
async def test_every_entry_requires_a_fresh_native_market_snapshot() -> None:
    reason = await _policy(latest=100.0, fresh={}).guard_reason(_decision())

    assert "Fresh pre-order native market fact is incomplete" in reason


@pytest.mark.asyncio
async def test_live_rules_canary_does_not_require_model_return_budget() -> None:
    decision = _decision()
    decision.raw_response = {
        "production_trade_gate": {
            "version": PRODUCTION_TRADE_GATE_VERSION,
            "mode": "live_rules_canary",
            "can_trade": True,
            "decision_authority": "rules",
            "model_can_influence": False,
        }
    }

    assert await _policy(latest=101.0).guard_reason(decision, "live") is None
    assert decision.raw_response["pre_execution_price_check"][
        "contract_lifecycle"
    ] == "live_rules_canary"


@pytest.mark.asyncio
async def test_dirty_analysis_fact_is_replaced_by_authoritative_execution_snapshot() -> None:
    async def fresh_feature(_symbol: str) -> Any:
        raise AssertionError("authoritative execution facts must avoid a duplicate feature refresh")

    async def execution_facts(_mode: str, _decision: DecisionOutput) -> dict[str, Any]:
        return {
            "production_eligible": True,
            "inst_id": "BTC-USDT-SWAP",
            "feature_snapshot": {"current_price": 100.1, "close": 100.1},
        }

    policy = EntryPriceGuardPolicy(
        fresh_feature_provider=fresh_feature,
        market_data_quality_reason_provider=lambda _snapshot, **kwargs: (
            "dirty analysis fact"
            if kwargs.get("stage_label") == "pre-order analysis snapshot"
            else None
        ),
        decision_age_seconds_provider=lambda _decision: 12.0,
        pre_order_execution_facts_provider=execution_facts,
    )

    reason = await policy.guard_reason(_decision(), "paper")

    assert reason is None


@pytest.mark.asyncio
async def test_dirty_authoritative_execution_fact_still_fails_closed() -> None:
    async def execution_facts(_mode: str, _decision: DecisionOutput) -> dict[str, Any]:
        return {
            "production_eligible": True,
            "inst_id": "BTC-USDT-SWAP",
            "feature_snapshot": {"current_price": 100.1, "close": 100.1},
        }

    policy = EntryPriceGuardPolicy(
        fresh_feature_provider=lambda _symbol: {},
        market_data_quality_reason_provider=lambda _snapshot, **_kwargs: "dirty execution fact",
        decision_age_seconds_provider=lambda _decision: 12.0,
        pre_order_execution_facts_provider=execution_facts,
    )

    reason = await policy.guard_reason(_decision(), "paper")

    assert "execution market fact is invalid" in reason
    assert "dirty execution fact" in reason


@pytest.mark.asyncio
async def test_missing_authoritative_return_budget_fails_closed() -> None:
    reason = await _policy(latest=100.0).guard_reason(
        _decision(return_lcb=0.0, expected_net=9.0)
    )
    assert "return budget is missing" in reason


@pytest.mark.asyncio
async def test_paper_market_refresh_defers_signal_budget_until_final_sizing() -> None:
    decision = _decision(return_lcb=0.0, expected_net=-1.0)

    assert await _policy(latest=150.0).guard_reason(decision, "paper") is None
    price_check = decision.raw_response["pre_execution_price_check"]
    assert price_check["allowed_adverse_move_fraction"] is None
    assert price_check["contract_lifecycle"] == "normal_paper_trade"
    assert price_check["production_permission"] is False
    assert price_check["profitability_gate_applied"] is False
    assert price_check["safety_scope"] == "market_integrity_only"
    assert decision.feature_snapshot["current_price"] == 150.0


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_key", ["paper_training", "paper_exploration", "paper_bootstrap_canary"])
async def test_legacy_paper_identity_cannot_change_price_guard(
    legacy_key: str,
) -> None:
    decision = _decision(return_lcb=-2.0, expected_net=-3.0)
    decision.raw_response[legacy_key] = {"authorized": True}

    assert await _policy(latest=125.0).guard_reason(decision, "paper") is None
    price_check = decision.raw_response["pre_execution_price_check"]
    assert price_check["contract_lifecycle"] == "normal_paper_trade"
    assert price_check["profitability_gate_applied"] is False
    assert decision.feature_snapshot["current_price"] == 125.0


@pytest.mark.asyncio
async def test_paper_still_fails_closed_without_fresh_market_fact() -> None:
    decision = _decision(return_lcb=-2.0, expected_net=-3.0)

    reason = await _policy(latest=0.0).guard_reason(decision, "paper")

    assert "fails closed" in reason


@pytest.mark.asyncio
async def test_pre_order_execution_facts_replace_market_and_fee_snapshot() -> None:
    async def fresh_feature(_symbol: str) -> dict[str, Any]:
        raise AssertionError("authoritative execution facts must avoid a duplicate feature refresh")

    async def execution_facts(mode: str, decision: DecisionOutput) -> dict[str, Any]:
        assert mode == "paper"
        assert decision.symbol == "BTC/USDT"
        return {
            "production_eligible": True,
            "inst_id": "BTC-USDT-SWAP",
            "reason": "ready",
            "feature_snapshot": {
                "current_price": 100.1,
                "bid": 100.0,
                "ask": 100.2,
                "mark_price": 100.1,
                "orderbook_bids": [[100.0, 2.0]],
                "orderbook_asks": [[100.2, 2.0]],
                "orderbook_bid_depth": 200.0,
                "orderbook_ask_depth": 200.4,
                "contract_value_base": 1.0,
                "taker_fee_rate": 0.0004,
            },
            "policy_provenance": {"source": "test_okx_native"},
        }

    decision = _decision(return_lcb=0.6)
    policy = EntryPriceGuardPolicy(
        fresh_feature_provider=fresh_feature,
        market_data_quality_reason_provider=lambda _snapshot, **_kwargs: None,
        decision_age_seconds_provider=lambda _decision: 12.0,
        pre_order_execution_facts_provider=execution_facts,
    )

    assert await policy.guard_reason(decision, "paper") is None
    assert decision.feature_snapshot["mark_price"] == 100.1
    assert decision.feature_snapshot["taker_fee_rate"] == 0.0004
    contract = decision.raw_response["pre_order_execution_facts"]
    assert contract["production_eligible"] is True
    assert contract["input_fingerprint"]
    proof = decision.raw_response["pre_execution_price_check"]["native_market_fact_proof"]
    assert proof["fresh_inst_id"] == "BTC-USDT-SWAP"
    assert proof["fresh_source_interface"] == "test_okx_native"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("blocker", "expected"),
    [
        ("okx_pre_order_account_direction_capacity_zero", "最大可开数量为 0 张"),
        ("okx_pre_order_account_order_capacity_unavailable", "可开额度暂时无法核验"),
    ],
)
async def test_account_capacity_blocker_is_explained_before_submission(
    blocker: str, expected: str,
) -> None:
    async def execution_facts(_mode: str, _decision: DecisionOutput) -> dict[str, Any]:
        return {
            "production_eligible": False,
            "reason": blocker,
            "reasons": [blocker],
        }

    policy = EntryPriceGuardPolicy(
        fresh_feature_provider=lambda _symbol: {"current_price": 100.0},
        market_data_quality_reason_provider=lambda _snapshot, **_kwargs: None,
        decision_age_seconds_provider=lambda _decision: 12.0,
        pre_order_execution_facts_provider=execution_facts,
    )
    reason = await policy.guard_reason(_decision(), "paper")
    assert expected in reason
    assert "未提交订单" in reason


@pytest.mark.asyncio
async def test_analysis_only_execution_facts_cannot_submit_even_paper_entry() -> None:
    async def execution_facts(_mode: str, _decision: DecisionOutput) -> dict[str, Any]:
        return {
            "production_eligible": False,
            "reason": "okx_private_entry_instrument_temporarily_unverified",
            "inst_id": "BTC-USDT-SWAP",
            "feature_snapshot": {"current_price": 100.1},
        }

    policy = EntryPriceGuardPolicy(
        fresh_feature_provider=lambda _symbol: {"current_price": 100.0},
        market_data_quality_reason_provider=lambda _snapshot, **_kwargs: None,
        decision_age_seconds_provider=lambda _decision: 12.0,
        pre_order_execution_facts_provider=execution_facts,
    )

    reason = await policy.guard_reason(_decision(), "paper")

    assert "execution facts are incomplete" in reason
    assert "temporarily_unverified" in reason


@pytest.mark.asyncio
async def test_repeated_guard_keeps_original_analysis_price_as_immutable_basis() -> None:
    calls = {"quality": 0}

    async def execution_facts(_mode: str, _decision: DecisionOutput) -> dict[str, Any]:
        return {
            "production_eligible": True,
            "inst_id": "BTC-USDT-SWAP",
            "feature_snapshot": {
                "current_price": 101.0,
                "bid": 100.9,
                "ask": 101.1,
                "mark_price": 101.0,
                "orderbook_bids": [[100.9, 2.0]],
                "orderbook_asks": [[101.1, 2.0]],
                "orderbook_bid_depth": 201.8,
                "orderbook_ask_depth": 202.2,
                "contract_value_base": 1.0,
                "taker_fee_rate": 0.0004,
            },
            "policy_provenance": {"source": "test_okx_native"},
        }

    def quality(snapshot: dict[str, Any], **_kwargs: Any) -> None:
        calls["quality"] += 1
        assert snapshot.get("current_price") == 101.0
        return None

    decision = _decision()
    policy = EntryPriceGuardPolicy(
        fresh_feature_provider=lambda _symbol: execution_facts("paper", decision),
        market_data_quality_reason_provider=quality,
        decision_age_seconds_provider=lambda _decision: 12.0,
        pre_order_execution_facts_provider=execution_facts,
    )

    assert await policy.guard_reason(decision, "paper") is None
    assert await policy.guard_reason(decision, "paper") is None
    assert calls["quality"] == 2
    assert decision.raw_response["pre_execution_price_check"]["snapshot_price"] == 100.0


@pytest.mark.asyncio
async def test_pre_order_execution_fact_instrument_mismatch_fails_closed() -> None:
    async def fresh_feature(_symbol: str) -> dict[str, Any]:
        return {
            "current_price": 100.0,
            "market_fact": {"native_identity": {"inst_id": "BTC-USDT-SWAP"}},
        }

    async def execution_facts(_mode: str, _decision: DecisionOutput) -> dict[str, Any]:
        return {
            "production_eligible": True,
            "inst_id": "ETH-USDT-SWAP",
            "feature_snapshot": {"current_price": 100.0},
        }

    policy = EntryPriceGuardPolicy(
        fresh_feature_provider=fresh_feature,
        market_data_quality_reason_provider=lambda _snapshot, **_kwargs: None,
        decision_age_seconds_provider=lambda _decision: 12.0,
        pre_order_execution_facts_provider=execution_facts,
    )

    reason = await policy.guard_reason(_decision(), "paper")
    assert "instrument mismatch" in reason


def _paper_decision(
    *, action: Action, price: float, expected: float,
) -> DecisionOutput:
    decision = _decision(expected_net=expected, return_lcb=-2.0)
    decision.action = action
    decision.feature_snapshot = {"current_price": price, "close": price}
    decision.raw_response["opportunity_score"] = {"score": 1.0}
    decision.raw_response["normal_paper_trade"] = {
        "version": NORMAL_PAPER_TRADE_VERSION,
        "symbol": decision.symbol,
        "side": action.value,
        "authorized": True,
        "selection_reason": "paper_training_entry",
        "expected_net_return_pct": expected,
        "objective_net_return_pct": -2.0,
        "production_permission": False,
    }
    return decision


def _paper_policy(
    *, latest: float, final_expected: float, with_score: bool = False,
) -> EntryPolicy:
    class FinalSizing:
        async def apply(self, decision, _mode, *, open_positions):
            decision.raw_response["profit_risk_sizing"] = {
                "production_eligible": True,
                "expected_net_return_pct": final_expected,
                "return_lcb_pct": -2.0,
                "final_notional_usdt": 100.0,
            }

    return EntryPolicy(
        entry_price_guard=_policy(latest=latest),
        entry_profit_risk_sizing=FinalSizing(),
        entry_opportunity_score=(
            EntryOpportunityScorePolicy(lambda _decision, _strategy: 1.0)
            if with_score else None
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "price", "latest", "expected"),
    [
        (Action.SHORT, 0.01467, 0.0145495, 0.39785122),  # IRYS, decision 816282
        (Action.SHORT, 1.301, 1.2885, 0.81351882),  # ATOM, decision 818618
        (Action.SHORT, 0.01607, 0.01605, 0.08390255),  # 2Z, decision 823310
        (Action.SHORT, 0.7216, 0.717, 0.20926527),  # DOT, decision 827036
        (Action.LONG, 100.0, 100.5, 0.2),
        (Action.LONG, 100.0, 101.0, 1.0),  # No positive edge remains at equality.
    ],
)
@pytest.mark.parametrize("with_score", [False, True])
async def test_paper_final_sizing_rejects_price_consumed_return(
    action: Action, price: float, latest: float, expected: float, with_score: bool,
) -> None:
    decision = _paper_decision(action=action, price=price, expected=expected)

    reason = await _paper_policy(
        latest=latest, final_expected=expected, with_score=with_score,
    ).prepare_dynamic_risk_contract(decision, "paper", [])

    assert reason is not None
    assert "预期净收益" in reason
    check = decision.raw_response["pre_execution_price_check"]
    assert check["remaining_expected_net_return_pct"] <= 0.0
    assert check["return_budget_basis"] == "final_sizing_expected_net_return_pct"
    assert check["production_permission"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("action,latest", [(Action.LONG, 100.1), (Action.SHORT, 99.9),
                                          (Action.LONG, 99.0), (Action.SHORT, 101.0)])
async def test_paper_positive_mean_with_negative_downside_keeps_entry(
    action: Action, latest: float,
) -> None:
    decision = _paper_decision(action=action, price=100.0, expected=0.2)

    reason = await _paper_policy(
        latest=latest, final_expected=0.2, with_score=True,
    ).prepare_dynamic_risk_contract(decision, "paper", [])

    assert reason is None
    assert decision.raw_response["pre_execution_price_check"][
        "remaining_expected_net_return_pct"
    ] > 0.0
    assert decision.raw_response["normal_paper_trade"]["objective_net_return_pct"] == -2.0
    assert decision.raw_response["profit_risk_sizing"]["final_notional_usdt"] == 100.0


@pytest.mark.asyncio
async def test_paper_price_budget_uses_final_costs_without_rebasing_original_price() -> None:
    decision = _paper_decision(action=Action.LONG, price=100.0, expected=0.8)
    policy = _paper_policy(latest=100.3, final_expected=0.5, with_score=True)
    assert await policy.prepare_dynamic_risk_contract(decision, "paper", []) is None
    # A later cost refresh consumes more of the original signal, not a signal
    # reset to the previously refreshed price of 100.3.
    policy.entry_profit_risk_sizing = _paper_policy(
        latest=100.3, final_expected=0.2,
    ).entry_profit_risk_sizing

    reason = await policy.prepare_dynamic_risk_contract(decision, "paper", [])

    assert reason is not None
    check = decision.raw_response["pre_execution_price_check"]
    assert check["snapshot_price"] == 100.0
    assert check["latest_price"] == 100.3
    assert check["return_budget_fraction"] == pytest.approx(0.002)
    assert check["remaining_expected_net_return_pct"] == pytest.approx(-0.1)


@pytest.mark.asyncio
async def test_paper_price_budget_can_use_improved_final_execution_cost() -> None:
    decision = _paper_decision(action=Action.LONG, price=100.0, expected=0.1)

    reason = await _paper_policy(
        latest=100.3, final_expected=0.5, with_score=True,
    ).prepare_dynamic_risk_contract(decision, "paper", [])

    assert reason is None
    assert decision.raw_response["pre_execution_price_check"][
        "remaining_expected_net_return_pct"
    ] == pytest.approx(0.2)


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["latest_price", "snapshot_price"])
async def test_final_paper_budget_cannot_use_missing_or_rebased_price(field: str) -> None:
    decision = _paper_decision(action=Action.LONG, price=100.0, expected=0.5)
    policy = _paper_policy(latest=100.1, final_expected=0.5, with_score=True)
    assert await policy.prepare_dynamic_risk_contract(decision, "paper", []) is None
    decision.raw_response["pre_execution_price_check"][field] = None

    assert policy.final_price_budget_reason(decision, "paper") is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("action,latest", [(Action.LONG, 102.1), (Action.SHORT, 97.9)])
async def test_quote_outside_venue_band_keeps_the_actual_rejection_reason(
    action: Action, latest: float,
) -> None:
    decision = _paper_decision(action=action, price=100.0, expected=5.0)
    assert await _paper_policy(latest=latest, final_expected=5.0).prepare_dynamic_risk_contract(
        decision, "paper", [],
    ) is None

    guard = paper_entry_execution_price_limit(
        decision, "paper", latest_price=latest, tick_size=0.1,
        price_limit={"buy_price_limit": 102.0, "sell_price_limit": 98.0},
    )

    assert guard["eligible"] is False
    assert guard["reason"] == "paper_execution_quote_outside_venue_price_limit"
    assert guard["remaining_expected_net_return_pct"] > 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "price,checked,submit,expected,tick",
    [
        (0.04274, 0.042725, 0.04264, 0.10041862, 0.00001),  # CRO 828451
        (0.09615, 0.096, 0.0958, 0.26865573, 0.00001),  # IMX 828485
    ],
)
async def test_final_execution_quote_cannot_bypass_a_previously_positive_budget(
    price: float, checked: float, submit: float, expected: float, tick: float,
) -> None:
    decision = _paper_decision(action=Action.SHORT, price=price, expected=expected)
    assert await _paper_policy(
        latest=checked, final_expected=expected,
    ).prepare_dynamic_risk_contract(decision, "paper", []) is None

    guard = paper_entry_execution_price_limit(
        decision, "paper", latest_price=submit, tick_size=tick,
    )

    assert guard["eligible"] is False
    assert guard["remaining_expected_net_return_pct"] < 0.0
    assert decision.raw_response["pre_execution_price_check"]["latest_price"] == checked


@pytest.mark.asyncio
async def test_sushi_market_fallback_cannot_fill_below_the_signal_budget() -> None:
    decision = _paper_decision(action=Action.SHORT, price=0.153, expected=0.21584328)
    assert await _paper_policy(
        latest=0.1528, final_expected=0.21584328,
    ).prepare_dynamic_risk_contract(decision, "paper", []) is None

    guard = paper_entry_execution_price_limit(
        decision, "paper", latest_price=0.1527, tick_size=0.0001,
    )

    assert guard["eligible"] is True
    assert guard["limit_price"] == pytest.approx(0.1527)
    assert guard["limit_price"] > 0.15240104956268222  # Actual unbounded fill, 828488.
    assert guard["remaining_expected_net_return_pct"] > 0.0
    assert decision.raw_response["normal_paper_trade"]["objective_net_return_pct"] < 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "price,quote,fill,expected,tick",
    [
        (0.05844, 0.0589612, 0.0589346552653749, 0.84443495, 0.0000001),
        (7.244, 7.242, 7.238231092436975, 1.10975784, 0.001),
        (44.56, 44.5249, 44.52009735099339, 0.16097076, 0.0001),
    ],
)
async def test_execution_cap_preserves_doge_avax_ltc_profitable_entry_prices(
    price: float, quote: float, fill: float, expected: float, tick: float,
) -> None:
    decision = _paper_decision(action=Action.SHORT, price=price, expected=expected)
    assert await _paper_policy(
        latest=quote, final_expected=expected,
    ).prepare_dynamic_risk_contract(decision, "paper", []) is None

    guard = paper_entry_execution_price_limit(
        decision, "paper", latest_price=quote, tick_size=tick,
    )

    assert guard["eligible"] is True
    assert fill >= guard["limit_price"]


@pytest.mark.asyncio
@pytest.mark.parametrize("action", [Action.LONG, Action.SHORT])
async def test_execution_limit_leaves_positive_budget_at_an_exact_tick_boundary(
    action: Action,
) -> None:
    decision = _paper_decision(action=action, price=100.0, expected=1.0)
    assert await _paper_policy(
        latest=100.0, final_expected=1.0,
    ).prepare_dynamic_risk_contract(decision, "paper", []) is None

    guard = paper_entry_execution_price_limit(
        decision, "paper", latest_price=100.0, tick_size=0.1,
    )

    assert guard["eligible"] is True
    assert guard["limit_price"] == pytest.approx(100.9 if action == Action.LONG else 99.1)
    assert guard["worst_remaining_expected_net_return_pct"] > 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action,price,quote,expected,tick,stop,target,limit",
    [
        (Action.SHORT, 0.9557, 0.9574, 1.18797556, 0.0001, 0.9696, 0.9452, 0.9453),
        (Action.LONG, 100.0, 100.0, 2.0, 0.1, 99.0, 101.0, 100.9),
        (Action.SHORT, 100.0, 100.0, 2.0, 0.1, 101.0, 99.0, 99.1),
        (Action.LONG, 100.0, 100.0, 2.0, 0.1, 99.0, 101.05, 101.0),
        (Action.SHORT, 100.0, 100.0, 2.0, 0.1, 101.0, 98.95, 99.0),
    ],
)
async def test_bounded_entry_stays_inside_attached_protection(
    action: Action, price: float, quote: float, expected: float, tick: float,
    stop: float, target: float, limit: float,
) -> None:
    decision = _paper_decision(action=action, price=price, expected=expected)
    assert await _paper_policy(
        latest=quote, final_expected=expected,
    ).prepare_dynamic_risk_contract(decision, "paper", []) is None
    protection = {"stop_loss_price": str(stop), "take_profit_price": str(target)}

    guard = paper_entry_execution_price_limit(
        decision, "paper", latest_price=quote, tick_size=tick, protection=protection,
    )

    assert guard["eligible"] is True
    assert guard["protection_compatible"] is True
    assert guard["limit_price"] == pytest.approx(limit)
    assert guard["worst_remaining_expected_net_return_pct"] > 0.0
    if action == Action.LONG:
        assert stop < guard["limit_price"] < target
        assert guard["limit_price"] <= guard["return_budget_limit_price"]
    else:
        assert target < guard["limit_price"] < stop
        assert guard["limit_price"] >= guard["return_budget_limit_price"]
    assert protection == {"stop_loss_price": str(stop), "take_profit_price": str(target)}


@pytest.mark.asyncio
@pytest.mark.parametrize("action", [Action.LONG, Action.SHORT])
async def test_protection_cannot_loosen_a_tighter_entry_return_limit(action: Action) -> None:
    decision = _paper_decision(action=action, price=100.0, expected=0.5)
    assert await _paper_policy(
        latest=100.0, final_expected=0.5,
    ).prepare_dynamic_risk_contract(decision, "paper", []) is None
    protection = {
        "stop_loss_price": 99.0 if action == Action.LONG else 101.0,
        "take_profit_price": 102.0 if action == Action.LONG else 98.0,
    }

    guard = paper_entry_execution_price_limit(
        decision, "paper", latest_price=100.0, tick_size=0.1, protection=protection,
    )

    assert guard["eligible"] is True
    assert guard["limit_price"] == pytest.approx(100.4 if action == Action.LONG else 99.6)
    assert guard["limit_price"] == guard["return_budget_limit_price"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action,price,quote,expected,tick,stop,target,buy_limit,sell_limit,limit",
    [
        (Action.SHORT, 0.9557, 0.9574, 1.18797556, 0.0001, 0.9696, 0.9452, 0.9668, 0.9475, 0.9475),
        (Action.LONG, 100.0, 100.0, 3.0, 0.1, 98.0, 104.0, 101.05, 99.0, 101.0),
        (Action.SHORT, 100.0, 100.0, 3.0, 0.1, 102.0, 96.0, 101.0, 98.95, 99.0),
    ],
)
async def test_bounded_entry_intersects_return_protection_and_venue_limits(
    action: Action, price: float, quote: float, expected: float, tick: float,
    stop: float, target: float, buy_limit: float, sell_limit: float, limit: float,
) -> None:
    decision = _paper_decision(action=action, price=price, expected=expected)
    assert await _paper_policy(
        latest=quote, final_expected=expected,
    ).prepare_dynamic_risk_contract(decision, "paper", []) is None

    guard = paper_entry_execution_price_limit(
        decision, "paper", latest_price=quote, tick_size=tick,
        protection={"stop_loss_price": stop, "take_profit_price": target},
        price_limit={"buy_price_limit": buy_limit, "sell_price_limit": sell_limit},
    )

    assert guard["eligible"] is True
    assert guard["limit_price"] == pytest.approx(limit)
    assert guard["venue_price_limit_compatible"] is True
    assert guard["protection_compatible"] is True
    assert sell_limit <= limit <= buy_limit
    assert guard["worst_remaining_expected_net_return_pct"] > 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("price_limit", [{}, {"buy_price_limit": 99.0, "sell_price_limit": 101.0}])
async def test_bounded_entry_requires_valid_venue_bounds_when_supplied(
    price_limit: dict[str, Any],
) -> None:
    decision = _paper_decision(action=Action.LONG, price=100.0, expected=1.0)
    assert await _paper_policy(
        latest=100.0, final_expected=1.0,
    ).prepare_dynamic_risk_contract(decision, "paper", []) is None

    guard = paper_entry_execution_price_limit(
        decision, "paper", latest_price=100.0, tick_size=0.1, price_limit=price_limit,
    )

    assert guard["eligible"] is False
    assert guard["reason"] == "paper_execution_venue_price_limit_incompatible"


@pytest.mark.asyncio
async def test_no_entry_when_venue_and_protection_have_no_common_price() -> None:
    decision = _paper_decision(action=Action.LONG, price=94.0, expected=20.0)
    assert await _paper_policy(
        latest=94.0, final_expected=20.0,
    ).prepare_dynamic_risk_contract(decision, "paper", []) is None

    guard = paper_entry_execution_price_limit(
        decision, "paper", latest_price=94.0, tick_size=0.1,
        protection={"stop_loss_price": 90.0, "take_profit_price": 95.0},
        price_limit={"buy_price_limit": 110.0, "sell_price_limit": 99.0},
    )

    assert guard["eligible"] is False
    assert guard["reason"] == "paper_execution_venue_price_limit_incompatible"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action,stop,target",
    [
        (Action.LONG, 100.0, 102.0),
        (Action.LONG, 99.0, 100.0),
        (Action.SHORT, 100.0, 98.0),
        (Action.SHORT, 101.0, 100.0),
        (Action.LONG, None, 102.0),
        (Action.SHORT, 101.0, float("nan")),
    ],
)
async def test_execution_limit_rejects_incompatible_protection(
    action: Action, stop: float | None, target: float,
) -> None:
    decision = _paper_decision(action=action, price=100.0, expected=1.0)
    assert await _paper_policy(
        latest=100.0, final_expected=1.0,
    ).prepare_dynamic_risk_contract(decision, "paper", []) is None

    guard = paper_entry_execution_price_limit(
        decision, "paper", latest_price=100.0, tick_size=0.1,
        protection={"stop_loss_price": stop, "take_profit_price": target},
    )

    assert guard["eligible"] is False
    assert guard["reason"] == "paper_execution_protection_incompatible"


@pytest.mark.asyncio
@pytest.mark.parametrize("tick", [0.0, float("nan")])
async def test_execution_limit_requires_a_real_tick_size(tick: float) -> None:
    decision = _paper_decision(action=Action.SHORT, price=100.0, expected=1.0)
    assert await _paper_policy(
        latest=100.0, final_expected=1.0,
    ).prepare_dynamic_risk_contract(decision, "paper", []) is None

    assert paper_entry_execution_price_limit(
        decision, "paper", latest_price=100.0, tick_size=tick,
    )["eligible"] is False


def test_execution_limit_does_not_add_a_new_live_or_exit_gate() -> None:
    decision = _paper_decision(action=Action.SHORT, price=100.0, expected=1.0)
    assert paper_entry_execution_price_limit(
        decision, "live", latest_price=1.0, tick_size=0.1,
    ) is None
    decision.action = Action.CLOSE_SHORT
    assert paper_entry_execution_price_limit(
        decision, "paper", latest_price=1.0, tick_size=0.1,
    ) is None
