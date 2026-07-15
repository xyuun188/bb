from __future__ import annotations

from typing import Any

import pytest

from ai_brain.base_model import Action, DecisionOutput
from services.entry_price_guard import EntryPriceGuardPolicy


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
async def test_invalid_analysis_fact_cannot_be_rescued_by_a_fresh_snapshot() -> None:
    async def fresh_feature(_symbol: str) -> Any:
        raise AssertionError("dirty analysis must be blocked before refresh")

    policy = EntryPriceGuardPolicy(
        fresh_feature_provider=fresh_feature,
        market_data_quality_reason_provider=lambda _snapshot, **_kwargs: "dirty fact",
        decision_age_seconds_provider=lambda _decision: 12.0,
    )

    reason = await policy.guard_reason(_decision())

    assert "analysis market fact is invalid" in reason


@pytest.mark.asyncio
async def test_missing_authoritative_return_budget_fails_closed() -> None:
    reason = await _policy(latest=100.0).guard_reason(
        _decision(return_lcb=0.0, expected_net=9.0)
    )
    assert "return budget is missing" in reason


@pytest.mark.asyncio
async def test_pre_order_execution_facts_replace_market_and_fee_snapshot() -> None:
    async def fresh_feature(_symbol: str) -> dict[str, Any]:
        return {
            "current_price": 100.1,
            "market_fact": {"native_identity": {"inst_id": "BTC-USDT-SWAP"}},
        }

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
