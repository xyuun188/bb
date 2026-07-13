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
    async def latest_price(_symbol: str) -> float:
        return latest

    async def fresh_feature(_symbol: str) -> Any:
        return fresh

    return EntryPriceGuardPolicy(
        latest_price_provider=latest_price,
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
async def test_fresh_snapshot_can_rebase_stale_analysis_without_fixed_rescue_threshold() -> None:
    decision = _decision(return_lcb=0.2)
    reason = await _policy(
        latest=101.0,
        fresh={"current_price": 101.0, "close": 101.0},
    ).guard_reason(decision)
    assert reason is None
    assert decision.feature_snapshot["current_price"] == 101.0


@pytest.mark.asyncio
async def test_missing_authoritative_return_budget_fails_closed() -> None:
    reason = await _policy(latest=100.0).guard_reason(
        _decision(return_lcb=0.0, expected_net=9.0)
    )
    assert "return budget is missing" in reason
