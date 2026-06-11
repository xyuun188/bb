from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ai_brain.base_model import Action, DecisionOutput
from services.entry_price_guard import EntryPriceGuardPolicy


def _decision(
    *,
    action: Action = Action.LONG,
    current_price: float = 100.0,
    raw_response: dict[str, Any] | None = None,
) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=action,
        confidence=0.8,
        reasoning="测试开仓",
        position_size_pct=0.05,
        suggested_leverage=3.0,
        raw_response=raw_response or {},
        feature_snapshot={"current_price": current_price, "close": current_price},
    )


def _feature(snapshot: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(to_dict=lambda: snapshot)


def _policy(
    *,
    latest_price: float,
    fresh_snapshot: dict[str, Any] | None = None,
    quality_reasons: dict[str, str | None] | None = None,
    block_calls: list[tuple[str, str, float]] | None = None,
    max_slippage_pct: float = 0.005,
) -> EntryPriceGuardPolicy:
    async def latest_price_provider(symbol: str) -> float:
        return latest_price

    async def fresh_feature_provider(symbol: str) -> Any:
        if fresh_snapshot is None:
            return None
        return _feature(fresh_snapshot)

    def market_data_quality_reason(snapshot: dict[str, Any], *, stage_label: str) -> str | None:
        return (quality_reasons or {}).get(stage_label)

    def record_block(symbol: str, reason: str, minutes: float) -> None:
        if block_calls is not None:
            block_calls.append((symbol, reason, minutes))

    return EntryPriceGuardPolicy(
        latest_price_provider=latest_price_provider,
        fresh_feature_provider=fresh_feature_provider,
        market_data_quality_reason_provider=market_data_quality_reason,
        decision_age_seconds_provider=lambda decision: 12.0,
        temporary_entry_block_recorder=record_block,
        temporary_block_minutes=8.0,
        config=SimpleNamespace(max_slippage_pct=max_slippage_pct),
    )


@pytest.mark.asyncio
async def test_entry_price_guard_blocks_when_latest_price_is_missing() -> None:
    reason = await _policy(latest_price=0.0).guard_reason(_decision())

    assert reason == "下单前没有重新拿到最新价格，系统不使用过期行情盲目下单，本次跳过。"


@pytest.mark.asyncio
async def test_entry_price_guard_blocks_adverse_long_chase_and_records_cooldown() -> None:
    block_calls: list[tuple[str, str, float]] = []
    decision = _decision(current_price=100.0)

    reason = await _policy(
        latest_price=103.0,
        block_calls=block_calls,
    ).guard_reason(decision)

    assert reason is not None
    assert "避免追高" in reason
    assert decision.raw_response["pre_execution_price_check"]["move_pct"] == 3.0
    assert decision.raw_response["pre_execution_price_recheck"]["rescued"] is False
    assert len(block_calls) == 1
    assert block_calls[0][0] == "BTC/USDT"
    assert block_calls[0][2] == 8.0


@pytest.mark.asyncio
async def test_entry_price_guard_rescues_small_drift_when_fresh_market_confirms() -> None:
    decision = _decision(
        current_price=100.0,
        raw_response={
            "opportunity_score": {
                "expected_net_return_pct": 0.82,
                "profit_quality_ratio": 0.5,
            }
        },
    )

    reason = await _policy(
        latest_price=100.4,
        fresh_snapshot={
            "current_price": 100.4,
            "close": 100.4,
            "returns_1": 0.0,
            "returns_5": 0.0,
        },
        max_slippage_pct=0.003,
    ).guard_reason(decision)

    assert reason is None
    assert decision.feature_snapshot["current_price"] == 100.4
    recheck = decision.raw_response["pre_execution_price_recheck"]
    assert recheck["triggered"] is True
    assert recheck["rescued"] is True


@pytest.mark.asyncio
async def test_entry_price_guard_rechecks_bad_snapshot_quality_before_blocking() -> None:
    reason = await _policy(
        latest_price=100.0,
        fresh_snapshot=None,
        quality_reasons={"下单前分析快照": "盘口数据异常"},
    ).guard_reason(_decision())

    assert reason is not None
    assert "行情质量复核未通过" in reason
