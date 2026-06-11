from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from ai_brain.base_model import Action, DecisionOutput
from services.exit_fee_churn_guard import ExitFeeChurnGuardPolicy
from services.forced_exit import ForcedExitPolicy


class _SessionContext:
    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _Repo:
    def __init__(self, positions: list[SimpleNamespace]) -> None:
        self._positions = positions

    async def get_matching_open_positions(self, **kwargs: Any) -> list[SimpleNamespace]:
        return self._positions


def _position(**overrides: Any) -> SimpleNamespace:
    data = {
        "symbol": "BTC/USDT",
        "quantity": 1.0,
        "entry_price": 100.0,
        "current_price": 100.0,
        "stop_loss_price": None,
        "take_profit_price": None,
        "created_at": datetime.now(UTC) - timedelta(minutes=10),
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def _decision(
    *,
    action: Action = Action.CLOSE_LONG,
    current_price: float = 100.0,
    confidence: float = 0.7,
    reasoning: str = "普通平仓",
    position_size_pct: float = 1.0,
    raw_response: dict[str, Any] | None = None,
) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=action,
        confidence=confidence,
        reasoning=reasoning,
        position_size_pct=position_size_pct,
        suggested_leverage=3.0,
        raw_response=raw_response or {},
        feature_snapshot={"current_price": current_price},
    )


def _policy(
    positions: list[SimpleNamespace],
    *,
    entry_fee: float = 0.0,
    invalidation: dict[str, Any] | None = None,
    position_peaks: dict[str, dict[str, Any]] | None = None,
) -> ExitFeeChurnGuardPolicy:
    async def entry_fee_provider(session: object, pos: Any, qty: float) -> float:
        return entry_fee

    return ExitFeeChurnGuardPolicy(
        session_factory=_SessionContext,
        model_execution_mode_provider=lambda model_name: "paper",
        entry_fee_provider=entry_fee_provider,
        invalidation_snapshot_provider=lambda decision, side, entry, price: invalidation or {},
        forced_exit_policy=ForcedExitPolicy(),
        position_peaks=position_peaks or {},
        position_peak_key_provider=lambda model, symbol, side: f"{model}|{symbol}|{side}",
        trade_repository_factory=lambda session: _Repo(positions),
    )


@pytest.mark.asyncio
async def test_aggregate_exit_guard_blocks_loss_repair_when_group_is_not_losing() -> None:
    decision = _decision(current_price=100.0, reasoning="单个分片浮亏，建议亏损修复止损")
    positions = [
        _position(entry_price=90.0, current_price=100.0),
        _position(entry_price=100.0, current_price=100.0),
    ]

    reason = await _policy(positions).guard_reason("ensemble_trader", decision)

    assert reason is not None
    assert "整体持仓保护" in reason
    guard = decision.raw_response["aggregate_exit_guard"]
    assert guard["applied"] is True
    assert guard["aggregate_gross_pnl"] == 10.0


@pytest.mark.asyncio
async def test_aggregate_exit_guard_allows_predictive_downside_stop_exit() -> None:
    decision = _decision(
        current_price=100.0,
        reasoning="AI 分析后续可能会跌，为了保护本金先止损平多。",
        raw_response={
            "close_evidence": {
                "should_close": True,
                "moderate_opposite_pressure": True,
                "preventive_exit": True,
            }
        },
    )
    positions = [
        _position(entry_price=90.0, current_price=100.0),
        _position(entry_price=100.0, current_price=100.0),
    ]

    reason = await _policy(positions).guard_reason("ensemble_trader", decision)

    assert reason is None
    assert "aggregate_exit_guard" not in decision.raw_response


@pytest.mark.asyncio
async def test_aggregate_exit_guard_allows_structured_predictive_downside_intent() -> None:
    decision = _decision(
        current_price=100.0,
        reasoning="loss stop before downside expands",
        raw_response={
            "exit_intent": "predictive_downside",
            "close_evidence": {"should_close": True},
        },
    )
    positions = [
        _position(entry_price=90.0, current_price=100.0),
        _position(entry_price=100.0, current_price=100.0),
    ]

    reason = await _policy(positions).guard_reason("ensemble_trader", decision)

    assert reason is None
    assert decision.raw_response["exit_intent"] == "predictive_downside"
    assert "aggregate_exit_guard" not in decision.raw_response


@pytest.mark.asyncio
async def test_aggregate_exit_guard_uses_net_pnl_after_fees() -> None:
    decision = _decision(current_price=100.1, reasoning="单个分片浮亏，建议亏损修复止损")
    positions = [_position(entry_price=100.0, current_price=100.1)]

    reason = await _policy(positions, entry_fee=0.20).guard_reason(
        "ensemble_trader",
        decision,
    )

    assert reason is None
    assert "aggregate_exit_guard" not in decision.raw_response


@pytest.mark.asyncio
async def test_entry_settlement_guard_blocks_ordinary_exit_soon_after_entry() -> None:
    decision = _decision(current_price=100.4, reasoning="普通 AI 降低风险")
    positions = [_position(created_at=datetime.now(UTC) - timedelta(seconds=30))]

    reason = await _policy(positions).guard_reason("ensemble_trader", decision)

    assert reason is not None
    assert "成交结算防抖窗口" in reason


@pytest.mark.asyncio
async def test_forced_exit_bypasses_entry_settlement_guard() -> None:
    decision = _decision(
        current_price=100.4,
        reasoning="快速风控要求强制平仓",
        raw_response={"forced_exit": True},
    )
    positions = [_position(created_at=datetime.now(UTC) - timedelta(seconds=30))]

    reason = await _policy(positions).guard_reason("ensemble_trader", decision)

    assert reason is None


@pytest.mark.asyncio
async def test_small_partial_profit_lock_is_blocked_when_planned_net_is_too_small() -> None:
    decision = _decision(
        current_price=103.2,
        confidence=0.82,
        reasoning="锁盈",
        position_size_pct=0.25,
    )
    positions = [
        _position(
            entry_price=100.0,
            current_price=103.2,
            created_at=datetime.now(UTC) - timedelta(minutes=12),
        )
    ]

    reason = await _policy(positions).guard_reason("ensemble_trader", decision)

    assert reason is not None
    assert "碎片化小额平仓" in reason
    guard = decision.raw_response["small_profit_lock_guard"]
    assert guard["applied"] is True
