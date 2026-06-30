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
async def test_low_quality_release_does_not_bypass_fee_churn_guard() -> None:
    decision = _decision(
        current_price=99.95,
        confidence=0.95,
        reasoning="策略纪律触发低质量持仓释放",
        raw_response={
            "forced_exit": True,
            "exit_intent": "hard_risk",
            "position_release_policy": {
                "source": "position_quality_capacity_release",
                "forced": True,
            },
            "close_evidence": {
                "forced_exit": True,
                "hard_risk": False,
                "source": "low_quality_position_release",
            },
        },
    )
    positions = [
        _position(
            entry_price=100.0,
            current_price=99.95,
            created_at=datetime.now(UTC) - timedelta(minutes=20),
        )
    ]

    reason = await _policy(positions, entry_fee=0.05).guard_reason(
        "ensemble_trader",
        decision,
    )

    assert reason is not None
    assert decision.raw_response["exit_intent"] == "capital_rotation"
    assert decision.raw_response["exit_quality"]["net_profit_after_fee"] < 0


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


@pytest.mark.asyncio
async def test_small_position_profit_lock_uses_dynamic_partial_floor() -> None:
    decision = _decision(
        action=Action.CLOSE_SHORT,
        current_price=0.1599,
        confidence=0.82,
        reasoning="小仓动态锁盈",
        position_size_pct=0.45,
        raw_response={
            "close_evidence": {
                "profit_protection": True,
                "small_position_profit_lock": True,
                "action_plan": "reduce",
            },
        },
    )
    positions = [
        _position(
            symbol="MET/USDT",
            quantity=150.0,
            entry_price=0.1713667,
            current_price=0.1599,
            created_at=datetime.now(UTC) - timedelta(hours=12),
        )
    ]

    reason = await _policy(positions).guard_reason("ensemble_trader", decision)

    assert reason is None
    assert "small_profit_lock_guard" not in decision.raw_response
    protection = decision.raw_response["execution_profit_protection"]
    assert protection["allow"] is True
    assert protection["small_position_lock"] is True


@pytest.mark.asyncio
async def test_winner_run_guard_blocks_ordinary_full_close_when_trend_still_valid() -> None:
    decision = _decision(
        current_price=104.0,
        confidence=0.72,
        reasoning="普通锁盈全平",
        position_size_pct=1.0,
    )
    positions = [
        _position(
            entry_price=100.0,
            current_price=104.0,
            created_at=datetime.now(UTC) - timedelta(minutes=20),
        )
    ]

    reason = await _policy(positions).guard_reason("ensemble_trader", decision)

    assert reason is not None
    assert "赢家持仓保护" in reason
    guard = decision.raw_response["winner_run_guard"]
    assert guard["applied"] is True
    assert guard["close_pct"] == 1.0
    assert guard["trend_still_valid"] is True
    assert guard["strong_lock"] is False
