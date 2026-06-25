from __future__ import annotations

from types import SimpleNamespace

import pytest

from ai_brain.base_model import Action, DecisionOutput
from services.decision_final_state_ensurer import DecisionFinalStateEnsurer


def _decision(action: Action = Action.LONG) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=action,
        confidence=0.8,
        reasoning="test final state",
        position_size_pct=0.05,
    )


def _ensurer() -> DecisionFinalStateEnsurer:
    return DecisionFinalStateEnsurer(
        execution_reason_unusable_checker=lambda reason: "unusable" in str(reason),
        execution_reason_recoverer=lambda _row: "恢复出的平仓原因",
        model_execution_mode_provider=lambda _model_name: "paper",
    )


@pytest.mark.asyncio
async def test_decision_final_state_ensurer_marks_pending_entry_without_order() -> None:
    row = SimpleNamespace(
        was_executed=False,
        execution_reason="正在提交 OKX：排序后进入执行",
        action="long",
    )
    results = {"decisions": []}
    flushed = False

    async def flush() -> None:
        nonlocal flushed
        flushed = True

    await _ensurer().ensure_row(
        row,
        order_count=0,
        symbol="BTC/USDT",
        model_name="ensemble_trader",
        decision=_decision(),
        results=results,
        flush_callback=flush,
    )

    assert flushed
    assert "45 秒内没有生成本地订单记录" in row.execution_reason
    assert results["decisions"][0]["execution_status"] == "error"
    assert results["decisions"][0]["is_paper"] is True


@pytest.mark.asyncio
async def test_decision_final_state_ensurer_keeps_pending_with_order_record() -> None:
    row = SimpleNamespace(
        was_executed=False,
        execution_reason="正在提交 OKX：已提交",
        action="long",
    )
    results = {"decisions": []}

    await _ensurer().ensure_row(
        row,
        order_count=1,
        symbol="BTC/USDT",
        model_name="ensemble_trader",
        decision=_decision(),
        results=results,
    )

    assert "本地订单记录已生成" in row.execution_reason
    assert results["decisions"] == []


@pytest.mark.asyncio
async def test_decision_final_state_ensurer_recovers_unusable_exit_reason() -> None:
    row = SimpleNamespace(
        was_executed=False,
        execution_reason="unusable",
        action="close_long",
    )
    results = {"decisions": []}

    await _ensurer().ensure_row(
        row,
        order_count=0,
        symbol="BTC/USDT",
        model_name="ensemble_trader",
        decision=_decision(Action.CLOSE_LONG),
        results=results,
    )

    assert row.execution_reason == "恢复出的平仓原因"
    assert results["decisions"][0]["execution_status"] == "skipped"


@pytest.mark.asyncio
async def test_decision_final_state_ensurer_finalizes_exit_when_recovery_has_no_reason() -> None:
    ensurer = DecisionFinalStateEnsurer(
        execution_reason_unusable_checker=lambda _reason: False,
        execution_reason_recoverer=lambda _row: None,
        model_execution_mode_provider=lambda _model_name: "paper",
    )
    row = SimpleNamespace(
        was_executed=False,
        execution_reason="",
        action="close_long",
    )
    results = {"decisions": []}
    flushed = False

    async def flush() -> None:
        nonlocal flushed
        flushed = True

    await ensurer.ensure_row(
        row,
        order_count=0,
        symbol="BTC/USDT",
        model_name="ensemble_trader",
        decision=_decision(Action.CLOSE_LONG),
        results=results,
        flush_callback=flush,
    )

    assert flushed
    assert "平仓裁决没有生成本地平仓委托" in row.execution_reason
    assert results["decisions"][0]["execution_status"] == "skipped"
    assert results["decisions"][0]["is_paper"] is True
