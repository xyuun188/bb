from __future__ import annotations

from typing import Any

from ai_brain.base_model import Action, DecisionOutput
from services.entry_immediate_execution import (
    DEFAULT_AUTO_SCAN_ENTRY_EXECUTION_REASON,
    EntryImmediateExecutionPlanner,
)


def _decision() -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.LONG,
        confidence=0.82,
        reasoning="entry",
    )


def _planner(
    *,
    immediate_reason: str | None = None,
    capacity_reason: str | None = None,
    calls: list[tuple[str, Any]] | None = None,
) -> EntryImmediateExecutionPlanner:
    call_log = calls if calls is not None else []

    def immediate_reason_provider(decision: DecisionOutput) -> str | None:
        call_log.append(("immediate", decision.symbol))
        return immediate_reason

    def capacity_reason_provider(
        model_name: str,
        decision: DecisionOutput,
        open_positions: list[dict[str, Any]],
        staged_entry_counts: dict[str, dict[Any, int]],
    ) -> str | None:
        call_log.append(("capacity", model_name, decision.symbol, len(open_positions)))
        return capacity_reason

    def reserve_capacity(
        model_name: str,
        decision: DecisionOutput,
        staged_entry_counts: dict[str, dict[Any, int]],
    ) -> None:
        call_log.append(("reserve", model_name, decision.symbol))
        staged_entry_counts.setdefault("reserved", {})[model_name] = 1

    return EntryImmediateExecutionPlanner(
        immediate_reason_provider=immediate_reason_provider,
        capacity_reason_provider=capacity_reason_provider,
        capacity_reserver=reserve_capacity,
    )


def test_entry_immediate_execution_planner_reserves_strong_signal() -> None:
    calls: list[tuple[str, Any]] = []
    staged_counts: dict[str, dict[Any, int]] = {}

    plan = _planner(immediate_reason="强信号", calls=calls).plan(
        model_name="ensemble_trader",
        decision=_decision(),
        open_positions=[],
        staged_entry_counts=staged_counts,
    )

    assert plan.should_execute is True
    assert plan.reason == "强信号"
    assert plan.is_strong_signal is True
    assert plan.capacity_reason is None
    assert staged_counts["reserved"]["ensemble_trader"] == 1
    assert calls == [
        ("immediate", "BTC/USDT"),
        ("capacity", "ensemble_trader", "BTC/USDT", 0),
        ("reserve", "ensemble_trader", "BTC/USDT"),
    ]


def test_entry_immediate_execution_planner_blocks_strong_signal_capacity() -> None:
    staged_counts: dict[str, dict[Any, int]] = {}

    plan = _planner(
        immediate_reason="强信号",
        capacity_reason="当前持仓数已达上限。",
    ).plan(
        model_name="ensemble_trader",
        decision=_decision(),
        open_positions=[{"symbol": "ETH/USDT"}],
        staged_entry_counts=staged_counts,
    )

    assert plan.should_execute is False
    assert plan.reason == "强信号未即时执行：当前持仓数已达上限。"
    assert plan.is_strong_signal is True
    assert plan.capacity_reason == "当前持仓数已达上限。"
    assert staged_counts == {}


def test_entry_immediate_execution_planner_uses_default_reason_for_regular_entry() -> None:
    staged_counts: dict[str, dict[Any, int]] = {}

    plan = _planner().plan(
        model_name="ensemble_trader",
        decision=_decision(),
        open_positions=[],
        staged_entry_counts=staged_counts,
    )

    assert plan.should_execute is True
    assert plan.reason == DEFAULT_AUTO_SCAN_ENTRY_EXECUTION_REASON
    assert plan.is_strong_signal is False
    assert staged_counts["reserved"]["ensemble_trader"] == 1


def test_entry_immediate_execution_planner_blocks_regular_entry_capacity() -> None:
    staged_counts: dict[str, dict[Any, int]] = {}

    plan = _planner(capacity_reason="当前持仓数已达上限。").plan(
        model_name="ensemble_trader",
        decision=_decision(),
        open_positions=[],
        staged_entry_counts=staged_counts,
    )

    assert plan.should_execute is False
    assert plan.reason == "开仓信号未即时执行：当前持仓数已达上限。"
    assert plan.is_strong_signal is False
    assert plan.capacity_reason == "当前持仓数已达上限。"
    assert staged_counts == {}
