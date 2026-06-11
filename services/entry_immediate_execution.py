"""Immediate entry execution planning for auto-scan decisions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import DecisionOutput

ImmediateReasonProvider = Callable[[DecisionOutput], str | None]
CapacityReasonProvider = Callable[
    [str, DecisionOutput, list[dict[str, Any]], dict[str, dict[Any, int]]],
    str | None,
]
CapacityReserver = Callable[[str, DecisionOutput, dict[str, dict[Any, int]]], None]

DEFAULT_AUTO_SCAN_ENTRY_EXECUTION_REASON = (
    "开仓信号已通过 AI 和执行前严重风险检查，立即进入下单流程；"
    "不再等待本轮候选排序，避免行情变化导致错过时机。"
)


@dataclass(frozen=True, slots=True)
class EntryImmediateExecutionPlan:
    """Decision for an auto-scan entry before actual order submission."""

    should_execute: bool
    reason: str
    is_strong_signal: bool
    capacity_reason: str | None = None


@dataclass(frozen=True, slots=True)
class EntryImmediateExecutionPlanner:
    """Plan immediate execution and staged capacity reservation for entry signals."""

    immediate_reason_provider: ImmediateReasonProvider
    capacity_reason_provider: CapacityReasonProvider
    capacity_reserver: CapacityReserver
    default_execution_reason: str = DEFAULT_AUTO_SCAN_ENTRY_EXECUTION_REASON

    def plan(
        self,
        *,
        model_name: str,
        decision: DecisionOutput,
        open_positions: list[dict[str, Any]],
        staged_entry_counts: dict[str, dict[Any, int]],
    ) -> EntryImmediateExecutionPlan:
        immediate_reason = self.immediate_reason_provider(decision)
        is_strong_signal = bool(immediate_reason)
        execution_reason = immediate_reason or self.default_execution_reason

        capacity_reason = self.capacity_reason_provider(
            model_name,
            decision,
            open_positions,
            staged_entry_counts,
        )
        if capacity_reason:
            prefix = "强信号未即时执行" if is_strong_signal else "开仓信号未即时执行"
            return EntryImmediateExecutionPlan(
                should_execute=False,
                reason=f"{prefix}：{capacity_reason}",
                is_strong_signal=is_strong_signal,
                capacity_reason=capacity_reason,
            )

        self.capacity_reserver(model_name, decision, staged_entry_counts)
        return EntryImmediateExecutionPlan(
            should_execute=True,
            reason=execution_reason,
            is_strong_signal=is_strong_signal,
        )
