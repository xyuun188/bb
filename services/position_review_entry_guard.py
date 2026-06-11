"""Entry guard for position-review decisions under account-level pauses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import DecisionOutput


@dataclass(frozen=True, slots=True)
class PositionReviewEntryGuardResult:
    """Result of blocking an entry/add signal during position review."""

    reason: str
    raw_response: dict[str, Any]


class PositionReviewEntryGuardPolicy:
    """Block new/add entries during position review when account risk is paused."""

    def block_reason(
        self,
        decision: DecisionOutput,
        pause_reason: str | None,
        *,
        after_risk_adjustment: bool = False,
    ) -> PositionReviewEntryGuardResult | None:
        if not decision.is_entry or not pause_reason:
            return None

        raw_response = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        raw_response["position_entry_guard"] = {
            "applied": True,
            "reason": "new_entry_paused_during_position_review",
            "pause_reason": pause_reason,
            "after_risk_adjustment": bool(after_risk_adjustment),
        }
        reason_detail = (
            "风控调整后的同方向加仓/新增仓位信号已跳过。"
            if after_risk_adjustment
            else "本次同方向加仓/新增仓位信号已跳过。"
        )
        reason = (
            "触发账户风险限制后，持仓复盘只允许平仓、减仓或继续持有，"
            f"{reason_detail}触发原因：{pause_reason}"
        )
        return PositionReviewEntryGuardResult(
            reason=reason,
            raw_response=raw_response,
        )
