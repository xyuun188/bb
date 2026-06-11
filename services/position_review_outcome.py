"""Outcome helpers for position-review decisions.

TradingService owns orchestration, but the wording and result payloads for
position-review skips should stay consistent across hold, guard, and fast-scan
branches. Keeping them here also prevents mojibake or ad-hoc English reasons
from leaking into persisted decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import DecisionOutput

POSITION_REVIEW_HOLD_DECISION_REASON = "持仓复盘结论为继续持有或暂不加仓，未提交订单。"
POSITION_REVIEW_HOLD_ALERT_REASON = "未提交订单：持仓复盘结论为继续持有或暂不加仓。"
POSITION_REVIEW_RISK_HOLD_DECISION_REASON = "持仓复盘经风控调整为观望，未提交订单。"
POSITION_REVIEW_RISK_HOLD_ALERT_REASON = "未提交订单：持仓复盘经风控调整为观望。"


def position_review_not_executed_reason(reason: str | None) -> str:
    """Format a consistent not-executed reason for risk-alert records."""

    detail = str(reason or "").strip() or "没有给出具体原因"
    return f"未执行：{detail}"


def _action_value(decision: DecisionOutput) -> str:
    return str(getattr(decision.action, "value", decision.action))


@dataclass(frozen=True, slots=True)
class PositionReviewOutcomePolicy:
    """Build public result payloads and reasons for position-review outcomes."""

    def hold_reason(
        self,
        *,
        after_risk_adjustment: bool = False,
        for_alert: bool = False,
    ) -> str:
        if after_risk_adjustment:
            return (
                POSITION_REVIEW_RISK_HOLD_ALERT_REASON
                if for_alert
                else POSITION_REVIEW_RISK_HOLD_DECISION_REASON
            )
        return (
            POSITION_REVIEW_HOLD_ALERT_REASON if for_alert else POSITION_REVIEW_HOLD_DECISION_REASON
        )

    def skipped_result(
        self,
        *,
        model_name: str,
        symbol: str,
        decision: DecisionOutput,
        reason: str,
        is_paper: bool,
        execution_status: str = "skipped",
    ) -> dict[str, Any]:
        return {
            "model": model_name,
            "symbol": symbol,
            "action": _action_value(decision),
            "approved": True,
            "confidence": decision.confidence,
            "executed": False,
            "execution_status": execution_status,
            "reason": reason,
            "is_paper": is_paper,
        }

    def fast_scan_result(
        self,
        *,
        model_name: str,
        symbol: str,
        reason: str,
        is_paper: bool,
    ) -> dict[str, Any]:
        return {
            "model": model_name,
            "symbol": symbol,
            "action": "hold",
            "approved": True,
            "confidence": 0.0,
            "executed": False,
            "execution_status": "fast_position_scan",
            "reason": reason,
            "is_paper": is_paper,
        }
