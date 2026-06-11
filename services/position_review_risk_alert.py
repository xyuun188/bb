"""Risk-alert policy for position-review decisions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ai_brain.base_model import DecisionOutput
from executor.base_executor import ExecutionResult, OrderStatus

FloatParser = Callable[[Any, float], float]
TextShortener = Callable[[Any, int], str]
ActionLabeler = Callable[[Any], str]
ExecutionReasonProvider = Callable[[ExecutionResult], str]

URGENT_RISK_TERMS = (
    "一票否决",
    "硬性否决",
    "禁止",
    "紧急",
    "立即",
    "极端",
    "异常",
    "黑天鹅",
    "爆仓",
    "止损",
    "平仓",
    "严重",
    "流动性",
    "高波动",
    "风险",
)


class PositionReviewRiskAlertPolicy:
    """Builds and stores position-review risk alerts without execution side effects."""

    def __init__(
        self,
        *,
        float_parser: FloatParser,
        text_shortener: TextShortener,
        action_labeler: ActionLabeler,
    ) -> None:
        self._float_parser = float_parser
        self._text_shortener = text_shortener
        self._action_labeler = action_labeler

    def build_alert(
        self,
        decision: DecisionOutput,
        positions: list[dict[str, Any]],
    ) -> str | None:
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        opinions = raw.get("opinions") or []
        if not isinstance(opinions, list):
            return None

        risk_opinion = next(
            (o for o in opinions if isinstance(o, dict) and o.get("model_name") == "risk_expert"),
            None,
        )
        if not risk_opinion:
            return None

        risk_action = str(risk_opinion.get("action") or "hold")
        risk_conf = self._float_parser(risk_opinion.get("confidence"), 0.0)
        risk_reason = self._text_shortener(risk_opinion.get("reasoning"), 220)
        urgent = (
            risk_action in {"close_long", "close_short"}
            or (risk_conf >= 0.70 and any(term in risk_reason for term in URGENT_RISK_TERMS))
            or (decision.is_exit and risk_conf >= 0.55)
        )
        if not urgent:
            return None

        position_bits = []
        for pos in positions[:3]:
            side = (
                "long"
                if pos.get("side") == "long"
                else "short" if pos.get("side") == "short" else str(pos.get("side") or "unknown")
            )
            position_bits.append(
                f"{side} 入场={pos.get('entry_price', '-')}，"
                f"数量={pos.get('quantity', '-')}，"
                f"浮盈亏={pos.get('unrealized_pnl', 0)}"
            )
        position_text = "；".join(position_bits) or "无仓位明细"
        return (
            f"持仓复盘风险告警：{decision.symbol} 当前仓位 {position_text}。"
            f"风险专家动作={self._action_labeler(risk_action)}，置信度={risk_conf:.0%}。"
            f"原因={risk_reason or '风险专家未给出具体原因'}。"
            f"最终复盘动作={self._action_labeler(decision.action)}。"
        )

    def attach(self, decision: DecisionOutput, message: str) -> None:
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        raw["position_review_risk_alert"] = {
            "message": message,
            "planned_action": decision.action.value,
        }
        decision.raw_response = raw

    @staticmethod
    def alert_context(decision: DecisionOutput) -> dict[str, Any] | None:
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        alert = raw.get("position_review_risk_alert")
        return alert if isinstance(alert, dict) else None

    def execution_result_text(
        self,
        decision: DecisionOutput,
        execution_result: ExecutionResult,
        execution_reason_provider: ExecutionReasonProvider,
    ) -> str:
        if execution_result.status == OrderStatus.FILLED:
            return (
                f"已执行完成：动作={self._action_labeler(decision.action)}，"
                f"数量={execution_result.quantity:g}，价格={execution_result.price:g}，"
                f"订单状态={execution_result.status.value}。"
            )
        return (
            f"执行未完成：动作={self._action_labeler(decision.action)}，"
            f"状态={execution_result.status.value}，"
            f"原因={execution_reason_provider(execution_result)}"
        )

    def risk_event_detail(
        self,
        decision: DecisionOutput,
        alert: dict[str, Any],
        result_text: str | None,
    ) -> str:
        return (
            f"{alert.get('message')}"
            f" 系统动作={self._action_labeler(decision.action)}。"
            f"执行结果={result_text or '无执行结果'}"
        )
