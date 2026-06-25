"""Execution-result classification and user-facing reason text.

The execution state machine needs consistent answers for three questions:

- did OKX confirm a real order?
- is an exit order still making progress?
- what should the dashboard show when an order did not count as executed?

Keeping those rules here prevents TradingService from owning exchange-error
wording and makes order-state behavior testable without building the full
orchestrator.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from executor.base_executor import ExecutionResult, OrderStatus
from services.okx_error_classifier import (
    is_okx_temporary_service_error,
    okx_temporary_service_error_message,
)
from web_dashboard.api.text_sanitize import sanitize_text


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def is_confirmed_native_full_close_result(result: ExecutionResult | None) -> bool:
    """OKX close-position may not return a normal order id; flat snapshot is confirmation."""

    if result is None or result.status != OrderStatus.FILLED:
        return False
    if result.quantity <= 0 or result.price <= 0:
        return False
    raw = result.raw_response if isinstance(result.raw_response, dict) else {}
    if not raw.get("okx_native_close_position"):
        return False

    before = _safe_float(raw.get("position_contracts_before"), 0.0)
    after = _safe_float(raw.get("position_contracts_after"), before)
    remaining = _safe_float(raw.get("remaining_contracts"), after)
    filled = _safe_float(raw.get("filled_contracts"), result.quantity)
    if filled <= 0:
        return False
    tolerance = max(before * 0.001, 1e-8) if before > 0 else 1e-8
    return min(after, remaining) <= tolerance


class ExecutionResultClassifier:
    """Classify exchange execution results and normalize failure reasons."""

    def __init__(
        self,
        *,
        untradable_exchange_error_checker: Callable[[str], bool] | None = None,
    ) -> None:
        self._untradable_exchange_error_checker = untradable_exchange_error_checker

    def reason_from_result(self, result: ExecutionResult | None) -> str:
        if result is None:
            return "交易接口未返回执行结果。"

        raw = result.raw_response or {}
        if isinstance(raw, dict) and raw.get("entry_tracking"):
            reason = self._entry_tracking_reason(result, raw)
            if reason:
                return reason

        if isinstance(raw, dict) and raw.get("exit_tracking"):
            reason = self._exit_tracking_reason(result, raw)
            if reason:
                return reason

        error = raw.get("error") if isinstance(raw, dict) else None
        raw_error = raw.get("raw_error") if isinstance(raw, dict) else None
        error_text = f"{error or ''} {raw_error or ''}"
        translated_error = self.translate_execution_error_text(error_text)
        if translated_error:
            return translated_error
        if self._is_untradable_exchange_error(error_text):
            return (
                "OKX 提示该交易对当前不可交易，可能受账户地区/合规限制影响；"
                "系统已暂时跳过该交易对，避免重复分析和下单。"
            )
        if self.is_no_exchange_position_error(error_text):
            return (
                "OKX 提示当前没有对应方向的可平仓位，可能已被 OKX 止盈/止损、"
                "手动平仓或刚刚同步延迟；本轮未重复提交。"
            )
        if error:
            return str(sanitize_text(error) or error)
        if result.status == OrderStatus.FILLED:
            return "订单已成交。"

        status_map = {
            OrderStatus.PENDING: "待成交",
            OrderStatus.OPEN: "挂单中",
            OrderStatus.PARTIAL: "部分成交",
            OrderStatus.CANCELLED: "已取消",
            OrderStatus.REJECTED: "已拒绝",
        }
        status = status_map.get(result.status, result.status.value)
        return f"订单状态为{status}，未计为已执行。"

    def translate_execution_error_text(self, text: str | None) -> str | None:
        message = str(text or "").strip()
        if not message:
            return None
        okx_detail = " ".join(self._extract_okx_error_fragments(message))
        normalized = f"{message} {okx_detail}".strip()
        normalized_lower = normalized.lower()
        if is_okx_temporary_service_error(normalized):
            return okx_temporary_service_error_message(normalized)
        if "51008" in normalized or "Insufficient USDT margin" in normalized:
            return (
                "OKX 返回错误码 51008：账户可用 USDT 保证金不足，订单没有提交成功。"
                "通常是当前持仓/挂单占用保证金过高、可用余额不足，或本轮计划仓位过大；"
                "系统应优先处理已有持仓的平仓/减仓，不再继续加仓。"
            )
        if "59670" in normalized or "more than 5 open orders" in normalized:
            return (
                "OKX 拒绝调整杠杆：该交易对当前挂单超过 5 条。"
                "系统会跳过重复杠杆设置，必要时只清理旧的非保护挂单后重试。"
            )
        if "51028" in normalized or "contract under delivery" in normalized_lower:
            return (
                "OKX 51028: Contract under delivery。该合约正在交割/结算，"
                "OKX 暂时拒绝开仓、减仓和平仓操作。系统不会把本地仓位标记为已平仓，"
                "并会暂停重复提交平仓，直到 OKX 结算完成或仓位同步确认最终状态。"
            )
        if "tptriggerpx" in normalized_lower and "error" in normalized_lower:
            return (
                "OKX 返回错误码 51000：保护止盈触发价 tpTriggerPx 无效，订单没有提交成功。"
                "通常是止盈价方向、精度或与当前价距离不符合 OKX 规则；"
                "系统需要重新计算保护止盈参数后再提交。"
            )
        if (
            "open interest" in normalized_lower
            and "platform" in normalized_lower
            and "limit" in normalized_lower
        ) or "has reached the platform's limit" in normalized:
            return (
                "OKX 拒绝开仓：该合约当前平台总持仓量已经达到 OKX 上限，"
                "交易所暂时不允许继续增加这个合约的新仓。"
                "这不是 AI 方向或下单数量计算错误；系统会临时跳过该币种，稍后等 OKX 限制解除再重新分析。"
            )
        return None

    @staticmethod
    def _extract_okx_error_fragments(message: str) -> list[str]:
        start = message.find("{")
        end = message.rfind("}")
        if start < 0 or end <= start:
            return []
        try:
            payload = json.loads(message[start : end + 1])
        except json.JSONDecodeError:
            return []
        rows = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(rows, dict):
            rows = [rows]
        if not isinstance(rows, list):
            rows = [payload] if isinstance(payload, dict) else []
        fragments: list[str] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            code = row.get("sCode") or row.get("code")
            message_text = row.get("sMsg") or row.get("msg") or row.get("message")
            if code:
                fragments.append(str(code))
            if message_text:
                fragments.append(str(message_text))
        return fragments

    @staticmethod
    def is_no_exchange_position_error(message: Any) -> bool:
        text = str(message or "").lower()
        return (
            "51169" in text
            or "don't have any positions in this direction" in text
            or "no matching position to close" in text
            or "没有对应方向" in text
            or "没有可平" in text
            or "可平仓位" in text
        )

    def result_has_no_exchange_position(self, result: ExecutionResult | None) -> bool:
        if result is None:
            return False
        raw = result.raw_response or {}
        pieces = [result.order_id, result.exchange_order_id]
        if isinstance(raw, dict):
            pieces.extend([raw.get("error"), raw.get("raw_error")])
        return self.is_no_exchange_position_error(" ".join(str(piece or "") for piece in pieces))

    @staticmethod
    def is_exit_tracking_execution(result: ExecutionResult | None) -> bool:
        raw = result.raw_response if result else None
        return bool(isinstance(raw, dict) and raw.get("exit_tracking"))

    def is_exit_progress_execution(self, result: ExecutionResult | None) -> bool:
        if not self.is_exit_tracking_execution(result):
            return False
        if result is None or result.status != OrderStatus.PARTIAL:
            return False
        order_id = str(result.exchange_order_id or "").strip()
        return bool(order_id and result.quantity > 0)

    @staticmethod
    def is_exchange_confirmed_execution(result: ExecutionResult | None) -> bool:
        """Only treat an execution as real after OKX returns a concrete order id."""

        if result is None or result.status != OrderStatus.FILLED:
            return False
        if is_confirmed_native_full_close_result(result):
            return True
        order_id = str(result.exchange_order_id or "").strip()
        if not order_id or order_id in {"hold", "rejected", "no_position"}:
            return False
        return bool(result.quantity > 0 and result.price > 0)

    def _entry_tracking_reason(
        self,
        result: ExecutionResult,
        raw: dict[str, Any],
    ) -> str | None:
        message = str(sanitize_text(raw.get("message")) or "").strip()
        remaining = _safe_float(raw.get("remaining_contracts"), 0.0)
        filled = _safe_float(raw.get("filled_contracts"), 0.0)
        if result.status == OrderStatus.PARTIAL:
            return message or (
                f"OKX 开仓委托已部分成交，已成交约 {filled:g} 张，"
                f"剩余约 {remaining:g} 张仍在追单；本地等待 OKX 仓位同步确认。"
            )
        if result.status in {OrderStatus.OPEN, OrderStatus.PENDING}:
            return message or (
                "OKX 开仓委托正在挂单或追单，尚未确认成交；"
                "系统不会先创建本地持仓，也不会重复提交同方向开仓单。"
            )
        return message or None

    def _exit_tracking_reason(
        self,
        result: ExecutionResult,
        raw: dict[str, Any],
    ) -> str | None:
        message = str(sanitize_text(raw.get("message")) or "").strip()
        remaining = _safe_float(raw.get("remaining_contracts"), 0.0)
        filled = _safe_float(raw.get("filled_contracts"), 0.0)
        if result.status == OrderStatus.PARTIAL:
            if remaining > 0:
                return (
                    message
                    or f"OKX 平仓已部分成交，仍剩约 {remaining:g} 张合约在处理；系统会继续同步，不会重复提交平仓单。"
                )
            return message or "OKX 平仓已部分成交，系统会继续同步最终成交结果。"
        if result.status in {OrderStatus.OPEN, OrderStatus.PENDING}:
            if filled > 0 or remaining > 0:
                return (
                    message
                    or f"OKX 平仓订单正在追单中，已成交约 {filled:g} 张，剩余约 {remaining:g} 张；系统不会重复提交。"
                )
            return message or "OKX 平仓订单正在追单或等待成交，系统不会重复提交平仓单。"
        return message or None

    def _is_untradable_exchange_error(self, message: str) -> bool:
        if self._untradable_exchange_error_checker is None:
            return False
        return bool(self._untradable_exchange_error_checker(message))
