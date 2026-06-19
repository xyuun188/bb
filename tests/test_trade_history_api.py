from __future__ import annotations

from web_dashboard.api.trades import (
    _execution_status_label,
    _readable_execution_reason,
    _repair_position_reason_hold_hours,
    _translate_execution_text,
)


def test_repair_position_reason_hold_hours_replaces_stale_zero_value() -> None:
    reason = "策略纪律触发低质量旧仓释放：hard_loss_pressure；质量分层=watch，质量分=56.0，持仓小时=0.0。"

    repaired = _repair_position_reason_hold_hours(reason, 68.5883 * 60)

    assert "持仓小时=68.5883" in repaired
    assert "持仓小时=0.0" not in repaired


def test_repair_position_reason_hold_hours_keeps_valid_existing_value() -> None:
    reason = "策略纪律触发低质量旧仓释放：loss_watch；持仓小时=70.1184。"

    repaired = _repair_position_reason_hold_hours(reason, 68.0 * 60)

    assert repaired == reason


def test_trade_history_translates_okx_50001_as_temporary_exchange_failure() -> None:
    reason = (
        'Max retries exceeded: okx {"code":"50001","data":[],'
        '"msg":"Service temporarily unavailable. Please try again later."}'
    )

    translated = _translate_execution_text(reason)

    assert "交易所服务临时不可用" in translated
    assert _execution_status_label("rejected", translated) == "交易所临时不可用"


def test_trade_detail_does_not_use_numeric_order_id_as_reason() -> None:
    reason = _readable_execution_reason(
        execution_reason="",
        reasoning="策略纪律触发低质量旧仓释放：signal_reversal。",
        exchange_order_id="3670054929945042944",
        status="filled",
    )

    assert "3670054929945042944" not in reason
    assert "策略纪律触发" in reason


def test_trade_detail_numeric_only_reason_falls_back_to_readable_success() -> None:
    reason = _readable_execution_reason(
        execution_reason="3670054929945042944",
        reasoning="",
        exchange_order_id="3670054929945042944",
        status="filled",
    )

    assert "3670054929945042944" not in reason
    assert "订单已成交" in reason
