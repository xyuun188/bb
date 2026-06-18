from __future__ import annotations

from web_dashboard.api.trades import (
    _execution_status_label,
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
