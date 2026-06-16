from __future__ import annotations

from web_dashboard.api.trades import _repair_position_reason_hold_hours


def test_repair_position_reason_hold_hours_replaces_stale_zero_value() -> None:
    reason = "策略纪律触发低质量旧仓释放：hard_loss_pressure；质量分层=watch，质量分=56.0，持仓小时=0.0。"

    repaired = _repair_position_reason_hold_hours(reason, 68.5883 * 60)

    assert "持仓小时=68.5883" in repaired
    assert "持仓小时=0.0" not in repaired


def test_repair_position_reason_hold_hours_keeps_valid_existing_value() -> None:
    reason = "策略纪律触发低质量旧仓释放：loss_watch；持仓小时=70.1184。"

    repaired = _repair_position_reason_hold_hours(reason, 68.0 * 60)

    assert repaired == reason
