from __future__ import annotations

from ai_brain.base_model import Action, DecisionOutput
from services.position_review_entry_guard import PositionReviewEntryGuardPolicy


def _decision(action: Action = Action.LONG) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=action,
        confidence=0.7,
        reasoning="test",
        position_size_pct=0.01,
        suggested_leverage=1.0,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        raw_response={"existing": True},
    )


def test_position_review_entry_guard_blocks_entry_when_pause_is_active() -> None:
    result = PositionReviewEntryGuardPolicy().block_reason(
        _decision(),
        "账户亏损暂停新开仓",
    )

    assert result is not None
    assert "持仓复盘只允许平仓、减仓或继续持有" in result.reason
    assert "本次同方向加仓/新增仓位信号已跳过" in result.reason
    assert result.raw_response["existing"] is True
    assert result.raw_response["position_entry_guard"] == {
        "applied": True,
        "reason": "new_entry_paused_during_position_review",
        "pause_reason": "账户亏损暂停新开仓",
        "after_risk_adjustment": False,
    }


def test_position_review_entry_guard_marks_after_risk_adjustment() -> None:
    result = PositionReviewEntryGuardPolicy().block_reason(
        _decision(),
        "账户亏损暂停新开仓",
        after_risk_adjustment=True,
    )

    assert result is not None
    assert "风控调整后的同方向加仓/新增仓位信号已跳过" in result.reason
    assert result.raw_response["position_entry_guard"]["after_risk_adjustment"] is True


def test_position_review_entry_guard_ignores_hold_or_missing_pause() -> None:
    policy = PositionReviewEntryGuardPolicy()

    assert policy.block_reason(_decision(Action.HOLD), "账户亏损暂停新开仓") is None
    assert policy.block_reason(_decision(), None) is None
