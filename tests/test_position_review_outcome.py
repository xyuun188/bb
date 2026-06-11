from __future__ import annotations

from ai_brain.base_model import Action, DecisionOutput
from services.position_review_outcome import (
    PositionReviewOutcomePolicy,
    position_review_not_executed_reason,
)


def _decision(action: Action = Action.CLOSE_LONG) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=action,
        confidence=0.73,
        reasoning="test",
        position_size_pct=0.0,
        suggested_leverage=1.0,
    )


def test_position_review_outcome_formats_hold_reasons() -> None:
    policy = PositionReviewOutcomePolicy()

    assert policy.hold_reason() == "持仓复盘结论为继续持有或暂不加仓，未提交订单。"
    assert policy.hold_reason(for_alert=True) == "未提交订单：持仓复盘结论为继续持有或暂不加仓。"
    assert (
        policy.hold_reason(after_risk_adjustment=True) == "持仓复盘经风控调整为观望，未提交订单。"
    )


def test_position_review_outcome_builds_skipped_result() -> None:
    result = PositionReviewOutcomePolicy().skipped_result(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        decision=_decision(),
        reason="fee guard",
        is_paper=True,
    )

    assert result == {
        "model": "ensemble_trader",
        "symbol": "BTC/USDT",
        "action": "close_long",
        "approved": True,
        "confidence": 0.73,
        "executed": False,
        "execution_status": "skipped",
        "reason": "fee guard",
        "is_paper": True,
    }


def test_position_review_outcome_builds_fast_scan_result() -> None:
    result = PositionReviewOutcomePolicy().fast_scan_result(
        model_name="ensemble_trader",
        symbol="ETH/USDT",
        reason="fast scan",
        is_paper=False,
    )

    assert result["action"] == "hold"
    assert result["execution_status"] == "fast_position_scan"
    assert result["is_paper"] is False


def test_position_review_not_executed_reason_is_chinese_and_non_empty() -> None:
    assert position_review_not_executed_reason("blocked") == "未执行：blocked"
    assert position_review_not_executed_reason("") == "未执行：没有给出具体原因"
