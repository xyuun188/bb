from __future__ import annotations

from typing import Any

import pytest

from ai_brain.base_model import Action, DecisionOutput
from services.position_review_entry_guard import PositionReviewEntryGuardResult
from services.position_review_outcome import PositionReviewOutcomePolicy
from services.position_review_result_recorder import PositionReviewResultRecorder


def _decision(action: Action = Action.HOLD) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=action,
        confidence=0.7,
        reasoning="review",
        raw_response={},
    )


def _recorder(calls: list[tuple[str, Any]]) -> PositionReviewResultRecorder:
    async def mark_reason(decision_id: int, reason: str) -> None:
        calls.append(("reason", decision_id, reason))

    async def mark_raw(decision_id: int, raw_response: dict[str, Any]) -> None:
        calls.append(("raw", decision_id, raw_response))

    async def log_risk(decision: DecisionOutput, model_name: str, reason: str) -> None:
        calls.append(("risk", decision.symbol, model_name, reason))

    return PositionReviewResultRecorder(
        outcome_policy=PositionReviewOutcomePolicy(),
        decision_reason_marker=mark_reason,
        decision_raw_response_marker=mark_raw,
        risk_result_logger=log_risk,
    )


@pytest.mark.asyncio
async def test_position_review_result_recorder_records_hold_reason_and_alert() -> None:
    calls: list[tuple[str, Any]] = []
    decision = _decision()

    await _recorder(calls).record_hold(
        decision=decision,
        model_name="ensemble_trader",
        decision_db_id=12,
        risk_alert="alert",
    )

    assert calls == [
        ("risk", "BTC/USDT", "ensemble_trader", "未提交订单：持仓复盘结论为继续持有或暂不加仓。"),
        ("reason", 12, "持仓复盘结论为继续持有或暂不加仓，未提交订单。"),
    ]


@pytest.mark.asyncio
async def test_position_review_result_recorder_records_entry_guard_and_skipped_result() -> None:
    calls: list[tuple[str, Any]] = []
    decision = _decision(Action.LONG)
    results = {"decisions": []}
    guard = PositionReviewEntryGuardResult(
        reason="entry paused",
        raw_response={"guarded": True},
    )

    await _recorder(calls).record_entry_guard(
        decision=decision,
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        model_mode="paper",
        decision_db_id=7,
        results=results,
        guard=guard,
    )

    assert decision.raw_response == {"guarded": True}
    assert calls == [
        ("raw", 7, {"guarded": True}),
        ("reason", 7, "entry paused"),
    ]
    assert results["decisions"][0]["execution_status"] == "skipped"
    assert results["decisions"][0]["reason"] == "entry paused"


@pytest.mark.asyncio
async def test_position_review_result_recorder_records_skip_optionally_appending() -> None:
    calls: list[tuple[str, Any]] = []
    results = {"decisions": []}
    decision = _decision(Action.CLOSE_LONG)

    await _recorder(calls).record_skip(
        decision=decision,
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        model_mode="paper",
        reason="fee guard",
        decision_db_id=8,
        results=results,
        risk_alert="alert",
        append_result=True,
    )

    assert calls == [
        ("reason", 8, "fee guard"),
        ("risk", "BTC/USDT", "ensemble_trader", "未执行：fee guard"),
    ]
    assert results["decisions"][0]["reason"] == "fee guard"


def test_position_review_result_recorder_appends_fast_scan_result() -> None:
    calls: list[tuple[str, Any]] = []
    results = {"decisions": []}

    _recorder(calls).append_fast_scan_result(
        results=results,
        model_name="ensemble_trader",
        symbol="ETH/USDT",
        reason="快速扫描继续观察",
        model_mode="paper",
    )

    assert calls == []
    assert results["decisions"] == [
        {
            "model": "ensemble_trader",
            "symbol": "ETH/USDT",
            "action": "hold",
            "approved": True,
            "confidence": 0.0,
            "executed": False,
            "execution_status": "fast_position_scan",
            "reason": "快速扫描继续观察",
            "is_paper": True,
        }
    ]
