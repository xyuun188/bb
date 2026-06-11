from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ai_brain.base_model import Action, DecisionOutput
from services.entry_capacity import EntryCapacityPolicy
from services.position_review_decision_processor import PositionReviewDecisionProcessor
from services.position_review_entry_guard import PositionReviewEntryGuardPolicy
from services.position_review_outcome import PositionReviewOutcomePolicy
from services.position_review_result_recorder import PositionReviewResultRecorder


def _decision(action: Action) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=action,
        confidence=0.75,
        reasoning="review",
        raw_response={},
    )


def _result_recorder(calls: list[tuple[str, Any]]) -> PositionReviewResultRecorder:
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


class _RiskAssessment:
    def __init__(
        self,
        calls: list[tuple[str, Any]],
        *,
        approved: bool = True,
        decision: DecisionOutput | None = None,
        reason: str = "",
    ) -> None:
        self.calls = calls
        self.approved = approved
        self.decision = decision
        self.reason = reason

    async def assess(self, **kwargs: Any) -> Any:
        self.calls.append(("assess", kwargs["decision"].action.value))
        return SimpleNamespace(
            approved=self.approved,
            decision=self.decision,
            rejection_reason=self.reason,
        )


def _processor(
    calls: list[tuple[str, Any]],
    *,
    risk_assessment: Any | None = None,
    max_positions: int = 99,
    fee_guard_reason: str | None = None,
) -> PositionReviewDecisionProcessor:
    async def fee_guard(model_name: str, decision: DecisionOutput) -> str | None:
        calls.append(("fee_guard", model_name, decision.action.value))
        return fee_guard_reason

    async def execute_candidate(*args: Any, **kwargs: Any) -> None:
        decision = args[2]
        calls.append(("execute", args[0], args[1], decision.action.value, bool(kwargs)))

    async def ensure_final(
        decision_id: int,
        symbol: str,
        model_name: str,
        decision: DecisionOutput,
        results: dict[str, Any] | None,
    ) -> None:
        calls.append(("ensure", decision_id, symbol, model_name, decision.action.value))

    async def account_balance(model_name: str) -> float:
        calls.append(("balance", model_name))
        return 1000.0

    return PositionReviewDecisionProcessor(
        entry_guard=PositionReviewEntryGuardPolicy(),
        entry_capacity=EntryCapacityPolicy(lambda symbol: str(symbol), lambda: max_positions),
        risk_assessment=risk_assessment or _RiskAssessment(calls),
        result_recorder=_result_recorder(calls),
        exit_fee_guard_reason_provider=fee_guard,
        candidate_executor=execute_candidate,
        final_state_ensurer=ensure_final,
        account_balance_provider=account_balance,
    )


@pytest.mark.asyncio
async def test_position_review_processor_records_hold_without_risk_assessment() -> None:
    calls: list[tuple[str, Any]] = []

    result = await _processor(calls).process(
        decision=_decision(Action.HOLD),
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        model_mode="paper",
        decision_db_id=7,
        open_positions=[],
        feature_vector=SimpleNamespace(),
        position_entry_pause_reason=None,
        risk_alert="alert",
        results={"decisions": []},
    )

    assert result.handled is True
    assert not any(call[0] == "assess" for call in calls)
    assert calls == [
        ("risk", "BTC/USDT", "ensemble_trader", "未提交订单：持仓复盘结论为继续持有或暂不加仓。"),
        ("reason", 7, "持仓复盘结论为继续持有或暂不加仓，未提交订单。"),
    ]


@pytest.mark.asyncio
async def test_position_review_processor_blocks_entry_when_position_entry_paused() -> None:
    calls: list[tuple[str, Any]] = []
    results = {"decisions": []}

    result = await _processor(calls).process(
        decision=_decision(Action.LONG),
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        model_mode="paper",
        decision_db_id=8,
        open_positions=[],
        feature_vector=SimpleNamespace(),
        position_entry_pause_reason="账户风险限制",
        risk_alert=None,
        results=results,
    )

    assert result.handled is True
    assert not any(call[0] == "assess" for call in calls)
    assert calls[0][0] == "raw"
    assert calls[1][0] == "reason"
    assert results["decisions"][0]["execution_status"] == "skipped"
    assert "账户风险限制" in results["decisions"][0]["reason"]


@pytest.mark.asyncio
async def test_position_review_processor_records_capacity_block_before_risk() -> None:
    calls: list[tuple[str, Any]] = []
    results = {"decisions": []}

    result = await _processor(calls, max_positions=1).process(
        decision=_decision(Action.LONG),
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        model_mode="paper",
        decision_db_id=9,
        open_positions=[{"model_name": "ensemble_trader", "symbol": "ETH/USDT"}],
        feature_vector=SimpleNamespace(),
        position_entry_pause_reason=None,
        risk_alert="alert",
        results=results,
    )

    assert result.handled is True
    assert not any(call[0] == "assess" for call in calls)
    assert any(call[0] == "risk" and "未执行：" in call[3] for call in calls)
    assert results["decisions"] == []


@pytest.mark.asyncio
async def test_position_review_processor_records_risk_rejection() -> None:
    calls: list[tuple[str, Any]] = []
    risk = _RiskAssessment(calls, approved=False, reason="风控拒绝")

    result = await _processor(calls, risk_assessment=risk).process(
        decision=_decision(Action.CLOSE_LONG),
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        model_mode="paper",
        decision_db_id=10,
        open_positions=[],
        feature_vector=SimpleNamespace(),
        position_entry_pause_reason=None,
        risk_alert=None,
        results={"decisions": []},
    )

    assert result.handled is True
    assert ("assess", "close_long") in calls
    assert ("reason", 10, "风控拒绝") in calls


@pytest.mark.asyncio
async def test_position_review_processor_records_fee_guard_skip_for_exit() -> None:
    calls: list[tuple[str, Any]] = []
    results = {"decisions": []}

    result = await _processor(calls, fee_guard_reason="手续费磨损").process(
        decision=_decision(Action.CLOSE_LONG),
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        model_mode="paper",
        decision_db_id=11,
        open_positions=[],
        feature_vector=SimpleNamespace(),
        position_entry_pause_reason=None,
        risk_alert="alert",
        results=results,
    )

    assert result.handled is True
    assert ("fee_guard", "ensemble_trader", "close_long") in calls
    assert results["decisions"][0]["reason"] == "手续费磨损"
    assert any(call[0] == "risk" and call[3] == "未执行：手续费磨损" for call in calls)


@pytest.mark.asyncio
async def test_position_review_processor_executes_exit_immediately_when_results_available() -> None:
    calls: list[tuple[str, Any]] = []

    result = await _processor(calls).process(
        decision=_decision(Action.CLOSE_LONG),
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        model_mode="paper",
        decision_db_id=12,
        open_positions=[{"symbol": "BTC/USDT"}],
        feature_vector=SimpleNamespace(),
        position_entry_pause_reason=None,
        risk_alert=None,
        results={"decisions": []},
    )

    assert result.handled is True
    assert result.executed_immediately is True
    assert ("execute", "BTC/USDT", "ensemble_trader", "close_long", True) in calls
    assert ("ensure", 12, "BTC/USDT", "ensemble_trader", "close_long") in calls


@pytest.mark.asyncio
async def test_position_review_processor_returns_exit_candidate_without_results() -> None:
    calls: list[tuple[str, Any]] = []

    result = await _processor(calls).process(
        decision=_decision(Action.CLOSE_LONG),
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        model_mode="paper",
        decision_db_id=13,
        open_positions=[],
        feature_vector=SimpleNamespace(),
        position_entry_pause_reason=None,
        risk_alert=None,
        results=None,
    )

    assert result.handled is False
    assert result.candidate is not None
    assert result.candidate[0] == "BTC/USDT"
    assert result.candidate[2].action == Action.CLOSE_LONG
    assert not any(call[0] == "execute" for call in calls)
