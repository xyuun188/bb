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


class _RiskAssessment:
    def __init__(self, calls: list[tuple[str, Any]], approved: bool = True) -> None:
        self.calls = calls
        self.approved = approved

    async def assess(self, **kwargs: Any) -> Any:
        self.calls.append(("assess", kwargs["decision"].action.value))
        return SimpleNamespace(
            approved=self.approved,
            decision=None,
            rejection_reason="risk_rejected" if not self.approved else "",
        )


def _processor(calls: list[tuple[str, Any]]) -> PositionReviewDecisionProcessor:
    async def mark_reason(decision_id: int, reason: str) -> None:
        calls.append(("reason", decision_id, reason))

    async def mark_raw(decision_id: int, raw_response: dict[str, Any]) -> None:
        calls.append(("raw", decision_id, raw_response))

    async def log_risk(decision: DecisionOutput, model_name: str, reason: str) -> None:
        calls.append(("risk", decision.symbol, model_name, reason))

    async def execute_candidate(*args: Any, **kwargs: Any) -> None:
        calls.append(("execute", args[0], args[1], args[2].action.value, bool(kwargs)))

    async def ensure_final(
        decision_id: int,
        symbol: str,
        model_name: str,
        decision: DecisionOutput,
        _results: dict[str, Any] | None,
    ) -> None:
        calls.append(("ensure", decision_id, symbol, model_name, decision.action.value))

    async def account_balance(model_name: str) -> float:
        calls.append(("balance", model_name))
        return 1000.0

    recorder = PositionReviewResultRecorder(
        outcome_policy=PositionReviewOutcomePolicy(),
        decision_reason_marker=mark_reason,
        decision_raw_response_marker=mark_raw,
        risk_result_logger=log_risk,
    )
    return PositionReviewDecisionProcessor(
        entry_guard=PositionReviewEntryGuardPolicy(),
        entry_capacity=EntryCapacityPolicy(lambda symbol: str(symbol)),
        risk_assessment=_RiskAssessment(calls),
        result_recorder=recorder,
        candidate_executor=execute_candidate,
        final_state_ensurer=ensure_final,
        account_balance_provider=account_balance,
    )


def _profitable_retrace_position() -> dict[str, Any]:
    return {
        "symbol": "BTC/USDT",
        "side": "long",
        "quantity": 10.0,
        "entry_price": 100.0,
        "current_price": 101.0,
        "unrealized_pnl": 10.0,
        "peak_unrealized_pnl": 20.0,
        "stop_loss_pct": 0.02,
        "entry_fee_usdt": 0.5,
    }


@pytest.mark.asyncio
async def test_hold_is_recorded_without_risk_assessment() -> None:
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
        risk_alert=None,
        results={"decisions": []},
    )

    assert result.handled is True
    assert not any(call[0] == "assess" for call in calls)


@pytest.mark.asyncio
async def test_entry_pause_blocks_before_risk_assessment() -> None:
    calls: list[tuple[str, Any]] = []

    result = await _processor(calls).process(
        decision=_decision(Action.LONG),
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        model_mode="paper",
        decision_db_id=8,
        open_positions=[],
        feature_vector=SimpleNamespace(),
        position_entry_pause_reason="account_safety_pause",
        risk_alert=None,
        results={"decisions": []},
    )

    assert result.handled is True
    assert not any(call[0] == "assess" for call in calls)


@pytest.mark.asyncio
async def test_exit_without_position_economics_is_skipped_by_dynamic_policy() -> None:
    calls: list[tuple[str, Any]] = []
    results = {"decisions": []}

    result = await _processor(calls).process(
        decision=_decision(Action.CLOSE_LONG),
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        model_mode="paper",
        decision_db_id=9,
        open_positions=[],
        feature_vector=SimpleNamespace(),
        position_entry_pause_reason=None,
        risk_alert=None,
        results=results,
    )

    assert result.handled is True
    assert "position_economics_missing" in results["decisions"][0]["reason"]
    assert not any(call[0] == "execute" for call in calls)


@pytest.mark.asyncio
async def test_profitable_retrace_exit_executes_with_dynamic_fraction() -> None:
    calls: list[tuple[str, Any]] = []
    decision = _decision(Action.CLOSE_LONG)

    result = await _processor(calls).process(
        decision=decision,
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        model_mode="paper",
        decision_db_id=10,
        open_positions=[_profitable_retrace_position()],
        feature_vector=SimpleNamespace(),
        position_entry_pause_reason=None,
        risk_alert=None,
        results={"decisions": []},
    )

    assert result.handled is True
    assert result.executed_immediately is True
    assert decision.position_size_pct == pytest.approx(0.5)
    assert any(call[0] == "execute" for call in calls)
    assert any(call[0] == "ensure" for call in calls)
