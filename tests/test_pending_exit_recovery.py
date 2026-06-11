from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from ai_brain.base_model import Action, DecisionOutput
from services.pending_exit_recovery import PendingExitDecisionRecoveryProcessor


def _row(**overrides: Any) -> dict[str, Any]:
    row = {
        "id": 11,
        "model_name": "ensemble_trader",
        "symbol": "BTC/USDT",
        "action": Action.CLOSE_LONG.value,
        "confidence": 0.72,
        "reasoning": "risk exit",
        "position_size_pct": 0.4,
        "suggested_leverage": 1.0,
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.1,
        "raw_response": {"exit_intent": "predictive_reversal"},
        "feature_snapshot": {"close": 100.0},
        "created_at": datetime(2026, 6, 9, 12, 0, tzinfo=UTC),
    }
    row.update(overrides)
    return row


def _processor(
    calls: list[tuple[str, Any]],
    rows: list[dict[str, Any]] | Exception,
) -> PendingExitDecisionRecoveryProcessor:
    async def load_pending(_cutoff: datetime) -> list[dict[str, Any]]:
        calls.append(("load", _cutoff))
        if isinstance(rows, Exception):
            raise rows
        return rows

    def set_stage(stage: str) -> None:
        calls.append(("stage", stage))

    async def execute_candidate(*args: Any, **kwargs: Any) -> None:
        decision = args[2]
        assert isinstance(decision, DecisionOutput)
        calls.append(
            (
                "execute",
                args[0],
                args[1],
                decision.action.value,
                args[4],
                bool(kwargs.get("open_positions")),
            )
        )

    return PendingExitDecisionRecoveryProcessor(
        set_loop_stage=set_stage,
        candidate_executor=execute_candidate,
        pending_loader=load_pending,
        clock=lambda: datetime(2026, 6, 9, 12, 10, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_pending_exit_recovery_returns_when_no_rows() -> None:
    calls: list[tuple[str, Any]] = []
    result = await _processor(calls, []).recover(
        results={"decisions": []},
        open_positions=[],
        round_decision_ids=set(),
    )

    assert result.loaded == 0
    assert result.executed == 0
    assert not any(call[0] == "stage" for call in calls)


@pytest.mark.asyncio
async def test_pending_exit_recovery_rebuilds_and_executes_exit_decision() -> None:
    calls: list[tuple[str, Any]] = []
    round_decision_ids: set[int] = set()

    result = await _processor(calls, [_row()]).recover(
        results={"decisions": []},
        open_positions=[{"symbol": "BTC/USDT"}],
        round_decision_ids=round_decision_ids,
    )

    assert result.loaded == 1
    assert result.executed == 1
    assert round_decision_ids == {11}
    assert ("stage", "recover_pending_exits") in calls
    assert ("execute", "BTC/USDT", "ensemble_trader", "close_long", 11, True) in calls


@pytest.mark.asyncio
async def test_pending_exit_recovery_skips_non_exit_rows() -> None:
    calls: list[tuple[str, Any]] = []
    result = await _processor(calls, [_row(action=Action.HOLD.value)]).recover(
        results={"decisions": []},
        open_positions=[],
        round_decision_ids=set(),
    )

    assert result.loaded == 1
    assert result.executed == 0
    assert result.skipped == 1
    assert not any(call[0] == "execute" for call in calls)


@pytest.mark.asyncio
async def test_pending_exit_recovery_loader_failure_does_not_raise() -> None:
    calls: list[tuple[str, Any]] = []
    result = await _processor(calls, RuntimeError("db offline")).recover(
        results={"decisions": []},
        open_positions=[],
        round_decision_ids=set(),
    )

    assert result.failed is True
    assert result.error == "db offline"
    assert not any(call[0] == "stage" for call in calls)
