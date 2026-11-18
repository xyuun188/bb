from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ai_brain.base_model import Action, DecisionOutput
from models.decision import AIDecision
from models.trade import Order
from services import pending_exit_recovery
from services.decision_persistence_service import DecisionPersistenceService
from services.decision_state import (
    EXIT_EXECUTION_CLAIM,
    DecisionStage,
    DecisionStageStatus,
    append_decision_stage,
)
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        {EXIT_EXECUTION_CLAIM: True},
        {"execution_result": {"exchange_confirmed": True, "exchange_order_id": "okx-123"}},
        append_decision_stage(
            {}, DecisionStage.EXCHANGE_SUBMIT, DecisionStageStatus.PENDING,
        ),
        append_decision_stage(
            {}, DecisionStage.EXCHANGE_CONFIRM, DecisionStageStatus.UNKNOWN,
        ),
    ],
)
async def test_pending_exit_recovery_does_not_replay_exchange_work(raw: dict[str, Any]) -> None:
    calls: list[tuple[str, Any]] = []
    round_decision_ids: set[int] = set()
    result = await _processor(calls, [_row(raw_response=raw), _row(id=12)]).recover(
        results={}, open_positions=[], round_decision_ids=round_decision_ids,
    )

    assert (result.executed, result.skipped) == (1, 1)
    assert round_decision_ids == {12}


@pytest.mark.asyncio
async def test_load_and_claim_exit_recheck_stale_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    created = datetime(2026, 6, 9, 12, 0)

    @asynccontextmanager
    async def session_context():
        async with sessions.begin() as session:
            yield session

    def row(decision_id: int, raw: dict[str, Any]) -> AIDecision:
        return AIDecision(
            id=decision_id, created_at=created, model_name="ensemble_trader",
            symbol="BTC/USDT", action=Action.CLOSE_LONG.value, confidence=0.8,
            raw_llm_response=raw,
        )

    try:
        async with engine.begin() as connection:
            await connection.run_sync(AIDecision.__table__.create)
            await connection.run_sync(Order.__table__.create)
        async with session_context() as session:
            session.add_all([row(i, {EXIT_EXECUTION_CLAIM: True}) for i in range(1, 12)])
            session.add(row(12, {"analysis_type": "position"}))
        monkeypatch.setattr(pending_exit_recovery, "get_session_ctx", session_context)
        cutoff = created.replace(tzinfo=UTC) - timedelta(minutes=1)
        pending = await pending_exit_recovery.load_pending_exit_decision_rows(cutoff)
        assert [item["id"] for item in pending] == [12]
        persistence = DecisionPersistenceService(
            normalize_symbol=lambda symbol: str(symbol),
            session_context_factory=session_context,
        )
        rebuild = PendingExitDecisionRecoveryProcessor._decision_from_row
        assert await persistence.claim_exit_execution(12, rebuild(pending[0], Action.CLOSE_LONG))
        stale = rebuild(pending[0], Action.CLOSE_LONG)
        assert not await persistence.claim_exit_execution(12, stale)
        assert EXIT_EXECUTION_CLAIM not in stale.raw_response
        assert await pending_exit_recovery.load_pending_exit_decision_rows(cutoff) == []
    finally:
        await engine.dispose()
