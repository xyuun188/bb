from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ai_brain.base_model import Action, DecisionOutput
from executor.base_executor import ExecutionResult, OrderStatus
from services.decision_state import DecisionStage, DecisionStageStatus
from services.market_decision_result_recorder import MarketDecisionResultRecorder
from services.market_queued_entry_processor import (
    QUEUED_ENTRY_PENDING_REASON,
    MarketQueuedEntryProcessor,
)


def _decision(symbol: str = "BTC/USDT") -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol=symbol,
        action=Action.LONG,
        confidence=0.8,
        reasoning="entry",
        raw_response={},
    )


def _result(status: OrderStatus) -> ExecutionResult:
    return ExecutionResult(
        order_id="local-1",
        symbol="BTC/USDT",
        side="long",
        order_type="market",
        quantity=1.0 if status == OrderStatus.FILLED else 0.0,
        price=100.0 if status == OrderStatus.FILLED else 0.0,
        status=status,
        exchange_order_id="exchange-1" if status == OrderStatus.FILLED else "rejected",
    )


def _processor(
    calls: list[tuple[str, Any]],
    *,
    claim_result: bool = True,
    execute_error: BaseException | None = None,
    execution_result: ExecutionResult | None = None,
) -> MarketQueuedEntryProcessor:
    async def claim(symbol: str, scope: str) -> bool:
        calls.append(("claim", symbol, scope))
        return claim_result

    def annotate(decision: DecisionOutput, **kwargs: Any) -> dict[str, Any]:
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        raw["candidate_selection"] = kwargs
        calls.append(("annotate", kwargs))
        return raw

    async def mark_raw(decision_id: int, raw_response: dict[str, Any]) -> None:
        calls.append(("raw", decision_id, dict(raw_response)))

    async def mark_reason(decision_id: int, reason: str) -> None:
        calls.append(("reason", decision_id, reason))

    async def mark_pending(decision_id: int, reason: str) -> None:
        calls.append(("pending", decision_id, reason))

    def set_stage(stage: str) -> None:
        calls.append(("stage", stage))

    def release(
        model_name: str,
        decision: DecisionOutput,
        staged_counts: dict[str, dict[Any, int]],
    ) -> None:
        calls.append(("release", model_name, decision.symbol))
        staged_counts.setdefault("reserved", {})[model_name] = 0

    async def execute_candidate(*args: Any, **kwargs: Any) -> ExecutionResult | None:
        decision = args[2]
        calls.append(("execute", args[0], args[1], decision.action.value, bool(kwargs)))
        if execute_error is not None:
            raise execute_error
        return execution_result

    async def ensure_final(
        decision_id: int,
        symbol: str,
        model_name: str,
        decision: DecisionOutput,
        results: dict[str, Any],
    ) -> None:
        calls.append(("ensure", decision_id, symbol, model_name, decision.action.value))

    return MarketQueuedEntryProcessor(
        normalize_symbol=lambda symbol: str(symbol).upper(),
        analysis_symbol_claimer=claim,
        annotate_candidate_selection=annotate,
        mark_decision_raw_response=mark_raw,
        mark_decision_reason=mark_reason,
        mark_decision_pending_execution=mark_pending,
        result_recorder=MarketDecisionResultRecorder(),
        model_execution_mode_provider=lambda _model: "paper",
        set_loop_stage=set_stage,
        candidate_executor=execute_candidate,
        final_state_ensurer=ensure_final,
        capacity_releaser=release,
        execution_confirmed_checker=lambda result: bool(
            result and result.status == OrderStatus.FILLED and result.exchange_order_id
        ),
    )


@pytest.mark.asyncio
async def test_market_queued_entry_processor_records_claim_skip() -> None:
    calls: list[tuple[str, Any]] = []
    results = {"decisions": []}

    result = await _processor(calls, claim_result=False).process(
        symbol="BTC/USDT",
        model_name="ensemble_trader",
        decision=_decision(),
        assessment=object(),
        decision_db_id=7,
        results=results,
        open_positions=[],
        claimed_symbol_keys=set(),
    )

    assert result.handled is True
    assert result.claimed_symbol is None
    assert result.execution_attempted is False
    assert results["decisions"][0]["execution_status"] == "skipped"
    assert "另一条分析流程" in results["decisions"][0]["reason"]
    assert _decision_state_status(calls, 7) == (
        DecisionStage.STRATEGY_ARBITRATION,
        DecisionStageStatus.SKIPPED,
    )
    assert ("reason", 7, results["decisions"][0]["reason"]) in calls
    assert not any(call[0] == "execute" for call in calls)


@pytest.mark.asyncio
async def test_market_queued_entry_processor_keeps_capacity_on_confirmed_execution() -> None:
    calls: list[tuple[str, Any]] = []
    results = {"decisions": []}
    staged_counts = {"reserved": {"ensemble_trader": 1}}

    result = await _processor(calls, execution_result=_result(OrderStatus.FILLED)).process(
        symbol="BTC/USDT",
        model_name="ensemble_trader",
        decision=_decision(),
        assessment=object(),
        decision_db_id=8,
        results=results,
        open_positions=[],
        claimed_symbol_keys=set(),
        staged_entry_counts=staged_counts,
    )

    assert result.handled is True
    assert result.claimed_symbol == "BTC/USDT"
    assert result.execution_attempted is True
    assert result.execution_confirmed is True
    assert staged_counts["reserved"]["ensemble_trader"] == 1
    assert ("pending", 8, QUEUED_ENTRY_PENDING_REASON) in calls
    assert ("execute", "BTC/USDT", "ensemble_trader", "long", True) in calls
    assert ("release", "ensemble_trader", "BTC/USDT") not in calls


@pytest.mark.asyncio
async def test_market_queued_entry_processor_releases_capacity_on_unconfirmed_execution() -> None:
    calls: list[tuple[str, Any]] = []
    staged_counts = {"reserved": {"ensemble_trader": 1}}

    result = await _processor(calls, execution_result=_result(OrderStatus.REJECTED)).process(
        symbol="BTC/USDT",
        model_name="ensemble_trader",
        decision=_decision(),
        assessment=object(),
        decision_db_id=8,
        results={"decisions": []},
        open_positions=[],
        claimed_symbol_keys=set(),
        staged_entry_counts=staged_counts,
    )

    assert result.execution_attempted is True
    assert result.execution_confirmed is False
    assert staged_counts["reserved"]["ensemble_trader"] == 0
    assert ("release", "ensemble_trader", "BTC/USDT") in calls


@pytest.mark.asyncio
async def test_market_queued_entry_processor_reuses_existing_claim() -> None:
    calls: list[tuple[str, Any]] = []

    result = await _processor(calls, execution_result=_result(OrderStatus.FILLED)).process(
        symbol="BTC/USDT",
        model_name="ensemble_trader",
        decision=_decision(),
        assessment=object(),
        decision_db_id=None,
        results={"decisions": []},
        open_positions=[],
        claimed_symbol_keys={"BTC/USDT"},
        staged_entry_counts={"reserved": {"ensemble_trader": 1}},
    )

    assert result.claimed_symbol is None
    assert result.execution_confirmed is True
    assert not any(call[0] == "claim" for call in calls)
    assert any(call[0] == "execute" for call in calls)


@pytest.mark.asyncio
async def test_market_queued_entry_processor_records_execution_error() -> None:
    calls: list[tuple[str, Any]] = []
    results = {"decisions": []}
    staged_counts = {"reserved": {"ensemble_trader": 1}}

    result = await _processor(calls, execute_error=RuntimeError("boom")).process(
        symbol="BTC/USDT",
        model_name="ensemble_trader",
        decision=_decision(),
        assessment=object(),
        decision_db_id=9,
        results=results,
        open_positions=[],
        claimed_symbol_keys=set(),
        staged_entry_counts=staged_counts,
    )

    assert result.execution_attempted is True
    assert result.execution_confirmed is False
    assert result.execution_error == "boom"
    assert staged_counts["reserved"]["ensemble_trader"] == 0
    assert results["decisions"][0]["execution_status"] == "error"
    assert "候选进入执行流程后异常中断" in results["decisions"][0]["reason"]
    assert _decision_state_status(calls, 9) == (
        DecisionStage.EXCHANGE_SUBMIT,
        DecisionStageStatus.FAILED,
    )
    assert any(call[0] == "reason" and call[1] == 9 for call in calls)


@pytest.mark.asyncio
async def test_market_queued_entry_processor_finalizes_cancelled_execution() -> None:
    calls: list[tuple[str, Any]] = []
    results = {"decisions": []}
    staged_counts = {"reserved": {"ensemble_trader": 1}}

    result = await _processor(calls, execute_error=asyncio.CancelledError()).process(
        symbol="LIT/USDT",
        model_name="ensemble_trader",
        decision=_decision("LIT/USDT"),
        assessment=object(),
        decision_db_id=10,
        results=results,
        open_positions=[],
        claimed_symbol_keys=set(),
        staged_entry_counts=staged_counts,
    )

    assert result.execution_attempted is True
    assert result.execution_confirmed is False
    assert result.execution_error == "cancelled"
    assert staged_counts["reserved"]["ensemble_trader"] == 0
    assert results["decisions"][0]["execution_status"] == "error"
    assert "被外层超时保护取消" in results["decisions"][0]["reason"]
    assert ("release", "ensemble_trader", "LIT/USDT") in calls
    assert any(
        call[0] == "reason" and call[1] == 10 and "被外层超时保护取消" in call[2]
        for call in calls
    )
    assert _decision_state_status(calls, 10) == (
        DecisionStage.EXCHANGE_SUBMIT,
        DecisionStageStatus.FAILED,
    )


def _decision_state_status(
    calls: list[tuple[str, Any]],
    decision_id: int,
) -> tuple[str, str] | None:
    for call in calls:
        if call[0] != "raw" or call[1] != decision_id:
            continue
        raw = call[2]
        if not isinstance(raw, dict):
            continue
        machine = raw.get("decision_state_machine")
        if isinstance(machine, dict):
            return str(machine.get("current_stage") or ""), str(machine.get("current_status") or "")
    return None
