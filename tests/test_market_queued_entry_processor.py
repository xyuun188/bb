from __future__ import annotations

from typing import Any

import pytest

from ai_brain.base_model import Action, DecisionOutput
from services.market_decision_result_recorder import MarketDecisionResultRecorder
from services.market_queued_entry_processor import (
    QUEUED_ENTRY_PENDING_REASON,
    MarketQueuedEntryProcessor,
)


def _decision() -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.LONG,
        confidence=0.8,
        reasoning="entry",
        raw_response={},
    )


def _processor(
    calls: list[tuple[str, Any]],
    *,
    claim_result: bool = True,
    execute_error: Exception | None = None,
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

    async def execute_candidate(*args: Any, **kwargs: Any) -> None:
        decision = args[2]
        calls.append(("execute", args[0], args[1], decision.action.value, bool(kwargs)))
        if execute_error is not None:
            raise execute_error

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
    assert ("reason", 7, results["decisions"][0]["reason"]) in calls
    assert not any(call[0] == "execute" for call in calls)


@pytest.mark.asyncio
async def test_market_queued_entry_processor_claims_and_executes() -> None:
    calls: list[tuple[str, Any]] = []
    results = {"decisions": []}

    result = await _processor(calls).process(
        symbol="BTC/USDT",
        model_name="ensemble_trader",
        decision=_decision(),
        assessment=object(),
        decision_db_id=8,
        results=results,
        open_positions=[],
        claimed_symbol_keys=set(),
    )

    assert result.handled is True
    assert result.claimed_symbol == "BTC/USDT"
    assert result.execution_attempted is True
    assert ("pending", 8, QUEUED_ENTRY_PENDING_REASON) in calls
    assert ("execute", "BTC/USDT", "ensemble_trader", "long", True) in calls
    assert ("ensure", 8, "BTC/USDT", "ensemble_trader", "long") in calls
    assert results["decisions"] == []


@pytest.mark.asyncio
async def test_market_queued_entry_processor_reuses_existing_claim() -> None:
    calls: list[tuple[str, Any]] = []

    result = await _processor(calls).process(
        symbol="BTC/USDT",
        model_name="ensemble_trader",
        decision=_decision(),
        assessment=object(),
        decision_db_id=None,
        results={"decisions": []},
        open_positions=[],
        claimed_symbol_keys={"BTC/USDT"},
    )

    assert result.claimed_symbol is None
    assert not any(call[0] == "claim" for call in calls)
    assert any(call[0] == "execute" for call in calls)


@pytest.mark.asyncio
async def test_market_queued_entry_processor_records_execution_error() -> None:
    calls: list[tuple[str, Any]] = []
    results = {"decisions": []}

    result = await _processor(calls, execute_error=RuntimeError("boom")).process(
        symbol="BTC/USDT",
        model_name="ensemble_trader",
        decision=_decision(),
        assessment=object(),
        decision_db_id=9,
        results=results,
        open_positions=[],
        claimed_symbol_keys=set(),
    )

    assert result.execution_attempted is True
    assert result.execution_error == "boom"
    assert results["decisions"][0]["execution_status"] == "error"
    assert "候选进入执行流程后异常中断" in results["decisions"][0]["reason"]
    assert any(call[0] == "reason" and call[1] == 9 for call in calls)
