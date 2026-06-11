from __future__ import annotations

from typing import Any

import pytest

from ai_brain.base_model import Action, DecisionOutput
from services.market_decision_result_recorder import MarketDecisionResultRecorder
from services.market_direct_entry_processor import MarketDirectEntryProcessor


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
    capacity_reason: str | None = None,
) -> MarketDirectEntryProcessor:
    def capacity(
        model_name: str,
        decision: DecisionOutput,
        open_positions: list[dict[str, Any]],
        staged_counts: dict[str, dict[Any, int]],
    ) -> str | None:
        calls.append(("capacity", model_name, decision.symbol, len(open_positions)))
        return capacity_reason

    def reserve(
        model_name: str,
        decision: DecisionOutput,
        staged_counts: dict[str, dict[Any, int]],
    ) -> None:
        calls.append(("reserve", model_name, decision.symbol))
        staged_counts.setdefault("reserved", {})[model_name] = 1

    def annotate(decision: DecisionOutput, **kwargs: Any) -> dict[str, Any]:
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        raw["candidate_selection"] = kwargs
        calls.append(("annotate", kwargs))
        return raw

    async def mark_raw(decision_id: int, raw_response: dict[str, Any]) -> None:
        calls.append(("raw", decision_id, dict(raw_response)))

    async def mark_reason(decision_id: int, reason: str) -> None:
        calls.append(("reason", decision_id, reason))

    def clear_no_opportunity(symbol: str) -> None:
        calls.append(("clear", symbol))

    async def execute_candidate(*args: Any, **kwargs: Any) -> None:
        decision = args[2]
        calls.append(("execute", args[0], args[1], decision.action.value, bool(kwargs)))

    return MarketDirectEntryProcessor(
        capacity_reason_provider=capacity,
        capacity_reserver=reserve,
        annotate_candidate_selection=annotate,
        mark_decision_raw_response=mark_raw,
        mark_decision_reason=mark_reason,
        result_recorder=MarketDecisionResultRecorder(),
        clear_market_no_opportunity_symbol=clear_no_opportunity,
        candidate_executor=execute_candidate,
    )


@pytest.mark.asyncio
async def test_market_direct_entry_processor_records_capacity_skip() -> None:
    calls: list[tuple[str, Any]] = []
    results = {"decisions": []}
    staged_counts: dict[str, dict[Any, int]] = {}

    result = await _processor(calls, capacity_reason="持仓已满").process(
        symbol="BTC/USDT",
        model_name="ensemble_trader",
        original_decision=_decision(),
        executed=_decision(),
        assessment=object(),
        decision_db_id=7,
        results=results,
        model_mode="paper",
        open_positions=[{"symbol": "ETH/USDT"}],
        staged_entry_counts=staged_counts,
    )

    assert result.handled is True
    assert result.execution_attempted is False
    assert result.reason == "持仓已满"
    assert results["decisions"][0]["execution_status"] == "skipped"
    assert ("reason", 7, "持仓已满") in calls
    assert not any(call[0] == "reserve" for call in calls)
    assert not staged_counts


@pytest.mark.asyncio
async def test_market_direct_entry_processor_reserves_and_executes() -> None:
    calls: list[tuple[str, Any]] = []
    results = {"decisions": []}
    staged_counts: dict[str, dict[Any, int]] = {}
    decision = _decision()

    result = await _processor(calls).process(
        symbol="BTC/USDT",
        model_name="ensemble_trader",
        original_decision=decision,
        executed=decision,
        assessment=object(),
        decision_db_id=8,
        results=results,
        model_mode="paper",
        open_positions=[],
        staged_entry_counts=staged_counts,
    )

    assert result.handled is True
    assert result.execution_attempted is True
    assert staged_counts["reserved"]["ensemble_trader"] == 1
    assert ("clear", "BTC/USDT") in calls
    assert ("execute", "BTC/USDT", "ensemble_trader", "long", True) in calls
    assert results["decisions"] == []
