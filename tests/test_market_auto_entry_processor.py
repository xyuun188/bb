from __future__ import annotations

from typing import Any

import pytest

from ai_brain.base_model import Action, DecisionOutput
from executor.base_executor import ExecutionResult, OrderStatus
from services.entry_immediate_execution import EntryImmediateExecutionPlanner
from services.market_auto_entry_processor import (
    ENTRY_EVIDENCE_SHADOW_ONLY_REASON,
    MarketAutoEntryProcessor,
)
from services.market_decision_result_recorder import MarketDecisionResultRecorder


def _decision() -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.LONG,
        confidence=0.8,
        reasoning="entry",
        raw_response={},
    )


def _filled_result() -> ExecutionResult:
    return ExecutionResult(
        order_id="local-1",
        symbol="BTC/USDT",
        side="long",
        order_type="market",
        quantity=1.0,
        price=100.0,
        status=OrderStatus.FILLED,
        exchange_order_id="exchange-1",
    )


def _rejected_result() -> ExecutionResult:
    return ExecutionResult(
        order_id="local-1",
        symbol="BTC/USDT",
        side="long",
        order_type="market",
        quantity=0.0,
        price=0.0,
        status=OrderStatus.REJECTED,
        exchange_order_id="rejected",
    )


def _processor(
    calls: list[tuple[str, Any]],
    *,
    gate_reason: str | None = None,
    immediate_reason: str | None = "强信号",
    capacity_reason: str | None = None,
    execute_error: Exception | None = None,
    execution_result: ExecutionResult | None = None,
) -> MarketAutoEntryProcessor:
    def score_candidate(decision: DecisionOutput, strategy: dict[str, Any] | None) -> float:
        calls.append(("score", decision.symbol, strategy))
        decision.raw_response["scored"] = True
        return 1.0

    def gate(decision: DecisionOutput) -> str | None:
        calls.append(("gate", decision.symbol))
        return gate_reason

    def immediate(decision: DecisionOutput) -> str | None:
        calls.append(("immediate", decision.symbol))
        return immediate_reason

    def capacity(
        model_name: str,
        decision: DecisionOutput,
        open_positions: list[dict[str, Any]],
        staged_counts: dict[str, dict[Any, int]],
    ) -> str | None:
        calls.append(("capacity", model_name, len(open_positions)))
        return capacity_reason

    def reserve(
        model_name: str,
        decision: DecisionOutput,
        staged_counts: dict[str, dict[Any, int]],
    ) -> None:
        calls.append(("reserve", model_name, decision.symbol))
        staged_counts.setdefault("reserved", {})[model_name] = 1

    def release(
        model_name: str,
        decision: DecisionOutput,
        staged_counts: dict[str, dict[Any, int]],
    ) -> None:
        calls.append(("release", model_name, decision.symbol))
        staged_counts.setdefault("reserved", {})[model_name] = 0

    def annotate(decision: DecisionOutput, **kwargs: Any) -> dict[str, Any]:
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        raw["candidate_selection"] = kwargs
        decision.raw_response = raw
        calls.append(("annotate", kwargs))
        return raw

    async def mark_raw(decision_id: int, raw_response: dict[str, Any]) -> None:
        calls.append(("raw", decision_id, dict(raw_response)))

    async def mark_reason(decision_id: int, reason: str) -> None:
        calls.append(("reason", decision_id, reason))

    async def mark_pending(decision_id: int, reason: str) -> None:
        calls.append(("pending", decision_id, reason))

    def clear_no_opportunity(symbol: str) -> None:
        calls.append(("clear", symbol))

    def set_stage(stage: str) -> None:
        calls.append(("stage", stage))

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

    return MarketAutoEntryProcessor(
        score_candidate=score_candidate,
        gate_reason=gate,
        immediate_execution=EntryImmediateExecutionPlanner(
            immediate_reason_provider=immediate,
            capacity_reason_provider=capacity,
            capacity_reserver=reserve,
        ),
        annotate_candidate_selection=annotate,
        mark_decision_raw_response=mark_raw,
        mark_decision_reason=mark_reason,
        mark_decision_pending_execution=mark_pending,
        result_recorder=MarketDecisionResultRecorder(),
        clear_market_no_opportunity_symbol=clear_no_opportunity,
        set_loop_stage=set_stage,
        candidate_executor=execute_candidate,
        final_state_ensurer=ensure_final,
        capacity_releaser=release,
        execution_confirmed_checker=lambda result: bool(
            result and result.status == OrderStatus.FILLED and result.exchange_order_id
        ),
    )


@pytest.mark.asyncio
async def test_market_auto_entry_processor_keeps_weak_evidence_shadow_only() -> None:
    calls: list[tuple[str, Any]] = []
    results = {"decisions": []}
    decision = _decision()
    decision.position_size_pct = 0.02
    decision.raw_response = {
        "opportunity_score": {
            "score": 1.2,
            "expected_net_return_pct": 0.9,
            "evidence_score": {
                "tier": "weak_conflict_probe",
                "effective_score": 42.0,
            },
        }
    }

    result = await _processor(calls, execution_result=_filled_result()).process(
        symbol="BTC/USDT",
        model_name="ensemble_trader",
        decision=decision,
        assessment=object(),
        decision_db_id=12,
        results=results,
        model_mode="paper",
        open_positions=[],
        staged_entry_counts={},
        strategy_mode_context=None,
    )

    assert result.handled is True
    assert result.execution_attempted is False
    assert result.reason == ENTRY_EVIDENCE_SHADOW_ONLY_REASON
    assert results["decisions"][0]["execution_status"] == "skipped"
    assert decision.raw_response["entry_evidence_shadow_only"]["shadow_only"] is True
    assert ("reason", 12, ENTRY_EVIDENCE_SHADOW_ONLY_REASON) in calls
    assert not any(call[0] in {"immediate", "reserve", "execute"} for call in calls)


@pytest.mark.asyncio
async def test_market_auto_entry_processor_blocks_legacy_tradeable_weak_probe() -> None:
    calls: list[tuple[str, Any]] = []
    results = {"decisions": []}
    decision = _decision()
    decision.position_size_pct = 0.02
    decision.raw_response = {
        "opportunity_score": {
            "score": 1.2,
            "expected_net_return_pct": 0.9,
            "evidence_score": {
                "tier": "weak_conflict_probe",
                "effective_score": 42.0,
                "tradeable_probe": True,
                "shadow_only": False,
            },
        }
    }

    result = await _processor(calls, execution_result=_filled_result()).process(
        symbol="BTC/USDT",
        model_name="ensemble_trader",
        decision=decision,
        assessment=object(),
        decision_db_id=12,
        results=results,
        model_mode="paper",
        open_positions=[],
        staged_entry_counts={},
        strategy_mode_context=None,
    )

    assert result.handled is True
    assert result.execution_attempted is False
    assert result.reason == ENTRY_EVIDENCE_SHADOW_ONLY_REASON
    shadow = decision.raw_response["entry_evidence_shadow_only"]
    assert shadow["applied"] is True
    assert shadow["legacy_tradeable_probe"] is True
    assert not any(call[0] in {"immediate", "reserve", "execute"} for call in calls)


@pytest.mark.asyncio
async def test_market_auto_entry_processor_records_gate_skip() -> None:
    calls: list[tuple[str, Any]] = []
    results = {"decisions": []}

    result = await _processor(calls, gate_reason="分数不足").process(
        symbol="BTC/USDT",
        model_name="ensemble_trader",
        decision=_decision(),
        assessment=object(),
        decision_db_id=7,
        results=results,
        model_mode="paper",
        open_positions=[],
        staged_entry_counts={},
        strategy_mode_context={"mode": "test"},
    )

    assert result.handled is True
    assert result.execution_attempted is False
    assert result.reason == "入场候选暂未满足执行条件：分数不足"
    assert results["decisions"][0]["execution_status"] == "skipped"
    assert ("reason", 7, "入场候选暂未满足执行条件：分数不足") in calls
    assert not any(call[0] == "execute" for call in calls)


@pytest.mark.asyncio
async def test_market_auto_entry_processor_preserves_dynamic_evidence_wait_reason() -> None:
    calls: list[tuple[str, Any]] = []
    results = {"decisions": []}
    wait_reason = "动态证据不足，本轮保持观望或极小探针，未提交 OKX 订单。"

    result = await _processor(calls, gate_reason=wait_reason).process(
        symbol="BTC/USDT",
        model_name="ensemble_trader",
        decision=_decision(),
        assessment=object(),
        decision_db_id=11,
        results=results,
        model_mode="paper",
        open_positions=[],
        staged_entry_counts={},
        strategy_mode_context=None,
    )

    assert result.handled is True
    assert result.execution_attempted is False
    assert result.reason == wait_reason
    assert "入场候选暂未满足执行条件" not in results["decisions"][0]["reason"]
    assert ("reason", 11, wait_reason) in calls


@pytest.mark.asyncio
async def test_market_auto_entry_processor_records_capacity_skip() -> None:
    calls: list[tuple[str, Any]] = []
    results = {"decisions": []}

    result = await _processor(calls, capacity_reason="持仓已满").process(
        symbol="BTC/USDT",
        model_name="ensemble_trader",
        decision=_decision(),
        assessment=object(),
        decision_db_id=8,
        results=results,
        model_mode="paper",
        open_positions=[{"symbol": "ETH/USDT"}],
        staged_entry_counts={},
        strategy_mode_context=None,
    )

    assert result.handled is True
    assert result.reason == "强信号未即时执行：持仓已满"
    assert results["decisions"][0]["reason"] == "强信号未即时执行：持仓已满"
    assert not any(call[0] == "reserve" for call in calls)


@pytest.mark.asyncio
async def test_market_auto_entry_processor_keeps_capacity_on_confirmed_execution() -> None:
    calls: list[tuple[str, Any]] = []
    results = {"decisions": []}
    staged_counts: dict[str, dict[Any, int]] = {}

    result = await _processor(calls, execution_result=_filled_result()).process(
        symbol="BTC/USDT",
        model_name="ensemble_trader",
        decision=_decision(),
        assessment=object(),
        decision_db_id=9,
        results=results,
        model_mode="paper",
        open_positions=[],
        staged_entry_counts=staged_counts,
        strategy_mode_context=None,
    )

    assert result.execution_attempted is True
    assert result.execution_confirmed is True
    assert staged_counts["reserved"]["ensemble_trader"] == 1
    assert ("release", "ensemble_trader", "BTC/USDT") not in calls
    assert ("pending", 9, "强信号") in calls
    assert ("ensure", 9, "BTC/USDT", "ensemble_trader", "long") in calls


@pytest.mark.asyncio
async def test_market_auto_entry_processor_releases_capacity_on_unconfirmed_execution() -> None:
    calls: list[tuple[str, Any]] = []
    results = {"decisions": []}
    staged_counts: dict[str, dict[Any, int]] = {}

    result = await _processor(calls, execution_result=_rejected_result()).process(
        symbol="BTC/USDT",
        model_name="ensemble_trader",
        decision=_decision(),
        assessment=object(),
        decision_db_id=9,
        results=results,
        model_mode="paper",
        open_positions=[],
        staged_entry_counts=staged_counts,
        strategy_mode_context=None,
    )

    assert result.execution_attempted is True
    assert result.execution_confirmed is False
    assert staged_counts["reserved"]["ensemble_trader"] == 0
    assert ("release", "ensemble_trader", "BTC/USDT") in calls


@pytest.mark.asyncio
async def test_market_auto_entry_processor_records_execution_error() -> None:
    calls: list[tuple[str, Any]] = []
    results = {"decisions": []}
    staged_counts: dict[str, dict[Any, int]] = {}

    result = await _processor(calls, execute_error=RuntimeError("boom")).process(
        symbol="BTC/USDT",
        model_name="ensemble_trader",
        decision=_decision(),
        assessment=object(),
        decision_db_id=10,
        results=results,
        model_mode="paper",
        open_positions=[],
        staged_entry_counts=staged_counts,
        strategy_mode_context=None,
    )

    assert result.handled is True
    assert result.execution_attempted is True
    assert result.execution_confirmed is False
    assert result.execution_error == "boom"
    assert results["decisions"][0]["execution_status"] == "error"
    assert "强信号已进入即时执行" in results["decisions"][0]["reason"]
    assert ("release", "ensemble_trader", "BTC/USDT") in calls
    assert any(call[0] == "reason" and call[1] == 10 for call in calls)
