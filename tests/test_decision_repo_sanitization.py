from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import db.repositories.decision_repo as decision_repo_module
from db.repositories.decision_repo import DecisionRepository
from services.decision_state import (
    EXIT_EXECUTION_CLAIM,
    DecisionStage,
    DecisionStageStatus,
    append_decision_stage,
)

BAD_REASON = "AI 选择观望，未提交订单。?"
GOOD_REASON = "AI 选择观望，未提交订单。"


@pytest.mark.asyncio
async def test_persisted_diagnostics_are_bounded_without_changing_execution_contracts() -> None:
    from copy import deepcopy

    normal_contract = {"paper_training_only": True, "objective_net_return_pct": -0.12,
                       "evidence": {"nested": {"exact": [1, 2, 3]}}}
    execution = {"suggested_leverage": 3, "position_size_pct": 2, "stop_loss_pct": 0.01}
    raw = {
        "normal_paper_trade": normal_contract, "execution_parameters": execution,
        "decision_state_machine": {"summary": {"final_stage": "exchange_confirm"}},
        "profit_risk_sizing": {"risk_budget_usdt": 50, "final_notional_usdt": 1000},
        "local_ai_tools": {"available": True, "prediction": {"history": [1] * 100_000}},
        "opportunity_score": {"score": 0.72, "diagnostics": {"transcript": "x" * 200_000}},
    }
    original = deepcopy(raw)
    session = FakeSession()
    repo = DecisionRepository(session)
    row = await repo.log_decision({"raw_llm_response": raw})
    assert row.raw_llm_response["normal_paper_trade"] == normal_contract
    assert row.raw_llm_response["execution_parameters"] == execution
    assert row.raw_llm_response["profit_risk_sizing"] == raw["profit_risk_sizing"]
    assert row.raw_llm_response["decision_state_machine"] == raw["decision_state_machine"]
    assert row.raw_llm_response["local_ai_tools"]["available"] is True
    assert "prediction" not in row.raw_llm_response["local_ai_tools"]
    assert row.raw_llm_response["opportunity_score"]["score"] == 0.72
    session.row = row
    await repo.update_raw_response(1, raw)
    assert row.suggested_leverage == 3
    assert row.position_size_pct == 2
    assert row.stop_loss_pct == 0.01
    assert raw == original


class FakeSession:
    def __init__(self, row: Any | None = None, rows: list[Any] | None = None) -> None:
        self.row = row
        self.rows = rows or []
        self.added: Any | None = None
        self.flush_count = 0

    def add(self, instance: Any) -> None:
        self.added = instance

    async def flush(self) -> None:
        self.flush_count += 1

    async def get(self, _model: Any, _row_id: int, **_kwargs: Any) -> Any:
        return self.row

    async def execute(self, _stmt: Any) -> Any:
        return FakeRowsResult(self.rows)


class FakeRowsResult:
    def __init__(self, rows: list[Any]) -> None:
        self.rows = rows

    def scalars(self) -> FakeRowsResult:
        return self

    def all(self) -> list[Any]:
        return self.rows


@pytest.mark.asyncio
async def test_stale_exit_payloads_cannot_replace_claimed_execution_state() -> None:
    original = {EXIT_EXECUTION_CLAIM: True, "dynamic_exit_policy": {"eligible": True}}
    row = SimpleNamespace(id=7, was_executed=False, execution_reason="", raw_llm_response=original)
    session = FakeSession(row=row, rows=[row])
    repo = DecisionRepository(session)
    stale = {"dynamic_exit_policy": {"eligible": False}}

    await repo.update_raw_response(7, stale)
    assert await repo.finalize_unresolved_decisions([(7, "stale fallback", stale)]) == 0
    assert row.raw_llm_response == original
    assert row.execution_reason == ""
    assert session.flush_count == 0

    owner_update = {**original, "execution_result": {"exchange_confirmed": True}}
    await repo.update_raw_response(7, owner_update)
    assert row.raw_llm_response == owner_update
    assert session.flush_count == 1


@pytest.mark.asyncio
async def test_stale_review_cannot_update_confirmed_exit_receipt_or_reason() -> None:
    original = {
        "dynamic_exit_policy": {"eligible": True, "close_fraction": 0.2},
        "execution_result": {
            "exchange_order_id": "close-order",
            "exchange_confirmed": True,
            "status": "partial",
        },
    }
    row = SimpleNamespace(
        action="close_long", was_executed=True, execution_reason=None,
        raw_llm_response=original,
    )
    session = FakeSession(row=row)
    repo = DecisionRepository(session)

    await repo.mark_execution_reason(7, "dynamic_exit_risk_target_already_realized")
    await repo.update_raw_response(7, {
        **original, "dynamic_exit_policy": {"eligible": False, "close_fraction": 0.0},
    })
    await repo.update_raw_response(7, {"dynamic_exit_policy": {"eligible": False}})
    assert row.raw_llm_response == original
    assert row.execution_reason is None
    assert session.flush_count == 0


@pytest.mark.asyncio
async def test_decision_repo_uses_unified_runtime_text_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []

    def fake_sanitize(value: Any) -> Any:
        calls.append(value)
        if isinstance(value, str):
            return f"unified:{value}"
        if isinstance(value, dict):
            return {"unified": value}
        return value

    monkeypatch.setattr(
        decision_repo_module,
        "sanitize_runtime_text",
        fake_sanitize,
        raising=False,
    )
    session = FakeSession()
    repo = DecisionRepository(session)  # type: ignore[arg-type]

    decision = await repo.log_decision(
        {
            "model_name": "ensemble_trader",
            "symbol": "BTC/USDT",
            "action": "hold",
            "confidence": 0.1,
            "reasoning": "raw reason",
            "execution_reason": "raw execution",
            "position_size_pct": 0.0,
            "suggested_leverage": 1.0,
            "stop_loss_pct": 0.0,
            "take_profit_pct": 0.0,
            "feature_snapshot": {"note": "raw feature"},
            "raw_llm_response": {"note": "raw llm"},
            "analysis_type": "market",
            "is_paper": True,
        }
    )

    assert decision.reasoning == "unified:raw reason"
    assert decision.execution_reason == "unified:raw execution"
    assert decision.feature_snapshot == {"unified": {"note": "raw feature"}}
    assert decision.raw_llm_response == {"unified": {"note": "raw llm"}}
    assert "raw reason" in calls


@pytest.mark.asyncio
async def test_log_decision_sanitizes_text_and_json_payloads() -> None:
    session = FakeSession()
    repo = DecisionRepository(session)  # type: ignore[arg-type]

    decision = await repo.log_decision(
        {
            "model_name": "ensemble_trader",
            "symbol": "BTC/USDT",
            "action": "hold",
            "confidence": 0.1,
            "reasoning": BAD_REASON,
            "execution_reason": BAD_REASON,
            "position_size_pct": 0.0,
            "suggested_leverage": 1.0,
            "stop_loss_pct": 0.0,
            "take_profit_pct": 0.0,
            "feature_snapshot": {"reason": BAD_REASON},
            "raw_llm_response": {"notes": [BAD_REASON]},
            "analysis_type": "market",
            "is_paper": True,
        }
    )

    assert session.added is decision
    assert decision.reasoning == GOOD_REASON
    assert decision.execution_reason == GOOD_REASON
    assert decision.feature_snapshot == {"reason": GOOD_REASON}
    assert decision.raw_llm_response == {"notes": [GOOD_REASON]}
    assert session.flush_count == 1


@pytest.mark.asyncio
async def test_decision_repo_update_methods_sanitize_text_and_json() -> None:
    row = SimpleNamespace(
        execution_reason=None,
        raw_llm_response=None,
        position_size_pct=0.25,
        suggested_leverage=5.0,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
    )
    session = FakeSession(row=row)
    repo = DecisionRepository(session)  # type: ignore[arg-type]

    await repo.mark_execution_reason(123, BAD_REASON)
    await repo.update_raw_response(
        123,
        {
            "reason": BAD_REASON,
            "execution_parameters": {
                "position_size_pct": 0.004,
                "suggested_leverage": 2.0,
                "stop_loss_pct": 0.012,
                "take_profit_pct": 0.044,
            },
        },
    )

    assert row.execution_reason == GOOD_REASON
    assert row.raw_llm_response == {
        "reason": GOOD_REASON,
        "execution_parameters": {
            "position_size_pct": 0.004,
            "suggested_leverage": 2.0,
            "stop_loss_pct": 0.012,
            "take_profit_pct": 0.044,
        },
    }
    assert row.position_size_pct == 0.004
    assert row.suggested_leverage == 2.0
    assert row.stop_loss_pct == 0.012
    assert row.take_profit_pct == 0.044
    assert session.flush_count == 2


@pytest.mark.asyncio
async def test_finalize_unresolved_decisions_writes_terminal_state_only_when_needed() -> None:
    unresolved = SimpleNamespace(
        id=1,
        was_executed=False,
        execution_reason="",
        raw_llm_response={"decision_state_machine": {"stages": []}},
        position_size_pct=0.25,
        suggested_leverage=5.0,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
    )
    executed = SimpleNamespace(
        id=2,
        was_executed=True,
        execution_reason="filled",
        raw_llm_response={},
        position_size_pct=0.25,
        suggested_leverage=5.0,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
    )
    terminal_raw = append_decision_stage(
        {},
        DecisionStage.RISK_CHECK,
        DecisionStageStatus.SKIPPED,
        "已有明确终态",
    )
    terminal = SimpleNamespace(
        id=3,
        was_executed=False,
        execution_reason="",
        raw_llm_response=terminal_raw,
        position_size_pct=0.25,
        suggested_leverage=5.0,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
    )
    exchange_pending_raw = append_decision_stage(
        {},
        DecisionStage.EXCHANGE_SUBMIT,
        DecisionStageStatus.PENDING,
        "正在提交 OKX",
    )
    exchange_pending = SimpleNamespace(
        id=4,
        was_executed=False,
        execution_reason="正在提交 OKX：等待交易所回报",
        raw_llm_response=exchange_pending_raw,
        position_size_pct=0.25,
        suggested_leverage=5.0,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
    )
    ai_completed_raw = append_decision_stage(
        {},
        DecisionStage.AI_ANALYSIS,
        DecisionStageStatus.COMPLETED,
        "AI 已完成分析并生成裁决。",
    )
    ai_completed = SimpleNamespace(
        id=5,
        was_executed=False,
        execution_reason="AI 已完成分析并生成裁决。",
        raw_llm_response=ai_completed_raw,
        position_size_pct=0.25,
        suggested_leverage=5.0,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
    )
    update_raw = append_decision_stage(
        {"execution_parameters": {"position_size_pct": 0.004}},
        DecisionStage.RISK_CHECK,
        DecisionStageStatus.SKIPPED,
        "轮次结束未进入下单",
    )
    session = FakeSession(rows=[unresolved, executed, terminal, exchange_pending, ai_completed])
    repo = DecisionRepository(session)  # type: ignore[arg-type]

    updated = await repo.finalize_unresolved_decisions(
        [
            (1, "轮次结束未进入下单", update_raw),
            (2, "不应覆盖已成交", update_raw),
            (3, "不应覆盖已终态", update_raw),
            (4, "不应覆盖已进入 OKX 提交", update_raw),
            (5, "轮次结束未进入下单", update_raw),
        ]
    )

    assert updated == 3
    assert unresolved.execution_reason == "轮次结束未进入下单"
    assert unresolved.raw_llm_response["decision_state_machine"]["summary"]["final_stage"] == (
        DecisionStage.RISK_CHECK
    )
    assert unresolved.raw_llm_response["decision_state_machine"]["summary"]["final_status"] == (
        DecisionStageStatus.SKIPPED
    )
    assert unresolved.position_size_pct == 0.004
    assert executed.execution_reason == "filled"
    assert terminal.execution_reason == "已有明确终态"
    assert exchange_pending.execution_reason == "正在提交 OKX：等待交易所回报"
    assert exchange_pending.raw_llm_response == exchange_pending_raw
    assert ai_completed.execution_reason == "轮次结束未进入下单"
    assert ai_completed.raw_llm_response["decision_state_machine"]["summary"]["final_stage"] == (
        DecisionStage.RISK_CHECK
    )
    assert session.flush_count == 1
