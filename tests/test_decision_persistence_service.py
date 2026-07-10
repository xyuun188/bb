from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

import services.decision_persistence_service as decision_persistence_module
from ai_brain.base_model import Action, DecisionOutput
from services.decision_persistence_service import DecisionPersistenceService
from services.decision_state import DecisionStage, DecisionStageStatus, append_decision_stage


def _decision(action: Action = Action.LONG) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT:USDT",
        action=action,
        confidence=0.71,
        reasoning="  测试 理由  ",
        position_size_pct=0.03,
        suggested_leverage=3.0,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        raw_response={"analysis_type": "market_scan"},
        feature_snapshot={"nan": float("nan"), "time": datetime(2026, 6, 10, tzinfo=UTC)},
    )


class FakeSessionContext:
    def __init__(self, session: Any) -> None:
        self.session = session

    async def __aenter__(self) -> Any:
        return self.session

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False


class FakeExecuteResult:
    def __init__(self, value: int) -> None:
        self.value = value

    def scalar(self) -> int:
        return self.value


class FakeSession:
    def __init__(self, *, order_count: int = 0, row: Any = None) -> None:
        self.order_count = order_count
        self.row = row

    async def execute(self, _stmt: Any) -> FakeExecuteResult:
        return FakeExecuteResult(self.order_count)

    async def get(self, _model: Any, _row_id: int) -> Any:
        return self.row


class FakeDecisionRepo:
    def __init__(self) -> None:
        self.logged: list[dict[str, Any]] = []
        self.marked_executed: list[tuple[int, float]] = []
        self.reasons: list[tuple[int, str | None]] = []
        self.raw_updates: list[tuple[int, dict[str, Any] | None]] = []
        self.fill_missing: list[tuple[list[int], str]] = []
        self.finalize_unresolved: list[list[tuple[int, str, dict[str, Any]]]] = []
        self.outcomes: list[tuple[int, str, float]] = []

    async def log_decision(self, data: dict[str, Any]) -> SimpleNamespace:
        self.logged.append(data)
        return SimpleNamespace(id=123)

    async def mark_executed(self, decision_id: int, execution_price: float) -> None:
        self.marked_executed.append((decision_id, execution_price))

    async def mark_execution_reason(self, decision_id: int, reason: str | None) -> None:
        self.reasons.append((decision_id, reason))

    async def update_raw_response(
        self, decision_id: int, raw_response: dict[str, Any] | None
    ) -> None:
        self.raw_updates.append((decision_id, raw_response))

    async def fill_missing_execution_reasons(self, ids: list[int], reason: str) -> None:
        self.fill_missing.append((ids, reason))

    async def finalize_unresolved_decisions(
        self,
        decision_updates: list[tuple[int, str, dict[str, Any]]],
    ) -> int:
        self.finalize_unresolved.append(decision_updates)
        return len(decision_updates)

    async def mark_outcome(self, decision_id: int, outcome: str, pnl_pct: float) -> None:
        self.outcomes.append((decision_id, outcome, pnl_pct))


def _service(session: FakeSession, repo: FakeDecisionRepo) -> DecisionPersistenceService:
    return DecisionPersistenceService(
        normalize_symbol=lambda symbol: str(symbol or "").replace(":USDT", ""),
        session_context_factory=lambda: FakeSessionContext(session),
        decision_repo_factory=lambda _session: repo,
    )


@pytest.mark.asyncio
async def test_decision_persistence_uses_unified_runtime_text_boundary(
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
        decision_persistence_module,
        "sanitize_runtime_text",
        fake_sanitize,
        raising=False,
    )
    repo = FakeDecisionRepo()
    service = _service(FakeSession(), repo)
    decision = _decision()
    decision.reasoning = "raw decision reason"
    decision.raw_response = {"analysis_type": "market_scan", "note": "raw llm"}
    decision.feature_snapshot = {"note": "raw feature"}

    await service.log_decision(decision, is_paper=True)
    await service.record_and_persist_stage(
        decision_id=9,
        decision=decision,
        stage=DecisionStage.RISK_CHECK,
        status=DecisionStageStatus.BLOCKED,
        reason="raw stage reason",
        data={"note": "raw stage data"},
    )
    await service.mark_reason(9, "raw final reason")
    await service.fill_missing_reasons([9], "raw missing reason")

    assert repo.logged[0]["reasoning"] == "unified:raw decision reason"
    assert repo.logged[0]["feature_snapshot"]["unified"]["note"] == "unified:raw feature"
    assert repo.logged[0]["raw_llm_response"]["unified"]["note"] == "unified:raw llm"
    assert "decision_state_machine" in repo.logged[0]["raw_llm_response"]["unified"]
    assert repo.raw_updates[-1][1]["unified"]["note"] == "unified:raw llm"
    assert repo.reasons == [(9, "unified:raw final reason")]
    assert repo.fill_missing == [([9], "unified:raw missing reason")]
    assert "raw decision reason" in calls


@pytest.mark.asyncio
async def test_log_decision_attaches_stage_and_sanitizes_payload() -> None:
    repo = FakeDecisionRepo()
    service = _service(FakeSession(), repo)
    decision = _decision()

    decision_id = await service.log_decision(decision, is_paper=True)

    assert decision_id == 123
    assert repo.logged[0]["symbol"] == "BTC/USDT"
    assert repo.logged[0]["analysis_type"] == "market"
    assert repo.logged[0]["is_paper"] is True
    assert repo.logged[0]["feature_snapshot"]["nan"] is None
    assert repo.logged[0]["feature_snapshot"]["time"] == "2026-06-10T00:00:00+00:00"
    machine = repo.logged[0]["raw_llm_response"]["decision_state_machine"]
    assert machine["current_stage"] == DecisionStage.AI_ANALYSIS
    assert machine["current_status"] == DecisionStageStatus.COMPLETED
    assert machine["last_reason"] == "AI 已完成分析并生成裁决。"


def test_analysis_type_detects_position_review_and_exits() -> None:
    hold = _decision(Action.HOLD)
    close = _decision(Action.CLOSE_LONG)

    assert DecisionPersistenceService.analysis_type(hold, {"analysis_type": "position_review"}) == (
        "position"
    )
    assert DecisionPersistenceService.analysis_type(hold, {"position_review": True}) == "position"
    assert DecisionPersistenceService.analysis_type(close, {}) == "position"
    assert (
        DecisionPersistenceService.analysis_type(
            hold,
            {"analysis_type": "entry_candidate"},
        )
        == "entry_candidate"
    )
    assert DecisionPersistenceService.analysis_type(hold, {}) == "market"


@pytest.mark.asyncio
async def test_record_and_persist_stage_updates_raw_response() -> None:
    repo = FakeDecisionRepo()
    service = _service(FakeSession(), repo)
    decision = _decision()

    raw = await service.record_and_persist_stage(
        decision_id=9,
        decision=decision,
        stage=DecisionStage.RISK_CHECK,
        status=DecisionStageStatus.BLOCKED,
        reason="风险拒绝",
        data={"bad": float("inf")},
        duration_sec=0.75,
    )

    assert repo.raw_updates == [(9, raw)]
    event = raw["decision_state_machine"]["stages"][-1]
    assert event["stage"] == DecisionStage.RISK_CHECK
    assert event["data"] == {"bad": None}
    assert event["duration_sec"] == 0.75


@pytest.mark.asyncio
async def test_mark_reason_recovers_unusable_legacy_text() -> None:
    repo = FakeDecisionRepo()
    service = _service(FakeSession(row=SimpleNamespace(id=7)), repo)

    await service.mark_reason(
        7,
        "原始说明已损坏",
        reason_recoverer=lambda row, fallback: f"recovered:{row.id}:{fallback}",
    )

    assert repo.reasons == [(7, "recovered:7:原始说明已损坏")]


@pytest.mark.asyncio
async def test_mark_pending_fill_missing_and_outcome_delegate_to_repo() -> None:
    repo = FakeDecisionRepo()
    service = _service(FakeSession(), repo)

    await service.mark_pending_execution(8, "下单中")
    await service.fill_missing_reasons({0, 8, 9}, "仍在处理中")
    await service.mark_executed(8, 123.4)
    await service.mark_outcome(8, "profit", 0.012)

    assert repo.reasons == [(8, "正在提交 OKX：下单中")]
    assert repo.fill_missing == [([8, 9], "仍在处理中")]
    assert repo.marked_executed == [(8, 123.4)]
    assert repo.outcomes == [(8, "profit", 0.012)]


@pytest.mark.asyncio
async def test_finalize_unresolved_decisions_adds_terminal_risk_skip() -> None:
    repo = FakeDecisionRepo()
    service = _service(FakeSession(), repo)
    decision = _decision()

    count = await service.finalize_unresolved_decisions({12: decision}, "轮次结束未进入下单")

    assert count == 1
    decision_id, reason, raw = repo.finalize_unresolved[0][0]
    assert decision_id == 12
    assert reason == "轮次结束未进入下单"
    machine = raw["decision_state_machine"]
    assert machine["current_stage"] == DecisionStage.RISK_CHECK
    assert machine["current_status"] == DecisionStageStatus.SKIPPED
    assert machine["summary"]["final_stage"] == DecisionStage.RISK_CHECK
    assert machine["summary"]["final_status"] == DecisionStageStatus.SKIPPED
    assert machine["stages"][-1]["data"]["skip_kind"] == "round_unresolved_terminal_skip"


@pytest.mark.asyncio
async def test_finalize_unresolved_decisions_preserves_existing_terminal_trail() -> None:
    repo = FakeDecisionRepo()
    service = _service(FakeSession(), repo)
    decision = _decision()
    decision.raw_response = append_decision_stage(
        decision.raw_response,
        DecisionStage.STRATEGY_ARBITRATION,
        DecisionStageStatus.SKIPPED,
        "已有同币种分析流程占用执行权。",
        {"skip_kind": "analysis_symbol_claimed"},
    )

    count = await service.finalize_unresolved_decisions({12: decision}, "round ended")

    assert count == 1
    _decision_id, reason, raw = repo.finalize_unresolved[0][0]
    assert reason == "已有同币种分析流程占用执行权。"
    machine = raw["decision_state_machine"]
    assert machine["current_stage"] == DecisionStage.STRATEGY_ARBITRATION
    assert machine["current_status"] == DecisionStageStatus.SKIPPED
    assert raw.get("skip_kind") != "round_unresolved_terminal_skip"


@pytest.mark.asyncio
async def test_duplicate_order_reason_describes_entry_and_exit_duplicates() -> None:
    repo = FakeDecisionRepo()

    entry_reason = await _service(FakeSession(order_count=2), repo).duplicate_order_reason(
        10, _decision(Action.LONG)
    )
    exit_reason = await _service(FakeSession(order_count=1), repo).duplicate_order_reason(
        11, _decision(Action.CLOSE_SHORT)
    )
    no_duplicate = await _service(FakeSession(order_count=0), repo).duplicate_order_reason(
        12, _decision(Action.HOLD)
    )

    assert entry_reason is not None and "重复开仓" in entry_reason
    assert exit_reason is not None and "重复平仓" in exit_reason
    assert no_duplicate is None
