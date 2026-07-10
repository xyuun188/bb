from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from services.decision_state import (
    DecisionStage,
    DecisionStageStatus,
    append_decision_stage,
    decision_state_from_raw,
)
from services.stale_entry_candidate_expirer import (
    STALE_ENTRY_MAINTENANCE_BATCH_LIMIT,
    STALE_ENTRY_MAINTENANCE_LOOKBACK,
    StaleEntryCandidateExpirer,
    action_label,
    is_pending_execution_reason,
    pending_execution_failed_reason,
    pending_execution_is_stale,
)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _row(
    *,
    row_id: int = 1,
    symbol: str = "BTC/USDT",
    action: str = "long",
    raw: Any | None = None,
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=row_id,
        symbol=symbol,
        action=action,
        raw_llm_response=raw if raw is not None else {},
        execution_reason="",
        created_at=created_at,
        updated_at=updated_at,
    )


def test_stale_entry_candidate_reason_helpers() -> None:
    assert action_label("long") == "做多"
    assert action_label("short") == "做空"
    assert is_pending_execution_reason("")
    assert is_pending_execution_reason("正在提交 OKX：下单中")
    assert not is_pending_execution_reason("风险检查拦截")
    assert "45 秒内没有生成本地订单记录" in pending_execution_failed_reason(
        "BTC/USDT",
        "long",
    )


@pytest.mark.asyncio
async def test_stale_entry_candidate_expirer_marks_waiting_rows() -> None:
    waiting = [
        _row(
            raw={
                "opportunity_score": {
                    "score": 0.7,
                    "min_score_required": 0.8,
                    "expected_net_return_pct": 0.2,
                }
            }
        )
    ]
    flushed = False

    async def order_count_provider(_decision_id: int) -> int:
        return 0

    async def flush() -> None:
        nonlocal flushed
        flushed = True

    expired = await StaleEntryCandidateExpirer(_float).expire_rows(
        waiting,
        [],
        order_count_provider=order_count_provider,
        flush_callback=flush,
    )

    assert expired == 1
    assert flushed
    assert "机会评分 0.7000 低于执行门槛 0.80" in waiting[0].execution_reason
    assert waiting[0].raw_llm_response["opportunity_score"]["selected_for_execution"] is False
    state = decision_state_from_raw(waiting[0].raw_llm_response)["summary"]
    assert state["final_stage"] == DecisionStage.RISK_CHECK
    assert state["final_status"] == DecisionStageStatus.SKIPPED
    assert waiting[0].raw_llm_response["skip_kind"] == "stale_entry_candidate_expired"


@pytest.mark.asyncio
async def test_stale_entry_expirer_repairs_old_expired_reason_with_pending_state() -> None:
    raw = append_decision_stage(
        {
            "opportunity_score": {
                "score": 0.7,
                "min_score_required": 0.8,
                "expected_net_return_pct": 0.2,
            }
        },
        DecisionStage.RISK_CHECK,
        DecisionStageStatus.PENDING,
        "已进入执行前严重风险检查。",
    )
    waiting = [
        _row(
            raw=raw,
        )
    ]
    waiting[0].execution_reason = (
        "候选排序超时后复核：BTC/USDT 本次做多机会评分 0.7000 "
        "低于执行门槛 0.80，旧信号不再执行，下一轮重新分析。"
    )

    async def order_count_provider(_decision_id: int) -> int:
        return 0

    expired = await StaleEntryCandidateExpirer(_float).expire_rows(
        waiting,
        [],
        order_count_provider=order_count_provider,
    )

    assert expired == 1
    state = decision_state_from_raw(waiting[0].raw_llm_response)["summary"]
    assert state["final_stage"] == DecisionStage.RISK_CHECK
    assert state["final_status"] == DecisionStageStatus.SKIPPED


@pytest.mark.asyncio
async def test_stale_candidate_loader_bounds_one_background_batch() -> None:
    statements: list[Any] = []

    class FakeResult:
        def mappings(self):
            return self

        def scalars(self):
            return self

        def all(self):
            return []

    class FakeSession:
        async def execute(self, statement):
            statements.append(statement)
            return FakeResult()

    expirer = StaleEntryCandidateExpirer(_float)
    since = datetime.utcnow() - STALE_ENTRY_MAINTENANCE_LOOKBACK
    assert await expirer._load_rows(
        FakeSession(),
        since=since,
        cutoff=datetime.utcnow(),
        reason_patterns=("waiting%",),
    ) == []

    assert statements[0]._limit_clause.value == STALE_ENTRY_MAINTENANCE_BATCH_LIMIT
    compiled = statements[0].compile()
    assert "ai_decisions.created_at >=" in str(compiled)
    assert since in compiled.params.values()

    assert await expirer._load_stale_open_state_rows(
        FakeSession(),
        since=since,
        cutoff=datetime.utcnow(),
    ) == []
    compiled_open_state = statements[1].compile()
    assert "ai_decisions.created_at >=" in str(compiled_open_state)
    assert since in compiled_open_state.params.values()


@pytest.mark.asyncio
async def test_stale_candidate_postgres_updates_patch_json_without_full_row_load() -> None:
    from sqlalchemy.dialects import postgresql

    row = _row(raw={"opportunity_score": {"score": 0.2}, "decision_state_machine": {}})
    expirer = StaleEntryCandidateExpirer(_float)
    expirer._apply_reason(
        row,
        "expired",
        stage=DecisionStage.RISK_CHECK,
        status=DecisionStageStatus.SKIPPED,
        skip_kind="stale_entry_candidate_expired",
        terminal=True,
    )

    class FakeSession:
        statement: Any | None = None
        payloads: Any | None = None

        async def execute(self, statement, payloads):
            self.statement = statement
            self.payloads = payloads

    session = FakeSession()
    await expirer._persist_postgres_updates(session, {int(row.id): row})

    compiled = session.statement.compile(dialect=postgresql.dialect())
    sql = str(compiled).lower()
    assert sql.startswith("update ai_decisions")
    assert "jsonb_set" in sql
    assert "select" not in sql
    assert session.payloads[0]["opportunity_patch"]["selection_reason"] == "expired"


def test_stale_entry_expirer_preserves_full_audit_payload_when_writing_projection() -> None:
    expirer = StaleEntryCandidateExpirer(_float)
    row = _row(
        raw={
            "opportunity_score": {"score": 0.2},
            "decision_state_machine": {"stages": []},
            "full_model_transcript": "preserve-me",
        }
    )
    expirer._apply_reason(
        row,
        "expired",
        stage=DecisionStage.RISK_CHECK,
        status=DecisionStageStatus.SKIPPED,
        skip_kind="stale_entry_candidate_expired",
        terminal=True,
    )

    from services.stale_entry_candidate_expirer import _merge_expired_raw_response

    merged = _merge_expired_raw_response(
        {
            "full_model_transcript": "preserve-me",
            "unrelated": {"keep": True},
            "opportunity_score": {"full_evidence": {"keep": True}},
            "decision_state_machine": {"historic_metadata": {"keep": True}},
        },
        row.raw_llm_response,
    )

    assert merged["full_model_transcript"] == "preserve-me"
    assert merged["unrelated"] == {"keep": True}
    assert merged["opportunity_score"]["full_evidence"] == {"keep": True}
    assert merged["decision_state_machine"]["historic_metadata"] == {"keep": True}
    assert merged["skip_kind"] == "stale_entry_candidate_expired"


@pytest.mark.asyncio
async def test_stale_entry_expirer_repairs_ai_analysis_completed_state() -> None:
    raw = append_decision_stage(
        {
            "opportunity_score": {
                "score": 0.7,
                "min_score_required": 0.8,
                "expected_net_return_pct": 0.2,
            }
        },
        DecisionStage.AI_ANALYSIS,
        DecisionStageStatus.COMPLETED,
        "AI analysis completed.",
    )
    waiting = [_row(raw=raw)]
    waiting[0].execution_reason = "AI analysis completed."

    async def order_count_provider(_decision_id: int) -> int:
        return 0

    expired = await StaleEntryCandidateExpirer(_float).expire_rows(
        waiting,
        [],
        order_count_provider=order_count_provider,
    )

    assert expired == 1
    state = decision_state_from_raw(waiting[0].raw_llm_response)["summary"]
    assert state["final_stage"] == DecisionStage.RISK_CHECK
    assert state["final_status"] == DecisionStageStatus.SKIPPED


@pytest.mark.asyncio
async def test_stale_entry_expirer_does_not_duplicate_terminal_state() -> None:
    raw = append_decision_stage(
        {
            "opportunity_score": {
                "score": 0.7,
                "min_score_required": 0.8,
                "expected_net_return_pct": 0.2,
            }
        },
        DecisionStage.RISK_CHECK,
        DecisionStageStatus.SKIPPED,
        "候选排序超时后复核：旧信号不再执行，下一轮重新分析。",
    )
    waiting = [_row(raw=raw)]

    async def order_count_provider(_decision_id: int) -> int:
        return 0

    expired = await StaleEntryCandidateExpirer(_float).expire_rows(
        waiting,
        [],
        order_count_provider=order_count_provider,
    )

    assert expired == 0
    assert len(waiting[0].raw_llm_response["decision_state_machine"]["stages"]) == 1


@pytest.mark.asyncio
async def test_stale_entry_candidate_expirer_marks_pending_rows_by_order_state() -> None:
    pending_without_order = _row(row_id=11, symbol="ETH/USDT", action="short")
    pending_with_order = _row(row_id=12, symbol="SOL/USDT", action="long")

    async def order_count_provider(decision_id: int) -> int:
        return 1 if decision_id == 12 else 0

    expired = await StaleEntryCandidateExpirer(_float).expire_rows(
        [],
        [pending_without_order, pending_with_order],
        order_count_provider=order_count_provider,
    )

    assert expired == 2
    assert "45 秒内没有生成本地订单记录" in pending_without_order.execution_reason
    assert "本地订单记录已生成" in pending_with_order.execution_reason
    without_order_state = decision_state_from_raw(pending_without_order.raw_llm_response)[
        "summary"
    ]
    with_order_state = decision_state_from_raw(pending_with_order.raw_llm_response)["summary"]
    assert without_order_state["final_stage"] == DecisionStage.LOCAL_SYNC
    assert without_order_state["final_status"] == DecisionStageStatus.SKIPPED
    assert pending_without_order.raw_llm_response["skip_kind"] == "pending_entry_execution_expired"
    assert (
        pending_with_order.raw_llm_response["opportunity_score"]["selected_for_execution"] is True
    )
    assert with_order_state["final_stage"] == DecisionStage.EXCHANGE_CONFIRM
    assert with_order_state["final_status"] == DecisionStageStatus.PENDING
    assert pending_with_order.raw_llm_response["skip_kind"] == "pending_exchange_order_status"


@pytest.mark.asyncio
async def test_pending_execution_expiry_uses_pending_state_time_not_decision_time() -> None:
    now = datetime(2026, 6, 30, 12, 0, 0)
    old_decision_recent_pending = _row(
        row_id=21,
        created_at=now - timedelta(minutes=5),
        updated_at=now - timedelta(seconds=10),
    )

    async def order_count_provider(_decision_id: int) -> int:
        return 0

    expired = await StaleEntryCandidateExpirer(_float).expire_rows(
        [],
        [old_decision_recent_pending],
        now=now,
        order_count_provider=order_count_provider,
    )

    assert expired == 0
    assert old_decision_recent_pending.execution_reason == ""
    assert not pending_execution_is_stale(old_decision_recent_pending, now)


@pytest.mark.asyncio
async def test_pending_execution_expiry_prefers_exchange_submit_stage_time() -> None:
    now = datetime(2026, 6, 30, 12, 0, 0)
    raw = append_decision_stage(
        {},
        DecisionStage.EXCHANGE_SUBMIT,
        DecisionStageStatus.PENDING,
        "正在提交 OKX 订单并等待交易所返回结果。",
        at=now - timedelta(seconds=46),
    )
    pending_row = _row(
        row_id=22,
        symbol="LIT/USDT",
        raw=raw,
        created_at=now - timedelta(minutes=5),
        updated_at=now - timedelta(seconds=5),
    )

    async def order_count_provider(_decision_id: int) -> int:
        return 0

    expired = await StaleEntryCandidateExpirer(_float).expire_rows(
        [],
        [pending_row],
        now=now,
        order_count_provider=order_count_provider,
    )

    assert expired == 1
    assert "45 秒内没有生成本地订单记录" in pending_row.execution_reason
    assert pending_execution_is_stale(pending_row, now)
    state = decision_state_from_raw(pending_row.raw_llm_response)["summary"]
    assert state["final_stage"] == DecisionStage.LOCAL_SYNC
    assert state["final_status"] == DecisionStageStatus.SKIPPED


@pytest.mark.asyncio
async def test_stale_expirer_uses_selected_side_net_not_aggregate() -> None:
    waiting = [
        _row(
            action="long",
            raw={
                "opportunity_score": {
                    "score": 1.4,
                    "min_score_required": 0.8,
                    "expected_net_return_pct": 1.2,
                },
                "entry_candidate_evidence": {
                    "long": {"expected_net_return_pct": -0.04},
                },
            },
        )
    ]

    async def order_count_provider(_decision_id: int) -> int:
        return 0

    expired = await StaleEntryCandidateExpirer(_float).expire_rows(
        waiting,
        [],
        order_count_provider=order_count_provider,
    )

    assert expired == 1
    assert "-0.0400%" in waiting[0].execution_reason
