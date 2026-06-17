from __future__ import annotations

from datetime import UTC, datetime

from services.decision_execution_trace import build_execution_trace
from services.decision_state import DecisionStage, DecisionStageStatus, append_decision_stage


def test_execution_trace_exposes_step_duration_and_failed_step() -> None:
    raw = append_decision_stage(
        {},
        DecisionStage.AI_ANALYSIS,
        DecisionStageStatus.COMPLETED,
        "AI 完成分析",
        duration_sec=1.2345,
    )
    raw = append_decision_stage(
        raw,
        DecisionStage.RISK_CHECK,
        DecisionStageStatus.BLOCKED,
        "OKX 最小下单数量不满足",
        data={"blocker": "okx_min_size", "okx_code": "51155"},
        duration_sec=0.42,
    )

    trace = build_execution_trace(raw, order_status="rejected", fallback_reason="51155")

    assert trace["final_result"]["success"] is False
    assert trace["failed_step"]["stage"] == DecisionStage.RISK_CHECK
    assert trace["failed_step"]["duration_sec"] == 0.42
    assert trace["execution_steps"][0]["duration_sec"] == 1.234
    assert any("最小" in item for item in trace["repair_suggestions"])


def test_execution_trace_synthesizes_legacy_order_snapshot() -> None:
    trace = build_execution_trace(
        {},
        order_status="filled",
        order_created_at=datetime(2026, 6, 17, 1, 0, 0, tzinfo=UTC),
        order_filled_at=datetime(2026, 6, 17, 1, 0, 3, tzinfo=UTC),
        fallback_reason="订单执行成功",
    )

    assert trace["final_result"]["success"] is True
    assert [step["stage"] for step in trace["execution_steps"]] == [
        DecisionStage.EXCHANGE_SUBMIT,
        DecisionStage.LOCAL_SYNC,
    ]
    assert trace["execution_steps"][0]["duration_sec"] == 3.0
    assert trace["execution_steps"][0]["data"]["source"] == "order_snapshot"
