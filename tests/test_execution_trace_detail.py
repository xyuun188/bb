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


def test_execution_trace_labels_okx_50001_as_transient_exchange_error() -> None:
    trace = build_execution_trace(
        {},
        order_status="rejected",
        fallback_reason=(
            'Max retries exceeded: okx {"code":"50001","data":[],'
            '"msg":"Service temporarily unavailable. Please try again later."}'
        ),
    )

    assert trace["final_result"]["status"] == "transient_exchange_error"
    assert trace["final_result"]["status_label"] == "交易所临时不可用"
    assert trace["repair_suggestions"] == [
        "OKX 交易所服务临时不可用：系统未拿到成交确认，本地仓位未改动；已临时跳过该币种，稍后自动重试。"
    ]


def test_no_position_projects_historical_failure_without_inventing_fill() -> None:
    raw = append_decision_stage(
        {}, DecisionStage.EXCHANGE_CONFIRM, DecisionStageStatus.FAILED,
        "OKX 当前没有对应方向的可平仓位。",
        data={"order_id": "no_position", "status": "rejected"},
    )
    trace = build_execution_trace(raw, order_status="rejected",
                                  fallback_reason="OKX 当前没有对应方向的可平仓位。")

    assert trace["final_result"]["status"] == "no_exchange_position"
    assert trace["final_result"]["success"] is False
    assert trace["final_result"]["status_label"] == "已同步，无需重复平仓"
    assert "没有对应方向" in trace["final_result"]["reason"]
    assert trace["failed_step"] is None
    assert trace["repair_suggestions"] == []
    assert trace["execution_steps"][0]["status"] == "skipped"
    assert trace["stage_events"][0]["status"] == "failed"


def test_no_position_does_not_hide_other_failure() -> None:
    raw = append_decision_stage(
        {}, DecisionStage.RISK_CHECK, DecisionStageStatus.FAILED, "invalid risk contract"
    )
    raw = append_decision_stage(
        raw, DecisionStage.EXCHANGE_CONFIRM, DecisionStageStatus.FAILED,
        "No matching position to close", data={"order_id": "no_position"},
    )
    trace = build_execution_trace(raw, order_status="rejected")

    assert trace["final_result"]["status"] == "failed"
    assert trace["failed_step"]["stage"] == DecisionStage.RISK_CHECK


def test_no_position_keeps_local_sync_failure_visible() -> None:
    raw = append_decision_stage(
        {}, DecisionStage.EXCHANGE_CONFIRM, DecisionStageStatus.FAILED,
        "No matching position to close", data={"order_id": "no_position"},
    )
    raw = append_decision_stage(
        raw, DecisionStage.LOCAL_SYNC, DecisionStageStatus.FAILED,
        "position reconciliation database write failed",
    )
    trace = build_execution_trace(raw, order_status="rejected")

    assert trace["final_result"]["status"] == "failed"
    assert trace["failed_step"]["stage"] == DecisionStage.LOCAL_SYNC
    assert trace["stage_events"][0]["status"] == "failed"


def test_no_position_snapshot_without_stage_events_is_terminal_skip() -> None:
    trace = build_execution_trace(
        {}, order_status="rejected", fallback_reason="No matching position to close"
    )

    assert trace["final_result"]["status"] == "no_exchange_position"
    assert trace["final_result"]["success"] is False
    assert trace["failed_step"] is None


def test_close_quantity_error_is_not_no_position() -> None:
    trace = build_execution_trace(
        {}, order_status="rejected", fallback_reason="可平仓位数量不符合交易所步长"
    )

    assert trace["final_result"]["status"] == "failed"
    assert trace["failed_step"] is not None
