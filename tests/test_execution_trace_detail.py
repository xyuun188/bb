from __future__ import annotations

from datetime import UTC, datetime

import pytest

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
        data={"blocker": "okx_min_size"},
        duration_sec=0.42,
    )

    trace = build_execution_trace(raw, order_status="rejected")

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

    assert trace["final_result"]["status"] == "local_sync_failed"
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


def test_okx_51155_is_a_compliance_rejection() -> None:
    reason = "OKX API error [51155]: You can't trade this pair due to local compliance restrictions."
    raw = append_decision_stage(
        {"execution_result": {"raw_response": {"raw_error": reason}}},
        DecisionStage.EXCHANGE_CONFIRM, DecisionStageStatus.FAILED, "OKX rejected the order",
    )

    trace = build_execution_trace(raw, order_status="rejected")

    assert trace["final_result"]["status"] == "failed"
    assert "51155" in trace["final_result"]["reason"]
    assert trace["stage_events"][0]["reason"] == "OKX rejected the order"
    assert trace["failed_step"]["stage"] == DecisionStage.EXCHANGE_CONFIRM
    assert len(trace["repair_suggestions"]) == 1
    assert "合规限制" in trace["repair_suggestions"][0]
    assert "最小" not in trace["repair_suggestions"][0]


@pytest.mark.parametrize(
    ("status", "confirm_status", "evidence", "expected"),
    [
        ("cancelled", "skipped", {"profit_lock_ioc_unfilled": True}, "profit_lock_unfilled"),
        ("cancelled", "failed", {"profit_lock_ioc_unfilled": False}, "failed"),
        ("rejected", "failed", {"profit_lock_ioc_unfilled": True}, "failed"),
        ("partial", "pending", {"profit_lock_ioc_unfilled": True}, "partial"),
        ("open", "pending", {"execution_transport_unknown": True}, "unknown"),
        ("pending", "pending", {}, "pending"),
    ],
)
def test_trace_distinguishes_exchange_outcomes(status, confirm_status, evidence, expected) -> None:
    raw = append_decision_stage(
        {"execution_result": {"raw_response": evidence}}, DecisionStage.EXCHANGE_CONFIRM,
        confirm_status, "Exchange confirmation",
    )
    raw = append_decision_stage(
        raw, DecisionStage.LOCAL_SYNC, DecisionStageStatus.SKIPPED, "No local position mutation",
    )

    trace = build_execution_trace(raw, order_status=status)

    assert trace["final_result"]["status"] == expected
    assert trace["final_result"]["success"] is False
    assert (trace["failed_step"] is not None) == (expected == "failed")
    assert bool(trace["repair_suggestions"]) == (expected == "failed")
    if expected == "profit_lock_unfilled":
        assert trace["final_result"]["status_label"] == "未成交"
        assert "零成交" in trace["final_result"]["reason"]
        assert trace["stage_events"][0]["reason"] == "Exchange confirmation"


@pytest.mark.parametrize("sync_status", ["failed", "skipped"])
def test_zero_fill_does_not_hide_local_order_persistence_failure(sync_status) -> None:
    raw = append_decision_stage(
        {
            "execution_result": {"raw_response": {"profit_lock_ioc_unfilled": True}},
            "local_order_persistence": {"status": "failed"},
        },
        DecisionStage.EXCHANGE_CONFIRM, DecisionStageStatus.SKIPPED, "No fill",
    )
    raw = append_decision_stage(raw, DecisionStage.LOCAL_SYNC, sync_status, "Local sync")

    trace = build_execution_trace(raw, order_status="cancelled")

    assert trace["final_result"]["status"] == "local_sync_failed"
    assert trace["failed_step"]["stage"] == DecisionStage.LOCAL_SYNC
    assert trace["stage_events"][-1]["status"] == sync_status
