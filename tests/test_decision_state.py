from services.decision_state import (
    DecisionStage,
    DecisionStageStatus,
    append_decision_stage,
    decision_state_from_raw,
)


def test_decision_state_records_ordered_summary():
    raw = append_decision_stage(
        {},
        DecisionStage.AI_ANALYSIS,
        DecisionStageStatus.COMPLETED,
        "AI 已完成分析。",
    )
    raw = append_decision_stage(
        raw,
        DecisionStage.RISK_CHECK,
        DecisionStageStatus.BLOCKED,
        "下单前价格偏移过大。",
    )

    machine = decision_state_from_raw(raw)
    summary = machine["summary"]

    assert summary["blocked"] is True
    assert summary["failed"] is False
    assert summary["final_stage"] == DecisionStage.RISK_CHECK
    assert summary["final_status"] == DecisionStageStatus.BLOCKED
    assert summary["final_reason"] == "下单前价格偏移过大。"
    assert [item["stage"] for item in summary["by_stage"]] == [
        DecisionStage.AI_ANALYSIS,
        DecisionStage.RISK_CHECK,
    ]
