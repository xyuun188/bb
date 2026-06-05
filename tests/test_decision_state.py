from ai_brain.base_model import Action, DecisionOutput
from services.decision_state import (
    DecisionStage,
    DecisionStageStatus,
    append_decision_stage,
    decision_state_from_raw,
)
from services.strategy_arbitration import arbitrate_decision


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
    assert [item["stage_label"] for item in summary["by_stage"]] == [
        "AI分析",
        "风控检查",
    ]


def _decision(action: Action) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=action,
        confidence=0.8,
        reasoning="测试决策",
        position_size_pct=0.05,
        suggested_leverage=3.0,
        raw_response={},
        feature_snapshot={},
    )


def test_strategy_arbitration_records_entry_exit_and_hold_intents():
    entry = arbitrate_decision(_decision(Action.LONG))
    assert entry.passed is True
    assert entry.data["intent"] == "entry"
    assert "开仓意图" in entry.reason

    exit_result = arbitrate_decision(_decision(Action.CLOSE_LONG))
    assert exit_result.passed is True
    assert exit_result.data["intent"] == "exit"
    assert "平仓/减仓意图" in exit_result.reason

    hold = arbitrate_decision(_decision(Action.HOLD))
    assert hold.passed is False
    assert hold.data["intent"] == "hold"
    assert "观望" in hold.reason
