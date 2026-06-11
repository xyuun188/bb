from types import SimpleNamespace

from services.decision_reason_recovery import DecisionReasonRecoveryPolicy


def test_decision_reason_recovery_builds_exit_reason_from_close_evidence() -> None:
    decision = SimpleNamespace(
        action="close_long",
        reasoning="模型认为趋势转弱",
        raw_llm_response={
            "close_evidence": {
                "action_plan": "reduce",
                "reason": "跌破短线结构",
                "position_unrealized_pnl": -1.23456,
            }
        },
    )

    reason = DecisionReasonRecoveryPolicy().recover(decision)

    assert reason is not None
    assert "AI 建议减仓" in reason
    assert "当时估算浮动盈亏 -1.2346 USDT" in reason
    assert "裁决依据：跌破短线结构" in reason


def test_decision_reason_recovery_uses_decision_reasoning_when_evidence_reason_missing() -> None:
    decision = SimpleNamespace(
        action="close_short",
        reasoning="空头动能衰减",
        raw_llm_response={
            "close_evidence": {
                "action_plan": "full_close",
                "position_unrealized_pnl": 2.0,
            }
        },
    )

    reason = DecisionReasonRecoveryPolicy().recover(decision)

    assert reason is not None
    assert "AI 建议全平" in reason
    assert "裁决依据：空头动能衰减" in reason


def test_decision_reason_recovery_returns_generic_exit_reason_without_detail() -> None:
    decision = SimpleNamespace(action="close_long", reasoning="", raw_llm_response={})

    assert (
        DecisionReasonRecoveryPolicy().recover(decision)
        == "平仓裁决已生成但本轮没有确认到 OKX 平仓订单结果。"
        "系统会继续以 OKX 实际仓位和执行记录为准同步；如果仓位仍存在，下一轮持仓复盘会重新评估并提交平仓。"
    )


def test_decision_reason_recovery_uses_fallback_for_non_exit() -> None:
    decision = SimpleNamespace(action="long", reasoning="", raw_llm_response={})

    assert DecisionReasonRecoveryPolicy().recover(decision, "fallback") == "fallback"
    assert DecisionReasonRecoveryPolicy().recover(None, "fallback") is None
