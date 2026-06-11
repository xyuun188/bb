from ai_brain.base_model import Action, DecisionOutput
from services.entry_priority import (
    MIN_ENTRY_OPPORTUNITY_SCORE,
    EntryExecutionPriorityPolicy,
)


def _decision(
    *,
    raw_response: dict | None = None,
    confidence: float = 0.8,
    position_size_pct: float = 0.02,
) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.LONG,
        confidence=confidence,
        reasoning="测试入场",
        position_size_pct=position_size_pct,
        suggested_leverage=3.0,
        raw_response=raw_response or {},
        feature_snapshot={"current_price": 100.0},
    )


def test_immediate_execution_reason_for_exceptional_entry() -> None:
    decision = _decision(
        confidence=0.87,
        raw_response={
            "opportunity_score": {
                "score": 4.3,
                "min_score_required": MIN_ENTRY_OPPORTUNITY_SCORE,
                "expected_net_return_pct": 1.25,
                "profit_quality_ratio": 1.55,
            }
        },
    )

    reason = EntryExecutionPriorityPolicy().immediate_execution_reason(decision)

    assert reason is not None
    assert reason.startswith("极强信号即时执行")
    assert "机会评分 4.30" in reason


def test_immediate_execution_reason_blocks_high_disagreement() -> None:
    decision = _decision(
        confidence=0.9,
        raw_response={
            "opportunity_score": {
                "score": 5.0,
                "expected_net_return_pct": 2.0,
                "profit_quality_ratio": 2.0,
                "high_disagreement": True,
            }
        },
    )

    assert EntryExecutionPriorityPolicy().immediate_execution_reason(decision) is None


def test_wait_sort_reason_includes_rank_and_threshold() -> None:
    decision = _decision(
        confidence=0.7,
        raw_response={
            "opportunity_score": {
                "score": 1.1,
                "min_score_required": 0.95,
                "expected_net_return_pct": 0.42,
            }
        },
    )

    reason = EntryExecutionPriorityPolicy().wait_sort_reason(
        decision,
        rank=2,
        candidate_count=5,
    )

    assert "历史候选排名参考 2/5" in reason
    assert "当前机会评分 1.10" in reason
    assert "执行门槛 0.95" in reason
