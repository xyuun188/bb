from ai_brain.base_model import Action, DecisionOutput
from services.entry_opportunity_gate import EntryOpportunityGatePolicy
from services.entry_post_crash_rebound import EntryPostCrashReboundGuardPolicy


def _decision(
    action: Action,
    *,
    feature_snapshot: dict | None = None,
) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=action,
        confidence=0.8,
        reasoning="entry",
        position_size_pct=0.05,
        suggested_leverage=3.0,
        raw_response={
            "opportunity_score": {
                "expected_net_return_pct": 0.7,
                "profit_quality_ratio": 1.1,
            }
        },
        feature_snapshot=feature_snapshot or {},
    )


def test_post_crash_rebound_guard_blocks_rebound_short() -> None:
    decision = _decision(
        Action.SHORT,
        feature_snapshot={
            "returns_1": 0.035,
            "returns_5": -0.19,
            "returns_20": -0.10,
        },
    )

    reason = EntryPostCrashReboundGuardPolicy().guard_reason(decision)

    assert reason is not None
    assert "暴跌后反弹保护" in reason
    guard = decision.raw_response["post_crash_rebound_guard"]
    assert guard["blocked"] is True
    assert guard["returns_1"] == 0.035
    assert guard["expected_net_return_pct"] == 0.7
    assert guard["profit_quality_ratio"] == 1.1


def test_post_crash_rebound_guard_allows_non_short_and_non_rebound() -> None:
    assert (
        EntryPostCrashReboundGuardPolicy().guard_reason(
            _decision(Action.LONG, feature_snapshot={"returns_1": 0.04, "returns_5": -0.2})
        )
        is None
    )
    assert (
        EntryPostCrashReboundGuardPolicy().guard_reason(
            _decision(Action.SHORT, feature_snapshot={"returns_1": 0.01, "returns_5": -0.2})
        )
        is None
    )


def test_entry_opportunity_gate_uses_injected_post_crash_rebound_guard() -> None:
    decision = _decision(
        Action.SHORT,
        feature_snapshot={
            "returns_1": 0.04,
            "returns_5": 0.0,
            "returns_20": -0.30,
        },
    )
    policy = EntryOpportunityGatePolicy(post_crash_rebound_guard=EntryPostCrashReboundGuardPolicy())

    reason = policy.gate_reason(decision)

    assert reason is not None
    assert decision.raw_response["post_crash_rebound_guard"]["blocked"] is True
