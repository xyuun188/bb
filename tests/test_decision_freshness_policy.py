from datetime import UTC, datetime, timedelta

from ai_brain.base_model import Action, DecisionOutput
from services.decision_freshness import (
    ENTRY_DECISION_MAX_AGE_SECONDS,
    ENTRY_EXCEPTIONAL_OPPORTUNITY_MAX_AGE_SECONDS,
    ENTRY_STRONG_OPPORTUNITY_MAX_AGE_SECONDS,
    EXIT_DECISION_MAX_AGE_SECONDS,
    PROFIT_PROTECTION_EXIT_MAX_AGE_SECONDS,
    DecisionFreshnessPolicy,
)


def _decision(
    action: Action,
    *,
    timestamp: datetime,
    confidence: float = 0.7,
    raw_response: dict | None = None,
    feature_snapshot: dict | None = None,
) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=action,
        confidence=confidence,
        reasoning="测试信号",
        position_size_pct=0.05,
        suggested_leverage=3.0,
        raw_response=raw_response or {},
        feature_snapshot=feature_snapshot or {},
        timestamp=timestamp,
    )


def test_entry_decision_stale_reason_uses_timing_not_market_snapshot() -> None:
    now = datetime(2026, 6, 8, 12, 0, tzinfo=UTC)
    completed_at = now - timedelta(seconds=ENTRY_DECISION_MAX_AGE_SECONDS + 5)
    fresh_market_snapshot = now - timedelta(seconds=10)
    decision = _decision(
        Action.LONG,
        timestamp=now,
        raw_response={"timing": {"decision_completed_at": completed_at.isoformat()}},
        feature_snapshot={"timestamp": fresh_market_snapshot.isoformat()},
    )
    policy = DecisionFreshnessPolicy(clock=lambda: now)

    reason = policy.stale_decision_reason(decision)

    assert reason is not None
    assert "AI开仓裁决完成到准备下单" in reason
    assert decision.raw_response["stale_decision_check"]["max_age_seconds"] == (
        ENTRY_DECISION_MAX_AGE_SECONDS
    )
    assert decision.raw_response["stale_decision_check"]["reference_time"] == (
        completed_at.isoformat()
    )


def test_strong_entry_uses_shorter_freshness_window() -> None:
    now = datetime(2026, 6, 8, 12, 0, tzinfo=UTC)
    decision = _decision(
        Action.LONG,
        timestamp=now - timedelta(seconds=ENTRY_STRONG_OPPORTUNITY_MAX_AGE_SECONDS + 1),
        confidence=0.76,
        raw_response={
            "opportunity_score": {
                "score": 3.1,
                "ai_expected_return_pct": 2.1,
                "reward_risk_ratio": 1.25,
            }
        },
    )
    policy = DecisionFreshnessPolicy(clock=lambda: now)

    assert policy.max_age_seconds(decision) == ENTRY_STRONG_OPPORTUNITY_MAX_AGE_SECONDS
    assert policy.stale_decision_reason(decision) is not None


def test_exceptional_entry_keeps_default_freshness_window() -> None:
    now = datetime(2026, 6, 8, 12, 0, tzinfo=UTC)
    decision = _decision(
        Action.LONG,
        timestamp=now - timedelta(seconds=ENTRY_EXCEPTIONAL_OPPORTUNITY_MAX_AGE_SECONDS - 1),
        confidence=0.86,
        raw_response={
            "opportunity_score": {
                "score": 6.0,
                "ai_expected_return_pct": 4.0,
                "reward_risk_ratio": 1.5,
            }
        },
    )
    policy = DecisionFreshnessPolicy(clock=lambda: now)

    assert policy.max_age_seconds(decision) == ENTRY_EXCEPTIONAL_OPPORTUNITY_MAX_AGE_SECONDS
    assert policy.stale_decision_reason(decision) is None


def test_profit_protection_exit_uses_extended_window() -> None:
    now = datetime(2026, 6, 8, 12, 0, tzinfo=UTC)
    decision = _decision(
        Action.CLOSE_LONG,
        timestamp=now - timedelta(seconds=EXIT_DECISION_MAX_AGE_SECONDS + 10),
        raw_response={"close_evidence": {"profit_protection": True}},
    )
    policy = DecisionFreshnessPolicy(clock=lambda: now)

    assert policy.max_age_seconds(decision) == PROFIT_PROTECTION_EXIT_MAX_AGE_SECONDS
    assert policy.stale_decision_reason(decision) is None


def test_forced_exit_bypasses_freshness_check() -> None:
    now = datetime(2026, 6, 8, 12, 0, tzinfo=UTC)
    decision = _decision(
        Action.CLOSE_LONG,
        timestamp=now - timedelta(seconds=PROFIT_PROTECTION_EXIT_MAX_AGE_SECONDS + 100),
    )
    policy = DecisionFreshnessPolicy(
        forced_exit_checker=lambda checked: checked is decision,
        clock=lambda: now,
    )

    assert policy.stale_decision_reason(decision) is None
    assert "stale_decision_check" not in decision.raw_response
