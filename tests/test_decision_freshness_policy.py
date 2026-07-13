from datetime import UTC, datetime, timedelta

from ai_brain.base_model import Action, DecisionOutput
from services.decision_freshness import DecisionFreshnessPolicy


def _decision(
    *,
    now: datetime,
    generated_at: datetime | None,
    valid_for_seconds: float | None,
    confidence: float = 0.7,
) -> DecisionOutput:
    provenance = {
        "source": "production_return_distribution",
        "generated_at": generated_at.isoformat() if generated_at else "",
    }
    if valid_for_seconds is not None:
        provenance["valid_for_seconds"] = valid_for_seconds
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.LONG,
        confidence=confidence,
        reasoning="test",
        raw_response={
            "opportunity_score": {
                "production_eligible": True,
                "policy_provenance": provenance,
            }
        },
        feature_snapshot={"timestamp": (now - timedelta(seconds=1)).isoformat()},
        timestamp=now,
    )


def test_entry_freshness_uses_return_provenance_instead_of_market_snapshot() -> None:
    now = datetime(2026, 7, 12, 12, 0, tzinfo=UTC)
    generated_at = now - timedelta(seconds=61)
    decision = _decision(
        now=now,
        generated_at=generated_at,
        valid_for_seconds=60,
    )
    policy = DecisionFreshnessPolicy(clock=lambda: now)

    reason = policy.stale_decision_reason(decision)

    assert reason is not None
    check = decision.raw_response["stale_decision_check"]
    assert check["valid_for_seconds"] == 60
    assert check["reason"] == "return_horizon_expired"
    assert check["reference_time"] == generated_at.isoformat()


def test_entry_freshness_changes_with_model_generated_horizon() -> None:
    now = datetime(2026, 7, 12, 12, 0, tzinfo=UTC)
    generated_at = now - timedelta(seconds=90)
    short_horizon = _decision(
        now=now,
        generated_at=generated_at,
        valid_for_seconds=60,
    )
    long_horizon = _decision(
        now=now,
        generated_at=generated_at,
        valid_for_seconds=120,
    )
    policy = DecisionFreshnessPolicy(clock=lambda: now)

    assert policy.stale_decision_reason(short_horizon) is not None
    assert policy.stale_decision_reason(long_horizon) is None


def test_entry_freshness_fails_closed_without_dynamic_horizon() -> None:
    now = datetime(2026, 7, 12, 12, 0, tzinfo=UTC)
    decision = _decision(
        now=now,
        generated_at=now,
        valid_for_seconds=None,
    )

    reason = DecisionFreshnessPolicy(clock=lambda: now).stale_decision_reason(decision)

    assert reason is not None
    assert "动态有效期" in reason
    assert decision.raw_response["stale_decision_check"]["reason"] == (
        "return_horizon_provenance_missing"
    )


def test_confidence_and_score_cannot_extend_return_horizon() -> None:
    now = datetime(2026, 7, 12, 12, 0, tzinfo=UTC)
    decision = _decision(
        now=now,
        generated_at=now - timedelta(seconds=61),
        valid_for_seconds=60,
        confidence=0.99,
    )
    decision.raw_response["opportunity_score"]["score"] = 999

    assert DecisionFreshnessPolicy(clock=lambda: now).stale_decision_reason(decision) is not None


def test_forced_exit_is_not_subject_to_entry_horizon() -> None:
    now = datetime(2026, 7, 12, 12, 0, tzinfo=UTC)
    decision = DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.CLOSE_LONG,
        confidence=0.5,
        reasoning="hard risk",
        timestamp=now - timedelta(days=1),
    )
    policy = DecisionFreshnessPolicy(
        forced_exit_checker=lambda checked: checked is decision,
        clock=lambda: now,
    )

    assert policy.stale_decision_reason(decision) is None
