from datetime import UTC, datetime, timedelta

from ai_brain.base_model import Action, DecisionOutput
from services.decision_freshness import DecisionFreshnessPolicy
from services.paper_training import build_paper_training_contract


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


def test_paper_canary_freshness_uses_version_bound_prediction_horizon() -> None:
    now = datetime(2026, 7, 17, 10, 0, tzinfo=UTC)
    decision = _decision(
        now=now,
        generated_at=datetime(2026, 7, 17, 8, 0, tzinfo=UTC),
        valid_for_seconds=0,
    )
    decision.raw_response["paper_bootstrap_canary"] = {
        "authorized": True,
        "requested": True,
        "execution_scope": "paper_only",
        "production_permission": False,
        "generated_at": "2026-07-17T09:56:00+00:00",
        "selected_observation": {"horizon_minutes": 10},
        "policy_provenance": {"generated_at": "2026-07-17T09:56:00+00:00"},
    }
    policy = DecisionFreshnessPolicy(clock=lambda: now)

    assert policy.max_age_seconds(decision) == 600.0
    assert policy.decision_reference_time(decision) == datetime(
        2026, 7, 17, 9, 56, tzinfo=UTC
    )
    assert policy.stale_decision_reason(decision) is None

    decision.raw_response["paper_bootstrap_canary"]["policy_provenance"][
        "generated_at"
    ] = "2026-07-17T09:49:00+00:00"
    assert policy.stale_decision_reason(decision) is not None


def test_paper_training_freshness_uses_observed_model_horizon() -> None:
    now = datetime(2026, 7, 22, 10, 0, tzinfo=UTC)
    decision = _decision(
        now=now,
        generated_at=now - timedelta(days=1),
        valid_for_seconds=0,
    )
    decision.raw_response["paper_training"] = build_paper_training_contract(
        symbol=decision.symbol,
        selected_side="long",
        signal_source="direction_competition_observation",
        horizon_minutes=10.0,
    )
    policy = DecisionFreshnessPolicy(clock=lambda: now)
    generated_at = policy.decision_reference_time(decision)
    policy.clock = lambda: generated_at + timedelta(seconds=599)

    assert policy.max_age_seconds(decision) == 600.0
    assert policy.stale_decision_reason(decision) is None

    policy.clock = lambda: generated_at + timedelta(seconds=601)
    assert policy.stale_decision_reason(decision) is not None


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
