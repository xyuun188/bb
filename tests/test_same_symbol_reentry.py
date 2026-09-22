from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from ai_brain.base_model import Action, DecisionOutput
from services.same_symbol_reentry import (
    ClosedLifecycleFact,
    SameSymbolReentryGuard,
)
from services.trading_policies import EntryPolicy


def _decision(
    now: datetime,
    *,
    selection_reason: str = "strategy_edge_selected",
    score: float = 0.2,
    expected_net: float = 0.3,
    evidence_at: datetime | None = None,
) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="PEPE/USDT",
        action=Action.SHORT,
        confidence=0.8,
        reasoning="reentry test",
        timestamp=now,
        raw_response={
            "normal_paper_trade": {
                "selection_reason": selection_reason,
                "prediction_horizon_minutes": 5.0,
            },
            "opportunity_score": {
                "score": score,
                "expected_net_return_pct": expected_net,
            },
        },
        feature_snapshot={
            "market_timestamp": (evidence_at or now).isoformat(),
        },
    )


def _closed(
    closed_at: datetime,
    *,
    pnl: float,
) -> ClosedLifecycleFact:
    return ClosedLifecycleFact(
        source="okx_position_history",
        source_id="history-1",
        symbol="PEPE/USDT",
        side="short",
        opened_at=closed_at - timedelta(minutes=15),
        closed_at=closed_at,
        realized_net_pnl_usdt=pnl,
        settlement_status="synced",
    )


def test_profitable_symbol_reentry_waits_ten_minutes() -> None:
    closed_at = datetime(2026, 9, 22, 4, 50, tzinfo=UTC)
    now = closed_at + timedelta(minutes=5)

    blocked = SameSymbolReentryGuard.assess(
        _decision(now),
        execution_mode="paper",
        latest_closed_lifecycle=_closed(closed_at, pnl=12.0),
        now=now,
    )
    allowed_at = closed_at + timedelta(minutes=10)
    allowed = SameSymbolReentryGuard.assess(
        _decision(allowed_at),
        execution_mode="paper",
        latest_closed_lifecycle=_closed(closed_at, pnl=12.0),
        now=allowed_at,
    )

    assert blocked.allowed is False
    assert blocked.reason == "same_symbol_reentry_cooldown_active"
    assert blocked.required_cooldown_seconds == 600.0
    assert blocked.remaining_cooldown_seconds == 300.0
    assert allowed.allowed is True


def test_non_profitable_symbol_reentry_waits_twenty_minutes() -> None:
    closed_at = datetime(2026, 9, 22, 4, 6, tzinfo=UTC)
    now = closed_at + timedelta(minutes=12)

    blocked = SameSymbolReentryGuard.assess(
        _decision(now),
        execution_mode="paper",
        latest_closed_lifecycle=_closed(closed_at, pnl=-3.0),
        now=now,
    )

    assert blocked.allowed is False
    assert blocked.required_cooldown_seconds == 1200.0
    assert blocked.remaining_cooldown_seconds == 480.0


def test_reentry_requires_market_evidence_newer_than_the_close() -> None:
    closed_at = datetime(2026, 9, 22, 4, 6, tzinfo=UTC)
    now = closed_at + timedelta(minutes=30)

    result = SameSymbolReentryGuard.assess(
        _decision(now, evidence_at=closed_at - timedelta(seconds=1)),
        execution_mode="paper",
        latest_closed_lifecycle=_closed(closed_at, pnl=4.0),
        now=now,
    )

    assert result.allowed is False
    assert result.reason == "same_symbol_reentry_market_evidence_not_newer_than_close"


def test_reentry_does_not_treat_decision_timestamp_as_market_evidence() -> None:
    closed_at = datetime(2026, 9, 22, 4, 6, tzinfo=UTC)
    now = closed_at + timedelta(minutes=30)
    decision = _decision(now)
    decision.feature_snapshot = {}

    result = SameSymbolReentryGuard.assess(
        decision,
        execution_mode="paper",
        latest_closed_lifecycle=_closed(closed_at, pnl=4.0),
        now=now,
    )

    assert result.allowed is False
    assert result.reason == "same_symbol_reentry_market_evidence_unavailable"
    assert result.decision_evidence_source == "decision.timestamp"


def test_negative_lcb_is_allowed_only_for_explicit_quality_observation() -> None:
    now = datetime(2026, 9, 22, 5, 30, tzinfo=UTC)

    observation = SameSymbolReentryGuard.assess(
        _decision(
            now,
            selection_reason="paper_quality_observation",
            score=-1.7,
            expected_net=0.2,
        ),
        execution_mode="paper",
        latest_closed_lifecycle=None,
        now=now,
    )
    invalid = SameSymbolReentryGuard.assess(
        _decision(
            now,
            selection_reason="strategy_edge_selected",
            score=-1.7,
            expected_net=0.2,
        ),
        execution_mode="paper",
        latest_closed_lifecycle=None,
        now=now,
    )

    assert observation.allowed is True
    assert observation.opportunity_contract_consistent is True
    assert invalid.allowed is False
    assert invalid.reason == "entry_opportunity_contract_inconsistent"


@pytest.mark.asyncio
async def test_entry_execution_policy_exposes_reentry_blocker_and_evidence() -> None:
    class Guard:
        async def evaluate(self, _decision: DecisionOutput, _mode: str) -> SimpleNamespace:
            return SimpleNamespace(
                allowed=False,
                reason="same_symbol_reentry_cooldown_active",
                to_dict=lambda: {"remaining_cooldown_seconds": 420.0},
            )

    now = datetime(2026, 9, 22, 5, 30, tzinfo=UTC)
    result = await EntryPolicy(same_symbol_reentry_guard=Guard()).evaluate(
        _decision(now),
        "ensemble_trader",
        "paper",
        [],
    )

    assert result.passed is False
    assert result.blocker == "same_symbol_reentry_guard"
    assert result.data["skip_kind"] == "same_symbol_reentry_guard"
    assert result.data["same_symbol_reentry_guard"]["remaining_cooldown_seconds"] == 420.0
