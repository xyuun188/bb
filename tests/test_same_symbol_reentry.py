from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from ai_brain.base_model import Action, DecisionOutput
from services.normal_paper_trade import (
    build_normal_paper_trade_contract,
    normal_paper_entry_contract_reasons,
    reconcile_normal_paper_trade_contract,
)
from services.same_symbol_reentry import (
    ClosedLifecycleFact,
    SameSymbolReentryGuard,
)
from services.trading_policies import EntryPolicy
from tests.normal_paper_test_fixtures import paper_quality_permissions


def _decision(
    now: datetime,
    *,
    selection_reason: str = "strategy_edge_selected",
    score: float = 0.2,
    expected_net: float | None = 0.3,
    downside_return: float | None = -0.2,
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
                "return_lcb_pct": downside_return,
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


def _validated_paper_decision(
    now: datetime,
    *,
    selection_reason: str = "strategy_edge_selected",
    horizon_minutes: float = 240.0,
    evidence_at: datetime | None = None,
    reconciled: bool = True,
) -> DecisionOutput:
    training = selection_reason == "paper_training_entry"
    contract = build_normal_paper_trade_contract(
        symbol="PEPE/USDT",
        side="short",
        selection_reason=selection_reason,
        direction_support={
            "eligible": True,
            "selected_side": "short",
            "prediction_horizon_minutes": horizon_minutes,
            "expected_net_return_pct": 0.3,
            "raw_expected_return_pct": 0.4,
            "objective_net_return_pct": 0.1,
            "loss_probability": 0.4,
            "quant_evidence_families": ["local_ml"],
            "quant_quality_permissions": paper_quality_permissions(),
            "strong_expert_opposition": False,
            "current_edge_validated": True,
            "paper_training_only": training,
            "paper_training_reasons": ["bounded_training_candidate"] if training else [],
            "aligned_expert_count": 2 if training else 0,
        },
    )
    if reconciled:
        contract = reconcile_normal_paper_trade_contract(
            contract,
            {
                "expected_net_return_pct": 0.3,
                "return_lcb_pct": 0.1,
                "loss_probability": 0.4,
                "return_distribution_contract": {
                    "objective_expected_return_pct": 0.2,
                    "lower_quantile_return_pct": 0.1,
                    "gross_market_distribution": {"raw_expected_return_pct": 0.4},
                },
            },
        )
    assert normal_paper_entry_contract_reasons(contract) == []
    decision = _decision(
        now,
        selection_reason=selection_reason,
        expected_net=0.3,
        downside_return=0.1,
        evidence_at=evidence_at,
    )
    decision.raw_response["normal_paper_trade"] = contract
    return decision


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


def test_validated_paper_edge_can_reenter_after_loss_floor_before_horizon() -> None:
    closed_at = datetime(2026, 9, 22, 4, 6, tzinfo=UTC)
    now = closed_at + timedelta(minutes=30)
    result = SameSymbolReentryGuard.assess(
        _validated_paper_decision(now),
        execution_mode="paper",
        latest_closed_lifecycle=_closed(closed_at, pnl=-3.0),
        now=now,
    )

    assert result.allowed is True
    assert result.reason == "same_symbol_reentry_cooldown_elapsed"
    assert result.required_cooldown_seconds == 1200.0
    assert result.remaining_cooldown_seconds == 0.0
    assert result.policy_provenance["prediction_horizon_cooldown_bypassed"] is True
    assert result.policy_provenance["cooldown_basis"] == (
        "paper_validated_edge_outcome_floor"
    )


def test_validated_paper_edge_still_waits_loss_floor() -> None:
    closed_at = datetime(2026, 9, 22, 4, 6, tzinfo=UTC)
    now = closed_at + timedelta(minutes=15)
    result = SameSymbolReentryGuard.assess(
        _validated_paper_decision(now),
        execution_mode="paper",
        latest_closed_lifecycle=_closed(closed_at, pnl=-3.0),
        now=now,
    )

    assert result.allowed is False
    assert result.reason == "same_symbol_reentry_cooldown_active"
    assert result.required_cooldown_seconds == 1200.0
    assert result.remaining_cooldown_seconds == 300.0


def test_unvalidated_paper_candidate_keeps_prediction_horizon_cooldown() -> None:
    closed_at = datetime(2026, 9, 22, 4, 6, tzinfo=UTC)
    now = closed_at + timedelta(minutes=30)
    decision = _decision(now)
    decision.raw_response["normal_paper_trade"]["prediction_horizon_minutes"] = 240.0
    result = SameSymbolReentryGuard.assess(
        decision,
        execution_mode="paper",
        latest_closed_lifecycle=_closed(closed_at, pnl=-3.0),
        now=now,
    )

    assert result.allowed is False
    assert result.reason == "same_symbol_reentry_cooldown_active"
    assert result.required_cooldown_seconds == 14_400.0
    assert result.policy_provenance["prediction_horizon_cooldown_bypassed"] is False


def test_unreconciled_invalid_or_mismatched_paper_contract_cannot_bypass_horizon() -> None:
    closed_at = datetime(2026, 9, 22, 4, 6, tzinfo=UTC)
    now = closed_at + timedelta(minutes=30)
    invalid_contract = _validated_paper_decision(now)
    invalid_contract.raw_response["normal_paper_trade"][
        "objective_net_return_pct"
    ] = -0.1
    wrong_side = _validated_paper_decision(now)
    wrong_side.action = Action.LONG
    unreconciled = _validated_paper_decision(now, reconciled=False)

    for decision in (invalid_contract, wrong_side, unreconciled):
        result = SameSymbolReentryGuard.assess(
            decision,
            execution_mode="paper",
            latest_closed_lifecycle=_closed(closed_at, pnl=-3.0),
            now=now,
        )
        assert result.allowed is False
        assert result.reason == "same_symbol_reentry_cooldown_active"
        assert result.required_cooldown_seconds == 14_400.0
        assert result.policy_provenance["prediction_horizon_cooldown_bypassed"] is False


def test_training_and_live_candidates_keep_prediction_horizon_cooldown() -> None:
    closed_at = datetime(2026, 9, 22, 4, 6, tzinfo=UTC)
    now = closed_at + timedelta(minutes=30)

    training = SameSymbolReentryGuard.assess(
        _validated_paper_decision(now, selection_reason="paper_training_entry"),
        execution_mode="paper",
        latest_closed_lifecycle=_closed(closed_at, pnl=-3.0),
        now=now,
    )
    live = SameSymbolReentryGuard.assess(
        _validated_paper_decision(now),
        execution_mode="live",
        latest_closed_lifecycle=_closed(closed_at, pnl=-3.0),
        now=now,
    )

    assert training.allowed is False
    assert training.required_cooldown_seconds == 14_400.0
    assert training.policy_provenance["prediction_horizon_cooldown_bypassed"] is False
    assert live.allowed is False
    assert live.required_cooldown_seconds == 14_400.0
    assert live.policy_provenance["prediction_horizon_cooldown_bypassed"] is False


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


def test_validated_paper_edge_still_requires_fresh_market_evidence() -> None:
    closed_at = datetime(2026, 9, 22, 4, 6, tzinfo=UTC)
    now = closed_at + timedelta(minutes=30)
    result = SameSymbolReentryGuard.assess(
        _validated_paper_decision(
            now,
            evidence_at=closed_at - timedelta(seconds=1),
        ),
        execution_mode="paper",
        latest_closed_lifecycle=_closed(closed_at, pnl=-3.0),
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


def test_quality_observation_selection_never_authorizes_reentry() -> None:
    now = datetime(2026, 9, 22, 5, 30, tzinfo=UTC)

    observation = SameSymbolReentryGuard.assess(
        _decision(
            now,
            selection_reason="paper_quality_observation",
            score=-1.7,
            expected_net=0.2,
            evidence_at=now,
        ),
        execution_mode="paper",
        latest_closed_lifecycle=None,
        now=now,
    )
    assert observation.allowed is False
    assert observation.opportunity_contract_consistent is False
    assert observation.reason == "entry_opportunity_contract_inconsistent"


def test_training_selection_uses_current_contract_for_reentry() -> None:
    now = datetime(2026, 9, 22, 5, 30, tzinfo=UTC)

    training = SameSymbolReentryGuard.assess(
        _decision(
            now,
            selection_reason="paper_training_entry",
            score=-1.7,
            expected_net=0.2,
            evidence_at=now,
        ),
        execution_mode="paper",
        latest_closed_lifecycle=None,
        now=now,
    )

    assert training.allowed is True
    assert training.opportunity_contract_consistent is True
    assert training.reason == "no_recent_closed_symbol_lifecycle"


def test_training_selection_rejects_non_positive_current_contract() -> None:
    now = datetime(2026, 9, 22, 5, 30, tzinfo=UTC)

    decision = _decision(
        now,
        selection_reason="paper_training_entry",
        score=0.2,
        expected_net=-0.01,
        downside_return=-0.2,
        evidence_at=now,
    )
    decision.raw_response["normal_paper_trade"].update(
        {
            "expected_net_return_pct": -0.01,
            "current_raw_expected_return_pct": -0.01,
            "objective_net_return_pct": -0.2,
        }
    )
    blocked = SameSymbolReentryGuard.assess(
        decision,
        execution_mode="paper",
        latest_closed_lifecycle=None,
        now=now,
    )

    assert blocked.allowed is False
    assert blocked.opportunity_contract_consistent is False
    assert blocked.reason == "entry_opportunity_contract_inconsistent"


def test_paper_downside_lcb_does_not_block_positive_current_return_reentry() -> None:
    now = datetime(2026, 9, 22, 5, 30, tzinfo=UTC)
    current = _decision(now, score=-1.7, expected_net=0.2, evidence_at=now)
    current.raw_response["opportunity_score"]["return_lcb_pct"] = -0.4
    allowed = SameSymbolReentryGuard.assess(
        current,
        execution_mode="paper",
        latest_closed_lifecycle=None,
        now=now,
    )
    assert allowed.allowed is True
    assert allowed.opportunity_contract_consistent is True


@pytest.mark.parametrize(
    ("expected_net", "downside_return"),
    [
        (0.0, -0.4),
        (-0.2, -0.4),
        (None, -0.4),
        (float("nan"), -0.4),
        (float("inf"), -0.4),
        (0.2, None),
        (0.2, float("nan")),
        (0.2, float("inf")),
        (0.2, -float("inf")),
    ],
)
def test_paper_reentry_requires_positive_current_mean_and_finite_downside(
    expected_net: float | None,
    downside_return: float | None,
) -> None:
    now = datetime(2026, 9, 22, 5, 30, tzinfo=UTC)

    result = SameSymbolReentryGuard.assess(
        _decision(
            now,
            expected_net=expected_net,
            downside_return=downside_return,
        ),
        execution_mode="paper",
        latest_closed_lifecycle=None,
        now=now,
    )

    assert result.allowed is False
    assert result.reason == "entry_opportunity_contract_inconsistent"
    assert result.opportunity_contract_consistent is False


@pytest.mark.parametrize(
    ("elapsed_minutes", "evidence_offset_seconds", "allowed", "reason"),
    [
        (5, 60, False, "same_symbol_reentry_cooldown_active"),
        (30, -1, False, "same_symbol_reentry_market_evidence_not_newer_than_close"),
        (30, 60, True, "same_symbol_reentry_cooldown_elapsed"),
    ],
)
def test_paper_negative_downside_keeps_cooldown_and_fresh_evidence_guards(
    elapsed_minutes: int,
    evidence_offset_seconds: int,
    allowed: bool,
    reason: str,
) -> None:
    closed_at = datetime(2026, 9, 22, 5, 0, tzinfo=UTC)
    now = closed_at + timedelta(minutes=elapsed_minutes)
    result = SameSymbolReentryGuard.assess(
        _decision(
            now,
            score=-1.7,
            expected_net=0.2,
            downside_return=-0.4,
            evidence_at=closed_at + timedelta(seconds=evidence_offset_seconds),
        ),
        execution_mode="paper",
        latest_closed_lifecycle=_closed(closed_at, pnl=-3.0),
        now=now,
    )

    assert result.allowed is allowed
    assert result.reason == reason
    assert result.opportunity_contract_consistent is True


def test_negative_opportunity_score_is_paper_only() -> None:
    now = datetime(2026, 9, 22, 5, 30, tzinfo=UTC)
    paper = SameSymbolReentryGuard.assess(
        _decision(now, score=-1.7, expected_net=0.2, downside_return=-0.4),
        execution_mode="paper",
        latest_closed_lifecycle=None,
        now=now,
    )
    live = SameSymbolReentryGuard.assess(
        _decision(now, score=-1.7, expected_net=0.2, downside_return=-0.4),
        execution_mode="live",
        latest_closed_lifecycle=None,
        now=now,
    )

    assert paper.allowed is True
    assert paper.opportunity_contract_consistent is True
    assert live.allowed is False
    assert live.reason == "entry_opportunity_contract_inconsistent"
    assert live.opportunity_contract_consistent is False


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
