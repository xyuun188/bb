from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from ai_brain.base_model import Action, DecisionOutput
from ai_brain.ensemble_coordinator import EnsembleCoordinator
from data_feed.feature_vector import FeatureVector


class _FakeDecisionMaker:
    calls = 0

    async def decide(self, _features, _context):
        type(self).calls += 1
        return DecisionOutput(
            model_name="decision_maker",
            symbol="CRCL/USDT",
            action=Action.SHORT,
            confidence=0.7,
            reasoning="same-side add probe",
            position_size_pct=0.02,
            suggested_leverage=5.0,
        )


class _FakeRegistry:
    def get(self, _name):
        return _FakeDecisionMaker()


@pytest.mark.asyncio


@pytest.mark.asyncio










@pytest.mark.asyncio


def test_position_close_evidence_locks_small_profitable_weak_continuation_position() -> None:
    coordinator = EnsembleCoordinator(SimpleNamespace())

    evidence = coordinator._position_close_evidence(
        current_side="short",
        close_action=Action.CLOSE_SHORT,
        exit_votes=[],
        risk_vetoes=[],
        score=0.0,
        raw_opinions=[],
        symbol_positions=[
            {
                "side": "short",
                "entry_price": 0.1713667,
                "current_price": 0.1599,
                "quantity": 150.0,
                "unrealized_pnl": 1.72,
                "stop_loss": 0.1911,
                "created_at": datetime.now(UTC) - timedelta(hours=12),
            }
        ],
        features=FeatureVector(
            symbol="MET/USDT",
            current_price=0.1599,
            returns_1=0.0,
            returns_5=0.0,
            volume_ratio=0.42,
            bb_pct=0.50,
            rsi_14=50.0,
        ),
        context={},
    )

    assert evidence["should_close"] is False
    assert evidence["dynamic_exit_policy"]["profit_retrace_ratio"] == 0.0
    provenance = evidence["dynamic_exit_policy"]["policy_provenance"]
    assert provenance["source"]
    assert provenance["observation_window"]
    assert provenance["sample_count"] >= 1
    assert provenance["generated_at"]
    assert provenance["strategy_version"]
    assert "fallback_reason" in provenance


def test_position_close_evidence_keeps_strong_small_winner_running() -> None:
    coordinator = EnsembleCoordinator(SimpleNamespace())

    evidence = coordinator._position_close_evidence(
        current_side="short",
        close_action=Action.CLOSE_SHORT,
        exit_votes=[],
        risk_vetoes=[],
        score=-0.35,
        raw_opinions=[],
        symbol_positions=[
            {
                "side": "short",
                "entry_price": 0.1713667,
                "current_price": 0.1599,
                "quantity": 150.0,
                "unrealized_pnl": 1.72,
                "stop_loss": 0.1911,
                "created_at": datetime.now(UTC) - timedelta(hours=12),
            }
        ],
        features=FeatureVector(
            symbol="MET/USDT",
            current_price=0.1599,
            returns_1=-0.002,
            returns_5=-0.003,
            volume_ratio=1.20,
            bb_pct=0.50,
            rsi_14=48.0,
        ),
        context={},
    )

    assert evidence["should_close"] is False
    assert evidence["dynamic_exit_policy"]["close_fraction"] == 0.0
    assert evidence["dynamic_exit_policy"]["policy_provenance"]["strategy_version"]


def test_position_close_evidence_prioritizes_crossed_planned_stop_over_hold() -> None:
    coordinator = EnsembleCoordinator(SimpleNamespace())

    evidence = coordinator._position_close_evidence(
        current_side="long",
        close_action=Action.CLOSE_LONG,
        exit_votes=[],
        risk_vetoes=[],
        score=0.8,
        raw_opinions=[
            {"model_name": "position_expert", "action": "hold", "confidence": 0.95}
        ],
        symbol_positions=[
            {
                "side": "long",
                "entry_price": 100.0,
                "current_price": 93.0,
                "quantity": 1.0,
                "unrealized_pnl": -7.0,
                "stop_loss": 94.0,
                "created_at": datetime.now(UTC) - timedelta(hours=2),
            }
        ],
        features=FeatureVector(symbol="INJ/USDT", current_price=93.0),
        context={},
    )

    assert evidence["planned_stop_crossed"] is True
    assert evidence["should_close"] is True
    assert evidence["action_plan"] == "full_close"
    assert evidence["position_size_pct"] == 1.0


def test_position_close_evidence_executes_dynamic_loss_reduction() -> None:
    coordinator = EnsembleCoordinator(SimpleNamespace())

    evidence = coordinator._position_close_evidence(
        current_side="long",
        close_action=Action.CLOSE_LONG,
        exit_votes=[],
        risk_vetoes=[],
        score=0.0,
        raw_opinions=[],
        symbol_positions=[
            {
                "side": "long",
                "entry_price": 100.0,
                "current_price": 99.2,
                "quantity": 10.0,
                "unrealized_pnl": -8.0,
                "entry_fee_usdt": 0.05,
                "stop_loss": 0.0,
                "created_at": datetime.now(UTC) - timedelta(hours=2),
            }
        ],
        features=FeatureVector(symbol="YB/USDT", current_price=99.2),
        context={},
    )

    assert evidence["position_loss"] is True
    assert evidence["should_close"] is True
    assert evidence["action_plan"] == "reduce"
    assert evidence["position_size_pct"] == pytest.approx(
        evidence["dynamic_loss_reduce_fraction"]
    )
    assert 0.0 < evidence["position_size_pct"] < 1.0
