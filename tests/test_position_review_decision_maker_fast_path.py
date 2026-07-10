from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from ai_brain.base_model import Action, DecisionOutput
from ai_brain.ensemble_coordinator import EnsembleCoordinator
from data_feed.feature_vector import FeatureVector


def _decision(action: Action, confidence: float = 0.7, raw: dict | None = None) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="CRCL/USDT",
        action=action,
        confidence=confidence,
        reasoning="test",
        raw_response=raw or {},
    )


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
async def test_final_trader_defers_when_market_analysis_budget_is_exhausted() -> None:
    _FakeDecisionMaker.calls = 0
    coordinator = EnsembleCoordinator(_FakeRegistry())
    preliminary = _decision(Action.LONG)

    result = await coordinator._apply_decision_maker(
        FeatureVector(symbol="CRCL/USDT"),
        {
            "_analysis_deadline_monotonic": asyncio.get_running_loop().time(),
            "_analysis_budget_scope": "market_ai",
            "_analysis_budget_seconds": 12.0,
        },
        preliminary,
        {},
        [],
        None,
    )

    assert result is preliminary
    assert _FakeDecisionMaker.calls == 0
    assert result.raw_response["decision_maker"]["status"] == "analysis_budget_deferred"
    assert result.raw_response["decision_maker_timing"]["status"] == "analysis_budget_deferred"


@pytest.mark.asyncio
async def test_final_trader_timeout_preserves_preliminary_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TimeoutDecisionMaker:
        async def decide(self, _features, _context):
            await asyncio.sleep(60)

    class TimeoutRegistry:
        def get(self, _name):
            return TimeoutDecisionMaker()

    async def timeout_wait_for(awaitable, **_kwargs):
        awaitable.close()
        raise TimeoutError()

    monkeypatch.setattr("ai_brain.ensemble_coordinator.asyncio.wait_for", timeout_wait_for)
    coordinator = EnsembleCoordinator(TimeoutRegistry())
    preliminary = _decision(Action.LONG)

    result = await coordinator._apply_decision_maker(
        FeatureVector(symbol="CRCL/USDT"),
        {},
        preliminary,
        {},
        [],
        None,
    )

    assert result is preliminary
    assert result.raw_response["decision_maker"]["status"] == "timeout"
    assert result.raw_response["decision_maker_timing"]["status"] == "timeout"


def test_position_hold_skips_decision_maker_when_exit_evidence_is_insufficient() -> None:
    coordinator = EnsembleCoordinator(SimpleNamespace())
    preliminary = _decision(
        Action.HOLD,
        raw={
            "position_review_policy": {"result": "hold"},
            "close_evidence": {
                "should_close": False,
                "action_plan": "hold",
                "hard_risk": False,
                "profit_protection": False,
                "support_count": 2,
                "strong_support_count": 2,
                "block_reason": "退出证据不足。",
            },
        },
    )
    opinions = {
        "position_expert": _decision(Action.CLOSE_LONG, 0.65),
        "risk_expert": _decision(Action.CLOSE_LONG, 0.70),
    }

    should_call = coordinator._should_call_decision_maker(
        preliminary,
        opinions,
        [],
        {"review_positions": True},
    )

    assert should_call is False
    fast_path = preliminary.raw_response["decision_maker_fast_path"]
    assert fast_path["applied"] is True
    assert fast_path["close_evidence"]["support_count"] == 2


def test_position_hold_keeps_decision_maker_for_protected_exit() -> None:
    coordinator = EnsembleCoordinator(SimpleNamespace())
    preliminary = _decision(
        Action.HOLD,
        raw={
            "position_review_policy": {"result": "hold"},
            "close_evidence": {
                "should_close": True,
                "action_plan": "hold",
                "hard_risk": True,
            },
        },
    )
    opinions = {"risk_expert": _decision(Action.CLOSE_LONG, 0.90)}

    should_call = coordinator._should_call_decision_maker(
        preliminary,
        opinions,
        [],
        {"review_positions": True},
    )

    assert should_call is True
    assert "decision_maker_fast_path" not in preliminary.raw_response


def test_position_hold_keeps_decision_maker_for_add_or_direction_signal() -> None:
    coordinator = EnsembleCoordinator(SimpleNamespace())
    preliminary = _decision(
        Action.HOLD,
        raw={
            "position_review_policy": {"result": "hold"},
            "close_evidence": {"should_close": False, "action_plan": "hold"},
        },
    )
    opinions = {"trend_expert": _decision(Action.LONG, 0.70)}

    should_call = coordinator._should_call_decision_maker(
        preliminary,
        opinions,
        [],
        {"review_positions": True},
    )

    assert should_call is True
    assert "decision_maker_fast_path" not in preliminary.raw_response


def test_entry_with_non_actionable_profit_evidence_skips_slow_decision_maker() -> None:
    coordinator = EnsembleCoordinator(SimpleNamespace())
    preliminary = _decision(
        Action.LONG,
        raw={
            "opportunity_score": {
                "expected_net_return_pct": -0.11,
                "profit_quality_ratio": -0.07,
            },
            "entry_candidate_evidence": {
                "long": {
                    "expected_net_return_pct": -0.10,
                    "profit_quality_ratio": -0.05,
                }
            },
        },
    )
    opinions = {
        "trend_expert": _decision(Action.LONG, 0.70),
        "momentum_expert": _decision(Action.LONG, 0.68),
    }

    should_call = coordinator._should_call_decision_maker(
        preliminary,
        opinions,
        [],
        {},
    )

    assert should_call is False
    fast_path = preliminary.raw_response["decision_maker_fast_path"]
    assert fast_path["mode"] == "pre_screened_non_actionable_entry"
    assert fast_path["blockers"] == [
        "non_positive_expected_net",
        "non_positive_profit_quality",
    ]


@pytest.mark.asyncio
async def test_position_hold_blocks_decision_maker_entry_when_add_evidence_denies() -> None:
    coordinator = EnsembleCoordinator(_FakeRegistry())
    preliminary = _decision(
        Action.HOLD,
        raw={
            "position_review_policy": {"result": "hold"},
            "close_evidence": {"should_close": False, "action_plan": "hold"},
            "add_evidence": {
                "should_add": False,
                "action_plan": "hold",
                "block_reason": "position is losing",
            },
        },
    )
    opinions = {
        "trend_expert": _decision(Action.SHORT, 0.68),
        "momentum_expert": _decision(Action.SHORT, 0.62),
    }

    result = await coordinator._apply_decision_maker(
        FeatureVector(symbol="CRCL/USDT"),
        {"review_positions": True},
        preliminary,
        opinions,
        [],
        None,
    )

    assert result.action == Action.HOLD
    assert result.raw_response["decision_maker"]["applied"] is False
    guard = result.raw_response["decision_maker_position_add_guard"]
    assert guard["applied"] is True
    assert guard["add_evidence"]["should_add"] is False


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

    assert evidence["should_close"] is True
    assert evidence["action_plan"] == "reduce"
    assert evidence["profit_protection"] is True
    assert evidence["small_position_profit_lock"] is True
    assert evidence["meaningful_reduce_lock"] is False
    assert evidence["small_position_profit_lock_fee_multiple"] >= 8.0
    assert "小仓动态锁盈" in evidence["reason"]


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
    assert evidence["small_position_profit_lock"] is False
    assert evidence["winner_run_protected"] is True
    assert "盈利仓继续奔跑" in evidence["block_reason"]
