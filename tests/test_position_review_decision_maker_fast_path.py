from __future__ import annotations

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
    async def decide(self, _features, _context):
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
