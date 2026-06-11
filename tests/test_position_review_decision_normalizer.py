from __future__ import annotations

from ai_brain.base_model import Action, DecisionOutput
from services.position_review_decision_normalizer import PositionReviewDecisionNormalizer


def _decision(action: Action) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=action,
        confidence=0.5,
        reasoning="原始理由",
        position_size_pct=0.04,
        suggested_leverage=4.0,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        raw_response={"source": "test"},
        feature_snapshot={"current_price": 100.0},
    )


def _normalizer() -> PositionReviewDecisionNormalizer:
    return PositionReviewDecisionNormalizer(
        lambda symbol: str(symbol or "").upper().replace("-SWAP", "")
    )


def test_position_review_decision_normalizer_keeps_non_entry_decisions() -> None:
    decision = _decision(Action.HOLD)

    assert _normalizer().normalize(decision, [{"symbol": "BTC/USDT", "side": "long"}]) is decision


def test_position_review_decision_normalizer_keeps_same_side_entry_as_add_candidate() -> None:
    decision = _decision(Action.LONG)

    normalized = _normalizer().normalize(
        decision,
        [{"symbol": "BTC/USDT", "side": "long"}],
    )

    assert normalized is decision
    assert "同方向信号" in normalized.reasoning


def test_position_review_decision_normalizer_turns_opposite_long_into_close_short() -> None:
    decision = _decision(Action.LONG)

    normalized = _normalizer().normalize(
        decision,
        [{"symbol": "BTC/USDT-SWAP", "side": "short"}],
    )

    assert normalized is not decision
    assert normalized.action == Action.CLOSE_SHORT
    assert normalized.confidence == 0.62
    assert normalized.position_size_pct == 1.0
    assert normalized.suggested_leverage == 1.0
    assert normalized.raw_response == {"source": "test"}
    assert "先平掉现有仓位" in normalized.reasoning


def test_position_review_decision_normalizer_turns_opposite_short_into_close_long() -> None:
    decision = _decision(Action.SHORT)
    decision.confidence = 0.7

    normalized = _normalizer().normalize(
        decision,
        [{"symbol": "BTC/USDT", "side": "long"}],
    )

    assert normalized.action == Action.CLOSE_LONG
    assert normalized.confidence == 0.7
