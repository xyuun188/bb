import inspect

from ai_brain.base_model import Action, DecisionOutput
from ai_brain.ensemble_coordinator import EnsembleCoordinator
from ai_brain.llm_agent import _apply_aggressive_hold_policy


def test_aggressive_hold_rewrite_does_not_raise_requested_leverage() -> None:
    decision = DecisionOutput(
        model_name="test",
        symbol="BTC/USDT",
        action=Action.HOLD,
        confidence=0.4,
        reasoning="test",
        suggested_leverage=2.0,
        feature_snapshot={
            "price_vs_sma20": 0.02,
            "price_vs_sma50": 0.03,
            "macd_diff": 1.0,
            "ema_12": 102.0,
            "ema_26": 100.0,
            "returns_5": 0.02,
            "returns_20": 0.04,
            "adx_14": 30.0,
            "volume_ratio": 2.0,
        },
    )

    _apply_aggressive_hold_policy(decision, [])

    assert decision.action == Action.LONG
    assert decision.suggested_leverage == 2.0


def test_entry_coordinator_has_no_confidence_based_leverage_floor() -> None:
    coordinator_source = inspect.getsource(EnsembleCoordinator)

    assert "_entry_min_leverage" not in coordinator_source
    assert "_entry_leverage_cap" not in coordinator_source
    assert "最低杠杆" not in coordinator_source
    assert "5-10x" not in coordinator_source
