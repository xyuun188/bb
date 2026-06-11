from ai_brain.base_model import Action, DecisionOutput
from services.exit_invalidation_snapshot import ExitInvalidationSnapshotPolicy


def _decision(snapshot: dict) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.CLOSE_LONG,
        confidence=0.8,
        reasoning="exit",
        raw_response={},
        feature_snapshot=snapshot,
    )


def test_exit_invalidation_snapshot_detects_long_thesis_break() -> None:
    policy = ExitInvalidationSnapshotPolicy(lambda: 1.0)

    result = policy.snapshot(
        _decision(
            {
                "atr_14": 0.5,
                "ema_12": 98.0,
                "ema_26": 100.0,
                "returns_5": -0.01,
                "returns_20": -0.02,
                "volume_ratio": 1.4,
                "price_vs_sma20": -0.01,
                "price_vs_sma50": -0.02,
            }
        ),
        "long",
        100.0,
        98.0,
    )

    assert result["severe"] is True
    assert result["key_break"] is True
    assert result["trend_reversal"] is True
    assert result["momentum_bad"] is True


def test_exit_invalidation_snapshot_detects_short_thesis_break() -> None:
    policy = ExitInvalidationSnapshotPolicy(lambda: 1.0)

    result = policy.snapshot(
        _decision(
            {
                "atr_14": 0.5,
                "ema_12": 102.0,
                "ema_26": 100.0,
                "returns_5": 0.01,
                "returns_20": 0.02,
                "volume_ratio": 1.4,
                "price_vs_sma20": 0.01,
                "price_vs_sma50": 0.02,
            }
        ),
        "short",
        100.0,
        102.0,
    )

    assert result["severe"] is True
    assert result["key_break"] is True
    assert result["trend_reversal"] is True
    assert result["momentum_bad"] is True


def test_exit_invalidation_snapshot_reports_no_severe_invalidation() -> None:
    policy = ExitInvalidationSnapshotPolicy(lambda: 1.0)

    result = policy.snapshot(_decision({"volume_ratio": 0.5}), "long", 100.0, 100.0)

    assert result["severe"] is False
    assert result["reason"] == "no severe invalidation"
