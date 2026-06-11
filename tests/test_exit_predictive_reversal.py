from services.exit_predictive_reversal import ExitPredictiveReversalPolicy


def test_exit_predictive_reversal_scores_long_adverse_window() -> None:
    policy = ExitPredictiveReversalPolicy()

    evidence = policy.evidence(
        side="long",
        returns_1=-0.007,
        returns_5=-0.015,
        returns_20=-0.012,
        volume_ratio=1.25,
        rsi_14=72.0,
        bb_pct=0.90,
        macd_diff=-0.1,
        adx_14=12.0,
    )

    assert evidence["score"] >= policy.full_exit_score
    assert evidence["level"] == "full_exit"
    assert "strong_short_window_against" in evidence["reasons"]
    assert "long_overheated_reversal" in evidence["reasons"]


def test_exit_predictive_reversal_rejects_unknown_side() -> None:
    policy = ExitPredictiveReversalPolicy()

    evidence = policy.evidence(
        side="hold",
        returns_1=-0.1,
        returns_5=-0.1,
        returns_20=-0.1,
        volume_ratio=9.0,
        rsi_14=90.0,
        bb_pct=1.0,
        macd_diff=-1.0,
        adx_14=1.0,
    )

    assert evidence == {"score": 0.0, "level": "none", "reasons": []}
