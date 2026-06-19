from __future__ import annotations

from services.training_data_quality import (
    DATA_QUALITY_VERSION,
    annotate_training_payload,
    assess_shadow_sample,
    assess_trade_sample,
)


def _shadow_sample(**overrides):
    sample = {
        "symbol": "BTC/USDT",
        "decision_action": "long",
        "decision_confidence": 0.72,
        "horizon_minutes": 30,
        "features": {
            "symbol": "BTC/USDT",
            "current_price": 100.0,
            "spread_pct": 0.03,
        },
        "long_return_pct": 0.42,
        "short_return_pct": -0.31,
        "best_action": "long",
        "missed_opportunity": False,
    }
    sample.update(overrides)
    return sample


def _trade_sample(**overrides):
    sample = {
        "source": "closed_position",
        "symbol": "BTC/USDT",
        "side": "long",
        "entry_price": 100.0,
        "exit_price": 101.0,
        "quantity": 0.1,
        "realized_pnl": 1.0,
        "fee_estimate": 0.05,
        "hold_minutes": 35.0,
        "outcome": "profit",
    }
    sample.update(overrides)
    return sample


def test_shadow_hold_samples_are_downweighted_not_deleted() -> None:
    assessment = assess_shadow_sample(_shadow_sample(decision_action="hold"))

    assert assessment.status == "downweighted"
    assert assessment.exclude_from_training is False
    assert assessment.weight < 1.0
    assert "hold_observation_downweighted" in assessment.reasons


def test_shadow_missing_features_are_excluded() -> None:
    assessment = assess_shadow_sample(_shadow_sample(features={}))

    assert assessment.status == "excluded"
    assert assessment.weight == 0.0
    assert "missing_features" in assessment.reasons


def test_trade_fast_loss_exit_is_downweighted_for_review() -> None:
    assessment = assess_trade_sample(_trade_sample(realized_pnl=-0.8, hold_minutes=1.5))

    assert assessment.status == "downweighted"
    assert assessment.exclude_from_training is False
    assert "fast_loss_exit_requires_review" in assessment.reasons


def test_manual_trade_samples_are_excluded() -> None:
    assessment = assess_trade_sample(_trade_sample(source="manual"))

    assert assessment.status == "excluded"
    assert "manual_or_test_trade" in assessment.reasons


def test_training_payload_returns_trainable_samples_and_quality_report() -> None:
    payload = annotate_training_payload(
        shadow_samples=[_shadow_sample(), _shadow_sample(features={})],
        trade_samples=[_trade_sample(), _trade_sample(source="manual")],
        sequence_samples=[],
        text_sentiment_samples=[],
    )

    assert len(payload["shadow_samples"]) == 1
    assert len(payload["trade_samples"]) == 1
    report = payload["quality_report"]
    assert report["data_quality_version"] == DATA_QUALITY_VERSION
    assert report["totals"]["excluded"] == 2
    assert report["by_kind"]["shadow"]["total"] == 2
    assert report["by_kind"]["trade"]["total"] == 2
