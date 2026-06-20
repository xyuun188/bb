from __future__ import annotations

from services.training_data_quality import (
    DATA_QUALITY_VERSION,
    annotate_training_payload,
    assess_shadow_sample,
    assess_trade_sample,
    governance_report,
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
        text_sentiment_samples=[
            {
                "source": "news",
                "platform": "scrapling:ethereum_blog",
                "text": "Ethereum upgrade launch with ecosystem impact",
                "sentiment_score": 0.2,
            }
        ],
    )

    assert len(payload["shadow_samples"]) == 1
    assert len(payload["trade_samples"]) == 1
    report = payload["quality_report"]
    assert report["data_quality_version"] == DATA_QUALITY_VERSION
    assert report["totals"]["excluded"] == 2
    assert report["totals"]["effective_weight"] > 0
    assert report["totals"]["effective_weight_ratio"] > 0
    assert report["by_kind"]["shadow"]["total"] == 2
    assert report["by_kind"]["trade"]["total"] == 2
    assert report["by_kind"]["shadow"]["actions"]["long"] == 2
    assert report["by_kind"]["shadow"]["trainable_actions"]["long"] == 1
    assert report["by_kind"]["trade"]["sources"]["closed_position"] == 1
    assert report["by_kind"]["trade"]["trainable_sources"]["closed_position"] == 1
    assert report["by_kind"]["text_sentiment"]["sources"]["scrapling:ethereum_blog"] == 1
    assert report["by_kind"]["text_sentiment"]["trainable_sources"]["scrapling:ethereum_blog"] == 1
    assert report["policy"]["hold_observation_penalty"] == 0.55
    governance = payload["governance_report"]
    assert governance["cleanup_mode"] == "quarantine_not_delete"
    assert governance["raw_records_preserved"] is True
    assert governance["quarantine_applied"] is True
    assert governance["requires_artifact_refresh"] is True
    assert "local_ai_tools" in governance["refresh_targets"]


def test_shadow_market_data_quality_issue_is_excluded_from_training() -> None:
    assessment = assess_shadow_sample(
        _shadow_sample(
            features={
                "symbol": "BTC/USDT",
                "current_price": 100.0,
                "spread_pct": 0.03,
                "market_data_quality": {
                    "code": "price_source_split",
                    "exclude_from_training": True,
                    "training_quality_reason": "market_data_quality:price_source_split",
                },
            }
        )
    )

    assert assessment.status == "excluded"
    assert assessment.weight == 0.0
    assert "market_data_quality:price_source_split" in assessment.reasons


def test_training_governance_report_preserves_raw_records_and_targets_refresh() -> None:
    report = governance_report(
        {
            "data_quality_version": DATA_QUALITY_VERSION,
            "totals": {
                "total": 10,
                "included": 6,
                "downweighted": 3,
                "excluded": 1,
                "effective_weight_ratio": 0.72,
            },
            "top_reasons": [{"reason": "trade:manual_or_test_trade", "count": 1}],
        }
    )

    assert report["status"] == "quarantined"
    assert report["raw_records_preserved"] is True
    assert report["excluded_sample_count"] == 1
    assert report["downweighted_sample_count"] == 3
    assert report["contamination_risk"] == "high"
    assert "vector_memory_reindex" in report["refresh_targets"]
