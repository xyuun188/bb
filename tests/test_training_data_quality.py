from __future__ import annotations

from datetime import UTC, datetime

from services.training_data_quality import (
    DATA_QUALITY_VERSION,
    annotate_training_payload,
    assess_sequence_sample,
    assess_shadow_sample,
    assess_text_sentiment_sample,
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
        "position_size_pct": 0.03,
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


def test_shadow_mojibake_payload_is_excluded() -> None:
    assessment = assess_shadow_sample(
        _shadow_sample(
            features={"current_price": 100.0, "spread_pct": 0.01, "note": "鏈轰細璇勫垎"}
        )
    )

    assert assessment.status == "excluded"
    assert assessment.weight == 0.0
    assert "mojibake_text" in assessment.reasons


def test_duplicate_training_sample_is_excluded() -> None:
    assessment = assess_shadow_sample(_shadow_sample(duplicate_count=2))

    assert assessment.status == "excluded"
    assert "duplicate_sample" in assessment.reasons


def test_trade_fast_loss_exit_is_downweighted_for_review() -> None:
    assessment = assess_trade_sample(_trade_sample(realized_pnl=-0.8, hold_minutes=1.5))

    assert assessment.status == "downweighted"
    assert assessment.exclude_from_training is False
    assert "fast_loss_exit_requires_review" in assessment.reasons


def test_manual_trade_samples_are_excluded() -> None:
    assessment = assess_trade_sample(_trade_sample(source="manual"))

    assert assessment.status == "excluded"
    assert "manual_or_test_trade" in assessment.reasons


def test_trade_missing_fee_and_micro_probe_are_excluded() -> None:
    no_fee = assess_trade_sample(_trade_sample(fee_estimate=None))
    micro_probe = assess_trade_sample(
        _trade_sample(position_size_pct=0.0003, evidence_tier="weak_conflict_probe")
    )

    assert no_fee.status == "excluded"
    assert "missing_fee_estimate" in no_fee.reasons
    assert micro_probe.status == "excluded"
    assert "weak_evidence_micro_probe" in micro_probe.reasons


def test_trade_mode_mixing_and_failed_close_are_excluded() -> None:
    mode_mixed = assess_trade_sample(_trade_sample(execution_mode="shadow"))
    bad_close = assess_trade_sample(_trade_sample(close_status="failed"))

    assert mode_mixed.status == "excluded"
    assert "execution_mode_mismatch" in mode_mixed.reasons
    assert bad_close.status == "excluded"
    assert "failed_close_status" in bad_close.reasons


def test_sequence_future_leakage_is_excluded() -> None:
    assessment = assess_sequence_sample(
        {
            "close_sequence": [100 + idx for idx in range(30)],
            "future_return_pct": 0.3,
            "timeframe": "1m",
            "feature_timestamp": datetime(2026, 6, 23, 1, 10, tzinfo=UTC),
            "label_timestamp": datetime(2026, 6, 23, 1, 0, tzinfo=UTC),
        }
    )

    assert assessment.status == "excluded"
    assert "future_leakage" in assessment.reasons


def test_shadow_feature_snapshot_future_leakage_is_excluded() -> None:
    assessment = assess_shadow_sample(
        _shadow_sample(
            features={
                "current_price": 100.0,
                "spread_pct": 0.01,
                "feature_timestamp": datetime(2026, 6, 23, 1, 10, tzinfo=UTC).isoformat(),
            },
            label_timestamp=datetime(2026, 6, 23, 1, 0, tzinfo=UTC).isoformat(),
        )
    )

    assert assessment.status == "excluded"
    assert "future_leakage" in assessment.reasons


def test_text_sentiment_mojibake_and_duplicate_samples_are_excluded() -> None:
    mojibake = assess_text_sentiment_sample(
        {
            "source": "news",
            "platform": "scrapling:okx",
            "text": "閺堣桨绱扮拠鍕瀻 market event summary",
            "sentiment_score": 0.2,
        }
    )
    duplicate = assess_text_sentiment_sample(
        {
            "source": "news",
            "platform": "scrapling:okx",
            "text": "OKX listing update with enough text for training",
            "sentiment_score": 0.2,
            "duplicate_of": "news:1",
        }
    )

    assert mojibake.status == "excluded"
    assert "mojibake_text" in mojibake.reasons
    assert duplicate.status == "excluded"
    assert "duplicate_sample" in duplicate.reasons


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


def test_shadow_price_reconciliation_warning_is_excluded_from_training() -> None:
    assessment = assess_shadow_sample(
        _shadow_sample(
            features={
                "symbol": "PROS/USDT",
                "current_price": 0.5666,
                "close": 0.5666,
                "indicator_close_price": 0.3902,
                "indicator_price_gap_pct": 45.18,
                "price_reconciliation_warning": (
                    "ticker_current_price_kept_indicator_close_diverged"
                ),
                "spread_pct": 0.03,
            }
        )
    )

    assert assessment.status == "excluded"
    assert assessment.weight == 0.0
    assert (
        "price_reconciliation:ticker_current_price_kept_indicator_close_diverged"
        in assessment.reasons
    )


def test_shadow_price_outside_24h_range_is_excluded_from_training() -> None:
    assessment = assess_shadow_sample(
        _shadow_sample(
            features={
                "symbol": "PROS/USDT",
                "current_price": 0.3902,
                "low_24h": 0.5491,
                "high_24h": 0.5707,
                "spread_pct": 0.03,
            }
        )
    )

    assert assessment.status == "excluded"
    assert assessment.weight == 0.0
    assert "price_outside_24h_range" in assessment.reasons


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
