"""Deterministic observed-input clocks for otherwise-valid synthetic samples."""

from datetime import datetime, timedelta

from core.feature_availability import COMPONENT_AVAILABILITY_VERSION, seal_feature_availability


def seal_synthetic_feature_availability(features: dict, *, decision_at: datetime) -> None:
    source_at = (decision_at - timedelta(seconds=9)).isoformat()
    receipt_at = (decision_at - timedelta(seconds=8)).isoformat()
    features.update({
        "original_decision_timestamp": decision_at.isoformat(),
        "technical_availability": {
            "version": COMPONENT_AVAILABILITY_VERSION,
            "available_at": receipt_at,
            "timeframes": "1m",
            "source_delivery_observed": True,
            "1m_closed_at": source_at,
            "1m_received_at": receipt_at,
            "input_fingerprint": "synthetic-technical-inputs",
        },
        "sentiment_availability": {
            "version": COMPONENT_AVAILABILITY_VERSION,
            "available_at": receipt_at,
            "news_input_count": 0,
            "news_clock_errors": 0,
            "social_input_count": 0,
            "social_clock_errors": 0,
        },
        "funding_rate_observed_at": source_at,
        "funding_rate_received_at": receipt_at,
        "account_fee_source_observed_at": source_at,
        "account_fee_received_at": receipt_at,
    })
    for name in ("market_fact", "orderbook_fact", "mark_price_fact", "index_price_fact"):
        features.setdefault(name, {"timestamp": source_at, "received_at": receipt_at})
    seal_feature_availability(features, frozen_at=decision_at - timedelta(seconds=5))
