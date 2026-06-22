from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from services.crypto_feature_coverage import summarize_crypto_feature_coverage


def _decision(
    *,
    symbol: str = "BTC/USDT",
    hours_ago: float = 1.0,
    feature_snapshot: dict | None = None,
) -> SimpleNamespace:
    now = datetime(2026, 6, 23, 12, 0, tzinfo=UTC)
    return SimpleNamespace(
        symbol=symbol,
        feature_snapshot=feature_snapshot or {},
        created_at=now - timedelta(hours=hours_ago),
    )


def test_feature_coverage_marks_missing_defaults_as_neutral_not_bullish() -> None:
    now = datetime(2026, 6, 23, 12, 0, tzinfo=UTC)
    report = summarize_crypto_feature_coverage(
        decisions=[
            _decision(
                feature_snapshot={
                    "timestamp": (now - timedelta(minutes=20)).isoformat(),
                    "current_price": 65000.0,
                    "close": 65000.0,
                    "funding_rate": 0.0,
                    "open_interest_value": 0.0,
                    "orderbook_bid_depth": 0.0,
                    "orderbook_ask_depth": 0.0,
                    "orderbook_imbalance": 0.0,
                    "news_sentiment_avg": 0.0,
                    "social_sentiment_avg": 0.0,
                    "sentiment_data_available": False,
                    "direct_sentiment_data_available": False,
                }
            )
        ],
        market_coverage={
            "klines": {
                "1m": {"rows": 10, "symbols": 1, "latest_at": now - timedelta(minutes=2)},
                "5m": {"rows": 0, "symbols": 0, "latest_at": None},
                "15m": {"rows": 0, "symbols": 0, "latest_at": None},
                "1h": {"rows": 0, "symbols": 0, "latest_at": None},
            },
            "ticker": {"count": 1, "latest_at": now - timedelta(minutes=2)},
            "news": {"count": 0, "latest_at": None},
            "social": {"count": 0, "latest_at": None},
        },
        now=now,
    )

    assert report["audit_only"] is True
    assert report["live_signal_mutation"] is False
    assert report["feature_defaults_are_neutral"] is True
    assert report["can_missing_features_drive_live_entry"] is False
    feature_states = {item["key"]: item for item in report["features"]}
    assert feature_states["funding_rate"]["status"] == "missing"
    assert "default_zero_without_presence_flag" in feature_states["funding_rate"]["reasons"]
    assert feature_states["orderbook_depth"]["status"] == "missing"
    assert feature_states["news"]["live_entry_influence"] == "blocked"
    assert "news" in report["missing_features"]
    assert "funding_rate" in report["neutralized_features"]
    assert report["feature_contribution_policy"]["missing_feature_policy"] == "neutral_blocked"


def test_feature_coverage_accepts_available_timestamped_crypto_features() -> None:
    now = datetime(2026, 6, 23, 12, 0, tzinfo=UTC)
    snapshot_time = now - timedelta(minutes=4)
    report = summarize_crypto_feature_coverage(
        decisions=[
            _decision(
                symbol="SOL/USDT",
                feature_snapshot={
                    "timestamp": snapshot_time.isoformat(),
                    "current_price": 140.0,
                    "close": 140.0,
                    "bid": 139.99,
                    "ask": 140.01,
                    "spread_pct": 0.014,
                    "returns_1": 0.001,
                    "returns_5": -0.002,
                    "returns_20": 0.005,
                    "technical_indicator_timeframe": "1h",
                    "short_returns_timeframe": "1m",
                    "abnormal_wick_count_72h": 2,
                    "abnormal_wick_max_pct": 4.2,
                    "abnormal_wick_recent_hours": 3.0,
                    "funding_rate": 0.0001,
                    "next_funding_time": "2026-06-23T16:00:00+00:00",
                    "open_interest_value": 1234567.0,
                    "orderbook_bid_depth": 30000.0,
                    "orderbook_ask_depth": 28000.0,
                    "orderbook_imbalance": 0.034,
                    "news_sentiment_avg": 0.12,
                    "social_sentiment_avg": -0.03,
                    "news_article_count": 3,
                    "social_mention_count": 8,
                    "sentiment_data_available": True,
                    "direct_sentiment_data_available": True,
                    "direct_news_item_count": 2,
                    "recent_news_items": [
                        {
                            "source": "okx_announcements",
                            "source_weight": 0.9,
                            "published_at": (now - timedelta(minutes=30)).isoformat(),
                            "symbols": ["SOL"],
                            "event_type": "exchange_announcement",
                        }
                    ],
                },
            )
        ],
        market_coverage={
            "klines": {
                "1m": {"rows": 10, "symbols": 1, "latest_at": now - timedelta(minutes=1)},
                "5m": {"rows": 10, "symbols": 1, "latest_at": now - timedelta(minutes=5)},
                "15m": {"rows": 10, "symbols": 1, "latest_at": now - timedelta(minutes=15)},
                "1h": {"rows": 10, "symbols": 1, "latest_at": now - timedelta(hours=1)},
            },
            "ticker": {"count": 1, "latest_at": now - timedelta(minutes=1)},
            "news": {"count": 3, "latest_at": now - timedelta(minutes=30)},
            "social": {"count": 8, "latest_at": now - timedelta(minutes=45)},
        },
        now=now,
    )

    assert report["status"] == "warning"
    feature_states = {item["key"]: item for item in report["features"]}
    assert feature_states["funding_rate"]["status"] == "available"
    assert feature_states["open_interest"]["status"] == "available"
    assert feature_states["orderbook_depth"]["status"] == "available"
    assert feature_states["news"]["status"] == "available"
    assert feature_states["event_calendar"]["status"] == "missing"
    assert feature_states["news"]["source"] == "feature_snapshot"
    assert feature_states["news"]["confidence"] > 0
    assert report["symbols_observed"] == ["SOL/USDT"]
    assert "event_calendar" in report["missing_features"]
    assert "funding_rate" not in report["missing_features"]
