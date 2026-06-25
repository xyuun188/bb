from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from services.crypto_feature_coverage import (
    CryptoFeatureCoverageService,
    summarize_crypto_feature_coverage,
)


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


def test_feature_coverage_accepts_open_1m_candle_with_collection_lag() -> None:
    now = datetime(2026, 6, 23, 12, 2, 24, tzinfo=UTC)
    report = summarize_crypto_feature_coverage(
        decisions=[
            _decision(
                feature_snapshot={
                    "timestamp": (now - timedelta(minutes=5)).isoformat(),
                    "current_price": 65000.0,
                    "close": 65000.0,
                }
            )
        ],
        market_coverage={
            "klines": {
                "1m": {"rows": 10, "symbols": 2, "latest_at": now - timedelta(seconds=144)},
                "5m": {"rows": 10, "symbols": 2, "latest_at": now - timedelta(minutes=5)},
                "15m": {"rows": 10, "symbols": 2, "latest_at": now - timedelta(minutes=15)},
                "1h": {"rows": 10, "symbols": 2, "latest_at": now - timedelta(hours=1)},
            },
            "ticker": {"count": 2, "latest_at": now - timedelta(minutes=1)},
            "news": {"count": 0, "latest_at": None},
            "social": {"count": 0, "latest_at": None},
        },
        now=now,
    )

    feature_states = {item["key"]: item for item in report["features"]}
    assert feature_states["kline_1m"]["status"] == "available"
    assert "kline_1m" not in report["stale_features"]


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


def test_feature_coverage_uses_recent_available_snapshot_per_feature() -> None:
    now = datetime(2026, 6, 23, 12, 0, tzinfo=UTC)
    report = summarize_crypto_feature_coverage(
        decisions=[
            _decision(
                symbol="ETH/USDT",
                hours_ago=0.1,
                feature_snapshot={
                    "timestamp": (now - timedelta(minutes=5)).isoformat(),
                    "current_price": 2500.0,
                    "close": 2500.0,
                },
            ),
            _decision(
                symbol="SOL/USDT",
                hours_ago=0.4,
                feature_snapshot={
                    "timestamp": (now - timedelta(minutes=20)).isoformat(),
                    "current_price": 140.0,
                    "close": 140.0,
                    "volatility_20": 0.032,
                    "abnormal_wick_count_72h": 0,
                    "abnormal_wick_max_pct": 0.0,
                    "abnormal_wick_recent_hours": 9999.0,
                },
            ),
        ],
        market_coverage={
            "klines": {
                "1m": {"rows": 10, "symbols": 1, "latest_at": now - timedelta(minutes=1)},
                "5m": {"rows": 10, "symbols": 1, "latest_at": now - timedelta(minutes=5)},
                "15m": {"rows": 10, "symbols": 1, "latest_at": now - timedelta(minutes=15)},
                "1h": {"rows": 10, "symbols": 1, "latest_at": now - timedelta(hours=1)},
            },
            "ticker": {"count": 1, "latest_at": now - timedelta(minutes=1)},
            "news": {"count": 0, "latest_at": None},
            "social": {"count": 0, "latest_at": None},
        },
        now=now,
    )

    feature_states = {item["key"]: item for item in report["features"]}
    assert feature_states["altcoin_volatility_risk"]["status"] == "available"
    assert feature_states["abnormal_wick"]["status"] == "available"
    assert feature_states["abnormal_wick"]["details"]["count_72h"] == 0
    assert "altcoin_volatility_risk" not in report["missing_features"]
    assert "abnormal_wick" not in report["missing_features"]


def test_feature_coverage_uses_recent_derivatives_and_market_anchor_evidence() -> None:
    now = datetime(2026, 6, 23, 12, 0, tzinfo=UTC)
    report = summarize_crypto_feature_coverage(
        decisions=[
            _decision(
                symbol="MASK/USDT",
                hours_ago=0.05,
                feature_snapshot={
                    "timestamp": (now - timedelta(minutes=3)).isoformat(),
                    "current_price": 2.4,
                    "close": 2.4,
                    "orderbook_bid_depth": 0.0,
                    "orderbook_ask_depth": 0.0,
                    "funding_rate": 0.0,
                    "next_funding_time": None,
                    "open_interest_contracts": 0.0,
                    "open_interest_value": 0.0,
                },
            ),
            _decision(
                symbol="AUCTION/USDT",
                hours_ago=0.2,
                feature_snapshot={
                    "timestamp": (now - timedelta(minutes=12)).isoformat(),
                    "current_price": 0.35,
                    "close": 0.35,
                    "orderbook_bid_depth": 180573.55,
                    "orderbook_ask_depth": 148706.2,
                    "orderbook_imbalance": 0.096,
                    "funding_rate": -0.0000296,
                    "next_funding_time": "2026-06-23T16:00:00+00:00",
                    "open_interest_contracts": 100487756.8,
                    "open_interest_value": 35442031.82,
                },
            ),
        ],
        market_coverage={
            "klines": {
                "1m": {"rows": 10, "symbols": 2, "latest_at": now - timedelta(minutes=1)},
                "5m": {"rows": 10, "symbols": 2, "latest_at": now - timedelta(minutes=5)},
                "15m": {"rows": 10, "symbols": 2, "latest_at": now - timedelta(minutes=15)},
                "1h": {"rows": 10, "symbols": 2, "latest_at": now - timedelta(hours=1)},
            },
            "ticker": {"count": 2, "latest_at": now - timedelta(minutes=1)},
            "btc_eth_anchor": {
                "btc": {
                    "change_24h_pct": 0.68,
                    "latest_at": now - timedelta(minutes=1),
                },
                "eth": {
                    "change_24h_pct": 1.39,
                    "latest_at": now - timedelta(minutes=1),
                },
            },
            "news": {"count": 0, "latest_at": None},
            "social": {"count": 0, "latest_at": None},
        },
        now=now,
    )

    feature_states = {item["key"]: item for item in report["features"]}
    assert feature_states["orderbook_depth"]["status"] == "available"
    assert feature_states["orderbook_depth"]["details"]["bid_depth"] == 180573.55
    assert feature_states["funding_rate"]["status"] == "available"
    assert feature_states["open_interest"]["status"] == "available"
    assert feature_states["btc_eth_anchor"]["status"] == "available"
    assert feature_states["btc_eth_anchor"]["source"] == "market_tickers"
    assert feature_states["btc_eth_anchor"]["details"] == {
        "btc_change_24h_pct": 0.68,
        "eth_change_24h_pct": 1.39,
    }
    assert "orderbook_depth" not in report["missing_features"]
    assert "funding_rate" not in report["missing_features"]
    assert "open_interest" not in report["missing_features"]
    assert "btc_eth_anchor" not in report["missing_features"]


def test_feature_coverage_accepts_dedicated_event_sources_from_market_coverage() -> None:
    now = datetime(2026, 6, 23, 12, 0, tzinfo=UTC)
    report = summarize_crypto_feature_coverage(
        decisions=[
            _decision(
                symbol="ETH/USDT",
                hours_ago=0.1,
                feature_snapshot={
                    "timestamp": (now - timedelta(minutes=5)).isoformat(),
                    "current_price": 2500.0,
                    "close": 2500.0,
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
            "event_calendar": {
                "count": 3,
                "latest_at": now - timedelta(minutes=30),
                "sources": [
                    {
                        "source": "okx_announcements",
                        "count": 2,
                        "latest_at": now - timedelta(minutes=30),
                    },
                    {
                        "source": "scrapling:ethereum_blog",
                        "count": 1,
                        "latest_at": now - timedelta(hours=2),
                    },
                ],
            },
            "news": {"count": 3, "latest_at": now - timedelta(minutes=30)},
            "social": {"count": 0, "latest_at": None},
        },
        now=now,
    )

    feature_states = {item["key"]: item for item in report["features"]}
    assert feature_states["event_calendar"]["status"] == "available"
    assert feature_states["event_calendar"]["source"] == "event_calendar"
    assert feature_states["event_calendar"]["details"]["item_count"] == 2
    assert "event_calendar" not in report["missing_features"]


def test_feature_coverage_does_not_treat_generic_news_as_event_calendar() -> None:
    now = datetime(2026, 6, 23, 12, 0, tzinfo=UTC)
    report = summarize_crypto_feature_coverage(
        decisions=[
            _decision(
                symbol="BTC/USDT",
                feature_snapshot={
                    "timestamp": (now - timedelta(minutes=5)).isoformat(),
                    "current_price": 65000.0,
                    "close": 65000.0,
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
            "event_calendar": {"count": 0, "latest_at": None, "sources": []},
            "news": {"count": 50, "latest_at": now - timedelta(minutes=10)},
            "social": {"count": 0, "latest_at": None},
        },
        now=now,
    )

    feature_states = {item["key"]: item for item in report["features"]}
    assert feature_states["news"]["status"] == "available"
    assert feature_states["event_calendar"]["status"] == "missing"
    assert "event_calendar" in report["missing_features"]


def test_feature_coverage_service_uses_lightweight_decision_projection() -> None:
    class FakeColumn:
        def label(self, name: str):
            return (self, name)

        def __ge__(self, _other):
            return (self, "gte")

        def desc(self):
            return (self, "desc")

    class FakeAIDecision:
        id = FakeColumn()
        symbol = FakeColumn()
        created_at = FakeColumn()
        feature_snapshot = FakeColumn()

    columns = CryptoFeatureCoverageService._decision_projection_columns(FakeAIDecision)

    assert columns == (
        FakeAIDecision.id,
        FakeAIDecision.symbol,
        FakeAIDecision.created_at,
        FakeAIDecision.feature_snapshot,
    )
