"""
Pytest fixtures and configuration for the AI trading system.
"""

import asyncio
import sys
from pathlib import Path

import pytest

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def event_loop():
    """Create a single event loop for all async tests."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_feature_vector():
    """Return a mock feature vector for testing models."""
    from data_feed.feature_vector import FeatureVector

    return FeatureVector(
        symbol="BTC/USDT",
        current_price=50000.0,
        rsi_14=45.0,
        rsi_7=42.0,
        macd=100.0,
        macd_signal=80.0,
        macd_diff=20.0,
        bb_upper=52000.0,
        bb_middle=50000.0,
        bb_lower=48000.0,
        volume_ratio=1.2,
        returns_1=0.001,
        returns_5=0.015,
        returns_20=-0.02,
        volatility_20=0.03,
        price_vs_sma20=0.01,
        price_vs_sma50=-0.02,
        news_sentiment_avg=0.35,
        social_sentiment_avg=0.25,
        social_mention_count=45,
        recent_headlines=["Bitcoin ETF inflows reach new high", "BTC price consolidates above 50K"],
    )
