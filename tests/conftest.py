"""
Pytest fixtures and configuration for the AI trading system.
"""

import asyncio
import os
import sys
from pathlib import Path

import pytest

# Test collection imports some operational scripts. Keep those imports from
# loading the host's production runtime environment before fixtures can run.
os.environ["BB_RUNTIME_ENV_PATH"] = str(
    Path(__file__).resolve().parent / ".runtime-env-disabled"
)

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def isolate_model_training_scheduler_state(tmp_path, monkeypatch):
    """Keep scheduler writes from tests out of the runtime data directory."""
    from services.model_training_state import ModelTrainingStateStore

    store = ModelTrainingStateStore(tmp_path / "model_training_scheduler_state.json")
    monkeypatch.setattr("services.ml_signal_service.MODEL_TRAINING_STATE_STORE", store)
    monkeypatch.setattr("services.trading_service.MODEL_TRAINING_STATE_STORE", store)
    data_collection = sys.modules.get("web_dashboard.api.data_collection")
    if data_collection is not None:
        monkeypatch.setattr(data_collection, "MODEL_TRAINING_STATE_STORE", store)


@pytest.fixture(autouse=True)
def isolate_okx_private_api_circuit_state():
    """Prevent process-wide circuit state from leaking between tests."""
    from executor.okx_executor import OKXExecutor

    OKXExecutor.reset_private_api_circuit_states()
    yield
    OKXExecutor.reset_private_api_circuit_states()


@pytest.fixture(autouse=True)
async def dispose_shared_db_engine_after_test():
    """Cancel test-owned background work before releasing the async DB engine."""
    yield
    current_task = asyncio.current_task()
    pending_tasks = [
        task
        for task in asyncio.all_tasks()
        if task is not current_task and not task.done()
    ]
    for task in pending_tasks:
        task.cancel()
    if pending_tasks:
        await asyncio.gather(*pending_tasks, return_exceptions=True)
    from db.session import close_db

    await close_db()


@pytest.fixture(scope="session")
def event_loop():
    """Create a single event loop for all async tests."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    loop = asyncio.new_event_loop()
    yield loop
    # Dispose SQLAlchemy/aiosqlite workers before closing the loop.  Otherwise
    # a worker can finish after loop shutdown and raise an unhandled callback
    # exception even though the test assertions all passed.
    try:
        from db.session import close_db

        loop.run_until_complete(close_db())
    finally:
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
