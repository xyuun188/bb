"""
Historical data loader for backtesting.
Loads OHLCV data from database or CCXT.
"""

from __future__ import annotations

import pandas as pd
import structlog

from data_feed.okx_rest_client import OKXRestClient

logger = structlog.get_logger(__name__)


async def load_historical_from_okx(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    limit: int = 500,
) -> pd.DataFrame:
    """Fetch historical OHLCV data from OKX via CCXT."""
    client = OKXRestClient()
    try:
        ohlcv = await client.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")
        logger.info("historical data loaded", symbol=symbol, rows=len(df))
        return df
    finally:
        await client.close()


async def load_historical_from_db(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    limit: int = 500,
) -> pd.DataFrame:
    """Load historical OHLCV data from database."""
    from db.repositories.market_repo import MarketRepository
    from db.session import get_session_ctx

    async with get_session_ctx() as session:
        repo = MarketRepository(session)
        klines = await repo.get_klines(symbol, timeframe, limit)

    if not klines:
        logger.warning("no historical data in DB, fetching from OKX")
        return await load_historical_from_okx(symbol, timeframe, limit)

    df = pd.DataFrame(
        [
            {
                "timestamp": k.open_time,
                "open": k.open,
                "high": k.high,
                "low": k.low,
                "close": k.close,
                "volume": k.volume,
            }
            for k in klines
        ]
    )
    df = df.set_index("timestamp")
    return df
