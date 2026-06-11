"""
Technical indicators computed from OHLCV data.
Uses the `ta` library for common indicators: RSI, MACD, Bollinger Bands, etc.

Each function takes a pandas DataFrame with columns: open, high, low, close, volume.
Returns the DataFrame with additional indicator columns appended.
"""

from __future__ import annotations

import pandas as pd
import ta

# Map of CCXT timeframe strings to readable names
TIMEFRAME_LABELS = {
    "1m": "1分钟",
    "5m": "5分钟",
    "15m": "15分钟",
    "30m": "30分钟",
    "1h": "1小时",
    "4h": "4小时",
    "1d": "日线",
}


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a comprehensive set of technical indicators.

    Args:
        df: DataFrame with columns [open, high, low, close, volume]

    Returns:
        DataFrame with added indicator columns.
    """
    if df.empty or len(df) < 20:
        return df

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    # --- Momentum ---
    df["rsi_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    df["rsi_7"] = ta.momentum.RSIIndicator(close=close, window=7).rsi()

    macd = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    df["stoch_k"] = ta.momentum.StochasticOscillator(
        high=high, low=low, close=close, window=14, smooth_window=3
    ).stoch()

    # --- Trend ---
    df["ema_12"] = ta.trend.EMAIndicator(close=close, window=12).ema_indicator()
    df["ema_26"] = ta.trend.EMAIndicator(close=close, window=26).ema_indicator()

    df["adx_14"] = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14).adx()

    # --- Volatility ---
    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    df["bb_pct"] = (close - df["bb_lower"]) / (df["bb_width"] + 1e-10)

    df["atr_14"] = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=14
    ).average_true_range()

    # --- Volume ---
    df["volume_sma_20"] = volume.rolling(window=20).mean()
    df["volume_ratio"] = volume / (df["volume_sma_20"] + 1e-10)
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()

    # --- Price derivatives ---
    df["returns_1"] = close.pct_change(1)
    df["returns_5"] = close.pct_change(5)
    df["returns_20"] = close.pct_change(20)
    df["volatility_20"] = df["returns_1"].rolling(20).std()

    # Price relative to moving averages
    df["sma_20"] = close.rolling(20).mean()
    df["sma_50"] = close.rolling(50).mean()
    df["sma_200"] = close.rolling(200).mean()
    df["price_vs_sma20"] = close / (df["sma_20"] + 1e-10) - 1
    df["price_vs_sma50"] = close / (df["sma_50"] + 1e-10) - 1

    return df


def extract_latest_features(df: pd.DataFrame) -> dict[str, float]:
    """Extract the latest (most recent) indicator values as a flat dict.

    Used by the feature vector builder in the main trading loop.
    """
    if df.empty:
        return {}

    latest = df.iloc[-1]
    feature_keys = [
        "rsi_14",
        "rsi_7",
        "macd",
        "macd_signal",
        "macd_diff",
        "stoch_k",
        "ema_12",
        "ema_26",
        "adx_14",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "bb_width",
        "bb_pct",
        "atr_14",
        "volume_ratio",
        "obv",
        "returns_1",
        "returns_5",
        "returns_20",
        "volatility_20",
        "price_vs_sma20",
        "price_vs_sma50",
    ]
    features = {}
    for key in feature_keys:
        val = latest.get(key)
        if pd.notna(val):
            features[key] = float(val)
        else:
            features[key] = 0.0

    features["close"] = float(latest.get("close", 0))
    features["volume"] = float(latest.get("volume", 0))
    return features
