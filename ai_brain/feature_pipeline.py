"""
Feature preprocessing pipeline.
Handles normalization, missing value imputation, and feature selection
for models that require structured numerical input (XGBoost, RL).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.preprocessing import RobustScaler

from config.settings import settings
from core.model_artifact_safety import dump_trusted_pickle, load_trusted_pickle

# Columns that should be present in the feature vector
TECHNICAL_FEATURE_COLS = [
    "rsi_14",
    "rsi_7",
    "macd",
    "macd_signal",
    "macd_diff",
    "stoch_k",
    "adx_14",
    "bb_pct",
    "bb_width",
    "atr_14",
    "volume_ratio",
    "returns_1",
    "returns_5",
    "returns_20",
    "volatility_20",
    "price_vs_sma20",
    "price_vs_sma50",
]

SENTIMENT_FEATURE_COLS = [
    "news_sentiment_avg",
    "social_sentiment_avg",
    "social_mention_count",
]

ALL_FEATURE_COLS = TECHNICAL_FEATURE_COLS + SENTIMENT_FEATURE_COLS

# Default values for missing features
DEFAULT_VALUES = {
    "rsi_14": 50.0,
    "rsi_7": 50.0,
    "macd": 0.0,
    "macd_signal": 0.0,
    "macd_diff": 0.0,
    "stoch_k": 50.0,
    "adx_14": 20.0,
    "bb_pct": 0.5,
    "bb_width": 0.01,
    "atr_14": 0.001,
    "volume_ratio": 1.0,
    "returns_1": 0.0,
    "returns_5": 0.0,
    "returns_20": 0.0,
    "volatility_20": 0.02,
    "price_vs_sma20": 0.0,
    "price_vs_sma50": 0.0,
    "news_sentiment_avg": 0.0,
    "social_sentiment_avg": 0.0,
    "social_mention_count": 0,
}


class FeaturePipeline:
    """Preprocesses feature vectors for ML model consumption.

    Handles:
    1. Feature extraction from FeatureVector or dict
    2. Missing value imputation with sensible defaults
    3. Robust scaling (fitted on historical data, saved to disk)
    """

    def __init__(self) -> None:
        self._scaler: RobustScaler | None = None
        self._fitted = False
        self._scaler_path = settings.data_dir / "feature_scaler.pkl"

    def extract_features(self, feature_dict: dict[str, Any]) -> np.ndarray:
        """Extract a 1D feature array from a flat feature dict."""
        values = []
        for col in ALL_FEATURE_COLS:
            val = feature_dict.get(col)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                val = DEFAULT_VALUES.get(col, 0.0)
            values.append(float(val))
        return np.array(values)

    def extract_batch(self, feature_dicts: list[dict[str, Any]]) -> np.ndarray:
        """Extract a 2D feature array from a list of feature dicts."""
        return np.array([self.extract_features(d) for d in feature_dicts])

    def fit(self, X: np.ndarray) -> None:
        """Fit the robust scaler on historical feature data."""
        self._scaler = RobustScaler()
        self._scaler.fit(X)
        self._fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale features using the fitted scaler."""
        if not self._fitted or self._scaler is None:
            # No scaler fitted yet; return as-is (or apply simple standardization)
            return X
        return self._scaler.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def save(self) -> None:
        if self._scaler:
            dump_trusted_pickle(
                self._scaler,
                self._scaler_path,
                trusted_root=settings.data_dir,
            )

    def load(self) -> bool:
        if self._scaler_path.exists():
            self._scaler = load_trusted_pickle(
                self._scaler_path,
                trusted_root=settings.data_dir,
                expected_type=RobustScaler,
            )
            self._fitted = True
            return True
        return False
