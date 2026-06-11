"""
XGBoost classifier model for trading decisions.
Uses technical indicators + sentiment as features.
Trains on historical data; supports online incremental retraining.

This model provides a non-LLM baseline that is fast, deterministic,
and has no external API dependency.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from ai_brain.base_model import AbstractAIModel, Action, DecisionOutput
from config.settings import settings
from core.model_artifact_safety import dump_trusted_pickle, load_trusted_pickle
from core.safe_output import safe_error_text

if TYPE_CHECKING:
    from data_feed.feature_vector import FeatureVector

logger = structlog.get_logger(__name__)

# Feature columns used by XGBoost (subset of FeatureVector fields)
FEATURE_COLUMNS = [
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
    "news_sentiment_avg",
    "social_sentiment_avg",
    "social_mention_count",
]

# Action labels: 0=hold, 1=long, 2=short, 3=close_long, 4=close_short
ACTION_TO_LABEL = {
    Action.HOLD: 0,
    Action.LONG: 1,
    Action.SHORT: 2,
    Action.CLOSE_LONG: 3,
    Action.CLOSE_SHORT: 4,
}
LABEL_TO_ACTION = {v: k for k, v in ACTION_TO_LABEL.items()}


class XGBoostModel(AbstractAIModel):
    """XGBoost-based trading decision model.

    Uses a multi-class classifier to predict the optimal action
    from feature vectors. Supports incremental retraining from
    stored decision outcomes.
    """

    name = "xgboost"

    def __init__(self, name: str | None = None) -> None:
        if name is not None:
            self.name = name
        self._model: xgb.XGBClassifier | None = None
        self._scaler: StandardScaler | None = None
        self._model_path = settings.data_dir / "xgboost_model.json"
        self._scaler_path = settings.data_dir / "xgboost_scaler.pkl"

    async def initialize(self) -> None:
        """Load saved model if exists, otherwise train a minimal seed model."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._init_sync)

    def _init_sync(self) -> None:
        if self._model_path.exists() and self._scaler_path.exists():
            try:
                self._model = xgb.XGBClassifier()
                self._model.load_model(str(self._model_path))
                self._scaler = load_trusted_pickle(
                    self._scaler_path,
                    trusted_root=settings.data_dir,
                    expected_type=StandardScaler,
                )
                logger.info("xgboost model loaded from disk")
            except Exception as e:
                logger.warning(
                    "failed to load xgboost model, creating seed",
                    error=safe_error_text(e),
                )
                self._create_seed_model()
        else:
            self._create_seed_model()

    def _create_seed_model(self) -> None:
        """Create a minimal seed model with synthetic training data."""
        logger.info("creating xgboost seed model")

        # Generate synthetic balanced data with simple rules
        np.random.seed(42)
        n_samples = 500

        X = np.zeros((n_samples, len(FEATURE_COLUMNS)))
        y = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            rsi = np.random.uniform(20, 80)
            macd = np.random.uniform(-50, 50)
            sentiment = np.random.uniform(-0.5, 0.5)
            vol_ratio = np.random.uniform(0.5, 2.0)
            returns_5 = np.random.uniform(-0.05, 0.05)

            X[i] = [
                rsi,
                rsi - np.random.uniform(-5, 5),
                macd,
                macd * 0.5,
                macd * 0.3,
                np.random.uniform(20, 80),
                np.random.uniform(10, 40),
                np.random.uniform(0.2, 0.8),
                np.random.uniform(0.01, 0.1),
                np.random.uniform(0.001, 0.01),
                vol_ratio,
                returns_5 * 0.2,
                returns_5,
                returns_5 * 4,
                np.random.uniform(0.01, 0.05),
                np.random.uniform(-0.03, 0.03),
                np.random.uniform(-0.05, 0.05),
                sentiment,
                sentiment * 0.7,
                np.random.randint(0, 100),
            ]

            # Simple heuristic labeling
            if sentiment > 0.25 and rsi < 60 and macd > 0:
                y[i] = 1  # long
            elif sentiment < -0.25 and rsi > 40 and macd < 0:
                y[i] = 2  # short
            else:
                y[i] = 0  # hold

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            objective="multi:softprob",
            num_class=5,
            eval_metric="mlogloss",
            random_state=42,
        )
        self._model.fit(X_scaled, y)
        logger.info("xgboost seed model trained")

    def _features_to_array(self, features: FeatureVector) -> np.ndarray:
        """Extract feature values into numpy array."""
        d = features.to_dict()
        values = []
        for col in FEATURE_COLUMNS:
            val = d.get(col, 0.0)
            if val is None:
                val = 0.0
            values.append(float(val))
        return np.array([values])

    async def decide(self, features: FeatureVector, context: dict[str, Any]) -> DecisionOutput:
        if self._model is None or self._scaler is None:
            await self.initialize()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._decide_sync, features, context)

    def _decide_sync(self, features: FeatureVector, context: dict) -> DecisionOutput:
        if self._model is None or self._scaler is None:
            raise RuntimeError("xgboost model is not initialized")
        model = self._model
        scaler = self._scaler

        X = self._features_to_array(features)
        X_scaled = scaler.transform(X)

        # Get probability distribution over all classes
        proba = model.predict_proba(X_scaled)[0]
        predicted_label = int(np.argmax(proba))
        confidence = float(proba[predicted_label])

        action = LABEL_TO_ACTION.get(predicted_label, Action.HOLD)

        # If confidence is too low, default to hold
        if confidence < 0.5:
            action = Action.HOLD
            confidence = 0.0

        reasoning = self._build_reasoning(proba, action, confidence, features)

        return DecisionOutput(
            model_name=self.name,
            symbol=features.symbol,
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            position_size_pct=0.1 if action.is_entry() else 0.0,
            suggested_leverage=1.0,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            feature_snapshot=features.to_dict(),
        )

    def _build_reasoning(
        self, proba: np.ndarray, action: Action, confidence: float, features: FeatureVector
    ) -> str:
        """Generate a human-readable reasoning trace."""
        top_3 = np.argsort(proba)[-3:][::-1]
        labels = [LABEL_TO_ACTION.get(i, Action.HOLD).value for i in top_3]
        scores = [float(proba[i]) for i in top_3]

        factors = []
        if features.rsi_14 > 70:
            factors.append("RSI overbought")
        elif features.rsi_14 < 30:
            factors.append("RSI oversold")
        if features.macd > features.macd_signal:
            factors.append("MACD bullish")
        else:
            factors.append("MACD bearish")
        if features.news_sentiment_avg > 0.3:
            factors.append("positive news sentiment")
        elif features.news_sentiment_avg < -0.3:
            factors.append("negative news sentiment")

        factor_str = ", ".join(factors) if factors else "mixed signals"
        return (
            f"XGBoost: {action.value} (confidence={confidence:.2f}). "
            f"Top classes: {labels[0]}={scores[0]:.2f}, {labels[1]}={scores[1]:.2f}, "
            f"{labels[2]}={scores[2]:.2f}. Factors: {factor_str}."
        )

    async def retrain(self, training_data: list[dict]) -> None:
        """Incrementally retrain the model with new labeled data.

        Args:
            training_data: List of dicts with 'features' (FeatureVector or dict)
                          and 'label' (Action or str).
        """
        if not training_data:
            return

        X_list = []
        y_list = []

        for row in training_data:
            fv = row.get("features", {})
            if isinstance(fv, dict):
                values = [float(fv.get(col, 0.0) or 0.0) for col in FEATURE_COLUMNS]
            else:
                values = self._features_to_array(fv)[0].tolist()
            X_list.append(values)

            label = row.get("label", "hold")
            if isinstance(label, str):
                label = Action.from_string(label)
            y_list.append(ACTION_TO_LABEL.get(label, 0))

        X = np.array(X_list)
        y = np.array(y_list)

        if len(set(y)) < 2:
            logger.warning("not enough class diversity for retraining")
            return

        if self._model is None or self._scaler is None:
            await self.initialize()
        if self._model is None or self._scaler is None:
            raise RuntimeError("xgboost model is not initialized")
        model = self._model
        scaler = self._scaler

        X_scaled = scaler.transform(X)

        # Incremental fit (warm start from existing model)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: model.fit(X_scaled, y, xgb_model=model.get_booster()),
        )

        # Save to disk
        await loop.run_in_executor(None, self._save)
        logger.info("xgboost retrained and saved", samples=len(X_list))

    def _save(self) -> None:
        if self._model:
            self._model.save_model(str(self._model_path))
        if self._scaler:
            dump_trusted_pickle(
                self._scaler,
                self._scaler_path,
                trusted_root=settings.data_dir,
            )

    async def shutdown(self) -> None:
        if self._model:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._save)
        self._model = None
        self._scaler = None
