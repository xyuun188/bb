"""
Sentiment analysis using FinBERT / transformers.
Scores text on a -1 to 1 scale (negative to positive) for financial news.

Note: FinBERT model loading is lazy and optional. If torch/transformers
is not available or fails to load, the decision model still works using
pre-computed sentiment scores from the data feed layer.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import structlog

from ai_brain.base_model import AbstractAIModel, Action, DecisionOutput

logger = structlog.get_logger(__name__)

_inference_executor = ThreadPoolExecutor(max_workers=2)


class FinBERTSentimentAnalyzer:
    """Thin wrapper around a FinBERT sentiment pipeline.

    Falls back gracefully if transformers/torch is unavailable.
    """

    def __init__(self) -> None:
        self._pipeline = None
        self._model_name = "ProsusAI/finbert"
        self._available = False

    async def initialize(self) -> None:
        """Load FinBERT model. Swallows errors if not available."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(_inference_executor, self._load_model)
            self._available = True
            logger.info("finbert model loaded", model=self._model_name)
        except Exception as e:
            logger.warning("finbert model not available, using cached sentiment", error=str(e))
            self._available = False

    def _load_model(self) -> None:
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
        self._pipeline = pipeline(
            "sentiment-analysis", model=model, tokenizer=tokenizer,
            truncation=True, max_length=512,
        )

    def _score_text(self, text: str) -> float:
        if not text or not self._pipeline:
            return 0.0
        try:
            result = self._pipeline(text[:512])[0]
            label = result["label"].lower()
            score = result["score"]
            return score if label == "positive" else (-score if label == "negative" else 0.0)
        except Exception:
            return 0.0

    async def score_text(self, text: str) -> float:
        if not self._available:
            return 0.0
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_inference_executor, self._score_text, text)

    async def score_batch(self, texts: list[str]) -> list[float]:
        if not self._available:
            return [0.0] * len(texts)
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(_inference_executor, self._score_text, t) for t in texts]
        return await asyncio.gather(*tasks)

    async def shutdown(self) -> None:
        self._pipeline = None


class FinBERTDecisionModel(AbstractAIModel):
    """Trading model based on sentiment scores (from FinBERT or cached).

    Decision rules:
    - News sentiment > 0.35 AND social > 0.2 -> LONG
    - News sentiment < -0.35 AND social < -0.2 -> SHORT
    - Extreme negative (< -0.6) -> CLOSE_LONG
    - Otherwise HOLD

    Uses pre-computed sentiment from FeatureVector (set by data_service).
    Works even without the FinBERT model loaded.
    """

    name = "finbert_sentiment"

    def __init__(self, name: str | None = None) -> None:
        if name is not None:
            self.name = name
        self._analyzer: FinBERTSentimentAnalyzer | None = None
        self._initialized = False

    async def initialize(self) -> None:
        try:
            self._analyzer = FinBERTSentimentAnalyzer()
            await self._analyzer.initialize()
        except Exception as e:
            logger.warning("finbert init failed, using cached sentiment", error=str(e))
            self._analyzer = None
        self._initialized = True

    async def decide(self, features: "FeatureVector", context: dict[str, Any]) -> DecisionOutput:
        news_sent = features.news_sentiment_avg
        social_sent = features.social_sentiment_avg
        mention_count = features.social_mention_count

        action = Action.HOLD
        confidence = 0.5
        reasoning = ""

        if news_sent > 0.35 and social_sent > 0.2:
            action = Action.LONG
            confidence = min(0.5 + (news_sent + social_sent) * 0.4, 0.9)
            reasoning = f"Bullish sentiment: news={news_sent:.2f}, social={social_sent:.2f}."
        elif news_sent < -0.35 and social_sent < -0.2:
            action = Action.SHORT
            confidence = min(0.5 + abs(news_sent + social_sent) * 0.4, 0.9)
            reasoning = f"Bearish sentiment: news={news_sent:.2f}, social={social_sent:.2f}."
        elif news_sent < -0.6:
            action = Action.CLOSE_LONG
            confidence = 0.8
            reasoning = f"Extreme negative sentiment ({news_sent:.2f}). Closing longs."
        elif mention_count > 50 and abs(news_sent) < 0.1:
            reasoning = f"High social activity ({mention_count} mentions) but neutral. Holding."
        else:
            reasoning = f"Neutral sentiment (news={news_sent:.2f}, social={social_sent:.2f}). Holding."

        abs_sent = (abs(news_sent) + abs(social_sent)) / 2
        position_size = min(abs_sent * 0.3, 0.15)

        return DecisionOutput(
            model_name=self.name,
            symbol=features.symbol,
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            position_size_pct=position_size if action.is_entry() else 0.0,
            suggested_leverage=1.0,
            stop_loss_pct=0.05,
            take_profit_pct=0.10,
            feature_snapshot=features.to_dict(),
        )

    async def shutdown(self) -> None:
        if self._analyzer:
            await self._analyzer.shutdown()
            self._analyzer = None
