"""
Feature vector builder.
Combines market data, technical indicators, sentiment, and social data
into a unified FeatureVector dataclass consumed by all AI models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class FeatureVector:
    """Unified feature snapshot for a symbol at a point in time.

    All AI models consume this same structure, ensuring fair comparison.
    """

    symbol: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    # --- Price data ---
    current_price: float = 0.0
    close: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    high_24h: float = 0.0
    low_24h: float = 0.0
    volume_24h: float = 0.0
    volume: float = 0.0
    change_24h_pct: float = 0.0
    spread_pct: float = 0.0

    # --- Technical indicators (from latest candle) ---
    rsi_14: float = 50.0
    rsi_7: float = 50.0
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_diff: float = 0.0
    stoch_k: float = 50.0
    ema_12: float = 0.0
    ema_26: float = 0.0
    adx_14: float = 20.0
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    bb_width: float = 0.0
    bb_pct: float = 0.5
    atr_14: float = 0.0
    volume_ratio: float = 1.0
    returns_1: float = 0.0
    returns_5: float = 0.0
    returns_20: float = 0.0
    volatility_20: float = 0.0
    price_vs_sma20: float = 0.0
    price_vs_sma50: float = 0.0
    abnormal_wick_count_72h: int = 0
    abnormal_wick_max_pct: float = 0.0
    abnormal_wick_recent_hours: float = 9999.0

    # --- Perpetual swap microstructure ---
    funding_rate: float = 0.0
    next_funding_time: str | None = None
    open_interest_contracts: float = 0.0
    open_interest_value: float = 0.0
    orderbook_bid_depth: float = 0.0
    orderbook_ask_depth: float = 0.0
    orderbook_imbalance: float = 0.0

    # --- Sentiment data ---
    news_sentiment_avg: float = 0.0
    social_sentiment_avg: float = 0.0
    social_mention_count: int = 0
    news_article_count: int = 0
    headline_count: int = 0
    sentiment_data_available: bool = False
    direct_sentiment_data_available: bool = False
    direct_news_item_count: int = 0
    market_news_item_count: int = 0
    news_sources: list[str] = field(default_factory=list)
    recent_headlines: list[str] = field(default_factory=list)
    recent_news_items: list[dict[str, Any]] = field(default_factory=list)

    # --- On-chain (future) ---
    exchange_inflow: float | None = None
    whale_txn_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a flat dict for storage or serialization."""
        from datetime import datetime as dt

        d = {}
        for f_name, f_value in self.__dict__.items():
            if f_name in ("recent_headlines", "recent_news_items"):
                continue
            if isinstance(f_value, dt):
                d[f_name] = f_value.isoformat()
            else:
                d[f_name] = f_value
        return d

    def to_llm_context(self) -> str:
        """Format as a human-readable context string for LLM prompts."""
        headlines_text = (
            "\n".join(f"  - {h}" for h in self.recent_headlines[:10])
            if self.recent_headlines
            else "  (no recent headlines)"
        )
        news_items_text = (
            "\n".join(
                "  - "
                + " | ".join(
                    str(part)
                    for part in (
                        item.get("source") or "-",
                        item.get("event_type") or "news",
                        item.get("impact_level") or 1,
                        item.get("title") or "-",
                    )
                )
                for item in self.recent_news_items[:8]
            )
            if self.recent_news_items
            else "  (no matched news items)"
        )

        return f"""Symbol: {self.symbol}
Current Price: {self.current_price:.4f}
24h Change: {self.change_24h_pct:.2f}%
Bid/Ask: {self.bid:.4f} / {self.ask:.4f}
Spread: {self.spread_pct:.4f}%
24h High/Low: {self.high_24h:.4f} / {self.low_24h:.4f}
24h Volume: {self.volume_24h:.2f}

Technical Indicators:
  RSI(14): {self.rsi_14:.1f} | RSI(7): {self.rsi_7:.1f}
  MACD: {self.macd:.6f} / Signal: {self.macd_signal:.6f} / Diff: {self.macd_diff:.6f}
  Stochastic K: {self.stoch_k:.1f}
  ADX(14): {self.adx_14:.1f}
  Bollinger Bands: Upper={self.bb_upper:.4f}, Middle={self.bb_middle:.4f}, Lower={self.bb_lower:.4f}
  BB Position: {self.bb_pct:.2f} (0=lower, 1=upper)
  ATR(14): {self.atr_14:.4f}
  Volume Ratio (vs 20MA): {self.volume_ratio:.2f}
  EMA 12/26: {self.ema_12:.4f} / {self.ema_26:.4f}
  Price vs SMA20: {self.price_vs_sma20*100:.2f}% | vs SMA50: {self.price_vs_sma50*100:.2f}%
  Returns: 1p={self.returns_1*100:.2f}%, 5p={self.returns_5*100:.2f}%, 20p={self.returns_20*100:.2f}%
  Volatility(20): {self.volatility_20*100:.2f}%
  Abnormal Wick: count72h={self.abnormal_wick_count_72h}, max={self.abnormal_wick_max_pct:.2f}%, recent_hours={self.abnormal_wick_recent_hours:.1f}

Perpetual Swap Data:
  Funding Rate: {self.funding_rate*100:.4f}%
  Open Interest: contracts={self.open_interest_contracts:.2f}, value={self.open_interest_value:.2f}
  Orderbook Depth: bid={self.orderbook_bid_depth:.2f}, ask={self.orderbook_ask_depth:.2f}, imbalance={self.orderbook_imbalance:.3f}

Market Sentiment:
  News Sentiment: {self.news_sentiment_avg:.3f} (-1 to 1)
  Social Sentiment: {self.social_sentiment_avg:.3f} (-1 to 1)
  Social Mention Count: {self.social_mention_count}
  News Article Count: {self.news_article_count}
  Sentiment Data Available: {self.sentiment_data_available}
  Direct News Items: {self.direct_news_item_count}
  Market Background News Items: {self.market_news_item_count}
  News Sources: {", ".join(self.news_sources[:5]) if self.news_sources else "none"}

Recent Headlines:
{headlines_text}

Matched News Items:
{news_items_text}
"""


def build_feature_vector(
    symbol: str,
    ticker: dict | None = None,
    indicators: dict[str, float] | None = None,
    sentiment_data: dict | None = None,
    headlines: list[str] | None = None,
    derivatives: dict | None = None,
) -> FeatureVector:
    """Factory function: assemble a FeatureVector from raw data sources."""

    fv = FeatureVector(symbol=symbol)

    # Ticker data
    if ticker:
        fv.current_price = ticker.get("last_price", 0)
        fv.close = fv.current_price
        fv.bid = ticker.get("bid", 0)
        fv.ask = ticker.get("ask", 0)
        fv.high_24h = ticker.get("high_24h", 0)
        fv.low_24h = ticker.get("low_24h", 0)
        fv.volume_24h = ticker.get("volume_24h", 0)
        fv.change_24h_pct = ticker.get("change_24h_pct", 0)
        fv.spread_pct = ticker.get("spread_pct", 0)

    # Technical indicators
    if indicators:
        for key, value in indicators.items():
            if hasattr(fv, key):
                setattr(fv, key, value)
        if fv.current_price <= 0 and fv.close > 0:
            fv.current_price = fv.close
        if fv.close <= 0 and fv.current_price > 0:
            fv.close = fv.current_price
        if fv.current_price > 0 and fv.close > 0:
            price_gap = abs(fv.current_price - fv.close) / max(fv.close, 1e-12)
            if price_gap > 0.20:
                fv.current_price = fv.close
                if fv.bid <= 0 or abs(fv.bid - fv.close) / max(fv.close, 1e-12) > 0.20:
                    fv.bid = fv.close
                if fv.ask <= 0 or abs(fv.ask - fv.close) / max(fv.close, 1e-12) > 0.20:
                    fv.ask = fv.close
                fv.spread_pct = 0.0
        if fv.bid <= 0 and fv.current_price > 0:
            fv.bid = fv.current_price
        if fv.ask <= 0 and fv.current_price > 0:
            fv.ask = fv.current_price

    # Sentiment
    if sentiment_data:
        fv.news_sentiment_avg = sentiment_data.get("news_sentiment", 0.0)
        fv.social_sentiment_avg = sentiment_data.get("social_sentiment", 0.0)
        fv.social_mention_count = sentiment_data.get("mention_count", 0)
        fv.news_article_count = sentiment_data.get("article_count", 0)
        fv.headline_count = sentiment_data.get("headline_count", fv.news_article_count)
        fv.sentiment_data_available = bool(sentiment_data.get("sentiment_data_available", False))
        fv.direct_sentiment_data_available = bool(
            sentiment_data.get("direct_sentiment_data_available", False)
        )
        fv.direct_news_item_count = int(sentiment_data.get("direct_news_item_count", 0) or 0)
        fv.market_news_item_count = int(sentiment_data.get("market_news_item_count", 0) or 0)
        fv.news_sources = list(sentiment_data.get("news_sources") or [])
        fv.recent_news_items = list(sentiment_data.get("news_items") or [])[:20]

    # Headlines
    if headlines:
        fv.recent_headlines = headlines

    if derivatives:
        for key, value in derivatives.items():
            if hasattr(fv, key):
                setattr(fv, key, value)

    return fv
