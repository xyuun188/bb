"""
Custom exception hierarchy for the trading system.
Each layer has its own exception base class for precise error handling.
"""


class TradingSystemError(Exception):
    """Root exception for the entire trading system."""


# --- Data Feed Errors ---
class DataFeedError(TradingSystemError):
    """Base for data feed layer errors."""


class WebSocketConnectionError(DataFeedError):
    """OKX WebSocket connection failure or unexpected disconnect."""


class NewsFetchError(DataFeedError):
    """Failed to fetch news from a source."""


# --- AI Brain Errors ---
class AIBrainError(TradingSystemError):
    """Base for AI decision layer errors."""


class ModelInferenceError(AIBrainError):
    """A model failed to produce a decision."""


class LLMResponseParseError(ModelInferenceError):
    """LLM returned a response that couldn't be parsed as valid JSON."""


class SentimentAnalysisError(AIBrainError):
    """Sentiment scoring pipeline failed."""


# --- Executor Errors ---
class ExecutorError(TradingSystemError):
    """Base for trade execution errors."""


class OrderPlacementError(ExecutorError):
    """Order was rejected or failed to place."""


class OrderCancellationError(ExecutorError):
    """Failed to cancel an order."""


class InsufficientBalanceError(ExecutorError):
    """Not enough balance to place the order."""


class ExchangeAPIError(ExecutorError):
    """OKX API returned an error response with optional structured facts."""

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        payload: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code or "") or None
        self.payload = dict(payload) if isinstance(payload, dict) else None


class RateLimitError(ExchangeAPIError):
    """OKX API rate limit exceeded."""


# --- Risk Manager Errors ---
class RiskManagerError(TradingSystemError):
    """Base for risk management errors."""


class PositionLimitExceeded(RiskManagerError):
    """Trade would exceed position size limit."""


class DailyLossLimitReached(RiskManagerError):
    """Daily maximum loss has been reached — trading halted."""


class BlackSwanTriggered(RiskManagerError):
    """Extreme negative sentiment detected — forced position closure."""


class CircuitBreakerTripped(RiskManagerError):
    """Circuit breaker activated — all trading suspended."""


# --- Configuration Errors ---
class ConfigError(TradingSystemError):
    """Invalid or missing configuration."""
