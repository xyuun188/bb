"""
Abstract base class for all AI trading models.
Every model must implement this interface to participate in the competition.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from data_feed.feature_vector import FeatureVector


class Action(StrEnum):
    LONG = "long"
    SHORT = "short"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
    HOLD = "hold"

    @classmethod
    def from_string(cls, s: str) -> Action:
        s = s.lower().strip()
        mapping = {
            "long": cls.LONG,
            "buy": cls.LONG,
            "open_long": cls.LONG,
            "short": cls.SHORT,
            "sell": cls.SHORT,
            "open_short": cls.SHORT,
            "close_long": cls.CLOSE_LONG,
            "close_short": cls.CLOSE_SHORT,
            "close": cls.CLOSE_LONG,  # default close = close long
            "hold": cls.HOLD,
            "wait": cls.HOLD,
            "none": cls.HOLD,
        }
        return mapping.get(s, cls.HOLD)

    def is_entry(self) -> bool:
        return self in (Action.LONG, Action.SHORT)

    def is_exit(self) -> bool:
        return self in (Action.CLOSE_LONG, Action.CLOSE_SHORT)


@dataclass
class DecisionOutput:
    """Standardized output from any AI model.

    This is the contract between the AI Brain and the Executor.
    """

    model_name: str
    symbol: str
    action: Action
    confidence: float  # 0.0 to 1.0
    reasoning: str  # human-readable explanation
    position_size_pct: float = 0.0  # fraction of available capital
    suggested_leverage: float = 1.0
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0
    suggested_holding_minutes: float = 0.0
    maximum_holding_minutes: float = 0.0
    suggested_close_fraction: float = 0.0
    cross_check_for: dict[str, Any] | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    raw_response: dict | None = None  # for debugging LLM models
    feature_snapshot: dict | None = None  # features at decision time

    @property
    def is_entry(self) -> bool:
        return self.action.is_entry()

    @property
    def is_exit(self) -> bool:
        return self.action.is_exit()

    @property
    def is_hold(self) -> bool:
        return self.action == Action.HOLD

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "symbol": self.symbol,
            "action": self.action.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "position_size_pct": self.position_size_pct,
            "suggested_leverage": self.suggested_leverage,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "suggested_holding_minutes": self.suggested_holding_minutes,
            "maximum_holding_minutes": self.maximum_holding_minutes,
            "suggested_close_fraction": self.suggested_close_fraction,
            "cross_check_for": self.cross_check_for,
        }


class AbstractAIModel(ABC):
    """All AI trading models must implement this interface.

    Adding a new model:
    1. Subclass AbstractAIModel
    2. Implement decide()
    3. Register with ModelRegistry
    """

    name: str  # Set in subclass, e.g., "trend_expert" or "risk_expert"

    @abstractmethod
    async def initialize(self) -> None:
        """Load models, connect to APIs, warm up caches."""

    @abstractmethod
    async def decide(self, features: FeatureVector, context: dict[str, Any]) -> DecisionOutput:
        """Given current market state and context, produce a trading decision.

        Args:
            features: The FeatureVector with all current market/sentiment data.
            context: Additional context (e.g., existing positions, account state).

        Returns:
            DecisionOutput with action, confidence, sizing, and reasoning.
        """

    @abstractmethod
    async def shutdown(self) -> None:
        """Release resources (model memory, API connections)."""

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, *args):
        await self.shutdown()
