"""
Abstract executor interface.
All executors (paper, OKX live) must implement this.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_brain.base_model import DecisionOutput


class OrderStatus(StrEnum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class ExecutionResult:
    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: float
    status: OrderStatus
    fee: float = 0.0
    pnl: float = 0.0
    exchange_order_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    raw_response: dict | None = None


class AbstractExecutor(ABC):
    """Interface for trade execution."""

    @abstractmethod
    async def place_order(
        self, decision: DecisionOutput, account_id: str | None = None
    ) -> ExecutionResult:
        """Execute a trading decision and return the result."""

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order."""

    @abstractmethod
    async def get_balance(self, asset: str = "USDT") -> float:
        """Get available balance for an asset."""

    @abstractmethod
    async def get_positions(self, symbol: str | None = None) -> list[dict]:
        """Get current open positions."""

    async def get_positions_strict(self, symbol: str | None = None) -> list[dict]:
        """Get positions and propagate failures for safety-critical callers."""
        return await self.get_positions(symbol)

    @abstractmethod
    async def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        """Get currently open orders."""

    async def get_open_orders_strict(self, symbol: str | None = None) -> list[dict]:
        """Get open orders and propagate failures for safety-critical callers."""
        return await self.get_open_orders(symbol)

    @abstractmethod
    async def initialize(self) -> None:
        """Connect to exchange / init virtual accounts."""

    @abstractmethod
    async def shutdown(self) -> None:
        """Disconnect / cleanup."""
