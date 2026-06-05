"""
Abstract executor interface.
All executors (paper, OKX live) must implement this.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class OrderStatus(str, Enum):
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
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    raw_response: dict | None = None


class AbstractExecutor(ABC):
    """Interface for trade execution."""

    @abstractmethod
    async def place_order(
        self, decision: "DecisionOutput", account_id: str | None = None
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

    @abstractmethod
    async def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        """Get currently open orders."""

    @abstractmethod
    async def initialize(self) -> None:
        """Connect to exchange / init virtual accounts."""

    @abstractmethod
    async def shutdown(self) -> None:
        """Disconnect / cleanup."""
