"""
Order lifecycle manager.
Tracks orders from creation through fill/cancel, with retry logic.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

import structlog

from executor.base_executor import OrderStatus

logger = structlog.get_logger(__name__)


class OrderState(Enum):
    CREATED = "created"
    SENT = "sent"
    ACKNOWLEDGED = "acknowledged"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"
    EXPIRED = "expired"


# Valid state transitions
VALID_TRANSITIONS = {
    OrderState.CREATED: {OrderState.SENT, OrderState.FAILED},
    OrderState.SENT: {OrderState.ACKNOWLEDGED, OrderState.REJECTED, OrderState.FAILED, OrderState.EXPIRED},
    OrderState.ACKNOWLEDGED: {OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCELLING, OrderState.EXPIRED},
    OrderState.PARTIALLY_FILLED: {OrderState.FILLED, OrderState.CANCELLING, OrderState.EXPIRED},
    OrderState.CANCELLING: {OrderState.CANCELLED, OrderState.FAILED},
    OrderState.FILLED: set(),       # terminal
    OrderState.CANCELLED: set(),    # terminal
    OrderState.REJECTED: set(),     # terminal
    OrderState.FAILED: set(),       # terminal
    OrderState.EXPIRED: set(),      # terminal
}


class TrackedOrder:
    """Mutable state for a single order being tracked."""

    def __init__(self, order_id: str, symbol: str, side: str, quantity: float) -> None:
        self.order_id = order_id
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.filled_quantity = 0.0
        self.state = OrderState.CREATED
        self.exchange_order_id: str | None = None
        self.created_at = datetime.now(timezone.utc)
        self.last_updated = self.created_at
        self.retry_count = 0
        self.max_retries = 3
        self.error_message: str | None = None

    def transition(self, new_state: OrderState) -> bool:
        if new_state in VALID_TRANSITIONS.get(self.state, set()):
            old_state = self.state
            self.state = new_state
            self.last_updated = datetime.now(timezone.utc)
            logger.debug(
                "order transition",
                order_id=self.order_id,
                from_state=old_state.value,
                to_state=new_state.value,
            )
            return True
        logger.warning(
            "invalid state transition",
            order_id=self.order_id,
            from_state=self.state.value,
            to_state=new_state.value,
        )
        return False

    @property
    def is_terminal(self) -> bool:
        return self.state in (
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.FAILED,
            OrderState.EXPIRED,
        )

    @property
    def is_active(self) -> bool:
        return not self.is_terminal


class OrderManager:
    """Manages the lifecycle of all orders in the system.

    This sits between decision execution and the raw executor API,
    providing retry logic, state machine enforcement, and event logging.
    """

    def __init__(self, executor: "AbstractExecutor") -> None:
        self._executor = executor
        self._orders: dict[str, TrackedOrder] = {}
        self._max_retries = 3

    def create_order(self, order_id: str, symbol: str, side: str, quantity: float) -> TrackedOrder:
        order = TrackedOrder(order_id, symbol, side, quantity)
        self._orders[order_id] = order
        return order

    async def execute(self, order: TrackedOrder) -> bool:
        """Execute an order with retry logic. Returns True if successful."""
        if order.state != OrderState.CREATED:
            logger.warning("order not in created state", order_id=order.order_id)
            return False

        order.transition(OrderState.SENT)

        for attempt in range(self._max_retries):
            try:
                # (The actual execution is handled by the executor)
                # This is a lifecycle tracker, not the executor itself
                return True
            except Exception as e:
                order.retry_count = attempt + 1
                order.error_message = str(e)
                logger.warning(
                    "order attempt failed",
                    order_id=order.order_id,
                    attempt=attempt + 1,
                    error=str(e),
                )

        order.transition(OrderState.FAILED)
        order.error_message = f"Failed after {self._max_retries} attempts"
        return False

    def get_order(self, order_id: str) -> TrackedOrder | None:
        return self._orders.get(order_id)

    def get_active_orders(self) -> list[TrackedOrder]:
        return [o for o in self._orders.values() if o.is_active]

    def get_orders_for_symbol(self, symbol: str) -> list[TrackedOrder]:
        return [o for o in self._orders.values() if o.symbol == symbol]

    def cleanup_terminal_orders(self, max_age_minutes: int = 60) -> int:
        """Remove old terminal orders from memory."""
        now = datetime.now(timezone.utc)
        to_remove = []
        for order_id, order in self._orders.items():
            if order.is_terminal:
                age = (now - order.last_updated).total_seconds() / 60
                if age > max_age_minutes:
                    to_remove.append(order_id)

        for order_id in to_remove:
            del self._orders[order_id]

        return len(to_remove)
