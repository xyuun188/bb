"""
Circuit breaker — halts all trading when risk thresholds are breached.
Protects against cascading losses from runaway algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum

import structlog

from config.settings import settings

logger = structlog.get_logger(__name__)


class BreakerState(StrEnum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Trading halted
    HALF_OPEN = "half_open"  # Testing if we can resume


@dataclass
class CircuitBreakerState:
    state: BreakerState = BreakerState.CLOSED
    tripped_at: datetime | None = None
    tripped_reason: str = ""
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    total_trades_today: int = 0
    last_reset_date: datetime = field(default_factory=lambda: datetime.now(UTC))


class CircuitBreaker:
    """Monitors risk metrics and trips when thresholds are breached.

    When OPEN:
    - All new position entries are rejected.
    - Existing positions may only be closed.
    - Dashboard shows HALTED status.

    Automatic reset: After cooldown period (default 1 hour), transitions to
    HALF_OPEN. If the next trade is profitable, transitions back to CLOSED.
    """

    def __init__(self, cooldown_minutes: int = 60) -> None:
        self.max_consecutive_losses = 5
        self.cooldown = timedelta(minutes=cooldown_minutes)
        self._state = CircuitBreakerState()

    @property
    def max_daily_loss_pct(self) -> float:
        return float(settings.max_daily_loss_pct or 0.05)

    @property
    def is_open(self) -> bool:
        self._check_daily_reset()
        # Auto-transition from OPEN to HALF_OPEN after cooldown
        if (
            self._state.state == BreakerState.OPEN
            and self._state.tripped_at
            and datetime.now(UTC) - self._state.tripped_at > self.cooldown
        ):
            self._state.state = BreakerState.HALF_OPEN
            logger.info("circuit breaker cooling down", state="half_open")
        return self._state.state == BreakerState.OPEN

    @property
    def is_half_open(self) -> bool:
        return self._state.state == BreakerState.HALF_OPEN

    @property
    def is_closed(self) -> bool:
        return self._state.state == BreakerState.CLOSED

    @property
    def state(self) -> BreakerState:
        self._check_daily_reset()
        return self._state.state

    def can_open_position(self) -> bool:
        """Check if new positions are allowed."""
        return self._state.state == BreakerState.CLOSED

    def can_close_position(self) -> bool:
        """Closing positions is always allowed."""
        return True

    def record_trade(self, pnl: float) -> None:
        """Record a completed trade result."""
        self._check_daily_reset()
        self._state.daily_pnl += pnl
        self._state.total_trades_today += 1

        if pnl < 0:
            self._state.consecutive_losses += 1
        else:
            self._state.consecutive_losses = 0
            # If we had a profitable trade in HALF_OPEN, reset
            if self._state.state == BreakerState.HALF_OPEN:
                self._state.state = BreakerState.CLOSED
                self._state.tripped_at = None
                self._state.tripped_reason = ""
                logger.info("circuit breaker reset to closed")

        self._evaluate()

    def _evaluate(self) -> None:
        """Check if breaker should be tripped."""
        if self._state.state != BreakerState.CLOSED:
            return

        # Check consecutive losses
        if self._state.consecutive_losses >= self.max_consecutive_losses:
            self._trip(f"Consecutive losses: {self._state.consecutive_losses}")

    def evaluate_daily_loss(self, account_balance: float) -> None:
        """Check if daily loss limit is breached."""
        self._check_daily_reset()
        if self._state.state != BreakerState.CLOSED:
            return

        loss_pct = abs(self._state.daily_pnl) / account_balance if account_balance > 0 else 0
        if self._state.daily_pnl < 0 and loss_pct > self.max_daily_loss_pct:
            self._trip(
                f"Daily loss limit reached: {self._state.daily_pnl:.2f} USD "
                f"({loss_pct*100:.1f}% > {self.max_daily_loss_pct*100:.1f}%)"
            )

    def _trip(self, reason: str) -> None:
        self._state.state = BreakerState.OPEN
        self._state.tripped_at = datetime.now(UTC)
        self._state.tripped_reason = reason
        logger.warning("circuit breaker tripped", reason=reason)

    def reset(self) -> None:
        """Manual reset (e.g., from dashboard)."""
        self._state = CircuitBreakerState()
        logger.info("circuit breaker manually reset")

    def _check_daily_reset(self) -> None:
        """Reset daily counters if it's a new day."""
        now = datetime.now(UTC)
        if now.date() > self._state.last_reset_date.date():
            self._state.daily_pnl = 0.0
            self._state.total_trades_today = 0
            self._state.last_reset_date = now
            logger.info("daily risk counters reset")

    def get_state(self) -> dict:
        self._check_daily_reset()
        return {
            "breaker_state": self._state.state.value,
            "tripped_at": self._state.tripped_at.isoformat() if self._state.tripped_at else None,
            "tripped_reason": self._state.tripped_reason,
            "daily_pnl": round(self._state.daily_pnl, 2),
            "consecutive_losses": self._state.consecutive_losses,
            "total_trades_today": self._state.total_trades_today,
            "can_open": self.can_open_position(),
        }
