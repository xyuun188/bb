"""
Stop-loss logic: hard stop-loss and trailing stop.
Evaluated on every tick for open positions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import structlog

from config.settings import settings

logger = structlog.get_logger(__name__)


class StopLossType(StrEnum):
    NONE = "none"
    HARD = "hard"
    TRAILING = "trailing"


@dataclass
class StopLossResult:
    triggered: bool
    stop_type: StopLossType = StopLossType.NONE
    exit_price: float = 0.0
    reason: str = ""
    loss_pct: float = 0.0


@dataclass
class TrailingStopState:
    """Mutable state tracking the trailing stop level for a position."""

    symbol: str
    side: str  # long or short
    entry_price: float
    highest_price: float  # highest seen since entry (for longs)
    lowest_price: float  # lowest seen since entry (for shorts)
    activation_price: float  # price at which trailing stop activates
    stop_price: float
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class StopLossManager:
    """Manages stop-loss evaluation for all open positions.

    Two modes:
    - Hard stop: Fixed price level from entry. Always active.
    - Trailing stop: Activates after price moves favorably by activation_pct,
      then trails by trailing_distance_pct.
    """

    def __init__(self) -> None:
        self._trailing_states: dict[str, TrailingStopState] = {}

    @property
    def hard_stop_loss_pct(self) -> float:
        return float(settings.hard_stop_loss_pct or 0.05)

    @property
    def trailing_activation(self) -> float:
        return float(settings.trailing_stop_activation or 0.03)

    @property
    def trailing_distance(self) -> float:
        return float(settings.trailing_stop_distance or 0.015)

    def init_trailing_stop(
        self, symbol: str, side: str, entry_price: float, current_price: float
    ) -> None:
        """Initialize trailing stop tracking for a new position."""
        if side == "long":
            activation = entry_price * (1 + self.trailing_activation)
            stop = entry_price * (1 - self.trailing_distance)
            state = TrailingStopState(
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                highest_price=max(entry_price, current_price),
                lowest_price=current_price,
                activation_price=activation,
                stop_price=stop,
            )
        else:  # short
            activation = entry_price * (1 - self.trailing_activation)
            stop = entry_price * (1 + self.trailing_distance)
            state = TrailingStopState(
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                highest_price=current_price,
                lowest_price=min(entry_price, current_price),
                activation_price=activation,
                stop_price=stop,
            )

        self._trailing_states[symbol] = state
        logger.debug("trailing stop initialized", symbol=symbol, side=side, stop=stop)

    def remove_trailing_stop(self, symbol: str) -> None:
        self._trailing_states.pop(symbol, None)

    def evaluate(
        self, symbol: str, side: str, entry_price: float, current_price: float
    ) -> StopLossResult:
        """Evaluate stop-loss conditions for a position.

        Returns StopLossResult indicating whether to exit and at what price.
        """
        # 1. Check hard stop-loss
        if side == "long":
            hard_stop_price = entry_price * (1 - self.hard_stop_loss_pct)
            if current_price <= hard_stop_price:
                loss_pct = (current_price - entry_price) / entry_price
                return StopLossResult(
                    triggered=True,
                    stop_type=StopLossType.HARD,
                    exit_price=current_price,
                    reason=f"Hard stop-loss triggered: {current_price:.4f} <= {hard_stop_price:.4f}",
                    loss_pct=loss_pct,
                )
        else:  # short
            hard_stop_price = entry_price * (1 + self.hard_stop_loss_pct)
            if current_price >= hard_stop_price:
                loss_pct = (entry_price - current_price) / entry_price
                return StopLossResult(
                    triggered=True,
                    stop_type=StopLossType.HARD,
                    exit_price=current_price,
                    reason=f"Hard stop-loss triggered: {current_price:.4f} >= {hard_stop_price:.4f}",
                    loss_pct=loss_pct,
                )

        # 2. Check trailing stop
        state = self._trailing_states.get(symbol)
        if state is None:
            return StopLossResult(triggered=False)

        if side == "long":
            # Update highest price seen
            if current_price > state.highest_price:
                state.highest_price = current_price

            # Check if trailing stop is active
            if state.highest_price >= state.activation_price:
                # Update trailing stop level
                new_stop = state.highest_price * (1 - self.trailing_distance)
                if new_stop > state.stop_price:
                    state.stop_price = new_stop

                # Check if price has fallen below trailing stop
                if current_price <= state.stop_price:
                    loss_pct = (current_price - entry_price) / entry_price
                    return StopLossResult(
                        triggered=True,
                        stop_type=StopLossType.TRAILING,
                        exit_price=current_price,
                        reason=f"Trailing stop triggered: {current_price:.4f} <= {state.stop_price:.4f}",
                        loss_pct=loss_pct,
                    )
        else:  # short
            if current_price < state.lowest_price:
                state.lowest_price = current_price

            if state.lowest_price <= state.activation_price:
                new_stop = state.lowest_price * (1 + self.trailing_distance)
                if new_stop < state.stop_price:
                    state.stop_price = new_stop

                if current_price >= state.stop_price:
                    loss_pct = (entry_price - current_price) / entry_price
                    return StopLossResult(
                        triggered=True,
                        stop_type=StopLossType.TRAILING,
                        exit_price=current_price,
                        reason=f"Trailing stop triggered: {current_price:.4f} >= {state.stop_price:.4f}",
                        loss_pct=loss_pct,
                    )

        return StopLossResult(triggered=False)

    def get_active_stops(self) -> dict[str, dict[str, Any]]:
        """Return all active trailing stops for dashboard display."""
        return {
            sym: {
                "side": s.side,
                "entry": s.entry_price,
                "stop": s.stop_price,
                "activation": s.activation_price,
                "highest": s.highest_price if s.side == "long" else s.lowest_price,
            }
            for sym, s in self._trailing_states.items()
        }
