"""
Real-time position tracker.
Maintains an in-memory cache of open positions with real-time PnL.
Queries the exchange or paper executor on startup to sync state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from core.safe_output import safe_error_text

if TYPE_CHECKING:
    from executor.base_executor import AbstractExecutor

logger = structlog.get_logger(__name__)


class PositionTracker:
    """In-memory cache of current positions with real-time PnL tracking.

    Provides:
    - O(1) lookup of positions by symbol
    - Real-time unrealized PnL updates as prices change
    - Stop-loss and take-profit breach detection
    """

    def __init__(self) -> None:
        self._positions: dict[str, list[dict]] = {}  # model_name -> positions
        self._price_cache: dict[str, float] = {}  # symbol -> last price

    async def sync_from_executor(
        self, executor: AbstractExecutor, model_name: str = "default"
    ) -> None:
        """Load current positions from the executor."""
        try:
            positions = await executor.get_positions()
            self._positions[model_name] = []
            for pos in positions:
                self._positions[model_name].append(
                    {
                        "symbol": pos.get("symbol", ""),
                        "side": pos.get("side", "long"),
                        "quantity": pos.get("contracts", pos.get("quantity", 0)),
                        "entry_price": pos.get("entryPrice", pos.get("entry_price", 0)),
                        "leverage": pos.get("leverage", 1.0),
                        "unrealized_pnl": pos.get("unrealizedPnl", 0),
                        "is_open": True,
                    }
                )
            logger.info(
                "positions synced", model=model_name, count=len(self._positions[model_name])
            )
        except Exception as e:
            logger.error(
                "position sync failed",
                model=model_name,
                error=safe_error_text(e),
            )

    def update_price(self, symbol: str, price: float) -> None:
        """Update the cached price and recalculate all PnLs."""
        self._price_cache[symbol] = price

        for _model_name, positions in self._positions.items():
            for pos in positions:
                if pos["symbol"] == symbol and pos["is_open"]:
                    entry = pos["entry_price"]
                    qty = pos["quantity"]
                    if pos["side"] == "long":
                        pos["unrealized_pnl"] = (price - entry) * qty
                    else:
                        pos["unrealized_pnl"] = (entry - price) * qty

    def get_positions(self, model_name: str | None = None) -> list[dict]:
        if model_name:
            return self._positions.get(model_name, [])
        all_pos = []
        for positions in self._positions.values():
            all_pos.extend(positions)
        return all_pos

    def get_position_for_symbol(self, symbol: str, model_name: str) -> dict | None:
        positions = self._positions.get(model_name, [])
        for pos in positions:
            if pos["symbol"] == symbol and pos["is_open"]:
                return pos
        return None

    def get_total_exposure(self, model_name: str) -> float:
        positions = self._positions.get(model_name, [])
        return sum(
            abs(p["quantity"] * self._price_cache.get(p["symbol"], p["entry_price"]))
            for p in positions
            if p["is_open"]
        )

    def get_total_unrealized_pnl(self, model_name: str) -> float:
        return sum(p.get("unrealized_pnl", 0) for p in self._positions.get(model_name, []))

    def check_stop_triggers(self, model_name: str) -> list[dict]:
        """Find positions that have breached stop-loss or take-profit levels."""
        triggered = []
        for pos in self._positions.get(model_name, []):
            if not pos["is_open"]:
                continue
            current = self._price_cache.get(pos["symbol"], pos["entry_price"])

            stop_loss = pos.get("stop_loss")
            take_profit = pos.get("take_profit")

            if stop_loss:
                if pos["side"] == "long" and current <= stop_loss:
                    triggered.append({**pos, "trigger": "stop_loss", "trigger_price": stop_loss})
                elif pos["side"] == "short" and current >= stop_loss:
                    triggered.append({**pos, "trigger": "stop_loss", "trigger_price": stop_loss})

            if take_profit:
                if pos["side"] == "long" and current >= take_profit:
                    triggered.append(
                        {**pos, "trigger": "take_profit", "trigger_price": take_profit}
                    )
                elif pos["side"] == "short" and current <= take_profit:
                    triggered.append(
                        {**pos, "trigger": "take_profit", "trigger_price": take_profit}
                    )

        return triggered

    def add_position(self, model_name: str, position: dict) -> None:
        self._positions.setdefault(model_name, []).append(position)

    def remove_position(self, model_name: str, position_id: str) -> bool:
        positions = self._positions.get(model_name, [])
        for _i, pos in enumerate(positions):
            if pos.get("id") == position_id:
                pos["is_open"] = False
                return True
        return False

    def get_summary(self) -> dict[str, Any]:
        summaries = {}
        for model_name, positions in self._positions.items():
            open_positions = [p for p in positions if p["is_open"]]
            total_pnl = sum(p.get("unrealized_pnl", 0) for p in open_positions)
            summaries[model_name] = {
                "open_positions": len(open_positions),
                "total_pnl": total_pnl,
                "positions": open_positions,
            }
        return summaries
