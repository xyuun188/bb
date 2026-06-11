from __future__ import annotations

from collections.abc import Callable
from typing import Any


class MemoryPositionStore:
    """Maintain in-memory paper positions without exposing TradingService internals."""

    def __init__(
        self,
        *,
        paper_executor_provider: Callable[[], Any | None],
        symbol_normalizer: Callable[[Any], str],
    ) -> None:
        self.paper_executor_provider = paper_executor_provider
        self.symbol_normalizer = symbol_normalizer

    def remove_open_position(self, model_name: str, symbol: str, side: str) -> None:
        paper_executor = self.paper_executor_provider()
        if not paper_executor:
            return
        positions = paper_executor._positions.get(model_name, [])
        target_symbol = self.symbol_normalizer(symbol)
        paper_executor._positions[model_name] = [
            position
            for position in positions
            if not (
                self.symbol_normalizer(position.get("symbol")) == target_symbol
                and position.get("side") == side
                and position.get("is_open")
            )
        ]
