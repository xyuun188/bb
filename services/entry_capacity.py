"""Entry capacity policy for staged and open positions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import Action, DecisionOutput

NormalizeSymbol = Callable[[Any], str | None]
MaxOpenPositionsProvider = Callable[[], int]


@dataclass(frozen=True, slots=True)
class EntryCapacityPolicy:
    """Limit new entries without blocking same-symbol adds that manage an existing position."""

    normalize_symbol: NormalizeSymbol
    max_open_positions_per_model_provider: MaxOpenPositionsProvider

    def empty_staged_counts(self) -> dict[str, dict[Any, int]]:
        """Return the per-round staged-entry counters used before orders are submitted."""

        return {"model_totals": {}, "symbol_side": {}, "side_totals": {}}

    def reason(
        self,
        model_name: str,
        decision: DecisionOutput,
        open_positions: list[dict],
        staged_entry_counts: dict[str, dict],
    ) -> str | None:
        if not decision.is_entry:
            return None

        side = "long" if decision.action == Action.LONG else "short"
        symbol_key = self.normalize_symbol(decision.symbol)
        staged_symbol_side = staged_entry_counts.get("symbol_side", {})
        staged_model_totals = staged_entry_counts.get("model_totals", {})
        existing_same_symbol = sum(
            1
            for position in open_positions
            if position.get("model_name") == model_name
            and self.normalize_symbol(position.get("symbol")) == symbol_key
            and position.get("side") == side
        )
        staged_key = (model_name, symbol_key, side)
        existing_same_symbol += int(staged_symbol_side.get(staged_key, 0))
        is_same_symbol_add = existing_same_symbol > 0

        model_open_count = sum(
            1 for position in open_positions if position.get("model_name") == model_name
        )
        model_open_count += int(staged_model_totals.get(model_name, 0))
        max_open_positions = int(self.max_open_positions_per_model_provider() or 0)
        if (
            not is_same_symbol_add
            and max_open_positions > 0
            and model_open_count >= max_open_positions
        ):
            return (
                "当前持仓数已达上限，暂停新开仓。"
                f"当前 {model_open_count} 笔，限制 {max_open_positions} 笔。"
            )
        return None

    def reserve_slot(
        self,
        model_name: str,
        decision: DecisionOutput,
        staged_entry_counts: dict[str, dict[Any, int]],
    ) -> None:
        """Reserve capacity for an entry selected earlier in the current round."""

        if not decision.is_entry:
            return

        staged_entry_counts.setdefault("model_totals", {})
        staged_entry_counts.setdefault("symbol_side", {})
        staged_entry_counts.setdefault("side_totals", {})

        staged_entry_counts["model_totals"][model_name] = (
            int(staged_entry_counts["model_totals"].get(model_name, 0)) + 1
        )
        side = "long" if decision.action == Action.LONG else "short"
        staged_entry_counts["side_totals"][side] = (
            int(staged_entry_counts["side_totals"].get(side, 0)) + 1
        )
        staged_key = (model_name, self.normalize_symbol(decision.symbol), side)
        staged_entry_counts["symbol_side"][staged_key] = (
            int(staged_entry_counts["symbol_side"].get(staged_key, 0)) + 1
        )
