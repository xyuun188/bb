"""Existing winning-position context for entry sizing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import Action, DecisionOutput


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class EntryExistingWinnerContextPolicy:
    """Summarize same-symbol, same-side open winners before adding to a position."""

    normalize_symbol: Callable[[Any], str]

    def context(
        self,
        decision: DecisionOutput,
        open_positions: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        if not decision.is_entry:
            return {"has_winner": False}

        side = "long" if decision.action == Action.LONG else "short"
        symbol_key = self.normalize_symbol(decision.symbol)
        matches = [
            pos
            for pos in (open_positions or [])
            if self.normalize_symbol(pos.get("symbol")) == symbol_key
            and str(pos.get("side") or "").lower() == side
            and pos.get("is_open", True)
        ]
        if not matches:
            return {"has_winner": False}

        total_notional = 0.0
        total_unrealized = 0.0
        total_quantity = 0.0
        for pos in matches:
            entry = _safe_float(pos.get("entry_price"), 0.0)
            current = _safe_float(pos.get("current_price"), entry)
            qty = abs(_safe_float(pos.get("quantity"), 0.0))
            contract_size = _safe_float(
                pos.get("contract_size") or pos.get("contractSize"),
                1.0,
            )
            direct_notional = abs(
                _safe_float(
                    pos.get("notional")
                    or pos.get("notional_usd")
                    or pos.get("notionalUsd")
                    or (pos.get("info") or {}).get("notionalUsd")
                    or (pos.get("info") or {}).get("notional")
                    or (pos.get("info") or {}).get("posValue"),
                    0.0,
                )
            )
            notional = (
                direct_notional
                if direct_notional > 0
                else qty * max(entry, current, 0.0) * (contract_size if contract_size > 0 else 1.0)
            )
            total_notional += max(notional, 0.0)
            total_quantity += qty
            total_unrealized += _safe_float(pos.get("unrealized_pnl"), 0.0)

        pnl_ratio = total_unrealized / max(total_notional, 1e-9)
        return {
            "has_winner": bool(total_unrealized > 0),
            "symbol": symbol_key,
            "side": side,
            "positions": len(matches),
            "quantity": round(total_quantity, 8),
            "notional_usdt": round(total_notional, 6),
            "unrealized_pnl": round(total_unrealized, 6),
            "pnl_ratio": round(pnl_ratio, 6),
        }
