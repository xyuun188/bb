"""Position profit-peak context for the AI position-review prompt."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

NormalizeSymbol = Callable[[Any], str]
AggregatePositionGroup = Callable[[list[dict[str, Any]], str, str, str], dict[str, Any]]
PositionPeakKeyProvider = Callable[[str, str, str], Any]
PositionPeaksProvider = Callable[[], Mapping[Any, dict[str, Any]]]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class PositionProfitPeakContextPolicy:
    """Expose per-position floating-profit peaks without coupling to TradingService."""

    normalize_symbol: NormalizeSymbol
    aggregate_position_group: AggregatePositionGroup
    position_peak_key: PositionPeakKeyProvider
    position_peaks_provider: PositionPeaksProvider
    default_model_name: str

    def context(
        self,
        model_name: str,
        symbol: str,
        positions: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        if not positions:
            return {}
        normalized = self.normalize_symbol(symbol)
        model = str(model_name or self.default_model_name)
        best: dict[str, Any] = {}
        by_side: dict[str, list[dict[str, Any]]] = {}
        for pos in positions or []:
            side = str(pos.get("side") or "").lower()
            if side not in {"long", "short"}:
                continue
            by_side.setdefault(side, []).append(pos)

        peaks = self.position_peaks_provider()
        for side, side_positions in by_side.items():
            pos = self.aggregate_position_group(side_positions, model, normalized or symbol, side)
            if not pos:
                continue
            key = self.position_peak_key(
                model, normalized or str(pos.get("symbol") or symbol), side
            )
            state = peaks.get(key) or {}
            peak = _safe_float(
                state.get("peak_unrealized_pnl", state.get("peak_pnl")),
                0.0,
            )
            current = _safe_float(pos.get("unrealized_pnl"), 0.0)
            peak = max(peak, current)
            if peak <= 0 and current <= 0:
                continue
            retrace_abs = max(peak - current, 0.0)
            retrace_ratio = retrace_abs / max(peak, 1e-9) if peak > 0 else 0.0
            item = {
                "model_name": model,
                "symbol": normalized or str(pos.get("symbol") or symbol),
                "side": side,
                "rows": len(side_positions),
                "quantity": round(_safe_float(pos.get("quantity"), 0.0), 8),
                "notional": round(_safe_float(pos.get("notional"), 0.0), 6),
                "peak_unrealized_pnl": round(peak, 6),
                "current_unrealized_pnl": round(current, 6),
                "profit_retrace_abs": round(retrace_abs, 6),
                "profit_retrace_ratio": round(retrace_ratio, 6),
                "peak_pnl_ratio": round(_safe_float(state.get("peak_pnl_ratio"), 0.0), 6),
                "updated_at": state.get("updated_at"),
            }
            if not best or _safe_float(item.get("profit_retrace_abs"), 0.0) > _safe_float(
                best.get("profit_retrace_abs"),
                0.0,
            ):
                best = item
        return best
