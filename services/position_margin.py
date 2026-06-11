from __future__ import annotations

from typing import Any


class PositionMarginCalculator:
    """Calculate isolated margin for position accounting."""

    @staticmethod
    def margin(notional_value: Any, leverage: Any) -> float:
        notional = _to_float(notional_value, 0.0)
        lev = max(_to_float(leverage, 1.0), 1.0)
        return notional / lev


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default
