"""Entry protection prices from current executable quotes."""

from __future__ import annotations

from math import isfinite
from typing import Any


def entry_protection_prices(
    *,
    side: str,
    reference_price: float,
    stop_loss_fraction: float,
    take_profit_fraction: float,
    current_quote: dict[str, Any] | None = None,
) -> tuple[float, float]:
    def positive(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        return number if isfinite(number) and number > 0.0 else 0.0

    stop = positive(stop_loss_fraction)
    take = positive(take_profit_fraction)
    if side not in {"long", "short"} or stop <= 0.0 or take <= 0.0:
        return 0.0, 0.0
    quote = current_quote or {}
    # Daily extremes and exchange order-price bands are not current prices.
    # Including them silently widens both the strategy stop and its target.
    references = [positive(reference_price)]
    references.extend(positive(quote.get(key)) for key in ("last", "close", "bid", "ask"))
    references = [value for value in references if value > 0.0]
    if not references:
        return 0.0, 0.0
    lower, upper = min(references), max(references)
    if side == "long":
        return lower * (1.0 - stop), upper * (1.0 + take)
    return upper * (1.0 + stop), lower * (1.0 - take)
