"""Normalize OKX ticker volume fields for swap market features."""

from __future__ import annotations

from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def first_positive_float(*values: Any) -> float:
    for value in values:
        parsed = safe_float(value, 0.0)
        if parsed > 0:
            return parsed
    return 0.0


def okx_swap_volume_fields(ticker: dict[str, Any] | None, price: float = 0.0) -> dict[str, Any]:
    """Return explicit contracts/base/quote volume fields for an OKX swap ticker.

    OKX swap tickers expose ``vol24h`` as contract count and ``volCcy24h`` as
    base-currency volume. Candidate ranking needs USDT notional, so it must not
    multiply contract count by price.
    """

    data = ticker or {}
    info = data.get("info") if isinstance(data.get("info"), dict) else {}
    last = first_positive_float(
        price,
        data.get("last_price"),
        data.get("last"),
        data.get("price"),
        data.get("close"),
        info.get("last"),
        info.get("lastPx"),
    )
    contracts = first_positive_float(
        data.get("volume_24h_contracts"),
        data.get("contract_volume_24h"),
        data.get("contracts_volume_24h"),
        data.get("vol24h"),
        info.get("vol24h"),
    )
    base = first_positive_float(
        data.get("volume_24h_base"),
        data.get("base_volume_24h"),
        data.get("baseVolume"),
        data.get("volCcy24h"),
        info.get("volCcy24h"),
    )
    quote = first_positive_float(
        data.get("volume_24h_quote"),
        data.get("quote_volume_24h"),
        data.get("quoteVolume"),
        data.get("volCcyQuote24h"),
        info.get("volCcyQuote24h"),
    )
    if quote <= 0 and base > 0 and last > 0:
        quote = base * last
    source = "quote" if quote > 0 else "base" if base > 0 else "contracts" if contracts > 0 else ""
    return {
        "volume_24h_contracts": contracts,
        "volume_24h_base": base,
        "volume_24h_quote": quote,
        "notional_24h_usdt": quote,
        "volume_24h_source": source,
    }
