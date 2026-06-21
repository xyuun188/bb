"""Trading symbol normalization helpers.

The application stores positions, orders, exchange fills, and dashboard rows from
multiple sources. OKX/CCXT can represent the same USDT swap as `MET/USDT`,
`MET/USDT:USDT`, or `MET-USDT-SWAP`. Keep that contract in one place so
matching logic does not silently miss fills and create synthetic closes.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def normalize_trading_symbol(symbol: Any) -> str:
    """Return the canonical app symbol, e.g. `MET/USDT` for OKX swap variants."""

    text = str(symbol or "").strip().upper().replace("_", "-")
    if not text:
        return ""
    if ":" in text:
        text = text.split(":", 1)[0]
    if text.endswith("/SWAP"):
        text = text[: -len("/SWAP")]
    if text.endswith("-SWAP"):
        text = text[: -len("-SWAP")]
    if "/" not in text and "-" in text:
        parts = [part for part in text.split("-") if part]
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
    if "/" in text:
        base, quote = text.split("/", 1)
        quote = quote.split("-", 1)[0]
        if base and quote:
            return f"{base}/{quote}"
    return text


def trading_symbol_variants(symbol: Any) -> set[str]:
    """Return known storage spellings for a trading symbol."""

    raw = str(symbol or "").strip()
    normalized = normalize_trading_symbol(raw)
    variants: set[str] = set()
    for value in (raw, raw.upper(), normalized):
        value = str(value or "").strip()
        if not value:
            continue
        variants.add(value)
        variants.add(value.upper())
        canonical = normalize_trading_symbol(value)
        if canonical:
            variants.add(canonical)
            variants.add(canonical.upper())
            dashed = canonical.replace("/", "-")
            variants.add(dashed)
            variants.add(dashed.upper())
            variants.add(f"{canonical}:USDT")
            variants.add(f"{canonical.upper()}:USDT")
            variants.add(f"{dashed}-SWAP")
            variants.add(f"{dashed.upper()}-SWAP")
    return {item for item in variants if item}


def symbol_query_variants(symbols: Iterable[Any]) -> set[str]:
    """Return storage variants for a collection of symbols."""

    variants: set[str] = set()
    for symbol in symbols:
        variants.update(trading_symbol_variants(symbol))
    return variants
