"""Trading symbol normalization helpers.

The application stores positions, orders, exchange fills, and dashboard rows from
multiple sources. OKX/CCXT can represent the same USDT swap as `MET/USDT`,
`MET/USDT:USDT`, or `MET-USDT-SWAP`. Keep that contract in one place so
matching logic does not silently miss fills and create synthetic closes.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def symbol_from_okx_inst_id(inst_id: Any, *, fallback: Any = "") -> str:
    """Return the app symbol represented by an OKX instrument id."""

    text = str(inst_id or "").strip().upper()
    if not text:
        return normalize_trading_symbol(fallback)
    if text.endswith("-SWAP"):
        text = text[: -len("-SWAP")]
    parts = [part for part in text.replace("_", "-").split("-") if part]
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return normalize_trading_symbol(fallback or text)


def symbol_from_okx_market(market: dict[str, Any] | None, *, fallback: Any = "") -> str:
    """Return the exchange-native display symbol for an OKX market payload.

    CCXT can expose OKX's H-USDT-SWAP contract as WLFI/USDT because OKX reports
    uly=WLFI-USDT while the actual instrument and contract value currency remain
    H. For trade facts and dashboard history, instId is the authoritative symbol.
    """

    payload = market if isinstance(market, dict) else {}
    info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
    inst_symbol = symbol_from_okx_inst_id(info.get("instId") or payload.get("id"))
    if inst_symbol:
        return inst_symbol
    return normalize_trading_symbol(fallback or payload.get("symbol"))


def symbol_from_okx_payload(payload: dict[str, Any] | None, *, fallback: Any = "") -> str:
    """Extract the authoritative OKX symbol from an order/position payload."""

    data = payload if isinstance(payload, dict) else {}
    info = data.get("info") if isinstance(data.get("info"), dict) else {}
    for candidate in (
        info.get("instId"),
        data.get("instId"),
        data.get("okx_inst_id"),
        data.get("okx_symbol"),
        data.get("symbol"),
    ):
        symbol = symbol_from_okx_inst_id(candidate)
        if symbol:
            return symbol
    return normalize_trading_symbol(fallback)


def okx_inst_id_from_symbol(symbol: Any) -> str:
    """Return the OKX USDT swap instrument id represented by an app symbol."""

    normalized = normalize_trading_symbol(symbol)
    if not normalized or "/" not in normalized:
        return ""
    return f"{normalized.replace('/', '-')}-SWAP".upper()


def okx_inst_id_from_payload(
    payload: dict[str, Any] | None,
    *,
    fallback: Any = "",
    include_fallback: bool = True,
) -> str:
    """Extract an authoritative OKX instId, falling back to the app symbol."""

    data = payload if isinstance(payload, dict) else {}
    info = data.get("info") if isinstance(data.get("info"), dict) else {}
    request_params = (
        data.get("request_params") if isinstance(data.get("request_params"), dict) else {}
    )
    native_close_fill = (
        data.get("native_close_fill") if isinstance(data.get("native_close_fill"), dict) else {}
    )
    native_fill_info = (
        native_close_fill.get("order_info")
        if isinstance(native_close_fill.get("order_info"), dict)
        else {}
    )
    close_fill = data.get("close_fill") if isinstance(data.get("close_fill"), dict) else {}
    close_fill_info = (
        close_fill.get("order_info") if isinstance(close_fill.get("order_info"), dict) else {}
    )
    for candidate in (
        info.get("instId"),
        data.get("instId"),
        data.get("okx_inst_id"),
        request_params.get("instId"),
        native_fill_info.get("instId"),
        native_close_fill.get("instId"),
        close_fill_info.get("instId"),
        close_fill.get("instId"),
    ):
        text = str(candidate or "").strip().upper()
        if text:
            if text.endswith("-SWAP"):
                return text
            symbol = symbol_from_okx_inst_id(text)
            inst_id = okx_inst_id_from_symbol(symbol)
            if inst_id:
                return inst_id
    if not include_fallback:
        return ""
    return okx_inst_id_from_symbol(
        fallback or data.get("canonical_exchange_symbol") or data.get("symbol")
    )


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
