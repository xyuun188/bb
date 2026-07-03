"""Shared filters for OKX swap instruments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

OKX_CRYPTO_INSTRUMENT_CATEGORY = "1"


def is_supported_usdt_swap_instrument(instrument: Mapping[str, Any]) -> bool:
    """Return whether an OKX instrument belongs in the real trading universe.

    OKX's SWAP instrument endpoint includes stock and commodity derivatives beside
    crypto perpetuals. Those symbols can have public tickers and contract rules, but
    the strategy stack and current OKX account configuration are crypto-swap only.
    Keep the new-entry universe aligned with what the executor can reliably trade.
    """

    inst_id = str(instrument.get("instId") or "")
    if not (
        instrument.get("instType") == "SWAP"
        and instrument.get("state") == "live"
        and instrument.get("ctType") == "linear"
        and instrument.get("settleCcy") == "USDT"
        and inst_id.endswith("-USDT-SWAP")
        and instrument.get("ctVal")
        and instrument.get("minSz")
        and instrument.get("tickSz")
    ):
        return False
    return is_crypto_swap_instrument(instrument)


def is_crypto_swap_instrument(instrument: Mapping[str, Any]) -> bool:
    """Return whether the OKX instrument is categorized as a crypto contract."""

    inst_category = str(instrument.get("instCategory") or "").strip()
    return inst_category == OKX_CRYPTO_INSTRUMENT_CATEGORY


def supported_usdt_swap_instruments(
    instruments: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> list[Mapping[str, Any]]:
    """Return supported instruments after applying OKX executor-style de-duplication.

    OKX demo can expose two instrument ids that CCXT parses to the same market
    symbol, for example LINEA-USDT-SWAP and SKY-USDT-SWAP both map through the
    same underlying `uly`. CCXT `set_markets()` keeps the later parsed market, so
    public symbol lists must apply the same overwrite rule or they can advertise a
    symbol the executor cannot look up by id.
    """

    selected: dict[str, Mapping[str, Any]] = {}
    for instrument in instruments:
        if not isinstance(instrument, Mapping) or not is_supported_usdt_swap_instrument(
            instrument
        ):
            continue
        selected[_okx_executor_market_key(instrument)] = instrument
    return list(selected.values())


def _okx_executor_market_key(instrument: Mapping[str, Any]) -> str:
    uly = str(instrument.get("uly") or "").strip().upper()
    if uly:
        return uly
    return str(instrument.get("instId") or "").strip().upper()
