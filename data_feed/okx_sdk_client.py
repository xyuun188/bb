"""Compatibility helpers routed through the unified OKX perpetual SDK adapter."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import structlog

from core.exceptions import ExchangeAPIError
from core.okx_instrument_filter import supported_usdt_swap_instruments
from core.safe_output import safe_error_text
from data_feed.okx_ticker_volume import okx_swap_volume_fields, safe_float
from services.okx_perpetual_sdk import OkxPerpetualSdkExchange

logger = structlog.get_logger(__name__)

SUSPICIOUS_CONTRACT_BASE_TOKENS = ("TEST", "DEMO", "DUMMY", "MOCK", "SAMPLE")


def _is_suspicious_contract_base(base: str | None) -> bool:
    value = str(base or "").upper()
    return bool(value and any(token in value for token in SUSPICIOUS_CONTRACT_BASE_TOKENS))


def _raise_okx_api_error(result: dict[str, Any], fallback: str = "OKX API error") -> None:
    code = safe_error_text(result.get("code") or "unknown", limit=40)
    message = safe_error_text(result.get("msg") or fallback, limit=240)
    raise ExchangeAPIError(f"OKX API error [{code}]: {message}")


def _make_exchange(mode: str) -> OkxPerpetualSdkExchange:
    return OkxPerpetualSdkExchange(mode)


def _ensure_okx_success(result: dict[str, Any], fallback: str = "OKX API error") -> None:
    if not isinstance(result, dict):
        raise ExchangeAPIError(f"{fallback}: unexpected response type {type(result).__name__}")
    if str(result.get("code") or "0") != "0":
        _raise_okx_api_error(result, fallback)


async def fetch_klines(
    symbol: str,
    bar: str = "1H",
    limit: int = 100,
    mode: str = "paper",
    inst_type: str = "SWAP",
) -> list[dict]:
    """
    Fetch candlestick data via OKX official SDK.
    Defaults to perpetual swap (SWAP) for accurate pricing.
    Returns list of {time, open, high, low, close, volume} in chronological order.
    """
    inst_type = str(inst_type or "SWAP").upper()
    if inst_type != "SWAP":
        raise ExchangeAPIError(f"Only OKX SWAP klines are supported, got {inst_type!r}")

    exchange = _make_exchange(mode)
    rows = await exchange.fetch_ohlcv(symbol, timeframe=bar, limit=limit)
    return [
        {
            "time": datetime.fromtimestamp(int(c[0]) / 1000, tz=UTC).isoformat(),
            "open": float(c[1]),
            "high": float(c[2]),
            "low": float(c[3]),
            "close": float(c[4]),
            "volume": float(c[5]),
        }
        for c in rows
        if len(c) >= 6
    ]


async def fetch_usdt_balance(mode: str = "paper") -> float | None:
    """Fetch USDT balance from OKX account using official SDK."""

    try:
        exchange = _make_exchange(mode)
        result = await exchange.privateGetAccountBalance({"ccy": "USDT"})
        _ensure_okx_success(result)
        data = result.get("data", [])
        if data:
            inner_details = data[0].get("details", [])
            for d in inner_details:
                return float(d.get("availBal", 0))
        return 0.0
    except Exception as e:
        logger.warning("fetch USDT balance failed", mode=mode, error=safe_error_text(e))
        return None


async def fetch_tickers(instType: str = "SWAP", mode: str = "paper") -> dict:
    """Fetch all tickers from OKX via official SDK."""

    inst_type = str(instType or "SWAP").upper()
    if inst_type != "SWAP":
        raise ExchangeAPIError(f"Only OKX SWAP tickers are supported, got {inst_type!r}")
    exchange = _make_exchange(mode)
    result = await exchange.publicGetMarketTickers({"instType": "SWAP"})
    _ensure_okx_success(result)
    supported_swap_inst_ids = await _fetch_supported_swap_inst_ids(mode, exchange=exchange)
    tickers = {}
    for t in result.get("data", []):
        inst_id = str(t.get("instId") or "").strip().upper()
        if supported_swap_inst_ids is not None and inst_id not in supported_swap_inst_ids:
            continue
        symbol = t.get("instId", "").replace("-", "/")
        last = safe_float(t.get("last"), 0.0)
        open24h = safe_float(t.get("open24h"), 0.0)
        change_pct = ((last - open24h) / open24h * 100) if open24h else 0
        volume_fields = okx_swap_volume_fields(t, last)
        tickers[symbol] = {
            "price": last,
            "change_24h": change_pct,
            "volume_24h": volume_fields["volume_24h_base"] or volume_fields["volume_24h_contracts"],
            **volume_fields,
            "bid": safe_float(t.get("bidPx"), 0.0),
            "ask": safe_float(t.get("askPx"), 0.0),
        }
    return tickers


async def _fetch_supported_swap_inst_ids(
    mode: str,
    *,
    exchange: OkxPerpetualSdkExchange | None = None,
) -> set[str] | None:
    try:
        ex = exchange or _make_exchange(mode)
        data = await ex.publicGetPublicInstruments({"instType": "SWAP"})
        _ensure_okx_success(data)
    except Exception as exc:
        logger.warning(
            "fetch OKX instrument metadata failed; using unfiltered SDK tickers",
            mode=mode,
            error=safe_error_text(exc),
        )
        return None
    return {
        str(inst.get("instId") or "").strip().upper()
        for inst in supported_usdt_swap_instruments(data.get("data", []))
    }


async def get_available_symbols(mode: str = "paper") -> list[dict[str, str]]:
    """Get available OKX USDT perpetual swaps via public endpoint."""

    exchange = _make_exchange(mode)
    data = await exchange.publicGetPublicInstruments({"instType": "SWAP"})
    _ensure_okx_success(data)
    symbols = []
    for inst in supported_usdt_swap_instruments(data.get("data", [])):
        inst_id = inst.get("instId", "")
        base = inst_id.removesuffix("-USDT-SWAP")
        if _is_suspicious_contract_base(base):
            continue
        symbols.append(
            {
                "symbol": f"{base}/USDT",
                "base": base,
                "quote": "USDT",
                "type": "swap",
                "id": inst_id,
                "ccxt_symbol": f"{base}/USDT:USDT",
            }
        )

    priority = {
        "BTC": 0,
        "ETH": 1,
        "SOL": 2,
        "XRP": 3,
        "BNB": 4,
        "DOGE": 5,
        "ADA": 6,
        "AVAX": 7,
        "LINK": 8,
        "SUI": 9,
        "LTC": 10,
        "BCH": 11,
        "DOT": 12,
        "TRX": 13,
        "TON": 14,
    }
    return sorted(symbols, key=lambda x: (priority.get(x["base"], 1000), x["symbol"]))
