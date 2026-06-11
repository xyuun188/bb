"""
OKX official Python SDK client for market data and account balance.
Uses python-okx with openapi.okx.com domain.
Wraps sync SDK calls in asyncio.to_thread for async compatibility.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC
from typing import Any

import structlog

from config.settings import settings
from core.exceptions import ConfigError, ExchangeAPIError
from core.safe_output import safe_error_text

logger = structlog.get_logger(__name__)

SUSPICIOUS_CONTRACT_BASE_TOKENS = ("TEST", "DEMO", "DUMMY", "MOCK", "SAMPLE")


def _requests_proxies() -> dict[str, str] | None:
    proxy = (
        os.environ.get("OKX_PROXY")
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("http_proxy")
    )
    if not proxy:
        return None
    return {"http": proxy, "https": proxy}


def _is_suspicious_contract_base(base: str | None) -> bool:
    value = str(base or "").upper()
    return bool(value and any(token in value for token in SUSPICIOUS_CONTRACT_BASE_TOKENS))


def _raise_okx_api_error(result: dict[str, Any], fallback: str = "OKX API error") -> None:
    code = safe_error_text(result.get("code") or "unknown", limit=40)
    message = safe_error_text(result.get("msg") or fallback, limit=240)
    raise ExchangeAPIError(f"OKX API error [{code}]: {message}")


def _make_market_api(mode: str) -> Any:
    """Create a MarketAPI instance for the given mode.

    NOTE: flag='1' means simulated/demo in the x-simulated-trading header.
    """
    import okx.MarketData as MarketData

    flag = "0" if mode == "live" else "1"
    return MarketData.MarketAPI(flag=flag, debug=False)


def _make_account_api(mode: str) -> Any:
    """Create an AccountAPI instance for the given mode.

    NOTE: The python-okx SDK maps flag directly to the x-simulated-trading header,
    where '1' = simulated/demo and '0' = real/live. So we use flag='1' for paper
    and flag='0' for live (inverse of the SDK's documented convention).
    """
    from okx.Account import AccountAPI

    creds = settings.get_okx_credentials(mode)
    flag = "0" if mode == "live" else "1"
    return AccountAPI(
        api_key=creds.get("api_key", ""),
        api_secret_key=creds.get("api_secret", ""),
        passphrase=creds.get("passphrase", ""),
        flag=flag,
        use_server_time=True,
        debug=False,
    )


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
    from datetime import datetime

    base = symbol.split("/")[0]
    if inst_type == "SWAP":
        instId = f"{base}-USDT-SWAP"
    else:
        instId = f"{base}-USDT"

    def _sync():
        api = _make_market_api(mode)
        result = api.get_candlesticks(instId=instId, bar=bar, limit=limit)
        if result.get("code") != "0":
            _raise_okx_api_error(result)
        raw = result.get("data", [])
        # OKX returns newest first; reverse to chronological order (left to right)
        raw.reverse()
        return [
            {
                "time": datetime.fromtimestamp(int(c[0]) / 1000, tz=UTC).isoformat(),
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5]),
            }
            for c in raw
        ]

    return await asyncio.to_thread(_sync)


async def fetch_usdt_balance(mode: str = "paper") -> float | None:
    """Fetch USDT balance from OKX account using official SDK."""

    def _sync():
        creds = settings.get_okx_credentials(mode)
        if not creds.get("api_key") or not creds.get("api_secret"):
            raise ConfigError("OKX API credentials are not configured")
        if not creds.get("passphrase"):
            raise ConfigError("OKX API passphrase is not configured")

        api = _make_account_api(mode)
        result = api.get_account_balance(ccy="USDT")
        if result.get("code") != "0":
            _raise_okx_api_error(result)
        data = result.get("data", [])
        if data:
            inner_details = data[0].get("details", [])
            for d in inner_details:
                return float(d.get("availBal", 0))
        return 0.0

    try:
        return await asyncio.to_thread(_sync)
    except Exception as e:
        logger.warning("fetch USDT balance failed", mode=mode, error=safe_error_text(e))
        return None


async def fetch_tickers(instType: str = "SPOT", mode: str = "paper") -> dict:
    """Fetch all tickers from OKX via official SDK."""

    def _sync():
        api = _make_market_api(mode)
        result = api.get_tickers(instType=instType)
        if result.get("code") != "0":
            _raise_okx_api_error(result)
        tickers = {}
        for t in result.get("data", []):
            symbol = t.get("instId", "").replace("-", "/")
            last = float(t.get("last", 0))
            open24h = float(t.get("open24h", 0))
            change_pct = ((last - open24h) / open24h * 100) if open24h else 0
            tickers[symbol] = {
                "price": last,
                "change_24h": change_pct,
                "volume_24h": float(t.get("vol24h", 0)),
                "bid": float(t.get("bidPx", 0)),
                "ask": float(t.get("askPx", 0)),
            }
        return tickers

    return await asyncio.to_thread(_sync)


async def get_available_symbols(mode: str = "paper") -> list[dict[str, str]]:
    """Get available OKX USDT perpetual swaps via public endpoint."""
    import requests

    def _sync():
        url = "https://www.okx.com/api/v5/public/instruments?instType=SWAP"
        resp = requests.get(url, timeout=10, proxies=_requests_proxies())
        data = resp.json()
        if data.get("code") != "0":
            _raise_okx_api_error(data)
        symbols = []
        for inst in data.get("data", []):
            inst_id = inst.get("instId", "")
            if (
                inst.get("settleCcy") == "USDT"
                and inst.get("ctType") == "linear"
                and inst.get("state") == "live"
                and inst_id.endswith("-USDT-SWAP")
            ):
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

    return await asyncio.to_thread(_sync)
