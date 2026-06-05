"""
Symbol management API — list available, add/remove monitored symbols.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config.settings import settings
from web_dashboard.api import dashboard as _dash

router = APIRouter()


class SymbolsUpdateRequest(BaseModel):
    symbols: list[str]


class SymbolRequest(BaseModel):
    symbol: str


@router.get("/symbols/available")
async def get_available_symbols():
    """Get all available USDT trading pairs from OKX via official SDK."""
    # Primary: OKX official SDK
    try:
        from data_feed.okx_sdk_client import get_available_symbols as sdk_symbols
        symbols = await sdk_symbols()
        if symbols:
            return {"count": len(symbols), "symbols": symbols}
    except Exception:
        pass

    # Fallback: DataService (CCXT)
    if _dash._data_service:
        try:
            symbols = await _dash._data_service.get_available_symbols()
            return {"count": len(symbols), "symbols": symbols}
        except Exception as e:
            return {"count": 0, "symbols": [], "error": str(e)}
    return {"count": 0, "symbols": [], "error": "Data service not initialized"}


@router.get("/symbols/active")
async def get_active_symbols():
    """Get currently monitored symbols."""
    return {"symbols": settings.symbols}


@router.post("/symbols/update")
async def update_symbols(req: SymbolsUpdateRequest):
    """Replace the entire monitored symbols list."""
    if not req.symbols:
        raise HTTPException(status_code=400, detail="Symbols list cannot be empty")

    old_symbols = set(settings.symbols)
    new_symbols = set(req.symbols)

    # Persist change
    settings.update_symbols(req.symbols)

    # Update WebSocket subscriptions if data service is available
    if _dash._data_service:
        ws = _dash._data_service.ws_client
        # Subscribe to new symbols
        for sym in new_symbols - old_symbols:
            try:
                await ws.subscribe_symbol(sym)
            except Exception:
                pass
        # Unsubscribe from removed symbols
        for sym in old_symbols - new_symbols:
            try:
                await ws.unsubscribe_symbol(sym)
            except Exception:
                pass

    return {"status": "ok", "symbols": settings.symbols}


@router.post("/symbols/add")
async def add_symbol(req: SymbolRequest):
    """Add a single symbol to the monitored list."""
    if req.symbol in settings.symbols:
        return {"status": "ok", "symbols": settings.symbols, "message": "Symbol already monitored"}

    new_symbols = settings.symbols + [req.symbol]
    settings.update_symbols(new_symbols)

    # Subscribe via WebSocket
    if _dash._data_service:
        try:
            await _dash._data_service.ws_client.subscribe_symbol(req.symbol)
        except Exception:
            pass

    return {"status": "ok", "symbols": settings.symbols}


@router.post("/symbols/remove")
async def remove_symbol(req: SymbolRequest):
    """Remove a single symbol from the monitored list."""
    if req.symbol not in settings.symbols:
        raise HTTPException(status_code=404, detail="Symbol not in monitored list")

    new_symbols = [s for s in settings.symbols if s != req.symbol]
    if not new_symbols:
        raise HTTPException(status_code=400, detail="Cannot remove last symbol")

    settings.update_symbols(new_symbols)

    # Unsubscribe via WebSocket
    if _dash._data_service:
        try:
            await _dash._data_service.ws_client.unsubscribe_symbol(req.symbol)
        except Exception:
            pass

    return {"status": "ok", "symbols": settings.symbols}
