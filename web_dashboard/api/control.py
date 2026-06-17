"""
System control API endpoints — mode switching, pause/resume, model selection.
"""

from __future__ import annotations

from datetime import UTC, datetime

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config.settings import ENSEMBLE_TRADER_NAME, settings
from core.safe_output import safe_error_text
from core.trading_mode import mode_manager
from web_dashboard.api import dashboard as _dash

router = APIRouter()
logger = structlog.get_logger(__name__)


class ModeSwitchRequest(BaseModel):
    mode: str  # "paper" or "live"


class SelectModelRequest(BaseModel):
    model_name: str


class ManualTradeRequest(BaseModel):
    symbol: str
    model_name: str | None = None


class ManualClosePositionRequest(BaseModel):
    reason: str | None = None


class ManualCloseAllPositionsRequest(BaseModel):
    mode: str | None = None
    reason: str | None = None


class ScanModeRequest(BaseModel):
    mode: str  # only "auto" is supported


def _normalize_okx_bar(timeframe: str) -> str:
    """Normalize UI/CCXT timeframes to OKX candle bar values."""
    value = (timeframe or "1H").strip()
    mapping = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "1h": "1H",
        "1H": "1H",
        "4h": "4H",
        "4H": "4H",
        "1d": "1D",
        "1D": "1D",
    }
    return mapping.get(value, mapping.get(value.lower(), "1H"))


def _okx_bar_to_ccxt_timeframe(bar: str) -> str:
    return {"1H": "1h", "4H": "4h", "1D": "1d"}.get(bar, bar)


def _missing_okx_credential_fields(mode: str) -> list[str]:
    creds = settings.get_okx_credentials(mode)
    required = {
        "api_key": "API Key",
        "api_secret": "API Secret",
        "passphrase": "Passphrase",
    }
    return [label for key, label in required.items() if not str(creds.get(key) or "").strip()]


def _assert_execution_account_configured(mode: str) -> None:
    missing = _missing_okx_credential_fields(mode)
    if not missing:
        return
    detail = {
        "message": f"{('实盘' if mode == 'live' else '模拟盘')}账户未配置完整，不能切换执行账户。",
        "mode": mode,
        "settings_tab": "okx",
        "missing_fields": missing,
    }
    raise HTTPException(status_code=409, detail=detail)


def _symbol_to_okx_inst_id(symbol: str, inst_type: str = "SWAP") -> str:
    """Convert BTC/USDT or BTC-USDT-SWAP to OKX instrument id."""
    normalized = (symbol or "BTC/USDT").strip().upper().replace("_", "-")
    if normalized.endswith("-SWAP") or normalized.endswith("-USDT"):
        return normalized

    base = normalized.split("/")[0].split("-")[0]
    return f"{base}-USDT-SWAP" if inst_type == "SWAP" else f"{base}-USDT"


async def _fetch_okx_public_klines(symbol: str, bar: str, limit: int) -> list[dict]:
    """Fetch OKX public candles directly so charts work without the data service."""
    import httpx

    inst_id = _symbol_to_okx_inst_id(symbol, inst_type="SWAP")
    safe_limit = max(1, min(int(limit or 100), 300))
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": inst_id, "bar": bar, "limit": str(safe_limit)}

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        payload = resp.json()

    if payload.get("code") != "0":
        raise RuntimeError(payload.get("msg") or f"OKX API error: {payload.get('code')}")

    rows = payload.get("data", [])
    rows = sorted(rows, key=lambda c: int(c[0]))
    return [
        {
            "time": datetime.fromtimestamp(int(c[0]) / 1000, tz=UTC).isoformat(),
            "open": float(c[1]),
            "high": float(c[2]),
            "low": float(c[3]),
            "close": float(c[4]),
            "volume": float(c[5]) if len(c) > 5 else 0.0,
        }
        for c in rows
    ]


@router.post("/control/mode")
async def switch_mode(req: ModeSwitchRequest):
    """Switch between paper and live trading modes."""
    mode = req.mode.lower()

    if mode not in ("paper", "live"):
        raise HTTPException(status_code=400, detail="Mode must be 'paper' or 'live'")

    _assert_execution_account_configured(mode)

    if mode == "paper":
        await mode_manager.switch_to_paper()
        return {
            "status": "ok",
            "message": "Switched to paper trading mode",
            "mode": "paper",
        }
    else:
        # Live mode uses the unified ensemble execution account.
        live_model = mode_manager.live_model_name or ENSEMBLE_TRADER_NAME
        mode_manager._live_model_name = live_model
        await mode_manager.switch_to_live(live_model)
        return {
            "status": "ok",
            "message": f"Switched to live trading mode (model: {mode_manager.live_model_name})",
            "mode": "live",
            "live_model": mode_manager.live_model_name,
        }


@router.post("/control/pause")
async def pause_trading():
    """Pause all trading."""
    await mode_manager.pause()
    return {"status": "ok", "message": "Trading paused"}


@router.post("/control/scan-mode")
async def switch_scan_mode(req: ScanModeRequest):
    """Keep scan mode on automatic portfolio discovery."""
    mode = req.mode.lower()
    if mode != "auto":
        raise HTTPException(status_code=400, detail="主面板只支持自动模式，手动扫描已移除。")
    await mode_manager.switch_to_auto()
    return {"status": "ok", "message": "已切换为自动模式", "scan_mode": "auto"}


@router.post("/control/resume")
async def resume_trading():
    """Resume trading."""
    await mode_manager.resume()
    return {"status": "ok", "message": "Trading resumed"}


@router.post("/control/select-model")
async def select_live_model(req: SelectModelRequest):
    """Select which model to use for live trading."""
    if req.model_name != ENSEMBLE_TRADER_NAME:
        raise HTTPException(
            status_code=400,
            detail=f"Live execution is fixed to '{ENSEMBLE_TRADER_NAME}'.",
        )
    mode_manager._live_model_name = ENSEMBLE_TRADER_NAME

    return {
        "status": "ok",
        "message": f"Live model set to '{ENSEMBLE_TRADER_NAME}'",
        "live_model": ENSEMBLE_TRADER_NAME,
    }


@router.get("/control/state")
async def get_control_state():
    """Get current control state (mode, pause, live model)."""
    state = mode_manager.get_state()
    if _dash._trading_service:
        state.update(_dash._trading_service.models.get_state())
        state.update(_dash._trading_service.get_stats())
    return state


@router.post("/trade/manual")
async def manual_trade(req: ManualTradeRequest):
    """Manual open-trade entry is disabled; auto mode only."""
    raise HTTPException(status_code=410, detail="主面板已移除手动开仓入口，仅保留自动模式。")


@router.post("/positions/{position_id}/close")
async def manual_close_position(position_id: int, req: ManualClosePositionRequest):
    """Manually close one open position directly through OKX."""
    if not _dash._trading_service:
        raise HTTPException(status_code=503, detail="Trading service not initialized")

    result = await _dash._trading_service.manual_close_position(
        position_id,
        reason=req.reason,
    )
    return result


@router.post("/positions/close-all")
async def manual_close_all_positions(req: ManualCloseAllPositionsRequest):
    """Manually close all open positions in the selected execution mode."""
    if not _dash._trading_service:
        raise HTTPException(status_code=503, detail="Trading service not initialized")

    mode = req.mode or mode_manager.mode.value
    if str(mode).lower() not in {"paper", "live"}:
        raise HTTPException(status_code=400, detail="mode must be 'paper' or 'live'")

    result = await _dash._trading_service.manual_close_all_positions(
        mode=mode,
        reason=req.reason,
    )
    return result


@router.get("/market/klines/{symbol:path}")
async def get_klines(symbol: str, timeframe: str = "1H", limit: int = 100):
    """Get recent kline data for charting via OKX official SDK."""
    bar = _normalize_okx_bar(timeframe)
    ccxt_timeframe = _okx_bar_to_ccxt_timeframe(bar)

    data = []
    try:
        from data_feed.okx_sdk_client import fetch_klines

        data = await fetch_klines(symbol, bar=bar, limit=limit, inst_type="SWAP")
    except Exception as exc:
        logger.warning(
            "failed to fetch klines from OKX SDK",
            symbol=symbol,
            error=safe_error_text(exc),
        )

    # Public OKX REST fallback works even when the trading/data services are stopped.
    if not data:
        try:
            data = await _fetch_okx_public_klines(symbol, bar=bar, limit=limit)
        except Exception as exc:
            logger.warning(
                "failed to fetch klines from OKX public REST",
                symbol=symbol,
                error=safe_error_text(exc),
            )

    # Fallback to CCXT if SDK fails
    if not data and _dash._data_service and _dash._data_service.rest_client:
        try:
            raw_klines = await _dash._data_service.rest_client.fetch_ohlcv(
                symbol, ccxt_timeframe, limit
            )
            data = [
                {
                    "time": datetime.fromtimestamp(k[0] / 1000, tz=UTC).isoformat(),
                    "open": k[1],
                    "high": k[2],
                    "low": k[3],
                    "close": k[4],
                    "volume": k[5],
                }
                for k in raw_klines
            ]
        except Exception as exc:
            logger.warning(
                "failed to fetch klines from CCXT fallback",
                symbol=symbol,
                error=safe_error_text(exc),
            )

    # Final fallback: DB cache
    if not data:
        try:
            from db.repositories.market_repo import MarketRepository
            from db.session import get_session_ctx

            async with get_session_ctx() as session:
                repo = MarketRepository(session)
                klines = await repo.get_klines(symbol, bar, limit)
                if not klines and bar != timeframe:
                    klines = await repo.get_klines(symbol, timeframe, limit)

            data = [
                {
                    "time": k.open_time.isoformat() if k.open_time else None,
                    "open": k.open,
                    "high": k.high,
                    "low": k.low,
                    "close": k.close,
                    "volume": k.volume,
                }
                for k in klines
            ]
        except Exception as exc:
            logger.warning(
                "failed to fetch klines from DB fallback",
                symbol=symbol,
                error=safe_error_text(exc),
            )

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "count": len(data),
        "data": data,
    }
