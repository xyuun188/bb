"""
Exchange & AI settings API — get/set OKX credentials (paper/live split), AI config, test connections.
"""

from __future__ import annotations

import asyncio
import copy
import json
import time
from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config.settings import ENSEMBLE_TRADER_NAME, settings
from core.model_runtime import (
    HIGH_RISK_REVIEW_TOKEN_CAP,
    HIGH_RISK_REVIEW_TOKEN_FLOOR,
    ensure_no_think_text,
    non_thinking_extra_body,
    uses_thinking_tags,
)
from core.safe_output import redact_output, safe_error_text
from core.secret_utils import is_masked_secret, mask_secret
from core.url_safety import normalize_http_base_url
from services.model_server_config import (
    ModelServerConfigError,
    ModelServerConfigNotConfigured,
    build_model_server_info_from_update,
    get_model_server_settings_public,
    load_model_server_info_from_secure_settings,
    save_model_server_settings,
)
from services.secure_runtime_config import (
    scrub_ai_model_env,
    secure_ai_model_key,
    set_runtime_secret,
    strip_secret_env_updates,
)
from services.server_monitor_status import ServerMonitorStatusService, clear_server_monitor_cache
from services.trading_params import DEFAULT_TRADING_PARAMS, ESTIMATED_TAKER_FEE_PCT
from web_dashboard.api import dashboard as _dash

router = APIRouter()
logger = structlog.get_logger(__name__)
_OKX_BALANCE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_OKX_BALANCE_ERROR_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_OKX_BALANCE_LOCKS: dict[str, asyncio.Lock] = {}
_OKX_BALANCE_TTL_SECONDS = 60.0
_OKX_BALANCE_STALE_SECONDS = 300.0
_OKX_BALANCE_ERROR_TTL_SECONDS = 5.0
_OKX_BALANCE_INITIALIZE_TIMEOUT_SECONDS = 5.0
_OKX_BALANCE_READ_TIMEOUT_SECONDS = 5.0
_MODEL_CONNECTION_ERROR_LIMIT = 700


# ── Request models ──


class OKXSettingsRequest(BaseModel):
    mode: str  # "paper" or "live"
    api_key: str | None = None
    api_secret: str | None = None
    passphrase: str | None = None


class OKXTestRequest(BaseModel):
    mode: str = "paper"  # "paper" or "live"


class AIModelRequest(BaseModel):
    name: str
    api_base: str | None = None
    api_key: str | None = None
    model: str | None = None
    balance: float | None = None  # legacy clients may send it; ignored for Phase 3 OKX truth
    execution_mode: str = "paper"  # "paper" or "live"


class AIModelTestRequest(BaseModel):
    name: str | None = None  # look up by name in settings.ai_models
    api_base: str | None = None
    api_key: str | None = None
    model: str | None = None


class ExecutionAccountRequest(BaseModel):
    mode: str = "paper"
    account_name: str | None = None
    allocated_balance: float | None = None  # legacy clients may send it; ignored intentionally
    max_loss_pct: float | None = None
    max_loss_usdt: float | None = None
    cooldown_loss_pct: float | None = None


class ModelServerSettingsRequest(BaseModel):
    host: str | None = None
    port: int | None = None
    username: str | None = None
    password: str | None = None


# ── Helpers ──


def _masked(m: dict) -> dict:
    """Return a copy with api_key masked."""
    mc = dict(m)
    mc["api_key"] = mask_secret(mc.get("api_key", ""))
    return mc


def _is_masked_secret(value: str | None) -> bool:
    return is_masked_secret(value)


def _normalize_api_base_or_400(
    value: str | None,
    *,
    field_name: str,
    allow_empty: bool = True,
) -> str:
    try:
        return normalize_http_base_url(
            value,
            field_name=field_name,
            allow_empty=allow_empty,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=safe_error_text(exc)) from exc


def _connection_error_text(value: Any) -> str:
    return safe_error_text(value, limit=_MODEL_CONNECTION_ERROR_LIMIT)


def _model_server_error(exc: Exception) -> str:
    return safe_error_text(exc, limit=500)


def _okx_mode_label(mode: str) -> str:
    return "OKX 实盘账户" if mode == "live" else "OKX 模拟盘账户"


def _empty_okx_snapshot(mode: str) -> dict[str, Any]:
    return {
        "available_balance": None,
        "used_balance": None,
        "total_balance": None,
        "cash_balance": None,
        "equity_balance": None,
        "allocatable_balance": None,
        "balance_error": None,
        "balance_source": _okx_mode_label(mode),
    }


def _okx_balance_error_text(exc: Exception) -> str:
    error = _connection_error_text(exc)
    lower = error.lower()
    if isinstance(exc, TimeoutError) or error in ("TimeoutError", "") or "timed out" in lower:
        return "OKX 余额响应超时，已优先返回缓存数据"
    return f"OKX 余额查询失败: {error}"


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_optional_float(*values: Any) -> float | None:
    for value in values:
        parsed = _optional_float(value)
        if parsed is not None:
            return parsed
    return None


def _okx_balance_result_from_raw_snapshot(
    mode: str,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    result = _empty_okx_snapshot(mode)
    available = _first_optional_float(snapshot.get("free"), snapshot.get("available_balance"))
    used = _first_optional_float(snapshot.get("used"), snapshot.get("used_balance"))
    total = _first_optional_float(snapshot.get("total"), snapshot.get("total_balance"))
    cash = _first_optional_float(snapshot.get("cash"), snapshot.get("cash_balance"), total)
    equity = _first_optional_float(snapshot.get("equity"), snapshot.get("equity_balance"), total)
    allocatable = _first_optional_float(
        snapshot.get("allocatable"),
        snapshot.get("allocatable_balance"),
        equity,
        total,
        available,
    )
    result.update(
        {
            "available_balance": available,
            "used_balance": used,
            "total_balance": total,
            "cash_balance": cash,
            "equity_balance": equity,
            "allocatable_balance": allocatable,
            "balance_source": snapshot.get("balance_source") or _okx_mode_label(mode),
        }
    )
    if snapshot.get("stale"):
        result["stale"] = True
        result["stale_age_seconds"] = snapshot.get("stale_age_seconds")
    for key in ("balance_status", "refresh_in_progress", "source"):
        if key in snapshot:
            result[key] = snapshot.get(key)
    if snapshot.get("error") or snapshot.get("balance_error"):
        result["balance_error"] = snapshot.get("balance_error") or snapshot.get("error")
        result["error_cached"] = bool(snapshot.get("error_cached"))
    return result


def _cached_okx_snapshot(
    mode: str,
    *,
    allow_stale: bool,
    error: str | None = None,
) -> dict[str, Any] | None:
    cached = _OKX_BALANCE_CACHE.get(mode)
    if not cached:
        return None
    cached_at, cached_value = cached
    age_seconds = time.time() - cached_at
    ttl_seconds = _OKX_BALANCE_STALE_SECONDS if allow_stale else _OKX_BALANCE_TTL_SECONDS
    if age_seconds > ttl_seconds:
        return None
    result = copy.deepcopy(cached_value)
    if allow_stale and age_seconds > _OKX_BALANCE_TTL_SECONDS:
        result["stale"] = True
        result["stale_age_seconds"] = round(age_seconds, 3)
    if error:
        result["balance_error"] = error
        result["error_cached"] = True
    return result


def _clear_okx_snapshot_caches(mode: str) -> None:
    mode = "live" if mode == "live" else "paper"
    _OKX_BALANCE_CACHE.pop(mode, None)
    _OKX_BALANCE_ERROR_CACHE.pop(mode, None)
    for cache_name in (
        "_dashboard_okx_balance_cache",
        "_dashboard_okx_balance_error_cache",
        "_dashboard_okx_position_cache",
        "_dashboard_okx_position_error_cache",
        "_exchange_mark_cache",
        "_exchange_open_symbol_cache",
    ):
        cache = getattr(_dash, cache_name, None)
        if isinstance(cache, dict):
            cache.pop(mode, None)


def _make_okx_executor(cls: Any, mode: str):
    """Create OKX executor in lightweight-balance mode, compatible with test doubles."""
    try:
        return cls(mode=mode, load_markets_on_initialize=False)
    except TypeError:
        return cls(mode=mode)


def _trading_service_cached_okx_snapshot(mode: str) -> dict[str, Any] | None:
    trading_service = getattr(_dash, "_trading_service", None)
    if not trading_service:
        return None
    peeker = getattr(trading_service, "peek_okx_balance_snapshot_for_mode", None)
    if not callable(peeker):
        return None
    selected_mode = "live" if mode == "live" else "paper"
    try:
        snapshot = peeker(selected_mode, allow_stale=True)
    except TypeError:
        snapshot = peeker(selected_mode)
    if not isinstance(snapshot, dict) or not snapshot:
        return None
    result = dict(snapshot)
    result.pop("error", None)
    result.pop("balance_error", None)
    result.pop("error_cached", None)
    return result


def _dashboard_cached_okx_snapshot(mode: str, *, allow_stale: bool) -> dict[str, Any] | None:
    selected_mode = "live" if mode == "live" else "paper"
    cached = getattr(_dash, "_dashboard_okx_balance_cache", {}).get(selected_mode)
    if not cached:
        return None
    cached_at, cached_snapshot = cached
    age_seconds = (datetime.now(UTC) - cached_at).total_seconds()
    ttl_seconds = _OKX_BALANCE_STALE_SECONDS if allow_stale else _OKX_BALANCE_TTL_SECONDS
    if age_seconds > ttl_seconds:
        return None
    snapshot = copy.deepcopy(cached_snapshot)
    if allow_stale and age_seconds > _OKX_BALANCE_TTL_SECONDS:
        snapshot["stale"] = True
        snapshot["stale_age_seconds"] = round(age_seconds, 3)
    return snapshot


async def _get_okx_usdt_snapshot(mode: str, force: bool = False) -> dict[str, Any]:
    """Fetch the real OKX USDT balance snapshot for paper/demo or live mode."""
    mode = "live" if mode == "live" else "paper"
    result = _empty_okx_snapshot(mode)

    creds = settings.get_okx_credentials(mode)
    if not creds.get("api_key"):
        result["balance_error"] = "未配置 OKX API Key"
        result["balance_status"] = "not_configured"
        return result

    cached = _cached_okx_snapshot(mode, allow_stale=False)
    if cached and not force:
        return cached

    cached_error = _OKX_BALANCE_ERROR_CACHE.get(mode)
    if cached_error and not force:
        cached_at, cached_value = cached_error
        if time.time() - cached_at <= _OKX_BALANCE_ERROR_TTL_SECONDS:
            stale = _cached_okx_snapshot(
                mode,
                allow_stale=True,
                error=cached_value.get("balance_error"),
            )
            return stale or copy.deepcopy(cached_value)

    lock = _OKX_BALANCE_LOCKS.setdefault(mode, asyncio.Lock())
    async with lock:
        cached = _cached_okx_snapshot(mode, allow_stale=False)
        if cached and not force:
            return cached

        if not force:
            shared_snapshot = _trading_service_cached_okx_snapshot(mode)
            if shared_snapshot:
                normalized = _okx_balance_result_from_raw_snapshot(mode, shared_snapshot)
                if not (
                    shared_snapshot.get("stale")
                    or shared_snapshot.get("error")
                    or shared_snapshot.get("balance_error")
                ):
                    _OKX_BALANCE_CACHE[mode] = (time.time(), copy.deepcopy(normalized))
                    _OKX_BALANCE_ERROR_CACHE.pop(mode, None)
                return normalized

            dashboard_cached = _dashboard_cached_okx_snapshot(mode, allow_stale=True)
            if dashboard_cached:
                normalized = _okx_balance_result_from_raw_snapshot(mode, dashboard_cached)
                if not (
                    dashboard_cached.get("stale")
                    or dashboard_cached.get("error")
                    or dashboard_cached.get("balance_error")
                ):
                    _OKX_BALANCE_CACHE[mode] = (time.time(), copy.deepcopy(normalized))
                    _OKX_BALANCE_ERROR_CACHE.pop(mode, None)
                return normalized

            dashboard_snapshot = await _dash._get_dashboard_okx_account_snapshot(mode)
            if isinstance(dashboard_snapshot, dict) and dashboard_snapshot:
                normalized = _okx_balance_result_from_raw_snapshot(mode, dashboard_snapshot)
                if not (
                    dashboard_snapshot.get("stale")
                    or dashboard_snapshot.get("error")
                    or dashboard_snapshot.get("balance_error")
                    or dashboard_snapshot.get("refresh_in_progress")
                ):
                    _OKX_BALANCE_CACHE[mode] = (time.time(), copy.deepcopy(normalized))
                    _OKX_BALANCE_ERROR_CACHE.pop(mode, None)
                return normalized

        from executor.okx_executor import OKXExecutor

        executor = _make_okx_executor(OKXExecutor, mode)
        try:
            async def fetch_snapshot() -> dict[str, Any]:
                await asyncio.wait_for(
                    executor.initialize(),
                    timeout=_OKX_BALANCE_INITIALIZE_TIMEOUT_SECONDS,
                )
                snapshot = await asyncio.wait_for(
                    executor.get_balance_snapshot("USDT"),
                    timeout=_OKX_BALANCE_READ_TIMEOUT_SECONDS,
                )
                if snapshot.get("error"):
                    raise RuntimeError(_connection_error_text(snapshot.get("error")))
                return snapshot

            snapshot = await fetch_snapshot()
            normalized = _okx_balance_result_from_raw_snapshot(mode, snapshot)
            _OKX_BALANCE_CACHE[mode] = (time.time(), copy.deepcopy(normalized))
            _OKX_BALANCE_ERROR_CACHE.pop(mode, None)
            return normalized
        except Exception as exc:
            error_text = _okx_balance_error_text(exc)
            stale = _cached_okx_snapshot(mode, allow_stale=True, error=error_text)
            failure = {**result, "balance_error": error_text, "error_cached": True}
            _OKX_BALANCE_ERROR_CACHE[mode] = (time.time(), copy.deepcopy(failure))
            return stale or failure
        finally:
            try:
                await executor.shutdown()
            except Exception as exc:
                logger.debug(
                    "OKX executor shutdown failed after balance snapshot",
                    mode=mode,
                    error=_connection_error_text(exc),
                )


async def _get_okx_usdt_balance(mode: str, force: bool = False) -> float | None:
    """Fetch OKX USDT account equity/balance for allocation validation."""
    snapshot = await _get_okx_usdt_snapshot(mode, force=force)
    if snapshot.get("balance_error"):
        return None
    return snapshot.get("allocatable_balance")


async def _paper_execution_account_summary() -> dict:
    """Return no synthetic account balance for OKX-backed paper mode."""
    return {
        "available_balance": None,
        "current_balance": None,
        "wallet_balance": None,
        "equity": None,
        "initial_balance": None,
        "used_margin": 0.0,
        "unrealized_pnl": 0.0,
        "total_pnl": None,
        "total_pnl_pct": None,
    }


async def _execution_account_status(mode: str) -> dict:
    mode = "live" if mode == "live" else "paper"
    cfg = settings.get_execution_account_config(mode)
    max_loss_pct = float(cfg.get("max_loss_pct") or 0.0)
    pnl_summary = await _dash._get_execution_pnl_summary(mode)
    okx_snapshot = await _get_okx_usdt_snapshot(mode)
    okx_available = okx_snapshot.get("available_balance")
    okx_allocatable = okx_snapshot.get("allocatable_balance")
    okx_balance_available = bool(
        okx_snapshot.get("equity_balance")
        or okx_snapshot.get("total_balance")
        or okx_allocatable
        or okx_available
    )
    account_equity = float(
        okx_snapshot.get("equity_balance")
        or okx_snapshot.get("total_balance")
        or okx_allocatable
        or okx_available
        or 0.0
    ) if okx_balance_available else 0.0
    okx_pnl = _dash._okx_equity_pnl_from_snapshot(
        current_equity=account_equity if okx_balance_available else None,
        pnl_summary=pnl_summary,
    )
    max_loss_usdt = (
        account_equity * max_loss_pct if account_equity > 0 and max_loss_pct > 0 else 0.0
    )
    risk_floor = max(account_equity - max_loss_usdt, 0.0) if account_equity > 0 else 0.0
    pause_reason = None
    if _dash._trading_service and _dash.mode_manager.mode.value == mode:
        pause_reason = getattr(_dash._trading_service, "_new_pair_pause_reasons", {}).get(
            ENSEMBLE_TRADER_NAME
        )
    if okx_snapshot.get("balance_error") and not pause_reason and not okx_balance_available:
        pause_reason = (
            f"未同步到 {okx_snapshot.get('balance_source')} 的实际余额，暂停分析新的交易对。"
        )
    total_pnl_for_risk = 0.0
    if (
        account_equity > 0
        and max_loss_usdt > 0
        and total_pnl_for_risk <= -max_loss_usdt
        and not pause_reason
    ):
        pause_reason = (
            f"{okx_snapshot.get('balance_source')} AI 执行账户累计盈亏 {total_pnl_for_risk:.2f} USDT "
            f"已达到最高亏损限制 {max_loss_pct * 100:.1f}%（{max_loss_usdt:.2f} USDT），暂停分析新的交易对。"
        )
    if not pause_reason and okx_pnl["today_equity_pnl"] is not None:
        pause_reason = _dash._cooldown_pause_reason_from_summary(
            {
                "today_risk_pnl": okx_pnl["today_equity_pnl"],
                "today_equity_pnl": okx_pnl["today_equity_pnl"],
                "today_total_pnl": okx_pnl["today_total_pnl"],
            },
            {**cfg, "max_loss_usdt": max_loss_usdt},
            okx_snapshot.get("balance_source") or "执行账户",
        )
    status = dict(cfg)
    status.update(
        {
            "allocated_balance": None,
            "account_balance_source_value": account_equity,
            "account_equity": account_equity,
            "max_loss_usdt": max_loss_usdt,
            "available_balance": okx_available,
            "equity": okx_snapshot.get("total_balance"),
            "used_margin": okx_snapshot.get("used_balance"),
            "unrealized_pnl": pnl_summary.get("unrealized_pnl", 0.0),
            "realized_profit": pnl_summary.get("realized_profit", 0.0),
            "realized_loss": pnl_summary.get("realized_loss", 0.0),
            "realized_pnl": pnl_summary.get("realized_pnl", 0.0),
            "today_realized_profit": pnl_summary.get("today_realized_profit", 0.0),
            "today_realized_loss": pnl_summary.get("today_realized_loss", 0.0),
            "today_realized_pnl": pnl_summary.get("today_realized_pnl", 0.0),
            "today_closed_realized_profit": pnl_summary.get("today_closed_realized_profit", 0.0),
            "today_closed_realized_loss": pnl_summary.get("today_closed_realized_loss", 0.0),
            "today_closed_realized_pnl": pnl_summary.get("today_closed_realized_pnl", 0.0),
            "today_equity_pnl": okx_pnl["today_equity_pnl"],
            "today_equity_baseline": pnl_summary.get("today_equity_baseline"),
            "today_equity_baseline_total_pnl": pnl_summary.get("today_equity_baseline_total_pnl"),
            "today_equity_baseline_at": pnl_summary.get("today_equity_baseline_at"),
            "today_equity_baseline_source": pnl_summary.get("today_equity_baseline_source"),
            "today_snapshot_date": pnl_summary.get("today_snapshot_date"),
            "today_total_pnl": okx_pnl["today_total_pnl"],
            "today_risk_pnl": pnl_summary.get("today_risk_pnl", 0.0),
            "cumulative_profit": pnl_summary.get("realized_profit", 0.0),
            "cumulative_loss": pnl_summary.get("realized_loss", 0.0),
            "total_pnl": okx_pnl["total_pnl"],
            "cumulative_total_pnl": okx_pnl["cumulative_total_pnl"],
            "total_pnl_pct": okx_pnl["total_pnl_pct"],
            "local_trade_total_pnl": pnl_summary.get("total_pnl", 0.0),
            "local_trade_today_pnl": pnl_summary.get("today_total_pnl", 0.0),
            "account_pnl_source": "okx_authoritative" if okx_balance_available else "okx_unavailable",
            "remaining_allocation": okx_available,
            "balance_error": okx_snapshot.get("balance_error"),
            "balance_status": okx_snapshot.get("balance_status"),
            "refresh_in_progress": bool(okx_snapshot.get("refresh_in_progress")),
            "balance_source": okx_snapshot.get("balance_source"),
            "okx_available_balance": okx_available,
            "okx_total_balance": okx_snapshot.get("total_balance"),
            "okx_cash_balance": okx_snapshot.get("cash_balance"),
            "okx_equity_balance": okx_snapshot.get("equity_balance"),
            "okx_used_balance": okx_snapshot.get("used_balance"),
            "max_allocatable_balance": okx_allocatable if okx_allocatable is not None else 0.0,
            "allocation_exceeds_balance": False,
            "risk_floor": risk_floor,
            "risk_paused": bool(pause_reason),
            "risk_pause_reason": pause_reason,
        }
    )
    if mode == "paper":
        summary = await _paper_execution_account_summary()
        status.update(
            {
                "paper_execution_available_balance": okx_available,
                "paper_execution_equity": summary.get("equity"),
                "paper_execution_used_margin": summary.get("used_margin"),
                "paper_execution_unrealized_pnl": pnl_summary.get("unrealized_pnl"),
                "initial_balance": None,
            }
        )
        return status

    return status


async def _sync_execution_account_to_paper_account() -> None:
    """No-op: OKX-backed paper accounts must not sync fixed local balances."""
    return


async def _sync_models_to_running_services() -> None:
    """Rebuild models from settings.ai_models and sync to running trading service."""
    if not _dash._trading_service or not _dash._trading_service.models:
        return

    import structlog

    log = structlog.get_logger(__name__)

    registry = _dash._trading_service.models
    old_names, new_names = await registry.sync_from_config()
    log.info("models synced from config", old=list(old_names), new=list(new_names))

    # Expert models analyze only; OKX-backed execution balances come from OKX.
    executor = _dash._trading_service.paper_executor
    if executor:
        executor._model_names = [ENSEMBLE_TRADER_NAME]

    # Update competition service active models and trigger evaluation
    if _dash._competition_service:
        _dash._competition_service.set_active_models([ENSEMBLE_TRADER_NAME])
        rankings = await _dash._competition_service.evaluate_all_models(force=True)
        log.info(
            "rankings updated", count=len(rankings), models=[r["model_name"] for r in rankings]
        )


# ── OKX Settings (split paper / live) ──


# ── Model Server Settings ──


@router.get("/settings/model-server")
async def get_model_server_settings():
    """Return encrypted model-server settings without exposing the password."""
    try:
        payload = await get_model_server_settings_public()
    except ModelServerConfigError as exc:
        raise HTTPException(status_code=503, detail=_model_server_error(exc)) from exc
    return payload.as_dict()


@router.post("/settings/model-server")
async def update_model_server_settings(req: ModelServerSettingsRequest):
    """Save model-server SSH settings for hardware/model monitoring."""
    if not str(req.host or "").strip():
        raise HTTPException(status_code=400, detail="模型服务器地址不能为空")
    if req.port is None:
        raise HTTPException(status_code=400, detail="模型服务器 SSH 端口不能为空")
    if not str(req.username or "").strip():
        raise HTTPException(status_code=400, detail="模型服务器用户名不能为空")

    try:
        payload = await save_model_server_settings(
            host=req.host or "",
            port=req.port,
            username=req.username or "",
            password=req.password,
        )
    except ModelServerConfigNotConfigured as exc:
        raise HTTPException(status_code=400, detail=_model_server_error(exc)) from exc
    except ModelServerConfigError as exc:
        raise HTTPException(status_code=400, detail=_model_server_error(exc)) from exc

    clear_server_monitor_cache()
    return {
        "status": "ok",
        "message": "模型服务器配置已加密保存。",
        "settings": payload.as_dict(),
    }


@router.post("/settings/model-server/test")
async def test_model_server_settings(req: ModelServerSettingsRequest | None = None):
    """Test the saved or submitted model-server settings."""
    try:
        if req and any(
            value is not None for value in (req.host, req.port, req.username, req.password)
        ):
            if (
                not str(req.host or "").strip()
                or req.port is None
                or not str(req.username or "").strip()
            ):
                raise ModelServerConfigNotConfigured("请先填写地址、端口和用户名。")
            info = await build_model_server_info_from_update(
                host=req.host or "",
                port=req.port,
                username=req.username or "",
                password=req.password,
            )
        else:
            info = await load_model_server_info_from_secure_settings()
    except ModelServerConfigNotConfigured as exc:
        raise HTTPException(status_code=400, detail=_model_server_error(exc)) from exc
    except ModelServerConfigError as exc:
        raise HTTPException(status_code=400, detail=_model_server_error(exc)) from exc

    service = ServerMonitorStatusService(info_loader=lambda _root: info)
    result = await asyncio.to_thread(service.collect_sync)
    return {
        "success": bool(result.get("available")),
        "status": result.get("status"),
        "message": result.get("message") or ("连接成功" if result.get("available") else "连接失败"),
        "host": info.host,
        "result": result,
    }


@router.get("/settings/okx")
async def get_okx_settings():
    """Return both paper and live OKX config with secrets masked."""
    paper_creds = settings.get_okx_credentials("paper")
    live_creds = settings.get_okx_credentials("live")
    return {
        "paper": {
            "api_key": mask_secret(paper_creds.get("api_key", "")),
            "has_secret": bool(paper_creds.get("api_secret")),
            "has_passphrase": bool(paper_creds.get("passphrase")),
        },
        "live": {
            "api_key": mask_secret(live_creds.get("api_key", "")),
            "has_secret": bool(live_creds.get("api_secret")),
            "has_passphrase": bool(live_creds.get("passphrase")),
        },
    }


@router.post("/settings/okx")
async def update_okx_settings(req: OKXSettingsRequest):
    """Update OKX credentials for a specific mode (paper or live)."""
    if req.mode not in ("paper", "live"):
        raise HTTPException(status_code=400, detail="mode must be 'paper' or 'live'")

    prefix = "OKX_PAPER" if req.mode == "paper" else "OKX_LIVE"
    updates = {}
    if req.api_key is not None and req.api_key.strip() and not _is_masked_secret(req.api_key):
        updates[f"{prefix}_API_KEY"] = req.api_key.strip()
        await set_runtime_secret(f"okx.{req.mode}.api_key", req.api_key.strip())
        setattr(settings, f"okx_{req.mode}_api_key", req.api_key.strip())
    if (
        req.api_secret is not None
        and req.api_secret.strip()
        and not _is_masked_secret(req.api_secret)
    ):
        updates[f"{prefix}_API_SECRET"] = req.api_secret.strip()
        await set_runtime_secret(f"okx.{req.mode}.api_secret", req.api_secret.strip())
        setattr(settings, f"okx_{req.mode}_api_secret", req.api_secret.strip())
    if req.passphrase is not None and not _is_masked_secret(req.passphrase):
        updates[f"{prefix}_PASSPHRASE"] = req.passphrase.strip()
        if req.passphrase.strip():
            await set_runtime_secret(f"okx.{req.mode}.passphrase", req.passphrase.strip())
        setattr(settings, f"okx_{req.mode}_passphrase", req.passphrase.strip())

    if updates:
        env_updates = strip_secret_env_updates(updates)
        if env_updates:
            settings.update_env_file(env_updates)
        _clear_okx_snapshot_caches(req.mode)

    # Reinitialize connections so displayed OKX balances immediately use the
    # latest credentials. Failures do not block saving the credentials.
    current_mode = settings.trading_mode.value
    if req.mode == current_mode:
        if _dash._data_service:
            try:
                await _dash._data_service.rest_client.reinitialize()
            except Exception as exc:
                logger.warning(
                    "failed to reinitialize data service after OKX credential update",
                    mode=req.mode,
                    error=_connection_error_text(exc),
                )
    if _dash._trading_service:
        from executor.okx_executor import OKXExecutor

        attr = "_okx_live" if req.mode == "live" else "_okx_paper"
        old_executor = getattr(_dash._trading_service, attr, None)
        if old_executor:
            try:
                await old_executor.shutdown()
            except Exception as exc:
                logger.debug(
                    "failed to shutdown previous OKX executor after credential update",
                    mode=req.mode,
                    error=_connection_error_text(exc),
                )
        new_executor = _make_okx_executor(OKXExecutor, req.mode)
        try:
            await new_executor.initialize()
            setattr(_dash._trading_service, attr, new_executor)
        except Exception as exc:
            logger.warning(
                "failed to initialize OKX executor after credential update",
                mode=req.mode,
                error=_connection_error_text(exc),
            )
            setattr(_dash._trading_service, attr, None)

    return {
        "status": "ok",
        "message": f"{req.mode} settings saved.",
        "updated_keys": list(updates.keys()),
    }


@router.post("/settings/okx/test")
async def test_okx_connection(req: OKXTestRequest):
    """Test OKX credentials for paper or live mode by fetching balance."""
    if req.mode not in ("paper", "live"):
        raise HTTPException(status_code=400, detail="mode must be 'paper' or 'live'")

    creds = settings.get_okx_credentials(req.mode)
    if not creds.get("api_key") or not creds.get("api_secret"):
        return {"success": False, "error": "请先配置 API Key 和 API Secret"}
    if not creds.get("passphrase"):
        return {"success": False, "error": "请填写 Passphrase（创建 OKX API Key 时设置的密码短语）"}

    from executor.okx_executor import OKXExecutor

    executor = _make_okx_executor(OKXExecutor, req.mode)
    try:
        await executor.initialize()
        snapshot = await executor.get_balance_snapshot("USDT")
        if snapshot.get("error"):
            return {
                "success": False,
                "error": f"连接失败: {_connection_error_text(snapshot.get('error'))}",
            }
        usdt = float(snapshot.get("free") or 0.0)
        used = float(snapshot.get("used") or 0.0)
        total = float(snapshot.get("total") or 0.0)
        cash = float(snapshot.get("cash") or total or 0.0)
        equity = float(snapshot.get("equity") or total or 0.0)
        allocatable = float(snapshot.get("allocatable") or equity or total or usdt)
        return {
            "success": True,
            "message": (
                f"连接成功 ({req.mode})。USDT 账户余额: {cash:.2f}，"
                f"权益: {equity:.2f}，可交易: {usdt:.2f}，占用: {used:.2f}"
            ),
            "balance_usdt": allocatable,
            "available_balance": usdt,
            "used_balance": used,
            "total_balance": total,
            "cash_balance": cash,
            "equity_balance": equity,
            "allocatable_balance": allocatable,
        }
    except Exception as exc:
        return {
            "success": False,
            "error": f"连接失败: {_connection_error_text(exc)}",
        }
    finally:
        try:
            await executor.shutdown()
        except Exception as exc:
            logger.debug(
                "OKX executor shutdown failed after connection test",
                mode=req.mode,
                error=_connection_error_text(exc),
            )


@router.get("/settings/okx/balance")
async def get_okx_balances():
    """Fetch USDT balance for both paper and live OKX accounts."""
    result: dict = {"paper": None, "live": None, "paper_error": None, "live_error": None}

    for mode in ("paper", "live"):
        creds = settings.get_okx_credentials(mode)
        if not creds.get("api_key"):
            result[f"{mode}_error"] = "未配置API密钥"
            continue
        snapshot = await _get_okx_usdt_snapshot(mode, force=True)
        if snapshot.get("balance_error"):
            result[f"{mode}_error"] = snapshot.get("balance_error")
        elif snapshot.get("allocatable_balance") is not None:
            result[mode] = snapshot.get("allocatable_balance")
        else:
            result[f"{mode}_error"] = "查询失败"

    return result


@router.get("/settings/execution-account")
async def get_execution_account_settings():
    """Return unified execution-account settings and current balances."""
    paper_status, live_status = await asyncio.gather(
        _execution_account_status("paper"),
        _execution_account_status("live"),
    )
    return {
        "paper": paper_status,
        "live": live_status,
    }


@router.post("/settings/execution-account")
async def update_execution_account_settings(req: ExecutionAccountRequest):
    """Update execution-account display name and percent-based risk controls."""
    mode = "live" if req.mode == "live" else "paper"
    updates: dict[str, str] = {}

    if req.account_name is not None:
        account_name = req.account_name.strip() or "多专家执行账户"
        settings.execution_account_name = account_name
        updates["EXECUTION_ACCOUNT_NAME"] = account_name

    if req.max_loss_pct is not None:
        if req.max_loss_pct < 0 or req.max_loss_pct > 1:
            raise HTTPException(status_code=400, detail="最高亏损比例必须在 0 到 1 之间")
        settings.execution_account_max_loss_pct[mode] = float(req.max_loss_pct)
        updates["EXECUTION_ACCOUNT_MAX_LOSS_PCT"] = json.dumps(
            settings.execution_account_max_loss_pct,
            ensure_ascii=False,
        )

    if req.cooldown_loss_pct is not None:
        if req.cooldown_loss_pct < 0 or req.cooldown_loss_pct > 1:
            raise HTTPException(status_code=400, detail="冷静期触发比例必须在 0 到 100% 之间")
        settings.execution_account_cooldown_loss_pct[mode] = float(req.cooldown_loss_pct)
        updates["EXECUTION_ACCOUNT_COOLDOWN_LOSS_PCT"] = json.dumps(
            settings.execution_account_cooldown_loss_pct,
            ensure_ascii=False,
        )

    if updates:
        settings.update_env_file(updates)

    paper_status, live_status = await asyncio.gather(
        _execution_account_status("paper"),
        _execution_account_status("live"),
    )

    return {
        "status": "ok",
        "message": "执行账户风控设置已保存；下单资金自动使用 OKX 当前可用余额。",
        "paper": paper_status,
        "live": live_status,
    }


# ── AI Model CRUD ──


@router.get("/settings/ai-models")
async def get_ai_models():
    """Return fixed expert model slots with api_key masked.

    Keep this endpoint lightweight for the settings page; OKX balance checks are
    intentionally handled by /settings/okx/balance and test endpoints.
    """
    models = []
    for m in settings.get_fixed_ai_models(include_empty=True):
        mc = dict(m)
        mc["api_key"] = mask_secret(mc.get("api_key", ""))
        mc["execution_mode"] = "analysis"
        models.append(mc)

    execution_accounts = {
        "paper": settings.get_execution_account_config("paper"),
        "live": settings.get_execution_account_config("live"),
    }
    okx_info: dict = {
        "paper_balance": None,
        "paper_error": None,
        "live_balance": None,
        "live_error": None,
        "skipped": True,
    }

    return {
        "models": models,
        "legacy": [],
        "execution_model": ENSEMBLE_TRADER_NAME,
        "execution_account": execution_accounts,
        "okx": okx_info,
    }


@router.post("/settings/ai-models")
async def add_ai_model_fixed(req: AIModelRequest):
    """Fixed expert slots cannot be added from the UI."""
    raise HTTPException(status_code=400, detail="模型槽位已固定，请直接编辑页面中的专家模型。")


@router.put("/settings/ai-models/{name}")
async def update_ai_model_fixed(name: str, req: AIModelRequest):
    """Update one fixed expert model slot."""
    try:
        updates: dict[str, Any] = {}
        if req.api_base is not None:
            updates["api_base"] = _normalize_api_base_or_400(
                req.api_base,
                field_name="AI model API base",
            )
        if req.model is not None:
            updates["model"] = req.model
        if req.api_key is not None and req.api_key.strip() and not _is_masked_secret(req.api_key):
            updates["api_key"] = req.api_key
            await set_runtime_secret(secure_ai_model_key(name), req.api_key.strip())
        updated = settings.set_fixed_ai_model(name, updates)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=safe_error_text(e)) from e

    settings.update_env_file(
        {"AI_MODELS": json.dumps(scrub_ai_model_env(settings.ai_models), ensure_ascii=False)}
    )
    await _sync_models_to_running_services()
    return {"status": "ok", "message": f"Model '{name}' updated.", "model": _masked(updated)}


@router.delete("/settings/ai-models/{name}")
async def delete_ai_model_fixed(name: str):
    """Fixed expert slots cannot be deleted."""
    raise HTTPException(
        status_code=400, detail="固定专家模型不能删除，只能清空 Key 或修改模型配置。"
    )


@router.post("/settings/ai-models")
async def add_ai_model(req: AIModelRequest):
    """Add a new AI model configuration (paper or live)."""
    if not req.name or not req.name.strip():
        raise HTTPException(status_code=400, detail="Model name is required")

    # Check for duplicate name
    for m in settings.ai_models:
        if m.get("name") == req.name.strip():
            raise HTTPException(status_code=400, detail=f"Model '{req.name}' already exists")

    mode = req.execution_mode or "paper"

    new_model: dict[str, Any] = {
        "name": req.name.strip(),
        "api_base": _normalize_api_base_or_400(
            req.api_base,
            field_name="AI model API base",
        ),
        "api_key": (req.api_key or "").strip(),
        "model": (req.model or "gpt-4").strip(),
        "execution_mode": mode,
    }
    if new_model["api_key"]:
        await set_runtime_secret(secure_ai_model_key(new_model["name"]), new_model["api_key"])

    settings.ai_models.append(new_model)
    env_updates = {
        "AI_MODELS": json.dumps(scrub_ai_model_env(settings.ai_models), ensure_ascii=False)
    }
    env_updates = strip_secret_env_updates(env_updates)
    settings.update_env_file(env_updates)

    await _sync_models_to_running_services()

    return {"status": "ok", "message": f"Model '{req.name}' added.", "model": _masked(new_model)}


@router.put("/settings/ai-models/{name}")
async def update_ai_model(name: str, req: AIModelRequest):
    """Update an existing AI model configuration."""
    for i, m in enumerate(settings.ai_models):
        if m.get("name") == name:
            updated = dict(m)
            if req.name and req.name.strip() and req.name.strip() != name:
                updated["name"] = req.name.strip()
            if req.api_base is not None:
                updated["api_base"] = _normalize_api_base_or_400(
                    req.api_base,
                    field_name="AI model API base",
                )
            if (
                req.api_key is not None
                and req.api_key.strip()
                and not _is_masked_secret(req.api_key)
            ):
                updated["api_key"] = req.api_key.strip()
                await set_runtime_secret(
                    secure_ai_model_key(updated.get("name", name)), req.api_key.strip()
                )
            if req.model is not None and req.model.strip():
                updated["model"] = req.model.strip()
            if req.execution_mode:
                updated["execution_mode"] = req.execution_mode
            updated.pop("balance", None)
            settings.ai_models[i] = updated
            env_updates = {
                "AI_MODELS": json.dumps(scrub_ai_model_env(settings.ai_models), ensure_ascii=False)
            }
            env_updates = strip_secret_env_updates(env_updates)
            settings.update_env_file(env_updates)
            await _sync_models_to_running_services()
            return {
                "status": "ok",
                "message": f"Model '{name}' updated.",
                "model": _masked(updated),
            }

    raise HTTPException(status_code=404, detail=f"Model '{name}' not found")


@router.delete("/settings/ai-models/{name}")
async def delete_ai_model(name: str):
    """Delete an AI model configuration."""
    # Check configured models first
    for i, m in enumerate(settings.ai_models):
        if m.get("name") == name:
            settings.ai_models.pop(i)
            settings.update_env_file(
                {
                    "AI_MODELS": json.dumps(
                        scrub_ai_model_env(settings.ai_models), ensure_ascii=False
                    )
                }
            )
            await _sync_models_to_running_services()
            return {"status": "ok", "message": f"Model '{name}' deleted."}

    # Legacy fallback: if no models configured and name matches legacy, clear it
    if not settings.ai_models and name == "llm_agent":
        settings.ai_api_key = ""
        settings.ai_api_base = ""
        settings.ai_model = ""
        settings.update_env_file(
            {
                "AI_MODELS": "[]",
                "AI_API_KEY": "",
                "AI_API_BASE": "",
                "AI_MODEL": "",
            }
        )
        await _sync_models_to_running_services()
        return {"status": "ok", "message": f"Legacy model '{name}' cleared."}

    raise HTTPException(status_code=404, detail=f"Model '{name}' not found")


@router.post("/settings/ai-models/test")
async def test_ai_model_connection(req: AIModelTestRequest):
    """Test an AI model's API connection by making a simple ChatOpenAI call."""
    # Resolve config: by name from settings.ai_models, or direct fields, or fallback
    api_base = req.api_base
    api_key = req.api_key
    model = req.model

    if req.name and (not api_base or not api_key or not model):
        for m in settings.get_fixed_ai_models(include_empty=True):
            if m.get("name") == req.name:
                api_base = api_base or m.get("api_base")
                api_key = api_key or m.get("api_key")
                model = model or m.get("model")
                break

    # Fallback to global settings
    api_base = api_base or settings.ai_api_base
    api_key = api_key or settings.ai_api_key
    model = model or settings.ai_model

    try:
        api_base = normalize_http_base_url(
            api_base,
            field_name="AI model API base",
        )
    except ValueError as exc:
        return {"success": False, "error": _connection_error_text(exc), "model": model}

    if not api_key:
        return {"success": False, "error": "No API key configured"}

    try:
        from langchain_core.messages import HumanMessage
        from langchain_openai import ChatOpenAI

        llm_kwargs: dict[str, Any] = {
            "base_url": api_base,
            "api_key": api_key,
            "model": model,
            "temperature": 0,
            "max_tokens": 10,
            "timeout": 15,
            "max_retries": 0,
        }
        if uses_thinking_tags(model):
            llm_kwargs["extra_body"] = non_thinking_extra_body()
            prompt = ensure_no_think_text("Hi")
        else:
            prompt = "Hi"

        llm = ChatOpenAI(
            **llm_kwargs,
        )
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        content = response.content if hasattr(response, "content") else str(response)
        safe_content = redact_output(content)[:100]
        return {
            "success": True,
            "message": f"Connection OK. Response: {safe_content}",
            "model": model,
        }
    except Exception as exc:
        return {"success": False, "error": _connection_error_text(exc), "model": model}


class IntervalRequest(BaseModel):
    interval_seconds: int


class ThresholdsRequest(BaseModel):
    decision_interval: int | None = None
    confidence_threshold: float | None = None
    max_position_pct: float | None = None
    max_leverage: float | None = None
    max_daily_loss_pct: float | None = None
    hard_stop_loss_pct: float | None = None
    max_open_positions_per_model: int | None = None
    max_same_symbol_positions_per_side: int | None = None
    total_margin_limit_pct: float | None = None
    max_slippage_pct: float | None = None
    local_ai_tools_enabled: bool | None = None
    local_ai_tools_api_base: str | None = None
    local_ai_tools_timeout_seconds: float | None = None
    local_ai_tools_circuit_breaker_failures: int | None = None
    local_ai_tools_circuit_breaker_cooldown_seconds: float | None = None
    high_risk_review_enabled: bool | None = None
    high_risk_review_api_base: str | None = None
    high_risk_review_api_key: str | None = None
    high_risk_review_model: str | None = None
    high_risk_review_timeout_seconds: float | None = None
    high_risk_review_max_tokens: int | None = None
    high_risk_review_circuit_breaker_failures: int | None = None
    high_risk_review_circuit_breaker_cooldown_seconds: float | None = None


def _effective_total_margin_limit_pct() -> float:
    configured = float(settings.max_total_margin_pct or 0.0)
    if configured > 0:
        return configured
    return float(settings.max_position_pct or 0.0) * 3


def _pct(value: float, digits: int = 2) -> str:
    return f"{float(value) * 100:.{digits}f}%"


def _ratio(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _threshold_item(
    *,
    key: str,
    label: str,
    current: Any,
    current_display: str | None = None,
    effective: Any | None = None,
    effective_display: str | None = None,
    unit: str = "",
    source: str = "",
    surface: str = "",
    bounds: dict[str, Any] | None = None,
    effect: str,
    increase_effect: str = "",
    decrease_effect: str = "",
    automation: str = "",
    reason: str = "",
    status: str = "active",
) -> dict[str, Any]:
    payload = {
        "key": key,
        "label": label,
        "current": current,
        "current_display": current_display if current_display is not None else str(current),
        "effective": current if effective is None else effective,
        "effective_display": (
            effective_display
            if effective_display is not None
            else (current_display if effective is None else str(effective))
        ),
        "unit": unit,
        "source": source,
        "surface": surface,
        "bounds": bounds or {},
        "effect": effect,
        "increase_effect": increase_effect,
        "decrease_effect": decrease_effect,
        "automation": automation,
        "reason": reason,
        "status": status,
    }
    return payload


def _threshold_catalog() -> dict[str, Any]:
    from risk_manager.engine import (
        MIN_ENTRY_CONFIDENCE_AFTER_FEES,
        MIN_REWARD_RISK_RATIO,
        MIN_TAKE_PROFIT_AFTER_COSTS,
    )

    strategy = DEFAULT_TRADING_PARAMS
    learning = strategy.strategy_learning
    total_margin_limit_pct = _effective_total_margin_limit_pct()
    confidence_reference = max(
        float(settings.confidence_threshold or 0.0),
        MIN_ENTRY_CONFIDENCE_AFTER_FEES,
    )

    manual_editable = [
        _threshold_item(
            key="decision_interval_seconds",
            label="决策间隔",
            current=settings.decision_interval_seconds,
            current_display=f"{settings.decision_interval_seconds} 秒",
            unit="seconds",
            source="DECISION_INTERVAL_SECONDS",
            surface="系统设置 - 交易参数",
            bounds={"min": 10, "max": 3600, "step": 5},
            effect="控制主循环触发市场分析和持仓复盘的基础频率。",
            increase_effect="调大会降低模型调用频率和成本，但行情响应会变慢。",
            decrease_effect="调小会更快响应行情，但会增加模型调用、接口压力和重复分析概率。",
            reason="属于资源节奏配置，系统无法替你决定可接受的成本和延迟。",
        ),
        _threshold_item(
            key="confidence_threshold",
            label="最低信心阈值",
            current=settings.confidence_threshold,
            current_display=_ratio(settings.confidence_threshold, 2),
            effective=confidence_reference,
            effective_display=_ratio(confidence_reference, 2),
            source="CONFIDENCE_THRESHOLD",
            surface="系统设置 - 交易参数",
            bounds={"min": 0.10, "max": 1.00, "step": 0.05},
            effect=(
                "影响专家提示、集成决策低信心处理和仓位收缩；手续费后参考线 "
                f"{MIN_ENTRY_CONFIDENCE_AFTER_FEES:.2f} 不会被手动调低。"
            ),
            increase_effect="调高会减少低信心候选和小单探针，但可能错过部分早期机会。",
            decrease_effect="调低会增加候选数量，但不会绕过净收益、容量、滑点和硬风控。",
            reason="这是交易风格偏好，不应由系统自动放松。",
        ),
        _threshold_item(
            key="total_margin_limit_pct",
            label="总保证金占用上限",
            current=settings.max_total_margin_pct,
            current_display=(
                _pct(settings.max_total_margin_pct)
                if float(settings.max_total_margin_pct or 0.0) > 0
                else "未显式配置"
            ),
            effective=total_margin_limit_pct,
            effective_display=_pct(total_margin_limit_pct),
            source="MAX_TOTAL_MARGIN_PCT",
            surface="系统设置 - 交易参数",
            bounds={"min": 0.10, "max": 1.00, "step": 0.01},
            effect="限制账户同时占用的保证金比例，达到后停止新开仓但仍复盘和平仓。",
            increase_effect="调高会允许更多仓位同时占用保证金，回撤和爆仓风险同步放大。",
            decrease_effect="调低会更早停止新开仓，资金更保守但轮转速度下降。",
            reason="这是账户最大风险承受能力，必须人工决定。",
        ),
        _threshold_item(
            key="max_slippage_pct",
            label="最大滑点上限",
            current=settings.max_slippage_pct,
            current_display=_pct(settings.max_slippage_pct),
            source="MAX_SLIPPAGE_PCT",
            surface="系统设置 - 交易参数",
            bounds={"min": 0.0002, "max": 0.02, "step": 0.0001},
            effect="这是下单和评分可接受滑点的安全上限；真实成本由盘口价差和深度动态估算。",
            increase_effect="调高会放过更差成交价，成交率可能升高但净收益更容易被吃掉。",
            decrease_effect="调低会拒绝更多流动性差的订单，成交更挑剔但可能错过急速行情。",
            reason="上限代表你愿意承受的最差成交偏差，不应由系统自动扩大。",
        ),
    ]

    manual_service_controls = [
        _threshold_item(
            key="local_ai_tools_timeout_seconds",
            label="本地量化 AI 工具超时",
            current=settings.local_ai_tools_timeout_seconds,
            current_display=f"{settings.local_ai_tools_timeout_seconds:.1f} 秒",
            source="LOCAL_AI_TOOLS_TIMEOUT_SECONDS",
            surface="系统设置 - 交易参数",
            bounds={"min": 0.2, "max": 15.0, "step": 0.1},
            effect="控制盈利、时序、情绪等本地辅助证据的等待时间。",
            increase_effect="调高会给本地工具更多时间返回证据，但会拖慢一轮分析。",
            decrease_effect="调低会更快跳过慢服务，但辅助证据缺失会变多。",
            reason="取决于本地/服务器工具性能，需要人工按部署环境配置。",
        ),
        _threshold_item(
            key="local_ai_tools_circuit_breaker_failures",
            label="本地工具熔断失败次数",
            current=settings.local_ai_tools_circuit_breaker_failures,
            source="LOCAL_AI_TOOLS_CIRCUIT_BREAKER_FAILURES",
            surface="系统设置 - 交易参数",
            bounds={"min": 1, "max": 20, "step": 1},
            effect="连续失败达到次数后，系统暂时跳过本地工具，避免拖垮主交易循环。",
            increase_effect="调高会更容忍偶发失败，但异常服务会拖慢更久。",
            decrease_effect="调低会更快隔离异常服务，但短暂网络抖动也可能触发熔断。",
        ),
        _threshold_item(
            key="local_ai_tools_circuit_breaker_cooldown_seconds",
            label="本地工具熔断冷却",
            current=settings.local_ai_tools_circuit_breaker_cooldown_seconds,
            current_display=f"{settings.local_ai_tools_circuit_breaker_cooldown_seconds:.0f} 秒",
            source="LOCAL_AI_TOOLS_CIRCUIT_BREAKER_COOLDOWN_SECONDS",
            surface="系统设置 - 交易参数",
            bounds={"min": 5, "max": 3600, "step": 5},
            effect="本地工具熔断后等待多久再尝试恢复。",
            increase_effect="调高会减少异常服务反复重试，但恢复变慢。",
            decrease_effect="调低会更快恢复，但故障未好时会更频繁重试。",
        ),
        _threshold_item(
            key="high_risk_review_timeout_seconds",
            label="高风险复核超时",
            current=settings.high_risk_review_timeout_seconds,
            current_display=f"{settings.high_risk_review_timeout_seconds:.0f} 秒",
            source="HIGH_RISK_REVIEW_TIMEOUT_SECONDS",
            surface="系统设置 - 交易参数",
            bounds={"min": 5, "max": 120, "step": 1},
            effect="控制重大冲突时等待复核模型的最长时间。",
            increase_effect="调高会给复核模型更多思考时间，但可能错过快速风控动作。",
            decrease_effect="调低会更快回到主模型决策，但复核结论缺失会变多。",
        ),
        _threshold_item(
            key="high_risk_review_max_tokens",
            label="高风险复核最大输出 Token",
            current=settings.high_risk_review_max_tokens,
            source="HIGH_RISK_REVIEW_MAX_TOKENS",
            surface="系统设置 - 交易参数",
            bounds={
                "min": HIGH_RISK_REVIEW_TOKEN_FLOOR,
                "max": HIGH_RISK_REVIEW_TOKEN_CAP,
                "step": 20,
            },
            effect="限制复核模型输出长度，防止长推理拖慢交易链路。",
            increase_effect="调高会允许更完整解释，但延迟和费用可能上升。",
            decrease_effect="调低会更快、更省，但复杂冲突可能解释不足。",
        ),
    ]

    manual_hard_guards = [
        _threshold_item(
            key="max_position_pct",
            label="单笔保证金上限",
            current=settings.max_position_pct,
            current_display=_pct(settings.max_position_pct),
            source="MAX_POSITION_PCT",
            surface="系统设置 - 交易参数",
            effect="限制单笔合约开仓保证金占账户权益的比例。",
            increase_effect="调高会放大单笔亏损和保证金占用。",
            decrease_effect="调低会让单笔更小，降低风险但盈利弹性也更低。",
            reason="硬风险上限只能人工改，系统不会自动放大。",
        ),
        _threshold_item(
            key="max_leverage",
            label="最大杠杆上限",
            current=settings.max_leverage,
            current_display=f"{settings.max_leverage:.1f}x",
            source="MAX_LEVERAGE",
            surface="系统设置 - 交易参数",
            effect="限制所有开仓最终杠杆；运行态会按证据质量再压到 5x/10x/20x 以内。",
            increase_effect="调高会允许更高名义敞口和更快亏损放大。",
            decrease_effect="调低会让高质量机会也只能低杠杆执行。",
            reason="杠杆是账户级硬风险，不允许自动放松。",
        ),
        _threshold_item(
            key="max_daily_loss_pct",
            label="日内最大亏损比例",
            current=settings.max_daily_loss_pct,
            current_display=_pct(settings.max_daily_loss_pct),
            source="MAX_DAILY_LOSS_PCT",
            surface="系统设置 - 交易参数",
            effect="触发账户熔断和执行账户亏损控制。",
            increase_effect="调高会延后停手，可能扩大当天亏损。",
            decrease_effect="调低会更早暂停新交易，保护本金但可能错过修复机会。",
            reason="亏损容忍度必须人工确认。",
        ),
        _threshold_item(
            key="hard_stop_loss_pct",
            label="硬止损比例",
            current=settings.hard_stop_loss_pct,
            current_display=_pct(settings.hard_stop_loss_pct),
            source="HARD_STOP_LOSS_PCT",
            surface="系统设置 - 交易参数",
            effect="用于止损管理和缺省风险测算。",
            increase_effect="调高会给仓位更大回撤空间，但单笔亏损可能扩大。",
            decrease_effect="调低会更快止损，降低尾部风险但更容易被普通波动洗出。",
            reason="止损深度不能交给系统自动放宽。",
        ),
        _threshold_item(
            key="max_same_symbol_positions_per_side",
            label="同币同向持仓组数上限",
            current=settings.max_same_symbol_positions_per_side,
            source="MAX_SAME_SYMBOL_POSITIONS_PER_SIDE",
            surface="系统设置 - 交易参数",
            effect="防止同一个币种同方向重复堆仓。",
            increase_effect="调高会允许同方向加速集中暴露。",
            decrease_effect="调低会更严格避免重复仓，但可能限制分批建仓。",
            reason="集中度硬上限不自动放松。",
        ),
        _threshold_item(
            key="max_open_positions_per_model",
            label="基础持仓组数上限",
            current=settings.max_open_positions_per_model,
            source="MAX_OPEN_POSITIONS_PER_MODEL",
            surface="系统设置 - 交易参数",
            effect="动态容量策略以此为基础计算运行持仓容量。",
            increase_effect="调高会扩大组合复杂度和保证金压力。",
            decrease_effect="调低会减少同时持仓数量，让系统更集中但机会覆盖变少。",
            reason="这是组合容量的人工上限，动态策略只能在其约束内调度。",
        ),
        _threshold_item(
            key="auto_scan_symbol_limit",
            label="自动扫描币种上限",
            current=settings.auto_scan_symbol_limit,
            source="AUTO_SCAN_SYMBOL_LIMIT",
            surface=".env / config.settings",
            effect="限制每轮候选池规模和行情特征抓取成本。",
            increase_effect="调高会扩大候选覆盖，但接口、模型和延迟成本上升。",
            decrease_effect="调低会更省资源，但可能漏掉更优交易对。",
            reason="这是资源预算，不应由系统在未获授权时自动扩大。",
        ),
    ]

    auto_tunable = [
        _threshold_item(
            key="min_entry_volume_ratio",
            label="入场量能参考线",
            current=settings.min_entry_volume_ratio,
            current_display=_ratio(settings.min_entry_volume_ratio, 2),
            effective="strategy_learning_runtime",
            effective_display="策略学习运行态自动计算",
            source="settings bootstrap + strategy_learning",
            surface="自动调度，不在交易参数页手动调整",
            bounds={
                "min": learning.entry_volume_ratio_min,
                "default": learning.entry_volume_ratio_default,
                "max": learning.entry_volume_ratio_max,
            },
            effect="用于候选排序、仓位和杠杆质量参考，不是硬开仓门槛。",
            automation="策略学习根据近期胜率、亏损、释放压力和市场状态动态收紧或放松。",
            reason="已自动化；配置值只在运行上下文缺失时作为启动默认值。",
        ),
        _threshold_item(
            key="min_entry_adx",
            label="入场 ADX 参考线",
            current=settings.min_entry_adx,
            current_display=_ratio(settings.min_entry_adx, 1),
            effective="strategy_learning_runtime",
            effective_display="策略学习运行态自动计算",
            source="settings bootstrap + strategy_learning",
            surface="自动调度，不在交易参数页手动调整",
            bounds={
                "min": learning.entry_adx_min,
                "default": learning.entry_adx_default,
                "max": learning.entry_adx_max,
            },
            effect="用于衡量趋势强度，影响排序、仓位和杠杆参考。",
            automation="策略学习按历史交易质量生成运行态 entry_filters。",
            reason="已自动化；人工长期固定容易让系统在震荡/趋势切换时走偏。",
        ),
        _threshold_item(
            key="strategy_profile_sizing",
            label="策略画像仓位参数",
            current="global_min_score_delta / position_size_multiplier / probe_fraction",
            effective="active_strategy_profile",
            effective_display="由策略学习画像和回滚机制管理",
            source="services.strategy_learning",
            surface="策略学习页/自动画像，不在交易参数页手动调整",
            bounds={
                "global_min_score_delta": [-0.25, 0.35],
                "position_size_multiplier": [0.10, 1.25],
                "probe_fraction": [0.0, 0.10],
                "max_probe_size_pct": [0.0, strategy.entry_risk_sizing.strategy_probe_cap_max_pct],
            },
            effect="控制候选门槛、小仓探针比例和仓位放大/收缩。",
            automation="系统根据策略画像表现启用、禁用或回滚，人工不直接改底层数值。",
            reason="属于学习策略，不应在交易参数页临时手动改。",
        ),
        _threshold_item(
            key="dynamic_position_capacity",
            label="动态持仓容量",
            current=settings.max_open_positions_per_model,
            effective="runtime_capacity",
            effective_display="按持仓质量、释放候选和回撤自动计算",
            source="services.dynamic_position_capacity",
            surface="自动调度，受基础持仓组数上限约束",
            effect="决定新开不同币种/方向仓位前是否仍有容量。",
            automation="低质量持仓多、回撤扩大时收缩；释放轮转需要时预留槽位。",
            reason="自动调度只在人工硬上限内运行，不会自行扩大账户风险边界。",
        ),
        _threshold_item(
            key="execution_cost_estimate",
            label="执行成本估算",
            current=ESTIMATED_TAKER_FEE_PCT,
            current_display=f"手续费常量 {_pct(ESTIMATED_TAKER_FEE_PCT, 3)} + 动态滑点",
            effective="orderbook_runtime",
            effective_display="按盘口价差、深度、失衡和滑点上限动态估算",
            source="services.execution_cost_model",
            surface="自动估算，最大滑点上限仍由人工配置",
            effect="用于净收益评分，避免把最大滑点误当每笔固定成本。",
            automation="系统逐笔从市场微结构估算，超过人工滑点上限则拒绝或降级。",
            reason="手续费常量不是后台阈值；滑点真实值应由盘口数据计算。",
        ),
    ]

    removed_or_deprecated = [
        _threshold_item(
            key="max_auto_trades_per_round",
            label="每轮最大自动交易数",
            current="removed",
            current_display="已删除",
            source="旧 MAX_AUTO_TRADES_PER_ROUND",
            effect="旧字段没有执行链读取，保留会误导系统以为它能限制开仓。",
            reason="已从 Settings 和 .env.example 移除；现在由动态容量、保证金上限和候选评分控制。",
            status="removed",
        ),
        _threshold_item(
            key="daily_profit_target_usdt_cny",
            label="每日盈利目标",
            current="removed",
            current_display="已删除",
            source="旧 DAILY_PROFIT_TARGET_USDT / DAILY_PROFIT_TARGET_CNY",
            effect="每日盈利目标不参与交易判断，不能用来驱动硬开仓或追单。",
            reason="已从 Settings 和 .env.example 移除；收益页只展示实际盈亏。",
            status="removed",
        ),
        _threshold_item(
            key="fee.estimated_taker_fee_pct",
            label="策略快照手续费包装字段",
            current="removed",
            current_display="已删除，真实常量为 " + _pct(ESTIMATED_TAKER_FEE_PCT, 3),
            source="services.trading_params",
            effect="旧字段只在快照里出现，没有独立行为读取。",
            reason="保留模块常量 ESTIMATED_TAKER_FEE_PCT，删除伪可调包装字段。",
            status="removed",
        ),
        _threshold_item(
            key="entry_opportunity_gate.selected_side_positive_net_hard_gate",
            label="未接入的正净收益硬开关",
            current="removed",
            current_display="已删除",
            source="services.trading_params.EntryOpportunityGateParams",
            effect="旧字段没有执行链读取，不能实际改变开仓判断。",
            reason="删除未接入字段，避免后续 AI 把它当成可用硬门槛。",
            status="removed",
        ),
    ]

    return {
        "status": "ok",
        "policy": {
            "name": "threshold_governance_v1",
            "hard_risk_auto_relax": False,
            "manual_inputs_require_effect_explanation": True,
            "auto_tunable_not_rendered_as_manual_inputs": True,
            "removed_fake_thresholds": True,
            "strategy_snapshot_version": strategy.version,
            "notes": [
                "能由策略学习和运行态证据自动计算的参数，不再放进交易参数页让人手动调。",
                "账户级硬风险上限只允许人工配置，系统不会为了多开仓自动放松。",
                "没有行为接入的旧字段已经删除或列入废弃清单，避免开发时读错。",
            ],
        },
        "manual_editable": manual_editable,
        "manual_service_controls": manual_service_controls,
        "manual_hard_guards": manual_hard_guards,
        "auto_tunable": auto_tunable,
        "removed_or_deprecated": removed_or_deprecated,
        "risk_references": {
            "min_entry_confidence_after_fees": MIN_ENTRY_CONFIDENCE_AFTER_FEES,
            "min_take_profit_after_costs": MIN_TAKE_PROFIT_AFTER_COSTS,
            "min_reward_risk_ratio": MIN_REWARD_RISK_RATIO,
        },
    }


@router.get("/settings/thresholds")
async def get_thresholds():
    """Get current decision interval and confidence threshold."""
    settings.refresh_runtime_env(force=True)
    total_margin_limit_pct = _effective_total_margin_limit_pct()
    return {
        "decision_interval": settings.decision_interval_seconds,
        "confidence_threshold": settings.confidence_threshold,
        "max_position_pct": settings.max_position_pct,
        "max_leverage": settings.max_leverage,
        "max_daily_loss_pct": settings.max_daily_loss_pct,
        "hard_stop_loss_pct": settings.hard_stop_loss_pct,
        "max_open_positions_per_model": settings.max_open_positions_per_model,
        "max_same_symbol_positions_per_side": settings.max_same_symbol_positions_per_side,
        "local_ai_tools_enabled": settings.local_ai_tools_enabled,
        "local_ai_tools_api_base": settings.local_ai_tools_api_base,
        "local_ai_tools_timeout_seconds": settings.local_ai_tools_timeout_seconds,
        "local_ai_tools_circuit_breaker_failures": settings.local_ai_tools_circuit_breaker_failures,
        "local_ai_tools_circuit_breaker_cooldown_seconds": settings.local_ai_tools_circuit_breaker_cooldown_seconds,
        "high_risk_review_enabled": settings.high_risk_review_enabled,
        "high_risk_review_api_base": settings.high_risk_review_api_base,
        "high_risk_review_api_key": mask_secret(settings.high_risk_review_api_key),
        "high_risk_review_has_api_key": bool(settings.high_risk_review_api_key),
        "high_risk_review_model": settings.high_risk_review_model,
        "high_risk_review_timeout_seconds": settings.high_risk_review_timeout_seconds,
        "high_risk_review_max_tokens": settings.high_risk_review_max_tokens,
        "high_risk_review_token_floor": HIGH_RISK_REVIEW_TOKEN_FLOOR,
        "high_risk_review_token_cap": HIGH_RISK_REVIEW_TOKEN_CAP,
        "high_risk_review_circuit_breaker_failures": settings.high_risk_review_circuit_breaker_failures,
        "high_risk_review_circuit_breaker_cooldown_seconds": settings.high_risk_review_circuit_breaker_cooldown_seconds,
        "total_margin_limit_pct": total_margin_limit_pct,
        "max_slippage_pct": settings.max_slippage_pct,
    }


@router.get("/settings/threshold-catalog")
async def get_threshold_catalog():
    """Return threshold governance with automation and manual-change impact notes."""
    settings.refresh_runtime_env(force=True)
    return _threshold_catalog()


@router.post("/settings/thresholds")
async def update_thresholds(req: ThresholdsRequest):
    """Update decision interval and/or confidence threshold dynamically."""
    updates = {}

    if req.decision_interval is not None:
        if req.decision_interval < 10:
            raise HTTPException(status_code=400, detail="Interval must be at least 10 seconds")
        if req.decision_interval > 3600:
            raise HTTPException(status_code=400, detail="Interval must be at most 3600 seconds")
        settings.decision_interval_seconds = req.decision_interval
        updates["DECISION_INTERVAL_SECONDS"] = str(req.decision_interval)

    if req.confidence_threshold is not None:
        if req.confidence_threshold < 0.1:
            raise HTTPException(status_code=400, detail="Confidence threshold must be at least 0.1")
        if req.confidence_threshold > 1.0:
            raise HTTPException(status_code=400, detail="Confidence threshold must be at most 1.0")
        settings.confidence_threshold = req.confidence_threshold
        updates["CONFIDENCE_THRESHOLD"] = str(req.confidence_threshold)

    if req.max_position_pct is not None:
        if req.max_position_pct < 0.005:
            raise HTTPException(status_code=400, detail="单笔保证金上限不能低于 0.5%")
        if req.max_position_pct > 0.50:
            raise HTTPException(status_code=400, detail="单笔保证金上限不能超过 50%")
        settings.max_position_pct = float(req.max_position_pct)
        updates["MAX_POSITION_PCT"] = str(settings.max_position_pct)

    if req.max_leverage is not None:
        if req.max_leverage < 1:
            raise HTTPException(status_code=400, detail="最大杠杆不能低于 1x")
        if req.max_leverage > 125:
            raise HTTPException(status_code=400, detail="最大杠杆不能超过 125x")
        settings.max_leverage = float(req.max_leverage)
        updates["MAX_LEVERAGE"] = str(settings.max_leverage)

    if req.max_daily_loss_pct is not None:
        if req.max_daily_loss_pct < 0.001:
            raise HTTPException(status_code=400, detail="日内最大亏损不能低于 0.1%")
        if req.max_daily_loss_pct > 0.50:
            raise HTTPException(status_code=400, detail="日内最大亏损不能超过 50%")
        settings.max_daily_loss_pct = float(req.max_daily_loss_pct)
        updates["MAX_DAILY_LOSS_PCT"] = str(settings.max_daily_loss_pct)

    if req.hard_stop_loss_pct is not None:
        if req.hard_stop_loss_pct < 0.001:
            raise HTTPException(status_code=400, detail="硬止损不能低于 0.1%")
        if req.hard_stop_loss_pct > 0.50:
            raise HTTPException(status_code=400, detail="硬止损不能超过 50%")
        settings.hard_stop_loss_pct = float(req.hard_stop_loss_pct)
        updates["HARD_STOP_LOSS_PCT"] = str(settings.hard_stop_loss_pct)

    if req.max_open_positions_per_model is not None:
        if req.max_open_positions_per_model < 1:
            raise HTTPException(status_code=400, detail="基础持仓组数上限不能低于 1")
        if req.max_open_positions_per_model > 200:
            raise HTTPException(status_code=400, detail="基础持仓组数上限不能超过 200")
        settings.max_open_positions_per_model = int(req.max_open_positions_per_model)
        updates["MAX_OPEN_POSITIONS_PER_MODEL"] = str(settings.max_open_positions_per_model)

    if req.max_same_symbol_positions_per_side is not None:
        if req.max_same_symbol_positions_per_side < 1:
            raise HTTPException(status_code=400, detail="同币同向持仓组数上限不能低于 1")
        if req.max_same_symbol_positions_per_side > 20:
            raise HTTPException(status_code=400, detail="同币同向持仓组数上限不能超过 20")
        settings.max_same_symbol_positions_per_side = int(req.max_same_symbol_positions_per_side)
        updates["MAX_SAME_SYMBOL_POSITIONS_PER_SIDE"] = str(
            settings.max_same_symbol_positions_per_side
        )

    if req.max_slippage_pct is not None:
        if req.max_slippage_pct < 0.0002:
            raise HTTPException(status_code=400, detail="最大滑点上限不能低于 0.02%")
        if req.max_slippage_pct > 0.02:
            raise HTTPException(status_code=400, detail="最大滑点上限不能超过 2%")
        settings.max_slippage_pct = float(req.max_slippage_pct)
        updates["MAX_SLIPPAGE_PCT"] = str(settings.max_slippage_pct)

    if req.local_ai_tools_enabled is not None:
        settings.local_ai_tools_enabled = bool(req.local_ai_tools_enabled)
        updates["LOCAL_AI_TOOLS_ENABLED"] = "true" if settings.local_ai_tools_enabled else "false"

    if req.local_ai_tools_api_base is not None:
        settings.local_ai_tools_api_base = _normalize_api_base_or_400(
            req.local_ai_tools_api_base,
            field_name="Local AI tools API base",
        )
        updates["LOCAL_AI_TOOLS_API_BASE"] = settings.local_ai_tools_api_base

    if req.local_ai_tools_timeout_seconds is not None:
        if req.local_ai_tools_timeout_seconds < 0.2 or req.local_ai_tools_timeout_seconds > 15:
            raise HTTPException(
                status_code=400, detail="Local AI tools timeout must be between 0.2 and 15 seconds"
            )
        settings.local_ai_tools_timeout_seconds = float(req.local_ai_tools_timeout_seconds)
        updates["LOCAL_AI_TOOLS_TIMEOUT_SECONDS"] = str(settings.local_ai_tools_timeout_seconds)

    if req.local_ai_tools_circuit_breaker_failures is not None:
        if (
            req.local_ai_tools_circuit_breaker_failures < 1
            or req.local_ai_tools_circuit_breaker_failures > 20
        ):
            raise HTTPException(
                status_code=400,
                detail="Local AI tools circuit breaker failures must be between 1 and 20",
            )
        settings.local_ai_tools_circuit_breaker_failures = int(
            req.local_ai_tools_circuit_breaker_failures
        )
        updates["LOCAL_AI_TOOLS_CIRCUIT_BREAKER_FAILURES"] = str(
            settings.local_ai_tools_circuit_breaker_failures
        )

    if req.local_ai_tools_circuit_breaker_cooldown_seconds is not None:
        cooldown = float(req.local_ai_tools_circuit_breaker_cooldown_seconds)
        if cooldown < 5 or cooldown > 3600:
            raise HTTPException(
                status_code=400,
                detail="Local AI tools circuit breaker cooldown must be between 5 and 3600 seconds",
            )
        settings.local_ai_tools_circuit_breaker_cooldown_seconds = cooldown
        updates["LOCAL_AI_TOOLS_CIRCUIT_BREAKER_COOLDOWN_SECONDS"] = str(cooldown)

    if req.high_risk_review_enabled is not None:
        settings.high_risk_review_enabled = bool(req.high_risk_review_enabled)
        updates["HIGH_RISK_REVIEW_ENABLED"] = (
            "true" if settings.high_risk_review_enabled else "false"
        )

    if req.high_risk_review_api_base is not None:
        settings.high_risk_review_api_base = _normalize_api_base_or_400(
            req.high_risk_review_api_base,
            field_name="High-risk review API base",
        )
        updates["HIGH_RISK_REVIEW_API_BASE"] = settings.high_risk_review_api_base

    if req.high_risk_review_api_key is not None:
        api_key = req.high_risk_review_api_key.strip()
        if api_key and not _is_masked_secret(api_key):
            settings.high_risk_review_api_key = api_key
            updates["HIGH_RISK_REVIEW_API_KEY"] = settings.high_risk_review_api_key
            await set_runtime_secret("high_risk_review.api_key", api_key)

    if req.high_risk_review_model is not None:
        settings.high_risk_review_model = req.high_risk_review_model.strip()
        updates["HIGH_RISK_REVIEW_MODEL"] = settings.high_risk_review_model

    if req.high_risk_review_timeout_seconds is not None:
        timeout_seconds = float(req.high_risk_review_timeout_seconds)
        if timeout_seconds < 5 or timeout_seconds > 120:
            raise HTTPException(
                status_code=400,
                detail="High-risk review timeout must be between 5 and 120 seconds",
            )
        settings.high_risk_review_timeout_seconds = timeout_seconds
        updates["HIGH_RISK_REVIEW_TIMEOUT_SECONDS"] = str(timeout_seconds)

    if req.high_risk_review_max_tokens is not None:
        max_tokens = int(req.high_risk_review_max_tokens)
        if max_tokens < HIGH_RISK_REVIEW_TOKEN_FLOOR or max_tokens > HIGH_RISK_REVIEW_TOKEN_CAP:
            raise HTTPException(
                status_code=400,
                detail=(
                    "High-risk review max tokens must be between "
                    f"{HIGH_RISK_REVIEW_TOKEN_FLOOR} and {HIGH_RISK_REVIEW_TOKEN_CAP}"
                ),
            )
        settings.high_risk_review_max_tokens = max_tokens
        updates["HIGH_RISK_REVIEW_MAX_TOKENS"] = str(max_tokens)

    if req.high_risk_review_circuit_breaker_failures is not None:
        failures = int(req.high_risk_review_circuit_breaker_failures)
        if failures < 1 or failures > 20:
            raise HTTPException(
                status_code=400,
                detail="High-risk review circuit breaker failures must be between 1 and 20",
            )
        settings.high_risk_review_circuit_breaker_failures = failures
        updates["HIGH_RISK_REVIEW_CIRCUIT_BREAKER_FAILURES"] = str(failures)

    if req.high_risk_review_circuit_breaker_cooldown_seconds is not None:
        cooldown = float(req.high_risk_review_circuit_breaker_cooldown_seconds)
        if cooldown < 5 or cooldown > 3600:
            raise HTTPException(
                status_code=400,
                detail="High-risk review circuit breaker cooldown must be between 5 and 3600 seconds",
            )
        settings.high_risk_review_circuit_breaker_cooldown_seconds = cooldown
        updates["HIGH_RISK_REVIEW_CIRCUIT_BREAKER_COOLDOWN_SECONDS"] = str(cooldown)

    if req.total_margin_limit_pct is not None:
        if req.total_margin_limit_pct < 0.10:
            raise HTTPException(status_code=400, detail="总保证金占用上限不能低于 10%")
        if req.total_margin_limit_pct > 1.0:
            raise HTTPException(status_code=400, detail="总保证金占用上限不能超过 100%")
        settings.max_total_margin_pct = float(req.total_margin_limit_pct)
        updates["MAX_TOTAL_MARGIN_PCT"] = str(settings.max_total_margin_pct)

    if updates:
        env_updates = strip_secret_env_updates(updates)
        if env_updates:
            settings.update_env_file(env_updates)

    total_margin_limit_pct = _effective_total_margin_limit_pct()
    return {
        "status": "ok",
        "message": "Settings updated.",
        "decision_interval": settings.decision_interval_seconds,
        "confidence_threshold": settings.confidence_threshold,
        "max_position_pct": settings.max_position_pct,
        "max_leverage": settings.max_leverage,
        "max_daily_loss_pct": settings.max_daily_loss_pct,
        "hard_stop_loss_pct": settings.hard_stop_loss_pct,
        "max_open_positions_per_model": settings.max_open_positions_per_model,
        "max_same_symbol_positions_per_side": settings.max_same_symbol_positions_per_side,
        "local_ai_tools_enabled": settings.local_ai_tools_enabled,
        "local_ai_tools_api_base": settings.local_ai_tools_api_base,
        "local_ai_tools_timeout_seconds": settings.local_ai_tools_timeout_seconds,
        "local_ai_tools_circuit_breaker_failures": settings.local_ai_tools_circuit_breaker_failures,
        "local_ai_tools_circuit_breaker_cooldown_seconds": settings.local_ai_tools_circuit_breaker_cooldown_seconds,
        "high_risk_review_enabled": settings.high_risk_review_enabled,
        "high_risk_review_api_base": settings.high_risk_review_api_base,
        "high_risk_review_api_key": mask_secret(settings.high_risk_review_api_key),
        "high_risk_review_has_api_key": bool(settings.high_risk_review_api_key),
        "high_risk_review_model": settings.high_risk_review_model,
        "high_risk_review_timeout_seconds": settings.high_risk_review_timeout_seconds,
        "high_risk_review_max_tokens": settings.high_risk_review_max_tokens,
        "high_risk_review_token_floor": HIGH_RISK_REVIEW_TOKEN_FLOOR,
        "high_risk_review_token_cap": HIGH_RISK_REVIEW_TOKEN_CAP,
        "high_risk_review_circuit_breaker_failures": settings.high_risk_review_circuit_breaker_failures,
        "high_risk_review_circuit_breaker_cooldown_seconds": settings.high_risk_review_circuit_breaker_cooldown_seconds,
        "total_margin_limit_pct": total_margin_limit_pct,
        "max_slippage_pct": settings.max_slippage_pct,
    }


@router.post("/settings/interval")
async def update_decision_interval(req: IntervalRequest):
    """Update the decision interval dynamically (no restart needed)."""
    if req.interval_seconds < 10:
        raise HTTPException(status_code=400, detail="Interval must be at least 10 seconds")
    if req.interval_seconds > 3600:
        raise HTTPException(status_code=400, detail="Interval must be at most 3600 seconds")

    settings.decision_interval_seconds = req.interval_seconds
    settings.update_env_file({"DECISION_INTERVAL_SECONDS": str(req.interval_seconds)})

    return {
        "status": "ok",
        "message": f"Decision interval updated to {req.interval_seconds}s",
        "decision_interval": req.interval_seconds,
    }
