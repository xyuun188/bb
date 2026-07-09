"""
Dashboard API endpoints — system status, market data, account balance.
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, overload

import structlog
from fastapi import APIRouter, Depends

from config.settings import (
    DECISION_MAKER_NAME,
    ENSEMBLE_TRADER_NAME,
    FIXED_AI_MODEL_SLOTS,
    settings,
)
from core.safe_output import safe_error_text
from core.symbols import (
    normalize_trading_symbol,
    okx_inst_id_from_symbol,
    symbol_from_okx_inst_id,
    symbol_query_variants,
)
from core.trading_mode import mode_manager
from data_feed.okx_rest_client import OKXRestClient
from data_feed.okx_ticker_volume import okx_swap_volume_fields
from executor.okx_executor import OKXExecutor
from services.account_accounting_service import (
    allocatable_balance_from_snapshot,
    balance_from_snapshot,
    tradeable_balance_from_snapshot,
)
from services.decision_reason_recovery import DecisionReasonRecoveryPolicy
from services.entry_signal_extraction import (
    enrich_signal_payload,
    first_tool_payload,
    unwrap_tool_payload,
)
from services.entry_signal_extraction import (
    expected_return_pct as signal_expected_return_pct,
)
from services.entry_signal_extraction import (
    payload_side as signal_payload_side,
)
from services.entry_signal_extraction import (
    signal_available as signal_payload_available,
)
from services.exchange_position_state import (
    exchange_position_display_valuation as _exchange_position_display_valuation,
)
from services.exchange_position_state import (
    exchange_snapshot_price as _exchange_snapshot_price,
)
from services.exchange_position_state import (
    exchange_snapshot_quantity as _exchange_snapshot_quantity,
)
from services.exchange_position_state import (
    exchange_snapshot_unrealized as _exchange_snapshot_unrealized,
)
from services.exchange_position_state import (
    parse_exchange_position_snapshot,
)
from services.execution_reason_localizer import localize_execution_reason
from services.manual_close_marker import MANUAL_CLOSE_LABEL, is_manual_close_order
from services.phase3_boundary import PHASE3_CLEAN_START_UTC, PHASE3_FIRST_CLEAN_DAY
from services.runtime_entry_filters import entry_filters_from_context
from services.server_monitor_status import get_server_monitor_status_async
from services.trading_params import DEFAULT_TRADING_PARAMS
from services.vector_memory import get_vector_memory_service
from web_dashboard.api.security import require_destructive_dashboard_confirmation
from web_dashboard.api.text_sanitize import sanitize_payload, sanitize_text

router = APIRouter()
logger = structlog.get_logger(__name__)
BEIJING_TZ = timezone(timedelta(hours=8))
OKX_AUTHORITATIVE_LEDGER_MODEL = "okx_authoritative_sync"
EXECUTION_LEDGER_MODEL_NAMES = (ENSEMBLE_TRADER_NAME, OKX_AUTHORITATIVE_LEDGER_MODEL)
LOCAL_ML_TRAINING_PARAMS = DEFAULT_TRADING_PARAMS.local_ml_training
STRATEGY_LEARNING_PARAMS = DEFAULT_TRADING_PARAMS.strategy_learning


# In-memory reference to the trading service (set by main loop)
_trading_service = None
_data_service = None
_competition_service = None
_local_ai_tools_status_client = None
_ml_signal_status_service = None
_EXCHANGE_MARK_CACHE_TTL_SECONDS = 15.0
_EXCHANGE_OPEN_SYMBOL_CACHE_TTL_SECONDS = 15.0
_PUBLIC_TICKER_CACHE_TTL_SECONDS = 10.0
_DASHBOARD_OKX_POSITION_READ_TIMEOUT_SECONDS = 3.0
_DASHBOARD_OKX_POSITION_INITIALIZE_TIMEOUT_SECONDS = 3.0
_DASHBOARD_OKX_BALANCE_READ_TIMEOUT_SECONDS = 5.0
_DASHBOARD_OKX_BALANCE_INITIALIZE_TIMEOUT_SECONDS = 5.0
_DASHBOARD_OKX_BALANCE_CACHE_TTL_SECONDS = 60.0
_DASHBOARD_OKX_BALANCE_STALE_CACHE_TTL_SECONDS = 300.0
_DASHBOARD_OKX_POSITION_STALE_CACHE_TTL_SECONDS = 180.0
_DASHBOARD_OKX_BALANCE_ERROR_CACHE_TTL_SECONDS = 30.0
_DASHBOARD_OKX_POSITION_ERROR_CACHE_TTL_SECONDS = 30.0
_DASHBOARD_HEAVY_CACHE_TTL_SECONDS = 60.0
_DASHBOARD_OKX_CONFIRMED_ORDER_STATUSES = {
    "okx_confirmed",
    "okx_only_backfilled",
    "okx_execution_result_confirmed",
}
_DASHBOARD_POSITION_HISTORY_ORDER_WINDOW_SECONDS = 10 * 60
_dashboard_okx_position_cache: dict[str, tuple[datetime, list[dict[str, Any]], Any | None]] = {}
_dashboard_okx_position_error_cache: dict[str, tuple[datetime, str, Any | None]] = {}
_dashboard_okx_position_locks: dict[str, asyncio.Lock] = {}
_dashboard_okx_position_refresh_tasks: dict[str, asyncio.Task[Any]] = {}
_exchange_mark_cache: dict[str, tuple[datetime, dict[tuple[str, str], dict[str, Any]]]] = {}
_exchange_open_symbol_cache: dict[str, tuple[datetime, set[str]]] = {}
_public_ticker_cache: dict[str, tuple[datetime, dict[str, dict]]] = {}
_dashboard_okx_balance_cache: dict[str, tuple[datetime, dict[str, Any]]] = {}
_dashboard_okx_balance_error_cache: dict[str, tuple[datetime, dict[str, Any], Any | None]] = {}
_dashboard_okx_balance_locks: dict[str, asyncio.Lock] = {}
_dashboard_okx_balance_refresh_tasks: dict[str, asyncio.Task[Any]] = {}
_dashboard_heavy_cache: dict[tuple[Any, ...], tuple[datetime, Any]] = {}
_dashboard_heavy_cache_locks: dict[tuple[Any, ...], asyncio.Lock] = {}
_DECISION_REASON_RECOVERY = DecisionReasonRecoveryPolicy()


def _dashboard_heavy_cache_get(
    key: tuple[Any, ...], ttl_seconds: float = _DASHBOARD_HEAVY_CACHE_TTL_SECONDS
) -> Any | None:
    cached = _dashboard_heavy_cache.get(key)
    if cached is None:
        return None
    cached_at, payload = cached
    if (datetime.now(UTC) - cached_at).total_seconds() > ttl_seconds:
        _dashboard_heavy_cache.pop(key, None)
        return None
    return copy.deepcopy(payload)


def _dashboard_heavy_cache_set(key: tuple[Any, ...], payload: Any) -> Any:
    _dashboard_heavy_cache[key] = (datetime.now(UTC), copy.deepcopy(payload))
    return payload


async def _dashboard_heavy_cached(
    key: tuple[Any, ...],
    builder: Callable[[], Awaitable[Any]],
    ttl_seconds: float = _DASHBOARD_HEAVY_CACHE_TTL_SECONDS,
) -> Any:
    cached = _dashboard_heavy_cache_get(key, ttl_seconds)
    if cached is not None:
        return cached
    lock = _dashboard_heavy_cache_locks.setdefault(key, asyncio.Lock())
    async with lock:
        cached = _dashboard_heavy_cache_get(key, ttl_seconds)
        if cached is not None:
            return cached
        payload = await builder()
        return _dashboard_heavy_cache_set(key, payload)


def _clear_dashboard_heavy_cache(*names: str) -> None:
    wanted = {name for name in names if name}
    if not wanted:
        _dashboard_heavy_cache.clear()
        return
    for key in list(_dashboard_heavy_cache):
        if key and key[0] in wanted:
            _dashboard_heavy_cache.pop(key, None)


def _log_dashboard_fallback(event: str, exc: Exception, **fields: Any) -> None:
    """Log a recoverable dashboard fallback without breaking the endpoint."""
    logger.debug(event, error=safe_error_text(exc), **fields)


def _dashboard_okx_account_label(mode: str) -> str:
    return "OKX 实盘账户" if mode == "live" else "OKX 模拟盘账户"


def _dashboard_okx_error_text(exc: Exception, *, resource: str) -> str:
    error = safe_error_text(exc)
    lower = error.lower()
    if isinstance(exc, TimeoutError) or error in ("TimeoutError", "") or "timed out" in lower:
        return f"OKX {resource}响应超时，已优先返回缓存数据"
    if "authorization" in lower or "api key" in lower or "signature" in lower:
        return f"OKX {resource}鉴权失败：{error}"
    return f"OKX {resource}读取失败：{error}"


def _dashboard_okx_balance_refreshing_snapshot(selected_mode: str) -> dict[str, Any]:
    """Return a non-error snapshot while OKX balance warms in the background."""

    selected_mode = "live" if selected_mode == "live" else "paper"
    return {
        "balance_source": _dashboard_okx_account_label(selected_mode),
        "source": "background_refresh",
        "refresh_in_progress": True,
        "balance_status": "refresh_pending",
    }


def _consume_dashboard_refresh_task(
    tasks: dict[str, asyncio.Task[Any]],
    selected_mode: str,
    *,
    label: str,
) -> Callable[[asyncio.Task[Any]], None]:
    def _done(task: asyncio.Task[Any]) -> None:
        if tasks.get(selected_mode) is task:
            tasks.pop(selected_mode, None)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception as exc:
            _log_dashboard_fallback(
                f"dashboard {label} background refresh failed",
                exc,
                mode=selected_mode,
            )

    return _done


async def _refresh_dashboard_okx_position_cache(selected_mode: str) -> None:
    selected_mode = "live" if selected_mode == "live" else "paper"
    executor = _dashboard_okx_executor_for_mode(selected_mode)
    executor_identity = executor
    positions = await _fetch_dashboard_okx_positions_uncached(selected_mode, executor=executor)
    normalized_positions = [dict(position) for position in positions or []]
    _dashboard_okx_position_cache[selected_mode] = (
        datetime.now(UTC),
        copy.deepcopy(normalized_positions),
        executor_identity,
    )
    _dashboard_okx_position_error_cache.pop(selected_mode, None)


def _start_dashboard_okx_position_refresh(selected_mode: str) -> None:
    task = _dashboard_okx_position_refresh_tasks.get(selected_mode)
    if task is not None and not task.done():
        return
    task = asyncio.create_task(_refresh_dashboard_okx_position_cache(selected_mode))
    _dashboard_okx_position_refresh_tasks[selected_mode] = task
    task.add_done_callback(
        _consume_dashboard_refresh_task(
            _dashboard_okx_position_refresh_tasks,
            selected_mode,
            label="okx position",
        )
    )


async def _fetch_dashboard_okx_balance_uncached(selected_mode: str) -> dict[str, Any] | None:
    selected_mode = "live" if selected_mode == "live" else "paper"
    fallback_executor = _make_lightweight_okx_executor(OKXExecutor, selected_mode)
    try:
        await asyncio.wait_for(
            fallback_executor.initialize(),
            timeout=_DASHBOARD_OKX_BALANCE_INITIALIZE_TIMEOUT_SECONDS,
        )
        snapshot = await asyncio.wait_for(
            fallback_executor.get_balance_snapshot("USDT"),
            timeout=_DASHBOARD_OKX_BALANCE_READ_TIMEOUT_SECONDS,
        )
        if not snapshot:
            raise RuntimeError("empty OKX balance snapshot")
        if snapshot.get("error"):
            raise RuntimeError(safe_error_text(snapshot.get("error")))
        return dict(snapshot)
    finally:
        try:
            await fallback_executor.shutdown()
        except Exception as exc:
            _log_dashboard_fallback(
                "dashboard summary okx fallback shutdown failed",
                exc,
                mode=selected_mode,
            )


async def _fetch_dashboard_okx_balance_uncached_with_total_budget(
    selected_mode: str,
) -> dict[str, Any] | None:
    selected_mode = "live" if selected_mode == "live" else "paper"
    try:
        return await asyncio.wait_for(
            _fetch_dashboard_okx_balance_uncached(selected_mode),
            timeout=(
                _DASHBOARD_OKX_BALANCE_INITIALIZE_TIMEOUT_SECONDS
                + _DASHBOARD_OKX_BALANCE_READ_TIMEOUT_SECONDS
                + 1.0
            ),
        )
    except TimeoutError as exc:
        raise TimeoutError("OKX balance background refresh exceeded total budget") from exc


async def _refresh_dashboard_okx_balance_cache(selected_mode: str) -> None:
    selected_mode = "live" if selected_mode == "live" else "paper"
    executor_identity = _dashboard_okx_executor_for_mode(selected_mode)
    try:
        snapshot = await _fetch_dashboard_okx_balance_uncached_with_total_budget(selected_mode)
        if snapshot:
            _dashboard_okx_balance_cache[selected_mode] = (
                datetime.now(UTC),
                copy.deepcopy(snapshot),
            )
            _dashboard_okx_balance_error_cache.pop(selected_mode, None)
    except Exception as exc:
        error = _dashboard_okx_error_text(exc, resource="余额")
        _dashboard_okx_balance_error_cache[selected_mode] = (
            datetime.now(UTC),
            {
                "error": error,
                "balance_error": error,
                "balance_source": _dashboard_okx_account_label(selected_mode),
                "source": "background_refresh",
                "error_cached": True,
            },
            executor_identity,
        )
        raise


def _start_dashboard_okx_balance_refresh(selected_mode: str) -> None:
    task = _dashboard_okx_balance_refresh_tasks.get(selected_mode)
    if task is not None and not task.done():
        return
    task = asyncio.create_task(_refresh_dashboard_okx_balance_cache(selected_mode))
    _dashboard_okx_balance_refresh_tasks[selected_mode] = task
    task.add_done_callback(
        _consume_dashboard_refresh_task(
            _dashboard_okx_balance_refresh_tasks,
            selected_mode,
            label="okx balance",
        )
    )


def _dashboard_okx_position_error_state(mode: str | None = None) -> dict[str, Any] | None:
    selected_mode = "live" if (mode or mode_manager.mode.value) == "live" else "paper"
    cached_error = _dashboard_okx_position_error_cache.get(selected_mode)
    if not cached_error:
        return None
    cached_at, cached_text, cached_executor_identity = cached_error
    age_seconds = (datetime.now(UTC) - cached_at).total_seconds()
    if age_seconds > _DASHBOARD_OKX_POSITION_ERROR_CACHE_TTL_SECONDS:
        return None
    executor = _dashboard_okx_executor_for_mode(selected_mode)
    executor_identity = executor
    if cached_executor_identity is not executor_identity:
        return None
    return {
        "mode": selected_mode,
        "error": cached_text,
        "age_seconds": round(age_seconds, 3),
    }


def _dashboard_okx_positions_temporarily_unavailable(mode: str | None = None) -> bool:
    return _dashboard_okx_position_error_state(mode) is not None


def _as_utc_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            value = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


@overload
def _safe_float(value: Any, default: None) -> float | None: ...


@overload
def _safe_float(value: Any, default: float = 0.0) -> float: ...


def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_float_value(value: Any, default: float = 0.0) -> float:
    result = _safe_float(value, default)
    return default if result is None else result


def _weighted_average(items: list[tuple[float, float]]) -> float:
    total_weight = sum(max(float(weight or 0.0), 0.0) for weight, _value in items)
    if total_weight <= 0:
        return 0.0
    return (
        sum(max(float(weight or 0.0), 0.0) * float(value or 0.0) for weight, value in items)
        / total_weight
    )


def _dashboard_position_key(symbol: Any, side: Any) -> tuple[str, str]:
    return (_normalize_dashboard_symbol(str(symbol or "")), str(side or "").lower())


def _exchange_snapshot_margin(snapshot: dict[str, Any]) -> float:
    return _safe_float(snapshot.get("margin_used"), 0.0) or 0.0


def _exchange_position_totals(
    exchange_mark_map: dict[tuple[str, str], dict[str, Any]],
    fallback_margin_by_key: dict[tuple[str, str], float] | None = None,
) -> dict[str, float | int]:
    unrealized = 0.0
    used_margin = 0.0
    fallback_margin_by_key = fallback_margin_by_key or {}
    for key, snapshot in exchange_mark_map.items():
        side = key[1]
        unrealized += _exchange_snapshot_unrealized(snapshot, side)
        margin = _exchange_snapshot_margin(snapshot)
        used_margin += margin if margin > 0 else fallback_margin_by_key.get(key, 0.0)
    return {
        "open_count": len(exchange_mark_map),
        "unrealized_pnl": round(unrealized, 8),
        "used_margin": round(used_margin, 8),
    }


def _local_position_margin(row: Any) -> float:
    quantity = _safe_float(getattr(row, "quantity", None), 0.0) or 0.0
    entry_price = _safe_float(getattr(row, "entry_price", None), 0.0) or 0.0
    leverage = max(_safe_float(getattr(row, "leverage", None), 1.0) or 1.0, 1.0)
    return (quantity * entry_price) / leverage


def _local_group_notional(quantity: Any, entry_price: Any) -> float:
    return (_safe_float(quantity, 0.0) or 0.0) * (_safe_float(entry_price, 0.0) or 0.0)


def _group_open_dashboard_positions(
    positions: list[dict[str, Any]],
    exchange_mark_map: dict[tuple[str, str], dict[str, Any]],
    *,
    mode: str | None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    order: list[tuple[str, str, str, str]] = []

    for item in positions:
        key_symbol, key_side = _dashboard_position_key(item.get("symbol"), item.get("side"))
        key = (
            str(item.get("model_name") or ""),
            str(item.get("mode") or mode or ""),
            key_symbol,
            key_side,
        )
        local_qty = _safe_float(item.get("local_quantity", item.get("quantity")), 0.0) or 0.0
        local_entry = (
            _safe_float(
                item.get("local_entry_price", item.get("entry_price")),
                0.0,
            )
            or 0.0
        )
        local_unrealized = (
            _safe_float(
                item.get("local_unrealized_pnl", item.get("unrealized_pnl")),
                0.0,
            )
            or 0.0
        )

        if key not in grouped:
            group = dict(item)
            group["symbol"] = key_symbol or item.get("symbol")
            group["side"] = key_side or item.get("side")
            group["position_ids"] = [item.get("id")] if item.get("id") is not None else []
            group["split_count"] = 1
            group["local_quantity"] = local_qty
            group["local_entry_price"] = local_entry
            group["local_unrealized_pnl"] = local_unrealized
            group["_local_notional"] = _local_group_notional(local_qty, local_entry)
            grouped[key] = group
            order.append(key)
            continue

        group = grouped[key]
        if item.get("id") is not None:
            group.setdefault("position_ids", []).append(item.get("id"))
        group["split_count"] = int(group.get("split_count") or 1) + 1
        group["local_quantity"] = (_safe_float(group.get("local_quantity"), 0.0) or 0.0) + local_qty
        group["local_unrealized_pnl"] = (
            _safe_float(group.get("local_unrealized_pnl"), 0.0) or 0.0
        ) + local_unrealized
        group["_local_notional"] = (
            _safe_float(group.get("_local_notional"), 0.0) or 0.0
        ) + _local_group_notional(local_qty, local_entry)
        if group["local_quantity"]:
            group["local_entry_price"] = group["_local_notional"] / group["local_quantity"]
        group["leverage"] = max(
            _safe_float(group.get("leverage"), 1.0) or 1.0,
            _safe_float(item.get("leverage"), 1.0) or 1.0,
        )

    for (symbol, side), snapshot in exchange_mark_map.items():
        if any(key[2] == symbol and key[3] == side for key in grouped):
            continue
        key = (ENSEMBLE_TRADER_NAME, "live" if mode == "live" else "paper", symbol, side)
        grouped[key] = {
            "id": None,
            "model_name": ENSEMBLE_TRADER_NAME,
            "mode": key[1],
            "symbol": symbol,
            "side": side,
            "quantity": _exchange_snapshot_quantity(snapshot),
            "entry_price": _safe_float(snapshot.get("entry_price"), 0.0) or 0.0,
            "current_price": _exchange_snapshot_price(snapshot),
            "change_24h": 0.0,
            "unrealized_pnl": _exchange_snapshot_unrealized(snapshot, side),
            "pnl_source": "okx_position",
            "local_quantity": 0.0,
            "local_entry_price": 0.0,
            "local_unrealized_pnl": 0.0,
            "realized_pnl": 0.0,
            "leverage": 1.0,
            "stop_loss": None,
            "take_profit": None,
            "is_open": True,
            "db_is_open": False,
            "exchange_synced": True,
            "close_status": "open",
            "close_status_label": "持有中",
            "close_status_source": "exchange",
            "position_status": "持有中",
            "opened_at": None,
            "closed_at": None,
            "position_ids": [],
            "split_count": 0,
        }
        order.append(key)

    result: list[dict[str, Any]] = []
    for key in order:
        group = grouped[key]
        symbol = key[2]
        side = key[3]
        local_qty = _safe_float(group.get("local_quantity"), 0.0) or 0.0
        local_entry = _safe_float(group.get("local_entry_price"), 0.0) or 0.0
        snapshot = exchange_mark_map.get((symbol, side))
        if snapshot:
            exchange_qty = _exchange_snapshot_quantity(snapshot)
            exchange_entry = _safe_float(snapshot.get("entry_price"), 0.0) or 0.0
            exchange_mark = _exchange_snapshot_price(snapshot)
            exchange_upl = _exchange_snapshot_unrealized(snapshot, side)
            group["quantity"] = exchange_qty if exchange_qty > 0 else local_qty
            group["entry_price"] = exchange_entry if exchange_entry > 0 else local_entry
            if exchange_mark > 0:
                group["current_price"] = exchange_mark
            group["unrealized_pnl"] = exchange_upl
            group["pnl_source"] = "okx_position"
            group["exchange_quantity"] = exchange_qty
            group["exchange_entry_price"] = exchange_entry
            group["exchange_mark_price"] = exchange_mark
            group["exchange_unrealized_pnl"] = exchange_upl
            group["exchange_margin_used"] = _exchange_snapshot_margin(snapshot)
        else:
            group["quantity"] = local_qty
            group["entry_price"] = local_entry
            group["unrealized_pnl"] = _safe_float(group.get("local_unrealized_pnl"), 0.0) or 0.0
            group["pnl_source"] = "local_group"
        position_ids = [pid for pid in group.get("position_ids", []) if pid is not None]
        group["position_ids"] = position_ids
        group["id"] = min(position_ids) if position_ids else None
        group["can_manual_close"] = (
            len(position_ids) == 1 and int(group.get("split_count") or 0) <= 1
        )
        group.pop("_local_notional", None)
        result.append(group)
    return result


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _analysis_pre_expert_skip(raw: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized pre-expert skip state for analysis-flow rendering."""
    fast_prefilter = _safe_dict(raw.get("fast_prefilter"))
    if fast_prefilter.get("skipped_llm"):
        return {
            "skipped": True,
            "kind": "market_prefilter",
            "label": "行情预检未进入专家",
            "reason": sanitize_text(
                fast_prefilter.get("reason")
                or "本轮行情数据或机会质量未通过预检，因此没有把该交易对送入大模型专家。"
            ),
        }
    position_fast_scan = _safe_dict(raw.get("position_fast_scan"))
    if position_fast_scan.get("skipped_llm"):
        return {
            "skipped": True,
            "kind": "position_fast_scan",
            "label": "持仓快速扫描未进入专家",
            "reason": sanitize_text(
                position_fast_scan.get("reason")
                or "本轮是持仓快速扫描；只有出现强平仓、强加仓或高风险信号时才进入专家深度复盘。"
            ),
        }
    return {"skipped": False, "kind": "", "label": "", "reason": ""}


def _execution_reason_is_unusable(reason: str | None) -> bool:
    text = str(reason or "").strip()
    if not text:
        return False
    if re.fullmatch(r"\d{12,}", text):
        return True
    return any(
        marker in text
        for marker in (
            "原始说明已损坏",
            "无法准确还原",
            "鍘嗗彶璁板綍",
            "鎹熷潖",
            "这条平仓决策没有找到对应的本地平仓委托记录",
        )
    )


def _recover_execution_reason_from_raw_decision(decision) -> str | None:
    return _DECISION_REASON_RECOVERY.recover(decision)


def _display_execution_reason(decision, order=None) -> str | None:
    reason = getattr(decision, "execution_reason", None) or _fallback_execution_reason(
        decision, order
    )
    sanitized = sanitize_text(localize_execution_reason(reason))
    if _execution_reason_is_unusable(str(sanitized or "")):
        recovered = _recover_execution_reason_from_raw_decision(decision)
        if recovered:
            return localize_execution_reason(recovered)
    return sanitized


def _display_opportunity_score(
    decision,
    raw: dict[str, Any],
    execution_reason: str | None = None,
) -> dict[str, Any] | None:
    opportunity = raw.get("opportunity_score") if isinstance(raw, dict) else None
    if not isinstance(opportunity, dict):
        return None
    payload = dict(opportunity)
    if bool(getattr(decision, "was_executed", False)):
        return payload
    reason_text = sanitize_text(
        localize_execution_reason(execution_reason or getattr(decision, "execution_reason", None))
    )
    final_state = str(
        payload.get("execution_final_state")
        or raw.get("stage_status")
        or raw.get("execution_status")
        or ""
    ).lower()
    if reason_text or final_state in {"skipped", "blocked", "rejected"}:
        payload["selected_for_execution"] = False
        if reason_text:
            payload["selection_reason"] = reason_text
        payload["execution_final_state"] = final_state or "skipped"
        payload.setdefault(
            "execution_final_blocker",
            raw.get("policy_blocker") or raw.get("skip_kind") or "not_executed",
        )
    return payload


def _side_from_action(action: str | None) -> str:
    value = str(action or "").lower()
    if "short" in value:
        return "short"
    if "long" in value:
        return "long"
    return "hold"


def _side_label(side: str | None) -> str:
    value = str(side or "").lower()
    if value == "long":
        return "做多"
    if value == "short":
        return "做空"
    return "观望"


def _model_side_from_payload(payload: dict | None, side_key: str = "best_side") -> str:
    if not isinstance(payload, dict):
        return ""
    value = str(payload.get(side_key) or payload.get("side") or "").lower()
    if not value:
        direction = str(
            payload.get("direction")
            or payload.get("forecast_direction")
            or payload.get("trend")
            or ""
        ).lower()
        if direction == "up":
            value = "long"
        elif direction == "down":
            value = "short"
    if not value:
        label = str(payload.get("label") or payload.get("sentiment") or "").lower()
        score = _safe_float(payload.get("score"), payload.get("sentiment_score", 0.0)) or 0.0
        if label in {"positive", "bullish"} or score > 0:
            value = "long"
        elif label in {"negative", "bearish"} or score < 0:
            value = "short"
    if value in {"long", "short"}:
        return value
    return ""


def _normalized_tool_payload(
    raw: dict[str, Any],
    canonical_name: str,
    *aliases: str,
) -> dict[str, Any]:
    payload = first_tool_payload(raw, canonical_name, *aliases)
    if payload:
        return payload
    source = _safe_dict(raw.get("local_ai_tools"))
    for key in (canonical_name, *aliases):
        value = source.get(key)
        if isinstance(value, dict):
            normalized = enrich_signal_payload(canonical_name, unwrap_tool_payload(value))
            return normalized or dict(value)
    return {}


def _normalized_local_ai_tools_payload(raw: dict[str, Any]) -> dict[str, Any] | None:
    source = _safe_dict(raw.get("local_ai_tools"))
    if not source:
        return None
    normalized = dict(source)
    normalized["profit_prediction"] = _normalized_tool_payload(
        raw,
        "profit_prediction",
        "profit_model",
        "server_profit",
        "server_profit_model",
        "profit",
    )
    normalized["time_series_prediction"] = _normalized_tool_payload(
        raw,
        "time_series_prediction",
        "timeseries_prediction",
        "sequence_prediction",
        "timeseries",
        "time_series",
    )
    normalized["sentiment_analysis"] = _normalized_tool_payload(
        raw,
        "sentiment_analysis",
        "sentiment_prediction",
        "sentiment_model",
        "sentiment",
    )
    normalized["exit_advice"] = _normalized_tool_payload(
        raw,
        "exit_advice",
        "exit_model",
        "position_exit",
        "exit",
    )
    return normalized


def _extract_primary_ml(raw: dict[str, Any]) -> dict[str, Any]:
    ml = _safe_dict(raw.get("ml_signal"))
    predictions = _safe_list(ml.get("predictions"))
    primary = _safe_dict(predictions[0] if predictions else {})
    best_side = _model_side_from_payload(primary)
    return {
        "available": bool(ml.get("available")) if ml else False,
        "side": best_side,
        "side_label": _side_label(best_side),
        "expected_return_pct": _safe_float(
            primary.get("best_expected_return_pct"), ml.get("expected_return_pct", 0.0)
        )
        or 0.0,
        "profit_edge_pct": _safe_float(
            primary.get("profit_edge_pct"), ml.get("profit_edge_pct", 0.0)
        )
        or 0.0,
        "win_rate": _safe_float(primary.get("best_win_rate"), 0.0) or 0.0,
        "influence_enabled": bool(ml.get("influence_enabled", True)),
        "summary": ml.get("suggestion") or ml.get("note") or "",
    }


def _extract_local_tools(raw: dict[str, Any]) -> dict[str, Any]:
    tools = _normalized_local_ai_tools_payload(raw) or {}
    profit = _safe_dict(tools.get("profit_prediction"))
    ts = _safe_dict(tools.get("time_series_prediction"))
    sentiment = _safe_dict(tools.get("sentiment_analysis"))
    profit_side = signal_payload_side(profit)
    ts_side = signal_payload_side(ts)
    sentiment_side = signal_payload_side(sentiment)
    return {
        "profit": {
            "available": signal_payload_available(profit),
            "side": profit_side,
            "side_label": _side_label(profit_side),
            "expected_return_pct": signal_expected_return_pct(profit, profit_side),
            "profit_quality_score": _safe_float(profit.get("profit_quality_score"), 0.0) or 0.0,
            "loss_probability": _safe_float(profit.get(f"{profit_side}_loss_probability"), 0.0)
            or 0.0,
            "model": profit.get("model") or "",
        },
        "timeseries": {
            "available": signal_payload_available(ts),
            "side": ts_side,
            "side_label": _side_label(ts_side),
            "expected_return_pct": signal_expected_return_pct(ts, ts_side),
            "horizon_minutes": ts.get("horizon_minutes") or ts.get("primary_horizon_minutes"),
            "model": ts.get("model") or "",
        },
        "sentiment": {
            "available": signal_payload_available(sentiment),
            "side": sentiment_side,
            "side_label": _side_label(sentiment_side),
            "score": _safe_float(sentiment.get("score"), sentiment.get("sentiment_score", 0.0))
            or 0.0,
            "expected_return_pct": signal_expected_return_pct(sentiment, sentiment_side),
            "summary": sentiment.get("summary") or sentiment.get("reason") or "",
            "model": sentiment.get("model") or "",
        },
    }


def _execution_tradeable_balance(
    *,
    okx_available: float | None,
    okx_allocatable: float | None,
    okx_equity: float | None,
    okx_total: float | None,
    fallback_available: float | None,
) -> float | None:
    snapshot = {
        "free": okx_available,
        "allocatable": okx_allocatable,
        "equity": okx_equity,
        "total": okx_total,
    }
    tradeable = tradeable_balance_from_snapshot(snapshot)
    if tradeable > 0:
        return tradeable
    return _safe_float(fallback_available, None)


def _build_decision_attribution(
    decision: Any, raw: dict[str, Any], experts: list[dict[str, Any]]
) -> dict[str, Any]:
    raw = _safe_dict(raw)
    action = str(getattr(decision, "action", "") or "").lower()
    side = _side_from_action(action)
    executed = bool(getattr(decision, "was_executed", False))
    opportunity = _safe_dict(raw.get("opportunity_score"))
    decision_maker = _safe_dict(raw.get("decision_maker"))
    close_evidence = _safe_dict(raw.get("close_evidence"))
    high_risk_review = _safe_dict(raw.get("high_risk_review"))
    ml = _extract_primary_ml(raw)
    local = _extract_local_tools(raw)

    support = 0
    oppose = 0
    hold = 0
    for item in experts or []:
        expert_side = _side_from_action(item.get("action"))
        if side in {"long", "short"} and expert_side == side:
            support += 1
        elif expert_side in {"long", "short"} and expert_side != side:
            oppose += 1
        else:
            hold += 1

    if action == "hold":
        final_reason = (
            getattr(decision, "execution_reason", None)
            or getattr(decision, "reasoning", None)
            or "最终选择观望。"
        )
    elif action in {"close_long", "close_short"}:
        final_reason = (
            close_evidence.get("reason")
            or close_evidence.get("block_reason")
            or getattr(decision, "execution_reason", None)
            or getattr(decision, "reasoning", None)
            or "持仓复盘给出平仓/减仓信号。"
        )
    elif executed:
        final_reason = (
            getattr(decision, "execution_reason", None)
            or "通过机会评分、风控检查和执行前检查，已提交订单。"
        )
    else:
        final_reason = (
            getattr(decision, "execution_reason", None)
            or opportunity.get("selection_reason")
            or decision_maker.get("reasoning")
            or getattr(decision, "reasoning", None)
            or "未达到执行条件。"
        )

    final_reason = localize_execution_reason(final_reason) or final_reason

    return {
        "action": action,
        "side": side,
        "side_label": _side_label(side),
        "executed": executed,
        "ai_experts": {
            "support_count": support,
            "oppose_count": oppose,
            "hold_count": hold,
            "summary": f"{support} 个专家同向，{oppose} 个反向，{hold} 个观望/非方向。",
        },
        "local_ml": ml,
        "server_profit": local.get("profit", {}),
        "timeseries": local.get("timeseries", {}),
        "sentiment": local.get("sentiment", {}),
        "opportunity_score": opportunity,
        "high_risk_review": high_risk_review,
        "decision_maker": {
            "status": decision_maker.get("status"),
            "action": decision_maker.get("action"),
            "confidence": decision_maker.get("confidence"),
            "reasoning": decision_maker.get("reasoning")
            or decision_maker.get("reason")
            or decision_maker.get("guard_reason"),
        },
        "close_evidence": close_evidence,
        "final_reason": final_reason,
    }


async def _get_today_ai_decision_count(mode: str) -> int:
    """Count AI decisions for the current Beijing calendar day."""
    from sqlalchemy import func, select

    from db.session import get_session_ctx
    from models.decision import AIDecision

    selected_mode = "live" if mode == "live" else "paper"
    now_local = datetime.now(BEIJING_TZ)
    start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    end_local = start_local + timedelta(days=1)
    start_utc = start_local.astimezone(UTC).replace(tzinfo=None)
    end_utc = end_local.astimezone(UTC).replace(tzinfo=None)

    async with get_session_ctx() as session:
        result = await session.execute(
            select(func.count(AIDecision.id)).where(
                AIDecision.is_paper.is_(selected_mode != "live"),
                AIDecision.created_at >= start_utc,
                AIDecision.created_at < end_utc,
            )
        )
        return int(result.scalar_one() or 0)


async def _recent_trading_activity_stats(hours: int = 6) -> dict[str, Any]:
    """Return recent DB activity as a heartbeat for split dashboard/trader deployments."""
    from sqlalchemy import func, select

    from db.session import get_session_ctx
    from models.decision import AIDecision
    from models.trade import Order

    since = datetime.now(UTC) - timedelta(hours=hours)
    async with get_session_ctx() as session:
        decision_row = (
            await session.execute(
                select(func.count(AIDecision.id), func.max(AIDecision.created_at)).where(
                    AIDecision.created_at >= since
                )
            )
        ).one()
        order_row = (
            await session.execute(
                select(func.count(Order.id), func.max(Order.created_at)).where(
                    Order.created_at >= since
                )
            )
        ).one()

    latest_values = [value for value in (decision_row[1], order_row[1]) if value is not None]
    latest_at = max(latest_values) if latest_values else None
    if latest_at and latest_at.tzinfo is None:
        latest_at = latest_at.replace(tzinfo=UTC)
    heartbeat_age_seconds = (
        max((datetime.now(UTC) - latest_at).total_seconds(), 0.0) if latest_at else None
    )
    return {
        "decision_count": int(decision_row[0] or 0),
        "order_count": int(order_row[0] or 0),
        "latest_decision_at": decision_row[1].isoformat() if decision_row[1] else None,
        "latest_order_at": order_row[1].isoformat() if order_row[1] else None,
        "latest_activity_at": latest_at.isoformat() if latest_at else None,
        "heartbeat_age_seconds": heartbeat_age_seconds,
        "window_hours": hours,
    }


def _load_trading_runtime_status() -> dict[str, Any]:
    """Load latest split-process trading runtime heartbeat from disk."""
    path = settings.data_dir / "trading_runtime_status.json"
    try:
        if not path.exists():
            return {}
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {}
        heartbeat_at = _as_utc_datetime(data.get("heartbeat_at"))
        if heartbeat_at is not None:
            data["heartbeat_age_seconds"] = max(
                (datetime.now(UTC) - heartbeat_at).total_seconds(),
                0.0,
            )
        elif path.exists():
            data["heartbeat_age_seconds"] = max(
                datetime.now(UTC).timestamp() - path.stat().st_mtime,
                0.0,
            )
        return data
    except Exception as exc:
        logger.debug(
            "failed to load trading runtime heartbeat",
            error=safe_error_text(exc),
        )
        return {}


async def _split_process_trading_stats(mode: str | None = None) -> dict[str, Any]:
    """Fallback stats when dashboard and trading engine run as separate services."""
    settings.refresh_runtime_env(force=True)
    runtime_status = _load_trading_runtime_status()
    try:
        activity = await _recent_trading_activity_stats()
    except Exception as exc:
        logger.warning(
            "failed to load split-process trading stats",
            error=safe_error_text(exc),
        )
        activity = {}
    runtime_age = runtime_status.get("heartbeat_age_seconds")
    heartbeat_age = (
        runtime_age if runtime_age is not None else activity.get("heartbeat_age_seconds")
    )
    running = heartbeat_age is not None and float(heartbeat_age) <= max(
        float(settings.decision_interval_seconds) * 4,
        180.0,
    )
    decision_interval = int(
        runtime_status.get("decision_interval") or settings.decision_interval_seconds
    )
    started_at = _as_utc_datetime(runtime_status.get("started_at"))
    uptime_seconds = int(runtime_status.get("uptime_seconds") or 0)
    if uptime_seconds <= 0 and started_at is not None:
        uptime_seconds = int(max((datetime.now(UTC) - started_at).total_seconds(), 0.0))
    last_round_started = _as_utc_datetime(runtime_status.get("last_round_started_at"))
    last_round_finished = _as_utc_datetime(runtime_status.get("last_round_finished_at"))
    round_active = bool(runtime_status.get("round_active", False))
    if last_round_started is not None and (
        last_round_finished is None or last_round_finished < last_round_started
    ):
        round_active = True
    round_running_seconds = (
        int(max((datetime.now(UTC) - last_round_started).total_seconds(), 0.0))
        if round_active and last_round_started is not None
        else 0
    )
    return {
        "running": running,
        "mode": runtime_status.get("mode")
        or ("live" if mode == "live" else mode_manager.mode.value),
        "paused": mode_manager.is_paused,
        "uptime_source": "split_process_heartbeat",
        "uptime_seconds": uptime_seconds,
        "started_at": runtime_status.get("started_at"),
        "heartbeat_at": runtime_status.get("heartbeat_at")
        or runtime_status.get("last_heartbeat_at"),
        "last_heartbeat_at": runtime_status.get("last_heartbeat_at")
        or runtime_status.get("heartbeat_at"),
        "decisions_total": int(activity.get("decision_count") or 0),
        "trades_total": int(activity.get("order_count") or 0),
        "recent_decisions": [],
        "recent_executions": [],
        "current_stage": runtime_status.get("current_stage")
        or ("split_process" if running else "unknown"),
        "current_stage_label": "独立交易进程运行中" if running else "等待交易心跳",
        "round_active": round_active,
        "round_running_seconds": round_running_seconds,
        "market_analysis_watchdog_seconds": int(
            runtime_status.get("market_analysis_watchdog_seconds")
            or settings.market_analysis_watchdog_seconds
        ),
        "position_analysis_watchdog_seconds": int(
            runtime_status.get("position_analysis_watchdog_seconds")
            or settings.position_analysis_watchdog_seconds
            or settings.market_analysis_watchdog_seconds
        ),
        "market_current_stage": runtime_status.get("market_current_stage"),
        "market_round_active": runtime_status.get("market_round_active"),
        "market_last_error": runtime_status.get("market_last_error"),
        "position_current_stage": runtime_status.get("position_current_stage"),
        "position_round_active": runtime_status.get("position_round_active"),
        "position_last_error": runtime_status.get("position_last_error"),
        "last_round_started_at": runtime_status.get("last_round_started_at")
        or activity.get("latest_activity_at"),
        "last_round_finished_at": runtime_status.get("last_round_finished_at")
        or activity.get("latest_activity_at"),
        "last_market_round_started_at": runtime_status.get("last_market_round_started_at"),
        "last_market_round_finished_at": runtime_status.get("last_market_round_finished_at"),
        "last_position_round_started_at": runtime_status.get("last_position_round_started_at"),
        "last_position_round_finished_at": runtime_status.get("last_position_round_finished_at"),
        "last_round_error": runtime_status.get("last_round_error"),
        "okx_authoritative_sync": runtime_status.get("okx_authoritative_sync"),
        "live_model": ENSEMBLE_TRADER_NAME,
        "models": [ENSEMBLE_TRADER_NAME],
        "risk": {},
        "decision_interval": decision_interval,
        "market_loop_interval_seconds": runtime_status.get("market_loop_interval_seconds"),
        "position_loop_interval_seconds": runtime_status.get("position_loop_interval_seconds"),
        "market_round_time_budget_seconds": runtime_status.get("market_round_time_budget_seconds"),
        "split_process_activity": activity,
        "runtime_status": runtime_status,
    }


async def _trading_stats_with_runtime_heartbeat(mode: str | None = None) -> dict[str, Any]:
    """Return trading stats and backfill split-process heartbeat fields when needed."""

    if _trading_service:
        stats = dict(_trading_service.get_stats(mode_filter=mode))
        if stats.get("market_loop_interval_seconds") is not None:
            return stats
        runtime_stats = await _split_process_trading_stats(mode)
        for key in (
            "decision_interval",
            "market_loop_interval_seconds",
            "position_loop_interval_seconds",
            "market_round_time_budget_seconds",
            "market_current_stage",
            "market_round_active",
            "position_current_stage",
            "position_round_active",
            "last_market_round_started_at",
            "last_market_round_finished_at",
            "last_position_round_started_at",
            "last_position_round_finished_at",
            "okx_authoritative_sync",
        ):
            if stats.get(key) is None and runtime_stats.get(key) is not None:
                stats[key] = runtime_stats[key]
        stats.setdefault("runtime_status", runtime_stats.get("runtime_status", {}))
        return stats
    return await _split_process_trading_stats(mode)


def _execution_risk_floor(allocated: float, max_loss_pct: float, max_loss_usdt: float) -> float:
    if allocated <= 0:
        return 0.0
    loss_limit = max_loss_usdt if max_loss_usdt > 0 else allocated * max_loss_pct
    return max(allocated - loss_limit, 0.0)


def _cooldown_pause_reason_from_summary(pnl_summary: dict, cfg: dict, source: str) -> str | None:
    max_loss_usdt = _safe_float(cfg.get("max_loss_usdt"), 0.0) or 0.0
    cooldown_loss_pct = _safe_float(cfg.get("cooldown_loss_pct"), 0.0) or 0.0
    trigger_loss = max_loss_usdt * cooldown_loss_pct
    if trigger_loss <= 0:
        return None

    today_risk_pnl = _safe_float(pnl_summary.get("today_risk_pnl"), 0.0) or 0.0
    if today_risk_pnl > -trigger_loss:
        return None

    today_equity = (
        _safe_float(
            pnl_summary.get("today_equity_pnl"),
            pnl_summary.get("today_total_pnl", 0.0),
        )
        or 0.0
    )
    today_realized = _safe_float(pnl_summary.get("today_realized_pnl"), 0.0) or 0.0
    unrealized = _safe_float(pnl_summary.get("unrealized_pnl"), 0.0) or 0.0
    return (
        "冷静期触发比例已达到，暂停分析新的交易对。"
        f"{source} 今日权益盈亏 {today_equity:.2f} USDT，"
        f"其中今日已平仓盈亏 {today_realized:.2f} USDT，"
        f"当前持仓浮动盈亏 {unrealized:.2f} USDT；触发线为最高亏损金额 "
        f"{max_loss_usdt:.2f} USDT 的 {cooldown_loss_pct * 100:.0f}%"
        f"（{trigger_loss:.2f} USDT）。已有持仓仍会继续复盘、止盈止损和平仓。"
    )


def _okx_equity_pnl_from_snapshot(
    *,
    current_equity: float | None,
    pnl_summary: dict | None,
) -> dict[str, Any]:
    summary = pnl_summary or {}
    equity = _safe_float(current_equity, None)
    baseline = _safe_float(summary.get("today_equity_baseline"), None)
    if equity is None or equity <= 0 or baseline is None or baseline <= 0:
        return {
            "today_total_pnl": None,
            "today_equity_pnl": None,
            "cumulative_total_pnl": None,
            "total_pnl": None,
            "total_pnl_pct": None,
        }
    today_pnl = equity - baseline
    return {
        "today_total_pnl": today_pnl,
        "today_equity_pnl": today_pnl,
        "cumulative_total_pnl": None,
        "total_pnl": None,
        "total_pnl_pct": None,
    }


async def _phase3_equity_pnl_for_mode(
    mode: str,
    *,
    current_equity: float | None,
) -> dict[str, Any]:
    try:
        from db.session import get_session_ctx
        from services.equity_baseline import phase3_equity_change_from_snapshots

        async with get_session_ctx() as session:
            return await phase3_equity_change_from_snapshots(
                session,
                mode=mode,
                model_name=ENSEMBLE_TRADER_NAME,
                current_equity=current_equity,
            )
    except Exception as exc:
        _log_dashboard_fallback("phase3 equity pnl fallback", exc, mode=mode)
        return {
            "phase3_equity_pnl": None,
            "phase3_equity_pnl_pct": None,
            "phase3_equity_baseline": None,
            "phase3_equity_baseline_at": None,
            "phase3_equity_baseline_source": "okx_unavailable",
            "phase3_equity_start_date": PHASE3_FIRST_CLEAN_DAY,
        }


async def _get_execution_pnl_summary(mode: str) -> dict:
    """Return local trade diagnostics plus OKX-backed margin usage.

    This summary is intentionally not an account-profit source. Account equity
    and displayed account PnL must come from OKX snapshots only.
    """
    from sqlalchemy import case, func, select

    from db.session import get_session_ctx
    from models.trade import Order, Position
    from services.trade_fact_trust import (
        closed_position_trade_fact_trusted_with_orders,
        orders_by_exchange_id,
        split_exchange_order_ids,
    )

    selected_mode = "live" if mode == "live" else "paper"
    allocated = 0.0
    okx_current_equity: float | None = None
    realized_profit = 0.0
    realized_loss = 0.0
    today_realized_profit = 0.0
    today_realized_loss = 0.0
    unrealized_pnl = 0.0
    used_margin = 0.0
    open_count = 0
    position_rows = []
    now_local = datetime.now(timezone(timedelta(hours=8)))
    start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    start_utc = start_local.astimezone(UTC)

    balance_result, marks_result, symbols_result = await asyncio.gather(
        _dashboard_okx_balance_snapshot_for_mode(selected_mode),
        _get_exchange_position_mark_map(selected_mode),
        _get_exchange_open_position_symbols(selected_mode),
        return_exceptions=True,
    )
    if isinstance(balance_result, Exception):
        _log_dashboard_fallback(
            "dashboard balance snapshot fallback",
            balance_result,
            mode=selected_mode,
        )
        snapshot = None
    else:
        snapshot = balance_result
    okx_current_equity = _safe_float(
        (snapshot or {}).get("equity") or (snapshot or {}).get("total"),
        None,
    )
    allocated = (
        _safe_float(
            (snapshot or {}).get("allocatable")
            or (snapshot or {}).get("equity")
            or (snapshot or {}).get("total")
            or (snapshot or {}).get("free"),
            0.0,
        )
        or 0.0
    )
    if isinstance(marks_result, Exception):
        _log_dashboard_fallback(
            "exchange mark snapshot fallback",
            marks_result,
            mode=selected_mode,
        )
        exchange_marks = {}
    else:
        exchange_marks = marks_result
    if isinstance(symbols_result, Exception):
        _log_dashboard_fallback(
            "exchange open symbol fallback",
            symbols_result,
            mode=selected_mode,
        )
        exchange_symbols = None
    else:
        exchange_symbols = symbols_result
    exchange_temporarily_unavailable = bool(
        not exchange_marks and _dashboard_okx_positions_temporarily_unavailable(selected_mode)
    )

    try:
        async with get_session_ctx() as session:
            filters = (
                Position.execution_mode == selected_mode,
                Position.model_name.in_(EXECUTION_LEDGER_MODEL_NAMES),
            )
            all_result = await session.execute(
                select(Position)
                .where(*filters)
                .order_by(Position.created_at.asc(), Position.id.asc())
            )
            position_rows = list(all_result.scalars().all())
            realized_result = await session.execute(
                select(
                    func.coalesce(
                        func.sum(
                            case(
                                (Position.realized_pnl >= 0, Position.realized_pnl),
                                else_=0.0,
                            )
                        ),
                        0.0,
                    ),
                    func.coalesce(
                        func.sum(
                            case(
                                (Position.realized_pnl < 0, -Position.realized_pnl),
                                else_=0.0,
                            )
                        ),
                        0.0,
                    ),
                ).where(*filters, Position.is_open.is_(False))
            )
            realized_profit, realized_loss = [
                float(value or 0.0) for value in realized_result.one()
            ]
            today_result = await session.execute(
                select(
                    func.coalesce(
                        func.sum(
                            case(
                                (Position.realized_pnl >= 0, Position.realized_pnl),
                                else_=0.0,
                            )
                        ),
                        0.0,
                    ),
                    func.coalesce(
                        func.sum(
                            case(
                                (Position.realized_pnl < 0, -Position.realized_pnl),
                                else_=0.0,
                            )
                        ),
                        0.0,
                    ),
                ).where(
                    *filters,
                    Position.is_open.is_(False),
                    Position.closed_at >= start_utc,
                )
            )
            today_realized_profit, today_realized_loss = [
                float(value or 0.0) for value in today_result.one()
            ]
            linked_order_ids = {
                order_id
                for pos in position_rows
                if not pos.is_open
                for value in (
                    getattr(pos, "entry_exchange_order_id", None),
                    getattr(pos, "close_exchange_order_id", None),
                )
                for order_id in split_exchange_order_ids(value)
            }
            linked_orders = []
            if linked_order_ids:
                linked_orders = list(
                    (
                        await session.execute(
                            select(Order).where(
                                Order.execution_mode == selected_mode,
                                Order.exchange_order_id.in_(sorted(linked_order_ids)),
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
            linked_orders_by_id = orders_by_exchange_id(linked_orders)
            trusted_closed_rows = [
                pos
                for pos in position_rows
                if not pos.is_open
                and closed_position_trade_fact_trusted_with_orders(pos, linked_orders_by_id)
            ]
            realized_profit = sum(
                max(float(pos.realized_pnl or 0.0), 0.0) for pos in trusted_closed_rows
            )
            realized_loss = sum(
                abs(min(float(pos.realized_pnl or 0.0), 0.0)) for pos in trusted_closed_rows
            )
            today_trusted_closed_rows = [
                pos
                for pos in trusted_closed_rows
                if (closed_at := _as_utc_datetime(pos.closed_at)) is not None
                and closed_at >= start_utc
            ]
            today_realized_profit = sum(
                max(float(pos.realized_pnl or 0.0), 0.0) for pos in today_trusted_closed_rows
            )
            today_realized_loss = sum(
                abs(min(float(pos.realized_pnl or 0.0), 0.0)) for pos in today_trusted_closed_rows
            )
            fallback_margin_by_key: dict[tuple[str, str], float] = {}
            fallback_unrealized_by_key: dict[tuple[str, str], float] = {}
            for pos in position_rows:
                if not pos.is_open:
                    continue
                key = _dashboard_position_key(pos.symbol, pos.side)
                if not exchange_temporarily_unavailable and (
                    exchange_symbols is None or key[0] not in exchange_symbols
                ):
                    continue
                fallback_margin_by_key[key] = fallback_margin_by_key.get(
                    key, 0.0
                ) + _local_position_margin(pos)
                fallback_unrealized_by_key[key] = fallback_unrealized_by_key.get(
                    key, 0.0
                ) + (_safe_float(pos.unrealized_pnl, 0.0) or 0.0)
            if exchange_marks:
                exchange_totals = _exchange_position_totals(exchange_marks, fallback_margin_by_key)
                open_count = int(exchange_totals["open_count"])
                unrealized_pnl = float(exchange_totals["unrealized_pnl"])
                used_margin = float(exchange_totals["used_margin"])
            elif exchange_temporarily_unavailable:
                open_count = len(fallback_margin_by_key)
                unrealized_pnl = float(sum(fallback_unrealized_by_key.values()))
                used_margin = float(sum(fallback_margin_by_key.values()))
    except Exception as exc:
        _log_dashboard_fallback(
            "execution pnl database summary fallback",
            exc,
            mode=selected_mode,
        )

    realized_pnl = realized_profit - realized_loss
    today_realized_pnl = today_realized_profit - today_realized_loss
    total_pnl = realized_pnl + unrealized_pnl
    today_total_pnl = None
    today_risk_pnl = None
    equity_baseline = {}
    phase3_equity = {}
    try:
        from services.equity_baseline import apply_daily_equity_baseline

        async with get_session_ctx() as session:
            equity_baseline = await apply_daily_equity_baseline(
                session,
                mode=selected_mode,
                model_name=ENSEMBLE_TRADER_NAME,
                allocated=allocated,
                positions=position_rows,
                realized_pnl=realized_pnl,
                unrealized_pnl=unrealized_pnl,
                total_pnl=total_pnl,
                current_equity=okx_current_equity,
            )
        today_total_pnl = _safe_float(equity_baseline.get("today_equity_pnl"), None)
        today_risk_pnl = today_total_pnl
    except Exception as exc:
        _log_dashboard_fallback(
            "daily equity baseline fallback",
            exc,
            mode=selected_mode,
        )
        equity_baseline = {}
    phase3_equity = await _phase3_equity_pnl_for_mode(
        selected_mode,
        current_equity=okx_current_equity,
    )
    return {
        "allocated_balance": allocated,
        "realized_profit": realized_profit,
        "realized_loss": realized_loss,
        "realized_pnl": realized_pnl,
        "today_realized_profit": today_realized_profit,
        "today_realized_loss": today_realized_loss,
        "today_realized_pnl": today_realized_pnl,
        "today_closed_realized_profit": today_realized_profit,
        "today_closed_realized_loss": today_realized_loss,
        "today_closed_realized_pnl": today_realized_pnl,
        "today_equity_pnl": today_total_pnl,
        "today_equity_baseline": _safe_float(equity_baseline.get("today_equity_baseline"), None),
        "today_equity_baseline_total_pnl": _safe_float(
            equity_baseline.get("today_equity_baseline_total_pnl"),
            None,
        ),
        "today_equity_baseline_at": equity_baseline.get("today_equity_baseline_at"),
        "today_equity_baseline_source": equity_baseline.get("today_equity_baseline_source"),
        "today_snapshot_date": equity_baseline.get("today_snapshot_date"),
        "phase3_equity_pnl": phase3_equity.get("phase3_equity_pnl"),
        "phase3_equity_pnl_pct": phase3_equity.get("phase3_equity_pnl_pct"),
        "phase3_equity_baseline": phase3_equity.get("phase3_equity_baseline"),
        "phase3_equity_baseline_at": phase3_equity.get("phase3_equity_baseline_at"),
        "phase3_equity_baseline_source": phase3_equity.get("phase3_equity_baseline_source"),
        "phase3_equity_start_date": phase3_equity.get("phase3_equity_start_date"),
        "today_total_pnl": today_total_pnl,
        "today_risk_pnl": today_risk_pnl,
        "unrealized_pnl": unrealized_pnl,
        "total_pnl": total_pnl,
        "cumulative_realized_pnl": realized_pnl,
        "cumulative_unrealized_pnl": unrealized_pnl,
        "cumulative_total_pnl": total_pnl,
        "used_margin": used_margin,
        "remaining_allocation": None,
        "open_positions": open_count,
    }


def _clamp_confidence(value: float | None) -> float:
    if value is None:
        return 0.0
    return max(min(float(value), 1.0), 0.0)


def _analysis_display_confidence(
    action: str, trade_confidence: float, experts: list[dict], raw: dict
) -> float:
    """Confidence shown on collaboration records.

    Trade confidence intentionally stays 0 for hold decisions so the executor
    cannot mistake an observation for an order. Analysis records need a
    separate confidence that reflects how strongly the experts supported the
    final collaboration outcome.
    """
    trade_confidence = _clamp_confidence(trade_confidence)
    if action != "hold" or trade_confidence > 0:
        return trade_confidence

    weighted_total = 0.0
    weight_sum = 0.0
    plain_confidences: list[float] = []
    for expert in experts:
        confidence = _safe_float(expert.get("confidence"), None)
        if confidence is None:
            continue
        confidence = _clamp_confidence(confidence)
        plain_confidences.append(confidence)
        weight = _safe_float(expert.get("weight"), 0.0) or 0.0
        if weight > 0:
            weighted_total += confidence * weight
            weight_sum += weight

    if weight_sum > 0:
        return _clamp_confidence(weighted_total / weight_sum)
    if plain_confidences:
        return _clamp_confidence(sum(plain_confidences) / len(plain_confidences))

    weighted_score = abs(_safe_float(raw.get("weighted_score"), 0.0) or 0.0)
    return _clamp_confidence(weighted_score)


def _build_execution_account_status(
    mode: str,
    paper_summary: dict | None = None,
    okx_account: dict | None = None,
    pnl_summary: dict | None = None,
) -> dict:
    """Build the single execution-account payload shown on the dashboard."""
    mode = "live" if mode == "live" else "paper"
    cfg = settings.get_execution_account_config(mode)
    pnl_summary = pnl_summary or {}
    max_loss_pct = float(cfg.get("max_loss_pct") or 0.0)
    okx_error = str(okx_account.get("error")) if okx_account and okx_account.get("error") else None
    raw_okx_available = _safe_float(okx_account.get("free"), None) if okx_account else None
    raw_okx_used = _safe_float(okx_account.get("used"), 0.0) if okx_account else None
    raw_okx_total = (
        _safe_float(okx_account.get("total"), raw_okx_available) if okx_account else None
    )
    raw_okx_cash = _safe_float(okx_account.get("cash"), raw_okx_total) if okx_account else None
    raw_okx_equity = _safe_float(okx_account.get("equity"), raw_okx_total) if okx_account else None
    raw_okx_allocatable = (
        _safe_float(okx_account.get("allocatable"), raw_okx_equity or raw_okx_total or raw_okx_available)
        if okx_account
        else None
    )
    okx_error_blocks_balance = bool(
        okx_error
        and max(
            _safe_float(raw_okx_available, 0.0),
            _safe_float(raw_okx_total, 0.0),
            _safe_float(raw_okx_cash, 0.0),
            _safe_float(raw_okx_equity, 0.0),
            _safe_float(raw_okx_allocatable, 0.0),
        )
        <= 0
    )
    okx_available = None if okx_error_blocks_balance else raw_okx_available
    okx_used = None if okx_error_blocks_balance else raw_okx_used
    okx_total = None if okx_error_blocks_balance else raw_okx_total
    okx_cash = None if okx_error_blocks_balance else raw_okx_cash
    okx_equity = None if okx_error_blocks_balance else raw_okx_equity
    okx_allocatable = None if okx_error_blocks_balance else raw_okx_allocatable
    okx_snapshot_for_balance = (
        {
            "free": okx_available,
            "used": okx_used,
            "total": okx_total,
            "cash": okx_cash,
            "equity": okx_equity,
            "allocatable": okx_allocatable,
        }
        if okx_account and not okx_error_blocks_balance
        else None
    )
    parsed_account_equity = balance_from_snapshot(okx_snapshot_for_balance)
    parsed_allocatable = allocatable_balance_from_snapshot(okx_snapshot_for_balance)
    parsed_tradeable = tradeable_balance_from_snapshot(okx_snapshot_for_balance)
    okx_balance_available = bool(okx_snapshot_for_balance and parsed_account_equity > 0)
    account_equity = parsed_account_equity if okx_balance_available else 0.0
    okx_pnl = _okx_equity_pnl_from_snapshot(
        current_equity=account_equity if okx_balance_available else None,
        pnl_summary=pnl_summary,
    )
    max_loss_usdt = (
        account_equity * max_loss_pct if account_equity > 0 and max_loss_pct > 0 else 0.0
    )
    risk_floor = _execution_risk_floor(account_equity, max_loss_pct, max_loss_usdt)
    pause_reason = None
    if _trading_service:
        pause_reason = getattr(_trading_service, "_new_pair_pause_reasons", {}).get(
            ENSEMBLE_TRADER_NAME
        )
    pause_reason = _translate_pause_reason(pause_reason)
    if not okx_error and not okx_balance_available:
        okx_error = "OKX balance unavailable"
    if okx_error and not pause_reason and not okx_balance_available:
        source = "OKX 实盘账户" if mode == "live" else "OKX 模拟盘账户"
        pause_reason = f"{source} 余额同步失败，系统不会分析新的交易对。原因：{okx_error}"
    total_pnl_for_risk = 0.0
    if (
        account_equity > 0
        and max_loss_usdt > 0
        and total_pnl_for_risk <= -max_loss_usdt
        and not pause_reason
    ):
        source = "OKX 实盘账户" if mode == "live" else "OKX 模拟盘账户"
        pause_reason = (
            f"{source} AI 执行账户累计盈亏 {total_pnl_for_risk:.2f} USDT 已达到最高亏损限制 "
            f"{max_loss_pct * 100:.1f}%（{max_loss_usdt:.2f} USDT），系统不会分析新的交易对。"
        )
    if not pause_reason and okx_pnl["today_equity_pnl"] is not None:
        source = "OKX 实盘账户" if mode == "live" else "OKX 模拟盘账户"
        pause_reason = _cooldown_pause_reason_from_summary(
            {
                "today_risk_pnl": okx_pnl["today_equity_pnl"],
                "today_equity_pnl": okx_pnl["today_equity_pnl"],
                "today_total_pnl": okx_pnl["today_total_pnl"],
            },
            {**cfg, "max_loss_usdt": max_loss_usdt},
            source,
        )

    payload = {
        **cfg,
        "model_name": ENSEMBLE_TRADER_NAME,
        "allocated_balance": None,
        "account_balance_source_value": account_equity,
        "account_equity": account_equity,
        "max_loss_usdt": max_loss_usdt,
        "risk_floor": risk_floor,
        "risk_paused": bool(pause_reason),
        "risk_pause_reason": pause_reason,
        "balance_error": okx_error,
        "okx_available_balance": okx_available,
        "okx_used_balance": okx_used,
        "okx_total_balance": okx_total,
        "okx_cash_balance": okx_cash,
        "okx_equity_balance": okx_equity,
        "max_allocatable_balance": parsed_allocatable,
        "allocation_exceeds_balance": False,
        "account_pnl_source": "okx_authoritative" if okx_balance_available else "okx_unavailable",
        "local_trade_total_pnl": _safe_float(
            pnl_summary.get("cumulative_total_pnl"),
            _safe_float(pnl_summary.get("total_pnl"), 0.0),
        ),
        "local_trade_today_pnl": _safe_float(pnl_summary.get("today_closed_realized_pnl"), 0.0),
        "positions": [],
        "open_positions": int(pnl_summary.get("open_positions") or 0),
        "realized_profit": _safe_float(pnl_summary.get("realized_profit"), 0.0),
        "realized_loss": _safe_float(pnl_summary.get("realized_loss"), 0.0),
        "realized_pnl": _safe_float(pnl_summary.get("realized_pnl"), 0.0),
        "today_realized_profit": _safe_float(pnl_summary.get("today_realized_profit"), 0.0),
        "today_realized_loss": _safe_float(pnl_summary.get("today_realized_loss"), 0.0),
        "today_realized_pnl": _safe_float(pnl_summary.get("today_realized_pnl"), 0.0),
        "today_closed_realized_profit": _safe_float(
            pnl_summary.get("today_closed_realized_profit"), 0.0
        ),
        "today_closed_realized_loss": _safe_float(
            pnl_summary.get("today_closed_realized_loss"), 0.0
        ),
        "today_closed_realized_pnl": _safe_float(pnl_summary.get("today_closed_realized_pnl"), 0.0),
        "today_equity_pnl": okx_pnl["today_equity_pnl"],
        "today_equity_baseline": _safe_float(pnl_summary.get("today_equity_baseline"), None),
        "today_equity_baseline_total_pnl": _safe_float(
            pnl_summary.get("today_equity_baseline_total_pnl"),
            None,
        ),
        "today_equity_baseline_at": pnl_summary.get("today_equity_baseline_at"),
        "today_equity_baseline_source": pnl_summary.get("today_equity_baseline_source"),
        "today_snapshot_date": pnl_summary.get("today_snapshot_date"),
        "phase3_equity_pnl": _safe_float(pnl_summary.get("phase3_equity_pnl"), None),
        "phase3_equity_pnl_pct": _safe_float(pnl_summary.get("phase3_equity_pnl_pct"), None),
        "phase3_equity_baseline": _safe_float(pnl_summary.get("phase3_equity_baseline"), None),
        "phase3_equity_baseline_at": pnl_summary.get("phase3_equity_baseline_at"),
        "phase3_equity_baseline_source": pnl_summary.get("phase3_equity_baseline_source"),
        "phase3_equity_start_date": pnl_summary.get("phase3_equity_start_date")
        or PHASE3_FIRST_CLEAN_DAY,
        "today_total_pnl": okx_pnl["today_total_pnl"],
        "today_risk_pnl": _safe_float(pnl_summary.get("today_risk_pnl"), None),
        "cumulative_profit": _safe_float(pnl_summary.get("realized_profit"), 0.0),
        "cumulative_loss": _safe_float(pnl_summary.get("realized_loss"), 0.0),
        "cumulative_realized_pnl": _safe_float(pnl_summary.get("cumulative_realized_pnl"), 0.0),
        "cumulative_unrealized_pnl": _safe_float(pnl_summary.get("cumulative_unrealized_pnl"), 0.0),
        "cumulative_total_pnl": okx_pnl["cumulative_total_pnl"],
        "total_pnl": okx_pnl["total_pnl"],
        "total_pnl_pct": okx_pnl["total_pnl_pct"],
        "remaining_allocation": _execution_tradeable_balance(
            okx_available=okx_available,
            okx_allocatable=okx_allocatable,
            okx_equity=okx_equity,
            okx_total=okx_total,
            fallback_available=None,
        )
        or parsed_tradeable
        or 0.0,
    }

    if mode == "paper":
        summary = paper_summary or {}
        unrealized = _safe_float(pnl_summary.get("unrealized_pnl"), 0.0) or 0.0
        available = _execution_tradeable_balance(
            okx_available=okx_available,
            okx_allocatable=okx_allocatable,
            okx_equity=okx_equity,
            okx_total=okx_total,
            fallback_available=None,
        )
        used_margin = okx_used
        wallet = (
            okx_cash if okx_cash is not None else (okx_total if okx_total is not None else None)
        )
        equity = (
            okx_equity if okx_equity is not None else (okx_total if okx_total is not None else None)
        )
        payload.update(
            {
                "available_balance": available,
                "current_balance": available,
                "tradeable_balance": available,
                "remaining_allocation": available,
                "wallet_balance": wallet,
                "equity": equity,
                "used_margin": used_margin,
                "position_margin_used": used_margin,
                "unrealized_pnl": unrealized,
                "total_pnl": okx_pnl["total_pnl"],
                "cumulative_total_pnl": okx_pnl["cumulative_total_pnl"],
                "total_pnl_pct": okx_pnl["total_pnl_pct"],
                "initial_balance": None,
                "paper_execution_available_balance": _safe_float(
                    available,
                    None,
                ),
                "paper_execution_used_margin": used_margin,
                "positions": summary.get("positions", []),
                "open_positions": int(pnl_summary.get("open_positions") or 0),
                "balance_snapshot_stale": bool(okx_account and okx_account.get("stale") is True),
                "balance_snapshot_age_seconds": (
                    _safe_float(okx_account.get("stale_age_seconds"), None) if okx_account else None
                ),
                "balance_source": "OKX 模拟盘账户" if okx_account else "模拟盘执行账户",
            }
        )
        return payload

    if okx_account and not okx_error:
        available = _execution_tradeable_balance(
            okx_available=okx_available,
            okx_allocatable=okx_allocatable,
            okx_equity=okx_equity,
            okx_total=okx_total,
            fallback_available=None,
        ) or (okx_available if okx_available is not None else okx_allocatable)
        used_margin = (
            okx_used if okx_used is not None else _safe_float(pnl_summary.get("used_margin"), 0.0)
        )
        total = okx_total
        unrealized = _safe_float(pnl_summary.get("unrealized_pnl"), 0.0)
        payload.update(
            {
                "available_balance": available,
                "current_balance": available,
                "tradeable_balance": available,
                "remaining_allocation": available,
                "wallet_balance": okx_cash if okx_cash is not None else total,
                "equity": okx_equity if okx_equity is not None else total,
                "used_margin": used_margin,
                "position_margin_used": used_margin,
                "unrealized_pnl": unrealized,
                "total_pnl": okx_pnl["total_pnl"],
                "cumulative_total_pnl": okx_pnl["cumulative_total_pnl"],
                "total_pnl_pct": okx_pnl["total_pnl_pct"],
                "initial_balance": None,
                "balance_source": "OKX 实盘账户",
                "balance_snapshot_stale": bool(okx_account.get("stale")),
                "balance_snapshot_age_seconds": _safe_float(
                    okx_account.get("stale_age_seconds"), None
                ),
            }
        )
    else:
        payload.update(
            {
                "allocated_balance": None,
                "account_equity": None,
                "max_loss_usdt": None,
                "risk_floor": None,
                "available_balance": None,
                "current_balance": None,
                "tradeable_balance": None,
                "remaining_allocation": None,
                "wallet_balance": None,
                "equity": None,
                "used_margin": None,
                "position_margin_used": None,
                "unrealized_pnl": None,
                "total_pnl": None,
                "total_pnl_pct": None,
                "initial_balance": None,
                "balance_source": "OKX 实盘账户",
                "balance_error": okx_error or "实盘账户未连接或余额查询失败",
                "balance_snapshot_stale": False,
                "balance_snapshot_age_seconds": None,
            }
        )
    return payload


def _order_status_label(status: str | None) -> str:
    status_map = {
        "filled": "执行成功",
        "partial": "部分成交",
        "open": "等待成交",
        "pending": "等待提交",
        "cancelled": "已取消",
        "canceled": "已取消",
        "rejected": "执行失败",
    }
    return status_map.get(str(status or "").lower(), str(status or "未知"))


def _translate_pause_reason(reason: str | None) -> str | None:
    text = str(reason or "").strip()
    if not text:
        return None
    if "Execution account reached max loss limit" in text:
        total_match = re.search(r"total_pnl=([-0-9.]+)\s*USDT", text)
        max_match = re.search(r"max_loss=([-0-9.]+)\s*USDT", text)
        pct_match = re.search(r"\(([-0-9.]+)%\)", text)
        total = total_match.group(1) if total_match else "-"
        max_loss = max_match.group(1) if max_match else "-"
        pct = pct_match.group(1) if pct_match else "-"
        return (
            f"执行账户已达到最高亏损限制：当前累计盈亏 {total} USDT，"
            f"最高允许亏损 {max_loss} USDT（{pct}%）。暂停分析新的交易对。"
        )
    if "Risk circuit breaker is open" in text:
        detail = text.split("reason=", 1)[1] if "reason=" in text else "触发风险阈值"
        return f"风险熔断已开启，暂停分析新的交易对。原因：{detail}"
    if "OKX usable balance snapshot is unavailable" in text:
        return "未获取到 OKX 可用余额快照，暂停分析新的交易对。"
    if "OKX equity/balance is unavailable" in text:
        return "未获取到 OKX 账户权益或余额，暂停分析新的交易对。"
    if "OKX tradable balance is too low" in text:
        available = re.search(r"available=([-0-9.]+)\s*USDT", text)
        required = re.search(r"minimum_required=([-0-9.]+)\s*USDT", text)
        return (
            f"OKX 可交易余额过低：当前可用 {available.group(1) if available else '-'} USDT，"
            f"最低需要 {required.group(1) if required else '-'} USDT，暂停分析新的交易对。"
        )
    return text


def _fallback_execution_reason(decision, order=None) -> str | None:
    return _fallback_execution_reason_clean(decision, order)


def _decision_raw_payload(decision) -> dict[str, Any]:
    raw = getattr(decision, "raw_llm_response", None)
    if not isinstance(raw, dict):
        raw = getattr(decision, "raw_response", None)
    return raw if isinstance(raw, dict) else {}


def _fallback_execution_reason_clean(decision, order=None) -> str | None:
    """Build a clean, non-mojibake execution reason for dashboard details."""

    if decision.was_executed:
        return None

    action = decision.action
    if action == "hold":
        return "AI 选择观望，未提交订单。"

    if action not in ("long", "short", "close_long", "close_short"):
        return "未保存具体未执行原因。"

    snapshot = decision.feature_snapshot or {}
    raw = _decision_raw_payload(decision)
    entry_filters = entry_filters_from_context(
        raw or {"entry_filters": snapshot.get("entry_filters")}
    )
    confidence = float(decision.confidence or 0.0)
    volume_ratio = float(snapshot.get("volume_ratio") or 0.0)
    adx_14 = float(snapshot.get("adx_14") or 0.0)
    price_vs_sma20 = float(snapshot.get("price_vs_sma20") or 0.0)
    price_vs_sma50 = float(snapshot.get("price_vs_sma50") or 0.0)

    if confidence < settings.confidence_threshold:
        return (
            f"分析信心偏低：当前 {confidence:.2f}，系统会降低排序或仓位；"
            "这不是固定开仓门槛，最终仍由收益质量、风险、OKX规则和动态调度共同决定。"
        )

    if action in ("long", "short"):
        if volume_ratio < entry_filters.min_entry_volume_ratio:
            return (
                f"量能低于本轮运行时参考：当前量能倍数 {volume_ratio:.2f}，"
                f"动态参考 {entry_filters.min_entry_volume_ratio:.2f}。"
                "该参考只影响排序/仓位，不是固定硬门槛。"
            )

        if adx_14 < entry_filters.min_entry_adx:
            return (
                f"趋势强度低于本轮运行时参考：当前 ADX {adx_14:.1f}，"
                f"动态参考 {entry_filters.min_entry_adx:.1f}。"
                "该参考只影响排序/仓位，不是固定硬门槛。"
            )

        if action == "long" and (price_vs_sma20 <= 0 or price_vs_sma50 <= 0):
            return "做多趋势未完全对齐：价格没有同时站上 SMA20 和 SMA50，本轮未提交订单。"
        if action == "short" and (price_vs_sma20 >= 0 or price_vs_sma50 >= 0):
            return "做空趋势未完全对齐：价格没有同时跌破 SMA20 和 SMA50，本轮未提交订单。"

        return (
            "未找到关联订单记录：这通常表示信号仍在候选排序、下单前检查、"
            "OKX规则校验或订单回写阶段被跳过；请查看本条执行步骤时间线定位具体节点。"
        )

    if action in ("close_long", "close_short"):
        if order is not None:
            status_label = _order_status_label(getattr(order, "status", None))
            exchange_order_id = str(getattr(order, "exchange_order_id", "") or "").strip()
            if exchange_order_id:
                return (
                    f"系统已向 OKX 提交平仓单，OKX 订单号 {exchange_order_id}，"
                    f"最终状态为 {status_label}。如果数量为 0，说明 OKX 未确认成交或仓位未减少。"
                )
            return (
                f"本地已生成平仓单，但未拿到 OKX 订单号，当前本地订单状态为 {status_label}。"
                "请以同时间附近的执行记录和 OKX 订单状态为准。"
            )
        recovered = _recover_execution_reason_from_raw_decision(decision)
        if recovered:
            return recovered
        return (
            "这条平仓决策没有找到对应的本地平仓委托记录，因此系统未把它视为已执行。"
            "请以 OKX 订单状态和执行记录为准。"
        )

    return "未保存具体未执行原因。"


def _humanize_expert_failure(reason: str | None) -> str:
    """Convert provider/runtime errors into short UI-friendly Chinese text."""
    text = str(reason or "").strip()
    lower = text.lower()
    if not text:
        return "本轮没有返回结果。"
    if "timeout" in lower or "timed out" in lower or "readtimeout" in lower:
        return "模型调用超时，本轮没有返回结果。"
    if "json" in lower or "decode" in lower or "parse" in lower:
        return "模型返回格式不符合 JSON 要求，本轮结果被丢弃。"
    if (
        "invalid response" in lower
        or "empty" in lower
        or "not supported" in lower
        or "不被该代理支持" in text
    ):
        return "模型接口返回空结果，可能是当前代理不支持这个模型名。"
    if "401" in lower or "unauthorized" in lower or "invalid api key" in lower:
        return "API Key 无效或没有权限，本轮没有返回结果。"
    if "403" in lower or "forbidden" in lower or "permission" in lower:
        return "模型或接口权限不足，本轮没有返回结果。"
    if "429" in lower or "rate limit" in lower or "too many requests" in lower:
        return "接口限流，本轮没有返回结果。"
    if "connect" in lower or "connection" in lower or "network" in lower:
        return "模型接口连接失败，本轮没有返回结果。"
    return "模型调用失败，本轮没有返回结果。"


def _decision_type(action: str | None) -> tuple[str, str]:
    if action in ("long", "short"):
        return "entry", "开仓决策"
    if action in ("close_long", "close_short"):
        return "exit", "平仓决策"
    if action == "hold":
        return "hold", "观望决策"
    return "other", "其他决策"


def set_services(trading_svc, data_svc, competition_svc):
    """Called by main loop to inject service references for API access."""
    global _trading_service, _data_service, _competition_service
    _trading_service = trading_svc
    _data_service = data_svc
    _competition_service = competition_svc


def _dashboard_okx_executor_for_mode(mode: str) -> Any | None:
    if not _trading_service:
        return None
    getter = getattr(_trading_service, "okx_executor_for_dashboard", None)
    if not callable(getter):
        return None
    return getter("live" if mode == "live" else "paper")


def _trading_service_cached_okx_balance_snapshot(mode: str) -> dict[str, Any] | None:
    if not _trading_service:
        return None
    peeker = getattr(_trading_service, "peek_okx_balance_snapshot_for_mode", None)
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


async def _fetch_dashboard_okx_positions_uncached(
    selected_mode: str,
    executor: Any | None = None,
) -> list[dict[str, Any]]:
    executor = executor if executor is not None else _dashboard_okx_executor_for_mode(selected_mode)
    if executor:
        fetch_strict = getattr(executor, "get_positions_strict", None)
        if callable(fetch_strict):
            return await asyncio.wait_for(
                fetch_strict(),
                timeout=_DASHBOARD_OKX_POSITION_READ_TIMEOUT_SECONDS,
            )

    fallback_executor = _make_lightweight_okx_executor(OKXExecutor, selected_mode)
    try:
        async def fetch_with_fallback() -> list[dict[str, Any]]:
            await fallback_executor.initialize()
            return await fallback_executor.get_positions_strict()

        return await asyncio.wait_for(
            fetch_with_fallback(),
            timeout=_DASHBOARD_OKX_POSITION_INITIALIZE_TIMEOUT_SECONDS,
        )
    finally:
        try:
            await fallback_executor.shutdown()
        except Exception as exc:
            _log_dashboard_fallback(
                "dashboard okx position fallback shutdown failed",
                exc,
                mode=selected_mode,
            )


async def _fetch_dashboard_okx_positions(selected_mode: str) -> list[dict[str, Any]]:
    """Fetch current OKX positions once per dashboard refresh window.

    The dashboard needs both open-symbol and mark-price views.  Without this
    cache, a single page refresh can hit the same OKX private positions API
    several times and amplify transient slowness into endpoint timeouts.
    """

    selected_mode = "live" if selected_mode == "live" else "paper"
    executor = _dashboard_okx_executor_for_mode(selected_mode)
    executor_identity = executor
    now = datetime.now(UTC)
    cached = _dashboard_okx_position_cache.get(selected_mode)
    if cached:
        cached_at, cached_value, cached_executor_identity = cached
        cache_age_seconds = (now - cached_at).total_seconds()
        if (
            cached_executor_identity is executor_identity
            and cache_age_seconds <= _EXCHANGE_MARK_CACHE_TTL_SECONDS
        ):
            return copy.deepcopy(cached_value)
        if (
            cached_executor_identity is executor_identity
            and cache_age_seconds <= _DASHBOARD_OKX_POSITION_STALE_CACHE_TTL_SECONDS
        ):
            _start_dashboard_okx_position_refresh(selected_mode)
            return copy.deepcopy(cached_value)

    cached_error = _dashboard_okx_position_error_cache.get(selected_mode)
    if cached_error:
        cached_at, cached_text, cached_executor_identity = cached_error
        if (
            cached_executor_identity is executor_identity
            and (now - cached_at).total_seconds()
            <= _DASHBOARD_OKX_POSITION_ERROR_CACHE_TTL_SECONDS
        ):
            if cached:
                return copy.deepcopy(cached[1])
            raise RuntimeError(cached_text)

    lock = _dashboard_okx_position_locks.setdefault(selected_mode, asyncio.Lock())
    async with lock:
        now = datetime.now(UTC)
        cached = _dashboard_okx_position_cache.get(selected_mode)
        if cached:
            cached_at, cached_value, cached_executor_identity = cached
            cache_age_seconds = (now - cached_at).total_seconds()
            if (
                cached_executor_identity is executor_identity
                and cache_age_seconds <= _EXCHANGE_MARK_CACHE_TTL_SECONDS
            ):
                return copy.deepcopy(cached_value)
            if (
                cached_executor_identity is executor_identity
                and cache_age_seconds <= _DASHBOARD_OKX_POSITION_STALE_CACHE_TTL_SECONDS
            ):
                _start_dashboard_okx_position_refresh(selected_mode)
                return copy.deepcopy(cached_value)
        cached_error = _dashboard_okx_position_error_cache.get(selected_mode)
        if cached_error:
            cached_at, cached_text, cached_executor_identity = cached_error
            if (
                cached_executor_identity is executor_identity
                and
                (now - cached_at).total_seconds()
                <= _DASHBOARD_OKX_POSITION_ERROR_CACHE_TTL_SECONDS
            ):
                if cached:
                    return copy.deepcopy(cached[1])
                raise RuntimeError(cached_text)

        try:
            positions = await _fetch_dashboard_okx_positions_uncached(
                selected_mode,
                executor=executor,
            )
        except Exception as exc:
            _dashboard_okx_position_error_cache[selected_mode] = (
                datetime.now(UTC),
                _dashboard_okx_error_text(exc, resource="持仓"),
                executor_identity,
            )
            if cached and cached[2] is executor_identity:
                return copy.deepcopy(cached[1])
            raise

        normalized_positions = [dict(position) for position in positions or []]
        _dashboard_okx_position_cache[selected_mode] = (
            datetime.now(UTC),
            copy.deepcopy(normalized_positions),
            executor_identity,
        )
        _dashboard_okx_position_error_cache.pop(selected_mode, None)
        return normalized_positions


async def _dashboard_okx_balance_snapshot_for_mode(mode: str) -> dict[str, Any] | None:
    return await _get_dashboard_okx_account_snapshot(mode)


def _trading_service_is_running() -> bool:
    if not _trading_service:
        return False
    is_running = getattr(_trading_service, "is_running", None)
    return bool(is_running()) if callable(is_running) else False


def _make_lightweight_okx_executor(cls: Any, mode: str):
    """Create an OKX executor for balance-only reads without loading market rules."""
    try:
        return cls(mode=mode, load_markets_on_initialize=False)
    except TypeError:
        return cls(mode=mode)


def _dashboard_ml_signal_service() -> Any | None:
    if _trading_service and getattr(_trading_service, "ml_signal_service", None):
        return _trading_service.ml_signal_service
    global _ml_signal_status_service
    if _ml_signal_status_service is None:
        from services.ml_signal_service import MLSignalService

        _ml_signal_status_service = MLSignalService()
    return _ml_signal_status_service


def _dashboard_local_ai_tools_client() -> Any | None:
    if _trading_service and getattr(_trading_service, "local_ai_tools", None):
        return _trading_service.local_ai_tools
    global _local_ai_tools_status_client
    if _local_ai_tools_status_client is None:
        from services.local_ai_tools_client import LocalAIToolsClient

        _local_ai_tools_status_client = LocalAIToolsClient()
    return _local_ai_tools_status_client


async def _completed_ml_shadow_sample_count() -> int:
    try:
        from services.ml_signal_service import count_shadow_training_rows

        db_count = int(await count_shadow_training_rows())
        if db_count >= 0:
            return db_count
    except Exception as exc:
        _log_dashboard_fallback("ml signal phase3 sample count fallback", exc)
    ml_signal_service = _dashboard_ml_signal_service()
    if not ml_signal_service:
        return 0
    counter = getattr(ml_signal_service, "completed_shadow_sample_count", None)
    if not callable(counter):
        return 0
    return int(await counter())


def _safe_int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default


def _explicit_phase3_count(status: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key not in status:
            continue
        value = _safe_int_value(status.get(key), 0)
        if value >= 0:
            return value
    return None


async def _completed_local_ai_shadow_backtest_total() -> int:
    if not _trading_service:
        try:
            from services.ml_signal_service import count_shadow_training_rows

            return int(await count_shadow_training_rows())
        except Exception as exc:
            _log_dashboard_fallback("local ai tools phase3 sample count fallback", exc)
            return 0
    counter = getattr(_trading_service, "completed_shadow_backtest_total", None)
    if not callable(counter):
        return 0
    return int(await counter())


def _reset_trading_decision_runtime_state() -> None:
    if not _trading_service:
        return
    reset = getattr(_trading_service, "reset_decision_runtime_state", None)
    if callable(reset):
        reset()


def _normalize_dashboard_symbol(symbol: str | None) -> str:
    """Normalize exchange/CCXT symbols to the dashboard ticker key format."""
    return normalize_trading_symbol(symbol)


def _dashboard_symbol_query_variants(symbols: set[str]) -> set[str]:
    """Return historical symbol spellings used across orders, decisions, and shadows."""
    return symbol_query_variants(symbols)


def _dashboard_split_exchange_order_ids(value: Any) -> set[str]:
    tokens = {str(value or "").strip()}
    if not next(iter(tokens), ""):
        return set()
    for separator in (",", ";", "|", "\n", "\t", " "):
        pieces: set[str] = set()
        for token in tokens:
            pieces.update(part.strip() for part in token.split(separator) if part.strip())
        tokens = pieces
    return {token for token in tokens if token}


async def _dashboard_closed_position_ledger_rows(
    session: Any,
    repo: Any,
    *,
    mode: str | None,
    model_names: tuple[str, ...] | None = None,
    page: int = 1,
    page_size: int = 20,
    paginate: bool = True,
) -> tuple[list[dict[str, Any]], int, int, int, str]:
    from sqlalchemy import select

    from models.account import OkxAccountBill
    from models.trade import Order, Position
    from services.okx_order_fact_sync import (
        OKX_SYNC_CONFIRMED,
        OKX_SYNC_EXECUTION_RESULT_CONFIRMED,
        OKX_SYNC_OKX_ONLY,
    )
    from services.okx_position_ledger_view import build_okx_position_ledger_groups
    from services.position_settlement import is_final_settlement_status

    if model_names:
        closed_result = await session.execute(
            select(Position)
            .where(
                Position.execution_mode == mode if mode else True,
                Position.model_name.in_(model_names),
                Position.is_open.is_(False),
            )
            .order_by(
                Position.closed_at.desc().nullslast(),
                Position.created_at.desc(),
            )
            .limit(5000)
        )
        closed_rows = list(closed_result.scalars().all())
    else:
        closed_rows = await repo.get_position_records(
            execution_mode=mode,
            limit=5000,
            offset=0,
            is_open=False,
        )
    closed_rows = [
        position
        for position in closed_rows
        if is_final_settlement_status(getattr(position, "settlement_status", None))
    ]
    position_history_rows = await _dashboard_okx_position_history_rows(
        mode=mode,
        closed_rows=closed_rows,
    )
    linked_order_ids = {
        token
        for position in closed_rows
        for value in (
            getattr(position, "entry_exchange_order_id", None),
            getattr(position, "close_exchange_order_id", None),
        )
        for token in _dashboard_split_exchange_order_ids(value)
    }
    official_symbols = {
        symbol
        for row in position_history_rows
        if isinstance(row, dict)
        for symbol in [
            symbol_from_okx_inst_id(
                str(row.get("instId") or row.get("inst_id") or "").strip().upper()
            )
            or normalize_trading_symbol(str(row.get("instId") or row.get("inst_id") or ""))
        ]
        if symbol
    }
    symbol_variants = _dashboard_symbol_query_variants(
        {
            _normalize_dashboard_symbol(str(getattr(position, "symbol", "") or ""))
            for position in closed_rows
            if getattr(position, "symbol", None)
        }
        | official_symbols
    )
    order_stmt = select(Order).where(Order.status == "filled")
    if mode:
        order_stmt = order_stmt.where(Order.execution_mode == mode)
    if symbol_variants:
        order_stmt = order_stmt.where(Order.symbol.in_(symbol_variants))
    else:
        order_stmt = order_stmt.where(Order.id == -1)
    order_rows = list(
        (
            await session.execute(
                order_stmt.order_by(
                    Order.filled_at.desc().nullslast(),
                    Order.created_at.desc(),
                ).limit(10000)
            )
        )
        .scalars()
        .all()
    )
    if linked_order_ids:
        order_rows = [
            order
            for order in order_rows
            if (
                _dashboard_split_exchange_order_ids(getattr(order, "exchange_order_id", None))
                & linked_order_ids
            )
            or str(getattr(order, "okx_sync_status", "") or "").strip()
            in {
                OKX_SYNC_CONFIRMED,
                OKX_SYNC_OKX_ONLY,
                OKX_SYNC_EXECUTION_RESULT_CONFIRMED,
            }
        ]
    if position_history_rows:
        account_bill_rows = await _dashboard_okx_account_bill_rows(
            session,
            closed_rows=closed_rows,
            mode=mode,
            account_bill_model=OkxAccountBill,
        )
        refreshed_groups = build_okx_position_ledger_groups(
            closed_rows,
            order_rows,
            account_bills=account_bill_rows,
            position_history_rows=position_history_rows,
            require_order_lifecycle_source_positions=True,
        )
        refreshed_groups = [
            group
            for group in refreshed_groups
            if is_final_settlement_status(getattr(group, "settlement_status", None))
        ]
        official_rows = _dashboard_position_history_official_rows_as_groups(
            position_history_rows,
            refreshed_groups,
            mode=mode,
            order_rows=order_rows,
            closed_rows=closed_rows,
        )
        if official_rows:
            official_total = len(official_rows)
            official_total_pages = (
                max(1, (official_total + page_size - 1) // page_size)
                if official_total
                else 1
            )
            page = min(max(int(page or 1), 1), official_total_pages)
            selected_official_rows = official_rows
            if paginate:
                start = (page - 1) * page_size
                selected_official_rows = official_rows[start : start + page_size]
            return (
                selected_official_rows,
                official_total,
                page,
                official_total_pages,
                "okx_positions_history_official",
            )
    return (
        [],
        0,
        1,
        1,
        "okx_positions_history_official_unavailable",
    )


def _closed_rows_for_selected_ledger_groups(
    closed_rows: list[Any],
    selected_groups: list[Any],
) -> list[Any]:
    selected_ids = {
        int(position_id)
        for group in selected_groups
        for position_id in group.position_ids
    }
    if not selected_ids:
        return []
    return [
        row
        for row in closed_rows
        if row.id is not None and int(row.id) in selected_ids
    ]


def _dashboard_ledger_group_refresh_key(group: Any) -> tuple[Any, ...]:
    return (
        getattr(group, "symbol", ""),
        getattr(group, "inst_id", ""),
        getattr(group, "side", ""),
        tuple(getattr(group, "position_ids", []) or []),
        tuple(getattr(group, "entry_order_ids", []) or []),
        tuple(getattr(group, "close_order_ids", []) or []),
    )


def _dashboard_ms_datetime(value: Any) -> datetime | None:
    number = _safe_float(value, 0.0) or 0.0
    if number <= 0:
        return None
    if number < 10_000_000_000:
        number *= 1000.0
    try:
        return datetime.fromtimestamp(number / 1000.0, tz=UTC)
    except (OSError, OverflowError, ValueError):
        return None


def _dashboard_position_history_side(row: dict[str, Any]) -> str:
    for key in ("posSide", "positionSide", "side", "direction"):
        value = str(row.get(key) or "").lower().strip()
        if value in {"long", "short"}:
            return value
    return ""


def _dashboard_position_history_close_status(row: dict[str, Any]) -> tuple[str, str]:
    close_type = str(row.get("type") or row.get("closeType") or "").strip()
    if close_type == "1":
        return "partial", "部分平仓"
    if close_type == "2":
        return "full", "全部平仓"
    closed_qty = _safe_float(row.get("closeTotalPos"), 0.0) or 0.0
    max_qty = _safe_float(row.get("openMaxPos"), 0.0) or 0.0
    if max_qty > 0 and closed_qty > 0 and closed_qty < max_qty:
        return "partial", "部分平仓"
    return "full", "全部平仓"


def _dashboard_position_history_group_id(row: dict[str, Any], mode: str | None) -> str:
    inst_id = str(row.get("instId") or row.get("inst_id") or "").strip().upper()
    pos_id = str(row.get("posId") or row.get("pos_id") or "").strip()
    c_time = str(row.get("cTime") or row.get("createdTime") or "").strip()
    u_time = str(row.get("uTime") or row.get("updatedTime") or "").strip()
    close_type = str(row.get("type") or row.get("closeType") or "").strip()
    raw = "|".join([mode or "", inst_id, pos_id, c_time, u_time, close_type])
    return raw.replace("/", "_").replace(":", "").replace("+", "")


def _dashboard_position_history_match_score(
    row: dict[str, Any],
    group: Any,
) -> int:
    row_inst_id = str(row.get("instId") or row.get("inst_id") or "").strip().upper()
    if row_inst_id and row_inst_id != str(getattr(group, "inst_id", "") or "").strip().upper():
        return -1
    score = 0
    if row_inst_id:
        score += 20
    row_side = _dashboard_position_history_side(row)
    if row_side and row_side == str(getattr(group, "side", "") or "").lower().strip():
        score += 15
    row_closed_qty = _safe_float(row.get("closeTotalPos"), 0.0) or 0.0
    group_closed_qty = _safe_float(getattr(group, "closed_quantity", None), 0.0) or 0.0
    if row_closed_qty > 0 and group_closed_qty > 0:
        denominator = max(abs(row_closed_qty), abs(group_closed_qty), 1e-12)
        if abs(row_closed_qty - group_closed_qty) / denominator <= 0.02:
            score += 30
    row_opened = _dashboard_ms_datetime(row.get("cTime") or row.get("createdTime"))
    group_opened = _as_utc_datetime(getattr(group, "opened_at", None))
    if row_opened and group_opened:
        delta = abs((row_opened - group_opened).total_seconds())
        if delta <= 300:
            score += max(0, 20 - int(delta // 30))
    row_updated = _dashboard_ms_datetime(row.get("uTime") or row.get("updatedTime"))
    group_closed = _as_utc_datetime(getattr(group, "closed_at", None))
    if row_updated and group_closed:
        delta = abs((row_updated - group_closed).total_seconds())
        if delta <= 3600:
            score += max(0, 20 - int(delta // 180))
    return score


def _dashboard_position_history_best_local_group(row: dict[str, Any], groups: list[Any]) -> Any | None:
    scored = [
        (score, group)
        for group in groups
        if (score := _dashboard_position_history_match_score(row, group)) >= 0
    ]
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    if scored[0][0] <= 0:
        return None
    return scored[0][1]


def _dashboard_linked_fill_unit_sums(
    fills: list[dict[str, Any]],
    order_ids: set[str],
) -> tuple[float, float]:
    base_quantity = 0.0
    contracts = 0.0
    for fill in fills:
        if not isinstance(fill, dict):
            continue
        order_id = str(fill.get("order_id") or "").strip()
        if order_ids and order_id not in order_ids:
            continue
        base_quantity += abs(_safe_float(fill.get("quantity"), 0.0) or 0.0)
        contracts += abs(_safe_float(fill.get("contracts"), 0.0) or 0.0)
    return base_quantity, contracts


def _dashboard_position_history_quantity_in_base_units(
    raw_quantity: Any,
    payload: dict[str, Any],
    *,
    order_id_key: str,
    fallback: Any,
) -> float:
    raw = _safe_float(raw_quantity, None)
    fallback_value = _safe_float(fallback, 0.0) or 0.0
    if raw is None or raw <= 0:
        return fallback_value
    linked_fills = payload.get("linked_fills") if isinstance(payload, dict) else []
    linked_fills = linked_fills if isinstance(linked_fills, list) else []
    order_ids = {str(item or "").strip() for item in payload.get(order_id_key, []) or []}
    base_quantity, contracts = _dashboard_linked_fill_unit_sums(linked_fills, order_ids)
    if base_quantity > 0:
        denominator = max(abs(raw), abs(base_quantity), 1e-12)
        if abs(raw - base_quantity) / denominator <= 0.02:
            return base_quantity
    if contracts > 0:
        denominator = max(abs(raw), abs(contracts), 1e-12)
        if abs(raw - contracts) / denominator <= 0.02:
            return base_quantity if base_quantity > 0 else fallback_value or raw
    if fallback_value > 0:
        denominator = max(abs(raw), abs(fallback_value), 1e-12)
        if abs(raw - fallback_value) / denominator <= 0.02:
            return fallback_value
    return raw


def _dashboard_position_history_status_label(status: str) -> str:
    close_type = "1" if status == "partial" else "2"
    return _dashboard_position_history_close_status({"type": close_type})[1]


def _dashboard_position_history_inferred_side(row: dict[str, Any]) -> str:
    explicit = _dashboard_position_history_side(row)
    if explicit:
        return explicit
    open_price = _safe_float(row.get("openAvgPx"), 0.0) or 0.0
    close_price = _safe_float(row.get("closeAvgPx"), 0.0) or 0.0
    pnl = _safe_float(row.get("pnl"), None)
    if pnl is None:
        pnl = _safe_float(row.get("realizedPnl"), None)
    if open_price <= 0 or close_price <= 0 or pnl is None or abs(close_price - open_price) <= 1e-12:
        return ""
    return "long" if (close_price > open_price) == (pnl >= 0) else "short"


def _dashboard_position_history_local_side(
    row: dict[str, Any],
    local_groups: list[Any],
    closed_rows: list[Any],
) -> str:
    explicit = _dashboard_position_history_side(row)
    if explicit:
        return explicit
    matched = _dashboard_position_history_best_local_group(row, local_groups)
    if matched is not None:
        side = str(getattr(matched, "side", "") or "").lower().strip()
        if side in {"long", "short"}:
            return side
    inst_id = str(row.get("instId") or row.get("inst_id") or "").strip().upper()
    pos_id = str(row.get("posId") or row.get("pos_id") or "").strip()
    opened_at = _dashboard_ms_datetime(row.get("cTime") or row.get("createdTime"))
    updated_at = _dashboard_ms_datetime(row.get("uTime") or row.get("updatedTime"))
    candidates: list[tuple[int, str]] = []
    for position in closed_rows:
        position_inst_id = (
            str(getattr(position, "okx_inst_id", "") or "").strip().upper()
            or okx_inst_id_from_symbol(getattr(position, "symbol", None))
            or ""
        )
        if inst_id and position_inst_id and position_inst_id != inst_id:
            continue
        score = 0
        if pos_id and str(getattr(position, "okx_pos_id", "") or "").strip() == pos_id:
            score += 20
        position_opened = _as_utc_datetime(getattr(position, "created_at", None))
        if opened_at and position_opened:
            delta = abs((opened_at - position_opened).total_seconds())
            if delta <= 1800:
                score += max(0, 10 - int(delta // 180))
        position_closed = _as_utc_datetime(getattr(position, "closed_at", None))
        if updated_at and position_closed:
            delta = abs((updated_at - position_closed).total_seconds())
            if delta <= 3600:
                score += max(0, 10 - int(delta // 360))
        side = str(getattr(position, "side", "") or "").lower().strip()
        if score > 0 and side in {"long", "short"}:
            candidates.append((score, side))
    if not candidates:
        return _dashboard_position_history_inferred_side(row)
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _dashboard_order_time(order: Any) -> datetime | None:
    return _as_utc_datetime(getattr(order, "filled_at", None)) or _as_utc_datetime(
        getattr(order, "created_at", None)
    )


def _dashboard_order_raw_fills(order: Any) -> dict[str, Any]:
    raw = getattr(order, "okx_raw_fills", None)
    return raw if isinstance(raw, dict) else {}


def _dashboard_order_base_quantity(order: Any) -> float:
    raw = _dashboard_order_raw_fills(order)
    for key in ("base_quantity", "baseQuantity", "fill_base_quantity"):
        value = _safe_float(raw.get(key), None)
        if value is not None and abs(value) > 0:
            return abs(value)
    return abs(_safe_float(getattr(order, "quantity", None), 0.0) or 0.0)


def _dashboard_order_contracts(order: Any) -> float:
    value = _safe_float(getattr(order, "okx_fill_contracts", None), None)
    if value is not None and abs(value) > 0:
        return abs(value)
    raw = _dashboard_order_raw_fills(order)
    for key in ("contracts", "fillSz", "fill_size"):
        value = _safe_float(raw.get(key), None)
        if value is not None and abs(value) > 0:
            return abs(value)
    contract_size = _safe_float(raw.get("contract_size"), 0.0) or 0.0
    base_quantity = _dashboard_order_base_quantity(order)
    if contract_size > 0 and base_quantity > 0:
        return abs(base_quantity / contract_size)
    return base_quantity


def _dashboard_order_fee_abs(order: Any) -> float:
    raw = _dashboard_order_raw_fills(order)
    value = _safe_float(raw.get("fee_abs"), None)
    if value is None:
        value = _safe_float(getattr(order, "fee", None), 0.0) or 0.0
    return abs(value)


def _dashboard_order_fill_pnl(order: Any) -> float | None:
    value = _safe_float(getattr(order, "okx_fill_pnl", None), None)
    if value is not None:
        return value
    raw = _dashboard_order_raw_fills(order)
    for key in ("fill_pnl", "fillPnl", "pnl"):
        value = _safe_float(raw.get(key), None)
        if value is not None:
            return value
    return None


def _dashboard_order_price(order: Any) -> float:
    raw = _dashboard_order_raw_fills(order)
    value = _safe_float(raw.get("avg_price"), None)
    if value is None:
        value = _safe_float(raw.get("fillPx"), None)
    if value is None:
        value = _safe_float(getattr(order, "price", None), 0.0)
    return value or 0.0


def _dashboard_order_inst_id(order: Any) -> str:
    raw = _dashboard_order_raw_fills(order)
    return (
        str(getattr(order, "okx_inst_id", "") or "").strip().upper()
        or str(raw.get("inst_id") or raw.get("instId") or "").strip().upper()
        or okx_inst_id_from_symbol(getattr(order, "symbol", None))
        or ""
    )


def _dashboard_order_okx_confirmed(order: Any) -> bool:
    return str(getattr(order, "okx_sync_status", "") or "").strip() in (
        _DASHBOARD_OKX_CONFIRMED_ORDER_STATUSES
    )


def _dashboard_order_matches_position_history_window(
    order: Any,
    *,
    inst_id: str,
    opened_at: datetime | None,
    updated_at: datetime | None,
) -> bool:
    order_inst_id = _dashboard_order_inst_id(order)
    if inst_id and order_inst_id and order_inst_id != inst_id:
        return False
    if not _dashboard_order_okx_confirmed(order):
        return False
    order_time = _dashboard_order_time(order)
    if order_time is None:
        return False
    window = timedelta(seconds=_DASHBOARD_POSITION_HISTORY_ORDER_WINDOW_SECONDS)
    if opened_at and order_time < opened_at - window:
        return False
    if updated_at and order_time > updated_at + window:
        return False
    return True


def _dashboard_quantities_match(
    left: float,
    right: float,
    *,
    tolerance_ratio: float = 0.02,
) -> bool:
    if left <= 0 or right <= 0:
        return False
    denominator = max(abs(left), abs(right), 1e-12)
    return abs(left - right) / denominator <= tolerance_ratio


def _dashboard_best_quantity_subset(
    orders: list[Any],
    target: float,
    quantity_getter: Callable[[Any], float],
) -> tuple[list[Any], bool]:
    if target <= 0 or not orders:
        return (orders, False)
    ordered = sorted(
        [order for order in orders if quantity_getter(order) > 0],
        key=lambda order: _dashboard_order_time(order) or datetime.max.replace(tzinfo=UTC),
    )
    total = sum(quantity_getter(order) for order in ordered)
    if _dashboard_quantities_match(total, target):
        return (ordered, True)
    if len(ordered) <= 18:
        best_subset: list[Any] = []
        best_delta = float("inf")

        def visit(index: int, current: list[Any], quantity: float) -> None:
            nonlocal best_subset, best_delta
            if quantity > target * 1.03:
                return
            if index >= len(ordered):
                delta = abs(quantity - target)
                if delta < best_delta:
                    best_subset = list(current)
                    best_delta = delta
                return
            order = ordered[index]
            current.append(order)
            visit(index + 1, current, quantity + quantity_getter(order))
            current.pop()
            visit(index + 1, current, quantity)

        visit(0, [], 0.0)
        best_quantity = sum(quantity_getter(order) for order in best_subset)
        if _dashboard_quantities_match(best_quantity, target):
            return (
                sorted(
                    best_subset,
                    key=lambda order: _dashboard_order_time(order)
                    or datetime.max.replace(tzinfo=UTC),
                ),
                True,
            )
    selected: list[Any] = []
    quantity = 0.0
    for order in ordered:
        order_quantity = quantity_getter(order)
        if quantity + order_quantity > target * 1.03 and selected:
            break
        selected.append(order)
        quantity += order_quantity
        if quantity >= target * 0.98:
            break
    return (selected or ordered, _dashboard_quantities_match(quantity, target))


def _dashboard_select_orders_by_official_quantity(
    orders: list[Any],
    target: float,
) -> tuple[list[Any], str]:
    selected, matched = _dashboard_best_quantity_subset(
        orders,
        target,
        _dashboard_order_base_quantity,
    )
    if matched:
        return selected, "base_quantity"
    selected, matched = _dashboard_best_quantity_subset(
        orders,
        target,
        _dashboard_order_contracts,
    )
    if matched:
        return selected, "contracts"
    return selected, "unmatched"


def _dashboard_linked_fill_from_order(order: Any) -> dict[str, Any]:
    raw = _dashboard_order_raw_fills(order)
    order_time = _dashboard_order_time(order)
    return {
        "side": str(getattr(order, "side", "") or "").lower(),
        "quantity": _dashboard_order_base_quantity(order),
        "contracts": _dashboard_order_contracts(order),
        "contract_size": _safe_float(raw.get("contract_size"), 1.0) or 1.0,
        "price": _dashboard_order_price(order),
        "pnl": _dashboard_order_fill_pnl(order),
        "pnl_pct": None,
        "fee": _dashboard_order_fee_abs(order),
        "order_id": str(getattr(order, "exchange_order_id", "") or "").strip(),
        "trade_id": str(getattr(order, "okx_trade_ids", "") or "").strip(),
        "filled_at": order_time.isoformat() if order_time else None,
        "okx_confirmed": _dashboard_order_okx_confirmed(order),
        "source": "okx_order_fact_cache",
    }


def _dashboard_order_ids(orders: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for order in sorted(
        orders,
        key=lambda item: _dashboard_order_time(item) or datetime.max.replace(tzinfo=UTC),
    ):
        order_id = str(getattr(order, "exchange_order_id", "") or "").strip()
        if order_id and order_id not in seen:
            result.append(order_id)
            seen.add(order_id)
    return result


def _dashboard_position_history_matching_position_ids(
    row: dict[str, Any],
    closed_rows: list[Any],
    *,
    side: str,
) -> list[int]:
    inst_id = str(row.get("instId") or row.get("inst_id") or "").strip().upper()
    pos_id = str(row.get("posId") or row.get("pos_id") or "").strip()
    opened_at = _dashboard_ms_datetime(row.get("cTime") or row.get("createdTime"))
    updated_at = _dashboard_ms_datetime(row.get("uTime") or row.get("updatedTime"))
    result: list[int] = []
    window = timedelta(seconds=_DASHBOARD_POSITION_HISTORY_ORDER_WINDOW_SECONDS)
    for position in closed_rows:
        row_id = getattr(position, "id", None)
        if row_id is None:
            continue
        position_inst_id = (
            str(getattr(position, "okx_inst_id", "") or "").strip().upper()
            or okx_inst_id_from_symbol(getattr(position, "symbol", None))
            or ""
        )
        if inst_id and position_inst_id and position_inst_id != inst_id:
            continue
        position_side = str(getattr(position, "side", "") or "").lower().strip()
        if side and position_side and position_side != side:
            continue
        if pos_id and str(getattr(position, "okx_pos_id", "") or "").strip() == pos_id:
            result.append(int(row_id))
            continue
        position_opened = _as_utc_datetime(getattr(position, "created_at", None))
        position_closed = _as_utc_datetime(getattr(position, "closed_at", None))
        if opened_at and position_opened and abs(position_opened - opened_at) > window:
            continue
        if updated_at and position_closed and abs(position_closed - updated_at) > window:
            continue
        if position_opened or position_closed:
            result.append(int(row_id))
    return sorted(set(result))


def _dashboard_position_history_order_payload(
    row: dict[str, Any],
    *,
    side: str,
    order_rows: list[Any],
) -> dict[str, Any]:
    inst_id = str(row.get("instId") or row.get("inst_id") or "").strip().upper()
    opened_at = _dashboard_ms_datetime(row.get("cTime") or row.get("createdTime"))
    updated_at = _dashboard_ms_datetime(row.get("uTime") or row.get("updatedTime"))
    raw_close_quantity = _safe_float(row.get("closeTotalPos"), 0.0) or 0.0
    raw_max_quantity = _safe_float(row.get("openMaxPos"), 0.0) or 0.0
    entry_side = "sell" if side == "short" else "buy" if side == "long" else ""
    close_side = "buy" if side == "short" else "sell" if side == "long" else ""
    candidates = [
        order
        for order in order_rows
        if _dashboard_order_matches_position_history_window(
            order,
            inst_id=inst_id,
            opened_at=opened_at,
            updated_at=updated_at,
        )
    ]
    entry_candidates = [
        order
        for order in candidates
        if not entry_side or str(getattr(order, "side", "") or "").lower() == entry_side
    ]
    close_candidates = [
        order
        for order in candidates
        if (
            not close_side
            or str(getattr(order, "side", "") or "").lower() == close_side
        )
        and (_dashboard_order_fill_pnl(order) is not None or _dashboard_order_raw_fills(order))
    ]
    selected_closes, close_match_source = _dashboard_select_orders_by_official_quantity(
        close_candidates,
        raw_close_quantity,
    )
    entry_match_source = "unmatched"
    if raw_max_quantity > 0:
        selected_entries, entry_match_source = _dashboard_select_orders_by_official_quantity(
            entry_candidates,
            raw_max_quantity,
        )
    else:
        first_close_at = min(
            (_dashboard_order_time(order) for order in selected_closes if _dashboard_order_time(order)),
            default=None,
        )
        selected_entries = [
            order
            for order in entry_candidates
            if first_close_at is None
            or (
                (order_time := _dashboard_order_time(order)) is not None
                and order_time <= first_close_at
            )
        ]
    close_base_quantity = sum(_dashboard_order_base_quantity(order) for order in selected_closes)
    close_contracts = sum(_dashboard_order_contracts(order) for order in selected_closes)
    entry_base_quantity = sum(_dashboard_order_base_quantity(order) for order in selected_entries)
    entry_contracts = sum(_dashboard_order_contracts(order) for order in selected_entries)
    close_quantity = raw_close_quantity
    if close_base_quantity > 0 and _dashboard_quantities_match(close_base_quantity, raw_close_quantity):
        close_quantity = close_base_quantity
    elif close_contracts > 0 and _dashboard_quantities_match(close_contracts, raw_close_quantity):
        close_quantity = close_base_quantity or raw_close_quantity
    max_quantity = raw_max_quantity or entry_base_quantity or close_quantity
    if raw_max_quantity > 0:
        if entry_base_quantity > 0 and _dashboard_quantities_match(entry_base_quantity, raw_max_quantity):
            max_quantity = entry_base_quantity
        elif entry_contracts > 0 and _dashboard_quantities_match(entry_contracts, raw_max_quantity):
            max_quantity = entry_base_quantity or raw_max_quantity
    elif entry_base_quantity > 0:
        max_quantity = max(entry_base_quantity, close_quantity)
    linked_orders = sorted(
        [*selected_entries, *selected_closes],
        key=lambda order: _dashboard_order_time(order) or datetime.max.replace(tzinfo=UTC),
    )
    return {
        "entry_order_ids": _dashboard_order_ids(selected_entries),
        "close_order_ids": _dashboard_order_ids(selected_closes),
        "linked_fills": [_dashboard_linked_fill_from_order(order) for order in linked_orders],
        "close_quantity": close_quantity,
        "max_quantity": max_quantity,
        "entry_fee": sum(_dashboard_order_fee_abs(order) for order in selected_entries),
        "close_fee": sum(_dashboard_order_fee_abs(order) for order in selected_closes),
        "close_fill_pnl": sum(
            _dashboard_order_fill_pnl(order) or 0.0 for order in selected_closes
        ),
        "entry_match_source": entry_match_source,
        "close_match_source": close_match_source,
    }


def _dashboard_position_history_official_rows_as_groups_legacy(
    rows: list[dict[str, Any]],
    local_groups: list[Any],
    *,
    mode: str | None,
) -> list[dict[str, Any]]:
    official_groups: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        inst_id = str(row.get("instId") or row.get("inst_id") or "").strip().upper()
        symbol = symbol_from_okx_inst_id(inst_id) or normalize_trading_symbol(inst_id)
        if not symbol:
            continue
        matched = _dashboard_position_history_best_local_group(row, local_groups)
        payload = (
            matched.as_dict(include_fills=True)
            if matched is not None and hasattr(matched, "as_dict")
            else {}
        )
        status, status_label = _dashboard_position_history_close_status(row)
        if status == "full" and str(payload.get("close_status") or "") == "partial":
            status, status_label = "partial", "部分平仓"
        opened_at = _dashboard_ms_datetime(row.get("cTime") or row.get("createdTime"))
        updated_at = _dashboard_ms_datetime(row.get("uTime") or row.get("updatedTime"))
        closed_at = None if status == "partial" else updated_at
        close_quantity = _dashboard_position_history_quantity_in_base_units(
            row.get("closeTotalPos"),
            payload,
            order_id_key="close_order_ids",
            fallback=payload.get("closed_quantity") or payload.get("quantity"),
        )
        max_quantity = _dashboard_position_history_quantity_in_base_units(
            row.get("openMaxPos"),
            payload,
            order_id_key="entry_order_ids",
            fallback=payload.get("max_position_quantity") or close_quantity,
        )
        realized_pnl = _safe_float(row.get("realizedPnl"), 0.0) or 0.0
        pnl_ratio = _safe_float(row.get("pnlRatio"), None)
        if pnl_ratio is not None:
            pnl_ratio *= 100.0
        payload.update(
            {
                "id": _dashboard_position_history_group_id(row, mode),
                "group_id": _dashboard_position_history_group_id(row, mode),
                "is_open": False,
                "symbol": symbol,
                "okx_inst_id": inst_id,
                "okx_pos_id": str(row.get("posId") or row.get("pos_id") or "").strip(),
                "side": _dashboard_position_history_side(row)
                or str(payload.get("side") or "").lower(),
                "leverage": _safe_float(row.get("lever"), payload.get("leverage", 1.0)) or 1.0,
                "position_status": status_label,
                "close_status": status,
                "close_status_label": status_label,
                "quantity": close_quantity,
                "max_position_quantity": max_quantity,
                "closed_quantity": close_quantity,
                "entry_price": _safe_float(row.get("openAvgPx"), payload.get("entry_price", 0.0))
                or 0.0,
                "current_price": _safe_float(row.get("closeAvgPx"), payload.get("current_price", 0.0))
                or 0.0,
                "average_entry_price": _safe_float(
                    row.get("openAvgPx"),
                    payload.get("average_entry_price", 0.0),
                )
                or 0.0,
                "average_close_price": _safe_float(
                    row.get("closeAvgPx"),
                    payload.get("average_close_price", 0.0),
                )
                or 0.0,
                "realized_pnl": realized_pnl,
                "realized_pnl_pct": pnl_ratio,
                "pnl_source": "okx_position_history_realized_pnl",
                "funding_fee": _safe_float(row.get("fundingFee"), payload.get("funding_fee", 0.0))
                or 0.0,
                "settlement_source": "okx_position_history_realized_pnl",
                "settlement_status": "okx_position_history",
                "opened_at": opened_at.isoformat() if opened_at else payload.get("opened_at"),
                "closed_at": closed_at.isoformat() if closed_at else None,
                "official_updated_at": updated_at.isoformat() if updated_at else None,
                "ledger_source": "okx_positions_history_official",
                "okx_position_history_row": dict(row),
            }
        )
        if "linked_fills" not in payload:
            payload["linked_fills"] = []
        payload["linked_order_count"] = len(payload.get("linked_fills") or [])
        payload.setdefault("entry_order_ids", [])
        payload.setdefault("close_order_ids", [])
        payload.setdefault("evidence_complete", True)
        payload.setdefault("trainable", True)
        payload.setdefault("evidence_gaps", [])
        official_groups.append(payload)
    return sorted(
        official_groups,
        key=lambda item: (
            _as_utc_datetime(item.get("official_updated_at"))
            or _as_utc_datetime(item.get("closed_at"))
            or _as_utc_datetime(item.get("opened_at"))
            or datetime.min.replace(tzinfo=UTC)
        ),
        reverse=True,
    )


def _dashboard_position_history_official_rows_as_groups(
    rows: list[dict[str, Any]],
    local_groups: list[Any],
    *,
    mode: str | None,
    order_rows: list[Any] | None = None,
    closed_rows: list[Any] | None = None,
) -> list[dict[str, Any]]:
    official_groups: list[dict[str, Any]] = []
    order_rows = list(order_rows or [])
    closed_rows = list(closed_rows or [])
    for row in rows:
        if not isinstance(row, dict):
            continue
        inst_id = str(row.get("instId") or row.get("inst_id") or "").strip().upper()
        symbol = symbol_from_okx_inst_id(inst_id) or normalize_trading_symbol(inst_id)
        if not symbol:
            continue
        side = _dashboard_position_history_local_side(row, local_groups, closed_rows)
        order_payload = _dashboard_position_history_order_payload(
            row,
            side=side,
            order_rows=order_rows,
        )
        status, status_label = _dashboard_position_history_close_status(row)
        opened_at = _dashboard_ms_datetime(row.get("cTime") or row.get("createdTime"))
        updated_at = _dashboard_ms_datetime(row.get("uTime") or row.get("updatedTime"))
        close_quantity = order_payload["close_quantity"]
        max_quantity = order_payload["max_quantity"]
        if max_quantity > 0 and close_quantity > 0 and close_quantity < max_quantity * 0.999:
            status = "partial"
            status_label = _dashboard_position_history_status_label(status)
        closed_at = None if status == "partial" else updated_at
        realized_pnl = _safe_float(row.get("realizedPnl"), 0.0) or 0.0
        pnl_ratio = _safe_float(row.get("pnlRatio"), None)
        if pnl_ratio is not None:
            pnl_ratio *= 100.0
        row_pnl = _safe_float(row.get("pnl"), None)
        close_fill_pnl = (
            row_pnl
            if row_pnl is not None
            else _safe_float(order_payload.get("close_fill_pnl"), 0.0) or 0.0
        )
        evidence_gaps: list[str] = []
        if not order_payload["entry_order_ids"]:
            evidence_gaps.append("missing_position_history_entry_orders")
        if not order_payload["close_order_ids"]:
            evidence_gaps.append("missing_position_history_close_orders")
        if order_payload["close_match_source"] == "unmatched":
            evidence_gaps.append("position_history_close_quantity_not_matched_to_orders")
        official_groups.append(
            {
                "id": _dashboard_position_history_group_id(row, mode),
                "group_id": _dashboard_position_history_group_id(row, mode),
                "is_open": False,
                "symbol": symbol,
                "okx_inst_id": inst_id,
                "okx_pos_id": str(row.get("posId") or row.get("pos_id") or "").strip(),
                "side": side,
                "leverage": _safe_float(row.get("lever"), 1.0) or 1.0,
                "position_status": status_label,
                "close_status": status,
                "close_status_label": status_label,
                "quantity": close_quantity,
                "max_position_quantity": max_quantity,
                "closed_quantity": close_quantity,
                "entry_price": _safe_float(row.get("openAvgPx"), 0.0) or 0.0,
                "current_price": _safe_float(row.get("closeAvgPx"), 0.0) or 0.0,
                "average_entry_price": _safe_float(row.get("openAvgPx"), 0.0) or 0.0,
                "average_close_price": _safe_float(row.get("closeAvgPx"), 0.0) or 0.0,
                "realized_pnl": realized_pnl,
                "realized_pnl_pct": pnl_ratio,
                "pnl_source": "okx_position_history_realized_pnl",
                "close_fill_pnl": close_fill_pnl,
                "entry_fee": order_payload["entry_fee"],
                "close_fee": order_payload["close_fee"],
                "funding_fee": _safe_float(row.get("fundingFee"), 0.0) or 0.0,
                "funding_bill_count": 0,
                "funding_fee_source": "okx_positions_history.fundingFee",
                "settlement_source": "okx_position_history_realized_pnl",
                "settlement_status": "okx_position_history",
                "realized_pnl_formula": "okx_position_history_realized_pnl_authoritative",
                "opened_at": opened_at.isoformat() if opened_at else None,
                "closed_at": closed_at.isoformat() if closed_at else None,
                "official_updated_at": updated_at.isoformat() if updated_at else None,
                "position_ids": _dashboard_position_history_matching_position_ids(
                    row,
                    closed_rows,
                    side=side,
                ),
                "entry_order_ids": order_payload["entry_order_ids"],
                "close_order_ids": order_payload["close_order_ids"],
                "linked_fills": order_payload["linked_fills"],
                "linked_order_count": len(order_payload["linked_fills"]),
                "evidence_complete": not evidence_gaps,
                "trainable": not evidence_gaps,
                "evidence_gaps": evidence_gaps,
                "ledger_source": "okx_positions_history_official",
                "okx_position_history_row": dict(row),
                "position_history_entry_match_source": order_payload["entry_match_source"],
                "position_history_close_match_source": order_payload["close_match_source"],
            }
        )
    return sorted(
        official_groups,
        key=lambda item: (
            _as_utc_datetime(item.get("official_updated_at"))
            or _as_utc_datetime(item.get("closed_at"))
            or _as_utc_datetime(item.get("opened_at"))
            or datetime.min.replace(tzinfo=UTC)
        ),
        reverse=True,
    )


async def _dashboard_okx_position_history_rows(
    *,
    mode: str | None,
    closed_rows: list[Any],
) -> list[dict[str, Any]]:
    selected_mode = "live" if mode == "live" else "paper"
    pos_ids = sorted(
        {
            str(getattr(row, "okx_pos_id", "") or "").strip()
            for row in closed_rows
            if str(getattr(row, "okx_pos_id", "") or "").strip()
        }
    )
    inst_ids = sorted(
        {
            (
                str(getattr(row, "okx_inst_id", "") or "").strip().upper()
                or okx_inst_id_from_symbol(getattr(row, "symbol", None))
                or ""
            )
            for row in closed_rows
            if (
                str(getattr(row, "okx_inst_id", "") or "").strip()
                or okx_inst_id_from_symbol(getattr(row, "symbol", None))
            )
        }
    )
    opened_values = [
        value
        for row in closed_rows
        if (value := _as_utc_datetime(getattr(row, "created_at", None))) is not None
    ]
    closed_values = [
        value
        for row in closed_rows
        if (value := _as_utc_datetime(getattr(row, "closed_at", None))) is not None
    ]
    reference_values = [*opened_values, *closed_values]
    since = (
        min(reference_values) - timedelta(days=1)
        if reference_values
        else datetime.now(UTC) - timedelta(days=14)
    )
    cache_key = (
        "okx_position_history_rows",
        selected_mode,
        tuple(pos_ids[:500]),
        tuple(inst_ids[:200]),
        since.date().isoformat(),
    )

    async def builder() -> list[dict[str, Any]]:
        from services.okx_native_facts import OkxNativeFactsClient

        executor = _dashboard_okx_executor_for_mode(selected_mode)
        owns_executor = executor is None
        if owns_executor:
            executor = _make_lightweight_okx_executor(OKXExecutor, selected_mode)
        try:
            if owns_executor:
                await executor.initialize()
            page_count = 5 if not pos_ids else 2
            rows = await asyncio.wait_for(
                OkxNativeFactsClient(executor).fetch_position_history_rows(
                    inst_ids=inst_ids or None,
                    pos_ids=pos_ids or None,
                    since=since,
                    strict=False,
                    limit=100,
                    max_pages=page_count,
                ),
                timeout=12.0,
            )
            wanted_pos_ids = set(pos_ids)
            wanted_inst_ids = set(inst_ids)
            return [
                dict(row)
                for row in rows
                if isinstance(row, dict)
                and (
                    not wanted_pos_ids
                    or str(row.get("posId") or row.get("pos_id") or "").strip()
                    in wanted_pos_ids
                )
                and (
                    not wanted_inst_ids
                    or str(row.get("instId") or row.get("inst_id") or "").strip().upper()
                    in wanted_inst_ids
                )
            ]
        except Exception as exc:
            _log_dashboard_fallback(
                "dashboard OKX position history official rows unavailable",
                exc,
                mode=selected_mode,
                inst_id_count=len(inst_ids),
                pos_id_count=len(pos_ids),
            )
            return []
        finally:
            if owns_executor and executor is not None:
                try:
                    await executor.shutdown()
                except Exception as exc:
                    _log_dashboard_fallback(
                        "dashboard OKX position history executor shutdown failed",
                        exc,
                        mode=selected_mode,
                    )

    return await _dashboard_heavy_cached(cache_key, builder, ttl_seconds=60.0)


def _is_live_position_open(position: dict) -> bool:
    raw_size = (
        position.get("contracts")
        or position.get("size")
        or position.get("positionAmt")
        or (position.get("info") or {}).get("pos")
        or (position.get("info") or {}).get("qty")
        or 0
    )
    try:
        return abs(float(raw_size)) > 0
    except (TypeError, ValueError):
        return bool(position.get("symbol"))


async def _get_open_position_symbols(mode: str | None = None) -> set[str]:
    """Return symbols that currently have open positions for the selected mode."""
    from sqlalchemy import select

    from db.session import get_session_ctx
    from models.trade import Position

    selected_mode = mode or mode_manager.mode.value
    symbols: set[str] = set()

    try:
        async with get_session_ctx() as session:
            result = await session.execute(
                select(Position.symbol)
                .where(
                    Position.execution_mode == selected_mode,
                    Position.model_name.in_(EXECUTION_LEDGER_MODEL_NAMES),
                    Position.is_open.is_(True),
                )
                .distinct()
            )
            symbols.update(_normalize_dashboard_symbol(row[0]) for row in result.all())
    except Exception as exc:
        _log_dashboard_fallback(
            "open position db symbol fallback",
            exc,
            mode=selected_mode,
        )

    if _dashboard_okx_executor_for_mode(selected_mode):
        exchange_symbols = await _get_exchange_open_position_symbols(selected_mode)
        if exchange_symbols:
            return {s for s in exchange_symbols if s}

    return {s for s in symbols if s}


async def _get_exchange_open_position_symbols(mode: str | None = None) -> set[str] | None:
    """Return actual exchange open-position symbols, or None when unavailable."""
    selected_mode = mode or mode_manager.mode.value
    now = datetime.now(UTC)
    cached = _exchange_open_symbol_cache.get(selected_mode)
    if cached:
        cached_at, cached_value = cached
        if (now - cached_at).total_seconds() <= _EXCHANGE_OPEN_SYMBOL_CACHE_TTL_SECONDS:
            return set(cached_value)

    try:
        positions = await _fetch_dashboard_okx_positions(selected_mode)
    except Exception as exc:
        _log_dashboard_fallback(
            "exchange open position symbols strict read unavailable",
            exc,
            mode=selected_mode,
            has_cached=bool(cached),
        )
        return set(cached[1]) if cached else None

    symbols: set[str] = set()
    for position in positions or []:
        if _is_live_position_open(position):
            symbols.add(_normalize_dashboard_symbol(position.get("symbol")))
    result = {s for s in symbols if s}
    _exchange_open_symbol_cache[selected_mode] = (now, result)
    return set(result)


async def _get_exchange_position_mark_map(
    mode: str | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Return exchange position mark-price snapshots keyed by (symbol, side)."""
    selected_mode = mode or mode_manager.mode.value
    now = datetime.now(UTC)
    cached = _exchange_mark_cache.get(selected_mode)
    if cached:
        cached_at, cached_value = cached
        if (now - cached_at).total_seconds() <= _EXCHANGE_MARK_CACHE_TTL_SECONDS:
            return cached_value

    try:
        positions = await _fetch_dashboard_okx_positions(selected_mode)
    except Exception as exc:
        _log_dashboard_fallback(
            "exchange mark map strict read unavailable",
            exc,
            mode=selected_mode,
            has_cached=bool(cached),
        )
        return cached[1] if cached else {}

    snapshots: dict[tuple[str, str], dict[str, Any]] = {}
    for position in positions or []:
        snapshot = parse_exchange_position_snapshot(
            position,
            symbol_normalizer=_normalize_dashboard_symbol,
        )
        if not snapshot:
            continue
        symbol = str(snapshot["symbol"])
        side = str(snapshot["side"])
        snapshots[(symbol, side)] = dict(snapshot)
    _exchange_mark_cache[selected_mode] = (now, snapshots)
    return snapshots


async def _get_dashboard_okx_account_snapshot(selected_mode: str) -> dict[str, Any] | None:
    """Fetch the OKX balance snapshot used by dashboard summary with bounded retries."""
    selected_mode = "live" if selected_mode == "live" else "paper"
    initial_executor = _dashboard_okx_executor_for_mode(selected_mode)
    executor_identity = initial_executor

    def cached_success(now: datetime, *, fresh_only: bool) -> dict[str, Any] | None:
        cached = _dashboard_okx_balance_cache.get(selected_mode)
        if not cached:
            return None
        cached_at, cached_value = cached
        age_seconds = (now - cached_at).total_seconds()
        if fresh_only and age_seconds > _DASHBOARD_OKX_BALANCE_CACHE_TTL_SECONDS:
            return None
        if not fresh_only and age_seconds > _DASHBOARD_OKX_BALANCE_STALE_CACHE_TTL_SECONDS:
            return None
        snapshot = copy.deepcopy(cached_value)
        if not fresh_only:
            snapshot["stale"] = True
            snapshot["stale_age_seconds"] = round(age_seconds, 3)
        return snapshot

    def cached_failure(now: datetime) -> dict[str, Any] | None:
        cached_error = _dashboard_okx_balance_error_cache.get(selected_mode)
        if not cached_error:
            return None
        cached_at, cached_value, cached_executor_identity = cached_error
        if (
            cached_executor_identity is not executor_identity
            or (now - cached_at).total_seconds()
            > _DASHBOARD_OKX_BALANCE_ERROR_CACHE_TTL_SECONDS
        ):
            return None
        stale_snapshot = cached_success(now, fresh_only=False)
        if stale_snapshot:
            stale_snapshot["error"] = cached_value.get("error")
            stale_snapshot["balance_error"] = cached_value.get("balance_error")
            stale_snapshot["error_cached"] = True
            return stale_snapshot
        return copy.deepcopy(cached_value)

    now = datetime.now(UTC)
    fresh_cached = cached_success(now, fresh_only=True)
    if fresh_cached:
        return fresh_cached
    error_cached = cached_failure(now)
    if error_cached:
        return error_cached
    shared_snapshot = _trading_service_cached_okx_balance_snapshot(selected_mode)
    if shared_snapshot:
        if not (
            shared_snapshot.get("stale")
            or shared_snapshot.get("error")
            or shared_snapshot.get("balance_error")
        ):
            _dashboard_okx_balance_cache[selected_mode] = (
                datetime.now(UTC),
                copy.deepcopy(shared_snapshot),
            )
            _dashboard_okx_balance_error_cache.pop(selected_mode, None)
        return shared_snapshot
    stale_cached = cached_success(datetime.now(UTC), fresh_only=False)
    if stale_cached:
        stale_cached["refresh_in_progress"] = True
        _start_dashboard_okx_balance_refresh(selected_mode)
        return stale_cached

    def cache_failure(
        exc: Exception,
        *,
        source: str,
        identity: Any | None = executor_identity,
    ) -> dict[str, Any]:
        error = _dashboard_okx_error_text(exc, resource="余额")
        failure = {
            "error": error,
            "balance_error": error,
            "balance_source": _dashboard_okx_account_label(selected_mode),
            "source": source,
            "error_cached": True,
        }
        _dashboard_okx_balance_error_cache[selected_mode] = (
            datetime.now(UTC),
            failure,
            identity,
        )
        return copy.deepcopy(failure)

    lock = _dashboard_okx_balance_locks.setdefault(selected_mode, asyncio.Lock())
    if lock.locked():
        stale_cached = cached_success(now, fresh_only=False)
        if stale_cached:
            stale_cached["stale"] = True
            stale_cached["refresh_in_progress"] = True
            return stale_cached
        return _dashboard_okx_balance_refreshing_snapshot(selected_mode)
    async with lock:
        now = datetime.now(UTC)
        fresh_cached = cached_success(now, fresh_only=True)
        if fresh_cached:
            return fresh_cached
        error_cached = cached_failure(now)
        if error_cached:
            return error_cached

        shared_snapshot = _trading_service_cached_okx_balance_snapshot(selected_mode)
        if shared_snapshot:
            if not (
                shared_snapshot.get("stale")
                or shared_snapshot.get("error")
                or shared_snapshot.get("balance_error")
            ):
                _dashboard_okx_balance_cache[selected_mode] = (
                    datetime.now(UTC),
                    copy.deepcopy(shared_snapshot),
                )
                _dashboard_okx_balance_error_cache.pop(selected_mode, None)
            return shared_snapshot
        stale_cached = cached_success(datetime.now(UTC), fresh_only=False)
        if stale_cached:
            stale_cached["refresh_in_progress"] = True
            _start_dashboard_okx_balance_refresh(selected_mode)
            return stale_cached
        _start_dashboard_okx_balance_refresh(selected_mode)
        return _dashboard_okx_balance_refreshing_snapshot(selected_mode)

        try:
            snapshot = await _fetch_dashboard_okx_balance_uncached(selected_mode)
            _dashboard_okx_balance_cache[selected_mode] = (
                datetime.now(UTC),
                copy.deepcopy(snapshot),
            )
            _dashboard_okx_balance_error_cache.pop(selected_mode, None)
            return snapshot
        except Exception as exc:
            _log_dashboard_fallback(
                "dashboard summary okx balance fallback",
                exc,
                mode=selected_mode,
                source="isolated_executor",
            )
            stale_snapshot = cached_success(datetime.now(UTC), fresh_only=False)
            if stale_snapshot:
                error = _dashboard_okx_error_text(exc, resource="余额")
                stale_snapshot["error"] = error
                stale_snapshot["balance_error"] = error
                stale_snapshot["error_cached"] = True
                _dashboard_okx_balance_error_cache[selected_mode] = (
                    datetime.now(UTC),
                    {
                        "error": error,
                        "balance_error": error,
                        "balance_source": _dashboard_okx_account_label(selected_mode),
                        "source": "isolated_executor",
                        "error_cached": True,
                    },
                        executor_identity,
                    )
                return stale_snapshot
            return cache_failure(exc, source="isolated_executor", identity=executor_identity)


async def _get_display_open_position_symbols(mode: str | None = None) -> set[str]:
    """Return OKX symbols that should be shown as currently held on the dashboard."""
    exchange_symbols = await _get_exchange_open_position_symbols(mode)
    if exchange_symbols is None:
        if _dashboard_okx_positions_temporarily_unavailable(mode):
            return await _get_open_position_symbols(mode)
        return set()
    return {symbol for symbol in exchange_symbols if symbol}


async def _get_open_position_prices(mode: str | None = None) -> dict[str, float]:
    """Return last known prices from open persisted positions."""
    from db.repositories.trade_repo import TradeRepository
    from db.session import get_session_ctx

    selected_mode = mode or mode_manager.mode.value
    prices: dict[str, float] = {}

    try:
        async with get_session_ctx() as session:
            repo = TradeRepository(session)
            rows = await repo.get_position_records(execution_mode=selected_mode, limit=1000)
            for p in rows:
                if not p.is_open:
                    continue
                symbol = _normalize_dashboard_symbol(p.symbol)
                price = float(p.current_price or p.entry_price or 0)
                if symbol and price > 0:
                    prices[symbol] = price
    except Exception as exc:
        _log_dashboard_fallback(
            "open position price fallback",
            exc,
            mode=selected_mode,
        )

    return prices


async def _build_tickers_for_open_positions(
    open_symbols: set[str],
    market_tickers: dict,
    mode: str | None = None,
) -> dict:
    tickers: dict = {}
    public_tickers = await _get_public_ticker_map(open_symbols)

    # For currently held positions, prefer the same account-side OKX price
    # source used by simulated/live execution. Demo-account prices can differ
    # from OKX production public tickers, so this keeps the dashboard aligned
    # with the user's OKX position page.
    exchange_mark_map = await _get_exchange_position_mark_map(mode)
    for symbol in open_symbols:
        snapshots = [
            data
            for (snap_symbol, _side), data in exchange_mark_map.items()
            if snap_symbol == symbol
        ]
        if not snapshots:
            continue
        snapshot = snapshots[0]
        price = _exchange_snapshot_price(snapshot)
        if price > 0:
            market_ticker = dict(market_tickers.get(symbol, {}) or {})
            public_ticker = dict(public_tickers.get(symbol, {}) or {})
            if not market_ticker and public_ticker:
                market_ticker = public_ticker
            market_change = _ticker_change_24h(market_ticker, None)
            public_change = _ticker_change_24h(public_ticker, None)
            change_24h = (
                market_change
                if market_change is not None and abs(market_change) > 1e-12
                else public_change if public_change is not None else market_change or 0.0
            )
            tickers[symbol] = {
                "price": price,
                "mark_price": snapshot.get("mark_price"),
                "index_price": snapshot.get("index_price"),
                "change_24h": change_24h,
                "volume_24h": _safe_float(market_ticker.get("volume_24h"), 0.0)
                or _safe_float(public_ticker.get("volume_24h"), 0.0),
                "bid": _safe_float(market_ticker.get("bid"), 0.0)
                or _safe_float(public_ticker.get("bid"), 0.0),
                "ask": _safe_float(market_ticker.get("ask"), 0.0)
                or _safe_float(public_ticker.get("ask"), 0.0),
            }

    for symbol in open_symbols - set(tickers):
        if symbol in market_tickers or symbol in public_tickers:
            tickers[symbol] = _merge_market_and_public_ticker(
                dict(market_tickers.get(symbol, {}) or {}),
                dict(public_tickers.get(symbol, {}) or {}),
            )

    if open_symbols - set(tickers):
        position_prices = await _get_open_position_prices(mode)
        for symbol in open_symbols - set(tickers):
            price = position_prices.get(symbol, 0)
            if price > 0:
                market_ticker = dict(market_tickers.get(symbol, {}) or {})
                public_ticker = dict(public_tickers.get(symbol, {}) or {})
                market_change = _ticker_change_24h(market_ticker, None)
                public_change = _ticker_change_24h(public_ticker, None)
                change_24h = (
                    market_change
                    if market_change is not None and abs(market_change) > 1e-12
                    else public_change if public_change is not None else market_change or 0.0
                )
                tickers[symbol] = {
                    "price": price,
                    "change_24h": change_24h,
                    "volume_24h": _safe_float(market_ticker.get("volume_24h"), 0.0)
                    or _safe_float(public_ticker.get("volume_24h"), 0.0),
                    "bid": _safe_float(market_ticker.get("bid"), 0.0)
                    or _safe_float(public_ticker.get("bid"), 0.0),
                    "ask": _safe_float(market_ticker.get("ask"), 0.0)
                    or _safe_float(public_ticker.get("ask"), 0.0),
                }

    return tickers


async def _build_dashboard_tickers(
    open_symbols: set[str],
    market_tickers: dict,
    mode: str | None = None,
) -> dict:
    open_position_tickers = await _build_tickers_for_open_positions(
        open_symbols,
        market_tickers,
        mode,
    )
    if open_position_tickers:
        return open_position_tickers
    return {}


async def _build_open_position_market_snapshot(mode: str | None = None) -> dict[str, Any]:
    """Build the single dashboard contract for current open-position market data.

    The dashboard must not infer current position tickers from account cards.
    Account cards summarize balances and counts; this payload is the canonical
    source for the real-time held-symbol list and prices.
    """
    selected_mode = mode or mode_manager.mode.value
    market_state: dict[str, Any] = {}
    if _data_service:
        market_state = _data_service.get_market_state() or {}
    market_tickers = market_state.get("tickers", {}) if isinstance(market_state, dict) else {}
    open_symbols = await _get_display_open_position_symbols(selected_mode)
    tickers = await _build_dashboard_tickers(open_symbols, market_tickers, selected_mode)
    open_positions = await _get_display_open_positions_snapshot(
        selected_mode,
        ticker_overrides=tickers,
    )
    if open_positions:
        position_tickers = _build_tickers_from_position_snapshot(
            open_positions,
            tickers,
        )
        if position_tickers:
            tickers = {**tickers, **position_tickers}
    return {
        **(market_state if isinstance(market_state, dict) else {}),
        "tickers": tickers,
        "position_symbols": sorted(open_symbols),
        "open_positions": open_positions,
        "open_position_count": len(open_positions),
        "ws_stats": (market_state.get("ws_stats", {}) if isinstance(market_state, dict) else {}),
    }


async def _get_display_open_positions_snapshot(
    mode: str | None = None,
    ticker_overrides: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return display-ready open positions without pagination for dashboard widgets."""
    from db.repositories.trade_repo import TradeRepository
    from db.session import get_session_ctx

    selected_mode = mode or mode_manager.mode.value
    exchange_mark_map = await _get_exchange_position_mark_map(selected_mode)
    exchange_temporarily_unavailable = bool(
        not exchange_mark_map and _dashboard_okx_positions_temporarily_unavailable(selected_mode)
    )
    exchange_symbols = (
        {symbol for symbol, _side in exchange_mark_map.keys()} if exchange_mark_map else None
    )
    market_tickers: dict[str, dict[str, Any]] = {}
    if _data_service:
        market_tickers = (_data_service.get_market_state() or {}).get("tickers", {}) or {}
    local_by_key: dict[tuple[str, str], Any] = {}
    positions: list[dict[str, Any]] = []
    try:
        async with get_session_ctx() as session:
            repo = TradeRepository(session)
            rows = await repo.get_position_records(
                execution_mode=selected_mode,
                limit=5000,
                offset=0,
                is_open=True,
            )
            rows_by_key: dict[tuple[str, str], list[Any]] = {}
            for row in rows:
                symbol = _normalize_dashboard_symbol(row.symbol)
                side = str(row.side or "").lower()
                if not symbol or not side:
                    continue
                exchange_key = (symbol, side)
                rows_by_key.setdefault(exchange_key, []).append(row)
                local_by_key.setdefault(exchange_key, row)

            for exchange_key, group_rows in rows_by_key.items():
                symbol, side = exchange_key
                p = group_rows[0]
                exchange_synced = bool(
                    exchange_symbols is not None and exchange_key in exchange_mark_map
                )
                if not exchange_synced and not exchange_temporarily_unavailable:
                    continue

                local_quantity = sum(
                    _safe_float(getattr(row, "quantity", None), 0.0) or 0.0 for row in group_rows
                )
                local_entry_price = _weighted_average(
                    [
                        (
                            _safe_float(getattr(row, "quantity", None), 0.0) or 0.0,
                            _safe_float(getattr(row, "entry_price", None), 0.0) or 0.0,
                        )
                        for row in group_rows
                    ]
                )
                local_unrealized_pnl = sum(
                    _safe_float(getattr(row, "unrealized_pnl", None), 0.0) or 0.0
                    for row in group_rows
                )
                current_price = p.current_price
                unrealized_pnl = local_unrealized_pnl
                entry_price = local_entry_price or p.entry_price
                quantity = local_quantity or p.quantity
                pnl_source = "local_db"
                snapshot = exchange_mark_map.get(exchange_key)
                market_ticker = dict(market_tickers.get(symbol, {}) or {})
                override_ticker = (
                    dict(ticker_overrides.get(symbol, {}) or {})
                    if isinstance(ticker_overrides, dict)
                    else {}
                )
                display_ticker = _merge_market_and_public_ticker(
                    market_ticker,
                    override_ticker,
                )
                change_24h = _ticker_change_24h(display_ticker, 0.0) or 0.0
                if snapshot:
                    valuation = _exchange_position_display_valuation(
                        snapshot,
                        side,
                        fallback_current_price=current_price,
                        fallback_unrealized_pnl=unrealized_pnl,
                        fallback_entry_price=entry_price,
                        fallback_quantity=quantity,
                    )
                    current_price = valuation["current_price"]
                    unrealized_pnl = valuation["unrealized_pnl"]
                    entry_price = valuation["entry_price"]
                    quantity = valuation["quantity"]
                    pnl_source = valuation["pnl_source"]

                positions.append(
                    {
                        "id": p.id,
                        "model_name": p.model_name,
                        "mode": p.execution_mode,
                        "symbol": symbol or p.symbol,
                        "side": side,
                        "quantity": quantity,
                        "entry_price": entry_price,
                        "current_price": current_price,
                        "change_24h": change_24h,
                        "unrealized_pnl": unrealized_pnl,
                        "pnl_source": pnl_source,
                        "local_quantity": local_quantity,
                        "local_entry_price": local_entry_price or p.entry_price,
                        "local_unrealized_pnl": local_unrealized_pnl,
                        "realized_pnl": sum(
                            _safe_float(getattr(row, "realized_pnl", None), 0.0) or 0.0
                            for row in group_rows
                        ),
                        "leverage": p.leverage,
                        "stop_loss": p.stop_loss_price,
                        "take_profit": p.take_profit_price,
                        "is_open": True,
                        "db_is_open": p.is_open,
                        "local_position_ids": [
                            getattr(row, "id", None)
                            for row in group_rows
                            if getattr(row, "id", None) is not None
                        ],
                        "merged_local_position_count": len(group_rows),
                        "exchange_synced": exchange_synced,
                        "exchange_temporarily_unavailable": bool(
                            exchange_temporarily_unavailable and not exchange_synced
                        ),
                        "close_status": "open",
                        "close_status_label": "持有中",
                        "close_status_source": "position",
                        "position_status": "持有中",
                        "opened_at": p.created_at.isoformat() if p.created_at else None,
                        "closed_at": p.closed_at.isoformat() if p.closed_at else None,
                    }
                )
    except Exception as exc:
        _log_dashboard_fallback(
            "open position snapshot fallback",
            exc,
            mode=selected_mode,
        )

    if exchange_mark_map:
        seen_keys = {
            (
                _normalize_dashboard_symbol(str(position.get("symbol") or "")),
                str(position.get("side") or "").lower(),
            )
            for position in positions
            if isinstance(position, dict)
        }
        for (symbol, side), snapshot in sorted(exchange_mark_map.items()):
            if (symbol, side) in seen_keys:
                continue
            local_position = local_by_key.get((symbol, side))
            market_ticker = dict(market_tickers.get(symbol, {}) or {})
            override_ticker = (
                dict(ticker_overrides.get(symbol, {}) or {})
                if isinstance(ticker_overrides, dict)
                else {}
            )
            display_ticker = _merge_market_and_public_ticker(
                market_ticker,
                override_ticker,
            )
            change_24h = _ticker_change_24h(display_ticker, 0.0) or 0.0
            valuation = _exchange_position_display_valuation(
                snapshot,
                side,
                fallback_current_price=getattr(local_position, "current_price", None),
                fallback_unrealized_pnl=getattr(local_position, "unrealized_pnl", None),
                fallback_entry_price=getattr(local_position, "entry_price", None),
                fallback_quantity=getattr(local_position, "quantity", None),
            )
            info = snapshot.get("info") if isinstance(snapshot.get("info"), dict) else {}
            opened_at = getattr(local_position, "created_at", None)
            positions.append(
                {
                    "id": getattr(local_position, "id", None),
                    "model_name": getattr(local_position, "model_name", None)
                    or "okx_native_position",
                    "mode": selected_mode,
                    "symbol": symbol,
                    "side": side,
                    "quantity": valuation["quantity"],
                    "contracts": snapshot.get("contracts"),
                    "contract_size": snapshot.get("contract_size"),
                    "entry_price": valuation["entry_price"],
                    "current_price": valuation["current_price"],
                    "change_24h": change_24h,
                    "unrealized_pnl": valuation["unrealized_pnl"],
                    "pnl_source": valuation["pnl_source"],
                    "local_quantity": getattr(local_position, "quantity", None),
                    "local_entry_price": getattr(local_position, "entry_price", None),
                    "local_unrealized_pnl": getattr(local_position, "unrealized_pnl", None),
                    "realized_pnl": (
                        getattr(local_position, "realized_pnl", 0.0)
                        if local_position is not None
                        else 0.0
                    ),
                    "leverage": getattr(local_position, "leverage", None)
                    or snapshot.get("leverage"),
                    "stop_loss": getattr(local_position, "stop_loss_price", None),
                    "take_profit": getattr(local_position, "take_profit_price", None),
                    "is_open": True,
                    "db_is_open": (
                        bool(getattr(local_position, "is_open", False))
                        if local_position is not None
                        else False
                    ),
                    "exchange_synced": True,
                    "exchange_temporarily_unavailable": False,
                    "close_status": "open",
                    "close_status_label": "持有中",
                    "close_status_source": "okx_current_position",
                    "position_status": "持有中",
                    "okx_inst_id": info.get("instId") or snapshot.get("okx_inst_id"),
                    "okx_pos_id": info.get("posId") or snapshot.get("okx_pos_id"),
                    "opened_at": opened_at.isoformat() if opened_at else None,
                    "closed_at": None,
                }
            )

    return _group_open_dashboard_positions(
        positions,
        exchange_mark_map,
        mode=selected_mode,
    )


def _build_tickers_from_position_snapshot(
    open_positions: list[dict[str, Any]],
    existing_tickers: dict[str, Any] | None = None,
) -> dict[str, dict[str, float]]:
    """Build ticker cards directly from open-position snapshots."""
    existing_tickers = existing_tickers or {}
    tickers: dict[str, dict[str, float]] = {}
    for position in open_positions:
        if not isinstance(position, dict):
            continue
        if position.get("is_open") is False:
            continue
        symbol = _normalize_dashboard_symbol(str(position.get("symbol") or ""))
        if not symbol:
            continue
        price = (
            _safe_float(position.get("current_price"), 0.0)
            or _safe_float(position.get("exchange_mark_price"), 0.0)
            or _safe_float(position.get("entry_price"), 0.0)
            or 0.0
        )
        if price <= 0:
            continue
        previous = existing_tickers.get(symbol, {}) if isinstance(existing_tickers, dict) else {}
        position_change = _ticker_change_24h(position, None)
        previous_change = _ticker_change_24h(previous, None) if isinstance(previous, dict) else None
        if position_change is not None and abs(position_change) > 1e-12:
            change_24h = position_change
        elif previous_change is not None:
            change_24h = previous_change
        else:
            change_24h = position_change or 0.0
        tickers[symbol] = {
            "price": price,
            "change_24h": change_24h,
            "volume_24h": _safe_float(
                previous.get("volume_24h") if isinstance(previous, dict) else None,
                0.0,
            )
            or 0.0,
            "bid": _safe_float(previous.get("bid") if isinstance(previous, dict) else None, 0.0)
            or 0.0,
            "ask": _safe_float(previous.get("ask") if isinstance(previous, dict) else None, 0.0)
            or 0.0,
        }
    return tickers


def _merge_market_and_public_ticker(market_ticker: dict, public_ticker: dict) -> dict:
    merged = {**public_ticker, **market_ticker}
    market_change = _ticker_change_24h(market_ticker, None)
    public_change = _ticker_change_24h(public_ticker, None)
    merged["change_24h"] = (
        market_change
        if market_change is not None and abs(market_change) > 1e-12
        else public_change if public_change is not None else market_change or 0.0
    )
    for key in (
        "volume_24h",
        "volume_24h_contracts",
        "volume_24h_base",
        "volume_24h_quote",
        "notional_24h_usdt",
        "volume_24h_source",
        "bid",
        "ask",
    ):
        if _safe_float(market_ticker.get(key), 0.0) == 0.0:
            merged[key] = public_ticker.get(key, market_ticker.get(key, 0.0))
    return merged


def _ticker_change_24h(ticker: dict[str, Any] | None, default: float | None = 0.0) -> float | None:
    if not isinstance(ticker, dict):
        return default
    for key in ("change_24h", "change24h", "change_24h_pct", "percentage"):
        if key not in ticker:
            continue
        value = ticker.get(key)
        if value is None:
            continue
        return _safe_float(value, 0.0)
    return default


async def _get_public_ticker_map(symbols: set[str]) -> dict[str, dict]:
    requested = sorted(s for s in symbols if s)
    if not requested:
        return {}

    cache_key = ",".join(requested)
    now = datetime.now(UTC)
    cached = _public_ticker_cache.get(cache_key)
    if cached:
        cached_at, cached_value = cached
        if (now - cached_at).total_seconds() <= _PUBLIC_TICKER_CACHE_TTL_SECONDS:
            return cached_value

    rest_client = getattr(_data_service, "rest_client", None) if _data_service else None
    owns_client = False
    if rest_client is None:
        rest_client = OKXRestClient()
        owns_client = True

    try:
        raw_tickers = await asyncio.wait_for(rest_client.fetch_tickers(requested), timeout=8.0)
    except Exception as exc:
        _log_dashboard_fallback(
            "public ticker fallback",
            exc,
            symbol_count=len(requested),
            has_cached=bool(cached),
        )
        raw_tickers = await _fetch_public_tickers_individually(rest_client, requested)
        if not raw_tickers:
            return cached[1] if cached else {}
    finally:
        if owns_client:
            try:
                await rest_client.close()
            except Exception as exc:
                _log_dashboard_fallback("public ticker client close fallback", exc)

    parsed = _parse_public_tickers(raw_tickers, set(requested))

    _public_ticker_cache[cache_key] = (now, parsed)
    return parsed


async def _fetch_public_tickers_individually(
    rest_client: Any, requested: list[str]
) -> dict[str, dict[str, Any]]:
    fetch_ticker = getattr(rest_client, "fetch_ticker", None)
    has_single_ticker = callable(fetch_ticker)
    if len(requested) == 1 and not has_single_ticker:
        return {}

    recovered: dict[str, dict[str, Any]] = {}
    for symbol in requested:
        try:
            if has_single_ticker:
                ticker = await asyncio.wait_for(fetch_ticker(symbol), timeout=3.0)
                raw_tickers = {symbol: ticker} if isinstance(ticker, dict) else {}
            else:
                raw_tickers = await asyncio.wait_for(
                    rest_client.fetch_tickers([symbol]), timeout=3.0
                )
        except Exception as exc:
            _log_dashboard_fallback(
                "public ticker symbol fallback",
                exc,
                symbol=symbol,
            )
            continue

        if isinstance(raw_tickers, dict):
            recovered.update(raw_tickers)

    return recovered


def _parse_public_tickers(
    raw_tickers: dict[str, Any] | None, requested_symbols: set[str]
) -> dict[str, dict]:
    parsed: dict[str, dict] = {}
    for raw_key, ticker in (raw_tickers or {}).items():
        if not isinstance(ticker, dict):
            continue
        info = ticker.get("info") or {}
        symbol = _normalize_dashboard_symbol(ticker.get("symbol") or info.get("instId") or raw_key)
        if symbol not in requested_symbols:
            continue
        price = _safe_float(
            ticker.get("price")
            or ticker.get("last_price")
            or ticker.get("last")
            or ticker.get("close")
            or info.get("last")
            or info.get("lastPx"),
            0.0,
        )
        volume_fields = okx_swap_volume_fields(ticker, price)
        parsed[symbol] = {
            "price": price,
            "change_24h": _okx_display_change_pct(ticker),
            "volume_24h": _safe_float(
                volume_fields.get("volume_24h_base")
                or ticker.get("baseVolume")
                or info.get("volCcy24h")
                or info.get("vol24h"),
                0.0,
            ),
            **volume_fields,
            "bid": _safe_float(ticker.get("bid") or info.get("bidPx"), 0.0),
            "ask": _safe_float(ticker.get("ask") or info.get("askPx"), 0.0),
        }
    return parsed


def _okx_display_change_pct(ticker: dict) -> float:
    """Use OKX's real 24h change, not the UTC+8 session-open helper field."""
    info = ticker.get("info") or {}
    for key in ("change_24h", "change24h", "change_24h_pct", "percentage"):
        value = ticker.get(key)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if abs(parsed) > 1e-12:
            return parsed
    try:
        price = float(
            ticker.get("price")
            or ticker.get("last_price")
            or ticker.get("last")
            or ticker.get("close")
            or info.get("last")
            or info.get("lastPx")
            or 0
        )
        baseline = float(
            ticker.get("open")
            or ticker.get("open24h")
            or info.get("open24h")
            or info.get("sodUtc8")
            or ticker.get("sodUtc8")
            or 0
        )
        if price > 0 and baseline > 0:
            return (price - baseline) / baseline * 100
    except (TypeError, ValueError):
        pass
    try:
        return float(
            ticker.get("change_24h")
            or ticker.get("change24h")
            or ticker.get("change_24h_pct")
            or ticker.get("percentage")
            or 0
        )
    except (TypeError, ValueError):
        return 0.0


@router.get("/status")
async def get_status(mode: str | None = None):
    """System status overview. Optional mode filter for cumulative counts."""
    trading_stats = await _trading_stats_with_runtime_heartbeat(mode)

    return {
        "status": "running" if bool(trading_stats.get("running")) else "stopped",
        "timestamp": datetime.now(UTC).isoformat(),
        **trading_stats,
        "mode": mode_manager.mode.value,
        "paused": mode_manager.is_paused,
        "scan_mode": mode_manager.scan_mode,
        "live_model": mode_manager.live_model_name,
    }


@router.get("/ml-signal/status")
async def get_ml_signal_status():
    try:
        ml_signal_service = _dashboard_ml_signal_service()
    except Exception as exc:
        return {
            "available": False,
            "status": "client_error",
            "error": safe_error_text(exc, limit=180),
            "message": "本地 ML 状态客户端初始化失败。",
        }
    if not ml_signal_service:
        return {"available": False, "status": "service_not_ready"}
    try:
        status = ml_signal_service.status()
    except Exception as exc:
        _log_dashboard_fallback("ml signal status fallback", exc)
        return {
            "available": False,
            "status": "status_error",
            "error": safe_error_text(exc, limit=180),
            "message": "本地 ML 状态读取失败；请检查模型文件、训练元数据和服务日志。",
        }
    if not isinstance(status, dict):
        return status

    raw_training_count = _safe_int_value(
        status.get("sample_count") or status.get("trained_sample_count"), 0
    )
    explicit_training_count = _explicit_phase3_count(
        status,
        "phase3_clean_trainable_shadow_sample_count",
        "phase3_clean_completed_shadow_sample_count",
    )
    if explicit_training_count is None:
        explicit_training_count = await _completed_ml_shadow_sample_count()
    training_count = int(explicit_training_count or 0)
    status["phase3_clean_trainable_shadow_sample_count"] = training_count
    status["training_shadow_sample_count"] = training_count
    status["raw_shadow_sample_count"] = raw_training_count
    status["legacy_shadow_sample_count"] = max(raw_training_count - training_count, 0)
    status.setdefault(
        "training_shadow_sample_limit",
        LOCAL_ML_TRAINING_PARAMS.training_shadow_sample_limit,
    )
    status.setdefault(
        "training_sample_note",
        "sample_count is the latest training window, not the all-time total.",
    )

    try:
        completed_total = _explicit_phase3_count(
            status,
            "phase3_clean_completed_shadow_sample_count",
            "phase3_clean_trainable_shadow_sample_count",
        )
        if completed_total is None:
            completed_total = await _completed_ml_shadow_sample_count()
        completed_total = int(completed_total or 0)
        completed_total = max(completed_total, training_count)
        status["phase3_clean_completed_shadow_sample_count"] = completed_total
        status["completed_shadow_sample_count"] = completed_total
        status["total_shadow_sample_count"] = completed_total

        auto_last = (
            status.get("auto_train_last_result")
            if isinstance(status.get("auto_train_last_result"), dict)
            else {}
        )
        auto_new = (
            auto_last.get("phase3_new_shadow_sample_count") if isinstance(auto_last, dict) else None
        )
        if auto_new is None:
            trained_cursor = _phase3_trained_shadow_cursor(status, completed_total)
            status["last_trained_phase3_shadow_sample_count"] = trained_cursor
            auto_new = max(int(completed_total) - trained_cursor, 0)
        status["phase3_new_shadow_sample_count"] = int(auto_new or 0)
        status["new_shadow_sample_count"] = int(auto_new or 0)
    except Exception as exc:
        _log_dashboard_fallback("ml signal sample count fallback", exc)
        status["phase3_clean_completed_shadow_sample_count"] = training_count
        status["completed_shadow_sample_count"] = training_count
        status["total_shadow_sample_count"] = training_count
        status["phase3_new_shadow_sample_count"] = 0
        status["new_shadow_sample_count"] = 0

    return status


def _phase3_trained_shadow_cursor(status: dict[str, Any], completed_total: int) -> int:
    """Return the trained shadow cursor using the same Phase 3 sample-count scale."""
    candidates = (
        status.get("last_trained_phase3_shadow_sample_count"),
        status.get("phase3_trained_shadow_sample_count"),
        status.get("last_trained_completed_shadow_sample_count"),
        status.get("last_trained_completed_sample_count"),
        status.get("sample_count"),
        status.get("trained_sample_count"),
    )
    for value in candidates:
        cursor = _optional_non_negative_int(value)
        if cursor is not None and cursor <= completed_total:
            return cursor
    return min(
        _safe_int_value(
            status.get("training_shadow_sample_count")
            or status.get("sample_count")
            or status.get("trained_sample_count"),
            0,
        ),
        completed_total,
    )


def _optional_non_negative_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


@router.get("/local-ai-tools/status")
async def get_local_ai_tools_status():
    try:
        local_ai_tools = _dashboard_local_ai_tools_client()
    except Exception as exc:
        return {
            "available": False,
            "status": "client_error",
            "error": safe_error_text(exc, limit=180),
            "message": "本地量化工具客户端初始化失败。",
        }
    if not local_ai_tools:
        return {"available": False, "status": "service_not_ready"}
    try:
        status = await local_ai_tools.status()
    except Exception as exc:
        _log_dashboard_fallback("local ai tools status fallback", exc)
        return {
            "available": False,
            "status": "status_error",
            "error": safe_error_text(exc, limit=180),
            "message": "本地量化工具状态读取失败；请检查 18001 隧道、API Key 和服务健康接口。",
        }
    if isinstance(status, dict):
        try:
            completed_total = await _completed_local_ai_shadow_backtest_total()
            service_shadow_count = _safe_int_value(
                status.get("shadow_sample_count") or status.get("training_shadow_sample_count"),
                0,
            )
            service_trade_count = _safe_int_value(status.get("trade_sample_count"), 0)
            explicit_phase3_count = _explicit_phase3_count(
                status,
                "phase3_clean_trainable_shadow_sample_count",
                "phase3_clean_completed_shadow_sample_count",
            )
            phase3_count = int(
                explicit_phase3_count if explicit_phase3_count is not None else completed_total
            )
            status["phase3_clean_trainable_shadow_sample_count"] = phase3_count
            status["phase3_clean_completed_shadow_sample_count"] = phase3_count
            status["shadow_sample_count"] = phase3_count
            status["completed_shadow_sample_count"] = phase3_count
            status["training_shadow_sample_count"] = phase3_count
            status["total_shadow_sample_count"] = phase3_count
            status["raw_shadow_sample_count"] = phase3_count
            status["legacy_shadow_sample_count"] = 0
            status["service_model_window_shadow_sample_count"] = service_shadow_count
            status["service_model_window_trade_sample_count"] = service_trade_count
            status["training_sample_source"] = "phase3_clean_completed_shadow_backtests"
            status.setdefault(
                "training_shadow_sample_limit",
                LOCAL_ML_TRAINING_PARAMS.training_shadow_sample_limit,
            )
        except Exception as exc:
            _log_dashboard_fallback("local ai tools shadow count fallback", exc)
    return status


@router.get("/server-monitor/status")
async def get_server_monitor_status():
    return await get_server_monitor_status_async()


@router.get("/dashboard/summary")
async def get_dashboard_summary():
    """Aggregate dashboard data."""
    market_state = await _build_open_position_market_snapshot(mode_manager.mode.value)

    okx_account = await _get_dashboard_okx_account_snapshot(mode_manager.mode.value)

    pnl_summary = await _get_execution_pnl_summary(mode_manager.mode.value)
    execution_account = _build_execution_account_status(
        mode_manager.mode.value,
        paper_summary=None,
        okx_account=okx_account,
        pnl_summary=pnl_summary,
    )
    account_summaries = [execution_account]

    trading_stats = await _trading_stats_with_runtime_heartbeat(mode_manager.mode.value)
    today_decisions_total = await _get_today_ai_decision_count(mode_manager.mode.value)

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "market": market_state,
        "model_rankings": [],
        "execution_account": execution_account,
        "accounts": account_summaries,
        "okx_account": okx_account,
        **trading_stats,
        "mode": mode_manager.mode.value,
        "paused": mode_manager.is_paused,
        "scan_mode": mode_manager.scan_mode,
        "today_decisions_total": today_decisions_total,
        "today_decisions_timezone": "Asia/Shanghai",
    }


@router.get("/dashboard/market")
async def get_market_data():
    """Current market prices and stats."""
    return await _build_open_position_market_snapshot(mode_manager.mode.value)


@router.get("/dashboard/positions")
async def get_positions(
    mode: str | None = None,
    page: int = 1,
    page_size: int = 20,
    open_only: bool = False,
    closed_only: bool = False,
):
    """All positions (open + closed) — 持仓记录. Optional mode filter (paper/live)."""
    from sqlalchemy import select

    from db.repositories.trade_repo import TradeRepository
    from db.session import get_session_ctx
    from models.decision import AIDecision
    from models.trade import Order
    from services.okx_position_ledger_view import build_okx_position_ledger_groups

    page = max(int(page or 1), 1)
    page_size = max(1, min(int(page_size or 20), 100))
    positions = []
    exchange_mark_map = {} if closed_only else await _get_exchange_position_mark_map(mode)
    exchange_temporarily_unavailable = bool(
        not closed_only
        and not exchange_mark_map
        and _dashboard_okx_positions_temporarily_unavailable(mode)
    )
    exchange_symbols = (
        {symbol for symbol, _side in exchange_mark_map.keys()} if exchange_mark_map else None
    )
    market_tickers: dict[str, dict[str, Any]] = {}
    if _data_service:
        market_tickers = (_data_service.get_market_state() or {}).get("tickers", {}) or {}
    public_tickers: dict[str, dict[str, Any]] = {}
    if open_only and not closed_only:
        open_symbols = await _get_display_open_position_symbols(mode)
        public_tickers = await _get_public_ticker_map(open_symbols)
    if not open_only and not closed_only:
        try:
            open_rows = await _get_display_open_positions_snapshot(mode)
        except Exception as exc:
            _log_dashboard_fallback(
                "combined positions open snapshot fallback",
                exc,
                mode=mode,
            )
            open_rows = []
        async with get_session_ctx() as session:
            repo = TradeRepository(session)
            closed_rows, closed_total, _closed_page, _closed_pages, closed_ledger_source = (
                await _dashboard_closed_position_ledger_rows(
                    session,
                    repo,
                    mode=mode,
                    page=1,
                    page_size=5000,
                    paginate=False,
                )
            )
        combined_positions = [*open_rows, *closed_rows]
        display_total = len(combined_positions)
        display_total_pages = (
            max(1, (display_total + page_size - 1) // page_size) if display_total else 1
        )
        page = min(page, display_total_pages)
        start = (page - 1) * page_size
        return {
            "positions": combined_positions[start : start + page_size],
            "count": len(combined_positions[start : start + page_size]),
            "open_count": len(open_rows),
            "closed_count": closed_total,
            "total": display_total,
            "page": page,
            "page_size": page_size,
            "total_pages": display_total_pages,
            "ledger_source": f"okx_current_positions_plus_{closed_ledger_source}",
        }
    async with get_session_ctx() as session:
        repo = TradeRepository(session)
        is_open_filter = True if open_only else False if closed_only else None
        if closed_only:
            positions, display_total, page, display_total_pages, ledger_source = (
                await _dashboard_closed_position_ledger_rows(
                    session,
                    repo,
                    mode=mode,
                    page=page,
                    page_size=page_size,
                    paginate=True,
                )
            )
            return {
                "positions": positions,
                "count": display_total,
                "total": display_total,
                "page": page,
                "page_size": page_size,
                "total_pages": display_total_pages,
                "ledger_source": ledger_source,
            }
            closed_rows = await repo.get_position_records(
                execution_mode=mode,
                limit=5000,
                offset=0,
                is_open=False,
            )
            linked_order_ids = {
                token
                for position in closed_rows
                for value in (
                    getattr(position, "entry_exchange_order_id", None),
                    getattr(position, "close_exchange_order_id", None),
                )
                for token in _dashboard_split_exchange_order_ids(value)
            }
            symbol_variants = _dashboard_symbol_query_variants(
                {
                    _normalize_dashboard_symbol(str(getattr(position, "symbol", "") or ""))
                    for position in closed_rows
                    if getattr(position, "symbol", None)
                }
            )
            order_stmt = select(Order).where(Order.status == "filled")
            if mode:
                order_stmt = order_stmt.where(Order.execution_mode == mode)
            if symbol_variants:
                order_stmt = order_stmt.where(Order.symbol.in_(symbol_variants))
            else:
                order_stmt = order_stmt.where(Order.id == -1)
            order_rows = list(
                (
                    await session.execute(
                        order_stmt.order_by(
                            Order.filled_at.desc().nullslast(),
                            Order.created_at.desc(),
                        ).limit(10000)
                    )
                )
                .scalars()
                .all()
            )
            if linked_order_ids:
                order_rows = [
                    order
                    for order in order_rows
                    if _dashboard_split_exchange_order_ids(
                        getattr(order, "exchange_order_id", None)
                    )
                    & linked_order_ids
                ]
            ledger_groups = build_okx_position_ledger_groups(closed_rows, order_rows)
            display_total = len(ledger_groups)
            display_total_pages = (
                max(1, (display_total + page_size - 1) // page_size) if display_total else 1
            )
            page = min(page, display_total_pages)
            start = (page - 1) * page_size
            positions = [
                group.as_dict(include_fills=True)
                for group in ledger_groups[start : start + page_size]
            ]
            return {
                "positions": positions,
                "count": display_total,
                "total": display_total,
                "page": page,
                "page_size": page_size,
                "total_pages": display_total_pages,
                "ledger_source": "okx_native_grouped_cache",
            }
        total = await repo.count_positions(execution_mode=mode, is_open=is_open_filter)
        open_count = await repo.count_positions(execution_mode=mode, is_open=True)
        # When showing current open positions, exchange reconciliation happens
        # after DB rows are loaded. Load all open rows first, filter against OKX,
        # then paginate the display list; otherwise page 1 can incorrectly cap
        # the total at 20.
        should_paginate_after_display_filter = bool(open_only or closed_only)
        total_pages = max(1, (total + page_size - 1) // page_size) if total else 1
        page = min(page, total_pages)
        offset = 0 if should_paginate_after_display_filter else (page - 1) * page_size
        limit = (
            min(max(total, page_size), 5000) if should_paginate_after_display_filter else page_size
        )
        rows = await repo.get_position_records(
            execution_mode=mode,
            limit=limit,
            offset=offset,
            is_open=is_open_filter,
        )
        open_sibling_rows = []
        if closed_only:
            open_sibling_rows = await repo.get_position_records(
                execution_mode=mode,
                limit=5000,
                offset=0,
                is_open=True,
            )
        sibling_positions = list(rows) + list(open_sibling_rows)
        closed_group_totals: dict[tuple, dict[str, Any]] = {}
        if closed_only:
            for sibling in sibling_positions:
                created = getattr(sibling, "created_at", None)
                created_second = (
                    created.replace(microsecond=0).isoformat() if created is not None else ""
                )
                group_key = (
                    getattr(sibling, "model_name", None),
                    getattr(sibling, "execution_mode", None),
                    _normalize_dashboard_symbol(str(getattr(sibling, "symbol", "") or "")),
                    str(getattr(sibling, "side", "") or "").lower(),
                    round(_safe_float(getattr(sibling, "entry_price", None), 0.0) or 0.0, 8),
                    created_second,
                )
                totals = closed_group_totals.setdefault(
                    group_key,
                    {"closed_qty": 0.0, "open_qty": 0.0, "closed_count": 0},
                )
                qty = _safe_float(getattr(sibling, "quantity", None), 0.0) or 0.0
                if getattr(sibling, "is_open", False):
                    totals["open_qty"] += qty
                else:
                    totals["closed_qty"] += qty
                    totals["closed_count"] += 1

        def closed_group_key_for(pos) -> tuple:
            created = getattr(pos, "created_at", None)
            created_second = (
                created.replace(microsecond=0).isoformat() if created is not None else ""
            )
            return (
                getattr(pos, "model_name", None),
                getattr(pos, "execution_mode", None),
                _normalize_dashboard_symbol(str(getattr(pos, "symbol", "") or "")),
                str(getattr(pos, "side", "") or "").lower(),
                round(_safe_float(getattr(pos, "entry_price", None), 0.0) or 0.0, 8),
                created_second,
            )

        def group_close_status_for(pos) -> tuple[str | None, str | None]:
            if not closed_only or getattr(pos, "is_open", False):
                return None, None
            totals = closed_group_totals.get(closed_group_key_for(pos)) or {}
            if (
                _safe_float(totals.get("open_qty"), 0.0) > 1e-8
                or int(totals.get("closed_count") or 0) > 1
            ):
                return "partial", "部分平仓"
            return None, None

        close_order_match_by_position_id: dict[int, dict[str, Any]] = {}
        close_order_match_window = timedelta(seconds=240)
        if closed_only and rows:
            closed_rows = [p for p in rows if not p.is_open and p.closed_at]
            close_symbols = sorted(
                {_normalize_dashboard_symbol(p.symbol) for p in closed_rows if p.symbol}
            )
            if close_symbols:
                close_symbol_variants = _dashboard_symbol_query_variants(set(close_symbols))
                min_closed = min((p.closed_at for p in closed_rows if p.closed_at), default=None)
                max_closed = max((p.closed_at for p in closed_rows if p.closed_at), default=None)
                order_stmt = select(Order).where(
                    Order.symbol.in_(close_symbol_variants),
                    Order.status == "filled",
                )
                if mode:
                    order_stmt = order_stmt.where(Order.execution_mode == mode)
                if min_closed:
                    order_stmt = order_stmt.where(
                        Order.created_at >= min_closed - close_order_match_window
                    )
                if max_closed:
                    order_stmt = order_stmt.where(
                        Order.created_at <= max_closed + close_order_match_window
                    )
                order_result = await session.execute(order_stmt)
                close_orders = list(order_result.scalars().all())
                close_orders_by_key: dict[tuple[str, str], list[Any]] = {}
                for order in close_orders:
                    order_key = (
                        _normalize_dashboard_symbol(order.symbol),
                        str(order.side or "").lower(),
                    )
                    close_orders_by_key.setdefault(order_key, []).append(order)
                decision_meta: dict[int, dict[str, Any]] = {}

                def status_from_order_decision(order) -> tuple[str | None, str | None]:
                    if is_manual_close_order(order):
                        return "manual", MANUAL_CLOSE_LABEL
                    meta = _safe_dict(decision_meta.get(order.decision_id or -1))
                    pct = (
                        _safe_float(
                            meta.get("position_size_pct"),
                            0.0,
                        )
                        or 0.0
                    )
                    if pct > 0:
                        if pct < 0.999:
                            return "partial", "部分平仓"
                        return "full", "全部平仓"
                    return None, None

                for pos in closed_rows:
                    close_side = "buy" if str(pos.side or "").lower() == "short" else "sell"
                    closed_at = pos.closed_at
                    closed_at_cmp = (
                        closed_at.replace(tzinfo=None)
                        if closed_at and closed_at.tzinfo
                        else closed_at
                    )
                    candidates = []
                    order_key = (_normalize_dashboard_symbol(pos.symbol), close_side)
                    for order in close_orders_by_key.get(order_key, []):
                        order_time = order.filled_at or order.created_at
                        order_time_cmp = (
                            order_time.replace(tzinfo=None)
                            if order_time and order_time.tzinfo
                            else order_time
                        )
                        if not closed_at_cmp or not order_time_cmp:
                            continue
                        delta = abs((order_time_cmp - closed_at_cmp).total_seconds())
                        if delta > close_order_match_window.total_seconds():
                            continue
                        order_qty = _safe_float(order.quantity, 0.0) or 0.0
                        position_qty = _safe_float(pos.quantity, 0.0) or 0.0
                        if order_qty > 0 and position_qty > order_qty * 1.05:
                            continue
                        price_delta = abs(
                            (_safe_float(order.price, 0.0) or 0.0)
                            - (_safe_float(pos.current_price, 0.0) or 0.0)
                        )
                        qty_delta = abs(order_qty - position_qty)
                        candidates.append((delta, price_delta, qty_delta, order))
                    if not candidates:
                        continue
                    matched_order = sorted(
                        candidates, key=lambda item: (item[0], item[1], item[2])
                    )[0][3]
                    close_order_match_by_position_id[pos.id] = {
                        "order": matched_order,
                        "source": "order",
                    }

                matched_decision_ids = {
                    match["order"].decision_id
                    for match in close_order_match_by_position_id.values()
                    if match.get("order") and match["order"].decision_id
                }
                if matched_decision_ids:
                    decision_result = await session.execute(
                        select(
                            AIDecision.id,
                            AIDecision.action,
                            AIDecision.position_size_pct,
                        ).where(AIDecision.id.in_(matched_decision_ids))
                    )
                    for decision in decision_result.all():
                        decision_meta[int(decision.id)] = {
                            "action": decision.action,
                            "position_size_pct": decision.position_size_pct,
                        }
                for match in close_order_match_by_position_id.values():
                    status, label = status_from_order_decision(match["order"])
                    match["status"] = status or "full"
                    match["label"] = label or "全部平仓"

        def close_status_for(pos) -> tuple[str, str]:
            if pos.is_open:
                return "open", "持有中"

            group_status, group_label = group_close_status_for(pos)
            if group_status:
                return group_status, str(group_label)

            order_status = close_order_match_by_position_id.get(pos.id)
            if order_status:
                return str(order_status["status"]), str(order_status["label"])

            # A system partial close is stored as a separate closed row for the
            # closed quantity, while the original older position keeps the
            # remaining quantity. Detect that shape without changing DB schema.
            closed_at = pos.closed_at
            created_at = pos.created_at
            if closed_at and closed_at.tzinfo is None:
                closed_at = closed_at.replace(tzinfo=UTC)
            if created_at and created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=UTC)
            same_time_close = False
            if closed_at and created_at:
                same_time_close = abs((closed_at - created_at).total_seconds()) <= 5
            if not same_time_close:
                return "full", "全部平仓"

            symbol_key = _normalize_dashboard_symbol(pos.symbol)
            side_key = str(pos.side or "").lower()
            entry = float(pos.entry_price or 0.0)
            tolerance = max(abs(entry) * 1e-6, 1e-8)
            for sibling in sibling_positions:
                if sibling.id == pos.id:
                    continue
                if (
                    sibling.model_name != pos.model_name
                    or sibling.execution_mode != pos.execution_mode
                ):
                    continue
                if _normalize_dashboard_symbol(sibling.symbol) != symbol_key:
                    continue
                if str(sibling.side or "").lower() != side_key:
                    continue
                sibling_entry = float(sibling.entry_price or 0.0)
                if entry > 0 and abs(sibling_entry - entry) > tolerance:
                    continue
                sibling_created_at = sibling.created_at
                if sibling_created_at and sibling_created_at.tzinfo is None:
                    sibling_created_at = sibling_created_at.replace(tzinfo=UTC)
                if sibling_created_at and created_at and sibling_created_at < created_at:
                    return "partial", "部分平仓"
            return "full", "全部平仓"

        for p in rows:
            current_price = p.current_price
            unrealized_pnl = p.unrealized_pnl
            realized_pnl = p.realized_pnl
            closed_at = p.closed_at
            entry_price = p.entry_price
            quantity = p.quantity
            pnl_source = "local_db"
            change_24h = 0.0
            if p.is_open:
                snapshot = exchange_mark_map.get(
                    (_normalize_dashboard_symbol(p.symbol), str(p.side or "").lower())
                )
                symbol_key = _normalize_dashboard_symbol(p.symbol)
                market_ticker = dict(market_tickers.get(symbol_key, {}) or {})
                public_ticker = dict(public_tickers.get(symbol_key, {}) or {})
                display_ticker = _merge_market_and_public_ticker(
                    market_ticker,
                    public_ticker,
                )
                change_24h = _ticker_change_24h(display_ticker, 0.0) or 0.0
                if snapshot:
                    valuation = _exchange_position_display_valuation(
                        snapshot,
                        str(p.side or "").lower(),
                        fallback_current_price=current_price,
                        fallback_unrealized_pnl=unrealized_pnl,
                        fallback_entry_price=entry_price,
                        fallback_quantity=quantity,
                    )
                    current_price = valuation["current_price"]
                    unrealized_pnl = valuation["unrealized_pnl"]
                    entry_price = valuation["entry_price"]
                    quantity = valuation["quantity"]
                    pnl_source = valuation["pnl_source"]

            exchange_key = _dashboard_position_key(p.symbol, p.side)
            exchange_synced = True
            if p.is_open:
                exchange_synced = bool(
                    exchange_symbols is not None and exchange_key in exchange_mark_map
                )
            display_is_open = bool(
                p.is_open and (exchange_synced or exchange_temporarily_unavailable)
            )
            if open_only and not display_is_open:
                continue
            matched_close = close_order_match_by_position_id.get(p.id)
            if closed_only and matched_close:
                close_order = matched_close.get("order")
                close_price = _safe_float(getattr(close_order, "price", None), 0.0) or 0.0
                position_quantity = _safe_float(quantity, 0.0) or 0.0
                if close_price > 0:
                    current_price = close_price
                closed_at = (
                    getattr(close_order, "filled_at", None)
                    or getattr(close_order, "created_at", None)
                    or closed_at
                )
                if close_price > 0 and position_quantity > 0:
                    realized_pnl = _safe_float(p.realized_pnl, 0.0) or 0.0
                    pnl_source = "position_realized_pnl"
            close_status, close_status_label = close_status_for(p)
            close_status_source = (
                "order" if p.id in close_order_match_by_position_id else "position"
            )

            positions.append(
                {
                    "id": p.id,
                    "model_name": p.model_name,
                    "mode": p.execution_mode,
                    "symbol": p.symbol,
                    "side": p.side,
                    "quantity": quantity,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "change_24h": change_24h,
                    "unrealized_pnl": unrealized_pnl,
                    "pnl_source": pnl_source,
                    "local_quantity": p.quantity,
                    "local_entry_price": p.entry_price,
                    "local_unrealized_pnl": p.unrealized_pnl,
                    "realized_pnl": realized_pnl,
                    "leverage": p.leverage,
                    "stop_loss": p.stop_loss_price,
                    "take_profit": p.take_profit_price,
                    "is_open": display_is_open,
                    "db_is_open": p.is_open,
                    "exchange_synced": exchange_synced,
                    "exchange_temporarily_unavailable": bool(
                        p.is_open and exchange_temporarily_unavailable and not exchange_synced
                    ),
                    "close_status": close_status,
                    "close_status_label": close_status_label,
                    "close_status_source": close_status_source,
                    "position_status": close_status_label,
                    "opened_at": p.created_at.isoformat() if p.created_at else None,
                    "closed_at": closed_at.isoformat() if closed_at else None,
                }
            )

    if closed_only:
        grouped_positions: list[dict] = []
        grouped_index: dict[tuple, dict] = {}
        for item in positions:
            closed_at_text = str(item.get("closed_at") or "")
            closed_second = closed_at_text.split(".")[0] if closed_at_text else ""
            close_price = round(_safe_float(item.get("current_price"), 0.0) or 0.0, 8)
            key = (
                item.get("model_name"),
                item.get("mode"),
                _normalize_dashboard_symbol(str(item.get("symbol") or "")),
                str(item.get("side") or "").lower(),
                closed_second,
                close_price,
            )
            quantity = _safe_float(item.get("quantity"), 0.0) or 0.0
            entry_price = _safe_float(item.get("entry_price"), 0.0) or 0.0
            realized_pnl = _safe_float(item.get("realized_pnl"), 0.0) or 0.0
            if key not in grouped_index:
                item = dict(item)
                item["position_ids"] = [item.get("id")]
                item["split_count"] = 1
                item["_entry_notional_for_avg"] = entry_price * quantity
                grouped_index[key] = item
                grouped_positions.append(item)
                continue

            group = grouped_index[key]
            group_qty = (_safe_float(group.get("quantity"), 0.0) or 0.0) + quantity
            group["_entry_notional_for_avg"] = (
                _safe_float(group.get("_entry_notional_for_avg"), 0.0) or 0.0
            ) + entry_price * quantity
            group["quantity"] = group_qty
            if group_qty:
                group["entry_price"] = group["_entry_notional_for_avg"] / group_qty
            group["realized_pnl"] = (
                _safe_float(group.get("realized_pnl"), 0.0) or 0.0
            ) + realized_pnl
            group["split_count"] = int(group.get("split_count") or 1) + 1
            group.setdefault("position_ids", []).append(item.get("id"))
            group["id"] = min(
                [pid for pid in group.get("position_ids", []) if pid is not None]
                or [group.get("id")]
            )
            order_based_status = (
                group.get("close_status_source") == "order"
                or item.get("close_status_source") == "order"
            )
            if order_based_status:
                group["close_status_source"] = "order"
                if group.get("close_status") == "manual" or item.get("close_status") == "manual":
                    group["close_status"] = "manual"
                elif (
                    group.get("close_status") == "partial" or item.get("close_status") == "partial"
                ):
                    group["close_status"] = "partial"
                elif group.get("close_status") == "full" or item.get("close_status") == "full":
                    group["close_status"] = "full"
            else:
                group["close_status"] = (
                    "partial"
                    if group.get("close_status") == "partial"
                    or item.get("close_status") == "partial"
                    else group.get("close_status")
                )
            group["close_status_label"] = (
                "部分平仓"
                if group.get("close_status") == "partial"
                else group.get("close_status_label")
            )
            if group.get("close_status") == "manual":
                group["close_status_label"] = MANUAL_CLOSE_LABEL
            if group.get("close_status") == "full":
                group["close_status_label"] = "全部平仓"
            group["position_status"] = group["close_status_label"]

        for item in grouped_positions:
            item.pop("_entry_notional_for_avg", None)
        positions = grouped_positions

    if open_only and not closed_only:
        positions = _group_open_dashboard_positions(
            positions,
            exchange_mark_map,
            mode=mode,
        )

    display_total = len(positions) if open_only or closed_only else total
    display_open_count = len(positions) if open_only else open_count
    display_total_pages = (
        max(1, (display_total + page_size - 1) // page_size) if display_total else 1
    )
    page = min(page, display_total_pages)
    if should_paginate_after_display_filter:
        start = (page - 1) * page_size
        positions = positions[start : start + page_size]
    return {
        "positions": positions,
        "count": display_open_count,
        "total": display_total,
        "page": page,
        "page_size": page_size,
        "total_pages": display_total_pages,
    }


@router.get("/decisions")
async def get_decisions(
    limit: int = 200,
    page: int = 1,
    page_size: int | None = None,
    model_name: str | None = None,
    action: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    was_executed: bool | None = None,
    is_paper: bool | None = None,
):
    """Get AI decisions from DB with optional filters.

    - start_date / end_date: ISO datetime strings for time range filter
    - model_name: filter by AI model name
    - action: filter by decision direction/action
    - was_executed: true=executed only, false=pending only, omit=all
    - is_paper: true=paper trading, false=live trading, omit=all
    """
    from datetime import datetime as dt

    from sqlalchemy import select

    from db.repositories.decision_repo import DecisionRepository
    from db.session import get_session_ctx
    from models.trade import Order

    effective_page_size = page_size if page_size is not None else limit
    effective_page_size = max(1, min(int(effective_page_size or 50), 500))
    page = max(int(page or 1), 1)
    offset = (page - 1) * effective_page_size
    start_dt = dt.fromisoformat(start_date) if start_date else None
    end_dt = dt.fromisoformat(end_date) if end_date else None
    allowed_actions = {"long", "short", "close_long", "close_short", "hold"}
    action_filter = (action or "").strip().lower() or None
    if action_filter not in allowed_actions:
        action_filter = None

    async with get_session_ctx() as session:
        repo = DecisionRepository(session)
        rows = await repo.get_recent_decisions(
            model_name=model_name,
            action=action_filter,
            limit=effective_page_size,
            offset=offset,
            start_date=start_dt,
            end_date=end_dt,
            was_executed=was_executed,
            is_paper=is_paper,
        )
        total = await repo.count_decisions(
            model_name=model_name,
            action=action_filter,
            start_date=start_dt,
            end_date=end_dt,
            was_executed=was_executed,
            is_paper=is_paper,
        )
        order_map = {}
        decision_ids = [d.id for d in rows]
        if decision_ids:
            order_result = await session.execute(
                select(Order)
                .where(Order.decision_id.in_(decision_ids))
                .order_by(Order.created_at.desc())
            )
            for order in order_result.scalars().all():
                if order.decision_id not in order_map:
                    order_map[order.decision_id] = order

    decisions = []
    for d in rows:
        decision_type, decision_type_label = _decision_type(d.action)
        raw = d.raw_llm_response if isinstance(d.raw_llm_response, dict) else {}
        order = order_map.get(d.id)
        display_reason = _display_execution_reason(d, order)
        order_quantity = _safe_float(getattr(order, "quantity", None), 0.0) if order else None
        order_price = _safe_float(getattr(order, "price", None), 0.0) if order else None
        order_notional = (
            order_quantity * order_price
            if order_quantity is not None
            and order_price is not None
            and order_quantity > 0
            and order_price > 0
            else None
        )
        decisions.append(
            sanitize_payload(
                {
                    "id": d.id,
                    "model_name": d.model_name,
                    "symbol": _normalize_dashboard_symbol(d.symbol),
                    "action": d.action,
                    "decision_type": decision_type,
                    "decision_type_label": decision_type_label,
                    "confidence": d.confidence,
                    "reasoning": sanitize_text(d.reasoning),
                    "position_size_pct": d.position_size_pct,
                    "position_size_pct_basis": "execution_account_available_margin",
                    "position_size_pct_label": "保证金占当前执行账户可用余额比例",
                    "suggested_leverage": d.suggested_leverage,
                    "was_executed": d.was_executed,
                    "execution_reason": display_reason,
                    "executed_at": d.executed_at.isoformat() if d.executed_at else None,
                    "execution_price": d.execution_price,
                    "order_quantity": order_quantity,
                    "order_price": order_price,
                    "order_notional_usdt": order_notional,
                    "order_status": getattr(order, "status", None) if order else None,
                    "exchange_order_id": (
                        getattr(order, "exchange_order_id", None) if order else None
                    ),
                    "created_at": d.created_at.isoformat() if d.created_at else None,
                    "outcome": d.outcome,
                    "is_paper": d.is_paper,
                    "opportunity_score": _display_opportunity_score(d, raw, display_reason),
                }
            )
        )

    total_pages = max(1, (total + effective_page_size - 1) // effective_page_size) if total else 1
    return {
        "decisions": decisions,
        "count": len(decisions),
        "total": total,
        "page": page,
        "page_size": effective_page_size,
        "total_pages": total_pages,
    }


def _opening_funnel_reason_bucket(reason: str | None) -> str:
    text = str(reason or "").lower()
    if not text:
        return "unknown"
    if any(
        token in text
        for token in (
            "预期净收益",
            "净收益",
            "不为正",
            "收益期望",
            "正期望",
            "盈利质量",
            "expected_net",
            "expected net",
            "expected return",
            "profit_quality",
            "profit quality",
            "system_pre_submit_rejection",
        )
    ):
        return "profit_expectancy"
    if any(
        token in text
        for token in (
            "动态证据不足",
            "保持观望",
            "极小探针",
            "entry_evidence_wait",
            "skip_kind",
        )
    ):
        return "waiting_queue"
    if any(
        token in text
        for token in (
            "动态证据",
            "证据评分",
            "候选评分未达",
            "机会评分",
            "entry evidence",
            "entry_evidence",
            "evidence",
            "探索下限",
            "硬拦",
            "弱冲突",
        )
    ):
        return "evidence_gate"
    if any(
        token in text
        for token in (
            "下单前价格",
            "价格已比",
            "允许偏移",
            "价格偏移",
            "即时刷新",
            "行情复核",
            "盘口",
            "动量",
            "追多",
            "追空",
            "risk",
            "风控",
            "熔断",
            "circuit",
            "confidence",
            "置信",
            "adx",
            "成交量",
            "volume",
            "趋势",
            "仓位",
            "capacity",
            "position limit",
            "止盈",
            "止损",
            "minimum",
            "too low",
            "余额",
            "balance",
        )
    ):
        return "risk_or_precheck"
    if any(token in text for token in ("等待", "排队", "candidate", "queue", "staged", "本轮")):
        return "waiting_queue"
    if any(
        token in text
        for token in (
            "okx",
            "订单",
            "order",
            "下单",
            "执行",
            "executor",
            "timeout",
            "超时",
            "接口",
            "api",
        )
    ):
        return "execution_or_exchange"
    if any(token in text for token in ("成本", "token", "budget", "预算")):
        return "ai_budget"
    return "other"


def _opening_funnel_is_repair_cleanup(decision) -> bool:
    """Rows created only to document a local repair must not pollute the entry funnel."""
    reason = str(getattr(decision, "execution_reason", "") or "")
    if "已清理修复前误记" in reason:
        return True
    if "未确认成交" in reason and "撤销" in reason:
        return True
    return False


@router.get("/opening-funnel")
async def get_opening_funnel(
    mode: str | None = None,
    hours: int = 24,
    limit: int = 500,
):
    try:
        return await _build_opening_funnel_payload(mode=mode, hours=hours, limit=limit)
    except Exception as exc:
        _log_dashboard_fallback("opening funnel fallback", exc)
        selected_mode = "live" if mode == "live" else "paper"
        capped_hours = max(1, min(int(hours or 24), 24 * 30))
        capped_limit = max(50, min(int(limit or 500), 2000))
        return sanitize_payload(
            {
                "mode": selected_mode,
                "window_hours": capped_hours,
                "sample_limit": capped_limit,
                "sampled_decisions": 0,
                "repair_cleanup_rows": 0,
                "market_scans": 0,
                "average_confidence": 0.0,
                "stages": {
                    "market_scans": 0,
                    "ai_entry_signals": 0,
                    "orders_created": 0,
                    "executed_entries": 0,
                },
                "rates": {
                    "signal_rate": 0.0,
                    "order_rate": 0.0,
                    "execution_rate": 0.0,
                    "overall_open_rate": 0.0,
                },
                "action_counts": {"hold": 0, "long": 0, "short": 0, "other": 0},
                "reason_buckets": {
                    "profit_expectancy": 0,
                    "evidence_gate": 0,
                    "risk_or_precheck": 0,
                    "waiting_queue": 0,
                    "execution_or_exchange": 0,
                    "ai_budget": 0,
                    "other": 0,
                    "unknown": 0,
                },
                "hold_count": 0,
                "no_order_after_signal": 0,
                "bottleneck": "api_error",
                "bottleneck_label": "漏斗接口异常，已返回诊断占位",
                "top_symbols": [],
                "recent_blocked": [],
                "generated_at": datetime.now(UTC).isoformat(),
                "status": "error",
                "detail": safe_error_text(exc, limit=180),
            }
        )


async def _build_opening_funnel_payload(
    mode: str | None = None,
    hours: int = 24,
    limit: int = 500,
):
    """Diagnose where new entries are filtered out before becoming positions."""
    from sqlalchemy import select

    from db.session import get_session_ctx
    from models.decision import AIDecision
    from models.trade import Order

    selected_mode = "live" if mode == "live" else "paper"
    is_paper = selected_mode == "paper"
    capped_hours = max(1, min(int(hours or 24), 24 * 30))
    capped_limit = max(50, min(int(limit or 500), 2000))
    since = max(datetime.now(UTC) - timedelta(hours=capped_hours), PHASE3_CLEAN_START_UTC)

    async with get_session_ctx() as session:
        stmt = (
            select(AIDecision)
            .where(
                AIDecision.model_name == ENSEMBLE_TRADER_NAME,
                AIDecision.is_paper == is_paper,
                AIDecision.created_at >= since,
                AIDecision.analysis_type.in_(("market", "entry_candidate")),
            )
            .order_by(AIDecision.created_at.desc())
            .limit(capped_limit)
        )
        result = await session.execute(stmt)
        rows = list(result.scalars().all())
        decision_ids = [row.id for row in rows]

        order_map: dict[int, Order] = {}
        if decision_ids:
            order_result = await session.execute(
                select(Order)
                .where(Order.decision_id.in_(decision_ids))
                .order_by(Order.created_at.desc())
            )
            for order in order_result.scalars().all():
                if order.decision_id is not None and order.decision_id not in order_map:
                    order_map[order.decision_id] = order

    market_rows = []
    repair_cleanup_rows = 0
    for row in rows:
        raw = row.raw_llm_response if isinstance(row.raw_llm_response, dict) else {}
        analysis_type = str(row.analysis_type or raw.get("analysis_type") or "").lower()
        if analysis_type == "position" or str(row.action or "").lower() in {
            "close_long",
            "close_short",
        }:
            continue
        if _opening_funnel_is_repair_cleanup(row):
            repair_cleanup_rows += 1
            continue
        market_rows.append(row)

    action_counts = {"hold": 0, "long": 0, "short": 0, "other": 0}
    reason_buckets = {
        "profit_expectancy": 0,
        "evidence_gate": 0,
        "risk_or_precheck": 0,
        "waiting_queue": 0,
        "execution_or_exchange": 0,
        "ai_budget": 0,
        "other": 0,
        "unknown": 0,
    }
    symbol_counts: dict[str, dict[str, int]] = {}
    recent_blocked: list[dict[str, Any]] = []

    entry_signals = 0
    orders_created = 0
    executed_entries = 0
    no_order_after_signal = 0
    confidence_total = 0.0
    confidence_count = 0

    for row in market_rows:
        action = str(row.action or "").lower()
        action_key = action if action in {"hold", "long", "short"} else "other"
        action_counts[action_key] += 1
        symbol = _normalize_dashboard_symbol(row.symbol)
        symbol_state = symbol_counts.setdefault(
            symbol or "-", {"scans": 0, "signals": 0, "executed": 0}
        )
        symbol_state["scans"] += 1
        try:
            confidence_total += float(row.confidence or 0.0)
            confidence_count += 1
        except (TypeError, ValueError):
            pass

        if action not in {"long", "short"}:
            continue

        entry_signals += 1
        symbol_state["signals"] += 1
        matched_order = order_map.get(row.id)
        if matched_order is not None:
            orders_created += 1
        if row.was_executed:
            executed_entries += 1
            symbol_state["executed"] += 1
            continue
        if matched_order is None:
            no_order_after_signal += 1

        reason = _display_execution_reason(row, matched_order)
        bucket = _opening_funnel_reason_bucket(reason)
        reason_buckets[bucket] += 1
        if len(recent_blocked) < 20:
            recent_blocked.append(
                sanitize_payload(
                    {
                        "id": row.id,
                        "created_at": row.created_at.isoformat() if row.created_at else None,
                        "symbol": symbol,
                        "action": action,
                        "confidence": row.confidence,
                        "reason_bucket": bucket,
                        "reason": reason or "未保存具体未执行原因。",
                        "has_order": matched_order is not None,
                    }
                )
            )

    total_scans = len(market_rows)
    hold_count = action_counts["hold"]
    signal_rate = (entry_signals / total_scans) if total_scans else 0.0
    order_rate = (orders_created / entry_signals) if entry_signals else 0.0
    execution_rate = (executed_entries / entry_signals) if entry_signals else 0.0
    overall_open_rate = (executed_entries / total_scans) if total_scans else 0.0

    bottleneck = "no_data"
    bottleneck_label = "暂无足够数据"
    if total_scans:
        if signal_rate < 0.08:
            bottleneck = "ai_hold"
            bottleneck_label = "AI 主要选择观望"
        elif entry_signals and order_rate < 0.5:
            top_bucket = max(reason_buckets.items(), key=lambda item: item[1])[0]
            bottleneck = top_bucket
            label_map = {
                "profit_expectancy": "收益期望不足，费后预期净收益未转正",
                "evidence_gate": "动态证据强冲突硬拦",
                "risk_or_precheck": "风控或入场预检未通过",
                "waiting_queue": "动态证据不足，观望等待",
                "execution_or_exchange": "执行层或交易所接口问题",
                "ai_budget": "AI 成本预算仍在拦截",
                "other": "未执行原因分散",
                "unknown": "缺少未执行原因",
            }
            bottleneck_label = label_map.get(top_bucket, "开仓信号未形成订单")
        elif execution_rate < 0.75:
            bottleneck = "execution"
            bottleneck_label = "订单生成后执行/成交不足"
        else:
            bottleneck = "healthy_selective"
            bottleneck_label = "漏斗正常，开仓少主要来自选择性交易"

    top_symbols = sorted(
        (
            {
                "symbol": symbol,
                "scans": counts["scans"],
                "signals": counts["signals"],
                "executed": counts["executed"],
                "signal_rate": counts["signals"] / counts["scans"] if counts["scans"] else 0.0,
                "execution_rate": (
                    counts["executed"] / counts["signals"] if counts["signals"] else 0.0
                ),
            }
            for symbol, counts in symbol_counts.items()
        ),
        key=lambda item: (item["signals"], item["scans"]),
        reverse=True,
    )[:12]

    return sanitize_payload(
        {
            "mode": selected_mode,
            "window_hours": capped_hours,
            "sample_limit": capped_limit,
            "sampled_decisions": len(rows),
            "repair_cleanup_rows": repair_cleanup_rows,
            "market_scans": total_scans,
            "average_confidence": (
                (confidence_total / confidence_count) if confidence_count else 0.0
            ),
            "stages": {
                "market_scans": total_scans,
                "ai_entry_signals": entry_signals,
                "orders_created": orders_created,
                "executed_entries": executed_entries,
            },
            "rates": {
                "signal_rate": signal_rate,
                "order_rate": order_rate,
                "execution_rate": execution_rate,
                "overall_open_rate": overall_open_rate,
            },
            "action_counts": action_counts,
            "reason_buckets": reason_buckets,
            "hold_count": hold_count,
            "no_order_after_signal": no_order_after_signal,
            "bottleneck": bottleneck,
            "bottleneck_label": bottleneck_label,
            "top_symbols": top_symbols,
            "recent_blocked": recent_blocked,
            "generated_at": datetime.now(UTC).isoformat(),
        }
    )


@router.get("/analysis-records")
async def get_analysis_records(
    limit: int = 50,
    page: int = 1,
    page_size: int | None = None,
    decision_id: int | None = None,
    analysis_type: str | None = None,
    include_detail: bool = True,
    symbol: str | None = None,
    expert_name: str | None = None,
    is_paper: bool | None = None,
):
    """Return one collaboration record per ensemble decision."""
    from sqlalchemy import func, select

    from db.repositories.decision_repo import DecisionRepository
    from db.session import get_session_ctx
    from models.decision import AIDecision
    from models.trade import Position

    effective_page_size = page_size if page_size is not None else limit
    effective_page_size = max(1, min(int(effective_page_size or 50), 200))
    page = max(int(page or 1), 1)
    offset = (page - 1) * effective_page_size
    normalized_analysis_type = str(analysis_type or "").lower()
    if normalized_analysis_type not in {"market", "position"}:
        normalized_analysis_type = ""
    selected_mode = (
        "paper" if is_paper is True else "live" if is_paper is False else mode_manager.mode.value
    )
    current_position_symbols: set[str] = set()
    if normalized_analysis_type == "position" and decision_id is None:
        current_position_symbols = await _get_display_open_position_symbols(selected_mode)

    async with get_session_ctx() as session:
        repo = DecisionRepository(session)
        needs_server_filter = bool(normalized_analysis_type or expert_name)
        db_type_filter = bool(normalized_analysis_type and not expert_name and decision_id is None)
        total_without_server_filter = 0
        if decision_id is not None:
            decision_stmt = select(AIDecision).where(
                AIDecision.id == decision_id,
                AIDecision.model_name == ENSEMBLE_TRADER_NAME,
            )
            if is_paper is not None:
                decision_stmt = decision_stmt.where(AIDecision.is_paper == is_paper)
            decision_result = await session.execute(decision_stmt)
            all_rows = list(decision_result.scalars().all())
        elif db_type_filter:
            filters = [
                AIDecision.model_name == ENSEMBLE_TRADER_NAME,
                AIDecision.analysis_type == normalized_analysis_type,
            ]
            if symbol:
                filters.append(AIDecision.symbol == symbol)
            elif normalized_analysis_type == "position":
                if current_position_symbols:
                    filters.append(AIDecision.symbol.in_(current_position_symbols))
                else:
                    filters.append(AIDecision.id == -1)
            if is_paper is not None:
                filters.append(AIDecision.is_paper == is_paper)
            decision_stmt = (
                select(AIDecision)
                .where(*filters)
                .order_by(AIDecision.created_at.desc())
                .offset(offset)
                .limit(effective_page_size)
            )
            decision_result = await session.execute(decision_stmt)
            all_rows = list(decision_result.scalars().all())
            count_result = await session.execute(select(func.count(AIDecision.id)).where(*filters))
            total_without_server_filter = int(count_result.scalar() or 0)
        else:
            all_rows = await repo.get_recent_decisions(
                model_name=ENSEMBLE_TRADER_NAME,
                symbol=symbol,
                limit=5000 if needs_server_filter else effective_page_size,
                offset=0 if needs_server_filter else offset,
                is_paper=is_paper,
            )
        if decision_id is not None:
            total_without_server_filter = len(all_rows)
        elif db_type_filter:
            pass
        elif not needs_server_filter:
            total_without_server_filter = await repo.count_decisions(
                model_name=ENSEMBLE_TRADER_NAME,
                is_paper=is_paper,
            )
        position_stmt = select(
            Position.execution_mode,
            Position.symbol,
        ).where(
            Position.model_name.in_(EXECUTION_LEDGER_MODEL_NAMES),
            Position.is_open.is_(True),
        )
        if is_paper is not None:
            position_stmt = position_stmt.where(
                Position.execution_mode == ("paper" if is_paper else "live")
            )
        position_result = await session.execute(position_stmt)
        open_position_keys = {
            (row.execution_mode, _normalize_dashboard_symbol(row.symbol))
            for row in position_result.all()
            if _normalize_dashboard_symbol(row.symbol)
        }

    def infer_analysis_type(decision, raw: dict) -> tuple[str, str]:
        explicit = str(raw.get("analysis_type") or "").lower()
        if explicit in {"position", "position_review", "holding", "holdings"}:
            return "position", "持仓分析"
        if explicit in {"market", "market_scan", "symbol_scan"}:
            return "market", "市场分析"

        if raw.get("position_review_policy") or raw.get("position_review"):
            return "position", "持仓分析"

        if decision.action in {"close_long", "close_short"}:
            return "position", "持仓分析"

        return "market", "市场分析"

    def infer_position_lifecycle(decision, analysis_type: str) -> tuple[str | None, str | None]:
        if analysis_type != "position":
            return None, None

        decision_mode = "paper" if decision.is_paper else "live"
        symbol_key = _normalize_dashboard_symbol(decision.symbol)
        if (decision_mode, symbol_key) in open_position_keys:
            return "holding", "持仓中"
        return "closed", "已平仓"

    records: list[dict[str, Any]] = []
    filtered_count = 0
    for d in all_rows:
        raw = _safe_dict(d.raw_llm_response)
        if not raw:
            continue
        opinions = raw.get("opinions") or []
        if not isinstance(opinions, list):
            continue
        model_timings = raw.get("model_timings") or []
        if not isinstance(model_timings, list):
            model_timings = []
        timings_by_name = {
            str(item.get("name")): item
            for item in model_timings
            if isinstance(item, dict) and item.get("name")
        }

        experts = []
        for opinion in opinions:
            if not isinstance(opinion, dict):
                continue
            name = opinion.get("model_name") or ""
            experts.append(
                {
                    "expert_name": name,
                    "expert_label": opinion.get("label") or name,
                    "role": opinion.get("role") or "",
                    "action": opinion.get("action") or "hold",
                    "confidence": opinion.get("confidence") or 0.0,
                    "weight": opinion.get("weight") or 0.0,
                    "reasoning": opinion.get("reasoning") or "",
                    "cross_check_for": opinion.get("cross_check_for"),
                    "timeout_fallback": bool(opinion.get("timeout_fallback")),
                    "latency": timings_by_name.get(str(name)),
                }
            )

        if expert_name and not any(e["expert_name"] == expert_name for e in experts):
            continue

        cross_validations = raw.get("cross_validations") or []
        consultation = raw.get("consultation")
        conflict_resolution = raw.get("conflict_resolution") or {}
        attempted_experts = raw.get("attempted_experts") or []
        failure_rows = raw.get("expert_failures") or []
        if not isinstance(attempted_experts, list):
            attempted_experts = []
        if not isinstance(failure_rows, list):
            failure_rows = []
        failures_by_name = {
            str(item.get("expert_name")): _humanize_expert_failure(item.get("reason"))
            for item in failure_rows
            if isinstance(item, dict) and item.get("expert_name")
        }
        cross_requested = sum(1 for e in experts if e.get("cross_check_for"))
        divergent = sum(1 for v in cross_validations if v.get("consistency") == "divergent")
        aligned = sum(1 for v in cross_validations if v.get("consistency") == "aligned")
        major_conflicts = sum(1 for v in cross_validations if v.get("major_conflict"))
        unavailable_validations = sum(
            1 for v in cross_validations if v.get("validation_status") == "target_missing"
        )
        completed_validations = sum(
            1 for v in cross_validations if v.get("validation_status", "completed") == "completed"
        )

        expected_experts = [
            {
                "expert_name": slot.get("name", ""),
                "expert_label": slot.get("label", slot.get("name", "")),
                "role": slot.get("role", ""),
            }
            for slot in FIXED_AI_MODEL_SLOTS
            if slot.get("name") != DECISION_MAKER_NAME
        ]
        returned_names = {e["expert_name"] for e in experts}
        attempted_names = {str(name) for name in attempted_experts}
        fast_scan_payload = _safe_dict(raw.get("position_fast_scan"))
        pre_expert_skip = _analysis_pre_expert_skip(raw)
        attempted_expert_count = (
            0 if pre_expert_skip.get("skipped") else (len(attempted_names) or len(expected_experts))
        )
        missing_experts = [
            {
                **e,
                "latency": timings_by_name.get(e["expert_name"]),
                "reason": (
                    pre_expert_skip.get("reason")
                    if pre_expert_skip.get("skipped")
                    else failures_by_name.get(e["expert_name"])
                    or (
                        "未发起调用，可能是该专家未启用或未配置 API Key。"
                        if attempted_names and e["expert_name"] not in attempted_names
                        else "本轮未返回结果，可能是模型调用失败、超时或返回格式不符合 JSON 要求。"
                    )
                ),
                "status": ("pre_expert_skipped" if pre_expert_skip.get("skipped") else "missing"),
                "skip_kind": pre_expert_skip.get("kind") or "",
            }
            for e in expected_experts
            if e["expert_name"] not in returned_names
        ]
        trade_confidence = _clamp_confidence(_safe_float(d.confidence, 0.0))
        display_confidence = _analysis_display_confidence(d.action, trade_confidence, experts, raw)
        analysis_type, analysis_type_label = infer_analysis_type(d, raw)
        position_lifecycle_status, position_lifecycle_label = infer_position_lifecycle(
            d, analysis_type
        )
        if normalized_analysis_type and analysis_type != normalized_analysis_type:
            continue
        if (
            normalized_analysis_type == "position"
            and decision_id is None
            and not symbol
            and _normalize_dashboard_symbol(d.symbol) not in current_position_symbols
        ):
            continue
        if expert_name and not any(e["expert_name"] == expert_name for e in experts):
            continue

        filtered_count += 1
        if needs_server_filter and not db_type_filter:
            if filtered_count <= offset:
                continue
            if len(records) >= effective_page_size:
                continue

        local_ai_tools_payload = _normalized_local_ai_tools_payload(raw)
        display_execution_reason = _display_execution_reason(d)
        vector_memory_context = (
            await get_vector_memory_service().similar_decision_context(d, raw)
            if include_detail and settings.vector_memory_enabled
            else {"enabled": False, "status": "disabled", "hits": []}
        )
        detail_payload = (
            {
                "experts": experts,
                "missing_experts": missing_experts,
                "cross_validations": cross_validations,
                "consultation": consultation,
                "decision_maker": (
                    raw.get("decision_maker")
                    if isinstance(raw.get("decision_maker"), dict)
                    else None
                ),
                "decision_attribution": _build_decision_attribution(d, raw, experts),
                "ml_signal": (
                    raw.get("ml_signal") if isinstance(raw.get("ml_signal"), dict) else None
                ),
                "local_ai_tools": local_ai_tools_payload,
                "agent_skills": (
                    raw.get("agent_skills") if isinstance(raw.get("agent_skills"), dict) else None
                ),
                "news_context": (
                    raw.get("news_context") if isinstance(raw.get("news_context"), dict) else None
                ),
                "opportunity_score": _display_opportunity_score(
                    d,
                    raw,
                    display_execution_reason,
                ),
                "position_review_policy": (
                    raw.get("position_review_policy")
                    if isinstance(raw.get("position_review_policy"), dict)
                    else {}
                ),
                "position_fast_scan": fast_scan_payload,
                "close_evidence": _safe_dict(raw.get("close_evidence")),
                "add_evidence": _safe_dict(raw.get("add_evidence")),
                "timing": _safe_dict(raw.get("timing")),
                "timing_breakdown": (
                    raw.get("timing_breakdown")
                    if isinstance(raw.get("timing_breakdown"), list)
                    else []
                ),
                "model_timings": model_timings,
                "latency_summary": (
                    raw.get("latency_summary")
                    if isinstance(raw.get("latency_summary"), dict)
                    else {}
                ),
                "conflict_resolution": (
                    conflict_resolution if isinstance(conflict_resolution, dict) else {}
                ),
                "vector_memory": vector_memory_context,
            }
            if include_detail
            else {}
        )

        record = {
            "id": str(d.id),
            "decision_id": d.id,
            "analysis_type": d.analysis_type or analysis_type,
            "analysis_type_label": (
                "持仓分析" if (d.analysis_type or analysis_type) == "position" else "市场分析"
            ),
            "position_lifecycle_status": position_lifecycle_status,
            "position_lifecycle_label": position_lifecycle_label,
            "created_at": d.created_at.isoformat() if d.created_at else None,
            "symbol": _normalize_dashboard_symbol(d.symbol),
            "expert_count": len(experts),
            "expected_expert_count": len(expected_experts),
            "attempted_expert_count": attempted_expert_count,
            "attempted_experts": attempted_experts,
            "expert_call_status": pre_expert_skip,
            "position_fast_scan": fast_scan_payload,
            "cross_requested": cross_requested,
            "cross_summary": {
                "total": len(cross_validations),
                "completed": completed_validations,
                "unavailable": unavailable_validations,
                "aligned": aligned,
                "divergent": divergent,
                "major_conflicts": major_conflicts,
            },
            "consultation_status": (
                consultation.get("status") if isinstance(consultation, dict) else None
            ),
            "validation_adjustment": raw.get("validation_adjustment"),
            "final_action": d.action,
            "final_confidence": display_confidence,
            "trade_confidence": trade_confidence,
            "confidence_note": (
                "观望/不下单时，信心度显示专家加权平均分析信心；内部下单信心仍为 0，避免误触发交易。"
                if d.action == "hold" and trade_confidence == 0.0
                else "信心度来自最终可执行裁决。"
            ),
            "final_reasoning": sanitize_text(d.reasoning),
            "position_size_pct": d.position_size_pct,
            "weighted_score": raw.get("weighted_score"),
            "disagreement": raw.get("disagreement"),
            "was_executed": d.was_executed,
            "execution_reason": display_execution_reason,
            "is_paper": d.is_paper,
            "flow_summary": (
                f"{pre_expert_skip.get('label')}：{pre_expert_skip.get('reason')}"
                if pre_expert_skip.get("skipped")
                else (
                    f"{len(experts)}/{len(expected_experts)} 个专家返回，"
                    f"{cross_requested} 个交叉验证请求，"
                    f"{completed_validations} 次完成，"
                    f"{unavailable_validations} 次无法验证，"
                    f"{major_conflicts} 个重大矛盾"
                )
            ),
        }
        record.update(detail_payload)
        records.append(sanitize_payload(record))

    total = (
        total_without_server_filter
        if db_type_filter
        else (filtered_count if needs_server_filter else total_without_server_filter)
    )
    total_pages = max(1, (total + effective_page_size - 1) // effective_page_size) if total else 1

    return {
        "records": records,
        "count": len(records),
        "total": total,
        "page": page,
        "page_size": effective_page_size,
        "total_pages": total_pages,
    }


@router.get("/strategy-learning")
async def get_strategy_learning(
    mode: str | None = None,
    hours: int = STRATEGY_LEARNING_PARAMS.default_lookback_hours,
    limit: int = STRATEGY_LEARNING_PARAMS.dashboard_default_limit,
    detail: str = "summary",
):
    """Return the active strategy-learning feedback and scheduler state."""
    from services.strategy_learning import StrategyLearningService

    selected_mode = "live" if str(mode or "").lower() == "live" else "paper"
    selected_detail = "full" if str(detail or "").lower() == "full" else "summary"
    strategy_params = STRATEGY_LEARNING_PARAMS
    capped_hours = max(
        1,
        min(
            int(hours or strategy_params.default_lookback_hours), strategy_params.max_lookback_hours
        ),
    )
    max_limit = (
        strategy_params.dashboard_full_limit
        if selected_detail == "full"
        else strategy_params.dashboard_summary_limit
    )
    capped_limit = max(
        strategy_params.min_dashboard_limit,
        min(int(limit or strategy_params.dashboard_default_limit), max_limit),
    )
    max_open_positions = int(settings.max_open_positions_per_model or 20)
    cache_key = (
        "strategy-learning",
        selected_mode,
        capped_hours,
        capped_limit,
        max_open_positions,
        selected_detail,
    )
    cached = _dashboard_heavy_cache_get(cache_key)
    if cached is not None:
        return sanitize_payload(cached)
    service = getattr(_trading_service, "strategy_learning_service", None)
    if service is None:
        service = StrategyLearningService()
    payload = await service.dashboard_payload(
        mode=selected_mode,
        hours=capped_hours,
        limit=capped_limit,
        max_open_positions=max_open_positions,
        detail=selected_detail,
    )
    return sanitize_payload(_dashboard_heavy_cache_set(cache_key, payload))


@router.post("/strategy-learning/profiles/{profile_id}/disabled")
async def set_strategy_learning_profile_disabled(
    profile_id: str,
    disabled: bool = True,
    reason: str | None = None,
):
    """Disable or re-enable one strategy profile."""
    from services.strategy_learning import StrategyLearningService

    service = getattr(_trading_service, "strategy_learning_service", None)
    if service is None:
        service = StrategyLearningService()
    state = service.set_profile_disabled(
        profile_id,
        disabled=bool(disabled),
        reason=reason or "dashboard_manual_control",
    )
    _clear_dashboard_heavy_cache("strategy-learning")
    return sanitize_payload({"profile_id": profile_id, "disabled": bool(disabled), "state": state})


@router.post("/strategy-learning/profiles/{profile_id}/activate")
async def activate_strategy_learning_profile(profile_id: str):
    """Manually select a strategy profile until rollback or another selection."""
    from services.strategy_learning import StrategyLearningService

    service = getattr(_trading_service, "strategy_learning_service", None)
    if service is None:
        service = StrategyLearningService()
    state = service.set_manual_active_profile(profile_id)
    _clear_dashboard_heavy_cache("strategy-learning")
    return sanitize_payload({"profile_id": profile_id, "state": state})


@router.post("/strategy-learning/rollback")
async def rollback_strategy_learning_profile():
    """Clear manual profile selection so the scheduler resumes automatic switching."""
    from services.strategy_learning import StrategyLearningService

    service = getattr(_trading_service, "strategy_learning_service", None)
    if service is None:
        service = StrategyLearningService()
    state = service.rollback_to_baseline()
    _clear_dashboard_heavy_cache("strategy-learning")
    return sanitize_payload({"profile_id": "auto", "state": state})


@router.get("/profit-attribution")
async def get_profit_attribution(
    mode: str | None = None,
    hours: int = 24,
    limit: int = 200,
):
    """Explain why recent closed trades made or lost money."""
    from sqlalchemy import or_, select

    from db.session import get_session_ctx
    from models.decision import AIDecision
    from models.learning import ShadowBacktest
    from models.trade import Order, Position
    from services.position_settlement import final_settlement_status_values
    from services.profit_attribution import (
        build_profit_attribution,
        match_entry_decisions_for_positions,
    )

    selected_mode = "live" if str(mode or "").lower() == "live" else "paper"
    is_paper = selected_mode == "paper"
    capped_hours = max(1, min(int(hours or 24), 720))
    max_rows = max(20, min(int(limit or 200), 1000))
    since = datetime.now(UTC) - timedelta(hours=capped_hours)
    cache_key = ("profit-attribution", selected_mode, capped_hours, max_rows)
    cached = _dashboard_heavy_cache_get(cache_key)
    if cached is not None:
        return sanitize_payload(cached)

    async with get_session_ctx() as session:
        position_result = await session.execute(
            select(Position)
            .where(
                Position.model_name.in_(EXECUTION_LEDGER_MODEL_NAMES),
                Position.execution_mode == selected_mode,
                Position.is_open.is_(False),
                Position.settlement_status.in_(final_settlement_status_values()),
                Position.closed_at.is_not(None),
                Position.closed_at >= since,
            )
            .order_by(Position.closed_at.desc(), Position.created_at.desc())
            .limit(max_rows)
        )
        positions = list(position_result.scalars().all())
        if not positions:
            empty_payload = {
                "mode": selected_mode,
                "window_hours": capped_hours,
                "summary": {
                    "trade_count": 0,
                    "total_closed_pnl": 0.0,
                    "win_count": 0,
                    "loss_count": 0,
                    "win_rate": 0.0,
                    "avg_win": 0.0,
                    "avg_loss": 0.0,
                    "profit_factor": 0.0,
                },
                "buckets": [],
                "records": [],
                "message": "最近窗口内暂无已平仓记录。",
            }
            return sanitize_payload(_dashboard_heavy_cache_set(cache_key, empty_payload))

        symbols = {p.symbol for p in positions if p.symbol}
        symbol_variants = _dashboard_symbol_query_variants(symbols)
        position_times = [
            normalized
            for p in positions
            if p.created_at and (normalized := _as_utc_datetime(p.created_at)) is not None
        ]
        earliest_position_time = min(position_times, default=since)
        order_since = min(since, earliest_position_time or since) - timedelta(hours=2)
        order_limit = max(max_rows * 20, 2000)
        order_result = await session.execute(
            select(Order)
            .where(
                Order.model_name.in_(EXECUTION_LEDGER_MODEL_NAMES),
                Order.execution_mode == selected_mode,
                Order.symbol.in_(symbol_variants) if symbol_variants else Order.id == -1,
                Order.created_at >= order_since,
            )
            .order_by(Order.filled_at.desc().nullslast(), Order.created_at.desc())
            .limit(order_limit)
        )
        orders = list(order_result.scalars().all())

        decision_ids = {int(o.decision_id) for o in orders if o.decision_id}
        decisions_by_id = {}
        if decision_ids:
            decision_result = await session.execute(
                select(AIDecision)
                .where(AIDecision.id.in_(decision_ids))
                .order_by(AIDecision.created_at.desc())
            )
            decisions_by_id.update(
                {int(row.id): row for row in decision_result.scalars().all() if row.id is not None}
            )
        if symbol_variants:
            fallback_decision_result = await session.execute(
                select(AIDecision)
                .where(
                    AIDecision.model_name == ENSEMBLE_TRADER_NAME,
                    AIDecision.symbol.in_(symbol_variants),
                    AIDecision.is_paper.is_(is_paper),
                    AIDecision.created_at >= order_since,
                )
                .order_by(AIDecision.created_at.desc())
                .limit(max(max_rows * 12, 1200))
            )
            decisions_by_id.update(
                {
                    int(row.id): row
                    for row in fallback_decision_result.scalars().all()
                    if row.id is not None and int(row.id) not in decisions_by_id
                }
            )
        decisions = list(decisions_by_id.values())

        entry_decisions = match_entry_decisions_for_positions(positions, orders, decisions)
        shadow_decision_ids = {int(d.id) for d in entry_decisions if d.id is not None}
        shadows_by_id = {}
        if shadow_decision_ids:
            shadow_result = await session.execute(
                select(ShadowBacktest)
                .where(ShadowBacktest.decision_id.in_(shadow_decision_ids))
                .order_by(ShadowBacktest.created_at.desc())
                .limit(max(len(shadow_decision_ids) * 4, max_rows * 10))
            )
            shadows_by_id.update(
                {int(row.id): row for row in shadow_result.scalars().all() if row.id is not None}
            )
        shadow_time_conditions = []
        for decision in entry_decisions:
            decision_time = _as_utc_datetime(decision.created_at)
            if not decision.symbol or not decision_time:
                continue
            decision_symbol_variants = _dashboard_symbol_query_variants({decision.symbol})
            shadow_time_conditions.append(
                (ShadowBacktest.symbol.in_(decision_symbol_variants))
                & (ShadowBacktest.execution_mode == selected_mode)
                & (ShadowBacktest.created_at >= decision_time - timedelta(minutes=30))
                & (ShadowBacktest.created_at <= decision_time + timedelta(minutes=30))
            )
        for position in positions:
            opened_at = _as_utc_datetime(position.created_at)
            if not position.symbol or not opened_at:
                continue
            position_symbol_variants = _dashboard_symbol_query_variants({position.symbol})
            shadow_time_conditions.append(
                (ShadowBacktest.symbol.in_(position_symbol_variants))
                & (ShadowBacktest.execution_mode == selected_mode)
                & (ShadowBacktest.created_at >= opened_at - timedelta(minutes=45))
                & (ShadowBacktest.created_at <= opened_at + timedelta(minutes=45))
            )
        if shadow_time_conditions:
            fallback_shadow_result = await session.execute(
                select(ShadowBacktest)
                .where(or_(*shadow_time_conditions))
                .order_by(ShadowBacktest.created_at.desc())
            )
            shadows_by_id.update(
                {
                    int(row.id): row
                    for row in fallback_shadow_result.scalars().all()
                    if row.id is not None and int(row.id) not in shadows_by_id
                }
            )
        shadows = list(shadows_by_id.values())

    payload = build_profit_attribution(positions, orders, decisions, shadows)
    result_payload = {
        "mode": selected_mode,
        "window_hours": capped_hours,
        "sample_limit": max_rows,
        "since": since.isoformat(),
        **payload,
        "message": "按已平仓真实盈亏，结合 AI 决策、订单、影子复盘和本地模型证据做交易级归因。",
    }
    return sanitize_payload(_dashboard_heavy_cache_set(cache_key, result_payload))


@router.get("/model-contribution/stats")
async def get_model_contribution_stats(
    mode: str | None = None,
    days: int = 7,
    limit: int = 2000,
):
    """Estimate which model signals helped or hurt realized PnL."""
    from sqlalchemy import select

    from db.session import get_session_ctx
    from models.decision import AIDecision
    from models.trade import Order, Position
    from services.model_contribution_performance import ModelContributionPerformanceService

    selected_mode = "live" if str(mode or "").lower() == "live" else "paper"
    since = datetime.now(UTC) - timedelta(days=max(1, min(int(days or 7), 90)))
    max_rows = max(50, min(int(limit or 2000), 10000))

    buckets: dict[str, dict[str, Any]] = {
        "local_ml_aligned": {"label": "本地 ML 同向"},
        "server_profit_aligned": {"label": "服务器盈利模型同向"},
        "timeseries_aligned": {"label": "时序预测同向"},
        "sentiment_aligned": {"label": "情绪预测同向"},
        "ai_only_ml_opposed": {"label": "AI 支持但 ML 反对"},
        "high_risk_review_approved": {"label": "高风险复核通过"},
    }
    stats: dict[str, dict[str, Any]] = {
        key: {
            **meta,
            "count": 0,
            "wins": 0,
            "losses": 0,
            "pnl": 0.0,
            "profit": 0.0,
            "loss": 0.0,
        }
        for key, meta in buckets.items()
    }

    def add(key: str, pnl: float) -> None:
        bucket = stats.get(key)
        if not bucket:
            return
        bucket["count"] += 1
        bucket["pnl"] += pnl
        if pnl >= 0:
            bucket["wins"] += 1
            bucket["profit"] += pnl
        else:
            bucket["losses"] += 1
            bucket["loss"] += abs(pnl)

    async with get_session_ctx() as session:
        from services.position_settlement import final_settlement_status_values

        position_result = await session.execute(
            select(Position)
            .where(
                Position.model_name.in_(EXECUTION_LEDGER_MODEL_NAMES),
                Position.execution_mode == selected_mode,
                Position.is_open.is_(False),
                Position.settlement_status.in_(final_settlement_status_values()),
                Position.closed_at.is_not(None),
                Position.closed_at >= since,
            )
            .order_by(Position.closed_at.desc())
            .limit(max_rows)
        )
        positions = list(position_result.scalars().all())
        if not positions:
            return {
                "mode": selected_mode,
                "days": days,
                "total_positions": 0,
                "lineage": {
                    "total_closed_positions": 0,
                    "filled_order_count": 0,
                    "orders_with_decision_id": 0,
                    "orders_with_loaded_decision": 0,
                    "matched_position_count": 0,
                    "unmatched_position_count": 0,
                    "match_rate": 0.0,
                    "reason": "no_closed_positions",
                    "ready_for_profit_learning": False,
                },
                "stats": list(stats.values()),
                "summary": "暂无已平仓样本，等新交易完成后会自动统计。",
            }

        symbols = {p.symbol for p in positions if p.symbol}
        symbol_variants = _dashboard_symbol_query_variants(symbols)
        order_result = await session.execute(
            select(Order)
            .where(
                Order.model_name.in_(EXECUTION_LEDGER_MODEL_NAMES),
                Order.execution_mode == selected_mode,
                Order.status == "filled",
                Order.symbol.in_(symbol_variants) if symbol_variants else Order.id == -1,
            )
            .order_by(Order.filled_at.desc(), Order.created_at.desc())
            .limit(max_rows * 3)
        )
        orders = list(order_result.scalars().all())
        decision_ids = list({int(o.decision_id) for o in orders if o.decision_id})
        decisions: dict[int, AIDecision] = {}
        if decision_ids:
            decision_result = await session.execute(
                select(AIDecision).where(AIDecision.id.in_(decision_ids))
            )
            decisions = {d.id: d for d in decision_result.scalars().all()}

    contribution_service = ModelContributionPerformanceService()
    lineage = contribution_service.build_lineage_diagnostics(positions, orders, decisions)

    def aware(value):
        if value and value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value

    for pos in positions:
        pos_created = aware(pos.created_at)
        pos_symbol = _normalize_dashboard_symbol(pos.symbol)
        candidates = []
        for order in orders:
            if _normalize_dashboard_symbol(order.symbol) != pos_symbol:
                continue
            if order.decision_id not in decisions:
                continue
            decision = decisions[order.decision_id]
            action_side = _side_from_action(decision.action)
            if action_side != str(pos.side or "").lower():
                continue
            order_time = aware(order.filled_at or order.created_at)
            if pos_created and order_time and abs((order_time - pos_created).total_seconds()) > 180:
                continue
            candidates.append(
                (
                    (
                        abs(((order_time or pos_created) - pos_created).total_seconds())
                        if pos_created and order_time
                        else 0
                    ),
                    decision,
                )
            )
        if not candidates:
            continue
        _, decision = sorted(candidates, key=lambda item: item[0])[0]
        raw = _safe_dict(decision.raw_llm_response)
        side = _side_from_action(decision.action)
        pnl = float(pos.realized_pnl or 0.0)
        ml = _extract_primary_ml(raw)
        local = _extract_local_tools(raw)
        if ml.get("side") == side:
            add("local_ml_aligned", pnl)
        elif (
            side in {"long", "short"}
            and ml.get("side") in {"long", "short"}
            and ml.get("side") != side
        ):
            add("ai_only_ml_opposed", pnl)
        if _safe_dict(local.get("profit")).get("side") == side:
            add("server_profit_aligned", pnl)
        if _safe_dict(local.get("timeseries")).get("side") == side:
            add("timeseries_aligned", pnl)
        if _safe_dict(local.get("sentiment")).get("side") == side:
            add("sentiment_aligned", pnl)
        review = _safe_dict(raw.get("high_risk_review"))
        if review.get("triggered") and review.get("approved") is True:
            add("high_risk_review_approved", pnl)

    result = []
    for item in stats.values():
        count = max(int(item["count"]), 0)
        item["pnl"] = round(float(item["pnl"]), 6)
        item["profit"] = round(float(item["profit"]), 6)
        item["loss"] = round(float(item["loss"]), 6)
        item["avg_pnl"] = round(item["pnl"] / count, 6) if count else 0.0
        item["win_rate"] = round(item["wins"] / count, 4) if count else 0.0
        item["profit_factor"] = (
            round(item["profit"] / item["loss"], 4)
            if item["loss"] > 0
            else (999.0 if item["profit"] > 0 else 0.0)
        )
        result.append(item)

    return {
        "mode": selected_mode,
        "days": days,
        "total_positions": len(positions),
        "lineage": lineage,
        "stats": result,
        "summary": "按已平仓真实盈亏回看各模型同向信号贡献，用于后续自动降权/加权。",
    }


@router.get("/expert-memories")
async def get_expert_memories(
    limit: int = 10,
    page_size: int = 10,
    memory_page: int = 1,
    reflection_page: int = 1,
    expert_name: str | None = None,
    symbol: str | None = None,
):
    """Return long-term expert memories and recent trade reflections."""
    from db.repositories.memory_repo import MemoryRepository
    from db.session import get_session_ctx

    size = max(min(int(page_size or limit or 10), 100), 1)
    memory_page = max(int(memory_page or 1), 1)
    reflection_page = max(int(reflection_page or 1), 1)
    memory_offset = (memory_page - 1) * size
    reflection_offset = (reflection_page - 1) * size

    async with get_session_ctx() as session:
        repo = MemoryRepository(session)
        memory_total = await repo.count_memories(
            expert_name=expert_name,
            symbol=symbol,
        )
        reflection_total = await repo.count_reflections(symbol=symbol)
        memories = await repo.list_memories(
            expert_name=expert_name,
            symbol=symbol,
            limit=size,
            offset=memory_offset,
        )
        reflections = await repo.list_reflections(
            symbol=symbol,
            limit=size,
            offset=reflection_offset,
        )

    memory_rows = [
        {
            "id": m.id,
            "expert_name": m.expert_name,
            "expert_label": m.expert_label,
            "symbol": m.symbol,
            "side": m.side,
            "memory_type": m.memory_type,
            "market_pattern": sanitize_text(m.market_pattern),
            "lesson": sanitize_text(m.lesson),
            "recommended_action": m.recommended_action,
            "confidence_adjustment": m.confidence_adjustment,
            "position_size_multiplier": m.position_size_multiplier,
            "evidence_count": m.evidence_count,
            "hit_count": m.hit_count,
            "success_count": m.success_count,
            "failure_count": m.failure_count,
            "confidence_score": m.confidence_score,
            "last_used_at": m.last_used_at.isoformat() if m.last_used_at else None,
            "created_at": m.created_at.isoformat() if m.created_at else None,
            "updated_at": m.updated_at.isoformat() if m.updated_at else None,
        }
        for m in memories
    ]
    reflection_rows = [
        {
            "id": r.id,
            "position_id": r.position_id,
            "symbol": r.symbol,
            "side": r.side,
            "entry_price": r.entry_price,
            "exit_price": r.exit_price,
            "quantity": r.quantity,
            "realized_pnl": r.realized_pnl,
            "fee_estimate": r.fee_estimate,
            "hold_minutes": r.hold_minutes,
            "closed_at": r.closed_at.isoformat() if r.closed_at else None,
            "outcome": r.outcome,
            "mistake_summary": sanitize_text(r.mistake_summary),
            "improvement_summary": sanitize_text(r.improvement_summary),
            "source": r.source,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in reflections
    ]
    return {
        "memories": memory_rows,
        "reflections": reflection_rows,
        "count": memory_total,
        "reflection_count": reflection_total,
        "pagination": {
            "page_size": size,
            "memory_page": memory_page,
            "memory_total": memory_total,
            "memory_total_pages": max((memory_total + size - 1) // size, 1),
            "reflection_page": reflection_page,
            "reflection_total": reflection_total,
            "reflection_total_pages": max((reflection_total + size - 1) // size, 1),
        },
        "daily_target": _daily_target_payload(),
    }


@router.get("/shadow-backtests")
async def get_shadow_backtests(
    limit: int = 10,
    page_size: int = 10,
    page: int = 1,
    status: str | None = None,
    symbol: str | None = None,
):
    """Return shadow backtest records shown in the dashboard."""
    from db.repositories.memory_repo import MemoryRepository
    from db.session import get_session_ctx

    size = max(min(int(page_size or limit or 10), 100), 1)
    page = max(int(page or 1), 1)
    offset = (page - 1) * size
    status_filter: str | None = str(status or "").strip().lower()
    if status_filter not in {"pending", "completed"}:
        status_filter = None
    cache_key = ("shadow-backtests", size, page, status_filter or "", symbol or "")
    cached = _dashboard_heavy_cache_get(cache_key)
    if cached is not None:
        return sanitize_payload(cached)

    async with get_session_ctx() as session:
        repo = MemoryRepository(session)
        total = await repo.count_shadow_backtests(status=status_filter, symbol=symbol)
        pending_total = await repo.count_shadow_backtests(status="pending", symbol=symbol)
        completed_total = await repo.count_shadow_backtests(status="completed", symbol=symbol)
        rows = await repo.list_shadow_backtests(
            status=status_filter,
            symbol=symbol,
            limit=size,
            offset=offset,
        )

    records: list[dict[str, Any]] = []
    for row in rows:
        long_return = _safe_float(row.long_return_pct, None)
        short_return = _safe_float(row.short_return_pct, None)
        best_action = row.best_action or None
        raw = _safe_dict(row.raw_llm_response)
        decision_maker = _safe_dict(raw.get("decision_maker"))
        decision_note = ""
        if str(row.decision_action or "").lower() == "hold":
            decision_note = str(decision_maker.get("reason") or "").strip()
            if not decision_note:
                weighted_score = _safe_float(raw.get("weighted_score"), 0.0) or 0.0
                if abs(weighted_score) < 1e-9 and _safe_float(row.decision_confidence, 0.0) <= 0:
                    decision_note = "当时没有形成可执行开仓信号，系统记录为观望样本。"
                else:
                    decision_note = "当时最终裁决为观望，用于复盘是否错过机会。"
        records.append(
            {
                "id": row.id,
                "decision_id": row.decision_id,
                "model_name": row.model_name,
                "execution_mode": row.execution_mode,
                "symbol": row.symbol,
                "analysis_type": row.analysis_type,
                "decision_action": row.decision_action,
                "decision_action_label": _action_label_text(row.decision_action),
                "decision_confidence": row.decision_confidence,
                "decision_note": decision_note,
                "decision_maker_status": decision_maker.get("status") or "",
                "entry_price": row.entry_price,
                "status": row.status,
                "status_label": "已完成" if row.status == "completed" else "等待复盘",
                "due_at": row.due_at.isoformat() if row.due_at else None,
                "horizon_minutes": row.horizon_minutes,
                "actual_price": row.actual_price,
                "long_return_pct": long_return,
                "short_return_pct": short_return,
                "best_action": best_action,
                "best_action_label": _action_label_text(best_action),
                "missed_opportunity": bool(row.missed_opportunity),
                "conclusion": _shadow_backtest_conclusion(
                    row.decision_action,
                    best_action,
                    bool(row.missed_opportunity),
                    long_return,
                    short_return,
                ),
                "note": row.note,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            }
        )

    payload = {
        "records": records,
        "count": total,
        "pending_count": pending_total,
        "completed_count": completed_total,
        "pagination": {
            "page": page,
            "page_size": size,
            "total": total,
            "total_pages": max((total + size - 1) // size, 1),
        },
    }
    return sanitize_payload(_dashboard_heavy_cache_set(cache_key, payload))


def _action_label_text(action: str | None) -> str:
    return {
        "long": "做多",
        "short": "做空",
        "close_long": "平多",
        "close_short": "平空",
        "hold": "观望",
        None: "-",
        "": "-",
    }.get(action, str(action or "-"))


def _shadow_backtest_conclusion(
    decision_action: str | None,
    best_action: str | None,
    missed_opportunity: bool,
    long_return_pct: float | None,
    short_return_pct: float | None,
) -> str:
    if missed_opportunity and best_action in {"long", "short"}:
        return f"观望错过{_action_label_text(best_action)}机会"
    if not best_action:
        return "等待结果"
    if best_action == "hold":
        return "观望较合理"
    if decision_action == best_action:
        return "方向有效"
    if decision_action in {"long", "short"} and best_action in {"long", "short"}:
        return f"方向偏差，实际更适合{_action_label_text(best_action)}"
    best_return = max(long_return_pct or 0.0, short_return_pct or 0.0)
    if best_return > 0:
        return f"实际更适合{_action_label_text(best_action)}"
    return "结果中性"


def _daily_target_payload() -> dict:
    cny_per_usdt = max(float(settings.cny_per_usdt_assumption or 7.2), 0.0001)
    return {
        "enabled": False,
        "target_currency": "USDT",
        "target_usdt": 0.0,
        "target_cny": 0.0,
        "cny_per_usdt_assumption": cny_per_usdt,
        "note": "每日目标已禁用，不参与交易判断。",
    }


@router.delete(
    "/decisions",
    dependencies=[Depends(require_destructive_dashboard_confirmation)],
)
async def clear_decisions():
    """Delete all AI decision records."""
    from db.repositories.decision_repo import DecisionRepository
    from db.session import get_session_ctx

    async with get_session_ctx() as session:
        repo = DecisionRepository(session)
        count = await repo.delete_all()
        await session.commit()

    _reset_trading_decision_runtime_state()

    return {"status": "ok", "message": f"Deleted {count} decision records", "deleted": count}


@router.get("/dashboard/account")
async def get_account_balance():
    """Account balances backed by OKX snapshots only.

    This legacy endpoint keeps the old response shape for clients, but Phase 3
    paper/live account truth must not come from virtual-account balances.
    """
    selected_mode = mode_manager.mode.value
    okx_account = await _get_dashboard_okx_account_snapshot(selected_mode)
    okx_error = str(okx_account.get("error")) if okx_account and okx_account.get("error") else None
    okx_equity = (
        None
        if okx_error
        else _safe_float(
            (okx_account or {}).get("equity") or (okx_account or {}).get("total"), None
        )
    )
    okx_available = None if okx_error else _safe_float((okx_account or {}).get("free"), None)

    return {
        "mode": selected_mode,
        "virtual_accounts": [],
        "live_balance": okx_equity if selected_mode == "live" else None,
        "okx_account": okx_account,
        "account_equity": okx_equity,
        "available_balance": okx_available,
        "balance_source": (
            "okx_authoritative" if okx_account and not okx_error else "okx_unavailable"
        ),
        "balance_error": okx_error,
    }


@router.get("/dashboard/pnl-history")
async def get_pnl_history(mode: str | None = None):
    """PnL equity curve history for each model."""
    selected_mode = (
        "live" if mode == "live" else "paper" if mode == "paper" else mode_manager.mode.value
    )
    okx_history = await _build_okx_equity_pnl_history(selected_mode)
    return {"history": _format_okx_equity_pnl_history(okx_history)}


@router.get("/dashboard/daily-pnl")
async def get_daily_pnl_records(mode: str | None = None, days: int = 30):
    """Daily execution PnL grouped by Beijing calendar day."""
    from sqlalchemy import select

    from db.repositories.trade_repo import TradeRepository
    from db.session import get_session_ctx
    from models.trade import Order

    selected_mode = "live" if mode == "live" else "paper"
    days = min(max(int(days or 30), 1), 180)
    today_local = datetime.now(BEIJING_TZ).date()
    phase3_start_day = datetime.fromisoformat(PHASE3_FIRST_CLEAN_DAY).date()
    start_day = max(today_local - timedelta(days=days - 1), phase3_start_day)
    start_local = datetime.combine(start_day, datetime.min.time(), tzinfo=BEIJING_TZ)
    start_utc = start_local.astimezone(UTC)

    record_count = max((today_local - start_day).days + 1, 1)
    records: dict[str, dict[str, Any]] = {
        (start_day + timedelta(days=offset)).isoformat(): {
            "date": (start_day + timedelta(days=offset)).isoformat(),
            "timezone": "Asia/Shanghai",
            "realized_profit": 0.0,
            "realized_loss": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "total_pnl": 0.0,
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "symbols": [],
            "symbol_pnl": {},
            "position_details": [],
            "order_count": 0,
            "filled_order_count": 0,
            "rejected_order_count": 0,
            "order_buy_count": 0,
            "order_sell_count": 0,
            "order_details": [],
            "cumulative_realized_pnl": 0.0,
            "cumulative_total_pnl": 0.0,
        }
        for offset in range(record_count)
    }
    cumulative_before = 0.0
    open_unrealized = 0.0

    try:
        exchange_marks = await _get_exchange_position_mark_map(selected_mode)
    except Exception as exc:
        _log_dashboard_fallback(
            "daily pnl exchange mark fallback",
            exc,
            mode=selected_mode,
        )
        exchange_marks = {}
    async with get_session_ctx() as session:
        repo = TradeRepository(session)
        (
            closed_ledger_rows,
            _closed_total,
            _closed_page,
            _closed_total_pages,
            _closed_ledger_source,
        ) = (
            await _dashboard_closed_position_ledger_rows(
                session,
                repo,
                mode=selected_mode,
                model_names=EXECUTION_LEDGER_MODEL_NAMES,
                page=1,
                page_size=5000,
                paginate=False,
            )
        )
        from models.account import ExecutionEquitySnapshot

        order_result = await session.execute(
            select(Order)
            .where(
                Order.execution_mode == selected_mode,
                Order.model_name.in_(EXECUTION_LEDGER_MODEL_NAMES),
            )
            .order_by(
                Order.filled_at.desc().nullslast(),
                Order.created_at.desc(),
                Order.id.desc(),
            )
            .limit(50000)
        )
        order_rows = list(order_result.scalars().all())

        snapshot_result = await session.execute(
            select(ExecutionEquitySnapshot)
            .where(
                ExecutionEquitySnapshot.model_name == ENSEMBLE_TRADER_NAME,
                ExecutionEquitySnapshot.mode == selected_mode,
                ExecutionEquitySnapshot.source == "okx_snapshot",
                ExecutionEquitySnapshot.snapshot_date >= start_day.isoformat(),
                ExecutionEquitySnapshot.snapshot_date >= PHASE3_FIRST_CLEAN_DAY,
            )
            .order_by(
                ExecutionEquitySnapshot.snapshot_date.asc(),
                ExecutionEquitySnapshot.snapshot_at.asc(),
                ExecutionEquitySnapshot.id.asc(),
            )
        )
        equity_snapshots = list(snapshot_result.scalars().all())

    for order in order_rows:
        order_time = _as_utc_datetime(getattr(order, "filled_at", None)) or _as_utc_datetime(
            getattr(order, "created_at", None)
        )
        if order_time is None or order_time < PHASE3_CLEAN_START_UTC:
            continue
        if order_time < start_utc:
            continue
        day = order_time.astimezone(BEIJING_TZ).date().isoformat()
        row = records.get(day)
        if not row:
            continue
        status = str(getattr(order, "status", "") or "").strip().lower()
        side = str(getattr(order, "side", "") or "").strip().lower()
        row["order_count"] += 1
        if status == "filled":
            row["filled_order_count"] += 1
        elif status == "rejected":
            row["rejected_order_count"] += 1
        if side == "buy":
            row["order_buy_count"] += 1
        elif side == "sell":
            row["order_sell_count"] += 1
        row["order_details"].append(_daily_pnl_order_detail(order, order_time=order_time))

    equity_by_date: dict[str, dict[str, Any]] = {}
    for snapshot in equity_snapshots:
        day = str(snapshot.snapshot_date or "")
        if not day or day not in records or day in equity_by_date:
            continue
        equity = _safe_float(snapshot.equity, None)
        if equity is None or equity <= 0:
            continue
        equity_by_date[day] = {
            "equity": equity,
            "snapshot_at": snapshot.snapshot_at.isoformat() if snapshot.snapshot_at else None,
            "source": snapshot.source or "okx_snapshot",
        }

    for ledger_row in closed_ledger_rows:
        if not _daily_pnl_ledger_row_has_okx_realized_pnl(ledger_row):
            continue
        pnl = float(_safe_float(ledger_row.get("realized_pnl"), 0.0) or 0.0)
        closed_at = _as_utc_datetime(ledger_row.get("closed_at"))
        if closed_at is None or closed_at < PHASE3_CLEAN_START_UTC:
            continue
        if closed_at < start_utc:
            cumulative_before += pnl
            continue
        day = closed_at.astimezone(BEIJING_TZ).date().isoformat()
        row = records.get(day)
        if not row:
            continue
        if pnl >= 0:
            row["realized_profit"] += pnl
            row["win_count"] += 1
        else:
            row["realized_loss"] += abs(pnl)
            row["loss_count"] += 1
        row["realized_pnl"] += pnl
        row["total_pnl"] += pnl
        row["trade_count"] += 1
        symbol = str(ledger_row.get("symbol") or "")
        if symbol and symbol not in row["symbols"]:
            row["symbols"].append(symbol)
        row["position_details"].append(
            _daily_pnl_ledger_position_detail(ledger_row, pnl=pnl, closed_at=closed_at)
        )
        if symbol:
            symbol_row = row["symbol_pnl"].setdefault(
                symbol,
                {
                    "symbol": symbol,
                    "realized_profit": 0.0,
                    "realized_loss": 0.0,
                    "realized_pnl": 0.0,
                    "trade_count": 0,
                    "win_count": 0,
                    "loss_count": 0,
                },
            )
            if pnl >= 0:
                symbol_row["realized_profit"] += pnl
                symbol_row["win_count"] += 1
            else:
                symbol_row["realized_loss"] += abs(pnl)
                symbol_row["loss_count"] += 1
            symbol_row["realized_pnl"] += pnl
            symbol_row["trade_count"] += 1

    if exchange_marks:
        open_unrealized = float(_exchange_position_totals(exchange_marks)["unrealized_pnl"])

    current_okx_equity: float | None = None
    current_okx_equity_at: str | None = None
    try:
        okx_account = await _get_dashboard_okx_account_snapshot(selected_mode)
        okx_error = (
            str(okx_account.get("error"))
            if isinstance(okx_account, dict) and okx_account.get("error")
            else None
        )
        if not okx_error and isinstance(okx_account, dict):
            current_okx_equity = _safe_float(
                okx_account.get("equity") or okx_account.get("total"),
                None,
            )
            current_okx_equity_at = (
                str(okx_account.get("timestamp") or okx_account.get("snapshot_at") or "") or None
            )
    except Exception as exc:
        _log_dashboard_fallback(
            "daily pnl current OKX equity unavailable",
            exc,
            mode=selected_mode,
        )

    cumulative = cumulative_before
    first_okx_equity = None
    previous_okx_equity = None
    for date_key in sorted(records):
        row = records[date_key]
        cumulative += row["realized_pnl"]
        row["realized_profit"] = round(row["realized_profit"], 8)
        row["realized_loss"] = round(row["realized_loss"], 8)
        row["realized_pnl"] = round(row["realized_pnl"], 8)
        equity_row = equity_by_date.get(date_key)
        if equity_row:
            equity = float(equity_row["equity"])
            if first_okx_equity is None:
                first_okx_equity = equity
            day_equity_pnl = 0.0 if previous_okx_equity is None else equity - previous_okx_equity
            previous_okx_equity = equity
            row["okx_equity"] = round(equity, 8)
            row["okx_equity_snapshot_at"] = equity_row.get("snapshot_at")
            row["okx_equity_source"] = equity_row.get("source") or "okx_snapshot"
            row["okx_equity_pnl"] = round(day_equity_pnl, 8)
            row["okx_equity_change"] = row["okx_equity_pnl"]
            row["okx_cumulative_equity_pnl"] = round(equity - first_okx_equity, 8)
            row["okx_cumulative_equity_change"] = row["okx_cumulative_equity_pnl"]
        else:
            row["okx_equity"] = None
            row["okx_equity_snapshot_at"] = None
            row["okx_equity_source"] = "okx_snapshot_missing"
            row["okx_equity_pnl"] = None
            row["okx_equity_change"] = None
            row["okx_cumulative_equity_pnl"] = None
            row["okx_cumulative_equity_change"] = None
        row["trade_realized_pnl"] = row["realized_pnl"]
        row["trade_cumulative_realized_pnl"] = round(cumulative, 8)
        if date_key == today_local.isoformat():
            if (
                current_okx_equity is not None
                and current_okx_equity > 0
                and first_okx_equity is not None
            ):
                latest_equity = float(current_okx_equity)
                today_baseline_row = equity_by_date.get(date_key)
                baseline_for_today = (
                    float(today_baseline_row["equity"])
                    if today_baseline_row and today_baseline_row.get("equity") is not None
                    else None
                )
                row["okx_equity"] = round(latest_equity, 8)
                row["okx_equity_snapshot_at"] = (
                    current_okx_equity_at or datetime.now(UTC).isoformat()
                )
                row["okx_equity_source"] = "okx_current_balance"
                row["okx_today_baseline_equity"] = (
                    round(baseline_for_today, 8) if baseline_for_today is not None else None
                )
                row["okx_today_baseline_at"] = (
                    today_baseline_row.get("snapshot_at") if today_baseline_row else None
                )
                row["okx_current_equity"] = round(latest_equity, 8)
                row["okx_current_equity_at"] = row["okx_equity_snapshot_at"]
                row["okx_equity_pnl_source"] = (
                    "current_equity_minus_today_baseline"
                    if baseline_for_today is not None
                    else "today_baseline_missing"
                )
                row["okx_equity_pnl"] = (
                    round(latest_equity - baseline_for_today, 8)
                    if baseline_for_today is not None
                    else None
                )
                row["okx_equity_change"] = row["okx_equity_pnl"]
                row["okx_cumulative_equity_pnl"] = round(latest_equity - first_okx_equity, 8)
                row["okx_cumulative_equity_change"] = row["okx_cumulative_equity_pnl"]
            row["unrealized_pnl"] = round(open_unrealized, 8)
            row["total_pnl"] = row["okx_equity_pnl"]
            row["cumulative_total_pnl"] = row["okx_cumulative_equity_pnl"]
        else:
            row["total_pnl"] = row["okx_equity_pnl"]
            row["cumulative_total_pnl"] = row["okx_cumulative_equity_pnl"]
        row["cumulative_realized_pnl"] = round(cumulative, 8)
        row["symbols"] = sorted(row["symbols"])
        row["position_details"] = sorted(
            row["position_details"],
            key=lambda item: item.get("closed_at") or "",
            reverse=True,
        )
        row["order_details"] = sorted(
            row["order_details"],
            key=lambda item: item.get("time") or "",
            reverse=True,
        )
        row["symbol_pnl"] = [
            {
                **symbol_row,
                "realized_profit": round(symbol_row["realized_profit"], 8),
                "realized_loss": round(symbol_row["realized_loss"], 8),
                "realized_pnl": round(symbol_row["realized_pnl"], 8),
            }
            for symbol_row in sorted(
                row["symbol_pnl"].values(),
                key=lambda item: abs(item["realized_pnl"]),
                reverse=True,
            )
        ]

    return {
        "mode": selected_mode,
        "timezone": "Asia/Shanghai",
        "phase3_start_date": PHASE3_FIRST_CLEAN_DAY,
        "pnl_source": "okx_equity_snapshots_and_okx_position_ledger",
        "start_date": start_day.isoformat(),
        "end_date": today_local.isoformat(),
        "records": [records[key] for key in sorted(records.keys(), reverse=True)],
    }


def _daily_pnl_ledger_row_has_okx_realized_pnl(row: dict[str, Any]) -> bool:
    """Only OKX-confirmed ledger PnL may feed daily realized PnL records."""
    source = str(row.get("pnl_source") or "").strip()
    if source == "okx_linked_order_net_pnl":
        return True
    if source == "okx_position_history_realized_pnl":
        return True
    if source == "okx_close_fill_net_pnl":
        return True
    if source == "okx_fill_pnl" and bool(row.get("evidence_complete")):
        return True
    if source.startswith("position_settlement_snapshot:okx_"):
        return True
    settlement_source = str(row.get("settlement_source") or "").strip()
    if settlement_source.startswith("position_settlement_snapshot:okx_"):
        return True
    if source == "position_settlement_snapshot":
        final_statuses = {"reconciled", "settled", "okx_position_history"}
        status = str(row.get("settlement_status") or "").strip()
        has_linked_order_evidence = bool(row.get("linked_order_count")) or (
            bool(row.get("entry_order_ids")) and bool(row.get("close_order_ids"))
        )
        return status in final_statuses and has_linked_order_evidence
    return False


def _daily_pnl_order_detail(order: Any, *, order_time: datetime | None) -> dict[str, Any]:
    raw_pnl = _safe_float(getattr(order, "okx_fill_pnl", None), None)
    fee = _safe_float(getattr(order, "fee", None), 0.0) or 0.0
    net_fill_pnl = None if raw_pnl is None else float(raw_pnl) - abs(float(fee or 0.0))
    return {
        "id": getattr(order, "id", None),
        "decision_id": getattr(order, "decision_id", None),
        "symbol": str(getattr(order, "symbol", "") or ""),
        "side": str(getattr(order, "side", "") or ""),
        "status": str(getattr(order, "status", "") or ""),
        "quantity": round(_safe_float(getattr(order, "quantity", None), 0.0) or 0.0, 8),
        "price": _safe_float(getattr(order, "price", None), None),
        "fee": round(fee, 8),
        "okx_fill_pnl": round(float(raw_pnl), 8) if raw_pnl is not None else None,
        "net_fill_pnl": round(net_fill_pnl, 8) if net_fill_pnl is not None else None,
        "time": order_time.isoformat() if order_time else None,
        "filled_at": (
            _as_utc_datetime(getattr(order, "filled_at", None)).isoformat()
            if _as_utc_datetime(getattr(order, "filled_at", None))
            else None
        ),
        "created_at": (
            _as_utc_datetime(getattr(order, "created_at", None)).isoformat()
            if _as_utc_datetime(getattr(order, "created_at", None))
            else None
        ),
        "exchange_order_id": str(getattr(order, "exchange_order_id", "") or ""),
        "okx_inst_id": str(getattr(order, "okx_inst_id", "") or ""),
        "okx_sync_status": str(getattr(order, "okx_sync_status", "") or ""),
    }


def _daily_pnl_ledger_position_detail(
    row: dict[str, Any],
    *,
    pnl: float,
    closed_at: datetime | None,
) -> dict[str, Any]:
    quantity = _safe_float(row.get("closed_quantity") or row.get("quantity"), 0.0) or 0.0
    entry_price = (
        _safe_float(row.get("average_entry_price"), None)
        or _safe_float(row.get("entry_price"), 0.0)
        or 0.0
    )
    exit_price = (
        _safe_float(row.get("average_close_price"), None)
        or _safe_float(row.get("current_price"), 0.0)
        or 0.0
    )
    side = str(row.get("side") or "")
    return {
        "id": row.get("id") or row.get("group_id"),
        "group_id": row.get("group_id") or row.get("id"),
        "symbol": str(row.get("symbol") or ""),
        "side": side,
        "side_label": _side_label(side),
        "quantity": round(quantity, 8),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "realized_pnl": round(float(pnl or 0.0), 8),
        "closed_at": closed_at.isoformat() if closed_at else None,
        "opened_at": row.get("opened_at"),
        "pnl_source": row.get("pnl_source"),
        "ledger_source": row.get("ledger_source"),
        "evidence_complete": bool(row.get("evidence_complete")),
        "position_ids": list(row.get("position_ids") or []),
        "entry_order_ids": list(row.get("entry_order_ids") or []),
        "close_order_ids": list(row.get("close_order_ids") or []),
    }


async def _dashboard_okx_account_bill_rows(
    session: Any,
    *,
    closed_rows: list[Any],
    mode: str | None,
    account_bill_model: Any,
) -> list[Any]:
    from sqlalchemy import select

    if not closed_rows:
        return []
    inst_ids = {
        str(getattr(position, "okx_inst_id", "") or "").strip().upper()
        for position in closed_rows
        if str(getattr(position, "okx_inst_id", "") or "").strip()
    }
    opened_values = [
        value
        for position in closed_rows
        if (value := _as_utc_datetime(getattr(position, "created_at", None))) is not None
    ]
    closed_values = [
        value
        for position in closed_rows
        if (value := _as_utc_datetime(getattr(position, "closed_at", None))) is not None
    ]
    if not inst_ids or not opened_values or not closed_values:
        return []
    start = min(opened_values) - timedelta(hours=1)
    end = max(closed_values) + timedelta(hours=1)
    stmt = select(account_bill_model).where(
        account_bill_model.inst_id.in_(sorted(inst_ids)),
        account_bill_model.bill_ts >= start.replace(tzinfo=None),
        account_bill_model.bill_ts <= end.replace(tzinfo=None),
    )
    if mode:
        stmt = stmt.where(account_bill_model.mode == ("live" if mode == "live" else "paper"))
    result = await session.execute(
        stmt.order_by(account_bill_model.bill_ts.asc(), account_bill_model.id.asc()).limit(10000)
    )
    return list(result.scalars().all())


def _daily_pnl_position_detail(
    position: Any,
    *,
    pnl: float,
    closed_at: datetime | None,
) -> dict[str, Any]:
    quantity = _safe_float(getattr(position, "quantity", None), 0.0) or 0.0
    entry_price = _safe_float(getattr(position, "entry_price", None), 0.0) or 0.0
    exit_price = _safe_float(getattr(position, "current_price", None), 0.0) or 0.0
    return {
        "id": getattr(position, "id", None),
        "symbol": str(getattr(position, "symbol", "") or ""),
        "side": str(getattr(position, "side", "") or ""),
        "side_label": _side_label(getattr(position, "side", "")),
        "quantity": round(quantity, 8),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "realized_pnl": round(float(pnl or 0.0), 8),
        "closed_at": closed_at.isoformat() if closed_at else None,
        "opened_at": (
            _as_utc_datetime(getattr(position, "created_at", None)).isoformat()
            if _as_utc_datetime(getattr(position, "created_at", None))
            else None
        ),
    }


async def _build_okx_equity_pnl_history(mode: str) -> dict[str, list[dict]]:
    """Return persisted OKX equity snapshots for the account curve."""
    from sqlalchemy import select

    from db.session import get_session_ctx
    from models.account import ExecutionEquitySnapshot

    selected_mode = "live" if mode == "live" else "paper"
    try:
        async with get_session_ctx() as session:
            result = await session.execute(
                select(ExecutionEquitySnapshot)
                .where(
                    ExecutionEquitySnapshot.model_name == ENSEMBLE_TRADER_NAME,
                    ExecutionEquitySnapshot.mode == selected_mode,
                    ExecutionEquitySnapshot.source == "okx_snapshot",
                    ExecutionEquitySnapshot.snapshot_date >= PHASE3_FIRST_CLEAN_DAY,
                )
                .order_by(
                    ExecutionEquitySnapshot.snapshot_at.asc(), ExecutionEquitySnapshot.id.asc()
                )
                .limit(500)
            )
            rows = list(result.scalars().all())
    except Exception as exc:
        _log_dashboard_fallback(
            "OKX equity history database fallback",
            exc,
            mode=selected_mode,
        )
        return {}

    if not rows:
        return {}
    return {
        ENSEMBLE_TRADER_NAME: [
            {
                "time": row.snapshot_at.isoformat() if row.snapshot_at else None,
                "equity": round(float(row.equity or 0.0), 8),
                "source": row.source,
            }
            for row in rows
            if _safe_float(row.equity, 0.0) and _safe_float(row.equity, 0.0) > 0
        ][-500:]
    }


def _format_okx_equity_pnl_history(history: dict[str, list[dict]]) -> dict:
    result = {}
    for model_name, snapshots in (history or {}).items():
        if not snapshots:
            continue
        initial = _safe_float(snapshots[0].get("equity"), 0.0) or 0.0
        result[model_name] = {
            "pnl_curve": [
                (
                    round(
                        ((_safe_float(s.get("equity"), initial) or initial) - initial)
                        / initial
                        * 100,
                        6,
                    )
                    if initial > 0
                    else 0
                )
                for s in snapshots
            ],
            "labels": [s.get("time") for s in snapshots],
            "source": "okx_equity_snapshots",
        }
    return result
