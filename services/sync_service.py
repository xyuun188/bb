"""OKX and local-position synchronization boundary."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from math import isclose
from types import SimpleNamespace
from typing import Any

import structlog
from sqlalchemy import select

from ai_brain.base_model import Action, DecisionOutput
from config.settings import ENSEMBLE_TRADER_NAME
from core.safe_output import safe_error_text
from core.symbols import (
    okx_inst_id_from_payload,
    symbol_from_okx_inst_id,
    trading_symbol_variants,
)
from core.trading_mode import mode_manager
from db.repositories.trade_repo import TradeRepository
from db.session import get_session_ctx
from executor.base_executor import OrderStatus
from models.decision import AIDecision
from models.learning import TradeReflection
from models.trade import Order, Position
from services.current_position_management import (
    CURRENT_POSITION_MANAGEMENT_KIND,
    CURRENT_POSITION_MANAGEMENT_VERSION,
)
from services.exchange_position_state import parse_exchange_position_snapshot
from services.paper_bootstrap_canary import build_paper_canary_position_lifecycle
from services.position_open_time import parse_position_time, serialize_position_time
from services.position_settlement import (
    SETTLEMENT_STATUS_SETTLING,
    apply_position_settlement_snapshot,
    build_position_settlement_snapshot,
    funding_fee_from_payload,
    proportional_signed_value,
    settlement_payload_fields,
)
from services.trade_fact_trust import TRUSTED_OKX_ORDER_SYNC_STATUSES

logger = structlog.get_logger(__name__)

UNCONFIRMED_EXCHANGE_CLOSE_GRACE_SECONDS = 180.0
OPEN_ORDER_SNAPSHOT_UNKNOWN_KIND = "unknown"
EXCHANGE_PROTECTION_MAP_TIMEOUT_SECONDS = 6.0
EXCHANGE_CLOSE_FILL_LOOKUP_TIMEOUT_SECONDS = 8.0
RECONCILE_OPTIONAL_DEADLINE_RESERVE_SECONDS = 0.75
RECONCILE_OPTIONAL_MIN_TIMEOUT_SECONDS = 0.5
RECONCILE_ORIGIN_SYSTEM_PROTECTION = "system_protection"
RECONCILE_ORIGIN_EXTERNAL_OKX = "external_okx_sync"
ORPHAN_QUARANTINE_REFLECTION_SOURCE = "okx_orphan_position_quarantine"
ORPHAN_QUARANTINE_CLOSE_PREFIX = "okx_orphan_quarantine:"
POSITION_PRICE_REFRESH_DB_LOAD_TIMEOUT_SECONDS = 1.5


def _consume_background_task_result(task: asyncio.Task[Any]) -> None:
    try:
        task.exception()
    except (asyncio.CancelledError, Exception):
        return


def _first_value(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _safe_positive(value: float, default: float = 0.0) -> float:
    return value if value > 0 else default


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _split_exchange_order_ids(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    tokens = [text]
    for separator in (",", ";", "|", "\n", "\t", " "):
        pieces: list[str] = []
        for token in tokens:
            pieces.extend(part.strip() for part in token.split(separator) if part.strip())
        tokens = pieces
    ordered: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def _merge_exchange_order_ids(*values: Any, max_length: int = 500) -> str:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        for token in _split_exchange_order_ids(value):
            if token in seen:
                continue
            seen.add(token)
            ordered.append(token)
    merged: list[str] = []
    for token in ordered:
        candidate = ",".join([*merged, token]) if merged else token
        if len(candidate) > max_length:
            break
        merged.append(token)
    return ",".join(merged)


def _merge_entry_legs(*values: Any) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen_order_ids: set[str] = set()
    for value in values:
        legs = _list_of_dicts(value)
        for leg in legs:
            order_id = str(leg.get("exchange_order_id") or "").strip()
            if order_id and order_id in seen_order_ids:
                continue
            if order_id:
                seen_order_ids.add(order_id)
            merged.append(dict(leg))
    return merged


def _merge_created_at(left: Any, right: Any) -> Any:
    if left in (None, ""):
        return right
    if right in (None, ""):
        return left
    try:
        left_dt = parse_position_time(left)
        right_dt = parse_position_time(right)
    except Exception:
        return left
    if left_dt is None:
        return right
    if right_dt is None:
        return left
    return left if left_dt <= right_dt else right


def _merge_local_position_candidates(
    candidates: list[dict[str, Any]],
    *,
    exchange_position: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not candidates:
        return {}
    exchange_pos_id = _okx_pos_id_from_position_payload(exchange_position or {})
    scoped_candidates = candidates
    if exchange_pos_id:
        matching = [
            candidate
            for candidate in candidates
            if str(candidate.get("okx_pos_id") or "").strip() == exchange_pos_id
        ]
        if matching:
            scoped_candidates = matching
    merged = dict(scoped_candidates[0])
    for candidate in scoped_candidates[1:]:
        for key in (
            "model_name",
            "stop_loss",
            "take_profit",
            "okx_inst_id",
            "okx_pos_id",
            "execution_mode",
        ):
            if merged.get(key) in (None, "") and candidate.get(key) not in (None, ""):
                merged[key] = candidate.get(key)
        merged["created_at"] = _merge_created_at(
            merged.get("created_at"), candidate.get("created_at")
        )
        merged["entry_exchange_order_id"] = _merge_exchange_order_ids(
            merged.get("entry_exchange_order_id"),
            candidate.get("entry_exchange_order_id"),
        )
        merged["entry_legs"] = _merge_entry_legs(
            merged.get("entry_legs"),
            candidate.get("entry_legs"),
        )
        if not _dict_value(merged.get("current_management_contract")):
            merged["current_management_contract"] = _dict_value(
                candidate.get("current_management_contract")
            )
        if not _dict_value(merged.get("paper_canary_lifecycle")):
            merged["paper_canary_lifecycle"] = _dict_value(candidate.get("paper_canary_lifecycle"))
        if _float_value(merged.get("entry_fee"), 0.0) <= 0:
            merged["entry_fee"] = candidate.get("entry_fee")

    return merged


def _is_missing_market_symbol_error(error: Any) -> bool:
    text = str(error or "").lower()
    return "does not have market symbol" in text or "bad symbol" in text


def _close_fill_order_info(close_fill: dict[str, Any] | None) -> dict[str, Any]:
    fill = _dict_value(close_fill)
    info = _dict_value(fill.get("order_info"))
    if info:
        return info
    raw = _dict_value(fill.get("raw"))
    return _dict_value(raw.get("info"))


def _okx_inst_id_from_position_payload(position_payload: dict[str, Any]) -> str:
    info = _dict_value(position_payload.get("info"))
    return okx_inst_id_from_payload(
        {
            **position_payload,
            "info": info,
            "instId": _first_value(info.get("instId"), position_payload.get("instId")),
        },
        fallback=position_payload.get("symbol"),
    )


def _okx_pos_id_from_position_payload(position_payload: dict[str, Any]) -> str:
    info = _dict_value(position_payload.get("info"))
    for candidate in (
        info.get("posId"),
        position_payload.get("posId"),
        position_payload.get("okx_pos_id"),
    ):
        value = str(candidate or "").strip()
        if value:
            return value
    return ""


def _okx_inst_id_from_close_fill(close_fill: dict[str, Any] | None, *, fallback: Any) -> str:
    fill = _dict_value(close_fill)
    info = _close_fill_order_info(fill)
    return okx_inst_id_from_payload({**fill, "info": info}, fallback=fallback)


def _symbol_from_okx_inst_id_or_fallback(okx_inst_id: Any, fallback: Any) -> str:
    value = str(okx_inst_id or "").strip().upper()
    if value:
        symbol = symbol_from_okx_inst_id(value)
        if symbol:
            return symbol
    return str(fallback or "")


def _okx_pos_id_from_close_fill(close_fill: dict[str, Any] | None) -> str:
    fill = _dict_value(close_fill)
    info = _close_fill_order_info(fill)
    for candidate in (info.get("posId"), fill.get("posId"), fill.get("okx_pos_id")):
        value = str(candidate or "").strip()
        if value:
            return value
    return ""


def _close_fill_has_protection_metadata(close_fill: dict[str, Any] | None) -> bool:
    fill = _dict_value(close_fill)
    info = _close_fill_order_info(fill)
    order_type = str(
        _first_value(
            fill.get("order_type"),
            info.get("ordType"),
            info.get("algoOrdType"),
            info.get("category"),
        )
        or ""
    ).lower()
    if order_type in {"oco", "conditional", "trigger", "move_order_stop"}:
        return True
    return bool(_first_value(fill.get("algo_id"), info.get("algoId"), info.get("algoClOrdId")))


def _exchange_reconcile_close_origin(
    position: Any,
    close_fill: dict[str, Any] | None,
) -> str:
    """Classify who initiated an exchange close discovered by reconciliation."""

    fill = _dict_value(close_fill)
    if not fill or fill.get("estimated"):
        return RECONCILE_ORIGIN_EXTERNAL_OKX
    if _close_fill_has_protection_metadata(fill):
        return RECONCILE_ORIGIN_SYSTEM_PROTECTION
    return RECONCILE_ORIGIN_EXTERNAL_OKX


def _okx_close_fill_fact_quantity(
    *,
    fill: dict[str, Any],
    order_info: dict[str, Any],
    contracts: float,
    contract_size: float,
    fallback_quantity: float,
) -> float:
    converted_quantity = (
        float(contracts) * float(contract_size)
        if float(contracts or 0.0) > 0 and float(contract_size or 0.0) > 0
        else 0.0
    )
    if converted_quantity > 0:
        return converted_quantity
    explicit_quantity = _float_value(
        _first_value(
            fill.get("base_quantity"),
            fill.get("baseQuantity"),
            fill.get("fill_quantity"),
            fill.get("filled_base_quantity"),
            fill.get("filledBaseQuantity"),
            fill.get("quantity"),
            order_info.get("base_quantity"),
            order_info.get("baseQuantity"),
        ),
        0.0,
    )
    if explicit_quantity > 0:
        return explicit_quantity
    return _float_value(fallback_quantity, 0.0)


async def _position_entry_order_is_exchange_close_fill(
    session: Any,
    position: Position,
) -> bool:
    entry_order_ids = _split_exchange_order_ids(getattr(position, "entry_exchange_order_id", None))
    if not entry_order_ids:
        return False
    result = await session.execute(
        select(Order).where(
            Order.execution_mode == getattr(position, "execution_mode", None),
            Order.exchange_order_id.in_(entry_order_ids),
        )
    )
    orders = list(result.scalars().all())
    for order in orders:
        raw = _dict_value(getattr(order, "okx_raw_fills", None))
        fill_pnl = _float_value(raw.get("fill_pnl") or getattr(order, "okx_fill_pnl", None), 0.0)
        contracts = _float_value(
            raw.get("contracts") or getattr(order, "okx_fill_contracts", None), 0.0
        )
        confirmed = (
            raw.get("fills_history_confirmed") is True
            or raw.get("execution_result_confirmed") is True
            or str(getattr(order, "okx_sync_status", "") or "").startswith("okx_")
        )
        if confirmed and contracts > 0 and abs(fill_pnl) > 1e-12:
            return True
    return False


async def _confirmed_local_close_fill_for_position(
    session: Any,
    position: Any,
    *,
    target_quantity: float | None = None,
) -> dict[str, Any]:
    """Reuse a persisted OKX-confirmed close order before querying remote history."""

    side = str(getattr(position, "side", "") or "").lower()
    close_action = "close_short" if side == "short" else "close_long"
    close_side = "buy" if side == "short" else "sell"
    symbol = str(getattr(position, "symbol", "") or "")
    variants = trading_symbol_variants(symbol) or {symbol}
    statement = (
        select(Order)
        .join(AIDecision, AIDecision.id == Order.decision_id)
        .where(
            Order.execution_mode == getattr(position, "execution_mode", None),
            Order.symbol.in_(variants),
            Order.side == close_side,
            Order.status == OrderStatus.FILLED.value,
            Order.exchange_order_id.is_not(None),
            Order.exchange_order_id != "",
            Order.okx_sync_status.in_(TRUSTED_OKX_ORDER_SYNC_STATUSES),
            AIDecision.action == close_action,
            AIDecision.was_executed.is_(True),
        )
        .order_by(Order.filled_at.desc().nullslast(), Order.created_at.desc())
        .limit(10)
    )
    opened_at = getattr(position, "created_at", None)
    if opened_at is not None:
        statement = statement.where(
            (Order.filled_at.is_(None) & (Order.created_at >= opened_at))
            | (Order.filled_at >= opened_at)
        )
    if not callable(getattr(session, "execute", None)):
        return {}
    result = await session.execute(statement)
    if not callable(getattr(result, "scalars", None)):
        return {}
    target = abs(_float_value(target_quantity, 0.0))
    for order in result.scalars().all():
        quantity = abs(_float_value(getattr(order, "quantity", None), 0.0))
        if target > 0 and (quantity <= 0 or quantity < target * 0.2):
            continue
        price = _float_value(getattr(order, "price", None), 0.0)
        if price <= 0:
            continue
        raw = _dict_value(getattr(order, "okx_raw_fills", None))
        return {
            "price": price,
            "fee": abs(_float_value(getattr(order, "fee", None), 0.0)),
            "order_id": str(getattr(order, "exchange_order_id", "") or ""),
            "timestamp": getattr(order, "filled_at", None) or getattr(order, "created_at", None),
            "quantity": quantity,
            "contracts": _float_value(
                getattr(order, "okx_fill_contracts", None) or raw.get("contracts"),
                0.0,
            ),
            "pnl": _float_value(
                (
                    getattr(order, "okx_fill_pnl", None)
                    if getattr(order, "okx_fill_pnl", None) is not None
                    else raw.get("fill_pnl")
                ),
                0.0,
            ),
            "source": "local_okx_confirmed_close_order",
            "order_info": _dict_value(raw.get("order_info")),
        }
    return {}


def _okx_close_fill_order_payload(
    *,
    model_name: str,
    execution_mode: str,
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    fee: float,
    decision_id: int | None,
    close_order_id: str,
    filled_at: datetime | None,
    close_fill: dict[str, Any] | None,
    okx_inst_id: str,
) -> dict[str, Any]:
    fill = _dict_value(close_fill)
    order_info = _close_fill_order_info(fill)
    trade_id = str(
        _first_value(
            fill.get("trade_id"),
            fill.get("tradeId"),
            order_info.get("tradeId"),
        )
        or ""
    ).strip()
    contracts = _float_value(fill.get("contracts"), 0.0)
    if contracts <= 0:
        contracts = _float_value(order_info.get("fillSz") or order_info.get("accFillSz"), 0.0)
    contract_size = _float_value(
        _first_value(
            fill.get("contract_size"),
            fill.get("contractSize"),
            order_info.get("ctVal"),
            order_info.get("contractSize"),
        ),
        0.0,
    )
    fact_quantity = _okx_close_fill_fact_quantity(
        fill=fill,
        order_info=order_info,
        contracts=contracts,
        contract_size=contract_size,
        fallback_quantity=quantity,
    )
    fill_pnl = _float_value(
        _first_value(fill.get("pnl"), fill.get("fillPnl"), order_info.get("fillPnl")),
        0.0,
    )
    timestamp = filled_at
    raw_timestamp = _first_value(
        fill.get("timestamp"), order_info.get("ts"), fill.get("timestamp_ms")
    )
    if timestamp is None and isinstance(raw_timestamp, datetime):
        timestamp = raw_timestamp
    raw_payload = {
        "source": fill.get("source") or "okx_reconcile_close_fill",
        "fills_history_confirmed": True,
        "order_id": close_order_id or None,
        "trade_ids": [trade_id] if trade_id else [],
        "inst_id": okx_inst_id or _okx_inst_id_from_close_fill(fill, fallback=symbol),
        "contracts": contracts or None,
        "contract_size": contract_size or fill.get("contract_size"),
        "contract_size_verified": contract_size > 0,
        "contract_size_source": "okx_close_fill" if contract_size > 0 else "missing",
        "base_quantity": fact_quantity,
        "avg_price": price,
        "fee_abs": abs(float(fee or 0.0)),
        "fill_pnl": fill_pnl,
        "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else raw_timestamp,
        "rows": [dict(order_info)] if order_info else [],
    }
    return {
        "model_name": model_name,
        "execution_mode": execution_mode,
        "symbol": symbol,
        "side": side,
        "order_type": "market",
        "quantity": fact_quantity,
        "price": price,
        "status": OrderStatus.FILLED.value,
        "fee": abs(float(fee or 0.0)),
        "decision_id": decision_id,
        "exchange_order_id": close_order_id or None,
        "filled_at": filled_at,
        "okx_inst_id": okx_inst_id or raw_payload["inst_id"],
        "okx_trade_ids": ",".join(raw_payload["trade_ids"]) if raw_payload["trade_ids"] else None,
        "okx_fill_contracts": contracts or None,
        "okx_fill_pnl": fill_pnl,
        "okx_state": "filled",
        "okx_sync_status": "okx_confirmed",
        "okx_synced_at": datetime.now(UTC),
        "okx_last_error": None,
        "okx_raw_fills": raw_payload,
    }


def _apply_okx_close_fill_order_payload(order: Order, payload: dict[str, Any]) -> None:
    for key, value in payload.items():
        if key == "decision_id" and value is None:
            continue
        setattr(order, key, value)


def _position_context_opened_at(position_payload: dict[str, Any], info: dict[str, Any]) -> Any:
    """Return a stable open time, never the OKX update time."""

    for value in (
        position_payload.get("created_at"),
        position_payload.get("opened_at"),
        position_payload.get("open_time"),
        position_payload.get("openTime"),
        info.get("cTime"),
        info.get("openTime"),
        info.get("posTime"),
        info.get("created_at"),
    ):
        parsed = parse_position_time(value)
        if parsed is not None:
            return serialize_position_time(parsed)
    return None


def _exchange_position_key(
    position_payload: dict[str, Any],
    *,
    symbol_normalizer: Callable[[Any], str],
) -> tuple[str, str] | None:
    snapshot = parse_exchange_position_snapshot(
        position_payload,
        symbol_normalizer=symbol_normalizer,
    )
    if snapshot:
        return str(snapshot["symbol"]), str(snapshot["side"])
    symbol = symbol_normalizer(position_payload.get("symbol"))
    side = str(position_payload.get("side") or "").lower()
    if not symbol or side not in {"long", "short"}:
        return None
    return symbol, side


def _position_execution_mode(position: Any) -> str:
    return (
        "live" if str(getattr(position, "execution_mode", "") or "").lower() == "live" else "paper"
    )


def normalized_open_position_context(
    position_payload: dict[str, Any],
    *,
    symbol_normalizer: Callable[[Any], str],
    float_parser: Callable[[Any, float], float],
) -> dict[str, Any]:
    raw_info = position_payload.get("info")
    info = raw_info if isinstance(raw_info, dict) else {}
    okx_inst_id = _okx_inst_id_from_position_payload(position_payload)
    okx_pos_id = _okx_pos_id_from_position_payload(position_payload)
    entry_exchange_order_id = str(position_payload.get("entry_exchange_order_id") or "").strip()
    entry_legs = _list_of_dicts(position_payload.get("entry_legs"))
    current_management_contract = _dict_value(position_payload.get("current_management_contract"))
    entry_fee = abs(float_parser(position_payload.get("entry_fee"), 0.0))
    exit_fee_rate = float_parser(
        current_management_contract.get("exit_fee_rate_proxy"),
        0.0,
    )
    snapshot = parse_exchange_position_snapshot(
        position_payload,
        symbol_normalizer=symbol_normalizer,
    )

    if snapshot:
        entry_price = float_parser(snapshot.get("entry_price"), 0.0)
        current_price = (
            float_parser(snapshot.get("mark_price"), 0.0)
            or float_parser(snapshot.get("last_price"), 0.0)
            or entry_price
        )
        quantity = float_parser(snapshot.get("quantity"), 0.0)
        contracts = float_parser(snapshot.get("contracts"), 0.0)
        contract_size = float_parser(snapshot.get("contract_size"), 1.0)
        contract_size_source = "exchange_position_snapshot"
        management_contracts = abs(float_parser(current_management_contract.get("contracts"), 0.0))
        management_quantity = abs(float_parser(current_management_contract.get("quantity"), 0.0))
        contract_symbol = symbol_normalizer(current_management_contract.get("symbol"))
        if (
            current_management_contract.get("contract_version")
            == CURRENT_POSITION_MANAGEMENT_VERSION
            and current_management_contract.get("kind") == CURRENT_POSITION_MANAGEMENT_KIND
            and current_management_contract.get("management_eligible") is True
            and not current_management_contract.get("blockers")
            and contract_symbol == snapshot.get("symbol")
            and str(current_management_contract.get("side") or "").lower() == snapshot.get("side")
            and contracts > 0
            and management_contracts > 0
            and management_quantity > 0
            and isclose(
                contracts,
                management_contracts,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ):
            contract_size = management_quantity / management_contracts
            quantity = contracts * contract_size
            contract_size_source = "current_management_contract_okx_contract_spec"
        direct_notional = abs(
            float_parser(
                position_payload.get("notional")
                or position_payload.get("notional_usd")
                or position_payload.get("notionalUsd")
                or info.get("notionalUsd")
                or info.get("notional")
                or info.get("posValue"),
                0.0,
            )
        )
        notional = direct_notional if direct_notional > 0 else abs(entry_price * quantity)
        unrealized = snapshot.get("upl")
        if unrealized is None:
            unrealized = position_payload.get(
                "unrealized_pnl", position_payload.get("unrealizedPnl", 0)
            )
        return {
            "model_name": position_payload.get("model_name", ""),
            "symbol": snapshot.get("symbol", ""),
            "side": snapshot.get("side", "long"),
            "entry_price": entry_price,
            "current_price": current_price,
            "quantity": quantity,
            "base_quantity": quantity,
            "contracts": contracts,
            "contract_size": contract_size,
            "contractSize": contract_size,
            "contract_size_source": contract_size_source,
            "leverage": float_parser(position_payload.get("leverage") or info.get("lever"), 1.0),
            "notional": notional,
            "notional_usd": notional,
            "margin": _first_value(
                position_payload.get("margin"),
                position_payload.get("initial_margin"),
                position_payload.get("initialMargin"),
                position_payload.get("margin_used"),
                snapshot.get("margin_used"),
                info.get("margin"),
                info.get("imr"),
            ),
            "initial_margin": _first_value(
                position_payload.get("initial_margin"),
                position_payload.get("initialMargin"),
                snapshot.get("margin_used"),
                info.get("imr"),
            ),
            "initialMargin": _first_value(
                position_payload.get("initialMargin"),
                position_payload.get("initial_margin"),
                snapshot.get("margin_used"),
                info.get("imr"),
            ),
            "unrealized_pnl": unrealized,
            "stop_loss": position_payload.get("stop_loss"),
            "take_profit": position_payload.get("take_profit"),
            "is_open": position_payload.get("is_open", True),
            "created_at": _position_context_opened_at(position_payload, info),
            "okx_inst_id": okx_inst_id,
            "okx_pos_id": okx_pos_id,
            "entry_exchange_order_id": entry_exchange_order_id,
            "entry_legs": entry_legs,
            "entry_fee": entry_fee,
            "entry_fee_usdt": entry_fee,
            "exit_fee_rate": exit_fee_rate,
            "current_management_contract": current_management_contract,
            "paper_canary_lifecycle": _dict_value(position_payload.get("paper_canary_lifecycle")),
            "execution_mode": position_payload.get("execution_mode"),
            "info": info,
        }

    entry_price = float_parser(
        position_payload.get("entry_price")
        or position_payload.get("entryPrice")
        or position_payload.get("avgPx"),
        0.0,
    )
    current_price = float_parser(
        position_payload.get("current_price")
        or position_payload.get("markPrice")
        or position_payload.get("lastPrice")
        or position_payload.get("entry_price")
        or position_payload.get("entryPrice")
        or position_payload.get("avgPx"),
        entry_price,
    )
    raw_quantity = abs(
        float_parser(
            position_payload.get("quantity")
            or position_payload.get("baseVolume")
            or info.get("baseBal"),
            0.0,
        )
    )
    contract_size = float_parser(
        position_payload.get("contract_size")
        or position_payload.get("contractSize")
        or info.get("ctVal"),
        1.0,
    )
    contracts = abs(
        float_parser(
            position_payload.get("contracts")
            or position_payload.get("sz")
            or position_payload.get("size")
            or position_payload.get("positionAmt")
            or info.get("pos")
            or info.get("qty"),
            0.0,
        )
    )
    quantity = (
        contracts * (contract_size if contract_size > 0 else 1.0) if contracts > 0 else raw_quantity
    )
    direct_notional = abs(
        float_parser(
            position_payload.get("notional")
            or position_payload.get("notional_usd")
            or position_payload.get("notionalUsd")
            or info.get("notionalUsd")
            or info.get("notional")
            or info.get("posValue"),
            0.0,
        )
    )
    notional = direct_notional if direct_notional > 0 else abs(entry_price * quantity)
    return {
        "model_name": position_payload.get("model_name", ""),
        "symbol": position_payload.get("symbol", ""),
        "side": position_payload.get("side", "long"),
        "entry_price": entry_price,
        "current_price": _safe_positive(current_price, entry_price),
        "quantity": quantity,
        "base_quantity": quantity,
        "raw_quantity": raw_quantity,
        "contracts": contracts,
        "contract_size": contract_size,
        "contractSize": contract_size,
        "leverage": float_parser(position_payload.get("leverage") or info.get("lever"), 1.0),
        "notional": notional,
        "notional_usd": notional,
        "margin": _first_value(
            position_payload.get("margin"),
            position_payload.get("initial_margin"),
            position_payload.get("initialMargin"),
            position_payload.get("margin_used"),
            info.get("margin"),
            info.get("imr"),
        ),
        "initial_margin": _first_value(
            position_payload.get("initial_margin"),
            position_payload.get("initialMargin"),
            info.get("imr"),
        ),
        "initialMargin": _first_value(
            position_payload.get("initialMargin"),
            position_payload.get("initial_margin"),
            info.get("imr"),
        ),
        "unrealized_pnl": position_payload.get(
            "unrealized_pnl", position_payload.get("unrealizedPnl", 0)
        ),
        "stop_loss": position_payload.get("stop_loss"),
        "take_profit": position_payload.get("take_profit"),
        "is_open": position_payload.get("is_open", True),
        "created_at": _position_context_opened_at(position_payload, info),
        "okx_inst_id": okx_inst_id,
        "okx_pos_id": okx_pos_id,
        "entry_exchange_order_id": entry_exchange_order_id,
        "entry_legs": entry_legs,
        "entry_fee": entry_fee,
        "entry_fee_usdt": entry_fee,
        "exit_fee_rate": exit_fee_rate,
        "current_management_contract": current_management_contract,
        "paper_canary_lifecycle": _dict_value(position_payload.get("paper_canary_lifecycle")),
        "execution_mode": position_payload.get("execution_mode"),
        "info": info,
    }


class OkxSyncService:
    """Owns OKX reconciliation and open-position context building."""

    def __init__(
        self,
        *,
        exchange_reconcile_lock: AbstractAsyncContextManager[Any] | None = None,
        round_error_recorder: Callable[[str], None] | None = None,
        symbol_normalizer: Callable[[Any], str] | None = None,
        model_execution_mode_provider: Callable[[str], str] | None = None,
        okx_executor_provider: Callable[[str], Awaitable[Any]] | None = None,
        float_parser: Callable[[Any, float], float] | None = None,
        paper_positions_provider: Callable[[], Awaitable[list[dict[str, Any]]]] | None = None,
        active_okx_provider: Callable[[], Any | None] | None = None,
        paper_okx_provider: Callable[[], Any | None] | None = None,
        exchange_position_open_checker: Callable[[dict[str, Any]], bool] | None = None,
        exchange_protection_map_provider: (
            Callable[[Any, list[dict]], Awaitable[dict[tuple[str, str], dict[str, Any]]]] | None
        ) = None,
        position_protection_fallback_provider: (
            Callable[..., Awaitable[dict[str, Any]]] | None
        ) = None,
        position_profit_peak_recorder: Callable[..., Any] | None = None,
        position_age_minutes_provider: Callable[[Any], float | None] | None = None,
        position_profit_peak_pruner: Callable[[list[dict[str, Any]]], None] | None = None,
        local_position_snapshot_syncer: Callable[..., bool] | None = None,
        datetime_from_ms_parser: Callable[[Any], datetime] | None = None,
        exchange_close_fill_finder: Callable[[Any], Awaitable[dict[str, Any]]] | None = None,
        fresh_feature_vector_provider: Callable[[str], Awaitable[Any | None]] | None = None,
        market_value_reader: Callable[[Any, str], Any] | None = None,
        entry_fee_provider: Callable[[Any, Any, float], Awaitable[float]] | None = None,
        exchange_sync_close_decision_logger: Callable[..., Awaitable[int | None]] | None = None,
        trade_reflection_recorder: Callable[..., Awaitable[None]] | None = None,
        position_margin_calculator: Callable[[float, float | None], float] | None = None,
        memory_position_remover: Callable[[str, str, str], None] | None = None,
    ) -> None:
        self.exchange_reconcile_lock = exchange_reconcile_lock
        self.round_error_recorder = round_error_recorder
        self.symbol_normalizer = symbol_normalizer
        self.model_execution_mode_provider = model_execution_mode_provider
        self.okx_executor_provider = okx_executor_provider
        self.float_parser = float_parser
        self.paper_positions_provider = paper_positions_provider
        self.active_okx_provider = active_okx_provider
        self.paper_okx_provider = paper_okx_provider
        self.exchange_position_open_checker = exchange_position_open_checker
        self.exchange_protection_map_provider = exchange_protection_map_provider
        self.position_protection_fallback_provider = position_protection_fallback_provider
        self.position_profit_peak_recorder = position_profit_peak_recorder
        self.position_age_minutes_provider = position_age_minutes_provider
        self.position_profit_peak_pruner = position_profit_peak_pruner
        self.local_position_snapshot_syncer = local_position_snapshot_syncer
        self.datetime_from_ms_parser = datetime_from_ms_parser
        self.exchange_close_fill_finder = exchange_close_fill_finder
        self.fresh_feature_vector_provider = fresh_feature_vector_provider
        self.market_value_reader = market_value_reader
        self.entry_fee_provider = entry_fee_provider
        self.exchange_sync_close_decision_logger = exchange_sync_close_decision_logger
        self.trade_reflection_recorder = trade_reflection_recorder
        self.position_margin_calculator = position_margin_calculator
        self.memory_position_remover = memory_position_remover
        self._reconcile_deadline_monotonic: float | None = None
        self._reconcile_degraded_rows: list[dict[str, Any]] | None = None

    def _required_exchange_reconcile_lock(self) -> AbstractAsyncContextManager[Any]:
        if self.exchange_reconcile_lock is None:
            raise RuntimeError("OkxSyncService requires exchange_reconcile_lock dependency")
        return self.exchange_reconcile_lock

    def _record_round_error(self, reason: str) -> None:
        if self.round_error_recorder is None:
            raise RuntimeError("OkxSyncService requires round_error_recorder dependency")
        self.round_error_recorder(reason)

    def _reconcile_seconds_remaining(self) -> float | None:
        deadline = self._reconcile_deadline_monotonic
        if not isinstance(deadline, (int, float)):
            return None
        return float(deadline) - asyncio.get_running_loop().time()

    def _reconcile_optional_timeout(self, requested_seconds: float) -> float | None:
        requested = max(float(requested_seconds or 0.0), 0.0)
        remaining = self._reconcile_seconds_remaining()
        if remaining is None:
            return requested
        available = remaining - RECONCILE_OPTIONAL_DEADLINE_RESERVE_SECONDS
        if available < RECONCILE_OPTIONAL_MIN_TIMEOUT_SECONDS:
            return None
        return max(
            RECONCILE_OPTIONAL_MIN_TIMEOUT_SECONDS,
            min(requested, available),
        )

    def _record_reconcile_degraded(
        self,
        *,
        kind: str,
        note: str,
        symbol: Any = None,
        side: Any = None,
        exchange_order_id: Any = None,
        error: Any = None,
    ) -> None:
        rows = self._reconcile_degraded_rows
        if rows is None:
            rows = []
            self._reconcile_degraded_rows = rows
        row: dict[str, Any] = {
            "kind": kind,
            "source": "okx_authoritative_current_position",
            "requires_attention": False,
            "degraded": True,
            "note": note,
        }
        if symbol not in (None, ""):
            row["symbol"] = symbol
        if side not in (None, ""):
            row["side"] = side
        if exchange_order_id not in (None, ""):
            row["exchange_order_id"] = exchange_order_id
        if error not in (None, ""):
            row["error"] = safe_error_text(error, limit=180)
        rows.append(row)

    def _with_reconcile_degraded_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        degraded_rows = self._reconcile_degraded_rows or []
        self._reconcile_degraded_rows = None
        if not degraded_rows:
            return rows
        return [*rows, *degraded_rows]

    def _required_symbol_normalizer(self) -> Callable[[Any], str]:
        if self.symbol_normalizer is None:
            raise RuntimeError("OkxSyncService requires symbol_normalizer dependency")
        return self.symbol_normalizer

    def _required_model_execution_mode_provider(self) -> Callable[[str], str]:
        if self.model_execution_mode_provider is None:
            raise RuntimeError("OkxSyncService requires model_execution_mode_provider dependency")
        return self.model_execution_mode_provider

    def _required_okx_executor_provider(self) -> Callable[[str], Awaitable[Any]]:
        if self.okx_executor_provider is None:
            raise RuntimeError("OkxSyncService requires okx_executor_provider dependency")
        return self.okx_executor_provider

    def _required_float_parser(self) -> Callable[[Any, float], float]:
        if self.float_parser is None:
            raise RuntimeError("OkxSyncService requires float_parser dependency")
        return self.float_parser

    def _required_active_okx_provider(self) -> Callable[[], Any | None]:
        if self.active_okx_provider is None:
            raise RuntimeError("OkxSyncService requires active_okx_provider dependency")
        return self.active_okx_provider

    def _required_paper_okx_provider(self) -> Callable[[], Any | None]:
        if self.paper_okx_provider is None:
            raise RuntimeError("OkxSyncService requires paper_okx_provider dependency")
        return self.paper_okx_provider

    def _required_exchange_position_open_checker(
        self,
    ) -> Callable[[dict[str, Any]], bool]:
        if self.exchange_position_open_checker is None:
            raise RuntimeError("OkxSyncService requires exchange_position_open_checker dependency")
        return self.exchange_position_open_checker

    def _required_exchange_protection_map_provider(
        self,
    ) -> Callable[[Any, list[dict]], Awaitable[dict[tuple[str, str], dict[str, Any]]]]:
        if self.exchange_protection_map_provider is None:
            raise RuntimeError(
                "OkxSyncService requires exchange_protection_map_provider dependency"
            )
        return self.exchange_protection_map_provider

    def _required_position_protection_fallback_provider(
        self,
    ) -> Callable[..., Awaitable[dict[str, Any]]]:
        if self.position_protection_fallback_provider is None:
            raise RuntimeError(
                "OkxSyncService requires position_protection_fallback_provider dependency"
            )
        return self.position_protection_fallback_provider

    def _required_position_profit_peak_recorder(self) -> Callable[..., Any]:
        if self.position_profit_peak_recorder is None:
            raise RuntimeError("OkxSyncService requires position_profit_peak_recorder dependency")
        return self.position_profit_peak_recorder

    def _required_position_age_minutes_provider(self) -> Callable[[Any], float | None]:
        if self.position_age_minutes_provider is None:
            raise RuntimeError("OkxSyncService requires position_age_minutes_provider dependency")
        return self.position_age_minutes_provider

    def _required_position_profit_peak_pruner(self) -> Callable[[list[dict[str, Any]]], None]:
        if self.position_profit_peak_pruner is None:
            raise RuntimeError("OkxSyncService requires position_profit_peak_pruner dependency")
        return self.position_profit_peak_pruner

    def _required_local_position_snapshot_syncer(self) -> Callable[..., bool]:
        if self.local_position_snapshot_syncer is None:
            raise RuntimeError("OkxSyncService requires local_position_snapshot_syncer dependency")
        return self.local_position_snapshot_syncer

    def _required_datetime_from_ms_parser(self) -> Callable[[Any], datetime]:
        if self.datetime_from_ms_parser is None:
            raise RuntimeError("OkxSyncService requires datetime_from_ms_parser dependency")
        return self.datetime_from_ms_parser

    def _required_exchange_close_fill_finder(
        self,
    ) -> Callable[[Any], Awaitable[dict[str, Any]]]:
        if self.exchange_close_fill_finder is None:
            raise RuntimeError("OkxSyncService requires exchange_close_fill_finder dependency")
        return self.exchange_close_fill_finder

    def _required_fresh_feature_vector_provider(self) -> Callable[[str], Awaitable[Any | None]]:
        if self.fresh_feature_vector_provider is None:
            raise RuntimeError("OkxSyncService requires fresh_feature_vector_provider dependency")
        return self.fresh_feature_vector_provider

    def _required_market_value_reader(self) -> Callable[[Any, str], Any]:
        if self.market_value_reader is None:
            raise RuntimeError("OkxSyncService requires market_value_reader dependency")
        return self.market_value_reader

    def _required_entry_fee_provider(self) -> Callable[[Any, Any, float], Awaitable[float]]:
        if self.entry_fee_provider is None:
            raise RuntimeError("OkxSyncService requires entry_fee_provider dependency")
        return self.entry_fee_provider

    def _required_exchange_sync_close_decision_logger(
        self,
    ) -> Callable[..., Awaitable[int | None]]:
        if self.exchange_sync_close_decision_logger is None:
            raise RuntimeError(
                "OkxSyncService requires exchange_sync_close_decision_logger dependency"
            )
        return self.exchange_sync_close_decision_logger

    def _required_trade_reflection_recorder(self) -> Callable[..., Awaitable[None]]:
        if self.trade_reflection_recorder is None:
            raise RuntimeError("OkxSyncService requires trade_reflection_recorder dependency")
        return self.trade_reflection_recorder

    def _required_position_margin_calculator(self) -> Callable[[float, float | None], float]:
        if self.position_margin_calculator is None:
            raise RuntimeError("OkxSyncService requires position_margin_calculator dependency")
        return self.position_margin_calculator

    def _required_memory_position_remover(self) -> Callable[[str, str, str], None]:
        if self.memory_position_remover is None:
            raise RuntimeError("OkxSyncService requires memory_position_remover dependency")
        return self.memory_position_remover

    async def reconcile_positions(
        self,
        reason: str,
        timeout_seconds: float = 25.0,
        *,
        lock_wait_seconds: float | None = None,
        record_timeout_error: bool = True,
    ) -> list[dict]:
        lock = self._required_exchange_reconcile_lock()
        wait_budget = (
            min(max(float(timeout_seconds), 0.1), 1.0)
            if lock_wait_seconds is None
            else max(float(lock_wait_seconds), 0.0)
        )
        try:
            await asyncio.wait_for(lock.acquire(), timeout=wait_budget)
        except TimeoutError:
            logger.info(
                "exchange position reconciliation already running; skipping duplicate request",
                reason=reason,
                lock_wait_seconds=round(wait_budget, 3),
            )
            return []

        previous_deadline = self._reconcile_deadline_monotonic
        self._reconcile_deadline_monotonic = asyncio.get_running_loop().time() + max(
            float(timeout_seconds or 0.0), 0.0
        )
        task = asyncio.create_task(self.reconcile_exchange_positions())
        try:
            done, _pending = await asyncio.wait({task}, timeout=timeout_seconds)
            if done:
                return task.result()

            timeout_reason = (
                f"exchange position reconciliation timed out during {reason}; "
                "the single-flight transaction continues in background"
            )
            if record_timeout_error:
                self._record_round_error(timeout_reason)
            logger.warning(timeout_reason)

            def finish_background_reconciliation(completed: asyncio.Task[Any]) -> None:
                self._reconcile_deadline_monotonic = previous_deadline
                if lock.locked():
                    lock.release()
                try:
                    rows = completed.result()
                except asyncio.CancelledError:
                    logger.warning("background exchange position reconciliation was cancelled")
                except Exception as exc:
                    logger.warning(
                        "background exchange position reconciliation failed",
                        error=safe_error_text(exc),
                    )
                else:
                    logger.info(
                        "background exchange position reconciliation completed",
                        reconciled_count=len(rows or []),
                    )

            task.add_done_callback(finish_background_reconciliation)
            return []
        except asyncio.CancelledError:
            task.cancel()
            task.add_done_callback(_consume_background_task_result)
            self._reconcile_deadline_monotonic = previous_deadline
            if lock.locked():
                lock.release()
            raise
        finally:
            if task.done():
                self._reconcile_deadline_monotonic = previous_deadline
                if lock.locked():
                    lock.release()

    async def refresh_position_prices(self, feature_vectors: dict[str, Any]) -> Any:
        """Update persisted open-position prices and unrealized PnL."""
        refresh_started = time.perf_counter()
        diagnostics: dict[str, Any] = {
            "source": "position_price_refresh",
            "exchange_snapshot_count": 0,
            "open_position_count": 0,
            "updated_position_count": 0,
        }
        record_position_profit_peak = self._required_position_profit_peak_recorder()
        position_age_minutes = self._required_position_age_minutes_provider()
        prune_position_profit_peaks = self._required_position_profit_peak_pruner()
        normalize_symbol = self.symbol_normalizer or (lambda value: str(value or ""))
        parse_float = self.float_parser or _float_value
        exchange_snapshots: dict[tuple[str, str, str], dict[str, Any]] = {}

        def pool_status(session: Any) -> str | None:
            try:
                bind = session.get_bind()
                pool = getattr(bind, "pool", None)
                status = pool.status() if pool is not None else None
                return str(status) if status else None
            except Exception:
                return None

        try:
            candidates: list[tuple[str, Any]] = []
            active_okx = self.active_okx_provider() if self.active_okx_provider else None
            active_mode = "live" if mode_manager.mode.value == "live" else "paper"
            if active_okx:
                candidates.append((active_mode, active_okx))
            paper_okx = self.paper_okx_provider() if self.paper_okx_provider else None
            if paper_okx and all(executor is not paper_okx for _mode, executor in candidates):
                candidates.append(("paper", paper_okx))

            async def fetch_source_positions(
                source_mode: str,
                executor: Any,
            ) -> tuple[str, list[dict[str, Any]]]:
                try:
                    exchange_positions = await asyncio.wait_for(
                        executor.get_positions_strict(),
                        timeout=4.0,
                    )
                except Exception as exc:
                    logger.debug(
                        "OKX position snapshot source unavailable during price refresh",
                        source_mode=source_mode,
                        error=safe_error_text(exc),
                    )
                    return source_mode, []
                return source_mode, list(exchange_positions or [])

            exchange_fetch_started = time.perf_counter()
            source_snapshots = await asyncio.gather(
                *(
                    fetch_source_positions(source_mode, executor)
                    for source_mode, executor in candidates
                )
            )
            diagnostics["exchange_snapshot_fetch_ms"] = round(
                (time.perf_counter() - exchange_fetch_started) * 1000,
                3,
            )
            for source_mode, exchange_positions in source_snapshots:
                for exchange_pos in exchange_positions or []:
                    snapshot = parse_exchange_position_snapshot(
                        exchange_pos,
                        symbol_normalizer=normalize_symbol,
                    )
                    if not snapshot:
                        continue
                    key = (source_mode, str(snapshot["symbol"]), str(snapshot["side"]))
                    exchange_snapshots[key] = snapshot
            diagnostics["exchange_snapshot_count"] = len(exchange_snapshots)
        except Exception as exc:
            diagnostics["exchange_snapshot_error"] = safe_error_text(exc, limit=180)
            logger.debug(
                "OKX position snapshot unavailable during price refresh; using feature prices",
                error=safe_error_text(exc),
            )
        try:
            async with get_session_ctx() as session:
                trade_repo = TradeRepository(session)
                diagnostics["database_pool_before_load"] = pool_status(session)
                load_started = time.perf_counter()
                try:
                    positions = await asyncio.wait_for(
                        trade_repo.get_open_positions(),
                        timeout=POSITION_PRICE_REFRESH_DB_LOAD_TIMEOUT_SECONDS,
                    )
                except TimeoutError:
                    diagnostics["database_deferred"] = {
                        "reason": "connection_pool_or_open_position_query_timeout",
                        "timeout_seconds": POSITION_PRICE_REFRESH_DB_LOAD_TIMEOUT_SECONDS,
                        "note": (
                            "Price persistence was deferred because the database was busy; "
                            "position review continues with the current round's feature data."
                        ),
                    }
                    return diagnostics
                diagnostics["load_open_positions_ms"] = round(
                    (time.perf_counter() - load_started) * 1000,
                    3,
                )
                diagnostics["database_pool_after_load"] = pool_status(session)
                diagnostics["open_position_count"] = len(positions)
                open_context: list[dict[str, Any]] = []
                price_updates: list[tuple[Any, float, float]] = []
                prepare_started = time.perf_counter()

                for pos in positions:
                    execution_mode = _position_execution_mode(pos)
                    key = (
                        execution_mode,
                        normalize_symbol(pos.symbol),
                        str(pos.side or "").lower(),
                    )
                    snapshot = exchange_snapshots.get(key)
                    if snapshot:
                        current_price = (
                            parse_float(snapshot.get("mark_price"), 0.0)
                            or parse_float(snapshot.get("last_price"), 0.0)
                            or pos.current_price
                            or pos.entry_price
                        )
                    else:
                        fv = feature_vectors.get(pos.symbol)
                        fv_price = getattr(fv, "current_price", None) if fv is not None else None
                        current_price = (
                            fv_price if fv_price else pos.current_price or pos.entry_price
                        )
                    if not current_price or current_price <= 0:
                        continue

                    snapshot_upl = parse_float(snapshot.get("upl"), 0.0) if snapshot else 0.0
                    if snapshot and snapshot.get("upl") is not None:
                        unrealized_pnl = snapshot_upl
                    elif pos.side == "short":
                        unrealized_pnl = (pos.entry_price - current_price) * pos.quantity
                    else:
                        unrealized_pnl = (current_price - pos.entry_price) * pos.quantity

                    price_updates.append((pos, float(current_price), float(unrealized_pnl)))
                    open_context.append(
                        {
                            "model_name": pos.model_name,
                            "symbol": pos.symbol,
                            "side": pos.side,
                            "current_price": float(current_price),
                            "entry_price": float(pos.entry_price or 0.0),
                            "unrealized_pnl": float(unrealized_pnl),
                            "created_at": pos.created_at,
                            "is_open": True,
                        }
                    )
                    record_position_profit_peak(
                        model_name=pos.model_name,
                        symbol=pos.symbol,
                        side=pos.side,
                        current_price=float(current_price),
                        entry_price=float(pos.entry_price or 0.0),
                        unrealized_pnl=float(unrealized_pnl),
                        hold_minutes=position_age_minutes(pos.created_at),
                        quantity=float(pos.quantity or 0.0),
                    )
                diagnostics["prepare_updates_ms"] = round(
                    (time.perf_counter() - prepare_started) * 1000,
                    3,
                )
                persist_started = time.perf_counter()
                update_many = getattr(trade_repo, "update_open_position_prices", None)
                if callable(update_many):
                    await update_many(price_updates)
                else:
                    # Compatibility for isolated test doubles and external repository adapters.
                    for pos, current_price, unrealized_pnl in price_updates:
                        await trade_repo.update_position_price(
                            pos.id,
                            current_price,
                            unrealized_pnl,
                        )
                diagnostics["persist_updates_ms"] = round(
                    (time.perf_counter() - persist_started) * 1000,
                    3,
                )
                diagnostics["updated_position_count"] = len(price_updates)
                prune_position_profit_peaks(open_context)
        except Exception as e:
            diagnostics["database_error"] = safe_error_text(e, limit=180)
            logger.warning("failed to refresh DB position prices", error=safe_error_text(e))
        diagnostics["total_ms"] = round((time.perf_counter() - refresh_started) * 1000, 3)
        return diagnostics

    async def has_matching_exchange_exit_position(
        self,
        model_name: str,
        decision: DecisionOutput,
    ) -> bool | None:
        """Check OKX before rejecting a close decision.

        Returns True when the exchange confirms a matching open position, False
        when the exchange snapshot is available and no match exists, and None
        when the snapshot is unavailable.
        """
        if not decision.is_exit:
            return True
        target_side = "long" if decision.action == Action.CLOSE_LONG else "short"
        normalize_symbol = self._required_symbol_normalizer()
        parse_float = self._required_float_parser()
        mode = self._required_model_execution_mode_provider()(model_name)
        try:
            executor = await self._required_okx_executor_provider()(mode)
            positions = await asyncio.wait_for(
                executor.get_positions_strict(decision.symbol),
                timeout=8.0,
            )
        except TimeoutError:
            logger.warning(
                "timed out checking OKX position before exit",
                model=model_name,
                symbol=decision.symbol,
            )
            return None
        except Exception as e:
            logger.warning(
                "failed to check OKX position before exit",
                model=model_name,
                symbol=decision.symbol,
                error=safe_error_text(e),
            )
            return None

        target_symbol = normalize_symbol(decision.symbol)
        for pos in positions or []:
            snapshot = parse_exchange_position_snapshot(
                pos,
                symbol_normalizer=normalize_symbol,
            )
            if snapshot:
                position_symbol = str(snapshot.get("symbol") or "")
                position_side = str(snapshot.get("side") or "").lower()
                quantity = parse_float(snapshot.get("contracts"), 0.0) or parse_float(
                    snapshot.get("quantity"),
                    0.0,
                )
            else:
                position_symbol = normalize_symbol(
                    _first_value(
                        _dict_value(pos.get("info")).get("instId"),
                        pos.get("instId"),
                        pos.get("symbol"),
                    )
                )
                position_side = str(
                    pos.get("side") or _dict_value(pos.get("info")).get("posSide") or ""
                ).lower()
                info = pos.get("info") if isinstance(pos.get("info"), dict) else {}
                quantity = parse_float(
                    pos.get("contracts")
                    or pos.get("size")
                    or pos.get("positionAmt")
                    or info.get("pos")
                    or info.get("qty"),
                    0.0,
                )
            if position_symbol != target_symbol:
                continue
            if position_side != target_side:
                continue
            if abs(quantity) > 0:
                return True
        return False

    async def active_exchange_order_for_local_position(
        self, pos: Position
    ) -> dict[str, Any] | None:
        """Return an active OKX order that can explain a temporary position mismatch."""
        normalize_symbol = self._required_symbol_normalizer()
        symbol = normalize_symbol(pos.symbol)
        side = str(pos.side or "").lower()
        if not symbol or side not in {"long", "short"}:
            return None

        entry_side = "buy" if side == "long" else "sell"
        exit_side = "sell" if side == "long" else "buy"
        timeout = self._reconcile_optional_timeout(8.0)
        if timeout is None:
            self._record_reconcile_degraded(
                kind="active_order_lookup_deferred",
                symbol=pos.symbol,
                side=pos.side,
                error="deadline_budget_exhausted",
                note=(
                    "OKX active-order lookup was deferred because the current-position "
                    "reconciliation round had already spent its safe time budget."
                ),
            )
            logger.info(
                "deferred active OKX order lookup because reconciliation budget is low",
                position_id=pos.id,
                symbol=pos.symbol,
                side=pos.side,
            )
            return {
                "kind": OPEN_ORDER_SNAPSHOT_UNKNOWN_KIND,
                "order_id": None,
                "side": None,
                "state": "deferred",
                "reduce_only": None,
                "error": "deadline_budget_exhausted",
            }
        try:
            executor = await self._required_okx_executor_provider()(pos.execution_mode or "paper")
            orders = await asyncio.wait_for(
                executor.get_open_orders_strict(pos.symbol),
                timeout=timeout,
            )
        except TimeoutError:
            self._record_reconcile_degraded(
                kind="active_order_lookup_unavailable",
                symbol=pos.symbol,
                side=pos.side,
                error="timeout",
                note=(
                    "OKX active-order lookup timed out; current positions stayed usable "
                    "and this optional mismatch check will be retried later."
                ),
            )
            logger.warning(
                "timed out checking active OKX orders before local close",
                position_id=pos.id,
                symbol=pos.symbol,
                side=pos.side,
            )
            return {
                "kind": OPEN_ORDER_SNAPSHOT_UNKNOWN_KIND,
                "order_id": None,
                "side": None,
                "state": "unavailable",
                "reduce_only": None,
                "error": "timeout",
            }
        except Exception as e:
            error_text = safe_error_text(e)
            if _is_missing_market_symbol_error(error_text):
                logger.warning(
                    "treat missing OKX market symbol as no active order for absent exchange position",
                    position_id=pos.id,
                    symbol=pos.symbol,
                    side=pos.side,
                    error=error_text,
                )
                return None
            logger.warning(
                "failed to check active OKX orders before local close",
                position_id=pos.id,
                symbol=pos.symbol,
                side=pos.side,
                error=error_text,
            )
            return {
                "kind": OPEN_ORDER_SNAPSHOT_UNKNOWN_KIND,
                "order_id": None,
                "side": None,
                "state": "unavailable",
                "reduce_only": None,
                "error": error_text,
            }

        active_states = {"", "open", "pending", "live", "partially_filled", "partial"}
        for order in orders or []:
            info = order.get("info") if isinstance(order.get("info"), dict) else {}
            order_symbol = normalize_symbol(order.get("symbol") or info.get("instId") or pos.symbol)
            if order_symbol != symbol:
                continue

            order_state = str(order.get("status") or info.get("state") or "").lower().strip()
            if order_state not in active_states:
                continue

            order_side = str(order.get("side") or info.get("side") or "").lower().strip()
            if order_side not in {"buy", "sell"}:
                continue

            reduce_only = order.get("reduceOnly")
            if reduce_only in (None, ""):
                reduce_only = info.get("reduceOnly")
            is_reduce_only = str(reduce_only).lower() == "true"
            ord_type = str(info.get("ordType") or order.get("type") or "").lower()

            if is_reduce_only and order_side == exit_side:
                return {
                    "kind": "exit",
                    "order_id": order.get("id") or info.get("ordId"),
                    "side": order_side,
                    "state": order_state or "open",
                    "reduce_only": True,
                }
            if (
                not is_reduce_only
                and order_side == entry_side
                and ord_type not in {"oco", "conditional", "trigger"}
            ):
                return {
                    "kind": "entry",
                    "order_id": order.get("id") or info.get("ordId"),
                    "side": order_side,
                    "state": order_state or "open",
                    "reduce_only": False,
                }
        return None

    async def _fetch_exchange_protection_map_with_timeout(
        self,
        provider: Callable[[Any, list[dict]], Awaitable[dict[tuple[str, str], dict[str, Any]]]],
        paper_okx: Any,
        exchange_positions: list[dict],
    ) -> dict[tuple[str, str], dict[str, Any]]:
        timeout = self._reconcile_optional_timeout(EXCHANGE_PROTECTION_MAP_TIMEOUT_SECONDS)
        if timeout is None:
            self._record_reconcile_degraded(
                kind="protection_order_map_deferred",
                error="deadline_budget_exhausted",
                note=(
                    "OKX TP/SL protection map lookup was deferred because the current-position "
                    "snapshot already consumed this reconciliation round's safe time budget."
                ),
            )
            logger.info(
                "deferred exchange protection order map because reconciliation budget is low"
            )
            return {}
        try:
            return await asyncio.wait_for(
                provider(paper_okx, exchange_positions),
                timeout=timeout,
            )
        except TimeoutError:
            reason = (
                "exchange protection order map timed out during reconciliation; "
                "continuing without exchange TP/SL map this round"
            )
            self._record_reconcile_degraded(
                kind="protection_order_map_unavailable",
                error="timeout",
                note=reason,
            )
            logger.warning(reason)
            return {}
        except Exception as exc:
            reason = (
                "exchange protection order map failed during reconciliation; "
                "continuing without exchange TP/SL map this round"
            )
            self._record_reconcile_degraded(
                kind="protection_order_map_unavailable",
                error=safe_error_text(exc),
                note=reason,
            )
            logger.warning(reason, error=safe_error_text(exc))
            return {}

    async def _find_exchange_close_fill_with_timeout(
        self,
        finder: Callable[[Any], Awaitable[dict[str, Any]]],
        position: Any,
        *,
        context: str,
    ) -> dict[str, Any]:
        timeout = self._reconcile_optional_timeout(EXCHANGE_CLOSE_FILL_LOOKUP_TIMEOUT_SECONDS)
        if timeout is None:
            reason = (
                f"exchange close-fill lookup deferred during {context}; "
                "current OKX positions were refreshed and historical close-fill repair "
                "will continue in a later sync round"
            )
            self._record_reconcile_degraded(
                kind="close_fill_lookup_deferred",
                symbol=getattr(position, "symbol", None),
                side=getattr(position, "side", None),
                error="deadline_budget_exhausted",
                note=reason,
            )
            logger.info(
                reason,
                position_id=getattr(position, "id", None),
                symbol=getattr(position, "symbol", None),
                side=getattr(position, "side", None),
            )
            return {"lookup_unavailable": True, "error": "deadline_budget_exhausted"}
        try:
            return await asyncio.wait_for(
                finder(position),
                timeout=timeout,
            )
        except TimeoutError:
            reason = (
                f"exchange close-fill lookup timed out during {context}; "
                "skipping local close reconciliation for this position this round"
            )
            self._record_reconcile_degraded(
                kind="close_fill_lookup_unavailable",
                symbol=getattr(position, "symbol", None),
                side=getattr(position, "side", None),
                error="timeout",
                note=reason,
            )
            logger.warning(
                reason,
                position_id=getattr(position, "id", None),
                symbol=getattr(position, "symbol", None),
                side=getattr(position, "side", None),
            )
            return {"lookup_unavailable": True, "error": "timeout"}
        except Exception as exc:
            error_text = safe_error_text(exc)
            if context == "missing exchange position" and _is_missing_market_symbol_error(
                error_text
            ):
                logger.warning(
                    "close-fill lookup skipped because OKX market symbol is missing",
                    position_id=getattr(position, "id", None),
                    symbol=getattr(position, "symbol", None),
                    side=getattr(position, "side", None),
                    error=error_text,
                )
                return {}
            reason = (
                f"exchange close-fill lookup failed during {context}; "
                "skipping local close reconciliation for this position this round"
            )
            self._record_reconcile_degraded(
                kind="close_fill_lookup_unavailable",
                symbol=getattr(position, "symbol", None),
                side=getattr(position, "side", None),
                error=error_text,
                note=reason,
            )
            logger.warning(
                reason,
                position_id=getattr(position, "id", None),
                symbol=getattr(position, "symbol", None),
                side=getattr(position, "side", None),
                error=error_text,
            )
            return {"lookup_unavailable": True, "error": error_text}

    async def reconcile_exchange_positions(self) -> list[dict]:
        """Reconcile local paper positions with actual OKX demo positions.

        OKX attached TP/SL orders can close positions without going through the
        AI decision loop. When that happens, close the local DB position using
        the exchange fill so the dashboard and account state remain truthful.
        """
        self._reconcile_degraded_rows = []
        normalize_symbol = self._required_symbol_normalizer()
        parse_float = self._required_float_parser()
        exchange_position_is_open = self._required_exchange_position_open_checker()
        paper_okx = self._required_paper_okx_provider()()
        if not paper_okx:
            return []
        fetch_exchange_protection_map = self._required_exchange_protection_map_provider()
        fallback_position_protection = self._required_position_protection_fallback_provider()
        sync_local_open_position_snapshot = self._required_local_position_snapshot_syncer()
        datetime_from_ms = self._required_datetime_from_ms_parser()
        find_exchange_close_fill = self._required_exchange_close_fill_finder()
        entry_fee_for_position = self._required_entry_fee_provider()
        log_exchange_sync_close_decision = self._required_exchange_sync_close_decision_logger()
        record_trade_reflection = self._required_trade_reflection_recorder()
        calculate_position_margin = self._required_position_margin_calculator()
        remove_memory_position = self._required_memory_position_remover()

        try:
            exchange_positions = await asyncio.wait_for(
                paper_okx.get_positions_strict(),
                timeout=6.0,
            )
        except TimeoutError:
            logger.warning("timed out fetching OKX positions for reconciliation")
            return []
        except Exception as e:
            logger.warning(
                "failed to fetch OKX positions for reconciliation",
                error=safe_error_text(e),
            )
            return []

        protection_by_key = await self._fetch_exchange_protection_map_with_timeout(
            fetch_exchange_protection_map,
            paper_okx,
            exchange_positions,
        )

        exchange_position_keys = {
            key
            for key in (
                _exchange_position_key(p, symbol_normalizer=normalize_symbol)
                for p in exchange_positions or []
                if exchange_position_is_open(p)
            )
            if key is not None
        }
        reconciled: list[dict] = []

        try:
            async with get_session_ctx() as session:
                trade_repo = TradeRepository(session)
                positions = await trade_repo.get_open_positions()
                local_open_keys = {
                    (
                        normalize_symbol(pos.symbol),
                        str(pos.side or "").lower(),
                    )
                    for pos in positions
                    if pos.execution_mode == "paper" and pos.is_open
                }

                for exchange_pos in exchange_positions or []:
                    if not exchange_position_is_open(exchange_pos):
                        continue
                    snapshot = parse_exchange_position_snapshot(
                        exchange_pos,
                        symbol_normalizer=normalize_symbol,
                    )
                    if not snapshot:
                        continue
                    symbol = str(snapshot["symbol"])
                    side = str(snapshot["side"])
                    key = (symbol, side)

                    info = exchange_pos.get("info") or {}
                    symbol_variants = trading_symbol_variants(symbol)
                    okx_inst_id = _okx_inst_id_from_position_payload(exchange_pos)
                    okx_pos_id = _okx_pos_id_from_position_payload(exchange_pos)
                    quantity = parse_float(snapshot.get("quantity"), 0.0)
                    entry_price = parse_float(snapshot.get("entry_price"), 0.0)
                    if quantity <= 0 or entry_price <= 0:
                        continue

                    current_price = (
                        parse_float(snapshot.get("mark_price"), 0.0)
                        or parse_float(snapshot.get("last_price"), 0.0)
                        or entry_price
                    )
                    leverage = (
                        parse_float(exchange_pos.get("leverage") or info.get("lever"), 1.0) or 1.0
                    )
                    exchange_unrealized = parse_float(snapshot.get("upl"), 0.0)
                    exchange_realized = parse_float(exchange_pos.get("realizedPnl"), 0.0)
                    protection = protection_by_key.get(key, {})
                    fallback_protection = await fallback_position_protection(
                        session,
                        symbol=symbol,
                        side=side,
                        entry_price=entry_price,
                    )
                    stop_loss_price = protection.get("stop_loss_price") or fallback_protection.get(
                        "stop_loss_price"
                    )
                    take_profit_price = protection.get(
                        "take_profit_price"
                    ) or fallback_protection.get("take_profit_price")

                    matching_local_positions = [
                        pos
                        for pos in positions
                        if (
                            pos.execution_mode == "paper"
                            and pos.is_open
                            and normalize_symbol(pos.symbol) == symbol
                            and str(pos.side or "").lower() == side
                        )
                    ]
                    if matching_local_positions:
                        local_quantity_before_by_id = {
                            getattr(pos, "id", id(pos)): parse_float(
                                getattr(pos, "quantity", 0.0),
                                0.0,
                            )
                            for pos in matching_local_positions
                        }
                        local_total_before = sum(
                            abs(quantity_before)
                            for quantity_before in local_quantity_before_by_id.values()
                        )
                        changed = sync_local_open_position_snapshot(
                            matching_local_positions,
                            exchange_quantity=quantity,
                            current_price=current_price,
                            entry_price=entry_price,
                            leverage=leverage,
                            exchange_unrealized=exchange_unrealized,
                            stop_loss_price=stop_loss_price,
                            take_profit_price=take_profit_price,
                        )
                        for local_position in matching_local_positions:
                            if okx_inst_id:
                                local_position.okx_inst_id = okx_inst_id
                            if okx_pos_id:
                                local_position.okx_pos_id = okx_pos_id
                        local_open_keys.add(key)
                        quantity_tolerance = max(
                            abs(local_total_before) * 0.001,
                            abs(quantity) * 0.001,
                            1e-8,
                        )
                        reduction_close_fill = None
                        if local_total_before > quantity + quantity_tolerance:
                            reduced_quantity = local_total_before - quantity
                            reduction_probe = SimpleNamespace(
                                symbol=symbol,
                                side=side,
                                quantity=reduced_quantity,
                                created_at=min(
                                    (
                                        getattr(pos, "created_at", None)
                                        for pos in matching_local_positions
                                        if getattr(pos, "created_at", None) is not None
                                    ),
                                    default=None,
                                ),
                            )
                            try:
                                reduction_close_fill = (
                                    await _confirmed_local_close_fill_for_position(
                                        session,
                                        matching_local_positions[0],
                                        target_quantity=reduced_quantity,
                                    )
                                )
                                if not reduction_close_fill:
                                    reduction_close_fill = (
                                        await self._find_exchange_close_fill_with_timeout(
                                            find_exchange_close_fill,
                                            reduction_probe,
                                            context="exchange quantity reduction",
                                        )
                                    )
                            except Exception as exc:
                                logger.warning(
                                    "failed to find OKX close fill for reduced open quantity",
                                    symbol=symbol,
                                    side=side,
                                    reduced_quantity=reduced_quantity,
                                    error=safe_error_text(exc),
                                )
                            if (reduction_close_fill or {}).get("lookup_unavailable"):
                                logger.warning(
                                    "skip exchange quantity reduction history because close fill lookup is unavailable",
                                    symbol=symbol,
                                    side=side,
                                    reduced_quantity=reduced_quantity,
                                )
                            else:
                                closed_slices = await self._record_exchange_quantity_reduction(
                                    session=session,
                                    trade_repo=trade_repo,
                                    positions=matching_local_positions,
                                    quantity_before_by_id=local_quantity_before_by_id,
                                    exchange_quantity=quantity,
                                    exit_price=current_price,
                                    close_fill=reduction_close_fill,
                                    entry_fee_for_position=entry_fee_for_position,
                                    log_exchange_sync_close_decision=log_exchange_sync_close_decision,
                                    record_trade_reflection=record_trade_reflection,
                                    calculate_position_margin=calculate_position_margin,
                                )
                                reconciled.extend(closed_slices)
                        if changed:
                            reconciled.append(
                                {
                                    "kind": "snapshot_update",
                                    "source": "okx_authoritative_current_position",
                                    "model_name": matching_local_positions[0].model_name,
                                    "symbol": symbol,
                                    "side": side,
                                    "quantity": quantity,
                                    "current_price": current_price,
                                    "note": "OKX 持仓数量或价格已变化，本地持仓快照已同步更新。",
                                }
                            )
                        continue

                    closed_position_result = await session.execute(
                        select(Position)
                        .where(
                            Position.execution_mode == "paper",
                            Position.symbol.in_(symbol_variants),
                            Position.side == side,
                            Position.is_open.is_(False),
                        )
                        .order_by(Position.created_at.desc())
                        .limit(1)
                    )
                    closed_position = closed_position_result.scalar_one_or_none()
                    if closed_position:
                        closed_position.is_open = True
                        closed_position.quantity = quantity
                        closed_position.entry_price = entry_price
                        closed_position.current_price = current_price
                        closed_position.leverage = leverage
                        closed_position.unrealized_pnl = exchange_unrealized
                        closed_position.realized_pnl = exchange_realized
                        closed_position.stop_loss_price = stop_loss_price
                        closed_position.take_profit_price = take_profit_price
                        closed_position.closed_at = None
                        closed_position.okx_inst_id = okx_inst_id or closed_position.okx_inst_id
                        closed_position.okx_pos_id = okx_pos_id or closed_position.okx_pos_id
                        closed_position.close_exchange_order_id = None
                        closed_position.updated_at = datetime.now(UTC)
                        local_open_keys.add(key)
                        reconciled.append(
                            {
                                "kind": "reopened_local_position",
                                "source": "okx_authoritative_current_position",
                                "model_name": closed_position.model_name,
                                "symbol": symbol,
                                "side": side,
                                "entry_price": entry_price,
                                "note": "OKX 仍有持仓，本地之前误记为已平仓，已重新打开本地持仓记录。",
                            }
                        )
                        logger.warning(
                            "reopened local position still open on OKX",
                            position_id=closed_position.id,
                            symbol=symbol,
                            side=side,
                        )
                        continue

                    entry_side = "buy" if side == "long" else "sell"
                    order_result = await session.execute(
                        select(Order)
                        .where(
                            Order.execution_mode == "paper",
                            Order.symbol.in_(symbol_variants),
                            Order.side == entry_side,
                            Order.exchange_order_id.is_not(None),
                            Order.exchange_order_id != "",
                            Order.status.in_(
                                [
                                    OrderStatus.OPEN.value,
                                    OrderStatus.PENDING.value,
                                    OrderStatus.PARTIAL.value,
                                    OrderStatus.FILLED.value,
                                ]
                            ),
                        )
                        .order_by(Order.created_at.desc())
                        .limit(1)
                    )
                    order = order_result.scalar_one_or_none()
                    if not order:
                        continue
                    if not stop_loss_price or not take_profit_price:
                        order_fallback_protection = await fallback_position_protection(
                            session,
                            symbol=symbol,
                            side=side,
                            entry_price=entry_price,
                            order=order,
                        )
                        stop_loss_price = stop_loss_price or order_fallback_protection.get(
                            "stop_loss_price"
                        )
                        take_profit_price = take_profit_price or order_fallback_protection.get(
                            "take_profit_price"
                        )

                    opened_at = datetime_from_ms(exchange_pos.get("timestamp") or info.get("cTime"))
                    order.status = OrderStatus.FILLED.value
                    order.quantity = order.quantity or quantity
                    order.price = order.price or entry_price
                    order.filled_at = order.filled_at or opened_at
                    position_symbol = _symbol_from_okx_inst_id_or_fallback(okx_inst_id, symbol)

                    position_payload = {
                        "model_name": order.model_name or ENSEMBLE_TRADER_NAME,
                        "execution_mode": "paper",
                        "symbol": position_symbol,
                        "side": side,
                        "quantity": quantity,
                        "entry_price": entry_price,
                        "current_price": current_price,
                        "leverage": leverage,
                        "unrealized_pnl": exchange_unrealized,
                        "realized_pnl": exchange_realized,
                        "stop_loss_price": stop_loss_price,
                        "take_profit_price": take_profit_price,
                    }
                    if okx_inst_id:
                        position_payload["okx_inst_id"] = okx_inst_id
                    if okx_pos_id:
                        position_payload["okx_pos_id"] = okx_pos_id
                    if order.exchange_order_id:
                        position_payload["entry_exchange_order_id"] = order.exchange_order_id
                    await trade_repo.open_position(position_payload)
                    local_open_keys.add(key)
                    reconciled.append(
                        {
                            "kind": "created_missing_local_position",
                            "source": "okx_authoritative_current_position",
                            "model_name": order.model_name or ENSEMBLE_TRADER_NAME,
                            "symbol": symbol,
                            "side": side,
                            "entry_price": entry_price,
                            "exchange_order_id": order.exchange_order_id,
                            "note": "OKX 已有持仓但本地缺失，已按执行订单补回持仓记录。",
                        }
                    )
                    logger.warning(
                        "synced missing local position from OKX",
                        symbol=symbol,
                        side=side,
                        order_id=order.exchange_order_id,
                    )

                for pos in positions:
                    if pos.execution_mode != "paper":
                        continue
                    if not pos.is_open:
                        continue
                    symbol = normalize_symbol(pos.symbol)
                    if (symbol, str(pos.side or "").lower()) in exchange_position_keys:
                        continue

                    try:
                        await session.refresh(pos)
                    except Exception as e:
                        logger.warning(
                            "failed to refresh local position before exchange reconciliation close",
                            position_id=pos.id,
                            symbol=pos.symbol,
                            side=pos.side,
                            error=safe_error_text(e),
                        )
                    if not pos.is_open:
                        logger.info(
                            "skip exchange reconciliation close; local position already closed",
                            position_id=pos.id,
                            symbol=pos.symbol,
                            side=pos.side,
                        )
                        continue

                    if await _position_entry_order_is_exchange_close_fill(session, pos):
                        now = datetime.now(UTC)
                        opened_at = pos.created_at
                        if opened_at and opened_at.tzinfo is None:
                            opened_at = opened_at.replace(tzinfo=UTC)
                        age_seconds = (now - opened_at).total_seconds() if opened_at else 0.0
                        quarantined = self._quarantine_orphan_local_position(
                            session=session,
                            pos=pos,
                            remove_memory_position=remove_memory_position,
                            age_seconds=age_seconds,
                            reason=(
                                "The local entry order is an OKX-confirmed close/reduce fill "
                                "with non-zero fillPnl, so this local open row is not an "
                                "exchange-backed current position."
                            ),
                        )
                        reconciled.append(
                            {
                                "kind": (
                                    "entry_order_close_fill_position_quarantined"
                                    if quarantined
                                    else "entry_order_close_fill_position_quarantine_skipped"
                                ),
                                "source": "okx_authoritative_current_position",
                                "model_name": pos.model_name,
                                "symbol": pos.symbol,
                                "side": pos.side,
                                "exchange_order_id": getattr(pos, "entry_exchange_order_id", None),
                                "requires_attention": False,
                                "training_policy": "exclude_until_manual_trust",
                                "note": (
                                    "The local row was created from an OKX close/reduce fill, "
                                    "not a clean entry fill; it was quarantined out of current "
                                    "positions and clean training."
                                ),
                            }
                        )
                        if quarantined:
                            logger.warning(
                                "quarantined local position created from OKX close fill",
                                position_id=pos.id,
                                symbol=pos.symbol,
                                side=pos.side,
                            )
                            continue

                    close_fill = await _confirmed_local_close_fill_for_position(
                        session,
                        pos,
                    )
                    if not close_fill:
                        close_fill = await self._find_exchange_close_fill_with_timeout(
                            find_exchange_close_fill,
                            pos,
                            context="missing exchange position",
                        )
                    if close_fill.get("lookup_unavailable"):
                        lookup_error = str(close_fill.get("error") or "").strip().lower()
                        temporary_lookup_unavailable = lookup_error in {
                            "timeout",
                            "deadline_budget_exhausted",
                        }
                        reconciled.append(
                            {
                                "kind": "close_fill_lookup_unavailable",
                                "source": "okx_authoritative_current_position",
                                "model_name": pos.model_name,
                                "symbol": pos.symbol,
                                "side": pos.side,
                                "exchange_order_id": None,
                                "requires_attention": not temporary_lookup_unavailable,
                                "degraded": temporary_lookup_unavailable,
                                "error": lookup_error or None,
                                "note": (
                                    "OKX close-fill lookup was temporarily unavailable; "
                                    "current positions stayed usable and historical close-fill "
                                    "repair will retry in a later sync round."
                                    if temporary_lookup_unavailable
                                    else (
                                        "OKX close-fill lookup was unavailable; local position "
                                        "was left open until a real close fill or stable exchange "
                                        "state is confirmed."
                                    )
                                ),
                            }
                        )
                        continue
                    if not close_fill.get("order_id"):
                        active_order = await self.active_exchange_order_for_local_position(pos)
                        if active_order:
                            if active_order.get("kind") == OPEN_ORDER_SNAPSHOT_UNKNOWN_KIND:
                                active_order_error = (
                                    str(active_order.get("error") or "").strip().lower()
                                )
                                temporary_order_unavailable = active_order_error in {
                                    "timeout",
                                    "deadline_budget_exhausted",
                                }
                                logger.warning(
                                    "skip synthetic local close because OKX open order snapshot is unavailable",
                                    position_id=pos.id,
                                    symbol=pos.symbol,
                                    side=pos.side,
                                    error=active_order.get("error"),
                                )
                                reconciled.append(
                                    {
                                        "kind": "active_order_snapshot_unavailable",
                                        "source": "okx_authoritative_current_position",
                                        "model_name": pos.model_name,
                                        "symbol": pos.symbol,
                                        "side": pos.side,
                                        "exchange_order_id": None,
                                        "requires_attention": not temporary_order_unavailable,
                                        "degraded": temporary_order_unavailable,
                                        "error": active_order_error or None,
                                        "note": (
                                            "OKX 暂时没有返回对应持仓，且挂单状态查询失败或超时；"
                                            "为避免把查询失败误判为没有挂单，本地不估算平仓，等待下一轮同步确认。"
                                        ),
                                    }
                                )
                                continue
                            logger.warning(
                                "skip synthetic local close because OKX still has active order",
                                position_id=pos.id,
                                symbol=pos.symbol,
                                side=pos.side,
                                order_id=active_order.get("order_id"),
                                order_kind=active_order.get("kind"),
                                order_state=active_order.get("state"),
                            )
                            reconciled.append(
                                {
                                    "kind": "active_exchange_order_present",
                                    "source": "okx_authoritative_current_position",
                                    "model_name": pos.model_name,
                                    "symbol": pos.symbol,
                                    "side": pos.side,
                                    "exchange_order_id": active_order.get("order_id"),
                                    "active_order_kind": active_order.get("kind"),
                                    "requires_attention": False,
                                    "note": (
                                        "OKX 暂时没有返回对应持仓，但仍存在挂单/追单中的"
                                        f"{'平仓' if active_order.get('kind') == 'exit' else '开仓'}委托；"
                                        "本地不估算平仓，等待 OKX 成交、撤单或下一轮同步确认。"
                                    ),
                                }
                            )
                            continue
                        now = datetime.now(UTC)
                        opened_at = pos.created_at
                        if opened_at and opened_at.tzinfo is None:
                            opened_at = opened_at.replace(tzinfo=UTC)
                        age_seconds = (now - opened_at).total_seconds() if opened_at else 0.0
                        if age_seconds < UNCONFIRMED_EXCHANGE_CLOSE_GRACE_SECONDS:
                            logger.warning(
                                "exchange position missing but close fill not found; waiting before local close",
                                position_id=pos.id,
                                symbol=pos.symbol,
                                side=pos.side,
                                age_seconds=round(age_seconds, 1),
                            )
                            continue

                        quarantined = self._quarantine_orphan_local_position(
                            session=session,
                            pos=pos,
                            remove_memory_position=remove_memory_position,
                            age_seconds=age_seconds,
                        )
                        reconciled.append(
                            {
                                "kind": (
                                    "orphan_local_position_quarantined"
                                    if quarantined
                                    else "missing_exchange_position_without_close_fill"
                                ),
                                "source": "okx_authoritative_current_position",
                                "model_name": pos.model_name,
                                "symbol": pos.symbol,
                                "side": pos.side,
                                "exchange_order_id": None,
                                "requires_attention": not quarantined,
                                "training_policy": (
                                    "exclude_until_manual_trust" if quarantined else None
                                ),
                                "note": (
                                    (
                                        "OKX current positions do not contain this local row, no "
                                        "matching close fill or active exchange order was found, and "
                                        "the row is older than the confirmation grace window; the "
                                        "local row was quarantined out of the current-position truth "
                                        "set and excluded from clean training."
                                    )
                                    if quarantined
                                    else (
                                        "OKX did not return this position and no matching close fill "
                                        "was found; the local position remains open until an exchange "
                                        "fill or authoritative position snapshot confirms closure."
                                    )
                                ),
                            }
                        )
                        if quarantined:
                            logger.warning(
                                "quarantined orphan local position missing on OKX",
                                position_id=pos.id,
                                symbol=pos.symbol,
                                side=pos.side,
                                age_seconds=round(age_seconds, 1),
                            )
                        else:
                            logger.warning(
                                "skip local close because OKX close fill is missing",
                                position_id=pos.id,
                                symbol=pos.symbol,
                                side=pos.side,
                            )
                        continue
                    exit_price = close_fill.get("price") or pos.current_price or pos.entry_price
                    if not exit_price or exit_price <= 0:
                        logger.warning(
                            "skip local close; invalid OKX close fill price",
                            position_id=pos.id,
                            symbol=pos.symbol,
                            side=pos.side,
                            exchange_order_id=close_fill.get("order_id"),
                        )
                        continue
                    close_fee = float(close_fill.get("fee") or 0.0)
                    close_order_id = str(close_fill.get("order_id") or "").strip()
                    close_okx_inst_id = _okx_inst_id_from_close_fill(
                        close_fill,
                        fallback=pos.symbol,
                    )
                    close_okx_pos_id = _okx_pos_id_from_close_fill(close_fill)
                    from services.okx_realized_pnl import gross_pnl_with_okx_override

                    gross_pnl, gross_pnl_source = gross_pnl_with_okx_override(
                        side=str(pos.side or "").lower(),
                        entry_price=pos.entry_price,
                        exit_price=exit_price,
                        close_qty=pos.quantity,
                        okx_payload=close_fill,
                        okx_total_qty=close_fill.get("quantity") if close_fill else pos.quantity,
                    )
                    close_side = "buy" if pos.side == "short" else "sell"
                    entry_fee = await entry_fee_for_position(session, pos, pos.quantity)
                    calculate_position_margin(pos.entry_price * pos.quantity, pos.leverage)
                    raw_funding_fee, funding_fee_source = funding_fee_from_payload(close_fill)
                    funding_fee = proportional_signed_value(
                        raw_funding_fee,
                        pos.quantity,
                        close_fill.get("quantity") if close_fill else pos.quantity,
                    )
                    settlement = build_position_settlement_snapshot(
                        close_fill_pnl=gross_pnl,
                        entry_fee=entry_fee,
                        close_fee=close_fee,
                        funding_fee=funding_fee,
                        status=SETTLEMENT_STATUS_SETTLING,
                        source="okx_authoritative_reconcile",
                        synced_at=close_fill.get("timestamp") or datetime.now(UTC),
                        raw={
                            "gross_pnl_source": gross_pnl_source,
                            "funding_fee_source": funding_fee_source,
                            "close_exchange_order_id": close_order_id,
                            "close_quantity": pos.quantity,
                            "fill_quantity": close_fill.get("quantity") if close_fill else None,
                        },
                    )
                    realized_pnl = settlement.realized_pnl

                    reconcile_origin = _exchange_reconcile_close_origin(pos, close_fill)
                    pos.is_open = False
                    pos.current_price = exit_price
                    pos.unrealized_pnl = 0.0
                    apply_position_settlement_snapshot(pos, settlement)
                    pos.closed_at = close_fill.get("timestamp") or datetime.now(UTC)
                    if close_okx_inst_id:
                        pos.okx_inst_id = close_okx_inst_id
                    if close_okx_pos_id:
                        pos.okx_pos_id = close_okx_pos_id
                    if close_order_id:
                        pos.close_exchange_order_id = close_order_id
                    decision_id = await log_exchange_sync_close_decision(
                        session=session,
                        pos=pos,
                        exit_price=exit_price,
                        realized_pnl=realized_pnl,
                        closed_at=pos.closed_at,
                        reason=(
                            "OKX 已返回平仓成交，系统同步为平仓记录；"
                            "这通常来自 OKX 止盈止损、手动平仓或交易所侧自动平仓。"
                        ),
                        close_fill={**close_fill, "reconcile_origin": reconcile_origin},
                        reconcile_origin=reconcile_origin,
                    )
                    await record_trade_reflection(
                        session,
                        pos,
                        exit_price=exit_price,
                        entry_fee=entry_fee,
                        close_fee=close_fee,
                        gross_pnl=gross_pnl,
                        source="okx_reconcile",
                        decision=None,
                    )

                    existing_close_order = None
                    if close_order_id:
                        existing_close_order_result = await session.execute(
                            select(Order)
                            .where(
                                Order.execution_mode == pos.execution_mode,
                                Order.exchange_order_id == close_order_id,
                            )
                            .limit(1)
                        )
                        existing_close_order = existing_close_order_result.scalar_one_or_none()

                    close_order_payload = _okx_close_fill_order_payload(
                        model_name=pos.model_name,
                        execution_mode=pos.execution_mode,
                        symbol=pos.symbol,
                        side=close_side,
                        quantity=pos.quantity,
                        price=exit_price,
                        fee=close_fee,
                        decision_id=decision_id,
                        close_order_id=close_order_id,
                        filled_at=pos.closed_at,
                        close_fill=close_fill,
                        okx_inst_id=close_okx_inst_id,
                    )
                    if existing_close_order is not None:
                        _apply_okx_close_fill_order_payload(
                            existing_close_order, close_order_payload
                        )
                    else:
                        await trade_repo.create_order(close_order_payload)

                    remove_memory_position(pos.model_name, pos.symbol, pos.side)

                    reconciled.append(
                        {
                            "kind": "closed_from_okx_close_fill",
                            "source": "okx_authoritative_current_position",
                            "model_name": pos.model_name,
                            "symbol": pos.symbol,
                            "side": pos.side,
                            "exit_price": exit_price,
                            "realized_pnl": realized_pnl,
                            "gross_pnl": gross_pnl,
                            "fees": entry_fee + close_fee,
                            "exchange_order_id": close_order_id or None,
                        }
                    )

        except Exception as e:
            logger.warning("exchange position reconciliation failed", error=safe_error_text(e))
            return self._with_reconcile_degraded_rows(reconciled)

        if reconciled:
            logger.info(
                "reconciled exchange-closed positions", count=len(reconciled), positions=reconciled
            )
        return self._with_reconcile_degraded_rows(reconciled)

    def _quarantine_orphan_local_position(
        self,
        *,
        session: Any,
        pos: Position,
        remove_memory_position: Callable[[str, str, str], None],
        age_seconds: float,
        reason: str | None = None,
    ) -> bool:
        """Move OKX-denied local open rows out of the current-position truth set.

        This deliberately does not invent an OKX close fill, close order, or PnL.
        The row is marked closed with a synthetic quarantine close id and excluded
        from clean training through a TradeReflection marker.
        """

        if str(getattr(pos, "execution_mode", "") or "").lower() != "paper":
            return False
        if not bool(getattr(pos, "is_open", False)):
            return False
        if str(getattr(pos, "close_exchange_order_id", "") or "").strip():
            return False
        position_id = int(getattr(pos, "id", 0) or 0)
        if position_id <= 0:
            return False
        now = datetime.now(UTC)
        current_price = _float_value(getattr(pos, "current_price", None), 0.0)
        if current_price <= 0:
            current_price = _float_value(getattr(pos, "entry_price", None), 0.0)
        old_unrealized_pnl = _float_value(getattr(pos, "unrealized_pnl", None), 0.0)
        pos.is_open = False
        if current_price > 0:
            pos.current_price = current_price
        pos.unrealized_pnl = 0.0
        pos.realized_pnl = 0.0
        pos.closed_at = now
        pos.close_exchange_order_id = f"{ORPHAN_QUARANTINE_CLOSE_PREFIX}{position_id}"
        if hasattr(pos, "updated_at"):
            pos.updated_at = now

        plan = {
            "position_id": position_id,
            "symbol": str(getattr(pos, "symbol", "") or ""),
            "side": str(getattr(pos, "side", "") or ""),
            "model_name": str(getattr(pos, "model_name", "") or ENSEMBLE_TRADER_NAME),
            "execution_mode": str(getattr(pos, "execution_mode", "") or "paper"),
            "quantity": abs(_float_value(getattr(pos, "quantity", None), 0.0)),
            "entry_price": _float_value(getattr(pos, "entry_price", None), 0.0),
            "old_current_price": current_price,
            "old_unrealized_pnl": old_unrealized_pnl,
            "age_seconds": round(float(age_seconds or 0.0), 6),
            "source": ORPHAN_QUARANTINE_REFLECTION_SOURCE,
            "reason": reason
            or (
                "OKX current positions do not contain this local open row, and no "
                "matching close fill or active exchange order was found after the "
                "confirmation grace window."
            ),
        }
        add = getattr(session, "add", None)
        if callable(add):
            add(
                TradeReflection(
                    position_id=position_id,
                    model_name=plan["model_name"],
                    execution_mode=plan["execution_mode"],
                    symbol=plan["symbol"],
                    side=plan["side"],
                    entry_price=plan["entry_price"],
                    exit_price=current_price,
                    quantity=plan["quantity"],
                    realized_pnl=0.0,
                    fee_estimate=0.0,
                    hold_minutes=max(float(age_seconds or 0.0) / 60.0, 0.0),
                    closed_at=now,
                    outcome="flat",
                    mistake_summary=("local orphan position was absent from OKX current positions"),
                    improvement_summary=(
                        "exclude this non-OKX-backed local row from training and current capacity"
                    ),
                    expert_lessons={
                        "source": ORPHAN_QUARANTINE_REFLECTION_SOURCE,
                        "training_policy": "exclude_until_manual_trust",
                        "repair_plan": plan,
                    },
                    source=ORPHAN_QUARANTINE_REFLECTION_SOURCE,
                )
            )
        try:
            remove_memory_position(
                str(getattr(pos, "model_name", "") or ENSEMBLE_TRADER_NAME),
                str(getattr(pos, "symbol", "") or ""),
                str(getattr(pos, "side", "") or ""),
            )
        except Exception as exc:
            logger.warning(
                "failed to remove quarantined orphan position from memory",
                position_id=position_id,
                error=safe_error_text(exc),
            )
        return True

    async def _record_exchange_quantity_reduction(
        self,
        *,
        session: Any,
        trade_repo: TradeRepository,
        positions: list[Position],
        quantity_before_by_id: dict[Any, float],
        exchange_quantity: float,
        exit_price: float,
        close_fill: dict[str, Any] | None,
        entry_fee_for_position: Callable[[Any, Any, float], Awaitable[float]],
        log_exchange_sync_close_decision: Callable[..., Awaitable[int | None]],
        record_trade_reflection: Callable[..., Awaitable[None]],
        calculate_position_margin: Callable[[float, float | None], float],
    ) -> list[dict[str, Any]]:
        """Persist closed history when OKX reports a smaller still-open position."""

        reconciled: list[dict[str, Any]] = []
        if not positions or exchange_quantity < 0 or exit_price <= 0:
            return reconciled

        for pos in positions:
            position_key = getattr(pos, "id", id(pos))
            before_qty = abs(float(quantity_before_by_id.get(position_key, 0.0) or 0.0))
            after_qty = abs(float(getattr(pos, "quantity", 0.0) or 0.0))
            closed_qty = max(before_qty - after_qty, 0.0)
            tolerance = max(before_qty * 0.001, after_qty * 0.001, 1e-8)
            if closed_qty <= tolerance:
                continue

            fill_quantity = abs(float((close_fill or {}).get("quantity") or 0.0))
            fill_matches_slice = bool(
                close_fill
                and fill_quantity > 0
                and abs(fill_quantity - closed_qty)
                <= max(fill_quantity * 0.05, closed_qty * 0.05, 1e-8)
            )
            if not fill_matches_slice:
                logger.warning(
                    "skip exchange quantity reduction history because no matching OKX close fill was found",
                    position_id=getattr(pos, "id", None),
                    symbol=pos.symbol,
                    side=pos.side,
                    closed_qty=closed_qty,
                    remaining_qty=after_qty,
                    exchange_quantity=exchange_quantity,
                    close_fill_quantity=fill_quantity,
                )
                continue
            close_price = (
                float((close_fill or {}).get("price") or 0.0) if fill_matches_slice else 0.0
            )
            close_price = close_price if close_price > 0 else exit_price
            close_fee = float((close_fill or {}).get("fee") or 0.0) if fill_matches_slice else 0.0
            close_order_id = (
                str((close_fill or {}).get("order_id") or "").strip() if fill_matches_slice else ""
            )
            if close_order_id:
                existing_slice_result = await session.execute(
                    select(Position).where(
                        Position.execution_mode == getattr(pos, "execution_mode", None),
                        Position.is_open.is_(False),
                        Position.close_exchange_order_id == close_order_id,
                    )
                )
                existing_slice_rows = (
                    existing_slice_result.scalars().all()
                    if callable(getattr(existing_slice_result, "scalars", None))
                    else []
                )
                existing_slice = next(
                    (
                        row
                        for row in existing_slice_rows
                        if isclose(
                            abs(_float_value(getattr(row, "quantity", None), 0.0)),
                            closed_qty,
                            rel_tol=1e-6,
                            abs_tol=1e-8,
                        )
                    ),
                    None,
                )
                if existing_slice is not None:
                    logger.info(
                        "skip duplicate exchange quantity reduction slice",
                        position_id=getattr(pos, "id", None),
                        existing_slice_id=getattr(existing_slice, "id", None),
                        exchange_order_id=close_order_id,
                        closed_qty=closed_qty,
                    )
                    continue
            close_okx_inst_id = _okx_inst_id_from_close_fill(
                close_fill,
                fallback=pos.symbol,
            )
            close_okx_pos_id = _okx_pos_id_from_close_fill(close_fill)
            closed_symbol = _symbol_from_okx_inst_id_or_fallback(close_okx_inst_id, pos.symbol)
            closed_at = (
                (close_fill or {}).get("timestamp")
                if fill_matches_slice and (close_fill or {}).get("timestamp")
                else datetime.now(UTC)
            )
            entry_fee = await entry_fee_for_position(session, pos, closed_qty)
            from services.okx_realized_pnl import gross_pnl_with_okx_override

            gross_pnl, gross_pnl_source = gross_pnl_with_okx_override(
                side=str(pos.side or "").lower(),
                entry_price=pos.entry_price,
                exit_price=close_price,
                close_qty=closed_qty,
                okx_payload=close_fill,
                okx_total_qty=(close_fill or {}).get("quantity") if close_fill else closed_qty,
            )
            close_side = "buy" if str(pos.side or "").lower() == "short" else "sell"
            raw_funding_fee, funding_fee_source = funding_fee_from_payload(close_fill)
            funding_fee = proportional_signed_value(
                raw_funding_fee,
                closed_qty,
                (close_fill or {}).get("quantity") if close_fill else closed_qty,
            )
            settlement = build_position_settlement_snapshot(
                close_fill_pnl=gross_pnl,
                entry_fee=entry_fee,
                close_fee=close_fee,
                funding_fee=funding_fee,
                status=SETTLEMENT_STATUS_SETTLING,
                source="okx_authoritative_reconcile",
                synced_at=closed_at,
                raw={
                    "gross_pnl_source": gross_pnl_source,
                    "funding_fee_source": funding_fee_source,
                    "close_exchange_order_id": close_order_id,
                    "close_quantity": closed_qty,
                    "fill_quantity": (close_fill or {}).get("quantity") if close_fill else None,
                    "partial_reduction": True,
                },
            )
            realized_pnl = settlement.realized_pnl
            closed_position_payload = {
                "model_name": pos.model_name,
                "execution_mode": pos.execution_mode,
                "symbol": closed_symbol,
                "side": pos.side,
                "quantity": closed_qty,
                "entry_price": pos.entry_price,
                "current_price": close_price,
                "leverage": pos.leverage,
                "unrealized_pnl": 0.0,
                "stop_loss_price": getattr(pos, "stop_loss_price", None),
                "take_profit_price": getattr(pos, "take_profit_price", None),
                "is_open": False,
                "closed_at": closed_at,
                "created_at": pos.created_at,
                **settlement_payload_fields(settlement),
            }
            okx_inst_id = getattr(pos, "okx_inst_id", None) or close_okx_inst_id
            okx_pos_id = getattr(pos, "okx_pos_id", None) or close_okx_pos_id
            entry_exchange_order_id = getattr(pos, "entry_exchange_order_id", None)
            if okx_inst_id:
                closed_position_payload["okx_inst_id"] = okx_inst_id
            if okx_pos_id:
                closed_position_payload["okx_pos_id"] = okx_pos_id
            if entry_exchange_order_id:
                closed_position_payload["entry_exchange_order_id"] = entry_exchange_order_id
            if close_order_id:
                closed_position_payload["close_exchange_order_id"] = close_order_id
            closed_position = await trade_repo.open_position(closed_position_payload)
            decision_pos = SimpleNamespace(
                **{
                    key: getattr(closed_position, key, closed_position_payload.get(key))
                    for key in (
                        "id",
                        "model_name",
                        "execution_mode",
                        "symbol",
                        "side",
                        "quantity",
                        "entry_price",
                        "current_price",
                        "leverage",
                        "realized_pnl",
                        "created_at",
                        "closed_at",
                        "stop_loss_price",
                        "take_profit_price",
                    )
                }
            )
            reconcile_origin = _exchange_reconcile_close_origin(
                decision_pos,
                close_fill if fill_matches_slice else None,
            )
            decision_id = await log_exchange_sync_close_decision(
                session=session,
                pos=decision_pos,
                exit_price=close_price,
                realized_pnl=realized_pnl,
                closed_at=closed_at,
                reason=(
                    "OKX reported a smaller still-open position; local history records "
                    "the reduced quantity as a closed slice."
                ),
                position_size_pct=(
                    min(max(closed_qty / before_qty, 0.0), 1.0) if before_qty > 0 else None
                ),
                close_fill={
                    **(close_fill if fill_matches_slice else {}),
                    "reconcile_origin": reconcile_origin,
                    "estimated": not fill_matches_slice,
                    "partial_reduction": True,
                    "price": close_price,
                    "quantity": closed_qty,
                    "remaining_quantity": exchange_quantity,
                    "gross_pnl": gross_pnl,
                    "entry_fee": entry_fee,
                    "fee": close_fee,
                    "pnl": realized_pnl,
                    "note": (
                        "exchange position quantity decreased while still open; "
                        "closed slice reconstructed from OKX remaining quantity"
                    ),
                },
                reconcile_origin=reconcile_origin,
            )
            await record_trade_reflection(
                session,
                closed_position,
                exit_price=close_price,
                entry_fee=entry_fee,
                close_fee=close_fee,
                gross_pnl=gross_pnl,
                source="okx_reconcile",
                decision=None,
            )
            existing_close_order = None
            if close_order_id:
                existing_close_order_result = await session.execute(
                    select(Order)
                    .where(
                        Order.execution_mode == getattr(pos, "execution_mode", None),
                        Order.exchange_order_id == close_order_id,
                    )
                    .limit(1)
                )
                existing_close_order = existing_close_order_result.scalar_one_or_none()
            close_order_payload = _okx_close_fill_order_payload(
                model_name=pos.model_name,
                execution_mode=pos.execution_mode,
                symbol=closed_symbol,
                side=close_side,
                quantity=closed_qty,
                price=close_price,
                fee=close_fee,
                decision_id=decision_id,
                close_order_id=close_order_id,
                filled_at=closed_at,
                close_fill=close_fill if fill_matches_slice else None,
                okx_inst_id=close_okx_inst_id,
            )
            if existing_close_order is not None:
                _apply_okx_close_fill_order_payload(existing_close_order, close_order_payload)
            else:
                await trade_repo.create_order(close_order_payload)
            reconciled.append(
                {
                    "kind": "quantity_reduction_closed_slice",
                    "source": "okx_authoritative_current_position",
                    "model_name": pos.model_name,
                    "symbol": closed_symbol,
                    "side": pos.side,
                    "exit_price": close_price,
                    "quantity": closed_qty,
                    "remaining_quantity": exchange_quantity,
                    "realized_pnl": realized_pnl,
                    "gross_pnl": gross_pnl,
                    "fees": entry_fee + close_fee,
                    "exchange_order_id": close_order_id or None,
                    "note": "OKX position quantity decreased; closed slice recorded locally.",
                }
            )
            logger.warning(
                "recorded exchange quantity reduction as closed position slice",
                position_id=getattr(pos, "id", None),
                symbol=closed_symbol,
                side=pos.side,
                closed_qty=closed_qty,
                remaining_qty=after_qty,
            )
        return reconciled

    async def get_local_open_positions_context(
        self,
        *,
        strict: bool = False,
    ) -> list[dict]:
        """Get locally known open positions without waiting for OKX current state."""
        normalize_symbol = self._required_symbol_normalizer()
        parse_float = self._required_float_parser()
        local_positions: list[dict[str, Any]] = []

        try:
            async with get_session_ctx() as session:
                repo = TradeRepository(session)
                db_positions = await repo.get_position_records(
                    execution_mode=mode_manager.mode.value,
                    limit=1000,
                    is_open=True,
                )
                decision_ids = {
                    int(decision_id)
                    for position in db_positions
                    for decision_id in _dict_value(
                        getattr(position, "current_management_contract", None)
                    ).get("original_entry_decision_ids", [])
                    if str(decision_id or "").isdigit() and int(decision_id) > 0
                }
                decisions_by_id: dict[int, Any] = {}
                if decision_ids:
                    decisions_result = await session.execute(
                        select(AIDecision).where(AIDecision.id.in_(decision_ids))
                    )
                    decisions_by_id = {
                        int(decision.id): decision for decision in decisions_result.scalars().all()
                    }
                for p in db_positions:
                    management_contract = _dict_value(
                        getattr(p, "current_management_contract", None)
                    )
                    canary_lifecycle = _dict_value(
                        management_contract.get("paper_canary_lifecycle")
                    )
                    if not canary_lifecycle:
                        for decision_id in management_contract.get(
                            "original_entry_decision_ids", []
                        ):
                            decision = decisions_by_id.get(
                                int(decision_id)
                                if str(decision_id or "").isdigit()
                                else -1
                            )
                            lifecycle = build_paper_canary_position_lifecycle(decision)
                            if lifecycle:
                                canary_lifecycle = lifecycle
                                break
                    local_positions.append(
                        {
                            "model_name": p.model_name,
                            "symbol": p.symbol,
                            "side": p.side,
                            "entry_price": p.entry_price,
                            "current_price": p.current_price or p.entry_price,
                            "quantity": p.quantity,
                            "leverage": p.leverage or 1.0,
                            "unrealized_pnl": p.unrealized_pnl,
                            "stop_loss": p.stop_loss_price,
                            "take_profit": p.take_profit_price,
                            "is_open": p.is_open,
                            "created_at": p.created_at,
                            "okx_inst_id": getattr(p, "okx_inst_id", None),
                            "okx_pos_id": getattr(p, "okx_pos_id", None),
                            "entry_exchange_order_id": getattr(p, "entry_exchange_order_id", None),
                            "entry_fee": getattr(p, "entry_fee", None),
                            "current_management_contract": management_contract,
                            "paper_canary_lifecycle": canary_lifecycle,
                            "execution_mode": getattr(p, "execution_mode", None),
                        }
                    )
        except Exception as e:
            logger.warning("failed to load DB positions for context", error=safe_error_text(e))
            if strict:
                raise
        normalized: list[dict[str, Any]] = []
        for position_payload in local_positions or []:
            normalized.append(
                normalized_open_position_context(
                    position_payload,
                    symbol_normalizer=normalize_symbol,
                    float_parser=parse_float,
                )
            )
        return normalized

    async def get_open_positions_context(self) -> list[dict]:
        """Get open positions from the active OKX account for trading context."""
        normalize_symbol = self._required_symbol_normalizer()
        parse_float = self._required_float_parser()
        active_okx_provider = self._required_active_okx_provider()
        exchange_position_is_open = self._required_exchange_position_open_checker()
        local_positions = await self.get_local_open_positions_context()

        active_okx = active_okx_provider()
        if not active_okx:
            logger.warning("active OKX executor unavailable; position context is fail-closed")
            return []

        try:
            okx_positions = await asyncio.wait_for(
                active_okx.get_positions_strict(),
                timeout=8.0,
            )
        except Exception as exc:
            logger.warning(
                "failed to fetch authoritative OKX position context; returning no positions",
                error=safe_error_text(exc),
            )
            return []

        local_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for local_position in local_positions:
            key = (
                normalize_symbol(local_position.get("symbol")),
                str(local_position.get("side") or "").lower(),
            )
            if not key[0] or key[1] not in {"long", "short"}:
                continue
            local_by_key.setdefault(key, []).append(local_position)

        positions: list[dict[str, Any]] = []
        for exchange_position in okx_positions or []:
            if not exchange_position_is_open(exchange_position):
                continue
            key = _exchange_position_key(exchange_position, symbol_normalizer=normalize_symbol)
            if key is None:
                continue
            payload = dict(exchange_position)
            raw_info = payload.get("info")
            if isinstance(raw_info, dict):
                payload["info"] = dict(raw_info)
            local_position = _merge_local_position_candidates(
                local_by_key.get(key, []),
                exchange_position=payload,
            )
            payload["model_name"] = (
                payload.get("model_name")
                or local_position.get("model_name")
                or ENSEMBLE_TRADER_NAME
            )
            # Preserve local risk metadata only; quantity, price, side, and PnL stay OKX-native.
            for source_key, target_key in (
                ("stop_loss", "stop_loss"),
                ("take_profit", "take_profit"),
                ("created_at", "created_at"),
                ("entry_exchange_order_id", "entry_exchange_order_id"),
                ("entry_legs", "entry_legs"),
                ("entry_fee", "entry_fee"),
                ("current_management_contract", "current_management_contract"),
                ("paper_canary_lifecycle", "paper_canary_lifecycle"),
                ("execution_mode", "execution_mode"),
            ):
                value = local_position.get(source_key)
                if value not in (None, "") and payload.get(target_key) in (None, ""):
                    payload[target_key] = value
            positions.append(payload)

        normalized: list[dict[str, Any]] = []
        for position_payload in positions or []:
            normalized.append(
                normalized_open_position_context(
                    position_payload,
                    symbol_normalizer=normalize_symbol,
                    float_parser=parse_float,
                )
            )
        return normalized
