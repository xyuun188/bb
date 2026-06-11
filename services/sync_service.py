"""OKX and local-position synchronization boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import select

from ai_brain.base_model import Action, DecisionOutput
from config.settings import ENSEMBLE_TRADER_NAME
from core.safe_output import safe_error_text
from core.trading_mode import mode_manager
from db.repositories.account_repo import AccountRepository
from db.repositories.trade_repo import TradeRepository
from db.session import get_session_ctx
from executor.base_executor import OrderStatus
from models.trade import Order, Position

logger = structlog.get_logger(__name__)

UNCONFIRMED_EXCHANGE_CLOSE_GRACE_SECONDS = 180.0
OPEN_ORDER_SNAPSHOT_UNKNOWN_KIND = "unknown"


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

    def _required_exchange_reconcile_lock(self) -> AbstractAsyncContextManager[Any]:
        if self.exchange_reconcile_lock is None:
            raise RuntimeError("OkxSyncService requires exchange_reconcile_lock dependency")
        return self.exchange_reconcile_lock

    def _record_round_error(self, reason: str) -> None:
        if self.round_error_recorder is None:
            raise RuntimeError("OkxSyncService requires round_error_recorder dependency")
        self.round_error_recorder(reason)

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

    def _required_paper_positions_provider(
        self,
    ) -> Callable[[], Awaitable[list[dict[str, Any]]]]:
        if self.paper_positions_provider is None:
            raise RuntimeError("OkxSyncService requires paper_positions_provider dependency")
        return self.paper_positions_provider

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

    async def reconcile_positions(self, reason: str, timeout_seconds: float = 25.0) -> list[dict]:
        try:

            async def _locked_reconcile() -> list[dict]:
                async with self._required_exchange_reconcile_lock():
                    return await self.reconcile_exchange_positions()

            return await asyncio.wait_for(_locked_reconcile(), timeout=timeout_seconds)
        except TimeoutError:
            timeout_reason = (
                f"exchange position reconciliation timed out during {reason}; "
                "continuing with local position state"
            )
            self._record_round_error(timeout_reason)
            logger.warning(timeout_reason)
            return []

    async def refresh_position_prices(self, feature_vectors: dict[str, Any]) -> Any:
        """Update persisted open-position prices and unrealized PnL."""
        record_position_profit_peak = self._required_position_profit_peak_recorder()
        position_age_minutes = self._required_position_age_minutes_provider()
        prune_position_profit_peaks = self._required_position_profit_peak_pruner()
        try:
            async with get_session_ctx() as session:
                trade_repo = TradeRepository(session)
                account_repo = AccountRepository(session)
                positions = await trade_repo.get_open_positions()
                open_context: list[dict[str, Any]] = []
                pnl_by_model: dict[str, float] = {}

                for pos in positions:
                    fv = feature_vectors.get(pos.symbol)
                    fv_price = getattr(fv, "current_price", None) if fv is not None else None
                    current_price = fv_price if fv_price else pos.current_price or pos.entry_price
                    if not current_price or current_price <= 0:
                        continue

                    if pos.side == "short":
                        unrealized_pnl = (pos.entry_price - current_price) * pos.quantity
                    else:
                        unrealized_pnl = (current_price - pos.entry_price) * pos.quantity

                    await trade_repo.update_position_price(
                        pos.id,
                        float(current_price),
                        float(unrealized_pnl),
                    )
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
                    )
                    pnl_by_model[pos.model_name] = pnl_by_model.get(pos.model_name, 0.0) + float(
                        unrealized_pnl
                    )

                for model_name, unrealized_pnl in pnl_by_model.items():
                    await account_repo.update_unrealized_pnl(model_name, round(unrealized_pnl, 8))
                prune_position_profit_peaks(open_context)
        except Exception as e:
            logger.warning("failed to refresh DB position prices", error=safe_error_text(e))

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
            if normalize_symbol(pos.get("symbol")) != target_symbol:
                continue
            if str(pos.get("side") or "").lower() != target_side:
                continue
            info = pos.get("info") if isinstance(pos.get("info"), dict) else {}
            quantity = parse_float(
                pos.get("contracts")
                or pos.get("size")
                or pos.get("positionAmt")
                or info.get("pos")
                or info.get("qty"),
                0.0,
            )
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
        try:
            executor = await self._required_okx_executor_provider()(pos.execution_mode or "paper")
            orders = await asyncio.wait_for(
                executor.get_open_orders_strict(pos.symbol),
                timeout=8.0,
            )
        except TimeoutError:
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

    async def reconcile_exchange_positions(self) -> list[dict]:
        """Reconcile local paper positions with actual OKX demo positions.

        OKX attached TP/SL orders can close positions without going through the
        AI decision loop. When that happens, close the local DB position using
        the exchange fill so the dashboard and account state remain truthful.
        """
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
        fresh_feature_vector_for_price_recheck = self._required_fresh_feature_vector_provider()
        read_market_value = self._required_market_value_reader()
        entry_fee_for_position = self._required_entry_fee_provider()
        log_exchange_sync_close_decision = self._required_exchange_sync_close_decision_logger()
        record_trade_reflection = self._required_trade_reflection_recorder()
        calculate_position_margin = self._required_position_margin_calculator()
        remove_memory_position = self._required_memory_position_remover()

        try:
            exchange_positions = await asyncio.wait_for(
                paper_okx.get_positions_strict(),
                timeout=10.0,
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

        protection_by_key = await fetch_exchange_protection_map(
            paper_okx,
            exchange_positions,
        )

        exchange_position_keys = {
            (
                normalize_symbol(p.get("symbol")),
                str(p.get("side") or "").lower(),
            )
            for p in exchange_positions or []
            if exchange_position_is_open(p)
        }
        exchange_position_keys.discard(("", ""))
        reconciled: list[dict] = []

        try:
            async with get_session_ctx() as session:
                trade_repo = TradeRepository(session)
                account_repo = AccountRepository(session)
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
                    symbol = normalize_symbol(exchange_pos.get("symbol"))
                    side = str(exchange_pos.get("side") or "").lower()
                    if not symbol or side not in {"long", "short"}:
                        continue
                    key = (symbol, side)

                    info = exchange_pos.get("info") or {}
                    contracts = parse_float(exchange_pos.get("contracts"), 0.0)
                    contract_size = parse_float(exchange_pos.get("contractSize"), 1.0) or 1.0
                    quantity = abs(contracts * contract_size)
                    entry_price = parse_float(
                        exchange_pos.get("entryPrice") or info.get("avgPx"),
                        0.0,
                    )
                    if quantity <= 0 or entry_price <= 0:
                        continue

                    current_price = parse_float(
                        exchange_pos.get("markPrice")
                        or exchange_pos.get("lastPrice")
                        or entry_price,
                        entry_price,
                    )
                    leverage = (
                        parse_float(exchange_pos.get("leverage") or info.get("lever"), 1.0) or 1.0
                    )
                    exchange_unrealized = parse_float(exchange_pos.get("unrealizedPnl"), 0.0)
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
                        local_open_keys.add(key)
                        if changed:
                            reconciled.append(
                                {
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
                            Position.symbol == symbol,
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
                        closed_position.updated_at = datetime.now(UTC)
                        local_open_keys.add(key)
                        reconciled.append(
                            {
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
                            Order.symbol == symbol,
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

                    await trade_repo.open_position(
                        {
                            "model_name": order.model_name or ENSEMBLE_TRADER_NAME,
                            "execution_mode": "paper",
                            "symbol": symbol,
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
                    )
                    local_open_keys.add(key)
                    reconciled.append(
                        {
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

                    close_fill = await find_exchange_close_fill(pos)
                    if not close_fill.get("order_id"):
                        active_order = await self.active_exchange_order_for_local_position(pos)
                        if active_order:
                            if active_order.get("kind") == OPEN_ORDER_SNAPSHOT_UNKNOWN_KIND:
                                logger.warning(
                                    "skip synthetic local close because OKX open order snapshot is unavailable",
                                    position_id=pos.id,
                                    symbol=pos.symbol,
                                    side=pos.side,
                                    error=active_order.get("error"),
                                )
                                reconciled.append(
                                    {
                                        "model_name": pos.model_name,
                                        "symbol": pos.symbol,
                                        "side": pos.side,
                                        "exchange_order_id": None,
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
                                    "model_name": pos.model_name,
                                    "symbol": pos.symbol,
                                    "side": pos.side,
                                    "exchange_order_id": active_order.get("order_id"),
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

                        exit_price = pos.current_price or pos.entry_price
                        fresh = await fresh_feature_vector_for_price_recheck(pos.symbol)
                        if fresh is not None:
                            fresh_price = parse_float(
                                read_market_value(fresh, "current_price")
                                or read_market_value(fresh, "close")
                                or read_market_value(fresh, "bid")
                                or read_market_value(fresh, "ask"),
                                0.0,
                            )
                            if fresh_price > 0:
                                exit_price = fresh_price
                        if pos.side == "short":
                            gross_pnl = (
                                float(pos.entry_price or 0.0) - float(exit_price or 0.0)
                            ) * float(pos.quantity or 0.0)
                            close_side = "buy"
                        else:
                            gross_pnl = (
                                float(exit_price or 0.0) - float(pos.entry_price or 0.0)
                            ) * float(pos.quantity or 0.0)
                            close_side = "sell"
                        entry_fee = await entry_fee_for_position(session, pos, pos.quantity)
                        realized_pnl = gross_pnl - entry_fee
                        decision_id = await log_exchange_sync_close_decision(
                            session=session,
                            pos=pos,
                            exit_price=exit_price,
                            realized_pnl=realized_pnl,
                            closed_at=now,
                            reason=(
                                "OKX 已没有这笔持仓，但没有查到对应平仓成交回报；"
                                "系统按交易所仓位状态同步为平仓，并用本地开仓价与同步平仓价估算盈亏。"
                            ),
                            close_fill={
                                "estimated": True,
                                "price": exit_price,
                                "gross_pnl": gross_pnl,
                                "entry_fee": entry_fee,
                                "fee": 0.0,
                                "pnl": realized_pnl,
                                "fresh_price_used": bool(fresh is not None),
                                "note": "close fill not found; realized pnl estimated from local entry and freshest available sync price",
                            },
                        )
                        pos.is_open = False
                        pos.current_price = exit_price
                        pos.unrealized_pnl = 0.0
                        pos.realized_pnl = realized_pnl
                        pos.closed_at = now
                        await trade_repo.create_order(
                            {
                                "model_name": pos.model_name,
                                "execution_mode": pos.execution_mode,
                                "symbol": pos.symbol,
                                "side": close_side,
                                "order_type": "market",
                                "quantity": pos.quantity,
                                "price": exit_price,
                                "status": OrderStatus.FILLED.value,
                                "fee": 0.0,
                                "decision_id": decision_id,
                                "exchange_order_id": None,
                                "filled_at": now,
                            }
                        )
                        reconciled.append(
                            {
                                "model_name": pos.model_name,
                                "symbol": pos.symbol,
                                "side": pos.side,
                                "exit_price": pos.current_price,
                                "realized_pnl": realized_pnl,
                                "exchange_order_id": None,
                                "note": "OKX 已无对应持仓，未查到平仓成交回报；本地已按同步价格估算盈亏并关闭仓位。",
                            }
                        )
                        logger.warning(
                            "closed unsynced local position; no OKX open position or close fill found",
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
                    if pos.side == "short":
                        gross_pnl = (pos.entry_price - exit_price) * pos.quantity
                        close_side = "buy"
                    else:
                        gross_pnl = (exit_price - pos.entry_price) * pos.quantity
                        close_side = "sell"
                    entry_fee = await entry_fee_for_position(session, pos, pos.quantity)
                    realized_pnl = gross_pnl - entry_fee - close_fee

                    pos.is_open = False
                    pos.current_price = exit_price
                    pos.unrealized_pnl = 0.0
                    pos.realized_pnl = realized_pnl
                    pos.closed_at = close_fill.get("timestamp") or datetime.now(UTC)
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
                        close_fill=close_fill,
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
                    close_order_id = str(close_fill.get("order_id") or "")
                    if close_order_id:
                        existing_close_order_result = await session.execute(
                            select(Order.id)
                            .where(
                                Order.execution_mode == pos.execution_mode,
                                Order.exchange_order_id == close_order_id,
                            )
                            .limit(1)
                        )
                        existing_close_order = existing_close_order_result.scalar_one_or_none()

                    if not existing_close_order:
                        await trade_repo.create_order(
                            {
                                "model_name": pos.model_name,
                                "execution_mode": pos.execution_mode,
                                "symbol": pos.symbol,
                                "side": close_side,
                                "order_type": "market",
                                "quantity": pos.quantity,
                                "price": exit_price,
                                "status": OrderStatus.FILLED.value,
                                "fee": close_fee,
                                "decision_id": decision_id,
                                "exchange_order_id": close_fill.get("order_id"),
                                "filled_at": pos.closed_at,
                            }
                        )

                    released_margin = calculate_position_margin(
                        pos.quantity * pos.entry_price,
                        pos.leverage,
                    )
                    await account_repo.update_balance(
                        pos.model_name,
                        released_margin + realized_pnl,
                        realized_pnl,
                    )
                    await account_repo.record_trade_result(pos.model_name, realized_pnl > 0)
                    remove_memory_position(pos.model_name, pos.symbol, pos.side)

                    reconciled.append(
                        {
                            "model_name": pos.model_name,
                            "symbol": pos.symbol,
                            "side": pos.side,
                            "exit_price": exit_price,
                            "realized_pnl": realized_pnl,
                            "gross_pnl": gross_pnl,
                            "fees": entry_fee + close_fee,
                            "exchange_order_id": close_fill.get("order_id"),
                        }
                    )

        except Exception as e:
            logger.warning("exchange position reconciliation failed", error=safe_error_text(e))
            return reconciled

        if reconciled:
            logger.info(
                "reconciled exchange-closed positions", count=len(reconciled), positions=reconciled
            )
        return reconciled

    async def get_open_positions_context(self) -> list[dict]:
        """Get open positions for LLM context (paper + live)."""
        normalize_symbol = self._required_symbol_normalizer()
        parse_float = self._required_float_parser()
        paper_positions_provider = self._required_paper_positions_provider()
        active_okx_provider = self._required_active_okx_provider()
        exchange_position_is_open = self._required_exchange_position_open_checker()
        positions: list[dict[str, Any]] = []

        try:
            async with get_session_ctx() as session:
                repo = TradeRepository(session)
                db_positions = await repo.get_position_records(
                    execution_mode=mode_manager.mode.value,
                    model_name=ENSEMBLE_TRADER_NAME,
                    limit=1000,
                )
                for p in db_positions:
                    if p.is_open:
                        positions.append(
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
                            }
                        )
        except Exception as e:
            logger.warning("failed to load DB positions for context", error=safe_error_text(e))

        # Keep in-memory paper positions as a fallback for positions opened before
        # persistence was introduced in this process.
        if not positions:
            positions.extend(await paper_positions_provider())

        # OKX positions for the active execution account.
        active_okx = active_okx_provider()
        if active_okx:
            try:
                okx_positions = await asyncio.wait_for(
                    active_okx.get_positions_strict(),
                    timeout=8.0,
                )
                exchange_keys = {
                    (
                        normalize_symbol(p.get("symbol")),
                        str(p.get("side") or "").lower(),
                    )
                    for p in (okx_positions or [])
                    if exchange_position_is_open(p)
                }
                positions = [
                    p
                    for p in positions
                    if (
                        normalize_symbol(p.get("symbol")),
                        str(p.get("side") or "").lower(),
                    )
                    in exchange_keys
                ]
                existing_keys = {
                    (
                        normalize_symbol(p.get("symbol")),
                        str(p.get("side") or "").lower(),
                    )
                    for p in positions
                }
                for p in okx_positions or []:
                    p["model_name"] = p.get("model_name") or ENSEMBLE_TRADER_NAME
                    key = (
                        normalize_symbol(p.get("symbol")),
                        str(p.get("side") or "").lower(),
                    )
                    if key not in existing_keys:
                        positions.append(p)
            except Exception as exc:
                logger.warning(
                    "failed to merge OKX position context; keeping local positions",
                    error=safe_error_text(exc),
                )

        normalized: list[dict[str, Any]] = []
        for position_payload in positions or []:
            raw_info = position_payload.get("info")
            info = raw_info if isinstance(raw_info, dict) else {}
            normalized.append(
                {
                    "model_name": position_payload.get("model_name", ""),
                    "symbol": position_payload.get("symbol", ""),
                    "side": position_payload.get("side", "long"),
                    "entry_price": (
                        position_payload.get("entry_price")
                        or position_payload.get("entryPrice")
                        or position_payload.get("avgPx")
                        or 0
                    ),
                    "current_price": (
                        position_payload.get("current_price")
                        or position_payload.get("markPrice")
                        or position_payload.get("lastPrice")
                        or position_payload.get("entry_price")
                        or position_payload.get("entryPrice")
                        or position_payload.get("avgPx")
                        or 0
                    ),
                    "quantity": (
                        position_payload.get("quantity")
                        or position_payload.get("contracts")
                        or position_payload.get("sz")
                        or 0
                    ),
                    "contracts": position_payload.get("contracts") or position_payload.get("sz"),
                    "contract_size": (
                        position_payload.get("contract_size")
                        or position_payload.get("contractSize")
                        or info.get("ctVal")
                    ),
                    "contractSize": (
                        position_payload.get("contractSize")
                        or position_payload.get("contract_size")
                        or info.get("ctVal")
                    ),
                    "leverage": parse_float(
                        position_payload.get("leverage") or info.get("lever"),
                        1.0,
                    ),
                    "notional": (
                        position_payload.get("notional")
                        or position_payload.get("notional_usd")
                        or position_payload.get("notionalUsd")
                        or info.get("notionalUsd")
                        or info.get("notional")
                        or info.get("posValue")
                        or 0
                    ),
                    "margin": (
                        position_payload.get("margin")
                        or position_payload.get("initial_margin")
                        or position_payload.get("initialMargin")
                        or position_payload.get("margin_used")
                        or info.get("margin")
                        or info.get("imr")
                    ),
                    "initial_margin": (
                        position_payload.get("initial_margin")
                        or position_payload.get("initialMargin")
                        or info.get("imr")
                    ),
                    "initialMargin": (
                        position_payload.get("initialMargin")
                        or position_payload.get("initial_margin")
                        or info.get("imr")
                    ),
                    "unrealized_pnl": position_payload.get(
                        "unrealized_pnl", position_payload.get("unrealizedPnl", 0)
                    ),
                    "stop_loss": position_payload.get("stop_loss"),
                    "take_profit": position_payload.get("take_profit"),
                    "is_open": position_payload.get("is_open", True),
                    "created_at": (
                        position_payload.get("created_at")
                        or position_payload.get("timestamp")
                        or position_payload.get("opened_at")
                        or info.get("cTime")
                        or info.get("uTime")
                    ),
                    "info": info,
                }
            )
        return normalized
