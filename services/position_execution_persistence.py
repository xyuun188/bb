"""Persistence boundary for position changes caused by executions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from typing import Any

import structlog

from ai_brain.base_model import Action, DecisionOutput
from core.safe_output import safe_error_text
from core.symbols import normalize_trading_symbol, symbol_from_okx_payload
from db.repositories.trade_repo import TradeRepository
from db.session import get_session_ctx
from services.order_position_reconciliation import reconcile_missing_closed_position_for_exit

logger = structlog.get_logger(__name__)

SessionContextFactory = Callable[[], AbstractAsyncContextManager[Any]]
TradeRepoFactory = Callable[[Any], TradeRepository]
ExecutionChecker = Callable[[Any | None], bool]
ExchangeBackedIdProvider = Callable[[Any, list[Any]], Awaitable[set[int]]]
EntryFeeForPosition = Callable[[Any, Any, float], Awaitable[float]]
ProportionalFee = Callable[[float | None, float, float], float]
TradeReflectionRecorder = Callable[..., Awaitable[None]]
PositionPeakRemover = Callable[[str, str, str], None]

POSITION_CLOSE_DUST_ABS_TOLERANCE = 1e-8
POSITION_CLOSE_DUST_REL_TOLERANCE = 1e-9


def _close_quantity_leaves_dust(
    *,
    position_quantity: float,
    close_quantity: float,
    result_quantity: float,
) -> bool:
    """Return true when a close leaves only floating-point dust."""

    remaining = max(float(position_quantity or 0.0) - float(close_quantity or 0.0), 0.0)
    tolerance = max(
        POSITION_CLOSE_DUST_ABS_TOLERANCE,
        abs(float(position_quantity or 0.0)) * POSITION_CLOSE_DUST_REL_TOLERANCE,
        abs(float(result_quantity or 0.0)) * POSITION_CLOSE_DUST_REL_TOLERANCE,
    )
    return remaining <= tolerance


def _quantity_is_dust(quantity: float, reference_quantity: float) -> bool:
    tolerance = max(
        POSITION_CLOSE_DUST_ABS_TOLERANCE,
        abs(float(reference_quantity or 0.0)) * POSITION_CLOSE_DUST_REL_TOLERANCE,
    )
    return abs(float(quantity or 0.0)) <= tolerance


class PositionExecutionPersistenceService:
    """Persist open/close position records while preserving execution semantics."""

    def __init__(
        self,
        *,
        exchange_confirmed_checker: ExecutionChecker,
        exit_progress_checker: ExecutionChecker,
        exchange_backed_id_provider: ExchangeBackedIdProvider,
        entry_fee_provider: EntryFeeForPosition,
        proportional_fee: ProportionalFee,
        trade_reflection_recorder: TradeReflectionRecorder,
        position_peak_remover: PositionPeakRemover,
        session_context_factory: SessionContextFactory = get_session_ctx,
        trade_repo_factory: TradeRepoFactory = TradeRepository,
    ) -> None:
        self._exchange_confirmed_checker = exchange_confirmed_checker
        self._exit_progress_checker = exit_progress_checker
        self._exchange_backed_id_provider = exchange_backed_id_provider
        self._entry_fee_provider = entry_fee_provider
        self._proportional_fee = proportional_fee
        self._trade_reflection_recorder = trade_reflection_recorder
        self._position_peak_remover = position_peak_remover
        self._session_context_factory = session_context_factory
        self._trade_repo_factory = trade_repo_factory

    async def persist(
        self,
        *,
        model_name: str,
        decision: DecisionOutput,
        result: Any,
        execution_mode: str,
    ) -> None:
        """Persist open and closed position records for paper/live parity."""

        exit_progress = decision.is_exit and self._exit_progress_checker(result)
        if not self._exchange_confirmed_checker(result) and not exit_progress:
            return
        if result.quantity <= 0 or result.price <= 0:
            return

        try:
            async with self._session_context_factory() as session:
                repo = self._trade_repo_factory(session)
                if decision.action in (Action.LONG, Action.SHORT):
                    await self._persist_entry(repo, model_name, decision, result, execution_mode)
                    return

                if decision.action in (Action.CLOSE_LONG, Action.CLOSE_SHORT):
                    await self._persist_exit(
                        session,
                        repo,
                        model_name,
                        decision,
                        result,
                        execution_mode,
                    )
        except Exception as exc:
            logger.error("failed to persist position", error=safe_error_text(exc))

    @staticmethod
    def _result_symbol(result: Any, decision: DecisionOutput) -> str:
        raw = getattr(result, "raw_response", None)
        raw = raw if isinstance(raw, dict) else {}
        explicit = normalize_trading_symbol(raw.get("canonical_exchange_symbol"))
        if explicit:
            return explicit
        exchange_symbol = symbol_from_okx_payload(raw, fallback=getattr(result, "symbol", None))
        if exchange_symbol:
            return exchange_symbol
        return normalize_trading_symbol(getattr(result, "symbol", None) or decision.symbol)

    @staticmethod
    async def _persist_entry(
        repo: TradeRepository,
        model_name: str,
        decision: DecisionOutput,
        result: Any,
        execution_mode: str,
    ) -> None:
        side = "long" if decision.action == Action.LONG else "short"
        symbol = PositionExecutionPersistenceService._result_symbol(result, decision)
        stop_loss = (
            result.price * (1 - decision.stop_loss_pct)
            if side == "long"
            else result.price * (1 + decision.stop_loss_pct)
        )
        take_profit = (
            result.price * (1 + decision.take_profit_pct)
            if side == "long"
            else result.price * (1 - decision.take_profit_pct)
        )
        await repo.open_position(
            {
                "model_name": model_name,
                "execution_mode": execution_mode,
                "symbol": symbol,
                "side": side,
                "quantity": result.quantity,
                "entry_price": result.price,
                "current_price": result.price,
                "leverage": decision.suggested_leverage,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0,
                "stop_loss_price": stop_loss,
                "take_profit_price": take_profit,
            }
        )

    async def _persist_exit(
        self,
        session: Any,
        repo: TradeRepository,
        model_name: str,
        decision: DecisionOutput,
        result: Any,
        execution_mode: str,
    ) -> None:
        side = "long" if decision.action == Action.CLOSE_LONG else "short"
        symbol = self._result_symbol(result, decision)
        positions = await repo.get_matching_open_positions(
            model_name=model_name,
            symbol=symbol,
            side=side,
            execution_mode=execution_mode,
        )
        if not positions:
            recovered = await reconcile_missing_closed_position_for_exit(
                session,
                model_name=model_name,
                execution_mode=execution_mode,
                decision=decision,
                result=result,
            )
            if recovered is not None:
                self._position_peak_remover(model_name, symbol, side)
                if result.pnl == 0.0 and recovered.plan.realized_pnl != 0.0:
                    result.pnl = recovered.plan.realized_pnl
                await self._record_reflection(
                    session,
                    recovered.position,
                    result,
                    recovered.plan.entry_fee_allocated,
                    recovered.plan.close_fee_allocated,
                    recovered.plan.gross_pnl,
                    decision,
                )
                await session.flush()
                logger.warning(
                    "recovered missing closed position from filled order pair",
                    symbol=recovered.plan.symbol,
                    side=recovered.plan.side,
                    quantity=recovered.plan.quantity,
                    entry_order_id=recovered.plan.entry_order_id,
                    close_order_id=recovered.plan.close_order_id,
                )
            return
        exchange_backed_ids = await self._exchange_backed_id_provider(session, positions)
        positions = sorted(
            positions,
            key=lambda position: (
                position.id not in exchange_backed_ids,
                position.created_at or datetime.min,
            ),
        )
        remaining_qty = result.quantity
        total_pnl = 0.0
        for position in positions:
            if remaining_qty <= 0 or _quantity_is_dust(remaining_qty, result.quantity):
                break
            close_qty = min(position.quantity, remaining_qty)
            closes_position = not (
                close_qty < position.quantity
                and not _close_quantity_leaves_dust(
                    position_quantity=position.quantity,
                    close_quantity=close_qty,
                    result_quantity=result.quantity,
                )
            )
            if closes_position:
                close_qty = position.quantity
            close_fee = self._proportional_fee(result.fee, close_qty, result.quantity)
            entry_fee = await self._entry_fee_provider(session, position, close_qty)
            if side == "long":
                gross_pnl = (result.price - position.entry_price) * close_qty
            else:
                gross_pnl = (position.entry_price - result.price) * close_qty
            pnl = gross_pnl - entry_fee - close_fee
            total_pnl += pnl
            if not closes_position:
                position.quantity -= close_qty
                remaining_qty = 0
                closed_pos = await repo.open_position(
                    {
                        "model_name": model_name,
                        "execution_mode": execution_mode,
                        "symbol": symbol,
                        "side": side,
                        "quantity": close_qty,
                        "entry_price": position.entry_price,
                        "current_price": result.price,
                        "leverage": position.leverage,
                        "unrealized_pnl": 0.0,
                        "realized_pnl": pnl,
                        "stop_loss_price": position.stop_loss_price,
                        "take_profit_price": position.take_profit_price,
                        "is_open": False,
                        "closed_at": result.timestamp,
                        "created_at": position.created_at,
                    }
                )
                await self._record_reflection(
                    session,
                    closed_pos,
                    result,
                    entry_fee,
                    close_fee,
                    gross_pnl,
                    decision,
                )
            else:
                self._position_peak_remover(model_name, symbol, side)
                position.is_open = False
                position.current_price = result.price
                position.unrealized_pnl = 0.0
                position.realized_pnl = pnl
                position.closed_at = result.timestamp
                remaining_qty -= close_qty
                await self._record_reflection(
                    session,
                    position,
                    result,
                    entry_fee,
                    close_fee,
                    gross_pnl,
                    decision,
                )
        if result.pnl == 0.0 and total_pnl != 0.0:
            result.pnl = total_pnl
        await session.flush()

    async def _record_reflection(
        self,
        session: Any,
        position: Any,
        result: Any,
        entry_fee: float,
        close_fee: float,
        gross_pnl: float,
        decision: DecisionOutput,
    ) -> None:
        await self._trade_reflection_recorder(
            session,
            position,
            exit_price=result.price,
            entry_fee=entry_fee,
            close_fee=close_fee,
            gross_pnl=gross_pnl,
            source="system_execution",
            decision=decision,
        )
