"""Persistence boundary for executed order logs."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any

import structlog

from ai_brain.base_model import DecisionOutput
from core.safe_output import safe_error_text
from core.symbols import (
    normalize_trading_symbol,
    okx_inst_id_from_payload,
    symbol_from_okx_inst_id,
    symbol_from_okx_payload,
)
from db.repositories.trade_repo import TradeRepository
from db.session import get_session_ctx
from executor.base_executor import OrderStatus
from services.execution_result_classifier import is_confirmed_native_full_close_result

logger = structlog.get_logger(__name__)

SessionContextFactory = Callable[[], AbstractAsyncContextManager[Any]]
TradeRepoFactory = Callable[[Any], TradeRepository]
ExecutionModeProvider = Callable[[str], str]


class TradeOrderLogService:
    """Persist order rows without leaking repository details into orchestration."""

    def __init__(
        self,
        *,
        execution_mode_provider: ExecutionModeProvider,
        session_context_factory: SessionContextFactory = get_session_ctx,
        trade_repo_factory: TradeRepoFactory = TradeRepository,
    ) -> None:
        self._execution_mode_provider = execution_mode_provider
        self._session_context_factory = session_context_factory
        self._trade_repo_factory = trade_repo_factory

    async def log_trade(
        self,
        result: Any,
        model_name: str,
        decision: DecisionOutput,
        decision_id: int | None = None,
    ) -> None:
        if self._should_skip_order_log(result):
            return
        try:
            async with self._session_context_factory() as session:
                repo = self._trade_repo_factory(session)
                symbol = self._result_symbol(result, decision)
                await repo.create_order(
                    {
                        "model_name": model_name,
                        "execution_mode": self._execution_mode_provider(model_name),
                        "symbol": symbol,
                        "side": result.side,
                        "order_type": result.order_type,
                        "quantity": result.quantity,
                        "price": result.price,
                        "status": result.status.value,
                        "fee": result.fee,
                        "decision_id": decision_id,
                        "exchange_order_id": result.exchange_order_id,
                        "filled_at": result.timestamp,
                    }
                )
        except Exception as exc:
            logger.error("failed to log trade", error=safe_error_text(exc))

    @staticmethod
    def _should_skip_order_log(result: Any) -> bool:
        raw = getattr(result, "raw_response", None)
        raw = raw if isinstance(raw, dict) else {}
        if raw.get("do_not_persist_order"):
            return True

        status = getattr(result, "status", None)
        status_value = getattr(status, "value", status)
        status_text = str(status_value or "").lower()
        quantity = TradeOrderLogService._safe_float(getattr(result, "quantity", 0.0), 0.0)
        price = TradeOrderLogService._safe_float(getattr(result, "price", 0.0), 0.0)
        active_or_filled = {
            OrderStatus.PENDING.value,
            OrderStatus.OPEN.value,
            OrderStatus.PARTIAL.value,
            OrderStatus.FILLED.value,
        }
        tracking_only = bool(raw.get("entry_tracking") or raw.get("exit_tracking"))
        if tracking_only and quantity <= 0:
            return True
        exchange_confirmed_statuses = {
            OrderStatus.PARTIAL.value,
            OrderStatus.FILLED.value,
        }
        if status_text in exchange_confirmed_statuses:
            exchange_order_id = str(getattr(result, "exchange_order_id", "") or "").strip()
            if not exchange_order_id or exchange_order_id in {"hold", "rejected", "no_position"}:
                return True
        if quantity <= 0 and status_text in active_or_filled:
            return True
        return price <= 0 and status_text in active_or_filled

    @staticmethod
    def _result_symbol(result: Any, decision: DecisionOutput) -> str:
        decision_symbol = normalize_trading_symbol(decision.symbol)
        status = getattr(result, "status", None)
        status_value = getattr(status, "value", status)
        status_text = str(status_value or "").lower()
        exchange_order_id = str(getattr(result, "exchange_order_id", "") or "").strip()
        exchange_confirmed = status_text in {
            OrderStatus.PARTIAL.value,
            OrderStatus.FILLED.value,
        } and (
            is_confirmed_native_full_close_result(result)
            or (
                bool(exchange_order_id)
                and exchange_order_id not in {"hold", "rejected", "no_position"}
            )
        )
        if not exchange_confirmed:
            return decision_symbol or normalize_trading_symbol(getattr(result, "symbol", None))

        raw = getattr(result, "raw_response", None)
        raw = raw if isinstance(raw, dict) else {}
        okx_inst_id = okx_inst_id_from_payload(raw, include_fallback=False)
        if okx_inst_id:
            return symbol_from_okx_inst_id(okx_inst_id)
        explicit = normalize_trading_symbol(raw.get("canonical_exchange_symbol"))
        if explicit:
            return explicit
        exchange_symbol = symbol_from_okx_payload(raw, fallback=getattr(result, "symbol", None))
        if exchange_symbol:
            return exchange_symbol
        return normalize_trading_symbol(getattr(result, "symbol", None) or decision.symbol)

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
