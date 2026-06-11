"""Persistence boundary for executed order logs."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any

import structlog

from ai_brain.base_model import DecisionOutput
from core.safe_output import safe_error_text
from db.repositories.trade_repo import TradeRepository
from db.session import get_session_ctx

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
        try:
            async with self._session_context_factory() as session:
                repo = self._trade_repo_factory(session)
                await repo.create_order(
                    {
                        "model_name": model_name,
                        "execution_mode": self._execution_mode_provider(model_name),
                        "symbol": result.symbol,
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
