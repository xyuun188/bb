from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from typing import Any

import structlog

from core.safe_output import safe_error_text
from db.repositories.account_repo import AccountRepository
from db.session import get_session_ctx
from executor.base_executor import ExecutionResult

logger = structlog.get_logger(__name__)


class AccountAccountingService:
    """Own account balance parsing and persistence boundaries."""

    def __init__(
        self,
        *,
        balance_snapshot_provider: Callable[[str], Awaitable[dict[str, Any] | None]],
        allocation_state_provider: Callable[[str], Awaitable[dict[str, Any]]],
        model_execution_mode_provider: Callable[[str], str],
        session_factory: Callable[[], AbstractAsyncContextManager[Any]] = get_session_ctx,
    ) -> None:
        self.balance_snapshot_provider = balance_snapshot_provider
        self.allocation_state_provider = allocation_state_provider
        self.model_execution_mode_provider = model_execution_mode_provider
        self.session_factory = session_factory

    async def account_balance(self, model_name: str) -> float:
        """Return account equity used as execution-risk denominator."""

        model_mode = self.model_execution_mode_provider(model_name)
        return await self.account_equity_for_risk(model_mode)

    async def account_equity_for_risk(self, mode: str) -> float:
        """Return exchange equity used by risk controls.

        OKX-backed paper/live accounts must not fall back to local virtual
        allocations. If OKX is unavailable, upstream guards pause new entries.
        """

        selected_mode = normalize_mode(mode)
        snapshot = await self.balance_snapshot_provider(selected_mode)
        balance = balance_from_snapshot(snapshot)
        if balance > 0:
            return balance
        return 0.0

    async def okx_available_balance_for_mode(self, mode: str) -> float | None:
        """Return exchange tradeable balance or None when no snapshot exists."""

        snapshot = await self.balance_snapshot_provider(normalize_mode(mode))
        if not snapshot:
            return None
        return tradeable_balance_from_snapshot(snapshot)

    async def allocated_order_balance(
        self,
        mode: str,
        _decision: Any | None = None,
    ) -> float:
        """Return balance visible to order sizing."""

        okx_available = await self.okx_available_balance_for_mode(normalize_mode(mode))
        if okx_available is None:
            return 0.0
        return max(float(okx_available or 0.0), 0.0)

    async def persist_balance_delta(
        self,
        model_name: str,
        balance_delta: float,
        realized_pnl_delta: float = 0.0,
    ) -> None:
        """Do not persist synthetic balance deltas for OKX-backed accounts."""

        return

    async def persist_account_update(
        self,
        model_name: str,
        _execution_model_name: str,
        result: ExecutionResult,
    ) -> None:
        """Record win/loss counters without mutating synthetic account balances."""

        try:
            async with self.session_factory() as session:
                repo = AccountRepository(session)
                await repo.record_trade_result(model_name, result.pnl > 0)
        except Exception as exc:
            logger.error("failed to persist account update", error=safe_error_text(exc))

    async def record_unrealized_pnl(self, model_name: str, unrealized_pnl: float) -> None:
        """Do not persist exchange-derived unrealized PnL into virtual accounts."""

        return


def normalize_mode(mode: str | None) -> str:
    return "live" if mode == "live" else "paper"


def allocatable_balance_from_snapshot(snapshot: dict[str, Any] | None) -> float:
    if not isinstance(snapshot, dict):
        return 0.0
    return max(
        _safe_float(snapshot.get("allocatable"), 0.0),
        _safe_float(snapshot.get("equity"), 0.0),
        _safe_float(snapshot.get("cash"), 0.0),
        _safe_float(snapshot.get("total"), 0.0),
        _safe_float(snapshot.get("free"), 0.0),
    )


def tradeable_balance_from_snapshot(snapshot: dict[str, Any] | None) -> float:
    if not isinstance(snapshot, dict):
        return 0.0
    free = _safe_float(snapshot.get("free"), 0.0)
    if free > 0:
        return free
    # OKX demo/swap accounts can report free=0 while equity/cash is usable.
    return allocatable_balance_from_snapshot(snapshot)


def balance_from_snapshot(snapshot: dict[str, Any] | None) -> float:
    if not isinstance(snapshot, dict):
        return 0.0
    for key in ("equity", "total", "allocatable", "cash", "free"):
        value = _safe_float(snapshot.get(key), 0.0)
        if value > 0:
            return value
    return 0.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
