from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import structlog

from config.settings import ENSEMBLE_TRADER_NAME
from core.safe_output import safe_error_text
from db.repositories.trade_repo import TradeRepository
from db.session import get_session_ctx
from services.equity_baseline import apply_daily_equity_baseline
from services.trade_fact_trust import closed_position_trade_fact_trusted

logger = structlog.get_logger(__name__)


BEIJING_TZ = timezone(timedelta(hours=8))


class ExecutionAllocationService:
    """Build the execution-account allocation and PnL state."""

    def __init__(
        self,
        *,
        balance_snapshot_provider: Callable[[str], Awaitable[dict[str, Any] | None]],
        active_executor_provider: Callable[[str], Any | None],
        exchange_position_open_checker: Callable[[dict[str, Any]], bool],
        symbol_normalizer: Callable[[Any], str],
        session_factory: Callable[[], AbstractAsyncContextManager[Any]] = get_session_ctx,
        equity_baseline_provider: Callable[..., Awaitable[dict[str, Any]]] = (
            apply_daily_equity_baseline
        ),
        model_name: str = ENSEMBLE_TRADER_NAME,
        now_provider: Callable[[], datetime] | None = None,
        exchange_positions_timeout_seconds: float = 8.0,
    ) -> None:
        self.balance_snapshot_provider = balance_snapshot_provider
        self.active_executor_provider = active_executor_provider
        self.exchange_position_open_checker = exchange_position_open_checker
        self.symbol_normalizer = symbol_normalizer
        self.session_factory = session_factory
        self.equity_baseline_provider = equity_baseline_provider
        self.model_name = model_name
        self.now_provider = now_provider or (lambda: datetime.now(UTC))
        self.exchange_positions_timeout_seconds = exchange_positions_timeout_seconds

    async def calculate(self, mode: str) -> dict[str, Any]:
        """Calculate execution PnL and exchange-backed budget state."""

        selected_mode = normalize_mode(mode)
        okx_snapshot = await self.balance_snapshot_provider(selected_mode)
        okx_available = snapshot_free_balance(okx_snapshot)
        allocated = snapshot_execution_equity(okx_snapshot, okx_available)
        current_equity = snapshot_account_equity(okx_snapshot)
        exchange_keys = await self._exchange_position_keys(selected_mode)
        start_utc = beijing_start_utc(self.now_provider())
        position_rows: list[Any] = []

        metrics = AllocationMetrics()
        try:
            async with self.session_factory() as session:
                rows = await TradeRepository(session).get_position_records(
                    execution_mode=selected_mode,
                    model_name=self.model_name,
                    limit=5000,
                )
                position_rows = list(rows)
                metrics = self._position_metrics(position_rows, exchange_keys, start_utc)
        except Exception as exc:
            logger.warning(
                "failed to calculate allocation state",
                mode=selected_mode,
                error=safe_error_text(exc),
            )

        realized_pnl = metrics.realized_profit - metrics.realized_loss
        today_realized_pnl = metrics.today_realized_profit - metrics.today_realized_loss
        total_pnl = realized_pnl + metrics.unrealized_pnl
        today_total_pnl: float | None = None
        today_risk_pnl: float | None = None
        equity_baseline: dict[str, Any] = {}
        try:
            async with self.session_factory() as session:
                equity_baseline = await self.equity_baseline_provider(
                    session,
                    mode=selected_mode,
                    model_name=self.model_name,
                    allocated=allocated,
                    positions=position_rows,
                    realized_pnl=realized_pnl,
                    unrealized_pnl=metrics.unrealized_pnl,
                    total_pnl=total_pnl,
                    current_equity=current_equity,
                )
            today_total_pnl = _safe_float(equity_baseline.get("today_equity_pnl"), None)
            today_risk_pnl = today_total_pnl
        except Exception as exc:
            logger.warning(
                "failed to calculate daily equity baseline",
                mode=selected_mode,
                error=safe_error_text(exc),
            )

        return {
            "allocated_balance": allocated,
            "used_margin": metrics.used_margin,
            "realized_profit": metrics.realized_profit,
            "realized_loss": metrics.realized_loss,
            "realized_pnl": realized_pnl,
            "today_realized_profit": metrics.today_realized_profit,
            "today_realized_loss": metrics.today_realized_loss,
            "today_realized_pnl": today_realized_pnl,
            "today_closed_realized_profit": metrics.today_realized_profit,
            "today_closed_realized_loss": metrics.today_realized_loss,
            "today_closed_realized_pnl": today_realized_pnl,
            "today_equity_pnl": today_total_pnl,
            "today_equity_baseline": equity_baseline.get("today_equity_baseline"),
            "today_equity_baseline_total_pnl": equity_baseline.get(
                "today_equity_baseline_total_pnl"
            ),
            "today_equity_baseline_at": equity_baseline.get("today_equity_baseline_at"),
            "today_equity_baseline_source": equity_baseline.get("today_equity_baseline_source"),
            "today_snapshot_date": equity_baseline.get("today_snapshot_date"),
            "today_total_pnl": today_total_pnl,
            "today_risk_pnl": today_risk_pnl,
            "unrealized_pnl": metrics.unrealized_pnl,
            "total_pnl": total_pnl,
            "remaining_allocation": okx_available,
        }

    async def _exchange_position_keys(self, mode: str) -> set[tuple[str, str]] | None:
        executor = self.active_executor_provider(mode)
        if executor is None:
            logger.warning("execution allocation OKX executor unavailable", mode=mode)
            return set()
        try:
            okx_positions = await asyncio.wait_for(
                self._fetch_exchange_positions(executor),
                timeout=self.exchange_positions_timeout_seconds,
            )
        except Exception as exc:
            logger.warning(
                "execution allocation strict OKX position snapshot unavailable",
                mode=mode,
                error=safe_error_text(exc),
            )
            return set()
        return {
            (
                self.symbol_normalizer(position.get("symbol")),
                str(position.get("side") or "").lower(),
            )
            for position in (okx_positions or [])
            if self.exchange_position_open_checker(position)
        }

    def _position_metrics(
        self,
        positions: list[Any],
        exchange_keys: set[tuple[str, str]],
        start_utc: datetime,
    ) -> AllocationMetrics:
        metrics = AllocationMetrics()
        for pos in positions:
            if getattr(pos, "is_open", False):
                if not self._position_exists_on_exchange(pos, exchange_keys):
                    continue
                leverage = max(float(getattr(pos, "leverage", 1.0) or 1.0), 1.0)
                metrics.used_margin += (
                    float(getattr(pos, "quantity", 0.0) or 0.0)
                    * float(getattr(pos, "entry_price", 0.0) or 0.0)
                ) / leverage
                metrics.unrealized_pnl += float(getattr(pos, "unrealized_pnl", 0.0) or 0.0)
                continue
            if not closed_position_trade_fact_trusted(pos):
                continue

            pnl = float(getattr(pos, "realized_pnl", 0.0) or 0.0)
            if pnl >= 0:
                metrics.realized_profit += pnl
            else:
                metrics.realized_loss += abs(pnl)
            closed_at = as_utc(getattr(pos, "closed_at", None))
            if closed_at and closed_at >= start_utc:
                if pnl >= 0:
                    metrics.today_realized_profit += pnl
                else:
                    metrics.today_realized_loss += abs(pnl)
        return metrics

    def _position_exists_on_exchange(
        self,
        pos: Any,
        exchange_keys: set[tuple[str, str]],
    ) -> bool:
        return (
            self.symbol_normalizer(getattr(pos, "symbol", None)),
            str(getattr(pos, "side", "") or "").lower(),
        ) in exchange_keys

    async def _fetch_exchange_positions(self, executor: Any) -> list[dict[str, Any]]:
        fetch_strict = getattr(executor, "get_positions_strict", None)
        if not callable(fetch_strict):
            raise RuntimeError("execution allocation requires get_positions_strict")
        return await fetch_strict()


class AllocationMetrics:
    def __init__(self) -> None:
        self.realized_profit = 0.0
        self.realized_loss = 0.0
        self.today_realized_profit = 0.0
        self.today_realized_loss = 0.0
        self.unrealized_pnl = 0.0
        self.used_margin = 0.0


def normalize_mode(mode: str | None) -> str:
    return "live" if mode == "live" else "paper"


def snapshot_free_balance(snapshot: dict[str, Any] | None) -> float:
    if not isinstance(snapshot, dict):
        return 0.0
    return _safe_float(snapshot.get("free"), 0.0)


def snapshot_execution_equity(
    snapshot: dict[str, Any] | None,
    fallback_free: float = 0.0,
) -> float:
    if not isinstance(snapshot, dict):
        return 0.0
    return _safe_float(
        snapshot.get("allocatable")
        or snapshot.get("equity")
        or snapshot.get("total")
        or fallback_free,
        0.0,
    )


def snapshot_account_equity(snapshot: dict[str, Any] | None) -> float:
    if not isinstance(snapshot, dict):
        return 0.0
    return _safe_float(
        snapshot.get("equity")
        or snapshot.get("total")
        or snapshot.get("allocatable")
        or snapshot.get("cash")
        or snapshot.get("free"),
        0.0,
    )


def beijing_start_utc(now: datetime) -> datetime:
    now_local = now.astimezone(BEIJING_TZ)
    start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    return start_local.astimezone(UTC)


def as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
