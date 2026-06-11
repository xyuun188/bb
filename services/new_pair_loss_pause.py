"""Loss-based pause rules for scanning new trading pairs."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import structlog

from config.settings import ENSEMBLE_TRADER_NAME
from core.safe_output import safe_error_text
from db.repositories.trade_repo import TradeRepository
from db.session import get_session_ctx
from services.equity_baseline import apply_daily_equity_baseline

logger = structlog.get_logger(__name__)

RECENT_LOSS_LOOKBACK_COUNT = 20
LOSS_STREAK_PAUSE_MINUTES = 30.0

BalanceSnapshotProvider = Callable[[str], Awaitable[dict[str, Any] | None]]
PositionRecordsProvider = Callable[[str, bool | None, int], Awaitable[list[Any]]]
DailyEquityBaselineProvider = Callable[
    [str, float, list[Any], float, float, float],
    Awaitable[dict[str, Any]],
]


def selected_execution_mode(mode: str) -> str:
    return "live" if mode == "live" else "paper"


@dataclass(slots=True)
class NewPairLossPausePolicy:
    """Decide whether new-pair analysis should pause after losses."""

    balance_snapshot_provider: BalanceSnapshotProvider
    position_records_provider: PositionRecordsProvider | None = None
    daily_equity_baseline_provider: DailyEquityBaselineProvider | None = None
    model_name: str = ENSEMBLE_TRADER_NAME
    recent_loss_lookback_count: int = RECENT_LOSS_LOOKBACK_COUNT
    loss_streak_pause_minutes: float = LOSS_STREAK_PAUSE_MINUTES

    async def cooldown_loss_pause_reason(
        self,
        mode: str,
        max_loss_usdt: float,
        cooldown_loss_pct: float,
    ) -> str | None:
        """Pause new-pair analysis when today's loss cooldown line is reached."""
        if max_loss_usdt <= 0 or cooldown_loss_pct <= 0:
            return None

        trigger_loss = max_loss_usdt * cooldown_loss_pct
        if trigger_loss <= 0:
            return None

        try:
            selected_mode = selected_execution_mode(mode)
            okx_snapshot = await self.balance_snapshot_provider(selected_mode)
            account_equity = float(
                (okx_snapshot or {}).get("allocatable")
                or (okx_snapshot or {}).get("equity")
                or (okx_snapshot or {}).get("total")
                or 0.0
            )
            now_local = datetime.now(timezone(timedelta(hours=8)))
            start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
            start_utc = start_local.astimezone(UTC)
            realized_pnl = 0.0
            today_realized = 0.0
            open_unrealized = 0.0
            rows = await self._position_records(
                selected_mode,
                is_open=None,
                limit=5000,
            )
            for pos in rows:
                if pos.is_open:
                    open_unrealized += float(pos.unrealized_pnl or 0.0)
                    continue
                pnl = float(pos.realized_pnl or 0.0)
                realized_pnl += pnl
                closed_at = getattr(pos, "closed_at", None)
                if closed_at:
                    if closed_at.tzinfo is None:
                        closed_at = closed_at.replace(tzinfo=UTC)
                    if closed_at >= start_utc:
                        today_realized += pnl
            equity_baseline = await self._daily_equity_baseline(
                selected_mode,
                account_equity,
                rows,
                realized_pnl,
                open_unrealized,
                realized_pnl + open_unrealized,
            )
            day_risk_pnl = float(equity_baseline.get("today_equity_pnl") or 0.0)
        except Exception as exc:
            logger.warning(
                "failed to check cooldown loss pause",
                error=safe_error_text(exc),
            )
            return None

        if day_risk_pnl > -trigger_loss:
            return None

        return (
            "New-pair analysis paused by realized daily loss guard. "
            f"Today equity PnL {day_risk_pnl:.2f} USDT, "
            f"realized {today_realized:.2f} USDT, open floating {open_unrealized:.2f} USDT. "
            f"Trigger is {cooldown_loss_pct * 100:.0f}% of max daily loss "
            f"{max_loss_usdt:.2f} USDT = {trigger_loss:.2f} USDT. "
            "Existing positions will continue to be reviewed."
        )

    async def recent_loss_streak_pause_reason(
        self,
        mode: str,
        max_loss_usdt: float,
        cooldown_loss_pct: float,
    ) -> str | None:
        """Pause briefly after a fresh losing streak exceeds the cooldown line."""
        if max_loss_usdt <= 0 or cooldown_loss_pct <= 0:
            return None

        trigger_loss = max_loss_usdt * cooldown_loss_pct
        if trigger_loss <= 0:
            return None

        try:
            rows = await self._position_records(
                selected_execution_mode(mode),
                is_open=False,
                limit=self.recent_loss_lookback_count,
            )
        except Exception as exc:
            logger.warning("failed to check recent loss streak", error=safe_error_text(exc))
            return None

        recent = [p for p in rows if p.closed_at is not None]
        if not recent:
            return None

        latest_closed = recent[0].closed_at
        if latest_closed and latest_closed.tzinfo is None:
            latest_closed = latest_closed.replace(tzinfo=UTC)
        minutes_since_latest = (
            (datetime.now(UTC) - latest_closed).total_seconds() / 60.0 if latest_closed else 9999.0
        )
        if minutes_since_latest >= self.loss_streak_pause_minutes:
            return None

        streak = 0
        streak_loss = 0.0
        for pos in recent:
            realized_pnl = float(pos.realized_pnl or 0.0)
            if realized_pnl < 0:
                streak += 1
                streak_loss += abs(realized_pnl)
            else:
                break

        if streak <= 0 or streak_loss < trigger_loss:
            return None

        remaining = max(self.loss_streak_pause_minutes - minutes_since_latest, 0.0)
        return (
            "New-pair analysis paused by consecutive realized losses. "
            f"Pause remains about {remaining:.0f} minutes. "
            f"Recent losing streak: {streak} trades, total loss {streak_loss:.2f} USDT. "
            f"Trigger is {cooldown_loss_pct * 100:.0f}% of max daily loss "
            f"{max_loss_usdt:.2f} USDT = {trigger_loss:.2f} USDT. "
            "Existing positions will continue to be monitored."
        )

    async def _position_records(
        self,
        mode: str,
        *,
        is_open: bool | None,
        limit: int,
    ) -> list[Any]:
        if self.position_records_provider is not None:
            return await self.position_records_provider(mode, is_open, limit)
        async with get_session_ctx() as session:
            rows = await TradeRepository(session).get_position_records(
                execution_mode=mode,
                model_name=self.model_name,
                is_open=is_open,
                limit=limit,
            )
        return list(rows)

    async def _daily_equity_baseline(
        self,
        mode: str,
        allocated: float,
        positions: list[Any],
        realized_pnl: float,
        unrealized_pnl: float,
        total_pnl: float,
    ) -> dict[str, Any]:
        if self.daily_equity_baseline_provider is not None:
            return await self.daily_equity_baseline_provider(
                mode,
                allocated,
                positions,
                realized_pnl,
                unrealized_pnl,
                total_pnl,
            )
        async with get_session_ctx() as session:
            return await apply_daily_equity_baseline(
                session,
                mode=mode,
                model_name=self.model_name,
                allocated=allocated,
                positions=positions,
                realized_pnl=realized_pnl,
                unrealized_pnl=unrealized_pnl,
                total_pnl=total_pnl,
            )
