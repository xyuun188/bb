"""Today's realized PnL split by long/short side.

The entry strategy mode uses this to bias posture from actual realized results
without letting the main trading orchestrator own database aggregation logic.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import structlog

from config.settings import ENSEMBLE_TRADER_NAME
from core.safe_output import safe_error_text
from db.repositories.trade_repo import TradeRepository
from db.session import get_session_ctx
from services.trade_fact_trust import closed_position_trade_fact_trusted

SessionFactory = Callable[[], Any]
TradeRepositoryFactory = Callable[[Any], TradeRepository]

logger = structlog.get_logger(__name__)


def _empty_side_bucket() -> dict[str, float]:
    return {
        "count": 0,
        "wins": 0,
        "losses": 0,
        "pnl": 0.0,
        "profit": 0.0,
        "loss": 0.0,
    }


def _normalize_local_now(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone(timedelta(hours=8)))
    return value.astimezone(timezone(timedelta(hours=8)))


def _aware_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class DailySidePerformanceService:
    """Calculate today's realized PnL buckets for long and short positions."""

    def __init__(
        self,
        *,
        session_factory: SessionFactory = get_session_ctx,
        trade_repository_factory: TradeRepositoryFactory = TradeRepository,
        model_name: str = ENSEMBLE_TRADER_NAME,
        limit: int = 5000,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._trade_repository_factory = trade_repository_factory
        self._model_name = model_name
        self._limit = int(limit)
        self._clock = clock or (lambda: datetime.now(timezone(timedelta(hours=8))))

    async def multiday_state(
        self, mode: str, *, lookback_days: float = 5.0
    ) -> dict[str, dict[str, float]]:
        """Return realized PnL split by side over the recent multi-day window."""

        selected_mode = "live" if mode == "live" else "paper"
        try:
            async with self._session_factory() as session:
                rows = await self._trade_repository_factory(session).get_position_records(
                    execution_mode=selected_mode,
                    model_name=self._model_name,
                    is_open=False,
                    limit=self._limit,
                )
        except Exception as exc:
            logger.warning(
                "failed to calculate multiday side performance",
                mode=selected_mode,
                error=safe_error_text(exc),
            )
            return self.build_window([], lookback_days=lookback_days)
        return self.build_window(rows, lookback_days=lookback_days)

    def build_window(
        self, rows: Iterable[Any], *, lookback_days: float = 5.0
    ) -> dict[str, dict[str, float]]:
        """Build side buckets from closed rows within the recent N-day window."""

        now_local = _normalize_local_now(self._clock())
        window_start_utc = (now_local - timedelta(days=max(lookback_days, 0.0))).astimezone(UTC)
        result: dict[str, dict[str, float]] = {
            "long": _empty_side_bucket(),
            "short": _empty_side_bucket(),
        }
        for pos in rows:
            closed_at = _aware_utc(getattr(pos, "closed_at", None))
            if closed_at is None or closed_at < window_start_utc:
                continue
            if not closed_position_trade_fact_trusted(pos):
                continue
            side = "short" if str(getattr(pos, "side", "") or "").lower() == "short" else "long"
            pnl = float(getattr(pos, "realized_pnl", 0.0) or 0.0)
            bucket = result[side]
            bucket["count"] += 1
            bucket["pnl"] += pnl
            if pnl >= 0:
                bucket["wins"] += 1
                bucket["profit"] += pnl
            else:
                bucket["losses"] += 1
                bucket["loss"] += abs(pnl)
        self._finalize(result)
        for bucket in result.values():
            bucket["lookback_days"] = round(float(lookback_days), 4)
        return result

    async def state(self, mode: str) -> dict[str, dict[str, float]]:
        """Return today's closed-position PnL split by side."""

        selected_mode = "live" if mode == "live" else "paper"
        try:
            async with self._session_factory() as session:
                rows = await self._trade_repository_factory(session).get_position_records(
                    execution_mode=selected_mode,
                    model_name=self._model_name,
                    is_open=False,
                    limit=self._limit,
                )
        except Exception as exc:
            logger.warning(
                "failed to calculate today side performance",
                mode=selected_mode,
                error=safe_error_text(exc),
            )
            return self.build([])
        return self.build(rows)

    def build(self, rows: Iterable[Any]) -> dict[str, dict[str, float]]:
        """Build side buckets from already-loaded closed position rows."""

        now_local = _normalize_local_now(self._clock())
        start_utc = now_local.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(UTC)
        result: dict[str, dict[str, float]] = {
            "long": _empty_side_bucket(),
            "short": _empty_side_bucket(),
        }
        for pos in rows:
            closed_at = _aware_utc(getattr(pos, "closed_at", None))
            if closed_at is None or closed_at < start_utc:
                continue
            if not closed_position_trade_fact_trusted(pos):
                continue
            side = "short" if str(getattr(pos, "side", "") or "").lower() == "short" else "long"
            pnl = float(getattr(pos, "realized_pnl", 0.0) or 0.0)
            bucket = result[side]
            bucket["count"] += 1
            bucket["pnl"] += pnl
            if pnl >= 0:
                bucket["wins"] += 1
                bucket["profit"] += pnl
            else:
                bucket["losses"] += 1
                bucket["loss"] += abs(pnl)
        self._finalize(result)
        return result

    @staticmethod
    def _finalize(result: dict[str, dict[str, float]]) -> None:
        for bucket in result.values():
            count = max(bucket["count"], 1)
            bucket["avg_pnl"] = bucket["pnl"] / count
            bucket["win_rate"] = bucket["wins"] / count
            for key, value in list(bucket.items()):
                if isinstance(value, float):
                    bucket[key] = round(value, 6)
