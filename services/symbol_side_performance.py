"""Recent realized PnL profiles by symbol and side.

This service feeds entry scoring and loss-cooldown policies with a compact
view of what has actually made or lost money recently.  Keeping it outside the
main trading orchestrator makes the feedback loop testable and easier to tune.
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
from services.trading_params import DEFAULT_TRADING_PARAMS, EntryLossCooldownParams

SessionFactory = Callable[[], Any]
TradeRepositoryFactory = Callable[[Any], TradeRepository]
NormalizeSymbol = Callable[[str | None], str | None]

DEFAULT_SYMBOL_SIDE_PROFILE_LOOKBACK = 2000
DEFAULT_SYMBOL_PROFIT_PROFILE_LOOKBACK_DAYS = 7.0

logger = structlog.get_logger(__name__)


def _ensure_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _today_start_utc(now_utc: datetime) -> datetime:
    now_local = now_utc.astimezone(timezone(timedelta(hours=8)))
    return now_local.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(UTC)


def _profile_template() -> dict[str, Any]:
    return {
        "count": 0,
        "wins": 0,
        "losses": 0,
        "pnl": 0.0,
        "profit": 0.0,
        "loss": 0.0,
        "largest_loss": 0.0,
        "today_count": 0,
        "today_pnl": 0.0,
        "today_loss": 0.0,
        "first_closed_at": None,
        "last_closed_at": None,
        "last_loss_at": None,
    }


class SymbolSidePerformanceService:
    """Build recent realized-PnL profiles used by entry feedback policies."""

    def __init__(
        self,
        *,
        session_factory: SessionFactory = get_session_ctx,
        trade_repository_factory: TradeRepositoryFactory = TradeRepository,
        normalize_symbol: NormalizeSymbol | None = None,
        model_name: str = ENSEMBLE_TRADER_NAME,
        lookback_limit: int = DEFAULT_SYMBOL_SIDE_PROFILE_LOOKBACK,
        lookback_days: float = DEFAULT_SYMBOL_PROFIT_PROFILE_LOOKBACK_DAYS,
        loss_cooldown_params: EntryLossCooldownParams = (
            DEFAULT_TRADING_PARAMS.entry_loss_cooldown
        ),
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._trade_repository_factory = trade_repository_factory
        self._normalize_symbol = normalize_symbol or (lambda symbol: str(symbol or ""))
        self._model_name = model_name
        self._lookback_limit = int(lookback_limit)
        self._lookback_days = float(lookback_days)
        self._loss_cooldown_params = loss_cooldown_params
        self._clock = clock or (lambda: datetime.now(UTC))

    async def recent(self, mode: str) -> dict[str, dict[str, Any]]:
        """Return recent realized performance for the selected execution mode."""

        selected_mode = "live" if mode == "live" else "paper"
        try:
            async with self._session_factory() as session:
                rows = await self._trade_repository_factory(session).get_position_records(
                    execution_mode=selected_mode,
                    model_name=self._model_name,
                    is_open=False,
                    limit=self._lookback_limit,
                )
        except Exception as exc:
            logger.warning(
                "failed to load symbol side performance rows",
                mode=selected_mode,
                error=safe_error_text(exc),
            )
            return {}
        return self.build_profiles(rows)

    def build_profiles(self, rows: Iterable[Any]) -> dict[str, dict[str, Any]]:
        """Build profiles from already-loaded closed position records."""

        now_utc = _ensure_utc(self._clock()) or datetime.now(UTC)
        today_start_utc = _today_start_utc(now_utc)
        window_start_utc = now_utc - timedelta(days=self._lookback_days)
        profiles: dict[str, dict[str, Any]] = {}

        def profile_for(key: str) -> dict[str, Any]:
            if key not in profiles:
                profiles[key] = _profile_template()
            return profiles[key]

        for pos in rows:
            closed_at = _ensure_utc(getattr(pos, "closed_at", None))
            if closed_at is None or closed_at < window_start_utc:
                continue
            symbol = self._normalize_symbol(getattr(pos, "symbol", None)) or str(
                getattr(pos, "symbol", "") or ""
            )
            if not symbol:
                continue
            side = "short" if str(getattr(pos, "side", "") or "").lower() == "short" else "long"
            pnl = float(getattr(pos, "realized_pnl", 0.0) or 0.0)
            closed_iso = closed_at.isoformat()
            for key in (f"{symbol}|{side}", f"{symbol}|all"):
                bucket = profile_for(key)
                bucket["count"] += 1
                bucket["pnl"] += pnl
                first_closed_at = bucket.get("first_closed_at")
                last_closed_at = bucket.get("last_closed_at")
                if not first_closed_at or closed_iso < str(first_closed_at):
                    bucket["first_closed_at"] = closed_iso
                if not last_closed_at or closed_iso > str(last_closed_at):
                    bucket["last_closed_at"] = closed_iso
                if closed_at >= today_start_utc:
                    bucket["today_count"] += 1
                    bucket["today_pnl"] += pnl
                    if pnl < 0:
                        bucket["today_loss"] += abs(pnl)
                if pnl >= 0:
                    bucket["wins"] += 1
                    bucket["profit"] += pnl
                else:
                    bucket["losses"] += 1
                    bucket["loss"] += abs(pnl)
                    bucket["largest_loss"] = min(float(bucket.get("largest_loss") or 0.0), pnl)
                    last_loss_at = bucket.get("last_loss_at")
                    if not last_loss_at or closed_iso > str(last_loss_at):
                        bucket["last_loss_at"] = closed_iso

        self._finalize_profiles(profiles, now_utc, today_start_utc)
        return profiles

    def _finalize_profiles(
        self,
        profiles: dict[str, dict[str, Any]],
        now_utc: datetime,
        today_start_utc: datetime,
    ) -> None:
        for key, bucket in profiles.items():
            count = max(int(bucket.get("count") or 0), 1)
            profit = float(bucket.get("profit") or 0.0)
            loss = float(bucket.get("loss") or 0.0)
            pnl = float(bucket.get("pnl") or 0.0)
            losses = int(bucket.get("losses") or 0)
            today_pnl = float(bucket.get("today_pnl") or 0.0)
            today_loss = float(bucket.get("today_loss") or 0.0)
            last_loss_age_hours = self._last_loss_age_hours(bucket, now_utc)
            recent_loss_cooldown_active = (
                last_loss_age_hours <= self._loss_cooldown_params.hard_cooldown_hours
            )
            is_symbol_profile = key.endswith("|all")
            cooldown, cooldown_reason = self._cooldown_state(
                is_symbol_profile=is_symbol_profile,
                recent_loss_cooldown_active=recent_loss_cooldown_active,
                pnl=pnl,
                profit=profit,
                loss=loss,
                losses=losses,
                today_pnl=today_pnl,
                today_loss=today_loss,
            )
            cooldown_remaining_hours = (
                max(
                    self._loss_cooldown_params.hard_cooldown_hours - last_loss_age_hours,
                    0.0,
                )
                if cooldown and not is_symbol_profile
                else 0.0
            )
            bucket.update(
                {
                    "avg_pnl": round(pnl / count, 6),
                    "win_rate": round(float(bucket.get("wins") or 0) / count, 6),
                    "profit_factor": (
                        round(profit / loss, 6) if loss > 0 else (999.0 if profit > 0 else 0.0)
                    ),
                    "cooldown": cooldown,
                    "cooldown_reason": cooldown_reason,
                    "last_loss_age_hours": round(last_loss_age_hours, 6),
                    "cooldown_remaining_hours": round(cooldown_remaining_hours, 6),
                    "lookback_days": self._lookback_days,
                    "age_seconds": round((now_utc - today_start_utc).total_seconds(), 3),
                }
            )
            for field in ("pnl", "profit", "loss", "largest_loss", "today_pnl", "today_loss"):
                bucket[field] = round(float(bucket.get(field) or 0.0), 6)

    @staticmethod
    def _last_loss_age_hours(bucket: dict[str, Any], now_utc: datetime) -> float:
        last_loss_at = bucket.get("last_loss_at")
        if not last_loss_at:
            return 9999.0
        try:
            parsed = datetime.fromisoformat(str(last_loss_at))
        except ValueError:
            return 0.0
        parsed_utc = _ensure_utc(parsed)
        if parsed_utc is None:
            return 0.0
        return max((now_utc - parsed_utc).total_seconds() / 3600.0, 0.0)

    def _cooldown_state(
        self,
        *,
        is_symbol_profile: bool,
        recent_loss_cooldown_active: bool,
        pnl: float,
        profit: float,
        loss: float,
        losses: int,
        today_pnl: float,
        today_loss: float,
    ) -> tuple[bool, str]:
        params = self._loss_cooldown_params
        if (
            is_symbol_profile
            and losses >= params.quarantine_min_losses
            and pnl <= -params.quarantine_loss_usdt
        ):
            return True, "该币种最近滚动真实亏损过大"
        if is_symbol_profile and (
            today_loss >= params.total_cooldown_loss_usdt
            or today_pnl <= -params.total_cooldown_loss_usdt
        ):
            return True, "该币种今天累计真实亏损超过限制"
        if (
            not is_symbol_profile
            and recent_loss_cooldown_active
            and (
                loss >= params.total_cooldown_loss_usdt
                or pnl <= -params.total_cooldown_loss_usdt
                or today_loss >= params.side_cooldown_loss_usdt
                or today_pnl <= -params.side_cooldown_loss_usdt
            )
        ):
            return True, "该币种这个方向的真实亏损已经超过限制"
        if (
            not is_symbol_profile
            and recent_loss_cooldown_active
            and (
                pnl <= -params.side_cooldown_loss_usdt
                or (losses >= 2 and pnl < 0 and loss > profit * 1.2)
            )
        ):
            return True, "该币种这个方向近期真实盈亏表现偏弱"
        return False, ""
