"""Observation-only fee-after performance profiles by symbol and side."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from math import sqrt
from typing import Any

import structlog

from config.settings import ENSEMBLE_TRADER_NAME
from core.safe_output import safe_error_text
from db.repositories.trade_repo import TradeRepository
from db.session import get_session_ctx
from services.trade_fact_trust import closed_position_trade_fact_trusted

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


def _profile_template() -> dict[str, Any]:
    return {
        "count": 0,
        "wins": 0,
        "losses": 0,
        "pnl": 0.0,
        "profit": 0.0,
        "loss": 0.0,
        "largest_loss": 0.0,
        "first_closed_at": None,
        "last_closed_at": None,
        "_returns": [],
    }


class SymbolSidePerformanceService:
    """Build read-only return distributions from trusted closed positions."""

    def __init__(
        self,
        *,
        session_factory: SessionFactory = get_session_ctx,
        trade_repository_factory: TradeRepositoryFactory = TradeRepository,
        normalize_symbol: NormalizeSymbol | None = None,
        model_name: str = ENSEMBLE_TRADER_NAME,
        lookback_limit: int = DEFAULT_SYMBOL_SIDE_PROFILE_LOOKBACK,
        lookback_days: float = DEFAULT_SYMBOL_PROFIT_PROFILE_LOOKBACK_DAYS,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._trade_repository_factory = trade_repository_factory
        self._normalize_symbol = normalize_symbol or (lambda symbol: str(symbol or ""))
        self._model_name = model_name
        self._lookback_limit = int(lookback_limit)
        self._lookback_days = float(lookback_days)
        self._clock = clock or (lambda: datetime.now(UTC))

    async def recent(self, mode: str) -> dict[str, dict[str, Any]]:
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
        now_utc = _ensure_utc(self._clock()) or datetime.now(UTC)
        window_start = now_utc - timedelta(days=self._lookback_days)
        profiles: dict[str, dict[str, Any]] = {}

        for position in rows:
            closed_at = _ensure_utc(getattr(position, "closed_at", None))
            if closed_at is None or closed_at < window_start:
                continue
            if not closed_position_trade_fact_trusted(position):
                continue
            symbol = self._normalize_symbol(getattr(position, "symbol", None)) or str(
                getattr(position, "symbol", "") or ""
            )
            if not symbol:
                continue
            side = (
                "short"
                if str(getattr(position, "side", "") or "").lower() == "short"
                else "long"
            )
            pnl = float(getattr(position, "realized_pnl", 0.0) or 0.0)
            closed_iso = closed_at.isoformat()
            for key in (f"{symbol}|{side}", f"{symbol}|all"):
                bucket = profiles.setdefault(key, _profile_template())
                bucket["count"] += 1
                bucket["pnl"] += pnl
                bucket["_returns"].append(pnl)
                bucket["wins" if pnl >= 0.0 else "losses"] += 1
                if pnl >= 0.0:
                    bucket["profit"] += pnl
                else:
                    bucket["loss"] += abs(pnl)
                    bucket["largest_loss"] = min(float(bucket["largest_loss"]), pnl)
                if not bucket["first_closed_at"] or closed_iso < bucket["first_closed_at"]:
                    bucket["first_closed_at"] = closed_iso
                if not bucket["last_closed_at"] or closed_iso > bucket["last_closed_at"]:
                    bucket["last_closed_at"] = closed_iso

        generated_at = now_utc.isoformat()
        for key, bucket in profiles.items():
            values = [float(value) for value in bucket.pop("_returns", [])]
            count = len(values)
            mean = sum(values) / count if count else 0.0
            if count > 1:
                variance = sum((value - mean) ** 2 for value in values) / (count - 1)
                uncertainty = sqrt(max(variance, 0.0) / count)
            else:
                uncertainty = abs(mean) if count else 0.0
            profit = float(bucket["profit"])
            loss = float(bucket["loss"])
            bucket.update(
                {
                    "avg_pnl": round(mean, 8),
                    "return_lcb_usdt": round(mean - uncertainty, 8),
                    "return_uncertainty_usdt": round(uncertainty, 8),
                    "win_rate": round(float(bucket["wins"]) / count, 8) if count else 0.0,
                    "profit_factor": round(profit / loss, 8) if loss > 0.0 else None,
                    "profile_scope": "symbol" if key.endswith("|all") else "symbol_side",
                    "production_permission": False,
                    "lookback_days": self._lookback_days,
                    "policy_provenance": {
                        "source": "trusted_closed_position_fee_after_pnl_distribution",
                        "observation_window": f"rolling_{self._lookback_days:g}_days",
                        "sample_count": count,
                        "generated_at": generated_at,
                        "strategy_version": "2026-07-12.symbol-side-return-observation.v1",
                        "fallback_reason": "" if count else "distribution_empty",
                    },
                }
            )
            for field in ("pnl", "profit", "loss", "largest_loss"):
                bucket[field] = round(float(bucket[field]), 8)
        return profiles
