from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from config.settings import ENSEMBLE_TRADER_NAME
from core.safe_output import safe_error_text

logger = structlog.get_logger(__name__)

DEFAULT_POSITION_PROFIT_PEAKS_STATE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "position_profit_peaks.json"
)


def _default_float_parser(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class PositionProfitPeakTracker:
    """Persist and update floating-profit peak state for open positions."""

    def __init__(
        self,
        *,
        path: Path = DEFAULT_POSITION_PROFIT_PEAKS_STATE_PATH,
        symbol_normalizer: Callable[[Any], str] | None = None,
        float_parser: Callable[[Any, float], float] | None = None,
    ) -> None:
        self.path = path
        self.symbol_normalizer = symbol_normalizer or (lambda symbol: str(symbol or ""))
        self.float_parser = float_parser or _default_float_parser
        self.peaks: dict[str, dict[str, Any]] = self._load()

    def key(self, model_name: str, symbol: str, side: str) -> str:
        return "|".join(
            [
                str(model_name or ENSEMBLE_TRADER_NAME),
                self.symbol_normalizer(symbol) or str(symbol or ""),
                str(side or "").lower(),
            ]
        )

    def update(
        self,
        *,
        model_name: str,
        symbol: str,
        side: str,
        current_price: float,
        entry_price: float,
        unrealized_pnl: float,
        hold_minutes: float | None,
        quantity: float | None = None,
    ) -> dict[str, Any]:
        key = self.key(model_name, symbol, side)
        now = datetime.now(UTC).isoformat()
        entry_price = self.float_parser(entry_price, 0.0)
        current_price = self.float_parser(current_price, 0.0)
        unrealized_pnl = self.float_parser(unrealized_pnl, 0.0)
        quantity_value = abs(self.float_parser(quantity, 0.0))
        hold_minutes = float(hold_minutes or 0.0)
        if entry_price <= 0 or current_price <= 0:
            return {}

        if side == "short":
            pnl_ratio = max((entry_price - current_price) / entry_price, 0.0)
        else:
            pnl_ratio = max((current_price - entry_price) / entry_price, 0.0)

        position_notional = abs(
            (current_price if current_price > 0 else entry_price) * quantity_value
        )
        state = self.peaks.get(key) or {}
        if state and not self._state_matches_position(
            state,
            entry_price=entry_price,
            quantity=quantity_value,
            position_notional=position_notional,
            hold_minutes=hold_minutes,
        ):
            state = {}
        state = state or {
            "peak_unrealized_pnl": unrealized_pnl,
            "peak_pnl_ratio": pnl_ratio,
            "last_unrealized_pnl": unrealized_pnl,
            "last_pnl_ratio": pnl_ratio,
            "updated_at": now,
            "hold_minutes": hold_minutes,
        }
        state["peak_unrealized_pnl"] = max(
            self.float_parser(state.get("peak_unrealized_pnl"), unrealized_pnl),
            unrealized_pnl,
        )
        state["peak_pnl_ratio"] = max(
            self.float_parser(state.get("peak_pnl_ratio"), pnl_ratio),
            pnl_ratio,
        )
        state["last_unrealized_pnl"] = unrealized_pnl
        state["last_pnl_ratio"] = pnl_ratio
        state["updated_at"] = now
        state["hold_minutes"] = hold_minutes
        state["entry_price"] = entry_price
        state["quantity"] = quantity_value
        state["position_notional"] = position_notional
        self.peaks[key] = state
        self.save()
        return state

    def _state_matches_position(
        self,
        state: dict[str, Any],
        *,
        entry_price: float,
        quantity: float,
        position_notional: float,
        hold_minutes: float,
    ) -> bool:
        if entry_price > 0.0 and self.float_parser(state.get("entry_price"), 0.0) <= 0.0:
            return False
        if quantity > 0.0 and self.float_parser(state.get("quantity"), 0.0) <= 0.0:
            return False
        if (
            position_notional > 0.0
            and self.float_parser(state.get("position_notional"), 0.0) <= 0.0
        ):
            return False
        stored_hold = self.float_parser(state.get("hold_minutes"), 0.0)
        return hold_minutes >= stored_hold

    def seconds_since_profit_exit(self, peak_state: dict[str, Any]) -> float:
        value = peak_state.get("last_profit_exit_at") if isinstance(peak_state, dict) else None
        if not value:
            return 0.0
        try:
            exited_at = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if exited_at.tzinfo is None:
                exited_at = exited_at.replace(tzinfo=UTC)
            return max((datetime.now(UTC) - exited_at).total_seconds(), 0.0)
        except (TypeError, ValueError):
            return 0.0

    def remember_profit_exit(self, model_name: str, symbol: str, side: str) -> None:
        key = self.key(model_name, symbol, side)
        state = self.peaks.get(key)
        if isinstance(state, dict):
            state["last_profit_exit_at"] = datetime.now(UTC).isoformat()
            state["profit_exit_count"] = int(state.get("profit_exit_count") or 0) + 1
            self.peaks[key] = state
            self.save()

    def remove(self, model_name: str, symbol: str, side: str) -> None:
        removed = self.peaks.pop(self.key(model_name, symbol, side), None)
        if removed is not None:
            self.save()

    def prune(self, open_positions: list[dict[str, Any]]) -> None:
        valid = {
            self.key(
                str(pos.get("model_name") or ENSEMBLE_TRADER_NAME),
                str(pos.get("symbol") or ""),
                str(pos.get("side") or ""),
            )
            for pos in open_positions or []
            if pos.get("is_open", True)
        }
        before = set(self.peaks.keys())
        for key in list(self.peaks.keys()):
            if key not in valid:
                self.peaks.pop(key, None)
        if set(self.peaks.keys()) != before:
            self.save()

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as fh:
                json.dump(self.peaks, fh, ensure_ascii=False, indent=2)
            tmp_path.replace(self.path)
        except OSError as exc:
            logger.warning("failed to save position profit peaks", error=safe_error_text(exc))

    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            if not self.path.exists():
                return {}
            with self.path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if not isinstance(payload, dict):
                return {}
            loaded: dict[str, dict[str, Any]] = {}
            for key, value in payload.items():
                if not isinstance(key, str) or not isinstance(value, dict):
                    continue
                loaded[key] = {
                    "peak_unrealized_pnl": float(value.get("peak_unrealized_pnl") or 0.0),
                    "peak_pnl_ratio": float(value.get("peak_pnl_ratio") or 0.0),
                    "last_unrealized_pnl": float(value.get("last_unrealized_pnl") or 0.0),
                    "last_pnl_ratio": float(value.get("last_pnl_ratio") or 0.0),
                    "updated_at": str(value.get("updated_at") or ""),
                    "hold_minutes": float(value.get("hold_minutes") or 0.0),
                    "last_profit_exit_at": value.get("last_profit_exit_at") or "",
                    "profit_exit_count": int(value.get("profit_exit_count") or 0),
                    "entry_price": float(value.get("entry_price") or 0.0),
                    "quantity": float(value.get("quantity") or 0.0),
                    "position_notional": float(value.get("position_notional") or 0.0),
                }
            logger.info("loaded position profit peaks", count=len(loaded), path=str(self.path))
            return loaded
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("failed to load position profit peaks", error=safe_error_text(exc))
            return {}
