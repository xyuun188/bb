"""
Trading mode management.
Handles paper/live mode switching with thread-safe state.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from enum import Enum


class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


class TradingModeManager:
    """Thread-safe singleton managing the current trading mode.

    In PAPER mode: all registered AI models trade against virtual accounts.
    In LIVE mode: only the best model executes real orders; others run silently.
    """

    _instance: TradingModeManager | None = None
    _lock: asyncio.Lock | None = None

    def __init__(self) -> None:
        self._mode: TradingMode = TradingMode.PAPER
        self._paused: bool = False
        self._scan_mode: str = "auto"  # "auto" or "manual"
        self._live_model_name: str | None = None
        self._mode_changed_at: datetime = datetime.utcnow()
        self._subscribers: list[callable] = []

    @classmethod
    def get_instance(cls) -> TradingModeManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def mode(self) -> TradingMode:
        return self._mode

    @property
    def is_paper(self) -> bool:
        return self._mode == TradingMode.PAPER

    @property
    def is_live(self) -> bool:
        return self._mode == TradingMode.LIVE

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def scan_mode(self) -> str:
        return self._scan_mode

    @property
    def is_auto_scan(self) -> bool:
        return self._scan_mode == "auto"

    @property
    def live_model_name(self) -> str | None:
        return self._live_model_name

    @property
    def mode_changed_at(self) -> datetime:
        return self._mode_changed_at

    async def switch_to_paper(self) -> None:
        self._mode = TradingMode.PAPER
        self._mode_changed_at = datetime.utcnow()
        await self._notify()

    async def switch_to_live(self, model_name: str) -> None:
        self._mode = TradingMode.LIVE
        self._live_model_name = model_name
        self._mode_changed_at = datetime.utcnow()
        await self._notify()

    async def pause(self) -> None:
        self._paused = True
        await self._notify()

    async def resume(self) -> None:
        self._paused = False
        await self._notify()

    async def switch_to_auto(self) -> None:
        self._scan_mode = "auto"
        self._mode_changed_at = datetime.utcnow()
        await self._notify()

    async def switch_to_manual(self) -> None:
        self._scan_mode = "manual"
        self._mode_changed_at = datetime.utcnow()
        await self._notify()

    def subscribe(self, callback: callable) -> None:
        """Register a callback invoked on mode/pause changes."""
        self._subscribers.append(callback)

    async def _notify(self) -> None:
        for cb in self._subscribers:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(self)
                else:
                    cb(self)
            except Exception:
                pass

    def get_state(self) -> dict:
        return {
            "mode": self._mode.value,
            "paused": self._paused,
            "scan_mode": self._scan_mode,
            "live_model_name": self._live_model_name,
            "mode_changed_at": self._mode_changed_at.isoformat(),
        }


# Convenience singleton access
mode_manager = TradingModeManager.get_instance()
