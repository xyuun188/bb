"""
Trading mode management.
Handles paper/live mode switching with process-shared control state.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from datetime import datetime
from enum import StrEnum
from inspect import isawaitable
from pathlib import Path
from typing import Any

import structlog

from core.safe_output import safe_error_text

logger = structlog.get_logger(__name__)

ModeSubscriber = Callable[["TradingModeManager"], Awaitable[None] | None]


class TradingMode(StrEnum):
    PAPER = "paper"
    LIVE = "live"


class TradingModeManager:
    """Thread-safe singleton managing the current trading mode.

    Dashboard and trading workers can run in separate processes. The control
    state is therefore persisted to a small JSON file so a dashboard pause is
    visible to the trading loop before it selects new market symbols.
    """

    _instance: TradingModeManager | None = None
    _lock: asyncio.Lock | None = None

    def __init__(self, *, state_path: Path | None = None) -> None:
        self._mode: TradingMode = TradingMode.PAPER
        self._paused: bool = False
        self._scan_mode: str = "auto"
        self._live_model_name: str | None = None
        self._mode_changed_at: datetime = datetime.utcnow()
        self._subscribers: list[ModeSubscriber] = []
        self._state_path = state_path or self._default_state_path()
        self._last_state_mtime: float = 0.0
        self._last_state_size: int = -1
        self._load_state_from_disk()

    @classmethod
    def get_instance(cls) -> TradingModeManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def mode(self) -> TradingMode:
        self._load_state_from_disk()
        return self._mode

    @property
    def is_paper(self) -> bool:
        return self.mode == TradingMode.PAPER

    @property
    def is_live(self) -> bool:
        return self.mode == TradingMode.LIVE

    @property
    def is_paused(self) -> bool:
        self._load_state_from_disk()
        return self._paused

    @property
    def scan_mode(self) -> str:
        self._load_state_from_disk()
        return self._scan_mode

    @property
    def is_auto_scan(self) -> bool:
        return self.scan_mode == "auto"

    @property
    def live_model_name(self) -> str | None:
        self._load_state_from_disk()
        return self._live_model_name

    @property
    def mode_changed_at(self) -> datetime:
        self._load_state_from_disk()
        return self._mode_changed_at

    async def switch_to_paper(self) -> None:
        self._mode = TradingMode.PAPER
        self._mode_changed_at = datetime.utcnow()
        self._persist_state_to_disk()
        await self._notify()

    async def switch_to_live(self, model_name: str) -> None:
        self._mode = TradingMode.LIVE
        self._live_model_name = model_name
        self._mode_changed_at = datetime.utcnow()
        self._persist_state_to_disk()
        await self._notify()

    async def pause(self) -> None:
        self._paused = True
        self._mode_changed_at = datetime.utcnow()
        self._persist_state_to_disk()
        await self._notify()

    async def resume(self) -> None:
        self._paused = False
        self._mode_changed_at = datetime.utcnow()
        self._persist_state_to_disk()
        await self._notify()

    async def switch_to_auto(self) -> None:
        self._scan_mode = "auto"
        self._mode_changed_at = datetime.utcnow()
        self._persist_state_to_disk()
        await self._notify()

    async def switch_to_manual(self) -> None:
        self._scan_mode = "auto"
        self._mode_changed_at = datetime.utcnow()
        self._persist_state_to_disk()
        await self._notify()

    def subscribe(self, callback: ModeSubscriber) -> None:
        """Register a callback invoked on mode/pause changes."""
        self._subscribers.append(callback)

    async def _notify(self) -> None:
        for cb in self._subscribers:
            try:
                result = cb(self)
                if isawaitable(result):
                    await result
            except Exception as exc:
                logger.warning("trading mode subscriber failed", error=safe_error_text(exc))

    def get_state(self) -> dict[str, Any]:
        self._load_state_from_disk()
        return self._serialize_state()

    @staticmethod
    def _default_state_path() -> Path:
        configured = os.getenv("BB_TRADING_CONTROL_STATE_PATH", "").strip()
        if configured:
            return Path(configured)
        return Path(__file__).resolve().parent.parent / "data" / "trading-control-state.json"

    def _serialize_state(self) -> dict[str, Any]:
        return {
            "mode": self._mode.value,
            "paused": self._paused,
            "scan_mode": self._scan_mode,
            "live_model_name": self._live_model_name,
            "mode_changed_at": self._mode_changed_at.isoformat(),
        }

    def _persist_state_to_disk(self) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(
                self._serialize_state(),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            temp_path = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
            temp_path.write_text(payload + "\n", encoding="utf-8")
            temp_path.replace(self._state_path)
            state_stat = self._state_path.stat()
            self._last_state_mtime = state_stat.st_mtime
            self._last_state_size = state_stat.st_size
        except Exception as exc:
            logger.warning(
                "failed to persist trading control state",
                path=str(self._state_path),
                error=safe_error_text(exc),
            )

    def _load_state_from_disk(self) -> None:
        try:
            try:
                state_stat = self._state_path.stat()
            except FileNotFoundError:
                return
            if (
                self._last_state_mtime == state_stat.st_mtime
                and self._last_state_size == state_stat.st_size
            ):
                return
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return
            mode = payload.get("mode")
            if mode in {TradingMode.PAPER.value, TradingMode.LIVE.value}:
                self._mode = TradingMode(mode)
            self._paused = bool(payload.get("paused", self._paused))
            scan_mode = str(payload.get("scan_mode") or self._scan_mode)
            self._scan_mode = "auto" if scan_mode != "auto" else scan_mode
            live_model_name = payload.get("live_model_name")
            self._live_model_name = str(live_model_name) if live_model_name else None
            changed_at = payload.get("mode_changed_at")
            if isinstance(changed_at, str) and changed_at:
                try:
                    self._mode_changed_at = datetime.fromisoformat(changed_at)
                except ValueError:
                    pass
            self._last_state_mtime = state_stat.st_mtime
            self._last_state_size = state_stat.st_size
        except Exception as exc:
            logger.warning(
                "failed to load trading control state",
                path=str(self._state_path),
                error=safe_error_text(exc),
            )


# Convenience singleton access
mode_manager = TradingModeManager.get_instance()
