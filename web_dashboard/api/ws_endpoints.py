"""
WebSocket manager for real-time dashboard updates.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog
from fastapi import WebSocket

from core.safe_output import safe_error_text

logger = structlog.get_logger(__name__)


class WebSocketManager:
    """Manages all connected WebSocket clients.

    Broadcasts updates to all connected dashboards:
    - Ticker prices
    - AI decisions
    - Trade executions
    - Risk alerts
    - Model rankings
    """

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._connections.add(ws)
        logger.info("ws client connected", total=len(self._connections))

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(ws)
        logger.info("ws client disconnected", total=len(self._connections))

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Send a message to all connected clients."""
        async with self._lock:
            connections = self._connections.copy()

        if not connections:
            return

        payload = json.dumps(message, default=str)
        dead: set[WebSocket] = set()

        for ws in connections:
            try:
                await ws.send_text(payload)
            except Exception as exc:
                logger.debug(
                    "ws send failed; marking connection dead",
                    error=safe_error_text(exc),
                )
                dead.add(ws)

        if dead:
            async with self._lock:
                self._connections -= dead

    async def broadcast_json(self, data: dict) -> None:
        await self.broadcast(data)

    async def close_all(self) -> None:
        async with self._lock:
            for ws in self._connections.copy():
                try:
                    await ws.close()
                except Exception as exc:
                    logger.debug("ws close failed", error=safe_error_text(exc))
            self._connections.clear()

    @property
    def connection_count(self) -> int:
        return len(self._connections)
