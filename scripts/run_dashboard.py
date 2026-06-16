#!/usr/bin/env python3
"""Run only the BB web dashboard process.

The trading engine can run in a separate process with DASHBOARD_INLINE_ENABLED=false.
This dashboard process reads the same database, protects HTTP/WebSocket access,
and fans Redis dashboard:update messages out to connected WebSocket clients.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structlog
import uvicorn

from config.settings import settings
from core.logging_config import setup_logging
from core.redis_runtime import create_redis_client
from core.safe_output import safe_error_text
from db.session import close_db, init_db
from services.secure_runtime_config import load_secure_settings_into_runtime
from web_dashboard.app import app, ws_manager

logger = structlog.get_logger("dashboard")


def _decode_pubsub_payload(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.debug("ignored invalid dashboard pubsub JSON", error=safe_error_text(exc))
            return None
    elif isinstance(raw, dict):
        payload = raw
    else:
        return None
    return payload if isinstance(payload, dict) else None


async def _redis_dashboard_listener(redis: Any | None) -> None:
    if redis is None:
        logger.warning("dashboard Redis listener disabled; no Redis client available")
        return
    pubsub = redis.pubsub()
    try:
        await pubsub.subscribe("dashboard:update")
        logger.info("dashboard subscribed to Redis updates", channel="dashboard:update")
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if not message:
                await asyncio.sleep(0.05)
                continue
            payload = _decode_pubsub_payload(message.get("data"))
            if payload:
                await ws_manager.broadcast(payload)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("dashboard Redis listener stopped", error=safe_error_text(exc))
    finally:
        try:
            await pubsub.unsubscribe("dashboard:update")
            await pubsub.close()
        except Exception as exc:
            logger.debug("dashboard Redis pubsub close failed", error=safe_error_text(exc))


async def _close_redis(redis: Any | None) -> None:
    if redis is None:
        return
    close = getattr(redis, "aclose", None) or getattr(redis, "close", None)
    if callable(close):
        result = close()
        if asyncio.iscoroutine(result):
            await result


async def main() -> None:
    setup_logging()
    await init_db()
    await load_secure_settings_into_runtime()
    redis = await create_redis_client()
    listener_task = asyncio.create_task(_redis_dashboard_listener(redis))

    logger.info(
        "starting dashboard process",
        host=settings.dashboard_host,
        port=settings.dashboard_port,
        auth_enabled=settings.dashboard_auth_enabled,
    )
    config = uvicorn.Config(
        app,
        host=settings.dashboard_host,
        port=settings.dashboard_port,
        log_level="info",
    )
    server = uvicorn.Server(config)
    try:
        await server.serve()
    finally:
        listener_task.cancel()
        try:
            await listener_task
        except asyncio.CancelledError:
            pass
        await _close_redis(redis)
        await close_db()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
