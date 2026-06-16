"""Redis runtime factory used by trading and dashboard processes."""

from __future__ import annotations

from typing import Any

import structlog

from config.settings import settings
from core.safe_output import safe_error_text

logger = structlog.get_logger(__name__)


async def create_redis_client() -> Any | None:
    """Return a real Redis client when configured, otherwise fakeredis fallback."""

    if not bool(settings.use_fakeredis):
        try:
            import redis.asyncio as redis_asyncio

            client = redis_asyncio.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            await client.ping()
            logger.info("connected to redis", redis_url=settings.redis_url)
            return client
        except Exception as exc:
            logger.warning(
                "redis connection failed; dashboard pubsub disabled",
                redis_url=settings.redis_url,
                error=safe_error_text(exc),
            )
            return None

    try:
        import fakeredis.aioredis

        return fakeredis.aioredis.FakeRedis()
    except Exception as exc:
        logger.warning(
            "fakeredis initialization failed; continuing without redis",
            error=safe_error_text(exc),
        )
        return None
