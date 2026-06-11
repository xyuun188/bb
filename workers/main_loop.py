"""
Main event loop — the heartbeat of the trading system.
Runs continuously, orchestrating the trading cycle.
"""

from __future__ import annotations

import asyncio
import signal
from typing import Any

import structlog

from ai_brain.model_factory import create_models_from_config
from ai_brain.model_registry import ModelRegistry
from config.settings import settings
from core.logging_config import setup_logging
from core.safe_output import safe_error_text
from core.trading_mode import mode_manager
from db.session import close_db, init_db
from services.competition_service import CompetitionService
from services.data_service import DataService
from services.notification_service import NotificationService
from services.trading_service import TradingService

logger = structlog.get_logger(__name__)


class MainLoop:
    """Central application runner.

    Initializes all services, manages lifecycle, handles graceful shutdown.
    """

    def __init__(self) -> None:
        self.model_registry = ModelRegistry()
        self.data_service = DataService()
        self.competition_service = CompetitionService()
        self.notification_service = NotificationService()

        # Redis client placeholder (will use fakeredis or real)
        self.redis: Any = None

        # Trading service (created after redis init)
        self.trading_service: TradingService | None = None

        self._running = False
        self._shutdown_event = asyncio.Event()

    async def initialize(self) -> None:
        """Initialize all subsystems in dependency order."""
        logger.info("initializing AI trading system...")

        # 1. Database
        await init_db()
        logger.info("database initialized")

        # 2. Redis (or fakeredis)
        await self._init_redis()

        # 3. Register AI models
        self._register_models()
        self.competition_service.set_active_models(self.model_registry.model_names)

        # 4. Data service (WebSocket + REST)
        await self.data_service.start()

        # 5. Trading service
        self.trading_service = TradingService(
            model_registry=self.model_registry,
            data_service=self.data_service,
            redis_client=self.redis,
        )
        await self.trading_service.initialize()

        # 6. Mode manager subscriptions
        mode_manager.subscribe(self._on_mode_change)

        logger.info("all systems initialized")

    def _register_models(self) -> None:
        """Register all AI models from configuration."""
        for model in create_models_from_config():
            self.model_registry.register(model)
        logger.info(
            "models registered",
            count=self.model_registry.model_count,
            names=self.model_registry.model_names,
        )

    async def _init_redis(self) -> None:
        """Initialize Redis connection or fakeredis."""
        if settings.use_fakeredis:
            import fakeredis.aioredis

            self.redis = fakeredis.aioredis.FakeRedis()
            logger.info("fakeredis initialized")
        else:
            try:
                import redis.asyncio as aioredis

                self.redis = await aioredis.from_url(settings.redis_url)
                await self.redis.ping()
                logger.info("redis connected")
            except Exception as e:
                logger.warning(
                    "redis connection failed, using fakeredis",
                    error=safe_error_text(e),
                )
                import fakeredis.aioredis

                self.redis = fakeredis.aioredis.FakeRedis()

    async def run(self) -> None:
        """Run the main application loop."""
        await self.initialize()

        self._running = True
        logger.info("=" * 60)
        logger.info("AI TRADING SYSTEM STARTED")
        logger.info(f"Mode: {mode_manager.mode.value}")
        logger.info(f"Symbols: {settings.symbols}")
        logger.info(f"Decision interval: {settings.decision_interval_seconds}s")
        logger.info(f"Dashboard: http://{settings.dashboard_host}:{settings.dashboard_port}")
        logger.info("=" * 60)

        # Start trading loop as a background task
        if self.trading_service is None:
            raise RuntimeError("trading service was not initialized")
        trading_task = asyncio.create_task(self.trading_service.start())

        # Periodic tasks
        evaluation_task = asyncio.create_task(self._periodic_model_evaluation())
        notification_task = asyncio.create_task(self._periodic_status_report())

        # Wait for shutdown signal
        await self._shutdown_event.wait()

        # Graceful shutdown
        logger.info("shutting down...")
        self._running = False

        # Cancel background tasks
        for task in [trading_task, evaluation_task, notification_task]:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        await self.shutdown()

    async def shutdown(self) -> None:
        """Gracefully shut down all subsystems."""
        if self.trading_service:
            await self.trading_service.stop()
        await self.data_service.stop()
        await self.model_registry.shutdown_all()
        await self.notification_service.close()
        await close_db()
        logger.info("system shutdown complete")

    async def _periodic_model_evaluation(self) -> None:
        """Periodically evaluate model performance and update rankings."""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Every hour
                rankings = await self.competition_service.evaluate_all_models()
                if rankings:
                    best = rankings[0]["model_name"]
                    logger.info("model rankings updated", best=best)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("model evaluation error", error=safe_error_text(e))

    async def _periodic_status_report(self) -> None:
        """Periodic status logging."""
        while self._running:
            try:
                await asyncio.sleep(600)  # Every 10 minutes
                if self.trading_service:
                    stats = self.trading_service.get_stats()
                    logger.info("status report", **stats)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("status report error", error=safe_error_text(exc))

    def _on_mode_change(self, manager) -> None:
        """Callback when trading mode changes."""
        logger.info(
            "trading mode changed",
            mode=manager.mode.value,
            live_model=manager.live_model_name,
        )

    def request_shutdown(self) -> None:
        """Signal the main loop to shut down."""
        self._shutdown_event.set()


# Signal handlers for graceful shutdown
_loop_instance: MainLoop | None = None


def _signal_handler(signum, frame):
    logger.info(f"received signal {signum}, shutting down...")
    if _loop_instance:
        _loop_instance.request_shutdown()


async def run_main_loop() -> None:
    global _loop_instance
    setup_logging()

    # Register signal handlers
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    _loop_instance = MainLoop()
    await _loop_instance.run()
