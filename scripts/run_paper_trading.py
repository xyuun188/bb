#!/usr/bin/env python3
"""
Launch the full paper trading system.
Starts: data feed, AI brain (all models), paper executor, and web dashboard.

Run: python scripts/run_paper_trading.py
Then open: http://localhost:8002
"""

import asyncio
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, TextIO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structlog
import uvicorn

from ai_brain.model_factory import create_models_from_config
from ai_brain.model_registry import ModelRegistry
from config.settings import ENSEMBLE_TRADER_NAME, settings
from core.logging_config import setup_logging
from core.redis_runtime import create_redis_client
from core.safe_output import safe_error_text
from db.session import close_db, init_db
from services.competition_service import CompetitionService
from services.data_service import DataService
from services.notification_service import NotificationService
from services.secure_runtime_config import load_secure_settings_into_runtime
from services.trading_service import TradingService
from web_dashboard.api.dashboard import (
    _build_tickers_for_open_positions,
    _get_open_position_symbols,
    set_services,
)
from web_dashboard.app import app, ws_manager

logger = structlog.get_logger(__name__)

LOCK_FILE = Path(__file__).resolve().parent.parent / "data" / "paper_trading.lock"
_lock_handle: TextIO | None = None


async def _send_dashboard_message(redis: Any | None, inline_dashboard: bool, message: dict[str, Any]) -> None:
    if inline_dashboard:
        await ws_manager.broadcast(message)
        return
    if redis is None:
        return
    try:
        await redis.publish("dashboard:update", json.dumps(message, default=str))
    except Exception as exc:
        logger.debug("dashboard Redis publish failed", error=safe_error_text(exc))


def _lock_file(handle: TextIO) -> None:
    """Acquire a non-blocking advisory lock on the current platform."""
    if os.name == "nt":
        msvcrt = importlib.import_module("msvcrt")
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return

    fcntl = importlib.import_module("fcntl")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def acquire_single_instance_lock() -> None:
    """Prevent two trading loops from running against the same account/db."""
    global _lock_handle
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    _lock_handle = open(LOCK_FILE, "a+", encoding="utf-8")
    try:
        _lock_file(_lock_handle)
    except OSError as exc:
        raise SystemExit(
            "Another paper trading instance is already running. "
            "Stop the existing process before starting a new one."
        ) from exc
    _lock_handle.seek(0)
    _lock_handle.truncate()
    _lock_handle.write(str(os.getpid()))
    _lock_handle.flush()


async def main():
    acquire_single_instance_lock()
    setup_logging()
    print("=" * 60)
    print("AI CRYPTO TRADING SYSTEM - PAPER TRADING MODE")
    print("=" * 60)
    print(f"Symbols: {settings.symbols}")
    model_names = [
        m.get("name", "llm_agent") for m in settings.get_fixed_ai_models(include_empty=False)
    ]
    print(f"AI Expert Models: {', '.join(model_names)} ({len(model_names)} models)")
    print(f"Execution Model: {ENSEMBLE_TRADER_NAME}")
    print(f"Default Virtual Balance: ${settings.initial_virtual_balance:,.0f}")
    inline_dashboard = bool(settings.dashboard_inline_enabled)
    if inline_dashboard:
        print(f"Dashboard: http://{settings.dashboard_host}:{settings.dashboard_port}")
    else:
        print("Dashboard: split process (Redis dashboard:update)")
    print("=" * 60)

    # Init database
    print("\n[1/5] Initializing database...")
    await init_db()
    await load_secure_settings_into_runtime()

    # Init services
    print("[2/5] Starting data service (OKX WebSocket)...")
    data_service = DataService()
    try:
        await data_service.start()
    except Exception as e:
        print(f"  WARNING: Data service start failed: {e}")
        print("  Continuing in offline mode (no live prices)...")

    # Register models
    print("[3/5] Registering AI models...")
    model_registry = ModelRegistry()
    for m in create_models_from_config():
        model_registry.register(m)
    await model_registry.initialize_all()

    # Competition & notifications
    competition_service = CompetitionService()
    competition_service.set_active_models([ENSEMBLE_TRADER_NAME])
    notification_service = NotificationService()

    redis = await create_redis_client()

    # Trading service
    print("[4/5] Starting trading service...")
    trading_service = TradingService(
        model_registry=model_registry,
        data_service=data_service,
        redis_client=redis,
    )
    await trading_service.initialize()

    # Initial model evaluation
    print("  Evaluating initial model rankings...")
    rankings = await competition_service.evaluate_all_models()
    if rankings:
        print(
            f"  Top model: {rankings[0]['model_name']} (score: {rankings[0].get('composite_score', 0):.4f})"
        )

    # Wire services to API
    set_services(trading_service, data_service, competition_service)

    # Start trading loop in background
    trading_task = asyncio.create_task(trading_service.start())

    # Periodic model evaluation
    async def periodic_evaluation():
        await asyncio.sleep(600)  # First eval after 10 minutes
        while trading_service._running:
            try:
                await competition_service.evaluate_all_models()
            except Exception as e:
                import structlog

                structlog.get_logger("paper_trading").error(
                    "periodic eval failed",
                    error=safe_error_text(e),
                )
            await asyncio.sleep(3600)

    eval_task = asyncio.create_task(periodic_evaluation())

    # Periodic WebSocket push for real-time dashboard updates
    async def periodic_ws_push():
        import structlog

        ws_log = structlog.get_logger("ws_push")
        await asyncio.sleep(2)  # Wait for dashboard to start
        snapshot_interval = 0
        while trading_service._running:
            try:
                market = data_service.get_market_state()
                tickers = market.get("tickers", {})

                # Update position prices so unrealized PnL stays current
                if trading_service.paper_executor:
                    for sym, ticker in tickers.items():
                        price = ticker.get("price", 0)
                        if price > 0:
                            await trading_service.paper_executor.update_market_prices(sym, price)

                # Record equity snapshot every 60s for PnL chart
                snapshot_interval += 2
                if snapshot_interval >= 60 and trading_service.paper_executor:
                    snapshot_interval = 0
                    await trading_service.record_equity_snapshot()

                open_symbols = await _get_open_position_symbols()
                tickers = await _build_tickers_for_open_positions(open_symbols, tickers)

                stats = trading_service.get_stats()
                await _send_dashboard_message(
                    redis,
                    inline_dashboard,
                    {
                        "type": "ticker_update",
                        "symbols": tickers,
                    }
                )
                await _send_dashboard_message(
                    redis,
                    inline_dashboard,
                    {
                        "type": "trading_round",
                        "decisions": stats.get("recent_decisions", []),
                        "executions": stats.get("recent_executions", []),
                        "stats": stats,
                    }
                )
            except Exception as e:
                ws_log.error("ws push failed", error=safe_error_text(e))
            await asyncio.sleep(2)

    ws_push_task = asyncio.create_task(periodic_ws_push())

    print("\n[5/5] Starting runtime...")
    try:
        if inline_dashboard:
            print(f"  Web dashboard on port {settings.dashboard_port}...")
            config = uvicorn.Config(
                app,
                host=settings.dashboard_host,
                port=settings.dashboard_port,
                log_level="info",
            )
            server = uvicorn.Server(config)
            await server.serve()
        else:
            print("  Trading engine only; dashboard runs in scripts/run_dashboard.py")
            await trading_task
    except KeyboardInterrupt:
        print("\nShutting down...")

    # Cleanup
    trading_service._running = False
    if not trading_task.done():
        trading_task.cancel()
    eval_task.cancel()
    ws_push_task.cancel()
    try:
        await trading_task
    except asyncio.CancelledError:
        pass
    try:
        await eval_task
    except asyncio.CancelledError:
        pass
    try:
        await ws_push_task
    except asyncio.CancelledError:
        pass

    await data_service.stop()
    await model_registry.shutdown_all()
    await notification_service.close()
    await close_db()
    print("Shutdown complete.")


if __name__ == "__main__":
    # Fix for Windows asyncio
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
