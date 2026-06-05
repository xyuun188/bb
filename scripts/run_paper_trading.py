#!/usr/bin/env python3
"""
Launch the full paper trading system.
Starts: data feed, AI brain (all models), paper executor, and web dashboard.

Run: python scripts/run_paper_trading.py
Then open: http://localhost:8002
"""

import asyncio
import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn

from config.settings import ENSEMBLE_TRADER_NAME, settings
from core.logging_config import setup_logging
from core.trading_mode import mode_manager
from db.session import init_db, close_db
from web_dashboard.app import app, ws_manager
from web_dashboard.api.dashboard import (
    _build_tickers_for_open_positions,
    _get_open_position_symbols,
    set_services,
)

# Service imports
from ai_brain.model_registry import ModelRegistry
from ai_brain.model_factory import create_models_from_config
from services.data_service import DataService
from services.trading_service import TradingService
from services.competition_service import CompetitionService
from services.notification_service import NotificationService


LOCK_FILE = Path(__file__).resolve().parent.parent / "data" / "paper_trading.lock"
_lock_handle = None


def acquire_single_instance_lock() -> None:
    """Prevent two trading loops from running against the same account/db."""
    global _lock_handle
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    _lock_handle = open(LOCK_FILE, "a+", encoding="utf-8")
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(_lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(_lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(
            "Another paper trading instance is already running. "
            "Stop the existing process before starting a new one."
        )
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
    model_names = [m.get("name", "llm_agent") for m in settings.get_fixed_ai_models(include_empty=False)]
    print(f"AI Expert Models: {', '.join(model_names)} ({len(model_names)} models)")
    print(f"Execution Model: {ENSEMBLE_TRADER_NAME}")
    print(f"Default Virtual Balance: ${settings.initial_virtual_balance:,.0f}")
    print(f"Dashboard: http://{settings.dashboard_host}:{settings.dashboard_port}")
    print("=" * 60)

    # Init database
    print("\n[1/5] Initializing database...")
    await init_db()

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

    # Redis (fakeredis)
    redis = None
    try:
        import fakeredis.aioredis
        redis = await fakeredis.aioredis.create_redis_pool()
    except Exception:
        pass

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
        print(f"  Top model: {rankings[0]['model_name']} (score: {rankings[0].get('composite_score', 0):.4f})")

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
                structlog.get_logger("paper_trading").error("periodic eval failed", error=str(e))
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
                await ws_manager.broadcast({
                    "type": "ticker_update",
                    "symbols": tickers,
                })
                await ws_manager.broadcast({
                    "type": "trading_round",
                    "decisions": stats.get("recent_decisions", []),
                    "executions": stats.get("recent_executions", []),
                    "stats": stats,
                })
            except Exception as e:
                ws_log.error("ws push failed", error=str(e))
            await asyncio.sleep(2)

    from datetime import datetime, timezone
    ws_push_task = asyncio.create_task(periodic_ws_push())

    # Start dashboard
    print(f"\n[5/5] Starting web dashboard on port {settings.dashboard_port}...")
    config = uvicorn.Config(
        app,
        host=settings.dashboard_host,
        port=settings.dashboard_port,
        log_level="info",
    )
    server = uvicorn.Server(config)

    # Run server (this blocks until shutdown)
    try:
        await server.serve()
    except KeyboardInterrupt:
        print("\nShutting down...")

    # Cleanup
    trading_service._running = False
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
    await close_db()
    print("Shutdown complete.")


if __name__ == "__main__":
    # Fix for Windows asyncio
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
