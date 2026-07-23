#!/usr/bin/env python3
"""
Launch the live trading system against the explicit OKX live account.

Run: python scripts/run_live_trading.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn

from ai_brain.model_factory import create_models_from_config
from ai_brain.model_registry import ModelRegistry
from config.settings import settings
from core.logging_config import setup_logging
from core.redis_runtime import create_redis_client
from core.secret_utils import secret_state
from db.session import close_db, init_db
from services.data_service import DataService
from services.secure_runtime_config import load_secure_settings_into_runtime
from services.trading_service import TradingService
from web_dashboard.api.dashboard import set_services
from web_dashboard.app import app


async def main():
    setup_logging()
    print("=" * 60)
    print("AI CRYPTO TRADING SYSTEM - LIVE TRADING MODE")
    print("OKX account: live")
    print("=" * 60)

    print("\n*** WARNING: LIVE TRADING MODE (REAL FUNDS) ***")
    print(f"OKX API Key: {secret_state(settings.okx_live_api_key)}")
    confirm = await asyncio.to_thread(input, "Type 'YES' to confirm live trading: ")
    if confirm != "YES":
        print("Aborted.")
        return

    # Init
    await init_db()
    await load_secure_settings_into_runtime()
    print("Database initialized.")

    data_service = DataService()
    try:
        await data_service.start()
    except Exception as e:
        print(f"WARNING: Data service start failed: {e}")

    # Models
    model_registry = ModelRegistry()
    for m in create_models_from_config():
        model_registry.register(m)
    await model_registry.initialize_all()

    redis = await create_redis_client()

    trading_service = TradingService(
        model_registry=model_registry,
        data_service=data_service,
        redis_client=redis,
    )
    await trading_service.initialize()
    set_services(trading_service, data_service)

    # Start trading
    trading_task = asyncio.create_task(trading_service.start())

    inline_dashboard = bool(settings.dashboard_inline_enabled)
    if inline_dashboard:
        print(f"\nDashboard: http://{settings.dashboard_host}:{settings.dashboard_port}")
    else:
        print("\nDashboard: split process (Redis dashboard:update)")

    try:
        if inline_dashboard:
            config = uvicorn.Config(
                app,
                host=settings.dashboard_host,
                port=settings.dashboard_port,
                log_level="info",
            )
            server = uvicorn.Server(config)
            await server.serve()
        else:
            await trading_task
    except KeyboardInterrupt:
        print("\nShutting down...")

    trading_service._running = False
    if not trading_task.done():
        trading_task.cancel()
    try:
        await trading_task
    except asyncio.CancelledError:
        pass
    await data_service.stop()
    await model_registry.shutdown_all()
    await close_db()
    print("Shutdown complete.")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
