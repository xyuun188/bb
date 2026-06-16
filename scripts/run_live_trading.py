#!/usr/bin/env python3
"""
Launch the live trading system (OKX demo or production).
Only the best-performing model from paper trading executes real orders.

Run: python scripts/run_live_trading.py
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structlog
import uvicorn

from ai_brain.model_factory import create_models_from_config
from ai_brain.model_registry import ModelRegistry
from config.settings import settings
from core.logging_config import setup_logging
from core.redis_runtime import create_redis_client
from core.safe_output import safe_error_text
from core.secret_utils import secret_state
from core.trading_mode import mode_manager
from db.session import close_db, init_db
from services.competition_service import CompetitionService
from services.data_service import DataService
from services.secure_runtime_config import load_secure_settings_into_runtime
from services.trading_service import TradingService
from web_dashboard.api.dashboard import set_services
from web_dashboard.app import app

logger = structlog.get_logger("live_trading")


async def main():
    setup_logging()
    print("=" * 60)
    print("AI CRYPTO TRADING SYSTEM - LIVE TRADING MODE")
    print(f"OKX Demo: {settings.okx_demo}")
    print("=" * 60)

    # Confirm before proceeding
    if not settings.okx_demo:
        print("\n*** WARNING: LIVE TRADING MODE (REAL FUNDS) ***")
        print(f"OKX API Key: {secret_state(settings.okx_api_key)}")
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

    # Competition
    competition_service = CompetitionService()
    rankings = await competition_service.evaluate_all_models()

    # Select best model for live trading
    if rankings:
        best_model = rankings[0]["model_name"]
        print(f"\nBest model from paper trading: {best_model}")
        print(f"  PnL: {rankings[0]['pnl_pct']:.2f}%")
        print(f"  Sharpe: {rankings[0]['sharpe_ratio']:.2f}")
        print(f"  Win Rate: {rankings[0]['win_rate']:.2f}%")

        mode_manager._live_model_name = best_model
        await mode_manager.switch_to_live(best_model)
    else:
        print("\nNo paper trading data found. Using LLM Agent as default.")
        mode_manager._live_model_name = "llm_agent"
        await mode_manager.switch_to_live("llm_agent")

    redis = await create_redis_client()

    trading_service = TradingService(
        model_registry=model_registry,
        data_service=data_service,
        redis_client=redis,
    )
    await trading_service.initialize()
    set_services(trading_service, data_service, competition_service)

    # Start trading
    trading_task = asyncio.create_task(trading_service.start())

    # Periodic model evaluation
    async def periodic_evaluation():
        await asyncio.sleep(600)
        while trading_service._running:
            try:
                await competition_service.evaluate_all_models()
            except Exception as e:
                logger.error("periodic eval failed", error=safe_error_text(e))
            await asyncio.sleep(3600)

    eval_task = asyncio.create_task(periodic_evaluation())

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
    eval_task.cancel()
    try:
        await trading_task
    except asyncio.CancelledError:
        pass
    try:
        await eval_task
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
