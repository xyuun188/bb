#!/usr/bin/env python3
"""Sample the durable 24/72-hour acceptance windows outside Dashboard."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.session import close_db, init_db
from services.continuous_observation import ContinuousObservationScheduler
from services.secure_runtime_config import load_secure_settings_into_runtime
from web_dashboard.api.dashboard import (
    CONTINUOUS_OBSERVATION_STORES,
    collect_continuous_observation_metrics,
    shutdown_dashboard_read_clients,
)


async def main() -> None:
    if os.name == "posix":
        try:
            os.nice(10)
        except OSError:
            pass

    await init_db(migrate_schema=False)
    await load_secure_settings_into_runtime()
    scheduler = ContinuousObservationScheduler(
        CONTINUOUS_OBSERVATION_STORES,
        collect_continuous_observation_metrics,
        interval_seconds=300.0,
        startup_delay_seconds=300.0,
    )
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stop_event.set)
        except (NotImplementedError, RuntimeError):
            pass

    await scheduler.start()
    try:
        await stop_event.wait()
    finally:
        await scheduler.stop()
        await shutdown_dashboard_read_clients()
        await close_db()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
