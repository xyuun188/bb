from __future__ import annotations

import asyncio
import signal
from types import SimpleNamespace

import pytest

from scripts import run_paper_trading


@pytest.mark.asyncio
async def test_headless_sigterm_finalizes_trading_before_closing_database(monkeypatch):
    calls = []
    loop = asyncio.get_running_loop()

    def add_signal_handler(signum, callback):
        assert signum == signal.SIGTERM
        loop.call_soon(callback)

    async def noop(*_args, **_kwargs):
        pass

    async def stop_data():
        calls.append("data_stopped")

    async def close_db():
        calls.append("db_closed")

    class TradingService:
        _running = True

        def __init__(self, **_kwargs):
            pass

        async def start(self):
            try:
                await asyncio.Event().wait()
            finally:
                assert not self._running
                calls.append("trading_loop_finalized")

        async def stop(self):
            calls.append("trading_resources_closed")

    monkeypatch.setattr(loop, "add_signal_handler", add_signal_handler)
    monkeypatch.setattr(loop, "remove_signal_handler", lambda signum: signum == signal.SIGTERM)
    monkeypatch.setattr(run_paper_trading.settings, "dashboard_inline_enabled", False)
    monkeypatch.setattr(run_paper_trading, "acquire_single_instance_lock", lambda: None)
    monkeypatch.setattr(run_paper_trading, "setup_logging", lambda: None)
    monkeypatch.setattr(run_paper_trading, "init_db", noop)
    monkeypatch.setattr(run_paper_trading, "load_secure_settings_into_runtime", noop)
    monkeypatch.setattr(run_paper_trading, "create_redis_client", noop)
    monkeypatch.setattr(run_paper_trading, "create_models_from_config", list)
    monkeypatch.setattr(run_paper_trading, "set_services", lambda *_args: None)
    monkeypatch.setattr(run_paper_trading, "repair_shadow_training_snapshots", noop)
    monkeypatch.setattr(
        run_paper_trading, "DataService", lambda: SimpleNamespace(start=noop, stop=stop_data)
    )
    monkeypatch.setattr(
        run_paper_trading, "ModelRegistry", lambda: SimpleNamespace(initialize_all=noop)
    )
    monkeypatch.setattr(run_paper_trading, "TradingService", TradingService)
    monkeypatch.setattr(run_paper_trading, "close_db", close_db)

    await asyncio.wait_for(run_paper_trading.main(), timeout=2.0)

    assert calls == [
        "trading_loop_finalized",
        "trading_resources_closed",
        "data_stopped",
        "db_closed",
    ]


@pytest.mark.asyncio
async def test_headless_runner_propagates_trading_failure(monkeypatch):
    loop = asyncio.get_running_loop()
    removed = []
    monkeypatch.setattr(loop, "add_signal_handler", lambda *_args: None)
    monkeypatch.setattr(loop, "remove_signal_handler", removed.append)

    async def fail():
        raise RuntimeError("trading startup failed")

    with pytest.raises(RuntimeError, match="trading startup failed"):
        await run_paper_trading._wait_for_headless_runtime(asyncio.create_task(fail()))

    assert removed == [signal.SIGTERM]
