from __future__ import annotations

import asyncio

import pytest

import db.session as session_module
from config.settings import settings


def test_session_gate_capacity_is_bounded_below_raw_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "database_pool_size", 16)
    monkeypatch.setattr(settings, "database_max_overflow", 24)
    monkeypatch.setattr(settings, "database_session_concurrency", 24)

    assert session_module._session_gate_capacity() == 24


@pytest.mark.asyncio
async def test_session_gate_serializes_bursts_without_leaking_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "database_session_concurrency", 1)
    previous_gate = session_module._session_gate
    previous_limit = session_module._session_gate_limit
    session_module._session_gate = None
    session_module._session_gate_limit = None
    entered: list[int] = []
    release = asyncio.Event()

    async def worker(index: int) -> None:
        gate = await session_module._acquire_session_slot()
        try:
            entered.append(index)
            await release.wait()
        finally:
            gate.release()

    try:
        first = asyncio.create_task(worker(1))
        await asyncio.sleep(0)
        second = asyncio.create_task(worker(2))
        await asyncio.sleep(0)
        assert entered == [1]
        release.set()
        await asyncio.gather(first, second)
        assert entered == [1, 2]
    finally:
        session_module._session_gate = previous_gate
        session_module._session_gate_limit = previous_limit
