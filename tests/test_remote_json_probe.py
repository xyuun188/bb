from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.remote_json_probe import collect_remote_json_snapshot


@pytest.mark.asyncio
async def test_probe_cancellation_closes_active_channel_and_releases_worker():
    command_started = threading.Event()
    connection_closed = threading.Event()
    worker_finished = threading.Event()
    kwargs_seen = {}

    class FakeSsh:
        def close(self):
            connection_closed.set()

    def connector(_root, **kwargs):
        kwargs_seen["connect"] = kwargs
        return FakeSsh()

    def executor(_ssh, _command, **kwargs):
        kwargs_seen["command"] = kwargs
        command_started.set()
        assert connection_closed.wait(3)
        worker_finished.set()
        raise OSError("channel closed")

    task = asyncio.create_task(collect_remote_json_snapshot(
        Path("."), object(), "readonly-probe",
        ssh_connector=connector, command_executor=executor,
        timeout_seconds=2, max_output_chars=1000,
    ))
    assert await asyncio.to_thread(command_started.wait, 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await asyncio.to_thread(worker_finished.wait, 1)
    assert kwargs_seen["connect"]["connect_retries"] == 0
    assert kwargs_seen["command"]["channel_open_retries"] == 0
    assert 0 < kwargs_seen["command"]["timeout"] <= 2


@pytest.mark.asyncio
async def test_probe_closes_connection_when_payload_is_not_an_object():
    closed = []
    with pytest.raises(RuntimeError, match="not an object"):
        await collect_remote_json_snapshot(
            Path("."), object(), "readonly-probe",
            ssh_connector=lambda *_args, **_kwargs: SimpleNamespace(
                close=lambda: closed.append(True)
            ),
            command_executor=lambda *_args, **_kwargs: SimpleNamespace(
                status=0, stdout="[]", stderr=""
            ),
            timeout_seconds=2, max_output_chars=1000,
        )
    assert closed == [True]
