from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import paramiko
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


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["connect", "channel"])
async def test_probe_recovers_transient_ssh_failure_with_one_new_connection(failure_stage):
    connections = []
    closed = []
    executed = []

    def connector(_root, **kwargs):
        connections.append(kwargs)
        if failure_stage == "connect" and len(connections) == 1:
            raise paramiko.SSHException("No existing session")
        return SimpleNamespace(close=lambda: closed.append(True))

    def executor(_ssh, _command, **kwargs):
        executed.append(kwargs)
        if failure_stage == "channel" and len(executed) == 1:
            raise paramiko.SSHException("Timeout opening channel.")
        return SimpleNamespace(status=0, stdout='{"runtime_ready":true}', stderr="")

    result = await collect_remote_json_snapshot(
        Path("."), object(), "readonly-probe",
        ssh_connector=connector, command_executor=executor,
        timeout_seconds=2, max_output_chars=1000,
    )

    assert result == {"runtime_ready": True}
    assert len(connections) == 2
    assert len(closed) == (1 if failure_stage == "connect" else 2)
    assert connections[1]["timeout"] < connections[0]["timeout"]
    assert 0 < executed[-1]["timeout"] < 2


@pytest.mark.asyncio
async def test_probe_does_not_retry_persistent_ssh_failure_indefinitely():
    attempts = []

    def connector(_root, **_kwargs):
        attempts.append(True)
        raise paramiko.SSHException("Error reading SSH protocol banner")

    with pytest.raises(paramiko.SSHException, match="protocol banner"):
        await collect_remote_json_snapshot(
            Path("."), object(), "readonly-probe",
            ssh_connector=connector, command_executor=lambda *_args, **_kwargs: None,
            timeout_seconds=2, max_output_chars=1000,
        )
    assert len(attempts) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("status, stdout", [(1, "failed"), (0, "not-json")])
async def test_probe_does_not_retry_executed_command_or_invalid_result(status, stdout):
    attempts = []
    closed = []

    def executor(*_args, **_kwargs):
        attempts.append(True)
        return SimpleNamespace(status=status, stdout=stdout, stderr="")

    with pytest.raises(RuntimeError):
        await collect_remote_json_snapshot(
            Path("."), object(), "readonly-probe",
            ssh_connector=lambda *_args, **_kwargs: SimpleNamespace(
                close=lambda: closed.append(True)
            ),
            command_executor=executor,
            timeout_seconds=2, max_output_chars=1000,
        )
    assert attempts == [True]
    assert closed == [True]


@pytest.mark.asyncio
async def test_probe_cancellation_during_connect_prevents_retry():
    connecting = threading.Event()
    release_connect = threading.Event()
    worker_finished = threading.Event()
    attempts = []

    def connector(_root, **_kwargs):
        attempts.append(True)
        connecting.set()
        try:
            assert release_connect.wait(2)
            raise paramiko.SSHException("No existing session")
        finally:
            worker_finished.set()

    task = asyncio.create_task(collect_remote_json_snapshot(
        Path("."), object(), "readonly-probe",
        ssh_connector=connector, command_executor=lambda *_args, **_kwargs: None,
        timeout_seconds=2, max_output_chars=1000,
    ))
    assert await asyncio.to_thread(connecting.wait, 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release_connect.set()
    assert await asyncio.to_thread(worker_finished.wait, 1)
    assert attempts == [True]


@pytest.mark.asyncio
async def test_probe_does_not_retry_authentication_failure():
    attempts = []

    def connector(_root, **_kwargs):
        attempts.append(True)
        raise paramiko.AuthenticationException("Authentication failed.")

    with pytest.raises(paramiko.AuthenticationException):
        await collect_remote_json_snapshot(
            Path("."), object(), "readonly-probe",
            ssh_connector=connector, command_executor=lambda *_args, **_kwargs: None,
            timeout_seconds=2, max_output_chars=1000,
        )
    assert attempts == [True]


@pytest.mark.asyncio
async def test_probe_closes_late_connection_without_executing_after_cancellation():
    connecting = threading.Event()
    release_connect = threading.Event()
    closed = threading.Event()
    executed = []

    def connector(_root, **_kwargs):
        connecting.set()
        assert release_connect.wait(2)
        return SimpleNamespace(close=closed.set)

    task = asyncio.create_task(collect_remote_json_snapshot(
        Path("."), object(), "readonly-probe",
        ssh_connector=connector,
        command_executor=lambda *_args, **_kwargs: executed.append(True),
        timeout_seconds=2, max_output_chars=1000,
    ))
    assert await asyncio.to_thread(connecting.wait, 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release_connect.set()
    assert await asyncio.to_thread(closed.wait, 1)
    assert executed == []
