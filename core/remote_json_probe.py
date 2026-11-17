"""Bounded, read-only JSON probes with cancellation-safe SSH ownership."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from core.remote_ssh import _is_transient_channel_open_error, _is_transient_connect_error
from core.safe_output import safe_error_text


async def collect_remote_json_snapshot(
    project_root: Path,
    info: Any,
    command: str,
    *,
    ssh_connector: Callable[..., Any],
    command_executor: Callable[..., Any],
    timeout_seconds: float,
    max_output_chars: int,
) -> dict[str, Any]:
    budget = max(float(timeout_seconds), 1.0)
    deadline = time.monotonic() + budget
    cancelled = threading.Event()
    ownership_lock = threading.Lock()
    active: dict[str, Any] = {}

    def collect() -> dict[str, Any]:
        for attempt in range(2):
            remaining = deadline - time.monotonic()
            if cancelled.is_set() or remaining <= 0:
                raise TimeoutError("Remote probe expired before SSH connection.")
            connect_timeout = min(8.0, remaining / 3.0)
            ssh = None
            opening_channel = False
            try:
                ssh = ssh_connector(
                    project_root,
                    timeout=connect_timeout,
                    banner_timeout=connect_timeout,
                    auth_timeout=connect_timeout,
                    connect_retries=0,
                    info=info,
                )
                with ownership_lock:
                    if cancelled.is_set():
                        raise TimeoutError("Remote probe expired during SSH connection.")
                    active["ssh"] = ssh
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Remote probe expired before command execution.")
                opening_channel = True
                result = command_executor(
                    ssh,
                    command,
                    timeout=remaining,
                    channel_open_retries=0,
                    max_output_chars=max_output_chars,
                )
                break
            except Exception as exc:
                retryable = (
                    _is_transient_channel_open_error(exc)
                    if opening_channel
                    else _is_transient_connect_error(exc)
                )
                if (
                    attempt > 0
                    or not retryable
                    or cancelled.is_set()
                    or deadline - time.monotonic() <= 0.35
                ):
                    raise
            finally:
                with ownership_lock:
                    active.pop("ssh", None)
                if ssh is not None:
                    ssh.close()
            # A new connection may recover a failed handshake/channel open, but
            # never restart a command after it has begun returning a result.
            if cancelled.wait(0.35):
                raise TimeoutError("Remote probe expired before SSH retry.")
        if result.status != 0:
            raise RuntimeError(
                safe_error_text(result.stderr or result.stdout or "Remote probe failed.")
            )
        try:
            payload = json.loads(str(result.stdout or "{}"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("Remote probe returned invalid JSON.") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Remote probe payload was not an object.")
        return payload

    try:
        return await asyncio.wait_for(asyncio.to_thread(collect), timeout=budget)
    finally:
        cancelled.set()
        with ownership_lock:
            ssh = active.pop("ssh", None)
        if ssh is not None:
            ssh.close()
