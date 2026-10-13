"""Bounded, read-only JSON probes with cancellation-safe SSH ownership."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

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
        connect_timeout = min(8.0, budget / 3.0)
        ssh = ssh_connector(
            project_root,
            timeout=connect_timeout,
            banner_timeout=connect_timeout,
            auth_timeout=connect_timeout,
            connect_retries=0,
            info=info,
        )
        try:
            with ownership_lock:
                if cancelled.is_set():
                    raise TimeoutError("Remote probe expired during SSH connection.")
                active["ssh"] = ssh
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Remote probe expired before command execution.")
            result = command_executor(
                ssh,
                command,
                timeout=remaining,
                channel_open_retries=0,
                max_output_chars=max_output_chars,
            )
        finally:
            with ownership_lock:
                active.pop("ssh", None)
            ssh.close()
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
