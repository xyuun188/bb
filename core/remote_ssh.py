"""Shared SSH connection helpers for remote model/server scripts.

The helper keeps credential parsing and host-key policy in one place. Unknown
SSH hosts are rejected; add the server fingerprint to system known_hosts or
``<project>/.ssh/known_hosts`` before running deployment scripts.
"""

from __future__ import annotations

import codecs
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.remote_server_info import RemoteServerInfo, load_remote_server_info
from core.safe_output import format_command_failure

DEFAULT_SSH_TIMEOUT_SECONDS = 20
DEFAULT_SSH_KEEPALIVE_SECONDS = 30
DEFAULT_REMOTE_COMMAND_TIMEOUT_SECONDS = 180
DEFAULT_REMOTE_OUTPUT_TEXT_LIMIT = 20_000
DEFAULT_REMOTE_COMMAND_CHANNEL_OPEN_RETRIES = 2
REMOTE_COMMAND_CHANNEL_RETRY_DELAY_SECONDS = 0.35
REMOTE_STREAM_READ_CHUNK_BYTES = 8192
TRANSIENT_CHANNEL_OPEN_ERROR_MARKERS = (
    "timeout opening channel",
    "ssh session not active",
)


@dataclass(frozen=True)
class RemoteCommandResult:
    """Decoded output from a remote SSH command."""

    status: int
    stdout: str
    stderr: str
    stdout_truncated: bool = False
    stderr_truncated: bool = False

    @property
    def combined_output(self) -> str:
        """Return stdout followed by stderr for human diagnostics."""
        return self.stdout + self.stderr


def configure_ssh_host_keys(ssh: Any, project_root: Path) -> None:
    """Load trusted host keys and reject unknown SSH hosts."""
    import paramiko

    ssh.load_system_host_keys()
    project_known_hosts = project_root / ".ssh" / "known_hosts"
    if project_known_hosts.exists():
        ssh.load_host_keys(str(project_known_hosts))
    ssh.set_missing_host_key_policy(paramiko.RejectPolicy())


def _enable_ssh_keepalive(ssh: Any, interval_seconds: int) -> None:
    """Keep idle SSH transports alive for long-running deploy and probe sessions."""
    try:
        interval = int(interval_seconds)
    except (TypeError, ValueError):
        interval = DEFAULT_SSH_KEEPALIVE_SECONDS
    if interval <= 0:
        return

    get_transport = getattr(ssh, "get_transport", None)
    if get_transport is None:
        return
    transport = get_transport()
    if transport is not None:
        transport.set_keepalive(interval)


def _remember_reconnect_metadata(
    ssh: Any,
    project_root: Path,
    connect_kwargs: dict[str, Any],
    keepalive_interval: int,
) -> None:
    ssh._bb_project_root = project_root
    ssh._bb_connect_kwargs = dict(connect_kwargs)
    ssh._bb_keepalive_interval = keepalive_interval


def connect_remote_ssh(
    project_root: Path,
    *,
    timeout: int = DEFAULT_SSH_TIMEOUT_SECONDS,
    banner_timeout: int | None = None,
    auth_timeout: int | None = None,
    keepalive_interval: int = DEFAULT_SSH_KEEPALIVE_SECONDS,
    info: RemoteServerInfo | None = None,
) -> Any:
    """Create a strict SSH connection from the local ignored server-info file."""
    import paramiko

    server_info = info or load_remote_server_info(project_root)
    ssh = paramiko.SSHClient()
    configure_ssh_host_keys(ssh, project_root)
    connect_kwargs: dict[str, Any] = {
        "hostname": server_info.host,
        "port": server_info.port,
        "username": server_info.username,
        "password": server_info.password,
        "timeout": timeout,
    }
    if banner_timeout is not None:
        connect_kwargs["banner_timeout"] = banner_timeout
    if auth_timeout is not None:
        connect_kwargs["auth_timeout"] = auth_timeout
    try:
        ssh.connect(**connect_kwargs)
        _enable_ssh_keepalive(ssh, keepalive_interval)
        _remember_reconnect_metadata(
            ssh,
            project_root,
            connect_kwargs,
            keepalive_interval,
        )
    except Exception:
        ssh.close()
        raise
    return ssh


def _normalize_output_limit(max_output_chars: int | None) -> int:
    try:
        limit = int(max_output_chars or 0)
    except (TypeError, ValueError):
        limit = DEFAULT_REMOTE_OUTPUT_TEXT_LIMIT
    if limit <= 0:
        return DEFAULT_REMOTE_OUTPUT_TEXT_LIMIT
    return min(limit, DEFAULT_REMOTE_OUTPUT_TEXT_LIMIT)


def _read_remote_stream(
    stream: Any,
    *,
    max_output_chars: int = DEFAULT_REMOTE_OUTPUT_TEXT_LIMIT,
) -> tuple[str, bool]:
    """Read and drain a Paramiko stream while retaining a bounded text prefix."""
    limit = _normalize_output_limit(max_output_chars)
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    parts: list[str] = []
    retained_chars = 0
    truncated = False

    while True:
        try:
            chunk = stream.read(REMOTE_STREAM_READ_CHUNK_BYTES)
        except TypeError:
            chunk = stream.read()
        if not chunk:
            break
        if isinstance(chunk, str):
            text = chunk
        else:
            text = decoder.decode(chunk, final=False)
        if retained_chars >= limit:
            truncated = True
            continue
        remaining = limit - retained_chars
        if len(text) > remaining:
            parts.append(text[:remaining])
            retained_chars = limit
            truncated = True
        else:
            parts.append(text)
            retained_chars += len(text)

    tail = decoder.decode(b"", final=True)
    if tail:
        if retained_chars < limit:
            remaining = limit - retained_chars
            parts.append(tail[:remaining])
            retained_chars += min(len(tail), remaining)
            truncated = truncated or len(tail) > remaining
        else:
            truncated = True

    output = "".join(parts)
    if truncated:
        output += f"\n...[remote stream truncated at {limit} characters]..."
    return output, truncated


def _is_transient_channel_open_error(exc: BaseException) -> bool:
    """Return whether Paramiko failed before the remote command was started."""
    try:
        import paramiko
    except Exception:
        return False

    ssh_exception = getattr(paramiko, "SSHException", None)
    if ssh_exception is None or not isinstance(exc, ssh_exception):
        return False

    message = str(exc).lower()
    return any(marker in message for marker in TRANSIENT_CHANNEL_OPEN_ERROR_MARKERS)


def _reconnect_ssh_client(ssh: Any) -> bool:
    """Reconnect a known SSH client after a channel-open failure."""
    project_root = getattr(ssh, "_bb_project_root", None)
    connect_kwargs = getattr(ssh, "_bb_connect_kwargs", None)
    keepalive_interval = getattr(
        ssh,
        "_bb_keepalive_interval",
        DEFAULT_SSH_KEEPALIVE_SECONDS,
    )
    if not isinstance(project_root, Path) or not isinstance(connect_kwargs, dict):
        return False

    ssh.close()
    configure_ssh_host_keys(ssh, project_root)
    ssh.connect(**connect_kwargs)
    _enable_ssh_keepalive(ssh, keepalive_interval)
    return True


def _command_channel_attempts(channel_open_retries: int) -> int:
    try:
        retries = int(channel_open_retries)
    except (TypeError, ValueError):
        retries = DEFAULT_REMOTE_COMMAND_CHANNEL_OPEN_RETRIES
    capped_retries = max(0, min(retries, DEFAULT_REMOTE_COMMAND_CHANNEL_OPEN_RETRIES))
    return capped_retries + 1


def exec_remote_command(
    ssh: Any,
    command: str,
    *,
    timeout: int = DEFAULT_REMOTE_COMMAND_TIMEOUT_SECONDS,
    max_output_chars: int = DEFAULT_REMOTE_OUTPUT_TEXT_LIMIT,
    channel_open_retries: int = DEFAULT_REMOTE_COMMAND_CHANNEL_OPEN_RETRIES,
) -> RemoteCommandResult:
    """Execute a remote command and return decoded output without raising."""
    attempts = _command_channel_attempts(channel_open_retries)
    for attempt_index in range(attempts):
        try:
            _stdin, stdout, stderr = ssh.exec_command(command, timeout=timeout)
            break
        except Exception as exc:
            is_last_attempt = attempt_index >= attempts - 1
            if is_last_attempt or not _is_transient_channel_open_error(exc):
                raise
            _reconnect_ssh_client(ssh)
            time.sleep(REMOTE_COMMAND_CHANNEL_RETRY_DELAY_SECONDS)
    out, out_truncated = _read_remote_stream(stdout, max_output_chars=max_output_chars)
    err, err_truncated = _read_remote_stream(stderr, max_output_chars=max_output_chars)
    status = stdout.channel.recv_exit_status()
    return RemoteCommandResult(
        status=status,
        stdout=out,
        stderr=err,
        stdout_truncated=out_truncated,
        stderr_truncated=err_truncated,
    )


def run_remote_text(
    ssh: Any,
    command: str,
    *,
    timeout: int = DEFAULT_REMOTE_COMMAND_TIMEOUT_SECONDS,
    check: bool = True,
    max_output_chars: int = DEFAULT_REMOTE_OUTPUT_TEXT_LIMIT,
    channel_open_retries: int = DEFAULT_REMOTE_COMMAND_CHANNEL_OPEN_RETRIES,
) -> str:
    """Execute a remote command and return combined output.

    When ``check`` is enabled, non-zero exits raise a redacted RuntimeError.
    """
    result = exec_remote_command(
        ssh,
        command,
        timeout=timeout,
        max_output_chars=max_output_chars,
        channel_open_retries=channel_open_retries,
    )
    if check and result.status != 0:
        raise RuntimeError(
            format_command_failure(result.status, command, result.stdout, result.stderr)
        )
    output = result.combined_output
    safe_limit = max(0, int(max_output_chars or 0))
    if safe_limit and len(output) > safe_limit:
        return output[:safe_limit] + f"\n...[remote output truncated at {safe_limit} characters]..."
    return output
