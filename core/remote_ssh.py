"""Shared SSH connection helpers for remote model/server scripts.

The helper keeps credential parsing and host-key policy in one place. Unknown
SSH hosts are rejected; add the server fingerprint to system known_hosts or
``<project>/.ssh/known_hosts`` before running deployment scripts.
"""

from __future__ import annotations

import codecs
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.remote_server_info import RemoteServerInfo, load_remote_server_info
from core.safe_output import format_command_failure

DEFAULT_SSH_TIMEOUT_SECONDS = 20
DEFAULT_REMOTE_COMMAND_TIMEOUT_SECONDS = 180
DEFAULT_REMOTE_OUTPUT_TEXT_LIMIT = 20_000
REMOTE_STREAM_READ_CHUNK_BYTES = 8192


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


def connect_remote_ssh(
    project_root: Path,
    *,
    timeout: int = DEFAULT_SSH_TIMEOUT_SECONDS,
    banner_timeout: int | None = None,
    auth_timeout: int | None = None,
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


def exec_remote_command(
    ssh: Any,
    command: str,
    *,
    timeout: int = DEFAULT_REMOTE_COMMAND_TIMEOUT_SECONDS,
    max_output_chars: int = DEFAULT_REMOTE_OUTPUT_TEXT_LIMIT,
) -> RemoteCommandResult:
    """Execute a remote command and return decoded output without raising."""
    _stdin, stdout, stderr = ssh.exec_command(command, timeout=timeout)
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
) -> str:
    """Execute a remote command and return combined output.

    When ``check`` is enabled, non-zero exits raise a redacted RuntimeError.
    """
    result = exec_remote_command(
        ssh,
        command,
        timeout=timeout,
        max_output_chars=max_output_chars,
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
