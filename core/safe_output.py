"""Terminal output helpers that redact secrets before rendering."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from typing import Any, TextIO

from core.secret_utils import redact_text

DEFAULT_ERROR_TEXT_LIMIT = 700
DEFAULT_COMMAND_TEXT_LIMIT = 1000
DEFAULT_COMMAND_STREAM_LIMIT = 4000


def redact_output(value: Any) -> str:
    """Return text safe for terminal, CI, and script diagnostics."""
    if isinstance(value, bytes):
        text = value.decode("utf-8", "replace")
    elif value is None:
        text = ""
    else:
        text = str(value)
    return redact_text(text)


def safe_error_text(
    value: Any,
    *,
    limit: int = DEFAULT_ERROR_TEXT_LIMIT,
    fallback: str | None = None,
) -> str:
    """Return a redacted, bounded error string suitable for logs and APIs."""
    if fallback is None:
        fallback = type(value).__name__ if value is not None else ""

    text = redact_output(value).strip() or fallback
    safe_limit = max(0, int(limit or 0))
    if safe_limit and len(text) > safe_limit:
        return f"{text[:safe_limit]}..."
    return text


def bounded_redacted_text(value: Any, *, limit: int) -> str:
    """Return redacted text capped to a deterministic character budget."""
    text = redact_output(value).strip()
    safe_limit = max(0, int(limit or 0))
    if safe_limit and len(text) > safe_limit:
        return f"{text[:safe_limit]}..."
    return text


def safe_response_error_text(
    response: Any,
    *,
    limit: int = DEFAULT_ERROR_TEXT_LIMIT,
) -> str:
    """Return a safe excerpt from an HTTP response error body."""
    try:
        parsed = response.json()
        if isinstance(parsed, Mapping):
            body = json.dumps(dict(parsed), ensure_ascii=False)
        else:
            body = json.dumps(parsed, ensure_ascii=False)
    except ValueError:
        body = getattr(response, "text", "")
    return safe_error_text(body, limit=limit)


def safe_print(
    *values: Any,
    sep: str = " ",
    end: str = "\n",
    file: TextIO | None = None,
    flush: bool = False,
) -> None:
    """Print values after redacting recognizable secrets."""
    stream = file or sys.stdout
    stream.write(sep.join(redact_output(value) for value in values))
    stream.write(end)
    if flush:
        stream.flush()


def format_command_failure(
    status: int,
    command: str,
    stdout: Any = "",
    stderr: Any = "",
    *,
    command_limit: int = DEFAULT_COMMAND_TEXT_LIMIT,
    stream_limit: int = DEFAULT_COMMAND_STREAM_LIMIT,
) -> str:
    """Format a redacted remote-command failure message."""
    safe_command = bounded_redacted_text(command, limit=command_limit)
    sections = [f"command failed ({status}): {safe_command}"]
    out = bounded_redacted_text(stdout, limit=stream_limit)
    err = bounded_redacted_text(stderr, limit=stream_limit)
    if out:
        sections.append(f"STDOUT:\n{out}")
    if err:
        sections.append(f"STDERR:\n{err}")
    return "\n".join(sections)
