"""Validation helpers for persisted HTTP service base URLs."""

from __future__ import annotations

import re
from urllib.parse import ParseResult, urlparse

_HTTP_SCHEMES = {"http", "https"}
_ENCODED_CONTROL_RE = re.compile(r"(?i)%0[0-9a-f]|%7f")


def _validate_common_http_url_parts(
    url: str,
    parsed: ParseResult,
    *,
    field_name: str,
) -> None:
    if any(ord(char) < 32 or ord(char) == 127 for char in url):
        raise ValueError(f"{field_name} must not contain control characters.")
    if any(char.isspace() for char in url):
        raise ValueError(f"{field_name} must not contain whitespace.")
    if _ENCODED_CONTROL_RE.search(url):
        raise ValueError(f"{field_name} must not contain encoded control characters.")
    if "\\" in url:
        raise ValueError(f"{field_name} must not contain backslashes.")

    if parsed.scheme not in _HTTP_SCHEMES or not parsed.netloc or not parsed.hostname:
        raise ValueError(f"{field_name} must be an absolute http(s) URL.")
    if "@" in parsed.netloc or parsed.username or parsed.password:
        raise ValueError(
            f"{field_name} must not include credentials; configure API keys separately."
        )
    if any(char.isspace() for char in parsed.hostname):
        raise ValueError(f"{field_name} host must not contain whitespace.")

    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{field_name} port must be between 1 and 65535.") from exc
    if port is not None and port < 1:
        raise ValueError(f"{field_name} port must be between 1 and 65535.")
    if _has_empty_port(parsed.netloc):
        raise ValueError(f"{field_name} port must not be empty.")


def _has_empty_port(netloc: str) -> bool:
    host_port = netloc.rsplit("@", 1)[-1]
    if host_port.startswith("["):
        closing = host_port.find("]")
        return closing >= 0 and host_port[closing + 1 :] == ":"
    return ":" in host_port and host_port.rsplit(":", 1)[1] == ""


def normalize_http_base_url(
    value: str | None,
    *,
    field_name: str = "HTTP service base URL",
    allow_empty: bool = False,
) -> str:
    """Return a normalized http(s) base URL safe to persist or call."""
    base = str(value or "").strip().rstrip("/")
    if not base:
        if allow_empty:
            return ""
        raise ValueError(f"{field_name} is required.")

    parsed = urlparse(base)
    _validate_common_http_url_parts(base, parsed, field_name=field_name)
    if parsed.query or parsed.fragment:
        raise ValueError(f"{field_name} must not include query strings or fragments.")

    return base


def normalize_external_http_url(
    value: str | None,
    *,
    field_name: str = "External HTTP URL",
    allow_empty: bool = True,
    max_length: int = 2048,
) -> str:
    """Return a safe absolute http(s) URL for outbound links.

    Unlike service base URLs, public article links may legitimately contain
    query strings or fragments, but credentials and script-like schemes are
    never valid for rendered dashboard links.
    """
    url = str(value or "").strip()
    if not url:
        if allow_empty:
            return ""
        raise ValueError(f"{field_name} is required.")
    if max_length > 0 and len(url) > max_length:
        raise ValueError(f"{field_name} is too long.")
    parsed = urlparse(url)
    _validate_common_http_url_parts(url, parsed, field_name=field_name)
    return url


def normalize_https_webhook_url(
    value: str | None,
    *,
    field_name: str = "Webhook URL",
    allow_empty: bool = True,
    max_length: int = 2048,
) -> str:
    """Return a safe HTTPS webhook URL with query strings allowed."""
    url = normalize_external_http_url(
        value,
        field_name=field_name,
        allow_empty=allow_empty,
        max_length=max_length,
    )
    if not url:
        return ""
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"{field_name} must use https.")
    if parsed.fragment:
        raise ValueError(f"{field_name} must not include fragments.")
    return url
