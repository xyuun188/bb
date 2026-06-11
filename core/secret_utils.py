"""Utilities for safe handling of secrets in logs, diagnostics, and UI payloads."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any

SENSITIVE_KEY_RE = re.compile(
    r"(api[_-]?key|secret|password|passphrase|token|authorization|access[_-]?key|"
    r"access[_-]?token|webhook)",
    re.IGNORECASE,
)
MASKED_SECRET_RE = re.compile(r"^\*{3,}[A-Za-z0-9_\-]{0,8}$")
URL_CREDENTIALS_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9+.-]*://)[^/\s:@]+(?::[^/\s@]*)?@")

SECRET_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(api\.telegram\.org/bot)[A-Za-z0-9:_\-]+"),
    re.compile(r"\b\d{6,}:[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9_\-\.=]{20,}"),
)
SECRET_KV_RE = re.compile(
    r"(?i)(['\"]?\b"
    r"(?:api[_-]?key|api[_-]?secret|secret|password|passphrase|token|authorization|"
    r"access[_-]?key|access[_-]?token|webhook)"
    r"\b['\"]?\s*[:=]\s*)(['\"]?)((?:bearer\s+)?[^'\"\s,;}]+)(['\"]?)"
)


def is_sensitive_key(key: str | None) -> bool:
    """Return True when a mapping key is likely to contain a secret."""
    return bool(SENSITIVE_KEY_RE.search(str(key or "")))


def mask_secret(value: Any, *, show_last: int = 0) -> str:
    """Mask a secret without exposing prefixes by default."""
    text = str(value or "")
    if not text:
        return ""
    if show_last <= 0:
        return "***"
    if len(text) <= show_last:
        return "***"
    return f"***{text[-show_last:]}"


def is_masked_secret(value: Any) -> bool:
    """Return True for UI/API secret placeholders such as *** or ****1234."""
    text = str(value or "").strip()
    return bool(text and MASKED_SECRET_RE.fullmatch(text))


def secret_state(value: Any) -> str:
    """Return a non-sensitive configured/missing label."""
    return "configured" if str(value or "").strip() else "missing"


def secret_fingerprint(value: Any, *, length: int = 12) -> str:
    """Return a stable non-reversible fingerprint for secret equality checks."""
    text = str(value or "")
    if not text:
        return ""
    safe_length = max(6, min(int(length or 12), 64))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:safe_length]


def redact_text(value: Any) -> str:
    """Redact recognizable secret values from free-form text."""
    text = str(value or "")
    text = URL_CREDENTIALS_RE.sub(r"\1***@", text)
    text = SECRET_KV_RE.sub(_redact_key_value_secret, text)
    for pattern in SECRET_VALUE_PATTERNS:
        text = pattern.sub(lambda match: f"{match.group(1)}***" if match.lastindex else "***", text)
    return text


def _redact_key_value_secret(match: re.Match[str]) -> str:
    return f"{match.group(1)}{match.group(2)}***{match.group(4)}"


def _redact_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return redact_mapping(value)
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(item) for item in value)
    if isinstance(value, set):
        return {_redact_value(item) for item in value}
    if isinstance(value, frozenset):
        return frozenset(_redact_value(item) for item in value)
    if isinstance(value, str):
        return redact_text(value)
    return value


def redact_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a shallow redacted copy of a mapping."""
    redacted: dict[str, Any] = {}
    for key, value in payload.items():
        if is_sensitive_key(str(key)):
            redacted[str(key)] = mask_secret(value)
        else:
            redacted[str(key)] = _redact_value(value)
    return redacted
