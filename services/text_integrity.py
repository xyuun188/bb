"""Runtime text integrity checks for records that enter storage or audit views.

The dashboard already has display-level sanitizers for old damaged records. This
module provides a service-level contract: detect suspicious mojibake, attempt only
safe deterministic repair, and return a structured result that callers can audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from web_dashboard.api.text_sanitize import sanitize_text


def _u(escaped: str) -> str:
    return escaped.encode("ascii").decode("unicode_escape")


_REPLACEMENT_CHAR = _u("\\ufffd")


_COMMON_MOJIBAKE_MARKERS = (
    _u("\\u93c8"),
    _u("\\u7487"),
    _u("\\u52eb\\u5a09"),
    _u("\\u7d30"),
    _u("\\u57ce"),
    _u("\\u951b"),
    _u("\\u9286"),
    _u("\\u95ab"),
    _u("\\u95c8"),
    _u("\\u9422"),
    _u("\\u93c4"),
    _u("\\u93c3"),
    _u("\\u934f"),
    _u("\\u6d60"),
    _u("\\u5bee"),
    _u("\\u95ba"),
    _u("\\u5823"),
    _u("\\u6868"),
    _u("\\u7ef1"),
    _u("\\u626e"),
    _u("\\u62e0"),
    _u("\\u9355"),
    _u("\\ue0a2"),
    _u("\\u703b"),
    _u("\\u4ed3?"),
    _REPLACEMENT_CHAR,
)

_MIN_MARKER_HITS = 2


@dataclass(frozen=True)
class TextIntegrityResult:
    """Result of a runtime text integrity pass."""

    original: str
    text: str
    changed: bool
    suspected: bool
    method: str
    reason: str


def _marker_hit_count(text: str) -> int:
    return sum(text.count(marker) for marker in _COMMON_MOJIBAKE_MARKERS)


def _looks_like_utf8_decoded_as_cjk_legacy(text: str) -> bool:
    if not text:
        return False
    if _REPLACEMENT_CHAR in text:
        return True
    return _marker_hit_count(text) >= _MIN_MARKER_HITS


def looks_like_mojibake(text: str | None) -> bool:
    """Return whether text likely contains mojibake rather than valid content."""

    if not text:
        return False
    value = str(text)
    return _looks_like_utf8_decoded_as_cjk_legacy(value)


def _deterministic_redecode(text: str) -> str | None:
    """Reverse common UTF-8 bytes decoded as GBK/CP936 when confidence improves."""

    original_score = _marker_hit_count(text) + text.count(_REPLACEMENT_CHAR) * 3
    if original_score <= 0:
        return None
    best: str | None = None
    best_score = original_score
    for encoding in ("gbk", "cp936"):
        try:
            candidate = text.encode(encoding, errors="strict").decode("utf-8", errors="strict")
        except UnicodeError:
            continue
        candidate_score = _marker_hit_count(candidate) + candidate.count(_REPLACEMENT_CHAR) * 3
        if candidate_score < best_score:
            best = candidate
            best_score = candidate_score
    return best


def repair_mojibake(text: str | None) -> TextIntegrityResult:
    """Repair text only when a deterministic encoding reversal is safer."""

    original = "" if text is None else str(text)
    if not original:
        return TextIntegrityResult(
            original=original,
            text=original,
            changed=False,
            suspected=False,
            method="unchanged",
            reason="empty_text",
        )

    suspected = looks_like_mojibake(original)
    if not suspected:
        sanitized = sanitize_text(original)
        return TextIntegrityResult(
            original=original,
            text=str(sanitized),
            changed=sanitized != original,
            suspected=False,
            method="unchanged" if sanitized == original else "control_text_sanitize",
            reason="clean_text",
        )

    sanitized = sanitize_text(original)
    if isinstance(sanitized, str) and sanitized != original and not looks_like_mojibake(sanitized):
        return TextIntegrityResult(
            original=original,
            text=sanitized,
            changed=True,
            suspected=True,
            method="known_replacement",
            reason="matched_existing_dashboard_sanitizer",
        )

    repaired = _deterministic_redecode(original)
    if repaired is not None and repaired != original and not looks_like_mojibake(repaired):
        return TextIntegrityResult(
            original=original,
            text=repaired,
            changed=True,
            suspected=True,
            method="deterministic_redecode",
            reason="utf8_bytes_decoded_as_gbk_reversed",
        )

    return TextIntegrityResult(
        original=original,
        text=original,
        changed=False,
        suspected=True,
        method="unrepairable",
        reason="suspected_mojibake_without_safe_repair",
    )


def sanitize_runtime_text(value: Any) -> Any:
    """Recursively sanitize runtime payloads before storing or presenting them."""

    if isinstance(value, str):
        return repair_mojibake(value).text
    if isinstance(value, list):
        return [sanitize_runtime_text(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_runtime_text(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize_runtime_text(item) for key, item in value.items()}
    return value
