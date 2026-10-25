"""Summarize bounded, redacted OKX public-market latency log tails."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

LATENCY_EVENT = "OKX public market request latency"
MAX_LOG_FILES = 6
MAX_BYTES_PER_LOG = 2 * 1024 * 1024
MAX_WINDOW_MINUTES = 240


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _event_payload(line: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(line)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("event") == LATENCY_EVENT:
        return payload
    for key in ("message", "MESSAGE"):
        message = payload.get(key)
        if not isinstance(message, str):
            continue
        try:
            nested = json.loads(message)
        except ValueError:
            continue
        if isinstance(nested, dict) and nested.get("event") == LATENCY_EVENT:
            return nested
    return None


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number >= 0 and number < float("inf") else None


def _metric(values: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "p50": None, "p95": None, "max": None}

    def percentile(fraction: float) -> float:
        index = min(round((len(ordered) - 1) * fraction), len(ordered) - 1)
        return round(ordered[index], 1)

    return {
        "count": len(ordered),
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "max": round(ordered[-1], 1),
    }


def summarize_public_market_latency_logs(
    log_tails: Iterable[tuple[str, bytes]],
    *,
    since: datetime,
    until: datetime,
) -> dict[str, Any]:
    """Aggregate latency events without returning raw log lines or payloads."""

    if since.tzinfo is None or until.tzinfo is None or since >= until:
        raise ValueError("latency audit window must be ordered and timezone-aware")
    start = since.astimezone(UTC)
    end = until.astimezone(UTC)
    accumulators: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "queue_wait_ms": [],
            "provider_duration_ms": [],
            "outcomes": Counter(),
            "priorities": Counter(),
        }
    )
    files = []
    matching_events = 0
    invalid_event_timestamps = 0
    truncated_files = 0

    for path, content in list(log_tails)[:MAX_LOG_FILES]:
        bounded = bytes(content[-MAX_BYTES_PER_LOG:])
        truncated = len(content) > len(bounded)
        truncated_files += int(truncated)
        lines = bounded.splitlines()
        if truncated and lines:
            lines = lines[1:]
        files.append(
            {
                "name": str(path).rsplit("/", 1)[-1][:32],
                "bytes_read": len(bounded),
                "tail_truncated": truncated,
                "complete_lines_read": len(lines),
            }
        )
        for raw_line in lines:
            payload = _event_payload(raw_line.decode("utf-8", errors="replace"))
            if payload is None:
                continue
            event_at = _timestamp(payload.get("timestamp"))
            if event_at is None:
                invalid_event_timestamps += 1
                continue
            if event_at < start or event_at >= end:
                continue
            method = payload.get("method")
            if not isinstance(method, str) or not method:
                continue
            matching_events += 1
            item = accumulators[method[:80]]
            for field in ("queue_wait_ms", "provider_duration_ms"):
                number = _finite_number(payload.get(field))
                if number is not None:
                    item[field].append(number)
            outcome = payload.get("outcome")
            if isinstance(outcome, str):
                item["outcomes"][outcome[:32]] += 1
            item["priorities"][str(payload.get("priority"))[:16]] += 1

    routes = {}
    for method, item in sorted(accumulators.items()):
        routes[method] = {
            "queue_wait_ms": _metric(item["queue_wait_ms"]),
            "provider_duration_ms": _metric(item["provider_duration_ms"]),
            "outcomes": dict(item["outcomes"]),
            "priorities": dict(item["priorities"]),
        }
    return {
        "window": {
            "since": start.isoformat(),
            "until_exclusive": end.isoformat(),
            "diagnostic_event_only": True,
            "latency_threshold_ms": 250.0,
        },
        "bounded_log_tails": {
            "max_files": MAX_LOG_FILES,
            "max_bytes_per_file": MAX_BYTES_PER_LOG,
            "truncated_file_count": truncated_files,
            "files": files,
        },
        "matching_latency_event_count": matching_events,
        "invalid_event_timestamp_count": invalid_event_timestamps,
        "routes": routes,
    }
