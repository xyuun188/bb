"""Shared marker contract for lossless runtime payload compaction."""

from __future__ import annotations

from typing import Any

RUNTIME_DATA_RETENTION_VERSION = "2026-07-18.runtime-data-retention.v1"
RUNTIME_DATA_RETENTION_SOURCE = "runtime_data_retention"
RUNTIME_DATA_RETENTION_APPLY_CONFIRMATION = "compact-runtime-data-v1"
RETENTION_MARKER_KEY = "_retention"
PRESERVE_AI_DECISION_PROJECTIONS_KEY = "preserve_ai_decision_projections"


def is_ai_decision_retention_payload(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    marker = value.get(RETENTION_MARKER_KEY)
    return bool(
        isinstance(marker, dict)
        and marker.get("version") == RUNTIME_DATA_RETENTION_VERSION
        and marker.get("source") == RUNTIME_DATA_RETENTION_SOURCE
        and marker.get(PRESERVE_AI_DECISION_PROJECTIONS_KEY) is True
    )
