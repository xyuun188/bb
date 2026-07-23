"""Canonical start boundary for the current clean training epoch."""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config.settings import settings

TRAINING_EPOCH_VERSION = "2026-07-24.v1"
TRAINING_EPOCH_FILENAME = "training_epoch.json"
CURRENT_TRAINING_EPOCH_POLICY = "current_training_epoch_only"


def training_epoch_path(root: Path | None = None) -> Path:
    return (root or settings.data_dir) / TRAINING_EPOCH_FILENAME


def _parse_epoch(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def load_training_epoch_start(path: Path | None = None) -> datetime:
    """Return the current clean epoch and fail closed when it is not initialized."""

    marker_path = path or training_epoch_path()
    if not marker_path.exists():
        raise RuntimeError(
            "training epoch marker is missing; run reset_training_derived_state.py first"
        )
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"training epoch marker is unreadable: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("version") != TRAINING_EPOCH_VERSION:
        raise RuntimeError("training epoch marker version is unsupported")
    started_at = _parse_epoch(payload.get("epoch_started_at"))
    if started_at is None:
        raise RuntimeError("training epoch marker has no valid epoch_started_at")
    return started_at


def write_training_epoch(
    path: Path | None = None,
    *,
    started_at: datetime | None = None,
    reset_id: str,
) -> dict[str, Any]:
    """Atomically publish a new epoch marker after derived data is removed."""

    marker_path = path or training_epoch_path()
    current = (started_at or datetime.now(UTC)).astimezone(UTC)
    payload: dict[str, Any] = {
        "version": TRAINING_EPOCH_VERSION,
        "epoch_started_at": current.isoformat(),
        "reset_id": str(reset_id),
        "policy": "raw_exchange_facts_preserved_derived_training_state_rebuilt",
    }
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker_path.with_name(f".{marker_path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, marker_path)
    return payload
