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
TRAINING_DATA_MIGRATION_VERSION = "2026-09-14.v1"
TRAINING_DATA_MIGRATION_FILENAME = "training_data_migration.json"
CURRENT_TRAINING_EPOCH_POLICY = "current_epoch_plus_approved_historical_rebuild"


def training_epoch_path(root: Path | None = None) -> Path:
    return (root or settings.data_dir) / TRAINING_EPOCH_FILENAME


def training_data_migration_path(root: Path | None = None) -> Path:
    return (root or settings.data_dir) / TRAINING_DATA_MIGRATION_FILENAME


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


def load_training_epoch(path: Path | None = None) -> dict[str, Any]:
    """Return the validated clean-epoch marker."""

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
    reset_id = str(payload.get("reset_id") or "").strip()
    if not reset_id:
        raise RuntimeError("training epoch marker has no reset_id")
    return {**payload, "epoch_started_at": started_at, "reset_id": reset_id}


def load_training_epoch_start(path: Path | None = None) -> datetime:
    """Return the current clean epoch and fail closed when it is not initialized."""

    return load_training_epoch(path)["epoch_started_at"]


def load_training_data_migration(
    path: Path | None = None,
    *,
    epoch_path: Path | None = None,
) -> dict[str, Any] | None:
    """Return an approved historical-data migration for the active epoch.

    A stale, malformed, or mismatched manifest never broadens the training
    window. The caller falls back to the clean epoch instead.
    """

    marker_path = path or training_data_migration_path()
    if not marker_path.exists():
        return None
    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    epoch = load_training_epoch(epoch_path)
    data_start = _parse_epoch(payload.get("training_data_started_at"))
    source_fingerprint = str(payload.get("source_fact_fingerprint") or "").strip().lower()
    quality_report_sha256 = str(payload.get("quality_report_sha256") or "").strip().lower()
    try:
        approved_counts = {
            str(key): max(int(value or 0), 0)
            for key, value in dict(payload.get("approved_sample_counts") or {}).items()
        }
        approved_total = int(payload.get("approved_sample_count_total") or 0)
    except (TypeError, ValueError):
        return None
    valid_hash_chars = set("0123456789abcdef")
    data_start_invalid = data_start is None or data_start >= epoch["epoch_started_at"]
    if any(
        (
            payload.get("version") != TRAINING_DATA_MIGRATION_VERSION,
            str(payload.get("reset_id") or "") != epoch["reset_id"],
            payload.get("status") != "approved",
            data_start_invalid,
            len(source_fingerprint) != 64,
            len(quality_report_sha256) != 64,
            not set(source_fingerprint) <= valid_hash_chars,
            not set(quality_report_sha256) <= valid_hash_chars,
            approved_total <= 0,
            approved_total != sum(approved_counts.values()),
            payload.get("live_routing_enabled") is not False,
        )
    ):
        return None
    return {
        **payload,
        "training_data_started_at": data_start,
        "approved_sample_counts": approved_counts,
        "approved_sample_count_total": approved_total,
    }


def training_data_scope(
    *,
    epoch_path: Path | None = None,
    migration_path: Path | None = None,
) -> dict[str, Any]:
    epoch = load_training_epoch(epoch_path)
    migration = load_training_data_migration(migration_path, epoch_path=epoch_path)
    data_start = (
        migration["training_data_started_at"]
        if migration is not None
        else epoch["epoch_started_at"]
    )
    return {
        "training_policy": CURRENT_TRAINING_EPOCH_POLICY,
        "training_epoch_started_at": epoch["epoch_started_at"].isoformat(),
        "training_epoch_reset_id": epoch["reset_id"],
        "training_data_started_at": data_start.isoformat(),
        "pre_epoch_data_training_allowed": migration is not None,
        "historical_migration_status": "approved" if migration is not None else "absent",
        "historical_migration_manifest": str(
            migration_path or training_data_migration_path()
        ),
        "approved_sample_counts": (
            dict(migration["approved_sample_counts"]) if migration is not None else {}
        ),
        "approved_sample_count_total": (
            int(migration["approved_sample_count_total"]) if migration is not None else 0
        ),
    }


def load_training_data_start(
    *,
    epoch_path: Path | None = None,
    migration_path: Path | None = None,
) -> datetime:
    scope = training_data_scope(epoch_path=epoch_path, migration_path=migration_path)
    parsed = _parse_epoch(scope["training_data_started_at"])
    if parsed is None:
        raise RuntimeError("training data scope has no valid start")
    return parsed


def write_training_data_migration(
    payload: dict[str, Any],
    path: Path | None = None,
    *,
    epoch_path: Path | None = None,
) -> dict[str, Any]:
    """Atomically publish a validated, non-live historical rebuild manifest."""

    epoch = load_training_epoch(epoch_path)
    marker_path = path or training_data_migration_path()
    value = {
        **payload,
        "version": TRAINING_DATA_MIGRATION_VERSION,
        "reset_id": epoch["reset_id"],
        "status": "approved",
        "live_routing_enabled": False,
    }
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker_path.with_name(f".{marker_path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, marker_path)
    if load_training_data_migration(marker_path, epoch_path=epoch_path) is None:
        marker_path.unlink(missing_ok=True)
        raise RuntimeError("training data migration manifest failed validation")
    return value


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
