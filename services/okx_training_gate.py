from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config.settings import settings
from core.safe_output import safe_error_text

OKX_DAILY_RECONCILIATION_REPORT_REL_PATH = (
    "okx_daily_reconciliation_reports/latest.json"
)
OKX_DAILY_RECONCILIATION_REPORT_MAX_AGE_SECONDS = 36 * 3600


def _parse_utc_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _age_seconds(value: datetime | None, *, now: datetime | None = None) -> float | None:
    if value is None:
        return None
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    return max((current.astimezone(UTC) - value).total_seconds(), 0.0)


def okx_training_refresh_gate(
    *,
    data_dir: Path | None = None,
    now: datetime | None = None,
    max_age_seconds: int = OKX_DAILY_RECONCILIATION_REPORT_MAX_AGE_SECONDS,
) -> dict[str, Any]:
    """Read-only gate before training/reindexing can consume OKX trade facts."""

    root = data_dir or settings.data_dir
    path = root / OKX_DAILY_RECONCILIATION_REPORT_REL_PATH
    base = {
        "source": "okx_daily_reconciliation",
        "path": str(path),
        "max_age_seconds": int(max_age_seconds),
        "read_only": True,
        "mutates_database": False,
    }
    if not path.exists():
        return {
            **base,
            "allowed": False,
            "reason": "okx_daily_reconciliation_report_missing",
            "status": "missing",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            **base,
            "allowed": False,
            "reason": "okx_daily_reconciliation_report_read_failed",
            "status": "read_failed",
            "error": safe_error_text(exc, limit=180),
        }
    if not isinstance(payload, dict):
        return {
            **base,
            "allowed": False,
            "reason": "okx_daily_reconciliation_report_invalid",
            "status": "invalid",
        }

    generated_at = _parse_utc_datetime(payload.get("generated_at"))
    age = _age_seconds(generated_at, now=now)
    gates = _safe_dict(payload.get("operational_gates"))
    ledger = _safe_dict(payload.get("issue_ledger"))
    stale = (
        age is None
        or age > int(max_age_seconds)
        or bool(payload.get("artifact_error"))
    )
    can_refresh_training = bool(payload.get("can_refresh_training"))
    requires_attention = bool(payload.get("requires_attention"))
    allowed = bool(can_refresh_training and not requires_attention and not stale)
    reason = "okx_daily_reconciliation_allows_training_refresh"
    if stale:
        reason = "okx_daily_reconciliation_report_stale"
    elif requires_attention:
        reason = "okx_daily_reconciliation_requires_attention"
    elif not can_refresh_training:
        reason = "okx_daily_reconciliation_training_blocked"

    return {
        **base,
        "allowed": allowed,
        "reason": reason,
        "status": payload.get("status") or "unknown",
        "generated_at": generated_at.isoformat() if generated_at else None,
        "age_seconds": None if age is None else round(age, 3),
        "can_open_new_entries": bool(payload.get("can_open_new_entries")),
        "can_refresh_training": can_refresh_training,
        "requires_attention": requires_attention,
        "entry_blocked": bool(gates.get("entry_blocked")),
        "training_blocked": bool(gates.get("training_blocked")),
        "attention_buckets": _safe_dict(gates.get("attention_buckets")),
        "issue_ledger_summary": _safe_dict(ledger.get("summary")),
        "entry_blockers": _safe_list(gates.get("entry_blockers")),
        "training_blockers": _safe_list(gates.get("training_blockers")),
        "attention_items": _safe_list(gates.get("attention_items")),
    }
