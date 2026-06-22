"""Automatic quarantine for shadow samples that must not train ML models.

Raw shadow backtest rows are part of the audit trail and should not be
physically deleted.  This module turns data-quality exclusions into a durable
row state (`status='quarantined'`) so local ML, server tools, and dashboards can
all consume the same clean training view.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select

from db.session import get_session_ctx
from models.learning import ShadowBacktest
from services.trading_params import DEFAULT_TRADING_PARAMS
from services.training_data_quality import SampleQualityAssessment, assess_shadow_sample

QUARANTINE_STATUS = "quarantined"
TRAINING_QUARANTINE_MARKER = "[training_quarantine]"
_LOCAL_ML_PARAMS = DEFAULT_TRADING_PARAMS.local_ml_training


def _safe_feature_snapshot(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def shadow_quality_sample(row: Any) -> dict[str, Any]:
    """Build the canonical data-quality payload from a shadow row."""

    return {
        "symbol": getattr(row, "symbol", ""),
        "analysis_type": getattr(row, "analysis_type", ""),
        "decision_action": getattr(row, "decision_action", ""),
        "decision_confidence": getattr(row, "decision_confidence", 0.0),
        "horizon_minutes": int(getattr(row, "horizon_minutes", 0) or 0),
        "features": _safe_feature_snapshot(getattr(row, "feature_snapshot", None)),
        "long_return_pct": getattr(row, "long_return_pct", None),
        "short_return_pct": getattr(row, "short_return_pct", None),
        "label_timestamp": getattr(row, "due_at", None),
        "best_action": getattr(row, "best_action", ""),
        "missed_opportunity": bool(getattr(row, "missed_opportunity", False)),
    }


def assess_shadow_row(row: Any) -> SampleQualityAssessment:
    """Assess whether a shadow row can safely enter training."""

    return assess_shadow_sample(shadow_quality_sample(row))


def note_with_quarantine_reason(note: str | None, reasons: tuple[str, ...]) -> str:
    """Append an idempotent quarantine marker to an existing note."""

    existing = str(note or "").strip()
    reason_text = ",".join(reasons) if reasons else "unknown"
    if TRAINING_QUARANTINE_MARKER in existing:
        return existing
    suffix = f"{TRAINING_QUARANTINE_MARKER} {reason_text}"
    return f"{existing}\n{suffix}" if existing else suffix


def quarantine_completed_shadow_row(
    row: Any,
    *,
    dry_run: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Quarantine one completed shadow row when quality rules exclude it."""

    assessment = assess_shadow_row(row)
    if not assessment.exclude_from_training:
        return {
            "applied": False,
            "status": assessment.status,
            "score": round(assessment.score, 4),
            "reasons": list(assessment.reasons),
        }

    if not dry_run:
        row.status = QUARANTINE_STATUS
        row.note = note_with_quarantine_reason(
            getattr(row, "note", ""),
            assessment.reasons,
        )
        row.updated_at = now or datetime.now(UTC)

    return {
        "applied": True,
        "status": assessment.status,
        "score": round(assessment.score, 4),
        "reasons": list(assessment.reasons),
    }


async def quarantine_dirty_shadow_samples(
    *,
    batch_size: int | None = None,
    max_batches: int | None = None,
    dry_run: bool = False,
    newest_first: bool = True,
    only_newer_than_id: int | None = None,
) -> dict[str, Any]:
    """Scan completed shadow samples and quarantine dirty training rows.

    The default order scans the newest rows first because model training also
    consumes the latest window.  Operators can pass ``newest_first=False`` for
    full historical cleanup from the oldest rows.
    """

    size = max(int(batch_size or _LOCAL_ML_PARAMS.auto_quarantine_batch_size), 1)
    batches = max(int(max_batches or _LOCAL_ML_PARAMS.auto_quarantine_max_batches), 1)
    scanned = 0
    quarantined = 0
    cursor_id: int | None = None
    reason_counts: Counter[str] = Counter()

    async with get_session_ctx() as session:
        for _batch_index in range(batches):
            stmt = select(ShadowBacktest).where(
                ShadowBacktest.status == "completed",
                ShadowBacktest.long_return_pct.is_not(None),
                ShadowBacktest.short_return_pct.is_not(None),
            )
            if only_newer_than_id is not None:
                stmt = stmt.where(ShadowBacktest.id > int(only_newer_than_id))
            if cursor_id is not None:
                if newest_first:
                    stmt = stmt.where(ShadowBacktest.id < cursor_id)
                else:
                    stmt = stmt.where(ShadowBacktest.id > cursor_id)
            stmt = stmt.order_by(
                ShadowBacktest.id.desc() if newest_first else ShadowBacktest.id.asc()
            ).limit(size)

            result = await session.execute(stmt)
            rows = list(result.scalars().all())
            if not rows:
                break

            for row in rows:
                row_id = int(getattr(row, "id", 0) or 0)
                cursor_id = (
                    row_id
                    if cursor_id is None
                    else (min(cursor_id, row_id) if newest_first else max(cursor_id, row_id))
                )
                scanned += 1
                outcome = quarantine_completed_shadow_row(row, dry_run=dry_run)
                if not outcome["applied"]:
                    continue
                quarantined += 1
                reason_counts.update(outcome.get("reasons") or ["unknown"])

            if not dry_run:
                await session.flush()

    return {
        "dry_run": dry_run,
        "newest_first": newest_first,
        "batch_size": size,
        "max_batches": batches,
        "only_newer_than_id": only_newer_than_id,
        "scanned": scanned,
        "quarantined": quarantined,
        "last_scanned_id": cursor_id,
        "top_reasons": [
            {"reason": reason, "count": count} for reason, count in reason_counts.most_common(20)
        ],
    }
