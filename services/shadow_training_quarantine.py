"""Automatic quarantine for shadow samples that must not train ML models.

Raw shadow backtest rows are part of the audit trail and should not be
physically deleted.  This module turns data-quality exclusions into a durable
row state (`status='quarantined'`) so local ML, server tools, and dashboards can
all consume the same clean training view.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.orm import load_only

from core.training_contracts import SHADOW_LABEL_VERSION
from db.session import get_session_ctx
from models.learning import ShadowBacktest
from services.trading_params import DEFAULT_TRADING_PARAMS
from services.training_data_quality import SampleQualityAssessment, assess_shadow_sample
from services.training_epoch import load_training_data_start

QUARANTINE_STATUS = "quarantined"
TRAINING_QUARANTINE_MARKER = "[training_quarantine]"
_LOCAL_ML_PARAMS = DEFAULT_TRADING_PARAMS.local_ml_training
_HISTORICAL_REQUALIFICATION_REASONS = {
    "market_data_quality:native_market_fact_invalid",
    "market_data_quality:orderbook_depth_invalid",
}
_TRAINING_QUARANTINE_REASON_PATTERN = re.compile(
    r"(?:^|\n)\[training_quarantine\][ \t]*([^\r\n]+)"
)


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

    compact_snapshot = getattr(row, "training_feature_snapshot", None)
    feature_snapshot = (
        compact_snapshot
        if compact_snapshot is not None
        else getattr(row, "feature_snapshot", None)
    )
    return {
        "id": int(getattr(row, "id", 0) or 0),
        "decision_id": int(getattr(row, "decision_id", 0) or 0) or None,
        "label_version": str(getattr(row, "label_version", "") or ""),
        "symbol": getattr(row, "symbol", ""),
        "analysis_type": getattr(row, "analysis_type", ""),
        "decision_action": getattr(row, "decision_action", ""),
        "decision_confidence": getattr(row, "decision_confidence", 0.0),
        "horizon_minutes": int(getattr(row, "horizon_minutes", 0) or 0),
        "features": _safe_feature_snapshot(feature_snapshot),
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
    training_start_override: datetime | None = None,
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
    newly_quarantined = 0
    already_quarantined = 0
    trainable = 0
    cursor_id: int | None = None
    reason_counts: Counter[str] = Counter()
    source_status_counts: Counter[str] = Counter()
    epoch_start = training_start_override or load_training_data_start()

    async with get_session_ctx() as session:
        audited_statuses = ("completed", "quarantined") if dry_run else ("completed",)
        total_candidate_count = int(
            (
                await session.execute(
                    select(func.count(ShadowBacktest.id)).where(
                        ShadowBacktest.status.in_(audited_statuses),
                        ShadowBacktest.created_at >= epoch_start,
                        ShadowBacktest.long_return_pct.is_not(None),
                        ShadowBacktest.short_return_pct.is_not(None),
                    )
                )
            ).scalar_one()
            or 0
        )
        for _batch_index in range(batches):
            stmt = select(ShadowBacktest).where(
                ShadowBacktest.status.in_(audited_statuses),
                ShadowBacktest.created_at >= epoch_start,
                ShadowBacktest.long_return_pct.is_not(None),
                ShadowBacktest.short_return_pct.is_not(None),
            ).options(
                load_only(
                    ShadowBacktest.id,
                    ShadowBacktest.decision_id,
                    ShadowBacktest.symbol,
                    ShadowBacktest.analysis_type,
                    ShadowBacktest.decision_action,
                    ShadowBacktest.decision_confidence,
                    ShadowBacktest.training_feature_snapshot,
                    ShadowBacktest.due_at,
                    ShadowBacktest.horizon_minutes,
                    ShadowBacktest.label_version,
                    ShadowBacktest.long_return_pct,
                    ShadowBacktest.short_return_pct,
                    ShadowBacktest.best_action,
                    ShadowBacktest.missed_opportunity,
                    ShadowBacktest.status,
                    ShadowBacktest.note,
                    ShadowBacktest.updated_at,
                )
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
                source_status = str(getattr(row, "status", "") or "unknown")
                source_status_counts[source_status] += 1
                outcome = quarantine_completed_shadow_row(row, dry_run=dry_run)
                if not outcome["applied"]:
                    trainable += 1
                    continue
                quarantined += 1
                if source_status == QUARANTINE_STATUS:
                    already_quarantined += 1
                else:
                    newly_quarantined += 1
                reason_counts.update(outcome.get("reasons") or ["unknown"])

            if not dry_run:
                await session.flush()

    return {
        "dry_run": dry_run,
        "newest_first": newest_first,
        "batch_size": size,
        "max_batches": batches,
        "only_newer_than_id": only_newer_than_id,
        "training_epoch_started_at": epoch_start.isoformat(),
        "scanned": scanned,
        "total_candidate_count": total_candidate_count,
        "coverage_complete": scanned >= total_candidate_count,
        "source_status_counts": dict(source_status_counts),
        "trainable": trainable,
        "quarantined": quarantined,
        "already_quarantined": already_quarantined,
        "new_quarantine_candidates": newly_quarantined,
        "last_scanned_id": cursor_id,
        "top_reasons": [
            {"reason": reason, "count": count} for reason, count in reason_counts.most_common(20)
        ],
    }


async def requalify_verified_historical_shadow_samples(
    *,
    batch_size: int = 250,
    max_batches: int = 20,
    dry_run: bool = True,
    only_newer_than_id: int | None = None,
    only_up_to_id: int | None = None,
) -> dict[str, Any]:
    """Restore only historical rows quarantined solely by obsolete quote quality."""

    size = max(int(batch_size or 250), 1)
    batches = max(int(max_batches or 20), 1)
    scanned = eligible = requalified = 0
    cursor_id = int(only_newer_than_id) if only_newer_than_id is not None else None
    candidate_ids: list[int] = []
    skipped_reasons: Counter[str] = Counter()
    coverage_complete = False
    epoch_start = load_training_data_start()
    quarantine_reason_filter = or_(
        *(
            ShadowBacktest.note.like(f"%{TRAINING_QUARANTINE_MARKER} {reason}%")
            for reason in _HISTORICAL_REQUALIFICATION_REASONS
        )
    )

    async with get_session_ctx() as session:
        for _batch_index in range(batches):
            stmt = (
                select(ShadowBacktest)
                .where(
                    ShadowBacktest.status == QUARANTINE_STATUS,
                    ShadowBacktest.created_at >= epoch_start,
                    ShadowBacktest.execution_mode == "paper",
                    ShadowBacktest.analysis_type == "market",
                    ShadowBacktest.label_version == SHADOW_LABEL_VERSION,
                    ShadowBacktest.long_return_pct.is_not(None),
                    ShadowBacktest.short_return_pct.is_not(None),
                    ShadowBacktest.training_feature_snapshot_version >= 1,
                    ShadowBacktest.training_feature_snapshot.is_not(None),
                    ShadowBacktest.training_feature_snapshot[
                        "historical_market_path_only"
                    ].as_boolean().is_(True),
                    quarantine_reason_filter,
                )
                .options(
                    load_only(
                        ShadowBacktest.id,
                        ShadowBacktest.training_feature_snapshot,
                        ShadowBacktest.training_feature_snapshot_version,
                        ShadowBacktest.label_version,
                        ShadowBacktest.decision_id,
                        ShadowBacktest.symbol,
                        ShadowBacktest.analysis_type,
                        ShadowBacktest.decision_action,
                        ShadowBacktest.decision_confidence,
                        ShadowBacktest.horizon_minutes,
                        ShadowBacktest.long_return_pct,
                        ShadowBacktest.short_return_pct,
                        ShadowBacktest.due_at,
                        ShadowBacktest.best_action,
                        ShadowBacktest.missed_opportunity,
                        ShadowBacktest.note,
                        ShadowBacktest.status,
                        ShadowBacktest.updated_at,
                    )
                )
                .order_by(ShadowBacktest.id.asc())
            )
            if cursor_id is not None:
                stmt = stmt.where(ShadowBacktest.id > cursor_id)
            if only_up_to_id is not None:
                stmt = stmt.where(ShadowBacktest.id <= int(only_up_to_id))
            stmt = stmt.limit(size)
            rows = list((await session.execute(stmt)).scalars().all())
            if not rows:
                coverage_complete = True
                break

            for row in rows:
                cursor_id = int(row.id)
                scanned += 1
                note = str(row.note or "")
                reasons = _TRAINING_QUARANTINE_REASON_PATTERN.findall(note)
                if (
                    len(reasons) != 1
                    or reasons[0].strip() not in _HISTORICAL_REQUALIFICATION_REASONS
                ):
                    skipped_reasons["quarantine_reason_not_solely_obsolete_quote_quality"] += 1
                    continue
                feature_snapshot = _safe_feature_snapshot(row.training_feature_snapshot)
                if feature_snapshot.get("historical_market_path_only") is not True:
                    skipped_reasons["not_historical_market_path_only"] += 1
                    continue

                assessment = assess_shadow_sample(
                    {
                        **shadow_quality_sample(row),
                        "features": feature_snapshot,
                    }
                )
                if (
                    assessment.status != "downweighted"
                    or assessment.reasons != ("historical_market_path_only",)
                ):
                    skipped_reasons.update(assessment.reasons or ("assessment_not_downweighted",))
                    continue
                eligible += 1
                if len(candidate_ids) < 50:
                    candidate_ids.append(int(row.id))
                if not dry_run:
                    row.status = "completed"
                    row.note = (
                        f"{note}\n[training_requalified] "
                        "verified_historical_path_after_market_quality_fix"
                    )
                    row.updated_at = datetime.now(UTC)
                    requalified += 1

            if not dry_run:
                await session.flush()
            if len(rows) < size:
                coverage_complete = True
                break

    return {
        "dry_run": dry_run,
        "batch_size": size,
        "max_batches": batches,
        "only_newer_than_id": only_newer_than_id,
        "only_up_to_id": only_up_to_id,
        "training_epoch_started_at": epoch_start.isoformat(),
        "scanned": scanned,
        "coverage_complete": coverage_complete,
        "eligible": eligible,
        "requalified": requalified,
        "last_scanned_id": cursor_id,
        "candidate_ids_sample": candidate_ids,
        "skipped_reasons": dict(skipped_reasons),
    }
