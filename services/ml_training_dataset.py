"""Authoritative, resource-bounded dataset access for local ML training."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import and_, func, or_, select

from db.session import get_read_session_ctx
from models.learning import ShadowBacktest
from services.ml_training_contract import MIN_TRAINING_DECISION_GROUP_COUNT
from services.training_data_quality import annotate_samples, assess_shadow_sample
from services.training_epoch import load_training_data_start

LOCAL_ML_TRAINING_MAX_DECISION_GROUPS = max(
    int(os.environ.get("LOCAL_ML_TRAINING_MAX_DECISION_GROUPS", "5000")),
    MIN_TRAINING_DECISION_GROUP_COUNT,
    2,
)
LOCAL_ML_TRAINING_READ_PAGE_SIZE = 500


@dataclass(frozen=True)
class ShadowTrainingRow:
    id: int
    decision_id: int | None
    created_at: datetime | None
    symbol: str
    analysis_type: str
    decision_action: str
    decision_confidence: float
    feature_snapshot: Any
    due_at: datetime | None
    horizon_minutes: int
    label_version: str
    long_return_pct: float | None
    short_return_pct: float | None
    best_action: str | None
    missed_opportunity: bool


def _parse_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def shadow_training_columns() -> tuple[Any, ...]:
    return (
        ShadowBacktest.id,
        ShadowBacktest.decision_id,
        ShadowBacktest.created_at,
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
    )


def shadow_training_row_from_mapping(mapping: Any) -> ShadowTrainingRow:
    confidence = _safe_float(mapping.get("decision_confidence"), 0.0)
    return ShadowTrainingRow(
        id=int(mapping.get("id") or 0),
        decision_id=int(mapping.get("decision_id") or 0) or None,
        created_at=mapping.get("created_at"),
        symbol=str(mapping.get("symbol") or ""),
        analysis_type=str(mapping.get("analysis_type") or ""),
        decision_action=str(mapping.get("decision_action") or ""),
        decision_confidence=float(confidence or 0.0),
        feature_snapshot=_parse_json(mapping.get("training_feature_snapshot")),
        due_at=mapping.get("due_at"),
        horizon_minutes=int(mapping.get("horizon_minutes") or 10),
        label_version=str(mapping.get("label_version") or ""),
        long_return_pct=mapping.get("long_return_pct"),
        short_return_pct=mapping.get("short_return_pct"),
        best_action=mapping.get("best_action"),
        missed_opportunity=bool(mapping.get("missed_opportunity")),
    )


def shadow_sort_key(row: Any) -> tuple[datetime, int]:
    created_at = getattr(row, "created_at", None)
    if not isinstance(created_at, datetime):
        created_at = datetime.fromtimestamp(0, UTC)
    elif created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    return created_at.astimezone(UTC), int(getattr(row, "id", 0) or 0)


def shadow_action(row: Any, field: str) -> str:
    return str(getattr(row, field, "") or "").lower().strip()


def _shadow_decision_confidence(row: Any) -> float:
    return float(_safe_float(getattr(row, "decision_confidence", 0.0), 0.0) or 0.0)


def shadow_quality_sample(row: Any) -> dict[str, Any]:
    return {
        "id": int(getattr(row, "id", 0) or 0),
        "decision_id": int(getattr(row, "decision_id", 0) or 0) or None,
        "label_version": str(getattr(row, "label_version", "") or ""),
        "symbol": getattr(row, "symbol", ""),
        "analysis_type": getattr(row, "analysis_type", ""),
        "decision_action": getattr(row, "decision_action", ""),
        "decision_confidence": _shadow_decision_confidence(row),
        "horizon_minutes": int(getattr(row, "horizon_minutes", 10) or 10),
        "features": _parse_json(getattr(row, "feature_snapshot", None)),
        "long_return_pct": _safe_float(getattr(row, "long_return_pct", None), None),
        "short_return_pct": _safe_float(getattr(row, "short_return_pct", None), None),
        "label_timestamp": getattr(row, "due_at", None),
        "best_action": getattr(row, "best_action", ""),
        "missed_opportunity": bool(getattr(row, "missed_opportunity", False)),
    }


def shadow_is_trainable_trade_opportunity(row: Any) -> bool:
    action = shadow_action(row, "decision_action")
    best_action = shadow_action(row, "best_action")
    if action in {"long", "short"}:
        return not assess_shadow_sample(shadow_quality_sample(row)).exclude_from_training
    missed = bool(getattr(row, "missed_opportunity", False)) and best_action in {
        "long",
        "short",
    }
    if not missed:
        return False
    return not assess_shadow_sample(shadow_quality_sample(row)).exclude_from_training


def select_shadow_training_rows(rows: list[Any]) -> list[Any]:
    """Select the latest quality-governed chronological training window."""

    deduped: dict[Any, Any] = {}
    for row in rows:
        deduped.setdefault(getattr(row, "id", id(row)), row)
    recent = sorted(deduped.values(), key=shadow_sort_key, reverse=True)
    return [row for row in recent if shadow_is_trainable_trade_opportunity(row)]


def shadow_training_candidate_filters(epoch_start: datetime) -> tuple[Any, ...]:
    return (
        ShadowBacktest.status == "completed",
        ShadowBacktest.created_at >= epoch_start,
        ShadowBacktest.long_return_pct.is_not(None),
        ShadowBacktest.short_return_pct.is_not(None),
        or_(
            ShadowBacktest.decision_action.in_(["long", "short"]),
            and_(
                ShadowBacktest.missed_opportunity.is_(True),
                ShadowBacktest.best_action.in_(["long", "short"]),
            ),
        ),
    )


async def load_shadow_training_rows() -> list[Any]:
    """Load a time-stratified, resource-bounded clean training window."""

    epoch_start = load_training_data_start()
    base_filters = shadow_training_candidate_filters(epoch_start)
    columns = shadow_training_columns()
    async with get_read_session_ctx() as session:
        identity_result = await session.stream(
            select(ShadowBacktest.id, ShadowBacktest.decision_id)
            .where(*base_filters)
            .order_by(ShadowBacktest.created_at.desc(), ShadowBacktest.id.desc())
        )
        group_ids: dict[int, list[int]] = {}
        async for mapping in identity_result.mappings():
            sample_id = int(mapping.get("id") or 0)
            decision_id = int(mapping.get("decision_id") or 0)
            if sample_id > 0:
                group_ids.setdefault(decision_id or -sample_id, []).append(sample_id)

        groups = list(group_ids.values())
        if len(groups) > LOCAL_ML_TRAINING_MAX_DECISION_GROUPS:
            budget = LOCAL_ML_TRAINING_MAX_DECISION_GROUPS
            selected_group_indexes = {
                round(index * (len(groups) - 1) / (budget - 1))
                for index in range(budget)
            }
            groups = [groups[index] for index in sorted(selected_group_indexes)]
        selected_ids = [sample_id for group in groups for sample_id in group]

        rows: list[ShadowTrainingRow] = []
        for offset in range(0, len(selected_ids), LOCAL_ML_TRAINING_READ_PAGE_SIZE):
            page_ids = selected_ids[offset : offset + LOCAL_ML_TRAINING_READ_PAGE_SIZE]
            result = await session.execute(
                select(*columns).where(ShadowBacktest.id.in_(page_ids))
            )
            rows.extend(
                shadow_training_row_from_mapping(row) for row in result.mappings()
            )
    return select_shadow_training_rows(rows)


async def count_shadow_training_rows() -> int:
    epoch_start = load_training_data_start()
    async with get_read_session_ctx() as session:
        result = await session.execute(
            select(func.count(ShadowBacktest.id)).where(
                ShadowBacktest.status == "completed",
                ShadowBacktest.created_at >= epoch_start,
                ShadowBacktest.long_return_pct.is_not(None),
                ShadowBacktest.short_return_pct.is_not(None),
            )
        )
        return int(result.scalar() or 0)


async def count_shadow_training_decision_groups() -> int:
    epoch_start = load_training_data_start()
    async with get_read_session_ctx() as session:
        result = await session.execute(
            select(
                func.count(
                    func.distinct(func.coalesce(ShadowBacktest.decision_id, ShadowBacktest.id))
                )
            ).where(*shadow_training_candidate_filters(epoch_start))
        )
        return int(result.scalar() or 0)


async def load_authoritative_trade_training_samples() -> list[dict[str, Any]]:
    """Load the bounded clean OKX lifecycle projection for realized calibration."""

    from scripts.train_local_ai_tools_models import _load_trade_samples

    annotated = annotate_samples(await _load_trade_samples(compact=True), "trade")
    return [sample for sample in annotated if not sample.get("exclude_from_training")]
