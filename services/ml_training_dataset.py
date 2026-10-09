"""Authoritative, resource-bounded dataset access for local ML training."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import and_, func, or_, select

from db.session import get_read_session_ctx
from models.learning import ShadowBacktest
from models.trade import OkxPositionHistory
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


async def probe_authoritative_trade_training_cursor() -> dict[str, Any]:
    """Detect settlement/link repairs without loading decision or raw payloads."""

    epoch_start = load_training_data_start()
    filters = (
        or_(
            OkxPositionHistory.updated_at_okx >= epoch_start,
            and_(
                OkxPositionHistory.updated_at_okx.is_(None),
                OkxPositionHistory.opened_at >= epoch_start,
            ),
        ),
    )
    columns = (
        OkxPositionHistory.id,
        OkxPositionHistory.row_identity,
        OkxPositionHistory.mode,
        OkxPositionHistory.opened_at,
        OkxPositionHistory.updated_at_okx,
        OkxPositionHistory.close_status,
        OkxPositionHistory.realized_pnl,
        OkxPositionHistory.pnl,
        OkxPositionHistory.pnl_ratio,
        OkxPositionHistory.fee,
        OkxPositionHistory.funding_fee,
        OkxPositionHistory.open_avg_px,
        OkxPositionHistory.close_avg_px,
        OkxPositionHistory.open_max_pos,
        OkxPositionHistory.close_total_pos,
        OkxPositionHistory.entry_order_ids,
        OkxPositionHistory.close_order_ids,
        OkxPositionHistory.linked_order_ids,
        OkxPositionHistory.position_ids,
        OkxPositionHistory.match_status,
        OkxPositionHistory.evidence_gaps,
    )
    digest = hashlib.sha256(epoch_start.isoformat().encode("utf-8"))
    count = 0
    async with get_read_session_ctx() as session:
        # Match the bounded lifecycle window used by the compact outcome loader.
        result = await session.stream(
            select(*columns)
            .where(*filters)
            .order_by(
                OkxPositionHistory.updated_at_okx.desc().nullslast(),
                OkxPositionHistory.opened_at.desc().nullslast(),
                OkxPositionHistory.id.desc(),
            )
            .limit(2000)
        )
        async for mapping in result.mappings():
            digest.update(
                json.dumps(
                    dict(mapping), sort_keys=True, default=str, separators=(",", ":")
                ).encode("utf-8")
            )
            count += 1
    return {
        "history_row_count": count,
        "available": True,
        "source": "bounded_okx_settlement_fact_projection",
        "fingerprint": digest.hexdigest(),
    }


def authoritative_trade_training_cursor(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Fingerprint the canonical clean outcomes, including repaired and losing facts."""

    identities = sorted(
        (
            str(sample.get("lifecycle_key") or sample.get("id") or ""),
            str(sample.get("closed_at") or sample.get("label_timestamp") or ""),
            json.dumps(
                {
                    "labels": sample.get("labels"),
                    "target": sample.get("profit_training_contract"),
                    "outcome_fingerprint": sample.get("outcome_fingerprint"),
                    "profit_supervision": sample.get("profit_supervision"),
                    "realized_pnl": sample.get("realized_pnl"),
                    "net_return": sample.get("net_return_after_all_cost_pct"),
                    "quality": sample.get("data_quality_status"),
                },
                sort_keys=True,
                default=str,
                separators=(",", ":"),
            ),
        )
        for sample in samples
    )
    return {
        "sample_count": len(samples),
        "fingerprint": hashlib.sha256(
            json.dumps(identities, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
