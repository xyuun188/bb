"""Bound large runtime payloads while preserving auditable trading facts."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import bindparam, func, literal, null, or_, select

from config.settings import settings
from core.runtime_data_retention_contract import (
    PRESERVE_AI_DECISION_PROJECTIONS_KEY,
    RETENTION_MARKER_KEY,
    RUNTIME_DATA_RETENTION_SOURCE,
    RUNTIME_DATA_RETENTION_VERSION,
)
from db.session import get_read_session_ctx, get_session_ctx
from models.decision import AIDecision
from models.learning import ShadowBacktest, StrategyLearningEvent
from models.trade import Order
from services.phase3_boundary import PHASE3_CLEAN_START_UTC

ACTIVE_STRATEGY_EVENT_STATUSES = ("active", "pending", "running")


@dataclass(frozen=True, slots=True)
class RuntimeDataRetentionPolicy:
    decision_raw_days: int = 14
    shadow_grace_days: int = 2
    strategy_event_days: int = 14
    keep_trainable_shadow_rows: int = 20_000
    batch_size: int = 250
    max_rows_per_table: int = 5_000
    batch_pause_seconds: float = 0.05

    def normalized(self) -> RuntimeDataRetentionPolicy:
        return RuntimeDataRetentionPolicy(
            decision_raw_days=max(int(self.decision_raw_days or 14), 7),
            shadow_grace_days=max(int(self.shadow_grace_days or 2), 1),
            strategy_event_days=max(int(self.strategy_event_days or 14), 7),
            keep_trainable_shadow_rows=max(
                int(self.keep_trainable_shadow_rows or 20_000),
                1_000,
            ),
            batch_size=max(1, min(int(self.batch_size or 250), 1_000)),
            max_rows_per_table=max(
                1,
                min(int(self.max_rows_per_table or 5_000), 50_000),
            ),
            batch_pause_seconds=max(
                0.0,
                min(float(self.batch_pause_seconds or 0.0), 2.0),
            ),
        )


class RuntimeDataRetentionService:
    """Compact redundant JSON without deleting decisions or exchange facts."""

    def __init__(self, policy: RuntimeDataRetentionPolicy | None = None) -> None:
        self.policy = (policy or RuntimeDataRetentionPolicy()).normalized()

    async def run(
        self,
        *,
        apply: bool = False,
        measure_reclaimable_bytes: bool = True,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current = _as_utc(now or datetime.now(UTC))
        sections = {
            "ai_decisions": await self._compact_decisions(
                current,
                apply=apply,
                measure_reclaimable_bytes=measure_reclaimable_bytes,
            ),
            "shadow_backtests": await self._compact_shadows(
                current,
                apply=apply,
                measure_reclaimable_bytes=measure_reclaimable_bytes,
            ),
            "strategy_learning_events": await self._compact_strategy_events(
                current,
                apply=apply,
                measure_reclaimable_bytes=measure_reclaimable_bytes,
            ),
        }
        return {
            "version": RUNTIME_DATA_RETENTION_VERSION,
            "generated_at": current.isoformat(),
            "mode": "apply" if apply else "dry_run",
            "mutates_database": bool(apply),
            "deletes_rows": False,
            "preserves": [
                "orders",
                "positions",
                "okx_position_history",
                "decision_rows",
                "shadow_labels",
                "compact_training_and_learning_projections",
            ],
            "policy": asdict(self.policy),
            "sections": sections,
            "summary": {
                "eligible_rows": sum(
                    int(section.get("eligible_rows") or 0)
                    for section in sections.values()
                ),
                "processed_rows": sum(
                    int(section.get("processed_rows") or 0)
                    for section in sections.values()
                ),
                "reclaimable_bytes_measured": all(
                    section.get("reclaimable_bytes_measured") is True
                    for section in sections.values()
                ),
                "estimated_reclaimable_bytes": (
                    sum(
                        int(section.get("estimated_reclaimable_bytes") or 0)
                        for section in sections.values()
                    )
                    if all(
                        section.get("reclaimable_bytes_measured") is True
                        for section in sections.values()
                    )
                    else None
                ),
            },
        }

    async def _compact_decisions(
        self,
        now: datetime,
        *,
        apply: bool,
        measure_reclaimable_bytes: bool,
    ) -> dict[str, Any]:
        cutoff = now - timedelta(days=self.policy.decision_raw_days)
        conditions = _decision_retention_conditions(cutoff)
        eligible, reclaimable = await _candidate_totals(
            AIDecision.id,
            conditions,
            size_expression=_json_size(AIDecision.raw_llm_response)
            + _json_size(AIDecision.feature_snapshot),
            measure_reclaimable_bytes=measure_reclaimable_bytes,
        )
        if not apply:
            return _section_summary(
                eligible=eligible,
                processed=0,
                reclaimable=reclaimable,
                cutoff=cutoff,
                max_rows=self.policy.max_rows_per_table,
            )

        processed = 0
        while processed < self.policy.max_rows_per_table:
            limit = min(
                self.policy.batch_size,
                self.policy.max_rows_per_table - processed,
            )
            async with get_read_session_ctx() as session:
                rows = list(
                    (
                        await session.execute(
                            select(
                                AIDecision.id,
                                AIDecision.decision_learning_snapshot,
                                (
                                    _json_size(AIDecision.raw_llm_response)
                                    + _json_size(AIDecision.feature_snapshot)
                                ).label(
                                    "original_payload_bytes"
                                ),
                            )
                            .where(*conditions)
                            .order_by(AIDecision.id.asc())
                            .limit(limit)
                        )
                    )
                    .mappings()
                    .all()
                )
            if not rows:
                break
            compacted_at = now.isoformat()
            updates = []
            for row in rows:
                payload = copy.deepcopy(row.get("decision_learning_snapshot") or {})
                payload[RETENTION_MARKER_KEY] = {
                    "version": RUNTIME_DATA_RETENTION_VERSION,
                    "source": RUNTIME_DATA_RETENTION_SOURCE,
                    "compacted_at": compacted_at,
                    "original_payload_bytes": int(row.get("original_payload_bytes") or 0),
                    PRESERVE_AI_DECISION_PROJECTIONS_KEY: True,
                    "feature_snapshot_removed": True,
                    "row_preserved": True,
                }
                updates.append(
                    {
                        "target_id": int(row["id"]),
                        "compacted_payload": payload,
                    }
                )
            async with get_session_ctx() as session:
                await session.execute(
                    AIDecision.__table__.update()
                    .where(AIDecision.id == bindparam("target_id"))
                    .values(
                        raw_llm_response=bindparam("compacted_payload"),
                        feature_snapshot=null(),
                        runtime_payload_compaction_version=RUNTIME_DATA_RETENTION_VERSION,
                        runtime_payload_compacted_at=now,
                    ),
                    updates,
                )
            processed += len(updates)
            await _pause(self.policy.batch_pause_seconds)

        return _section_summary(
            eligible=eligible,
            processed=processed,
            reclaimable=reclaimable,
            cutoff=cutoff,
            max_rows=self.policy.max_rows_per_table,
        )

    async def _compact_shadows(
        self,
        now: datetime,
        *,
        apply: bool,
        measure_reclaimable_bytes: bool,
    ) -> dict[str, Any]:
        cutoff = now - timedelta(days=self.policy.shadow_grace_days)
        conditions = _shadow_retention_conditions(
            cutoff,
            keep_rows=self.policy.keep_trainable_shadow_rows,
        )
        eligible, reclaimable = await _candidate_totals(
            ShadowBacktest.id,
            conditions,
            size_expression=(
                _json_size(ShadowBacktest.raw_llm_response)
                + _json_size(ShadowBacktest.feature_snapshot)
                - _json_size(ShadowBacktest.training_feature_snapshot)
            ),
            measure_reclaimable_bytes=measure_reclaimable_bytes,
        )
        if not apply:
            return _section_summary(
                eligible=eligible,
                processed=0,
                reclaimable=reclaimable,
                cutoff=cutoff,
                max_rows=self.policy.max_rows_per_table,
            )

        processed = 0
        while processed < self.policy.max_rows_per_table:
            limit = min(
                self.policy.batch_size,
                self.policy.max_rows_per_table - processed,
            )
            async with get_read_session_ctx() as session:
                rows = list(
                    (
                        await session.execute(
                            select(
                                ShadowBacktest.id,
                                ShadowBacktest.training_feature_snapshot,
                                (
                                    _json_size(ShadowBacktest.raw_llm_response)
                                    + _json_size(ShadowBacktest.feature_snapshot)
                                ).label("original_payload_bytes"),
                            )
                            .where(*conditions)
                            .order_by(ShadowBacktest.id.asc())
                            .limit(limit)
                        )
                    )
                    .mappings()
                    .all()
                )
            if not rows:
                break
            compacted_at = now.isoformat()
            updates = []
            for row in rows:
                retention_payload = {
                    RETENTION_MARKER_KEY: {
                        "version": RUNTIME_DATA_RETENTION_VERSION,
                        "source": RUNTIME_DATA_RETENTION_SOURCE,
                        "compacted_at": compacted_at,
                        "original_payload_bytes": int(
                            row.get("original_payload_bytes") or 0
                        ),
                        "training_feature_snapshot_preserved": True,
                        "labels_preserved": True,
                        "row_preserved": True,
                    }
                }
                updates.append(
                    {
                        "target_id": int(row["id"]),
                        "training_payload": copy.deepcopy(
                            row.get("training_feature_snapshot") or {}
                        ),
                        "retention_payload": retention_payload,
                    }
                )
            async with get_session_ctx() as session:
                await session.execute(
                    ShadowBacktest.__table__.update()
                    .where(ShadowBacktest.id == bindparam("target_id"))
                    .values(
                        feature_snapshot=bindparam("training_payload"),
                        raw_llm_response=bindparam("retention_payload"),
                        runtime_payload_compaction_version=RUNTIME_DATA_RETENTION_VERSION,
                        runtime_payload_compacted_at=now,
                    ),
                    updates,
                )
            processed += len(updates)
            await _pause(self.policy.batch_pause_seconds)

        return _section_summary(
            eligible=eligible,
            processed=processed,
            reclaimable=reclaimable,
            cutoff=cutoff,
            max_rows=self.policy.max_rows_per_table,
        )

    async def _compact_strategy_events(
        self,
        now: datetime,
        *,
        apply: bool,
        measure_reclaimable_bytes: bool,
    ) -> dict[str, Any]:
        cutoff = now - timedelta(days=self.policy.strategy_event_days)
        conditions = _strategy_event_retention_conditions(cutoff)
        size_expression = sum(
            (
                _json_size(column)
                for column in (
                    StrategyLearningEvent.strategy_snapshot,
                    StrategyLearningEvent.market_state,
                    StrategyLearningEvent.side_weights,
                    StrategyLearningEvent.expert_integrity,
                    StrategyLearningEvent.attribution,
                )
            ),
            start=0,
        )
        eligible, reclaimable = await _candidate_totals(
            StrategyLearningEvent.id,
            conditions,
            size_expression=size_expression,
            measure_reclaimable_bytes=measure_reclaimable_bytes,
        )
        if not apply:
            return _section_summary(
                eligible=eligible,
                processed=0,
                reclaimable=reclaimable,
                cutoff=cutoff,
                max_rows=self.policy.max_rows_per_table,
            )

        processed = 0
        while processed < self.policy.max_rows_per_table:
            limit = min(
                self.policy.batch_size,
                self.policy.max_rows_per_table - processed,
            )
            async with get_read_session_ctx() as session:
                ids = list(
                    (
                        await session.execute(
                            select(StrategyLearningEvent.id)
                            .where(*conditions)
                            .order_by(StrategyLearningEvent.id.asc())
                            .limit(limit)
                        )
                    )
                    .scalars()
                    .all()
                )
            if not ids:
                break
            async with get_session_ctx() as session:
                await session.execute(
                    StrategyLearningEvent.__table__.update()
                    .where(StrategyLearningEvent.id.in_(ids))
                    .values(
                        strategy_snapshot=null(),
                        market_state=null(),
                        side_weights=null(),
                        expert_integrity=null(),
                        attribution=null(),
                    )
                )
            processed += len(ids)
            await _pause(self.policy.batch_pause_seconds)

        return _section_summary(
            eligible=eligible,
            processed=processed,
            reclaimable=reclaimable,
            cutoff=cutoff,
            max_rows=self.policy.max_rows_per_table,
        )


def _decision_retention_conditions(cutoff: datetime) -> tuple[Any, ...]:
    linked_order = select(Order.id).where(Order.decision_id == AIDecision.id).exists()
    return (
        AIDecision.created_at < cutoff,
        AIDecision.was_executed.is_(False),
        AIDecision.decision_learning_snapshot_version >= 1,
        AIDecision.decision_learning_snapshot.is_not(None),
        AIDecision.runtime_payload_compaction_version.is_distinct_from(
            RUNTIME_DATA_RETENTION_VERSION
        ),
        ~linked_order,
    )


def _shadow_retention_conditions(
    cutoff: datetime,
    *,
    keep_rows: int,
) -> tuple[Any, ...]:
    keep_ids = (
        select(ShadowBacktest.id.label("id"))
        .where(
            ShadowBacktest.status == "completed",
            ShadowBacktest.created_at >= PHASE3_CLEAN_START_UTC,
            ShadowBacktest.long_return_pct.is_not(None),
            ShadowBacktest.short_return_pct.is_not(None),
            or_(
                ShadowBacktest.decision_action.in_(("long", "short")),
                (
                    ShadowBacktest.missed_opportunity.is_(True)
                    & ShadowBacktest.best_action.in_(("long", "short"))
                ),
            ),
        )
        .order_by(ShadowBacktest.created_at.desc(), ShadowBacktest.id.desc())
        .limit(keep_rows)
        .subquery()
    )
    return (
        ShadowBacktest.created_at < cutoff,
        ShadowBacktest.status.in_(("completed", "quarantined")),
        ShadowBacktest.training_feature_snapshot_version >= 1,
        ShadowBacktest.training_feature_snapshot.is_not(None),
        ~ShadowBacktest.id.in_(select(keep_ids.c.id)),
        ShadowBacktest.runtime_payload_compaction_version.is_distinct_from(
            RUNTIME_DATA_RETENTION_VERSION
        ),
    )


def _strategy_event_retention_conditions(cutoff: datetime) -> tuple[Any, ...]:
    return (
        StrategyLearningEvent.created_at < cutoff,
        StrategyLearningEvent.event_status.not_in(ACTIVE_STRATEGY_EVENT_STATUSES),
        or_(
            StrategyLearningEvent.strategy_snapshot.is_not(None),
            StrategyLearningEvent.market_state.is_not(None),
            StrategyLearningEvent.side_weights.is_not(None),
            StrategyLearningEvent.expert_integrity.is_not(None),
            StrategyLearningEvent.attribution.is_not(None),
        ),
    )


async def _candidate_totals(
    id_column: Any,
    conditions: tuple[Any, ...],
    *,
    size_expression: Any,
    measure_reclaimable_bytes: bool,
) -> tuple[int, int | None]:
    async with get_read_session_ctx() as session:
        if "postgresql" in settings.database_url and measure_reclaimable_bytes:
            result = await session.execute(
                select(
                    func.count(id_column),
                    func.coalesce(func.sum(func.greatest(size_expression, 0)), 0),
                ).where(*conditions)
            )
            count, size = result.one()
            return int(count or 0), int(size or 0)
        count = await session.scalar(select(func.count(id_column)).where(*conditions))
        return int(count or 0), 0 if measure_reclaimable_bytes else None


def _json_size(column: Any) -> Any:
    if "postgresql" in settings.database_url:
        return func.coalesce(func.pg_column_size(column), 0)
    return literal(0)


def _section_summary(
    *,
    eligible: int,
    processed: int,
    reclaimable: int | None,
    cutoff: datetime,
    max_rows: int,
) -> dict[str, Any]:
    capped = int(processed) >= int(max_rows) and int(eligible) > int(processed)
    return {
        "eligible_rows": int(eligible),
        "processed_rows": int(processed),
        "remaining_estimate": max(int(eligible) - int(processed), 0),
        "reclaimable_bytes_measured": reclaimable is not None,
        "estimated_reclaimable_bytes": (
            int(reclaimable) if reclaimable is not None else None
        ),
        "cutoff": _as_utc(cutoff).isoformat(),
        "max_rows_per_run": int(max_rows),
        "bounded_by_max_rows": capped,
        "dry_run_would_be_bounded": int(eligible) > int(max_rows),
    }


async def _pause(seconds: float) -> None:
    if seconds > 0:
        await asyncio.sleep(seconds)


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
