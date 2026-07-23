from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Float, cast, func, or_, select

from core.symbols import normalize_trading_symbol
from core.training_contracts import (
    SHADOW_LABEL_VERSION,
    is_authoritative_expert_memory_extra,
)
from db.repositories.base import BaseRepository
from models.learning import ExpertMemory, ShadowBacktest, TradeReflection
from services.profit_training_contract import PROFIT_TRAINING_TARGET
from services.text_integrity import looks_like_mojibake, sanitize_runtime_text

DAMAGED_MEMORY_MARKERS = (
    "\u539f\u59cb\u8bf4\u660e\u5df2\u635f\u574f",
    "\u65e0\u6cd5\u51c6\u786e\u8fd8\u539f",
    "raw note is damaged",
)


class MemoryRepository(BaseRepository):
    """Repository for expert long-term memories and trade reflections."""

    model = ExpertMemory

    async def get_relevant_memories_for_experts(
        self,
        expert_names: list[str],
        symbol: str,
        side: str | None = None,
        *,
        per_expert_limit: int = 12,
    ) -> dict[str, list[ExpertMemory]]:
        """Load prompt memories for all experts with one database round trip."""

        names = list(dict.fromkeys(str(name or "").strip() for name in expert_names))
        names = [name for name in names if name]
        if not names:
            return {}
        limit = max(int(per_expert_limit or 0), 1)
        candidate_limit = limit * 3
        authority_rank = cast(
            ExpertMemory.extra["authority_rank"].as_string(),
            Float,
        )
        ranked_candidates = (
            select(
                ExpertMemory.id.label("memory_id"),
                func.row_number()
                .over(
                    partition_by=ExpertMemory.expert_name,
                    order_by=(
                        authority_rank.desc().nullslast(),
                        ExpertMemory.updated_at.desc().nullslast(),
                        ExpertMemory.id.desc(),
                    ),
                )
                .label("expert_rank"),
            )
            .where(
                ExpertMemory.expert_name.in_(names),
                ExpertMemory.is_active.is_(True),
                or_(ExpertMemory.symbol.is_(None), ExpertMemory.symbol == symbol),
            )
            .subquery()
        )
        stmt = (
            select(ExpertMemory)
            .join(
                ranked_candidates,
                ExpertMemory.id == ranked_candidates.c.memory_id,
            )
            .where(ranked_candidates.c.expert_rank <= candidate_limit)
        )
        result = await self.session.execute(stmt)
        symbol_norm = normalize_trading_symbol(symbol)
        side_norm = str(side or "").lower()
        grouped: dict[str, list[ExpertMemory]] = {name: [] for name in names}
        for row in result.scalars().all():
            expert_name = str(row.expert_name or "")
            if expert_name not in grouped:
                continue
            if row.symbol and normalize_trading_symbol(row.symbol) != symbol_norm:
                continue
            if row.side and side_norm and str(row.side or "").lower() != side_norm:
                continue
            if _memory_row_usable(row):
                grouped[expert_name].append(row)
        return {
            name: sorted(rows, key=_memory_authority_sort_key, reverse=True)[:limit]
            for name, rows in grouped.items()
            if rows
        }

    async def mark_memories_used(self, memory_ids: list[int]) -> None:
        if not memory_ids:
            return
        result = await self.session.execute(
            select(ExpertMemory).where(ExpertMemory.id.in_(memory_ids))
        )
        now = datetime.now(UTC)
        for memory in result.scalars().all():
            memory.hit_count = int(memory.hit_count or 0) + 1
            memory.last_used_at = now
        await self.session.flush()

    async def upsert_memory(self, data: dict[str, Any]) -> ExpertMemory:
        data = _normalize_memory_payload(dict(data or {}))
        memory_key = str(data.get("memory_key") or "").strip()
        existing = None
        if memory_key:
            result = await self.session.execute(
                select(ExpertMemory)
                .where(
                    ExpertMemory.expert_name == data.get("expert_name"),
                    ExpertMemory.memory_key == memory_key,
                )
                .limit(1)
            )
            existing = result.scalar_one_or_none()

        if existing:
            existing.evidence_count = int(existing.evidence_count or 0) + int(
                data.get("evidence_count", 1) or 1
            )
            existing.success_count = int(existing.success_count or 0) + int(
                data.get("success_count", 0) or 0
            )
            existing.failure_count = int(existing.failure_count or 0) + int(
                data.get("failure_count", 0) or 0
            )
            existing.confidence_score = existing.evidence_count / (existing.evidence_count + 1.0)
            existing.lesson = str(data.get("lesson") or existing.lesson or "")
            existing.market_pattern = str(
                data.get("market_pattern") or existing.market_pattern or ""
            )
            existing.recommended_action = str(
                data.get("recommended_action") or existing.recommended_action or ""
            )
            existing.source_position_id = (
                data.get("source_position_id") or existing.source_position_id
            )
            existing.extra = _merge_memory_outcomes(existing.extra, data.get("extra"))
            if data.get("is_active") is False:
                existing.is_active = False
            existing.updated_at = datetime.now(UTC)
            await self.session.flush()
            return existing

        data["extra"] = _merge_memory_outcomes(None, data.get("extra"))
        memory = ExpertMemory(**data)
        self.session.add(memory)
        await self.session.flush()
        return memory

    async def create_reflection(self, data: dict[str, Any]) -> TradeReflection | None:
        data = _normalize_reflection_payload(dict(data or {}))
        position_id = int(data.get("position_id") or 0)
        if position_id:
            result = await self.session.execute(
                select(TradeReflection).where(TradeReflection.position_id == position_id).limit(1)
            )
            existing = result.scalar_one_or_none()
            if existing:
                return None
        reflection = TradeReflection(**data)
        self.session.add(reflection)
        await self.session.flush()
        return reflection

    async def get_reflection_by_position_id(
        self, position_id: int
    ) -> TradeReflection | None:
        if int(position_id or 0) <= 0:
            return None
        result = await self.session.execute(
            select(TradeReflection)
            .where(TradeReflection.position_id == int(position_id))
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def update_reflection(
        self,
        reflection: TradeReflection,
        data: dict[str, Any],
    ) -> TradeReflection:
        normalized = _normalize_reflection_payload(dict(data or {}))
        for key, value in normalized.items():
            if key in {"id", "position_id", "created_at"}:
                continue
            if hasattr(reflection, key):
                setattr(reflection, key, value)
        reflection.updated_at = datetime.now(UTC)
        await self.session.flush()
        return reflection

    async def create_shadow_backtest(self, data: dict[str, Any]) -> ShadowBacktest:
        data = _normalize_shadow_backtest_payload(dict(data or {}))
        data["label_version"] = str(data.get("label_version") or SHADOW_LABEL_VERSION)
        decision_id = int(data.get("decision_id") or 0)
        horizon_minutes = int(data.get("horizon_minutes") or 0)
        if decision_id > 0 and horizon_minutes > 0:
            existing = (
                await self.session.execute(
                    select(ShadowBacktest)
                    .where(
                        ShadowBacktest.decision_id == decision_id,
                        ShadowBacktest.horizon_minutes == horizon_minutes,
                        ShadowBacktest.label_version == data["label_version"],
                    )
                    .limit(1)
                )
            ).scalar_one_or_none()
            if existing is not None:
                return existing
        row = ShadowBacktest(**data)
        self.session.add(row)
        await self.session.flush()
        return row

    async def get_due_shadow_backtests(self, limit: int = 200) -> list[ShadowBacktest]:
        now = datetime.now(UTC)
        stmt = (
            select(ShadowBacktest)
            .where(
                ShadowBacktest.status == "pending",
                ShadowBacktest.due_at <= now,
            )
            .order_by(ShadowBacktest.due_at.asc(), ShadowBacktest.id.asc())
            .limit(max(int(limit or 200), 1))
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_pending_shadow_backtests_by_ids(
        self, shadow_backtest_ids: list[int]
    ) -> list[ShadowBacktest]:
        """Reload due rows for a short write transaction after price collection."""

        ids = {int(row_id or 0) for row_id in shadow_backtest_ids}
        ids = sorted(row_id for row_id in ids if row_id > 0)
        if not ids:
            return []
        result = await self.session.execute(
            select(ShadowBacktest).where(
                ShadowBacktest.id.in_(ids),
                ShadowBacktest.status == "pending",
            )
        )
        return list(result.scalars().all())

    async def complete_shadow_backtest(
        self,
        row: ShadowBacktest,
        *,
        actual_price: float,
        long_return_pct: float,
        short_return_pct: float,
        best_action: str,
        missed_opportunity: bool,
        note: str = "",
    ) -> ShadowBacktest:
        row.actual_price = actual_price
        row.long_return_pct = long_return_pct
        row.short_return_pct = short_return_pct
        row.best_action = best_action
        row.missed_opportunity = missed_opportunity
        row.note = str(sanitize_runtime_text(note) or "")
        row.status = "completed"
        row.updated_at = datetime.now(UTC)
        await self.session.flush()
        return row

    async def list_shadow_backtests(
        self,
        status: str | None = None,
        symbol: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ShadowBacktest]:
        stmt = (
            select(ShadowBacktest)
            .order_by(ShadowBacktest.created_at.desc(), ShadowBacktest.id.desc())
            .offset(max(int(offset or 0), 0))
            .limit(max(int(limit or 100), 1))
        )
        if status:
            stmt = stmt.where(ShadowBacktest.status == status)
        if symbol:
            stmt = stmt.where(ShadowBacktest.symbol == symbol)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def count_shadow_backtests(
        self,
        status: str | None = None,
        symbol: str | None = None,
    ) -> int:
        stmt = select(func.count(ShadowBacktest.id))
        if status:
            stmt = stmt.where(ShadowBacktest.status == status)
        if symbol:
            stmt = stmt.where(ShadowBacktest.symbol == symbol)
        result = await self.session.execute(stmt)
        return int(result.scalar() or 0)

    async def list_memories(
        self,
        expert_name: str | None = None,
        symbol: str | None = None,
        active_only: bool = True,
        limit: int = 200,
        offset: int = 0,
    ) -> list[ExpertMemory]:
        stmt = (
            select(ExpertMemory)
            .order_by(
                ExpertMemory.updated_at.desc().nullslast(),
                ExpertMemory.created_at.desc(),
            )
            .offset(max(int(offset or 0), 0))
            .limit(max(int(limit or 200), 1))
        )
        if expert_name:
            stmt = stmt.where(ExpertMemory.expert_name == expert_name)
        if symbol:
            stmt = stmt.where(ExpertMemory.symbol == symbol)
        if active_only:
            stmt = stmt.where(ExpertMemory.is_active.is_(True))
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def count_memories(
        self,
        expert_name: str | None = None,
        symbol: str | None = None,
        active_only: bool = True,
    ) -> int:
        stmt = select(func.count(ExpertMemory.id))
        if expert_name:
            stmt = stmt.where(ExpertMemory.expert_name == expert_name)
        if symbol:
            stmt = stmt.where(ExpertMemory.symbol == symbol)
        if active_only:
            stmt = stmt.where(ExpertMemory.is_active.is_(True))
        result = await self.session.execute(stmt)
        return int(result.scalar() or 0)

    async def list_reflections(
        self,
        symbol: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[TradeReflection]:
        stmt = (
            select(TradeReflection)
            .order_by(
                TradeReflection.closed_at.desc().nullslast(),
                TradeReflection.created_at.desc(),
                TradeReflection.id.desc(),
            )
            .offset(max(int(offset or 0), 0))
            .limit(max(int(limit or 100), 1))
        )
        if symbol:
            stmt = stmt.where(TradeReflection.symbol == symbol)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def count_reflections(self, symbol: str | None = None) -> int:
        stmt = select(func.count(TradeReflection.id))
        if symbol:
            stmt = stmt.where(TradeReflection.symbol == symbol)
        result = await self.session.execute(stmt)
        return int(result.scalar() or 0)


def _blend(old: float, new: float) -> float:
    return (old * 0.7) + (new * 0.3)


def _normalize_memory_payload(data: dict[str, Any]) -> dict[str, Any]:
    supported = {column.name for column in ExpertMemory.__table__.columns}
    unknown = sorted(set(data).difference(supported))
    if unknown:
        raise ValueError(f"unsupported expert memory fields: {','.join(unknown)}")
    for key in ("lesson", "market_pattern"):
        value = data.get(key)
        if value is not None:
            data[key] = str(sanitize_runtime_text(value) or "")
    for key in ("recommended_action", "extra"):
        if key in data:
            data[key] = sanitize_runtime_text(data.get(key))
    if not is_authoritative_expert_memory_extra(data.get("extra")):
        raise ValueError("expert memory requires a complete authoritative OKX outcome contract")
    if not _memory_text_usable(data.get("lesson")) or not _memory_text_usable(
        data.get("market_pattern")
    ):
        data["is_active"] = False
    return data


def _normalize_reflection_payload(data: dict[str, Any]) -> dict[str, Any]:
    for key in ("mistake_summary", "improvement_summary", "expert_lessons"):
        if key in data:
            data[key] = sanitize_runtime_text(data.get(key))
    if "source" in data:
        data["source"] = str(sanitize_runtime_text(data.get("source")) or "")[:40]
    return data


def _normalize_shadow_backtest_payload(data: dict[str, Any]) -> dict[str, Any]:
    for key in ("feature_snapshot", "raw_llm_response"):
        if key in data:
            data[key] = sanitize_runtime_text(data.get(key))
    return data


def _memory_row_usable(row: ExpertMemory) -> bool:
    return (
        is_authoritative_expert_memory_extra(getattr(row, "extra", None))
        and _memory_text_usable(getattr(row, "lesson", None))
        and _memory_text_usable(getattr(row, "market_pattern", None))
    )


def _memory_outcome_aggregation(extra: Any) -> dict[str, Any]:
    source = extra if isinstance(extra, dict) else {}
    value = source.get("outcome_aggregation")
    return value if isinstance(value, dict) else {}


def _merge_memory_outcomes(existing_extra: Any, new_extra: Any) -> dict[str, Any]:
    existing = dict(existing_extra) if isinstance(existing_extra, dict) else {}
    incoming = dict(new_extra) if isinstance(new_extra, dict) else {}
    previous = _memory_outcome_aggregation(existing)
    if previous.get("return_unit") != "percentage_points":
        previous = {}
    if not is_authoritative_expert_memory_extra(incoming):
        return {**existing, **incoming}

    realized_pnl = _finite_float(incoming.get("realized_pnl"), None)
    net_return = _finite_float(incoming.get(PROFIT_TRAINING_TARGET), None)
    if realized_pnl is None or net_return is None:
        return {**existing, **incoming}

    source_position_id = int(_finite_float(incoming.get("source_position_id"), 0.0) or 0)
    source_position_ids = [
        int(value)
        for value in previous.get("source_position_ids", [])
        if int(_finite_float(value, 0.0) or 0) > 0
    ]
    outcome_id = str(incoming.get("outcome_id") or "").strip()
    source_outcome_ids = [
        str(value)
        for value in previous.get("source_outcome_ids", [])
        if str(value or "").strip()
    ]
    if outcome_id and outcome_id in source_outcome_ids:
        return {**existing, **incoming, "outcome_aggregation": previous}
    if not outcome_id and source_position_id > 0 and source_position_id in source_position_ids:
        return {**existing, **incoming, "outcome_aggregation": previous}

    previous_count = int(previous.get("count") or 0)
    count = previous_count + 1
    total_pnl = float(previous.get("total_realized_net_pnl_usdt") or 0.0) + float(
        realized_pnl or 0.0
    )
    total_return = float(previous.get("total_net_return_pct") or 0.0) + float(
        net_return or 0.0
    )
    squared_return_sum = float(previous.get("squared_net_return_sum_pct2") or 0.0) + float(
        net_return or 0.0
    ) ** 2
    avg_return = total_return / count
    if count > 1:
        return_variance = max(
            (squared_return_sum - count * avg_return**2) / (count - 1),
            0.0,
        )
        return_lcb = avg_return - (return_variance / count) ** 0.5
    else:
        return_lcb = avg_return
    previous_worst = _finite_float(previous.get("worst_net_return_pct"), None)
    worst_return = min(
        float(net_return or 0.0),
        float(previous_worst) if previous_worst is not None else float(net_return or 0.0),
    )
    positive_count = int(previous.get("positive_count") or 0) + int(
        (realized_pnl or 0.0) > 0
    )
    negative_count = int(previous.get("negative_count") or 0) + int(
        (realized_pnl or 0.0) < 0
    )
    gross_profit = float(previous.get("gross_profit_usdt") or 0.0) + max(
        float(realized_pnl or 0.0), 0.0
    )
    gross_loss = float(previous.get("gross_loss_usdt") or 0.0) + max(
        -float(realized_pnl or 0.0), 0.0
    )
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
    aggregation = {
        "objective": PROFIT_TRAINING_TARGET,
        "objective_version": "2026-07-23.v1",
        "return_unit": "percentage_points",
        "count": count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "conflict": positive_count > 0 and negative_count > 0,
        "total_realized_net_pnl_usdt": round(total_pnl, 8),
        "avg_realized_net_pnl_usdt": round(total_pnl / count, 8),
        "total_net_return_pct": round(total_return, 8),
        "avg_net_return_pct": round(avg_return, 8),
        "squared_net_return_sum_pct2": round(squared_return_sum, 8),
        "return_lcb_pct": round(return_lcb, 8),
        "worst_net_return_pct": round(worst_return, 8),
        "gross_profit_usdt": round(gross_profit, 8),
        "gross_loss_usdt": round(gross_loss, 8),
        "profit_factor": round(profit_factor, 8) if profit_factor is not None else None,
        "profit_factor_defined": profit_factor is not None,
        "latest_source_position_id": incoming.get("source_position_id"),
        "source_position_ids": [
            *source_position_ids,
            *([source_position_id] if source_position_id > 0 else []),
        ][-2000:],
        "source_outcome_ids": [
            *source_outcome_ids,
            *([outcome_id] if outcome_id else []),
        ][-2000:],
    }
    return {**existing, **incoming, "outcome_aggregation": aggregation}


def _finite_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        parsed = float(value)
        return parsed if parsed == parsed and parsed not in {float("inf"), float("-inf")} else default
    except (TypeError, ValueError):
        return default


def _memory_authority_sort_key(memory: ExpertMemory) -> tuple[float, float, float, float]:
    extra = memory.extra if isinstance(memory.extra, dict) else {}
    aggregation = _memory_outcome_aggregation(extra)
    authority_rank = _finite_float(extra.get("authority_rank"), 0.0) or 0.0
    updated_at = getattr(memory, "updated_at", None) or getattr(memory, "created_at", None)
    timestamp = updated_at.timestamp() if isinstance(updated_at, datetime) else 0.0
    return_lcb = _finite_float(aggregation.get("return_lcb_pct"), float("-inf"))
    worst_return = _finite_float(aggregation.get("worst_net_return_pct"), float("-inf"))
    return (
        float(authority_rank),
        timestamp,
        float(return_lcb if return_lcb is not None else float("-inf")),
        float(worst_return if worst_return is not None else float("-inf")),
    )


def _memory_text_usable(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    lowered = text.lower()
    if any(marker.lower() in lowered for marker in DAMAGED_MEMORY_MARKERS):
        return False
    return not looks_like_mojibake(text) and not _has_suspicious_unicode(text)


def _has_suspicious_unicode(text: str) -> bool:
    suspicious = 0
    for ch in text:
        code = ord(ch)
        if (
            0x0300 <= code <= 0x036F
            or 0x0370 <= code <= 0x03FF
            or 0x0400 <= code <= 0x052F
            or 0x0590 <= code <= 0x05FF
            or 0x0600 <= code <= 0x06FF
            or 0x0700 <= code <= 0x074F
            or 0x1100 <= code <= 0x11FF
            or 0x1E00 <= code <= 0x1EFF
            or 0xAC00 <= code <= 0xD7AF
        ):
            suspicious += 1
    return suspicious >= 2
