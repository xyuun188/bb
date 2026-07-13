from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, or_, select

from db.repositories.base import BaseRepository
from models.learning import ExpertMemory, ShadowBacktest, TradeReflection
from services.text_integrity import looks_like_mojibake, sanitize_runtime_text

DAMAGED_MEMORY_MARKERS = (
    "\u539f\u59cb\u8bf4\u660e\u5df2\u635f\u574f",
    "\u65e0\u6cd5\u51c6\u786e\u8fd8\u539f",
    "raw note is damaged",
)


class MemoryRepository(BaseRepository):
    """Repository for expert long-term memories and trade reflections."""

    model = ExpertMemory

    async def get_relevant_memories(
        self,
        expert_name: str,
        symbol: str,
        side: str | None = None,
    ) -> list[ExpertMemory]:
        stmt = (
            select(ExpertMemory)
            .where(
                ExpertMemory.expert_name == expert_name,
                ExpertMemory.is_active.is_(True),
                or_(ExpertMemory.symbol.is_(None), ExpertMemory.symbol == symbol),
            )
            .order_by(
                ExpertMemory.symbol.desc(),
                ExpertMemory.confidence_score.desc(),
                ExpertMemory.evidence_count.desc(),
                ExpertMemory.updated_at.desc().nullslast(),
                ExpertMemory.created_at.desc(),
            )
        )
        result = await self.session.execute(stmt)
        rows = list(result.scalars().all())
        symbol_norm = _norm_symbol(symbol)
        side_norm = str(side or "").lower()
        filtered = [
            row
            for row in rows
            if (not row.symbol or _norm_symbol(row.symbol) == symbol_norm)
            and (not row.side or not side_norm or str(row.side or "").lower() == side_norm)
            and _memory_row_usable(row)
        ]
        return filtered

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
            existing.confidence_adjustment = 0.0
            existing.position_size_multiplier = 1.0
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

    async def create_shadow_backtest(self, data: dict[str, Any]) -> ShadowBacktest:
        data = _normalize_shadow_backtest_payload(dict(data or {}))
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


def _norm_symbol(symbol: str | None) -> str:
    if not symbol:
        return ""
    value = str(symbol).split(":")[0]
    if value.endswith("-SWAP"):
        value = value[:-5]
    if "/" not in value and "-" in value:
        parts = value.split("-")
        if len(parts) >= 2:
            value = f"{parts[0]}/{parts[1]}"
    return value.upper()


def _normalize_memory_payload(data: dict[str, Any]) -> dict[str, Any]:
    # The production database still has NOT NULL legacy columns. Persist only
    # neutral compatibility values so old callers cannot restore policy influence.
    data["confidence_adjustment"] = 0.0
    data["position_size_multiplier"] = 1.0
    for key in ("lesson", "market_pattern"):
        value = data.get(key)
        if value is not None:
            data[key] = str(sanitize_runtime_text(value) or "")
    for key in ("recommended_action", "extra"):
        if key in data:
            data[key] = sanitize_runtime_text(data.get(key))
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
    return _memory_text_usable(getattr(row, "lesson", None)) and _memory_text_usable(
        getattr(row, "market_pattern", None)
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
    realized_pnl = _finite_float(incoming.get("realized_pnl"), None)
    net_return = _finite_float(incoming.get("net_return_after_cost_pct"), None)
    if realized_pnl is None or net_return is None:
        return {**existing, **incoming}

    source_position_id = int(_finite_float(incoming.get("source_position_id"), 0.0) or 0)
    source_position_ids = [
        int(value)
        for value in previous.get("source_position_ids", [])
        if int(_finite_float(value, 0.0) or 0) > 0
    ]
    if source_position_id > 0 and source_position_id in source_position_ids:
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
    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 1e-12
        else (3.0 if gross_profit > 0 else 0.0)
    )
    aggregation = {
        "objective": "maximize_expected_realized_net_return_after_cost",
        "objective_version": "2026-07-12.v1",
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
        "profit_factor": round(profit_factor, 8),
        "latest_source_position_id": incoming.get("source_position_id"),
        "source_position_ids": [
            *source_position_ids,
            *([source_position_id] if source_position_id > 0 else []),
        ][-2000:],
    }
    return {**existing, **incoming, "outcome_aggregation": aggregation}


def _finite_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        parsed = float(value)
        return parsed if parsed == parsed and parsed not in {float("inf"), float("-inf")} else default
    except (TypeError, ValueError):
        return default


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
