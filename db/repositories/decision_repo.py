from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import func, or_, select

from db.repositories.base import BaseRepository
from models.decision import AIDecision
from web_dashboard.api.text_sanitize import sanitize_payload, sanitize_text


class DecisionRepository(BaseRepository):
    """Repository for AI decisions."""

    model = AIDecision

    async def log_decision(self, data: dict) -> AIDecision:
        decision = AIDecision(**_sanitize_decision_payload(data))
        self.session.add(decision)
        await self.session.flush()
        return decision

    async def get_recent_decisions(
        self,
        model_name: str | None = None,
        symbol: str | None = None,
        action: str | None = None,
        limit: int = 50,
        offset: int = 0,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        was_executed: bool | None = None,
        is_paper: bool | None = None,
    ) -> list[AIDecision]:
        stmt = (
            select(AIDecision)
            .order_by(AIDecision.created_at.desc())
            .offset(max(int(offset or 0), 0))
            .limit(limit)
        )
        if model_name:
            stmt = stmt.where(AIDecision.model_name == model_name)
        if symbol:
            stmt = stmt.where(AIDecision.symbol == symbol)
        if action:
            stmt = stmt.where(AIDecision.action == action)
        if start_date:
            stmt = stmt.where(AIDecision.created_at >= start_date)
        if end_date:
            stmt = stmt.where(AIDecision.created_at <= end_date)
        if was_executed is not None:
            stmt = stmt.where(AIDecision.was_executed == was_executed)
        if is_paper is not None:
            stmt = stmt.where(AIDecision.is_paper == is_paper)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def mark_executed(self, decision_id: int, execution_price: float) -> AIDecision | None:
        decision = await self.get(decision_id)
        if decision:
            decision.was_executed = True
            decision.execution_reason = None
            decision.executed_at = datetime.utcnow()
            decision.execution_price = execution_price
            await self.session.flush()
        return decision

    async def mark_execution_reason(
        self, decision_id: int, reason: str | None
    ) -> AIDecision | None:
        decision = await self.get(decision_id)
        if decision:
            decision.execution_reason = sanitize_text(reason)
            await self.session.flush()
        return decision

    async def update_raw_response(
        self, decision_id: int, raw_response: dict | None
    ) -> AIDecision | None:
        decision = await self.get(decision_id)
        if decision:
            clean_response = sanitize_payload(raw_response)
            decision.raw_llm_response = clean_response
            _sync_execution_parameters(decision, clean_response)
            await self.session.flush()
        return decision

    async def fill_missing_execution_reasons(self, decision_ids: list[int], reason: str) -> int:
        if not decision_ids:
            return 0
        stmt = select(AIDecision).where(
            AIDecision.id.in_(decision_ids),
            AIDecision.was_executed.is_(False),
            or_(
                AIDecision.execution_reason.is_(None),
                AIDecision.execution_reason == "",
                AIDecision.execution_reason.like("已进入本轮开仓候选排序%"),
                AIDecision.execution_reason.like("本轮执行仍在处理中%"),
                AIDecision.execution_reason.like("正在提交 OKX%"),
                AIDecision.execution_reason.like("Execution still pending this round%"),
                AIDecision.execution_reason.like("本轮还在分析或排队中%"),
            ),
        )
        result = await self.session.execute(stmt)
        rows = list(result.scalars().all())
        clean_reason = sanitize_text(reason)
        for decision in rows:
            decision.execution_reason = clean_reason
        if rows:
            await self.session.flush()
        return len(rows)

    async def mark_outcome(
        self, decision_id: int, outcome: str, pnl_pct: float
    ) -> AIDecision | None:
        decision = await self.get(decision_id)
        if decision:
            decision.outcome = outcome
            decision.outcome_pnl_pct = pnl_pct
            await self.session.flush()
        return decision

    async def get_decision_accuracy(self, model_name: str, since: datetime | None = None) -> float:
        """Calculate what fraction of executed decisions were profitable."""
        stmt = select(func.count(AIDecision.id)).where(
            AIDecision.model_name == model_name,
            AIDecision.was_executed.is_(True),
        )
        if since:
            stmt = stmt.where(AIDecision.created_at >= since)
        total = (await self.session.execute(stmt)).scalar() or 0
        if total == 0:
            return 0.0

        win_stmt = stmt.where(AIDecision.outcome == "profit")
        wins = (await self.session.execute(win_stmt)).scalar() or 0
        return wins / total

    async def count_decisions(
        self,
        model_name: str | None = None,
        action: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        was_executed: bool | None = None,
        is_paper: bool | None = None,
    ) -> int:
        """Count decisions matching the given filters (ignoring limit)."""
        stmt = select(func.count(AIDecision.id))
        if model_name:
            stmt = stmt.where(AIDecision.model_name == model_name)
        if action:
            stmt = stmt.where(AIDecision.action == action)
        if start_date:
            stmt = stmt.where(AIDecision.created_at >= start_date)
        if end_date:
            stmt = stmt.where(AIDecision.created_at <= end_date)
        if was_executed is not None:
            stmt = stmt.where(AIDecision.was_executed == was_executed)
        if is_paper is not None:
            stmt = stmt.where(AIDecision.is_paper == is_paper)
        result = await self.session.execute(stmt)
        return result.scalar() or 0

    async def delete_all(self) -> int:
        """Delete all AI decision records. Returns count of deleted rows."""
        from sqlalchemy import delete

        result = await self.session.execute(delete(AIDecision))
        await self.session.flush()
        return result.rowcount


def _sanitize_decision_payload(data: dict) -> dict:
    clean = dict(data or {})
    for key in ("reasoning", "execution_reason"):
        if key in clean:
            clean[key] = sanitize_text(clean.get(key))
    for key in ("feature_snapshot", "raw_llm_response"):
        if key in clean:
            clean[key] = sanitize_payload(clean.get(key))
    return clean


def _sync_execution_parameters(decision: AIDecision, raw_response: Any) -> None:
    if not isinstance(raw_response, dict):
        return
    parameters = raw_response.get("execution_parameters")
    if not isinstance(parameters, dict):
        parameters = raw_response.get("profit_risk_sizing")
    if not isinstance(parameters, dict):
        return
    for field_name in (
        "position_size_pct",
        "suggested_leverage",
        "stop_loss_pct",
        "take_profit_pct",
    ):
        if field_name not in parameters:
            continue
        value = _optional_float(parameters.get(field_name))
        if value is not None:
            setattr(decision, field_name, value)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
