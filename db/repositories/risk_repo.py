from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, select

from db.repositories.base import BaseRepository
from models.risk import ModelPerformanceSnapshot, RiskEvent


class RiskRepository(BaseRepository):
    """Repository for risk events and model performance snapshots."""

    async def log_risk_event(self, data: dict) -> RiskEvent:
        event = RiskEvent(**data)
        self.session.add(event)
        await self.session.flush()
        return event

    async def get_recent_events(self, limit: int = 50) -> list[RiskEvent]:
        result = await self.session.execute(
            select(RiskEvent).order_by(RiskEvent.created_at.desc()).limit(limit)
        )
        return list(result.scalars().all())

    async def save_performance_snapshot(self, data: dict) -> ModelPerformanceSnapshot:
        snap = ModelPerformanceSnapshot(**data, evaluated_at=datetime.utcnow())
        self.session.add(snap)
        await self.session.flush()
        return snap

    async def get_latest_snapshot(
        self, model_name: str
    ) -> ModelPerformanceSnapshot | None:
        result = await self.session.execute(
            select(ModelPerformanceSnapshot)
            .where(ModelPerformanceSnapshot.model_name == model_name)
            .order_by(ModelPerformanceSnapshot.evaluated_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_all_latest_snapshots(self) -> list[ModelPerformanceSnapshot]:
        """Return the most recent snapshot for each model."""
        from sqlalchemy import and_

        subq = (
            select(
                ModelPerformanceSnapshot.model_name,
                func.max(ModelPerformanceSnapshot.evaluated_at).label("max_eval"),
            )
            .group_by(ModelPerformanceSnapshot.model_name)
            .subquery()
        )
        result = await self.session.execute(
            select(ModelPerformanceSnapshot).join(
                subq,
                and_(
                    ModelPerformanceSnapshot.model_name == subq.c.model_name,
                    ModelPerformanceSnapshot.evaluated_at == subq.c.max_eval,
                ),
            )
        )
        return list(result.scalars().all())

    async def get_historical_snapshots(
        self, model_name: str, limit: int = 100
    ) -> list[ModelPerformanceSnapshot]:
        result = await self.session.execute(
            select(ModelPerformanceSnapshot)
            .where(ModelPerformanceSnapshot.model_name == model_name)
            .order_by(ModelPerformanceSnapshot.evaluated_at.asc())
            .limit(limit)
        )
        return list(result.scalars().all())
