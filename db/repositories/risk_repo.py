from __future__ import annotations

from sqlalchemy import select

from db.repositories.base import BaseRepository
from models.risk import RiskEvent


class RiskRepository(BaseRepository):
    """Repository for risk events."""

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
