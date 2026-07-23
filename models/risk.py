from sqlalchemy import JSON, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from models.base import Base, TimestampMixin


class RiskEvent(Base, TimestampMixin):
    __tablename__ = "risk_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String(50), index=True)
    severity: Mapped[str] = mapped_column(String(10))  # warn / critical / halt
    symbol: Mapped[str | None] = mapped_column(String(20), nullable=True)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    triggered_by_model: Mapped[str | None] = mapped_column(String(50), nullable=True)
