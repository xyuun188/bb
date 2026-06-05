from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.dialects.sqlite import JSON
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


class ModelPerformanceSnapshot(Base, TimestampMixin):
    __tablename__ = "model_performance_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(50), index=True)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    total_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    pnl_pct: Mapped[float] = mapped_column(Float, default=0.0)
    sharpe_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    max_drawdown: Mapped[float] = mapped_column(Float, default=0.0)
    win_rate: Mapped[float] = mapped_column(Float, default=0.0)
    total_trades: Mapped[int] = mapped_column(Integer, default=0)
    decision_accuracy: Mapped[float] = mapped_column(Float, default=0.0)
    volatility: Mapped[float] = mapped_column(Float, default=0.0)
    rank: Mapped[int] = mapped_column(Integer, default=0)
