from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column

from models.base import Base, TimestampMixin


class AIDecision(Base, TimestampMixin):
    __tablename__ = "ai_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(50), index=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    action: Mapped[str] = mapped_column(String(20))
    confidence: Mapped[float] = mapped_column(Float)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    position_size_pct: Mapped[float] = mapped_column(Float, default=0.0)
    suggested_leverage: Mapped[float] = mapped_column(Float, default=1.0)
    stop_loss_pct: Mapped[float] = mapped_column(Float, default=0.0)
    take_profit_pct: Mapped[float] = mapped_column(Float, default=0.0)
    feature_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    raw_llm_response: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    analysis_type: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)
    is_paper: Mapped[bool] = mapped_column(Boolean, default=True)
    was_executed: Mapped[bool] = mapped_column(Boolean, default=False)
    execution_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    execution_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    outcome: Mapped[str | None] = mapped_column(String(20), nullable=True)  # profit/loss/flat
    outcome_pnl_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
