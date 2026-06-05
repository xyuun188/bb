from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column

from models.base import Base, TimestampMixin


class ExpertMemory(Base, TimestampMixin):
    __tablename__ = "expert_memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    expert_name: Mapped[str] = mapped_column(String(50), index=True)
    expert_label: Mapped[str] = mapped_column(String(50), default="")
    symbol: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)
    side: Mapped[str | None] = mapped_column(String(10), nullable=True)
    memory_type: Mapped[str] = mapped_column(String(30), default="lesson")
    market_pattern: Mapped[str] = mapped_column(Text, default="")
    lesson: Mapped[str] = mapped_column(Text, default="")
    recommended_action: Mapped[str] = mapped_column(String(40), default="reduce_risk")
    confidence_adjustment: Mapped[float] = mapped_column(Float, default=0.0)
    position_size_multiplier: Mapped[float] = mapped_column(Float, default=1.0)
    evidence_count: Mapped[int] = mapped_column(Integer, default=1)
    hit_count: Mapped[int] = mapped_column(Integer, default=0)
    success_count: Mapped[int] = mapped_column(Integer, default=0)
    failure_count: Mapped[int] = mapped_column(Integer, default=0)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.5)
    memory_key: Mapped[str] = mapped_column(String(160), index=True)
    source_position_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    extra: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class TradeReflection(Base, TimestampMixin):
    __tablename__ = "trade_reflections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    position_id: Mapped[int] = mapped_column(Integer, index=True)
    model_name: Mapped[str] = mapped_column(String(50), index=True)
    execution_mode: Mapped[str] = mapped_column(String(10), index=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    side: Mapped[str] = mapped_column(String(10))
    entry_price: Mapped[float] = mapped_column(Float, default=0.0)
    exit_price: Mapped[float] = mapped_column(Float, default=0.0)
    quantity: Mapped[float] = mapped_column(Float, default=0.0)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    fee_estimate: Mapped[float] = mapped_column(Float, default=0.0)
    hold_minutes: Mapped[float] = mapped_column(Float, default=0.0)
    outcome: Mapped[str] = mapped_column(String(20), default="flat")
    mistake_summary: Mapped[str] = mapped_column(Text, default="")
    improvement_summary: Mapped[str] = mapped_column(Text, default="")
    expert_lessons: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    source: Mapped[str] = mapped_column(String(40), default="system")


class ShadowBacktest(Base, TimestampMixin):
    """Delayed outcome sample for AI decisions.

    It records what the market did after a decision without placing any order.
    The data is later used for missed-opportunity and bad-entry analysis.
    """

    __tablename__ = "shadow_backtests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    decision_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    model_name: Mapped[str] = mapped_column(String(50), index=True)
    execution_mode: Mapped[str] = mapped_column(String(10), index=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    analysis_type: Mapped[str] = mapped_column(String(20), default="market", index=True)
    decision_action: Mapped[str] = mapped_column(String(20), default="hold")
    decision_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    entry_price: Mapped[float] = mapped_column(Float, default=0.0)
    feature_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    raw_llm_response: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    due_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    horizon_minutes: Mapped[int] = mapped_column(Integer, default=10, index=True)
    actual_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    long_return_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    short_return_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_action: Mapped[str | None] = mapped_column(String(20), nullable=True)
    missed_opportunity: Mapped[bool] = mapped_column(Boolean, default=False)
    note: Mapped[str] = mapped_column(Text, default="")
