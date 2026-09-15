"""Durable operator commands consumed only by the trading process."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, Index, Integer, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column

from models.base import Base, TimestampMixin


class TradingControlCommand(Base, TimestampMixin):
    """An auditable command sent from the Dashboard to the trading worker.

    A command is never retried after a worker interruption. The operator must
    explicitly submit a new command once the actual OKX state is reconciled.
    """

    __tablename__ = "trading_control_commands"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    command_type: Mapped[str] = mapped_column(String(40), index=True)
    status: Mapped[str] = mapped_column(String(40), default="queued", index=True)
    execution_mode: Mapped[str] = mapped_column(String(10), index=True)
    position_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    position_key: Mapped[int] = mapped_column(Integer, default=0)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index(
            "uq_active_trading_control_command",
            "command_type",
            "execution_mode",
            "position_key",
            unique=True,
            postgresql_where=text("status IN ('queued', 'running')"),
            sqlite_where=text("status IN ('queued', 'running')"),
        ),
    )
