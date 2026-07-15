from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from models.base import Base, TimestampMixin


class Order(Base, TimestampMixin):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(50), index=True)
    execution_mode: Mapped[str] = mapped_column(String(10))  # paper or live
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    side: Mapped[str] = mapped_column(String(10))  # buy or sell
    order_type: Mapped[str] = mapped_column(String(10))  # market or limit
    quantity: Mapped[float] = mapped_column(Float)
    price: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    fee: Mapped[float] = mapped_column(Float, default=0.0)
    decision_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    exchange_order_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    filled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    okx_inst_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    okx_trade_ids: Mapped[str | None] = mapped_column(String(500), nullable=True)
    okx_fill_contracts: Mapped[float | None] = mapped_column(Float, nullable=True)
    okx_fill_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    okx_state: Mapped[str | None] = mapped_column(String(40), nullable=True)
    okx_sync_status: Mapped[str | None] = mapped_column(String(40), nullable=True)
    okx_synced_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    okx_last_error: Mapped[str | None] = mapped_column(String(500), nullable=True)
    okx_raw_fills: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class Position(Base, TimestampMixin):
    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(50), index=True)
    execution_mode: Mapped[str] = mapped_column(String(10))
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    side: Mapped[str] = mapped_column(String(10))  # long or short
    quantity: Mapped[float] = mapped_column(Float)
    entry_price: Mapped[float] = mapped_column(Float)
    current_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    leverage: Mapped[float] = mapped_column(Float, default=1.0)
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    close_fill_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    entry_fee: Mapped[float] = mapped_column(Float, default=0.0)
    close_fee: Mapped[float] = mapped_column(Float, default=0.0)
    funding_fee: Mapped[float] = mapped_column(Float, default=0.0)
    settlement_status: Mapped[str | None] = mapped_column(String(40), nullable=True)
    settlement_source: Mapped[str | None] = mapped_column(String(80), nullable=True)
    settlement_synced_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    settlement_raw: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    current_management_contract: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    stop_loss_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    take_profit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_open: Mapped[bool] = mapped_column(Boolean, default=True)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    okx_inst_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    okx_pos_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    entry_exchange_order_id: Mapped[str | None] = mapped_column(String(500), nullable=True)
    close_exchange_order_id: Mapped[str | None] = mapped_column(String(500), nullable=True)


class OkxPositionHistory(Base, TimestampMixin):
    """Local mirror of OKX account positions-history rows.

    This table stores OKX's official historical position lifecycle records as
    the dashboard/training truth. Local Position rows can still help with order
    matching, but they should not be the source of historical PnL truth.
    """

    __tablename__ = "okx_position_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    mode: Mapped[str] = mapped_column(String(10), index=True)
    row_identity: Mapped[str] = mapped_column(String(260), index=True)
    inst_id: Mapped[str] = mapped_column(String(64), index=True)
    symbol: Mapped[str] = mapped_column(String(30), index=True)
    pos_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    pos_side: Mapped[str | None] = mapped_column(String(20), nullable=True)
    side: Mapped[str | None] = mapped_column(String(10), nullable=True)
    close_type: Mapped[str | None] = mapped_column(String(10), nullable=True)
    close_status: Mapped[str] = mapped_column(String(20), default="full")
    opened_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at_okx: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    open_avg_px: Mapped[float] = mapped_column(Float, default=0.0)
    close_avg_px: Mapped[float] = mapped_column(Float, default=0.0)
    open_max_pos: Mapped[float] = mapped_column(Float, default=0.0)
    close_total_pos: Mapped[float] = mapped_column(Float, default=0.0)
    leverage: Mapped[float] = mapped_column(Float, default=1.0)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    pnl: Mapped[float] = mapped_column(Float, default=0.0)
    pnl_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    funding_fee: Mapped[float] = mapped_column(Float, default=0.0)
    fee: Mapped[float] = mapped_column(Float, default=0.0)
    entry_order_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)
    close_order_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)
    linked_order_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)
    position_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)
    match_status: Mapped[str] = mapped_column(String(160), default="unmatched")
    evidence_gaps: Mapped[list | None] = mapped_column(JSON, nullable=True)
    source: Mapped[str] = mapped_column(String(80), default="okx_position_history")
    raw_row: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    sync_status: Mapped[str] = mapped_column(String(40), default="synced")
    last_sync_error: Mapped[str | None] = mapped_column(String(500), nullable=True)
    synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint("mode", "row_identity", name="uq_okx_position_history_mode_row"),
    )
