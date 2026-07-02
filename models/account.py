from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from models.base import Base, TimestampMixin


class VirtualAccount(Base, TimestampMixin):
    __tablename__ = "virtual_accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    initial_balance: Mapped[float] = mapped_column(Float, default=100000.0)
    current_balance: Mapped[float] = mapped_column(Float, default=100000.0)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    total_trades: Mapped[int] = mapped_column(Integer, default=0)
    winning_trades: Mapped[int] = mapped_column(Integer, default=0)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def total_equity(self) -> float:
        return self.current_balance + self.unrealized_pnl

    @property
    def total_pnl_pct(self) -> float:
        if self.initial_balance == 0:
            return 0.0
        return (self.total_equity - self.initial_balance) / self.initial_balance


class ExecutionEquitySnapshot(Base, TimestampMixin):
    __tablename__ = "execution_equity_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    mode: Mapped[str] = mapped_column(String(10), index=True)
    model_name: Mapped[str] = mapped_column(String(50), index=True)
    snapshot_date: Mapped[str] = mapped_column(String(10), index=True)
    snapshot_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    equity: Mapped[float] = mapped_column(Float, default=0.0)
    total_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    source: Mapped[str] = mapped_column(String(20), default="observed")

    __table_args__ = (
        UniqueConstraint(
            "mode",
            "model_name",
            "snapshot_date",
            name="uq_execution_equity_snapshot_day",
        ),
    )


class OkxAccountBill(Base, TimestampMixin):
    __tablename__ = "okx_account_bills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    mode: Mapped[str] = mapped_column(String(10), index=True)
    bill_id: Mapped[str] = mapped_column(String(180), index=True)
    inst_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    pos_side: Mapped[str | None] = mapped_column(String(20), nullable=True)
    ccy: Mapped[str] = mapped_column(String(20), default="USDT")
    bill_type: Mapped[str | None] = mapped_column(String(40), nullable=True)
    bill_sub_type: Mapped[str | None] = mapped_column(String(40), nullable=True)
    bill_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    balance_change: Mapped[float] = mapped_column(Float, default=0.0)
    pnl: Mapped[float] = mapped_column(Float, default=0.0)
    fee: Mapped[float] = mapped_column(Float, default=0.0)
    funding_fee: Mapped[float] = mapped_column(Float, default=0.0)
    source: Mapped[str] = mapped_column(String(40), default="okx_account_bills")
    raw_bill: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    __table_args__ = (
        UniqueConstraint("mode", "bill_id", name="uq_okx_account_bill_mode_bill_id"),
    )
