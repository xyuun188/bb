import copy
from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text, event, inspect
from sqlalchemy.orm import Mapped, mapped_column

from core.training_contracts import SHADOW_LABEL_VERSION
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
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
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
    training_feature_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    training_feature_snapshot_version: Mapped[int] = mapped_column(Integer, default=0)
    raw_llm_response: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    due_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    horizon_minutes: Mapped[int] = mapped_column(Integer, default=10, index=True)
    label_version: Mapped[str] = mapped_column(
        String(80),
        default=SHADOW_LABEL_VERSION,
        index=True,
    )
    actual_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    long_return_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    short_return_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_action: Mapped[str | None] = mapped_column(String(20), nullable=True)
    missed_opportunity: Mapped[bool] = mapped_column(Boolean, default=False)
    note: Mapped[str] = mapped_column(Text, default="")


_TRAINING_FEATURE_MAX_STRING_CHARS = 512
_TRAINING_FEATURE_MAX_DICT_DEPTH = 1
_MISSING_TRAINING_FEATURE = object()


def _compact_training_feature_value(value: object, *, depth: int = 0) -> object:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= _TRAINING_FEATURE_MAX_STRING_CHARS else _MISSING_TRAINING_FEATURE
    if isinstance(value, dict) and depth < _TRAINING_FEATURE_MAX_DICT_DEPTH:
        compact: dict[str, object] = {}
        for key, nested_value in value.items():
            if not isinstance(key, str) or len(key) > 120:
                continue
            compact_value = _compact_training_feature_value(nested_value, depth=depth + 1)
            if compact_value is not _MISSING_TRAINING_FEATURE:
                compact[key] = compact_value
        return compact if compact else _MISSING_TRAINING_FEATURE
    return _MISSING_TRAINING_FEATURE


def _compact_training_feature_snapshot(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    compact: dict[str, object] = {}
    for key, raw_value in value.items():
        if not isinstance(key, str) or len(key) > 120:
            continue
        compact_value = _compact_training_feature_value(raw_value)
        if compact_value is not _MISSING_TRAINING_FEATURE:
            compact[key] = copy.deepcopy(compact_value)
    return compact


@event.listens_for(ShadowBacktest, "before_insert")
@event.listens_for(ShadowBacktest, "before_update")
def _sync_training_feature_snapshot(_mapper, _connection, target: ShadowBacktest) -> None:
    """Keep SQLite/tests aligned with the PostgreSQL training-snapshot trigger."""

    state = inspect(target)
    if state.persistent and not state.attrs.feature_snapshot.history.has_changes():
        return
    target.training_feature_snapshot = _compact_training_feature_snapshot(target.feature_snapshot)
    target.training_feature_snapshot_version = 1


class StrategyProfileSnapshot(Base, TimestampMixin):
    """Versioned strategy profile snapshot used by scheduler attribution."""

    __tablename__ = "strategy_profile_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    execution_mode: Mapped[str] = mapped_column(String(10), default="paper", index=True)
    profile_id: Mapped[str] = mapped_column(String(80), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1, index=True)
    label: Mapped[str] = mapped_column(String(120), default="")
    status: Mapped[str] = mapped_column(String(30), default="candidate", index=True)
    source: Mapped[str] = mapped_column(String(60), default="feedback_generator")
    description: Mapped[str] = mapped_column(Text, default="")
    params: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    promotion: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    backtest_metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    shadow_validation: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    probe_state: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    scheduler_reason: Mapped[str] = mapped_column(Text, default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    is_disabled: Mapped[bool] = mapped_column(Boolean, default=False, index=True)


class StrategyLearningEvent(Base, TimestampMixin):
    """Auditable event for decisions, blocks, executions, and manual closes."""

    __tablename__ = "strategy_learning_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(50), index=True)
    execution_mode: Mapped[str] = mapped_column(String(10), default="paper", index=True)
    symbol: Mapped[str | None] = mapped_column(String(30), nullable=True, index=True)
    side: Mapped[str | None] = mapped_column(String(10), nullable=True, index=True)
    action: Mapped[str | None] = mapped_column(String(30), nullable=True, index=True)
    event_type: Mapped[str] = mapped_column(String(50), index=True)
    event_status: Mapped[str] = mapped_column(String(30), default="recorded", index=True)
    severity: Mapped[str] = mapped_column(String(12), default="info")
    reason: Mapped[str] = mapped_column(Text, default="")
    decision_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    order_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    position_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    profile_id: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    profile_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    scheduler_reason: Mapped[str] = mapped_column(Text, default="")
    strategy_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    market_state: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    side_weights: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    expert_integrity: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    attribution: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    exclude_from_training: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
