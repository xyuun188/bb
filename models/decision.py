import copy
from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text, event
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
    # Keep report-only model evidence out of the large raw decision payload so
    # dashboard health reads never have to decompress full model transcripts.
    model_health_timings: Mapped[list | None] = mapped_column(JSON, nullable=True)
    model_health_fallback_timings: Mapped[list | None] = mapped_column(JSON, nullable=True)
    model_health_experts: Mapped[list | None] = mapped_column(JSON, nullable=True)
    model_health_opinions: Mapped[list | None] = mapped_column(JSON, nullable=True)
    model_health_has_ml_signal: Mapped[bool] = mapped_column(Boolean, default=False)
    model_health_has_local_ml_signal: Mapped[bool] = mapped_column(Boolean, default=False)
    model_health_has_local_ai_tools: Mapped[bool] = mapped_column(Boolean, default=False)
    model_health_snapshot_version: Mapped[int] = mapped_column(Integer, default=0)
    decision_learning_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    decision_learning_snapshot_version: Mapped[int] = mapped_column(Integer, default=0)
    analysis_type: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)
    is_paper: Mapped[bool] = mapped_column(Boolean, default=True)
    was_executed: Mapped[bool] = mapped_column(Boolean, default=False)
    execution_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    execution_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    outcome: Mapped[str | None] = mapped_column(String(20), nullable=True)  # profit/loss/flat
    outcome_pnl_pct: Mapped[float | None] = mapped_column(Float, nullable=True)


def _model_health_snapshot_value(raw: object, key: str) -> list | None:
    if not isinstance(raw, dict):
        return None
    value = raw.get(key)
    return copy.deepcopy(value) if isinstance(value, list) else None


def _model_health_snapshot_present(raw: object, key: str) -> bool:
    if not isinstance(raw, dict):
        return False
    value = raw.get(key)
    return value not in (None, {}, [])


_LEARNING_SNAPSHOT_MAX_STRING_CHARS = 1_024
_LEARNING_SNAPSHOT_MAX_CONTAINER_BYTES = 16_384
_LEARNING_SNAPSHOT_MAX_DEPTH = 2
_MISSING_LEARNING_VALUE = object()


def _compact_learning_value(value: object, *, depth: int = 0) -> object:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= _LEARNING_SNAPSHOT_MAX_STRING_CHARS else _MISSING_LEARNING_VALUE
    if depth >= _LEARNING_SNAPSHOT_MAX_DEPTH:
        return _MISSING_LEARNING_VALUE
    if isinstance(value, dict):
        compact: dict[str, object] = {}
        for key, nested_value in value.items():
            if not isinstance(key, str) or len(key) > 120:
                continue
            compact_value = _compact_learning_value(nested_value, depth=depth + 1)
            if compact_value is not _MISSING_LEARNING_VALUE:
                compact[key] = compact_value
        return compact if compact else _MISSING_LEARNING_VALUE
    if isinstance(value, list) and len(value) <= 32:
        compact_list = [
            compact_value
            for item in value
            if (compact_value := _compact_learning_value(item, depth=depth + 1))
            is not _MISSING_LEARNING_VALUE
        ]
        return compact_list
    return _MISSING_LEARNING_VALUE


def _compact_decision_learning_snapshot(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        return {}
    compact: dict[str, object] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or len(key) > 120:
            continue
        compact_value = _compact_learning_value(value)
        if compact_value is not _MISSING_LEARNING_VALUE:
            compact[key] = copy.deepcopy(compact_value)
    return compact


@event.listens_for(AIDecision, "before_insert")
@event.listens_for(AIDecision, "before_update")
def _sync_model_health_snapshot(_mapper, _connection, target: AIDecision) -> None:
    """Keep SQLite/tests aligned with the PostgreSQL write trigger."""

    raw = target.raw_llm_response
    target.model_health_timings = _model_health_snapshot_value(raw, "model_timings")
    target.model_health_fallback_timings = _model_health_snapshot_value(raw, "_model_timings")
    target.model_health_experts = _model_health_snapshot_value(raw, "experts")
    target.model_health_opinions = _model_health_snapshot_value(raw, "opinions")
    target.model_health_has_ml_signal = _model_health_snapshot_present(raw, "ml_signal")
    target.model_health_has_local_ml_signal = _model_health_snapshot_present(raw, "local_ml_signal")
    target.model_health_has_local_ai_tools = _model_health_snapshot_present(raw, "local_ai_tools")
    target.model_health_snapshot_version = 1
    target.decision_learning_snapshot = _compact_decision_learning_snapshot(raw)
    target.decision_learning_snapshot_version = 1
