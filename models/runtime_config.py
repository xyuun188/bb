"""Immutable runtime configuration snapshots and activation history."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Index, Integer, String, Text, event, inspect
from sqlalchemy.orm import Mapped, mapped_column

from core.experiment_contracts import ExperimentContractError
from core.promotion_contracts import verify_runtime_config_snapshot
from models.base import Base, TimestampMixin


class RuntimeConfigVersion(Base, TimestampMixin):
    __tablename__ = "runtime_config_versions"
    __table_args__ = (Index("idx_runtime_config_environment_created", "environment", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version_id: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    environment: Mapped[str] = mapped_column(String(40), index=True)
    config_sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    snapshot: Mapped[dict[str, Any]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(30), default="registered", index=True)
    parent_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    changed_by: Mapped[str] = mapped_column(String(160), default="system")
    change_reason: Mapped[str] = mapped_column(Text, default="")
    activation_by: Mapped[str | None] = mapped_column(String(160), nullable=True)
    activation_reason: Mapped[str] = mapped_column(Text, default="")
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deactivated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


_IMMUTABLE = (
    "version_id",
    "environment",
    "config_sha256",
    "snapshot",
    "parent_version",
    "changed_by",
    "change_reason",
)


@event.listens_for(RuntimeConfigVersion, "before_insert")
def _validate_insert(_mapper: Any, _connection: Any, target: RuntimeConfigVersion) -> None:
    verify_runtime_config_snapshot(target.snapshot)
    if target.version_id != target.snapshot.get("version_id") or target.config_sha256 != target.snapshot.get("config_sha256"):
        raise ExperimentContractError("runtime config identity mismatch")
    for name in ("environment", "parent_version", "changed_by", "change_reason"):
        snapshot_name = name
        if getattr(target, name) != target.snapshot.get(snapshot_name):
            raise ExperimentContractError(f"runtime config field {name} mismatches snapshot")


@event.listens_for(RuntimeConfigVersion, "before_update")
def _validate_update(_mapper: Any, _connection: Any, target: RuntimeConfigVersion) -> None:
    state = inspect(target)
    changed = [name for name in _IMMUTABLE if state.attrs[name].history.has_changes()]
    if changed:
        raise ExperimentContractError("runtime config immutable fields cannot be updated: " + ", ".join(changed))
    verify_runtime_config_snapshot(target.snapshot)
