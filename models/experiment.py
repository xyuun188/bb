"""Persistent registry for immutable research experiment evidence."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Index, Integer, String, Text, event, inspect
from sqlalchemy.orm import Mapped, mapped_column

from core.experiment_contracts import (
    EXPERIMENT_CONTRACT_VERSION,
    ExperimentContractError,
    verify_experiment_result,
    verify_experiment_spec,
)
from models.base import Base, TimestampMixin


class ExperimentRun(Base, TimestampMixin):
    """One immutable spec plus its append-only execution outcome."""

    __tablename__ = "experiment_runs"
    __table_args__ = (
        Index("idx_experiment_runs_strategy_created", "strategy_id", "created_at"),
        Index("idx_experiment_runs_dataset_created", "dataset_id", "created_at"),
        Index("idx_experiment_runs_status_created", "status", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    contract_version: Mapped[str] = mapped_column(
        String(80),
        default=EXPERIMENT_CONTRACT_VERSION,
    )
    experiment_type: Mapped[str] = mapped_column(String(60), index=True)
    status: Mapped[str] = mapped_column(String(30), default="registered", index=True)
    strategy_id: Mapped[str] = mapped_column(String(120), index=True)
    strategy_version: Mapped[str] = mapped_column(String(120), index=True)
    parameter_set_id: Mapped[str] = mapped_column(String(80), index=True)
    dataset_id: Mapped[str] = mapped_column(String(80), index=True)
    git_commit: Mapped[str] = mapped_column(String(40), index=True)
    spec_sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    spec: Mapped[dict[str, Any]] = mapped_column(JSON)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    result_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    artifact_path: Mapped[str] = mapped_column(Text, default="")
    failure_reason: Mapped[str] = mapped_column(Text, default="")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


_IMMUTABLE_EXPERIMENT_FIELDS = (
    "experiment_id",
    "contract_version",
    "experiment_type",
    "strategy_id",
    "strategy_version",
    "parameter_set_id",
    "dataset_id",
    "git_commit",
    "spec_sha256",
    "spec",
    "artifact_path",
)


@event.listens_for(ExperimentRun, "before_insert")
def _validate_experiment_insert(_mapper: Any, _connection: Any, target: ExperimentRun) -> None:
    verify_experiment_spec(target.spec)
    strategy = target.spec["strategy"]
    parameters = target.spec["parameters"]
    dataset = target.spec["dataset"]
    expected = {
        "experiment_id": target.spec["experiment_id"],
        "contract_version": target.spec["contract_version"],
        "experiment_type": target.spec["experiment_type"],
        "strategy_id": strategy["strategy_id"],
        "strategy_version": strategy["strategy_version"],
        "parameter_set_id": parameters["parameter_set_id"],
        "dataset_id": dataset["dataset_id"],
        "git_commit": strategy["git_commit"],
        "spec_sha256": target.spec["spec_sha256"],
    }
    for name, expected_value in expected.items():
        if getattr(target, name) != expected_value:
            raise ExperimentContractError(f"experiment registry field {name} mismatches spec")
    _validate_result_if_present(target)


@event.listens_for(ExperimentRun, "before_update")
def _validate_experiment_update(_mapper: Any, _connection: Any, target: ExperimentRun) -> None:
    state = inspect(target)
    changed = [name for name in _IMMUTABLE_EXPERIMENT_FIELDS if state.attrs[name].history.has_changes()]
    if changed:
        raise ExperimentContractError(
            "immutable experiment fields cannot be updated: " + ", ".join(changed)
        )
    verify_experiment_spec(target.spec)
    _validate_result_if_present(target)


def _validate_result_if_present(target: ExperimentRun) -> None:
    if target.result is None:
        if target.result_sha256:
            raise ExperimentContractError("result_sha256 cannot exist without result evidence")
        return
    verify_experiment_result(target.result, target.spec)
    if target.result_sha256 != target.result["result_sha256"]:
        raise ExperimentContractError("registry result_sha256 mismatches result evidence")
