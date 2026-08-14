"""Lifecycle operations for immutable experiment specifications and results."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.experiment_contracts import (
    ExperimentContractError,
    verify_experiment_result,
    verify_experiment_spec,
)
from models.experiment import ExperimentRun


class ExperimentRegistry:
    """Register immutable specs while allowing only controlled status transitions."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def register(
        self,
        spec: dict[str, Any],
        *,
        artifact_path: str = "",
    ) -> ExperimentRun:
        verify_experiment_spec(spec)
        experiment_id = str(spec["experiment_id"])
        existing = await self.get(experiment_id)
        if existing is not None:
            if existing.spec_sha256 != spec["spec_sha256"] or existing.spec != spec:
                raise ExperimentContractError(
                    "experiment_id already exists with different immutable evidence"
                )
            if artifact_path and existing.artifact_path != artifact_path:
                raise ExperimentContractError(
                    "experiment artifact path cannot change after registration"
                )
            return existing

        strategy = spec["strategy"]
        parameters = spec["parameters"]
        dataset = spec["dataset"]
        row = ExperimentRun(
            experiment_id=experiment_id,
            contract_version=str(spec["contract_version"]),
            experiment_type=str(spec["experiment_type"]),
            status="registered",
            strategy_id=str(strategy["strategy_id"]),
            strategy_version=str(strategy["strategy_version"]),
            parameter_set_id=str(parameters["parameter_set_id"]),
            dataset_id=str(dataset["dataset_id"]),
            git_commit=str(strategy["git_commit"]),
            spec_sha256=str(spec["spec_sha256"]),
            spec=spec,
            artifact_path=str(artifact_path or ""),
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def get(self, experiment_id: str) -> ExperimentRun | None:
        return (
            await self.session.execute(
                select(ExperimentRun).where(ExperimentRun.experiment_id == experiment_id)
            )
        ).scalar_one_or_none()

    async def mark_running(self, experiment_id: str) -> ExperimentRun:
        row = await self._required(experiment_id)
        if row.status == "complete":
            return row
        if row.status not in {"registered", "running", "failed"}:
            raise ExperimentContractError(
                f"experiment cannot start from status {row.status!r}"
            )
        row.status = "running"
        row.result = None
        row.result_sha256 = None
        row.started_at = row.started_at or datetime.now(UTC)
        row.completed_at = None
        row.failure_reason = ""
        await self.session.flush()
        return row

    async def complete(
        self,
        experiment_id: str,
        result: dict[str, Any],
    ) -> ExperimentRun:
        row = await self._required(experiment_id)
        verify_experiment_result(result, row.spec)
        if result.get("status") != "complete":
            raise ExperimentContractError("complete transition requires a complete result")
        if row.status == "complete":
            if row.result != result or row.result_sha256 != result["result_sha256"]:
                raise ExperimentContractError(
                    "completed experiment cannot be replaced with different result evidence"
                )
            return row
        if row.status not in {"registered", "running", "failed"}:
            raise ExperimentContractError(
                f"experiment cannot complete from status {row.status!r}"
            )
        row.status = "complete"
        row.result = result
        row.result_sha256 = str(result["result_sha256"])
        row.started_at = row.started_at or datetime.now(UTC)
        row.completed_at = datetime.now(UTC)
        row.failure_reason = ""
        await self.session.flush()
        return row

    async def fail(
        self,
        experiment_id: str,
        result: dict[str, Any],
        *,
        reason: str,
    ) -> ExperimentRun:
        row = await self._required(experiment_id)
        verify_experiment_result(result, row.spec)
        if result.get("status") != "failed":
            raise ExperimentContractError("fail transition requires a failed result")
        if row.status == "complete":
            raise ExperimentContractError("completed experiment cannot transition to failed")
        row.status = "failed"
        row.result = result
        row.result_sha256 = str(result["result_sha256"])
        row.started_at = row.started_at or datetime.now(UTC)
        row.completed_at = datetime.now(UTC)
        row.failure_reason = str(reason or "")[:4000]
        await self.session.flush()
        return row

    async def invalidate(self, experiment_id: str, *, reason: str) -> ExperimentRun:
        """Preserve completed evidence while excluding a provenance-defective run."""

        row = await self._required(experiment_id)
        safe_reason = str(reason or "").strip()
        if not safe_reason:
            raise ExperimentContractError("experiment invalidation requires a reason")
        if row.status == "invalidated":
            if row.failure_reason != safe_reason[:4000]:
                raise ExperimentContractError(
                    "invalidated experiment reason cannot be replaced"
                )
            return row
        if row.status not in {"complete", "failed"}:
            raise ExperimentContractError(
                f"experiment cannot be invalidated from status {row.status!r}"
            )
        row.status = "invalidated"
        row.failure_reason = safe_reason[:4000]
        await self.session.flush()
        return row

    async def _required(self, experiment_id: str) -> ExperimentRun:
        row = await self.get(str(experiment_id or ""))
        if row is None:
            raise ExperimentContractError(f"unknown experiment_id {experiment_id!r}")
        return row
