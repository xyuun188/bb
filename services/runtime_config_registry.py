"""Versioned runtime configuration with auditable activation and rollback."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.experiment_contracts import ExperimentContractError
from core.promotion_contracts import build_runtime_config_snapshot, verify_runtime_config_snapshot
from models.runtime_config import RuntimeConfigVersion


class RuntimeConfigRegistry:
    """Register immutable snapshots and make activation an explicit operation."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, version_id: str) -> RuntimeConfigVersion | None:
        return (
            await self.session.execute(
                select(RuntimeConfigVersion).where(RuntimeConfigVersion.version_id == str(version_id or ""))
            )
        ).scalar_one_or_none()

    async def register(
        self,
        values: dict[str, Any],
        *,
        environment: str,
        parent_version: str | None = None,
        changed_by: str = "system",
        change_reason: str = "",
    ) -> RuntimeConfigVersion:
        snapshot = build_runtime_config_snapshot(
            values,
            environment=environment,
            parent_version=parent_version,
            changed_by=changed_by,
            change_reason=change_reason,
        )
        existing = await self.get(snapshot["version_id"])
        if existing is not None:
            if existing.snapshot != snapshot:
                raise ExperimentContractError("runtime config version already contains different evidence")
            return existing
        row = RuntimeConfigVersion(
            version_id=snapshot["version_id"],
            environment=snapshot["environment"],
            config_sha256=snapshot["config_sha256"],
            snapshot=snapshot,
            status="registered",
            parent_version=snapshot.get("parent_version"),
            changed_by=snapshot["changed_by"],
            change_reason=snapshot["change_reason"],
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def activate(self, version_id: str, *, activated_by: str, reason: str = "") -> RuntimeConfigVersion:
        row = await self._required(version_id)
        verify_runtime_config_snapshot(row.snapshot)
        current_result = await self.session.execute(
            select(RuntimeConfigVersion).where(
                RuntimeConfigVersion.environment == row.environment,
                RuntimeConfigVersion.status == "active",
            )
        )
        now = datetime.now(UTC)
        for current in current_result.scalars().all():
            if current.version_id != row.version_id:
                current.status = "superseded"
                current.deactivated_at = now
        row.status = "active"
        row.activated_at = now
        row.activation_by = str(activated_by or "system")
        row.activation_reason = str(reason or "activated")[:4000]
        await self.session.flush()
        return row

    async def rollback(self, version_id: str, *, rolled_back_by: str, reason: str) -> RuntimeConfigVersion:
        target = await self._required(version_id)
        if not target.parent_version:
            raise ExperimentContractError("configuration rollback requires a parent version")
        parent = await self._required(target.parent_version)
        if parent.environment != target.environment:
            raise ExperimentContractError("configuration rollback environment mismatch")
        await self.activate(parent.version_id, activated_by=rolled_back_by, reason=f"rollback:{reason}")
        target.status = "rolled_back"
        target.deactivated_at = datetime.now(UTC)
        await self.session.flush()
        return parent

    async def _required(self, version_id: str) -> RuntimeConfigVersion:
        row = await self.get(version_id)
        if row is None:
            raise ExperimentContractError(f"unknown runtime config version {version_id!r}")
        return row
