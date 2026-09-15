"""Cross-process Dashboard to trading-worker command handoff."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import structlog
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from core.safe_output import safe_error_text
from db.session import get_session_ctx
from models.trading_control import TradingControlCommand

logger = structlog.get_logger(__name__)

COMMAND_CLOSE_POSITION = "close_position"
COMMAND_CLOSE_ALL_POSITIONS = "close_all_positions"
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_REQUIRES_RECONFIRMATION = "requires_reconfirmation"


class TradingControlCommandQueue:
    """Persist and safely consume manual close commands."""

    def __init__(self, worker_id: str | None = None) -> None:
        self.worker_id = worker_id or f"trading-worker-{uuid4().hex[:12]}"

    @staticmethod
    def _mode(mode: str | None) -> str:
        return "live" if str(mode or "").lower() == "live" else "paper"

    @staticmethod
    def _serialize(command: TradingControlCommand) -> dict[str, Any]:
        def stamp(value: datetime | None) -> str | None:
            return value.isoformat() if value is not None else None

        return {
            "command_id": command.id,
            "command_type": command.command_type,
            "status": command.status,
            "mode": command.execution_mode,
            "position_id": command.position_id,
            "reason": command.reason,
            "worker_id": command.worker_id,
            "created_at": stamp(command.created_at),
            "claimed_at": stamp(command.claimed_at),
            "completed_at": stamp(command.completed_at),
            "result": command.result,
            "failure_reason": command.failure_reason,
        }

    async def enqueue_close_all(self, *, mode: str | None, reason: str | None) -> dict[str, Any]:
        return await self._enqueue(
            command_type=COMMAND_CLOSE_ALL_POSITIONS,
            mode=mode,
            position_id=None,
            reason=reason,
        )

    async def enqueue_close_position(
        self,
        *,
        position_id: int,
        mode: str | None,
        reason: str | None,
    ) -> dict[str, Any]:
        return await self._enqueue(
            command_type=COMMAND_CLOSE_POSITION,
            mode=mode,
            position_id=int(position_id),
            reason=reason,
        )

    async def _enqueue(
        self,
        *,
        command_type: str,
        mode: str | None,
        position_id: int | None,
        reason: str | None,
    ) -> dict[str, Any]:
        command = TradingControlCommand(
            id=str(uuid4()),
            command_type=command_type,
            status=STATUS_QUEUED,
            execution_mode=self._mode(mode),
            position_id=position_id,
            position_key=int(position_id or 0),
            reason=(reason or "").strip()[:2000] or None,
        )
        async with get_session_ctx() as session:
            session.add(command)
            try:
                await session.commit()
            except IntegrityError:
                # The database partial unique index is the authority here:
                # another request has already asked this one worker to close
                # the same target and it must remain a single exchange action.
                await session.rollback()
                active = await session.scalars(
                    select(TradingControlCommand)
                    .where(
                        TradingControlCommand.command_type == command_type,
                        TradingControlCommand.execution_mode == self._mode(mode),
                        TradingControlCommand.position_key == int(position_id or 0),
                        TradingControlCommand.status.in_((STATUS_QUEUED, STATUS_RUNNING)),
                    )
                    .order_by(TradingControlCommand.created_at.asc())
                    .limit(1)
                )
                existing = active.first()
                if existing is None:
                    raise
                return self._serialize(existing)
            await session.refresh(command)
            return self._serialize(command)

    async def get(self, command_id: str) -> dict[str, Any] | None:
        async with get_session_ctx() as session:
            command = await session.get(TradingControlCommand, str(command_id))
            return self._serialize(command) if command is not None else None

    async def recover_interrupted_commands(self) -> int:
        """Freeze commands left running by a stopped worker; never replay them."""
        async with get_session_ctx() as session:
            rows = await session.scalars(
                select(TradingControlCommand).where(
                    TradingControlCommand.status == STATUS_RUNNING
                )
            )
            commands = list(rows)
            now = datetime.now(UTC)
            for command in commands:
                command.status = STATUS_REQUIRES_RECONFIRMATION
                command.completed_at = now
                command.failure_reason = (
                    "Trading worker stopped before this command could be verified. "
                    "Reconcile OKX positions, then submit a new command if a close is still needed."
                )
            if commands:
                await session.commit()
            return len(commands)

    async def _claim_next(self) -> dict[str, Any] | None:
        """Claim exactly one command inside the owning worker transaction."""
        async with get_session_ctx() as session:
            statement = (
                select(TradingControlCommand)
                .where(TradingControlCommand.status == STATUS_QUEUED)
                .order_by(TradingControlCommand.created_at.asc())
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            command = (await session.scalars(statement)).first()
            if command is None:
                return None
            command.status = STATUS_RUNNING
            command.worker_id = self.worker_id
            command.claimed_at = datetime.now(UTC)
            await session.commit()
            return self._serialize(command)

    async def _complete(self, command_id: str, result: dict[str, Any]) -> None:
        async with get_session_ctx() as session:
            command = await session.get(TradingControlCommand, command_id)
            if command is None or command.status != STATUS_RUNNING:
                return
            command.status = STATUS_COMPLETED
            command.completed_at = datetime.now(UTC)
            command.result = result
            command.failure_reason = None
            await session.commit()

    async def _require_reconfirmation(self, command_id: str, exc: BaseException) -> None:
        async with get_session_ctx() as session:
            command = await session.get(TradingControlCommand, command_id)
            if command is None or command.status != STATUS_RUNNING:
                return
            command.status = STATUS_REQUIRES_RECONFIRMATION
            command.completed_at = datetime.now(UTC)
            command.failure_reason = (
                "The close request ended unexpectedly and was not retried automatically. "
                f"Reconcile OKX before submitting another command: {safe_error_text(exc)}"
            )
            await session.commit()

    async def process_once(
        self,
        *,
        close_all: Callable[..., Awaitable[dict[str, Any]]],
        close_position: Callable[..., Awaitable[dict[str, Any]]],
    ) -> bool:
        """Process one command and report whether useful work was found."""
        command = await self._claim_next()
        if command is None:
            return False
        try:
            if command["command_type"] == COMMAND_CLOSE_ALL_POSITIONS:
                result = await close_all(mode=command["mode"], reason=command["reason"])
            elif command["command_type"] == COMMAND_CLOSE_POSITION:
                result = await close_position(
                    int(command["position_id"]), reason=command["reason"]
                )
            else:
                raise ValueError(f"unsupported command type: {command['command_type']}")
        except Exception as exc:
            logger.exception(
                "trading control command requires reconfirmation",
                command_id=command["command_id"],
                error=safe_error_text(exc),
            )
            await self._require_reconfirmation(command["command_id"], exc)
            return True
        await self._complete(command["command_id"], result)
        return True
