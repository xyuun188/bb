"""Recover pending exit decisions that were recorded but not executed."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import structlog
from sqlalchemy import func, or_, select

from ai_brain.base_model import Action, DecisionOutput
from core.safe_output import safe_error_text
from db.session import get_session_ctx
from models.decision import AIDecision
from models.trade import Order

logger = structlog.get_logger(__name__)

PendingExitLoader = Callable[[datetime], Awaitable[list[dict[str, Any]]]]
LoopStageSetter = Callable[[str], None]
CandidateExecutor = Callable[..., Awaitable[Any]]
Clock = Callable[[], datetime]

PENDING_EXIT_REASON_PREFIX = "本轮还在分析或排队中"


def _legacy_prefix_variants(prefix: str) -> tuple[str, ...]:
    damaged = prefix.encode("utf-8").decode("gbk", errors="replace")
    variants = (
        prefix,
        damaged,
        damaged.replace("\ufffd", "?"),
        damaged.replace("\ufffd", "\ue103"),
    )
    return tuple(dict.fromkeys(variants))


PENDING_EXIT_REASON_PREFIXES = _legacy_prefix_variants(PENDING_EXIT_REASON_PREFIX)


@dataclass(frozen=True, slots=True)
class PendingExitRecoveryResult:
    """Summary of one pending-exit recovery pass."""

    loaded: int = 0
    executed: int = 0
    skipped: int = 0
    failed: bool = False
    error: str | None = None


async def load_pending_exit_decision_rows(cutoff: datetime) -> list[dict[str, Any]]:
    """Load recent exit decisions that have no order and no final execution reason."""

    pending: list[dict[str, Any]] = []
    async with get_session_ctx() as session:
        stmt = (
            select(AIDecision)
            .where(
                AIDecision.was_executed.is_(False),
                AIDecision.action.in_([Action.CLOSE_LONG.value, Action.CLOSE_SHORT.value]),
                AIDecision.created_at >= cutoff.replace(tzinfo=None),
                or_(
                    AIDecision.execution_reason.is_(None),
                    AIDecision.execution_reason == "",
                    *[
                        AIDecision.execution_reason.like(f"{prefix}%")
                        for prefix in PENDING_EXIT_REASON_PREFIXES
                    ],
                ),
            )
            .order_by(AIDecision.created_at.asc())
            .limit(10)
        )
        rows = list((await session.execute(stmt)).scalars().all())
        for row in rows:
            order_count = (
                await session.execute(
                    select(func.count(Order.id)).where(Order.decision_id == row.id)
                )
            ).scalar() or 0
            if order_count > 0:
                continue
            pending.append(
                {
                    "id": row.id,
                    "model_name": row.model_name,
                    "symbol": row.symbol,
                    "action": row.action,
                    "confidence": row.confidence,
                    "reasoning": row.reasoning or "",
                    "position_size_pct": row.position_size_pct,
                    "suggested_leverage": row.suggested_leverage,
                    "stop_loss_pct": row.stop_loss_pct,
                    "take_profit_pct": row.take_profit_pct,
                    "raw_response": row.raw_llm_response,
                    "feature_snapshot": row.feature_snapshot,
                    "created_at": row.created_at,
                }
            )
    return pending


@dataclass(frozen=True, slots=True)
class PendingExitDecisionRecoveryProcessor:
    """Rebuild and execute pending exit decisions through an execution boundary."""

    set_loop_stage: LoopStageSetter
    candidate_executor: CandidateExecutor
    pending_loader: PendingExitLoader = load_pending_exit_decision_rows
    clock: Clock = lambda: datetime.now(UTC)
    lookback: timedelta = timedelta(minutes=10)

    async def recover(
        self,
        *,
        results: dict[str, Any],
        open_positions: list[dict[str, Any]],
        round_decision_ids: set[int],
    ) -> PendingExitRecoveryResult:
        cutoff = self.clock() - self.lookback
        try:
            pending = await self.pending_loader(cutoff)
        except Exception as exc:
            error_text = safe_error_text(exc, limit=160)
            logger.error("failed to load pending exit decisions", error=error_text)
            return PendingExitRecoveryResult(failed=True, error=error_text)

        if not pending:
            return PendingExitRecoveryResult()

        self.set_loop_stage("recover_pending_exits")
        logger.warning("recovering pending exit decisions", count=len(pending))

        executed = 0
        skipped = 0
        for item in pending:
            action = Action.from_string(str(item.get("action", "")))
            if not action.is_exit():
                skipped += 1
                continue

            decision = self._decision_from_row(item, action)
            decision_db_id = int(item["id"])
            round_decision_ids.add(decision_db_id)
            await self.candidate_executor(
                decision.symbol,
                decision.model_name,
                decision,
                SimpleNamespace(warnings=[]),
                decision_db_id,
                results,
                open_positions=open_positions,
            )
            executed += 1

        return PendingExitRecoveryResult(
            loaded=len(pending),
            executed=executed,
            skipped=skipped,
        )

    @staticmethod
    def _decision_from_row(item: dict[str, Any], action: Action) -> DecisionOutput:
        created_at = item.get("created_at")
        if isinstance(created_at, datetime):
            timestamp = created_at if created_at.tzinfo else created_at.replace(tzinfo=UTC)
        else:
            timestamp = datetime.now(UTC)

        return DecisionOutput(
            model_name=str(item["model_name"]),
            symbol=str(item["symbol"]),
            action=action,
            confidence=float(item.get("confidence") or 0.0),
            reasoning=str(item.get("reasoning") or ""),
            position_size_pct=float(item.get("position_size_pct") or 1.0),
            suggested_leverage=float(item.get("suggested_leverage") or 1.0),
            stop_loss_pct=float(item.get("stop_loss_pct") or 0.05),
            take_profit_pct=float(item.get("take_profit_pct") or 0.10),
            timestamp=timestamp,
            raw_response=(
                item["raw_response"] if isinstance(item.get("raw_response"), dict) else {}
            ),
            feature_snapshot=(
                item["feature_snapshot"] if isinstance(item.get("feature_snapshot"), dict) else {}
            ),
        )
