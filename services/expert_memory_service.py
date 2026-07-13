from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import select

from ai_brain.base_model import DecisionOutput
from config.settings import ENSEMBLE_TRADER_NAME, FIXED_AI_MODEL_SLOTS, settings
from core.safe_output import safe_error_text
from db.repositories.memory_repo import MemoryRepository
from db.session import get_session_ctx
from models.learning import TradeReflection
from models.trade import Order, Position
from services.manual_close_marker import position_has_manual_close_order
from services.memory_feedback import MemoryFeedbackPolicy
from services.trade_fact_trust import closed_position_trade_fact_trusted

logger = structlog.get_logger(__name__)


def _exchange_order_ids(value: Any) -> tuple[str, ...]:
    tokens = {
        token.strip()
        for token in str(value or "").replace(";", ",").split(",")
        if token.strip()
    }
    return tuple(sorted(tokens))


def _reflection_lifecycle_key(position: Any) -> str:
    entry_ids = _exchange_order_ids(getattr(position, "entry_exchange_order_id", None))
    close_ids = _exchange_order_ids(getattr(position, "close_exchange_order_id", None))
    if entry_ids or close_ids:
        return "|".join(
            (
                str(getattr(position, "execution_mode", "") or ""),
                str(getattr(position, "symbol", "") or ""),
                str(getattr(position, "side", "") or ""),
                ",".join(entry_ids),
                ",".join(close_ids),
            )
        )
    return f"position:{int(getattr(position, 'id', 0) or 0)}"


def _reflection_position_rank(position: Any) -> tuple[int, int, int, int]:
    settlement_source = str(getattr(position, "settlement_source", "") or "").lower()
    settlement_status = str(getattr(position, "settlement_status", "") or "").lower()
    source_rank = {
        "okx_position_history": 4,
        "okx_position_history_settlement": 3,
        "okx_order_fact_sync": 2,
        "system_execution": 1,
    }.get(settlement_source, 0)
    status_rank = int(settlement_status in {"okx_position_history", "reconciled", "settled"})
    link_rank = int(bool(_exchange_order_ids(getattr(position, "entry_exchange_order_id", None))))
    link_rank += int(bool(_exchange_order_ids(getattr(position, "close_exchange_order_id", None))))
    return source_rank, status_rank, link_rank, int(getattr(position, "id", 0) or 0)


class ExpertMemoryService:
    """Own expert memory retrieval, weight calibration, and trade reflections."""

    def __init__(
        self,
        *,
        session_factory: Callable[[], AbstractAsyncContextManager[Any]] = get_session_ctx,
        memory_enabled_provider: Callable[[], bool] | None = None,
        model_slots: Sequence[dict[str, Any]] | None = None,
        ensemble_model_name: str = ENSEMBLE_TRADER_NAME,
    ) -> None:
        self.session_factory = session_factory
        self.memory_enabled_provider = memory_enabled_provider or (
            lambda: bool(settings.expert_memory_enabled)
        )
        self.model_slots = tuple(model_slots or FIXED_AI_MODEL_SLOTS)
        self.ensemble_model_name = ensemble_model_name
        self.memory_feedback_policy = MemoryFeedbackPolicy()

    async def context(self, symbol: str) -> dict[str, Any]:
        """Fetch compact long-term memories and expert weight hints for prompts."""

        if not self.memory_enabled_provider():
            return _empty_memory_context()

        by_expert: dict[str, list[dict[str, Any]]] = {}
        flat: list[dict[str, Any]] = []
        used_ids: list[int] = []
        try:
            async with self.session_factory() as session:
                repo = MemoryRepository(session)
                for slot in self.model_slots:
                    expert_name = str(slot.get("name") or "")
                    if not expert_name:
                        continue
                    rows = await repo.get_relevant_memories(
                        expert_name=expert_name,
                        symbol=symbol,
                    )
                    serialized = [serialize_memory(row) for row in rows]
                    if serialized:
                        by_expert[expert_name] = serialized
                        flat.extend(serialized)
                        used_ids.extend([row.id for row in rows if row.id])
                await repo.mark_memories_used(used_ids)
        except Exception as exc:
            logger.warning(
                "failed to fetch expert memories",
                symbol=symbol,
                error=safe_error_text(exc),
            )
            return _empty_memory_context()

        return {
            "expert_memories": by_expert,
            "expert_memories_flat": flat,
            "memory_feedback": self.memory_feedback_policy.build(flat),
        }

    async def record_trade_reflection_in_session(
        self,
        session: Any,
        pos: Any,
        *,
        exit_price: float,
        entry_fee: float,
        close_fee: float,
        gross_pnl: float,
        source: str,
        decision: DecisionOutput | None = None,
    ) -> bool:
        """Create a compact post-trade reflection and update expert memories."""

        if not self.memory_enabled_provider():
            return False
        if not closed_position_trade_fact_trusted(pos):
            logger.warning(
                "skip trade reflection for untrusted closed position fact",
                position_id=getattr(pos, "id", None),
                symbol=getattr(pos, "symbol", None),
            )
            return False
        try:
            realized_pnl = float(pos.realized_pnl or 0.0)
            entry_price = float(pos.entry_price or 0.0)
            quantity = float(pos.quantity or 0.0)
            notional = abs(entry_price * quantity)
            pnl_pct = realized_pnl / notional if notional > 0 else 0.0
            funding_fee = float(getattr(pos, "funding_fee", 0.0) or 0.0)
            cost_complete = bool(
                notional > 0
                and entry_fee is not None
                and close_fee is not None
                and hasattr(pos, "funding_fee")
            )
            hold_minutes = position_hold_minutes(pos)
            outcome = "profit" if realized_pnl > 0 else "loss" if realized_pnl < 0 else "flat"
            pattern = reflection_pattern(pos, pnl_pct, hold_minutes)
            mistake, improvement = reflection_summary(pos, outcome, pnl_pct, hold_minutes)
            expert_lessons = build_expert_lessons(
                pos=pos,
                outcome=outcome,
                pnl_pct=pnl_pct,
                hold_minutes=hold_minutes,
                pattern=pattern,
                decision=decision,
                model_slots=self.model_slots,
            )
            repo = MemoryRepository(session)
            reflection = await repo.create_reflection(
                {
                    "position_id": int(pos.id or 0),
                    "model_name": pos.model_name,
                    "execution_mode": pos.execution_mode,
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "entry_price": entry_price,
                    "exit_price": float(exit_price or 0.0),
                    "quantity": quantity,
                    "realized_pnl": realized_pnl,
                    "fee_estimate": abs(float(entry_fee or 0.0)) + abs(float(close_fee or 0.0)),
                    "hold_minutes": hold_minutes,
                    "closed_at": getattr(pos, "closed_at", None),
                    "outcome": outcome,
                    "mistake_summary": mistake,
                    "improvement_summary": improvement,
                    "expert_lessons": expert_lessons,
                    "source": source,
                }
            )
            if reflection is None:
                reflection = await repo.get_reflection_by_position_id(int(pos.id or 0))
            if reflection is None:
                return False

            for lesson in expert_lessons.values():
                await repo.upsert_memory(
                    {
                        **lesson,
                        "source_position_id": int(pos.id or 0),
                        "extra": {
                            "reflection_id": reflection.id,
                            "realized_pnl": realized_pnl,
                            "pnl_pct": pnl_pct,
                            "pnl_pct_deprecated_ratio": True,
                            "net_return_after_cost_pct": pnl_pct * 100.0,
                            "objective": "maximize_expected_realized_net_return_after_cost",
                            "objective_version": "2026-07-12.v1",
                            "cost_complete": cost_complete,
                            "production_evidence_eligible": cost_complete,
                            "source": source,
                            "hold_minutes": hold_minutes,
                            "gross_pnl": gross_pnl,
                            "entry_fee": entry_fee,
                            "close_fee": close_fee,
                            "funding_fee": funding_fee,
                            "source_position_id": int(pos.id or 0),
                            "settlement_status": getattr(pos, "settlement_status", None),
                            "settlement_source": getattr(pos, "settlement_source", None),
                        },
                    }
                )
            return True
        except Exception as exc:
            logger.warning(
                "failed to record trade reflection",
                position_id=getattr(pos, "id", None),
                symbol=getattr(pos, "symbol", None),
                error=safe_error_text(exc),
            )
            return False

    async def backfill_trade_reflections(self, execution_mode: str) -> dict[str, Any]:
        """Create expert memories from already closed positions after restart."""

        if not self.memory_enabled_provider():
            return {"status": "disabled", "processed": 0}
        try:
            async with self.session_factory() as session:
                rows = list(
                    (
                        await session.execute(
                            select(Position)
                            .where(
                                Position.execution_mode == execution_mode,
                                Position.is_open.is_(False),
                                Position.closed_at.is_not(None),
                            )
                            .order_by(Position.closed_at.desc(), Position.id.desc())
                            .limit(1000)
                        )
                    )
                    .scalars()
                    .all()
                )
                symbols = {pos.symbol for pos in rows if pos.symbol}
                manual_close_orders = []
                if symbols:
                    manual_close_result = await session.execute(
                        select(Order).where(
                            Order.execution_mode == execution_mode,
                            Order.status == "filled",
                            Order.symbol.in_(symbols),
                            Order.exchange_order_id.like("manual_close:%"),
                        )
                    )
                    manual_close_orders = list(manual_close_result.scalars().all())
                eligible = [
                    pos
                    for pos in rows
                    if pos.closed_at
                    and not position_has_manual_close_order(pos, manual_close_orders)
                    and closed_position_trade_fact_trusted(pos)
                ]
                reflection_rows = list(
                    (
                        await session.execute(
                            select(TradeReflection).where(
                                TradeReflection.position_id.in_(
                                    [int(pos.id) for pos in eligible if int(pos.id or 0) > 0]
                                )
                            )
                        )
                    )
                    .scalars()
                    .all()
                ) if eligible else []
                reflected_position_ids = {
                    int(row.position_id) for row in reflection_rows if int(row.position_id or 0) > 0
                }
                grouped: dict[str, list[Any]] = {}
                for pos in eligible:
                    grouped.setdefault(_reflection_lifecycle_key(pos), []).append(pos)

                processed = 0
                for group in grouped.values():
                    pos = next(
                        (
                            row
                            for row in group
                            if int(row.id or 0) in reflected_position_ids
                        ),
                        max(group, key=_reflection_position_rank),
                    )
                    recorded = await self.record_trade_reflection_in_session(
                        session,
                        pos,
                        exit_price=float(pos.current_price or pos.entry_price or 0.0),
                        entry_fee=float(getattr(pos, "entry_fee", 0.0) or 0.0),
                        close_fee=float(getattr(pos, "close_fee", 0.0) or 0.0),
                        gross_pnl=float(
                            getattr(pos, "close_fill_pnl", 0.0)
                            or getattr(pos, "realized_pnl", 0.0)
                            or 0.0
                        ),
                        source="authoritative_settlement_backfill",
                        decision=None,
                    )
                    processed += int(recorded)
                report = {
                    "status": "completed",
                    "scanned": len(rows),
                    "eligible": len(eligible),
                    "unique_lifecycles": len(grouped),
                    "duplicate_rows_skipped": len(eligible) - len(grouped),
                    "processed": processed,
                }
                logger.info("trade reflection backfill completed", **report)
                return report
        except Exception as exc:
            logger.warning("failed to backfill trade reflections", error=safe_error_text(exc))
            return {"status": "error", "processed": 0, "error": safe_error_text(exc)}


def serialize_memory(memory: Any) -> dict[str, Any]:
    return {
        "id": memory.id,
        "expert_name": memory.expert_name,
        "expert_label": memory.expert_label,
        "symbol": memory.symbol,
        "side": memory.side,
        "memory_type": memory.memory_type,
        "market_pattern": memory.market_pattern,
        "lesson": memory.lesson,
        "recommended_action": memory.recommended_action,
        "evidence_count": int(memory.evidence_count or 0),
        "success_count": int(getattr(memory, "success_count", 0) or 0),
        "failure_count": int(getattr(memory, "failure_count", 0) or 0),
        "confidence_score": float(memory.confidence_score or 0.0),
        "extra": memory.extra or {},
        "created_at": memory.created_at.isoformat() if memory.created_at else None,
        "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
    }


def position_hold_minutes(pos: Any) -> float:
    opened = getattr(pos, "created_at", None)
    closed = getattr(pos, "closed_at", None) or datetime.now(UTC)
    if opened is None:
        return 0.0
    if opened.tzinfo is None:
        opened = opened.replace(tzinfo=UTC)
    if closed.tzinfo is None:
        closed = closed.replace(tzinfo=UTC)
    return max((closed - opened).total_seconds() / 60.0, 0.0)


def reflection_pattern(pos: Any, pnl_pct: float, hold_minutes: float) -> str:
    side_label = "做多" if str(pos.side).lower() == "long" else "做空"
    leverage = float(getattr(pos, "leverage", 1.0) or 1.0)
    return (
        f"{pos.symbol} {side_label}，费后收益率={pnl_pct:.6%}，"
        f"持仓分钟={hold_minutes:.2f}，杠杆={leverage:.2f}x"
    )


def reflection_summary(
    pos: Any,
    outcome: str,
    pnl_pct: float,
    hold_minutes: float,
) -> tuple[str, str]:
    side_label = "做多" if str(pos.side).lower() == "long" else "做空"
    observation = (
        f"{pos.symbol} {side_label} 权威结算结果={outcome}，"
        f"费后收益率={pnl_pct:.6%}，持仓分钟={hold_minutes:.2f}。"
    )
    return observation, "仅作为训练与复盘事实；不得直接调整方向、仓位、杠杆或退出。"


def build_expert_lessons(
    *,
    pos: Any,
    outcome: str,
    pnl_pct: float,
    hold_minutes: float,
    pattern: str,
    decision: DecisionOutput | None = None,
    model_slots: Sequence[dict[str, Any]] = FIXED_AI_MODEL_SLOTS,
) -> dict[str, dict[str, Any]]:
    del decision
    side = str(pos.side or "").lower()
    symbol = str(pos.symbol or "")
    is_profit = outcome == "profit"
    is_loss = outcome == "loss"
    memory_type = "fee_after_outcome_observation"
    evidence_success = 1 if is_profit else 0
    evidence_failure = 1 if is_loss else 0

    labels = {
        str(slot["name"]): slot.get("label", slot["name"])
        for slot in model_slots
        if slot.get("name")
    }
    base_key = f"{symbol}|{side}|{memory_type}|{lesson_bucket(pnl_pct, hold_minutes)}"
    lesson = (
        f"权威费后结果事实：symbol={symbol}, side={side}, outcome={outcome}, "
        f"net_return_after_cost_pct={pnl_pct * 100.0:.8f}, "
        f"hold_minutes={hold_minutes:.4f}, pattern={pattern}。"
    )
    result: dict[str, dict[str, Any]] = {}
    for expert_name in labels:
        result[expert_name] = {
            "expert_name": expert_name,
            "expert_label": labels.get(expert_name, expert_name),
            "symbol": symbol,
            "side": side,
            "memory_type": memory_type,
            "market_pattern": pattern,
            "lesson": lesson,
            "recommended_action": "observation_only",
            "evidence_count": 1,
            "success_count": evidence_success,
            "failure_count": evidence_failure,
            "memory_key": f"{expert_name}|{base_key}",
        }
    return result


def lesson_bucket(pnl_pct: float, hold_minutes: float) -> str:
    del pnl_pct, hold_minutes
    return "canonical_fee_after_outcome"


def _empty_memory_context() -> dict[str, Any]:
    return {
        "expert_memories": {},
        "expert_memories_flat": [],
        "memory_feedback": MemoryFeedbackPolicy().build([]),
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
