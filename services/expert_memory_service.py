from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from typing import Any

import structlog

from config.settings import ENSEMBLE_TRADER_NAME, FIXED_AI_MODEL_SLOTS, settings
from core.safe_output import safe_error_text
from db.repositories.memory_repo import MemoryRepository
from db.session import get_session_ctx
from services.authoritative_trade_outcome import (
    AUTHORITATIVE_TRADE_OUTCOME_AUTHORITY,
    AUTHORITATIVE_TRADE_OUTCOME_VERSION,
    build_authoritative_trade_outcome,
    load_authoritative_trade_outcomes,
)
from services.memory_feedback import MemoryFeedbackPolicy
from services.profit_training_contract import PROFIT_TRAINING_TARGET
from services.trade_fact_trust import closed_position_trade_fact_trusted
from services.training_epoch import load_training_epoch_start

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
        authoritative_outcome_loader: Callable[..., Any] = load_authoritative_trade_outcomes,
    ) -> None:
        self.session_factory = session_factory
        self.memory_enabled_provider = memory_enabled_provider or (
            lambda: bool(settings.expert_memory_enabled)
        )
        self.model_slots = tuple(model_slots or FIXED_AI_MODEL_SLOTS)
        self.ensemble_model_name = ensemble_model_name
        self.authoritative_outcome_loader = authoritative_outcome_loader
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
                expert_names = [
                    str(slot.get("name") or "")
                    for slot in self.model_slots
                    if str(slot.get("name") or "")
                ]
                memories_by_expert = await repo.get_relevant_memories_for_experts(
                    expert_names,
                    symbol,
                )
                for slot in self.model_slots:
                    expert_name = str(slot.get("name") or "")
                    if not expert_name:
                        continue
                    rows = memories_by_expert.get(expert_name, [])
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
        source: str,
    ) -> bool:
        """Create a reflection link that authoritative OKX backfill will replace."""

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
            hold_minutes = position_hold_minutes(pos)
            outcome = "profit" if realized_pnl > 0 else "loss" if realized_pnl < 0 else "flat"
            mistake, improvement = reflection_summary(pos, outcome, pnl_pct, hold_minutes)
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
                    "expert_lessons": {},
                    "source": source,
                }
            )
            if reflection is None:
                reflection = await repo.get_reflection_by_position_id(int(pos.id or 0))
            if reflection is None:
                return False
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
        """Bind reflections and memories to canonical OKX outcome events."""

        try:
            outcomes = await self.authoritative_outcome_loader(
                mode=execution_mode,
                since=load_training_epoch_start(),
            )
            complete_outcomes = [
                outcome
                for outcome in outcomes
                if outcome.get("settlement_fact_trusted") is True
                and outcome.get("outcome_complete") is True
            ]
            async with self.session_factory() as session:
                processed = 0
                for outcome in complete_outcomes:
                    processed += int(
                        await self._record_authoritative_outcome_in_session(session, outcome)
                    )
                report = {
                    "status": "completed",
                    "outcome_version": AUTHORITATIVE_TRADE_OUTCOME_VERSION,
                    "scanned": len(outcomes),
                    "eligible": len(complete_outcomes),
                    "unique_lifecycles": len(
                        {str(item.get("lifecycle_key") or "") for item in complete_outcomes}
                    ),
                    "complete_before_reflection_sync": len(complete_outcomes),
                    "processed": processed,
                }
                logger.info("trade reflection backfill completed", **report)
                return report
        except Exception as exc:
            logger.warning("failed to backfill trade reflections", error=safe_error_text(exc))
            return {"status": "error", "processed": 0, "error": safe_error_text(exc)}

    async def _record_authoritative_outcome_in_session(
        self,
        session: Any,
        outcome: dict[str, Any],
    ) -> bool:
        position_id = int(outcome.get("position_id") or 0)
        if position_id <= 0:
            return False
        repo = MemoryRepository(session)
        net_return_pct = _safe_float(outcome.get(PROFIT_TRAINING_TARGET), 0.0)
        realized_pnl = _safe_float(outcome.get("realized_pnl"), 0.0)
        result_label = "profit" if realized_pnl > 0 else "loss" if realized_pnl < 0 else "flat"
        symbol = str(outcome.get("symbol") or "")
        side = str(outcome.get("side") or "").lower()
        hold_minutes = _safe_float(outcome.get("holding_minutes"), 0.0)
        reflection_data = {
            "position_id": position_id,
            "model_name": str(outcome.get("model_name") or self.ensemble_model_name),
            "execution_mode": str(outcome.get("execution_mode") or "paper"),
            "symbol": symbol,
            "side": side,
            "entry_price": _safe_float(outcome.get("entry_price"), 0.0),
            "exit_price": _safe_float(outcome.get("close_price"), 0.0),
            "quantity": _safe_float(outcome.get("quantity"), 0.0),
            "realized_pnl": realized_pnl,
            "fee_estimate": _safe_float(outcome.get("entry_fee"), 0.0)
            + _safe_float(outcome.get("close_fee"), 0.0),
            "hold_minutes": hold_minutes,
            "closed_at": _parse_iso_datetime(outcome.get("label_timestamp")),
            "outcome": result_label,
            "mistake_summary": (
                f"{symbol} {side} authoritative fee-after outcome: "
                f"net PnL={realized_pnl:.8f} USDT, return={net_return_pct:.8f}%."
            ),
            "improvement_summary": (
                "The model must recalibrate expected return, uncertainty and tail execution risk "
                "from this outcome; it must not hard-code a direction from one trade."
            ),
            "source": "authoritative_trade_outcome",
        }
        reflection = await repo.get_reflection_by_position_id(position_id)
        if reflection is None:
            reflection = await repo.create_reflection(reflection_data)
        elif reflection is not None:
            reflection = await repo.update_reflection(reflection, reflection_data)
        if reflection is None:
            return False

        canonical = build_authoritative_trade_outcome(outcome, reflection=reflection)
        if canonical.get("outcome_complete") is not True:
            return False
        lesson_text = (
            f"Authoritative fee-after outcome {canonical['outcome_id']}: symbol={symbol}, "
            f"side={side}, result={result_label}, net_return_pct={net_return_pct:.8f}, "
            f"realized_net_pnl_usdt={realized_pnl:.8f}."
        )
        for slot in self.model_slots:
            expert_name = str(slot.get("name") or "")
            if not expert_name:
                continue
            await repo.upsert_memory(
                {
                    "expert_name": expert_name,
                    "expert_label": str(slot.get("label") or expert_name),
                    "symbol": symbol,
                    "side": side,
                    "memory_type": "authoritative_trade_outcome",
                    "market_pattern": f"{symbol}|{side}|authoritative_fee_after_outcome",
                    "lesson": lesson_text,
                    "recommended_action": "observation_only_revalidate_distribution",
                    "evidence_count": 1,
                    "success_count": int(realized_pnl > 0),
                    "failure_count": int(realized_pnl < 0),
                    "memory_key": f"{expert_name}|{symbol}|{side}|authoritative_trade_outcome",
                    "source_position_id": position_id,
                    "extra": {
                        "reflection_id": int(reflection.id or 0),
                        "source": "authoritative_trade_outcome",
                        "source_position_id": position_id,
                        "outcome_id": canonical["outcome_id"],
                        "outcome_version": canonical["outcome_version"],
                        "outcome_fingerprint": canonical["outcome_fingerprint"],
                        "authority_level": AUTHORITATIVE_TRADE_OUTCOME_AUTHORITY,
                        "authority_rank": 100,
                        "realized_pnl": realized_pnl,
                        "net_return_after_all_cost_pct": net_return_pct,
                        "objective": "maximize_expected_realized_net_return_after_cost",
                        "objective_version": "2026-07-12.v1",
                        "cost_complete": True,
                        "production_evidence_eligible": True,
                        "hold_minutes": hold_minutes,
                        "attribution": canonical.get("attribution"),
                        "counterfactual_production_weight": 0.0,
                    },
                }
            )
        return True


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


def reflection_summary(
    pos: Any,
    outcome: str,
    pnl_pct: float,
    hold_minutes: float,
) -> tuple[str, str]:
    side_label = "做多" if str(pos.side).lower() == "long" else "做空"
    observation = (
        f"{pos.symbol} {side_label} 本地平仓记录={outcome}，"
        f"暂存收益率={pnl_pct:.6%}，持仓分钟={hold_minutes:.2f}；等待 OKX 权威结果回写。"
    )
    return observation, "暂存记录不进入专家记忆、训练或晋升；仅等待权威 outcome 覆盖。"


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


def _parse_iso_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
