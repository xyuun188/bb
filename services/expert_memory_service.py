from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import structlog
from sqlalchemy import select

from ai_brain.base_model import DecisionOutput
from config.settings import ENSEMBLE_TRADER_NAME, FIXED_AI_MODEL_SLOTS, settings
from core.safe_output import safe_error_text
from db.repositories.memory_repo import MemoryRepository
from db.repositories.trade_repo import TradeRepository
from db.session import get_session_ctx
from models.decision import AIDecision
from models.trade import Order, Position
from services.manual_close_marker import position_has_manual_close_order
from services.memory_feedback import MemoryFeedbackPolicy

logger = structlog.get_logger(__name__)


class ExpertMemoryService:
    """Own expert memory retrieval, weight calibration, and trade reflections."""

    def __init__(
        self,
        *,
        session_factory: Callable[[], AbstractAsyncContextManager[Any]] = get_session_ctx,
        memory_enabled_provider: Callable[[], bool] | None = None,
        memory_limit_provider: Callable[[], int] | None = None,
        model_slots: Sequence[dict[str, Any]] | None = None,
        ensemble_model_name: str = ENSEMBLE_TRADER_NAME,
    ) -> None:
        self.session_factory = session_factory
        self.memory_enabled_provider = memory_enabled_provider or (
            lambda: bool(settings.expert_memory_enabled)
        )
        self.memory_limit_provider = memory_limit_provider or (
            lambda: int(settings.expert_memory_per_prompt or 4)
        )
        self.model_slots = tuple(model_slots or FIXED_AI_MODEL_SLOTS)
        self.ensemble_model_name = ensemble_model_name
        self._realized_weight_cache: dict[str, Any] = {"expires_at": None, "weights": {}}
        self.memory_feedback_policy = MemoryFeedbackPolicy()

    async def context(self, symbol: str) -> dict[str, Any]:
        """Fetch compact long-term memories and expert weight hints for prompts."""

        if not self.memory_enabled_provider():
            return _empty_memory_context()

        limit = max(1, int(self.memory_limit_provider() or 4))
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
                        limit=limit,
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

        dynamic_weights = dynamic_expert_weights_from_memories(by_expert, self.model_slots)
        realized_weights = await self.realized_expert_weight_adjustments()
        for expert_name, realized in realized_weights.items():
            if expert_name not in dynamic_weights:
                dynamic_weights[expert_name] = realized
                continue
            current = dynamic_weights[expert_name]
            base_weight = _safe_float(current.get("base_weight"), realized.get("base_weight", 1.0))
            memory_multiplier = _safe_float(current.get("multiplier"), 1.0)
            realized_multiplier = _safe_float(realized.get("multiplier"), 1.0)
            combined = min(max(memory_multiplier * realized_multiplier, 0.65), 1.30)
            current.update(
                {
                    "multiplier": round(combined, 4),
                    "effective_weight": round(base_weight * combined, 4),
                    "realized_pnl": realized.get("realized_pnl", 0.0),
                    "realized_count": realized.get("realized_count", 0),
                    "reason": (
                        f"{current.get('reason') or ''} 实盘/模拟已实现盈亏校准："
                        f"{realized.get('reason') or '暂无'}"
                    ),
                }
            )

        return {
            "expert_memories": by_expert,
            "expert_memories_flat": flat,
            "dynamic_expert_weights": dynamic_weights,
            "memory_feedback": self.memory_feedback_policy.build(flat),
        }

    async def realized_expert_weight_adjustments(self) -> dict[str, dict[str, Any]]:
        """Calibrate expert weights from today's realized same-side PnL."""

        now = datetime.now(UTC)
        expires_at = self._realized_weight_cache.get("expires_at")
        if isinstance(expires_at, datetime) and expires_at > now:
            return self._realized_weight_cache.get("weights") or {}

        slot_weights = {
            str(slot.get("name") or ""): float(slot.get("weight", 1.0) or 1.0)
            for slot in self.model_slots
            if slot.get("name")
        }
        stats: dict[str, dict[str, Any]] = {
            name: {"pnl": 0.0, "profit": 0.0, "loss": 0.0, "count": 0, "wins": 0, "losses": 0}
            for name in slot_weights
        }
        start_utc = (
            datetime.now(timezone(timedelta(hours=8)))
            .replace(hour=0, minute=0, second=0, microsecond=0)
            .astimezone(UTC)
        )

        try:
            async with self.session_factory() as session:
                positions_result = await session.execute(
                    select(Position)
                    .where(
                        Position.model_name == self.ensemble_model_name,
                        Position.is_open.is_(False),
                        Position.closed_at.is_not(None),
                        Position.closed_at >= start_utc,
                    )
                    .order_by(Position.closed_at.desc())
                    .limit(800)
                )
                positions = list(positions_result.scalars().all())
                if not positions:
                    self._realized_weight_cache = {
                        "expires_at": now + timedelta(minutes=15),
                        "weights": {},
                    }
                    return {}

                symbols = {pos.symbol for pos in positions if pos.symbol}
                manual_close_orders = []
                if symbols:
                    manual_close_result = await session.execute(
                        select(Order).where(
                            Order.model_name == self.ensemble_model_name,
                            Order.status == "filled",
                            Order.symbol.in_(symbols),
                            Order.exchange_order_id.like("manual_close:%"),
                        )
                    )
                    manual_close_orders = list(manual_close_result.scalars().all())
                positions = [
                    pos
                    for pos in positions
                    if not position_has_manual_close_order(pos, manual_close_orders)
                ]
                if not positions:
                    self._realized_weight_cache = {
                        "expires_at": now + timedelta(minutes=15),
                        "weights": {},
                    }
                    return {}
                symbols = {pos.symbol for pos in positions if pos.symbol}
                order_symbol_filter = Order.symbol.in_(symbols) if symbols else Order.id == -1
                orders_result = await session.execute(
                    select(Order)
                    .where(
                        Order.model_name == self.ensemble_model_name,
                        Order.status == "filled",
                        Order.decision_id.is_not(None),
                        order_symbol_filter,
                    )
                    .order_by(Order.filled_at.desc(), Order.created_at.desc())
                    .limit(2400)
                )
                orders = list(orders_result.scalars().all())
                decision_ids = [order.decision_id for order in orders if order.decision_id]
                decisions: dict[int, AIDecision] = {}
                if decision_ids:
                    decisions_result = await session.execute(
                        select(AIDecision).where(AIDecision.id.in_(decision_ids))
                    )
                    decisions = {
                        decision.id: decision for decision in decisions_result.scalars().all()
                    }
        except Exception as exc:
            logger.warning(
                "failed to calculate realized expert weights",
                error=safe_error_text(exc),
            )
            return {}

        for pos in positions:
            pos_created = _aware_utc(pos.created_at)
            pos_side = str(pos.side or "").lower()
            candidates: list[tuple[float, AIDecision]] = []
            for order in orders:
                if order.symbol != pos.symbol or order.decision_id not in decisions:
                    continue
                decision = decisions[order.decision_id]
                decision_side = _entry_side_from_action(decision.action)
                if decision_side != pos_side:
                    continue
                order_time = _aware_utc(order.filled_at or order.created_at)
                if (
                    pos_created
                    and order_time
                    and abs((order_time - pos_created).total_seconds()) > 180
                ):
                    continue
                distance = (
                    abs(((order_time or pos_created) - pos_created).total_seconds())
                    if pos_created and order_time
                    else 0.0
                )
                candidates.append((distance, decision))
            if not candidates:
                continue
            _, decision = sorted(candidates, key=lambda item: item[0])[0]
            raw = _safe_dict(decision.raw_llm_response)
            opinions = _safe_list(raw.get("opinions"))
            pnl = float(pos.realized_pnl or 0.0)
            for opinion in opinions:
                if not isinstance(opinion, dict):
                    continue
                name = str(opinion.get("model_name") or "")
                if name not in stats:
                    continue
                action = str(opinion.get("action") or "").lower()
                if action != pos_side:
                    continue
                bucket = stats[name]
                bucket["pnl"] += pnl
                bucket["count"] += 1
                if pnl >= 0:
                    bucket["wins"] += 1
                    bucket["profit"] += pnl
                else:
                    bucket["losses"] += 1
                    bucket["loss"] += abs(pnl)

        result = _realized_weight_result(stats, slot_weights)
        self._realized_weight_cache = {
            "expires_at": now + timedelta(minutes=15),
            "weights": result,
        }
        return result

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
    ) -> None:
        """Create a compact post-trade reflection and update expert memories."""

        if not self.memory_enabled_provider():
            return
        try:
            realized_pnl = float(pos.realized_pnl or 0.0)
            entry_price = float(pos.entry_price or 0.0)
            quantity = float(pos.quantity or 0.0)
            notional = abs(entry_price * quantity)
            pnl_pct = realized_pnl / notional if notional > 0 else 0.0
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
                return

            for lesson in expert_lessons.values():
                await repo.upsert_memory(
                    {
                        **lesson,
                        "source_position_id": int(pos.id or 0),
                        "extra": {
                            "reflection_id": reflection.id,
                            "realized_pnl": realized_pnl,
                            "pnl_pct": pnl_pct,
                            "hold_minutes": hold_minutes,
                            "gross_pnl": gross_pnl,
                            "entry_fee": entry_fee,
                            "close_fee": close_fee,
                        },
                    }
                )
        except Exception as exc:
            logger.warning(
                "failed to record trade reflection",
                position_id=getattr(pos, "id", None),
                symbol=getattr(pos, "symbol", None),
                error=safe_error_text(exc),
            )

    async def backfill_trade_reflections(self, execution_mode: str) -> None:
        """Create expert memories from already closed positions after restart."""

        if not self.memory_enabled_provider():
            return
        try:
            async with self.session_factory() as session:
                repo = TradeRepository(session)
                rows = await repo.get_position_records(
                    execution_mode=execution_mode,
                    model_name=self.ensemble_model_name,
                    is_open=False,
                    limit=200,
                )
                symbols = {pos.symbol for pos in rows if pos.symbol}
                manual_close_orders = []
                if symbols:
                    manual_close_result = await session.execute(
                        select(Order).where(
                            Order.model_name == self.ensemble_model_name,
                            Order.execution_mode == execution_mode,
                            Order.status == "filled",
                            Order.symbol.in_(symbols),
                            Order.exchange_order_id.like("manual_close:%"),
                        )
                    )
                    manual_close_orders = list(manual_close_result.scalars().all())
                for pos in rows:
                    if not pos.closed_at:
                        continue
                    if position_has_manual_close_order(pos, manual_close_orders):
                        continue
                    await self.record_trade_reflection_in_session(
                        session,
                        pos,
                        exit_price=float(pos.current_price or pos.entry_price or 0.0),
                        entry_fee=0.0,
                        close_fee=0.0,
                        gross_pnl=float(pos.realized_pnl or 0.0),
                        source="startup_backfill",
                        decision=None,
                    )
        except Exception as exc:
            logger.warning("failed to backfill trade reflections", error=safe_error_text(exc))


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
        "confidence_adjustment": float(memory.confidence_adjustment or 0.0),
        "position_size_multiplier": float(memory.position_size_multiplier or 1.0),
        "evidence_count": int(memory.evidence_count or 0),
        "success_count": int(getattr(memory, "success_count", 0) or 0),
        "failure_count": int(getattr(memory, "failure_count", 0) or 0),
        "confidence_score": float(memory.confidence_score or 0.0),
        "extra": memory.extra or {},
        "created_at": memory.created_at.isoformat() if memory.created_at else None,
        "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
    }


def dynamic_expert_weights_from_memories(
    by_expert: dict[str, list[dict[str, Any]]],
    model_slots: Sequence[dict[str, Any]] = FIXED_AI_MODEL_SLOTS,
) -> dict[str, dict[str, Any]]:
    """Conservative long-term-memory based expert weight adjustment."""

    result: dict[str, dict[str, Any]] = {}
    slot_weights = {
        str(slot.get("name") or ""): float(slot.get("weight", 1.0) or 1.0)
        for slot in model_slots
        if slot.get("name")
    }
    for expert_name, base_weight in slot_weights.items():
        memories = [memory for memory in by_expert.get(expert_name, []) if isinstance(memory, dict)]
        if not memories:
            result[expert_name] = {
                "base_weight": base_weight,
                "multiplier": 1.0,
                "effective_weight": base_weight,
                "memory_count": 0,
                "evidence_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "reason": "暂无足够历史样本，使用基础权重。",
            }
            continue

        evidence = sum(max(int(memory.get("evidence_count", 1) or 1), 1) for memory in memories)
        success = sum(max(int(memory.get("success_count", 0) or 0), 0) for memory in memories)
        failure = sum(max(int(memory.get("failure_count", 0) or 0), 0) for memory in memories)
        weighted_adjustment = 0.0
        weight_sum = 0.0
        for memory in memories:
            confidence_score = min(max(float(memory.get("confidence_score", 0.5) or 0.5), 0.1), 1.0)
            memory_evidence = max(int(memory.get("evidence_count", 1) or 1), 1)
            weight = confidence_score * min(memory_evidence, 6)
            weighted_adjustment += float(memory.get("confidence_adjustment", 0.0) or 0.0) * weight
            weight_sum += weight

        average_adjustment = weighted_adjustment / weight_sum if weight_sum > 0 else 0.0
        performance_edge = ((success + 1) / (success + failure + 2)) - 0.5
        raw_multiplier = 1.0 + average_adjustment * 0.70 + performance_edge * 0.35
        if evidence < 2 and success + failure < 2:
            raw_multiplier = 1.0

        multiplier = min(max(raw_multiplier, 0.70), 1.15)
        if failure >= success + 2:
            multiplier = min(multiplier, 0.90)
        elif success >= failure + 3:
            multiplier = max(multiplier, 1.05)

        if multiplier > 1.03:
            reason = f"近期记忆中成功样本较多或正向教训更稳定，权重提高到 {multiplier:.2f} 倍。"
        elif multiplier < 0.97:
            reason = f"近期记忆提示该专家相关场景亏损偏多，权重降到 {multiplier:.2f} 倍。"
        else:
            reason = "历史样本未显示明显优势，保持基础权重。"

        result[expert_name] = {
            "base_weight": base_weight,
            "multiplier": round(multiplier, 4),
            "effective_weight": round(base_weight * multiplier, 4),
            "memory_count": len(memories),
            "evidence_count": evidence,
            "success_count": success,
            "failure_count": failure,
            "reason": reason,
        }
    return result


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
    speed = "极短持仓" if hold_minutes < 5 else "短线持仓" if hold_minutes < 30 else "较长持仓"
    loss_level = (
        "大亏" if pnl_pct <= -0.01 else "小亏" if pnl_pct < 0 else "盈利" if pnl_pct > 0 else "打平"
    )
    leverage = float(getattr(pos, "leverage", 1.0) or 1.0)
    return f"{pos.symbol} {side_label}，{speed}，{leverage:.1f}x，{loss_level}"


def reflection_summary(
    pos: Any,
    outcome: str,
    pnl_pct: float,
    hold_minutes: float,
) -> tuple[str, str]:
    side_label = "做多" if str(pos.side).lower() == "long" else "做空"
    if outcome == "loss":
        mistake = (
            f"{pos.symbol} {side_label} 最终亏损 {pnl_pct:.2%}，"
            "说明入场后的趋势延续、成交量配合或退出时机至少有一项不足。"
        )
        improvement = (
            "下次同类场景需要提高入场质量要求，优先降低仓位和杠杆；"
            "如果短时间内没有走出利润缓冲，持仓专家应更早要求复盘。"
        )
    elif outcome == "profit":
        mistake = f"{pos.symbol} {side_label} 本次盈利，" "说明该方向在当时条件下存在可执行边际。"
        improvement = (
            "保留这类有效条件，但仍需确认成交量、趋势强度和止损收益比，" "不允许盲目放大。"
        )
    else:
        mistake = f"{pos.symbol} {side_label} 基本打平，收益没有明显覆盖机会成本。"
        improvement = "下次同类场景降低优先级，只有当趋势、动量和成交量更明确时才开仓。"
    if hold_minutes < 5 and outcome != "profit":
        improvement += " 本次持仓很短即退出，说明入场点或止盈止损距离可能过窄。"
    return mistake, improvement


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
    is_loss = outcome == "loss"
    big_loss = pnl_pct <= -0.01
    is_profit = outcome == "profit"
    adjustment = -0.12 if big_loss else -0.08 if is_loss else 0.03 if is_profit else -0.03
    size_multiplier = 0.45 if big_loss else 0.60 if is_loss else 1.0 if is_profit else 0.80
    memory_type = "loss_lesson" if is_loss else "profit_pattern" if is_profit else "flat_lesson"
    recommended = (
        "reduce_risk" if is_loss else "keep_with_filters" if is_profit else "wait_for_better_setup"
    )
    evidence_success = 1 if is_profit else 0
    evidence_failure = 1 if is_loss else 0

    labels = {
        str(slot["name"]): slot.get("label", slot["name"])
        for slot in model_slots
        if slot.get("name")
    }
    side_label = "做多" if side == "long" else "做空"
    outcome_text = {"loss": "亏损", "profit": "盈利", "flat": "打平"}.get(outcome, outcome)
    base_key = f"{symbol}|{side}|{memory_type}|{lesson_bucket(pnl_pct, hold_minutes)}"
    lessons = {
        "trend_expert": (
            f"{symbol} {side_label} 在场景[{pattern}]下结果为{outcome_text}。"
            "下次只判断方向质量：均线方向、ADX、MACD 和突破结构，不直接决定仓位。"
        ),
        "momentum_expert": (
            f"{symbol} {side_label} 在场景[{pattern}]下结果为{outcome_text}。"
            "下次优先看预期净收益、手续费覆盖、亏损概率和盈亏比，不只看胜率。"
        ),
        "sentiment_expert": (
            f"{symbol} {side_label} 在场景[{pattern}]下结果为{outcome_text}。"
            "下次核对 1/5/10/30 分钟路径、延续风险、反转风险和事件冲击后再判断执行时机。"
        ),
        "position_expert": (
            f"{symbol} {side_label} 持仓 {hold_minutes:.1f} 分钟后结果为{outcome_text}。"
            "下次检查是否该锁盈、亏损能否修复、亏损是否扩大，以及是否值得加仓或减仓。"
        ),
        "risk_expert": (
            f"{symbol} {side_label} 在场景[{pattern}]下结果为{outcome_text}。"
            "下次检查异常插针、流动性、极端波动、保证金限制和交易所约束后再放行风险。"
        ),
    }
    result: dict[str, dict[str, Any]] = {}
    for expert_name, lesson in lessons.items():
        result[expert_name] = {
            "expert_name": expert_name,
            "expert_label": labels.get(expert_name, expert_name),
            "symbol": symbol,
            "side": side,
            "memory_type": memory_type,
            "market_pattern": pattern,
            "lesson": lesson,
            "recommended_action": recommended,
            "confidence_adjustment": adjustment,
            "position_size_multiplier": size_multiplier,
            "evidence_count": 1,
            "success_count": evidence_success,
            "failure_count": evidence_failure,
            "confidence_score": 0.65 if is_loss else 0.55,
            "memory_key": f"{expert_name}|{base_key}",
        }
    return result


def lesson_bucket(pnl_pct: float, hold_minutes: float) -> str:
    pnl_bucket = (
        "big_loss"
        if pnl_pct <= -0.01
        else "loss" if pnl_pct < 0 else "profit" if pnl_pct > 0 else "flat"
    )
    time_bucket = "fast" if hold_minutes < 5 else "short" if hold_minutes < 30 else "long"
    return f"{pnl_bucket}|{time_bucket}"


def _realized_weight_result(
    stats: dict[str, dict[str, Any]],
    slot_weights: dict[str, float],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, bucket in stats.items():
        count = int(bucket["count"])
        if count < 3:
            continue
        pnl = float(bucket["pnl"])
        avg_pnl = pnl / count
        win_rate = bucket["wins"] / count
        profit_factor = (
            float(bucket["profit"]) / float(bucket["loss"])
            if float(bucket["loss"]) > 0
            else (3.0 if float(bucket["profit"]) > 0 else 0.0)
        )
        expectancy_component = max(min(avg_pnl / 8.0, 0.24), -0.30)
        factor_component = max(min((profit_factor - 1.0) * 0.12, 0.14), -0.18)
        win_component = max(min((win_rate - 0.5) * 0.06, 0.03), -0.03)
        raw_multiplier = 1.0 + expectancy_component + factor_component + win_component
        multiplier = min(max(raw_multiplier, 0.65), 1.30)
        result[name] = {
            "base_weight": slot_weights.get(name, 1.0),
            "multiplier": round(multiplier, 4),
            "effective_weight": round(slot_weights.get(name, 1.0) * multiplier, 4),
            "realized_count": count,
            "realized_pnl": round(pnl, 6),
            "win_rate": round(win_rate, 4),
            "avg_pnl": round(avg_pnl, 6),
            "profit_factor": round(profit_factor, 4),
            "reason": (
                f"北京时间今日同向参与 {count} 笔，真实盈亏 {pnl:.2f}U，"
                f"胜率 {win_rate:.0%}，权重调整到 {multiplier:.2f} 倍。"
            ),
        }
    return result


def _entry_side_from_action(action: Any) -> str:
    value = str(action or "").lower()
    if value == "short":
        return "short"
    if value == "long":
        return "long"
    return ""


def _aware_utc(value: datetime | None) -> datetime | None:
    if value and value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _empty_memory_context() -> dict[str, Any]:
    return {
        "expert_memories": {},
        "expert_memories_flat": [],
        "dynamic_expert_weights": {},
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
