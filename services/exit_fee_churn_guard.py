"""Exit fee churn and early-close protection."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog

from ai_brain.base_model import Action, DecisionOutput
from config.settings import settings
from core.safe_output import safe_error_text
from db.repositories.trade_repo import TradeRepository
from services.exit_strategy_policy import exit_strategy_policy_from_context
from services.exit_intent import (
    PROFIT_EXIT_INTENTS,
    PROTECTIVE_DOWNSIDE_INTENTS,
    ExitIntent,
    classify_exit_intent,
)
from services.trading_params import DEFAULT_TRADING_PARAMS, ESTIMATED_TAKER_FEE_PCT

logger = structlog.get_logger(__name__)

MIN_DISCRETIONARY_HOLD_MINUTES = 4.0
ENTRY_SETTLEMENT_EXIT_GUARD_SECONDS = 120.0
DISCRETIONARY_CLOSE_CONFIDENCE = 0.66
PROFIT_PROTECTION_MIN_NET_PNL_RATIO = 0.004
PROFIT_PROTECTION_STRONG_NET_PNL_RATIO = 0.010
PROFIT_PROTECTION_MIN_NET_USDT = 3.00
PROFIT_PROTECTION_MIN_FEE_MULTIPLE = 4.0
PROFIT_PROTECTION_STRONG_FEE_MULTIPLE = 5.0
PROFIT_DRAWDOWN_PARTIAL_RETRACE = 0.38
FAST_RISK_REDUCE_LOSS_PCT = 0.012
_EXIT_PARAMS = DEFAULT_TRADING_PARAMS.ensemble_exit_decision
SMALL_POSITION_PROFIT_LOCK_MAX_NOTIONAL_USDT = (
    _EXIT_PARAMS.small_position_profit_lock_max_notional_usdt
)
SMALL_POSITION_PROFIT_LOCK_MIN_PNL_RATIO = (
    _EXIT_PARAMS.small_position_profit_lock_min_pnl_ratio
)
SMALL_POSITION_PROFIT_LOCK_MIN_FEE_MULTIPLE = (
    _EXIT_PARAMS.small_position_profit_lock_min_fee_multiple
)
SMALL_POSITION_PROFIT_LOCK_MIN_NET_USDT = (
    _EXIT_PARAMS.small_position_profit_lock_min_net_usdt
)
SMALL_POSITION_PROFIT_LOCK_MIN_PLANNED_NET_USDT = (
    _EXIT_PARAMS.small_position_profit_lock_min_planned_net_usdt
)
SMALL_POSITION_PROFIT_LOCK_PARTIAL_FEE_MULTIPLE = (
    _EXIT_PARAMS.small_position_profit_lock_partial_fee_multiple
)
SMALL_POSITION_PROFIT_LOCK_PARTIAL_NOTIONAL_RATIO = (
    _EXIT_PARAMS.small_position_profit_lock_partial_notional_ratio
)

PROTECTIVE_DOWNSIDE_EXIT_TEXT_TERMS = (
    "可能会跌",
    "可能下跌",
    "后续下跌",
    "后面可能会跌",
    "继续下跌",
    "下行风险",
    "趋势转弱",
    "趋势走弱",
    "趋势失效",
    "趋势破坏",
    "反向压力",
    "反转风险",
    "预防性",
    "保护本金",
    "避免转亏",
    "避免回撤",
    "避免回吐",
    "先撤退",
    "先离场",
    "主动撤退",
    "风险扩大",
    "downside",
    "trend failure",
    "trend invalid",
    "reversal risk",
    "capital protection",
)


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(float(value), low), high)


def _protective_downside_exit_intent(
    *,
    reasoning_text: str,
    close_evidence: dict[str, Any],
) -> bool:
    """Detect AI exits that protect capital from expected downside, not shard loss repair."""

    if close_evidence.get("loss_repair") or close_evidence.get("loss_repair_evidence"):
        return False
    structured = bool(
        close_evidence.get("predictive_reversal_exit")
        or close_evidence.get("predictive_full_exit")
        or close_evidence.get("strong_opposite_pressure")
        or close_evidence.get("moderate_opposite_pressure")
        or close_evidence.get("trend_failure")
        or close_evidence.get("trend_invalidation")
        or close_evidence.get("thesis_invalidated")
        or close_evidence.get("capital_protection")
        or close_evidence.get("preventive_exit")
    )
    text = reasoning_text.lower()
    textual = any(term.lower() in text for term in PROTECTIVE_DOWNSIDE_EXIT_TEXT_TERMS)
    return structured or (bool(close_evidence.get("should_close")) and textual) or textual


def exit_profit_protection_state(
    *,
    net_now: float,
    notional: float,
    fee_buffer: float,
    confidence: float,
    age_minutes: float,
    min_net_pnl_ratio: float = PROFIT_PROTECTION_MIN_NET_PNL_RATIO,
    min_fee_multiple: float = PROFIT_PROTECTION_MIN_FEE_MULTIPLE,
    min_net_usdt: float = PROFIT_PROTECTION_MIN_NET_USDT,
    strong_net_pnl_ratio: float = PROFIT_PROTECTION_STRONG_NET_PNL_RATIO,
    strong_fee_multiple: float = PROFIT_PROTECTION_STRONG_FEE_MULTIPLE,
    discretionary_confidence: float = DISCRETIONARY_CLOSE_CONFIDENCE,
    min_hold_minutes: float = MIN_DISCRETIONARY_HOLD_MINUTES,
) -> dict[str, Any]:
    """Allow profitable exits with dynamic thresholds instead of a fixed USDT target."""

    abs_notional = abs(float(notional or 0.0))
    fee_buffer = max(float(fee_buffer or 0.0), 0.0)
    net_now = float(net_now or 0.0)
    confidence = float(confidence or 0.0)
    pnl_ratio = net_now / abs_notional if abs_notional > 0 else 0.0
    standard_min_net_profit = max(
        abs_notional * min_net_pnl_ratio,
        fee_buffer * min_fee_multiple,
        min_net_usdt,
    )
    small_position_min_net_profit = max(
        abs_notional * SMALL_POSITION_PROFIT_LOCK_MIN_PNL_RATIO,
        fee_buffer * SMALL_POSITION_PROFIT_LOCK_MIN_FEE_MULTIPLE,
        SMALL_POSITION_PROFIT_LOCK_MIN_NET_USDT,
    )
    small_position_lock = (
        0 < abs_notional <= SMALL_POSITION_PROFIT_LOCK_MAX_NOTIONAL_USDT
        and pnl_ratio >= SMALL_POSITION_PROFIT_LOCK_MIN_PNL_RATIO
        and net_now >= small_position_min_net_profit
        and net_now / max(fee_buffer, 1e-9) >= SMALL_POSITION_PROFIT_LOCK_MIN_FEE_MULTIPLE
        and age_minutes >= min_hold_minutes
    )
    min_net_profit = (
        min(standard_min_net_profit, small_position_min_net_profit)
        if small_position_lock
        else standard_min_net_profit
    )
    strong_net_profit = max(
        abs_notional * strong_net_pnl_ratio,
        fee_buffer * strong_fee_multiple,
        min_net_profit * 1.5,
    )
    mature_enough = age_minutes >= min_hold_minutes
    normal_lock = mature_enough and net_now >= min_net_profit
    strong_lock = net_now >= strong_net_profit and confidence >= discretionary_confidence
    early_lock = False
    allow = bool(net_now > 0 and (normal_lock or strong_lock or early_lock))
    return {
        "allow": allow,
        "net_pnl": round(net_now, 8),
        "pnl_ratio": round(pnl_ratio, 6),
        "notional": round(abs_notional, 8),
        "fee_buffer": round(fee_buffer, 8),
        "min_net_profit": round(min_net_profit, 8),
        "standard_min_net_profit": round(standard_min_net_profit, 8),
        "small_position_lock": bool(small_position_lock),
        "small_position_min_net_profit": round(small_position_min_net_profit, 8),
        "strong_net_profit": round(strong_net_profit, 8),
        "confidence": round(confidence, 4),
        "age_minutes": round(age_minutes, 3),
        "mature_enough": mature_enough,
        "normal_lock": bool(normal_lock),
        "strong_lock": bool(strong_lock),
        "early_lock": bool(early_lock),
        "rule": "按仓位名义价值比例和手续费倍数动态判断锁盈，不使用固定 8-10U 作为大浮盈门槛。",
        "reason": (
            "扣除双边手续费后达到动态锁盈线，允许主动平仓。"
            if allow
            else "扣除双边手续费后未达到动态锁盈线。"
        ),
    }


@dataclass(slots=True)
class ExitFeeChurnGuardPolicy:
    """Skip weak discretionary exits that only create fee churn."""

    session_factory: Callable[[], Any]
    model_execution_mode_provider: Callable[[str], str]
    entry_fee_provider: Callable[[Any, Any, float], Awaitable[float]]
    invalidation_snapshot_provider: Callable[[DecisionOutput, str, float, float], dict[str, Any]]
    forced_exit_policy: Any
    position_peaks: dict[str, dict[str, Any]]
    position_peak_key_provider: Callable[[str, str, str], str]
    trade_repository_factory: Callable[[Any], Any] = TradeRepository
    config: Any = field(default_factory=lambda: settings)

    async def guard_reason(self, model_name: str, decision: DecisionOutput) -> str | None:
        """Return a Chinese blocker reason when an exit is likely fee churn."""

        if not decision.is_exit:
            return None

        target_side = "long" if decision.action == Action.CLOSE_LONG else "short"
        model_mode = self.model_execution_mode_provider(model_name)
        snapshot = decision.feature_snapshot or {}
        current_price = _safe_float(
            snapshot.get("current_price", snapshot.get("close", 0.0)),
            0.0,
        )

        try:
            async with self.session_factory() as session:
                repo = self.trade_repository_factory(session)
                positions = await repo.get_matching_open_positions(
                    model_name=model_name,
                    symbol=decision.symbol,
                    side=target_side,
                    execution_mode=model_mode,
                )
                if not positions:
                    return None

                raw = _safe_dict(decision.raw_response)
                close_evidence = _safe_dict(raw.get("close_evidence"))
                execution_profit = _safe_dict(raw.get("execution_profit_protection"))
                fast_trigger = str(raw.get("fast_risk_trigger") or "")
                reasoning_text = str(decision.reasoning or "")
                exit_intent = classify_exit_intent(decision)
                raw = _safe_dict(decision.raw_response)
                close_evidence = _safe_dict(raw.get("close_evidence"))
                profit_exit_intent = bool(
                    close_evidence.get("profit_protection")
                    or execution_profit.get("allow")
                    or fast_trigger.startswith("profit_drawdown")
                    or any(term in reasoning_text for term in ("锁盈", "利润保护", "浮盈", "止盈"))
                )
                loss_exit_intent = bool(
                    any(
                        term in reasoning_text
                        for term in ("亏损", "浮亏", "扩亏", "止损", "未实现盈亏为负")
                    )
                    or close_evidence.get("loss_repair")
                    or close_evidence.get("loss_repair_evidence")
                )
                protective_downside_exit_intent = _protective_downside_exit_intent(
                    reasoning_text=reasoning_text,
                    close_evidence=close_evidence,
                )
                profit_exit_intent = profit_exit_intent or exit_intent in PROFIT_EXIT_INTENTS
                loss_exit_intent = bool(
                    exit_intent == ExitIntent.LOSS_REPAIR
                    or (loss_exit_intent and exit_intent == ExitIntent.ORDINARY)
                )
                protective_downside_exit_intent = bool(
                    protective_downside_exit_intent or exit_intent in PROTECTIVE_DOWNSIDE_INTENTS
                )

                aggregate_qty = 0.0
                aggregate_entry_value = 0.0
                aggregate_gross_pnl = 0.0
                aggregate_entry_fee = 0.0
                aggregate_close_fee = 0.0
                aggregate_hit_stop = False
                aggregate_hit_profit = False
                for aggregate_pos in positions:
                    pos_qty = abs(float(aggregate_pos.quantity or 0.0))
                    pos_entry = float(aggregate_pos.entry_price or 0.0)
                    if pos_qty <= 0 or pos_entry <= 0:
                        continue
                    aggregate_qty += pos_qty
                    aggregate_entry_value += pos_entry * pos_qty
                    if current_price <= 0:
                        current_price = float(aggregate_pos.current_price or pos_entry)
                    pos_gross = (
                        (current_price - pos_entry) * pos_qty
                        if target_side == "long"
                        else (pos_entry - current_price) * pos_qty
                    )
                    aggregate_gross_pnl += pos_gross
                    aggregate_entry_fee += await self.entry_fee_provider(
                        session, aggregate_pos, pos_qty
                    )
                    aggregate_close_fee += abs(current_price * pos_qty) * ESTIMATED_TAKER_FEE_PCT
                    if target_side == "long":
                        aggregate_hit_stop = aggregate_hit_stop or bool(
                            aggregate_pos.stop_loss_price
                            and current_price <= aggregate_pos.stop_loss_price
                        )
                        aggregate_hit_profit = aggregate_hit_profit or bool(
                            aggregate_pos.take_profit_price
                            and current_price >= aggregate_pos.take_profit_price
                        )
                    else:
                        aggregate_hit_stop = aggregate_hit_stop or bool(
                            aggregate_pos.stop_loss_price
                            and current_price >= aggregate_pos.stop_loss_price
                        )
                        aggregate_hit_profit = aggregate_hit_profit or bool(
                            aggregate_pos.take_profit_price
                            and current_price <= aggregate_pos.take_profit_price
                        )

                aggregate_entry = aggregate_entry_value / max(aggregate_qty, 1e-9)
                aggregate_net_pnl = aggregate_gross_pnl - aggregate_entry_fee - aggregate_close_fee
                aggregate_invalidation = (
                    self.invalidation_snapshot_provider(
                        decision,
                        target_side,
                        aggregate_entry,
                        current_price,
                    )
                    if aggregate_qty > 0 and aggregate_entry > 0 and current_price > 0
                    else {}
                )
                forced_exit = self.forced_exit_policy.is_forced_exit(decision)
                if (
                    aggregate_qty > 0
                    and aggregate_net_pnl >= 0
                    and loss_exit_intent
                    and not profit_exit_intent
                    and not protective_downside_exit_intent
                    and not forced_exit
                    and not aggregate_hit_stop
                    and not aggregate_hit_profit
                    and not bool(aggregate_invalidation.get("severe"))
                ):
                    raw["aggregate_exit_guard"] = {
                        "applied": True,
                        "target_side": target_side,
                        "fragments": len(positions),
                        "current_price": round(current_price, 8),
                        "aggregate_entry_price": round(aggregate_entry, 8),
                        "aggregate_gross_pnl": round(aggregate_gross_pnl, 6),
                        "aggregate_net_pnl": round(aggregate_net_pnl, 6),
                        "loss_exit_intent": True,
                        "profit_exit_intent": False,
                        "protective_downside_exit_intent": False,
                        "exit_intent": exit_intent.value,
                        "reason": "同币种同方向整体不亏，禁止按单个分片浮亏触发亏损修复平仓。",
                    }
                    decision.raw_response = raw
                    return (
                        f"整体持仓保护：{decision.symbol} {target_side} 当前共有 {len(positions)} 个分片，"
                        f"按整体均价 {aggregate_entry:.6g} 和最新价 {current_price:.6g} 估算整体浮盈 "
                        f"{aggregate_gross_pnl:.4f}U（扣费后约 {aggregate_net_pnl:.4f}U）。"
                        "本次平仓理由来自单个分片亏损/止损描述，但整体仓位扣费后并不亏，"
                        "未触发硬止损、止盈或严重趋势失效，因此不执行该亏损平仓。"
                    )

                pos = positions[0]
                qty = float(pos.quantity or 0.0)
                entry_price = float(pos.entry_price or 0.0)
                if qty <= 0 or entry_price <= 0:
                    return None
                if current_price <= 0:
                    current_price = float(pos.current_price or entry_price)

                notional = qty * entry_price
                if target_side == "long":
                    gross_pnl = (current_price - entry_price) * qty
                    hit_stop = bool(pos.stop_loss_price and current_price <= pos.stop_loss_price)
                    hit_profit = bool(
                        pos.take_profit_price and current_price >= pos.take_profit_price
                    )
                else:
                    gross_pnl = (entry_price - current_price) * qty
                    hit_stop = bool(pos.stop_loss_price and current_price >= pos.stop_loss_price)
                    hit_profit = bool(
                        pos.take_profit_price and current_price <= pos.take_profit_price
                    )

                if hit_stop or hit_profit:
                    return None

                hard_stop_loss = -abs(notional) * float(self.config.hard_stop_loss_pct or 0.05)
                if gross_pnl <= hard_stop_loss:
                    return None

                invalidation = self.invalidation_snapshot_provider(
                    decision,
                    target_side,
                    entry_price,
                    current_price,
                )
                severe_invalidation = bool(invalidation.get("severe"))

                entry_fee = await self.entry_fee_provider(session, pos, qty)
                estimated_close_fee = max(
                    abs(current_price * qty) * ESTIMATED_TAKER_FEE_PCT,
                    entry_fee if entry_fee > 0 else 0.0,
                )
                net_now = gross_pnl - entry_fee - estimated_close_fee

                opened_at = pos.created_at
                if opened_at and opened_at.tzinfo is None:
                    opened_at = opened_at.replace(tzinfo=UTC)
                age_minutes = (
                    (datetime.now(UTC) - opened_at).total_seconds() / 60.0 if opened_at else 9999.0
                )
                fee_buffer = entry_fee + estimated_close_fee
                fee_coverage_multiple = net_now / max(fee_buffer, 1e-9)
                continuation_valid = not bool(
                    invalidation.get("key_break")
                    or invalidation.get("trend_reversal")
                    or (invalidation.get("momentum_bad") and invalidation.get("volume_confirms"))
                )
                peak_state = self.position_peaks.get(
                    self.position_peak_key_provider(model_name, pos.symbol, target_side),
                    {},
                )
                peak_net = _safe_float(peak_state.get("peak_unrealized_pnl"), gross_pnl)
                drawdown_from_peak = max(peak_net - gross_pnl, 0.0)
                drawdown_ratio = (
                    drawdown_from_peak / max(abs(peak_net), 1e-9) if peak_net > 0 else 0.0
                )
                raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
                strategy_exit_policy = exit_strategy_policy_from_context(raw)
                profit_lock_multiplier = _safe_float(
                    strategy_exit_policy.get("profit_lock_min_usdt_multiplier"),
                    1.0,
                )
                winner_hold_strength = _safe_float(
                    strategy_exit_policy.get("winner_hold_strength"),
                    0.0,
                )
                loser_exit_strength = _safe_float(
                    strategy_exit_policy.get("loser_exit_strength"),
                    0.0,
                )
                dynamic_min_hold_minutes = MIN_DISCRETIONARY_HOLD_MINUTES * _clamp(
                    1.0
                    + winner_hold_strength * 0.35
                    - max(loser_exit_strength, 0.0) * 0.22
                    - max(1.0 - profit_lock_multiplier, 0.0) * 0.30,
                    0.70,
                    1.65,
                )
                dynamic_exit_confidence_floor = _clamp(
                    DISCRETIONARY_CLOSE_CONFIDENCE - max(loser_exit_strength, 0.0) * 0.10,
                    0.56,
                    DISCRETIONARY_CLOSE_CONFIDENCE,
                )
                winner_run_drawdown_limit = PROFIT_DRAWDOWN_PARTIAL_RETRACE * _clamp(
                    1.0
                    - winner_hold_strength * 0.22
                    + max(1.0 - profit_lock_multiplier, 0.0) * 0.35,
                    0.65,
                    1.15,
                )
                raw["exit_quality"] = {
                    "net_profit_after_fee": round(net_now, 8),
                    "fee_coverage_multiple": round(fee_coverage_multiple, 4),
                    "continuation_score": round(
                        0.0 if severe_invalidation else (0.75 if continuation_valid else 0.35), 4
                    ),
                    "trend_still_valid": bool(continuation_valid),
                    "drawdown_from_peak": round(drawdown_from_peak, 8),
                    "drawdown_from_peak_ratio": round(drawdown_ratio, 6),
                    "invalidation": invalidation,
                    "strategy_exit_policy": strategy_exit_policy,
                    "dynamic_min_hold_minutes": round(dynamic_min_hold_minutes, 4),
                    "dynamic_exit_confidence_floor": round(
                        dynamic_exit_confidence_floor,
                        4,
                    ),
                    "winner_run_drawdown_limit": round(winner_run_drawdown_limit, 6),
                }
                decision.raw_response = raw

                forced_exit = self.forced_exit_policy.is_forced_exit(decision)
                if (
                    age_minutes * 60.0 < ENTRY_SETTLEMENT_EXIT_GUARD_SECONDS
                    and not hit_stop
                    and not hit_profit
                    and not forced_exit
                    and not protective_downside_exit_intent
                ):
                    return (
                        f"平仓保护：该仓位刚开仓 {age_minutes:.2f} 分钟，仍在成交结算防抖窗口内，"
                        "普通 AI 减仓、止盈或降低风险建议暂不执行，避免开仓后同轮或数秒内反复部分平仓。"
                    )

                if forced_exit:
                    return None

                if severe_invalidation:
                    return None

                protected_by_okx = bool(pos.stop_loss_price or pos.take_profit_price)
                confidence = float(decision.confidence or 0.0)
                fee_buffer = entry_fee + estimated_close_fee
                invalidation_confirmed = bool(
                    (invalidation.get("key_break") and invalidation.get("momentum_bad"))
                    or (invalidation.get("trend_reversal") and invalidation.get("momentum_bad"))
                )
                early_hard_risk_confirmed = bool(
                    hit_stop
                    or hit_profit
                    or forced_exit
                    or severe_invalidation
                    or protective_downside_exit_intent
                    or invalidation_confirmed
                )

                if (
                    exit_intent == ExitIntent.CAPITAL_ROTATION
                    and net_now < 0
                    and not early_hard_risk_confirmed
                ):
                    raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
                    raw["capital_rotation_loss_guard"] = {
                        "applied": True,
                        "net_profit_after_fee": round(net_now, 8),
                        "trend_still_valid": bool(continuation_valid),
                        "exit_intent": exit_intent.value,
                        "reason": (
                            "capital rotation cannot realize a net loss without hard-risk or "
                            "trend-invalidation evidence"
                        ),
                    }
                    decision.raw_response = raw
                    return (
                        f"仓位轮动保护：{decision.symbol} 当前扣费后预计净亏 {net_now:.4f} USDT，"
                        "未触发硬止损、止盈、严重趋势失效或预测下行风险。低质量仓位释放只能在不制造净亏损"
                        "或已有明确风险证据时执行，本轮继续持有。"
                    )

                if (
                    age_minutes < dynamic_min_hold_minutes
                    and not invalidation_confirmed
                    and not protective_downside_exit_intent
                ):
                    early_strong_profit = (
                        net_now > 0
                        and abs(notional) > 0
                        and net_now
                        >= max(
                            abs(notional) * PROFIT_PROTECTION_STRONG_NET_PNL_RATIO,
                            fee_buffer * PROFIT_PROTECTION_STRONG_FEE_MULTIPLE,
                        )
                        and confidence >= dynamic_exit_confidence_floor
                    )
                    early_deep_loss = (
                        net_now < 0
                        and abs(notional) > 0
                        and abs(net_now) >= abs(notional) * FAST_RISK_REDUCE_LOSS_PCT
                        and confidence >= dynamic_exit_confidence_floor
                        and early_hard_risk_confirmed
                    )
                    if not early_strong_profit and not early_deep_loss:
                        return (
                            f"平仓保护：该仓位只持有 {age_minutes:.1f} 分钟，"
                            "未触发止损、止盈或趋势严重失效。普通 AI 平仓建议暂不执行，"
                            "避免刚开仓就因短线噪音频繁切仓。"
                        )

                profit_protection = exit_profit_protection_state(
                    net_now=net_now,
                    notional=notional,
                    fee_buffer=fee_buffer,
                    confidence=confidence,
                    age_minutes=age_minutes,
                    min_net_pnl_ratio=PROFIT_PROTECTION_MIN_NET_PNL_RATIO * profit_lock_multiplier,
                    min_fee_multiple=PROFIT_PROTECTION_MIN_FEE_MULTIPLE
                    * _clamp(0.90 + (profit_lock_multiplier - 1.0) * 0.65, 0.75, 1.35),
                    min_net_usdt=PROFIT_PROTECTION_MIN_NET_USDT * profit_lock_multiplier,
                    strong_net_pnl_ratio=PROFIT_PROTECTION_STRONG_NET_PNL_RATIO
                    * _clamp(0.95 + (profit_lock_multiplier - 1.0) * 0.55, 0.80, 1.35),
                    strong_fee_multiple=PROFIT_PROTECTION_STRONG_FEE_MULTIPLE
                    * _clamp(0.95 + (profit_lock_multiplier - 1.0) * 0.55, 0.80, 1.35),
                    discretionary_confidence=dynamic_exit_confidence_floor,
                    min_hold_minutes=dynamic_min_hold_minutes,
                )
                if profit_protection["allow"]:
                    close_pct = min(max(float(decision.position_size_pct or 1.0), 0.0), 1.0)
                    planned_lock_net = net_now * close_pct
                    ordinary_profit_intent = exit_intent in {
                        ExitIntent.PROFIT_PROTECTION,
                        ExitIntent.CAPITAL_ROTATION,
                        ExitIntent.ORDINARY,
                    }
                    if (
                        close_pct >= 0.999
                        and ordinary_profit_intent
                        and continuation_valid
                        and drawdown_ratio < winner_run_drawdown_limit
                        and not profit_protection.get("strong_lock")
                        and not protective_downside_exit_intent
                    ):
                        raw = (
                            decision.raw_response if isinstance(decision.raw_response, dict) else {}
                        )
                        raw["execution_profit_protection"] = profit_protection
                        raw["winner_run_guard"] = {
                            "applied": True,
                            "close_pct": round(close_pct, 4),
                            "net_profit_after_fee": round(net_now, 8),
                            "drawdown_from_peak_ratio": round(drawdown_ratio, 6),
                            "trend_still_valid": bool(continuation_valid),
                            "strong_lock": bool(profit_protection.get("strong_lock")),
                            "exit_intent": exit_intent.value,
                            "reason": (
                                "ordinary full close would realize only an early winner while "
                                "continuation is still valid"
                            ),
                        }
                        decision.raw_response = raw
                        return (
                            f"赢家持仓保护：{decision.symbol} 当前扣费后盈利 {net_now:.4f}U，"
                            f"峰值回撤 {drawdown_ratio:.0%}，趋势尚未确认失效，且未达到强锁盈线。"
                            "本次不执行普通全平，继续让优势仓位运行；若后续出现明显回撤、趋势失效、"
                            "硬风险或交易所止盈止损触发，再允许平仓。"
                        )
                    standard_meaningful_partial_lock = max(
                        PROFIT_PROTECTION_MIN_NET_USDT * profit_lock_multiplier,
                        fee_buffer
                        * max(
                            PROFIT_PROTECTION_MIN_FEE_MULTIPLE
                            * _clamp(0.95 + (profit_lock_multiplier - 1.0) * 0.55, 0.85, 1.35),
                            6.0,
                        ),
                        abs(notional)
                        * max(PROFIT_PROTECTION_MIN_NET_PNL_RATIO * profit_lock_multiplier, 0.008),
                    )
                    small_position_partial_lock = bool(
                        close_evidence.get("small_position_profit_lock")
                        or profit_protection.get("small_position_lock")
                    )
                    small_position_meaningful_partial_lock = max(
                        SMALL_POSITION_PROFIT_LOCK_MIN_PLANNED_NET_USDT * profit_lock_multiplier,
                        fee_buffer
                        * SMALL_POSITION_PROFIT_LOCK_PARTIAL_FEE_MULTIPLE
                        * profit_lock_multiplier
                        * close_pct,
                        abs(notional)
                        * SMALL_POSITION_PROFIT_LOCK_PARTIAL_NOTIONAL_RATIO
                        * profit_lock_multiplier
                        * close_pct,
                    )
                    meaningful_partial_lock = (
                        min(
                            standard_meaningful_partial_lock,
                            small_position_meaningful_partial_lock,
                        )
                        if small_position_partial_lock
                        else standard_meaningful_partial_lock
                    )
                    if 0.0 < close_pct < 0.999 and planned_lock_net < meaningful_partial_lock:
                        raw = (
                            decision.raw_response if isinstance(decision.raw_response, dict) else {}
                        )
                        raw["execution_profit_protection"] = profit_protection
                        raw["small_profit_lock_guard"] = {
                            "applied": True,
                            "close_pct": round(close_pct, 4),
                            "net_profit_after_fee": round(net_now, 8),
                            "planned_lock_net": round(planned_lock_net, 8),
                            "meaningful_partial_lock": round(meaningful_partial_lock, 8),
                            "standard_meaningful_partial_lock": round(
                                standard_meaningful_partial_lock, 8
                            ),
                            "small_position_meaningful_partial_lock": round(
                                small_position_meaningful_partial_lock, 8
                            ),
                            "small_position_partial_lock": small_position_partial_lock,
                            "reason": "本次部分锁盈预计落袋利润太小，继续持有等待更有意义的锁盈或明确反转。",
                        }
                        decision.raw_response = raw
                        return (
                            f"锁盈保护：当前整仓扣费后预计盈利 {net_now:.4f} USDT，"
                            f"但本次只计划平 {close_pct:.0%}，预计实际落袋约 {planned_lock_net:.4f} USDT，"
                            f"低于动态有效锁盈线 {meaningful_partial_lock:.4f} USDT。"
                            "为避免碎片化小额平仓和手续费消耗，本次不执行普通部分锁盈；"
                            "等待更大浮盈、明确回撤/反转，或交易所止盈止损触发。"
                        )
                    raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
                    raw["execution_profit_protection"] = profit_protection
                    decision.raw_response = raw
                    return None

                if protected_by_okx and confidence < dynamic_exit_confidence_floor:
                    if net_now > 0:
                        return (
                            "平仓保护：该仓位已有 OKX 止盈/止损托底，当前扣费后仍盈利，"
                            f"但主动平仓信号强度只有 {confidence:.0%}，趋势尚未确认失效，继续持有。"
                        )
                    return (
                        "平仓保护：该仓位已经有 OKX 止盈/止损保护，"
                        f"当前信号强度 {confidence:.0%}，未达到主动干预门槛 "
                        f"{DISCRETIONARY_CLOSE_CONFIDENCE:.0%}。"
                        "为避免和交易所止盈止损互相打架，本轮继续持有。"
                    )

                if net_now < 0 and confidence < dynamic_exit_confidence_floor:
                    invalidation_pressure = self.invalidation_snapshot_provider(
                        decision,
                        target_side,
                        entry_price,
                        current_price,
                    )
                    if (
                        invalidation_pressure.get("key_break")
                        and invalidation_pressure.get("momentum_bad")
                    ) or (
                        invalidation_pressure.get("trend_reversal")
                        and invalidation_pressure.get("momentum_bad")
                    ):
                        return None
                    return (
                        "平仓保护：当前按手续费后预计净亏"
                        f" {net_now:.4f} USDT，未触发硬止损或止盈。"
                        f"持仓 {age_minutes:.1f} 分钟，信号强度 {confidence:.0%}。"
                        "小幅浮亏不会直接平仓，需要关键位跌破、量能恶化或趋势反转确认。"
                    )

                if net_now > 0 and net_now < fee_buffer * 1.5:
                    return (
                        "平仓保护：当前盈利尚未明显覆盖双边手续费，"
                        f"预计净盈亏 {net_now:.4f} USDT，持仓 {age_minutes:.1f} 分钟，"
                        "先继续观察，等待止盈、止损或更强信号。"
                    )
                if (
                    net_now > 0
                    and continuation_valid
                    and drawdown_ratio < winner_run_drawdown_limit
                    and confidence < 0.82
                ):
                    return (
                        "平仓保护：扣费后仍有盈利，但趋势延续证据没有失效，"
                        f"净收益 {net_now:.4f} USDT，手续费覆盖 {fee_coverage_multiple:.1f} 倍，"
                        f"峰值回撤 {drawdown_ratio:.0%}。本轮继续持有，等待明确反转或明显回撤再平。"
                    )
        except Exception as exc:
            logger.warning(
                "exit fee churn guard failed",
                symbol=decision.symbol,
                error=safe_error_text(exc),
            )
        return None
