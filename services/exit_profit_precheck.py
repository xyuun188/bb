"""Pre-execution profit-lock validation for exits."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ai_brain.base_model import Action, DecisionOutput
from services.exit_intent import PROTECTIVE_DOWNSIDE_INTENTS, classify_exit_intent
from services.trading_params import DEFAULT_TRADING_PARAMS, ESTIMATED_TAKER_FEE_PCT

PROFIT_PROTECTION_MIN_NET_USDT = 3.00
_EXIT_PARAMS = DEFAULT_TRADING_PARAMS.ensemble_exit_decision


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class ExitProfitPrecheckPolicy:
    """Recheck fresh profit before executing pure lock-profit exits."""

    latest_price_provider: Callable[[str], Awaitable[float]]
    normalize_symbol: Callable[[Any], str]
    min_net_usdt: float = PROFIT_PROTECTION_MIN_NET_USDT

    async def guard_reason(
        self,
        decision: DecisionOutput,
        open_positions: list[dict[str, Any]] | None,
    ) -> str | None:
        if not decision.is_exit:
            return None

        raw = _safe_dict(decision.raw_response)
        close_evidence = _safe_dict(raw.get("close_evidence"))
        execution_profit = _safe_dict(raw.get("execution_profit_protection"))
        exit_intent = classify_exit_intent(decision)
        raw = _safe_dict(decision.raw_response)
        close_evidence = _safe_dict(raw.get("close_evidence"))
        profit_exit = bool(close_evidence.get("profit_protection") or execution_profit.get("allow"))
        if not profit_exit:
            return None

        target_side = "long" if decision.action == Action.CLOSE_LONG else "short"
        target_symbol = self.normalize_symbol(decision.symbol)
        matches = []
        for pos in open_positions or []:
            if str(pos.get("model_name") or "") != decision.model_name:
                continue
            if str(pos.get("side") or "").lower() != target_side:
                continue
            if self.normalize_symbol(pos.get("symbol")) != target_symbol:
                continue
            matches.append(pos)
        if not matches:
            return None

        latest_price = await self.latest_price_provider(decision.symbol)
        if latest_price <= 0:
            return "利润保护平仓前未能重新获取最新价格，系统不使用过期浮盈判断执行锁盈单。"

        estimated_unrealized = 0.0
        reported_unrealized = 0.0
        reported_available = False
        total_qty = 0.0
        total_notional = 0.0
        for pos in matches:
            qty = abs(
                _safe_float(pos.get("quantity") or pos.get("contracts") or pos.get("sz"), 0.0)
            )
            info = _safe_dict(pos.get("info"))
            contract_size = _safe_float(
                pos.get("contract_size") or pos.get("contractSize") or info.get("ctVal"),
                1.0,
            )
            qty_for_pnl = qty * (contract_size if contract_size > 0 else 1.0)
            entry = _safe_float(
                pos.get("entry_price") or pos.get("entryPrice") or pos.get("avgPx"),
                0.0,
            )
            if qty_for_pnl <= 0 or entry <= 0:
                continue
            gross = (
                (latest_price - entry) * qty_for_pnl
                if target_side == "long"
                else (entry - latest_price) * qty_for_pnl
            )
            estimated_unrealized += gross
            total_qty += qty_for_pnl
            total_notional += entry * qty_for_pnl
            reported = _safe_float(
                (
                    pos.get("unrealized_pnl")
                    if pos.get("unrealized_pnl") is not None
                    else (
                        pos.get("unrealizedPnl")
                        if pos.get("unrealizedPnl") is not None
                        else (
                            pos.get("upl")
                            if pos.get("upl") is not None
                            else (
                                info.get("upl")
                                if info.get("upl") is not None
                                else info.get("unrealizedPnl")
                            )
                        )
                    )
                ),
                0.0,
            )
            if abs(reported) > 1e-12:
                reported_available = True
                reported_unrealized += reported
        if total_qty <= 0:
            return None

        # OKX swap positions may expose quantity as contract count while the DB
        # stores base quantity. Prefer the already-synced PnL when it agrees on
        # direction, and use the larger positive value to avoid blocking valid
        # lock-profit exits because of unit conversion differences.
        total_unrealized = estimated_unrealized
        if reported_available:
            if estimated_unrealized < -1e-9 and reported_unrealized > 0:
                total_unrealized = estimated_unrealized
            elif estimated_unrealized > 0 and reported_unrealized > 0:
                total_unrealized = max(estimated_unrealized, reported_unrealized)
            else:
                total_unrealized = reported_unrealized

        standard_min_profit = max(self.min_net_usdt * 0.25, 0.05)
        estimated_fee_buffer = max(total_notional * ESTIMATED_TAKER_FEE_PCT * 2.0, 0.0)
        small_position_min_profit = max(
            _EXIT_PARAMS.small_position_profit_lock_min_net_usdt,
            total_notional * _EXIT_PARAMS.small_position_profit_lock_min_pnl_ratio,
            estimated_fee_buffer * _EXIT_PARAMS.small_position_profit_lock_min_fee_multiple,
        )
        small_position_profit_lock = bool(
            close_evidence.get("small_position_profit_lock")
            and 0 < total_notional <= _EXIT_PARAMS.small_position_profit_lock_max_notional_usdt
        )
        min_profit = (
            min(standard_min_profit, small_position_min_profit)
            if small_position_profit_lock
            else standard_min_profit
        )
        if total_unrealized > min_profit:
            return None

        non_profit_exit_evidence = bool(
            close_evidence.get("hard_risk")
            or close_evidence.get("raw_hard_risk")
            or close_evidence.get("position_loss")
            or close_evidence.get("strong_opposite_pressure")
            or close_evidence.get("moderate_opposite_pressure")
            or close_evidence.get("profit_retrace_protection")
            or close_evidence.get("predictive_reversal_exit")
            or close_evidence.get("predictive_full_exit")
            or exit_intent in PROTECTIVE_DOWNSIDE_INTENTS
        )
        raw["execution_profit_protection_guard"] = {
            "applied": not non_profit_exit_evidence,
            "latest_price": latest_price,
            "target_side": target_side,
            "estimated_unrealized_pnl_from_price": round(estimated_unrealized, 6),
            "reported_unrealized_pnl": (
                round(reported_unrealized, 6) if reported_available else None
            ),
            "estimated_unrealized_pnl": round(total_unrealized, 6),
            "min_required_profit": round(min_profit, 6),
            "standard_min_required_profit": round(standard_min_profit, 6),
            "small_position_profit_lock": small_position_profit_lock,
            "small_position_min_required_profit": round(small_position_min_profit, 6),
            "total_notional": round(total_notional, 6),
            "non_profit_exit_evidence": non_profit_exit_evidence,
            "reason": (
                "最新价格复核显示该仓位已不满足纯锁盈条件；但存在趋势反转/硬风险证据，"
                "本次不再按锁盈不足拦截，继续交给平仓执行。"
                if non_profit_exit_evidence
                else "最新价格复核显示该仓位已不满足纯锁盈条件。"
            ),
        }
        decision.raw_response = raw
        if non_profit_exit_evidence:
            return None
        return (
            f"利润保护执行前复核未通过：按最新价格 {latest_price:g} 估算该仓位浮盈为 "
            f"{total_unrealized:.4f}U，未达到锁盈所需最小浮盈 {min_profit:.4f}U；"
            "本次不按锁定利润路径平仓。"
        )
