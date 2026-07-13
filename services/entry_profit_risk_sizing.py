"""Dynamic fee-after-return entry sizing.

This module is the only production sizing boundary before the final return
execution adjudication. Sizing uses current return, cost, stop distance,
portfolio exposure, and account state only.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from typing import Any

from ai_brain.base_model import DecisionOutput
from services.dynamic_leverage_allocator import DynamicLeverageAllocator, DynamicLeverageInput

EntryProfitRiskSizingEvaluator = Callable[
    [DecisionOutput, str, list[dict[str, Any]]],
    Awaitable[None],
]
EntryBalanceProvider = Callable[[str, DecisionOutput | None], Awaitable[float | None]]


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if isfinite(number) else default


def _clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _normalized_ratio(value: Any) -> float:
    number = max(_safe_float(value, 0.0), 0.0)
    return number / 100.0 if number > 1.0 else number


def _production_source_count(opportunity: dict[str, Any]) -> int:
    breakdown = _safe_dict(opportunity.get("expected_net_breakdown"))
    return sum(
        1
        for component in _safe_list(breakdown.get("components"))
        if _safe_dict(component).get("production_eligible") is True
    )


def _portfolio_exposure_fraction(
    positions: list[dict[str, Any]],
    *,
    balance: float,
) -> float:
    exposure = 0.0
    for position in positions:
        item = _safe_dict(position)
        explicit = _safe_float(
            item.get("portfolio_exposure_pct", item.get("position_size_pct")),
            0.0,
        )
        if explicit > 0:
            exposure += _normalized_ratio(explicit)
            continue
        notional = abs(
            _safe_float(
                item.get("notional_usdt", item.get("position_value_usdt")),
                0.0,
            )
        )
        if notional > 0 and balance > 0:
            exposure += notional / balance
    return _clamp(exposure)


def _atr_ratio(decision: DecisionOutput) -> float:
    snapshot = _safe_dict(decision.feature_snapshot)
    explicit = _normalized_ratio(snapshot.get("atr_pct"))
    if explicit > 0:
        return explicit
    atr = max(_safe_float(snapshot.get("atr_14"), 0.0), 0.0)
    price = max(
        _safe_float(snapshot.get("current_price", snapshot.get("close")), 0.0),
        0.0,
    )
    return atr / price if atr > 0 and price > 0 else 0.0


@dataclass(slots=True)
class EntryProfitRiskSizingPolicy:
    """Generate entry size and leverage from current return and account state."""

    evaluator: EntryProfitRiskSizingEvaluator | None = None
    allocated_order_balance: EntryBalanceProvider | None = None
    dynamic_leverage_allocator: Any | None = None

    async def apply(
        self,
        decision: DecisionOutput,
        model_mode: str,
        open_positions: list[dict[str, Any]] | None = None,
    ) -> None:
        if self.evaluator is not None:
            await self.evaluator(decision, model_mode, open_positions or [])
            return
        if not decision.is_entry:
            return
        if self.allocated_order_balance is None:
            raise RuntimeError(
                "EntryProfitRiskSizingPolicy requires allocated_order_balance"
            )

        raw = _safe_dict(decision.raw_response)
        opportunity = _safe_dict(raw.get("opportunity_score"))
        execution_cost = _safe_dict(opportunity.get("execution_cost"))
        expected_net = _safe_float(
            opportunity.get("expected_net_return_pct"),
            float("nan"),
        )
        expected_loss = max(_safe_float(opportunity.get("expected_loss_pct"), 0.0), 0.0)
        loss_probability = _clamp(
            _safe_float(opportunity.get("server_profit_loss_probability"), 1.0)
        )
        tail_risk = _clamp(_safe_float(opportunity.get("tail_risk_score"), 1.0))
        profit_quality = max(_safe_float(opportunity.get("profit_quality_ratio"), 0.0), 0.0)
        source_count = _production_source_count(opportunity)
        balance = max(
            _safe_float(await self.allocated_order_balance(model_mode, decision), 0.0),
            0.0,
        )
        declared_stop = _normalized_ratio(decision.stop_loss_pct)
        declared_take_profit = _normalized_ratio(decision.take_profit_pct)
        atr_ratio = _atr_ratio(decision)
        expected_loss_distance = expected_loss / 100.0
        cost_pct = max(_safe_float(execution_cost.get("total_pct"), 0.0), 0.0)
        cost_distance = cost_pct / 100.0
        stress_stop = max(
            declared_stop,
            atr_ratio,
            expected_loss_distance,
            cost_distance,
        )
        portfolio_exposure = _portfolio_exposure_fraction(
            open_positions or [],
            balance=balance,
        )
        available_exposure = 1.0 - portfolio_exposure

        positive_return = max(expected_net, 0.0) if isfinite(expected_net) else 0.0
        dynamic_take_profit = max(
            declared_take_profit,
            (positive_return + cost_pct) / 100.0,
        )
        return_quality = positive_return / max(
            positive_return + expected_loss + cost_pct,
            1e-12,
        )
        survival_quality = (1.0 - loss_probability) * (1.0 - tail_risk)
        account_budget_fraction = _clamp(
            return_quality * survival_quality * available_exposure
        )
        max_loss = balance * account_budget_fraction * stress_stop

        reasons: list[str] = []
        if balance <= 0:
            reasons.append("allocated_account_balance_missing")
        if source_count <= 0:
            reasons.append("production_return_observations_missing")
        if not isfinite(expected_net) or expected_net <= 0:
            reasons.append("fee_after_expected_return_not_positive")
        if execution_cost.get("production_eligible") is not True or cost_pct <= 0:
            reasons.append("production_execution_cost_missing")
        if stress_stop <= 0:
            reasons.append("dynamic_stop_distance_missing")
        if dynamic_take_profit <= 0:
            reasons.append("dynamic_take_profit_distance_missing")
        if account_budget_fraction <= 0 or max_loss <= 0:
            reasons.append("dynamic_account_risk_budget_zero")

        eligible = not reasons
        base_size = account_budget_fraction if eligible else 0.0
        allocator = self.dynamic_leverage_allocator or DynamicLeverageAllocator()
        leverage_decision = allocator.allocate(
            DynamicLeverageInput(
                symbol=decision.symbol,
                requested_leverage=_safe_float(decision.suggested_leverage, 1.0),
                system_max_leverage=max(
                    _safe_float(decision.suggested_leverage, 1.0),
                    1.0,
                ),
                balance=balance,
                position_size_pct=base_size,
                stress_stop_loss_pct=stress_stop,
                max_loss_usdt=max_loss,
                expected_net_return_pct=positive_return,
                profit_quality_ratio=profit_quality,
                loss_probability=loss_probability,
                tail_risk_score=tail_risk,
                aligned_source_count=source_count,
                atr_pct=atr_ratio,
                execution_cost=execution_cost,
                portfolio_exposure_pct=portfolio_exposure,
            )
        )
        leverage = float(leverage_decision.final_integer_leverage) if eligible else 1.0
        final_size = (
            min(
                base_size,
                max_loss / max(balance * leverage * stress_stop, 1e-12),
            )
            if eligible
            else 0.0
        )
        planned_loss = balance * final_size * leverage * stress_stop
        generated_at = datetime.now(UTC).isoformat()
        provenance = {
            "source": "fee_after_return_cost_stop_distance_account_and_portfolio_state",
            "observation_window": "current_decision_and_open_portfolio",
            "sample_count": source_count,
            "generated_at": generated_at,
            "strategy_version": "2026-07-12.dynamic-entry-risk-budget.v1",
            "fallback_reason": ",".join(reasons),
        }
        sizing = {
            "production_eligible": eligible,
            "reason": "dynamic_return_risk_budget_ready" if eligible else ",".join(reasons),
            "account_balance_usdt": round(balance, 8),
            "position_size_pct": round(final_size, 8),
            "planned_stop_loss_usdt": round(planned_loss, 8),
            "max_stop_loss_usdt": round(max_loss, 8),
            "declared_stop_loss_pct": round(declared_stop, 8),
            "stress_stop_loss_pct": round(stress_stop, 8),
            "declared_take_profit_pct": round(declared_take_profit, 8),
            "dynamic_take_profit_pct": round(dynamic_take_profit, 8),
            "atr_pct": round(atr_ratio, 8),
            "return_quality": round(return_quality, 8),
            "survival_quality": round(survival_quality, 8),
            "portfolio_exposure_pct": round(portfolio_exposure, 8),
            "available_exposure_pct": round(available_exposure, 8),
            "account_budget_fraction": round(account_budget_fraction, 8),
            "final_notional_usdt": round(balance * final_size * leverage, 8),
            "expected_profit_usdt": round(
                balance * final_size * leverage * positive_return / 100.0,
                8,
            ),
            "dynamic_leverage_decision": leverage_decision.to_dict(),
            "policy_provenance": provenance,
        }
        raw["profit_risk_sizing"] = sizing
        raw["dynamic_leverage_decision"] = leverage_decision.to_dict()
        decision.raw_response = raw
        decision.position_size_pct = final_size
        decision.suggested_leverage = leverage
        decision.stop_loss_pct = stress_stop
        decision.take_profit_pct = dynamic_take_profit
