"""Unified dynamic exit sizing from fee-after PnL and downside pressure."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from math import isfinite
from typing import Any

from ai_brain.base_model import Action, DecisionOutput
from services.current_position_management import (
    ALLOWED_MANAGEMENT_ACTIONS,
    CURRENT_POSITION_MANAGEMENT_KIND,
    CURRENT_POSITION_MANAGEMENT_VERSION,
    current_position_management_contract_complete,
)
from services.dynamic_policy_values import continuous_budget_fraction
from services.paper_bootstrap_canary import assess_paper_canary_position_horizon
from services.paper_training import assess_paper_training_position_horizon


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if isfinite(number) else default


def _clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _normalized_symbol(value: Any) -> str:
    return str(value or "").upper().replace("/", "").replace("-", "").replace(":USDT", "")


def _position_side(position: dict[str, Any]) -> str:
    side = str(
        position.get("side")
        or position.get("position_side")
        or _safe_dict(position.get("info")).get("posSide")
        or ""
    ).lower()
    return "long" if side in {"long", "buy"} else "short" if side in {"short", "sell"} else ""


@dataclass(frozen=True, slots=True)
class DynamicExitAssessment:
    eligible: bool
    reason: str
    close_fraction: float
    hard_risk: bool
    gross_unrealized_pnl_usdt: float
    fee_after_unrealized_pnl_usdt: float
    fee_buffer_usdt: float
    execution_cost_complete: bool
    current_management_contract_complete: bool
    profit_retrace_ratio: float
    stop_risk_usage: float
    continuation_deterioration: float
    opposite_pressure: float
    portfolio_exposure_pressure: float
    planned_stop_crossed: bool
    paper_canary_horizon_elapsed: bool
    paper_canary_horizon_minutes: int
    paper_canary_expires_at: str | None
    paper_training_horizon_elapsed: bool
    paper_training_horizon_minutes: float
    paper_training_expires_at: str | None
    current_management_contract_versions: tuple[str, ...]
    policy_provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _matching_positions(
    decision: DecisionOutput,
    positions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    target_side = "long" if decision.action == Action.CLOSE_LONG else "short"
    symbol = _normalized_symbol(decision.symbol)
    return [
        item
        for item in positions
        if isinstance(item, dict)
        and (
            not _normalized_symbol(item.get("symbol"))
            or _normalized_symbol(item.get("symbol")) == symbol
        )
        and _position_side(item) == target_side
    ]


def assess_dynamic_exit(
    decision: DecisionOutput,
    positions: list[dict[str, Any]],
) -> DynamicExitAssessment:
    raw = _safe_dict(decision.raw_response)
    matches = _matching_positions(decision, positions)

    gross_pnl = 0.0
    notional = 0.0
    entry_fees = 0.0
    exit_fee_rates: list[float] = []
    planned_risk = 0.0
    peak_profit = 0.0
    planned_stop_crossed = False
    management_contracts: list[dict[str, Any]] = []
    management_pressure_values: list[float] = []
    canary_horizon_assessments: list[dict[str, Any]] = []
    training_horizon_assessments: list[dict[str, Any]] = []
    for position in matches:
        qty = abs(
            _safe_float(
                position.get("quantity", position.get("qty", position.get("contracts"))),
                0.0,
            )
        )
        entry = max(
            _safe_float(
                position.get("entry_price", position.get("entryPrice", position.get("avgPx"))),
                0.0,
            ),
            0.0,
        )
        current = max(
            _safe_float(
                position.get(
                    "current_price",
                    position.get("markPrice", position.get("lastPrice")),
                ),
                0.0,
            ),
            0.0,
        )
        position_notional = abs(
            _safe_float(
                position.get("notional_usdt", position.get("position_value_usdt")),
                0.0,
            )
        )
        if position_notional <= 0 and qty > 0 and current > 0:
            position_notional = qty * current
        reported = position.get("unrealized_pnl", position.get("unrealizedPnl"))
        if reported is not None:
            pnl = _safe_float(reported, 0.0)
        elif qty > 0 and entry > 0 and current > 0:
            pnl = (
                (current - entry) * qty
                if _position_side(position) == "long"
                else (entry - current) * qty
            )
        else:
            pnl = 0.0
        gross_pnl += pnl
        notional += position_notional
        entry_fees += max(
            _safe_float(position.get("entry_fee_usdt", position.get("entry_fee")), 0.0),
            0.0,
        )
        raw_fee_rate = max(
            _safe_float(
                position.get(
                    "exit_fee_rate",
                    position.get("taker_fee_rate", position.get("fee_rate")),
                ),
                0.0,
            ),
            0.0,
        )
        if raw_fee_rate > 0.0:
            exit_fee_rates.append(raw_fee_rate)
        elif position_notional > 0.0:
            actual_entry_fee = max(
                _safe_float(
                    position.get("entry_fee_usdt", position.get("entry_fee")),
                    0.0,
                ),
                0.0,
            )
            if actual_entry_fee > 0.0:
                exit_fee_rates.append(actual_entry_fee / position_notional)
        stop_distance = max(_safe_float(position.get("stop_loss_pct"), 0.0), 0.0)
        stop_price = max(_safe_float(position.get("stop_loss"), 0.0), 0.0)
        if stop_distance <= 0 and stop_price > 0 and entry > 0:
            stop_distance = abs(entry - stop_price) / entry
        if stop_price > 0 and current > 0:
            planned_stop_crossed = planned_stop_crossed or (
                (_position_side(position) == "long" and current <= stop_price)
                or (_position_side(position) == "short" and current >= stop_price)
            )
        planned_risk += position_notional * stop_distance
        peak_profit = max(
            peak_profit,
            _safe_float(
                position.get("peak_unrealized_pnl", position.get("peak_pnl_usdt")),
                0.0,
            ),
        )
        management = _safe_dict(position.get("current_management_contract"))
        management_contracts.append(management)
        canary_horizon_assessments.append(assess_paper_canary_position_horizon(position))
        training_horizon_assessments.append(
            assess_paper_training_position_horizon(position)
        )
        if management.get("management_eligible") is True and not management.get("blockers"):
            management_pressure_values.append(
                _clamp(_safe_float(management.get("portfolio_concentration_pressure"), 0.0))
            )

    execution_cost = _safe_dict(raw.get("execution_cost"))
    round_trip_cost_pct = max(_safe_float(execution_cost.get("total_pct"), 0.0), 0.0)
    if not exit_fee_rates and execution_cost.get("production_eligible") is True:
        exit_fee_rates.append(round_trip_cost_pct / 200.0)
    execution_cost_complete = bool(exit_fee_rates or notional <= 0.0)
    exit_fee_rate = sum(exit_fee_rates) / len(exit_fee_rates) if exit_fee_rates else 0.0
    close_fee = notional * exit_fee_rate
    fee_buffer = entry_fees + close_fee
    net_pnl = gross_pnl - fee_buffer
    peak_profit = max(peak_profit, gross_pnl)
    retrace = _clamp((peak_profit - gross_pnl) / peak_profit) if peak_profit > 0 else 0.0
    hard_risk = planned_stop_crossed
    elapsed_canary_horizons = [
        item
        for item in canary_horizon_assessments
        if item.get("authorized") is True and item.get("elapsed") is True
    ]
    paper_canary_horizon_elapsed = bool(elapsed_canary_horizons)
    elapsed_training_horizons = [
        item
        for item in training_horizon_assessments
        if item.get("authorized") is True and item.get("elapsed") is True
    ]
    paper_training_horizon_elapsed = bool(elapsed_training_horizons)
    model_horizon_elapsed = bool(
        paper_canary_horizon_elapsed or paper_training_horizon_elapsed
    )
    stop_usage = (
        _clamp(max(-net_pnl, 0.0) / planned_risk)
        if planned_risk > 0
        else _clamp(max(-net_pnl, 0.0) / notional) if notional > 0 else 0.0
    )
    target_side = "long" if decision.action == Action.CLOSE_LONG else "short"
    feature_snapshot = _safe_dict(decision.feature_snapshot)
    market_returns = [
        _safe_float(feature_snapshot.get(name), 0.0)
        for name in ("returns_1", "returns_5", "returns_20")
    ]
    total_move = sum(abs(value) for value in market_returns)
    adverse_move = sum(
        abs(value)
        for value in market_returns
        if (target_side == "long" and value < 0.0) or (target_side == "short" and value > 0.0)
    )
    continuation = _clamp(adverse_move / total_move) if total_move > 0.0 else 0.0
    opposite = 0.0
    current_management_contract_complete = bool(
        matches
        and len(management_contracts) == len(matches)
        and all(
            contract.get("management_eligible") is True
            and contract.get("contract_version") == CURRENT_POSITION_MANAGEMENT_VERSION
            and contract.get("kind") == CURRENT_POSITION_MANAGEMENT_KIND
            and contract.get("entry_fee_evidence_complete") is True
            and contract.get("protection_evidence_complete") is True
            and contract.get("can_expand_position") is False
            and contract.get("can_increase_leverage") is False
            and tuple(contract.get("allowed_actions") or ()) == ALLOWED_MANAGEMENT_ACTIONS
            and not contract.get("blockers")
            and current_position_management_contract_complete(position, contract)
            for position, contract in zip(matches, management_contracts, strict=True)
        )
    )
    adverse_position_pressure = continuous_budget_fraction(
        retrace,
        stop_usage,
        continuation,
        opposite,
    )
    portfolio_pressure = (
        max(management_pressure_values, default=0.0) * adverse_position_pressure
        if current_management_contract_complete
        else 0.0
    )
    close_fraction = (
        1.0
        if hard_risk or model_horizon_elapsed
        else continuous_budget_fraction(
            retrace,
            stop_usage,
            continuation,
            opposite,
            portfolio_pressure,
        )
    )
    reasons: list[str] = []
    if not matches:
        reasons.append("position_economics_missing")
    if (
        not hard_risk
        and not model_horizon_elapsed
        and matches
        and not current_management_contract_complete
    ):
        reasons.append("current_position_management_contract_incomplete")
    if not hard_risk and not model_horizon_elapsed and close_fraction <= 0:
        reasons.append("dynamic_exit_pressure_zero")
    if (
        not hard_risk
        and not model_horizon_elapsed
        and gross_pnl > 0
        and net_pnl <= 0
        and stop_usage <= 0
        and continuation <= 0
    ):
        reasons.append("fee_after_profit_not_positive")
    if not hard_risk and not model_horizon_elapsed and not execution_cost_complete:
        reasons.append("exit_execution_cost_missing")
    eligible = not reasons
    provenance = {
        "source": (
            "current_position_takeover_fee_after_pnl_peak_planned_stop_market_and_portfolio_facts"
        ),
        "observation_window": "current_position_review",
        "sample_count": len(matches),
        "generated_at": datetime.now(UTC).isoformat(),
        "strategy_version": "2026-07-15.dynamic-exit-authoritative-facts.v3",
        "fallback_reason": ",".join(reasons),
    }
    return DynamicExitAssessment(
        eligible=eligible,
        reason="dynamic_exit_policy_passed" if eligible else ",".join(reasons),
        close_fraction=round(close_fraction if eligible else 0.0, 8),
        hard_risk=hard_risk,
        gross_unrealized_pnl_usdt=round(gross_pnl, 8),
        fee_after_unrealized_pnl_usdt=round(net_pnl, 8),
        fee_buffer_usdt=round(fee_buffer, 8),
        execution_cost_complete=execution_cost_complete,
        current_management_contract_complete=current_management_contract_complete,
        profit_retrace_ratio=round(retrace, 8),
        stop_risk_usage=round(stop_usage, 8),
        continuation_deterioration=round(continuation, 8),
        opposite_pressure=round(opposite, 8),
        portfolio_exposure_pressure=round(portfolio_pressure, 8),
        planned_stop_crossed=planned_stop_crossed,
        paper_canary_horizon_elapsed=paper_canary_horizon_elapsed,
        paper_canary_horizon_minutes=max(
            (int(item.get("horizon_minutes") or 0) for item in elapsed_canary_horizons),
            default=0,
        ),
        paper_canary_expires_at=next(
            (
                str(item.get("expires_at"))
                for item in elapsed_canary_horizons
                if item.get("expires_at")
            ),
            None,
        ),
        paper_training_horizon_elapsed=paper_training_horizon_elapsed,
        paper_training_horizon_minutes=max(
            (
                _safe_float(item.get("horizon_minutes"), 0.0)
                for item in elapsed_training_horizons
            ),
            default=0.0,
        ),
        paper_training_expires_at=next(
            (
                str(item.get("expires_at"))
                for item in elapsed_training_horizons
                if item.get("expires_at")
            ),
            None,
        ),
        current_management_contract_versions=tuple(
            sorted(
                {
                    str(contract.get("contract_version") or "")
                    for contract in management_contracts
                    if str(contract.get("contract_version") or "")
                }
            )
        ),
        policy_provenance=provenance,
    )


def apply_dynamic_exit(
    decision: DecisionOutput,
    positions: list[dict[str, Any]],
) -> DynamicExitAssessment:
    assessment = assess_dynamic_exit(decision, positions)
    raw = _safe_dict(decision.raw_response)
    raw["dynamic_exit_policy"] = assessment.to_dict()
    raw["close_fraction"] = assessment.close_fraction
    raw["action_plan"] = (
        "close"
        if assessment.close_fraction >= 1.0
        else "reduce" if assessment.close_fraction > 0.0 else "hold"
    )
    decision.raw_response = raw
    decision.position_size_pct = assessment.close_fraction
    return assessment
