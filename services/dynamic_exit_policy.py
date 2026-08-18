"""Unified dynamic exit sizing from fee-after PnL and downside pressure."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
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
from services.execution_cost_model import funding_cost_estimate
from services.paper_bootstrap_canary import assess_paper_canary_position_horizon
from services.paper_training import assess_paper_training_position_horizon

EARLY_EXIT_OBSERVATION_MINUTES = 10.0
MIN_AUTOMATED_EXIT_FRACTION = 0.05
MIN_EARLY_MODEL_EXIT_PRESSURE = MIN_AUTOMATED_EXIT_FRACTION * 2.0


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


def _parse_utc_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                numeric = float(text)
            except (TypeError, ValueError):
                return None
            if numeric > 100_000_000_000:
                numeric /= 1000.0
            try:
                parsed = datetime.fromtimestamp(numeric, UTC)
            except (OverflowError, OSError, ValueError):
                return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class DynamicExitAssessment:
    eligible: bool
    reason: str
    close_fraction: float
    hard_risk: bool
    gross_unrealized_pnl_usdt: float
    fee_after_unrealized_pnl_usdt: float
    funding_fee_usdt: float
    settled_funding_fee: float
    expected_future_funding_cashflow: float
    current_lifecycle_net_pnl: float
    projected_hold_net_pnl: float
    next_funding_time: str | None
    funding_fee_included: bool
    funding_evidence_status: str
    funding_evidence_eligible: bool
    funding_loss_budget_crossed: bool
    projected_funding_budget_crossed: bool
    projected_funding_cost_usdt: float
    projected_funding_risk_usage: float
    funding_cost_projection_eligible: bool
    funding_cost_projection_reason: str
    lifecycle_net_pnl_usdt: float
    fee_buffer_usdt: float
    estimated_exit_cost_usdt: float
    execution_cost_complete: bool
    current_management_contract_complete: bool
    profit_retrace_ratio: float
    profit_lock_pressure: float
    stop_risk_usage: float
    continuation_deterioration: float
    opposite_pressure: float
    replacement_opportunity_eligible: bool
    replacement_symbol: str | None
    replacement_side: str | None
    replacement_expected_net_return_pct: float
    replacement_return_lcb_pct: float
    replacement_advantage_pct: float
    replacement_pressure: float
    portfolio_exposure_pressure: float
    model_requested_close_fraction: float
    model_exit_confidence: float
    model_exit_pressure: float
    planned_stop_crossed: bool
    position_age_minutes: float | None
    position_age_evidence_complete: bool
    early_exit_observation_active: bool
    economic_exit_evidence_complete: bool
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
    model_exit = _safe_dict(raw.get("model_exit_recommendation"))
    model_requested_close_fraction = _clamp(
        max(
            _safe_float(model_exit.get("requested_close_fraction"), 0.0),
            _safe_float(decision.suggested_close_fraction, 0.0),
        )
    )
    model_exit_confidence = _clamp(_safe_float(model_exit.get("confidence"), decision.confidence))
    model_exit_pressure = model_requested_close_fraction * model_exit_confidence
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
    funding_fee_observed = 0.0
    funding_fee_eligible = 0.0
    funding_bill_count = 0
    funding_contract_count = 0
    eligible_funding_contract_count = 0
    seen_funding_contracts: set[str] = set()
    canary_horizon_assessments: list[dict[str, Any]] = []
    training_horizon_assessments: list[dict[str, Any]] = []
    position_ages_minutes: list[float] = []
    observed_at = datetime.now(UTC)
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
        funding_identity = str(
            _safe_dict(management.get("policy_provenance")).get("contract_fingerprint")
            or ",".join(
                str(value) for value in management.get("position_fragment_ids") or []
            )
            or id(management)
        )
        if funding_identity not in seen_funding_contracts:
            seen_funding_contracts.add(funding_identity)
            funding_contract_count += 1
            contract_funding_fee = _safe_float(
                management.get(
                    "settled_funding_fee",
                    management.get("funding_fee_usdt"),
                ),
                0.0,
            )
            contract_bill_count = max(
                int(_safe_float(management.get("funding_bill_count"), 0.0)),
                0,
            )
            funding_fee_observed += contract_funding_fee
            funding_bill_count += contract_bill_count
            if management.get("funding_evidence_eligible") is True:
                eligible_funding_contract_count += 1
                funding_fee_eligible += contract_funding_fee
        canary_horizon_assessments.append(assess_paper_canary_position_horizon(position))
        training_horizon_assessments.append(assess_paper_training_position_horizon(position))
        opened_at = _parse_utc_datetime(
            position.get("created_at")
            or position.get("opened_at")
            or position.get("entry_time")
            or position.get("entry_timestamp")
        )
        if opened_at is not None:
            position_ages_minutes.append(
                max((observed_at - opened_at).total_seconds(), 0.0) / 60.0
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
    estimated_exit_cost_rate = max(
        exit_fee_rate,
        round_trip_cost_pct / 200.0 if execution_cost.get("production_eligible") is True else 0.0,
    )
    estimated_exit_cost = notional * estimated_exit_cost_rate
    price_fee_after_pnl = gross_pnl - fee_buffer
    funding_evidence_eligible = bool(
        funding_contract_count > 0
        and eligible_funding_contract_count == funding_contract_count
    )
    funding_fee_included = funding_evidence_eligible
    included_funding_fee = funding_fee_eligible if funding_fee_included else 0.0
    estimated_exit_slippage = (
        notional * max(_safe_float(execution_cost.get("slippage_pct"), 0.0), 0.0) / 200.0
        if execution_cost.get("production_eligible") is True
        else 0.0
    )
    current_lifecycle_net_pnl = (
        price_fee_after_pnl + included_funding_fee - estimated_exit_slippage
    )
    lifecycle_net_pnl = current_lifecycle_net_pnl
    peak_profit = max(peak_profit, gross_pnl)
    retrace = _clamp((peak_profit - gross_pnl) / peak_profit) if peak_profit > 0 else 0.0
    target_side = "long" if decision.action == Action.CLOSE_LONG else "short"
    feature_snapshot = _safe_dict(decision.feature_snapshot)
    funding_interval_minutes = _safe_float(
        feature_snapshot.get("funding_interval_minutes"),
        0.0,
    )
    if funding_interval_minutes <= 0.0:
        funding_interval_hours = _safe_float(
            feature_snapshot.get("funding_interval_hours"),
            0.0,
        )
        funding_interval_minutes = funding_interval_hours * 60.0
    funding_projection = funding_cost_estimate(
        feature_snapshot,
        side=target_side,
        horizon_minutes=funding_interval_minutes,
    )
    projected_funding_cost = (
        notional * float(funding_projection.adverse_cost_pct) / 100.0
        if funding_projection.production_eligible
        and funding_projection.adverse_cost_pct is not None
        else 0.0
    )
    expected_future_funding_cashflow = (
        notional * float(funding_projection.signed_cashflow_pct) / 100.0
        if funding_projection.production_eligible
        and funding_projection.signed_cashflow_pct is not None
        else 0.0
    )
    expected_future_price_return_pct = _safe_float(
        model_exit.get("expected_future_price_return_pct"),
        0.0,
    )
    expected_future_price_pnl = notional * expected_future_price_return_pct / 100.0
    projected_hold_net_pnl = (
        current_lifecycle_net_pnl
        + expected_future_price_pnl
        + expected_future_funding_cashflow
    )
    funding_evidence_status = (
        "complete"
        if funding_fee_included and funding_projection.production_eligible
        else "settled_complete_future_unavailable"
        if funding_fee_included
        else "settled_funding_unavailable"
    )
    funding_adjusted_gross_pnl = gross_pnl + min(included_funding_fee, 0.0)
    funding_adjusted_loss = max(-funding_adjusted_gross_pnl, 0.0)
    remaining_planned_risk = max(planned_risk - funding_adjusted_loss, 0.0)
    funding_loss_budget_crossed = bool(
        funding_fee_included
        and planned_risk > 0.0
        and funding_adjusted_loss + 1e-9 >= planned_risk
    )
    projected_funding_budget_crossed = bool(
        funding_projection.production_eligible
        and planned_risk > 0.0
        and projected_funding_cost > 0.0
        and projected_funding_cost + 1e-9 >= remaining_planned_risk
    )
    projected_funding_risk_usage = (
        _clamp(projected_funding_cost / planned_risk)
        if planned_risk > 0.0
        else 0.0
    )
    hard_risk = bool(
        planned_stop_crossed
        or funding_loss_budget_crossed
        or projected_funding_budget_crossed
    )
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
    # Entry and estimated exit fees are already/inevitably paid costs, not
    # evidence that market price has consumed the planned stop budget.
    stop_usage = (
        _clamp(funding_adjusted_loss / planned_risk)
        if planned_risk > 0
        else _clamp(funding_adjusted_loss / notional)
        if notional > 0
        else 0.0
    )
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
    adverse_direction_share = (
        _clamp(adverse_move / total_move) if total_move > 0.0 else 0.0
    )
    # Directional agreement is not loss magnitude. Scale it by the fraction of
    # the position's planned stop budget already consumed so tiny aligned moves
    # cannot force a full close in one review.
    continuation = adverse_direction_share * stop_usage
    replacement = _safe_dict(raw.get("stronger_opportunity"))
    replacement_cost = _safe_dict(replacement.get("execution_cost"))
    replacement_expected = max(
        _safe_float(replacement.get("expected_net_return_pct"), 0.0),
        0.0,
    )
    replacement_lcb = max(
        _safe_float(replacement.get("return_lcb_pct"), 0.0),
        0.0,
    )
    replacement_expected_loss_value = _safe_float(
        replacement.get("expected_loss_pct"),
        float("nan"),
    )
    replacement_expected_loss = (
        max(replacement_expected_loss_value, 0.0)
        if isfinite(replacement_expected_loss_value)
        else 0.0
    )
    replacement_eligible = bool(
        str(raw.get("execution_mode") or "").lower() == "paper"
        and replacement.get("available") is True
        and replacement.get("production_eligible") is True
        and replacement.get("execution_scope") == "paper_only"
        and replacement.get("production_permission") is False
        and replacement.get("creates_order") is False
        and replacement.get("can_increase_leverage") is False
        and replacement_expected > 0.0
        and replacement_lcb > 0.0
        and isfinite(replacement_expected_loss_value)
        and replacement_expected_loss_value >= 0.0
        and replacement_cost.get("production_eligible") is True
        and _normalized_symbol(replacement.get("symbol"))
        and _normalized_symbol(replacement.get("symbol"))
        != _normalized_symbol(decision.symbol)
        and str(replacement.get("side") or "").lower() in {"long", "short"}
    )
    current_fee_after_return_pct = (
        lifecycle_net_pnl / notional * 100.0 if notional > 0 else 0.0
    )
    replacement_advantage = (
        max(replacement_lcb - current_fee_after_return_pct, 0.0)
        if replacement_eligible
        else 0.0
    )
    replacement_scale = (
        abs(replacement_lcb)
        + abs(current_fee_after_return_pct)
        + replacement_expected_loss
    )
    replacement_pressure = (
        _clamp(replacement_advantage / replacement_scale)
        if replacement_scale > 0.0
        else 0.0
    )
    opposite = replacement_pressure
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
    funding_profit_lock_eligible = bool(
        funding_fee_included and abs(included_funding_fee) > 1e-12
    )
    profit_lock_pressure = (
        _clamp(max(lifecycle_net_pnl, 0.0) / planned_risk)
        if funding_profit_lock_eligible and planned_risk > 0.0
        else 0.0
    )
    # The prediction horizon is a label deadline, not position-exit authority.
    # Keep its elapsed state in the assessment for audit and training only.
    close_fraction = (
        1.0
        if hard_risk
        else continuous_budget_fraction(
            retrace,
            stop_usage,
            continuation,
            opposite,
            portfolio_pressure,
            model_exit_pressure,
            profit_lock_pressure,
        )
    )
    position_age_evidence_complete = bool(
        matches and len(position_ages_minutes) == len(matches)
    )
    position_age_minutes = min(position_ages_minutes) if position_ages_minutes else None
    early_exit_observation_active = bool(
        not hard_risk
        and close_fraction > 0.0
        and (
            not position_age_evidence_complete
            or position_age_minutes is None
            or position_age_minutes < EARLY_EXIT_OBSERVATION_MINUTES
        )
    )
    economic_exit_evidence_complete = not early_exit_observation_active
    reasons: list[str] = []
    if not matches:
        reasons.append("position_economics_missing")
    if not hard_risk and matches and not current_management_contract_complete:
        reasons.append("current_position_management_contract_incomplete")
    if not hard_risk and close_fraction <= 0:
        reasons.append("dynamic_exit_pressure_zero")
    if (
        not hard_risk
        and 0.0 < close_fraction
        and close_fraction + 1e-9 < MIN_AUTOMATED_EXIT_FRACTION
    ):
        reasons.append("dynamic_exit_fraction_below_execution_minimum")
    if (
        not hard_risk
        and gross_pnl > 0
        and lifecycle_net_pnl <= 0
        and stop_usage <= 0
        and continuation <= 0
    ):
        reasons.append("fee_after_profit_not_positive")
    if not hard_risk and not execution_cost_complete:
        reasons.append("exit_execution_cost_missing")
    if not hard_risk and matches and not position_age_evidence_complete:
        reasons.append("position_age_evidence_missing")
    elif not hard_risk and early_exit_observation_active:
        reasons.append("minimum_position_observation_not_elapsed")
    eligible = not reasons
    provenance = {
        "source": (
            "current_position_takeover_lifecycle_pnl_funding_peak_planned_stop_market_portfolio_and_replacement_facts"
        ),
        "observation_window": "current_position_review",
        "sample_count": len(matches),
        "generated_at": datetime.now(UTC).isoformat(),
        "strategy_version": "2026-08-19.dynamic-exit-lifecycle-guard.v15",
        "fallback_reason": ",".join(reasons),
        "early_exit_observation_minutes": EARLY_EXIT_OBSERVATION_MINUTES,
        "minimum_automated_exit_fraction": MIN_AUTOMATED_EXIT_FRACTION,
        "minimum_early_model_exit_pressure": MIN_EARLY_MODEL_EXIT_PRESSURE,
        "funding_fee_source": "current_position_management_contract",
        "funding_cost_projection_source": "current_okx_funding_rate_next_interval",
    }
    return DynamicExitAssessment(
        eligible=eligible,
        reason="dynamic_exit_policy_passed" if eligible else ",".join(reasons),
        close_fraction=round(close_fraction if eligible else 0.0, 8),
        hard_risk=hard_risk,
        gross_unrealized_pnl_usdt=round(gross_pnl, 8),
        fee_after_unrealized_pnl_usdt=round(price_fee_after_pnl, 8),
        funding_fee_usdt=round(funding_fee_observed, 8),
        settled_funding_fee=round(funding_fee_observed, 8),
        expected_future_funding_cashflow=round(
            expected_future_funding_cashflow,
            8,
        ),
        current_lifecycle_net_pnl=round(current_lifecycle_net_pnl, 8),
        projected_hold_net_pnl=round(projected_hold_net_pnl, 8),
        next_funding_time=funding_projection.next_funding_time,
        funding_fee_included=funding_fee_included,
        funding_evidence_status=funding_evidence_status,
        funding_evidence_eligible=funding_evidence_eligible,
        funding_loss_budget_crossed=funding_loss_budget_crossed,
        projected_funding_budget_crossed=projected_funding_budget_crossed,
        projected_funding_cost_usdt=round(projected_funding_cost, 8),
        projected_funding_risk_usage=round(projected_funding_risk_usage, 8),
        funding_cost_projection_eligible=funding_projection.production_eligible,
        funding_cost_projection_reason=funding_projection.reason,
        lifecycle_net_pnl_usdt=round(lifecycle_net_pnl, 8),
        fee_buffer_usdt=round(fee_buffer, 8),
        estimated_exit_cost_usdt=round(estimated_exit_cost, 8),
        execution_cost_complete=execution_cost_complete,
        current_management_contract_complete=current_management_contract_complete,
        profit_retrace_ratio=round(retrace, 8),
        profit_lock_pressure=round(profit_lock_pressure, 8),
        stop_risk_usage=round(stop_usage, 8),
        continuation_deterioration=round(continuation, 8),
        opposite_pressure=round(opposite, 8),
        replacement_opportunity_eligible=replacement_eligible,
        replacement_symbol=(
            str(replacement.get("symbol")) if replacement_eligible else None
        ),
        replacement_side=(
            str(replacement.get("side")) if replacement_eligible else None
        ),
        replacement_expected_net_return_pct=round(replacement_expected, 8),
        replacement_return_lcb_pct=round(replacement_lcb, 8),
        replacement_advantage_pct=round(replacement_advantage, 8),
        replacement_pressure=round(replacement_pressure, 8),
        portfolio_exposure_pressure=round(portfolio_pressure, 8),
        model_requested_close_fraction=round(model_requested_close_fraction, 8),
        model_exit_confidence=round(model_exit_confidence, 8),
        model_exit_pressure=round(model_exit_pressure, 8),
        planned_stop_crossed=planned_stop_crossed,
        position_age_minutes=(
            round(position_age_minutes, 8) if position_age_minutes is not None else None
        ),
        position_age_evidence_complete=position_age_evidence_complete,
        early_exit_observation_active=early_exit_observation_active,
        economic_exit_evidence_complete=economic_exit_evidence_complete,
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
            (_safe_float(item.get("horizon_minutes"), 0.0) for item in elapsed_training_horizons),
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
        else "reduce"
        if assessment.close_fraction > 0.0
        else "hold"
    )
    decision.raw_response = raw
    decision.position_size_pct = assessment.close_fraction
    return assessment


def attach_dynamic_exit_observation(
    decision: DecisionOutput,
    positions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Attach fee-after position economics without changing a hold decision."""

    symbol = _normalized_symbol(decision.symbol)
    sides = tuple(
        side
        for side in ("long", "short")
        if any(
            isinstance(position, dict)
            and (
                not _normalized_symbol(position.get("symbol"))
                or _normalized_symbol(position.get("symbol")) == symbol
            )
            and _position_side(position) == side
            for position in positions
        )
    )
    by_side: dict[str, dict[str, Any]] = {}
    for side in sides:
        probe = replace(
            decision,
            action=(Action.CLOSE_LONG if side == "long" else Action.CLOSE_SHORT),
            raw_response=dict(_safe_dict(decision.raw_response)),
            position_size_pct=0.0,
        )
        by_side[side] = assess_dynamic_exit(probe, positions).to_dict()

    if len(by_side) == 1:
        summary = dict(next(iter(by_side.values())))
    elif by_side:
        children = list(by_side.values())
        additive_fields = (
            "gross_unrealized_pnl_usdt",
            "fee_after_unrealized_pnl_usdt",
            "funding_fee_usdt",
            "settled_funding_fee",
            "expected_future_funding_cashflow",
            "current_lifecycle_net_pnl",
            "projected_hold_net_pnl",
            "lifecycle_net_pnl_usdt",
            "fee_buffer_usdt",
            "estimated_exit_cost_usdt",
            "projected_funding_cost_usdt",
        )
        summary = {
            field: round(sum(_safe_float(item.get(field), 0.0) for item in children), 8)
            for field in additive_fields
        }
        evidence_complete = all(
            item.get("funding_evidence_eligible") is True for item in children
        )
        future_complete = all(
            item.get("funding_cost_projection_eligible") is True for item in children
        )
        summary.update(
            {
                "eligible": False,
                "reason": "observation_only_hold_by_side",
                "close_fraction": 0.0,
                "hard_risk": any(item.get("hard_risk") is True for item in children),
                "next_funding_time": next(
                    (
                        item.get("next_funding_time")
                        for item in children
                        if item.get("next_funding_time")
                    ),
                    None,
                ),
                "funding_fee_included": evidence_complete,
                "funding_evidence_eligible": evidence_complete,
                "funding_evidence_status": (
                    "complete"
                    if evidence_complete and future_complete
                    else "settled_complete_future_unavailable"
                    if evidence_complete
                    else "settled_funding_unavailable"
                ),
                "funding_cost_projection_eligible": future_complete,
                "funding_cost_projection_reason": (
                    "current_direction_funding_cashflow_ready"
                    if future_complete
                    else "one_or_more_position_sides_unavailable"
                ),
                "profit_lock_pressure": max(
                    (_safe_float(item.get("profit_lock_pressure"), 0.0) for item in children),
                    default=0.0,
                ),
                "observed_close_fraction_by_side": {
                    side: _safe_float(item.get("close_fraction"), 0.0)
                    for side, item in by_side.items()
                },
            }
        )
    else:
        summary = {
            "eligible": False,
            "reason": "position_economics_missing",
            "close_fraction": 0.0,
            "settled_funding_fee": 0.0,
            "expected_future_funding_cashflow": 0.0,
            "current_lifecycle_net_pnl": 0.0,
            "projected_hold_net_pnl": 0.0,
            "next_funding_time": None,
            "funding_fee_included": False,
            "funding_evidence_eligible": False,
            "funding_evidence_status": "settled_funding_unavailable",
            "funding_cost_projection_eligible": False,
            "funding_cost_projection_reason": "position_economics_missing",
            "profit_lock_pressure": 0.0,
        }

    summary.update(
        {
            "observation_only": True,
            "observed_action": decision.action.value,
            "by_side": by_side,
        }
    )
    raw = dict(_safe_dict(decision.raw_response))
    raw["dynamic_exit_policy"] = summary
    decision.raw_response = raw
    return summary
