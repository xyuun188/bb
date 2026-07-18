"""Paper-only bootstrap lifecycle for collecting model-bound trade evidence.

This module never grants production permission.  It selects an already governed
canary artifact for OKX demo sampling, applies a small independent risk budget,
and fails closed when frequency, portfolio, or loss-streak guards are unavailable.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import isclose, isfinite
from typing import Any

from sqlalchemy import select

from ai_brain.base_model import Action, DecisionOutput
from config.settings import settings
from db.session import get_read_session_ctx
from models.decision import AIDecision
from services.entry_profit_risk_sizing import (
    build_portfolio_risk_snapshot,
    select_okx_leverage_tier,
)

PAPER_BOOTSTRAP_CANARY_VERSION = "2026-07-17.paper-bootstrap-canary.v1"
PAPER_BOOTSTRAP_SIZING_VERSION = "2026-07-17.paper-bootstrap-sizing.v1"
PAPER_BOOTSTRAP_MAX_OPEN_POSITIONS = 1
PAPER_BOOTSTRAP_MAX_DAILY_ENTRIES = 4
PAPER_BOOTSTRAP_MAX_CONSECUTIVE_LOSSES = 2
PAPER_BOOTSTRAP_SINGLE_TRADE_EQUITY_RISK = 0.0005
PAPER_BOOTSTRAP_PORTFOLIO_EQUITY_RISK = 0.001

BalanceProvider = Callable[[str, DecisionOutput], Awaitable[float | None]]
ExchangeFactsProvider = Callable[
    [str, DecisionOutput, list[dict[str, Any]]],
    Awaitable[dict[str, Any]],
]
HistoryProvider = Callable[[], Awaitable[list[Any]]]


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def _positive(value: Any) -> float:
    result = _finite(value)
    return max(result or 0.0, 0.0)


def _normalized_ratio(value: Any) -> float:
    result = _positive(value)
    return result / 100.0 if result > 1.0 else result


def _row_value(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _row_raw(row: Any) -> dict[str, Any]:
    return _safe_dict(
        _row_value(row, "raw_llm_response")
        or _row_value(row, "raw_response")
        or _row_value(row, "decision_learning_snapshot")
    )


def _as_utc(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _model_distribution(signal: dict[str, Any], side: str) -> dict[str, Any]:
    predictions = _safe_list(signal.get("predictions"))
    primary = _safe_dict(predictions[0] if predictions else {})
    contracts = _safe_dict(
        primary.get("return_distribution_contract") or signal.get("return_distribution_contract")
    )
    return _safe_dict(contracts.get(side))


def select_paper_bootstrap_candidate(context: dict[str, Any] | None) -> dict[str, Any]:
    """Select a paper sample by governed information value, not production PnL claims."""

    ctx = _safe_dict(context)
    blocked = {
        "version": PAPER_BOOTSTRAP_CANARY_VERSION,
        "authorized": False,
        "requested": False,
        "execution_scope": "paper_only",
        "production_permission": False,
    }
    if str(ctx.get("trading_mode") or "").lower() != "paper":
        return {**blocked, "reason": "paper_execution_mode_required"}

    candidate_evidence = _safe_dict(ctx.get("entry_candidate_evidence"))
    production_count = sum(
        int(_positive(_safe_dict(candidate_evidence.get(side)).get("production_source_count")))
        for side in ("long", "short")
    )
    if production_count > 0:
        return {**blocked, "reason": "production_return_source_available"}

    signal = _safe_dict(ctx.get("ml_signal"))
    readiness = _safe_dict(signal.get("paper_canary"))
    if (
        signal.get("paper_canary_authorized") is not True
        or signal.get("artifact_lifecycle") != "canary"
        or readiness.get("authorized") is not True
        or readiness.get("execution_scope") != "paper_only"
        or readiness.get("production_permission") is not False
    ):
        return {**blocked, "reason": "governed_paper_canary_artifact_unavailable"}

    eligible_sides = {
        str(side).lower()
        for side in _safe_list(readiness.get("eligible_sides"))
        if str(side).lower() in {"long", "short"}
    }
    observations: list[dict[str, Any]] = []
    for side in sorted(eligible_sides):
        distribution = _model_distribution(signal, side)
        raw_expected = _finite(distribution.get("raw_expected_return_pct"))
        objective_expected = _finite(distribution.get("objective_expected_return_pct"))
        lower_quantile = _finite(distribution.get("lower_quantile_return_pct"))
        dispersion = _positive(distribution.get("dispersion_pct"))
        if raw_expected is None or objective_expected is None or lower_quantile is None:
            continue
        execution_cost = _safe_dict(_safe_dict(candidate_evidence.get(side)).get("execution_cost"))
        current_cost = _positive(execution_cost.get("total_pct"))
        observations.append(
            {
                "side": side,
                "raw_expected_return_pct": raw_expected,
                "objective_expected_return_pct": objective_expected,
                "lower_quantile_return_pct": lower_quantile,
                "dispersion_pct": dispersion,
                "current_execution_cost_pct": current_cost,
                "observed_net_return_pct": raw_expected - current_cost,
                "horizon_minutes": int(_positive(distribution.get("horizon_minutes")) or 10),
                "distribution_member_count": int(
                    _positive(distribution.get("distribution_member_count"))
                ),
                "source_authority": distribution.get("source_authority"),
            }
        )
    if not observations:
        return {**blocked, "reason": "paper_canary_distribution_incomplete"}

    observations.sort(
        key=lambda item: (
            item["objective_expected_return_pct"],
            item["observed_net_return_pct"],
            item["lower_quantile_return_pct"],
        ),
        reverse=True,
    )
    selected = observations[0]
    if len(observations) > 1:
        score_gap = (
            selected["objective_expected_return_pct"]
            - observations[1]["objective_expected_return_pct"]
        )
        if score_gap <= 0:
            return {**blocked, "reason": "paper_canary_direction_not_identifiable"}
    else:
        score_gap = abs(selected["objective_expected_return_pct"])
    uncertainty = max(
        selected["dispersion_pct"],
        abs(selected["objective_expected_return_pct"] - selected["lower_quantile_return_pct"]),
    )
    confidence = score_gap / max(score_gap + uncertainty, 1e-12)
    now = datetime.now(UTC)
    return {
        "version": PAPER_BOOTSTRAP_CANARY_VERSION,
        "authorized": True,
        "requested": True,
        "execution_scope": "paper_only",
        "production_permission": False,
        "purpose": "collect_version_bound_authoritative_cost_and_return_samples",
        "selected_side": selected["side"],
        "selected_observation": selected,
        "side_observations": observations,
        "direction_score_gap": score_gap,
        "confidence": max(0.0, min(confidence, 1.0)),
        "artifact_version": signal.get("model_version"),
        "artifact_lifecycle": signal.get("artifact_lifecycle"),
        "source_model": "local_ml_profit_quality",
        "source_sample_count": max(
            int(signal.get("trained_sample_count") or 0),
            int(selected.get("distribution_member_count") or 0),
        ),
        "generated_at": now.isoformat(),
        "policy_provenance": {
            "source": "governed_shadow_market_distribution_for_paper_bootstrap",
            "observation_window": "current_model_artifact_and_pre_order_market_snapshot",
            "sample_count": max(
                int(signal.get("trained_sample_count") or 0),
                int(selected.get("distribution_member_count") or 0),
            ),
            "generated_at": now.isoformat(),
            "strategy_version": PAPER_BOOTSTRAP_CANARY_VERSION,
            "fallback_reason": "",
        },
    }


@dataclass(frozen=True, slots=True)
class PaperBootstrapAssessment:
    eligible: bool
    reason: str
    details: dict[str, Any]


class PaperBootstrapCanaryPolicy:
    """Build and validate an independently bounded OKX-demo risk contract."""

    def __init__(
        self,
        *,
        allocated_order_balance: BalanceProvider,
        exchange_risk_facts: ExchangeFactsProvider,
        history_provider: HistoryProvider | None = None,
    ) -> None:
        self.allocated_order_balance = allocated_order_balance
        self.exchange_risk_facts = exchange_risk_facts
        self.history_provider = history_provider or self._load_history

    @staticmethod
    def is_claimed(decision: DecisionOutput) -> bool:
        contract = _safe_dict(_safe_dict(decision.raw_response).get("paper_bootstrap_canary"))
        return bool(
            decision.is_entry
            and contract.get("version") == PAPER_BOOTSTRAP_CANARY_VERSION
            and contract.get("requested") is True
        )

    async def preflight(
        self,
        decision: DecisionOutput,
        model_mode: str,
        open_positions: list[dict[str, Any]],
    ) -> PaperBootstrapAssessment:
        """Evaluate lifecycle guards before an entry action is persisted."""

        raw = _safe_dict(decision.raw_response)
        contract = _safe_dict(raw.get("paper_bootstrap_canary"))
        reasons = PaperBootstrapCanaryPolicy._contract_reasons(
            decision,
            model_mode,
            contract,
        )
        runtime_guard = await self._runtime_guard(contract, open_positions)
        reasons.extend(runtime_guard["blocking_reasons"])
        reasons = list(dict.fromkeys(reasons))
        eligible = not reasons
        contract = dict(contract)
        contract["runtime_guard"] = runtime_guard
        contract["runtime_preflight_authorized"] = eligible
        contract["runtime_preflight_blocking_reasons"] = reasons
        if not eligible:
            contract["runtime_authorized"] = False
            contract["runtime_blocking_reasons"] = reasons
            raw["profit_risk_sizing"] = {
                "production_eligible": False,
                "contract_lifecycle": "paper_bootstrap_canary",
                "reasons": reasons,
                "policy_provenance": {
                    **_safe_dict(contract.get("policy_provenance")),
                    "fallback_reason": ",".join(reasons),
                },
            }
            decision.position_size_pct = 0.0
            decision.suggested_leverage = 1.0
        raw["paper_bootstrap_canary"] = contract
        decision.raw_response = raw
        return PaperBootstrapAssessment(
            eligible=eligible,
            reason=(
                "paper_bootstrap_canary_runtime_preflight_ready" if eligible else ",".join(reasons)
            ),
            details={
                "contract": contract,
                "runtime_guard": runtime_guard,
                "sizing": _safe_dict(raw.get("profit_risk_sizing")),
            },
        )

    @staticmethod
    def demote_blocked_candidate_to_hold(
        decision: DecisionOutput,
        assessment: PaperBootstrapAssessment,
    ) -> bool:
        """Persist a blocked canary as hold while retaining its shadow direction."""

        if assessment.eligible or not PaperBootstrapCanaryPolicy.is_claimed(decision):
            return False
        candidate_action = decision.action.value
        raw = _safe_dict(decision.raw_response)
        contract = _safe_dict(raw.get("paper_bootstrap_canary"))
        contract["candidate_action"] = candidate_action
        contract["persisted_action"] = Action.HOLD.value
        contract["execution_intent"] = "observation_only_hold"
        contract["candidate_blocking_reason"] = assessment.reason
        raw["paper_bootstrap_canary"] = contract
        raw["paper_bootstrap_canary_observation"] = {
            "candidate_action": candidate_action,
            "persisted_action": Action.HOLD.value,
            "selected_side": contract.get("selected_side"),
            "reason": assessment.reason,
            "shadow_direction_preserved": True,
            "exchange_submission_allowed": False,
        }
        decision.action = Action.HOLD
        decision.position_size_pct = 0.0
        decision.suggested_leverage = 1.0
        decision.stop_loss_pct = 0.0
        decision.take_profit_pct = 0.0
        decision.reasoning = (
            "paper canary \u8fd0\u884c\u65f6\u98ce\u63a7\u672a\u6388\u6743\u672c\u8f6e\u5f00\u4ed3\uff0c"
            "\u5019\u9009\u65b9\u5411\u4ec5\u4fdd\u7559\u4e3a\u89c2\u5bdf\u8bc1\u636e\uff0c\u672c\u8f6e\u88c1\u51b3\u4e3a\u89c2\u671b\u3002"
        )
        decision.raw_response = raw
        return True

    async def prepare(
        self,
        decision: DecisionOutput,
        model_mode: str,
        open_positions: list[dict[str, Any]],
    ) -> PaperBootstrapAssessment:
        preflight = await self.preflight(decision, model_mode, open_positions)
        if not preflight.eligible:
            return preflight
        raw = _safe_dict(decision.raw_response)
        contract = _safe_dict(raw.get("paper_bootstrap_canary"))
        runtime_guard = _safe_dict(preflight.details.get("runtime_guard"))
        reasons: list[str] = []
        facts: dict[str, Any] = {}
        allocated_margin = 0.0
        if not reasons:
            try:
                facts = _safe_dict(
                    await self.exchange_risk_facts(model_mode, decision, open_positions)
                )
                allocated_margin = _positive(
                    await self.allocated_order_balance(model_mode, decision)
                )
            except Exception as exc:
                reasons.append(f"paper_canary_exchange_facts_unavailable:{type(exc).__name__}")

        if not reasons:
            self._attach_risk_contract(
                decision,
                contract=contract,
                facts=facts,
                allocated_margin=allocated_margin,
                runtime_guard=runtime_guard,
                open_positions=open_positions,
            )
            sizing = _safe_dict(_safe_dict(decision.raw_response).get("profit_risk_sizing"))
            reasons.extend(str(item) for item in _safe_list(sizing.get("reasons")) if item)

        eligible = not reasons
        contract = dict(contract)
        contract["runtime_guard"] = runtime_guard
        contract["runtime_authorized"] = eligible
        contract["runtime_blocking_reasons"] = list(dict.fromkeys(reasons))
        raw = _safe_dict(decision.raw_response)
        raw["paper_bootstrap_canary"] = contract
        if not eligible:
            raw["profit_risk_sizing"] = {
                "production_eligible": False,
                "contract_lifecycle": "paper_bootstrap_canary",
                "reasons": list(dict.fromkeys(reasons)),
                "policy_provenance": {
                    **_safe_dict(contract.get("policy_provenance")),
                    "fallback_reason": ",".join(dict.fromkeys(reasons)),
                },
            }
            decision.position_size_pct = 0.0
            decision.suggested_leverage = 1.0
        decision.raw_response = raw
        return PaperBootstrapAssessment(
            eligible=eligible,
            reason=(
                "paper_bootstrap_canary_contract_ready"
                if eligible
                else ",".join(dict.fromkeys(reasons))
            ),
            details={
                "contract": contract,
                "runtime_guard": runtime_guard,
                "sizing": _safe_dict(raw.get("profit_risk_sizing")),
            },
        )

    @staticmethod
    def assess(decision: DecisionOutput, model_mode: str) -> PaperBootstrapAssessment:
        raw = _safe_dict(decision.raw_response)
        contract = _safe_dict(raw.get("paper_bootstrap_canary"))
        sizing = _safe_dict(raw.get("profit_risk_sizing"))
        reasons = PaperBootstrapCanaryPolicy._contract_reasons(
            decision,
            model_mode,
            contract,
        )
        if contract.get("runtime_authorized") is not True:
            reasons.append("paper_canary_runtime_guard_not_authorized")
        if sizing.get("contract_lifecycle") != "paper_bootstrap_canary":
            reasons.append("paper_canary_sizing_lifecycle_mismatch")
        if sizing.get("production_eligible") is not True:
            reasons.append("paper_canary_risk_contract_ineligible")
        risk_budget = _positive(sizing.get("risk_budget_usdt"))
        planned_loss = _positive(sizing.get("planned_stressed_loss_usdt"))
        stress = _positive(sizing.get("stressed_loss_fraction"))
        final_notional = _positive(sizing.get("final_notional_usdt"))
        target_notional = _positive(sizing.get("target_notional_usdt"))
        final_margin = _positive(sizing.get("final_margin_usdt"))
        position_size = _positive(sizing.get("position_size_pct"))
        fingerprint = str(
            _safe_dict(sizing.get("policy_provenance")).get("contract_fingerprint") or ""
        )
        if risk_budget <= 0 or planned_loss <= 0 or planned_loss > risk_budget + 1e-8:
            reasons.append("paper_canary_risk_budget_invalid")
        if stress <= 0 or not isclose(
            planned_loss,
            final_notional * stress,
            rel_tol=1e-9,
            abs_tol=1e-8,
        ):
            reasons.append("paper_canary_stressed_loss_algebra_mismatch")
        if final_notional <= 0 or final_notional > target_notional + 1e-8:
            reasons.append("paper_canary_notional_invalid")
        if final_margin <= 0 or position_size <= 0 or not fingerprint:
            reasons.append("paper_canary_sizing_identity_incomplete")
        if not isclose(position_size, _positive(decision.position_size_pct), abs_tol=1e-8):
            reasons.append("paper_canary_decision_size_mismatch")
        reasons = list(dict.fromkeys(reasons))
        return PaperBootstrapAssessment(
            eligible=not reasons,
            reason="paper_bootstrap_canary_contract_ready" if not reasons else ",".join(reasons),
            details={"contract": contract, "sizing": sizing, "blocking_reasons": reasons},
        )

    @staticmethod
    def _contract_reasons(
        decision: DecisionOutput,
        model_mode: str,
        contract: dict[str, Any],
    ) -> list[str]:
        reasons: list[str] = []
        side = "long" if decision.action == Action.LONG else "short"
        if model_mode != "paper":
            reasons.append("paper_canary_live_execution_forbidden")
        if contract.get("authorized") is not True:
            reasons.append("paper_canary_not_authorized")
        if contract.get("execution_scope") != "paper_only":
            reasons.append("paper_canary_scope_invalid")
        if contract.get("production_permission") is not False:
            reasons.append("paper_canary_production_permission_invalid")
        if contract.get("artifact_lifecycle") != "canary":
            reasons.append("paper_canary_artifact_lifecycle_invalid")
        if contract.get("selected_side") != side:
            reasons.append("paper_canary_selected_side_mismatch")
        provenance = _safe_dict(contract.get("policy_provenance"))
        if (
            not provenance.get("source")
            or not provenance.get("observation_window")
            or int(_positive(provenance.get("sample_count"))) <= 0
            or not provenance.get("generated_at")
            or provenance.get("strategy_version") != PAPER_BOOTSTRAP_CANARY_VERSION
            or provenance.get("fallback_reason")
        ):
            reasons.append("paper_canary_provenance_incomplete")
        return reasons

    async def _runtime_guard(
        self,
        contract: dict[str, Any],
        open_positions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        account_open_positions = [
            item
            for item in open_positions
            if item.get("is_open", True) is not False and _positive(item.get("quantity", 1.0)) > 0
        ]
        reasons: list[str] = []
        try:
            history = list(await self.history_provider())
        except Exception as exc:
            history = []
            reasons.append(f"paper_canary_history_unavailable:{type(exc).__name__}")

        canary_rows = [
            row
            for row in history
            if _safe_dict(_row_raw(row).get("paper_bootstrap_canary")).get("authorized") is True
        ]
        open_canary_rows = [
            row for row in canary_rows if not str(_row_value(row, "outcome") or "").strip()
        ]
        if len(open_canary_rows) >= PAPER_BOOTSTRAP_MAX_OPEN_POSITIONS:
            reasons.append("paper_canary_open_position_limit_reached")
        now = datetime.now(UTC)
        today_start = datetime(now.year, now.month, now.day, tzinfo=UTC)
        daily_entries = sum(
            1
            for row in canary_rows
            if (
                _as_utc(_row_value(row, "executed_at") or _row_value(row, "created_at"))
                or datetime.min.replace(tzinfo=UTC)
            )
            >= today_start
        )
        if daily_entries >= PAPER_BOOTSTRAP_MAX_DAILY_ENTRIES:
            reasons.append("paper_canary_daily_entry_budget_exhausted")

        completed = [row for row in canary_rows if _row_value(row, "outcome")]
        consecutive_losses = 0
        for row in completed:
            if str(_row_value(row, "outcome") or "").lower() != "loss":
                break
            consecutive_losses += 1
        if consecutive_losses >= PAPER_BOOTSTRAP_MAX_CONSECUTIVE_LOSSES:
            reasons.append("paper_canary_consecutive_loss_circuit_open")

        last_executed = _as_utc(
            next(
                (
                    _row_value(row, "executed_at") or _row_value(row, "created_at")
                    for row in canary_rows
                    if _row_value(row, "executed_at") or _row_value(row, "created_at")
                ),
                None,
            )
        )
        horizon_minutes = int(
            _positive(_safe_dict(contract.get("selected_observation")).get("horizon_minutes")) or 10
        )
        cooldown_seconds = max(
            horizon_minutes * 60,
            max(int(settings.decision_interval_seconds or 60), 1) * 5,
        )
        if isinstance(last_executed, datetime):
            if now - last_executed < timedelta(seconds=cooldown_seconds):
                reasons.append("paper_canary_cooldown_active")
        return {
            "blocking_reasons": list(dict.fromkeys(reasons)),
            "open_position_count": len(open_canary_rows),
            "max_open_positions": PAPER_BOOTSTRAP_MAX_OPEN_POSITIONS,
            "account_open_position_count": len(account_open_positions),
            "open_position_source": "executed_canary_decisions_without_outcome",
            "daily_entry_count": daily_entries,
            "max_daily_entries": PAPER_BOOTSTRAP_MAX_DAILY_ENTRIES,
            "consecutive_loss_count": consecutive_losses,
            "max_consecutive_losses": PAPER_BOOTSTRAP_MAX_CONSECUTIVE_LOSSES,
            "cooldown_seconds": cooldown_seconds,
            "last_executed_at": (
                last_executed.isoformat() if isinstance(last_executed, datetime) else None
            ),
        }

    def _attach_risk_contract(
        self,
        decision: DecisionOutput,
        *,
        contract: dict[str, Any],
        facts: dict[str, Any],
        allocated_margin: float,
        runtime_guard: dict[str, Any],
        open_positions: list[dict[str, Any]],
    ) -> None:
        raw = _safe_dict(decision.raw_response)
        snapshot = _safe_dict(decision.feature_snapshot)
        pre_order = _safe_dict(raw.get("pre_order_execution_facts"))
        available_from_facts = _positive(facts.get("available_margin_usdt"))
        available_margin = (
            min(allocated_margin, available_from_facts)
            if allocated_margin > 0 and available_from_facts > 0
            else available_from_facts
        )
        equity = _positive(facts.get("account_equity_usdt"))
        price = max(
            _positive(snapshot.get("current_price")),
            _positive(snapshot.get("close")),
        )
        atr_fraction = _positive(snapshot.get("atr_14")) / price if price > 0 else 0.0
        volatility_fraction = _normalized_ratio(snapshot.get("volatility_20"))
        wick_fraction = _positive(snapshot.get("abnormal_wick_max_pct")) / 100.0
        opportunity = _safe_dict(raw.get("opportunity_score"))
        execution_cost = _safe_dict(opportunity.get("execution_cost"))
        cost_pct = max(
            _positive(execution_cost.get("total_pct")),
            _positive(pre_order.get("total_cost_pct")),
        )
        stress = max(
            _normalized_ratio(decision.stop_loss_pct),
            atr_fraction,
            volatility_fraction,
            wick_fraction,
            cost_pct * 3.0 / 100.0,
        )
        reasons: list[str] = []
        if facts.get("production_eligible") is not True:
            reasons.append("paper_canary_exchange_risk_facts_ineligible")
        if pre_order and pre_order.get("production_eligible") is not True:
            reasons.append("paper_canary_pre_order_facts_ineligible")
        if equity <= 0 or available_margin <= 0:
            reasons.append("paper_canary_account_capacity_unavailable")
        if stress <= 0:
            reasons.append("paper_canary_stressed_loss_fraction_missing")

        risk_budget = equity * PAPER_BOOTSTRAP_SINGLE_TRADE_EQUITY_RISK
        portfolio_budget = equity * PAPER_BOOTSTRAP_PORTFOLIO_EQUITY_RISK
        target_notional = risk_budget / stress if stress > 0 else 0.0
        final_notional = min(target_notional, available_margin)
        final_margin = final_notional
        position_size = final_margin / available_margin if available_margin > 0 else 0.0
        planned_loss = final_notional * stress
        if final_notional <= 0 or planned_loss <= 0 or planned_loss > risk_budget + 1e-8:
            reasons.append("paper_canary_independent_risk_budget_zero")

        contract_specs = _safe_dict(facts.get("contract_specs"))
        account_portfolio, account_portfolio_blockers = build_portfolio_risk_snapshot(
            open_positions,
            candidate_side=str(contract.get("selected_side") or ""),
            contract_specs=contract_specs,
        )
        account_portfolio["valuation_blockers"] = account_portfolio_blockers
        target_inst_id = str(facts.get("target_inst_id") or "").strip()
        target_contract_spec = _safe_dict(contract_specs.get(target_inst_id))
        leverage_tier_selection = select_okx_leverage_tier(
            facts.get("leverage_tiers"),
            target_notional_usdt=final_notional,
            mark_price=price,
            contract_spec=target_contract_spec,
        )
        if leverage_tier_selection.get("production_eligible") is not True:
            reasons.append(
                "paper_canary_leverage_tier_ineligible:"
                + str(leverage_tier_selection.get("reason") or "unknown")
            )

        generated_at = datetime.now(UTC).isoformat()
        selected_observation = _safe_dict(contract.get("selected_observation"))
        expected_net_return = _finite(selected_observation.get("observed_net_return_pct"))
        canary_portfolio = {
            "scope": "paper_bootstrap_canary_positions_only",
            "current_stressed_loss_usdt": 0.0,
            "current_margin_usdt": 0.0,
            "gross_notional_usdt": 0.0,
            "same_side_notional_usdt": 0.0,
            "direction_concentration": 0.0,
            "positions": [],
        }
        fingerprint_payload = {
            "artifact_version": contract.get("artifact_version"),
            "symbol": decision.symbol,
            "side": contract.get("selected_side"),
            "equity": round(equity, 8),
            "available_margin": round(available_margin, 8),
            "stress": round(stress, 8),
            "risk_budget": round(risk_budget, 8),
            "final_notional": round(final_notional, 8),
            "pre_order_fingerprint": pre_order.get("input_fingerprint"),
        }
        provenance = {
            "source": "paper_bootstrap_canary_independent_risk_budget",
            "observation_window": "current_okx_demo_account_market_and_canary_runtime_guard",
            "sample_count": max(int(contract.get("source_sample_count") or 0), 1),
            "generated_at": generated_at,
            "strategy_version": PAPER_BOOTSTRAP_SIZING_VERSION,
            "fallback_reason": ",".join(reasons),
            "contract_fingerprint": _canonical_sha256(fingerprint_payload),
        }
        sizing = {
            "contract_version": PAPER_BOOTSTRAP_SIZING_VERSION,
            "production_eligible": not reasons,
            "reason": (
                "paper_bootstrap_canary_independent_risk_budget_ready"
                if not reasons
                else ",".join(dict.fromkeys(reasons))
            ),
            "contract_lifecycle": "paper_bootstrap_canary",
            "execution_scope": "paper_only",
            "production_permission": False,
            "account_equity_usdt": round(equity, 8),
            "available_margin_usdt": round(available_margin, 8),
            "risk_budget_usdt": risk_budget,
            "single_trade_risk_budget_usdt": risk_budget,
            "portfolio_risk_budget_usdt": portfolio_budget,
            "remaining_portfolio_risk_budget_usdt": portfolio_budget,
            "current_portfolio_stressed_loss_usdt": 0.0,
            "planned_stressed_loss_usdt": planned_loss,
            "stressed_loss_fraction": stress,
            "target_notional_usdt": target_notional,
            "final_notional_usdt": final_notional,
            "final_margin_usdt": final_margin,
            "position_size_pct": position_size,
            "expected_net_return_pct": expected_net_return,
            "expected_profit_usdt": (
                final_notional * expected_net_return / 100.0
                if expected_net_return is not None
                else None
            ),
            "portfolio_risk_snapshot": canary_portfolio,
            "account_portfolio_risk_snapshot": account_portfolio,
            "exchange_contract_specs": contract_specs,
            "exchange_risk_facts_provenance": facts.get("policy_provenance"),
            "entry_instrument_availability": facts.get("entry_instrument_availability"),
            "leverage_tier_selection": leverage_tier_selection,
            "runtime_guard": runtime_guard,
            "reasons": reasons,
            "units": {
                "money": "USDT",
                "returns": "percentage_points",
                "fractions": "decimal_ratio",
                "position_size_pct": "available_margin_fraction",
                "notional": "USDT",
            },
            "policy_provenance": provenance,
        }
        raw["profit_risk_sizing"] = sizing
        raw["execution_cost_sizing_pass"] = {
            "order_size_complete": bool(not reasons and final_notional > 0),
            "impact_basis_notional_usdt": final_notional,
            "final_notional_usdt": final_notional,
            "contract_lifecycle": "paper_bootstrap_canary",
        }
        decision.raw_response = raw
        decision.position_size_pct = position_size if not reasons else 0.0
        decision.suggested_leverage = 1.0
        if not reasons:
            decision.stop_loss_pct = stress
            decision.take_profit_pct = max(stress * 1.5, cost_pct * 2.0 / 100.0)

    @staticmethod
    async def _load_history() -> list[Any]:
        async with get_read_session_ctx() as session:
            result = await session.execute(
                select(AIDecision)
                .where(
                    AIDecision.is_paper.is_(True),
                    AIDecision.action.in_(("long", "short")),
                    AIDecision.was_executed.is_(True),
                )
                .order_by(AIDecision.created_at.desc())
                .limit(200)
            )
        return list(result.scalars().all())
