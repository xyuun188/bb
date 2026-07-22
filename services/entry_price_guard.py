"""Dynamic pre-execution entry price validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from math import isfinite
from typing import Any

from ai_brain.base_model import Action, DecisionOutput
from core.symbols import okx_inst_id_from_symbol
from services.paper_bootstrap_canary import (
    PAPER_BOOTSTRAP_CANARY_VERSION,
    PaperBootstrapCanaryPolicy,
)
from services.paper_training import is_paper_training_decision


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return number if isfinite(number) else default


def _feature_snapshot(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is not None and hasattr(value, "to_dict"):
        snapshot = value.to_dict()
        return snapshot if isinstance(snapshot, dict) else {}
    return {}


@dataclass(slots=True)
class EntryPriceGuardPolicy:
    """Require current adverse drift to fit inside authoritative return budget."""

    fresh_feature_provider: Callable[[str], Awaitable[Any]]
    market_data_quality_reason_provider: Callable[..., str | None]
    decision_age_seconds_provider: Callable[[DecisionOutput], float]
    pre_order_execution_facts_provider: (
        Callable[[str, DecisionOutput], Awaitable[dict[str, Any]]] | None
    ) = None

    async def guard_reason(
        self,
        decision: DecisionOutput,
        model_mode: str = "",
    ) -> str | None:
        if not decision.is_entry:
            return None

        snapshot = _safe_dict(decision.feature_snapshot)
        quality_reason = self.market_data_quality_reason_provider(
            snapshot,
            stage_label="pre-order analysis snapshot",
        )
        if quality_reason:
            return f"Pre-order analysis market fact is invalid; entry fails closed: {quality_reason}"

        execution_facts: dict[str, Any] = {}
        if self.pre_order_execution_facts_provider is not None:
            try:
                execution_facts = await self.pre_order_execution_facts_provider(
                    model_mode,
                    decision,
                )
            except Exception:
                return "Authoritative pre-order execution facts are unavailable; entry fails closed."
            if execution_facts.get("production_eligible") is not True:
                return (
                    "Authoritative pre-order execution facts are incomplete; entry fails closed: "
                    f"{execution_facts.get('reason') or 'unknown'}"
                )
            execution_snapshot = _safe_dict(execution_facts.get("feature_snapshot"))
            if not execution_snapshot:
                return "Authoritative pre-order execution snapshot is missing; entry fails closed."
            execution_inst_id = str(execution_facts.get("inst_id") or "").upper()
            expected_inst_id = okx_inst_id_from_symbol(decision.symbol).upper()
            if (
                not execution_inst_id
                or not expected_inst_id
                or execution_inst_id != expected_inst_id
            ):
                return "Pre-order market fact and execution fact instrument mismatch; entry fails closed."
            fresh = execution_snapshot
        else:
            fresh = await self._fresh_valid_snapshot(decision.symbol)
            if not fresh:
                return "Fresh pre-order native market fact is incomplete; entry fails closed."

        snapshot_price = _safe_float(snapshot.get("current_price") or snapshot.get("close"))
        if snapshot_price <= 0:
            return "Pre-order analysis price is missing; entry fails closed."
        latest_price = _safe_float(fresh.get("current_price") or fresh.get("close"))
        if latest_price <= 0:
            return "Fresh pre-order native price is unavailable; entry fails closed."

        paper_canary = bool(
            str(model_mode or "").lower() == "paper"
            and PaperBootstrapCanaryPolicy.is_claimed(decision)
        )
        paper_training = bool(
            str(model_mode or "").lower() == "paper"
            and is_paper_training_decision(decision)
        )
        return_budget = None if paper_training else (
            self._paper_canary_price_budget_fraction(decision)
            if paper_canary
            else self._return_budget_fraction(decision)
        )
        if not paper_training and (return_budget is None or return_budget <= 0):
            return (
                "Authoritative paper bootstrap distribution drift budget is missing; "
                "entry fails closed."
                if paper_canary
                else "Authoritative fee-after return budget is missing; entry fails closed."
            )

        move = (latest_price - snapshot_price) / snapshot_price
        adverse = self._adverse_move(decision.action, move)
        allowed = None if paper_training else return_budget
        raw = _safe_dict(decision.raw_response)
        analysis_fact = _safe_dict(snapshot.get("market_fact"))
        fresh_fact = _safe_dict(fresh.get("market_fact"))
        raw["pre_execution_price_check"] = {
            "snapshot_price": snapshot_price,
            "latest_price": latest_price,
            "adverse_move_fraction": round(adverse, 8),
            "return_budget_fraction": (
                round(return_budget, 8) if return_budget is not None else None
            ),
            "allowed_adverse_move_fraction": (
                round(allowed, 8) if allowed is not None else None
            ),
            "decision_age_seconds": round(self.decision_age_seconds_provider(decision), 3),
            "contract_lifecycle": (
                "paper_training"
                if paper_training
                else "paper_bootstrap_canary"
                if paper_canary
                else "production_return"
            ),
            "production_permission": False if paper_canary or paper_training else True,
            "profitability_gate_applied": not paper_training,
            "safety_scope": (
                "market_integrity_only"
                if paper_training
                else "market_integrity_and_return_budget"
            ),
            "policy_provenance": {
                "source": (
                    "paper_training_market_integrity_only"
                    if paper_training
                    else
                    "paper_bootstrap_empirical_distribution_uncertainty"
                    if paper_canary
                    else "authoritative_fee_after_return_lcb"
                ),
                "observation_window": "current_pre_order_refresh",
                "sample_count": (
                    self._paper_training_sample_count(decision)
                    if paper_training
                    else self._paper_canary_sample_count(decision)
                    if paper_canary
                    else self._return_sample_count(decision)
                ),
                "generated_at": raw.get("generated_at") or "decision_runtime",
                "strategy_version": "2026-07-12.dynamic-price-budget.v1",
                "fallback_reason": "",
            },
            "native_market_fact_proof": {
                "analysis_fact_id": analysis_fact.get("fact_id"),
                "fresh_fact_id": fresh_fact.get("fact_id"),
                "analysis_inst_id": _safe_dict(
                    analysis_fact.get("native_identity")
                ).get("inst_id"),
                "fresh_inst_id": _safe_dict(fresh_fact.get("native_identity")).get(
                    "inst_id"
                )
                or execution_facts.get("inst_id"),
                "fresh_source_timestamp_ms": fresh_fact.get("source_timestamp_ms")
                or execution_facts.get("ticker_source_timestamp_ms"),
                "fresh_source_interface": fresh_fact.get("source_interface")
                or _safe_dict(execution_facts.get("policy_provenance")).get("source"),
            },
        }
        if paper_training or (allowed is not None and adverse <= allowed):
            public_execution_facts = {
                key: value
                for key, value in execution_facts.items()
                if key not in {"feature_snapshot", "fee_snapshot"}
            }
            if execution_facts:
                fingerprint_payload = {
                    "facts": public_execution_facts,
                    "feature_snapshot": _safe_dict(execution_facts.get("feature_snapshot")),
                }
                public_execution_facts["input_fingerprint"] = hashlib.sha256(
                    json.dumps(
                        fingerprint_payload,
                        ensure_ascii=True,
                        sort_keys=True,
                        default=str,
                    ).encode("utf-8")
                ).hexdigest()
                raw["pre_order_execution_facts"] = public_execution_facts
                fresh["pre_order_execution_facts"] = public_execution_facts
            decision.feature_snapshot = {**snapshot, **fresh}
            decision.raw_response = raw
            return None

        decision.raw_response = raw

        return (
            "Current adverse price movement exceeds the authoritative fee-after return "
            "budget."
        )

    async def _fresh_valid_snapshot(self, symbol: str) -> dict[str, Any]:
        snapshot = _feature_snapshot(await self.fresh_feature_provider(symbol))
        if not snapshot:
            return {}
        reason = self.market_data_quality_reason_provider(
            snapshot,
            stage_label="pre-order refreshed market snapshot",
        )
        return {} if reason else snapshot

    @staticmethod
    def _adverse_move(action: Action, move: float) -> float:
        if action == Action.LONG:
            return max(move, 0.0)
        if action == Action.SHORT:
            return max(-move, 0.0)
        return 0.0

    @staticmethod
    def _side(decision: DecisionOutput) -> str:
        return "long" if decision.action == Action.LONG else "short"

    def _side_evidence(self, decision: DecisionOutput) -> dict[str, Any]:
        raw = _safe_dict(decision.raw_response)
        evidence = _safe_dict(raw.get("entry_candidate_evidence"))
        side_evidence = _safe_dict(evidence.get(self._side(decision)))
        if side_evidence:
            return side_evidence
        return _safe_dict(_safe_dict(raw.get("authoritative_return_candidate")).get("side_evidence"))

    def _return_budget_fraction(self, decision: DecisionOutput) -> float:
        evidence = self._side_evidence(decision)
        if evidence.get("production_eligible") is not True:
            return 0.0
        expected_net = _safe_float(evidence.get("expected_net_return_pct"))
        return_lcb = _safe_float(evidence.get("return_lcb_pct"))
        if expected_net <= 0 or return_lcb <= 0:
            return 0.0
        return min(expected_net, return_lcb) / 100.0

    @staticmethod
    def _paper_canary_contract(decision: DecisionOutput) -> dict[str, Any]:
        raw = _safe_dict(decision.raw_response)
        contract = _safe_dict(raw.get("paper_bootstrap_canary"))
        if (
            contract.get("version") != PAPER_BOOTSTRAP_CANARY_VERSION
            or contract.get("authorized") is not True
            or contract.get("requested") is not True
            or contract.get("execution_scope") != "paper_only"
            or contract.get("production_permission") is not False
        ):
            return {}
        return contract

    def _paper_canary_price_budget_fraction(self, decision: DecisionOutput) -> float:
        observation = _safe_dict(
            self._paper_canary_contract(decision).get("selected_observation")
        )
        dispersion_pct = max(_safe_float(observation.get("dispersion_pct")), 0.0)
        objective_pct = _safe_float(observation.get("objective_expected_return_pct"))
        lower_quantile_pct = _safe_float(observation.get("lower_quantile_return_pct"))
        quantile_gap_pct = abs(objective_pct - lower_quantile_pct)
        if dispersion_pct <= 0 or quantile_gap_pct <= 0:
            return 0.0
        return min(dispersion_pct, quantile_gap_pct) / 100.0

    def _paper_canary_sample_count(self, decision: DecisionOutput) -> int:
        contract = self._paper_canary_contract(decision)
        return max(int(_safe_float(contract.get("source_sample_count"))), 0)

    def _return_sample_count(self, decision: DecisionOutput) -> int:
        return max(int(_safe_float(self._side_evidence(decision).get("production_source_count"))), 0)

    @staticmethod
    def _paper_training_sample_count(decision: DecisionOutput) -> int:
        raw = _safe_dict(decision.raw_response)
        contract = _safe_dict(raw.get("paper_training"))
        provenance = _safe_dict(contract.get("policy_provenance"))
        return max(int(_safe_float(provenance.get("sample_count"))), 0)
