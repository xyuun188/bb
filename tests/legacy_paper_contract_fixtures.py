"""Test-only builders for immutable contracts emitted by retired runtimes."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from math import sqrt
from typing import Any

from services.normal_paper_trade import (
    LEGACY_NORMAL_PAPER_TRADE_V4_VERSION,
    NORMAL_PAPER_TRADE_LEVERAGE_POLICY,
    NORMAL_PAPER_TRADE_MAX_SINGLE_TRADE_RISK_FRACTION,
)
from services.normal_paper_trade import (
    _contract_fingerprint_payload as _normal_v4_fingerprint_payload,
)
from services.normal_paper_trade import _fingerprint as _normal_v4_fingerprint
from services.paper_exploration import (
    PAPER_EXPLORATION_MAX_PORTFOLIO_RISK_FRACTION,
    PAPER_EXPLORATION_MAX_SINGLE_TRADE_RISK_FRACTION,
    PAPER_EXPLORATION_VERSION,
)
from services.paper_exploration import (
    _contract_fingerprint_payload as _exploration_fingerprint_payload,
)
from services.paper_exploration import (
    _fingerprint as _exploration_fingerprint,
)
from services.paper_training import (
    PAPER_TRAINING_MAX_PORTFOLIO_RISK_FRACTION,
    PAPER_TRAINING_MAX_SINGLE_TRADE_RISK_FRACTION,
    PAPER_TRAINING_VERSION,
)
from services.paper_training import (
    _contract_fingerprint_payload as _training_fingerprint_payload,
)
from services.paper_training import (
    _fingerprint as _training_fingerprint,
)

HISTORICAL_NORMAL_PAPER_TRADE_VERSION = "2026-07-22.normal-paper-trade.v1"


def build_legacy_normal_paper_v4_trade_contract(
    *,
    symbol: str,
    side: str,
    expected_net_return_pct: float = 0.2,
    objective_net_return_pct: float = -0.1,
    horizon_minutes: float = 30.0,
) -> dict[str, Any]:
    contract = {
        "version": LEGACY_NORMAL_PAPER_TRADE_V4_VERSION,
        "authorized": True,
        "trade_mode": "paper",
        "execution_scope": "paper_only",
        "entry_type": "normal_strategy_trade",
        "trade_kind": "normal_strategy_trade",
        "production_permission": False,
        "decision_authority": "ensemble",
        "selection_reason": "strategy_edge_selected",
        "symbol": symbol,
        "side": side,
        "prediction_horizon_minutes": horizon_minutes,
        "valid_for_seconds": horizon_minutes * 60.0,
        "expected_net_return_pct": expected_net_return_pct,
        "objective_net_return_pct": objective_net_return_pct,
        "loss_probability": 0.4,
        "quant_evidence_families": ["local_ml"],
        "strong_expert_opposition": False,
        "single_trade_risk_fraction_cap": (
            NORMAL_PAPER_TRADE_MAX_SINGLE_TRADE_RISK_FRACTION
        ),
        "leverage_policy": NORMAL_PAPER_TRADE_LEVERAGE_POLICY,
        "model_leverage_role": "upper_bound_when_explicit",
        "uses_shared_order_pipeline": True,
        "uses_shared_position_ledger": True,
        "continuous_training_after_trusted_settlement": True,
        "training_eligibility_source": (
            "trusted_settlement_and_task_specific_training_contract"
        ),
        "risk_override_permission": False,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    contract["contract_fingerprint"] = _normal_v4_fingerprint(
        _normal_v4_fingerprint_payload(contract)
    )
    return contract


def build_legacy_paper_training_contract(
    *,
    symbol: str,
    selected_side: str,
    signal_source: str,
    expected_net_return_pct: float | None = None,
    return_lcb_pct: float | None = None,
    feature_opportunity_score: float | None = None,
    horizon_minutes: float | None = None,
    policy_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    horizon = float(horizon_minutes or 0.0)
    valid_for_seconds = horizon * 60.0
    generated_at = datetime.now(UTC).isoformat()
    contract = {
        "version": PAPER_TRAINING_VERSION,
        "authorized": True,
        "execution_scope": "paper_only",
        "production_permission": False,
        "trade_kind": "normal_paper_training_trade",
        "trade_is_normal": True,
        "continuous_training_after_settlement": True,
        "loss_tolerant_for_training": True,
        "risk_profile": "cold_start_exploration",
        "single_trade_risk_fraction_cap": PAPER_TRAINING_MAX_SINGLE_TRADE_RISK_FRACTION,
        "portfolio_risk_fraction_cap": PAPER_TRAINING_MAX_PORTFOLIO_RISK_FRACTION,
        "separate_sampling_order": False,
        "purpose": "execute_one_normal_bounded_paper_trade_and_learn_after_settlement",
        "symbol": symbol,
        "selected_side": selected_side,
        "signal_source": signal_source,
        "expected_net_return_pct": expected_net_return_pct,
        "return_lcb_pct": return_lcb_pct,
        "feature_opportunity_score": feature_opportunity_score,
        "prediction_horizon_minutes": horizon,
        "valid_for_seconds": valid_for_seconds,
        "sample_target": None,
        "daily_sample_quota": None,
        "selection_reason": "paper_training_bootstrap_without_profit_gate",
        "policy_provenance": {
            "source": "paper_directional_observation_before_strategy_promotion",
            "observation_window": "current_pre_order_paper_training_round",
            "sample_count": 1,
            "generated_at": generated_at,
            "strategy_version": PAPER_TRAINING_VERSION,
            "valid_for_seconds": valid_for_seconds,
            "prediction_horizon_minutes": horizon,
            "fallback_reason": "",
            "upstream_return_provenance": policy_provenance or {},
        },
    }
    contract["contract_fingerprint"] = _training_fingerprint(
        _training_fingerprint_payload(contract)
    )
    return contract


def build_legacy_normal_paper_trade_contract(
    *,
    symbol: str,
    side: str,
    route_kind: str = "evidence_best",
    horizon_minutes: float = 30.0,
) -> dict[str, Any]:
    generated_at = datetime.now(UTC).isoformat()
    contract = {
        "version": HISTORICAL_NORMAL_PAPER_TRADE_VERSION,
        "authorized": True,
        "execution_scope": "paper_only",
        "live_execution_permission": False,
        "trade_kind": "normal_paper_trade",
        "route_kind": route_kind,
        "symbol": symbol,
        "side": side,
        "prediction_horizon_minutes": horizon_minutes,
        "valid_for_seconds": horizon_minutes * 60.0,
        "uses_shared_order_pipeline": True,
        "uses_shared_position_ledger": True,
        "separate_sampling_order": False,
        "continuous_training_after_trusted_settlement": True,
        "training_eligibility_source": "trusted_settlement_and_training_quarantine",
        "order_creation_owner": "ensemble_trader_unified_decision",
        "risk_override_permission": False,
        "sample_target": None,
        "daily_sample_quota": None,
        "generated_at": generated_at,
    }
    fingerprint_payload = {key: value for key, value in contract.items() if key != "generated_at"}
    encoded = json.dumps(
        fingerprint_payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    contract["contract_fingerprint"] = hashlib.sha256(encoded).hexdigest()
    return contract


def build_legacy_paper_exploration_contract(
    candidate_evidence: dict[str, Any],
    *,
    symbol: str,
    independent_direction_support: dict[str, Any] | None = None,
) -> dict[str, Any]:
    exploration = candidate_evidence.get("paper_exploration") or {}
    selected = exploration.get("selected") or {}
    reliable_count = int(selected.get("reliable_evidence_count") or 0)
    allocation = float(
        selected.get("exploration_allocation_multiplier")
        or max(0.10, 1.0 / sqrt(1.0 + reliable_count / 20.0))
    )
    generated_at = datetime.now(UTC).isoformat()
    contract = {
        "version": PAPER_EXPLORATION_VERSION,
        "authorized": True,
        "execution_scope": "paper_only",
        "production_permission": False,
        "trade_kind": "normal_trade_with_bounded_exploration_risk",
        "trade_is_normal": True,
        "continuous_training_after_settlement": True,
        "purpose": "execute_positive_mean_uncertain_paper_opportunity_and_learn_after_settlement",
        "symbol": symbol,
        "selected_side": exploration.get("preferred_side"),
        "intervention_scope": selected.get("intervention_scope") or "bounded_return_uncertainty",
        "expected_net_return_pct": selected.get("expected_net_return_pct"),
        "objective_net_return_pct": selected.get("objective_net_return_pct"),
        "return_lcb_pct": selected.get("return_lcb_pct"),
        "lcb_gap_ratio": selected.get("lcb_gap_ratio"),
        "loss_probability": selected.get("loss_probability"),
        "tail_risk_score": selected.get("tail_risk_score"),
        "return_source_count": selected.get("return_source_count"),
        "historical_evidence_count": selected.get("historical_evidence_count"),
        "validated_route_evidence_count": selected.get("validated_route_evidence_count"),
        "reliable_evidence_count": reliable_count,
        "exploration_maturity_source": selected.get("exploration_maturity_source"),
        "exploration_maturity_evidence": selected.get("exploration_maturity_evidence"),
        "exploration_allocation_multiplier": allocation,
        "feature_opportunity_score": selected.get("feature_opportunity_score"),
        "information_value_score": selected.get("information_value_score"),
        "independent_direction_support": independent_direction_support or {},
        "prediction_horizon_minutes": selected.get("prediction_horizon_minutes"),
        "valid_for_seconds": selected.get("valid_for_seconds"),
        "single_trade_risk_fraction_cap": PAPER_EXPLORATION_MAX_SINGLE_TRADE_RISK_FRACTION
        * allocation,
        "portfolio_risk_fraction_cap": PAPER_EXPLORATION_MAX_PORTFOLIO_RISK_FRACTION * allocation,
        "leverage_cap": 1,
        "sample_target": None,
        "daily_sample_quota": None,
        "selection_reason": exploration.get("reason"),
        "policy_provenance": {
            "source": "current_cost_complete_return_distribution_and_ranked_market_features",
            "observation_window": "current_pre_order_paper_candidate",
            "sample_count": int(selected.get("return_source_count") or 0),
            "generated_at": generated_at,
            "strategy_version": PAPER_EXPLORATION_VERSION,
            "fallback_reason": "",
            "upstream_return_provenance": selected.get("policy_provenance") or {},
        },
    }
    contract["contract_fingerprint"] = _exploration_fingerprint(
        _exploration_fingerprint_payload(contract)
    )
    return contract
