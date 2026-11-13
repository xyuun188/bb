"""Go/no-go evaluation for the dynamic fee-after return architecture."""

from __future__ import annotations

import math
from typing import Any

from services.ml_prediction_contract import ENTRY_COHORT_VERSION, PREDICTION_SCORE_POLICY_VERSION
from services.normal_paper_trade import (
    NORMAL_PAPER_TRADE_TRAINING_PORTFOLIO_RISK_FRACTION,
    NORMAL_PAPER_TRADE_TRAINING_SINGLE_TRADE_RISK_FRACTION,
)
from services.profit_training_contract import PROFIT_TRAINING_TARGET


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _first_nested_metric(value: Any, names: tuple[str, ...], *, depth: int = 0) -> float | None:
    """Find one named scalar in bounded audit payloads without coupling to UI shape."""

    if depth > 6:
        return None
    if isinstance(value, dict):
        for name in names:
            if name in value:
                number = _safe_float(value.get(name))
                if number is not None:
                    return number
        for nested in value.values():
            number = _first_nested_metric(nested, names, depth=depth + 1)
            if number is not None:
                return number
    elif isinstance(value, list):
        for nested in value[:40]:
            number = _first_nested_metric(nested, names, depth=depth + 1)
            if number is not None:
                return number
    return None


def _model_profitability_evidence(
    training: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Select the current local ML artifact before any legacy tool bundle.

    ``local_ai_tools`` is an older, separately promoted model bundle. If the
    audit supplies ``local_ml_profit_quality``, its explicit metrics and
    side-level evidence are authoritative even when the legacy bundle still
    reports a different profit factor.
    """

    local_ml = _safe_dict(training.get("local_ml_profit_quality"))
    if not local_ml:
        for row in _safe_list(_safe_dict(training.get("model_registry")).get("models")):
            if isinstance(row, dict) and row.get("model_id") == "local_ml_profit_quality":
                local_ml = row
                break
    if local_ml and local_ml.get("available") is not False:
        sides = _safe_dict(
            local_ml.get("sides")
            or local_ml.get("side_metrics")
        )
        current_entry_cohort = (
            local_ml.get("entry_cohort_version") == ENTRY_COHORT_VERSION
            and local_ml.get("prediction_score_policy_version") == PREDICTION_SCORE_POLICY_VERSION
        )
        portfolio = _safe_dict(local_ml.get("portfolio")) if current_entry_cohort else {}
        if _safe_int(portfolio.get("count")) <= 0:
            portfolio = {}
        side_evidence: dict[str, dict[str, Any]] = {}
        for side in ("long", "short"):
            metrics = _safe_dict(sides.get(side)) if current_entry_cohort else {}
            profit_factor = _safe_float(metrics.get("profit_factor"))
            return_lcb = _safe_float(metrics.get("return_lcb_pct"))
            mean_return = _safe_float(metrics.get("mean_return_pct"))
            side_evidence[side] = {
                "profit_factor": profit_factor,
                "return_lcb_pct": return_lcb,
                "mean_return_pct": mean_return,
                "tail_loss_rate": _safe_float(metrics.get("tail_loss_rate")),
                "count": _safe_int(metrics.get("count")),
            }
        return (
            {
                "source": "local_ml_profit_quality",
                "artifact_version": local_ml.get("artifact_version"),
                "walk_forward_status": local_ml.get("walk_forward_status"),
                "walk_forward_stable": local_ml.get("walk_forward_stable"),
                "fallback_used": local_ml.get("fallback_used"),
                "lifecycle": local_ml.get("lifecycle"),
                "sides": side_evidence,
                "evidence_scope": local_ml.get("evidence_scope"),
                "prediction_score_policy_version": local_ml.get("prediction_score_policy_version"),
                "count": _safe_int(portfolio.get("count")),
                "profit_factor": _safe_float(portfolio.get("profit_factor")),
                "return_lcb_pct": _safe_float(portfolio.get("return_lcb_pct")),
                "mean_return_pct": _safe_float(portfolio.get("mean_return_pct")),
            },
            "local_ml_profit_quality",
        )
    legacy = _safe_dict(training.get("local_ai_tools"))
    return (legacy or training), "local_ai_tools" if legacy else "model_training"


def _blocker(code: str, message: str, evidence: Any | None = None) -> dict[str, Any]:
    return {"code": code, "message": message, "evidence": evidence}


def _warning(code: str, message: str, evidence: Any | None = None) -> dict[str, Any]:
    return {"code": code, "message": message, "evidence": evidence}


def _cards_by_key(cards: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(card.get("key") or ""): card
        for card in cards
        if isinstance(card, dict) and card.get("key")
    }


def _details(card: dict[str, Any]) -> dict[str, Any]:
    return _safe_dict(card.get("details"))


def _deferred_section_keys(cards: list[dict[str, Any]]) -> set[str]:
    """Return sections represented by a grouped timeout/deferred card."""

    keys: set[str] = set()
    for card in cards:
        if not str(card.get("key") or "").startswith("system_audit_deferred:"):
            continue
        details = _details(card)
        for key in _safe_list(details.get("affected_section_keys")):
            if key:
                keys.add(str(key))
        primary = details.get("primary_section_key")
        if primary:
            keys.add(str(primary))
    return keys


def evaluate_phase3_go_no_go_cards(cards: list[dict[str, Any]]) -> dict[str, Any]:
    """Require current return/cost provenance and real execution facts."""

    by_key = _cards_by_key(cards)
    deferred_keys = _deferred_section_keys(cards)
    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    passed: list[str] = []
    deferred_required: list[str] = []

    required_cards = {
        "okx_trade_fact_integrity": "authoritative OKX trade facts",
        "trade_execution_contract": "dynamic return execution contract",
        "position_capacity_release": "hard capacity and dynamic exit audit",
        "model_training": "fee-after return model training",
        "phase3_model_server_readiness": "model server readiness",
    }
    for key, label in required_cards.items():
        card = by_key.get(key)
        if not card:
            if key in deferred_keys:
                deferred_required.append(key)
                continue
            blockers.append(_blocker(f"{key}_missing", f"Missing {label} audit card."))
            continue
        status = str(card.get("status") or "unknown").lower()
        if status == "critical":
            blockers.append(
                _blocker(
                    f"{key}_critical",
                    f"{label} is critical.",
                    evidence=card.get("summary") or _details(card),
                )
            )
        elif status == "warning":
            warnings.append(
                _warning(
                    f"{key}_warning",
                    f"{label} reports warnings.",
                    evidence=card.get("summary") or _details(card),
                )
            )
        else:
            passed.append(f"{key}_available")

    if deferred_required:
        blockers.append(
            _blocker(
                "required_audits_deferred",
                "Required audit sections did not complete within the collection budget; retry before changing readiness.",
                {"sections": sorted(deferred_required)},
            )
        )

    trade_card = by_key.get("trade_execution_contract", {})
    trade = _details(trade_card)
    trade_summary = _safe_dict(trade.get("current_summary")) or _safe_dict(
        trade.get("summary")
    )
    trade_policy = _safe_dict(trade.get("policy"))
    required_true_trade_policy = (
        "paper_entry_requires_positive_expected_net_return",
        "paper_normal_entry_requires_positive_return_lcb",
        "paper_validated_entry_requires_positive_return_lcb",
        "paper_downside_return_is_sizing_stress",
        "paper_quality_observation_requires_positive_expected_net_return",
        "paper_quality_observation_allows_non_positive_return_lcb",
        "paper_quality_observation_is_shadow_only",
        "paper_training_entry_is_bounded_paper_only",
        "paper_training_entry_authorizes_normal_entry",
        "paper_training_entry_requires_positive_size_aware_net_return",
        "paper_training_entry_requires_positive_return_lcb",
        "paper_training_entry_requires_finite_downside_stress",
        "paper_entry_requires_historical_fee_after_quality_permission",
        "paper_entry_requires_profit_factor",
        "paper_entry_requires_current_execution_cost",
        "live_entry_requires_production_trade_gate",
        "live_entry_requires_positive_fee_after_return",
        "live_entry_requires_positive_return_lcb",
        "live_entry_requires_current_execution_cost",
        "entry_requires_dynamic_risk_budget",
        "entry_requires_complete_provenance",
        "exit_requires_position_economics",
        "exit_requires_dynamic_close_fraction",
        "filled_order_link_required",
    )
    required_false_trade_policy = (
        "paper_entry_requires_model_promotion",
        "paper_quality_observation_authorizes_normal_entry",
        "paper_training_entry_is_shadow_only",
        "paper_training_entry_allows_non_positive_return_lcb",
        "paper_training_entry_allows_non_positive_fee_after_return",
        "paper_training_entry_production_permission",
    )
    missing_policy = [
        key for key in required_true_trade_policy if trade_policy.get(key) is not True
    ]
    missing_policy.extend(
        key for key in required_false_trade_policy if trade_policy.get(key) is not False
    )
    required_numeric_trade_policy = {
        "paper_training_entry_single_trade_risk_fraction": (
            NORMAL_PAPER_TRADE_TRAINING_SINGLE_TRADE_RISK_FRACTION
        ),
        "paper_training_entry_portfolio_risk_fraction": (
            NORMAL_PAPER_TRADE_TRAINING_PORTFOLIO_RISK_FRACTION
        ),
    }
    invalid_numeric_policy = [
        key
        for key, expected in required_numeric_trade_policy.items()
        if (
            _safe_float(trade_policy.get(key)) is None
            or not math.isclose(
                float(_safe_float(trade_policy.get(key))),
                expected,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )
    ]
    report_available = trade.get("report_available", True)
    if report_available is False:
        blockers.append(
            _blocker(
                "dynamic_return_contract_unavailable",
                "Dynamic return execution contract report is unavailable.",
                evidence={
                    "error": trade.get("error"),
                    "error_type": trade.get("error_type"),
                    "timeout": bool(trade.get("timeout")),
                },
            )
        )
    # An unavailable audit cannot prove or disprove individual policy fields.
    # Keep the single availability blocker instead of misreporting a timeout as
    # a malformed contract payload.
    if (
        "trade_execution_contract" not in deferred_required
        and report_available is not False
        and (missing_policy or invalid_numeric_policy)
    ):
        blockers.append(
            _blocker(
                "dynamic_return_contract_policy_incomplete",
                "Execution audit does not prove all return/cost/provenance invariants.",
                evidence={
                    "missing": missing_policy,
                    "invalid_numeric": invalid_numeric_policy,
                },
            )
        )
    violation_count = _safe_int(trade_summary.get("contract_violation_count"))
    if "trade_execution_contract" not in deferred_required and report_available is not False and violation_count > 0:
        blockers.append(
            _blocker(
                "dynamic_return_contract_current_violations",
                "Current runtime window contains executed contract violations.",
                evidence=trade_summary,
            )
        )
    elif "trade_execution_contract" not in deferred_required and report_available is not False and trade_summary:
        passed.append("dynamic_return_contract_current_window_clean")

    current_realized_net = _safe_float(trade_summary.get("realized_net_pnl_usdt"))
    authoritative_settlements = bool(
        trade_summary.get("realized_pnl_source") == "okx_position_history_realized_pnl"
        and _safe_int(trade_summary.get("closed_position_count")) > 0
    )
    if (
        "trade_execution_contract" not in deferred_required
        and report_available is not False
        and authoritative_settlements
        and current_realized_net is not None
        and current_realized_net <= 0.0
    ):
        blockers.append(
            _blocker(
                "current_authoritative_return_not_positive",
                "Current settled trades have not earned a positive net return after costs.",
                evidence={
                    "source": trade_summary["realized_pnl_source"],
                    "closed_position_count": trade_summary["closed_position_count"],
                    "realized_net_pnl_usdt": current_realized_net,
                    "realized_net_pnl_by_side_usdt": trade_summary.get(
                        "realized_net_pnl_by_side_usdt"
                    ),
                    "scope": "current_system_execution_window_not_model_attribution",
                },
            )
        )

    capacity = _details(by_key.get("position_capacity_release", {}))
    capacity_policy = _safe_dict(capacity.get("policy"))
    if (
        "position_capacity_release" not in deferred_required
        and capacity_policy.get("strategy_learning_cannot_expand_capacity") is not True
    ):
        blockers.append(
            _blocker(
                "hard_capacity_boundary_unproven",
                "Capacity audit does not prove that strategy learning cannot expand hard capacity.",
            )
        )
    economics_gaps = _safe_int(capacity.get("position_economics_incomplete_count"))
    exit_gaps = _safe_int(capacity.get("executed_dynamic_exit_contract_gap_count"))
    if "position_capacity_release" not in deferred_required and (economics_gaps or exit_gaps):
        blockers.append(
            _blocker(
                "position_economics_or_exit_contract_gap",
                "Open-position economics or executed dynamic exit contracts are incomplete.",
                evidence={
                    "position_economics_incomplete_count": economics_gaps,
                    "executed_dynamic_exit_contract_gap_count": exit_gaps,
                },
            )
        )

    training = _details(by_key.get("model_training", {}))
    objective = str(
        training.get("optimization_target")
        or _safe_dict(training.get("policy")).get("optimization_target")
        or ""
    )
    if objective and objective != PROFIT_TRAINING_TARGET:
        blockers.append(
            _blocker(
                "model_training_objective_mismatch",
                "Model training is not governed by authoritative all-cost net return.",
                evidence={"optimization_target": objective},
            )
        )

    paper_resume_card = by_key.get("phase3_paper_resume_preflight")
    if paper_resume_card is not None:
        paper_resume = _details(paper_resume_card)
        paper_resume_status = str(paper_resume_card.get("status") or "unknown").lower()
        paper_resume_preflight_satisfied = (
            paper_resume.get("can_resume_paper") is True
            or (
                paper_resume.get("consumed_after_resume") is True
                and paper_resume.get("effective_blockers") == []
            )
        )
        if (
            "phase3_paper_resume_preflight" not in deferred_required
            and paper_resume_status != "critical"
            and not paper_resume_preflight_satisfied
        ):
            blockers.append(
                _blocker(
                    "paper_resume_preflight_not_ready",
                    "Paper trading resume preflight has not granted a controlled resume.",
                    evidence={
                        "status": paper_resume.get("status"),
                        "blockers": _safe_list(paper_resume.get("blockers")),
                        "consumed_after_resume": paper_resume.get(
                            "consumed_after_resume"
                        ),
                        "effective_blockers": _safe_list(
                            paper_resume.get("effective_blockers")
                        ),
                    },
                )
            )

    local_ai_tools = _safe_dict(training.get("local_ai_tools"))
    local_ml = _safe_dict(training.get("local_ml_profit_quality"))
    live_ml_ready = (
        local_ml.get("live_ml_ready")
        if local_ml
        else training.get("live_ml_ready")
    )
    if live_ml_ready is None and not local_ml:
        live_ml_ready = local_ai_tools.get("live_ml_ready")
    # Older/read-only audit cards may omit this optional promotion field. Only
    # an explicit false is a production-readiness blocker; absence is handled
    # by the dedicated model-training card and profitability evidence.
    if "model_training" not in deferred_required and live_ml_ready is False:
        blockers.append(
            _blocker(
                "local_ml_not_live_ready",
                "Local ML has not passed the production profitability gate.",
                evidence={"live_ml_ready": live_ml_ready},
            )
        )

    model_critical_items = _safe_list(training.get("model_critical_items"))
    if model_critical_items:
        blockers.append(
            _blocker(
                "model_training_resource_failure",
                "One or more required local ML models failed training or resource validation.",
                evidence={"items": model_critical_items[:8]},
            )
        )

    profitability_evidence, profitability_source = _model_profitability_evidence(
        training
    )
    profit_factor = _safe_float(profitability_evidence.get("profit_factor"))
    if profit_factor is None and profitability_source != "local_ml_profit_quality":
        profit_factor = _first_nested_metric(
            profitability_evidence,
            ("profit_factor", "top_profit_factor", "top_long_profit_factor"),
        )
    return_lcb = _safe_float(profitability_evidence.get("return_lcb_pct"))
    if return_lcb is None and profitability_source != "local_ml_profit_quality":
        return_lcb = _first_nested_metric(
            profitability_evidence,
            ("return_lcb_pct", "return_lcb", "return_lower_bound", "fee_after_return_lcb_pct"),
        )
    mean_fee_after_return = _safe_float(
        profitability_evidence.get("mean_return_pct")
    )
    if mean_fee_after_return is None and profitability_source != "local_ml_profit_quality":
        mean_fee_after_return = _first_nested_metric(
            profitability_evidence,
            (
                "mean_fee_after_return_pct",
                "avg_fee_after_return_pct",
                "average_fee_after_return_pct",
                "fee_after_mean_return_pct",
            ),
        )
    if profitability_source == "local_ml_profit_quality":
        if profit_factor is None:
            blockers.append(
                _blocker(
                    "model_profit_factor_evidence_missing",
                    "Local ML profitability evidence is missing its profit factor.",
                    evidence={"source": profitability_source},
                )
            )
        if return_lcb is None:
            blockers.append(
                _blocker(
                    "model_return_lcb_evidence_missing",
                    "Local ML profitability evidence is missing its return LCB.",
                    evidence={"source": profitability_source},
                )
            )
        if mean_fee_after_return is None:
            blockers.append(
                _blocker(
                    "model_fee_after_return_evidence_missing",
                    "Local ML profitability evidence is missing its mean fee-after return.",
                    evidence={"source": profitability_source},
                )
            )
    if profit_factor is not None and profit_factor <= 1.0:
        blockers.append(
            _blocker(
                "model_profit_factor_below_unity",
                "The evaluated model loses more than it earns after costs.",
                evidence={
                    "profit_factor": profit_factor,
                    "required_minimum": 1.0,
                    "source": profitability_source,
                },
            )
        )
    if return_lcb is not None and return_lcb <= 0.0:
        blockers.append(
            _blocker(
                "model_return_lcb_not_positive",
                "The evaluated model return lower confidence bound is not positive.",
                evidence={
                    "return_lcb_pct": return_lcb,
                    "required_minimum_exclusive": 0.0,
                    "source": profitability_source,
                },
            )
        )
    if mean_fee_after_return is not None and mean_fee_after_return <= 0.0:
        blockers.append(
            _blocker(
                "model_fee_after_return_not_positive",
                "The evaluated model mean return is not positive after all costs.",
                evidence={
                    "mean_fee_after_return_pct": mean_fee_after_return,
                    "required_minimum_exclusive": 0.0,
                    "source": profitability_source,
                },
            )
        )

    blocker_codes = {item["code"] for item in blockers}
    warning_codes = {item["code"] for item in warnings}
    return {
        "ready": not blockers,
        "status": "go" if not blockers else "no_go",
        "blockers": blockers,
        "warnings": warnings,
        "passed_checks": passed,
        "blocker_codes": sorted(blocker_codes),
        "warning_codes": sorted(warning_codes),
        "summary": {
            "blocker_count": len(blockers),
            "warning_count": len(warnings),
            "current_contract_violation_count": violation_count,
            "position_economics_incomplete_count": economics_gaps,
            "executed_dynamic_exit_contract_gap_count": exit_gaps,
            "live_ml_ready": live_ml_ready is True,
            "model_profitability_source": profitability_source,
            "model_profitability_scope": profitability_evidence.get("evidence_scope"),
            "model_profitability_sample_count": profitability_evidence.get("count"),
            "model_profitability_sides": profitability_evidence.get("sides"),
            "model_fallback_used": profitability_evidence.get("fallback_used"),
            "model_profit_factor": profit_factor,
            "model_return_lcb_pct": return_lcb,
            "model_mean_fee_after_return_pct": mean_fee_after_return,
        },
        "policy": {
            "optimization_target": PROFIT_TRAINING_TARGET,
            "current_authoritative_loss_blocks_go": True,
            "positive_current_pnl_alone_grants_go": False,
            "win_rate_is_diagnostic_only": True,
            "expert_memory_strategy_learning_are_observation_only": True,
            "fixed_strategy_thresholds_forbidden": True,
        },
    }
