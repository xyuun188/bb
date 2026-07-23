"""Distribution-driven model promotion for fee-after returns."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any

from config.settings import settings
from services.profit_supervision import (
    AUTHORITATIVE_REALIZED_RETURN_TASK,
    COUNTERFACTUAL_EXECUTION_COST_TASK,
    MARKET_OPPORTUNITY_TASK,
    PRODUCTION_RETURN_COMBINATION_VERSION,
    PROFIT_SUPERVISION_VERSION,
    weighted_distribution,
)
from services.profit_training_contract import PROFIT_TRAINING_TARGET
from services.return_objective import (
    RETURN_LABEL_NAME,
    RETURN_LABEL_VERSION,
    RETURN_OBJECTIVE_NAME,
    RETURN_OBJECTIVE_VERSION,
)

PAPER_OBSERVATION_REPORT_REL_PATH = "phase3_paper_resume_observation_reports/latest.json"
RETURN_PROMOTION_POLICY_VERSION = "2026-07-24.model-owned-return-promotion.v3"
logger = logging.getLogger(__name__)


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if isfinite(result) else default


def load_latest_paper_observation_report(root: Path | None = None) -> dict[str, Any]:
    root_candidate = (root or Path.cwd()) / "data" / PAPER_OBSERVATION_REPORT_REL_PATH
    candidates = (
        [root_candidate, settings.data_dir / PAPER_OBSERVATION_REPORT_REL_PATH]
        if root is not None
        else [settings.data_dir / PAPER_OBSERVATION_REPORT_REL_PATH, root_candidate]
    )
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.debug("paper observation report unavailable at %s: %s", path, exc)
            continue
        if isinstance(payload, dict):
            payload.setdefault("available", True)
            payload.setdefault("report_path", str(path))
            return payload
    return {
        "available": False,
        "status": "missing",
        "can_use_for_promotion": False,
        "candidate_paths": [str(path) for path in candidates],
    }


def _model_action(sample: dict[str, Any]) -> str:
    action = str(sample.get("model_shadow_action") or "").strip().lower()
    if action in {"buy", "open_long"}:
        return "long"
    if action in {"sell", "open_short"}:
        return "short"
    return action


def build_return_objective_report(
    *,
    trade_samples: list[dict[str, Any]] | None = None,
    shadow_samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    shadow_candidates = [
        _safe_dict(sample) for sample in (shadow_samples or []) if isinstance(sample, dict)
    ]
    trade_candidates = [
        _safe_dict(sample) for sample in (trade_samples or []) if isinstance(sample, dict)
    ]

    def task(sample: dict[str, Any], name: str) -> dict[str, Any]:
        supervision = _safe_dict(sample.get("profit_supervision"))
        if supervision.get("version") != PROFIT_SUPERVISION_VERSION:
            return {}
        return _safe_dict(_safe_dict(supervision.get("tasks")).get(name))

    def weighted_pairs(
        samples: list[dict[str, Any]],
        task_name: str,
        field: str,
    ) -> list[tuple[Any, Any]]:
        return [
            (current.get(field), sample.get("sample_weight", 1.0))
            for sample in samples
            if (current := task(sample, task_name)).get("eligible") is True
        ]

    def profit_contract(sample: dict[str, Any]) -> dict[str, Any]:
        realized = task(sample, AUTHORITATIVE_REALIZED_RETURN_TASK)
        return _safe_dict(realized.get("profit_training_contract"))

    market_long = weighted_pairs(
        shadow_candidates,
        MARKET_OPPORTUNITY_TASK,
        "long_gross_market_return_pct",
    )
    market_short = weighted_pairs(
        shadow_candidates,
        MARKET_OPPORTUNITY_TASK,
        "short_gross_market_return_pct",
    )
    shadow_cost_long = weighted_pairs(
        shadow_candidates,
        COUNTERFACTUAL_EXECUTION_COST_TASK,
        "long_total_cost_pct",
    )
    shadow_cost_short = weighted_pairs(
        shadow_candidates,
        COUNTERFACTUAL_EXECUTION_COST_TASK,
        "short_total_cost_pct",
    )
    all_actual_return_pairs = weighted_pairs(
        trade_candidates,
        AUTHORITATIVE_REALIZED_RETURN_TASK,
        PROFIT_TRAINING_TARGET,
    )
    model_live_return_pairs = [
        (realized.get(PROFIT_TRAINING_TARGET), sample.get("sample_weight", 1.0))
        for sample in trade_candidates
        if (realized := task(sample, AUTHORITATIVE_REALIZED_RETURN_TASK)).get("eligible")
        is True
        and profit_contract(sample).get("decision_authority") == "model"
    ]
    model_shadow_return_pairs: list[tuple[Any, Any]] = []
    for sample in shadow_candidates:
        supervision = _safe_dict(sample.get("profit_supervision"))
        labels = _safe_dict(supervision.get("fee_after_return_labels"))
        action = _model_action(sample)
        if labels.get("complete") is not True or action not in {"long", "short"}:
            continue
        model_shadow_return_pairs.append(
            (
                labels.get(f"{action}_net_return_after_all_cost_pct"),
                sample.get("sample_weight", 1.0),
            )
        )
    model_return_pairs = [*model_shadow_return_pairs, *model_live_return_pairs]
    actual_cost_pairs = weighted_pairs(
        trade_candidates,
        COUNTERFACTUAL_EXECUTION_COST_TASK,
        "total_cost_pct",
    )
    actual_slippage_pairs = weighted_pairs(
        trade_candidates,
        COUNTERFACTUAL_EXECUTION_COST_TASK,
        "slippage_pct",
    )
    returns = [
        (float(value), float(weight))
        for value, weight in model_return_pairs
        if _safe_float(value, None) is not None and (_safe_float(weight, 0.0) or 0.0) > 0
    ]
    model_distribution = weighted_distribution(returns)
    profit = sum(max(value, 0.0) * weight for value, weight in returns)
    loss = abs(sum(min(value, 0.0) * weight for value, weight in returns))
    avg_return = _safe_float(model_distribution.get("expected"), None)
    lower_hinge = _safe_float(model_distribution.get("lower_hinge"), None)
    negative_weight = sum(weight for value, weight in returns if value < 0)
    downside_mean = (
        sum(abs(value) * weight for value, weight in returns if value < 0)
        / negative_weight
        if negative_weight > 0
        else 0.0
    )
    profit_factor = profit / loss if loss > 0 else None
    alignment_counts: dict[str, int] = {}
    for sample in trade_candidates:
        contract = profit_contract(sample)
        if contract.get("decision_authority") != "rules":
            continue
        alignment = str(contract.get("model_shadow_alignment") or "").strip()
        if alignment:
            alignment_counts[alignment] = alignment_counts.get(alignment, 0) + 1
    blockers: list[str] = []
    if not market_long or not market_short:
        blockers.append("shadow_market_opportunity_distribution_missing")
    if not shadow_cost_long or not shadow_cost_short:
        blockers.append("counterfactual_execution_cost_distribution_missing")
    if not returns:
        blockers.append("model_attributed_return_distribution_missing")
    if not actual_cost_pairs:
        blockers.append("authoritative_execution_cost_distribution_missing")
    if not actual_slippage_pairs:
        blockers.append("authoritative_slippage_distribution_missing")
    if avg_return is None or avg_return <= 0:
        blockers.append("average_net_return_after_all_cost_not_positive")
    if lower_hinge is None or lower_hinge <= 0:
        blockers.append("empirical_return_lower_hinge_not_positive")
    if profit_factor is None:
        blockers.append("profit_factor_undefined")
    elif profit_factor <= 1.0:
        blockers.append("profit_factor_not_above_break_even")
    generated_at = datetime.now(UTC).isoformat()
    return {
        "available": bool(returns),
        "promotion_ready": not blockers,
        "objective_name": RETURN_OBJECTIVE_NAME,
        "objective_version": RETURN_OBJECTIVE_VERSION,
        "label_name": RETURN_LABEL_NAME,
        "label_version": RETURN_LABEL_VERSION,
        "optimization_target": PROFIT_TRAINING_TARGET,
        "sample_count": len(returns),
        "effective_sample_size": model_distribution.get("effective_sample_size"),
        "model_shadow_return_sample_count": len(model_shadow_return_pairs),
        "model_live_return_sample_count": len(model_live_return_pairs),
        "actual_realized_return_sample_count": len(all_actual_return_pairs),
        "shadow_market_opportunity_sample_count": min(len(market_long), len(market_short)),
        "shadow_counterfactual_cost_sample_count": min(
            len(shadow_cost_long), len(shadow_cost_short)
        ),
        "shadow_samples_are_actual_returns": False,
        "excluded_cost_incomplete_count": len(trade_candidates) - len(actual_cost_pairs),
        "separated_distributions": {
            "market_opportunity": {
                "source_authority": "shadow_native_market_path",
                "long_gross_return_pct": weighted_distribution(market_long),
                "short_gross_return_pct": weighted_distribution(market_short),
            },
            "counterfactual_execution_cost": {
                "source_authority": "shadow_live_microstructure",
                "long_total_cost_pct": weighted_distribution(shadow_cost_long),
                "short_total_cost_pct": weighted_distribution(shadow_cost_short),
            },
            "authoritative_realized_trade": {
                "source_authority": "okx_position_history",
                PROFIT_TRAINING_TARGET: weighted_distribution(all_actual_return_pairs),
                "execution_cost_pct": weighted_distribution(actual_cost_pairs),
                "slippage_pct": weighted_distribution(actual_slippage_pairs),
            },
            "model_attributed_return": {
                "source_authority": (
                    "model_shadow_native_market_path_plus_model_live_okx_position_history"
                ),
                PROFIT_TRAINING_TARGET: model_distribution,
                "shadow": weighted_distribution(model_shadow_return_pairs),
                "live": weighted_distribution(model_live_return_pairs),
            },
        },
        "rules_canary_attribution": {
            "decision_authority": "rules",
            "model_shadow_alignment_counts": alignment_counts,
            "included_in_model_return_distribution": False,
            "purpose": "diagnostic_and_training_alignment_only",
        },
        "production_combination": {
            "version": PRODUCTION_RETURN_COMBINATION_VERSION,
            "formula": (
                "market_opportunity_distribution-live_execution_cost_distribution-"
                "authoritative_slippage_tail_excess"
            ),
            "ready": not any(
                reason.endswith("distribution_missing") for reason in blockers
            ),
        },
        "average_net_return_after_all_cost_pct": (
            round(avg_return, 8) if avg_return is not None else None
        ),
        "median_net_return_after_all_cost_pct": model_distribution.get("median"),
        "empirical_return_lower_hinge_pct": (
            round(float(lower_hinge), 8) if lower_hinge is not None else None
        ),
        "downside_mean_pct": round(downside_mean, 8),
        "profit_factor": round(profit_factor, 8) if profit_factor is not None else None,
        "gross_positive_return_pct": round(profit, 8),
        "gross_negative_return_pct": round(loss, 8),
        "blocking_reasons": blockers,
        "policy_provenance": {
            "source": (
                "model_shadow_fee_after_returns_plus_model_authoritative_okx_returns"
            ),
            "observation_window": "provided_non_overlapping_training_evaluation_samples",
            "sample_count": len(returns),
            "generated_at": generated_at,
            "strategy_version": RETURN_PROMOTION_POLICY_VERSION,
            "fallback_reason": "" if not blockers else ";".join(blockers),
        },
    }


def build_phase3_promotion_recommendation(
    *,
    training_mode: str,
    quality_report: dict[str, Any] | None,
    governance_report: dict[str, Any] | None,
    paper_observation_report: dict[str, Any] | None = None,
    completed_shadow_sample_count: int = 0,
    completed_trade_sample_count: int = 0,
    return_objective_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    quality = _safe_dict(quality_report)
    governance = _safe_dict(governance_report)
    paper = _safe_dict(paper_observation_report)
    return_report = _safe_dict(return_objective_report)
    totals = _safe_dict(quality.get("totals"))
    total = _safe_int(totals.get("total"))
    trainable = _safe_int(governance.get("trainable_sample_count"))
    contamination = str(governance.get("contamination_risk") or "unknown").lower()
    effective_weight = _safe_float(totals.get("effective_weight_ratio"), None)
    blockers: list[str] = []
    if total <= 0 or trainable <= 0:
        blockers.append("no_trainable_samples")
    if contamination != "low":
        blockers.append(
            "high_contamination_risk"
            if contamination == "high"
            else "contamination_risk_unverified"
        )
    if effective_weight is not None and effective_weight <= 0:
        blockers.append("effective_training_weight_zero")
    if not bool(paper.get("can_use_for_promotion")):
        blockers.append("paper_observation_not_healthy")
    for unsafe_key in ("starts_trading_service", "submits_orders", "changes_model_routing"):
        if bool(paper.get(unsafe_key)):
            blockers.append(f"paper_observation_unsafe:{unsafe_key}")

    canary_blockers = list(dict.fromkeys(blockers))
    active_blockers = list(canary_blockers)
    if return_report.get("promotion_ready") is not True:
        active_blockers.extend(
            str(reason)
            for reason in _safe_list(return_report.get("blocking_reasons"))
            if reason
        )
        if not return_report:
            active_blockers.append("return_objective_report_missing")
    if str(training_mode or "").lower() != "walk_forward":
        active_blockers.append("walk_forward_required")
    active_blockers = list(dict.fromkeys(active_blockers))
    recommended_stage = (
        "active" if not active_blockers else "canary" if not canary_blockers else "shadow"
    )
    if contamination != "low":
        recommended_stage = "degraded"

    return {
        "policy": RETURN_PROMOTION_POLICY_VERSION,
        "optimization_target": PROFIT_TRAINING_TARGET,
        "training_mode": str(training_mode or "shadow").lower(),
        "recommended_stage": recommended_stage,
        "canary_ready": not canary_blockers,
        "canary_execution_scope": "paper_only",
        "canary_production_permission": False,
        "live_ml_ready": not active_blockers,
        "canary_blocking_reasons": canary_blockers,
        "active_blocking_reasons": active_blockers,
        "live_blocking_reasons": active_blockers,
        "observed_sample_counts": {
            "completed_shadow_sample_count": int(completed_shadow_sample_count or 0),
            "completed_trade_sample_count": int(completed_trade_sample_count or 0),
            "counts_are_diagnostic_only": True,
        },
        "quality_gate": {
            "total_samples": total,
            "trainable_sample_count": trainable,
            "effective_weight_ratio": effective_weight,
            "contamination_risk": contamination,
        },
        "return_objective_gate": return_report,
        "paper_observation_gate": {
            "required": True,
            "status": paper.get("status") or "missing",
            "can_use_for_promotion": bool(paper.get("can_use_for_promotion")),
        },
    }
