"""Bounded strategy blueprints generated from trained model artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

MODEL_STRATEGY_BLUEPRINT_VERSION = "2026-07-21.trained-model-strategy.v2"


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _canonical_id(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def build_model_strategy_blueprint(
    *,
    metadata: dict[str, Any] | None,
    readiness: dict[str, Any] | None,
    activation: dict[str, Any] | None,
    artifact_version: str | None = None,
) -> dict[str, Any]:
    """Turn one trained artifact into a declarative, paper-only strategy."""

    model = _safe_dict(metadata)
    ready = _safe_dict(readiness)
    active = _safe_dict(activation)
    paper_gate = _safe_dict(ready.get("paper_canary"))
    stage = str(active.get("activation_stage") or "unregistered").lower()
    version = str(
        artifact_version
        or model.get("artifact_version")
        or model.get("version")
        or model.get("trained_at")
        or "unversioned"
    )
    production_sides = {
        str(side).lower()
        for side in _safe_list(active.get("live_enabled_sides"))
        if str(side).lower() in {"long", "short"}
    }
    paper_sides = {
        str(side).lower()
        for side in _safe_list(paper_gate.get("eligible_sides"))
        if str(side).lower() in {"long", "short"}
    }
    live_ml_ready = bool(
        stage in {"canary", "active"}
        and active.get("live_ml_ready") is True
        and production_sides
    )
    paper_canary_authorized = bool(
        stage == "canary"
        and active.get("paper_canary_authorized") is True
        and paper_gate.get("authorized") is True
        and paper_gate.get("execution_scope") == "paper_only"
        and paper_sides
    )
    eligible_sides = sorted(production_sides or paper_sides)
    paper_execution_eligible = bool(
        live_ml_ready or paper_canary_authorized
    )
    champion_comparison = _safe_dict(active.get("champion_comparison"))
    oos_evaluation = _safe_dict(model.get("oos_return_evaluation"))
    model_quality = {
        side: {
            key: _safe_dict(oos_evaluation.get(side)).get(key)
            for key in (
                "avg_return_pct",
                "return_lcb_pct",
                "profit_factor",
                "cvar_10_pct",
                "max_drawdown_pct",
            )
        }
        for side in eligible_sides
    }
    identity = {
        "blueprint_version": MODEL_STRATEGY_BLUEPRINT_VERSION,
        "model_id": "local_ml_profit_quality",
        "model_version": version,
        "training_data_sha256": model.get("training_data_sha256"),
        "eligible_sides": eligible_sides,
    }
    blockers: list[str] = []
    if stage not in {"canary", "active"}:
        blockers.append("trained_model_not_canary_or_active")
    if not eligible_sides:
        blockers.append("trained_model_has_no_governed_side")
    if not paper_execution_eligible:
        blockers.append("trained_model_not_authorized_for_paper_strategy")
    return {
        "version": MODEL_STRATEGY_BLUEPRINT_VERSION,
        "strategy_id": f"trained_model_strategy_{_canonical_id(identity)}",
        "model_id": "local_ml_profit_quality",
        "model_version": version,
        "training_data_sha256": model.get("training_data_sha256"),
        "trained_at": model.get("trained_at"),
        "artifact_stage": stage,
        "generated_at": datetime.now(UTC).isoformat(),
        "source": "trained_model_artifact",
        "execution_scope": "paper_only",
        "eligible_sides": eligible_sides,
        "paper_execution_eligible": paper_execution_eligible,
        "live_execution_permission": False,
        "blocking_reasons": blockers,
        "model_quality": {
            "eligible_sides": model_quality,
            "champion_comparison": champion_comparison,
            "comparison_accepted": champion_comparison.get("accepted") is True,
            "comparison_reason": champion_comparison.get("reason"),
        },
        "entry_policy": {
            "direction_source": "trained_model_return_distribution",
            "require_current_fee_after_expected_return_positive": True,
            "require_current_fee_after_return_lcb_positive": True,
            "require_current_execution_cost_complete": True,
            "require_actual_trade_calibration": True,
            "historical_replay_uses_exact_model_inference": True,
        },
        "exit_policy": {
            "owner": "dynamic_fee_after_exit",
            "model_may_weaken_hard_exit": False,
            "historical_replay_horizon_minutes": 10,
            "historical_replay_exit": "model_primary_prediction_horizon",
        },
        "risk_policy": {
            "owner": "dynamic_risk_and_exchange_contracts",
            "model_may_change_size_or_leverage": False,
            "model_may_bypass_order_deduplication": False,
        },
        "training_evidence": {
            "shadow_sample_count": model.get("training_shadow_sample_count"),
            "fit_sample_count": model.get("train_count"),
            "holdout_sample_count": model.get("test_count"),
            "horizons": _safe_list(model.get("horizons")),
            "partition_policy": model.get("evaluation_group_policy"),
            "training_data_sha256": model.get("training_data_sha256"),
            "strategy_replay_holdout": _safe_dict(
                model.get("strategy_replay_holdout")
            ),
        },
        "historical_replay_policy": {
            "model_fit_rows_can_promote": False,
            "candidate_development_and_exam_must_be_disjoint": True,
            "cost_complete_shadow_required": True,
            "replay_available_without_model_promotion": True,
            "live_execution_permission": False,
        },
    }


def paper_strategy_replay_available(blueprint: dict[str, Any] | None) -> bool:
    """Allow paper evaluation of a complete artifact without granting execution."""

    strategy = _safe_dict(blueprint)
    evidence = _safe_dict(strategy.get("training_evidence"))
    return bool(
        strategy.get("execution_scope") == "paper_only"
        and strategy.get("live_execution_permission") is False
        and str(strategy.get("model_version") or "")
        and _safe_list(strategy.get("eligible_sides"))
        and str(strategy.get("trained_at") or "")
        and int(evidence.get("holdout_sample_count") or 0) > 0
    )


def _normalize_symbol(value: Any) -> str:
    return str(value or "").upper().replace("-", "/").split(":", 1)[0]


def _market_regime(strategy_context: dict[str, Any]) -> str:
    learning = _safe_dict(strategy_context.get("strategy_learning"))
    runtime = _safe_dict(learning.get("runtime"))
    direct = runtime.get("current_market_regime")
    if direct:
        return str(direct).lower()
    regime = _safe_dict(strategy_context.get("market_regime"))
    return str(
        regime.get("mode")
        or regime.get("regime")
        or regime.get("label")
        or ""
    ).lower()


def paper_strategy_authorization(
    strategy_context: dict[str, Any] | None,
    signal: dict[str, Any] | None,
    *,
    symbol: str,
    side: str,
) -> dict[str, Any]:
    """Validate that the active paper champion owns this model signal route."""

    context = _safe_dict(strategy_context)
    learning = _safe_dict(context.get("strategy_learning"))
    runtime = _safe_dict(learning.get("runtime"))
    champion = _safe_dict(
        runtime.get("paper_strategy_champion")
        or learning.get("paper_strategy_champion")
        or context.get("paper_strategy_champion")
    )
    model_signal = _safe_dict(signal)
    result = {
        "eligible": False,
        "reason": "paper_strategy_champion_unavailable",
        "execution_scope": "paper_only",
        "profile_id": champion.get("profile_id"),
    }
    if champion.get("active") is not True:
        return result
    if champion.get("paper_execution_permission") is not True:
        return {**result, "reason": "paper_strategy_execution_permission_missing"}
    if champion.get("live_execution_permission") is not False:
        return {**result, "reason": "paper_strategy_live_boundary_invalid"}
    execution_mode = str(
        context.get("execution_mode")
        or context.get("trading_mode")
        or champion.get("execution_scope")
        or ""
    ).lower()
    if execution_mode not in {"paper", "paper_only"}:
        return {**result, "reason": "paper_execution_mode_required"}

    selector = _safe_dict(champion.get("selector"))
    expected_side = str(selector.get("side") or "").lower()
    if expected_side and expected_side != side:
        return {**result, "reason": "paper_strategy_side_not_selected"}
    expected_symbol = _normalize_symbol(selector.get("symbol"))
    if expected_symbol and expected_symbol != _normalize_symbol(symbol):
        return {**result, "reason": "paper_strategy_symbol_not_selected"}
    expected_regime = str(selector.get("market_regime") or "").lower()
    if expected_regime and expected_regime != _market_regime(context):
        return {**result, "reason": "paper_strategy_regime_not_selected"}

    model_version = str(model_signal.get("model_version") or "")
    champion_version = str(champion.get("model_version") or "")
    if not model_version or model_version != champion_version:
        return {**result, "reason": "paper_strategy_model_version_mismatch"}
    lifecycle = str(model_signal.get("artifact_lifecycle") or "").lower()
    if lifecycle not in {"canary", "active"}:
        return {**result, "reason": "paper_strategy_model_lifecycle_ineligible"}

    live_ml_ready = model_signal.get("live_ml_ready") is True
    paper_gate = _safe_dict(model_signal.get("paper_canary"))
    canary_ready = bool(
        model_signal.get("paper_canary_authorized") is True
        and paper_gate.get("authorized") is True
        and paper_gate.get("execution_scope") == "paper_only"
        and side
        in {
            str(value).lower()
            for value in _safe_list(paper_gate.get("eligible_sides"))
        }
    )
    if not live_ml_ready and not canary_ready:
        return {**result, "reason": "paper_strategy_model_authorization_missing"}
    return {
        **result,
        "eligible": True,
        "reason": "active_trained_model_paper_strategy",
        "model_version": model_version,
        "selector": selector,
    }
