"""Canonical model identity and lifecycle reporting.

This module deliberately separates training from inference and shadow evaluation.
A reachable endpoint is never sufficient evidence that a model was trained.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from services.profit_training_contract import PROFIT_TRAINING_TARGET

MODEL_TRAINING_REGISTRY_VERSION = "2026-07-12.v2"


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_int(value: Any) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _fee_after_evaluation(
    bucket: dict[str, Any] | None,
    *,
    missing_reason: str = "fee_after_return_evaluation_missing",
) -> dict[str, Any]:
    evidence = _safe_dict(bucket)
    count = _safe_int(evidence.get("count"))
    if count <= 0:
        return {
            "evaluation_mode": "not_evaluated",
            "evaluation_objective": "fee_after_realized_net_pnl_and_profit_factor",
            "evaluation_sample_count": 0,
            "quality_state": "promotion_blocked",
            "blocking_reasons": [missing_reason],
            "realized_net_pnl_usdt": 0.0,
            "avg_realized_net_pnl_usdt": 0.0,
            "profit_factor": None,
        }
    pnl = float(evidence.get("pnl") or 0.0)
    profit_factor_raw = evidence.get("profit_factor")
    try:
        profit_factor = (
            float(profit_factor_raw) if profit_factor_raw is not None else None
        )
    except (TypeError, ValueError):
        profit_factor = None
    blockers = []
    if pnl <= 0:
        blockers.append("realized_net_pnl_non_positive")
    if profit_factor is None:
        blockers.append("profit_factor_undefined")
    elif profit_factor < 1.0:
        blockers.append("profit_factor_below_unity")
    return {
        "evaluation_mode": "fee_after_shadow_evaluated",
        "evaluation_objective": "fee_after_realized_net_pnl_and_profit_factor",
        "evaluation_sample_count": count,
        "quality_state": "promotion_blocked" if blockers else "fee_after_positive",
        "blocking_reasons": blockers,
        "realized_net_pnl_usdt": round(pnl, 6),
        "avg_realized_net_pnl_usdt": round(float(evidence.get("avg_pnl") or 0.0), 6),
        "profit_factor": (
            round(profit_factor, 6) if profit_factor is not None else None
        ),
    }


def _finquant_specialization_verified(
    slot: dict[str, Any],
    evidence: dict[str, Any],
) -> bool:
    if slot.get("specialization_evidence_verified") is False:
        return False
    if str(evidence.get("verification_status") or "") != "verified":
        return False
    if str(evidence.get("identity_verified") or "").lower() != "true":
        return False
    if str(evidence.get("legacy_read_only") or "").lower() == "true":
        return False
    if evidence.get("objective_name") != "maximize_expected_realized_net_return_after_cost":
        return False
    if evidence.get("objective_version") != "2026-07-12.v1":
        return False
    if evidence.get("preference_contract_version") != "bb_finquant_return_preference.v1":
        return False
    if "trl_dpo_return_preference" not in str(evidence.get("training_stages") or ""):
        return False
    required_text = (
        "adapter_version",
        "adapter_path",
        "specialization_manifest",
        "specialization_id",
        "dataset_version",
        "source_code_version",
        "base_model_repo",
        "trained_at",
        "objective_name",
        "objective_version",
        "preference_contract_version",
        "training_stages",
    )
    if any(not str(evidence.get(key) or "").strip() for key in required_text):
        return False
    required_hashes = (
        "adapter_sha256",
        "manifest_sha256",
        "dataset_sha256",
        "dataset_lineage_sha256",
        "dataset_manifest_sha256",
        "source_script_sha256",
        "trainer_code_sha256",
        "base_model_config_sha256",
        "inference_base_model_config_sha256",
    )
    for key in required_hashes:
        value = str(evidence.get(key) or "").lower()
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            return False
    return True


def _local_ml_row(status: dict[str, Any]) -> dict[str, Any]:
    available = bool(status.get("available"))
    live = bool(status.get("allow_live_position_influence") or status.get("influence_enabled"))
    readiness = str(status.get("readiness_state") or status.get("status") or "unknown")
    if live:
        lifecycle = "live"
    elif available and readiness in {"degraded", "blocked", "promotion_blocked"}:
        lifecycle = "promotion_blocked"
    elif available:
        lifecycle = "trained"
    else:
        lifecycle = "not_trained"
    return {
        "model_id": "local_ml_profit_quality",
        "display_name": "Local ML profit quality",
        "model_family": "sklearn RandomForest/Dummy classifier-regressor pipelines",
        "task": "after_cost_entry_profit_quality",
        "trainable": True,
        "training_owner": "platform_paper_runtime",
        "runtime_role": "entry_filter_and_ranking",
        "lifecycle": lifecycle,
        "runtime_available": available,
        "artifact_available": available,
        "trained_at": status.get("trained_at"),
        "sample_count": _safe_int(
            status.get("training_shadow_sample_count") or status.get("sample_count")
        ),
        "live_influence": live,
        "quality_state": readiness,
        "blocking_reasons": _safe_list(status.get("blocking_reason_codes")),
        "identity_verified": available,
        "alias_only": False,
    }


_LOCAL_TOOL_MODELS = (
    (
        "local_ai_profit_prediction",
        "Local AI profit prediction",
        "profit",
        "after_cost_long_short_expected_return",
    ),
    (
        "local_ai_loss_filter",
        "Local AI loss filter",
        "loss_filter",
        "side_specific_loss_probability",
    ),
    (
        "local_ai_timeseries",
        "Local AI multi-horizon timeseries",
        "timeseries",
        "multi_horizon_return_forecast",
    ),
    (
        "local_ai_sequence",
        "Local AI sequence model",
        "deep_timeseries",
        "sequence_return_forecast",
    ),
    (
        "local_ai_sentiment_calibration",
        "Local AI sentiment calibration",
        "deep_sentiment",
        "event_sentiment_return_calibration",
    ),
    (
        "local_ai_exit_profile",
        "Local AI exit profile",
        "exit",
        "position_exit_attribution",
    ),
)


def _local_tool_rows(status: dict[str, Any]) -> list[dict[str, Any]]:
    models = _safe_dict(status.get("models"))
    bundle_available = bool(
        status.get("model_bundle_available") or status.get("trained_models_available")
    )
    runtime_available = bool(status.get("service_available", status.get("available")))
    promotion = _safe_dict(status.get("promotion_recommendation"))
    active_ready = bool(promotion.get("active_ready", promotion.get("live_ready")))
    canary_ready = bool(promotion.get("canary_ready"))
    stage = str(status.get("model_stage") or status.get("training_mode") or "shadow")
    rows: list[dict[str, Any]] = []
    for model_id, display_name, model_key, task in _LOCAL_TOOL_MODELS:
        model_name = str(models.get(model_key) or "").strip()
        artifact_available = bool(bundle_available and model_name)
        if artifact_available and (active_ready or stage in {"active", "live"}):
            lifecycle = "active"
        elif artifact_available and (canary_ready or stage == "canary"):
            lifecycle = "canary"
        elif artifact_available:
            lifecycle = "promotion_blocked"
        elif runtime_available:
            lifecycle = "not_trained"
        else:
            lifecycle = "service_unavailable"
        rows.append(
            {
                "model_id": model_id,
                "display_name": display_name,
                "model_family": model_name or "unknown",
                "task": task,
                "trainable": True,
                "training_owner": "phase3_quant_api",
                "runtime_role": model_key,
                "lifecycle": lifecycle,
                "runtime_available": runtime_available,
                "artifact_available": artifact_available,
                "trained_at": status.get("trained_at"),
                "sample_count": _safe_int(
                    status.get("trade_sample_count")
                    if model_key == "exit"
                    else status.get("shadow_sample_count")
                ),
                "live_influence": lifecycle == "live",
                "quality_state": stage,
                "blocking_reasons": _safe_list(promotion.get("live_blocking_reasons")),
                "identity_verified": artifact_available,
                "alias_only": False,
            }
        )
    return rows


def _specialist_rows(
    local_tools_status: dict[str, Any],
    specialist_report: dict[str, Any],
) -> list[dict[str, Any]]:
    report_rows = [
        row
        for row in _safe_list(specialist_report.get("models"))
        if isinstance(row, dict)
    ]

    def report_for(*names: str) -> dict[str, Any]:
        expected = {name.lower() for name in names}
        for row in report_rows:
            if str(row.get("model") or "").strip().lower() in expected:
                return row
        return {}

    transformer = _safe_dict(local_tools_status.get("transformers_sentiment_backend"))
    specs = (
        (
            "timesfm_2_5",
            "TimesFM 2.5",
            "google/timesfm-2.5-200m-pytorch",
            ("google/timesfm-2.5-200m-pytorch", "timesfm-2.5-primary"),
            "pretrained_timeseries_forecast",
            True,
        ),
        (
            "chronos_2",
            "Chronos-2",
            "amazon/chronos-2",
            ("amazon/chronos-2", "chronos-2-shadow-challenger"),
            "pretrained_timeseries_challenger",
            True,
        ),
        (
            "finbert",
            "FinBERT",
            "ProsusAI/finbert",
            ("ProsusAI/finbert",),
            "pretrained_sentiment_inference",
            bool(transformer.get("available")),
        ),
        (
            "finbert_tone",
            "FinBERT Tone",
            "yiyanghkust/finbert-tone",
            ("yiyanghkust/finbert-tone",),
            "pretrained_sentiment_challenger",
            bool(transformer.get("available")),
        ),
    )
    rows: list[dict[str, Any]] = []
    for model_id, display_name, model_family, report_names, task, runtime_hint in specs:
        report = report_for(*report_names)
        inference_count = _safe_int(report.get("actual_inference_count"))
        promotion_ready = bool(report.get("promotion_ready"))
        runtime_available = bool(runtime_hint and (report or model_id == "finbert"))
        identity_verified = bool(runtime_available or inference_count > 0)
        lifecycle = "inference_only" if runtime_available else "service_unavailable"
        evaluation_mode = (
            "shadow_evaluating"
            if report
            else "not_evaluated" if runtime_available else "unavailable"
        )
        rows.append(
            {
                "model_id": model_id,
                "display_name": display_name,
                "model_family": model_family,
                "task": task,
                "trainable": False,
                "training_mode": "inference_only",
                "evaluation_mode": evaluation_mode,
                "fine_tune_available": False,
                "project_adapter_available": False,
                "training_owner": None,
                "runtime_role": "specialist_evidence",
                "lifecycle": lifecycle,
                "runtime_available": runtime_available,
                "artifact_available": runtime_available,
                "trained_at": None,
                "sample_count": inference_count,
                "live_influence": False,
                "quality_state": (
                    "promotion_ready_for_canary_review"
                    if promotion_ready
                    else "promotion_blocked"
                    if report
                    else "not_evaluated"
                ),
                "blocking_reasons": _safe_list(report.get("promotion_blockers")),
                "identity_verified": identity_verified,
                "alias_only": False,
                "actual_inference_count": inference_count,
                "fallback_count": _safe_int(report.get("fallback_count")),
                "authoritative_sample_count": _safe_int(
                    report.get("authoritative_direction_aligned_count")
                ),
                "evaluation_objective": "net_return_after_all_cost_profit_factor_tail_loss",
                PROFIT_TRAINING_TARGET: _first_present(
                    report.get("authoritative_avg_return_after_all_cost_pct"),
                    report.get("avg_shadow_return_after_all_cost_pct"),
                ),
                "profit_factor": report.get("profit_factor"),
                "authoritative_profit_factor": report.get(
                    "authoritative_profit_factor"
                ),
                "tail_loss_count": _safe_int(report.get("tail_loss_count")),
                "authoritative_tail_loss_count": _safe_int(
                    report.get("authoritative_tail_loss_count")
                ),
                "promotion_ready": promotion_ready,
                "evaluation_generated_at": specialist_report.get("generated_at"),
            }
        )
    return rows


def _llm_rows(
    model_server_report: dict[str, Any],
    contribution_performance: dict[str, Any],
) -> list[dict[str, Any]]:
    slot_reports = {
        str(row.get("slot") or ""): row
        for row in _safe_list(
            model_server_report.get("required_slots")
            or model_server_report.get("slot_reports")
            or model_server_report.get("identity_slots")
        )
        if isinstance(row, dict)
    }
    runtime_reports = {
        str(row.get("slot") or ""): row
        for row in _safe_list(model_server_report.get("manifest_services"))
        if isinstance(row, dict) and str(row.get("slot") or "")
    }
    finquant_slot = _safe_dict(slot_reports.get("llm_expert_pool"))
    decision_slot = _safe_dict(slot_reports.get("llm_decision_maker"))
    risk_slot = _safe_dict(slot_reports.get("llm_high_risk_review"))

    def current_slot_runtime(slot: dict[str, Any]) -> bool:
        runtime = _safe_dict(runtime_reports.get(str(slot.get("slot") or ""))) or slot
        return bool(
            runtime.get("service_active")
            and runtime.get("endpoint_ready")
        )

    specialization = _safe_dict(finquant_slot.get("specialization_evidence"))
    finquant_runtime = current_slot_runtime(finquant_slot)
    finquant_specialized = _finquant_specialization_verified(finquant_slot, specialization)
    finquant = {
        "model_id": "bb_finquant_expert_14b",
        "display_name": "BB-FinQuant-Expert-14B",
        "model_family": str(finquant_slot.get("base_model_carrier") or "Qwen3-14B base carrier"),
        "task": "quant_expert_reasoning",
        "trainable": True,
        "training_owner": "bb_finquant_qlora_pipeline",
        "runtime_role": "expert_pool",
        "lifecycle": "trained" if finquant_specialized else "promotion_blocked",
        "runtime_available": finquant_runtime,
        "artifact_available": finquant_specialized,
        "trained_at": specialization.get("trained_at"),
        "sample_count": _safe_int(specialization.get("sample_count")),
        "live_influence": False,
        "quality_state": "specialized" if finquant_specialized else "specialization_missing",
        "blocking_reasons": [] if finquant_specialized else ["finquant_specialization_missing"],
        "identity_verified": finquant_specialized,
        "alias_only": bool(finquant_runtime and not finquant_specialized),
        "specialization_evidence": specialization,
    }
    base_rows = [
        {
            "model_id": "qwen3_14b_trade",
            "display_name": str(decision_slot.get("served_model_name") or "Decision model"),
            "model_family": str(decision_slot.get("repo_id") or "phase3 decision model"),
            "task": "trade_reasoning_fallback",
            "runtime_available": current_slot_runtime(decision_slot),
        },
        {
            "model_id": "deepseek_r1_14b_risk",
            "display_name": str(risk_slot.get("served_model_name") or "Risk review model"),
            "model_family": str(risk_slot.get("repo_id") or "phase3 risk review model"),
            "task": "risk_review",
            "runtime_available": current_slot_runtime(risk_slot),
        },
        {
            "model_id": "deepseek_online_decision",
            "display_name": "Online DeepSeek decision model",
            "model_family": "provider_managed_deepseek",
            "task": "final_decision",
            "runtime_available": True,
        },
    ]
    normalized_base_rows = []
    for row in base_rows:
        evaluation_bucket = None
        missing_reason = "model_specific_fee_after_attribution_missing"
        if row["model_id"] == "deepseek_r1_14b_risk":
            evaluation_bucket = _safe_dict(contribution_performance.get("high_risk_review"))
            missing_reason = "high_risk_review_fee_after_evaluation_missing"
        elif row["model_id"] == "deepseek_online_decision":
            evaluation_bucket = _safe_dict(contribution_performance.get("decision_llm"))
            missing_reason = "decision_llm_fee_after_evaluation_missing"
        evaluation = _fee_after_evaluation(
            evaluation_bucket,
            missing_reason=missing_reason,
        )
        normalized_base_rows.append(
            {
                **row,
                **evaluation,
                "trainable": False,
                "training_owner": None,
                "runtime_role": row["task"],
                "lifecycle": (
                    "inference_only" if row["runtime_available"] else "service_unavailable"
                ),
                "artifact_available": bool(row["runtime_available"]),
                "trained_at": None,
                "sample_count": evaluation["evaluation_sample_count"],
                "live_influence": bool(
                    row["model_id"] == "deepseek_online_decision" and row["runtime_available"]
                ),
                "identity_verified": bool(row["runtime_available"]),
                "alias_only": False,
            }
        )
    return [finquant, *normalized_base_rows]


def build_model_training_registry(
    *,
    local_ml_status: dict[str, Any] | None = None,
    local_tools_status: dict[str, Any] | None = None,
    specialist_report: dict[str, Any] | None = None,
    model_server_report: dict[str, Any] | None = None,
    contribution_performance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one truthful model lifecycle view for APIs and audits."""

    local_ml = _safe_dict(local_ml_status)
    local_tools = _safe_dict(local_tools_status)
    specialist = _safe_dict(specialist_report)
    server = _safe_dict(model_server_report)
    contributions = _safe_dict(contribution_performance)
    models = [
        _local_ml_row(local_ml),
        *_local_tool_rows(local_tools),
        *_specialist_rows(local_tools, specialist),
        *_llm_rows(server, contributions),
    ]
    lifecycle_counts = Counter(str(row.get("lifecycle") or "unknown") for row in models)
    trainable_count = sum(1 for row in models if bool(row.get("trainable")))
    alias_only = [row["model_id"] for row in models if bool(row.get("alias_only"))]
    identity_failures = [
        row["model_id"]
        for row in models
        if bool(row.get("runtime_available")) and not bool(row.get("identity_verified"))
    ]
    return {
        "version": MODEL_TRAINING_REGISTRY_VERSION,
        "policy": "endpoint_availability_is_not_training_evidence",
        "models": models,
        "summary": {
            "model_count": len(models),
            "trainable_count": trainable_count,
            "inference_or_evaluation_only_count": len(models) - trainable_count,
            "lifecycle_counts": dict(lifecycle_counts),
            "alias_only_count": len(alias_only),
            "alias_only_models": alias_only,
            "identity_failure_count": len(identity_failures),
            "identity_failure_models": identity_failures,
        },
    }
