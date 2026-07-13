from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from services.artifact_retirement_audit import (
    PHASE3_ARTIFACT_POLICY_ID,
    PHASE3_REQUIRED_PROMOTION_FLOW,
    PHASE3_REQUIRED_TRAINING_POLICY,
)


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _status(value: Any) -> str:
    return str(value or "").strip().lower()


def _append_unique(items: list[str], value: str) -> None:
    if value and value not in items:
        items.append(value)


@dataclass(frozen=True)
class Phase3RebuildReadinessService:
    """Read-only gate before any confirmed Phase 3 artifact rebuild can write files."""

    def report(
        self,
        *,
        local_ai_tools: dict[str, Any] | None,
        governance: dict[str, Any] | None,
        historical_trade_fact_audit: dict[str, Any] | None,
        artifact_retirement_audit: dict[str, Any] | None,
        runtime_probe: dict[str, Any] | None = None,
        requested_persist_artifact: bool = False,
        confirm_phase3_rebuild: bool = False,
    ) -> dict[str, Any]:
        started_at = datetime.now(UTC)
        local_tools = _safe_dict(local_ai_tools)
        governance_report = _safe_dict(governance)
        historical = _safe_dict(historical_trade_fact_audit)
        artifacts = _safe_dict(artifact_retirement_audit)
        runtime = _safe_dict(runtime_probe)

        blockers: list[str] = []
        warnings: list[str] = []
        passed: list[str] = []

        shadow_count = _safe_int(
            local_tools.get("shadow_sample_count")
            or local_tools.get("trainable_sample_count")
            or local_tools.get("total_trainable_count")
        )
        trade_count = _safe_int(
            local_tools.get("trainable_trade_sample_count")
            or local_tools.get("trade_sample_count")
        )
        quality_report = _safe_dict(local_tools.get("quality_report"))
        totals = _safe_dict(quality_report.get("totals"))
        excluded_total = _safe_int(totals.get("excluded"))
        effective_weight_ratio = _safe_float(totals.get("effective_weight_ratio"))

        if shadow_count <= 1:
            blockers.append("shadow_training_distribution_unavailable")
        else:
            passed.append("shadow_train_and_holdout_available")
        if trade_count <= 0:
            blockers.append("authoritative_trade_distribution_unavailable")
        else:
            passed.append("authoritative_trade_distribution_available")

        governance_status = _status(governance_report.get("status"))
        contamination_risk = _status(governance_report.get("contamination_risk"))
        if governance_status in {"error", "unavailable", "failed"}:
            blockers.append("clean_training_view_unavailable")
        elif contamination_risk == "high":
            blockers.append("high_contamination_risk")
        elif governance_status in {"quarantined", "downweighted"} or excluded_total:
            warnings.append("clean_training_view_has_quarantined_or_downweighted_samples")
        else:
            passed.append("clean_training_view_available")

        historical_status = _status(historical.get("status"))
        if historical_status in {"", "unavailable", "error"}:
            blockers.append("historical_trade_fact_audit_unavailable")
        elif historical_status != "clean":
            if _safe_int(historical.get("trainable_closed_positions")) <= 0:
                blockers.append("no_clean_closed_trade_facts")
            warnings.append("historical_trade_facts_require_quarantine")
        else:
            passed.append("historical_trade_facts_clean")

        artifact_status = _status(artifacts.get("status"))
        if artifact_status in {"", "unavailable", "error"}:
            blockers.append("artifact_retirement_audit_unavailable")
        elif (
            artifact_status in {"retired_required", "untrusted", "blocked"}
            or _safe_int(artifacts.get("unresolved_artifact_count")) > 0
        ):
            blockers.append("unresolved_or_untrusted_artifacts_block_rebuild")
        elif _safe_int(artifacts.get("retired_legacy_count")) > 0:
            warnings.append("legacy_artifacts_preserved_read_only")
        else:
            passed.append("artifact_retirement_audit_ready")

        runtime_status = _status(runtime.get("status"))
        if runtime_status in {"critical", "error"}:
            blockers.append("model_runtime_probe_critical")
        elif runtime_status == "warning":
            warnings.append("model_runtime_probe_warning")
        elif runtime_status == "ok":
            passed.append("model_runtime_probe_ok")

        evaluation_policy = _safe_dict(local_tools.get("evaluation_policy"))
        promotion_flow = (
            local_tools.get("promotion_flow")
            or evaluation_policy.get("promotion_flow")
            or PHASE3_REQUIRED_PROMOTION_FLOW
        )
        if promotion_flow != PHASE3_REQUIRED_PROMOTION_FLOW:
            blockers.append("promotion_flow_not_phase3")
        else:
            passed.append("promotion_flow_ok")
        if bool(local_tools.get("live_mutation") or evaluation_policy.get("live_mutation")):
            blockers.append("live_mutation_must_remain_disabled_for_rebuild_gate")
        else:
            passed.append("live_mutation_disabled")

        if requested_persist_artifact and not confirm_phase3_rebuild:
            blockers.append("confirmed_rebuild_required_for_artifact_write")
        if not requested_persist_artifact:
            warnings.append("preflight_only_no_artifact_write_requested")
        elif confirm_phase3_rebuild:
            passed.append("explicit_rebuild_confirmation_present")

        if totals and effective_weight_ratio <= 0.0:
            blockers.append("effective_training_weight_zero")

        can_persist = not blockers and bool(requested_persist_artifact and confirm_phase3_rebuild)
        can_run_confirmed_rebuild = not blockers
        status = "ready" if can_run_confirmed_rebuild else "blocked"
        if status == "ready" and warnings:
            status = "ready_with_warnings"

        target_states = {
            "ml_signal": {
                "can_run_confirmed_rebuild": can_run_confirmed_rebuild,
                "can_persist_artifact": can_persist,
                "artifact_policy_id": PHASE3_ARTIFACT_POLICY_ID,
                "training_policy": PHASE3_REQUIRED_TRAINING_POLICY,
                "target_stage": "shadow",
            },
            "local_ai_tools": {
                "can_run_confirmed_rebuild": can_run_confirmed_rebuild,
                "can_persist_artifact": can_persist,
                "artifact_policy_id": PHASE3_ARTIFACT_POLICY_ID,
                "training_policy": PHASE3_REQUIRED_TRAINING_POLICY,
                "target_stage": "shadow",
            },
        }

        for target in target_states.values():
            if target["can_persist_artifact"]:
                _append_unique(passed, "artifact_write_gate_open_for_confirmed_rebuild")

        return {
            "status": status,
            "read_only": True,
            "audit_only": True,
            "mutates_database": False,
            "writes_artifacts": False,
            "live_mutation": False,
            "can_run_confirmed_rebuild": can_run_confirmed_rebuild,
            "can_persist_artifact": can_persist,
            "requires_persist_artifact_flag": True,
            "requires_confirm_phase3_rebuild": True,
            "requested_persist_artifact": bool(requested_persist_artifact),
            "confirm_phase3_rebuild": bool(confirm_phase3_rebuild),
            "artifact_policy_id": PHASE3_ARTIFACT_POLICY_ID,
            "training_policy": PHASE3_REQUIRED_TRAINING_POLICY,
            "promotion_flow": PHASE3_REQUIRED_PROMOTION_FLOW,
            "blockers": list(dict.fromkeys(blockers)),
            "warnings": list(dict.fromkeys(warnings)),
            "passed_checks": list(dict.fromkeys(passed)),
            "sample_floor": {
                "shadow_sample_count": shadow_count,
                "trade_sample_count": trade_count,
                "effective_weight_ratio": round(effective_weight_ratio, 4),
                "distribution_requirement": "non_empty_train_holdout_and_authoritative_trade",
            },
            "source_status": {
                "governance_status": governance_status or "unknown",
                "contamination_risk": contamination_risk or "unknown",
                "historical_trade_fact_status": historical_status or "unknown",
                "artifact_retirement_status": artifact_status or "unknown",
                "runtime_probe_status": runtime_status or "unknown",
            },
            "target_artifacts": target_states,
            "next_action": (
                "run_preflight_until_blockers_clear"
                if blockers
                else (
                    "run_confirmed_phase3_rebuild_when_operator_is_ready"
                    if not requested_persist_artifact
                    else "confirmed_rebuild_may_persist_shadow_artifacts"
                )
            ),
            "checked_at": datetime.now(UTC).isoformat(),
            "duration_seconds": round((datetime.now(UTC) - started_at).total_seconds(), 6),
        }
