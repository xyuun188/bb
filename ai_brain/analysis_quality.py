"""Canonical expert-call and analysis-completeness contracts."""

from __future__ import annotations

from collections import Counter
from typing import Any

from ai_brain.base_model import DecisionOutput

EXPERT_CALL_STATUSES = (
    "completed",
    "timeout",
    "parse_failed",
    "empty",
    "unavailable",
    "skipped",
)

_FALLBACK_MARKERS = (
    "timeout_fallback",
    "local_fallback",
    "market_fast_prefilter",
    "batch_expert_fallback",
    "analysis_budget_deferred",
)


def _text_status(status: Any) -> str:
    return str(status or "").strip().lower()


def _failure_status(reason: Any) -> str:
    text = str(reason or "").lower()
    if "timeout" in text or "timed out" in text or "超时" in text:
        return "timeout"
    if any(token in text for token in ("json", "parse", "format", "invalid", "截断", "格式")):
        return "parse_failed"
    if any(token in text for token in ("empty", "空返回", "空响应")):
        return "empty"
    return "unavailable"


def _decision_fallback(decision: DecisionOutput | None) -> bool:
    if not isinstance(decision, DecisionOutput):
        return False
    raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
    return any(bool(raw.get(marker)) for marker in _FALLBACK_MARKERS)


def _normalized_status(
    timing: dict[str, Any] | None,
    decision: DecisionOutput | None,
    failure_reason: str,
    attempted: bool,
) -> str:
    timing = timing if isinstance(timing, dict) else {}
    raw_status = _text_status(timing.get("status"))
    reason = str(timing.get("reason") or failure_reason or "")

    if "timeout" in raw_status:
        return "timeout"
    if raw_status in {"invalid", "parse_failed"}:
        return "parse_failed"
    if raw_status in {"empty"}:
        return "empty"
    if raw_status in {
        "analysis_budget_deferred",
        "fast_prefilter",
        "pre_expert_skipped",
    }:
        return "skipped"
    if raw_status in {
        "failed",
        "independent_provider_failed",
        "unavailable",
        "circuit_breaker_fallback",
        "partial_batch_fallback",
    }:
        return _failure_status(reason)
    if _decision_fallback(decision):
        raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
        if raw.get("timeout_fallback"):
            return "timeout"
        if raw.get("analysis_budget_deferred") or raw.get("market_fast_prefilter"):
            return "skipped"
        return "unavailable"
    if failure_reason:
        return _failure_status(failure_reason)
    if isinstance(decision, DecisionOutput):
        if not str(decision.reasoning or "").strip():
            return "empty"
        return "completed"
    return "empty" if attempted else "unavailable"


def _timings_by_name(timings: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in timings or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        current = selected.get(name)
        if current is None or row.get("replaces_batch_decision") or not current.get(
            "replaces_batch_decision"
        ):
            selected[name] = row
    return selected


def build_expert_call_contract(
    *,
    expected_names: list[str] | tuple[str, ...],
    attempted_names: list[str] | tuple[str, ...] | None,
    opinions: dict[str, DecisionOutput] | None,
    timings: list[dict[str, Any]] | None,
    failures: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Describe every expected expert without treating fallbacks as successes."""

    expected = [str(name) for name in expected_names if str(name)]
    attempted = {str(name) for name in attempted_names or [] if str(name)}
    returned = opinions if isinstance(opinions, dict) else {}
    timing_by_name = _timings_by_name(timings)
    failure_by_name = {
        str(row.get("expert_name")): str(row.get("reason") or "")
        for row in failures or []
        if isinstance(row, dict) and row.get("expert_name")
    }
    slots: list[dict[str, Any]] = []
    for name in expected:
        decision = returned.get(name)
        timing = timing_by_name.get(name, {})
        failure_reason = failure_by_name.get(name, "")
        was_attempted = name in attempted
        status = _normalized_status(timing, decision, failure_reason, was_attempted)
        reason = str(
            timing.get("reason")
            or failure_reason
            or (decision.reasoning if isinstance(decision, DecisionOutput) and status != "completed" else "")
            or ""
        ).strip()
        slot = {
            "name": name,
            "status": status,
            "attempted": was_attempted,
            "returned": isinstance(decision, DecisionOutput),
            "usable": status == "completed",
            "reason": reason[:500],
            "duration_sec": timing.get("duration_sec"),
            "provider_model": timing.get("provider_model"),
        }
        if isinstance(decision, DecisionOutput):
            slot["action"] = decision.action.value
            slot["confidence"] = float(decision.confidence or 0.0)
        slots.append(slot)

    counts = Counter(slot["status"] for slot in slots)
    status_counts = {status: int(counts.get(status, 0)) for status in EXPERT_CALL_STATUSES}
    expert_complete = bool(slots) and status_counts["completed"] == len(slots)
    first_problem = next((slot for slot in slots if slot["status"] != "completed"), None)
    return {
        "version": "2026-08-17.analysis-quality.v1",
        "expected_expert_count": len(slots),
        "attempted_expert_count": sum(1 for slot in slots if slot["attempted"]),
        "returned_expert_count": sum(1 for slot in slots if slot["returned"]),
        "successful_expert_count": status_counts["completed"],
        "status_counts": status_counts,
        "experts": slots,
        "expert_complete": expert_complete,
        "cross_validation_complete": False,
        "analysis_complete": False,
        "decision_eligible": False,
        "result": "unclear",
        "reason_code": "cross_validation_pending" if expert_complete else "insufficient_evidence",
        "reason": (
            "专家均已返回有效结构化结果，等待交叉验证。"
            if expert_complete
            else str((first_problem or {}).get("reason") or "必需专家未全部返回有效结果。")
        ),
    }


def usable_expert_opinions(
    opinions: dict[str, DecisionOutput], contract: dict[str, Any]
) -> dict[str, DecisionOutput]:
    usable = {
        str(slot.get("name"))
        for slot in contract.get("experts", [])
        if isinstance(slot, dict) and slot.get("usable")
    }
    return {name: decision for name, decision in opinions.items() if name in usable}


def finalize_analysis_quality(
    contract: dict[str, Any],
    validations: list[dict[str, Any]] | None,
    *,
    final_action: str,
    consultation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = dict(contract)
    completed_names = [
        str(slot.get("name"))
        for slot in result.get("experts", [])
        if isinstance(slot, dict) and slot.get("status") == "completed"
    ]
    expected_pairs = len(completed_names) * (len(completed_names) - 1) // 2
    completed_pairs: set[tuple[str, str]] = set()
    major_conflict_pairs: set[tuple[str, str]] = set()
    unavailable = 0
    for row in validations or []:
        if not isinstance(row, dict):
            continue
        pair = [str(name) for name in row.get("expert_pair") or [] if str(name)]
        if len(pair) != 2:
            continue
        if row.get("validation_status", "completed") == "completed":
            normalized_pair = tuple(sorted(pair))
            completed_pairs.add(normalized_pair)
            if row.get("major_conflict") or row.get("needs_resolution"):
                major_conflict_pairs.add(normalized_pair)
        else:
            unavailable += 1
    cross_complete = expected_pairs > 0 and len(completed_pairs) >= expected_pairs and unavailable == 0
    consultation_payload = consultation if isinstance(consultation, dict) else {}
    resolution_status = str(consultation_payload.get("resolution_status") or "").lower()
    resolved_action = str(consultation_payload.get("resolved_action") or "").lower()
    resolved_pairs: set[tuple[str, str]] = set()
    for raw_pair in consultation_payload.get("resolved_conflict_pairs") or []:
        if not isinstance(raw_pair, (list, tuple)):
            continue
        pair = tuple(sorted(str(name) for name in raw_pair if str(name)))
        if len(pair) == 2:
            resolved_pairs.add(pair)
    conflict_resolution_complete = not major_conflict_pairs or bool(
        consultation_payload.get("status") == "completed"
        and resolution_status == "resolved"
        and resolved_action == str(final_action or "").lower()
        and major_conflict_pairs.issubset(resolved_pairs)
    )
    unresolved_major_conflicts = major_conflict_pairs - resolved_pairs
    analysis_complete = (
        bool(result.get("expert_complete"))
        and cross_complete
        and conflict_resolution_complete
    )
    result["cross_validation"] = {
        "expected_pair_count": expected_pairs,
        "completed_pair_count": len(completed_pairs),
        "unavailable_pair_count": unavailable,
        "coverage_ratio": round(len(completed_pairs) / expected_pairs, 4) if expected_pairs else 0.0,
        "major_conflict_count": len(major_conflict_pairs),
        "unresolved_major_conflict_count": len(unresolved_major_conflicts),
        "conflicts_resolved": conflict_resolution_complete,
        "resolution_status": (
            "not_required" if not major_conflict_pairs else resolution_status or "unresolved"
        ),
        "resolved_action": resolved_action or None,
    }
    result["cross_validation_complete"] = cross_complete
    result["analysis_complete"] = analysis_complete
    result["decision_eligible"] = analysis_complete
    if analysis_complete:
        result["result"] = final_action if final_action in {"long", "short", "hold"} else "hold"
        result["reason_code"] = "analysis_complete"
        result["reason"] = "必需专家结果和两两交叉验证均完整。"
    else:
        result["result"] = "unclear"
        if not result.get("expert_complete"):
            result["reason_code"] = "insufficient_evidence"
        elif not cross_complete:
            result["reason_code"] = "cross_validation_unresolved"
            result["reason"] = "专家结果已返回，但交叉验证覆盖不完整或存在不可用核验。"
        else:
            result["reason_code"] = "direction_conflict"
            result["reason"] = "交叉验证发现重大方向冲突，深度会诊未明确解决全部冲突。"
    return result
