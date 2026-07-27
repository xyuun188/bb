"""Independent directional confirmation for executable entries.

Quant models may propose a side, but correlated outputs from the same runtime
family count once.  Executable entries also require directional confirmation
from the expert analysis; all-HOLD analysis remains Shadow-only.
"""

from __future__ import annotations

import hashlib
import json
from math import isfinite
from typing import Any

INDEPENDENT_DIRECTION_SUPPORT_VERSION = (
    "2026-07-27.independent-direction-support.v1"
)
MIN_ALIGNED_EXPERT_COUNT = 2
MIN_INDEPENDENT_SUPPORT_GROUP_COUNT = 2


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _quant_family(source: str) -> str:
    normalized = str(source or "").strip().lower()
    if normalized == "local_ml":
        return "local_ml"
    if normalized in {"server_profit", "timeseries", "sentiment"}:
        return "local_ai_tools"
    return normalized


def _expert_group(item: dict[str, Any]) -> str:
    source_group = str(item.get("source_group") or "").strip()
    if source_group:
        return source_group
    provider = str(item.get("provider_model") or "").strip()
    if provider:
        return f"llm:{provider}"
    return f"expert:{item.get('model_name') or 'unknown'}"


def _fingerprint_payload(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in (
            "version",
            "eligible",
            "selected_side",
            "positive_quant_sources",
            "quant_evidence_families",
            "aligned_expert_count",
            "opposition_expert_count",
            "hold_expert_count",
            "auditable_expert_count",
            "aligned_expert_groups",
            "opposition_expert_groups",
            "independent_support_groups",
            "independent_support_group_count",
            "blocking_reasons",
            "production_permission",
        )
    }


def assess_directional_entry_support(
    direction_competition: dict[str, Any] | None,
    expert_opinions: list[dict[str, Any]] | None,
    selected_side: str,
) -> dict[str, Any]:
    """Require positive quant evidence plus non-HOLD expert confirmation."""

    side = str(selected_side or "").lower()
    opposite_side = "short" if side == "long" else "long"
    competition = _dict(direction_competition)
    side_evidence = _dict(competition.get(side))
    evidence_rows = side_evidence.get("evidence")
    if not isinstance(evidence_rows, list):
        evidence_rows = []

    positive_rows = [
        item
        for item in evidence_rows
        if isinstance(item, dict)
        and str(item.get("source") or "").strip()
        and (_float(item.get("raw_expected_return_pct")) or 0.0) > 0.0
        and (_float(item.get("objective_expected_return_pct")) or 0.0) > 0.0
        and (_float(item.get("horizon_minutes")) or 0.0) > 0.0
    ]
    positive_quant_sources = sorted(
        {str(item.get("source") or "").strip() for item in positive_rows}
    )
    quant_families = sorted(
        {
            family
            for family in (_quant_family(source) for source in positive_quant_sources)
            if family
        }
    )

    auditable_experts = [
        item
        for item in expert_opinions or []
        if isinstance(item, dict)
        and (_float(item.get("effective_weight")) or 0.0) > 0.0
        and item.get("trace_only_fallback") is not True
        and str(item.get("reasoning") or "").strip()
        and str(item.get("action") or "").lower() in {"long", "short", "hold"}
    ]
    aligned = [
        item for item in auditable_experts if str(item.get("action") or "").lower() == side
    ]
    opposition = [
        item
        for item in auditable_experts
        if str(item.get("action") or "").lower() == opposite_side
    ]
    holds = [
        item
        for item in auditable_experts
        if str(item.get("action") or "").lower() == "hold"
    ]
    aligned_groups = sorted({_expert_group(item) for item in aligned})
    opposition_groups = sorted({_expert_group(item) for item in opposition})
    support_groups = sorted(
        {f"quant:{family}" for family in quant_families}
        | {f"expert:{group}" for group in aligned_groups}
    )

    blockers: list[str] = []
    if side not in {"long", "short"}:
        blockers.append("direction_support_side_missing")
    if not quant_families:
        blockers.append("direction_support_positive_quant_evidence_missing")
    if len(auditable_experts) < 3:
        blockers.append("direction_support_expert_analysis_incomplete")
    if auditable_experts and len(holds) == len(auditable_experts):
        blockers.append("direction_support_experts_all_hold")
    if len(aligned) < MIN_ALIGNED_EXPERT_COUNT:
        blockers.append("direction_support_aligned_experts_insufficient")
    if len(aligned) <= len(opposition):
        blockers.append("direction_support_expert_opposition_not_resolved")
    if len(support_groups) < MIN_INDEPENDENT_SUPPORT_GROUP_COUNT:
        blockers.append("direction_support_independent_groups_insufficient")

    result = {
        "version": INDEPENDENT_DIRECTION_SUPPORT_VERSION,
        "eligible": not blockers,
        "reason": (
            "independent_positive_direction_support_ready"
            if not blockers
            else blockers[0]
        ),
        "selected_side": side if side in {"long", "short"} else "neutral",
        "positive_quant_sources": positive_quant_sources,
        "quant_evidence_families": quant_families,
        "aligned_expert_count": len(aligned),
        "opposition_expert_count": len(opposition),
        "hold_expert_count": len(holds),
        "auditable_expert_count": len(auditable_experts),
        "aligned_expert_groups": aligned_groups,
        "opposition_expert_groups": opposition_groups,
        "independent_support_groups": support_groups,
        "independent_support_group_count": len(support_groups),
        "blocking_reasons": list(dict.fromkeys(blockers)),
        "production_permission": False,
    }
    result["contract_fingerprint"] = _fingerprint(_fingerprint_payload(result))
    return result


def directional_entry_support_reasons(value: Any, selected_side: str) -> list[str]:
    support = _dict(value)
    reasons: list[str] = []
    if support.get("version") != INDEPENDENT_DIRECTION_SUPPORT_VERSION:
        reasons.append("direction_support_version_invalid")
    if support.get("eligible") is not True:
        reasons.append("direction_support_not_eligible")
    if support.get("selected_side") != str(selected_side or "").lower():
        reasons.append("direction_support_side_mismatch")
    if not support.get("quant_evidence_families"):
        reasons.append("direction_support_positive_quant_evidence_missing")
    if int(support.get("aligned_expert_count") or 0) < MIN_ALIGNED_EXPERT_COUNT:
        reasons.append("direction_support_aligned_experts_insufficient")
    if int(support.get("aligned_expert_count") or 0) <= int(
        support.get("opposition_expert_count") or 0
    ):
        reasons.append("direction_support_expert_opposition_not_resolved")
    if int(support.get("independent_support_group_count") or 0) < (
        MIN_INDEPENDENT_SUPPORT_GROUP_COUNT
    ):
        reasons.append("direction_support_independent_groups_insufficient")
    if support.get("blocking_reasons"):
        reasons.append("direction_support_contains_blockers")
    if support.get("contract_fingerprint") != _fingerprint(
        _fingerprint_payload(support)
    ):
        reasons.append("direction_support_fingerprint_mismatch")
    return list(dict.fromkeys(reasons))
