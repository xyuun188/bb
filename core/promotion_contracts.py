"""Evidence and configuration contracts for controlled strategy promotion."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from core.experiment_contracts import ExperimentContractError, content_sha256
from core.secret_utils import is_sensitive_key

PROMOTION_CONTRACT_VERSION = "bb.promotion-review.v1"
CONFIG_CONTRACT_VERSION = "bb.runtime-config.v1"
PROMOTION_STAGES = ("candidate", "shadow", "paper", "live")
PROMOTION_TRANSITIONS = {
    "candidate": {"shadow"},
    "shadow": {"paper"},
    "paper": {"live"},
    "live": set(),
}


def evaluate_promotion_gate(
    evidence: Mapping[str, Any],
    *,
    minimum_return_lcb_pct: float = 0.0,
    minimum_profit_factor: float = 1.0,
    minimum_trade_count: int = 20,
) -> dict[str, Any]:
    """Evaluate promotion evidence without changing routing or model pointers."""

    metrics = evidence.get("metrics") if isinstance(evidence.get("metrics"), Mapping) else {}
    blockers: list[str] = []
    return_lcb = _float(metrics.get("authoritative_return_lcb_pct", metrics.get("return_lcb_pct")))
    profit_factor = _float(metrics.get("authoritative_profit_factor", metrics.get("profit_factor")))
    trade_count = _int(metrics.get("authoritative_trade_count", metrics.get("trade_count")))
    if return_lcb is None or return_lcb <= minimum_return_lcb_pct:
        blockers.append("authoritative_return_lcb_not_positive")
    if profit_factor is None or profit_factor < minimum_profit_factor:
        blockers.append("authoritative_profit_factor_below_threshold")
    if trade_count is None or trade_count < minimum_trade_count:
        blockers.append("insufficient_authoritative_trade_count")
    for key, blocker in (
        ("walk_forward_stability", "walk_forward_not_stable"),
        ("market_regime_stability", "market_regime_not_stable"),
        ("rolling_distribution_stability", "rolling_distribution_not_stable"),
        ("return_completeness_verified", "return_completeness_not_verified"),
        ("okx_fact_linkage_verified", "okx_fact_linkage_not_verified"),
    ):
        if evidence.get(key) is not True:
            blockers.append(blocker)
    return {
        "contract_version": PROMOTION_CONTRACT_VERSION,
        "promotion_ready": not blockers,
        "decision": "manual_review_required" if not blockers else "blocked",
        "blockers": blockers,
        "observed": {
            "authoritative_return_lcb_pct": return_lcb,
            "authoritative_profit_factor": profit_factor,
            "authoritative_trade_count": trade_count,
        },
        "thresholds": {
            "minimum_return_lcb_pct": minimum_return_lcb_pct,
            "minimum_profit_factor": minimum_profit_factor,
            "minimum_trade_count": minimum_trade_count,
        },
        "automatic_live_promotion": False,
    }


def build_promotion_review(
    *,
    artifact_id: str,
    strategy_id: str,
    strategy_version: str,
    from_stage: str,
    to_stage: str,
    evidence: Mapping[str, Any],
    reviewer: str | None = None,
    review_reason: str = "",
) -> dict[str, Any]:
    if from_stage not in PROMOTION_STAGES or to_stage not in PROMOTION_STAGES:
        raise ExperimentContractError("unknown promotion stage")
    if to_stage not in PROMOTION_TRANSITIONS[from_stage]:
        raise ExperimentContractError(f"invalid promotion transition {from_stage}->{to_stage}")
    gate = evaluate_promotion_gate(evidence)
    if to_stage == "live" and not gate["promotion_ready"]:
        raise ExperimentContractError("live promotion requires a passing evidence gate")
    content = {
        "contract_version": PROMOTION_CONTRACT_VERSION,
        "immutable": True,
        "artifact_id": str(artifact_id),
        "strategy_id": str(strategy_id),
        "strategy_version": str(strategy_version),
        "from_stage": from_stage,
        "to_stage": to_stage,
        "evidence": _safe_json(dict(evidence)),
        "gate": gate,
        "reviewer": str(reviewer or "") or None,
        "review_reason": str(review_reason or "")[:4000],
        "created_at": datetime.now(UTC).isoformat(),
    }
    return {**content, "review_id": f"promotion_{content_sha256(content)[:24]}", "review_sha256": content_sha256(content)}


def verify_promotion_review(review: Mapping[str, Any]) -> None:
    payload = dict(review or {})
    review_id = str(payload.pop("review_id", ""))
    recorded = str(payload.pop("review_sha256", ""))
    if payload.get("contract_version") != PROMOTION_CONTRACT_VERSION or payload.get("immutable") is not True:
        raise ExperimentContractError("unsupported promotion review contract")
    actual = content_sha256(payload)
    if actual != recorded or review_id != f"promotion_{actual[:24]}":
        raise ExperimentContractError("promotion review SHA-256 mismatch")
    if payload.get("to_stage") not in PROMOTION_TRANSITIONS.get(str(payload.get("from_stage")), set()):
        raise ExperimentContractError("promotion review transition is invalid")


def build_runtime_config_snapshot(
    values: Mapping[str, Any],
    *,
    environment: str,
    parent_version: str | None = None,
    changed_by: str = "system",
    change_reason: str = "",
) -> dict[str, Any]:
    safe_values = _safe_json(dict(values))
    content = {
        "contract_version": CONFIG_CONTRACT_VERSION,
        "immutable": True,
        "environment": str(environment),
        "values": safe_values,
        "parent_version": str(parent_version or "") or None,
        "changed_by": str(changed_by or "system"),
        "change_reason": str(change_reason or "")[:4000],
        "created_at": datetime.now(UTC).isoformat(),
    }
    digest = content_sha256(content)
    return {**content, "version_id": f"config_{digest[:24]}", "config_sha256": digest}


def verify_runtime_config_snapshot(snapshot: Mapping[str, Any]) -> None:
    payload = dict(snapshot or {})
    version_id = str(payload.pop("version_id", ""))
    recorded = str(payload.pop("config_sha256", ""))
    if payload.get("contract_version") != CONFIG_CONTRACT_VERSION or payload.get("immutable") is not True:
        raise ExperimentContractError("unsupported runtime config contract")
    actual = content_sha256(payload)
    if actual != recorded or version_id != f"config_{actual[:24]}":
        raise ExperimentContractError("runtime config SHA-256 mismatch")


def _safe_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            key_text = str(key)
            if is_sensitive_key(key_text):
                result[key_text] = "[REDACTED]"
            else:
                result[key_text] = _safe_json(item)
        return result
    if isinstance(value, (list, tuple, set)):
        return [_safe_json(item) for item in value]
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    return value


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
