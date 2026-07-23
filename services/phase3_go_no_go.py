"""Go/no-go evaluation for the dynamic fee-after return architecture."""

from __future__ import annotations

from typing import Any

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


def evaluate_phase3_go_no_go_cards(cards: list[dict[str, Any]]) -> dict[str, Any]:
    """Require current return/cost provenance and real execution facts."""

    by_key = _cards_by_key(cards)
    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    passed: list[str] = []

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

    trade_card = by_key.get("trade_execution_contract", {})
    trade = _details(trade_card)
    trade_summary = _safe_dict(trade.get("current_summary")) or _safe_dict(
        trade.get("summary")
    )
    trade_policy = _safe_dict(trade.get("policy"))
    required_trade_policy = (
        "entry_requires_positive_fee_after_return",
        "entry_requires_positive_return_lcb",
        "entry_requires_live_execution_cost",
        "entry_requires_dynamic_risk_budget",
        "entry_requires_complete_provenance",
        "exit_requires_position_economics",
        "exit_requires_dynamic_close_fraction",
        "filled_order_link_required",
    )
    missing_policy = [key for key in required_trade_policy if trade_policy.get(key) is not True]
    if trade.get("report_available") is False:
        blockers.append(
            _blocker(
                "dynamic_return_contract_unavailable",
                "Dynamic return execution contract report is unavailable.",
            )
        )
    if missing_policy:
        blockers.append(
            _blocker(
                "dynamic_return_contract_policy_incomplete",
                "Execution audit does not prove all return/cost/provenance invariants.",
                evidence={"missing": missing_policy},
            )
        )
    violation_count = _safe_int(trade_summary.get("contract_violation_count"))
    if violation_count > 0:
        blockers.append(
            _blocker(
                "dynamic_return_contract_current_violations",
                "Current runtime window contains executed contract violations.",
                evidence=trade_summary,
            )
        )
    elif trade_summary:
        passed.append("dynamic_return_contract_current_window_clean")

    capacity = _details(by_key.get("position_capacity_release", {}))
    capacity_policy = _safe_dict(capacity.get("policy"))
    if capacity_policy.get("strategy_learning_cannot_expand_capacity") is not True:
        blockers.append(
            _blocker(
                "hard_capacity_boundary_unproven",
                "Capacity audit does not prove that strategy learning cannot expand hard capacity.",
            )
        )
    economics_gaps = _safe_int(capacity.get("position_economics_incomplete_count"))
    exit_gaps = _safe_int(capacity.get("executed_dynamic_exit_contract_gap_count"))
    if economics_gaps or exit_gaps:
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
        },
        "policy": {
            "optimization_target": PROFIT_TRAINING_TARGET,
            "win_rate_is_diagnostic_only": True,
            "expert_memory_strategy_learning_are_observation_only": True,
            "fixed_strategy_thresholds_forbidden": True,
        },
    }
