"""Read-only Phase 3 go/no-go aggregation gate."""

from __future__ import annotations

from typing import Any


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _blocker(code: str, message: str, *, evidence: Any | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {"code": code, "severity": "blocking", "message": message}
    if evidence is not None:
        item["evidence"] = evidence
    return item


def _warning(code: str, message: str, *, evidence: Any | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {"code": code, "severity": "warning", "message": message}
    if evidence is not None:
        item["evidence"] = evidence
    return item


def _card_by_key(cards: list[dict[str, Any]], key: str) -> dict[str, Any]:
    for card in cards:
        if str(card.get("key") or "") == key:
            return card
    return {}


def _card_status(card: dict[str, Any]) -> str:
    return str(card.get("status") or "missing").lower()


def _card_details(card: dict[str, Any]) -> dict[str, Any]:
    return _safe_dict(card.get("details"))


def _blocker_codes(blockers: list[Any]) -> set[str]:
    codes: set[str] = set()
    for item in blockers:
        if isinstance(item, dict):
            code = str(item.get("code") or "").strip()
        else:
            code = str(item or "").strip()
        if code:
            codes.add(code)
    return codes


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


def _profit_first_brain_coverage_gaps(ranking: dict[str, Any]) -> list[str]:
    recommendations = _safe_dict(ranking.get("brain_recommendations"))
    coverage = _safe_dict(recommendations.get("brain_output_coverage"))
    required = (
        "source_weights",
        "strategy_weights",
        "lane_threshold_recommendations",
        "size_promotion_demotion",
        "no_entry_threshold_recommendations",
        "exit_policy_adjustments",
        "shadow_canary_live_decisions",
    )
    return [key for key in required if coverage.get(key) is not True]


def _specialist_tail_risk_models(models: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in models:
        row = _safe_dict(raw)
        if not row:
            continue
        tail_loss_count = _safe_int(row.get("tail_loss_count"))
        blockers = _safe_list(row.get("promotion_blockers"))
        if tail_loss_count <= 0 and "false_signal_loss_exceeds_floor" not in blockers:
            continue
        rows.append(
            {
                "model": row.get("model"),
                "tool": row.get("tool"),
                "promotion_blockers": blockers[:6],
                "tail_loss_count": tail_loss_count,
                "worst_realized_return_pct": row.get("worst_realized_return_pct"),
                "tail_loss_symbols": _safe_list(row.get("tail_loss_symbols"))[:6],
                "worst_samples": _safe_list(row.get("worst_samples"))[:3],
            }
        )
    rows.sort(
        key=lambda item: (
            _safe_int(item.get("tail_loss_count")),
            abs(_safe_float(item.get("worst_realized_return_pct"))),
        ),
        reverse=True,
    )
    return rows[:6]


def evaluate_phase3_go_no_go_cards(cards: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Phase 3 audit cards into a conservative next-step gate."""

    cards_by_key = {str(card.get("key") or ""): card for card in cards}
    server_migration = _card_details(_card_by_key(cards, "phase3_server_migration"))
    model_readiness = _card_details(_card_by_key(cards, "phase3_model_server_readiness"))
    preflight = _card_details(_card_by_key(cards, "phase3_paper_resume_preflight"))
    observation = _card_details(_card_by_key(cards, "phase3_paper_resume_observation"))
    training = _card_details(_card_by_key(cards, "model_training"))
    ranking_card = _card_by_key(cards, "profit_first_ranking")
    ranking = _card_details(ranking_card)
    ranking_summary = _safe_dict(ranking.get("summary"))
    governance_card = _card_by_key(cards, "profit_first_governance")
    governance = _card_details(governance_card)
    governance_summary = _safe_dict(governance.get("summary"))
    recovery_blockers_card = _card_by_key(cards, "profit_first_recovery_blockers")
    recovery_blockers = _card_details(recovery_blockers_card)
    recovery_blockers_summary = _safe_dict(recovery_blockers.get("summary"))
    trade_contract_card = _card_by_key(cards, "trade_execution_contract")
    trade_contract = _card_details(trade_contract_card)
    trade_contract_policy = _safe_dict(trade_contract.get("policy"))
    trade_contract_summary = _safe_dict(trade_contract.get("summary"))
    trade_contract_current = _safe_dict(trade_contract.get("current_summary"))
    trade_contract_window = trade_contract_current or trade_contract_summary
    trade_contract_unresolved_violation_count = _safe_int(
        trade_contract_window.get("historical_recovery_quarantine_unresolved_count"),
        _safe_int(trade_contract_window.get("contract_violation_count")),
    )
    trade_contract_quarantined_violation_count = _safe_int(
        trade_contract_window.get("historical_recovery_quarantined_violation_count")
    )
    specialist = _safe_dict(training.get("specialist_shadow_evaluation"))
    specialist_summary = _safe_dict(specialist.get("summary"))

    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    passed: list[str] = []
    observation_status = str(observation.get("status") or "missing").lower()
    paper_active = bool(observation.get("paper_active"))
    preflight_consumed_after_resume = (
        bool(preflight)
        and not bool(preflight.get("can_resume_paper"))
        and paper_active
        and observation_status in {"critical", "warming_up", "healthy"}
    )
    platform_model_endpoints_ready = "phase3_model_server_platform_endpoints_ready" in set(
        str(item) for item in _safe_list(preflight.get("passed_checks"))
    )
    endpoint_verified_model_server = bool(
        platform_model_endpoints_ready
        and paper_active
        and observation_status in {"warming_up", "healthy"}
    )

    required_cards = {
        "phase3_server_migration": "server resource-release/migration gate",
        "phase3_model_server_readiness": "model-server readiness gate",
        "phase3_paper_resume_preflight": "paper-resume hard gate",
        "phase3_paper_resume_observation": "paper observation gate",
        "model_training": "model training/promotion gate",
        "trade_execution_contract": "Profit-First trade execution contract gate",
        "profit_first_ranking": "Profit-First realized-PnL ranking gate",
        "profit_first_governance": "Profit-First no-entry and losing-exit governance gate",
    }
    for key, label in required_cards.items():
        card = cards_by_key.get(key)
        if not card:
            blockers.append(_blocker(f"{key}_missing", f"Missing {label}."))
            continue
        if _card_status(card) == "critical":
            if key == "phase3_paper_resume_preflight" and preflight_consumed_after_resume:
                passed.append("paper_resume_preflight_consumed_after_resume")
                continue
            if (
                key in {"phase3_model_server_readiness", "phase3_server_migration"}
                and endpoint_verified_model_server
            ):
                warnings.append(
                    _warning(
                        f"{key}_remote_audit_unverified",
                        "Remote model-server audit is unverified, but platform model endpoints are healthy during active paper observation.",
                        evidence={"summary": card.get("summary")},
                    )
                )
                passed.append(f"{key}_platform_endpoint_verified")
                continue
            if key == "model_training" and _card_details(card).get("hard_failure") is False:
                warnings.append(
                    _warning(
                        "model_training_observing_not_hard_failure",
                        "Model training is in a controlled observation state, not a hard blocker.",
                        evidence={"summary": card.get("summary")},
                    )
                )
                passed.append("model_training_available")
                continue
            if key == "profit_first_ranking" and ranking.get("report_available") is not False:
                ranking_hard_items = [
                    item
                    for item in _safe_list(ranking.get("blockers"))
                    if _safe_dict(item).get("severity") == "blocking"
                ]
                if not any(
                    not _ranking_blocker_is_lane_scoped(item)
                    for item in ranking_hard_items
                ):
                    warnings.append(
                        _warning(
                            "profit_first_ranking_affected_lanes_contained",
                            (
                                "Profit-First ranking is critical only for identified "
                                "losing lanes; those lanes remain ineligible for promotion "
                                "or budget increase."
                            ),
                            evidence={
                                "disable_count": _safe_int(
                                    ranking_summary.get("disable_count")
                                ),
                                "contained_blockers": ranking_hard_items[:8],
                            },
                        )
                    )
                    passed.append("profit_first_ranking_lane_containment_active")
                    continue
            if (
                key == "trade_execution_contract"
                and trade_contract_unresolved_violation_count <= 0
                and trade_contract_quarantined_violation_count > 0
            ):
                warnings.append(
                    _warning(
                        "trade_execution_contract_historical_quarantined",
                        (
                            "Trade contract card is critical only because historical "
                            "Profit-First recovery samples remain quarantined and excluded "
                            "from clean training/promotion."
                        ),
                        evidence={
                            "unresolved_violation_count": trade_contract_unresolved_violation_count,
                            "quarantined_violation_count": trade_contract_quarantined_violation_count,
                            "summary": card.get("summary"),
                        },
                    )
                )
                passed.append("trade_execution_contract_historical_quarantine_accounted")
                continue
            blockers.append(
                _blocker(
                    f"{key}_critical",
                    f"{label} is critical.",
                    evidence={"summary": card.get("summary")},
                )
            )
        else:
            passed.append(f"{key}_available")

    if not trade_contract_card:
        blockers.append(
            _blocker(
                "profit_first_trade_contract_missing",
                "Profit-First trade execution contract audit is missing.",
            )
        )
    else:
        if trade_contract.get("report_available") is False:
            blockers.append(
                _blocker(
                    "profit_first_trade_contract_unavailable",
                    "Profit-First trade execution contract audit failed and cannot prove resume safety.",
                    evidence={
                        "error": trade_contract.get("error"),
                        "summary": trade_contract_summary,
                    },
                )
            )
        elif not bool(trade_contract_policy.get("entry_requires_profit_first_trade_plan")):
            blockers.append(
                _blocker(
                    "profit_first_trade_plan_policy_missing",
                    "Trade execution policy must require profit_first_trade_plan before resume.",
                    evidence={"policy": trade_contract_policy},
                )
            )
        else:
            passed.append("profit_first_trade_plan_policy_active")
        if (
            _card_status(trade_contract_card) == "critical"
            and trade_contract_unresolved_violation_count > 0
        ):
            blockers.append(
                _blocker(
                    "profit_first_trade_contract_critical",
                    "Profit-First trade execution contract has hard violations.",
                    evidence={
                        "summary": trade_contract_card.get("summary"),
                        "violation_reason_counts": trade_contract.get("violation_reason_counts"),
                    },
                )
            )

    profit_first_window = trade_contract_window
    profit_first_hard_counts = {
        "profit_first_plan_missing_count": _safe_int(
            profit_first_window.get(
                "profit_first_plan_missing_count_unresolved",
                profit_first_window.get("profit_first_plan_missing_count"),
            )
        ),
        "profit_first_plan_incomplete_count": _safe_int(
            profit_first_window.get(
                "profit_first_plan_incomplete_count_unresolved",
                profit_first_window.get("profit_first_plan_incomplete_count"),
            )
        ),
        "shadow_lane_executed_count": _safe_int(
            profit_first_window.get(
                "shadow_lane_executed_count_unresolved",
                profit_first_window.get("shadow_lane_executed_count"),
            )
        ),
        "profit_first_position_ladder_missing_count": _safe_int(
            profit_first_window.get(
                "profit_first_position_ladder_missing_count_unresolved",
                profit_first_window.get("profit_first_position_ladder_missing_count"),
            )
        ),
        "exit_plan_reference_missing_count": _safe_int(
            profit_first_window.get(
                "exit_plan_reference_missing_count_unresolved",
                profit_first_window.get("exit_plan_reference_missing_count"),
            )
        ),
        "exit_plan_failure_reason_missing_count": _safe_int(
            profit_first_window.get(
                "exit_plan_failure_reason_missing_count_unresolved",
                profit_first_window.get("exit_plan_failure_reason_missing_count"),
            )
        ),
        "low_payoff_meaningful_size_count": _safe_int(
            profit_first_window.get(
                "low_payoff_meaningful_size_count_unresolved",
                profit_first_window.get("low_payoff_meaningful_size_count"),
            )
        ),
        "profit_first_lane_size_above_max_count": _safe_int(
            profit_first_window.get(
                "profit_first_lane_size_above_max_count_unresolved",
                profit_first_window.get("profit_first_lane_size_above_max_count"),
            )
        ),
        "probe_loss_brake_bypassed_count": _safe_int(
            profit_first_window.get(
                "probe_loss_brake_bypassed_count_unresolved",
                profit_first_window.get("probe_loss_brake_bypassed_count"),
            )
        ),
        "meaningful_lane_tiny_without_budget_reason_count": _safe_int(
            profit_first_window.get(
                "meaningful_lane_tiny_without_budget_reason_count_unresolved",
                profit_first_window.get("meaningful_lane_tiny_without_budget_reason_count"),
            )
        ),
    }
    if any(profit_first_hard_counts.values()):
        blockers.append(
            _blocker(
                "profit_first_trade_plan_current_window_violations",
                "Current trade contract window has Profit-First plan violations.",
                evidence=profit_first_hard_counts,
            )
        )
    elif profit_first_window:
        passed.append("profit_first_trade_plan_current_window_clean")

    if not ranking_card:
        blockers.append(
            _blocker(
                "profit_first_ranking_missing",
                "Profit-First realized-PnL ranking audit is missing.",
            )
        )
    else:
        unsafe_ranking_flags = {
            "live_mutation": bool(ranking.get("live_mutation")),
            "live_weight_mutation": bool(ranking.get("live_weight_mutation")),
            "live_sizing_mutation": bool(ranking.get("live_sizing_mutation")),
            "can_change_model_routing": bool(ranking.get("can_change_model_routing")),
            "can_change_strategy_weight": bool(ranking.get("can_change_strategy_weight")),
            "can_increase_live_size": bool(ranking.get("can_increase_live_size")),
        }
        if ranking.get("report_available") is False:
            blockers.append(
                _blocker(
                    "profit_first_ranking_unavailable",
                    "Profit-First ranking audit failed and cannot prove realized-PnL readiness.",
                    evidence={
                        "error": ranking.get("error"),
                        "status": ranking.get("status"),
                        "summary": ranking_summary,
                    },
                )
            )
        elif any(unsafe_ranking_flags.values()):
            blockers.append(
                _blocker(
                    "profit_first_ranking_not_read_only",
                    "Profit-First ranking must be read-only before resume.",
                    evidence=unsafe_ranking_flags,
                )
            )
        elif not bool(ranking.get("ranking_ready")):
            blockers.append(
                _blocker(
                    "profit_first_ranking_not_ready",
                    "Profit-First ranking has not produced realized-PnL evidence yet.",
                    evidence={
                        "status": ranking.get("status"),
                        "summary": ranking_summary,
                    },
                )
            )
        else:
            passed.append("profit_first_ranking_ready")
            brain_gaps = _profit_first_brain_coverage_gaps(ranking)
            if brain_gaps:
                blockers.append(
                    _blocker(
                        "profit_first_brain_output_coverage_missing",
                        "Profit-First ranking must expose complete brain-training outputs before resume.",
                        evidence={
                            "missing_outputs": brain_gaps,
                            "brain_recommendations": _safe_dict(
                                ranking.get("brain_recommendations")
                            ),
                        },
                    )
                )
            else:
                passed.append("profit_first_brain_output_coverage_complete")

        disable_count = _safe_int(ranking_summary.get("disable_count"))
        hard_ranking_blockers = [
            item
            for item in _safe_list(ranking.get("blockers"))
            if _safe_dict(item).get("severity") == "blocking"
        ]
        unscoped_hard_ranking_blockers = [
            item
            for item in hard_ranking_blockers
            if not _ranking_blocker_is_lane_scoped(item)
        ]
        lane_scoped_hard_ranking_blockers = [
            item
            for item in hard_ranking_blockers
            if _ranking_blocker_is_lane_scoped(item)
        ]
        if unscoped_hard_ranking_blockers:
            blockers.append(
                _blocker(
                    "profit_first_ranking_has_disable_blockers",
                    "Profit-First ranking contains an unscoped hard blocker that cannot be contained to one model/strategy lane.",
                    evidence={
                        "disable_count": disable_count,
                        "blockers": unscoped_hard_ranking_blockers[:8],
                    },
                )
            )
        elif disable_count or lane_scoped_hard_ranking_blockers:
            warnings.append(
                _warning(
                    "profit_first_ranking_lanes_contained",
                    "Losing model/strategy/lane combinations remain disabled or shadow-only without blocking healthy paper runtime.",
                    evidence={
                        "disable_count": disable_count,
                        "contained_blockers": lane_scoped_hard_ranking_blockers[:8],
                        "containment_policy": (
                            "affected lanes cannot receive promotion or budget increase"
                        ),
                    },
                )
            )
        demote_count = _safe_int(ranking_summary.get("demote_count"))
        if demote_count:
            warnings.append(
                _warning(
                    "profit_first_ranking_has_demotions",
                    "Some model/strategy/lane combinations require demotion or reduced budget.",
                    evidence={
                        "demote_count": demote_count,
                        "blockers": _safe_list(ranking.get("blockers"))[:8],
                    },
                )
            )

    if not governance_card:
        blockers.append(
            _blocker(
                "profit_first_governance_missing",
                "Profit-First no-entry and losing-exit governance audit is missing.",
            )
        )
    else:
        governance_unsafe_flags = {
            "live_mutation": bool(governance.get("live_mutation")),
            "live_entry_mutation": bool(governance.get("live_entry_mutation")),
            "live_exit_mutation": bool(governance.get("live_exit_mutation")),
            "live_weight_mutation": bool(governance.get("live_weight_mutation")),
            "live_sizing_mutation": bool(governance.get("live_sizing_mutation")),
            "can_submit_orders": bool(governance.get("can_submit_orders")),
            "can_start_trading_service": bool(governance.get("can_start_trading_service")),
            "can_change_model_routing": bool(governance.get("can_change_model_routing")),
            "can_change_strategy_weight": bool(governance.get("can_change_strategy_weight")),
            "can_increase_live_size": bool(governance.get("can_increase_live_size")),
        }
        if governance.get("report_available") is False or str(governance.get("status") or "") == "unavailable":
            blockers.append(
                _blocker(
                    "profit_first_governance_unavailable",
                    "Profit-First governance audit failed and cannot prove no-entry/loss-exit readiness.",
                    evidence={
                        "error": governance.get("error"),
                        "status": governance.get("status"),
                        "summary": governance_summary,
                    },
                )
            )
        elif any(governance_unsafe_flags.values()):
            blockers.append(
                _blocker(
                    "profit_first_governance_not_read_only",
                    "Profit-First governance must be read-only before resume.",
                    evidence=governance_unsafe_flags,
                )
            )
        elif _safe_list(governance.get("missing_brain_outputs")):
            blockers.append(
                _blocker(
                    "profit_first_governance_brain_outputs_missing",
                    "Profit-First governance must include complete brain outputs before resume.",
                    evidence={
                        "missing_outputs": _safe_list(governance.get("missing_brain_outputs")),
                        "summary": governance_summary,
                    },
                )
            )
        else:
            passed.append("profit_first_governance_ready")
        if _safe_int(governance_summary.get("no_entry_sample_count")) == 0:
            warnings.append(
                _warning(
                    "profit_first_governance_no_entry_samples_empty",
                    "Profit-First governance has no recent no-entry samples yet.",
                    evidence={"summary": governance_summary},
                )
            )
        if _safe_int(governance_summary.get("losing_exit_sample_count")) == 0:
            warnings.append(
                _warning(
                    "profit_first_governance_losing_exit_samples_empty",
                    "Profit-First governance has no recent losing-exit samples yet.",
                    evidence={"summary": governance_summary},
                )
            )

    if recovery_blockers_card:
        if _safe_int(recovery_blockers.get("blocking_item_count")) > 0:
            blockers.append(
                _blocker(
                    "profit_first_recovery_blockers_not_clear",
                    "Profit-First recovery has concrete blockers that must be repaired, disabled, or quarantined before resume.",
                    evidence={
                        "summary": recovery_blockers_summary,
                        "blocking_item_count": _safe_int(
                            recovery_blockers.get("blocking_item_count")
                        ),
                        "items": _safe_list(recovery_blockers.get("items"))[:8],
                    },
                )
            )
        elif bool(recovery_blockers.get("resume_clear")):
            passed.append("profit_first_recovery_blockers_clear")

    if bool(server_migration.get("phase3_go_live_blocked")):
        if endpoint_verified_model_server:
            warnings.append(
                _warning(
                    "server_migration_remote_audit_unverified",
                    "Server migration remote audit is unverified, but Phase 3 platform endpoints are healthy for paper observation.",
                    evidence={
                        "blockers": _safe_list(server_migration.get("blockers"))[:8],
                        "warnings": _safe_list(server_migration.get("warnings"))[:8],
                    },
                )
            )
            passed.append("server_migration_platform_endpoint_verified")
        else:
            blockers.append(
                _blocker(
                    "server_migration_go_live_blocked",
                    "Phase 3 server resource-release/migration gate still blocks go-live.",
                    evidence={
                        "blockers": _safe_list(server_migration.get("blockers"))[:8],
                        "warnings": _safe_list(server_migration.get("warnings"))[:8],
                    },
                )
            )
    elif server_migration:
        passed.append("server_migration_ready")

    if bool(model_readiness.get("phase3_model_service_go_live_blocked")):
        if endpoint_verified_model_server:
            warnings.append(
                _warning(
                    "model_server_remote_audit_unverified",
                    "Model-server manifest audit is unverified, but platform model endpoints are healthy for paper observation.",
                    evidence={
                        "status": model_readiness.get("status"),
                        "blockers": _safe_list(model_readiness.get("blockers"))[:8],
                    },
                )
            )
            passed.append("model_server_platform_endpoint_verified")
        else:
            blockers.append(
                _blocker(
                    "model_server_go_live_blocked",
                    "Phase 3 model-server readiness gate still blocks service go-live.",
                    evidence={
                        "status": model_readiness.get("status"),
                        "blockers": _safe_list(model_readiness.get("blockers"))[:8],
                    },
                )
            )
    elif bool(model_readiness.get("runtime_ready")):
        passed.append("model_server_runtime_ready")
    elif model_readiness:
        blockers.append(
            _blocker(
                "model_server_runtime_not_ready",
                "Phase 3 model-server runtime is not ready.",
                evidence={"status": model_readiness.get("status")},
            )
        )

    can_resume_paper = bool(preflight.get("can_resume_paper"))
    if preflight and not can_resume_paper and not preflight_consumed_after_resume:
        blockers.append(
            _blocker(
                "paper_resume_preflight_not_ready",
                "Paper cannot resume until the hard preflight returns can_resume_paper=true.",
                evidence={
                    "status": preflight.get("status"),
                    "blockers": _safe_list(preflight.get("blockers"))[:8],
                },
            )
        )
    elif can_resume_paper:
        passed.append("paper_resume_preflight_ready")
    elif preflight_consumed_after_resume:
        passed.append("paper_resume_preflight_consumed_after_resume")

    observation_safe = not any(
        bool(observation.get(key))
        for key in (
            "starts_trading_service",
            "submits_orders",
            "changes_model_routing",
            "live_mutation",
        )
    )
    if observation and not observation_safe:
        blockers.append(
            _blocker(
                "paper_observation_contract_unsafe",
                "Paper observation contract is unsafe and cannot justify promotion.",
                evidence={
                    "starts_trading_service": observation.get("starts_trading_service"),
                    "submits_orders": observation.get("submits_orders"),
                    "changes_model_routing": observation.get("changes_model_routing"),
                    "live_mutation": observation.get("live_mutation"),
                },
            )
        )
    elif observation_status == "critical":
        blockers.append(
            _blocker(
                "paper_observation_critical",
                "Paper observation reports critical post-resume evidence.",
                evidence={"blockers": _safe_list(observation.get("blockers"))[:8]},
            )
        )
    elif observation_status in {"waiting_for_resume", "warming_up"}:
        warnings.append(
            _warning(
                f"paper_observation_{observation_status}",
                "Paper observation is not promotion-ready yet.",
                evidence={
                    "status": observation_status,
                    "paper_active": observation.get("paper_active"),
                    "can_use_for_promotion": observation.get("can_use_for_promotion"),
                },
            )
        )
    elif observation_status == "healthy" and bool(observation.get("can_use_for_promotion")):
        passed.append("paper_observation_healthy")

    local_tools = _safe_dict(training.get("local_ai_tools"))
    promotion = _safe_dict(local_tools.get("promotion_recommendation"))
    paper_gate = _safe_dict(promotion.get("paper_observation_gate"))
    specialist_promotion_ready_count = int(
        specialist_summary.get("promotion_ready_count")
        if specialist_summary.get("promotion_ready_count") is not None
        else specialist.get("promotion_ready_count") or 0
    )
    specialist_blocked_count = int(
        specialist_summary.get("blocked_count")
        if specialist_summary.get("blocked_count") is not None
        else specialist.get("blocked_count") or 0
    )
    specialist_models = _safe_list(specialist.get("models"))
    specialist_tail_risk_models = _specialist_tail_risk_models(specialist_models)
    specialist_available = bool(specialist.get("available"))
    if promotion:
        if not bool(promotion.get("canary_ready")):
            warnings.append(
                _warning(
                    "model_promotion_canary_not_ready",
                    "Model promotion is still shadow-only.",
                    evidence={
                        "recommended_stage": promotion.get("recommended_stage"),
                        "canary_blocking_reasons": _safe_list(
                            promotion.get("canary_blocking_reasons")
                        )[:8],
                    },
                )
            )
        else:
            passed.append("model_promotion_canary_candidate_ready")
        if bool(promotion.get("live_ready")):
            warnings.append(
                _warning(
                    "model_promotion_live_ready_requires_operator",
                    "Live readiness is only an evidence state; operator approval is still required.",
                )
            )
    if paper_gate:
        passed.append("model_promotion_has_paper_observation_gate")
        if paper_gate.get("required") is not True:
            blockers.append(
                _blocker(
                    "model_promotion_paper_gate_not_required",
                    "Promotion policy must require the paper observation gate.",
                    evidence=paper_gate,
                )
            )
    elif promotion:
        blockers.append(
            _blocker(
                "model_promotion_missing_paper_gate",
                "Promotion recommendation is missing the paper observation gate.",
            )
        )
    canary_evidence_requested = (
        observation_status == "healthy"
        and bool(observation.get("can_use_for_promotion"))
        and bool(promotion.get("canary_ready"))
    )
    specialist_canary_blocked = False
    if not specialist_available:
        warnings.append(
            _warning(
                "specialist_shadow_evaluation_missing",
                "Specialist shadow evaluation is missing; canary cannot use professional-model evidence yet.",
            )
        )
        specialist_canary_blocked = canary_evidence_requested
    elif specialist_promotion_ready_count <= 0:
        warnings.append(
            _warning(
                "specialist_shadow_no_promotion_ready_model",
                "Specialist shadow models are still collecting evidence and cannot enter canary yet.",
                evidence={
                    "promotion_ready_count": specialist_promotion_ready_count,
                    "blocked_count": specialist_blocked_count,
                    "top_blocked_reasons": _safe_list(
                        specialist_summary.get("top_blocked_reasons")
                    )[:8],
                    "tail_risk_models": specialist_tail_risk_models,
                },
            )
        )
        specialist_canary_blocked = canary_evidence_requested
    else:
        passed.append("specialist_shadow_has_promotion_ready_model")

    if blockers:
        status = "blocked"
        next_step = "stay_shadow_fix_blockers"
        can_start_paper = False
        can_enter_canary = False
    elif observation_status == "healthy" and bool(observation.get("can_use_for_promotion")):
        status = "paper_observation_healthy"
        next_step = (
            "stay_shadow_improve_specialists"
            if specialist_canary_blocked
            else "operator_review_for_canary"
        )
        can_start_paper = False
        can_enter_canary = (
            bool(promotion.get("canary_ready")) and not specialist_canary_blocked
            if promotion
            else False
        )
    elif paper_active or observation_status == "warming_up":
        status = "post_resume_observing"
        next_step = "continue_post_resume_observation"
        can_start_paper = False
        can_enter_canary = False
    elif can_resume_paper:
        status = "paper_resume_ready"
        next_step = "resume_paper_pending_operator_approval"
        can_start_paper = True
        can_enter_canary = False
    else:
        status = "blocked"
        next_step = "stay_shadow_collect_evidence"
        can_start_paper = False
        can_enter_canary = False

    raw_promotion_canary_ready = (
        bool(promotion.get("canary_ready")) if promotion.get("canary_ready") is not None else None
    )
    effective_promotion_canary_ready = (
        raw_promotion_canary_ready and not specialist_canary_blocked
        if raw_promotion_canary_ready is not None
        else None
    )

    return {
        "status": status,
        "next_step": next_step,
        "read_only": True,
        "audit_only": True,
        "starts_trading_service": False,
        "submits_orders": False,
        "changes_model_routing": False,
        "live_mutation": False,
        "can_start_paper_with_operator_approval": can_start_paper,
        "can_enter_canary_with_operator_approval": can_enter_canary,
        "can_enter_live": False,
        "blockers": blockers,
        "warnings": warnings,
        "passed_checks": list(dict.fromkeys(passed)),
        "inputs": {
            "card_keys": sorted(cards_by_key),
            "server_migration_status": server_migration.get("status"),
            "model_server_status": model_readiness.get("status"),
            "preflight_status": preflight.get("status"),
            "can_resume_paper": preflight.get("can_resume_paper"),
            "preflight_consumed_after_resume": preflight_consumed_after_resume,
            "paper_observation_status": observation_status,
            "paper_active": paper_active,
            "paper_can_use_for_promotion": observation.get("can_use_for_promotion"),
            "promotion_recommended_stage": promotion.get("recommended_stage"),
            "promotion_canary_ready": effective_promotion_canary_ready,
            "raw_model_promotion_canary_ready": raw_promotion_canary_ready,
            "profit_first_ranking_ready": bool(ranking.get("ranking_ready")),
            "profit_first_ranking_disable_count": _safe_int(ranking_summary.get("disable_count")),
            "profit_first_ranking_demote_count": _safe_int(ranking_summary.get("demote_count")),
            "profit_first_ranking_promote_candidate_count": _safe_int(
                ranking_summary.get("promote_candidate_count")
            ),
            "profit_first_governance_status": governance.get("status"),
            "profit_first_governance_no_entry_sample_count": _safe_int(
                governance_summary.get("no_entry_sample_count")
            ),
            "profit_first_governance_losing_exit_sample_count": _safe_int(
                governance_summary.get("losing_exit_sample_count")
            ),
            "profit_first_recovery_blocking_item_count": _safe_int(
                recovery_blockers.get("blocking_item_count")
            ),
            "profit_first_recovery_resume_clear": bool(recovery_blockers.get("resume_clear")),
            "specialist_promotion_ready_count": specialist_promotion_ready_count,
            "specialist_blocked_count": specialist_blocked_count,
            "specialist_canary_blocked": specialist_canary_blocked,
            "specialist_tail_loss_model_count": len(specialist_tail_risk_models),
            "specialist_tail_loss_total": sum(
                _safe_int(item.get("tail_loss_count")) for item in specialist_tail_risk_models
            ),
        },
    }


def _ranking_blocker_is_lane_scoped(item: Any) -> bool:
    item = _safe_dict(item)
    if not str(item.get("code") or "").startswith("strategy_"):
        return False
    evidence = _safe_dict(item.get("evidence"))
    return any(
        str(evidence.get(key) or "").strip()
        for key in (
            "model_name",
            "strategy_profile_id",
            "symbol",
            "side",
            "decision_lane",
        )
    )
