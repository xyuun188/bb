from __future__ import annotations

from typing import Any

from services.phase3_go_no_go import evaluate_phase3_go_no_go_cards


def _card(key: str, status: str, details: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": key,
        "title": key,
        "status": status,
        "summary": f"{key} {status}",
        "details": details,
    }


def _base_cards(
    *,
    can_resume_paper: bool = True,
    preflight_status: str | None = None,
    preflight_blockers: list[Any] | None = None,
    observation_status: str = "waiting_for_resume",
    observation_can_promote: bool = False,
    promotion_canary_ready: bool = False,
    specialist_promotion_ready_count: int = 1,
    server_blocked: bool = False,
    model_blocked: bool = False,
    platform_model_endpoints_ready: bool = False,
    profit_first_plan_missing_count: int = 0,
    profit_first_plan_incomplete_count: int = 0,
    historical_recovery_quarantined_violation_count: int = 0,
    shadow_lane_executed_count: int = 0,
    profit_first_position_ladder_missing_count: int = 0,
    exit_plan_reference_missing_count: int = 0,
    low_payoff_meaningful_size_count: int = 0,
    probe_loss_brake_bypassed_count: int = 0,
    profit_first_policy_active: bool = True,
    ranking_ready: bool = True,
    ranking_demote_count: int = 0,
    ranking_disable_count: int = 0,
    ranking_live_mutation: bool = False,
    governance_status: str = "ready",
    governance_live_mutation: bool = False,
    governance_missing_brain_outputs: list[str] | None = None,
    recovery_blocking_item_count: int = 0,
) -> list[dict[str, Any]]:
    cards = [
        _card(
            "phase3_server_migration",
            "ok",
            {
                "status": "ready",
                "phase3_go_live_blocked": server_blocked,
                "blockers": ["legacy_service"] if server_blocked else [],
            },
        ),
        _card(
            "phase3_model_server_readiness",
            "ok",
            {
                "status": "ready",
                "runtime_ready": not model_blocked,
                "phase3_model_service_go_live_blocked": model_blocked,
                "blockers": ["runtime_not_ready"] if model_blocked else [],
            },
        ),
        _card(
            "phase3_paper_resume_preflight",
            preflight_status or ("ok" if can_resume_paper else "warning"),
            {
                "status": "ready" if can_resume_paper else "blocked",
                "can_resume_paper": can_resume_paper,
                "blockers": ([] if can_resume_paper else preflight_blockers or ["okx_difference"]),
                "passed_checks": (
                    ["phase3_model_server_platform_endpoints_ready"]
                    if platform_model_endpoints_ready
                    else []
                ),
            },
        ),
        _card(
            "phase3_paper_resume_observation",
            "warning" if observation_status != "healthy" else "ok",
            {
                "status": observation_status,
                "paper_active": observation_status != "waiting_for_resume",
                "can_use_for_promotion": observation_can_promote,
                "starts_trading_service": False,
                "submits_orders": False,
                "changes_model_routing": False,
                "live_mutation": False,
                "blockers": [],
                "warnings": [],
            },
        ),
        _card(
            "model_training",
            "warning",
            {
                "local_ai_tools": {
                    "promotion_recommendation": {
                        "recommended_stage": "canary" if promotion_canary_ready else "shadow",
                        "canary_ready": promotion_canary_ready,
                        "live_ready": False,
                        "canary_blocking_reasons": (
                            []
                            if promotion_canary_ready
                            else ["paper_observation_not_healthy:waiting_for_resume"]
                        ),
                        "paper_observation_gate": {
                            "required": True,
                            "status": observation_status,
                            "can_use_for_promotion": observation_can_promote,
                        },
                    }
                },
                "specialist_shadow_evaluation": {
                    "available": True,
                    "eligible_shadow_count": 40,
                    "summary": {
                        "promotion_ready_count": specialist_promotion_ready_count,
                        "blocked_count": 0 if specialist_promotion_ready_count else 1,
                        "top_blocked_reasons": (
                            [{"reason": "false_signal_loss_exceeds_floor", "count": 1}]
                            if not specialist_promotion_ready_count
                            else []
                        ),
                    },
                    "models": [
                        {
                            "model": "timesfm_shadow_challenger",
                            "tool": "time_series_prediction",
                            "promotion_ready": bool(specialist_promotion_ready_count),
                            "promotion_blockers": (
                                []
                                if specialist_promotion_ready_count
                                else ["false_signal_loss_exceeds_floor"]
                            ),
                            "tail_loss_count": 12 if not specialist_promotion_ready_count else 0,
                            "worst_realized_return_pct": (
                                -8.7 if not specialist_promotion_ready_count else 0
                            ),
                            "tail_loss_symbols": (
                                [{"symbol": "ACT/USDT", "count": 7}]
                                if not specialist_promotion_ready_count
                                else []
                            ),
                            "worst_samples": (
                                [
                                    {
                                        "shadow_backtest_id": 1996,
                                        "symbol": "ACT/USDT",
                                        "predicted_side": "short",
                                        "actual_best_side": "long",
                                        "actual_return_pct": -8.7,
                                    }
                                ]
                                if not specialist_promotion_ready_count
                                else []
                            ),
                        }
                    ],
                },
            },
        ),
        _card(
            "trade_execution_contract",
            (
                "critical"
                if (
                    profit_first_plan_missing_count
                    or profit_first_plan_incomplete_count
                    or shadow_lane_executed_count
                    or not profit_first_policy_active
                )
                else "ok"
            ),
            {
                "audit_only": True,
                "live_entry_mutation": False,
                "live_exit_mutation": False,
                "can_bypass_risk_controls": False,
                "policy": {
                    "entry_requires_profit_first_trade_plan": profit_first_policy_active,
                    "profit_first_missing_plan_is_hard_violation": True,
                    "profit_first_shadow_lane_cannot_execute": True,
                },
                "summary": {
                    "decision_count": 12,
                    "executed_entry_count": 2,
                    "profit_first_plan_missing_count": profit_first_plan_missing_count,
                    "profit_first_plan_missing_count_unresolved": profit_first_plan_missing_count,
                    "profit_first_plan_incomplete_count": profit_first_plan_incomplete_count,
                    "profit_first_plan_incomplete_count_unresolved": (
                        0
                        if historical_recovery_quarantined_violation_count
                        else profit_first_plan_incomplete_count
                    ),
                    "shadow_lane_executed_count": shadow_lane_executed_count,
                    "shadow_lane_executed_count_unresolved": (
                        0
                        if historical_recovery_quarantined_violation_count
                        else shadow_lane_executed_count
                    ),
                    "profit_first_position_ladder_missing_count": (
                        profit_first_position_ladder_missing_count
                    ),
                    "profit_first_position_ladder_missing_count_unresolved": (
                        profit_first_position_ladder_missing_count
                    ),
                    "exit_plan_reference_missing_count": exit_plan_reference_missing_count,
                    "exit_plan_reference_missing_count_unresolved": (
                        exit_plan_reference_missing_count
                    ),
                    "low_payoff_meaningful_size_count": low_payoff_meaningful_size_count,
                    "low_payoff_meaningful_size_count_unresolved": (
                        low_payoff_meaningful_size_count
                    ),
                    "probe_loss_brake_bypassed_count": probe_loss_brake_bypassed_count,
                    "probe_loss_brake_bypassed_count_unresolved": (
                        probe_loss_brake_bypassed_count
                    ),
                    "historical_recovery_quarantined_violation_count": (
                        historical_recovery_quarantined_violation_count
                    ),
                    "historical_recovery_quarantine_unresolved_count": (
                        0
                        if historical_recovery_quarantined_violation_count
                        else (
                            profit_first_plan_missing_count
                            + profit_first_plan_incomplete_count
                            + shadow_lane_executed_count
                            + profit_first_position_ladder_missing_count
                            + exit_plan_reference_missing_count
                            + low_payoff_meaningful_size_count
                            + probe_loss_brake_bypassed_count
                        )
                    ),
                    "profit_first_plan_derived_count": 0,
                    "contract_violation_count": (
                        profit_first_plan_missing_count
                        + profit_first_plan_incomplete_count
                        + shadow_lane_executed_count
                        + profit_first_position_ladder_missing_count
                        + exit_plan_reference_missing_count
                        + low_payoff_meaningful_size_count
                        + probe_loss_brake_bypassed_count
                    ),
                },
                "current_summary": {
                    "decision_count": 12,
                    "executed_entry_count": 2,
                    "profit_first_plan_missing_count": profit_first_plan_missing_count,
                    "profit_first_plan_missing_count_unresolved": profit_first_plan_missing_count,
                    "profit_first_plan_incomplete_count": profit_first_plan_incomplete_count,
                    "profit_first_plan_incomplete_count_unresolved": (
                        0
                        if historical_recovery_quarantined_violation_count
                        else profit_first_plan_incomplete_count
                    ),
                    "shadow_lane_executed_count": shadow_lane_executed_count,
                    "shadow_lane_executed_count_unresolved": (
                        0
                        if historical_recovery_quarantined_violation_count
                        else shadow_lane_executed_count
                    ),
                    "profit_first_position_ladder_missing_count": (
                        profit_first_position_ladder_missing_count
                    ),
                    "profit_first_position_ladder_missing_count_unresolved": (
                        profit_first_position_ladder_missing_count
                    ),
                    "exit_plan_reference_missing_count": exit_plan_reference_missing_count,
                    "exit_plan_reference_missing_count_unresolved": (
                        exit_plan_reference_missing_count
                    ),
                    "low_payoff_meaningful_size_count": low_payoff_meaningful_size_count,
                    "low_payoff_meaningful_size_count_unresolved": (
                        low_payoff_meaningful_size_count
                    ),
                    "probe_loss_brake_bypassed_count": probe_loss_brake_bypassed_count,
                    "probe_loss_brake_bypassed_count_unresolved": (
                        probe_loss_brake_bypassed_count
                    ),
                    "historical_recovery_quarantined_violation_count": (
                        historical_recovery_quarantined_violation_count
                    ),
                    "historical_recovery_quarantine_unresolved_count": (
                        0
                        if historical_recovery_quarantined_violation_count
                        else (
                            profit_first_plan_missing_count
                            + profit_first_plan_incomplete_count
                            + shadow_lane_executed_count
                            + profit_first_position_ladder_missing_count
                            + exit_plan_reference_missing_count
                            + low_payoff_meaningful_size_count
                            + probe_loss_brake_bypassed_count
                        )
                    ),
                    "profit_first_plan_derived_count": 0,
                    "contract_violation_count": (
                        profit_first_plan_missing_count
                        + profit_first_plan_incomplete_count
                        + shadow_lane_executed_count
                        + profit_first_position_ladder_missing_count
                        + exit_plan_reference_missing_count
                        + low_payoff_meaningful_size_count
                        + probe_loss_brake_bypassed_count
                    ),
                },
                "violation_reason_counts": {},
            },
        ),
        _card(
            "profit_first_ranking",
            "critical" if ranking_disable_count else "ok" if ranking_ready else "warning",
            {
                "status": "ready" if ranking_ready else "collecting_evidence",
                "audit_only": True,
                "read_only": True,
                "live_mutation": ranking_live_mutation,
                "live_weight_mutation": False,
                "live_sizing_mutation": False,
                "can_change_model_routing": False,
                "can_change_strategy_weight": False,
                "can_increase_live_size": False,
                "ranking_ready": ranking_ready,
                "summary": {
                    "closed_position_count": 24 if ranking_ready else 0,
                    "leaderboard_row_count": 4 if ranking_ready else 0,
                    "promote_candidate_count": 1 if ranking_ready else 0,
                    "demote_count": ranking_demote_count,
                    "disable_count": ranking_disable_count,
                    "blocker_count": ranking_disable_count + ranking_demote_count,
                },
                "blockers": (
                    [
                        {
                            "code": "strategy_disable",
                            "severity": "blocking",
                            "evidence": {"strategy_profile_id": "losing_profile"},
                        }
                    ]
                    if ranking_disable_count
                    else []
                ),
                "brain_recommendations": {
                    "brain_output_coverage": {
                        "source_weights": True,
                        "strategy_weights": True,
                        "lane_threshold_recommendations": True,
                        "size_promotion_demotion": True,
                        "no_entry_threshold_recommendations": True,
                        "exit_policy_adjustments": True,
                        "shadow_canary_live_decisions": True,
                    }
                },
            },
        ),
        _card(
            "profit_first_governance",
            "ok" if governance_status == "ready" else "warning",
            {
                "status": governance_status,
                "audit_only": True,
                "read_only": True,
                "live_mutation": governance_live_mutation,
                "live_entry_mutation": False,
                "live_exit_mutation": False,
                "live_weight_mutation": False,
                "live_sizing_mutation": False,
                "can_submit_orders": False,
                "can_start_trading_service": False,
                "can_change_model_routing": False,
                "can_change_strategy_weight": False,
                "can_increase_live_size": False,
                "missing_brain_outputs": governance_missing_brain_outputs or [],
                "summary": {
                    "ranking_ready": ranking_ready,
                    "no_entry_sample_count": 6,
                    "losing_exit_sample_count": 2,
                    "no_entry_diagnosis": "mixed_blockers_review_top_reasons",
                    "missing_brain_output_count": len(governance_missing_brain_outputs or []),
                },
            },
        ),
    ]
    if recovery_blocking_item_count:
        cards.append(
            _card(
                "profit_first_recovery_blockers",
                "critical",
                {
                    "status": "blocked",
                    "resume_clear": False,
                    "blocking_item_count": recovery_blocking_item_count,
                    "summary": {
                        "contract_blocker_count": recovery_blocking_item_count,
                        "ranking_blocker_count": 0,
                        "okx_blocker_count": 0,
                    },
                    "items": [
                        {
                            "category": "trade_contract",
                            "severity": "blocking",
                            "code": "missing_profit_first_trade_plan",
                        }
                    ],
                },
            )
        )
    return cards


def test_phase3_go_no_go_allows_only_controlled_paper_resume_when_preflight_ready() -> None:
    report = evaluate_phase3_go_no_go_cards(_base_cards())

    assert report["status"] == "paper_resume_ready"
    assert report["next_step"] == "resume_paper_pending_operator_approval"
    assert report["can_start_paper_with_operator_approval"] is True
    assert report["can_enter_canary_with_operator_approval"] is False
    assert report["starts_trading_service"] is False
    assert "paper_observation_waiting_for_resume" in {item["code"] for item in report["warnings"]}


def test_phase3_go_no_go_allows_operator_canary_review_after_healthy_observation() -> None:
    report = evaluate_phase3_go_no_go_cards(
        _base_cards(
            observation_status="healthy",
            observation_can_promote=True,
            promotion_canary_ready=True,
        )
    )

    assert report["status"] == "paper_observation_healthy"
    assert report["next_step"] == "operator_review_for_canary"
    assert report["can_start_paper_with_operator_approval"] is False
    assert report["can_enter_canary_with_operator_approval"] is True
    assert report["can_enter_live"] is False
    assert "paper_observation_healthy" in report["passed_checks"]
    assert "specialist_shadow_has_promotion_ready_model" in report["passed_checks"]


def test_phase3_go_no_go_blocks_canary_when_specialist_models_are_not_ready() -> None:
    report = evaluate_phase3_go_no_go_cards(
        _base_cards(
            observation_status="healthy",
            observation_can_promote=True,
            promotion_canary_ready=True,
            specialist_promotion_ready_count=0,
        )
    )

    assert report["status"] == "paper_observation_healthy"
    assert report["next_step"] == "stay_shadow_improve_specialists"
    assert report["can_enter_canary_with_operator_approval"] is False
    assert "specialist_shadow_no_promotion_ready_model" not in {
        item["code"] for item in report["blockers"]
    }
    assert "specialist_shadow_no_promotion_ready_model" in {
        item["code"] for item in report["warnings"]
    }
    warning = next(
        item
        for item in report["warnings"]
        if item["code"] == "specialist_shadow_no_promotion_ready_model"
    )
    assert warning["evidence"]["tail_risk_models"][0]["model"] == "timesfm_shadow_challenger"
    assert (
        warning["evidence"]["tail_risk_models"][0]["tail_loss_symbols"][0]["symbol"] == "ACT/USDT"
    )
    assert (
        warning["evidence"]["tail_risk_models"][0]["worst_samples"][0]["shadow_backtest_id"] == 1996
    )
    assert report["inputs"]["specialist_promotion_ready_count"] == 0
    assert report["inputs"]["specialist_canary_blocked"] is True
    assert report["inputs"]["specialist_tail_loss_total"] == 12
    assert report["inputs"]["raw_model_promotion_canary_ready"] is True
    assert report["inputs"]["promotion_canary_ready"] is False


def test_phase3_go_no_go_keeps_paper_resume_available_while_specialists_learn() -> None:
    report = evaluate_phase3_go_no_go_cards(_base_cards(specialist_promotion_ready_count=0))

    assert report["status"] == "paper_resume_ready"
    assert report["can_start_paper_with_operator_approval"] is True
    assert report["can_enter_canary_with_operator_approval"] is False
    assert "specialist_shadow_no_promotion_ready_model" not in {
        item["code"] for item in report["blockers"]
    }
    assert "specialist_shadow_no_promotion_ready_model" in {
        item["code"] for item in report["warnings"]
    }


def test_phase3_go_no_go_blocks_when_foundation_gates_block() -> None:
    report = evaluate_phase3_go_no_go_cards(
        _base_cards(server_blocked=True, model_blocked=True, can_resume_paper=False)
    )

    assert report["status"] == "blocked"
    assert report["next_step"] == "stay_shadow_fix_blockers"
    assert report["can_start_paper_with_operator_approval"] is False
    codes = {item["code"] for item in report["blockers"]}
    assert "server_migration_go_live_blocked" in codes
    assert "model_server_go_live_blocked" in codes
    assert "paper_resume_preflight_not_ready" in codes


def test_phase3_go_no_go_blocks_resume_when_profit_first_plan_violates_contract() -> None:
    report = evaluate_phase3_go_no_go_cards(
        _base_cards(
            profit_first_plan_missing_count=1,
            profit_first_plan_incomplete_count=1,
            shadow_lane_executed_count=1,
        )
    )

    assert report["status"] == "blocked"
    assert report["can_start_paper_with_operator_approval"] is False
    blocker_codes = {item["code"] for item in report["blockers"]}
    assert "profit_first_trade_contract_critical" in blocker_codes
    assert "profit_first_trade_plan_current_window_violations" in blocker_codes
    assert "profit_first_trade_plan_policy_active" in report["passed_checks"]


def test_phase3_go_no_go_allows_resume_when_only_historical_recovery_quarantine_remains() -> None:
    report = evaluate_phase3_go_no_go_cards(
        _base_cards(
            profit_first_plan_incomplete_count=3,
            shadow_lane_executed_count=3,
            historical_recovery_quarantined_violation_count=6,
        )
    )

    assert report["status"] == "paper_resume_ready"
    assert report["can_start_paper_with_operator_approval"] is True
    blocker_codes = {item["code"] for item in report["blockers"]}
    warning_codes = {item["code"] for item in report["warnings"]}
    assert "profit_first_trade_contract_critical" not in blocker_codes
    assert "profit_first_trade_plan_current_window_violations" not in blocker_codes
    assert "trade_execution_contract_historical_quarantined" in warning_codes
    assert "profit_first_trade_plan_current_window_clean" in report["passed_checks"]


def test_phase3_go_no_go_blocks_when_trade_contract_audit_unavailable() -> None:
    cards = _base_cards()
    for card in cards:
        if card["key"] == "trade_execution_contract":
            card["status"] = "warning"
            card["details"] = {
                "report_available": False,
                "audit_only": True,
                "live_entry_mutation": False,
                "live_exit_mutation": False,
                "can_bypass_risk_controls": False,
                "error": "database unavailable",
                "policy": {
                    "entry_requires_profit_first_trade_plan": True,
                    "profit_first_probe_loss_brake_must_block_execution": True,
                },
                "summary": {"report_available": False, "contract_violation_count": 0},
            }

    report = evaluate_phase3_go_no_go_cards(cards)

    blocker_codes = {item["code"] for item in report["blockers"]}
    assert report["status"] == "blocked"
    assert "profit_first_trade_contract_unavailable" in blocker_codes
    assert "profit_first_trade_plan_policy_missing" not in blocker_codes


def test_phase3_go_no_go_blocks_resume_when_ladder_or_exit_binding_violates_contract() -> None:
    report = evaluate_phase3_go_no_go_cards(
        _base_cards(
            profit_first_position_ladder_missing_count=1,
            exit_plan_reference_missing_count=1,
            low_payoff_meaningful_size_count=1,
        )
    )

    assert report["status"] == "blocked"
    assert report["can_start_paper_with_operator_approval"] is False
    blocker = next(
        item
        for item in report["blockers"]
        if item["code"] == "profit_first_trade_plan_current_window_violations"
    )
    assert blocker["evidence"]["profit_first_position_ladder_missing_count"] == 1
    assert blocker["evidence"]["exit_plan_reference_missing_count"] == 1
    assert blocker["evidence"]["low_payoff_meaningful_size_count"] == 1


def test_phase3_go_no_go_blocks_resume_when_probe_loss_brake_was_bypassed() -> None:
    report = evaluate_phase3_go_no_go_cards(
        _base_cards(probe_loss_brake_bypassed_count=1)
    )

    assert report["status"] == "blocked"
    assert report["can_start_paper_with_operator_approval"] is False
    blocker = next(
        item
        for item in report["blockers"]
        if item["code"] == "profit_first_trade_plan_current_window_violations"
    )
    assert blocker["evidence"]["probe_loss_brake_bypassed_count"] == 1


def test_phase3_go_no_go_blocks_resume_when_profit_first_ranking_not_ready() -> None:
    report = evaluate_phase3_go_no_go_cards(_base_cards(ranking_ready=False))

    assert report["status"] == "blocked"
    assert report["can_start_paper_with_operator_approval"] is False
    blocker_codes = {item["code"] for item in report["blockers"]}
    assert "profit_first_ranking_not_ready" in blocker_codes
    assert report["inputs"]["profit_first_ranking_ready"] is False


def test_phase3_go_no_go_blocks_resume_when_profit_first_ranking_audit_unavailable() -> None:
    cards = _base_cards()
    for card in cards:
        if card["key"] == "profit_first_ranking":
            card["status"] = "warning"
            card["details"] = {
                "report_available": False,
                "status": "unavailable",
                "audit_only": True,
                "read_only": True,
                "live_mutation": False,
                "live_weight_mutation": False,
                "live_sizing_mutation": False,
                "can_change_model_routing": False,
                "can_change_strategy_weight": False,
                "can_increase_live_size": False,
                "ranking_ready": False,
                "error": "database unavailable",
                "summary": {"report_available": False},
            }

    report = evaluate_phase3_go_no_go_cards(cards)

    blocker_codes = {item["code"] for item in report["blockers"]}
    assert report["status"] == "blocked"
    assert "profit_first_ranking_unavailable" in blocker_codes
    assert "profit_first_ranking_not_ready" not in blocker_codes
    assert report["inputs"]["profit_first_ranking_ready"] is False


def test_phase3_go_no_go_blocks_resume_when_profit_first_ranking_disables_profile() -> None:
    report = evaluate_phase3_go_no_go_cards(
        _base_cards(ranking_disable_count=1, ranking_demote_count=1)
    )

    assert report["status"] == "blocked"
    assert report["can_start_paper_with_operator_approval"] is False
    blocker_codes = {item["code"] for item in report["blockers"]}
    assert "profit_first_ranking_has_disable_blockers" in blocker_codes
    assert report["inputs"]["profit_first_ranking_disable_count"] == 1
    assert report["inputs"]["profit_first_ranking_demote_count"] == 1


def test_phase3_go_no_go_surfaces_recovery_blocker_checklist() -> None:
    report = evaluate_phase3_go_no_go_cards(_base_cards(recovery_blocking_item_count=3))

    assert report["status"] == "blocked"
    blocker_codes = {item["code"] for item in report["blockers"]}
    assert "profit_first_recovery_blockers_not_clear" in blocker_codes
    assert report["inputs"]["profit_first_recovery_blocking_item_count"] == 3
    assert report["inputs"]["profit_first_recovery_resume_clear"] is False


def test_phase3_go_no_go_blocks_when_profit_first_ranking_can_mutate_live() -> None:
    report = evaluate_phase3_go_no_go_cards(_base_cards(ranking_live_mutation=True))

    assert report["status"] == "blocked"
    blocker_codes = {item["code"] for item in report["blockers"]}
    assert "profit_first_ranking_not_read_only" in blocker_codes


def test_phase3_go_no_go_blocks_when_profit_first_brain_outputs_are_incomplete() -> None:
    cards = _base_cards()
    for card in cards:
        if card["key"] == "profit_first_ranking":
            card["details"]["brain_recommendations"] = {
                "brain_output_coverage": {
                    "source_weights": True,
                    "strategy_weights": True,
                }
            }

    report = evaluate_phase3_go_no_go_cards(cards)

    blocker_codes = {item["code"] for item in report["blockers"]}
    assert report["status"] == "blocked"
    assert "profit_first_brain_output_coverage_missing" in blocker_codes
    blocker = next(
        item
        for item in report["blockers"]
        if item["code"] == "profit_first_brain_output_coverage_missing"
    )
    assert "exit_policy_adjustments" in blocker["evidence"]["missing_outputs"]


def test_phase3_go_no_go_blocks_when_profit_first_governance_unavailable() -> None:
    cards = _base_cards(governance_status="unavailable")
    for card in cards:
        if card["key"] == "profit_first_governance":
            card["details"]["report_available"] = False
            card["details"]["error"] = "database unavailable"

    report = evaluate_phase3_go_no_go_cards(cards)

    blocker_codes = {item["code"] for item in report["blockers"]}
    assert report["status"] == "blocked"
    assert "profit_first_governance_unavailable" in blocker_codes
    assert report["inputs"]["profit_first_governance_status"] == "unavailable"


def test_phase3_go_no_go_blocks_when_profit_first_governance_can_mutate_live() -> None:
    report = evaluate_phase3_go_no_go_cards(
        _base_cards(governance_live_mutation=True)
    )

    blocker_codes = {item["code"] for item in report["blockers"]}
    assert report["status"] == "blocked"
    assert "profit_first_governance_not_read_only" in blocker_codes


def test_phase3_go_no_go_blocks_when_profit_first_governance_brain_outputs_missing() -> None:
    report = evaluate_phase3_go_no_go_cards(
        _base_cards(governance_missing_brain_outputs=["exit_policy_adjustments"])
    )

    blocker_codes = {item["code"] for item in report["blockers"]}
    assert report["status"] == "blocked"
    assert "profit_first_governance_brain_outputs_missing" in blocker_codes


def test_phase3_go_no_go_observes_after_controlled_paper_start() -> None:
    report = evaluate_phase3_go_no_go_cards(
        _base_cards(
            can_resume_paper=False,
            preflight_status="critical",
            preflight_blockers=[
                {
                    "code": "paper_trading_already_active",
                    "message": "Preflight must run before paper starts.",
                }
            ],
            observation_status="warming_up",
        )
    )

    assert report["status"] == "post_resume_observing"
    assert report["next_step"] == "continue_post_resume_observation"
    assert report["can_start_paper_with_operator_approval"] is False
    assert report["can_enter_canary_with_operator_approval"] is False
    assert report["can_enter_live"] is False
    assert report["blockers"] == []
    assert "paper_resume_preflight_consumed_after_resume" in report["passed_checks"]
    assert report["inputs"]["preflight_consumed_after_resume"] is True


def test_phase3_go_no_go_does_not_block_on_observing_model_training_card() -> None:
    cards = _base_cards(
        can_resume_paper=False,
        preflight_status="critical",
        preflight_blockers=[
            {
                "code": "paper_trading_already_active",
                "message": "Preflight must run before paper starts.",
            }
        ],
        observation_status="healthy",
        observation_can_promote=True,
    )
    for card in cards:
        if card["key"] == "model_training":
            card["status"] = "critical"
            card["details"]["hard_failure"] = False
            card["details"]["observing"] = True

    report = evaluate_phase3_go_no_go_cards(cards)

    assert report["status"] == "paper_observation_healthy"
    assert "model_training_available" in report["passed_checks"]
    assert "model_training_critical" not in {item["code"] for item in report["blockers"]}
    assert "model_training_observing_not_hard_failure" in {
        item["code"] for item in report["warnings"]
    }


def test_phase3_go_no_go_observes_with_platform_model_endpoints_when_remote_audits_unverified() -> (
    None
):
    cards = _base_cards(
        can_resume_paper=False,
        preflight_status="critical",
        preflight_blockers=[{"code": "paper_trading_already_active"}],
        observation_status="warming_up",
        server_blocked=True,
        model_blocked=True,
        platform_model_endpoints_ready=True,
    )
    for card in cards:
        if card["key"] in {"phase3_server_migration", "phase3_model_server_readiness"}:
            card["status"] = "critical"
            card["summary"] = "remote audit unavailable"
            card["details"]["status"] = "unverified"
            card["details"]["blockers"] = [{"code": "model_server_config_error"}]

    report = evaluate_phase3_go_no_go_cards(cards)

    blocker_codes = {item["code"] for item in report["blockers"]}
    warning_codes = {item["code"] for item in report["warnings"]}
    assert report["status"] == "post_resume_observing"
    assert report["next_step"] == "continue_post_resume_observation"
    assert "server_migration_go_live_blocked" not in blocker_codes
    assert "model_server_go_live_blocked" not in blocker_codes
    assert "phase3_server_migration_remote_audit_unverified" in warning_codes
    assert "phase3_model_server_readiness_remote_audit_unverified" in warning_codes
    assert "server_migration_remote_audit_unverified" in warning_codes
    assert "model_server_remote_audit_unverified" in warning_codes
    assert "model_server_platform_endpoint_verified" in report["passed_checks"]
