from scripts import inspect_online_strategy_health


def test_strategy_health_remote_template_is_valid_python() -> None:
    template = inspect_online_strategy_health.REMOTE_SCRIPT_TEMPLATE.replace(
        "__WINDOW_MINUTES__", "120"
    ).replace(
        "__SUMMARY_ONLY__",
        "False",
    )

    compile(template, "<inspect_online_strategy_health_remote_template>", "exec")


def test_strategy_health_remote_command_uses_unique_temp_files() -> None:
    command = inspect_online_strategy_health._build_remote_command(120, token="abc123")

    assert "/data/bb/app/tmp/codex-strategy-health/sample_120_abc123.py" in command
    assert "/data/bb/app/tmp/codex-strategy-health/launcher_120_abc123.py" in command
    assert "mkdir -p /data/bb/app/tmp/codex-strategy-health" in command
    assert "chmod 0750 /data/bb/app/tmp/codex-strategy-health" in command
    assert "codex_strategy_sample.py" not in command
    assert "codex_strategy_launcher.py" not in command
    assert "__WINDOW_MINUTES__" not in command


def test_strategy_health_remote_command_can_write_result_file() -> None:
    result_path = inspect_online_strategy_health._remote_result_path(120, "abc123")
    command = inspect_online_strategy_health._build_remote_command(
        120,
        token="abc123",
        output_path=result_path,
    )

    assert result_path == "/data/bb/app/tmp/codex-strategy-health/result_120_abc123.json"
    assert (
        f'python3 /data/bb/app/tmp/codex-strategy-health/launcher_120_abc123.py /data/bb/app/tmp/codex-strategy-health/sample_120_abc123.py > "{result_path}"'
        in command
    )
    assert f'chmod 0640 "{result_path}"' in command
    assert f"printf '%s\\n' \"{result_path}\"" in command
    assert "rm -f /data/bb/app/tmp/codex-strategy-health/sample_120_abc123.py" in command


def test_strategy_health_remote_command_can_emit_summary_only() -> None:
    command = inspect_online_strategy_health._build_remote_command(
        120, token="abc123", summary=True
    )

    assert "SUMMARY_ONLY = True" in command
    assert "__SUMMARY_ONLY__" not in command
    assert "MARKET_SYMBOL_ONLY = False" in command
    assert "__MARKET_SYMBOL_ONLY__" not in command
    assert "ENTRY_ONLY = False" in command
    assert "__ENTRY_ONLY__" not in command
    assert "elif SUMMARY_ONLY:" in command
    assert "output = summary_report(report)" in command
    assert "json.loads(out)" not in command


def test_strategy_health_summary_report_uses_compact_market_diagnostics() -> None:
    report = {
        "window_minutes": 20,
        "generated_at": "2026-06-23T18:40:00+00:00",
        "counts": {"decisions": 10, "orders": 0, "open_positions": 5},
        "trade_execution_contract": {
            "status": "ok",
            "audit_only": True,
            "can_bypass_risk_controls": False,
            "summary": {"executed_entry_count": 0},
            "fast_loss_samples": [{"id": i} for i in range(10)],
            "violations": [{"id": i} for i in range(10)],
        },
        "local_ml_readiness": {
            "status": "degraded",
            "readiness_state": "degraded",
            "allow_live_position_influence": False,
            "blocking_reason_codes": ["long_pr_auc_below_threshold"],
            "metrics": {"sample_count": 6994},
        },
        "market_symbol_diagnostics": {
            "market_decision_count": 18,
            "market_unique_symbol_count": 14,
            "market_top_symbols": [{"symbol": f"S{i}/USDT", "count": i} for i in range(20)],
            "candidate_funnel_sample_count": 18,
            "candidate_funnel_window": {
                "count": 18,
                "metric_stats": {
                    "scan_symbol_count": {
                        "count": 18,
                        "median": 120,
                        "p75": 120,
                        "max": 120,
                        "positive": 18,
                        "zero": 0,
                        "min": 120,
                    }
                },
                "filtered_symbol_counts": [{"symbol": f"F{i}/USDT", "count": i} for i in range(20)],
            },
            "market_analysis_progress": {
                "read_only": True,
                "count": 3,
                "processed_index_stats": {"count": 3, "median": 1, "max": 2},
                "ranked_market_symbol_count_stats": {"count": 3, "median": 8},
                "full_round_elapsed_before_ai_stats": {"count": 3, "median": 45.0},
                "market_ai_elapsed_before_symbol_stats": {"count": 3, "median": 8.5},
                "budget_used_ratio_before_ai_stats": {"count": 3, "median": 0.72},
                "market_ai_budget_used_ratio_before_symbol_stats": {
                    "count": 3,
                    "median": 0.31,
                },
                "latest": {
                    "symbol": "BZ/USDT",
                    "processed_index": 0,
                    "ranked_market_symbol_count": 8,
                    "market_ai_elapsed_seconds_before_symbol": 8.5,
                    "full_round_elapsed_seconds_before_ai": 45.0,
                    "budget_clock_scope": "market_ai_phase",
                    "large_unused_payload": "kept because aggregate is already compact",
                },
            },
            "candidate_filter_outcomes": {
                "sampled_symbol_count": 5,
                "market_entry_after_filter_count": 0,
                "reason_counts": [{"value": f"r{i}", "count": i} for i in range(20)],
                "symbol_examples": [{"symbol": f"E{i}/USDT"} for i in range(20)],
            },
            "latest_candidate_funnel": {
                "ranked_symbol_sample": [{"symbol": f"L{i}/USDT"} for i in range(20)],
                "market_budget_rotation": {
                    "read_only": True,
                    "is_entry_gate": False,
                    "applied": True,
                    "start_symbol": "L3/USDT",
                },
            },
        },
        "closed_position_pnl_diagnostics": {
            "read_only": True,
            "closed_count": 8,
            "win_count": 3,
            "loss_count": 5,
            "flat_count": 0,
            "win_rate": 0.375,
            "total_realized_pnl": -2.45,
            "avg_realized_pnl": -0.30625,
            "profit_factor": 0.42,
            "samples": [{"id": i, "symbol": f"P{i}/USDT"} for i in range(10)],
        },
        "rejected_order_examples": [{"id": i} for i in range(10)],
        "fast_loss_positions": [{"id": i} for i in range(10)],
    }

    compact = inspect_online_strategy_health._summarize_report(report)
    diagnostics = compact["market_symbol_diagnostics"]

    assert len(compact["trade_execution_contract"]["fast_loss_samples"]) == 5
    assert len(compact["trade_execution_contract"]["violations"]) == 5
    assert len(compact["rejected_order_examples"]) == 5
    assert len(compact["fast_loss_positions"]) == 5
    assert compact["closed_position_pnl_diagnostics"]["closed_count"] == 8
    assert compact["closed_position_pnl_diagnostics"]["total_realized_pnl"] == -2.45
    assert len(compact["closed_position_pnl_diagnostics"]["samples"]) == 5
    assert len(diagnostics["market_top_symbols"]) == 8
    assert "latest_candidate_funnel" not in diagnostics
    assert (
        diagnostics["candidate_funnel_window"]["metric_stats"]["scan_symbol_count"]["median"] == 120
    )
    assert "min" not in diagnostics["candidate_funnel_window"]["metric_stats"]["scan_symbol_count"]
    assert len(diagnostics["candidate_funnel_window"]["filtered_symbol_counts"]) == 6
    assert diagnostics["market_analysis_progress"]["read_only"] is True
    assert diagnostics["market_analysis_progress"]["count"] == 3
    assert diagnostics["market_analysis_progress"]["latest"]["processed_index"] == 0
    assert (
        diagnostics["market_analysis_progress"]["latest"]["budget_clock_scope"] == "market_ai_phase"
    )
    assert (
        diagnostics["market_analysis_progress"]["market_ai_elapsed_before_symbol_stats"]["median"]
        == 8.5
    )
    assert len(diagnostics["candidate_filter_outcomes"]["reason_counts"]) == 6
    assert len(diagnostics["candidate_filter_outcomes"]["symbol_examples"]) == 2


def test_strategy_health_remote_command_can_emit_market_symbol_only() -> None:
    command = inspect_online_strategy_health._build_remote_command(
        120, token="abc123", market_symbol_only=True
    )

    assert "SUMMARY_ONLY = False" in command
    assert "MARKET_SYMBOL_ONLY = True" in command
    assert "ENTRY_ONLY = False" in command
    assert "__SUMMARY_ONLY__" not in command
    assert "__MARKET_SYMBOL_ONLY__" not in command
    assert "__ENTRY_ONLY__" not in command
    assert "if MARKET_SYMBOL_ONLY:" in command
    assert "output = market_symbol_only_report(report)" in command


def test_strategy_health_remote_command_can_emit_entry_only() -> None:
    command = inspect_online_strategy_health._build_remote_command(
        120, token="abc123", entry_only=True
    )

    assert "SUMMARY_ONLY = False" in command
    assert "MARKET_SYMBOL_ONLY = False" in command
    assert "ENTRY_ONLY = True" in command
    assert "__SUMMARY_ONLY__" not in command
    assert "__MARKET_SYMBOL_ONLY__" not in command
    assert "__ENTRY_ONLY__" not in command
    assert "if ENTRY_ONLY:" in command
    assert "output = entry_only_report(report)" in command


def test_strategy_health_entry_only_report_is_compact_and_keeps_guards() -> None:
    report = {
        "window_minutes": 120,
        "generated_at": "2026-06-23T14:10:00+00:00",
        "counts": {
            "decisions": 393,
            "market_decisions": 121,
            "entry_decisions": 7,
            "market_entry_decisions": 7,
            "executed_entries": 1,
            "orders": 2,
            "filled_orders": 2,
            "failed_orders": 0,
            "rejected_orders": 0,
            "positions_created": 1,
            "positions_closed": 1,
            "open_positions": 6,
            "fast_loss_close_under_15m": 0,
        },
        "trade_execution_contract": {
            "status": "ok",
            "can_bypass_risk_controls": False,
            "summary": {
                "decision_count": 393,
                "executed_entry_count": 1,
                "contract_violation_count": 0,
                "weak_evidence_executed_count": 0,
                "negative_expected_executed_count": 0,
                "fast_loss_count": 0,
                "fast_loss_without_strong_exit_count": 0,
                "reentry_without_strong_unlock_count": 0,
            },
        },
        "local_ml_readiness": {
            "status": "degraded",
            "readiness_state": "degraded",
            "allow_live_position_influence": False,
            "blocking_reason_codes": ["dirty_sample_ratio_high"],
            "metrics": {
                "sample_count": 19971,
                "test_count": 4993,
                "dirty_sample_ratio": 0.7571,
                "long_pr_auc": 0.333,
                "short_pr_auc": 0.364,
                "top_long_avg_return_pct": -0.058,
                "top_short_avg_return_pct": -0.076,
                "training_data_version": "2026-06-23.v3",
                "required_training_data_version": "2026-06-23.v3",
                "large_unused_metric": "omit",
            },
            "training_window_composition": {
                "sample_count": 19971,
                "decision_action_counts": {"hold": 12000, "long": 3000, "short": 4971},
                "data_quality_status_counts": {"included": 4800, "downweighted": 15171},
            },
        },
        "closed_position_pnl_diagnostics": {
            "read_only": True,
            "closed_count": 6,
            "win_count": 2,
            "loss_count": 4,
            "flat_count": 0,
            "win_rate": 0.333333,
            "total_realized_pnl": -1.72,
            "avg_realized_pnl": -0.286667,
            "profit_factor": 0.51,
            "fast_loss_close_under_15m": 1,
            "samples": [{"id": i, "symbol": f"C{i}/USDT"} for i in range(8)],
        },
        "entry_evidence_thresholds": {"weak_probe": 40, "exploration": 55},
        "market_entry_final_skip_kind_counts": {
            "entry_evidence_shadow_only": 3,
            "executed": 1,
        },
        "market_entry_evidence_tier_counts": {"weak_conflict_probe": 4, "exploration": 1},
        "market_entry_evidence_component_status_counts": {"ml_signal:degraded": 4},
        "market_entry_probe_recommendation_counts": {"memory_watchlist_needs_probe_threshold": 3},
        "market_entry_probe_conversion_ready_counts": {"not_ready": 3},
        "market_entry_probe_conversion_block_reason_counts": {
            "expected_net_below_probe_threshold": 3
        },
        "market_entry_profit_probe_block_kind_counts": {"probe_threshold_not_met": 2},
        "market_entry_profit_probe_block_reason_counts": {"expected_net_below_probe_threshold": 2},
        "market_entry_expected_net_component_stats": {"shadow_memory": {"count": 2}},
        "position_size_pct_stats": {"count": 7, "median": 0.01},
        "quality_tier_counts": {"probe": 5, "base": 2},
        "low_payoff_entry_count": 4,
        "low_payoff_reason_counts": {
            "expected_net_below_min": 3,
            "evidence_low_payoff_quality": 2,
        },
        "low_payoff_missing_reason_count": 1,
        "entry_analysis_type_counts": {"entry_candidate": 5, "market": 2},
        "entry_analysis_type_skip_kind_counts": {
            "entry_candidate:executed": 1,
            "entry_candidate:high_risk_review_blocked": 1,
            "market:entry_evidence_shadow_only": 2,
        },
        "entry_analysis_type_evidence_tier_counts": {
            "entry_candidate:exploration": 4,
            "market:weak_conflict_probe": 2,
        },
        "entry_analysis_type_quality_tier_counts": {
            "entry_candidate:probe": 4,
            "market:unknown": 2,
        },
        "entry_analysis_type_low_payoff_counts": {"entry_candidate": 4},
        "entry_analysis_type_low_payoff_reason_counts": {
            "entry_candidate:expected_net_below_min": 3,
            "entry_candidate:evidence_low_payoff_quality": 2,
        },
        "entry_analysis_type_low_payoff_missing_reason_counts": {
            "entry_candidate": 1,
        },
        "entry_analysis_type_notional_floor_blocked_counts": {
            "entry_candidate:low_payoff_quality_cap": 2,
        },
        "entry_analysis_type_metric_stats": {
            "entry_candidate": {
                "expected_net_return_pct": {"count": 5, "median": 0.71},
                "position_size_pct": {"count": 5, "median": 0.0021},
            },
            "market": {
                "expected_net_return_pct": {"count": 2, "median": -0.01},
            },
        },
        "notional_floor_blocked_counts": {"尾部风险偏高，不抬高仓位": 1},
        "high_risk_review_status_counts": {"completed": 2},
        "high_risk_review_trigger_counts": {"triggered": 2},
        "high_risk_review_approved_counts": {"approved_false": 1, "approved_true": 1},
        "high_risk_review_reason_counts": {"risk asymmetric": 1},
        "high_risk_review_trigger_reason_counts": {
            "today_recovery_after_loss": 1,
            "ml_ai_direction_conflict": 1,
        },
        "executed_entry_sizing_diagnostics": {
            "read_only": True,
            "executed_entry_count": 2,
            "market_executed_entry_count": 1,
            "filled_order_count": 1,
            "missing_order_count": 1,
            "order_status_counts": {"filled": 1, "missing_order": 1},
            "evidence_tier_counts": {"exploration": 1, "normal": 1},
            "sizing_quality_tier_counts": {"probe": 1, "base": 1},
            "sizing_reason_tag_counts": {
                "evidence_tier:exploration": 1,
                "sizing_quality:probe": 1,
                "low_payoff:expected_net_below_min": 1,
                "low_payoff:evidence_low_payoff_quality": 1,
                "strategy_probe_cap_applied": 1,
            },
            "order_notional_stats": {"count": 1, "median": 180},
            "sizing_final_notional_stats": {"count": 2, "median": 180},
            "notional_fill_ratio_stats": {"count": 1, "median": 1.0},
            "decision_position_size_pct_stats": {"count": 2, "median": 0.012},
            "decision_leverage_stats": {"count": 2, "median": 3},
            "expected_net_stats": {"count": 2, "positive": 2},
            "profit_quality_stats": {"count": 2, "median": 1.1},
            "loss_probability_stats": {"count": 2, "median": 0.42},
            "tail_risk_stats": {"count": 2, "median": 0.61},
            "samples": [
                {
                    "id": 1,
                    "time": "2026-06-23T14:00:00+00:00",
                    "symbol": "PLTR/USDT",
                    "action": "long",
                    "analysis_type": "market",
                    "decision": {
                        "position_size_pct": 0.012,
                        "suggested_leverage": 3,
                        "was_executed": True,
                        "executed_at": "2026-06-23T14:00:03+00:00",
                        "execution_price": 180,
                        "large_unused_decision": "omit",
                    },
                    "order": {
                        "id": 501,
                        "status": "filled",
                        "side": "buy",
                        "quantity": 1,
                        "price": 180,
                        "notional": 180,
                        "filled_at": "2026-06-23T14:00:04+00:00",
                        "large_unused_order": "omit",
                    },
                    "evidence": {
                        "tier": "exploration",
                        "effective_score": 56,
                        "entry_evidence_score": 56,
                        "expected_net_return_pct": 0.4,
                        "aggregate_expected_net_return_pct": 0.43,
                        "ai_expected_return_policy": (
                            "probe_original_hold_without_independent_support"
                        ),
                        "ai_expected_return_weight": 0.0,
                        "ai_expected_return_independent_probe_support": [],
                        "profit_quality_ratio": 1.1,
                        "loss_probability": 0.42,
                        "tail_risk_score": 0.61,
                        "tradeable_probe": True,
                        "shadow_only": False,
                        "large_unused_evidence": "omit",
                    },
                    "sizing": {
                        "position_size_pct": 0.012,
                        "leverage": 3,
                        "final_notional_usdt": 180,
                        "quality_tier": "probe",
                        "low_payoff_quality": True,
                        "low_payoff_reasons": [
                            "expected_net_below_min",
                            "evidence_low_payoff_quality",
                        ],
                        "notional_floor_applied": False,
                        "notional_floor_blocked": "",
                        "meaningful_size_reason": "probe-quality cap",
                        "strategy_probe_cap_applied": True,
                        "strategy_max_probe_size_pct": 0.012,
                        "strategy_reason": "recent learning keeps probe cap",
                        "large_unused_sizing": "omit",
                    },
                    "sizing_reason_tags": [
                        "evidence_tier:exploration",
                        "sizing_quality:probe",
                        "low_payoff:expected_net_below_min",
                        "low_payoff:evidence_low_payoff_quality",
                        "strategy_probe_cap_applied",
                    ],
                    "notional_gap_usdt": 0,
                    "notional_fill_ratio": 1.0,
                    "execution_result": {
                        "source": "okx_live",
                        "status": "filled",
                        "exchange_confirmed": True,
                        "okx_symbol": "PLTR-USDT-SWAP",
                        "planned_order_contracts": 1,
                        "planned_base_quantity": 1,
                        "execution_blocker": "",
                        "okx_rejection": False,
                        "system_pre_submit_rejection": False,
                        "large_unused_execution_result": "omit",
                    },
                    "raw_payload": {"must": "omit"},
                }
            ]
            + [{"id": i, "symbol": f"BIG{i}/USDT"} for i in range(20)],
            "diagnostic_boundary": "Read-only executed-entry sizing/order diagnostics",
            "large_unused_payload": {"must": "omit"},
        },
        "entry_examples": [
            {
                "id": 1,
                "time": "2026-06-23T14:00:00+00:00",
                "symbol": "PLTR/USDT",
                "action": "long",
                "analysis_type": "market",
                "executed": True,
                "reason": "executed",
                "state": {"final_stage": "local_sync", "final_status": "completed"},
                "evidence": {
                    "tier": "exploration",
                    "effective_score": 56,
                    "entry_evidence_score": 56,
                    "score": 72,
                    "min_score_required": 68,
                    "expected_net_return_pct": 0.4,
                    "aggregate_expected_net_return_pct": 0.43,
                    "ai_expected_return_policy": (
                        "probe_original_hold_without_independent_support"
                    ),
                    "ai_expected_return_weight": 0.0,
                    "ai_expected_return_independent_probe_support": [],
                    "profit_quality_ratio": 1.1,
                    "loss_probability": 0.42,
                    "tail_risk_score": 0.61,
                    "aligned_support_sources": ["timeseries"],
                    "large_unused_evidence": "omit",
                },
                "probe": {
                    "recommendation": "memory_watchlist_needs_probe_threshold",
                    "probe_conversion_ready": False,
                    "probe_conversion_block_reasons": ["expected_net_below_probe_threshold"],
                    "probe_conversion_thresholds": {"min_expected_net_return_pct": 0.35},
                    "evidence_profit_probe_blocked": {
                        "blocked": True,
                        "block_kind": "probe_threshold_not_met",
                        "block_reasons": ["expected_net_below_probe_threshold"],
                        "reason": "retained for shadow learning",
                        "expected_net_return_pct": 0.31,
                        "profit_quality_ratio": 0.42,
                        "loss_probability": 0.51,
                        "tail_risk_score": 0.37,
                        "thresholds": {"min_expected_net_return_pct": 0.35},
                        "large_unused_probe_block": "omit",
                    },
                    "large_unused_probe": "omit",
                },
                "sizing": {
                    "position_size_pct": 0.012,
                    "leverage": 3,
                    "final_notional_usdt": 180,
                    "quality_tier": "probe",
                    "low_payoff_quality": False,
                    "notional_floor_applied": False,
                    "notional_floor_blocked": "",
                    "large_unused_sizing": "omit",
                },
                "order": {"status": "filled", "quantity": 1, "price": 180, "notional": 180},
            },
            {
                "id": 2,
                "time": "2026-06-23T14:01:00+00:00",
                "symbol": "CRCL/USDT",
                "action": "long",
                "analysis_type": "entry_candidate",
                "executed": False,
                "reason": "high risk review blocked entry candidate",
                "state": {
                    "final_stage": "risk_check",
                    "final_status": "blocked",
                    "blocked": True,
                    "failed": False,
                },
                "evidence": {
                    "tier": "exploration",
                    "effective_score": 52,
                    "entry_evidence_score": 52,
                    "score": 65,
                    "min_score_required": 68,
                    "expected_net_return_pct": 0.71,
                    "profit_quality_ratio": 0.98,
                    "loss_probability": 0.48,
                    "tail_risk_score": 0.24,
                },
                "probe": {},
                "sizing": {
                    "position_size_pct": 0.0021,
                    "leverage": 3,
                    "final_notional_usdt": 30,
                    "quality_tier": "probe",
                    "low_payoff_quality": True,
                    "low_payoff_reasons": ["expected_net_below_min"],
                    "notional_floor_applied": False,
                    "notional_floor_blocked": "low_payoff_quality_cap",
                },
                "high_risk_review": {
                    "triggered": True,
                    "status": "completed",
                    "approved": False,
                    "hard_review_required": True,
                    "reasons": ["today_recovery_after_loss"],
                    "reason": "risk asymmetric",
                    "confidence": 0.76,
                },
                "order": None,
            },
            {
                "id": 3,
                "symbol": "BTC/USDT",
                "analysis_type": "position_review",
                "executed": False,
                "evidence": {"tier": "normal"},
                "sizing": {"position_size_pct": 0.05},
            },
        ],
    }

    compact = inspect_online_strategy_health._entry_only_report(report)

    assert compact["trade_execution_contract"]["status"] == "ok"
    assert compact["trade_execution_contract"]["can_bypass_risk_controls"] is False
    assert compact["closed_position_pnl_diagnostics"]["closed_count"] == 6
    assert compact["closed_position_pnl_diagnostics"]["total_realized_pnl"] == -1.72
    assert len(compact["closed_position_pnl_diagnostics"]["samples"]) == 5
    assert compact["local_ml_readiness"]["allow_live_position_influence"] is False
    assert "large_unused_metric" not in compact["local_ml_readiness"]["metrics"]
    assert (
        compact["local_ml_readiness"]["training_window_composition"]["decision_action_counts"][
            "hold"
        ]
        == 12000
    )
    assert compact["market_entry_final_skip_kind_counts"]["executed"] == 1
    assert len(compact["entry_examples"]) == 2
    example = compact["entry_examples"][0]
    assert example["skip_kind"] == "executed"
    assert example["evidence"]["expected_net_return_pct"] == 0.4
    assert example["evidence"]["aggregate_expected_net_return_pct"] == 0.43
    assert (
        example["evidence"]["ai_expected_return_policy"]
        == "probe_original_hold_without_independent_support"
    )
    assert example["evidence"]["ai_expected_return_weight"] == 0.0
    assert example["evidence"]["ai_expected_return_independent_probe_support"] == []
    assert (
        compact["entry_ai_expected_return_policy_counts"][
            "probe_original_hold_without_independent_support"
        ]
        == 1
    )
    assert "large_unused_evidence" not in example["evidence"]
    assert (
        compact["market_entry_probe_conversion_block_reason_counts"][
            "expected_net_below_probe_threshold"
        ]
        == 3
    )
    assert compact["market_entry_profit_probe_block_kind_counts"]["probe_threshold_not_met"] == 2
    assert example["probe"]["probe_conversion_ready"] is False
    assert example["probe"]["probe_conversion_block_reasons"] == [
        "expected_net_below_probe_threshold"
    ]
    assert (
        example["probe"]["evidence_profit_probe_blocked"]["block_kind"] == "probe_threshold_not_met"
    )
    assert "large_unused_probe" not in example["probe"]
    assert "large_unused_probe_block" not in example["probe"]["evidence_profit_probe_blocked"]
    assert example["sizing"]["final_notional_usdt"] == 180
    assert "large_unused_sizing" not in example["sizing"]
    executed = compact["executed_entry_sizing_diagnostics"]
    assert executed["read_only"] is True
    assert executed["executed_entry_count"] == 2
    assert executed["market_executed_entry_count"] == 1
    assert executed["order_status_counts"]["filled"] == 1
    assert executed["sizing_reason_tag_counts"]["strategy_probe_cap_applied"] == 1
    assert executed["sizing_reason_tag_counts"]["low_payoff:expected_net_below_min"] == 1
    assert executed["order_notional_stats"]["median"] == 180
    assert compact["low_payoff_entry_count"] == 4
    assert compact["low_payoff_reason_counts"]["expected_net_below_min"] == 3
    assert compact["low_payoff_missing_reason_count"] == 1
    assert compact["entry_analysis_type_counts"]["entry_candidate"] == 5
    assert compact["entry_analysis_type_skip_kind_counts"]["entry_candidate:executed"] == 1
    assert (
        compact["entry_analysis_type_skip_kind_counts"]["entry_candidate:high_risk_review_blocked"]
        == 1
    )
    assert (
        compact["entry_analysis_type_low_payoff_reason_counts"][
            "entry_candidate:expected_net_below_min"
        ]
        == 3
    )
    assert (
        compact["entry_analysis_type_metric_stats"]["entry_candidate"]["position_size_pct"][
            "median"
        ]
        == 0.0021
    )
    assert compact["high_risk_review_status_counts"]["completed"] == 2
    assert compact["high_risk_review_approved_counts"]["approved_false"] == 1
    assert compact["high_risk_review_trigger_reason_counts"]["today_recovery_after_loss"] == 1
    assert compact["notional_floor_blocked_counts"] == [
        {"value": "尾部风险偏高，不抬高仓位", "count": 1}
    ]
    assert compact["high_risk_review_reason_counts"] == [{"value": "risk asymmetric", "count": 1}]
    candidate_example = compact["entry_examples"][1]
    assert candidate_example["analysis_type"] == "entry_candidate"
    assert candidate_example["skip_kind"] == "high_risk_review_blocked"
    assert candidate_example["high_risk_review"]["approved"] is False
    assert candidate_example["high_risk_review"]["reasons"] == ["today_recovery_after_loss"]
    assert candidate_example["sizing"]["low_payoff_reasons"] == ["expected_net_below_min"]
    assert all(item["analysis_type"] != "position_review" for item in compact["entry_examples"])
    assert len(executed["samples"]) == 3
    assert (
        executed["ai_expected_return_policy_counts"][
            "probe_original_hold_without_independent_support"
        ]
        == 1
    )
    executed_sample = executed["samples"][0]
    assert executed_sample["order"]["notional"] == 180
    assert executed_sample["decision"]["suggested_leverage"] == 3
    assert executed_sample["evidence"]["expected_net_return_pct"] == 0.4
    assert executed_sample["evidence"]["aggregate_expected_net_return_pct"] == 0.43
    assert (
        executed_sample["evidence"]["ai_expected_return_policy"]
        == "probe_original_hold_without_independent_support"
    )
    assert executed_sample["evidence"]["ai_expected_return_weight"] == 0.0
    assert executed_sample["evidence"]["ai_expected_return_independent_probe_support"] == []
    assert executed_sample["sizing"]["strategy_probe_cap_applied"] is True
    assert executed_sample["sizing"]["low_payoff_reasons"] == [
        "expected_net_below_min",
        "evidence_low_payoff_quality",
    ]
    assert executed_sample["sizing_reason_tags"] == [
        "evidence_tier:exploration",
        "sizing_quality:probe",
        "low_payoff:expected_net_below_min",
        "low_payoff:evidence_low_payoff_quality",
        "strategy_probe_cap_applied",
    ]
    assert executed_sample["execution_result"]["exchange_confirmed"] is True
    assert "large_unused_payload" not in executed
    assert "raw_payload" not in executed_sample
    assert "large_unused_execution_result" not in executed_sample["execution_result"]
    assert "Read-only compact market entry" in compact["diagnostic_boundary"]


def test_strategy_health_market_symbol_only_report_is_compact_and_keeps_guards() -> None:
    report = {
        "window_minutes": 60,
        "generated_at": "2026-06-23T12:44:20+00:00",
        "counts": {
            "decisions": 211,
            "market_decisions": 64,
            "market_entry_decisions": 2,
            "orders": 0,
            "failed_orders": 0,
            "rejected_orders": 0,
            "positions_created": 0,
            "positions_closed": 0,
            "open_positions": 6,
            "fast_loss_close_under_15m": 0,
        },
        "trade_execution_contract": {
            "status": "ok",
            "can_bypass_risk_controls": False,
            "summary": {
                "decision_count": 211,
                "executed_entry_count": 0,
                "contract_violation_count": 0,
                "weak_evidence_executed_count": 0,
                "negative_expected_executed_count": 0,
                "fast_loss_count": 0,
                "fast_loss_without_strong_exit_count": 0,
                "reentry_without_strong_unlock_count": 0,
            },
        },
        "local_ml_readiness": {
            "status": "degraded",
            "readiness_state": "degraded",
            "allow_live_position_influence": False,
            "blocking_reason_codes": ["dirty_sample_ratio_high"],
            "metrics": {
                "sample_count": 19971,
                "dirty_sample_ratio": 0.7571,
                "long_pr_auc": 0.338,
                "short_pr_auc": 0.366,
                "top_long_avg_return_pct": 0.0245,
                "top_short_avg_return_pct": -0.0048,
                "training_data_version": "2026-06-23.v3",
                "required_training_data_version": "2026-06-23.v3",
                "large_unused_metric": "omit",
            },
            "training_window_composition": {
                "sample_count": 19971,
                "decision_action_counts": {"hold": 12000},
            },
        },
        "market_symbol_diagnostics": {
            "market_decision_count": 64,
            "market_unique_symbol_count": 27,
            "market_top_symbols": [{"symbol": f"S{i}/USDT", "count": i} for i in range(20)],
            "candidate_funnel_sample_count": 42,
            "candidate_funnel_window": {
                "count": 42,
                "rank_underfilled_count": 7,
                "metric_stats": {
                    "scan_symbol_count": {
                        "count": 42,
                        "median": 120,
                        "p75": 120,
                        "max": 120,
                        "positive": 42,
                        "zero": 0,
                        "min": 120,
                        "large": "omit",
                    },
                    "rank_selected_count": {
                        "count": 42,
                        "median": 2,
                        "p75": 3,
                        "max": 8,
                        "positive": 42,
                        "zero": 0,
                        "min": 1,
                    },
                },
                "filtered_out_reason_counts": [
                    {"value": f"filter_reason_{i}", "count": i} for i in range(20)
                ],
                "selected_symbol_counts": [
                    {"symbol": f"SEL{i}/USDT", "count": i} for i in range(20)
                ],
                "filtered_symbol_counts": [
                    {"symbol": f"FIL{i}/USDT", "count": i} for i in range(20)
                ],
                "large_window_payload": ["omit"],
            },
            "candidate_filter_outcomes": {
                "read_only": True,
                "sampled_symbol_count": 9,
                "sampled_occurrence_count": 14,
                "market_entry_after_filter_count": 3,
                "market_entry_after_filter_symbol_count": 2,
                "positive_expected_net_after_filter_count": 1,
                "executed_after_filter_count": 0,
                "category_counts": [{"value": f"category_{i}", "count": i} for i in range(20)],
                "reason_counts": [{"value": f"reason_{i}", "count": i} for i in range(20)],
                "sampled_symbol_counts": [{"symbol": f"SF{i}/USDT", "count": i} for i in range(20)],
                "outcome_symbol_counts": [{"symbol": f"OF{i}/USDT", "count": i} for i in range(20)],
                "positive_expected_net_symbol_counts": [
                    {"symbol": f"PF{i}/USDT", "count": i} for i in range(20)
                ],
                "skip_kind_counts": [{"value": f"skip_{i}", "count": i} for i in range(20)],
                "evidence_tier_counts": [{"value": f"tier_{i}", "count": i} for i in range(20)],
                "expected_net_stats": {"count": 3, "positive": 1},
                "symbol_examples": [{"symbol": f"SE{i}/USDT", "large": "omit"} for i in range(20)],
                "market_entry_examples": [
                    {"symbol": f"ME{i}/USDT", "large": "omit"} for i in range(20)
                ],
                "diagnostic_boundary": "Read-only replay",
                "large_unused_payload": ["omit"],
            },
            "market_analysis_progress": {
                "read_only": True,
                "count": 4,
                "symbol_counts": [{"symbol": f"P{i}/USDT", "count": i} for i in range(12)],
                "processed_index_stats": {"count": 4, "median": 1.5, "max": 3},
                "ranked_market_symbol_count_stats": {"count": 4, "median": 8},
                "full_round_elapsed_before_ai_stats": {"count": 4, "median": 58.0},
                "market_ai_elapsed_before_symbol_stats": {"count": 4, "median": 9.0},
                "budget_used_ratio_before_ai_stats": {"count": 4, "median": 0.83},
                "market_ai_budget_used_ratio_before_symbol_stats": {
                    "count": 4,
                    "median": 0.33,
                },
                "latest": {
                    "symbol": "P0/USDT",
                    "processed_index": 0,
                    "ranked_market_symbol_count": 8,
                    "remaining_after_this_symbol": 7,
                    "market_ai_elapsed_seconds_before_symbol": 9.0,
                    "full_round_elapsed_seconds_before_ai": 58.0,
                    "budget_clock_scope": "market_ai_phase",
                },
                "diagnostic_boundary": "Read-only market AI throughput aggregate",
            },
            "latest_candidate_funnel": {
                "scan_symbol_count": 120,
                "feature_fetch_requested_count": 12,
                "feature_fetch_budget": {
                    "read_only": True,
                    "is_entry_gate": False,
                    "selected_market_feature_fetch_count": 48,
                    "selected_total_feature_fetch_count": 48,
                    "pool_min": 48,
                    "pool_max": 64,
                    "unused_large_payload": ["omit"],
                },
                "feature_valid_count": 12,
                "market_symbol_budget": 2,
                "rank_selected_count": 2,
                "rank_filtered_out_reason_counts": [
                    {"reason": f"reason_{i}", "count": i} for i in range(20)
                ],
                "ranked_symbol_sample": [
                    {
                        "symbol": f"R{i}/USDT",
                        "selected": i < 2,
                        "filter_metrics": {"notional_24h": i * 1000, "extra": "omit"},
                    }
                    for i in range(20)
                ],
                "filtered_symbol_sample": [
                    {"symbol": f"F{i}/USDT", "filter_reasons": ["x"]} for i in range(20)
                ],
                "analysis_budget": {
                    "budget_source": "strategy_learning",
                    "market_limit_policy": "position_first_low_risk_underfilled",
                    "market_symbol_limit": 2,
                    "market_limit_diagnostics": {
                        "position_group_count": 6,
                        "market_caps": {"large": "omit"},
                    },
                    "unused_large_payload": ["omit"],
                },
                "market_budget_rotation": {
                    "read_only": True,
                    "is_entry_gate": False,
                    "applied": True,
                    "start_symbol": "R1/USDT",
                    "ranked_symbol_count": 8,
                    "deferred_symbol_count": 6,
                },
            },
        },
    }

    compact = inspect_online_strategy_health._market_symbol_only_report(report)

    assert compact["trade_execution_contract"]["status"] == "ok"
    assert compact["trade_execution_contract"]["can_bypass_risk_controls"] is False
    assert compact["local_ml_readiness"]["allow_live_position_influence"] is False
    assert compact["local_ml_readiness"]["metrics"]["dirty_sample_ratio"] == 0.7571
    assert compact["local_ml_readiness"]["training_window_composition"]["sample_count"] == 19971
    assert "large_unused_metric" not in compact["local_ml_readiness"]["metrics"]
    diagnostics = compact["market_symbol_diagnostics"]
    assert len(diagnostics["market_top_symbols"]) == 8
    window = diagnostics["candidate_funnel_window"]
    assert window["count"] == 42
    assert window["rank_underfilled_count"] == 7
    assert window["metric_stats"]["scan_symbol_count"]["median"] == 120
    assert "min" not in window["metric_stats"]["scan_symbol_count"]
    assert len(window["filtered_out_reason_counts"]) == 6
    assert len(window["selected_symbol_counts"]) == 6
    assert "large_window_payload" not in window
    outcomes = diagnostics["candidate_filter_outcomes"]
    assert outcomes["read_only"] is True
    assert outcomes["sampled_symbol_count"] == 9
    assert outcomes["market_entry_after_filter_count"] == 3
    assert outcomes["positive_expected_net_after_filter_count"] == 1
    assert len(outcomes["reason_counts"]) == 6
    assert len(outcomes["sampled_symbol_counts"]) == 6
    assert len(outcomes["symbol_examples"]) == 2
    assert len(outcomes["market_entry_examples"]) == 2
    assert outcomes["expected_net_stats"]["positive"] == 1
    assert "large_unused_payload" not in outcomes
    progress = diagnostics["market_analysis_progress"]
    assert progress["read_only"] is True
    assert progress["count"] == 4
    assert progress["latest"]["processed_index"] == 0
    assert progress["latest"]["ranked_market_symbol_count"] == 8
    assert progress["budget_used_ratio_before_ai_stats"]["median"] == 0.83
    assert progress["market_ai_elapsed_before_symbol_stats"]["median"] == 9.0
    assert progress["market_ai_budget_used_ratio_before_symbol_stats"]["median"] == 0.33
    assert progress["latest"]["budget_clock_scope"] == "market_ai_phase"
    latest = diagnostics["latest_candidate_funnel"]
    assert latest["scan_symbol_count"] == 120
    assert latest["feature_valid_count"] == 12
    assert latest["feature_fetch_budget"]["selected_market_feature_fetch_count"] == 48
    assert latest["feature_fetch_budget"]["is_entry_gate"] is False
    assert "unused_large_payload" not in latest["feature_fetch_budget"]
    assert len(latest["rank_filtered_out_reason_counts"]) == 6
    assert len(latest["ranked_symbol_sample"]) == 2
    assert len(latest["filtered_symbol_sample"]) == 2
    assert latest["ranked_symbol_sample"][1]["notional_24h"] == 1000
    assert "filter_metrics" not in latest["ranked_symbol_sample"][0]
    assert latest["analysis_budget"]["budget_source"] == "strategy_learning"
    assert "unused_large_payload" not in latest["analysis_budget"]
    assert latest["market_budget_rotation"]["read_only"] is True
    assert latest["market_budget_rotation"]["is_entry_gate"] is False
    assert latest["market_budget_rotation"]["applied"] is True
    assert latest["market_budget_rotation"]["start_symbol"] == "R1/USDT"
    budget_diag = latest["analysis_budget"]["market_limit_diagnostics"]
    assert budget_diag["position_group_count"] == 6
    assert "market_caps" not in budget_diag
    assert "Read-only compact market symbol" in compact["diagnostic_boundary"]


def test_strategy_health_report_splits_market_and_position_review_decisions() -> None:
    template = inspect_online_strategy_health.REMOTE_SCRIPT_TEMPLATE

    assert "def analysis_type(decision):" in template
    assert '"analysis_type_counts": dict(analysis_type_counts.most_common(20))' in template
    assert (
        '"analysis_type_action_counts": dict(analysis_type_action_counts.most_common(40))'
        in template
    )
    assert '"entry_candidate_evidence_by_type": dict(' in template
    assert "entry_candidate_evidence_by_type.most_common(20)" in template
    assert '"market_decisions": len(market_decisions)' in template
    assert '"market_entry_decisions": len(market_entry_decisions)' in template
    assert '"analysis_type": analysis_type(d)' in template


def test_strategy_health_counts_entry_candidate_evidence_only_for_entry_decisions() -> None:
    template = inspect_online_strategy_health.REMOTE_SCRIPT_TEMPLATE

    assert "for d in entry_decisions:\n        raw = safe_dict(d.raw_llm_response)" in template
    assert "for d in decisions:\n        raw = safe_dict(d.raw_llm_response)" not in template


def test_strategy_health_report_exposes_market_entry_evidence_chain_stats() -> None:
    template = inspect_online_strategy_health.REMOTE_SCRIPT_TEMPLATE

    assert "def expected_net_components(decision):" in template
    assert "market_entry_score_gaps.append" in template
    assert "market_entry_component_contributions" in template
    assert '"market_entry_score_gap_stats": stats(market_entry_score_gaps)' in template
    assert (
        '"market_entry_profit_quality_stats": stats(market_entry_profit_quality_values)' in template
    )
    assert (
        '"market_entry_loss_probability_stats": stats(market_entry_loss_probabilities)' in template
    )
    assert '"market_entry_tail_risk_stats": stats(market_entry_tail_risks)' in template
    assert '"market_entry_expected_net_component_stats": {' in template


def test_strategy_health_report_exposes_entry_execution_blocking_contract() -> None:
    template = inspect_online_strategy_health.REMOTE_SCRIPT_TEMPLATE

    assert "def evidence_components(decision):" in template
    assert "def entry_skip_kind(decision):" in template
    assert "market_entry_evidence_effective_scores.append" in template
    assert "market_entry_evidence_tier_counts" in template
    assert "market_entry_final_skip_kind_counts" in template
    assert "market_entry_evidence_component_status_counts" in template
    assert '"market_entry_opportunity_score_gap_stats": stats(market_entry_score_gaps)' in template
    assert '"market_entry_evidence_effective_score_stats": stats(' in template
    assert '"market_entry_evidence_shadow_only_count": market_entry_shadow_only_count' in template
    assert (
        '"market_entry_evidence_tradeable_probe_count": market_entry_tradeable_probe_count'
        in template
    )


def test_strategy_health_report_exposes_market_symbol_funnel_diagnostics() -> None:
    template = inspect_online_strategy_health.REMOTE_SCRIPT_TEMPLATE

    assert "def counter_rows(counter, limit=20):" in template
    assert "def symbol_counter_rows(counter, limit=20):" in template
    assert "def top_share(counter, top_n=3):" in template
    assert "def candidate_funnel_aggregate(funnels):" in template
    assert "def market_analysis_progress(decision):" in template
    assert "def market_analysis_progress_aggregate(decisions):" in template
    assert "market_ai_elapsed_before_symbol_stats" in template
    assert "budget_clock_scope" in template
    assert "def candidate_filter_outcome_diagnostics(" in template
    assert "candidate_filter_outcomes = candidate_filter_outcome_diagnostics(" in template
    assert "market_progress = market_analysis_progress_aggregate(market_decisions)" in template
    assert "market_symbol_counts = Counter(" in template
    assert "market_entry_symbol_counts = Counter(" in template
    assert "market_candidate_funnels = [" in template
    assert "latest_candidate_funnel = market_candidate_funnels[0]" in template
    assert "candidate_funnel_window = candidate_funnel_aggregate(" in template
    assert '"market_symbol_diagnostics": market_symbol_diagnostics' in template
    assert '"market_unique_symbol_count": len(market_symbol_counts)' in template
    assert '"market_entry_unique_symbol_count": len(market_entry_symbol_counts)' in template
    assert '"market_top_symbols": symbol_counter_rows(market_symbol_counts)' in template
    assert '"market_entry_top_symbols": symbol_counter_rows(market_entry_symbol_counts)' in template
    assert '"market_entry_skip_kind_counts": counter_rows(' in template
    assert '"market_entry_tier_counts": counter_rows(' in template
    assert '"entry_unique_to_market_unique_ratio": roundv(' in template
    assert '"candidate_funnel_sample_count": len(market_candidate_funnels)' in template
    assert '"latest_candidate_funnel": latest_candidate_funnel' in template
    assert '"candidate_funnel_window": candidate_funnel_window' in template
    assert '"market_analysis_progress": market_progress' in template
    assert '"candidate_filter_outcomes": candidate_filter_outcomes' in template
    assert '"filtered_out_reason_counts": counter_rows(filtered_reasons, 12)' in template
    assert '"outside_budget_symbol_counts": symbol_counter_rows(' in template
    assert '"positive_expected_net_after_filter_count": positive_expected_net_count' in template
    assert "Positive outcomes here only" in template
    assert "Read-only symbol funnel diagnostics" in template


def test_strategy_health_classifies_market_entry_execution_outcomes() -> None:
    template = inspect_online_strategy_health.REMOTE_SCRIPT_TEMPLATE
    assert 'bool(getattr(decision, "was_executed", False))' in template
    assert 'final_stage == "local_sync" and final_status == "completed"' in template
    assert 'return "executed"' in template
    assert 'final_status in {"skipped", "failed"}' in template
    assert 'return "exchange_not_confirmed"' in template
    assert 'if data.get("skip_kind")' in template
    assert 'return str(data.get("skip_kind"))' in template


def test_strategy_health_report_exposes_entry_score_breakdown_and_relief_diagnostics() -> None:
    template = inspect_online_strategy_health.REMOTE_SCRIPT_TEMPLATE

    assert "ENTRY_EVIDENCE_SCORE_WEAK_PROBE" in template
    assert "market_entry_evidence_raw_scores" in template
    assert "market_entry_evidence_score_offsets" in template
    assert "market_entry_evidence_component_point_stats" in template
    assert "market_entry_evidence_relief_applied_counts" in template
    assert "market_entry_advisory_wait_reason_counts" in template
    assert '"entry_evidence_thresholds": {' in template
    assert '"market_entry_evidence_raw_score_stats": stats(' in template
    assert '"market_entry_evidence_score_offset_stats": stats(' in template


def test_strategy_health_report_exposes_rejected_order_diagnostics() -> None:
    template = inspect_online_strategy_health.REMOTE_SCRIPT_TEMPLATE

    assert "def order_execution_result(decision):" in template
    assert "def round_optional(value, digits=6):" in template
    assert '"planned_order_contracts": round_optional(' in template
    assert '"planned_base_quantity": round_optional(' in template
    assert "order_status_counts = Counter" in template
    assert "rejected_orders = [" in template
    assert "non_filled_orders = [" in template
    assert '"order_status_counts": dict(order_status_counts.most_common(20))' in template
    assert '"non_filled_orders": len(non_filled_orders)' in template
    assert '"rejected_orders": len(rejected_orders)' in template
    assert '"rejected_order_examples": rejected_order_examples' in template
    assert '"execution_result": order_execution_result(d)' in template


def test_strategy_health_report_exposes_executed_entry_sizing_diagnostics() -> None:
    template = inspect_online_strategy_health.REMOTE_SCRIPT_TEMPLATE

    assert "def executed_entry_sizing_diagnostics(entry_rows, order_by_decision):" in template
    assert "executed_entry_sizing_reason_tags(ev, sz)" in template
    assert '"executed_entry_sizing_diagnostics": executed_entry_sizing_diagnostics(' in template
    assert '"order_notional_stats": stats(order_notionals)' in template
    assert '"decision_leverage_stats": stats(leverage_values)' in template
    assert '"sizing_reason_tag_counts": dict(reason_tag_counts.most_common(20))' in template
    assert "low_payoff_reason_counts = Counter()" in template
    assert '"low_payoff_reason_counts": dict(low_payoff_reason_counts.most_common(20))' in template
    assert '"low_payoff_missing_reason_count": low_payoff_missing_reason_count' in template
    assert "Read-only executed-entry sizing/order diagnostics" in template


def test_strategy_health_report_exposes_local_ml_readiness_summary() -> None:
    template = inspect_online_strategy_health.REMOTE_SCRIPT_TEMPLATE

    assert "from services.ml_signal_service import MLSignalService" in template
    assert "def local_ml_readiness_summary():" in template
    assert '"local_ml_readiness": local_ml_readiness_summary()' in template
    assert '"allow_live_position_influence"' in template
    assert '"blocking_reason_codes"' in template
    assert '"quality_top_reasons"' in template
    assert '"quality_by_kind"' in template
    assert '"quality_top_actions"' in template
    assert '"quality_top_timeframes"' in template
    assert '"training_window_composition"' in template


def test_strategy_health_report_exposes_trade_execution_contract_summary() -> None:
    template = inspect_online_strategy_health.REMOTE_SCRIPT_TEMPLATE

    assert "from services.trade_execution_contract import TradeExecutionContractService" in template
    assert "async def trade_execution_contract_summary():" in template
    assert "TradeExecutionContractService().report(" in template
    assert "since=since" in template
    assert "trade_contract = await trade_execution_contract_summary()" in template
    assert '"trade_execution_contract": trade_contract' in template
    assert '"can_bypass_risk_controls"' in template
    assert '"contract_violation_count"' in template
    assert '"weak_evidence_executed_count"' in template
    assert '"negative_expected_executed_count"' in template
    assert '"fast_loss_without_strong_exit_count"' in template
    assert '"reentry_without_strong_unlock_count"' in template


def test_strategy_health_contract_samples_are_json_safe() -> None:
    template = inspect_online_strategy_health.REMOTE_SCRIPT_TEMPLATE

    assert "def json_safe(value):" in template
    assert '"violations": json_safe(safe_list(report.get("violations"))[:10])' in template
    assert (
        '"fast_loss_samples": json_safe(safe_list(report.get("fast_loss_samples"))[:10])'
        in template
    )


def test_strategy_health_summary_keeps_stop_signal_fields() -> None:
    report = {
        "window_minutes": 120,
        "generated_at": "2026-06-23T08:11:45+00:00",
        "counts": {
            "decisions": 400,
            "orders": 6,
            "filled_orders": 5,
            "failed_orders": 1,
            "rejected_orders": 1,
            "fast_loss_close_under_15m": 1,
            "open_positions": 6,
        },
        "order_status_counts": {"filled": 5, "rejected": 1},
        "local_ml_readiness": {
            "status": "degraded",
            "readiness_state": "degraded",
            "allow_live_position_influence": False,
            "blocking_reason_codes": ["dirty_sample_ratio_high"],
            "metrics": {"dirty_sample_ratio": 0.75},
            "training_window_composition": {
                "sample_count": 100,
                "decision_action_counts": {"hold": 90, "long": 5, "short": 5},
            },
        },
        "trade_execution_contract": {
            "status": "ok",
            "audit_only": True,
            "can_bypass_risk_controls": False,
            "summary": {
                "decision_count": 401,
                "executed_entry_count": 4,
                "contract_violation_count": 0,
                "weak_evidence_executed_count": 0,
                "negative_expected_executed_count": 0,
                "fast_loss_count": 1,
                "fast_loss_without_strong_exit_count": 0,
                "reentry_without_strong_unlock_count": 0,
            },
            "violation_reason_counts": {},
        },
        "market_symbol_diagnostics": {
            "market_decision_count": 122,
            "market_unique_symbol_count": 34,
            "market_entry_count": 8,
            "market_entry_unique_symbol_count": 3,
            "candidate_funnel_sample_count": 4,
            "latest_candidate_funnel": {"feature_valid_count": 12},
            "candidate_funnel_window": {
                "count": 4,
                "filtered_out_reason_counts": [
                    {"value": "analysis_volume_ratio_below_floor", "count": 7}
                ],
            },
        },
        "closed_position_pnl_diagnostics": {
            "read_only": True,
            "closed_count": 4,
            "win_count": 1,
            "loss_count": 3,
            "total_realized_pnl": -0.88,
            "samples": [{"id": i} for i in range(6)],
        },
        "rejected_order_examples": [{"order_id": 1, "symbol": "BTC/USDT"}],
        "fast_loss_positions": [{"id": 2, "symbol": "ETH/USDT"}],
    }

    summary = inspect_online_strategy_health._summarize_report(report)

    assert summary["counts"]["rejected_orders"] == 1
    assert summary["counts"]["fast_loss_close_under_15m"] == 1
    assert (
        summary["trade_execution_contract"]["summary"]["fast_loss_without_strong_exit_count"] == 0
    )
    assert summary["trade_execution_contract"]["can_bypass_risk_controls"] is False
    assert summary["market_symbol_diagnostics"]["market_entry_unique_symbol_count"] == 3
    assert summary["market_symbol_diagnostics"]["candidate_funnel_sample_count"] == 4
    assert summary["market_symbol_diagnostics"]["candidate_funnel_window"]["count"] == 4
    assert summary["closed_position_pnl_diagnostics"]["closed_count"] == 4
    assert summary["closed_position_pnl_diagnostics"]["total_realized_pnl"] == -0.88
    assert len(summary["closed_position_pnl_diagnostics"]["samples"]) == 5
    assert summary["rejected_order_examples"] == [{"order_id": 1, "symbol": "BTC/USDT"}]
    assert summary["fast_loss_positions"] == [{"id": 2, "symbol": "ETH/USDT"}]
    assert summary["local_ml_readiness"]["allow_live_position_influence"] is False
    assert summary["local_ml_readiness"]["training_window_composition"]["sample_count"] == 100


def test_strategy_health_shadow_only_examples_use_final_entry_evidence_contract() -> None:
    template = inspect_online_strategy_health.REMOTE_SCRIPT_TEMPLATE

    assert "def closed_position_pnl_diagnostics(closed_rows):" in template
    assert "def compact_closed_position_pnl_diagnostics(value):" in template
    assert "_compact_closed_position_pnl_diagnostics" not in template
    assert "def is_shadow_only_entry_decision(decision):" in template
    assert (
        "def normalize_relief_for_final_contract(relief, final_shadow_only, final_tier, final_score):"
        in template
    )
    assert "if is_shadow_only_entry_decision(d):" in template
    assert '"positive_net_probe_relief": normalize_relief_for_final_contract(' in template
    assert 'ev["positive_net_probe_relief"].get("shadow_only")' not in template
