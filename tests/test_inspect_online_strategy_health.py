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


def test_strategy_health_remote_command_can_emit_summary_only() -> None:
    command = inspect_online_strategy_health._build_remote_command(
        120, token="abc123", summary=True
    )

    assert "SUMMARY_ONLY = True" in command
    assert "__SUMMARY_ONLY__" not in command
    assert "output = summary_report(report) if SUMMARY_ONLY else report" in command
    assert "json.loads(out)" not in command


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
    assert "order_status_counts = Counter" in template
    assert "rejected_orders = [" in template
    assert "non_filled_orders = [" in template
    assert '"order_status_counts": dict(order_status_counts.most_common(20))' in template
    assert '"non_filled_orders": len(non_filled_orders)' in template
    assert '"rejected_orders": len(rejected_orders)' in template
    assert '"rejected_order_examples": rejected_order_examples' in template
    assert '"execution_result": order_execution_result(d)' in template


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
    assert summary["rejected_order_examples"] == [{"order_id": 1, "symbol": "BTC/USDT"}]
    assert summary["fast_loss_positions"] == [{"id": 2, "symbol": "ETH/USDT"}]
    assert summary["local_ml_readiness"]["allow_live_position_influence"] is False


def test_strategy_health_shadow_only_examples_use_final_entry_evidence_contract() -> None:
    template = inspect_online_strategy_health.REMOTE_SCRIPT_TEMPLATE

    assert "def is_shadow_only_entry_decision(decision):" in template
    assert (
        "def normalize_relief_for_final_contract(relief, final_shadow_only, final_tier, final_score):"
        in template
    )
    assert "if is_shadow_only_entry_decision(d):" in template
    assert '"positive_net_probe_relief": normalize_relief_for_final_contract(' in template
    assert 'ev["positive_net_probe_relief"].get("shadow_only")' not in template
