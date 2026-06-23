from scripts import inspect_online_strategy_health


def test_strategy_health_remote_command_uses_unique_temp_files() -> None:
    command = inspect_online_strategy_health._build_remote_command(120, token="abc123")

    assert "/data/bb/app/tmp/codex-strategy-health/sample_120_abc123.py" in command
    assert "/data/bb/app/tmp/codex-strategy-health/launcher_120_abc123.py" in command
    assert "mkdir -p /data/bb/app/tmp/codex-strategy-health" in command
    assert "chmod 0750 /data/bb/app/tmp/codex-strategy-health" in command
    assert "codex_strategy_sample.py" not in command
    assert "codex_strategy_launcher.py" not in command
    assert "__WINDOW_MINUTES__" not in command


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


def test_strategy_health_report_exposes_local_ml_readiness_summary() -> None:
    template = inspect_online_strategy_health.REMOTE_SCRIPT_TEMPLATE

    assert "from services.ml_signal_service import MLSignalService" in template
    assert "def local_ml_readiness_summary():" in template
    assert '"local_ml_readiness": local_ml_readiness_summary()' in template
    assert '"allow_live_position_influence"' in template
    assert '"blocking_reason_codes"' in template
    assert '"quality_top_reasons"' in template


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
