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
