import pytest

from scripts import inspect_online_strategy_health


def test_remote_template_compiles_and_uses_dynamic_return_contract() -> None:
    template = (
        inspect_online_strategy_health.REMOTE_SCRIPT_TEMPLATE.replace(
            "__WINDOW_MINUTES__", "120"
        )
        .replace("__SUMMARY_ONLY__", "False")
        .replace("__MARKET_SYMBOL_ONLY__", "False")
        .replace("__ENTRY_ONLY__", "False")
    )

    compile(template, "<strategy-health>", "exec")
    assert "TradeExecutionContractService" in template
    assert "get_model_training_registry_status" in template
    assert "realized_fee_after_return" in template
    assert "timedelta(minutes=WINDOW_MINUTES)" in template
    assert 'APP_ROOT = Path("/data/bb/app")' in template
    assert "_inherit_dashboard_runtime_environment()" in template
    assert "evidence_tier" not in template
    assert "tradeable_probe" not in template


def test_remote_command_keeps_paths_scoped_and_quotes_output() -> None:
    result_path = inspect_online_strategy_health._remote_result_path(120, "abc123")
    command = inspect_online_strategy_health._build_remote_command(
        120,
        token="abc123",
        summary=True,
        output_path=result_path,
    )

    assert "/data/bb/app/tmp/codex-strategy-health" in command
    assert "SUMMARY_ONLY = True" in command
    assert result_path in command
    assert "systemd-run --quiet --wait --pipe --collect" in command
    assert "EnvironmentFile=/etc/bb/bb-runtime.env" in command
    assert "--property=User=bb" in command
    assert "chown bb:bb" in command


def test_remote_command_rejects_injection_tokens_and_external_output_paths() -> None:
    with pytest.raises(ValueError):
        inspect_online_strategy_health._build_remote_command(120, token="bad;rm")
    with pytest.raises(ValueError):
        inspect_online_strategy_health._build_remote_command(
            120,
            token="safe",
            output_path="/outside/out.json",
        )


def test_summary_exposes_return_contract_and_ml_readiness() -> None:
    summary = inspect_online_strategy_health._summarize_report(
        {
            "generated_at": "2026-07-12T00:00:00Z",
            "window_minutes": 120,
            "trade_execution_contract": {
                "summary": {"contract_violation_count": 2},
                "violation_reason_counts": {"live_execution_cost_incomplete": 2},
                "policy": {"entry_requires_positive_return_lcb": True},
            },
            "local_ml_readiness": {
                "readiness_state": "degraded",
                "allow_live_position_influence": False,
            },
            "model_training_registry": {
                "summary": {"trainable_count": 8},
                "models": [
                    {
                        "model_id": "local_ml_profit_quality",
                        "trainable": True,
                        "lifecycle": "trained",
                    }
                ],
            },
        }
    )

    assert summary["optimization_target"] == "realized_fee_after_return"
    assert summary["contract_summary"]["contract_violation_count"] == 2
    assert summary["ml_live_influence"] is False
    assert summary["model_training_summary"]["trainable_count"] == 8
    assert summary["trainable_models"][0]["model_id"] == "local_ml_profit_quality"
