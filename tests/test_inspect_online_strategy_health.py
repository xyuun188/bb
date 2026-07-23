import pytest

from scripts import inspect_online_strategy_health
from services.profit_training_contract import PROFIT_TRAINING_TARGET


def test_decode_remote_json_ignores_runtime_logs_before_final_payload() -> None:
    output = (
        "2026-07-15 10:20:00 [info] status probe started\n"
        '{"event":"runtime_log","level":"info"}\n'
        '{"generated_at":"2026-07-15T10:20:01+00:00","audit_only":true}\n'
    )

    payload = inspect_online_strategy_health._decode_remote_json(output)

    assert payload == {
        "generated_at": "2026-07-15T10:20:01+00:00",
        "audit_only": True,
    }


def test_remote_template_compiles_and_uses_dynamic_return_contract() -> None:
    template = (
        inspect_online_strategy_health.REMOTE_SCRIPT_TEMPLATE.replace(
            "__WINDOW_MINUTES__", "120"
        )
        .replace("__SUMMARY_ONLY__", "False")
        .replace("__MARKET_SYMBOL_ONLY__", "False")
        .replace("__ENTRY_ONLY__", "False")
        .replace("__REPLAY_ONLY__", "False")
    )

    compile(template, "<strategy-health>", "exec")
    assert "TradeExecutionContractService" in template
    assert "get_model_training_registry_status" in template
    assert "okx_training_refresh_gate" in template
    assert "get_data_collection_status" in template
    assert "get_strategy_learning" in template
    assert "get_expert_memories" in template
    assert "get_shadow_backtests" in template
    assert "audit_protection_order_integrity" in template
    assert "get_positions_strict" in template
    assert PROFIT_TRAINING_TARGET in template
    assert "timedelta(minutes=WINDOW_MINUTES)" in template
    assert 'APP_ROOT = Path("/data/bb/app")' in template
    assert "_inherit_dashboard_runtime_environment()" in template
    assert "evidence_tier" not in template
    assert "tradeable_probe" not in template


def test_replay_only_command_skips_unrelated_online_audits() -> None:
    command = inspect_online_strategy_health._build_remote_command(
        120,
        token="replay123",
        replay_only=True,
    )

    assert "REPLAY_ONLY = True" in command
    assert "if REPLAY_ONLY:" in command
    assert 'get_strategy_learning(mode="paper", detail="summary")' in command


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
                "strategy_blueprint": {
                    "strategy_id": "trained-model-v1",
                    "execution_scope": "paper_only",
                    "paper_execution_eligible": False,
                    "live_execution_permission": False,
                },
            },
            "model_training_registry": {
                "summary": {"trainable_count": 8},
                "scheduler_state": {
                    "status": "warning",
                    "heartbeat_stale": True,
                    "models": {
                        "local_ml_profit_quality": {
                            "state": "failed",
                            "history": [
                                {"event": "started"},
                                {"event": "failed", "reason": "error"},
                            ],
                        }
                    },
                },
                "models": [
                    {
                        "model_id": "local_ml_profit_quality",
                        "trainable": True,
                        "lifecycle": "trained",
                    }
                ],
            },
            "okx_training_refresh_gate": {
                "allowed": False,
                "reason": "okx_daily_reconciliation_report_stale",
            },
            "data_collection": {
                "checked_at": "2026-07-15T10:00:00+00:00",
                "sources": [{"id": "market", "status": "ok"}],
                "training": {
                    "local_ai_tools": {"completed_trade_sample_count": 61},
                    "governance": {"cleanup_effective": True},
                },
            },
            "shadow_maturity": {
                "count": 20,
                "pending_count": 6,
                "completed_count": 14,
            },
            "strategy_learning": {
                "optimization_target": "maximize_authoritative_fee_after_return_rate",
                "paper_strategy_champion": {
                    "active": False,
                    "status": "base_strategy",
                    "live_execution_permission": False,
                    "reason": "no_validated_trained_model_strategy",
                },
                "feedback": {"generated_at": "2026-07-15T10:00:00+00:00"},
                "schedule": {
                    "scheduler_mode": "shadow_validation",
                    "candidate_count": 6,
                    "governed_candidate_count": 0,
                    "rejected_candidate_count": 6,
                    "current_production_strategy": {"id": "dynamic-return-contract"},
                    "runtime": {
                        "production_influence_enabled": False,
                        "can_authorize_entry": False,
                    },
                },
            },
            "expert_learning": {
                "count": 30,
                "reflection_count": 20,
                "authoritative_outcome_contract": {
                    "loaded_count": 497,
                    "complete_count": 61,
                    "actual_outcome_overrides_shadow": True,
                    "shadow_production_weight": 0.0,
                },
                "reflections": [],
            },
            "open_positions": {
                "total": 5,
                "count": 5,
                "protection_inventory": {"missing_keys": []},
            },
            "latest_okx_reconciliation": {
                "generated_at": "2026-07-15T09:51:36+00:00",
                "status": "warning",
                "can_open_new_entries": True,
                "can_refresh_training": True,
                "requires_attention": False,
                "issue_ledger": {"summary": {"unresolved": 1}},
            },
        }
    )

    assert summary["optimization_target"] == PROFIT_TRAINING_TARGET
    assert summary["contract_summary"]["contract_violation_count"] == 2
    assert summary["ml_live_influence"] is False
    assert summary["model_strategy_blueprint"]["execution_scope"] == "paper_only"
    assert summary["model_strategy_blueprint"]["live_execution_permission"] is False
    assert summary["model_training_summary"]["trainable_count"] == 8
    assert summary["training_scheduler_state"]["heartbeat_stale"] is True
    assert summary["training_scheduler_state"]["models"][
        "local_ml_profit_quality"
    ]["recent_history"][-1]["event"] == "failed"
    assert summary["okx_training_refresh_gate"]["allowed"] is False
    closed_loop = summary["profit_closed_loop"]
    assert closed_loop["shadow_maturity"]["completed_total"] == 14
    assert closed_loop["strategy_scheduler"]["candidate_count"] == 6
    assert closed_loop["strategy_scheduler"]["production_influence_enabled"] is False
    assert closed_loop["strategy_scheduler"]["paper_strategy_champion"][
        "active"
    ] is False
    assert closed_loop["authoritative_settlement"]["outcome_contract"][
        "actual_outcome_overrides_shadow"
    ] is True
    assert closed_loop["positions_and_protection"]["open_total"] == 5
    assert closed_loop["positions_and_protection"]["protection_inventory"][
        "missing_keys"
    ] == []
    assert closed_loop["okx_reconciliation"]["can_refresh_training"] is True
    assert summary["trainable_models"][0]["model_id"] == "local_ml_profit_quality"
