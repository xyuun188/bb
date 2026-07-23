from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import pytest

from web_dashboard.api import system_audit

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_removed_fixed_policy_and_global_fallback_tokens_cannot_return_to_production() -> None:
    forbidden = {
        "max_position_pct",
        "max_daily_loss_pct",
        "hard_stop_loss_pct",
        "max_open_positions_per_model",
        "max_same_symbol_positions_per_side",
        "max_total_margin_pct",
        "execution_account_max_loss_pct",
        "execution_account_max_loss_usdt",
        "shadow_memory_min_return_pct",
        "training_shadow_sample_limit",
        "training_trade_sample_limit",
        "training_sequence_sample_limit",
        "training_text_sample_limit",
        "MIN_TIMESERIES_SEQUENCE_LENGTH",
        "PROFIT_PROTECTION_MIN_NET_USDT",
        "AUTO_TRADE_MIN_NOTIONAL_24H",
        "AUTO_TRADE_MIN_VOLUME_RATIO",
        "AUTO_TRADE_MAX_VOLATILITY_20",
        "AUTO_TRADE_MAX_ABS_CHANGE_24H",
        "ABNORMAL_WICK_TAIL_RISK_MAX_PCT",
        "ai_market_fast_prefilter_enabled",
        "ai_market_fast_prefilter_min_expected_return_pct",
        "ai_market_fast_prefilter_max_loss_probability",
        "DEFAULT_MIN_CREATED_SHADOW_SAMPLES",
        "DEFAULT_MIN_COMPLETED_SHADOW_SAMPLES",
        "min_created_shadow_samples",
        "min_completed_shadow_samples",
        "max_open_positions: int | None",
        "OLD_TAKEOVER",
        "old_takeover",
        "temporary_old_server_takeover_allowed",
        "temporary_old_server_role",
        "ATTACHED_PROTECTION_MIN_STOP_PCT",
        "ATTACHED_PROTECTION_MIN_TAKE_PROFIT_PCT",
        "ATTACHED_PROTECTION_MIN_TRIGGER_GAP_PCT",
        "fee_after_sample_floor_not_met",
        "strategy_learning_min_trade_count_target",
        "vector_memory_min_score",
        "VECTOR_MEMORY_MIN_SCORE",
        "old_model_server_fast_fallback",
        "sync_legacy_local_ai_tools_key",
        "deploy_old_model_server_takeover",
        "switch_model_server_profile",
        "load_local_ai_tools_api_key_from_model_server",
        "_REMOTE_LOCAL_AI_TOOLS_KEY_COMMAND",
        "MODEL_SERVER_ACTIVE_PROFILE",
        "OLD_PROFILE_FILENAME",
        "_load_old_server_info",
        "expert_memory_per_prompt",
        "expert_memory_limit",
        "memory_limit_provider",
        "_is_auto_tradeable_feature",
        "_is_auto_analysis_candidate_feature",
    }
    removed_settings = re.compile(
        r"settings\.(?:ai_api_key|ai_api_base|ai_model|okx_api_key|okx_api_secret|okx_passphrase)\b"
    )
    roots = (
        "config",
        "ai_brain",
        "services",
        "risk_manager",
        "executor",
        "web_dashboard/api",
        "scripts",
    )
    violations: list[str] = []
    for root in roots:
        for path in (PROJECT_ROOT / root).rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                if token in source:
                    violations.append(f"{path.relative_to(PROJECT_ROOT)}:{token}")
            if removed_settings.search(source):
                violations.append(f"{path.relative_to(PROJECT_ROOT)}:removed_global_setting")

    assert violations == []


def test_specialist_promotion_evaluation_cannot_reintroduce_fixed_row_limits() -> None:
    service_source = (PROJECT_ROOT / "services/specialist_shadow_evaluation.py").read_text(
        encoding="utf-8"
    )
    runner_source = (PROJECT_ROOT / "scripts/run_specialist_shadow_evaluation.py").read_text(
        encoding="utf-8"
    )
    timer_source = (
        PROJECT_ROOT / "scripts/install_specialist_shadow_evaluation_timer.py"
    ).read_text(encoding="utf-8")

    assert "DEFAULT_LIMIT" not in service_source
    assert ".limit(" not in service_source
    assert "--authoritative-limit" not in runner_source
    assert "--limit" not in runner_source
    assert "--limit" not in timer_source


def test_training_pipelines_cannot_reintroduce_sample_limit_controls() -> None:
    sources = [
        (PROJECT_ROOT / path).read_text(encoding="utf-8")
        for path in (
            "scripts/train_local_ai_tools_models.py",
            "scripts/train_ml_signal_model.py",
            "scripts/evaluate_ml_training_windows.py",
            "scripts/finquant_expert_lora_training.py",
            "scripts/run_phase3_rebuild_preflight.py",
        )
    ]
    forbidden = (
        "--shadow-limit",
        "--trade-limit",
        "--sequence-limit",
        "--text-limit",
        "--memory-limit",
        "--max-samples",
        'parser.add_argument("--limit"',
        "shadow_limit",
        "trade_limit",
        "sequence_limit",
        "text_limit",
        "memory_limit",
        "max_samples",
        "candidate_multiplier",
    )

    assert [token for source in sources for token in forbidden if token in source] == []


def test_win_rate_accuracy_and_auc_remain_diagnostic_only() -> None:
    decision_roots = (
        PROJECT_ROOT / "ai_brain",
        PROJECT_ROOT / "services",
        PROJECT_ROOT / "risk_manager",
        PROJECT_ROOT / "executor",
    )
    violations: list[str] = []
    metric_names = ("win_rate", "accuracy", "auc", "pr_auc")
    for root in decision_roots:
        for path in root.rglob("*.py"):
            tree = compile(path.read_text(encoding="utf-8"), str(path), "exec", ast.PyCF_ONLY_AST)
            for node in ast.walk(tree):
                if not isinstance(node, (ast.If, ast.IfExp, ast.While, ast.Assert)):
                    continue
                condition = ast.unparse(node.test).lower()
                if any(name in condition for name in metric_names):
                    violations.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}:{condition}")

    assert violations == []


def test_model_server_env_routes_are_generically_purged() -> None:
    source = (PROJECT_ROOT / "scripts/sync_to_online_server.py").read_text(encoding="utf-8")

    assert "MODEL_SERVER_ACTIVE_PROFILE" not in source
    assert "startswith('MODEL_SERVER_')" in source
    assert "old_model_server_fast_fallback" not in source


def _required_go_no_go_cards() -> list[dict[str, Any]]:
    return [
        {"key": "okx_trade_fact_integrity", "status": "ok", "details": {}},
        {
            "key": "trade_execution_contract",
            "status": "ok",
            "details": {
                "report_available": True,
                "policy": {
                    "entry_requires_positive_fee_after_return": True,
                    "entry_requires_positive_return_lcb": True,
                    "entry_requires_live_execution_cost": True,
                    "entry_requires_dynamic_risk_budget": True,
                    "entry_requires_complete_provenance": True,
                    "exit_requires_position_economics": True,
                    "exit_requires_dynamic_close_fraction": True,
                    "filled_order_link_required": True,
                },
                "summary": {"contract_violation_count": 0},
            },
        },
        {
            "key": "position_capacity_release",
            "status": "ok",
            "details": {
                "policy": {"strategy_learning_cannot_expand_capacity": True},
                "position_economics_incomplete_count": 0,
                "executed_dynamic_exit_contract_gap_count": 0,
            },
        },
        {
            "key": "model_training",
            "status": "ok",
            "details": {"optimization_target": "net_return_after_all_cost_pct"},
        },
        {"key": "phase3_model_server_readiness", "status": "ok", "details": {}},
    ]


@pytest.mark.asyncio
async def test_trade_execution_contract_audit_is_critical_for_executed_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeService:
        async def report(self, **_kwargs: Any) -> dict[str, Any]:
            return {
                "audit_only": False,
                "read_only": False,
                "live_entry_mutation": True,
                "live_exit_mutation": True,
                "can_bypass_risk_controls": True,
                "summary": {
                    "executed_entry_count": 1,
                    "entry_contract_ready_count": 0,
                    "executed_exit_count": 0,
                    "exit_contract_ready_count": 0,
                    "contract_violation_count": 1,
                    "realized_net_pnl_usdt": -2.0,
                },
                "violations": [{"reason": "entry_return_contract_incomplete"}],
            }

    monkeypatch.setattr(system_audit, "TradeExecutionContractService", FakeService)

    card = await system_audit._trade_execution_contract_audit()

    assert card["status"] == "critical"
    assert card["details"]["audit_only"] is True
    assert card["details"]["read_only"] is True
    assert card["details"]["live_entry_mutation"] is False
    assert card["details"]["can_bypass_risk_controls"] is False


@pytest.mark.asyncio
async def test_shadow_missed_opportunity_audit_never_authorizes_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeService:
        async def report(self, **_kwargs: Any) -> dict[str, Any]:
            return {
                "summary": {
                    "completed_count": 4,
                    "missed_count": 2,
                    "observe_only_count": 1,
                    "executed_return_contract_gap_count": 1,
                },
                "return_observations": [{"symbol": "BTC/USDT", "side": "long"}],
                "executed_return_contract_gaps": [{"decision_id": 7, "reason": "gap"}],
                "blocked_reason_counts": {"gap": 1},
            }

    monkeypatch.setattr(system_audit, "ShadowMissedOpportunityClosedLoopService", FakeService)

    card = await system_audit._shadow_missed_opportunity_audit()

    assert card["status"] == "warning"
    assert card["details"]["read_only"] is True
    observation = card["details"]["return_observations"][0]
    assert observation["observation_only"] is True
    assert observation["can_authorize_entry"] is False
    assert observation["can_change_size_or_leverage"] is False


@pytest.mark.asyncio
async def test_position_capacity_audit_is_critical_for_economics_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeService:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def report(self) -> dict[str, Any]:
            return {
                "open_position_count": 2,
                "position_economics_complete_count": 1,
                "position_economics_incomplete_count": 1,
                "dynamic_exit_decision_count": 1,
                "executed_dynamic_exit_contract_gap_count": 0,
                "policy": {"strategy_learning_cannot_expand_capacity": True},
            }

    monkeypatch.setattr(system_audit, "PositionCapacityReleaseAuditService", FakeService)

    card = await system_audit._position_capacity_release_audit()

    assert card["status"] == "critical"
    assert card["details"]["read_only"] is True
    assert card["details"]["live_sizing_mutation"] is False


def test_phase3_audit_card_uses_dynamic_go_no_go_contract() -> None:
    card = system_audit._phase3_go_no_go_audit_from_cards(_required_go_no_go_cards())

    assert card["status"] == "ok"
    assert card["details"]["status"] == "go"
    assert card["details"]["ready"] is True
    assert card["evidence"][0] == {"label": "已就绪", "value": True}


@pytest.mark.asyncio
async def test_phase3_handoff_audit_is_permissionless_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeService:
        def report(self) -> dict[str, Any]:
            return {
                "status": "dynamic_return_ready",
                "ready": True,
                "audit_only": True,
                "read_only": True,
                "production_permission": False,
                "starts_trading_service": False,
                "submits_orders": False,
                "changes_model_routing": False,
                "live_mutation": False,
                "blockers": [],
                "warnings": [{"code": "observation_report_missing"}],
            }

    monkeypatch.setattr(system_audit, "Phase3StageHandoffService", FakeService)

    card = await system_audit._phase3_stage_handoff_audit()

    assert card["status"] == "warning"
    assert card["details"]["production_permission"] is False
    assert card["details"]["observing"] is True
    assert card["evidence"][0] == {"label": "已就绪", "value": True}


@pytest.mark.asyncio
async def test_strategy_closed_loop_is_critical_for_execution_contract_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRootService:
        async def report(self) -> dict[str, Any]:
            return {"live_ml_blocked_count": 0}

    class FakeContractService:
        async def report(self, **_kwargs: Any) -> dict[str, Any]:
            return {
                "summary": {
                    "contract_violation_count": 1,
                    "realized_net_pnl_usdt": -1.0,
                }
            }

    monkeypatch.setattr(system_audit, "StrategySignalRootCauseAuditService", FakeRootService)
    monkeypatch.setattr(system_audit, "TradeExecutionContractService", FakeContractService)

    card = await system_audit._strategy_closed_loop_audit()

    assert card["status"] == "critical"
    assert card["details"]["audit_only"] is True
    assert card["details"]["live_mutation"] is False
