from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from services.trading_params import DEFAULT_TRADING_PARAMS
from web_dashboard.api import system_audit

AuditFactory = Callable[[], Awaitable[dict[str, Any]]]


def _async_card(
    key: str,
    status: str,
    summary: str,
    *,
    title: str | None = None,
    evidence_value: int = 1,
) -> AuditFactory:
    async def factory() -> dict[str, Any]:
        return system_audit._audit_card(
            key,
            title or key,
            status,
            summary,
            evidence=[{"label": "样本", "value": evidence_value}],
            next_actions=[f"处理 {key}"],
        )

    return factory


@pytest.mark.asyncio
async def test_system_audit_status_aggregates_root_causes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        system_audit,
        "_shadow_missed_opportunity_audit",
        _async_card("shadow_missed_opportunity", "ok", "Shadow missed loop normal"),
    )
    monkeypatch.setattr(
        system_audit,
        "_trade_execution_contract_audit",
        _async_card("trade_execution_contract", "ok", "Trade execution contract normal"),
    )
    monkeypatch.setattr(
        system_audit,
        "_trade_loop_audit",
        _async_card("trade_loop", "critical", "交易主循环卡住", title="交易闭环"),
    )
    monkeypatch.setattr(
        system_audit,
        "_okx_reconciliation_audit",
        _async_card("okx_reconciliation", "ok", "OKX 对账正常", title="OKX 历史对账"),
    )
    monkeypatch.setattr(
        system_audit,
        "_position_price_integrity_audit",
        _async_card(
            "position_price_integrity",
            "ok",
            "持仓价格一致",
            title="持仓价格一致性",
        ),
    )
    monkeypatch.setattr(
        system_audit,
        "_market_data_audit",
        _async_card("market_data", "warning", "K线过期", title="行情与 K线"),
    )
    monkeypatch.setattr(
        system_audit,
        "_strategy_quality_audit",
        _async_card("strategy_quality", "ok", "策略质量正常", title="策略质量"),
    )
    monkeypatch.setattr(
        system_audit,
        "_strategy_closed_loop_audit",
        _async_card(
            "strategy_closed_loop",
            "ok",
            "策略闭环正常",
            title="策略闭环审计",
        ),
    )
    monkeypatch.setattr(
        system_audit,
        "_model_training_audit",
        _async_card("model_training", "warning", "模型未就绪", title="模型与训练"),
    )
    monkeypatch.setattr(
        system_audit,
        "_model_expert_health_audit",
        _async_card("model_expert_health", "ok", "模型/专家体检正常", title="模型/专家体检"),
    )
    monkeypatch.setattr(
        system_audit,
        "_model_expert_competition_audit",
        _async_card("model_expert_competition", "ok", "模型/专家竞赛正常", title="模型/专家竞赛"),
    )
    monkeypatch.setattr(
        system_audit,
        "_crypto_feature_coverage_audit",
        _async_card(
            "crypto_feature_coverage", "ok", "Crypto 特征覆盖正常", title="数字货币特征覆盖"
        ),
    )
    monkeypatch.setattr(
        system_audit,
        "_model_dynamic_routing_audit",
        _async_card("model_dynamic_routing", "ok", "动态路由正常", title="模型动态路由"),
    )
    monkeypatch.setattr(
        system_audit,
        "_strategy_gate_contract_audit",
        lambda: system_audit._audit_card(
            "strategy_gate_contract", "策略门槛契约", "ok", "运行时契约正常"
        ),
    )
    monkeypatch.setattr(
        system_audit,
        "_source_visible_text_audit",
        lambda: system_audit._audit_card(
            "visible_text_encoding", "中文显示与乱码回归", "ok", "无乱码"
        ),
    )
    monkeypatch.setattr(
        system_audit,
        "_runtime_text_integrity_audit",
        lambda: system_audit._audit_card(
            "runtime_text_integrity", "运行时文本完整性", "ok", "无新增乱码"
        ),
    )

    payload = await system_audit.system_audit_status()

    assert payload["status"] == "critical"
    assert payload["status_label"] == "异常"
    assert payload["summary"] == {
        "cards": 16,
        "critical": 1,
        "warning": 2,
        "ok": 13,
        "findings": 3,
        "nodes": 18,
    }
    assert [card["status"] for card in payload["cards"]] == [
        "critical",
        "warning",
        "warning",
        "ok",
        "ok",
        "ok",
        "ok",
        "ok",
        "ok",
        "ok",
        "ok",
        "ok",
        "ok",
        "ok",
        "ok",
        "ok",
    ]
    assert [item["key"] for item in payload["root_causes"]] == [
        "trade_loop",
        "market_data",
        "model_training",
    ]
    assert all(item["owner_path"] for item in payload["root_causes"])
    assert all(item["state"] in {"unresolved", "observing"} for item in payload["root_causes"])
    node_keys = {node["key"] for node in payload["nodes"]}
    assert "strategy_gate_contract" in node_keys
    assert "strategy_closed_loop" in node_keys
    assert "model_expert_health" in node_keys
    assert "model_expert_competition" in node_keys
    assert "crypto_feature_coverage" in node_keys
    assert "model_dynamic_routing" in node_keys
    assert "shadow_missed_opportunity" in node_keys
    assert "visible_text_encoding" in node_keys
    assert "runtime_text_integrity" in node_keys
    assert all(node["owner_path"] for node in payload["nodes"])
    assert all(node["state"] in {"fixed", "unresolved", "observing"} for node in payload["nodes"])
    assert all(node["state_label"] for node in payload["nodes"])
    card_keys = {card["key"] for card in payload["cards"]}
    assert "strategy_gate_contract" in card_keys
    assert "strategy_closed_loop" in card_keys
    assert "model_expert_health" in card_keys
    assert "model_expert_competition" in card_keys
    assert "crypto_feature_coverage" in card_keys
    assert "model_dynamic_routing" in card_keys
    assert "shadow_missed_opportunity" in card_keys
    assert "trade_execution_contract" in card_keys
    assert "visible_text_encoding" in card_keys
    assert "runtime_text_integrity" in card_keys
    assert all(card["owner_path"] for card in payload["cards"])
    assert payload["issue_ledger"]["summary"] == {
        "fixed": 13,
        "unresolved": 3,
        "observing": 0,
        "total": 16,
    }
    assert [item["key"] for item in payload["issue_ledger"]["unresolved"]] == [
        "trade_loop",
        "market_data",
        "model_training",
    ]
    assert "strategy_closed_loop" in {item["key"] for item in payload["issue_ledger"]["fixed"]}
    assert "只读巡检" in payload["safety_note"]
    assert "人工确认" in payload["safety_note"]


@pytest.mark.asyncio
async def test_audit_maybe_async_times_out_slow_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def slow_audit() -> dict[str, Any]:
        import asyncio

        await asyncio.sleep(0.05)
        return system_audit._audit_card("slow", "slow", "ok", "late")

    monkeypatch.setattr(system_audit, "SYSTEM_AUDIT_SECTION_TIMEOUT_SECONDS", 0.01)

    with pytest.raises(TimeoutError):
        await system_audit._audit_maybe_async(slow_audit)


@pytest.mark.asyncio
async def test_system_audit_status_wraps_failed_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        system_audit,
        "_shadow_missed_opportunity_audit",
        _async_card("shadow_missed_opportunity", "ok", "Shadow missed loop normal"),
    )
    monkeypatch.setattr(
        system_audit,
        "_trade_execution_contract_audit",
        _async_card("trade_execution_contract", "ok", "Trade execution contract normal"),
    )

    async def failed_audit() -> dict[str, Any]:
        raise RuntimeError("boom")

    monkeypatch.setattr(system_audit, "_trade_loop_audit", failed_audit)
    monkeypatch.setattr(
        system_audit,
        "_okx_reconciliation_audit",
        _async_card("okx_reconciliation", "ok", "OKX 对账正常"),
    )
    monkeypatch.setattr(
        system_audit,
        "_position_price_integrity_audit",
        _async_card("position_price_integrity", "ok", "持仓价格一致"),
    )
    monkeypatch.setattr(
        system_audit,
        "_market_data_audit",
        _async_card("market_data", "ok", "行情正常"),
    )
    monkeypatch.setattr(
        system_audit,
        "_strategy_quality_audit",
        _async_card("strategy_quality", "ok", "策略质量正常"),
    )
    monkeypatch.setattr(
        system_audit,
        "_strategy_closed_loop_audit",
        _async_card("strategy_closed_loop", "ok", "策略闭环正常"),
    )
    monkeypatch.setattr(
        system_audit,
        "_model_training_audit",
        _async_card("model_training", "ok", "模型正常"),
    )
    monkeypatch.setattr(
        system_audit,
        "_model_expert_health_audit",
        _async_card("model_expert_health", "ok", "模型/专家体检正常"),
    )
    monkeypatch.setattr(
        system_audit,
        "_model_expert_competition_audit",
        _async_card("model_expert_competition", "ok", "模型/专家竞赛正常"),
    )
    monkeypatch.setattr(
        system_audit,
        "_crypto_feature_coverage_audit",
        _async_card("crypto_feature_coverage", "ok", "Crypto 特征覆盖正常"),
    )
    monkeypatch.setattr(
        system_audit,
        "_model_dynamic_routing_audit",
        _async_card("model_dynamic_routing", "ok", "动态路由正常"),
    )
    monkeypatch.setattr(
        system_audit,
        "_strategy_gate_contract_audit",
        lambda: system_audit._audit_card(
            "strategy_gate_contract", "策略门槛契约", "ok", "运行时契约正常"
        ),
    )
    monkeypatch.setattr(
        system_audit,
        "_source_visible_text_audit",
        lambda: system_audit._audit_card(
            "visible_text_encoding", "中文显示与乱码回归", "ok", "无乱码"
        ),
    )
    monkeypatch.setattr(
        system_audit,
        "_runtime_text_integrity_audit",
        lambda: system_audit._audit_card(
            "runtime_text_integrity", "运行时文本完整性", "ok", "无新增乱码"
        ),
    )

    payload = await system_audit.system_audit_status()

    assert payload["status"] == "warning"
    assert payload["summary"]["warning"] == 1
    assert payload["root_causes"][0]["title"] == "巡检模块"
    assert payload["root_causes"][0]["key"] == "trade_loop"
    assert payload["root_causes"][0]["owner_path"] == "services/trading_service.py"
    assert payload["root_causes"][0]["severity"] == "warning"
    assert payload["cards"][0]["details"]["error"]


@pytest.mark.asyncio
async def test_model_training_audit_does_not_run_full_self_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_data_collection_status() -> dict[str, Any]:
        return {
            "training": {
                "local_ai_tools": {
                    "available": True,
                    "status": "ready",
                    "shadow_sample_count": 123,
                    "trade_sample_count": 7,
                    "text_sentiment_sample_count": 11,
                },
                "governance": {"status": "clean"},
            },
            "sources": [],
        }

    async def fake_runtime_status() -> dict[str, Any]:
        return {
            "ai_models": [
                {
                    "model": "qwen3-14b-trade",
                    "available": True,
                    "endpoint_ok": True,
                    "model_available": True,
                    "latency_ms": 12,
                }
            ],
            "local_ai_tools": {"available": True, "api_base": "http://127.0.0.1:18001"},
        }

    monkeypatch.setattr(
        system_audit.data_collection_api,
        "get_data_collection_status",
        fake_data_collection_status,
    )
    monkeypatch.setattr(system_audit, "collect_platform_runtime_status", fake_runtime_status)

    card = await system_audit._model_training_audit()

    assert not hasattr(system_audit, "system_self_check")
    assert card["status"] == "ok"
    assert card["details"]["runtime_probe"]["status"] == "ok"


@pytest.mark.asyncio
async def test_model_expert_health_audit_reports_read_only_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModelExpertHealthService:
        async def report(self, *, hours: int = 72, limit: int = 1000) -> dict[str, Any]:
            return {
                "audit_only": True,
                "live_weight_mutation": False,
                "summary": {
                    "components": 3,
                    "recommended_state_counts": {
                        "keep": 1,
                        "reduce": 1,
                        "disable": 1,
                    },
                },
                "components": {
                    "trend_expert": {
                        "recommended_state": "reduce",
                        "state_reasons": ["negative_adopted_pnl"],
                        "stability": {"json_error_rate": 0.0, "no_return_rate": 0.0},
                    },
                    "risk_expert": {
                        "recommended_state": "disable",
                        "state_reasons": ["json_error_rate_high"],
                        "stability": {"json_error_rate": 0.6, "no_return_rate": 0.6},
                    },
                },
            }

    monkeypatch.setattr(
        system_audit,
        "ModelExpertHealthService",
        lambda: FakeModelExpertHealthService(),
    )

    card = await system_audit._model_expert_health_audit()

    assert card["key"] == "model_expert_health"
    assert card["status"] == "warning"
    assert card["details"]["audit_only"] is True
    assert card["details"]["live_weight_mutation"] is False
    assert card["details"]["recommended_state_counts"]["disable"] == 1
    assert card["details"]["top_components"][0]["name"] == "risk_expert"
    assert any(item["label"] == "需降权" for item in card["evidence"])


@pytest.mark.asyncio
async def test_model_expert_health_status_endpoint_is_read_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModelExpertHealthService:
        async def report(self, *, hours: int = 72, limit: int = 1200) -> dict[str, Any]:
            return {
                "audit_only": False,
                "live_weight_mutation": True,
                "windows_hours": [24, 72],
                "summary": {"components": 1},
                "components": {"trend_expert": {"recommended_state": "keep"}},
            }

    monkeypatch.setattr(
        system_audit,
        "ModelExpertHealthService",
        lambda: FakeModelExpertHealthService(),
    )

    report = await system_audit.model_expert_health_status(hours=24, limit=200)

    assert report["audit_only"] is True
    assert report["live_weight_mutation"] is False
    assert report["components"]["trend_expert"]["recommended_state"] == "keep"


@pytest.mark.asyncio
async def test_model_expert_competition_audit_never_allows_live_weight_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModelExpertCompetitionService:
        async def report(self, *, hours: int = 72, limit: int = 1200) -> dict[str, Any]:
            return {
                "audit_only": True,
                "live_weight_mutation": False,
                "can_apply_live_weight": False,
                "baseline": {"sample_count": 12, "net_pnl_pct": -0.3},
                "layers": {
                    "offline_replay": {"baseline_available": True, "sample_count": 12},
                    "shadow_competition": {"available": True, "sample_count": 8},
                    "sim_ab": {"available": False, "sample_count": 0},
                },
                "blocking_reasons": [],
                "competitors": {
                    "trend_expert": {
                        "recommended_weight_action": "increase_shadow_weight",
                        "baseline_delta": {"net_pnl_pct": 0.8},
                        "can_apply_live_weight": True,
                    },
                    "risk_expert": {
                        "recommended_weight_action": "pause_shadow",
                        "baseline_delta": {"net_pnl_pct": -0.5},
                        "can_apply_live_weight": False,
                    },
                },
            }

    monkeypatch.setattr(
        system_audit,
        "ModelExpertCompetitionService",
        lambda: FakeModelExpertCompetitionService(),
    )

    card = await system_audit._model_expert_competition_audit()
    endpoint_report = await system_audit.model_expert_competition_status(hours=24, limit=200)

    assert card["key"] == "model_expert_competition"
    assert card["status"] == "warning"
    assert card["details"]["can_apply_live_weight"] is False
    assert card["details"]["live_weight_mutation"] is False
    assert card["details"]["recommended_weight_action_counts"]["pause_shadow"] == 1
    assert card["details"]["top_competitors"][0]["can_apply_live_weight"] is False
    assert endpoint_report["audit_only"] is True
    assert endpoint_report["live_weight_mutation"] is False
    assert endpoint_report["can_apply_live_weight"] is False


@pytest.mark.asyncio
async def test_model_dynamic_routing_audit_and_endpoint_force_read_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDynamicRoutingService:
        async def report(self, *, hours: int = 72, limit: int = 1200) -> dict[str, Any]:
            return {
                "audit_only": False,
                "live_route_mutation": True,
                "can_apply_live_route": True,
                "summary": {
                    "route_plan_count": 3,
                    "shadow_only_count": 2,
                    "estimated_call_reduction": 4,
                    "unsafe_live_mutation_attempts": 1,
                },
                "blocking_reason_counts": {"competition_baseline_missing": 2},
                "safety_observations": {"weak_evidence_executed_count": 1},
            }

    monkeypatch.setattr(
        system_audit,
        "ModelDynamicRoutingService",
        lambda: FakeDynamicRoutingService(),
    )

    card = await system_audit._model_dynamic_routing_audit()
    endpoint_report = await system_audit.model_dynamic_routing_status(hours=24, limit=200)

    assert card["key"] == "model_dynamic_routing"
    assert card["status"] == "warning"
    assert card["details"]["audit_only"] is True
    assert card["details"]["live_route_mutation"] is False
    assert card["details"]["can_apply_live_route"] is False
    assert card["details"]["unsafe_live_mutation_attempts"] == 1
    assert endpoint_report["audit_only"] is True
    assert endpoint_report["live_route_mutation"] is False
    assert endpoint_report["can_apply_live_route"] is False


@pytest.mark.asyncio
async def test_shadow_missed_opportunity_audit_and_endpoint_force_read_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []

    class FakeShadowMissedOpportunityService:
        async def report(self, *, hours: int = 72, limit: int = 1200) -> dict[str, Any]:
            calls.append((hours, limit))
            return {
                "audit_only": False,
                "live_entry_mutation": True,
                "can_bypass_risk_controls": True,
                "weak_evidence_execution_allowed": True,
                "global_missed_count_can_drive_entries": True,
                "summary": {
                    "missed_count": 7,
                    "adopted_count": 1,
                    "probe_count": 1,
                    "blocked_count": 5,
                    "weak_evidence_executed_count": 1,
                },
                "blocked_reason_counts": {"high_risk_evidence": 2},
                "probe_candidates": [
                    {"symbol": "BTC/USDT", "side": "long", "status": "probe_ready"}
                ],
                "adopted": [],
                "blocked": [],
            }

    monkeypatch.setattr(
        system_audit,
        "ShadowMissedOpportunityClosedLoopService",
        lambda: FakeShadowMissedOpportunityService(),
    )

    card = await system_audit._shadow_missed_opportunity_audit()
    endpoint_report = await system_audit.shadow_missed_opportunity_status(hours=24, limit=200)

    assert calls == [(24, 200), (24, 200)]
    assert card["key"] == "shadow_missed_opportunity"
    assert card["status"] == "warning"
    assert card["details"]["audit_only"] is True
    assert card["details"]["live_entry_mutation"] is False
    assert card["details"]["can_bypass_risk_controls"] is False
    assert card["details"]["weak_evidence_execution_allowed"] is False
    assert card["details"]["global_missed_count_can_drive_entries"] is False
    assert card["details"]["summary"]["weak_evidence_executed_count"] == 1
    assert endpoint_report["audit_only"] is True
    assert endpoint_report["live_entry_mutation"] is False
    assert endpoint_report["can_bypass_risk_controls"] is False
    assert endpoint_report["weak_evidence_execution_allowed"] is False
    assert endpoint_report["global_missed_count_can_drive_entries"] is False


@pytest.mark.asyncio
async def test_trade_execution_contract_audit_and_endpoint_force_read_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []

    class FakeTradeExecutionContractService:
        async def report(self, *, hours: int = 24, limit: int = 500) -> dict[str, Any]:
            calls.append((hours, limit))
            return {
                "audit_only": False,
                "live_entry_mutation": True,
                "live_exit_mutation": True,
                "can_bypass_risk_controls": True,
                "summary": {
                    "executed_entry_count": 4,
                    "missing_entry_explanation_count": 1,
                    "missing_sizing_explanation_count": 1,
                    "weak_evidence_executed_count": 1,
                    "negative_expected_executed_count": 1,
                    "fast_loss_count": 2,
                    "fast_loss_without_strong_exit_count": 1,
                    "reentry_without_strong_unlock_count": 1,
                    "contract_violation_count": 5,
                },
                "violation_reason_counts": {
                    "weak_evidence_executed": 1,
                    "non_positive_expected_net_executed": 1,
                    "fast_loss_without_strong_exit": 1,
                    "reentry_without_strong_unlock": 1,
                },
                "entry_explanations": [{"decision_id": 1, "violations": []}],
                "fast_loss_samples": [{"id": 10, "symbol": "BTC/USDT"}],
                "violations": [{"reason": "weak_evidence_executed"}],
            }

    monkeypatch.setattr(
        system_audit,
        "TradeExecutionContractService",
        lambda: FakeTradeExecutionContractService(),
    )

    card = await system_audit._trade_execution_contract_audit()
    endpoint_report = await system_audit.trade_execution_contract_status(hours=24, limit=200)

    assert calls == [(24, 500), (24, 200)]
    assert card["key"] == "trade_execution_contract"
    assert card["status"] == "critical"
    assert card["details"]["audit_only"] is True
    assert card["details"]["live_entry_mutation"] is False
    assert card["details"]["live_exit_mutation"] is False
    assert card["details"]["can_bypass_risk_controls"] is False
    assert card["details"]["summary"]["contract_violation_count"] == 5
    evidence = {item["label"]: item["value"] for item in card["evidence"]}
    assert evidence["弱证据执行"] == 1
    assert evidence["负期望执行"] == 1
    assert evidence["快亏缺强证据"] == 1
    assert evidence["复开缺解锁"] == 1
    assert endpoint_report["audit_only"] is True
    assert endpoint_report["live_entry_mutation"] is False
    assert endpoint_report["live_exit_mutation"] is False
    assert endpoint_report["can_bypass_risk_controls"] is False


@pytest.mark.asyncio
async def test_trade_execution_contract_audit_treats_historical_only_as_observing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_started_at = datetime(2026, 6, 22, 2, 0, tzinfo=UTC)
    calls: list[tuple[int, int, datetime | None]] = []

    historical_report = {
        "audit_only": False,
        "live_entry_mutation": True,
        "live_exit_mutation": True,
        "can_bypass_risk_controls": True,
        "summary": {
            "executed_entry_count": 4,
            "missing_entry_explanation_count": 1,
            "missing_sizing_explanation_count": 0,
            "weak_evidence_executed_count": 2,
            "negative_expected_executed_count": 0,
            "fast_loss_count": 1,
            "fast_loss_without_strong_exit_count": 0,
            "reentry_without_strong_unlock_count": 0,
            "small_size_without_reason_count": 0,
            "contract_violation_count": 3,
        },
        "violation_reason_counts": {"weak_evidence_executed": 2},
        "entry_explanations": [{"decision_id": 1, "violations": ["weak_evidence_executed"]}],
        "fast_loss_samples": [{"id": 10, "symbol": "BTC/USDT"}],
        "violations": [{"reason": "weak_evidence_executed"}],
    }
    current_report = {
        "audit_only": False,
        "live_entry_mutation": True,
        "live_exit_mutation": True,
        "can_bypass_risk_controls": True,
        "summary": {
            "decision_count": 12,
            "executed_entry_count": 0,
            "missing_entry_explanation_count": 0,
            "missing_sizing_explanation_count": 0,
            "weak_evidence_executed_count": 0,
            "negative_expected_executed_count": 0,
            "fast_loss_count": 0,
            "fast_loss_without_strong_exit_count": 0,
            "reentry_without_strong_unlock_count": 0,
            "small_size_without_reason_count": 0,
            "contract_violation_count": 0,
        },
        "violation_reason_counts": {},
        "entry_explanations": [],
        "fast_loss_samples": [],
        "violations": [],
        "query_policy": {"db_time_filter": True},
    }

    class FakeTradeExecutionContractService:
        async def report(
            self,
            *,
            hours: int = 24,
            limit: int = 500,
            since: datetime | None = None,
        ) -> dict[str, Any]:
            calls.append((hours, limit, since))
            return current_report if since is not None else historical_report

    monkeypatch.setattr(
        system_audit,
        "_load_trading_runtime_audit_window",
        lambda: {
            "available": True,
            "started_at": runtime_started_at,
            "started_at_iso": runtime_started_at.isoformat(),
            "heartbeat_at": datetime(2026, 6, 22, 2, 5, tzinfo=UTC),
            "heartbeat_at_iso": "2026-06-22T02:05:00+00:00",
            "running": True,
            "mode": "paper",
            "decision_interval": 30,
        },
    )
    monkeypatch.setattr(
        system_audit,
        "TradeExecutionContractService",
        lambda: FakeTradeExecutionContractService(),
    )

    card = await system_audit._trade_execution_contract_audit()
    ledger = system_audit._issue_ledger_from_cards([card])

    assert calls == [(24, 500, None), (24, 500, runtime_started_at)]
    assert card["status"] == "warning"
    assert card["details"]["summary"]["contract_violation_count"] == 3
    assert card["details"]["current_summary"]["contract_violation_count"] == 0
    assert card["details"]["current_runtime_window"] == {
        "available": True,
        "started_at": "2026-06-22T02:00:00+00:00",
        "heartbeat_at": "2026-06-22T02:05:00+00:00",
        "running": True,
        "mode": "paper",
        "decision_interval": 30,
        "decision_count": 12,
        "executed_entry_count": 0,
        "weak_evidence_executed_count": 0,
        "negative_expected_executed_count": 0,
        "fast_loss_without_strong_exit_count": 0,
        "reentry_without_strong_unlock_count": 0,
        "soft_violation_count": 0,
        "hard_violation_count": 0,
        "contract_violation_count": 0,
        "historical_legacy_issues": True,
    }
    assert ledger["summary"] == {"fixed": 0, "unresolved": 0, "observing": 1, "total": 1}
    assert ledger["observing"][0]["key"] == "trade_execution_contract"


@pytest.mark.asyncio
async def test_crypto_feature_coverage_audit_and_endpoint_force_read_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCryptoFeatureCoverageService:
        async def report(self, *, hours: int = 24, limit: int = 1000) -> dict[str, Any]:
            return {
                "audit_only": False,
                "live_signal_mutation": True,
                "can_missing_features_drive_live_entry": True,
                "feature_defaults_are_neutral": False,
                "status": "warning",
                "features": [
                    {
                        "key": "funding_rate",
                        "status": "missing",
                        "live_entry_influence": "eligible",
                        "reasons": ["default_zero_without_presence_flag"],
                    }
                ],
                "missing_features": ["funding_rate"],
                "stale_features": [],
                "neutralized_features": ["funding_rate"],
                "symbols_observed": ["BTC/USDT"],
                "feature_contribution_policy": {"missing_feature_policy": "unsafe"},
            }

    monkeypatch.setattr(
        system_audit,
        "CryptoFeatureCoverageService",
        lambda: FakeCryptoFeatureCoverageService(),
    )

    card = await system_audit._crypto_feature_coverage_audit()
    endpoint_report = await system_audit.crypto_feature_coverage_status(hours=12, limit=200)

    assert card["key"] == "crypto_feature_coverage"
    assert card["status"] == "warning"
    assert card["details"]["audit_only"] is True
    assert card["details"]["live_signal_mutation"] is False
    assert card["details"]["can_missing_features_drive_live_entry"] is False
    assert card["details"]["feature_defaults_are_neutral"] is True
    assert card["details"]["missing_features"] == ["funding_rate"]
    assert endpoint_report["audit_only"] is True
    assert endpoint_report["live_signal_mutation"] is False
    assert endpoint_report["can_missing_features_drive_live_entry"] is False
    assert endpoint_report["feature_defaults_are_neutral"] is True
    assert (
        endpoint_report["feature_contribution_policy"]["missing_feature_policy"]
        == "neutral_blocked"
    )


@pytest.mark.asyncio
async def test_model_training_optional_sources_are_observing_not_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_data_collection_status() -> dict[str, Any]:
        return {
            "training": {
                "local_ai_tools": {
                    "available": True,
                    "status": "learning_only",
                    "shadow_sample_count": 19979,
                    "trade_sample_count": 1588,
                    "text_sentiment_sample_count": 8000,
                },
                "governance": {"status": "ok"},
            },
            "sources": [
                {
                    "key": "cryptopanic",
                    "name": "CryptoPanic",
                    "group": "api",
                    "enabled": False,
                    "status": "not_configured",
                },
                {
                    "key": "coinmarketcal",
                    "name": "CoinMarketCal",
                    "group": "api",
                    "enabled": False,
                    "status": "not_configured",
                },
                {
                    "key": "newsapi",
                    "name": "NewsAPI",
                    "group": "api",
                    "enabled": False,
                    "status": "not_configured",
                },
            ],
        }

    async def fake_runtime_status() -> dict[str, Any]:
        return {
            "ai_models": [
                {"model": "qwen3-14b-trade", "available": True},
                {"model": "deepseek-r1-14b-risk", "available": True},
            ],
            "local_ai_tools": {"available": True, "api_base": "http://127.0.0.1:18001"},
        }

    monkeypatch.setattr(
        system_audit.data_collection_api,
        "get_data_collection_status",
        fake_data_collection_status,
    )
    monkeypatch.setattr(system_audit, "collect_platform_runtime_status", fake_runtime_status)

    card = await system_audit._model_training_audit()
    ledger = system_audit._issue_ledger_from_cards([card])

    assert card["status"] == "warning"
    assert card["details"]["hard_failure"] is False
    assert card["details"]["optional_source_warning_count"] == 3
    assert ledger["summary"] == {"fixed": 0, "unresolved": 0, "observing": 1, "total": 1}
    assert ledger["observing"][0]["key"] == "model_training"


@pytest.mark.asyncio
async def test_strategy_quality_audit_reports_short_adjustment_samples(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    from db.session import close_db, get_session_ctx, init_db
    from models.decision import AIDecision

    await close_db()
    db_path = tmp_path / "audit.db"
    now = datetime(2026, 6, 22, 3, 0, tzinfo=UTC)
    monkeypatch.setattr(
        system_audit.settings,
        "database_url",
        f"sqlite+aiosqlite:///{db_path.as_posix()}",
    )
    monkeypatch.setattr(system_audit, "_now", lambda: now)

    await init_db()
    try:
        raw = {
            "opportunity_score": {
                "expected_net_return_pct": 1.4,
                "evidence_score": {
                    "tier": "normal",
                    "short_evidence_adjustment": {
                        "mode": "strong_current_short_evidence",
                        "score_offset": 0.0,
                        "size_multiplier": 1.0,
                    },
                },
            }
        }
        async with get_session_ctx() as session:
            session.add(
                AIDecision(
                    model_name="test_model",
                    symbol="BTC/USDT",
                    action="short",
                    reasoning="short evidence release",
                    position_size_pct=0.04,
                    confidence=0.9,
                    raw_llm_response=raw,
                    was_executed=False,
                    created_at=now - timedelta(minutes=5),
                )
            )

        card = await system_audit._strategy_quality_audit()

        assert card["details"]["short_released_adjustment_count"] == 1
        assert card["details"]["short_conservative_adjustment_count"] == 0
        assert card["details"]["short_released_adjustment_samples"][0]["symbol"] == "BTC/USDT"
        evidence = {item["label"]: item["value"] for item in card["evidence"]}
        assert evidence["做空强证据放开"] == 1
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_model_training_unconfigured_local_tools_are_observing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_data_collection_status() -> dict[str, Any]:
        return {
            "training": {
                "local_ai_tools": {
                    "available": False,
                    "status": "disabled",
                    "shadow_sample_count": 0,
                    "trade_sample_count": 0,
                    "text_sentiment_sample_count": 0,
                },
                "governance": {"status": "error"},
            },
            "sources": [],
        }

    async def fake_runtime_status() -> dict[str, Any]:
        return {
            "ai_models": [],
            "local_ai_tools": {"available": False, "configured": False},
        }

    monkeypatch.setattr(
        system_audit.data_collection_api,
        "get_data_collection_status",
        fake_data_collection_status,
    )
    monkeypatch.setattr(system_audit, "collect_platform_runtime_status", fake_runtime_status)

    card = await system_audit._model_training_audit()
    ledger = system_audit._issue_ledger_from_cards([card])

    assert card["status"] == "warning"
    assert card["details"]["hard_failure"] is False
    assert card["details"]["observing"] is True
    assert ledger["summary"] == {"fixed": 0, "unresolved": 0, "observing": 1, "total": 1}


@pytest.mark.asyncio
async def test_model_training_auth_failure_remains_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_data_collection_status() -> dict[str, Any]:
        return {
            "training": {
                "local_ai_tools": {
                    "available": False,
                    "status": "error",
                    "shadow_sample_count": 0,
                    "trade_sample_count": 0,
                    "text_sentiment_sample_count": 0,
                },
                "governance": {"status": "ok"},
            },
            "sources": [],
        }

    async def fake_runtime_status() -> dict[str, Any]:
        return {
            "ai_models": [{"model": "qwen3-14b-trade", "available": True}],
            "local_ai_tools": {
                "available": False,
                "api_base": "http://127.0.0.1:18001",
                "health": {"status_category": "auth_failed"},
            },
        }

    monkeypatch.setattr(
        system_audit.data_collection_api,
        "get_data_collection_status",
        fake_data_collection_status,
    )
    monkeypatch.setattr(system_audit, "collect_platform_runtime_status", fake_runtime_status)

    card = await system_audit._model_training_audit()
    ledger = system_audit._issue_ledger_from_cards([card])

    assert card["status"] == "critical"
    assert card["details"]["hard_failure"] is True
    assert ledger["summary"] == {"fixed": 0, "unresolved": 1, "observing": 0, "total": 1}
    assert ledger["unresolved"][0]["key"] == "model_training"


@pytest.mark.asyncio
async def test_model_training_runtime_probe_timeout_is_observing_when_training_is_usable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_data_collection_status() -> dict[str, Any]:
        return {
            "training": {
                "local_ai_tools": {
                    "available": True,
                    "status": "learning_only",
                    "shadow_sample_count": 19979,
                    "trade_sample_count": 1588,
                    "text_sentiment_sample_count": 8000,
                },
                "governance": {"status": "ok"},
            },
            "sources": [],
        }

    async def timeout_runtime_status() -> dict[str, Any]:
        raise TimeoutError()

    monkeypatch.setattr(
        system_audit.data_collection_api,
        "get_data_collection_status",
        fake_data_collection_status,
    )
    monkeypatch.setattr(system_audit, "collect_platform_runtime_status", timeout_runtime_status)

    card = await system_audit._model_training_audit()
    ledger = system_audit._issue_ledger_from_cards([card])

    assert card["status"] == "warning"
    assert card["details"]["runtime_probe"]["timeout"] is True
    assert card["details"]["hard_failure"] is False
    assert ledger["summary"] == {"fixed": 0, "unresolved": 0, "observing": 1, "total": 1}
    assert ledger["observing"][0]["key"] == "model_training"


@pytest.mark.asyncio
async def test_okx_reconciliation_audit_reuses_short_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def fake_collect_missing_closed_position_plans(days: int) -> list[Any]:
        nonlocal calls
        calls += 1
        assert days == 14
        return [
            SimpleNamespace(
                symbol="PROS/USDT",
                side="long",
                quantity=1.0,
                realized_pnl=-0.82,
                close_order_id="close-1",
                closed_at=datetime(2026, 6, 22, tzinfo=UTC),
            )
        ]

    monkeypatch.setattr(system_audit, "_okx_reconciliation_cache", None)
    monkeypatch.setattr(
        system_audit,
        "collect_missing_closed_position_plans",
        fake_collect_missing_closed_position_plans,
    )

    first = await system_audit._okx_reconciliation_audit()
    second = await system_audit._okx_reconciliation_audit()

    assert calls == 1
    assert first["details"]["cache"]["hit"] is False
    assert second["details"]["cache"]["hit"] is True
    assert second["details"]["missing_closed_positions"] == 1


@pytest.mark.asyncio
async def test_okx_reconciliation_timeout_is_observing_not_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def slow_collect_missing_closed_position_plans(days: int) -> list[Any]:
        raise TimeoutError()

    monkeypatch.setattr(system_audit, "_okx_reconciliation_cache", None)
    monkeypatch.setattr(
        system_audit,
        "collect_missing_closed_position_plans",
        slow_collect_missing_closed_position_plans,
    )

    card = await system_audit._okx_reconciliation_audit()
    ledger = system_audit._issue_ledger_from_cards([card])

    assert card["status"] == "warning"
    assert card["details"]["timeout"] is True
    assert ledger["summary"] == {"fixed": 0, "unresolved": 0, "observing": 1, "total": 1}
    assert ledger["observing"][0]["key"] == "okx_reconciliation"


@pytest.mark.asyncio
async def test_trade_loop_recent_restart_without_decisions_is_observing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    from db.session import close_db, init_db

    await close_db()
    db_path = tmp_path / "audit.db"
    now = datetime(2026, 6, 22, 4, 30, tzinfo=UTC)
    started_at = now - timedelta(seconds=75)
    monkeypatch.setattr(
        system_audit.settings,
        "database_url",
        f"sqlite+aiosqlite:///{db_path.as_posix()}",
    )
    monkeypatch.setattr(system_audit, "_now", lambda: now)
    monkeypatch.setattr(
        system_audit,
        "_load_trading_runtime_audit_window",
        lambda: {
            "available": True,
            "started_at": started_at,
            "started_at_iso": started_at.isoformat(),
            "heartbeat_at": now - timedelta(seconds=5),
            "heartbeat_at_iso": (now - timedelta(seconds=5)).isoformat(),
            "running": True,
            "mode": "paper",
            "decision_interval": 30,
        },
    )

    await init_db()
    try:
        card = await system_audit._trade_loop_audit()
        ledger = system_audit._issue_ledger_from_cards([card])

        assert card["status"] == "warning"
        assert card["details"]["cold_start"] is True
        assert ledger["summary"] == {"fixed": 0, "unresolved": 0, "observing": 1, "total": 1}
        assert ledger["observing"][0]["key"] == "trade_loop"
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_strategy_closed_loop_audit_separates_active_runtime_window(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    from db.session import close_db, get_session_ctx, init_db
    from models.decision import AIDecision
    from models.trade import Order

    await close_db()
    db_path = tmp_path / "audit.db"
    monkeypatch.setattr(
        system_audit.settings,
        "database_url",
        f"sqlite+aiosqlite:///{db_path.as_posix()}",
    )
    monkeypatch.setattr(system_audit.settings, "_data_dir", tmp_path, raising=False)

    await init_db()
    try:
        old_created_at = datetime(2026, 6, 22, 1, 0, tzinfo=UTC)
        runtime_started_at = datetime(2026, 6, 22, 2, 0, tzinfo=UTC)
        monkeypatch.setattr(
            system_audit,
            "_load_trading_runtime_audit_window",
            lambda: {
                "available": True,
                "started_at": runtime_started_at,
                "started_at_iso": runtime_started_at.isoformat(),
                "heartbeat_at": datetime(2026, 6, 22, 2, 5, tzinfo=UTC),
                "heartbeat_at_iso": "2026-06-22T02:05:00+00:00",
                "running": True,
                "mode": "paper",
                "decision_interval": 30,
            },
        )
        raw = {
            "opportunity_score": {
                "expected_net_return_pct": 0.7,
                "evidence_score": {
                    "tier": "weak_conflict_probe",
                    "components": [
                        {
                            "source": "ml",
                            "status": "ignored",
                            "reason": "ML 学习观察模式",
                        }
                    ],
                },
            }
        }
        async with get_session_ctx() as session:
            old_decision = AIDecision(
                model_name="test_model",
                symbol="OLD/USDT",
                action="long",
                reasoning="old weak entry",
                position_size_pct=0.001,
                confidence=0.1,
                raw_llm_response=raw,
                was_executed=True,
                created_at=old_created_at,
            )
            new_decision = AIDecision(
                model_name="test_model",
                symbol="NEW/USDT",
                action="hold",
                reasoning="new hold",
                position_size_pct=0.0,
                confidence=0.1,
                raw_llm_response={},
                was_executed=False,
                created_at=runtime_started_at,
            )
            session.add_all([old_decision, new_decision])
            await session.flush()
            session.add(
                Order(
                    model_name="test_model",
                    execution_mode="paper",
                    decision_id=old_decision.id,
                    symbol="OLD/USDT",
                    side="long",
                    order_type="market",
                    quantity=1,
                    price=1,
                    status="filled",
                    created_at=old_created_at,
                )
            )

        card = await system_audit._strategy_closed_loop_audit()

        assert card["details"]["weak_executed_count"] == 1
        assert card["details"]["current_runtime_window"]["started_at"] == (
            "2026-06-22T02:00:00+00:00"
        )
        assert card["details"]["current_runtime_window"]["entry_decision_count"] == 0
        assert card["details"]["current_runtime_window"]["weak_executed_count"] == 0
        assert card["details"]["current_runtime_window"]["historical_legacy_issues"] is True
        assert card["details"]["ml_influence_reason"]["top_reasons"][0] == {
            "reason": "ML 学习观察模式",
            "count": 1,
        }
        ledger = system_audit._issue_ledger_from_cards([card])
        assert ledger["summary"] == {"fixed": 0, "unresolved": 0, "observing": 1, "total": 1}
        assert ledger["observing"][0]["key"] == "strategy_closed_loop"
        assert "历史" in ledger["observing"][0]["state_label"]
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_runtime_text_integrity_audit_reports_recent_suspects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_collect_runtime_text_integrity_report(
        *, hours: int, limit_per_table: int, example_limit: int
    ) -> dict[str, Any]:
        assert hours == system_audit.AUDIT_WINDOWS["strategy_hours"]
        assert limit_per_table > 0
        assert example_limit > 0
        return {
            "status": "warning",
            "scanned_records": 5,
            "suspected_records": 2,
            "suspected_fields": 3,
            "repairable_count": 1,
            "by_table": {"ai_decisions": {"suspected_records": 2}},
            "examples": [{"table": "ai_decisions", "field": "execution_reason"}],
            "policy": {"dry_run": True, "mutates_database": False},
        }

    monkeypatch.setattr(
        system_audit,
        "collect_runtime_text_integrity_report",
        fake_collect_runtime_text_integrity_report,
    )

    card = await system_audit._runtime_text_integrity_audit()

    assert card["key"] == "runtime_text_integrity"
    assert card["status"] == "warning"
    assert card["details"]["suspected_records"] == 2
    assert card["details"]["policy"]["dry_run"] is True
    assert {item["label"]: item["value"] for item in card["evidence"]} == {
        "扫描记录": 5,
        "疑似记录": 2,
        "疑似字段": 3,
        "可自动修复字段": 1,
    }
    assert "写入边界" in " ".join(card["next_actions"])


def test_strategy_gate_contract_audit_tracks_parameterized_strategy_constants() -> None:
    card = system_audit._strategy_gate_contract_audit()

    assert card["status"] == "ok"
    assert card["details"]["trading_parameter_version"] == DEFAULT_TRADING_PARAMS.version
    assert card["details"]["hidden_strategy_constant_count"] == 0
    assert card["details"]["ensemble_top_level_constant_count"] > 0
    assert "ACTION_SCORE" in card["details"]["allowed_top_level_constants"]
    assert any(item["label"] == "策略参数版本" for item in card["evidence"])


def test_issue_ledger_moves_strategy_quality_to_observing_when_closed_loop_is_legacy() -> None:
    strategy_quality = system_audit._audit_card(
        "strategy_quality",
        "策略质量",
        "warning",
        "存在快亏平、弱证据误执行或多数开仓候选净收益为负。",
    )
    strategy_closed_loop = system_audit._audit_card(
        "strategy_closed_loop",
        "策略闭环审计",
        "warning",
        "24小时历史窗口仍有遗留问题；当前运行窗口暂未复现硬执行错误，需继续观察新样本。",
        details={
            "current_runtime_window": {
                "historical_legacy_issues": True,
                "weak_executed_count": 0,
                "fast_loss_under_15m_count": 0,
            },
            "diagnostics": {
                "current_weak_executed": False,
                "current_no_high_quality_entries": False,
                "current_fast_loss_cluster": False,
                "current_ml_not_effective": False,
            },
        },
    )

    ledger = system_audit._issue_ledger_from_cards([strategy_quality, strategy_closed_loop])

    assert ledger["summary"] == {"fixed": 0, "unresolved": 0, "observing": 2, "total": 2}
    assert {item["key"] for item in ledger["observing"]} == {
        "strategy_quality",
        "strategy_closed_loop",
    }
