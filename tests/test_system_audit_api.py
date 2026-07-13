from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.trade import Position
from services.trading_params import DEFAULT_TRADING_PARAMS
from web_dashboard.api import dashboard as dashboard_api
from web_dashboard.api import system_audit

AuditFactory = Callable[[], Awaitable[dict[str, Any]]]


@pytest.fixture(autouse=True)
def _stable_model_training_scheduler_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        system_audit.MODEL_TRAINING_STATE_STORE,
        "read",
        lambda: {
            "status": "ok",
            "state_file_available": True,
            "heartbeat_stale": False,
            "training_timeout_exceeded": False,
            "models": {},
            "schedulers": {},
        },
    )


def _trade_contract_details_ok() -> dict[str, Any]:
    return {
        "audit_only": True,
        "read_only": True,
        "live_entry_mutation": False,
        "live_exit_mutation": False,
        "can_bypass_risk_controls": False,
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
        "summary": {
            "decision_count": 12,
            "executed_entry_count": 2,
            "entry_contract_ready_count": 2,
            "executed_exit_count": 1,
            "exit_contract_ready_count": 1,
            "contract_violation_count": 0,
            "realized_net_pnl_usdt": 1.2,
        },
        "violation_reason_counts": {},
    }












def _patch_okx_daily_report_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    payload: dict[str, Any] | None = None,
) -> None:
    data_dir = tmp_path / "data"
    latest_path = data_dir / "okx_daily_reconciliation_reports" / "latest.json"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    if payload is not None:
        latest_path.write_text(json.dumps(payload), encoding="utf-8")

    class SettingsProxy:
        def __init__(self, wrapped: Any) -> None:
            self._wrapped = wrapped
            self.data_dir = data_dir

        def __getattr__(self, name: str) -> Any:
            return getattr(self._wrapped, name)

    monkeypatch.setattr(system_audit, "settings", SettingsProxy(system_audit.settings))


def _patch_specialist_shadow_report_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    payload: dict[str, Any] | None = None,
) -> None:
    data_dir = tmp_path / "data"
    latest_path = data_dir / "phase3" / "specialist_shadow_evaluation_latest.json"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    if payload is not None:
        latest_path.write_text(json.dumps(payload), encoding="utf-8")

    class SettingsProxy:
        def __init__(self, wrapped: Any) -> None:
            self._wrapped = wrapped
            self.data_dir = data_dir

        def __getattr__(self, name: str) -> Any:
            return getattr(self._wrapped, name)

    monkeypatch.setattr(system_audit, "settings", SettingsProxy(system_audit.settings))


def _patch_phase3_resume_reports_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    *,
    now: datetime,
) -> None:
    data_dir = tmp_path / "data"
    go_no_go_path = data_dir / "phase3_go_no_go_reports" / "latest.json"
    preflight_path = data_dir / "phase3_paper_resume_preflight_reports" / "latest.json"
    go_no_go_path.parent.mkdir(parents=True, exist_ok=True)
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    go_no_go_path.write_text(
        json.dumps(
            {
                "status": "paper_resume_ready",
                "checked_at": now.isoformat(),
                "go_no_go": {
                    "status": "paper_resume_ready",
                    "can_start_paper_with_operator_approval": True,
                    "blockers": [],
                },
            }
        ),
        encoding="utf-8",
    )
    preflight_path.write_text(
        json.dumps(
            {
                "status": "ready",
                "checked_at": now.isoformat(),
                "can_resume_paper": True,
                "blockers": [],
            }
        ),
        encoding="utf-8",
    )

    class SettingsProxy:
        def __init__(self, wrapped: Any) -> None:
            self._wrapped = wrapped
            self.data_dir = data_dir

        def __getattr__(self, name: str) -> Any:
            return getattr(self._wrapped, name)

    monkeypatch.setattr(system_audit, "settings", SettingsProxy(system_audit.settings))


def _async_card(
    key: str,
    status: str,
    summary: str,
    *,
    title: str | None = None,
    evidence_value: int = 1,
) -> AuditFactory:
    async def factory() -> dict[str, Any]:
        if key == "trade_execution_contract":
            details = _trade_contract_details_ok()
        else:
            details = {}
        return system_audit._audit_card(
            key,
            title or key,
            status,
            summary,
            details=details,
            evidence=[{"label": "样本", "value": evidence_value}],
            next_actions=[f"处理 {key}"],
        )

    return factory


def _patch_historical_trade_fact_audit(
    monkeypatch: pytest.MonkeyPatch,
    *,
    report: dict[str, Any] | None = None,
) -> None:
    payload = report or {
        "status": "clean",
        "read_only": True,
        "audit_only": True,
        "cleanup_mode": "quarantine_not_delete",
        "training_policy": "clean_training_view_only",
        "checked_closed_positions": 0,
        "trainable_closed_positions": 0,
        "quarantined_closed_positions": 0,
        "can_delete_history": False,
        "can_apply_repair": False,
    }

    class FakeHistoricalTradeFactAuditService:
        def __init__(self, *, lookback_days: int, limit: int) -> None:
            assert lookback_days == system_audit.HISTORICAL_TRADE_FACT_AUDIT_DAYS
            assert limit == system_audit.HISTORICAL_TRADE_FACT_AUDIT_LIMIT

        async def report(self) -> dict[str, Any]:
            return dict(payload)

    monkeypatch.setattr(
        system_audit,
        "HistoricalTradeFactAuditService",
        FakeHistoricalTradeFactAuditService,
    )


def _patch_artifact_retirement_audit(
    monkeypatch: pytest.MonkeyPatch,
    *,
    report: dict[str, Any] | None = None,
) -> None:
    payload = report or {
        "status": "ready",
        "read_only": True,
        "audit_only": True,
        "raw_artifacts_preserved": True,
        "can_delete_artifacts": False,
        "training_policy": "clean_training_view_only",
        "artifact_count": 1,
        "phase3_compatible_count": 1,
        "retired_or_untrusted_count": 0,
        "status_counts": {"phase3_compatible": 1},
        "artifacts": [],
        "retired_or_untrusted_samples": [],
    }

    class FakeArtifactRetirementAuditService:
        async def report(self) -> dict[str, Any]:
            return dict(payload)

    monkeypatch.setattr(
        system_audit,
        "ArtifactRetirementAuditService",
        FakeArtifactRetirementAuditService,
    )


def _patch_phase3_server_migration_audit(
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: str = "ok",
) -> None:
    monkeypatch.setattr(
        system_audit,
        "_phase3_server_migration_audit",
        _async_card(
            "phase3_server_migration",
            status,
            "Phase 3 server reset/migration gate normal",
            title="Phase 3 server reset/migration gate",
        ),
    )


def _patch_phase3_model_server_readiness_audit(
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: str = "ok",
) -> None:
    monkeypatch.setattr(
        system_audit,
        "_phase3_model_server_readiness_audit",
        _async_card(
            "phase3_model_server_readiness",
            status,
            "Phase 3 quant model-server readiness normal",
            title="Phase 3 quant model-server readiness",
        ),
    )


def _patch_phase3_paper_resume_preflight_audit(
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: str = "ok",
) -> None:
    monkeypatch.setattr(
        system_audit,
        "_phase3_paper_resume_preflight_audit",
        _async_card(
            "phase3_paper_resume_preflight",
            status,
            "Phase 3 paper resume hard gate normal",
            title="Phase 3 paper resume hard gate",
        ),
    )


def _patch_phase3_paper_resume_observation_audit(
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: str = "ok",
) -> None:
    monkeypatch.setattr(
        system_audit,
        "_phase3_paper_resume_observation_audit",
        _async_card(
            "phase3_paper_resume_observation",
            status,
            "Phase 3 paper observation normal",
            title="Phase 3 paper observation",
        ),
    )


def _patch_phase3_stage_handoff_audit(
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: str = "warning",
    stage_status: str = "paper_start_ready",
) -> None:
    async def factory() -> dict[str, Any]:
        return system_audit._audit_card(
            "phase3_stage_handoff",
            "Phase 3 stage handoff",
            status,
            "Phase 3 stage handoff is paper-start ready and waiting for operator approval.",
            details={
                "status": stage_status,
                "stage": "paper_start_pending_operator_approval",
                "read_only": True,
                "audit_only": True,
                "starts_trading_service": False,
                "submits_orders": False,
                "changes_model_routing": False,
                "live_mutation": False,
                "can_start_paper_with_operator_approval": stage_status == "paper_start_ready",
                "can_enter_canary_with_operator_approval": False,
                "can_enter_live": False,
                "blockers": [],
                "warnings": [],
            },
            evidence=[
                {"label": "Stage", "value": "paper_start_pending_operator_approval"},
                {"label": "Can start paper", "value": stage_status == "paper_start_ready"},
                {"label": "Can enter canary", "value": False},
                {"label": "Can enter live", "value": False},
            ],
        )

    monkeypatch.setattr(system_audit, "_phase3_stage_handoff_audit", factory)




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
async def test_system_audit_runs_heavy_diagnostics_serially_after_regular_sections() -> None:
    import asyncio

    running = 0
    max_running = 0
    events: list[str] = []

    async def make_card(key: str, delay: float = 0.0):
        nonlocal running, max_running
        running += 1
        max_running = max(max_running, running)
        events.append(f"start:{key}")
        if delay:
            await asyncio.sleep(delay)
        events.append(f"finish:{key}")
        running -= 1
        return system_audit._audit_card(key, key, "ok", f"{key} ok")

    specs = [
        ("model_expert_health", lambda: make_card("model_expert_health", 0.01)),
        ("model_expert_competition", lambda: make_card("model_expert_competition", 0.01)),
        ("runtime_text_integrity", lambda: make_card("runtime_text_integrity", 0.01)),
    ]

    result = await system_audit._run_audit_specs(specs, max_concurrency=1)

    assert set(result) == {
        "model_expert_health",
        "model_expert_competition",
        "runtime_text_integrity",
    }
    assert max_running == 1
    assert events == [
        "start:model_expert_health",
        "finish:model_expert_health",
        "start:model_expert_competition",
        "finish:model_expert_competition",
        "start:runtime_text_integrity",
        "finish:runtime_text_integrity",
    ]


@pytest.mark.asyncio
async def test_system_audit_runs_database_diagnostics_serially() -> None:
    import asyncio

    running = 0
    max_running = 0
    events: list[str] = []

    async def make_card(key: str):
        nonlocal running, max_running
        running += 1
        max_running = max(max_running, running)
        events.append(f"start:{key}")
        await asyncio.sleep(0)
        events.append(f"finish:{key}")
        running -= 1
        return system_audit._audit_card(key, key, "ok", f"{key} ok")

    specs = [
        (key, lambda key=key: make_card(key))
        for key in (
            "trade_loop",
            "market_data",
            "strategy_quality",
            "model_training",
            "crypto_feature_coverage",
        )
    ]

    result = await system_audit._run_audit_specs(specs, max_concurrency=1)

    assert set(result) == {key for key, _factory in specs}
    assert max_running == 1
    assert events == [
        "start:trade_loop",
        "finish:trade_loop",
        "start:market_data",
        "finish:market_data",
        "start:strategy_quality",
        "finish:strategy_quality",
        "start:model_training",
        "finish:model_training",
        "start:crypto_feature_coverage",
        "finish:crypto_feature_coverage",
    ]


@pytest.mark.asyncio
async def test_system_audit_collect_uses_process_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    running = 0
    max_running = 0
    calls: list[str] = []

    async def fake_collect_unlocked(
        *,
        record_history: bool = True,
        source: str = "api",
    ) -> dict[str, Any]:
        nonlocal running, max_running
        running += 1
        max_running = max(max_running, running)
        calls.append(f"start:{source}")
        await asyncio.sleep(0)
        calls.append(f"finish:{source}")
        running -= 1
        return {
            "status": "ok",
            "summary": {},
            "cards": [],
            "nodes": [],
            "source": source,
            "record_history": record_history,
        }

    monkeypatch.setattr(
        system_audit,
        "_collect_system_audit_status_unlocked",
        fake_collect_unlocked,
    )

    first, second = await asyncio.gather(
        system_audit.collect_system_audit_status(record_history=False, source="first"),
        system_audit.collect_system_audit_status(record_history=False, source="second"),
    )

    assert first["source"] == "first"
    assert second["source"] == "second"
    assert max_running == 1
    assert calls == [
        "start:first",
        "finish:first",
        "start:second",
        "finish:second",
    ]




@pytest.mark.asyncio
async def test_phase3_server_migration_audit_blocks_go_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePhase3ServerMigrationAuditService:
        def __init__(self, *, timeout_seconds: int) -> None:
            assert timeout_seconds == system_audit.PHASE3_SERVER_MIGRATION_AUDIT_TIMEOUT_SECONDS

        async def report(self) -> dict[str, Any]:
            return {
                "status": "blocked",
                "read_only": True,
                "audit_only": True,
                "can_mutate_remote": False,
                "can_delete_remote_data": False,
                "phase3_go_live_blocked": True,
                "remote_probe_available": True,
                "forbidden_path_count": 1,
                "forbidden_service_count": 1,
                "legacy_process_count": 0,
                "migration_manifest": {"present": False, "item_count": 0},
                "blockers": [
                    {
                        "code": "reset_marker_missing",
                        "severity": "blocking",
                        "message": "full reset evidence is missing",
                    }
                ],
                "warnings": [],
            }

    monkeypatch.setattr(
        system_audit,
        "Phase3ServerMigrationAuditService",
        FakePhase3ServerMigrationAuditService,
    )

    card = await system_audit._phase3_server_migration_audit()
    state, label = system_audit._issue_ledger_state(card, cards_by_key={card["key"]: card})

    assert card["key"] == "phase3_server_migration"
    assert card["status"] == "warning"
    assert card["details"]["phase3_go_live_blocked"] is True
    assert card["details"]["can_delete_remote_data"] is False
    assert card["evidence"][0]["value"] is True
    assert state == "unresolved"
    assert "未修复" in label or "鏈慨澶" in label


@pytest.mark.asyncio
async def test_phase3_server_migration_does_not_observe_legacy_takeover_as_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePhase3ServerMigrationAuditService:
        def __init__(self, *, timeout_seconds: int) -> None:
            assert timeout_seconds == system_audit.PHASE3_SERVER_MIGRATION_AUDIT_TIMEOUT_SECONDS

        async def report(self) -> dict[str, Any]:
            return {
                "status": "ready",
                "read_only": True,
                "audit_only": True,
                "can_mutate_remote": False,
                "can_delete_remote_data": False,
                "phase3_go_live_blocked": False,
                "deployment_contract": "old_one_gpu_timesfm_takeover",
                "old_takeover": {
                    "active": True,
                    "service_ready": True,
                    "endpoint_ready": True,
                    "artifact_ready": True,
                },
                "forbidden_path_count": 0,
                "forbidden_service_count": 0,
                "legacy_process_count": 0,
                "legacy_data_path_count": 3,
                "migration_manifest": {"present": False, "item_count": 0},
                "blockers": [],
                "warnings": [
                    {
                        "code": "old_model_server_takeover_active",
                        "severity": "warning",
                        "message": "temporary takeover remains active",
                    }
                ],
            }

    monkeypatch.setattr(
        system_audit,
        "Phase3ServerMigrationAuditService",
        FakePhase3ServerMigrationAuditService,
    )

    card = await system_audit._phase3_server_migration_audit()
    state, label = system_audit._issue_ledger_state(card, cards_by_key={card["key"]: card})

    assert card["status"] == "warning"
    assert card["details"]["observing"] is False
    assert "under observation" not in card["summary"]
    assert state == "unresolved"
    assert label


@pytest.mark.asyncio
async def test_phase3_model_server_readiness_audit_warns_service_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePhase3ModelServerReadinessAuditService:
        def __init__(self, *, timeout_seconds: int) -> None:
            assert timeout_seconds == system_audit.PHASE3_MODEL_SERVER_READINESS_TIMEOUT_SECONDS

        async def report(self) -> dict[str, Any]:
            return {
                "status": "artifact_ready_service_pending",
                "read_only": True,
                "audit_only": True,
                "can_mutate_remote": False,
                "can_start_services": False,
                "can_change_live_routing": False,
                "live_routing_enabled": False,
                "artifact_ready": True,
                "runtime_ready": False,
                "phase3_model_service_go_live_blocked": True,
                "gpu_count": 8,
                "required_slot_ready_count": 6,
                "required_slot_count": 6,
                "active_endpoint_count": 0,
                "blockers": [],
                "warnings": [
                    {
                        "code": "model_services_not_running",
                        "severity": "warning",
                        "message": "services pending",
                    }
                ],
            }

    monkeypatch.setattr(
        system_audit,
        "Phase3ModelServerReadinessAuditService",
        FakePhase3ModelServerReadinessAuditService,
    )

    card = await system_audit._phase3_model_server_readiness_audit()
    state, _label = system_audit._issue_ledger_state(card, cards_by_key={card["key"]: card})

    assert card["key"] == "phase3_model_server_readiness"
    assert card["status"] == "warning"
    assert card["details"]["artifact_ready"] is True
    assert card["details"]["runtime_ready"] is False
    assert card["details"]["phase3_model_service_go_live_blocked"] is True
    assert card["evidence"][0]["value"] is True
    assert card["evidence"][1]["value"] is False
    assert state == "unresolved"


@pytest.mark.asyncio
async def test_phase3_model_server_readiness_audit_uses_verified_latest_when_probe_env_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified_latest = {
        "status": "ready",
        "artifact_ready": True,
        "runtime_ready": True,
        "phase3_model_service_go_live_blocked": False,
        "remote_probe_available": True,
        "gpu_count": 8,
        "required_slot_ready_count": 6,
        "required_slot_count": 6,
        "active_endpoint_count": 3,
        "blockers": [],
        "warnings": [],
    }

    class FakePhase3ModelServerReadinessAuditService:
        def __init__(self, *, timeout_seconds: int) -> None:
            assert timeout_seconds == system_audit.PHASE3_MODEL_SERVER_READINESS_TIMEOUT_SECONDS

        async def report(self) -> dict[str, Any]:
            return {
                "status": "unverified",
                "artifact_ready": False,
                "runtime_ready": False,
                "phase3_model_service_go_live_blocked": True,
                "remote_probe_available": False,
                "error": "BB_SECURE_SETTINGS_KEY is required for encrypted settings",
                "blockers": [
                    {
                        "code": "model_server_config_error",
                        "severity": "blocking",
                        "message": "config unavailable",
                    }
                ],
                "warnings": [],
            }

    monkeypatch.setattr(
        system_audit,
        "_load_phase3_model_server_readiness_latest_report",
        lambda: dict(verified_latest),
    )
    monkeypatch.setattr(
        system_audit,
        "Phase3ModelServerReadinessAuditService",
        FakePhase3ModelServerReadinessAuditService,
    )

    card = await system_audit._phase3_model_server_readiness_audit()
    state, _label = system_audit._issue_ledger_state(card, cards_by_key={card["key"]: card})

    assert card["status"] == "ok"
    assert card["details"]["status"] == "ready"
    assert card["details"]["runtime_ready"] is True
    assert card["evidence"][3]["value"] == 8
    assert state == "fixed"


@pytest.mark.asyncio
async def test_phase3_paper_resume_preflight_audit_blocks_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePhase3PaperResumePreflightService:
        def __init__(self, *, model_server_timeout_seconds: int, **_kwargs: Any) -> None:
            assert model_server_timeout_seconds == (
                system_audit.PHASE3_MODEL_SERVER_READINESS_TIMEOUT_SECONDS
            )

        async def report(self) -> dict[str, Any]:
            return {
                "status": "blocked",
                "read_only": True,
                "audit_only": True,
                "mutates_database": False,
                "starts_trading_service": False,
                "submits_orders": False,
                "changes_model_routing": False,
                "can_resume_paper": False,
                "requires_operator_start": True,
                "blockers": [
                    {
                        "code": "okx_authoritative_sync_has_differences",
                        "severity": "blocking",
                        "message": "OKX differs from local facts.",
                    }
                ],
                "warnings": [],
                "summary": {
                    "okx_issue_count": 1,
                    "model_server_runtime_ready": True,
                    "phase3_quant_api_available": True,
                },
            }

    monkeypatch.setattr(
        system_audit,
        "Phase3PaperResumePreflightService",
        FakePhase3PaperResumePreflightService,
    )

    card = await system_audit._phase3_paper_resume_preflight_audit()
    state, _label = system_audit._issue_ledger_state(card, cards_by_key={card["key"]: card})

    assert card["key"] == "phase3_paper_resume_preflight"
    assert card["status"] == "critical"
    assert card["details"]["can_resume_paper"] is False
    assert card["details"]["starts_trading_service"] is False
    assert card["details"]["submits_orders"] is False
    assert card["evidence"][0]["value"] is False
    assert card["evidence"][1]["value"] == 1
    assert state == "unresolved"


@pytest.mark.asyncio
async def test_phase3_paper_resume_preflight_audit_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePhase3PaperResumePreflightService:
        def __init__(self, *, model_server_timeout_seconds: int, **_kwargs: Any) -> None:
            assert model_server_timeout_seconds == (
                system_audit.PHASE3_MODEL_SERVER_READINESS_TIMEOUT_SECONDS
            )

        async def report(self) -> dict[str, Any]:
            return {
                "status": "ready",
                "read_only": True,
                "audit_only": True,
                "mutates_database": False,
                "starts_trading_service": False,
                "submits_orders": False,
                "changes_model_routing": False,
                "can_resume_paper": True,
                "requires_operator_start": True,
                "blockers": [],
                "warnings": [],
                "summary": {
                    "okx_issue_count": 0,
                    "model_server_runtime_ready": True,
                    "phase3_quant_api_available": True,
                },
            }

    monkeypatch.setattr(
        system_audit,
        "Phase3PaperResumePreflightService",
        FakePhase3PaperResumePreflightService,
    )

    card = await system_audit._phase3_paper_resume_preflight_audit()

    assert card["status"] == "ok"
    assert card["details"]["can_resume_paper"] is True
    assert card["details"]["requires_operator_start"] is True


@pytest.mark.asyncio
async def test_phase3_paper_resume_preflight_audit_is_consumed_after_paper_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePhase3PaperResumePreflightService:
        def __init__(self, *, model_server_timeout_seconds: int, **_kwargs: Any) -> None:
            assert model_server_timeout_seconds == (
                system_audit.PHASE3_MODEL_SERVER_READINESS_TIMEOUT_SECONDS
            )

        async def report(self) -> dict[str, Any]:
            return {
                "status": "blocked",
                "read_only": True,
                "audit_only": True,
                "mutates_database": False,
                "starts_trading_service": False,
                "submits_orders": False,
                "changes_model_routing": False,
                "can_resume_paper": False,
                "requires_operator_start": True,
                "blockers": [
                    {
                        "code": "paper_trading_already_active",
                        "severity": "blocking",
                        "message": "Preflight was already consumed.",
                    },
                    {
                        "code": "okx_authoritative_pull_unavailable",
                        "severity": "blocking",
                        "message": "Observation owns current OKX checks after resume.",
                    },
                ],
                "warnings": [],
                "summary": {
                    "okx_issue_count": 0,
                    "model_server_runtime_ready": True,
                    "phase3_quant_api_available": True,
                },
                "inputs": {
                    "platform_server": {
                        "services": [
                            {"name": "bb-paper-trading.service", "active": True},
                        ]
                    }
                },
            }

    monkeypatch.setattr(
        system_audit,
        "Phase3PaperResumePreflightService",
        FakePhase3PaperResumePreflightService,
    )

    card = await system_audit._phase3_paper_resume_preflight_audit()
    state, _label = system_audit._issue_ledger_state(card, cards_by_key={card["key"]: card})

    assert card["status"] == "warning"
    assert card["details"]["consumed_after_resume"] is True
    assert card["details"]["observing"] is True
    assert state == "observing"


@pytest.mark.asyncio
async def test_phase3_paper_resume_observation_audit_waits_for_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePhase3PaperResumeObservationService:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def report(self) -> dict[str, Any]:
            return {
                "status": "waiting_for_resume",
                "read_only": True,
                "audit_only": True,
                "starts_trading_service": False,
                "submits_orders": False,
                "paper_active": False,
                "blockers": [],
                "warnings": [{"code": "paper_trading_not_active"}],
                "summary": {
                    "created_shadow_count": 0,
                    "completed_shadow_count": 0,
                    "specialist_eligible_shadow_count": 0,
                },
            }

    monkeypatch.setattr(
        system_audit,
        "Phase3PaperResumeObservationService",
        FakePhase3PaperResumeObservationService,
    )

    card = await system_audit._phase3_paper_resume_observation_audit()
    state, _label = system_audit._issue_ledger_state(card, cards_by_key={card["key"]: card})

    assert card["key"] == "phase3_paper_resume_observation"
    assert card["status"] == "warning"
    assert card["details"]["observing"] is True
    assert card["details"]["starts_trading_service"] is False
    assert card["details"]["submits_orders"] is False
    assert card["evidence"][0]["value"] is False
    assert state == "observing"


@pytest.mark.asyncio
async def test_phase3_paper_resume_observation_audit_reports_healthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePhase3PaperResumeObservationService:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def report(self) -> dict[str, Any]:
            return {
                "status": "healthy",
                "read_only": True,
                "audit_only": True,
                "starts_trading_service": False,
                "submits_orders": False,
                "paper_active": True,
                "blockers": [],
                "warnings": [],
                "summary": {
                    "created_shadow_count": 12,
                    "completed_shadow_count": 3,
                    "specialist_eligible_shadow_count": 2,
                },
            }

    monkeypatch.setattr(
        system_audit,
        "Phase3PaperResumeObservationService",
        FakePhase3PaperResumeObservationService,
    )

    card = await system_audit._phase3_paper_resume_observation_audit()

    assert card["status"] == "ok"
    assert card["details"]["paper_active"] is True
    assert card["details"]["observing"] is False
    assert card["evidence"][3]["value"] == 12








@pytest.mark.asyncio
async def test_model_training_audit_does_not_run_full_self_check(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    async def fake_data_collection_status(
        include_feature_coverage: bool = True,
    ) -> dict[str, Any]:
        return {
            "training": {
                "local_ai_tools": {
                    "available": True,
                    "status": "ready",
                    "shadow_sample_count": 123,
                    "trade_sample_count": 7,
                    "text_sentiment_sample_count": 11,
                    "training_mode": "walk_forward",
                    "model_stage": "canary",
                    "trained_models_available": True,
                    "trained_at": "2026-07-09T05:00:00+00:00",
                    "evaluation_policy": {
                        "promotion_flow": "shadow_to_canary_to_live",
                        "live_mutation": False,
                        "requires_walk_forward": True,
                    },
                    "promotion_recommendation": {
                        "recommended_stage": "canary",
                        "live_ready": False,
                    },
                    "models": {"profit": "ExtraTreesRegressor"},
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
    _patch_artifact_retirement_audit(monkeypatch)
    _patch_specialist_shadow_report_path(
        monkeypatch,
        tmp_path,
        {
            "available": True,
            "generated_at": "2026-06-27T00:00:00+00:00",
            "completed_count": 20,
            "eligible_shadow_count": 12,
            "model_count": 1,
            "live_mutation": False,
            "promotion_flow": "shadow_to_canary_to_live",
            "summary": {"promotion_ready_count": 0, "blocked_count": 1},
            "models": [
                {
                    "tool": "time_series_prediction",
                    "model": "timesfm_shadow_challenger",
                    "actual_inference_count": 12,
                    "promotion_ready": False,
                    "promotion_blockers": ["specialist_shadow_sample_floor_not_met"],
                }
            ],
        },
    )

    _patch_historical_trade_fact_audit(
        monkeypatch,
        report={
            "status": "dirty",
            "read_only": True,
            "audit_only": True,
            "cleanup_mode": "quarantine_not_delete",
            "training_policy": "clean_training_view_only",
            "checked_closed_positions": 9,
            "trainable_closed_positions": 7,
            "quarantined_closed_positions": 2,
            "can_delete_history": False,
            "can_apply_repair": False,
        },
    )

    card = await system_audit._model_training_audit()

    assert not hasattr(system_audit, "system_self_check")
    assert card["status"] == "ok"
    assert card["details"]["runtime_probe"]["status"] == "ok"
    assert card["details"]["phase3_training_governance"]["training_mode"] == "walk_forward"
    assert card["details"]["phase3_training_governance"]["model_stage"] == "canary"
    assert card["details"]["phase3_training_governance"]["live_mutation"] is False
    assert card["details"]["local_ai_tools"]["models"]["profit"] == "ExtraTreesRegressor"
    assert (
        card["details"]["local_ai_tools"]["promotion_recommendation"]["recommended_stage"]
        == "canary"
    )
    specialist = card["details"]["specialist_shadow_evaluation"]
    assert specialist["available"] is True
    assert specialist["eligible_shadow_count"] == 12
    assert specialist["models"][0]["model"] == "timesfm_shadow_challenger"
    health = card["details"]["training_health_summary"]
    assert health["policy"] == "distinguish_project_training_from_pretrained_or_alias_models"
    assert health["status_counts"]["trained_canary_ready"] == 1
    assert health["status_counts"]["shadow_inference_evaluated_not_promoted"] == 1
    assert health["items"][0]["kind"] == "project_trained_bundle"
    assert health["items"][1]["kind"] == "specialist_shadow_evidence_source"
    assert card["evidence"][-1]["value"] == 12
    historical = card["details"]["historical_trade_fact_audit"]
    assert historical["training_policy"] == "clean_training_view_only"
    assert historical["cleanup_mode"] == "quarantine_not_delete"
    assert historical["quarantined_closed_positions"] == 2
    assert historical["can_delete_history"] is False
    assert historical["can_apply_repair"] is False
    readiness = card["details"]["phase3_rebuild_readiness"]
    assert readiness["read_only"] is True
    assert readiness["writes_artifacts"] is False
    assert readiness["can_persist_artifact"] is False
    assert readiness["target_artifacts"]["local_ai_tools"]["target_stage"] == "shadow"


@pytest.mark.asyncio
async def test_model_training_audit_skips_duplicate_feature_coverage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    calls: list[bool] = []

    async def fake_data_collection_status(
        include_feature_coverage: bool = True,
    ) -> dict[str, Any]:
        calls.append(include_feature_coverage)
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
            "ai_models": [],
            "local_ai_tools": {"available": True, "api_base": "http://127.0.0.1:18001"},
        }

    monkeypatch.setattr(
        system_audit.data_collection_api,
        "get_data_collection_status",
        fake_data_collection_status,
    )
    monkeypatch.setattr(system_audit, "collect_platform_runtime_status", fake_runtime_status)
    _patch_historical_trade_fact_audit(monkeypatch)
    _patch_artifact_retirement_audit(monkeypatch)
    _patch_specialist_shadow_report_path(
        monkeypatch,
        tmp_path,
        {
            "available": True,
            "completed_count": 0,
            "eligible_shadow_count": 0,
            "model_count": 0,
            "live_mutation": False,
            "summary": {"promotion_ready_count": 0, "blocked_count": 0},
            "models": [],
        },
    )

    card = await system_audit._model_training_audit()

    assert calls == [False]
    assert card["status"] == "ok"


@pytest.mark.asyncio
async def test_model_training_audit_surfaces_missing_specialist_shadow_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    async def fake_data_collection_status(
        include_feature_coverage: bool = True,
    ) -> dict[str, Any]:
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
            "ai_models": [],
            "local_ai_tools": {"available": True, "api_base": "http://127.0.0.1:18001"},
        }

    monkeypatch.setattr(
        system_audit.data_collection_api,
        "get_data_collection_status",
        fake_data_collection_status,
    )
    monkeypatch.setattr(system_audit, "collect_platform_runtime_status", fake_runtime_status)
    _patch_historical_trade_fact_audit(monkeypatch)
    _patch_artifact_retirement_audit(monkeypatch)
    _patch_specialist_shadow_report_path(monkeypatch, tmp_path, None)

    card = await system_audit._model_training_audit()

    assert card["status"] == "warning"
    specialist = card["details"]["specialist_shadow_evaluation"]
    assert specialist["available"] is False
    assert specialist["reason"] == "specialist_shadow_evaluation_report_missing"
    assert specialist["live_mutation"] is False


@pytest.mark.asyncio
async def test_model_training_audit_reports_retired_artifacts_as_rebuild_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_data_collection_status(
        include_feature_coverage: bool = True,
    ) -> dict[str, Any]:
        assert include_feature_coverage is False
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
            "ai_models": [{"model": "qwen3-14b-trade", "available": True}],
            "local_ai_tools": {"available": True, "api_base": "http://127.0.0.1:18001"},
        }

    monkeypatch.setattr(
        system_audit.data_collection_api,
        "get_data_collection_status",
        fake_data_collection_status,
    )
    monkeypatch.setattr(system_audit, "collect_platform_runtime_status", fake_runtime_status)
    _patch_historical_trade_fact_audit(monkeypatch)
    _patch_artifact_retirement_audit(
        monkeypatch,
        report={
            "status": "retired_required",
            "read_only": True,
            "audit_only": True,
            "raw_artifacts_preserved": True,
            "can_delete_artifacts": False,
            "training_policy": "clean_training_view_only",
            "artifact_count": 2,
            "phase3_compatible_count": 0,
            "retired_or_untrusted_count": 2,
            "status_counts": {"retired_legacy": 2},
            "retired_or_untrusted_samples": [
                {
                    "relative_path": "ml_signal/winrate_model.joblib",
                    "classification": "retired_legacy",
                    "preserved": True,
                    "can_influence_live": False,
                }
            ],
        },
    )

    card = await system_audit._model_training_audit()
    ledger = system_audit._issue_ledger_from_cards([card])

    assert card["status"] == "warning"
    assert card["details"]["hard_failure"] is False
    assert card["details"]["observing"] is True
    artifact_report = card["details"]["artifact_retirement_audit"]
    assert artifact_report["status"] == "retired_required"
    assert artifact_report["can_delete_artifacts"] is False
    assert artifact_report["retired_or_untrusted_count"] == 2
    readiness = card["details"]["phase3_rebuild_readiness"]
    assert readiness["read_only"] is True
    assert readiness["can_persist_artifact"] is False
    assert readiness["live_mutation"] is False
    assert "unresolved_or_untrusted_artifacts_block_rebuild" in readiness["blockers"]
    assert any(item["label"] == "Rebuild gate" for item in card["evidence"])
    assert any(
        item["label"] == "Retired artifact" and item["value"] == 2 for item in card["evidence"]
    )
    assert ledger["summary"] == {"fixed": 0, "unresolved": 0, "observing": 1, "total": 1}


@pytest.mark.asyncio
async def test_model_training_audit_runs_database_reports_serially(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    import asyncio

    active_db_reports = 0
    max_active_db_reports = 0
    events: list[str] = []

    async def fake_data_collection_status(
        include_feature_coverage: bool = True,
    ) -> dict[str, Any]:
        nonlocal active_db_reports, max_active_db_reports
        assert include_feature_coverage is False
        active_db_reports += 1
        max_active_db_reports = max(max_active_db_reports, active_db_reports)
        events.append("start:data_collection")
        await asyncio.sleep(0)
        events.append("end:data_collection")
        active_db_reports -= 1
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
        await asyncio.sleep(0)
        return {
            "ai_models": [],
            "local_ai_tools": {"available": True, "api_base": "http://127.0.0.1:18001"},
        }

    class FakeHistoricalTradeFactAuditService:
        def __init__(self, *, lookback_days: int, limit: int) -> None:
            assert lookback_days == system_audit.HISTORICAL_TRADE_FACT_AUDIT_DAYS
            assert limit == system_audit.HISTORICAL_TRADE_FACT_AUDIT_LIMIT

        async def report(self) -> dict[str, Any]:
            nonlocal active_db_reports, max_active_db_reports
            active_db_reports += 1
            max_active_db_reports = max(max_active_db_reports, active_db_reports)
            events.append("start:historical_trade_facts")
            await asyncio.sleep(0)
            events.append("end:historical_trade_facts")
            active_db_reports -= 1
            return {
                "status": "clean",
                "read_only": True,
                "audit_only": True,
                "cleanup_mode": "quarantine_not_delete",
                "training_policy": "clean_training_view_only",
                "checked_closed_positions": 0,
                "trainable_closed_positions": 0,
                "quarantined_closed_positions": 0,
                "can_delete_history": False,
                "can_apply_repair": False,
            }

    monkeypatch.setattr(
        system_audit.data_collection_api,
        "get_data_collection_status",
        fake_data_collection_status,
    )
    monkeypatch.setattr(system_audit, "collect_platform_runtime_status", fake_runtime_status)
    monkeypatch.setattr(
        system_audit,
        "HistoricalTradeFactAuditService",
        FakeHistoricalTradeFactAuditService,
    )
    _patch_artifact_retirement_audit(monkeypatch)
    _patch_specialist_shadow_report_path(
        monkeypatch,
        tmp_path,
        {
            "available": True,
            "completed_count": 0,
            "eligible_shadow_count": 0,
            "model_count": 0,
            "live_mutation": False,
            "summary": {"promotion_ready_count": 0, "blocked_count": 0},
            "models": [],
        },
    )

    card = await system_audit._model_training_audit()

    assert card["status"] == "ok"
    assert max_active_db_reports == 1
    assert events == [
        "start:data_collection",
        "end:data_collection",
        "start:historical_trade_facts",
        "end:historical_trade_facts",
    ]


@pytest.mark.asyncio
async def test_model_expert_health_audit_reports_read_only_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []

    class FakeModelExpertHealthService:
        async def report(self, *, hours: int = 72, limit: int = 1000) -> dict[str, Any]:
            calls.append((hours, limit))
            return {
                "audit_only": True,
                "live_weight_mutation": False,
                "summary": {
                    "components": 3,
                    "recommended_state_counts": {
                        "observation_only": 3,
                    },
                },
                "components": {
                    "trend_expert": {
                        "recommended_state": "observation_only",
                        "state_reasons": ["production_permission_false"],
                        "stability": {"json_error_rate": 0.0, "no_return_rate": 0.0},
                    },
                    "risk_expert": {
                        "recommended_state": "observation_only",
                        "state_reasons": ["production_permission_false"],
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

    assert calls == [(system_audit.MODEL_EXPERT_AUDIT_HOURS, system_audit.MODEL_EXPERT_AUDIT_LIMIT)]
    assert card["key"] == "model_expert_health"
    assert card["status"] == "ok"
    assert card["details"]["audit_only"] is True
    assert card["details"]["live_weight_mutation"] is False
    assert card["details"]["recommended_state_counts"]["observation_only"] == 3
    assert card["details"]["top_components"] == []
    assert any(item["label"] == "仅观察" for item in card["evidence"])


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
                "components": {"trend_expert": {"recommended_state": "legacy_keep"}},
            }

    monkeypatch.setattr(
        system_audit,
        "ModelExpertHealthService",
        lambda: FakeModelExpertHealthService(),
    )

    report = await system_audit.model_expert_health_status(hours=24, limit=200)

    assert report["audit_only"] is True
    assert report["live_weight_mutation"] is False
    assert report["components"]["trend_expert"]["recommended_state"] == "observation_only"
    assert report["components"]["trend_expert"]["production_permission"] is False


@pytest.mark.asyncio
async def test_model_expert_competition_audit_never_allows_live_weight_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []

    class FakeModelExpertCompetitionService:
        async def report(self, *, hours: int = 72, limit: int = 1200) -> dict[str, Any]:
            calls.append((hours, limit))
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
                        "recommended_weight_action": "observation_only",
                        "baseline_delta": {"net_pnl_pct": 0.8},
                        "can_apply_live_weight": True,
                    },
                    "risk_expert": {
                        "recommended_weight_action": "observation_only",
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

    assert calls == [
        (system_audit.MODEL_EXPERT_AUDIT_HOURS, system_audit.MODEL_EXPERT_AUDIT_LIMIT),
        (24, 200),
    ]
    assert card["key"] == "model_expert_competition"
    assert card["status"] == "ok"
    assert card["details"]["can_apply_live_weight"] is False
    assert card["details"]["live_weight_mutation"] is False
    assert card["details"]["recommended_weight_action_counts"]["observation_only"] == 2
    assert card["details"]["top_competitors"][0]["can_apply_live_weight"] is False
    assert endpoint_report["audit_only"] is True
    assert endpoint_report["live_weight_mutation"] is False
    assert endpoint_report["can_apply_live_weight"] is False


@pytest.mark.asyncio
async def test_model_dynamic_routing_audit_and_endpoint_force_read_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []

    class FakeDynamicRoutingService:
        async def report(self, *, hours: int = 72, limit: int = 1200) -> dict[str, Any]:
            calls.append((hours, limit))
            return {
                "audit_only": False,
                "live_route_mutation": True,
                "can_apply_live_route": True,
                "summary": {
                    "route_plan_count": 3,
                    "shadow_only_count": 2,
                    "canary_ready_count": 1,
                    "live_ready_count": 0,
                    "live_blocked_count": 3,
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

    assert calls == [
        (system_audit.MODEL_EXPERT_AUDIT_HOURS, system_audit.MODEL_EXPERT_AUDIT_LIMIT),
        (24, 200),
    ]
    assert card["key"] == "model_dynamic_routing"
    assert card["status"] == "warning"
    assert card["details"]["audit_only"] is True
    assert card["details"]["live_route_mutation"] is False
    assert card["details"]["can_apply_live_route"] is False
    assert card["details"]["unsafe_live_mutation_attempts"] == 1
    assert card["details"]["promotion_gate"]["canary_ready_count"] == 1
    assert card["details"]["promotion_gate"]["live_ready_count"] == 0
    assert card["details"]["promotion_gate"]["live_blocked_count"] == 3
    assert endpoint_report["audit_only"] is True
    assert endpoint_report["live_route_mutation"] is False
    assert endpoint_report["can_apply_live_route"] is False
    assert endpoint_report["promotion_gate"]["live_blocked_count"] == 3


@pytest.mark.asyncio
async def test_high_risk_review_audit_and_endpoint_force_read_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []

    class FakeHighRiskReviewAuditService:
        async def report(self, *, hours: int = 72, limit: int = 1200) -> dict[str, Any]:
            calls.append((hours, limit))
            return {
                "audit_only": False,
                "read_only": False,
                "live_entry_mutation": True,
                "can_bypass_risk_controls": True,
                "can_force_open": True,
                "entry_decision_count": 6,
                "review_payload_count": 3,
                "hard_review_required_count": 2,
                "blocked_count": 1,
                "executed_without_required_review_count": 1,
                "status_counts": {"pending": 1},
                "trigger_counts": {"triggered": 2},
                "approved_counts": {"approved_false": 1},
                "reason_counts": {"large_position": 1},
                "samples": [{"id": 9, "symbol": "BTC/USDT", "can_force_open": True}],
                "policy": {"hard_review_must_approve_before_execution": False},
            }

    monkeypatch.setattr(
        system_audit,
        "HighRiskReviewAuditService",
        lambda: FakeHighRiskReviewAuditService(),
    )

    card = await system_audit._high_risk_review_audit()
    endpoint_report = await system_audit.high_risk_review_audit_status(hours=24, limit=200)

    assert calls == [
        (system_audit.MODEL_EXPERT_AUDIT_HOURS, system_audit.MODEL_EXPERT_AUDIT_LIMIT),
        (24, 200),
    ]
    assert card["key"] == "high_risk_review_audit"
    assert card["status"] == "critical"
    assert card["details"]["audit_only"] is True
    assert card["details"]["read_only"] is True
    assert card["details"]["live_entry_mutation"] is False
    assert card["details"]["can_bypass_risk_controls"] is False
    assert card["details"]["can_force_open"] is False
    assert card["details"]["summary"]["executed_without_required_review_count"] == 1
    assert endpoint_report["audit_only"] is True
    assert endpoint_report["read_only"] is True
    assert endpoint_report["live_entry_mutation"] is False
    assert endpoint_report["can_bypass_risk_controls"] is False
    assert endpoint_report["can_force_open"] is False
    assert endpoint_report["hard_review_must_approve_before_execution"] is True
    assert endpoint_report["samples"][0]["can_force_open"] is False




@pytest.mark.asyncio
async def test_strong_opportunity_audit_and_endpoint_force_read_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []

    class FakeStrongOpportunityService:
        def __init__(self, *, lookback_hours: int = 24, limit: int = 500) -> None:
            calls.append((lookback_hours, limit))

        async def report(self) -> dict[str, Any]:
            return {
                "audit_only": False,
                "live_entry_mutation": True,
                "live_sizing_mutation": True,
                "can_bypass_risk_controls": True,
                "can_force_open": True,
                "can_apply_live_sizing": True,
                "lookback_hours": 24,
                "checked_decisions": 9,
                "entry_decisions": 6,
                "strong_candidate_count": 0,
                "executed_strong_candidate_count": 0,
                "near_miss_count": 2,
                "blocker_counts": {"expected_net_below_strong_threshold": 2},
                "evidence_tier_counts": {"small": 2},
                "side_counts": {"short": 2},
                "thresholds": {"min_expected_net_pct": 0.8},
                "strong_candidates": [
                    {
                        "symbol": "BTC/USDT",
                        "can_bypass_risk_controls": True,
                        "can_force_open": True,
                        "can_apply_live_sizing": True,
                    }
                ],
                "near_misses": [
                    {
                        "symbol": "ETH/USDT",
                        "can_bypass_risk_controls": True,
                        "can_force_open": True,
                        "can_apply_live_sizing": True,
                    }
                ],
            }

    monkeypatch.setattr(system_audit, "StrongOpportunityService", FakeStrongOpportunityService)

    card = await system_audit._strong_opportunity_audit()
    endpoint_report = await system_audit.strong_opportunity_status(hours=12, limit=120)

    assert calls == [(24, 500), (12, 120)]
    assert card["key"] == "strong_opportunity"
    assert card["status"] == "warning"
    assert card["details"]["audit_only"] is True
    assert card["details"]["live_entry_mutation"] is False
    assert card["details"]["live_sizing_mutation"] is False
    assert card["details"]["can_bypass_risk_controls"] is False
    assert card["details"]["can_force_open"] is False
    assert card["details"]["can_apply_live_sizing"] is False
    assert card["details"]["near_misses"][0]["can_apply_live_sizing"] is False
    assert endpoint_report["audit_only"] is True
    assert endpoint_report["live_entry_mutation"] is False
    assert endpoint_report["live_sizing_mutation"] is False
    assert endpoint_report["can_bypass_risk_controls"] is False
    assert endpoint_report["can_force_open"] is False
    assert endpoint_report["can_apply_live_sizing"] is False
    assert endpoint_report["near_misses"][0]["can_force_open"] is False




@pytest.mark.asyncio
async def test_strategy_signal_root_cause_audit_forces_read_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []

    class FakeStrategySignalRootCauseAuditService:
        def __init__(self, *, lookback_hours: int = 24, limit: int = 500) -> None:
            calls.append((lookback_hours, limit))

        async def report(self) -> dict[str, Any]:
            return {
                "status": "warning",
                "summary": "Signal blockers found.",
                "audit_only": False,
                "read_only": False,
                "live_entry_mutation": True,
                "live_sizing_mutation": True,
                "live_leverage_mutation": True,
                "can_force_open": True,
                "can_override_thresholds": True,
                "can_change_ml_readiness": True,
                "can_bypass_risk_controls": True,
                "entry_decision_count": 25,
                "high_quality_entry_count": 0,
                "ml": {"usable_rate": 0.0},
                "server_profit": {"negative_or_opposite_count": 20},
                "shadow_missed_opportunity": {"missed_count": 30},
                "root_causes": [
                    {
                        "code": "ml_not_contributing",
                        "can_force_open": True,
                        "can_override_thresholds": True,
                        "can_change_ml_readiness": True,
                        "can_bypass_risk_controls": True,
                    }
                ],
                "next_actions": ["Fix ML readiness first."],
            }

    monkeypatch.setattr(
        system_audit,
        "StrategySignalRootCauseAuditService",
        FakeStrategySignalRootCauseAuditService,
    )

    card = await system_audit._strategy_signal_root_cause_audit()

    assert calls == [(24, 500)]
    assert card["key"] == "strategy_signal_root_cause"
    assert card["status"] == "warning"
    assert card["details"]["audit_only"] is True
    assert card["details"]["read_only"] is True
    assert card["details"]["live_entry_mutation"] is False
    assert card["details"]["live_sizing_mutation"] is False
    assert card["details"]["live_leverage_mutation"] is False
    assert card["details"]["can_force_open"] is False
    assert card["details"]["can_override_thresholds"] is False
    assert card["details"]["can_change_ml_readiness"] is False
    assert card["details"]["can_bypass_risk_controls"] is False
    assert card["details"]["root_causes"][0]["can_force_open"] is False
    assert card["details"]["root_causes"][0]["can_override_thresholds"] is False
    assert card["evidence"][0] == {"label": "开仓候选", "value": 25}
    assert card["evidence"][-1] == {"label": "根因数", "value": 1}










def test_okx_integrity_authoritative_timeout_is_observing_when_runtime_sync_is_healthy() -> None:
    card = {
        "key": "okx_trade_fact_integrity",
        "title": "OKX/本地交易事实一致性",
        "status": "warning",
        "summary": "OKX authoritative pull timed out, but runtime sync is healthy.",
        "details": {
            "issue_count": 0,
            "critical_count": 0,
            "warning_count": 0,
            "severity_counts": {},
            "position_fact_link_repair": {"candidate_link_count": 0},
            "okx_authoritative_sync": {
                "okx_pull_available": False,
                "manual_review_count": 0,
                "repairable_count": 0,
                "severity_counts": {},
            },
            "runtime_okx_entry_gate": {
                "entry_blocked": False,
                "sync_status": "ok",
                "last_requires_attention_count": 0,
            },
        },
    }

    ledger = system_audit._issue_ledger_from_cards([card])

    assert ledger["summary"] == {"fixed": 0, "unresolved": 0, "observing": 1, "total": 1}
    assert ledger["observing"][0]["key"] == "okx_trade_fact_integrity"












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
async def test_crypto_feature_coverage_audit_marks_cold_start_as_observing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCryptoFeatureCoverageService:
        async def report(self, *, hours: int = 24, limit: int = 1000) -> dict[str, Any]:
            return {
                "audit_only": True,
                "live_signal_mutation": False,
                "can_missing_features_drive_live_entry": False,
                "feature_defaults_are_neutral": True,
                "status": "warning",
                "decision_sample_count": 0,
                "feature_snapshot_count": 0,
                "waiting_for_decision_samples": True,
                "features": [
                    {"key": "kline_1m", "status": "available"},
                    {"key": "ticker", "status": "available"},
                    {"key": "funding_rate", "status": "missing"},
                ],
                "missing_features": ["funding_rate"],
                "stale_features": [],
                "neutralized_features": ["funding_rate"],
                "symbols_observed": [],
                "feature_contribution_policy": {"missing_feature_policy": "neutral_blocked"},
            }

    monkeypatch.setattr(
        system_audit,
        "CryptoFeatureCoverageService",
        lambda: FakeCryptoFeatureCoverageService(),
    )

    card = await system_audit._crypto_feature_coverage_audit()
    state, label = system_audit._issue_ledger_state(card, cards_by_key={})

    assert card["status"] == "warning"
    assert card["details"]["waiting_for_decision_samples"] is True
    assert card["details"]["feature_snapshot_count"] == 0
    assert state == "observing"
    assert "观察" in label


def test_market_data_warmup_warning_is_observing() -> None:
    card = system_audit._audit_card(
        "market_data",
        "行情与 K线",
        "warning",
        "行情预热扩容中",
        details={
            "warmup_observing": True,
            "missing_timeframes": [],
            "stale_timeframes": ["1m"],
            "covered_timeframes": ["5m", "15m", "1h"],
        },
    )

    state, label = system_audit._issue_ledger_state(card, cards_by_key={})

    assert state == "observing"
    assert "market-data warmup" in label


def test_market_data_zero_coverage_warning_remains_unresolved() -> None:
    card = system_audit._audit_card(
        "market_data",
        "行情与 K线",
        "warning",
        "行情缺失",
        details={
            "warmup_observing": False,
            "missing_timeframes": ["1m", "5m", "15m", "1h"],
            "covered_timeframes": [],
        },
    )

    state, _label = system_audit._issue_ledger_state(card, cards_by_key={})

    assert state == "unresolved"






@pytest.mark.asyncio
async def test_model_training_optional_sources_are_observing_not_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_data_collection_status(
        include_feature_coverage: bool = True,
    ) -> dict[str, Any]:
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
    _patch_historical_trade_fact_audit(monkeypatch)
    _patch_artifact_retirement_audit(monkeypatch)

    card = await system_audit._model_training_audit()
    ledger = system_audit._issue_ledger_from_cards([card])

    assert card["status"] == "warning"
    assert card["summary"] == "模型服务可用；可选增强数据源未配置、模型仍在学习观察。"
    assert card["details"]["hard_failure"] is False
    assert card["details"]["optional_source_warning_count"] == 3
    assert ledger["summary"] == {"fixed": 0, "unresolved": 0, "observing": 1, "total": 1}
    assert ledger["observing"][0]["key"] == "model_training"


@pytest.mark.asyncio
async def test_model_training_ready_tools_optional_sources_summary_is_specific(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_data_collection_status(
        include_feature_coverage: bool = True,
    ) -> dict[str, Any]:
        return {
            "training": {
                "local_ai_tools": {
                    "available": True,
                    "status": "ready",
                    "shadow_sample_count": 19991,
                    "trade_sample_count": 1604,
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
                }
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
    _patch_historical_trade_fact_audit(monkeypatch)
    _patch_artifact_retirement_audit(monkeypatch)

    card = await system_audit._model_training_audit()

    assert card["status"] == "warning"
    assert card["summary"] == "模型服务可用；可选增强数据源未配置。"
    assert card["details"]["local_ai_tools"]["status"] == "ready"
    assert card["details"]["hard_failure"] is False




@pytest.mark.asyncio
async def test_model_training_unconfigured_local_tools_are_observing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_data_collection_status(
        include_feature_coverage: bool = True,
    ) -> dict[str, Any]:
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
    _patch_historical_trade_fact_audit(monkeypatch)
    _patch_artifact_retirement_audit(monkeypatch)

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
    async def fake_data_collection_status(
        include_feature_coverage: bool = True,
    ) -> dict[str, Any]:
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
    _patch_historical_trade_fact_audit(monkeypatch)
    _patch_artifact_retirement_audit(monkeypatch)

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
    async def fake_data_collection_status(
        include_feature_coverage: bool = True,
    ) -> dict[str, Any]:
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
    _patch_historical_trade_fact_audit(monkeypatch)
    _patch_artifact_retirement_audit(monkeypatch)

    card = await system_audit._model_training_audit()
    ledger = system_audit._issue_ledger_from_cards([card])

    assert card["status"] == "warning"
    assert card["details"]["runtime_probe"]["timeout"] is True
    assert card["details"]["hard_failure"] is False
    assert ledger["summary"] == {"fixed": 0, "unresolved": 0, "observing": 1, "total": 1}
    assert ledger["observing"][0]["key"] == "model_training"


@pytest.mark.asyncio
async def test_model_training_status_timeout_is_observing_when_runtime_tools_are_healthy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    async def fake_data_collection_status(
        include_feature_coverage: bool = True,
    ) -> dict[str, Any]:
        return {
            "training": {
                "local_ai_tools": {
                    "available": False,
                    "status": "timeout",
                    "shadow_sample_count": 0,
                    "trade_sample_count": 0,
                    "text_sentiment_sample_count": 0,
                },
                "governance": {
                    "status": "ok",
                    "local_ai_tools": {
                        "phase3_clean_trainable_sample_count": 541,
                        "trainable_sample_count": 541,
                        "trade_sample_count": 57,
                    },
                },
            },
            "sources": [],
        }

    async def fake_runtime_status() -> dict[str, Any]:
        return {
            "ai_models": [{"model": "qwen3-32b-trade", "available": True}],
            "local_ai_tools": {
                "available": True,
                "configured": True,
                "api_base": "http://127.0.0.1:18001",
                "child_endpoints": {
                    "profit_prediction": {"available": True},
                    "time_series_prediction": {"available": True},
                    "sentiment_analysis": {"available": True},
                    "exit_advice": {"available": True},
                },
            },
        }

    monkeypatch.setattr(
        system_audit.data_collection_api,
        "get_data_collection_status",
        fake_data_collection_status,
    )
    monkeypatch.setattr(system_audit, "collect_platform_runtime_status", fake_runtime_status)
    _patch_historical_trade_fact_audit(
        monkeypatch,
        report={
            "status": "dirty",
            "read_only": True,
            "audit_only": True,
            "cleanup_mode": "quarantine_not_delete",
            "training_policy": "clean_training_view_only",
            "checked_closed_positions": 57,
            "trainable_closed_positions": 57,
            "quarantined_closed_positions": 0,
            "can_delete_history": False,
            "can_apply_repair": False,
        },
    )
    _patch_artifact_retirement_audit(monkeypatch)
    _patch_specialist_shadow_report_path(
        monkeypatch,
        tmp_path,
        {
            "available": True,
            "completed_count": 0,
            "eligible_shadow_count": 0,
            "model_count": 0,
            "live_mutation": False,
            "summary": {"promotion_ready_count": 0, "blocked_count": 0},
            "models": [],
        },
    )

    card = await system_audit._model_training_audit()
    ledger = system_audit._issue_ledger_from_cards([card])

    assert card["status"] == "warning"
    assert card["details"]["hard_failure"] is False
    assert card["details"]["observing"] is True
    assert card["details"]["runtime_probe"]["local_ai_tools_available"] is True
    assert card["details"]["phase3_rebuild_readiness"]["sample_floor"]["shadow_sample_count"] == 541
    assert ledger["summary"] == {"fixed": 0, "unresolved": 0, "observing": 1, "total": 1}


@pytest.mark.asyncio
async def test_okx_reconciliation_audit_reuses_short_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def fake_okx_reconciliation_light_scan(
        days: int,
        max_close_orders: int | None = None,
    ) -> Any:
        nonlocal calls
        calls += 1
        assert days == 14
        assert max_close_orders == system_audit.OKX_RECONCILIATION_AUDIT_MAX_CLOSE_ORDERS
        return SimpleNamespace(
            plans=[
                SimpleNamespace(
                    symbol="PROS/USDT",
                    side="long",
                    quantity=1.0,
                    realized_pnl=-0.82,
                    close_order_id="close-1",
                    closed_at=datetime(2026, 6, 22, tzinfo=UTC),
                )
            ],
            lookback_days=14,
            candidate_order_count=1,
            scanned_order_count=1,
            truncated=False,
            max_close_orders=None,
            duration_seconds=0.012,
            scan_mode="light_close_order_link_summary",
        )

    monkeypatch.setattr(system_audit, "_okx_reconciliation_cache", None)
    monkeypatch.setattr(
        system_audit,
        "_okx_reconciliation_light_scan",
        fake_okx_reconciliation_light_scan,
    )

    first = await system_audit._okx_reconciliation_audit()
    second = await system_audit._okx_reconciliation_audit()

    assert calls == 1
    assert first["details"]["cache"]["hit"] is False
    assert first["details"]["candidate_close_order_count"] == 1
    assert first["details"]["classification_counts"] == {}
    assert first["details"]["repairable_count"] == 1
    assert first["details"]["manual_review_count"] == 0
    assert first["details"]["skipped_candidate_count"] == 0
    assert first["details"]["unscanned_candidate_count"] == 0
    assert first["details"]["root_cause_summary"]["status"] == "dirty"
    assert first["details"]["root_cause_summary"]["repairable_count"] == 1
    assert first["details"]["root_cause_summary"]["training_policy"] == (
        "exclude_dirty_or_unclassified_trade_facts"
    )
    assert first["details"]["training_data_policy"] == {
        "raw_records_preserved": True,
        "cleanup_mode": "quarantine_not_delete",
        "policy": "exclude_dirty_or_unclassified_trade_facts",
        "requires_training_rebuild": True,
    }
    assert first["details"]["sample_plans"][0]["classification"] == {}
    assert second["details"]["cache"]["hit"] is True
    assert second["details"]["missing_closed_positions"] == 1


@pytest.mark.asyncio
async def test_okx_reconciliation_timeout_is_observing_not_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def slow_okx_reconciliation_light_scan(
        days: int,
        max_close_orders: int | None = None,
    ) -> Any:
        assert max_close_orders == system_audit.OKX_RECONCILIATION_AUDIT_MAX_CLOSE_ORDERS
        raise TimeoutError()

    monkeypatch.setattr(system_audit, "_okx_reconciliation_cache", None)
    monkeypatch.setattr(
        system_audit,
        "_okx_reconciliation_light_scan",
        slow_okx_reconciliation_light_scan,
    )

    card = await system_audit._okx_reconciliation_audit()
    ledger = system_audit._issue_ledger_from_cards([card])

    assert card["status"] == "ok"
    assert card["details"]["timeout"] is True
    assert ledger["summary"] == {"fixed": 1, "unresolved": 0, "observing": 0, "total": 1}


@pytest.mark.asyncio
async def test_okx_reconciliation_light_scan_reports_unlinked_close_orders(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    from db.session import close_db, get_session_ctx, init_db
    from models.decision import AIDecision
    from models.trade import Order, Position

    await close_db()
    db_path = tmp_path / "audit.db"
    now = datetime(2026, 6, 22, 4, 0, tzinfo=UTC)
    monkeypatch.setattr(
        system_audit.settings,
        "database_url",
        f"sqlite+aiosqlite:///{db_path.as_posix()}",
    )
    monkeypatch.setattr(system_audit, "_now", lambda: now)
    monkeypatch.setattr(system_audit, "_okx_reconciliation_cache", None)

    await init_db()
    try:
        async with get_session_ctx() as session:
            decision = AIDecision(
                model_name="test_model",
                symbol="PROS/USDT",
                action="close_long",
                confidence=0.8,
                raw_llm_response={},
                was_executed=True,
                created_at=(now - timedelta(minutes=5)).replace(tzinfo=None),
            )
            linked_decision = AIDecision(
                model_name="test_model",
                symbol="BTC/USDT",
                action="close_short",
                confidence=0.8,
                raw_llm_response={},
                was_executed=True,
                created_at=(now - timedelta(minutes=4)).replace(tzinfo=None),
            )
            session.add_all([decision, linked_decision])
            await session.flush()
            session.add_all(
                [
                    Order(
                        model_name="test_model",
                        execution_mode="paper",
                        decision_id=decision.id,
                        symbol="PROS/USDT",
                        side="sell",
                        order_type="market",
                        quantity=9,
                        price=0.3,
                        status="filled",
                        exchange_order_id="close-missing",
                        filled_at=(now - timedelta(minutes=5)).replace(tzinfo=None),
                        created_at=(now - timedelta(minutes=5)).replace(tzinfo=None),
                    ),
                    Order(
                        model_name="test_model",
                        execution_mode="paper",
                        decision_id=linked_decision.id,
                        symbol="BTC/USDT",
                        side="buy",
                        order_type="market",
                        quantity=1,
                        price=60000,
                        status="filled",
                        exchange_order_id="close-linked",
                        filled_at=(now - timedelta(minutes=4)).replace(tzinfo=None),
                        created_at=(now - timedelta(minutes=4)).replace(tzinfo=None),
                    ),
                    Position(
                        model_name="test_model",
                        execution_mode="paper",
                        symbol="BTC/USDT",
                        side="short",
                        quantity=1,
                        entry_price=61000,
                        current_price=60000,
                        is_open=False,
                        realized_pnl=1000,
                        close_exchange_order_id="close-linked",
                        closed_at=(now - timedelta(minutes=4)).replace(tzinfo=None),
                        created_at=(now - timedelta(minutes=20)).replace(tzinfo=None),
                    ),
                ]
            )

        card = await system_audit._okx_reconciliation_audit()

        assert card["status"] == "warning"
        assert card["details"]["scan_mode"] == "light_close_order_link_summary"
        assert card["details"]["candidate_close_order_count"] == 2
        assert card["details"]["scanned_close_order_count"] == 2
        assert card["details"]["missing_closed_positions"] == 1
        assert card["details"]["manual_review_count"] == 1
        assert card["details"]["repairable_count"] == 0
        assert card["details"]["classification_counts"]["linked"] == 1
        assert card["details"]["root_cause_summary"]["status"] == "dirty"
        assert card["details"]["root_cause_summary"]["manual_review_count"] == 1
        assert card["details"]["root_cause_summary"]["root_causes"][0]["code"] == (
            "manual_review_required"
        )
        assert card["details"]["training_data_policy"]["cleanup_mode"] == ("quarantine_not_delete")
        assert card["details"]["training_data_policy"]["requires_training_rebuild"] is True
        assert card["details"]["sample_plans"][0]["exchange_order_id"] == "close-missing"
    finally:
        await close_db()


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
async def test_trade_loop_paused_market_analysis_is_observing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    from db.session import close_db, init_db

    await close_db()
    db_path = tmp_path / "audit.db"
    now = datetime(2026, 6, 22, 5, 30, tzinfo=UTC)
    started_at = now - timedelta(minutes=45)
    heartbeat_at = now - timedelta(seconds=4)
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
            "heartbeat_at": heartbeat_at,
            "heartbeat_at_iso": heartbeat_at.isoformat(),
            "running": True,
            "paused": True,
            "mode": "paper",
            "scan_mode": "auto",
            "decision_interval": 30,
            "current_stage": "idle",
            "market_current_stage": "idle",
            "market_round_active": False,
            "last_market_round_started_at": None,
        },
    )

    await init_db()
    try:
        card = await system_audit._trade_loop_audit()
        ledger = system_audit._issue_ledger_from_cards([card])

        assert card["status"] == "warning"
        assert card["details"]["market_analysis_paused"] is True
        assert card["details"]["runtime_window"]["paused"] is True
        assert card["details"]["runtime_window"]["scan_mode"] == "auto"
        assert "paused" in card["summary"].lower()
        assert ledger["summary"] == {"fixed": 0, "unresolved": 0, "observing": 1, "total": 1}
        assert ledger["observing"][0]["key"] == "trade_loop"
    finally:
        await close_db()




@pytest.mark.asyncio
async def test_trade_loop_orderless_but_healthy_runtime_is_observing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    from db.session import close_db, init_db
    from models.decision import AIDecision

    await close_db()
    db_path = tmp_path / "audit.db"
    now = datetime(2026, 6, 22, 6, 30, tzinfo=UTC)
    started_at = now - timedelta(minutes=45)
    heartbeat_at = now - timedelta(seconds=4)
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
            "heartbeat_at": heartbeat_at,
            "heartbeat_at_iso": heartbeat_at.isoformat(),
            "running": True,
            "paused": False,
            "mode": "paper",
            "scan_mode": "auto",
            "decision_interval": 30,
            "current_stage": "idle",
            "market_current_stage": "idle",
            "market_round_active": False,
        },
    )

    await init_db()
    try:
        from db.session import get_session_ctx

        async with get_session_ctx() as session:
            session.add_all(
                AIDecision(
                    model_name="test",
                    symbol=f"TEST{i}/USDT",
                    action="hold",
                    confidence=0.1,
                    raw_llm_response={},
                    created_at=(now - timedelta(minutes=i % 90)).replace(tzinfo=None),
                )
                for i in range(40)
            )
            await session.commit()

        card = await system_audit._trade_loop_audit()
        ledger = system_audit._issue_ledger_from_cards([card])

        assert card["status"] == "warning"
        assert card["details"]["orderless_observation"] is True
        assert card["details"]["runtime_heartbeat_fresh"] is True
        assert ledger["summary"] == {"fixed": 0, "unresolved": 0, "observing": 1, "total": 1}
        assert ledger["observing"][0]["state_label"] == "观察项 / 有分析但当前未触发订单"
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


def test_issue_ledger_observes_strategy_closed_loop_history_and_sample_warnings() -> None:
    strategy_closed_loop = system_audit._audit_card(
        "strategy_closed_loop",
        "Strategy closed loop",
        "warning",
        "Historical or sample-limited warning only.",
        details={
            "current_runtime_window": {
                "historical_legacy_issues": False,
                "weak_executed_count": 0,
                "fast_loss_under_15m_count": 0,
                "entry_decision_count": 2,
                "high_quality_entry_count": 0,
            },
            "diagnostics": {
                "current_weak_executed": False,
                "current_no_high_quality_entries": False,
                "current_fast_loss_cluster": False,
                "current_ml_not_effective": False,
                "shadow_only_executed": False,
                "executed_without_order": False,
                "historical_weak_executed": False,
                "historical_no_high_quality_entries": False,
                "historical_fast_loss_cluster": False,
                "historical_ml_not_effective": True,
                "insufficient_effectiveness_samples": True,
                "historical_legacy_issues": False,
            },
        },
    )

    ledger = system_audit._issue_ledger_from_cards([strategy_closed_loop])

    assert ledger["summary"] == {"fixed": 0, "unresolved": 0, "observing": 1, "total": 1}
    assert ledger["observing"][0]["key"] == "strategy_closed_loop"


def test_issue_ledger_observes_strategy_closed_loop_current_ml_only_warning() -> None:
    strategy_closed_loop = system_audit._audit_card(
        "strategy_closed_loop",
        "Strategy closed loop",
        "warning",
        "Current runtime ML is still not contributing.",
        details={
            "current_runtime_window": {
                "historical_legacy_issues": False,
                "weak_executed_count": 0,
                "fast_loss_under_15m_count": 0,
                "entry_decision_count": 14,
                "high_quality_entry_count": 7,
                "ml_usable_rate": 0.0,
            },
            "diagnostics": {
                "current_weak_executed": False,
                "current_no_high_quality_entries": False,
                "current_fast_loss_cluster": False,
                "current_ml_not_effective": True,
                "shadow_only_executed": False,
                "executed_without_order": False,
                "historical_ml_not_effective": True,
                "insufficient_effectiveness_samples": True,
            },
        },
    )

    ledger = system_audit._issue_ledger_from_cards([strategy_closed_loop])

    assert ledger["summary"] == {"fixed": 0, "unresolved": 0, "observing": 1, "total": 1}
    assert ledger["observing"][0]["key"] == "strategy_closed_loop"


def test_issue_ledger_keeps_strategy_closed_loop_current_quality_warning_unresolved() -> None:
    strategy_closed_loop = system_audit._audit_card(
        "strategy_closed_loop",
        "Strategy closed loop",
        "warning",
        "Current runtime still has quality problems.",
        details={
            "current_runtime_window": {
                "historical_legacy_issues": False,
                "weak_executed_count": 0,
                "fast_loss_under_15m_count": 0,
                "entry_decision_count": 24,
                "high_quality_entry_count": 0,
            },
            "diagnostics": {
                "current_weak_executed": False,
                "current_no_high_quality_entries": True,
                "current_fast_loss_cluster": False,
                "current_ml_not_effective": False,
                "shadow_only_executed": False,
                "executed_without_order": False,
                "historical_ml_not_effective": True,
                "insufficient_effectiveness_samples": True,
            },
        },
    )

    ledger = system_audit._issue_ledger_from_cards([strategy_closed_loop])

    assert ledger["summary"] == {"fixed": 0, "unresolved": 1, "observing": 0, "total": 1}
    assert ledger["unresolved"][0]["key"] == "strategy_closed_loop"


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


def test_issue_ledger_marks_shadow_only_governance_warnings_as_observing() -> None:
    cards = [
        system_audit._audit_card(
            "model_expert_health",
            "模型/专家体检",
            "warning",
            "发现组件需要影子观察。",
            details={
                "audit_only": True,
                "live_weight_mutation": False,
                "disable_or_replace_count": 0,
                "reduce_weight_count": 0,
            },
        ),
        system_audit._audit_card(
            "model_expert_competition",
            "模型/专家竞赛",
            "warning",
            "竞赛缺少 baseline 样本。",
            details={
                "audit_only": True,
                "live_weight_mutation": False,
                "can_apply_live_weight": False,
            },
        ),
        system_audit._audit_card(
            "model_dynamic_routing",
            "模型动态路由",
            "warning",
            "动态路由仍处于影子阶段。",
            details={
                "audit_only": True,
                "live_route_mutation": False,
                "can_apply_live_route": False,
                "unsafe_live_mutation_attempts": 0,
            },
        ),
        system_audit._audit_card(
            "high_risk_review_audit",
            "High-risk review",
            "warning",
            "Hard-review gate is observing blocked entries.",
            details={
                "audit_only": True,
                "read_only": True,
                "live_entry_mutation": False,
                "can_bypass_risk_controls": False,
                "can_force_open": False,
                "summary": {"executed_without_required_review_count": 0},
            },
        ),
        system_audit._audit_card(
            "crypto_feature_coverage",
            "数字货币特征覆盖",
            "warning",
            "缺失特征已中性阻断。",
            details={
                "audit_only": True,
                "live_signal_mutation": False,
                "can_missing_features_drive_live_entry": False,
                "feature_defaults_are_neutral": True,
            },
        ),
        system_audit._audit_card(
            "shadow_missed_opportunity",
            "Shadow missed opportunity",
            "warning",
            "Missed opportunity loop is observing.",
            details={
                "audit_only": True,
                "live_entry_mutation": False,
                "can_bypass_risk_controls": False,
                "weak_evidence_execution_allowed": False,
                "global_missed_count_can_drive_entries": False,
            },
        ),
        system_audit._audit_card(
            "strong_opportunity",
            "Strong opportunity",
            "warning",
            "Strong opportunity classifier is shadow-only.",
            details={
                "audit_only": True,
                "live_entry_mutation": False,
                "live_sizing_mutation": False,
                "can_bypass_risk_controls": False,
                "can_force_open": False,
                "can_apply_live_sizing": False,
            },
        ),
        system_audit._audit_card(
            "position_capacity_release",
            "Position capacity release",
            "warning",
            "Capacity release audit is observing.",
            details={
                "audit_only": True,
                "live_exit_mutation": False,
                "live_entry_mutation": False,
                "live_sizing_mutation": False,
                "can_force_close": False,
                "can_close_winners": False,
                "can_bypass_risk_controls": False,
            },
        ),
        system_audit._audit_card(
            "strategy_signal_root_cause",
            "策略信号根因",
            "warning",
            "Strategy signal root-cause audit is observing.",
            details={
                "audit_only": True,
                "read_only": True,
                "live_entry_mutation": False,
                "live_sizing_mutation": False,
                "live_leverage_mutation": False,
                "can_force_open": False,
                "can_override_thresholds": False,
                "can_change_ml_readiness": False,
                "can_bypass_risk_controls": False,
            },
        ),
    ]

    ledger = system_audit._issue_ledger_from_cards(cards)

    assert ledger["summary"] == {"fixed": 0, "unresolved": 0, "observing": 9, "total": 9}
    assert {item["key"] for item in ledger["observing"]} == {
        "model_expert_health",
        "model_expert_competition",
        "model_dynamic_routing",
        "high_risk_review_audit",
        "crypto_feature_coverage",
        "shadow_missed_opportunity",
        "strong_opportunity",
        "position_capacity_release",
        "strategy_signal_root_cause",
    }


def test_strong_opportunity_warning_is_observing_and_linked_to_strategy_node() -> None:
    card = system_audit._audit_card(
        "strong_opportunity",
        "Strong opportunity",
        "warning",
        "Strong opportunity classifier is shadow-only.",
        details={
            "audit_only": True,
            "live_entry_mutation": False,
            "live_sizing_mutation": False,
            "can_bypass_risk_controls": False,
            "can_force_open": False,
            "can_apply_live_sizing": False,
        },
    )

    ledger = system_audit._issue_ledger_from_cards([card])
    nodes = {node["key"]: node for node in system_audit._build_audit_nodes([card])}

    assert ledger["summary"] == {"fixed": 0, "unresolved": 0, "observing": 1, "total": 1}
    assert ledger["observing"][0]["key"] == "strong_opportunity"
    assert nodes["strong_opportunity"]["state"] == "observing"
    assert nodes["strong_opportunity"]["display_status"] == "warning"
    assert "strong_opportunity" in nodes["strategy_decision"]["card_keys"]
    assert nodes["strategy_decision"]["state"] == "observing"


def test_high_risk_review_warning_is_observing_and_linked_to_risk_nodes() -> None:
    card = system_audit._audit_card(
        "high_risk_review_audit",
        "High-risk review",
        "warning",
        "Hard-review gate is observing blocked entries.",
        details={
            "audit_only": True,
            "read_only": True,
            "live_entry_mutation": False,
            "can_bypass_risk_controls": False,
            "can_force_open": False,
            "summary": {"executed_without_required_review_count": 0},
        },
    )

    ledger = system_audit._issue_ledger_from_cards([card])
    nodes = {node["key"]: node for node in system_audit._build_audit_nodes([card])}

    assert ledger["summary"] == {"fixed": 0, "unresolved": 0, "observing": 1, "total": 1}
    assert ledger["observing"][0]["key"] == "high_risk_review_audit"
    assert nodes["high_risk_review_audit"]["state"] == "observing"
    assert nodes["high_risk_review_audit"]["display_status"] == "warning"
    assert "high_risk_review_audit" in nodes["strategy_decision"]["card_keys"]
    assert "high_risk_review_audit" in nodes["risk_guard"]["card_keys"]
    assert nodes["strategy_decision"]["state"] == "observing"
    assert nodes["risk_guard"]["state"] == "observing"


def test_position_capacity_release_warning_is_observing_and_linked_to_strategy_node() -> None:
    card = system_audit._audit_card(
        "position_capacity_release",
        "Position capacity release",
        "warning",
        "Capacity release audit is observing.",
        details={
            "audit_only": True,
            "live_exit_mutation": False,
            "live_entry_mutation": False,
            "live_sizing_mutation": False,
            "can_force_close": False,
            "can_close_winners": False,
            "can_bypass_risk_controls": False,
        },
    )

    ledger = system_audit._issue_ledger_from_cards([card])
    nodes = {node["key"]: node for node in system_audit._build_audit_nodes([card])}

    assert ledger["summary"] == {"fixed": 0, "unresolved": 0, "observing": 1, "total": 1}
    assert ledger["observing"][0]["key"] == "position_capacity_release"
    assert nodes["position_capacity_release"]["state"] == "observing"
    assert nodes["position_capacity_release"]["display_status"] == "warning"
    assert "position_capacity_release" in nodes["strategy_decision"]["card_keys"]
    assert nodes["strategy_decision"]["state"] == "observing"


def test_strategy_signal_root_cause_warning_is_observing_and_linked_to_strategy_node() -> None:
    card = system_audit._audit_card(
        "strategy_signal_root_cause",
        "策略信号根因",
        "warning",
        "Signal chain blockers are under read-only audit.",
        details={
            "audit_only": True,
            "read_only": True,
            "live_entry_mutation": False,
            "live_sizing_mutation": False,
            "live_leverage_mutation": False,
            "can_force_open": False,
            "can_override_thresholds": False,
            "can_change_ml_readiness": False,
            "can_bypass_risk_controls": False,
        },
    )

    ledger = system_audit._issue_ledger_from_cards([card])
    nodes = {node["key"]: node for node in system_audit._build_audit_nodes([card])}

    assert ledger["summary"] == {"fixed": 0, "unresolved": 0, "observing": 1, "total": 1}
    assert ledger["observing"][0]["key"] == "strategy_signal_root_cause"
    assert nodes["strategy_signal_root_cause"]["state"] == "observing"
    assert nodes["strategy_signal_root_cause"]["display_status"] == "warning"
    assert "strategy_signal_root_cause" in nodes["strategy_decision"]["card_keys"]
    assert "strategy_signal_root_cause" in nodes["strategy_closed_loop"]["card_keys"]
    assert nodes["strategy_decision"]["state"] == "observing"


def test_strategy_signal_root_cause_safety_wrapper_locks_scheduler_samples() -> None:
    report = system_audit._safe_strategy_signal_root_cause_report(
        {
            "status": "warning",
            "root_causes": [{"code": "dynamic_capacity_constrained"}],
            "scheduler": {
                "latest_samples": [
                    {
                        "symbol": "BTC/USDT",
                        "can_force_open": True,
                        "can_override_thresholds": True,
                        "can_bypass_risk_controls": True,
                    }
                ],
            },
        }
    )

    assert report["read_only"] is True
    assert report["audit_only"] is True
    assert report["live_entry_mutation"] is False
    assert report["can_force_open"] is False
    assert report["can_override_thresholds"] is False
    assert report["can_bypass_risk_controls"] is False
    assert report["root_causes"][0]["can_force_open"] is False
    scheduler = report["scheduler"]
    assert scheduler["read_only"] is True
    assert scheduler["audit_only"] is True
    assert scheduler["live_entry_mutation"] is False
    assert scheduler["live_sizing_mutation"] is False
    assert scheduler["live_leverage_mutation"] is False
    assert scheduler["can_force_open"] is False
    assert scheduler["can_override_thresholds"] is False
    assert scheduler["can_bypass_risk_controls"] is False
    sample = scheduler["latest_samples"][0]
    assert sample["can_force_open"] is False
    assert sample["can_override_thresholds"] is False
    assert sample["can_bypass_risk_controls"] is False


@pytest.mark.asyncio
async def test_okx_trade_fact_integrity_audit_marks_nodes_and_ledger(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    _patch_okx_daily_report_path(
        monkeypatch,
        tmp_path,
        {
            "status": "warning",
            "generated_at": datetime.now(UTC).isoformat(),
            "requires_attention": False,
            "can_open_new_entries": False,
            "can_refresh_training": True,
            "operational_gates": {
                "entry_blocked": True,
                "training_blocked": False,
                "attention_buckets": {"entry": 1, "training": 0, "manual_review": 0},
                "entry_blockers": [
                    {
                        "code": "trading_runtime_heartbeat_stale",
                        "card_key": "okx_trade_fact_integrity",
                        "status": "runtime_heartbeat_stale",
                        "requires_attention": False,
                    }
                ],
                "training_blockers": [],
                "attention_items": [],
            },
            "issue_ledger": {"summary": {"fixed": 3, "observing": 1, "unresolved": 0, "total": 4}},
        },
    )

    class FakeTradeFactIntegrityService:
        async def audit(self) -> dict[str, Any]:
            return {
                "read_only": True,
                "status": "critical",
                "checked_orders": 4,
                "checked_positions": 2,
                "issue_count": 2,
                "critical_count": 1,
                "warning_count": 1,
                "issues": [
                    {
                        "kind": "symbol_alias_mismatch",
                        "severity": "critical",
                        "order_id": 101,
                        "expected_symbol": "H/USDT",
                        "symbol": "WLFI/USDT",
                    }
                ],
            }

    monkeypatch.setattr(
        system_audit,
        "OkxTradeFactIntegrityService",
        lambda **_kwargs: FakeTradeFactIntegrityService(),
    )

    async def fake_collect_position_fact_link_scan_report(**_kwargs: Any) -> Any:
        return SimpleNamespace(
            lookback_days=14,
            candidate_link_count=0,
            repairable_count=0,
            manual_review_count=0,
            classification_counts={"repairable": 0, "manual_review": 0},
            scanned_position_count=0,
            max_positions=300,
            truncated=False,
            diagnostics=[],
        )

    monkeypatch.setattr(
        system_audit,
        "collect_position_fact_link_scan_report",
        fake_collect_position_fact_link_scan_report,
    )

    async def fake_okx_authoritative_sync_summary() -> dict[str, Any]:
        return {
            "status": "ok",
            "read_only": True,
            "audit_only": True,
            "okx_pull_available": True,
            "issue_count": 0,
            "repairable_count": 0,
            "manual_review_count": 0,
            "okx_fill_order_count": 2,
            "okx_position_count": 1,
            "apply_policy": {
                "can_write_database": False,
                "requires_allowlisted_apply": True,
                "requires_backup": True,
            },
        }

    monkeypatch.setattr(
        system_audit,
        "_okx_authoritative_sync_summary",
        fake_okx_authoritative_sync_summary,
    )

    card = await system_audit._okx_trade_fact_integrity_audit()
    ledger = system_audit._issue_ledger_from_cards([card])
    nodes = {node["key"]: node for node in system_audit._build_audit_nodes([card])}

    assert card["status"] == "critical"
    assert card["details"]["read_only"] is True
    assert card["details"]["live_repair_mutation"] is False
    assert card["details"]["okx_authoritative_sync"]["read_only"] is True
    assert card["details"]["okx_authoritative_sync"]["can_write_database"] is False
    assert card["details"]["okx_authoritative_sync"]["apply_policy"]["requires_backup"] is True
    assert card["details"]["position_fact_link_repair"]["candidate_link_count"] == 0
    assert card["details"]["position_fact_link_repair"]["max_positions"] == 300
    daily_report = card["details"]["daily_reconciliation_report"]
    assert daily_report["available"] is True
    assert daily_report["stale"] is False
    assert daily_report["can_open_new_entries"] is False
    assert daily_report["can_refresh_training"] is True
    assert daily_report["attention_buckets"]["entry"] == 1
    assert any(item["label"] == "Daily report" for item in card["evidence"])
    assert ledger["summary"] == {"fixed": 0, "unresolved": 1, "observing": 0, "total": 1}
    assert ledger["unresolved"][0]["key"] == "okx_trade_fact_integrity"
    assert nodes["okx_execution"]["display_status"] == "critical"
    assert nodes["position_sync"]["display_status"] == "critical"
    assert nodes["training_data"]["display_status"] == "critical"
    assert "okx_trade_fact_integrity" in nodes["okx_execution"]["card_keys"]
    assert "okx_trade_fact_integrity" in nodes["position_sync"]["card_keys"]
    assert "okx_trade_fact_integrity" in nodes["training_data"]["card_keys"]


@pytest.mark.asyncio
async def test_okx_trade_fact_integrity_warns_when_position_links_are_repairable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTradeFactIntegrityService:
        async def audit(self) -> dict[str, Any]:
            return {
                "read_only": True,
                "status": "ok",
                "checked_orders": 4,
                "checked_positions": 2,
                "issue_count": 0,
                "critical_count": 0,
                "warning_count": 0,
                "issues": [],
            }

    monkeypatch.setattr(
        system_audit,
        "OkxTradeFactIntegrityService",
        lambda **_kwargs: FakeTradeFactIntegrityService(),
    )

    async def fake_collect_position_fact_link_scan_report(**_kwargs: Any) -> Any:
        return SimpleNamespace(
            lookback_days=14,
            candidate_link_count=1,
            repairable_count=1,
            manual_review_count=0,
            classification_counts={"repairable": 1, "manual_review": 0},
            scanned_position_count=1,
            max_positions=300,
            truncated=False,
            diagnostics=[
                {
                    "status": "repairable",
                    "reason": "deterministic_position_order_match",
                    "position_id": 10,
                }
            ],
        )

    monkeypatch.setattr(
        system_audit,
        "collect_position_fact_link_scan_report",
        fake_collect_position_fact_link_scan_report,
    )

    async def fake_okx_authoritative_sync_summary() -> dict[str, Any]:
        return {
            "status": "ok",
            "read_only": True,
            "audit_only": True,
            "okx_pull_available": True,
            "issue_count": 0,
            "repairable_count": 0,
            "manual_review_count": 0,
            "okx_fill_order_count": 0,
            "okx_position_count": 0,
            "apply_policy": {
                "can_write_database": False,
                "requires_allowlisted_apply": True,
                "requires_backup": True,
            },
        }

    monkeypatch.setattr(
        system_audit,
        "_okx_authoritative_sync_summary",
        fake_okx_authoritative_sync_summary,
    )

    card = await system_audit._okx_trade_fact_integrity_audit()

    assert card["status"] == "warning"
    assert card["details"]["position_fact_link_repair"]["repairable_count"] == 1
    assert card["details"]["position_fact_link_repair"]["diagnostics"][0]["position_id"] == 10


@pytest.mark.asyncio
async def test_okx_trade_fact_integrity_ignores_superseded_link_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTradeFactIntegrityService:
        async def audit(self) -> dict[str, Any]:
            return {
                "read_only": True,
                "status": "ok",
                "checked_orders": 4,
                "checked_positions": 2,
                "issue_count": 2,
                "critical_count": 0,
                "warning_count": 0,
                "severity_counts": {"info": 2},
                "issues": [
                    {
                        "kind": "superseded_position_residual",
                        "severity": "info",
                        "position_id": 846,
                    },
                    {
                        "kind": "superseded_position_residual",
                        "severity": "info",
                        "position_id": 848,
                    },
                ],
            }

    monkeypatch.setattr(
        system_audit,
        "OkxTradeFactIntegrityService",
        lambda **_kwargs: FakeTradeFactIntegrityService(),
    )

    async def fake_collect_position_fact_link_scan_report(**_kwargs: Any) -> Any:
        return SimpleNamespace(
            lookback_days=14,
            candidate_link_count=2,
            repairable_count=0,
            manual_review_count=2,
            classification_counts={"manual_review": 2},
            scanned_position_count=2,
            max_positions=300,
            truncated=False,
            diagnostics=[
                {"status": "manual_review", "position_id": 846},
                {"status": "manual_review", "position_id": 848},
            ],
        )

    monkeypatch.setattr(
        system_audit,
        "collect_position_fact_link_scan_report",
        fake_collect_position_fact_link_scan_report,
    )

    async def fake_okx_authoritative_sync_summary() -> dict[str, Any]:
        return {
            "status": "ok",
            "read_only": True,
            "audit_only": True,
            "okx_pull_available": True,
            "issue_count": 0,
            "repairable_count": 0,
            "manual_review_count": 0,
            "okx_fill_order_count": 0,
            "okx_position_count": 0,
        }

    monkeypatch.setattr(
        system_audit,
        "_okx_authoritative_sync_summary",
        fake_okx_authoritative_sync_summary,
    )
    monkeypatch.setattr(
        system_audit,
        "_load_trading_runtime_status_for_audit",
        lambda: {
            "available": True,
            "running": True,
            "heartbeat_age_seconds": 1.0,
            "okx_authoritative_sync": {
                "status": "ok",
                "last_error": None,
                "last_requires_attention_count": 0,
            },
        },
    )
    monkeypatch.setattr(
        system_audit,
        "_load_okx_daily_reconciliation_report_summary",
        lambda: {
            "available": True,
            "status": "ok",
            "stale": False,
            "requires_attention": False,
            "can_open_new_entries": True,
            "can_refresh_training": True,
            "entry_blocked": False,
            "training_blocked": False,
            "attention_buckets": {},
        },
    )

    card = await system_audit._okx_trade_fact_integrity_audit()
    ledger = system_audit._issue_ledger_from_cards([card])

    assert card["status"] == "ok"
    assert card["details"]["unresolved_position_fact_link_candidate_count"] == 0
    assert ledger["summary"] == {"fixed": 1, "unresolved": 0, "observing": 0, "total": 1}


@pytest.mark.asyncio
async def test_okx_trade_fact_integrity_warns_on_authoritative_sync_issues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTradeFactIntegrityService:
        async def audit(self) -> dict[str, Any]:
            return {
                "read_only": True,
                "status": "ok",
                "checked_orders": 2,
                "checked_positions": 1,
                "issue_count": 0,
                "critical_count": 0,
                "warning_count": 0,
                "issues": [],
            }

    monkeypatch.setattr(
        system_audit,
        "OkxTradeFactIntegrityService",
        lambda **_kwargs: FakeTradeFactIntegrityService(),
    )

    async def fake_collect_position_fact_link_scan_report(**_kwargs: Any) -> Any:
        return SimpleNamespace(
            lookback_days=14,
            candidate_link_count=0,
            repairable_count=0,
            manual_review_count=0,
            classification_counts={"repairable": 0, "manual_review": 0},
            scanned_position_count=0,
            max_positions=300,
            truncated=False,
            diagnostics=[],
        )

    async def fake_okx_authoritative_sync_summary() -> dict[str, Any]:
        return {
            "status": "critical",
            "read_only": True,
            "audit_only": True,
            "okx_pull_available": True,
            "issue_count": 1,
            "repairable_count": 0,
            "manual_review_count": 1,
            "okx_fill_order_count": 1,
            "okx_position_count": 1,
            "issues": [
                {
                    "kind": "okx_fill_missing_local_order",
                    "classification": "manual_review",
                    "severity": "critical",
                    "exchange_order_id": "okx-only-fill",
                }
            ],
            "apply_policy": {
                "can_write_database": False,
                "requires_allowlisted_apply": True,
                "requires_backup": True,
            },
        }

    monkeypatch.setattr(
        system_audit,
        "collect_position_fact_link_scan_report",
        fake_collect_position_fact_link_scan_report,
    )
    monkeypatch.setattr(
        system_audit,
        "_okx_authoritative_sync_summary",
        fake_okx_authoritative_sync_summary,
    )

    card = await system_audit._okx_trade_fact_integrity_audit()

    assert card["status"] == "warning"
    assert card["details"]["okx_authoritative_sync"]["issue_count"] == 1
    assert card["details"]["okx_authoritative_sync"]["can_write_database"] is False
    assert card["details"]["okx_authoritative_sync"]["live_repair_mutation"] is False
    assert any(item["label"] == "OKX API facts" for item in card["evidence"])


@pytest.mark.asyncio
async def test_okx_trade_fact_integrity_surfaces_runtime_entry_gate_blocker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTradeFactIntegrityService:
        async def audit(self) -> dict[str, Any]:
            return {
                "read_only": True,
                "status": "ok",
                "checked_orders": 2,
                "checked_positions": 1,
                "issue_count": 0,
                "critical_count": 0,
                "warning_count": 0,
                "issues": [],
            }

    monkeypatch.setattr(
        system_audit,
        "OkxTradeFactIntegrityService",
        lambda **_kwargs: FakeTradeFactIntegrityService(),
    )

    async def fake_collect_position_fact_link_scan_report(**_kwargs: Any) -> Any:
        return SimpleNamespace(
            lookback_days=14,
            candidate_link_count=0,
            repairable_count=0,
            manual_review_count=0,
            classification_counts={"repairable": 0, "manual_review": 0},
            scanned_position_count=0,
            max_positions=300,
            truncated=False,
            diagnostics=[],
        )

    async def fake_okx_authoritative_sync_summary() -> dict[str, Any]:
        return {
            "status": "ok",
            "read_only": True,
            "audit_only": True,
            "okx_pull_available": True,
            "issue_count": 0,
            "repairable_count": 0,
            "manual_review_count": 0,
            "okx_fill_order_count": 0,
            "okx_position_count": 0,
            "apply_policy": {
                "can_write_database": False,
                "requires_allowlisted_apply": True,
                "requires_backup": True,
            },
        }

    monkeypatch.setattr(
        system_audit,
        "collect_position_fact_link_scan_report",
        fake_collect_position_fact_link_scan_report,
    )
    monkeypatch.setattr(
        system_audit,
        "_okx_authoritative_sync_summary",
        fake_okx_authoritative_sync_summary,
    )
    monkeypatch.setattr(
        system_audit,
        "_load_trading_runtime_status_for_audit",
        lambda: {
            "available": True,
            "running": True,
            "heartbeat_age_seconds": 12.0,
            "okx_authoritative_sync": {
                "status": "stale",
                "last_error": None,
                "last_success_at": "2026-06-26T20:00:00+00:00",
                "last_failure_at": None,
                "last_result_count": 2,
                "last_result_kinds": {
                    "missing_exchange_position_without_close_fill": 1,
                },
                "last_requires_attention_count": 0,
                "last_samples": [
                    {
                        "kind": "missing_exchange_position_without_close_fill",
                        "symbol": "SPK/USDT",
                        "side": "short",
                        "requires_attention": True,
                        "note": "waiting for OKX close fill",
                    }
                ],
                "source": "okx_private_api_current_positions",
            },
        },
    )

    card = await system_audit._okx_trade_fact_integrity_audit()

    runtime_gate = card["details"]["runtime_okx_entry_gate"]
    assert card["status"] == "warning"
    assert "阻断新开仓" in card["summary"]
    assert runtime_gate["entry_blocked"] is True
    assert runtime_gate["blocker"] == "okx_authoritative_sync_unhealthy"
    assert runtime_gate["status"] == "stale"
    assert runtime_gate["last_samples"][0]["symbol"] == "SPK/USDT"
    assert any(
        item["label"] == "Entry blocked" and item["value"] is True for item in card["evidence"]
    )
    assert any("只允许平仓" in action for action in card["next_actions"])


def test_okx_runtime_entry_gate_blocks_inactive_runtime() -> None:
    gate = system_audit._okx_runtime_entry_gate_summary(
        {
            "available": True,
            "running": False,
            "heartbeat_age_seconds": 5.0,
            "decision_interval": 30,
            "okx_authoritative_sync": {
                "status": "ok",
                "last_requires_attention_count": 0,
                "source": "okx_private_api_current_positions",
            },
        }
    )

    assert gate["entry_blocked"] is True
    assert gate["status"] == "runtime_inactive"
    assert gate["sync_status"] == "ok"
    assert gate["blocker"] == "trading_runtime_inactive"
    assert "交易运行时未运行" in gate["reason"]
    assert gate["heartbeat_fresh_limit_seconds"] == 180.0


def test_okx_daily_reconciliation_report_summary_marks_stale(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    _patch_okx_daily_report_path(
        monkeypatch,
        tmp_path,
        {
            "status": "ok",
            "generated_at": datetime(2026, 6, 20, tzinfo=UTC).isoformat(),
            "requires_attention": False,
            "can_open_new_entries": True,
            "can_refresh_training": True,
            "operational_gates": {
                "entry_blocked": False,
                "training_blocked": False,
                "attention_buckets": {"entry": 0, "training": 0, "manual_review": 0},
            },
            "issue_ledger": {"summary": {"fixed": 4, "observing": 0, "unresolved": 0, "total": 4}},
        },
    )
    monkeypatch.setattr(
        system_audit,
        "_now",
        lambda: datetime(2026, 6, 27, tzinfo=UTC),
    )

    report = system_audit._load_okx_daily_reconciliation_report_summary()

    assert report["available"] is True
    assert report["status"] == "ok"
    assert report["stale"] is True
    assert report["can_open_new_entries"] is True
    assert report["can_refresh_training"] is True
    assert report["issue_ledger_summary"]["unresolved"] == 0


def test_okx_runtime_entry_gate_blocks_stale_runtime_heartbeat() -> None:
    gate = system_audit._okx_runtime_entry_gate_summary(
        {
            "available": True,
            "running": True,
            "heartbeat_age_seconds": 600.0,
            "decision_interval": 30,
            "okx_authoritative_sync": {
                "status": "ok",
                "last_requires_attention_count": 0,
                "source": "okx_private_api_current_positions",
            },
        }
    )

    assert gate["entry_blocked"] is True
    assert gate["status"] == "runtime_heartbeat_stale"
    assert gate["sync_status"] == "ok"
    assert gate["blocker"] == "trading_runtime_heartbeat_stale"
    assert "心跳已过期" in gate["reason"]
    assert gate["heartbeat_fresh_limit_seconds"] == 180.0


def test_issue_ledger_treats_runtime_only_okx_entry_block_as_observing() -> None:
    card = system_audit._audit_card(
        "okx_trade_fact_integrity",
        "OKX trade fact integrity",
        "warning",
        "Trading runtime is intentionally inactive; new entries are blocked.",
        details={
            "issue_count": 0,
            "critical_count": 0,
            "position_fact_link_repair": {"candidate_link_count": 0},
            "okx_authoritative_sync": {"issue_count": 0},
            "runtime_okx_entry_gate": {
                "entry_blocked": True,
                "blocker": "trading_runtime_inactive",
                "status": "runtime_inactive",
                "sync_status": "ok",
            },
        },
    )

    ledger = system_audit._issue_ledger_from_cards([card])

    assert ledger["summary"] == {"fixed": 0, "unresolved": 0, "observing": 1, "total": 1}
    assert ledger["observing"][0]["key"] == "okx_trade_fact_integrity"


@pytest.mark.asyncio
async def test_position_price_integrity_reports_unmatched_okx_and_local_positions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'position-price-integrity.db').as_posix()}",
    )
    await init_db()

    class FakeExecutor:
        async def get_positions_strict(self) -> list[dict[str, Any]]:
            return [
                {
                    "symbol": "BTC/USDT:USDT",
                    "side": "long",
                    "contracts": 1.0,
                    "markPrice": 110.0,
                    "entryPrice": 100.0,
                    "info": {
                        "instId": "BTC-USDT-SWAP",
                        "pos": "1",
                        "ctVal": "1",
                        "avgPx": "100",
                        "markPx": "110",
                        "upl": "10",
                    },
                },
                {
                    "symbol": "ETH/USDT:USDT",
                    "side": "net",
                    "contracts": 2.0,
                    "markPrice": 90.0,
                    "entryPrice": 100.0,
                    "info": {
                        "instId": "ETH-USDT-SWAP",
                        "posSide": "net",
                        "pos": "-2",
                        "ctVal": "1",
                        "avgPx": "100",
                        "markPx": "90",
                        "upl": "20",
                    },
                },
            ]

    def fake_executor(mode: str) -> FakeExecutor | None:
        return FakeExecutor() if mode == "paper" else None

    monkeypatch.setattr(
        dashboard_api,
        "_dashboard_okx_executor_for_mode",
        fake_executor,
    )
    now = datetime(2026, 6, 26, tzinfo=UTC)
    try:
        async with get_session_ctx() as session:
            session.add_all(
                [
                    Position(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="BTC/USDT",
                        side="long",
                        quantity=1.0,
                        entry_price=100.0,
                        current_price=100.0,
                        unrealized_pnl=0.0,
                        realized_pnl=0.0,
                        is_open=True,
                        created_at=now,
                    ),
                    Position(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="SOL/USDT",
                        side="long",
                        quantity=3.0,
                        entry_price=50.0,
                        current_price=52.0,
                        unrealized_pnl=6.0,
                        realized_pnl=0.0,
                        is_open=True,
                        created_at=now,
                    ),
                ]
            )
        card = await system_audit._position_price_integrity_audit()
    finally:
        await close_db()

    details = card["details"]
    root = details["root_cause_summary"]
    assert card["status"] == "critical"
    assert root["status"] == "dirty"
    assert root["mismatch_count"] == 3
    assert root["split_count"] == 1
    assert root["local_only_count"] == 1
    assert root["exchange_only_count"] == 1
    assert root["root_cause_counts"]["mark_price_mismatch"] == 1
    assert root["root_cause_counts"]["local_open_position_missing_on_okx"] == 1
    assert root["root_cause_counts"]["okx_open_position_missing_locally"] == 1
    assert root["read_only"] is True
    assert root["live_repair_mutation"] is False
    assert details["splits"][0]["symbol"] == "BTC/USDT"
    assert details["splits"][0]["okx_side_inference"] == "ccxt_side"
    assert details["local_only_positions"][0]["symbol"] == "SOL/USDT"
    assert details["exchange_only_positions"][0]["symbol"] == "ETH/USDT"
    assert details["exchange_only_positions"][0]["okx_pos_side"] == "net"
    assert details["exchange_only_positions"][0]["okx_raw_pos"] == "-2"
    assert details["exchange_only_positions"][0]["okx_signed_position_size"] == -2.0
    assert details["exchange_only_positions"][0]["okx_side_inference"] == "okx_net_signed_pos"
    assert details["okx_pos_side_counts"]["net"] == 1
    assert details["okx_side_inference_counts"]["okx_net_signed_pos"] == 1
    assert details["read_only"] is True
    assert details["live_repair_mutation"] is False


def test_audit_nodes_use_issue_state_for_display_status() -> None:
    cards = [
        system_audit._audit_card(
            "strategy_closed_loop",
            "Strategy closed loop",
            "critical",
            "Historical weak entries were found, but the current window is clean.",
            details={
                "current_runtime_window": {"historical_legacy_issues": True},
                "diagnostics": {
                    "current_weak_executed": False,
                    "current_no_high_quality_entries": False,
                    "current_fast_loss_cluster": False,
                    "current_ml_not_effective": False,
                    "shadow_only_executed": False,
                    "executed_without_order": False,
                },
            },
        ),
        system_audit._audit_card(
            "trade_loop",
            "Trade loop",
            "critical",
            "Runtime loop is stuck.",
        ),
        system_audit._audit_card(
            "strategy_gate_contract",
            "Strategy gate contract",
            "ok",
            "Gate contract is healthy.",
        ),
    ]

    nodes = {node["key"]: node for node in system_audit._build_audit_nodes(cards)}

    assert nodes["strategy_closed_loop"]["status"] == "critical"
    assert nodes["strategy_closed_loop"]["state"] == "observing"
    assert nodes["strategy_closed_loop"]["display_status"] == "warning"
    assert nodes["runtime_loop"]["state"] == "unresolved"
    assert nodes["runtime_loop"]["display_status"] == "critical"
    assert nodes["strategy_gate_contract"]["state"] == "fixed"
    assert nodes["strategy_gate_contract"]["display_status"] == "ok"
def test_latest_audit_snapshot_serializes_nested_datetimes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    monkeypatch.setattr(system_audit.settings, "_data_dir", tmp_path, raising=False)
    checked_at = datetime(2026, 7, 11, 10, 0, tzinfo=UTC)
    payload = {
        "status": "ok",
        "checked_at": checked_at.isoformat(),
        "cards": [{"details": {"settlement_synced_at": checked_at}}],
    }

    system_audit._store_latest_audit_snapshot(payload)
    loaded = system_audit._load_latest_audit_snapshot()

    assert loaded is not None
    loaded_at, loaded_payload = loaded
    assert loaded_at == checked_at
    assert loaded_payload["cards"][0]["details"]["settlement_synced_at"] == (
        checked_at.isoformat()
    )
