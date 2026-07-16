from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from scripts import run_phase3_paper_resume_preflight as preflight_cli
from services.phase3_paper_resume_preflight import (
    Phase3PaperResumePreflightService,
    evaluate_phase3_paper_resume_preflight_inputs,
)


def _okx_sync_clean() -> dict[str, Any]:
    return {
        "status": "ok",
        "read_only": True,
        "audit_only": True,
        "okx_pull_available": True,
        "issue_count": 0,
        "manual_review_count": 0,
        "repairable_count": 0,
    }


def _okx_integrity_clean() -> dict[str, Any]:
    return {
        "status": "ok",
        "read_only": True,
        "audit_only": True,
        "critical_count": 0,
        "warning_count": 0,
    }


def _model_server_ready() -> dict[str, Any]:
    return {
        "status": "ready",
        "read_only": True,
        "audit_only": True,
        "artifact_ready": True,
        "runtime_ready": True,
        "phase3_model_service_go_live_blocked": False,
    }


def _platform_runtime_ready() -> dict[str, Any]:
    return {
        "local_ai_tools": {
            "available": True,
            "api_base": "http://127.0.0.1:18001",
            "health": {"ok": True, "service": "phase3_quant_api"},
            "tunnel_contract": {"ok": True},
            "child_endpoints": {
                "profit_prediction": {"available": True},
                "time_series_prediction": {"available": True},
                "sentiment_analysis": {"available": True},
                "exit_advice": {"available": True},
            },
        },
        "ai_models": [],
    }


def _platform_runtime_ready_with_models() -> dict[str, Any]:
    runtime = _platform_runtime_ready()
    runtime["ai_models"] = [
        {
            "model": "qwen3-14b-trade",
            "available": True,
            "models": ["qwen3-14b-trade"],
        },
        {
            "model": "deepseek-r1-14b-risk",
            "available": True,
            "models": ["deepseek-r1-14b-risk"],
        },
        {
            "model": "BB-FinQuant-Expert-14B",
            "available": True,
            "models": ["BB-FinQuant-Expert-14B"],
        },
    ]
    return runtime


def _platform_server_ready(*, paper_active: bool = False) -> dict[str, Any]:
    return {
        "available": True,
        "status": "ok",
        "services": [
            {"name": "bb-dashboard.service", "active": True, "status": "active"},
            {"name": "bb-model-tunnels.service", "active": True, "status": "active"},
            {
                "name": "bb-paper-trading.service",
                "active": paper_active,
                "status": "active" if paper_active else "inactive",
            },
        ],
    }


def _specialist_ready() -> dict[str, Any]:
    return {
        "available": True,
        "status": "ok",
        "generated_at": datetime.now(UTC).isoformat(),
        "live_mutation": False,
        "completed_count": 0,
        "eligible_shadow_count": 0,
    }


def _account_equity_ready() -> dict[str, Any]:
    return {
        "available": True,
        "status": "ok",
        "read_only": True,
        "audit_only": True,
        "source": "okx_snapshot",
        "equity": 4998.15,
        "total": 4998.15,
        "cash": 4998.15,
        "allocatable": 4998.15,
        "free": 4988.0,
    }


def _ready_inputs(**overrides: dict[str, Any]) -> dict[str, dict[str, Any]]:
    inputs = {
        "okx_authoritative_sync": _okx_sync_clean(),
        "okx_trade_fact_integrity": _okx_integrity_clean(),
        "model_server_readiness": _model_server_ready(),
        "platform_runtime": _platform_runtime_ready(),
        "platform_server": _platform_server_ready(),
        "specialist_shadow_evaluation": _specialist_ready(),
        "account_equity_truth": _account_equity_ready(),
    }
    inputs.update(overrides)
    return inputs


def test_phase3_paper_resume_preflight_ready_when_all_hard_gates_pass() -> None:
    report = evaluate_phase3_paper_resume_preflight_inputs(**_ready_inputs())

    assert report["status"] == "ready"
    assert report["can_resume_paper"] is True
    assert report["read_only"] is True
    assert report["starts_trading_service"] is False
    assert report["submits_orders"] is False
    assert report["changes_model_routing"] is False
    assert report["requires_operator_start"] is True
    assert report["blockers"] == []
    assert "okx_authoritative_sync_clean" in report["passed_checks"]
    assert "okx_account_equity_truth_available" in report["passed_checks"]
    assert "phase3_quant_api_endpoints_ready" in report["passed_checks"]


def test_phase3_paper_resume_preflight_blocks_okx_differences() -> None:
    okx = _okx_sync_clean()
    okx["issue_count"] = 1
    okx["issues"] = [{"kind": "okx_open_position_missing_locally"}]

    report = evaluate_phase3_paper_resume_preflight_inputs(
        **_ready_inputs(okx_authoritative_sync=okx)
    )

    assert report["status"] == "blocked"
    assert report["can_resume_paper"] is False
    assert "okx_authoritative_sync_has_differences" in {
        item["code"] for item in report["blockers"]
    }


def test_phase3_paper_resume_preflight_blocks_unhealthy_child_endpoint() -> None:
    runtime = _platform_runtime_ready()
    runtime["local_ai_tools"]["child_endpoints"]["exit_advice"]["available"] = False

    report = evaluate_phase3_paper_resume_preflight_inputs(
        **_ready_inputs(platform_runtime=runtime)
    )

    assert report["status"] == "blocked"
    assert "phase3_quant_api_child_endpoints_unavailable" in {
        item["code"] for item in report["blockers"]
    }


def test_phase3_paper_resume_preflight_blocks_without_okx_account_equity_truth() -> None:
    report = evaluate_phase3_paper_resume_preflight_inputs(
        **_ready_inputs(account_equity_truth={"available": False, "source": "okx_snapshot"})
    )

    assert report["status"] == "blocked"
    assert report["can_resume_paper"] is False
    assert "okx_account_equity_unavailable" in {
        item["code"] for item in report["blockers"]
    }


def test_phase3_paper_resume_preflight_blocks_local_account_equity_estimate() -> None:
    estimate = _account_equity_ready()
    estimate["source"] = "local_estimate"

    report = evaluate_phase3_paper_resume_preflight_inputs(
        **_ready_inputs(account_equity_truth=estimate)
    )

    assert report["status"] == "blocked"
    assert "okx_account_equity_not_exchange_truth" in {
        item["code"] for item in report["blockers"]
    }


def test_phase3_paper_resume_preflight_blocks_if_paper_already_running() -> None:
    report = evaluate_phase3_paper_resume_preflight_inputs(
        **_ready_inputs(platform_server=_platform_server_ready(paper_active=True))
    )

    assert report["status"] == "blocked"
    assert "paper_trading_already_active" in {item["code"] for item in report["blockers"]}


def test_phase3_paper_resume_preflight_allows_platform_model_endpoints_when_remote_audit_unverified() -> None:
    model_server = {
        "status": "unverified",
        "runtime_ready": False,
        "artifact_ready": False,
        "phase3_model_service_go_live_blocked": True,
        "blockers": [{"code": "model_server_config_error"}],
    }

    report = evaluate_phase3_paper_resume_preflight_inputs(
        **_ready_inputs(
            model_server_readiness=model_server,
            platform_runtime=_platform_runtime_ready_with_models(),
        )
    )

    warning_codes = {item["code"] for item in report["warnings"]}
    blocker_codes = {item["code"] for item in report["blockers"]}
    assert report["status"] == "ready_with_warnings"
    assert report["can_resume_paper"] is True
    assert "phase3_model_server_remote_audit_unverified" in warning_codes
    assert "phase3_model_server_runtime_not_ready" not in blocker_codes
    assert "phase3_model_server_platform_endpoints_ready" in report["passed_checks"]


def test_phase3_paper_resume_preflight_blocks_stale_specialist_report() -> None:
    specialist = _specialist_ready()
    specialist["generated_at"] = "2026-06-27T00:00:00+00:00"

    report = evaluate_phase3_paper_resume_preflight_inputs(
        **_ready_inputs(specialist_shadow_evaluation=specialist),
        specialist_report_max_age_seconds=1,
    )

    assert report["status"] == "blocked"
    assert "specialist_shadow_evaluation_stale" in {
        item["code"] for item in report["blockers"]
    }


@pytest.mark.asyncio
async def test_phase3_paper_resume_preflight_service_uses_injected_providers() -> None:
    service = Phase3PaperResumePreflightService(
        okx_sync_provider=_okx_sync_clean,
        okx_integrity_provider=_okx_integrity_clean,
        model_server_provider=_model_server_ready,
        platform_runtime_provider=_platform_runtime_ready,
        platform_server_provider=_platform_server_ready,
        specialist_shadow_provider=_specialist_ready,
        account_equity_provider=_account_equity_ready,
    )

    report = await service.report()

    assert report["status"] == "ready"
    assert report["can_resume_paper"] is True


def test_phase3_paper_resume_preflight_cli_writes_latest_report(tmp_path) -> None:
    report = {
        "status": "ready",
        "checked_at": "2026-06-27T08:30:00+00:00",
        "read_only": True,
        "can_resume_paper": True,
    }

    artifacts = preflight_cli.write_report(report, tmp_path, indent=2)

    latest_path = tmp_path / "latest.json"
    report_path = tmp_path / artifacts["report_path"].split("\\")[-1]
    assert latest_path.exists()
    assert report_path.exists()
    payload = json.loads(latest_path.read_text(encoding="utf-8"))
    assert payload["report_artifacts"] == artifacts
    assert payload["can_resume_paper"] is True
