from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import run_phase3_paper_resume_observation as observation_cli
from services.phase3_paper_resume_observation import (
    Phase3PaperResumeObservationService,
    evaluate_phase3_paper_resume_observation_inputs,
)


def _samples(**overrides: Any) -> dict[str, Any]:
    payload = {
        "status": "ok",
        "created_shadow_count": 8,
        "completed_shadow_count": 2,
        "created_order_count": 1,
        "filled_order_count": 1,
        "open_position_count": 0,
        "trade_reflection_count": 0,
    }
    payload.update(overrides)
    return payload


def _platform_server(*, paper_active: bool = True) -> dict[str, Any]:
    return {
        "available": True,
        "status": "ok",
        "services": [
            {"name": "bb-dashboard.service", "active": True},
            {"name": "bb-model-tunnels.service", "active": True},
            {"name": "bb-paper-trading.service", "active": paper_active},
        ],
    }


def _platform_runtime() -> dict[str, Any]:
    return {
        "local_ai_tools": {
            "available": True,
            "child_endpoints": {
                "profit_prediction": {"available": True},
                "time_series_prediction": {"available": True},
                "sentiment_analysis": {"available": True},
                "exit_advice": {"available": True},
            },
        }
    }


def _okx_clean() -> dict[str, Any]:
    return {
        "status": "ok",
        "okx_pull_available": True,
        "issue_count": 0,
        "read_only": True,
    }


def _trading_runtime_clean() -> dict[str, Any]:
    return {
        "available": True,
        "running": True,
        "heartbeat_age_seconds": 12.0,
        "decision_interval": 30,
        "okx_authoritative_sync": {
            "status": "ok",
            "last_requires_attention_count": 0,
            "last_success_age_seconds": 20.0,
            "stale_after_seconds": 180.0,
            "last_success_at": "2026-07-04T00:00:00+00:00",
        },
    }


def _specialist() -> dict[str, Any]:
    return {
        "available": True,
        "generated_at": datetime.now(UTC).isoformat(),
        "eligible_shadow_count": 2,
        "live_mutation": False,
    }


def _preflight() -> dict[str, Any]:
    return {"status": "ready", "can_resume_paper": True}


def _ready_inputs(**overrides: dict[str, Any]) -> dict[str, dict[str, Any]]:
    values = {
        "sample_summary": _samples(),
        "platform_server": _platform_server(),
        "platform_runtime": _platform_runtime(),
        "okx_authoritative_sync": _okx_clean(),
        "specialist_shadow_evaluation": _specialist(),
        "latest_preflight": _preflight(),
    }
    values.update(overrides)
    return values


def test_paper_resume_observation_is_healthy_when_samples_and_gates_pass() -> None:
    report = evaluate_phase3_paper_resume_observation_inputs(**_ready_inputs())

    assert report["status"] == "healthy"
    assert report["paper_active"] is True
    assert report["can_use_for_promotion"] is True
    assert report["starts_trading_service"] is False
    assert report["submits_orders"] is False
    assert report["blockers"] == []
    assert "post_resume_shadow_counts_are_diagnostic_only" in report["passed_checks"]


def test_paper_resume_observation_waits_when_paper_is_inactive() -> None:
    report = evaluate_phase3_paper_resume_observation_inputs(
        **_ready_inputs(platform_server=_platform_server(paper_active=False))
    )

    assert report["status"] == "waiting_for_resume"
    assert report["paper_active"] is False
    assert report["can_use_for_promotion"] is False
    assert "paper_trading_not_active" in {item["code"] for item in report["warnings"]}


def test_paper_resume_observation_blocks_on_okx_difference() -> None:
    okx = _okx_clean()
    okx["issue_count"] = 1
    okx["issues"] = [{"kind": "okx_open_position_missing_locally"}]

    report = evaluate_phase3_paper_resume_observation_inputs(
        **_ready_inputs(
            okx_authoritative_sync=okx,
            trading_runtime_status=_trading_runtime_clean(),
        )
    )

    assert report["status"] == "critical"
    assert "okx_authoritative_sync_has_post_resume_differences" in {
        item["code"] for item in report["blockers"]
    }


def test_paper_resume_observation_warns_on_read_only_okx_pull_timeout_when_runtime_is_clean() -> None:
    okx = _okx_clean()
    okx.update(
        {
            "status": "warning",
            "okx_pull_available": False,
            "fetch_errors": ["positions: TimeoutError"],
            "error": "TimeoutError",
        }
    )

    report = evaluate_phase3_paper_resume_observation_inputs(
        **_ready_inputs(
            okx_authoritative_sync=okx,
            trading_runtime_status=_trading_runtime_clean(),
        )
    )

    assert report["status"] == "warming_up"
    assert report["blockers"] == []
    assert report["summary"]["runtime_okx_sync_clean"] is True
    assert "okx_runtime_authoritative_sync_clean_after_resume" in report["passed_checks"]
    assert "okx_authoritative_audit_pull_degraded_runtime_clean" in {
        item["code"] for item in report["warnings"]
    }


def test_paper_resume_observation_blocks_on_okx_pull_timeout_when_runtime_is_not_clean() -> None:
    okx = _okx_clean()
    okx.update(
        {
            "status": "warning",
            "okx_pull_available": False,
            "fetch_errors": ["positions: TimeoutError"],
            "error": "TimeoutError",
        }
    )
    runtime = _trading_runtime_clean()
    runtime["okx_authoritative_sync"]["status"] = "stale"

    report = evaluate_phase3_paper_resume_observation_inputs(
        **_ready_inputs(
            okx_authoritative_sync=okx,
            trading_runtime_status=runtime,
        )
    )

    assert report["status"] == "critical"
    assert report["summary"]["runtime_okx_sync_clean"] is False
    assert "okx_authoritative_sync_unhealthy_after_resume" in {
        item["code"] for item in report["blockers"]
    }


def test_paper_resume_observation_does_not_use_fixed_sample_floors() -> None:
    report = evaluate_phase3_paper_resume_observation_inputs(
        **_ready_inputs(sample_summary=_samples(created_shadow_count=1, completed_shadow_count=0))
    )

    assert report["status"] == "healthy"
    assert report["can_use_for_promotion"] is True
    codes = {item["code"] for item in report["warnings"]}
    assert "post_resume_shadow_sample_floor_not_met" not in codes
    assert "post_resume_completed_shadow_floor_not_met" not in codes
    assert "post_resume_shadow_counts_are_diagnostic_only" in report["passed_checks"]


def test_paper_resume_observation_treats_active_paper_preflight_as_consumed() -> None:
    preflight = {
        "status": "blocked",
        "can_resume_paper": False,
        "blockers": [{"code": "paper_trading_already_active"}],
    }

    report = evaluate_phase3_paper_resume_observation_inputs(
        **_ready_inputs(latest_preflight=preflight)
    )

    codes = {item["code"] for item in report["warnings"]}
    assert "latest_preflight_not_ready" not in codes
    assert "latest_preflight_consumed_after_resume" in report["passed_checks"]
    assert report["summary"]["preflight_consumed_after_resume"] is True


@pytest.mark.asyncio
async def test_paper_resume_observation_service_uses_injected_providers() -> None:
    service = Phase3PaperResumeObservationService(
        sample_summary_provider=_samples,
        platform_server_provider=_platform_server,
        platform_runtime_provider=_platform_runtime,
        okx_sync_provider=_okx_clean,
        specialist_shadow_provider=_specialist,
        latest_preflight_provider=_preflight,
    )

    report = await service.report()

    assert report["status"] == "healthy"
    assert report["summary"]["completed_shadow_count"] == 2


def test_paper_resume_observation_cli_writes_latest_report(tmp_path) -> None:
    report = {
        "status": "waiting_for_resume",
        "checked_at": "2026-06-27T10:00:00+00:00",
        "read_only": True,
    }

    artifacts = observation_cli.write_report(report, tmp_path, indent=2)

    latest_path = tmp_path / "latest.json"
    report_path = tmp_path / artifacts["report_path"].split("\\")[-1]
    assert latest_path.exists()
    assert report_path.exists()
    payload = json.loads(latest_path.read_text(encoding="utf-8"))
    assert payload["report_artifacts"] == artifacts


@pytest.mark.asyncio
async def test_paper_resume_observation_cli_drops_runtime_user_before_collection(
    monkeypatch,
) -> None:
    events: list[str] = []

    async def fake_collect(**_kwargs: Any) -> dict[str, Any]:
        events.append("collect")
        return {"status": "warming_up", "checked_at": datetime.now(UTC).isoformat()}

    def fake_drop(*, project_root: Path) -> bool:
        assert project_root == observation_cli.ROOT
        events.append("drop")
        return True

    monkeypatch.setattr(observation_cli, "drop_privileges_to_runtime_user_if_needed", fake_drop)
    monkeypatch.setattr(observation_cli, "collect_phase3_paper_resume_observation", fake_collect)
    monkeypatch.setattr(
        observation_cli,
        "parse_args",
        lambda: SimpleNamespace(
            observation_hours=2,
            report_max_age_seconds=7200,
            json_indent=0,
            output_dir=None,
            stdout_only=True,
            fail_on_critical=False,
        ),
    )

    exit_code = await observation_cli._main()

    assert exit_code == 0
    assert events == ["drop", "collect"]
