from __future__ import annotations

import json
from typing import Any

import pytest

from scripts import run_phase3_rebuild_preflight as preflight


def test_phase3_rebuild_preflight_imports_online_runtime_bootstrap() -> None:
    source = preflight.ROOT.joinpath("scripts", "run_phase3_rebuild_preflight.py").read_text(
        encoding="utf-8"
    )

    assert "from scripts.runtime_env_bootstrap import" in source
    assert "load_runtime_env_files(project_root=ROOT)" in source
    assert "drop_privileges_to_runtime_user_if_needed(project_root=ROOT)" in source


def _training_payload() -> dict[str, Any]:
    quality_report = {
        "data_quality_version": "test.v1",
        "totals": {
            "total": 580,
            "included": 580,
            "downweighted": 0,
            "excluded": 0,
            "effective_weight_ratio": 0.92,
        },
    }
    governance_report = {
        "status": "clean",
        "contamination_risk": "low",
        "trainable_sample_count": 580,
        "excluded_sample_count": 0,
    }
    return {
        "payload": {
            "shadow_samples": [{"id": index} for index in range(500)],
            "trade_samples": [{"id": index} for index in range(80)],
            "sequence_samples": [{"id": index} for index in range(12)],
            "text_sentiment_samples": [{"id": index} for index in range(7)],
            "quality_report": quality_report,
            "governance_report": governance_report,
        },
        "completed_shadow_sample_count": 500,
        "completed_trade_sample_count": 80,
        "raw_shadow_sample_count": 500,
        "trainable_shadow_sample_count": 500,
        "raw_trade_sample_count": 84,
        "trainable_trade_sample_count": 80,
        "quarantined_trade_sample_count": 4,
        "sequence_sample_count": 12,
        "text_sentiment_sample_count": 7,
    }


async def _fake_training_payload(**_kwargs: Any) -> dict[str, Any]:
    return _training_payload()


async def _fake_historical_report(**_kwargs: Any) -> dict[str, Any]:
    return {
        "status": "clean",
        "read_only": True,
        "training_policy": "clean_training_view_only",
        "trainable_closed_positions": 80,
        "quarantined_closed_positions": 0,
    }


async def _fake_artifact_report() -> dict[str, Any]:
    return {
        "status": "retired_required",
        "read_only": True,
        "audit_only": True,
        "retired_or_untrusted_count": 2,
    }


async def _fake_runtime_report(*, include_runtime_probe: bool) -> dict[str, Any]:
    return {"status": "ok", "included": include_runtime_probe}


def _patch_preflight_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preflight, "_collect_training_payload", _fake_training_payload)
    monkeypatch.setattr(preflight, "_historical_trade_fact_report", _fake_historical_report)
    monkeypatch.setattr(preflight, "_artifact_retirement_report", _fake_artifact_report)
    monkeypatch.setattr(preflight, "_runtime_probe_report", _fake_runtime_report)
    monkeypatch.setattr(
        preflight,
        "load_latest_paper_observation_report",
        lambda **_kwargs: {
            "available": True,
            "status": "healthy",
            "paper_active": True,
            "can_use_for_promotion": True,
            "starts_trading_service": False,
            "submits_orders": False,
            "changes_model_routing": False,
            "blockers": [],
            "warnings": [],
        },
    )


@pytest.mark.asyncio
async def test_phase3_rebuild_preflight_is_read_only_and_lists_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_preflight_inputs(monkeypatch)

    report = await preflight.collect_phase3_rebuild_preflight(
        shadow_limit=500,
        trade_limit=80,
        sequence_limit=12,
        text_limit=7,
        local_ai_tools_base_url="http://127.0.0.1:8001",
    )

    assert report["status"] == "ready_with_warnings"
    assert report["read_only"] is True
    assert report["mutates_database"] is False
    assert report["writes_artifacts"] is False
    assert report["starts_trading_service"] is False
    assert report["readiness"]["can_run_confirmed_rebuild"] is True
    assert report["readiness"]["can_persist_artifact"] is False
    assert report["paper_observation_report"]["status"] == "healthy"
    assert "paper_observation_report_missing" not in report["promotion_recommendation"][
        "canary_blocking_reasons"
    ]
    assert report["training_summary"]["trainable_shadow_sample_count"] == 500
    assert report["commands"]["ml_signal"]["preflight_command"] == (
        "python scripts/train_ml_signal_model.py"
    )
    assert "--persist-artifact --confirm-phase3-rebuild" in report["commands"]["ml_signal"][
        "confirmed_rebuild_command"
    ]
    assert "--base-url http://127.0.0.1:8001" in report["commands"]["local_ai_tools"][
        "preflight_command"
    ]
    assert report["operator_sequence"][0].startswith("Review this report")


def test_phase3_rebuild_preflight_writes_dated_and_latest_report(tmp_path) -> None:
    report = {
        "status": "blocked",
        "checked_at": "2026-06-27T00:45:00+00:00",
        "read_only": True,
        "writes_artifacts": False,
        "readiness": {"status": "blocked"},
    }

    artifacts = preflight.write_report(report, tmp_path, indent=2)

    report_path = tmp_path / artifacts["report_path"].split("\\")[-1]
    latest_path = tmp_path / "latest.json"
    assert report_path.exists()
    assert latest_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["report_artifacts"] == artifacts
    assert payload["writes_artifacts"] is False
    assert latest_path.read_text(encoding="utf-8") == report_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_phase3_rebuild_preflight_blocks_unconfirmed_artifact_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_preflight_inputs(monkeypatch)

    report = await preflight.collect_phase3_rebuild_preflight(
        shadow_limit=500,
        trade_limit=80,
        sequence_limit=12,
        text_limit=7,
        requested_persist_artifact=True,
        confirm_phase3_rebuild=False,
    )

    assert report["status"] == "blocked"
    assert report["readiness"]["can_run_confirmed_rebuild"] is False
    assert report["readiness"]["can_persist_artifact"] is False
    assert "confirmed_rebuild_required_for_artifact_write" in report["readiness"]["blockers"]


@pytest.mark.asyncio
async def test_phase3_rebuild_preflight_reports_confirmed_rebuild_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_preflight_inputs(monkeypatch)

    report = await preflight.collect_phase3_rebuild_preflight(
        shadow_limit=500,
        trade_limit=80,
        sequence_limit=12,
        text_limit=7,
        requested_persist_artifact=True,
        confirm_phase3_rebuild=True,
    )

    assert report["status"] == "ready_with_warnings"
    assert report["readiness"]["can_run_confirmed_rebuild"] is True
    assert report["readiness"]["can_persist_artifact"] is True
    assert report["readiness"]["target_artifacts"]["ml_signal"]["target_stage"] == "shadow"
    assert report["readiness"]["live_mutation"] is False


@pytest.mark.asyncio
async def test_phase3_rebuild_preflight_returns_structured_blocked_report_on_collection_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_training_payload(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("database unavailable")

    async def fail_historical_report(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("history audit unavailable")

    async def fail_artifact_report() -> dict[str, Any]:
        raise RuntimeError("artifact audit unavailable")

    monkeypatch.setattr(preflight, "_collect_training_payload", fail_training_payload)
    monkeypatch.setattr(preflight, "_historical_trade_fact_report", fail_historical_report)
    monkeypatch.setattr(preflight, "_artifact_retirement_report", fail_artifact_report)
    monkeypatch.setattr(preflight, "_runtime_probe_report", _fake_runtime_report)

    report = await preflight.collect_phase3_rebuild_preflight(
        shadow_limit=1,
        trade_limit=1,
        sequence_limit=1,
        text_limit=1,
    )

    assert report["status"] == "blocked"
    assert report["read_only"] is True
    assert report["writes_artifacts"] is False
    assert report["collection_errors"] == {
        "training_payload": "database unavailable",
        "historical_trade_fact_audit": "history audit unavailable",
        "artifact_retirement_audit": "artifact audit unavailable",
    }
    assert "clean_training_view_unavailable" in report["readiness"]["blockers"]
    assert "historical_trade_fact_audit_unavailable" in report["readiness"]["blockers"]
    assert "artifact_retirement_audit_unavailable" in report["readiness"]["blockers"]


@pytest.mark.asyncio
async def test_phase3_rebuild_preflight_cli_writes_report_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_preflight_inputs(monkeypatch)

    monkeypatch.setattr(
        preflight,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "shadow_limit": 500,
                "trade_limit": 80,
                "sequence_limit": 12,
                "text_limit": 7,
                "historical_audit_days": 180,
                "historical_audit_limit": 5000,
                "include_runtime_probe": False,
                "local_ai_tools_base_url": "",
                "persist_artifact": False,
                "confirm_phase3_rebuild": False,
                "json_indent": 0,
                "output_dir": tmp_path,
                "stdout_only": False,
                "fail_on_blocked": False,
            },
        )(),
    )

    exit_code = await preflight._main()

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["report_artifacts"]["latest_path"].endswith("latest.json")
    assert payload["writes_artifacts"] is False
    assert (tmp_path / "latest.json").exists()


@pytest.mark.asyncio
async def test_phase3_rebuild_preflight_cli_stdout_only_does_not_write_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_preflight_inputs(monkeypatch)

    monkeypatch.setattr(
        preflight,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "shadow_limit": 500,
                "trade_limit": 80,
                "sequence_limit": 12,
                "text_limit": 7,
                "historical_audit_days": 180,
                "historical_audit_limit": 5000,
                "include_runtime_probe": False,
                "local_ai_tools_base_url": "",
                "persist_artifact": False,
                "confirm_phase3_rebuild": False,
                "json_indent": 0,
                "output_dir": tmp_path,
                "stdout_only": True,
                "fail_on_blocked": False,
            },
        )(),
    )

    exit_code = await preflight._main()

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert "report_artifacts" not in payload
    assert not (tmp_path / "latest.json").exists()
