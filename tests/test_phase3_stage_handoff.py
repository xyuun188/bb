from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from services.phase3_stage_handoff import evaluate_phase3_stage_handoff_inputs


def _inputs() -> dict:
    now = datetime.now(UTC).isoformat()
    return {
        "go_no_go_report": {
            "available": True,
            "checked_at": now,
            "go_no_go": {"ready": True, "status": "go", "blockers": []},
        },
        "observation_report": {"available": True, "checked_at": now},
        "specialist_shadow_report": {
            "available": True,
            "checked_at": now,
            "live_mutation": False,
            "production_permission": False,
        },
        "rebuild_preflight_report": {"available": True, "checked_at": now},
        "okx_daily_report": {
            "available": True,
            "checked_at": now,
            "issue_ledger_summary": {"unresolved": 0},
        },
    }


def test_phase3_stage_handoff_is_permissionless_when_return_gate_is_ready() -> None:
    report = evaluate_phase3_stage_handoff_inputs(**_inputs())

    assert report["status"] == "dynamic_return_ready"
    assert report["ready"] is True
    assert report["audit_only"] is True
    assert report["read_only"] is True
    assert report["production_permission"] is False
    assert report["starts_trading_service"] is False
    assert report["submits_orders"] is False
    assert report["changes_model_routing"] is False
    assert report["changes_model_promotion"] is False


def test_phase3_stage_handoff_blocks_when_return_gate_is_not_ready() -> None:
    payload = _inputs()
    payload["go_no_go_report"]["go_no_go"] = {
        "ready": False,
        "status": "no_go",
        "blockers": [{"code": "execution_contract_gap"}],
    }

    report = evaluate_phase3_stage_handoff_inputs(**payload)

    assert report["status"] == "blocked"
    assert "dynamic_return_gate_not_ready" in {
        item["code"] for item in report["blockers"]
    }


def test_phase3_stage_handoff_blocks_stale_gate_and_unresolved_okx_facts() -> None:
    payload = _inputs()
    payload["go_no_go_report"]["checked_at"] = (
        datetime.now(UTC) - timedelta(hours=4)
    ).isoformat()
    payload["okx_daily_report"]["issue_ledger_summary"]["unresolved"] = 2

    report = evaluate_phase3_stage_handoff_inputs(
        **payload,
        report_max_age_seconds=60,
    )

    codes = {item["code"] for item in report["blockers"]}
    assert "go_no_go_report_stale" in codes
    assert "okx_trade_facts_unresolved" in codes


def test_phase3_stage_handoff_blocks_specialist_production_claim() -> None:
    payload = _inputs()
    payload["specialist_shadow_report"]["production_permission"] = True

    report = evaluate_phase3_stage_handoff_inputs(**payload)

    assert report["status"] == "blocked"
    assert "specialist_observation_boundary_violated" in {
        item["code"] for item in report["blockers"]
    }


def test_phase3_stage_handoff_report_is_json_serializable() -> None:
    report = evaluate_phase3_stage_handoff_inputs(**_inputs())
    assert json.loads(json.dumps(report))["status"] == "dynamic_return_ready"
