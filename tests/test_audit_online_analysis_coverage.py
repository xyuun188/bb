from __future__ import annotations

from datetime import UTC, datetime, timedelta

from scripts import audit_online_analysis_coverage as audit


def _report(now: datetime) -> dict:
    window_started_at = now - timedelta(minutes=30)
    return {
        "generated_at": now.isoformat(),
        "window_minutes": 30,
        "window_started_at": window_started_at.isoformat(),
        "mode": "paper",
        "paused": False,
        "services": {
            "trading": {
                "active_state": "active",
                "sub_state": "running",
                "restart_count": 0,
                "active_enter_at": (window_started_at - timedelta(minutes=1)).isoformat(),
            },
            "dashboard": {
                "active_state": "active",
                "sub_state": "running",
                "restart_count": 0,
                "active_enter_at": (window_started_at - timedelta(minutes=1)).isoformat(),
            },
        },
        "coverage": {
            "monitoring_active": True,
            "coverage_window_evaluable": True,
            "coverage_window_met": True,
            "overdue_count": 0,
            "candidate_coverage_evidence_available": True,
            "candidate_selection_generated_at": (now - timedelta(seconds=30)).isoformat(),
            "candidate_count": 12,
            "candidate_coverage_target_seconds": 30 * 60,
            "coverage_due_count": 0,
            "coverage_due_symbols": [],
        },
        "market": {
            "decision_count": 12,
            "max_activity_gap_seconds": 120.0,
        },
        "position_review": {
            "open_position_count": 2,
            "decision_count": 8,
            "last_decision_at": (now - timedelta(seconds=30)).isoformat(),
        },
    }


def test_assessment_accepts_active_coverage_and_position_review() -> None:
    now = datetime(2026, 7, 25, 16, 0, tzinfo=UTC)

    assessment = audit.assess_coverage_report(_report(now), now=now)

    assert assessment["ready"] is True
    assert assessment["blockers"] == []


def test_assessment_rejects_paused_or_unevaluable_coverage() -> None:
    now = datetime(2026, 7, 25, 16, 0, tzinfo=UTC)
    report = _report(now)
    report["paused"] = True
    report["coverage"] = {
        "monitoring_active": False,
        "coverage_window_evaluable": False,
        "coverage_window_met": None,
        "overdue_count": 0,
    }
    report["market"] = {
        "decision_count": 0,
        "max_activity_gap_seconds": 1800.0,
    }

    assessment = audit.assess_coverage_report(report, now=now)

    assert assessment["ready"] is False
    assert "paper_trading_paused" in assessment["blockers"]
    assert "market_coverage_window_not_evaluable" in assessment["blockers"]
    assert "market_analysis_activity_missing" in assessment["blockers"]


def test_assessment_requires_current_position_review_when_positions_exist() -> None:
    now = datetime(2026, 7, 25, 16, 0, tzinfo=UTC)
    report = _report(now)
    report["position_review"]["last_decision_at"] = (now - timedelta(minutes=10)).isoformat()

    assessment = audit.assess_coverage_report(report, now=now)

    assert assessment["ready"] is False
    assert assessment["blockers"] == ["position_review_activity_stale"]


def test_assessment_requires_full_window_of_service_continuity() -> None:
    now = datetime(2026, 7, 25, 16, 0, tzinfo=UTC)
    report = _report(now)
    report["services"]["trading"]["active_enter_at"] = (now - timedelta(minutes=5)).isoformat()

    assessment = audit.assess_coverage_report(report, now=now)

    assert assessment["ready"] is False
    assert assessment["blockers"] == ["trading_service_continuity_unproven"]


def test_assessment_rejects_unresolved_due_market_candidates() -> None:
    now = datetime(2026, 7, 25, 16, 0, tzinfo=UTC)
    report = _report(now)
    report["coverage"]["coverage_window_met"] = False
    report["coverage"]["coverage_due_count"] = 1
    report["coverage"]["coverage_due_symbols"] = ["ETH/USDT"]

    assessment = audit.assess_coverage_report(report, now=now)

    assert assessment["ready"] is False
    assert "market_coverage_window_not_met" in assessment["blockers"]
    assert "market_candidate_coverage_has_unresolved_due_symbols" in assessment["blockers"]


def test_assessment_rejects_stale_candidate_selection_evidence() -> None:
    now = datetime(2026, 7, 25, 16, 0, tzinfo=UTC)
    report = _report(now)
    report["coverage"]["candidate_selection_generated_at"] = (
        now - timedelta(minutes=11)
    ).isoformat()

    assessment = audit.assess_coverage_report(report, now=now)

    assert assessment["ready"] is False
    assert assessment["blockers"] == ["market_candidate_coverage_evidence_stale"]


def test_assessment_rejects_shortened_acceptance_window() -> None:
    now = datetime(2026, 7, 25, 16, 0, tzinfo=UTC)
    report = _report(now)
    report["window_minutes"] = 5
    report["window_started_at"] = (now - timedelta(minutes=5)).isoformat()

    assessment = audit.assess_coverage_report(report, now=now)

    assert assessment["ready"] is False
    assert "observation_window_too_short" in assessment["blockers"]


def test_remote_script_compiles_and_reads_only_runtime_evidence() -> None:
    script = audit._remote_script(remote_app_dir="/data/bb/app", window_minutes=30)

    compile(script, "<online-analysis-coverage>", "exec")
    assert "trading_runtime_status.json" in script
    assert "market_analysis_deferred" in script
    assert "AIDecision.analysis_type" in script
    assert "Position.is_open.is_(True)" in script
    assert "bb-paper-trading.service" in script
    assert "ActiveEnterTimestamp" in script
    assert "strict=True" not in script


def test_decode_remote_json_ignores_logs_before_report() -> None:
    payload = audit._decode_remote_json(
        'runtime log\n{"event":"noise"}\n{"mode":"paper","paused":true}\n'
    )

    assert payload == {"mode": "paper", "paused": True}
