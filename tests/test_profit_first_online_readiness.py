from __future__ import annotations

from scripts import verify_profit_first_online_readiness as verify


def test_profit_first_online_readiness_remote_command_is_read_only() -> None:
    command = verify._remote_validation_command("/data/bb/app")

    forbidden = [
        "systemctl start bb-paper-trading.service",
        "systemctl restart bb-paper-trading.service",
        "systemctl enable --now bb-paper-trading.service",
        "place_order",
        "set_leverage",
        "live_weight",
    ]
    for text in forbidden:
        assert text not in command

    assert "systemctl is-active bb-paper-trading.service || true" in command
    assert "scripts/run_profit_first_governance_report.py --stdout-only" in command
    assert "scripts/run_phase3_go_no_go_report.py --stdout-only" in command
    assert "scripts/plan_profit_first_recovery_repairs.py --stdout-only" in command
    assert "scripts/plan_profit_first_historical_recovery_package.py" in command
    assert "--skip-current-blockers" in command
    assert '"recovery_repair_plan": {' in command
    assert '"historical_recovery_package": {' in command
    assert '"blocking_actions": [' in command
    assert '"target": item.get("target") if isinstance(item.get("target"), dict) else {}' in command
    assert '"mutates_database": recovery_plan.get("mutates_database")' in command
    assert '"mutates_database": historical_package.get("mutates_database")' in command
    assert '"starts_trading_service": False' in command
    assert '"submits_orders": False' in command
    assert '"changes_model_routing": False' in command
    assert '"changes_live_sizing": False' in command
    assert "resumed_observing" in command
    assert "post_resume_observing" in command
    assert "paper_observation_healthy" in command


def test_profit_first_online_readiness_connection_failure_is_not_resume_ready(monkeypatch) -> None:
    def fail_connect(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(verify, "connect_remote_ssh", fail_connect)

    report = verify.collect_online_readiness()

    assert report["validation_status"] == "unavailable"
    assert report["resume_allowed_by_this_check"] is False
    assert report["read_only"] is True
    assert report["starts_trading_service"] is False
    assert report["submits_orders"] is False
