from __future__ import annotations

from scripts import install_model_candidate_evidence_refresh_timer as installer


def test_candidate_evidence_refresh_timer_is_bounded_and_fail_closed() -> None:
    service = installer.render_service()
    timer = installer.render_timer()

    assert "User=linux" in service
    assert "refresh_model_candidate_evidence.py" in service
    assert "TimeoutStartSec=45min" in service
    assert "bb-phase3-llm-target.service" in service
    assert "OnCalendar=*-*-* 03:35:00" in timer
    assert "Persistent=true" in timer
    assert "RandomizedDelaySec=1200" in timer


def test_candidate_evidence_refresh_installer_dry_run_does_not_connect(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        installer,
        "load_model_server_info_from_platform",
        lambda *_args: (_ for _ in ()).throw(AssertionError("dry run connected")),
    )

    installer.install_timer(dry_run=True)

    output = capsys.readouterr().out
    assert installer.SERVICE_NAME in output
    assert "OnCalendar=*-*-* 03:35:00" in output
