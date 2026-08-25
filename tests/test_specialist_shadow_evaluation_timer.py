from __future__ import annotations

from scripts.install_specialist_shadow_evaluation_timer import (
    DEFAULT_TIMER_MINUTES,
    REPORT_DIR,
    REPORT_OWNER,
    SERVICE_NAME,
    TIMER_NAME,
    install_command,
    render_service,
    render_timer,
)


def test_specialist_shadow_evaluation_service_is_read_only_oneshot() -> None:
    service = render_service(hours=72)

    assert "Type=oneshot" in service
    assert "User=bb" in service
    assert "scripts/run_specialist_shadow_evaluation.py" in service
    assert "--hours 72" in service
    assert "--limit" not in service
    assert f"--output-dir {REPORT_DIR}" in service
    assert "bb-paper-trading.service" not in service
    assert "systemctl start" not in service
    assert "live" not in service.lower()
    assert "Environment=MALLOC_ARENA_MAX=2" in service
    assert "Environment=OMP_NUM_THREADS=1" in service
    assert "MemoryHigh=2G" in service
    assert "MemoryMax=3G" in service
    assert "TasksMax=128" in service


def test_specialist_shadow_evaluation_timer_runs_periodically() -> None:
    timer = render_timer(minutes=15)

    assert f"Unit={SERVICE_NAME}" in timer
    assert "OnActiveSec=15min" in timer
    assert "OnUnitActiveSec=15min" in timer
    assert "OnBootSec" not in timer
    assert "Persistent=true" in timer


def test_specialist_shadow_evaluation_timer_default_stays_inside_report_freshness_window() -> None:
    assert DEFAULT_TIMER_MINUTES == 90
    timer = render_timer()
    assert f"OnUnitActiveSec={DEFAULT_TIMER_MINUTES}min" in timer
    assert DEFAULT_TIMER_MINUTES < 2 * 60


def test_specialist_shadow_evaluation_install_command_keeps_paper_inactive_probe() -> None:
    command = install_command(minutes=30, hours=168)

    assert SERVICE_NAME in command
    assert TIMER_NAME in command
    assert "systemctl reset-failed bb-specialist-shadow-evaluation.service" in command
    assert "systemctl enable --now bb-specialist-shadow-evaluation.timer" in command
    assert "systemctl start bb-specialist-shadow-evaluation.service" not in command
    assert f"install -d -o 'bb' -g 'bb' -m 0775 '{REPORT_DIR}'" in command
    assert f"test -d '{REPORT_DIR}'" in command
    assert "specialist_shadow_evaluation_latest.json" not in command
    assert "systemctl is-active bb-paper-trading.service" in command
    assert "systemctl start bb-paper-trading.service" not in command


def test_specialist_shadow_evaluation_timer_writes_primary_phase3_data_dir() -> None:
    assert REPORT_DIR == "/data/bb/app/data/phase3"
    assert REPORT_OWNER == "bb:bb"
