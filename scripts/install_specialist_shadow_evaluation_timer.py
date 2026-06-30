"""Install the Phase 3 specialist shadow evaluation systemd timer online."""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text
from core.safe_output import safe_print

REMOTE_APP_DIR = "/data/bb/app"
SERVICE_NAME = "bb-specialist-shadow-evaluation.service"
TIMER_NAME = "bb-specialist-shadow-evaluation.timer"
REPORT_DIR = "/data/bb/app/data/phase3"
REPORT_OWNER = "bb:bb"


def sh(value: str) -> str:
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


def render_service(*, hours: int = 168, limit: int = 2000) -> str:
    return (
        textwrap.dedent(
            f"""
            [Unit]
            Description=BB Phase 3 Specialist Shadow Evaluation Report
            After=network-online.target postgresql.service
            Wants=network-online.target

            [Service]
            Type=oneshot
            User=bb
            WorkingDirectory={REMOTE_APP_DIR}
            EnvironmentFile=-{REMOTE_APP_DIR}/.env
            EnvironmentFile=-/etc/bb/bb-runtime.env
            ExecStart={REMOTE_APP_DIR}/.venv/bin/python {REMOTE_APP_DIR}/scripts/run_specialist_shadow_evaluation.py --hours {int(hours)} --limit {int(limit)} --output-dir {REPORT_DIR}
            """
        ).strip()
        + "\n"
    )


def render_timer(*, minutes: int = 30) -> str:
    return (
        textwrap.dedent(
            f"""
            [Unit]
            Description=Run BB Phase 3 Specialist Shadow Evaluation every {int(minutes)} minutes

            [Timer]
            OnBootSec=5min
            OnUnitActiveSec={int(minutes)}min
            AccuracySec=1min
            Persistent=true
            Unit={SERVICE_NAME}

            [Install]
            WantedBy=timers.target
            """
        ).strip()
        + "\n"
    )


def install_command(*, minutes: int, hours: int, limit: int) -> str:
    service = render_service(hours=hours, limit=limit)
    timer = render_timer(minutes=minutes)
    return "\n".join(
        [
            "set -e",
            f"cd {sh(REMOTE_APP_DIR)}",
            "./.venv/bin/python -m py_compile scripts/run_specialist_shadow_evaluation.py services/specialist_shadow_evaluation.py",
            f"install -d -o {sh(REPORT_OWNER.split(':', 1)[0])} -g {sh(REPORT_OWNER.split(':', 1)[1])} -m 0775 {sh(REPORT_DIR)}",
            f"cat > /tmp/{SERVICE_NAME} <<'UNIT'\n{service}UNIT",
            f"cat > /tmp/{TIMER_NAME} <<'UNIT'\n{timer}UNIT",
            f"install -m 0644 /tmp/{SERVICE_NAME} /etc/systemd/system/{SERVICE_NAME}",
            f"install -m 0644 /tmp/{TIMER_NAME} /etc/systemd/system/{TIMER_NAME}",
            "systemctl daemon-reload",
            f"systemctl enable --now {TIMER_NAME}",
            f"systemctl start {SERVICE_NAME}",
            f"systemctl is-active {TIMER_NAME}",
            f"systemctl status {SERVICE_NAME} --no-pager -n 20 || true",
            f"test -f {sh(REPORT_DIR + '/specialist_shadow_evaluation_latest.json')}",
            "printf 'paper='; systemctl is-active bb-paper-trading.service || true",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install a read-only Phase 3 specialist shadow evaluation timer."
    )
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--hours", type=int, default=168)
    parser.add_argument("--limit", type=int, default=2000)
    args = parser.parse_args()

    ssh = connect_remote_ssh(ROOT, timeout=20)
    try:
        output = run_remote_text(
            ssh,
            install_command(minutes=args.minutes, hours=args.hours, limit=args.limit),
            timeout=180,
        )
        safe_print(output)
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
