#!/usr/bin/env python3
"""Install the online systemd timer for the Phase 3 model-server readiness report."""

from __future__ import annotations

import argparse
import posixpath
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402

REMOTE_APP_DIR = "/data/bb/app"
REMOTE_OWNER = "bb:bb"
SERVICE_NAME = "bb-phase3-model-server-readiness.service"
TIMER_NAME = "bb-phase3-model-server-readiness.timer"
REMOTE_RUNTIME_ENV_PATH = "/etc/bb/bb-runtime.env"
REPORT_DIR_REL = "data/phase3_model_server_readiness_reports"
DEFAULT_ON_CALENDAR = "*-*-* 00:55:00"


def _remote_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _owner_parts(owner: str) -> tuple[str, str]:
    user, _sep, group = str(owner or REMOTE_OWNER).partition(":")
    user = user or "bb"
    group = group or user
    return user, group


def render_service(*, remote_app_dir: str = REMOTE_APP_DIR, owner: str = REMOTE_OWNER) -> str:
    user, group = _owner_parts(owner)
    return f"""[Unit]
Description=BB Phase 3 model-server readiness report
After=network-online.target postgresql.service redis-server.service redis.service
Wants=network-online.target

[Service]
Type=oneshot
User={user}
Group={group}
WorkingDirectory={remote_app_dir}
EnvironmentFile=-{remote_app_dir}/.env
EnvironmentFile={REMOTE_RUNTIME_ENV_PATH}
ExecStart=/bin/bash -lc 'cd {remote_app_dir} && if [ -x .venv/bin/python ]; then PY=.venv/bin/python; elif [ -x venv/bin/python ]; then PY=venv/bin/python; else PY=python3; fi; exec "$PY" scripts/run_phase3_model_server_readiness_audit.py --json-indent 0'
"""


def render_timer(*, on_calendar: str = DEFAULT_ON_CALENDAR) -> str:
    return f"""[Unit]
Description=Run BB Phase 3 model-server readiness report

[Timer]
OnCalendar={on_calendar}
Persistent=true
RandomizedDelaySec=300
Unit={SERVICE_NAME}

[Install]
WantedBy=timers.target
"""


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-app-dir", default=REMOTE_APP_DIR)
    parser.add_argument("--owner", default=REMOTE_OWNER)
    parser.add_argument("--on-calendar", default=DEFAULT_ON_CALENDAR)
    parser.add_argument("--run-now", action="store_true", help="Start the oneshot once after install.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _upload_text(ssh, remote_path: str, content: str) -> None:
    sftp = ssh.open_sftp()
    try:
        with sftp.file(remote_path, "w") as remote:
            remote.write(content)
        sftp.chmod(remote_path, 0o644)
    finally:
        sftp.close()


def install_timer(
    *,
    remote_app_dir: str = REMOTE_APP_DIR,
    owner: str = REMOTE_OWNER,
    on_calendar: str = DEFAULT_ON_CALENDAR,
    run_now: bool = False,
    dry_run: bool = False,
) -> None:
    service = render_service(remote_app_dir=remote_app_dir, owner=owner)
    timer = render_timer(on_calendar=on_calendar)
    safe_print({"service": SERVICE_NAME, "timer": TIMER_NAME, "on_calendar": on_calendar})
    if dry_run:
        safe_print(service)
        safe_print(timer)
        return

    user, group = _owner_parts(owner)
    report_dir = posixpath.join(remote_app_dir, REPORT_DIR_REL)
    ssh = connect_remote_ssh(ROOT, timeout=20)
    try:
        run_remote_text(
            ssh,
            " && ".join(
                [
                    f"install -d -o {_remote_quote(user)} -g {_remote_quote(group)} -m 0755 {_remote_quote(report_dir)}",
                    f"chown -R {_remote_quote(owner)} {_remote_quote(report_dir)}",
                ]
            ),
            timeout=60,
            check=True,
        )
        staged_service = f"/tmp/{SERVICE_NAME}"
        staged_timer = f"/tmp/{TIMER_NAME}"
        _upload_text(ssh, staged_service, service)
        _upload_text(ssh, staged_timer, timer)
        commands = [
            f"install -m 0644 {_remote_quote(staged_service)} /etc/systemd/system/{SERVICE_NAME}",
            f"install -m 0644 {_remote_quote(staged_timer)} /etc/systemd/system/{TIMER_NAME}",
            "systemctl daemon-reload",
            f"systemctl enable --now {TIMER_NAME}",
            f"systemctl is-enabled {TIMER_NAME}",
            f"systemctl is-active {TIMER_NAME}",
        ]
        if run_now:
            commands.append(f"systemctl start {SERVICE_NAME} || true")
            commands.append(f"systemctl status {SERVICE_NAME} --no-pager -l || true")
        safe_print(run_remote_text(ssh, " && ".join(commands), timeout=240, check=True))
    finally:
        ssh.close()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    install_timer(
        remote_app_dir=args.remote_app_dir,
        owner=args.owner,
        on_calendar=args.on_calendar,
        run_now=bool(args.run_now),
        dry_run=bool(args.dry_run),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
