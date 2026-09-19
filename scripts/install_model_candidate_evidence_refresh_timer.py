#!/usr/bin/env python3
"""Install automatic target-model evidence refresh on the model server."""

from __future__ import annotations

import argparse
import posixpath
import secrets
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.model_server_bridge import load_model_server_info_from_platform  # noqa: E402
from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402

SERVICE_NAME = "bb-model-candidate-evidence-refresh.service"
TIMER_NAME = "bb-model-candidate-evidence-refresh.timer"
REMOTE_ROOT = "/data/BB"
REMOTE_OWNER = "linux:linux"
DEFAULT_ON_CALENDAR = "*-*-* 03:35:00"
REMOTE_STAGING_DIR = f"{REMOTE_ROOT}/runtime/systemd-unit-stage"


def _quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def render_service() -> str:
    return f"""[Unit]
Description=BB target-model candidate evidence refresh
After=network-online.target bb-phase3-llm-target.service
Wants=network-online.target

[Service]
Type=oneshot
User=linux
Group=linux
WorkingDirectory={REMOTE_ROOT}
ExecStart={REMOTE_ROOT}/envs/target-inference/bin/python {REMOTE_ROOT}/scripts/refresh_model_candidate_evidence.py
TimeoutStartSec=45min
Nice=10
IOSchedulingClass=best-effort
IOSchedulingPriority=6
"""


def render_timer(*, on_calendar: str = DEFAULT_ON_CALENDAR) -> str:
    return f"""[Unit]
Description=Refresh BB target-model evidence before expiry

[Timer]
OnCalendar={on_calendar}
Persistent=true
RandomizedDelaySec=1200
Unit={SERVICE_NAME}

[Install]
WantedBy=timers.target
"""


def _upload(ssh, source: Path, destination: str, *, mode: int = 0o755) -> None:
    temporary = f"{destination}.bb-upload-{secrets.token_hex(8)}"
    sftp = ssh.open_sftp()
    try:
        sftp.put(str(source), temporary)
        sftp.chmod(temporary, mode)
        try:
            sftp.posix_rename(temporary, destination)
        except (AttributeError, OSError):
            try:
                sftp.remove(destination)
            except OSError:
                pass
            sftp.rename(temporary, destination)
    finally:
        try:
            sftp.remove(temporary)
        except OSError:
            pass
        sftp.close()


def install_timer(*, on_calendar: str = DEFAULT_ON_CALENDAR, run_now: bool = False, dry_run: bool = False) -> None:
    service = render_service()
    timer = render_timer(on_calendar=on_calendar)
    if dry_run:
        safe_print(service)
        safe_print(timer)
        return

    info = load_model_server_info_from_platform(ROOT)
    ssh = connect_remote_ssh(ROOT, timeout=20, info=info)
    nonce = secrets.token_hex(10)
    staged_service = posixpath.join(REMOTE_STAGING_DIR, f"{SERVICE_NAME}.{nonce}")
    staged_timer = posixpath.join(REMOTE_STAGING_DIR, f"{TIMER_NAME}.{nonce}")
    try:
        run_remote_text(
            ssh,
            f"install -d -o linux -g linux -m 0755 {REMOTE_ROOT}/scripts {REMOTE_ROOT}/core {REMOTE_ROOT}/runtime && "
            f"install -d -o linux -g linux -m 0700 {REMOTE_STAGING_DIR}",
            timeout=60,
            check=True,
        )
        artifacts = (
            (ROOT / "scripts/refresh_model_candidate_evidence.py", f"{REMOTE_ROOT}/scripts/refresh_model_candidate_evidence.py"),
            (ROOT / "scripts/probe_model_candidate_runtime.py", f"{REMOTE_ROOT}/scripts/probe_model_candidate_runtime.py"),
            (ROOT / "scripts/validate_model_candidate.py", f"{REMOTE_ROOT}/scripts/validate_model_candidate.py"),
            (ROOT / "core/model_candidate_manifest.py", f"{REMOTE_ROOT}/core/model_candidate_manifest.py"),
            (ROOT / "core/model_topology.py", f"{REMOTE_ROOT}/core/model_topology.py"),
        )
        for source, destination in artifacts:
            _upload(ssh, source, destination)
        for remote_path, content in ((staged_service, service), (staged_timer, timer)):
            sftp = ssh.open_sftp()
            try:
                with sftp.file(remote_path, "w") as handle:
                    handle.write(content)
                sftp.chmod(remote_path, stat.S_IRUSR | stat.S_IWUSR)
            finally:
                sftp.close()
        commands = [
            f"sudo install -m 0644 {_quote(staged_service)} /etc/systemd/system/{SERVICE_NAME}",
            f"sudo install -m 0644 {_quote(staged_timer)} /etc/systemd/system/{TIMER_NAME}",
            "sudo systemctl daemon-reload",
            f"sudo systemctl enable --now {TIMER_NAME}",
            f"sudo systemctl is-active {TIMER_NAME}",
            f"rm -f -- {_quote(staged_service)} {_quote(staged_timer)}",
        ]
        if run_now:
            commands.extend(
                [
                    f"sudo systemctl start {SERVICE_NAME}",
                    f"sudo systemctl status {SERVICE_NAME} --no-pager -l || true",
                ]
            )
        safe_print(run_remote_text(ssh, " && ".join(commands), timeout=3000, check=True))
    finally:
        ssh.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--on-calendar", default=DEFAULT_ON_CALENDAR)
    parser.add_argument("--run-now", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    install_timer(on_calendar=args.on_calendar, run_now=args.run_now, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
