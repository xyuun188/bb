#!/usr/bin/env python3
"""Install the independent Local AI Tools shadow-training systemd timer."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

SERVICE_NAME = "bb-local-ai-tools-auto-train.service"
TIMER_NAME = "bb-local-ai-tools-auto-train.timer"


def _unit_text(app_root: Path) -> tuple[str, str]:
    python = app_root / ".venv" / "bin" / "python"
    service = f"""[Unit]
Description=BB Local AI Tools shadow-training check
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=bb
Group=bb
WorkingDirectory={app_root}
EnvironmentFile=-{app_root / '.env'}
EnvironmentFile=-/etc/bb/bb-runtime.env
Environment=PYTHONUNBUFFERED=1
Environment=LOCAL_AI_TOOLS_TRAINING_MEMORY_LIMIT_BYTES=6442450944
ExecStart={python} {app_root / 'scripts' / 'run_local_ai_tools_auto_train.py'}
Nice=10
IOSchedulingClass=best-effort
MemoryHigh=4G
MemoryMax=6G
CPUQuota=100%
TasksMax=128
OOMPolicy=kill
"""
    timer = f"""[Unit]
Description=Run BB Local AI Tools shadow training periodically

[Timer]
OnBootSec=12min
OnUnitActiveSec=30min
RandomizedDelaySec=120s
Persistent=true
Unit={SERVICE_NAME}

[Install]
WantedBy=timers.target
"""
    return service, timer


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-root", default=os.environ.get("BB_APP_ROOT", "/data/bb/app"))
    parser.add_argument("--user", action="store_true")
    parser.add_argument("--remove", action="store_true")
    args = parser.parse_args()

    app_root = Path(args.app_root).expanduser().resolve()
    runner = app_root / "scripts" / "run_local_ai_tools_auto_train.py"
    if not runner.is_file():
        raise SystemExit(f"application root is invalid: {app_root}")

    if args.user:
        unit_dir = Path.home() / ".config" / "systemd" / "user"
        systemctl = ["systemctl", "--user"]
    else:
        if os.geteuid() != 0:
            raise SystemExit("system install requires root; use --user for a user unit")
        unit_dir = Path("/etc/systemd/system")
        systemctl = ["systemctl"]

    service_path = unit_dir / SERVICE_NAME
    timer_path = unit_dir / TIMER_NAME
    unit_dir.mkdir(parents=True, exist_ok=True)

    if args.remove:
        _run(systemctl + ["disable", "--now", TIMER_NAME])
        service_path.unlink(missing_ok=True)
        timer_path.unlink(missing_ok=True)
        _run(systemctl + ["daemon-reload"])
        return 0

    service, timer = _unit_text(app_root)
    service_path.write_text(service, encoding="utf-8")
    timer_path.write_text(timer, encoding="utf-8")
    _run(systemctl + ["daemon-reload"])
    _run(systemctl + ["enable", "--now", TIMER_NAME])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
