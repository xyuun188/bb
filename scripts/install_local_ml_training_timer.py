#!/usr/bin/env python3
"""Install the persistent local-ML auto-training systemd timer.

The timer is intentionally separate from the trading process.  A failed
training run is observable in journald and never takes the execution loop down.
Run as root for a system unit, or pass ``--user`` for a user unit.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def _unit_text(app_root: Path) -> tuple[str, str]:
    python = app_root / ".venv" / "bin" / "python"
    service = f"""[Unit]
Description=BB local ML auto-training check
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
Environment=LOCAL_ML_TRAINING_MEMORY_LIMIT_BYTES=8589934592
ExecStart={python} {app_root / 'scripts' / 'run_local_ml_auto_train.py'}
Nice=10
IOSchedulingClass=best-effort
MemoryHigh=6G
MemoryMax=8G
CPUQuota=150%
TasksMax=128
OOMPolicy=kill
"""
    timer = """[Unit]
Description=Run BB local ML auto-training periodically

[Timer]
OnBootSec=5min
OnUnitActiveSec=15min
RandomizedDelaySec=90s
Persistent=true
Unit=bb-local-ml-auto-train.service

[Install]
WantedBy=timers.target
"""
    return service, timer


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-root", default=os.environ.get("BB_APP_ROOT", "/data/bb/app"))
    parser.add_argument("--user", action="store_true", help="install under the current user's systemd instance")
    parser.add_argument("--remove", action="store_true")
    args = parser.parse_args()

    app_root = Path(args.app_root).expanduser().resolve()
    if not (app_root / "scripts" / "run_local_ml_auto_train.py").is_file():
        raise SystemExit(f"application root is invalid: {app_root}")

    if args.user:
        unit_dir = Path.home() / ".config" / "systemd" / "user"
        systemctl = ["systemctl", "--user"]
    else:
        if os.geteuid() != 0:
            raise SystemExit("system install requires root; use --user for a user unit")
        unit_dir = Path("/etc/systemd/system")
        systemctl = ["systemctl"]

    service_path = unit_dir / "bb-local-ml-auto-train.service"
    timer_path = unit_dir / "bb-local-ml-auto-train.timer"
    unit_dir.mkdir(parents=True, exist_ok=True)

    if args.remove:
        _run(systemctl + ["disable", "--now", "bb-local-ml-auto-train.timer"])
        service_path.unlink(missing_ok=True)
        timer_path.unlink(missing_ok=True)
        _run(systemctl + ["daemon-reload"])
        return 0

    service, timer = _unit_text(app_root)
    service_path.write_text(service, encoding="utf-8")
    timer_path.write_text(timer, encoding="utf-8")
    _run(systemctl + ["daemon-reload"])
    _run(systemctl + ["enable", "--now", "bb-local-ml-auto-train.timer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
