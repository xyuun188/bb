"""Install the resource-bounded, paper-only strategy governance worker."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402

SERVICE_NAME = "bb-strategy-replay.service"
TIMER_NAME = "bb-strategy-replay.timer"


def render_service(app_dir: str = "/data/bb/app", owner: str = "bb") -> str:
    return textwrap.dedent(f"""\
        [Unit]
        Description=BB Paper Strategy Exact Replay Governance
        After=network-online.target postgresql.service
        Wants=network-online.target

        [Service]
        Type=oneshot
        User={owner.split(':', 1)[0]}
        WorkingDirectory={app_dir}
        EnvironmentFile=-{app_dir}/.env
        EnvironmentFile=-/etc/bb/bb-runtime.env
        Environment=OMP_NUM_THREADS=1
        Environment=MKL_NUM_THREADS=1
        Environment=OPENBLAS_NUM_THREADS=1
        Environment=NUMEXPR_NUM_THREADS=1
        Environment=MALLOC_ARENA_MAX=2
        Environment=DATABASE_POOL_SIZE=2
        Environment=DATABASE_MAX_OVERFLOW=0
        Environment=DATABASE_SESSION_CONCURRENCY=2
        Environment=DATABASE_POOL_TIMEOUT_SECONDS=3
        Environment=BB_TRAINING_READ_STATEMENT_TIMEOUT_MS=30000
        Environment=BB_TRAINING_IDLE_TRANSACTION_TIMEOUT_MS=15000
        ExecStart={app_dir}/.venv/bin/python {app_dir}/scripts/run_strategy_replay_governance.py
        MemoryAccounting=true
        MemoryHigh=1G
        MemoryMax=2G
        CPUQuota=50%
        Nice=10
        IOSchedulingClass=idle
        TasksMax=64
        TimeoutStartSec=11min
        TimeoutStopSec=15s
        KillMode=control-group
        """)


def render_timer() -> str:
    return textwrap.dedent(f"""\
        [Unit]
        Description=Evaluate BB paper strategies every 30 minutes

        [Timer]
        OnActiveSec=3min
        OnUnitInactiveSec=30min
        AccuracySec=30s
        Unit={SERVICE_NAME}

        [Install]
        WantedBy=timers.target
        """)


def install_command(app_dir: str = "/data/bb/app", owner: str = "bb") -> str:
    return "\n".join((
        "set -e",
        f"cat > /etc/systemd/system/{SERVICE_NAME} <<'REPLAY_UNIT'",
        render_service(app_dir, owner).rstrip(),
        "REPLAY_UNIT",
        f"cat > /etc/systemd/system/{TIMER_NAME} <<'REPLAY_UNIT'",
        render_timer().rstrip(),
        "REPLAY_UNIT",
        "systemctl daemon-reload",
        f"systemctl enable --now {TIMER_NAME}",
        f"systemctl is-active {TIMER_NAME}",
    ))


if __name__ == "__main__":
    ssh = connect_remote_ssh(ROOT, timeout=20)
    try:
        safe_print(run_remote_text(ssh, install_command(), timeout=60))
    finally:
        ssh.close()
