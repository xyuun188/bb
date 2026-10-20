"""Run a bounded training preflight through the normal lease and quality gates."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402

REMOTE_CODE = """
import contextlib
import io
import json
import signal
import sys
from datetime import UTC, datetime
from scripts import train_ml_signal_model as trainer
from services.strategy_replay_governance import write_replay_governance
from config.settings import settings
from core.safe_output import safe_error_text

def terminate(signum, frame):
    raise TimeoutError("bounded training preflight was terminated")

signal.signal(signal.SIGTERM, terminate)

original = trainer.run_training
captured = {}

async def capture(**kwargs):
    result = await original(**kwargs)
    captured.update(result)
    return result

trainer.run_training = capture
sys.argv = ["scripts/train_ml_signal_model.py"]
output = io.StringIO()
failure = None
try:
    with contextlib.redirect_stdout(output):
        exit_code = trainer.main()
except Exception as exc:
    exit_code = 2
    failure = safe_error_text(exc, limit=1000)
metadata = captured.get("metadata") or {}
report = {
    "status": "failed" if failure else "complete" if exit_code == 0 else "skipped",
    "exit_code": exit_code,
    "generated_at": datetime.now(UTC).isoformat(),
    "diagnostic_only": True,
    "artifact_persisted": False,
    "promotion_authorized": False,
    "submits_orders": False,
    "frame_sample_count": captured.get("frame_sample_count"),
    "loaded_row_count": captured.get("loaded_row_count"),
    "authoritative_trade_sample_count": captured.get("authoritative_trade_sample_count"),
    "metadata": {
        key: metadata.get(key) for key in (
            "trained_at", "shadow_selection_policy", "training_data_sha256",
            "source_code_sha256", "training_window_composition",
            "decision_group_partition", "walk_forward_report", "metrics",
            "holdout_horizon_diagnostics", "counterfactual_cost_holdout",
        )
    },
}
if failure:
    report["error"] = failure
if exit_code != 0:
    report["result"] = output.getvalue()[-3000:]
write_replay_governance(
    report, path=settings.data_dir / "strategy_replay/training_preflight.json"
)
compact = {**report, "metadata": {
    key: value for key, value in report["metadata"].items()
    if key not in {"walk_forward_report", "holdout_horizon_diagnostics"}
}}
print(json.dumps(compact, allow_nan=False, default=str))
raise SystemExit(exit_code)
"""


def main() -> int:
    command = [
        "systemd-run",
        "--quiet",
        "--wait",
        "--pipe",
        "--collect",
        "--property=WorkingDirectory=/data/bb/app",
        "--property=User=bb",
        "--property=EnvironmentFile=-/data/bb/app/.env",
        "--property=EnvironmentFile=/etc/bb/bb-runtime.env",
        "--property=CPUQuota=200%",
        "--property=MemoryMax=2G",
        "--property=RuntimeMaxSec=3600",
        "--property=TimeoutStopSec=30",
        "--property=KillMode=control-group",
        "--property=Nice=10",
        "--setenv=OMP_NUM_THREADS=1",
        "--setenv=OPENBLAS_NUM_THREADS=1",
        "--setenv=MKL_NUM_THREADS=1",
        "--setenv=DATABASE_POOL_SIZE=2",
        "--setenv=DATABASE_MAX_OVERFLOW=0",
        "/data/bb/app/.venv/bin/python",
        "-c",
        REMOTE_CODE,
    ]
    ssh = connect_remote_ssh(ROOT, timeout=25)
    try:
        output = run_remote_text(
            ssh, shlex.join(command), timeout=3660, max_output_chars=18000
        )
    finally:
        ssh.close()
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
