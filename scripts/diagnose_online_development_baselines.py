"""Run fixed development-only baselines with bounded online resources."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402

EARLIEST_EXAM_START = "2026-09-26T22:11:51.501105+00:00"

REMOTE_CODE = """
import asyncio
import json
from datetime import UTC, datetime
from config.settings import settings
from db.session import close_db
from services.ml_development_baselines import build_development_baseline_report
from services.ml_nested_development import build_nested_development_report
from services.ml_development_snapshot import (
    load_development_snapshot, make_development_snapshot, save_development_snapshot,
)
from services.ml_horizon_development import build_horizon_development_report
from services.ml_signal_service import build_training_frame
from services.ml_training_dataset import load_shadow_training_rows
from services.strategy_replay_governance import write_replay_governance

NESTED = __NESTED__
HORIZONS = __HORIZONS__
CUTOFF = __CUTOFF__

async def main():
    try:
        directory = settings.data_dir / "strategy_replay"
        if HORIZONS:
            snapshot_path = directory / "frozen_development_v1.json"
            if snapshot_path.exists():
                development, snapshot = load_development_snapshot(snapshot_path, cutoff=CUTOFF)
            else:
                rows = await load_shadow_training_rows()
                snapshot = make_development_snapshot(build_training_frame(rows), cutoff=CUTOFF)
                development, snapshot = save_development_snapshot(snapshot, path=snapshot_path)
            report = build_horizon_development_report(development, snapshot=snapshot)
        else:
            rows = await load_shadow_training_rows()
            builder = build_nested_development_report if NESTED else build_development_baseline_report
            report = builder(build_training_frame(rows))
        report["generated_at"] = datetime.now(UTC).isoformat()
        name = (
            "horizon_development.json" if HORIZONS else
            "nested_development.json" if NESTED else "development_baselines.json"
        )
        write_replay_governance(
            report, path=directory / name
        )
        if HORIZONS:
            folds = [
                {
                    **fold,
                    "selection": {
                        **fold["selection"],
                        "candidate_count": len(fold["selection"]["candidates"]),
                        "eligible_candidate_count": sum(
                            candidate["inner_selection_eligible"]
                            for candidate in fold["selection"]["candidates"]
                        ),
                        "candidates": [],
                    },
                } for fold in report["folds"]
            ]
        elif NESTED:
            folds = [
                {
                    **fold,
                    "selection": {
                        **fold["selection"],
                        "candidates": [
                            {
                                "alpha": candidate["alpha"],
                                "net_threshold_pct": candidate["net_threshold_pct"],
                                "inner_selection_eligible": candidate["inner_selection_eligible"],
                                "evidence": {
                                    key: candidate["evidence"].get(key) for key in (
                                        "selected_count", "avg_return_pct", "return_lcb_pct",
                                        "profit_factor", "average_return_per_opportunity_pct",
                                    )
                                },
                            } for candidate in fold["selection"]["candidates"]
                        ],
                    },
                } for fold in report["folds"]
            ]
        else:
            folds = [
                {key: value for key, value in fold.items() if key != "models"}
                for fold in report["folds"]
            ]
        print(json.dumps({**report, "folds": folds}, allow_nan=False))
    finally:
        await close_db()

asyncio.run(main())
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--nested", action="store_true")
    mode.add_argument("--horizons", action="store_true")
    args = parser.parse_args()
    command = [
        "systemd-run", "--quiet", "--wait", "--pipe", "--collect",
        "--property=WorkingDirectory=/data/bb/app",
        "--property=User=bb",
        "--property=EnvironmentFile=-/data/bb/app/.env",
        "--property=EnvironmentFile=/etc/bb/bb-runtime.env",
        "--property=CPUQuota=100%",
        "--property=MemoryMax=2G",
        "--property=RuntimeMaxSec=900",
        "--property=KillMode=control-group",
        "--property=Nice=10",
        "--setenv=OMP_NUM_THREADS=1",
        "--setenv=OPENBLAS_NUM_THREADS=1",
        "--setenv=MKL_NUM_THREADS=1",
        "--setenv=DATABASE_POOL_SIZE=2",
        "--setenv=DATABASE_MAX_OVERFLOW=0",
        "/data/bb/app/.venv/bin/python", "-c",
        REMOTE_CODE.replace("__NESTED__", str(args.nested)).replace(
            "__HORIZONS__", str(args.horizons)
        ).replace("__CUTOFF__", repr(EARLIEST_EXAM_START)),
    ]
    ssh = connect_remote_ssh(ROOT, timeout=25)
    try:
        output = run_remote_text(
            ssh, shlex.join(command), timeout=930, max_output_chars=24000
        )
    finally:
        ssh.close()
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
