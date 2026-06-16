"""Deploy the approved dual-14B vLLM services for trading experts.

This script replaces the old single 32B main service with:
- Qwen3-14B-AWQ on port 8000 for trend/momentum/final decision.
- DeepSeek-R1-Distill-Qwen-14B-AWQ on port 8002 for sentiment/position/risk.

Use --plan-only first to review the exact known service/model paths before any
remote cleanup command is executed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ai_service_spec import (  # noqa: E402
    DEEPSEEK_R1_14B_RISK_SERVICE,
    QWEN3_14B_TRADE_SERVICE,
    QWEN3_MAIN_REMOTE_MODEL_CLEANUP_PATHS,
    qwen3_main_cleanup_command,
)
from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402
from core.model_server_bridge import load_model_server_info_from_platform  # noqa: E402

DUAL_14B_SPECS = (QWEN3_14B_TRADE_SERVICE, DEEPSEEK_R1_14B_RISK_SERVICE)


def _print_plan() -> None:
    safe_print("Dual-14B deployment plan:")
    safe_print("  Cleanup will remove only these known obsolete model paths:")
    for path in QWEN3_MAIN_REMOTE_MODEL_CLEANUP_PATHS:
        safe_print(f"    - {path}")
    safe_print("  New services:")
    for spec in DUAL_14B_SPECS:
        safe_print(
            f"    - {spec.service_name}: {spec.served_model_name} "
            f"port={spec.port} model_dir={spec.model_dir} "
            f"gpu_memory_utilization={spec.gpu_memory_utilization:.2f} "
            f"max_model_len={spec.max_model_len} max_num_seqs={spec.max_num_seqs}"
        )


def _upload_text(ssh, remote_path: str, content: str) -> None:
    sftp = ssh.open_sftp()
    try:
        with sftp.file(remote_path, "w") as remote:
            remote.write(content)
    finally:
        sftp.close()


def _deploy_one(ssh, spec) -> None:
    _upload_text(ssh, spec.download_script_path, spec.render_download_script())
    safe_print(run_remote_text(ssh, spec.download_and_run_command(), timeout=7200))

    _upload_text(ssh, spec.start_script_path, spec.render_start_script())
    _upload_text(ssh, spec.staged_service_path, spec.render_systemd_service())
    safe_print(
        run_remote_text(
            ssh,
            spec.install_and_restart_command(sleep_seconds=8, tail_lines=80),
            timeout=420,
        )
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print service/model paths without connecting to the remote server.",
    )
    args = parser.parse_args(argv)

    _print_plan()
    if args.plan_only:
        return

    info = load_model_server_info_from_platform(ROOT)
    ssh = connect_remote_ssh(ROOT, timeout=20, info=info)
    try:
        safe_print(run_remote_text(ssh, qwen3_main_cleanup_command()))
        for spec in DUAL_14B_SPECS:
            _deploy_one(ssh, spec)
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
