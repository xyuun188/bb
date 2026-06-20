"""Inspect remote AI service scripts and Python environments."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.model_server_bridge import load_model_server_info_from_platform  # noqa: E402
from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402

APPROVED_SERVICES = (
    "qwen3-14b-trade.service",
    "deepseek-r1-14b-risk.service",
    "local-ai-tools.service",
)
APPROVED_SCRIPT_PATHS = (
    "/data/trade_ai/scripts/start_qwen3_14b_trade.sh",
    "/data/trade_ai/scripts/start_deepseek_r1_14b_risk.sh",
)
DEPRECATED_SERVICES = (
    "qwen3-14b.service",
    "qwen3-32b-main.service",
    "qwen3-32b-review.service",
    "deepseek-14b-main.service",
    "deepseek-32b-main.service",
)


def main() -> None:
    info = load_model_server_info_from_platform(ROOT)
    ssh = connect_remote_ssh(ROOT, timeout=15, info=info)
    try:
        cmd = "\n".join(
            [
                "echo '--- python envs ---'",
                "find /home /data /opt \\( -path '*/envs/trade_vllm/bin/python' -o -path '*/envs/trade_ml/bin/python' \\) 2>/dev/null || true",
                "echo '--- approved services ---'",
                *[
                    f"echo '### {service}'; systemctl cat {service} --no-pager || true"
                    for service in APPROVED_SERVICES
                ],
                "echo '--- deprecated service leftovers ---'",
                *[
                    f"echo '### {service}'; systemctl cat {service} --no-pager || true"
                    for service in DEPRECATED_SERVICES
                ],
                "echo '--- scripts ---'",
                "ls -lah /data/trade_ai/scripts || true",
                "echo '--- approved start scripts ---'",
                *[
                    f"echo '### {path}'; sed -n '1,220p' {path} 2>/dev/null || true"
                    for path in APPROVED_SCRIPT_PATHS
                ],
                "echo '--- local tools api header ---'",
                "sed -n '1,80p' /data/trade_ai/tools/local_ai_tools_api.py 2>/dev/null || true",
            ]
        )
        safe_print(run_remote_text(ssh, cmd, timeout=120, check=False))
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
