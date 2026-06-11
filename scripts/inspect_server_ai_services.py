"""Inspect remote AI service scripts and Python environments."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402


def main() -> None:
    ssh = connect_remote_ssh(ROOT, timeout=15)
    try:
        cmd = "\n".join(
            [
                "echo '--- python envs ---'",
                "find /home /data /opt \\( -path '*/envs/trade_vllm/bin/python' -o -path '*/envs/trade_ml/bin/python' \\) 2>/dev/null || true",
                "echo '--- current qwen3 main service ---'",
                "systemctl cat qwen3-32b-main.service --no-pager || true",
                "echo '--- current local tools service ---'",
                "systemctl cat local-ai-tools.service --no-pager || true",
                "echo '--- deprecated service leftovers ---'",
                "systemctl cat qwen3-14b.service --no-pager || true",
                "systemctl cat qwen3-32b-review.service --no-pager || true",
                "systemctl cat deepseek-14b-main.service --no-pager || true",
                "systemctl cat deepseek-32b-main.service --no-pager || true",
                "echo '--- scripts ---'",
                "ls -lah /data/trade_ai/scripts || true",
                "echo '--- qwen3 main start script ---'",
                "sed -n '1,240p' /data/trade_ai/scripts/start_qwen3_32b_main.sh 2>/dev/null || true",
                "echo '--- local tools api header ---'",
                "sed -n '1,80p' /data/trade_ai/tools/local_ai_tools_api.py 2>/dev/null || true",
            ]
        )
        safe_print(run_remote_text(ssh, cmd, timeout=120, check=False))
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
