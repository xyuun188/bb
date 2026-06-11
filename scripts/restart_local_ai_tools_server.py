"""Restart remote local-AI-tools service and print health."""

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
                "sudo systemctl restart local-ai-tools.service",
                "sleep 3",
                "systemctl is-active local-ai-tools.service",
                "curl -s http://127.0.0.1:8001/health",
                "echo",
                "grep -n 'review_backend\\|RISK_REVIEW_BASE\\|local_review_backend' /data/trade_ai/tools/local_ai_tools_api.py | head -20",
            ]
        )
        safe_print(run_remote_text(ssh, cmd, timeout=90, check=False))
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
