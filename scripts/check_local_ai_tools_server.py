"""Print remote local-AI-tools service diagnostics."""

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
                "systemctl status local-ai-tools.service --no-pager -l || true",
                "echo '--- err log ---'",
                "tail -n 160 /data/trade_ai/logs/local_ai_tools_api.err.log 2>/dev/null || true",
                "echo '--- out log ---'",
                "tail -n 80 /data/trade_ai/logs/local_ai_tools_api.log 2>/dev/null || true",
                "echo '--- deps ---'",
                "/home/linux/anaconda3/envs/trade_ml/bin/python -c 'import fastapi,uvicorn,httpx,numpy; print(\"deps-ok\")' || true",
                "echo '--- local health ---'",
                "curl -sS http://127.0.0.1:8001/health || true",
            ]
        )
        safe_print(run_remote_text(ssh, cmd, timeout=120, check=False))
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
