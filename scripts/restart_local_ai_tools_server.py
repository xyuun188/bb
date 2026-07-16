"""Restart the Phase 3 quant API and print its bounded health summary."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.model_server_bridge import load_model_server_info_from_platform  # noqa: E402
from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402


def main() -> None:
    info = load_model_server_info_from_platform(ROOT)
    ssh = connect_remote_ssh(ROOT, timeout=15, info=info)
    try:
        cmd = "\n".join(
            [
                "sudo systemctl restart bb-phase3-quant-api.service",
                "sleep 3",
                "systemctl is-active bb-phase3-quant-api.service",
                "set -a; . /data/BB/env/phase3.env; set +a; "
                'curl -sS -H "Authorization: Bearer ${LOCAL_AI_TOOLS_API_KEY}" '
                "http://127.0.0.1:8101/health",
            ]
        )
        safe_print(run_remote_text(ssh, cmd, timeout=90, check=False))
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
