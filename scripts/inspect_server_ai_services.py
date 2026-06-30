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
    "bb-phase3-llm-decision.service",
    "bb-phase3-llm-risk-review.service",
    "bb-phase3-llm-expert.service",
)
APPROVED_SCRIPT_PATHS = (
    "/data/BB/scripts/start_bb-phase3-llm-decision.sh",
    "/data/BB/scripts/start_bb-phase3-llm-risk-review.sh",
    "/data/BB/scripts/start_bb-phase3-llm-expert.sh",
)
DEPRECATED_SERVICES = (
    "local-ai-tools.service",
    "qwen3-14b-trade.service",
    "deepseek-r1-14b-risk.service",
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
                "find /data/BB /home /opt \\( -path '*/envs/phase3-quant/bin/python' -o -path '*/envs/trade_vllm/bin/python' -o -path '*/envs/trade_ml/bin/python' \\) 2>/dev/null || true",
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
                "echo '--- phase3 scripts/manifests ---'",
                "ls -lah /data/BB/scripts /data/BB/manifests 2>/dev/null || true",
                "echo '--- phase3 quant API health ---'",
                "curl -fsS --max-time 8 http://127.0.0.1:8101/health || true",
                "echo '--- approved start scripts ---'",
                *[
                    f"echo '### {path}'; sed -n '1,220p' {path} 2>/dev/null || true"
                    for path in APPROVED_SCRIPT_PATHS
                ],
                "echo '--- service manifest ---'",
                "sed -n '1,220p' /data/BB/manifests/phase3_model_service_manifest.json 2>/dev/null || true",
            ]
        )
        safe_print(run_remote_text(ssh, cmd, timeout=120, check=False))
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
