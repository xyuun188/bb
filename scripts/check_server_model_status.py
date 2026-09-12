"""Read-only status check for the single Qwen3.8-27B model host.

Retired 14B entries below are reported as deprecated service names and must
remain inactive; they are never started or used as routing fallbacks.
The optional legacy audit profile is intentionally removed from the CLI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.model_server_bridge import load_model_server_info_from_platform  # noqa: E402
from core.model_topology import DEFAULT_MODEL_TOPOLOGY_PROFILE  # noqa: E402
from core.phase3_model_contract import (  # noqa: E402
    PHASE3_APPROVED_RUNTIME_MODEL_PATHS,
    PHASE3_MODEL_SERVER_SERVICES,
)
from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402

VLLM_SERVICES = PHASE3_MODEL_SERVER_SERVICES
PHASE3_QUANT_API_PORT = 8101
TARGET_CANDIDATE_MANIFEST = "/data/BB/manifests/target_model_candidate.json"
MODEL_DIRS = (*PHASE3_APPROVED_RUNTIME_MODEL_PATHS, "/data/BB/models/finquant_lora/current.json")
RETIRED_MODEL_SERVICES = (
    "local-ai-tools.service",
    "qwen3-14b.service",
    "qwen3-14b-trade.service",
    "qwen3-32b-main.service",
    "qwen3-32b-review.service",
    "deepseek-r1-14b-risk.service",
    "deepseek-14b-main.service",
    "deepseek-32b-main.service",
)


def _target_status_command() -> str:
    service, _model, port = PHASE3_MODEL_SERVER_SERVICES[0]
    model_path = PHASE3_APPROVED_RUNTIME_MODEL_PATHS[0]
    return "\n".join(
        [
            "echo '--- target_single_model profile ---'",
            f"printf '{service} '; systemctl is-active {service} || true",
            f"test -f '{TARGET_CANDIDATE_MANIFEST}' && echo 'candidate manifest: present' || echo 'candidate manifest: missing'",
            f"test -d '{model_path}' && du -sh '{model_path}' || true",
            f"candidate_model=$(python3 -c \"import json; print(json.load(open('{TARGET_CANDIDATE_MANIFEST}'))['model_id'])\" 2>/dev/null || true)",
            "echo \"candidate model: ${candidate_model:-unavailable}\"",
            f"curl -fsS --max-time 8 http://127.0.0.1:{port}/v1/models || true",
            f"echo '--- phase3 quant API port {PHASE3_QUANT_API_PORT} ---'",
            f"curl -fsS --max-time 8 http://127.0.0.1:{PHASE3_QUANT_API_PORT}/health || true",
            "echo '--- retired 14B services must be inactive or missing ---'",
            *[f"printf '{name} '; systemctl is-active {name} || true" for name in RETIRED_MODEL_SERVICES],
            "echo '--- disk/gpu ---'",
            "df -h /data || true",
            "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("target_single_model",), default=DEFAULT_MODEL_TOPOLOGY_PROFILE)
    parser.parse_args(argv)
    info = load_model_server_info_from_platform(ROOT)
    ssh = connect_remote_ssh(ROOT, timeout=15, info=info)
    try:
        command = _target_status_command()
        safe_print(run_remote_text(ssh, command, timeout=180, check=False))
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
