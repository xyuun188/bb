"""Inspect remote AI service scripts and Python environments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.model_server_bridge import load_model_server_info_from_platform  # noqa: E402
from core.model_topology import DEFAULT_MODEL_TOPOLOGY_PROFILE  # noqa: E402
from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402

RETIRED_MODEL_SERVICES = (
    "local-ai-tools.service",
    "qwen3-14b-trade.service",
    "deepseek-r1-14b-risk.service",
    "qwen3-14b.service",
    "qwen3-32b-main.service",
    "qwen3-32b-review.service",
    "deepseek-14b-main.service",
    "deepseek-32b-main.service",
)

TARGET_SERVICE = "bb-phase3-llm-target.service"
TARGET_SCRIPT_PATH = "/data/BB/scripts/start_target_single_model.sh"
TARGET_CANDIDATE_MANIFEST = "/data/BB/manifests/target_model_candidate.json"


def _target_command() -> str:
    return "\n".join(
        [
            "echo '--- target_single_model service ---'",
            f"systemctl cat {TARGET_SERVICE} --no-pager || true",
            "echo '--- retired services ---'",
            *[f"echo '### {service}'; systemctl cat {service} --no-pager || true" for service in RETIRED_MODEL_SERVICES],
            "echo '--- phase3 scripts/manifests ---'",
            "ls -lah /data/BB/scripts /data/BB/manifests 2>/dev/null || true",
            "echo '--- phase3 quant API health ---'",
            "curl -fsS --max-time 8 http://127.0.0.1:8101/health || true",
            "echo '--- target start script ---'",
            f"sed -n '1,220p' {TARGET_SCRIPT_PATH} 2>/dev/null || true",
            "echo '--- target candidate manifest ---'",
            f"sed -n '1,260p' {TARGET_CANDIDATE_MANIFEST} 2>/dev/null || true",
            "echo '--- target service manifest ---'",
            "sed -n '1,260p' /data/BB/manifests/phase3_model_service_manifest.json 2>/dev/null || true",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("target_single_model",), default=DEFAULT_MODEL_TOPOLOGY_PROFILE)
    parser.parse_args(argv)
    info = load_model_server_info_from_platform(ROOT)
    ssh = connect_remote_ssh(ROOT, timeout=15, info=info)
    try:
        cmd = _target_command()
        safe_print(run_remote_text(ssh, cmd, timeout=120, check=False))
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
