"""Check remote model service status for the approved dual-14B deployment."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.model_server_bridge import load_model_server_info_from_platform  # noqa: E402
from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402

VLLM_SERVICES = (
    ("qwen3-14b-trade.service", "qwen3-14b-trade", 8000),
    ("deepseek-r1-14b-risk.service", "deepseek-r1-14b-risk", 8002),
)

LOCAL_AI_TOOLS_SERVICE = "local-ai-tools.service"
MODEL_DIRS = (
    "/data/trade_models/Qwen/Qwen3-14B-AWQ",
    "/data/trade_models/DeepSeek/deepseek-r1-distill-qwen-14b-awq",
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
                "echo '--- approved services ---'",
                *[
                    f"printf '{service} '; systemctl is-active {service} || true"
                    for service, _model, _port in VLLM_SERVICES
                ],
                f"printf '{LOCAL_AI_TOOLS_SERVICE} '; "
                f"systemctl is-active {LOCAL_AI_TOOLS_SERVICE} || true",
                "echo '--- deprecated services should be inactive or missing ---'",
                *[
                    f"printf '{service} '; systemctl is-active {service} || true"
                    for service in DEPRECATED_SERVICES
                ],
                "echo '--- vllm models endpoint ---'",
                *[
                    (
                        f"echo '--- {model} port {port} ---'; "
                        f"curl -fsS --max-time 8 http://127.0.0.1:{port}/v1/models || true"
                    )
                    for _service, model, port in VLLM_SERVICES
                ],
                "echo '--- local tools health ---'",
                "curl -fsS --max-time 8 http://127.0.0.1:8001/health || true",
                "echo '--- approved model files ---'",
                *[f"du -sh {path} 2>/dev/null || true" for path in MODEL_DIRS],
                "echo '--- download processes ---'",
                "pgrep -af 'huggingface|modelscope|snapshot_download|Qwen3-14B|deepseek-r1-distill-qwen-14b' || true",
                "echo '--- qwen3 14b log ---'",
                "tail -n 80 /data/trade_ai/logs/qwen3_14b_trade.log 2>/dev/null || true",
                "echo '--- deepseek r1 14b log ---'",
                "tail -n 80 /data/trade_ai/logs/deepseek_r1_14b_risk.log 2>/dev/null || true",
                "echo '--- disk/gpu ---'",
                "df -h /data || true",
                "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true",
            ]
        )
        safe_print(run_remote_text(ssh, cmd, timeout=180, check=False))
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
