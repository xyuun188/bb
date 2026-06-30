"""Check remote model service status for the Phase 3 quant deployment."""

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
    ("bb-phase3-llm-decision.service", "qwen3-32b-trade", 8000),
    ("bb-phase3-llm-risk-review.service", "deepseek-r1-14b-risk", 8002),
    ("bb-phase3-llm-expert.service", "BB-FinQuant-Expert-14B", 8003),
)

PHASE3_QUANT_API_PORT = 8101
MODEL_DIRS = (
    "/data/BB/models/llm_decision_maker/Qwen--Qwen3-32B-AWQ",
    "/data/BB/models/llm_high_risk_review/casperhansen--deepseek-r1-distill-qwen-14b-awq",
    "/data/BB/models/llm_expert_pool/Qwen--Qwen3-14B-AWQ",
)
DEPRECATED_SERVICES = (
    "local-ai-tools.service",
    "qwen3-14b.service",
    "qwen3-14b-trade.service",
    "qwen3-32b-main.service",
    "qwen3-32b-review.service",
    "deepseek-r1-14b-risk.service",
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
                f"echo '--- phase3 quant API port {PHASE3_QUANT_API_PORT} ---'",
                f"curl -fsS --max-time 8 http://127.0.0.1:{PHASE3_QUANT_API_PORT}/health || true",
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
                "tail -n 80 /data/BB/logs/services/bb-phase3-llm-decision.log 2>/dev/null || true",
                "echo '--- deepseek r1 risk log ---'",
                "tail -n 80 /data/BB/logs/services/bb-phase3-llm-risk-review.log 2>/dev/null || true",
                "echo '--- qwen3 expert pool log ---'",
                "tail -n 80 /data/BB/logs/services/bb-phase3-llm-expert.log 2>/dev/null || true",
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
