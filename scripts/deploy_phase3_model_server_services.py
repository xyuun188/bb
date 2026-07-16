#!/usr/bin/env python3
"""Install or start Phase 3 shadow-only model services on the quant model server."""

from __future__ import annotations

import argparse
import json
import posixpath
import sys
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402

PHASE3_ROOT = "/data/BB"
PYTHON_BIN = f"{PHASE3_ROOT}/envs/phase3-quant/bin/python"
ENV_FILE = f"{PHASE3_ROOT}/env/phase3.env"
SCRIPT_DIR = f"{PHASE3_ROOT}/scripts"
SERVICE_DIR = f"{PHASE3_ROOT}/services/systemd"
LOG_DIR = f"{PHASE3_ROOT}/logs/services"
MANIFEST_PATH = f"{PHASE3_ROOT}/manifests/phase3_model_service_manifest.json"
POLICY_ID = "phase3_quant_model_services_shadow_only_2026_06_27"


@dataclass(frozen=True)
class Phase3ServiceSpec:
    slot: str
    role: str
    service_name: str
    served_model_name: str
    model_dir: str
    port: int
    cuda_visible_devices: str
    max_model_len: int
    gpu_memory_utilization: float
    max_num_seqs: int
    tensor_parallel_size: int = 1
    quantization: str = "awq_marlin"

    @property
    def start_script_path(self) -> str:
        return f"{SCRIPT_DIR}/start_{self.service_name.removesuffix('.service')}.sh"

    @property
    def staged_service_path(self) -> str:
        return f"{SERVICE_DIR}/{self.service_name}"

    @property
    def log_path(self) -> str:
        return f"{LOG_DIR}/{self.service_name.removesuffix('.service')}.log"

    def render_start_script(self) -> str:
        return textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            source {sh(ENV_FILE)}
            export CUDA_VISIBLE_DEVICES={sh(self.cuda_visible_devices)}
            export VLLM_WORKER_MULTIPROC_METHOD=spawn
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
            export VLLM_CONFIG_ROOT={sh(f"{PHASE3_ROOT}/runtime/vllm")}
            export VLLM_USAGE_STATS_SERVER=
            exec {sh(PYTHON_BIN)} -m vllm.entrypoints.openai.api_server \\
              --host 127.0.0.1 \\
              --port {self.port} \\
              --model {sh(self.model_dir)} \\
              --served-model-name {sh(self.served_model_name)} \\
              --trust-remote-code \\
              --max-model-len {self.max_model_len} \\
              --gpu-memory-utilization {self.gpu_memory_utilization:.2f} \\
              --quantization {sh(self.quantization)} \\
              --max-num-seqs {self.max_num_seqs} \\
              --tensor-parallel-size {self.tensor_parallel_size} >> {sh(self.log_path)} 2>&1
            """
        )

    def render_systemd_service(self) -> str:
        return textwrap.dedent(
            f"""\
            [Unit]
            Description=BB Phase 3 shadow model service - {self.role}
            After=network-online.target
            Wants=network-online.target

            [Service]
            Type=simple
            User=root
            WorkingDirectory={PHASE3_ROOT}
            ExecStart={self.start_script_path}
            Restart=on-failure
            RestartSec=10
            Environment=CUDA_VISIBLE_DEVICES={self.cuda_visible_devices}
            Environment=BB_PHASE3_ROOT={PHASE3_ROOT}
            LimitNOFILE=65535

            [Install]
            WantedBy=multi-user.target
            """
        )

    def manifest_item(self) -> dict[str, Any]:
        item = asdict(self)
        item.update(
            {
                "start_script_path": self.start_script_path,
                "staged_service_path": self.staged_service_path,
                "log_path": self.log_path,
                "api_base": f"http://127.0.0.1:{self.port}/v1",
                "models_endpoint": f"http://127.0.0.1:{self.port}/v1/models",
                "shadow_only": True,
                "live_routing_enabled": False,
            }
        )
        return item


PHASE3_SERVICE_SPECS = (
    Phase3ServiceSpec(
        slot="llm_decision_maker",
        role="decision_fallback_and_finquant_carrier",
        service_name="bb-phase3-llm-decision.service",
        served_model_name="qwen3-14b-trade",
        model_dir="/data/trade_models/Qwen/Qwen3-14B-AWQ",
        port=8000,
        cuda_visible_devices="0",
        max_model_len=4096,
        gpu_memory_utilization=0.72,
        max_num_seqs=2,
        tensor_parallel_size=1,
    ),
    Phase3ServiceSpec(
        slot="llm_expert_pool",
        role="expert_pool_shadow",
        service_name="bb-phase3-llm-expert.service",
        served_model_name="BB-FinQuant-Expert-14B",
        model_dir="/data/BB/models/finquant_lora/current.json",
        port=8003,
        cuda_visible_devices="0",
        max_model_len=4096,
        gpu_memory_utilization=0.72,
        max_num_seqs=2,
    ),
    Phase3ServiceSpec(
        slot="llm_high_risk_review",
        role="high_risk_review_shadow",
        service_name="bb-phase3-llm-risk-review.service",
        served_model_name="deepseek-r1-14b-risk",
        model_dir="/data/trade_models/DeepSeek/deepseek-r1-distill-qwen-14b-awq",
        port=8002,
        cuda_visible_devices="0",
        max_model_len=4096,
        gpu_memory_utilization=0.72,
        max_num_seqs=2,
    ),
)


def sh(value: str | int | float) -> str:
    text = str(value)
    return "'" + text.replace("'", "'\"'\"'") + "'"


def render_manifest() -> str:
    payload = {
        "schema_version": 1,
        "policy_id": POLICY_ID,
        "phase3_root": PHASE3_ROOT,
        "shadow_only": True,
        "live_routing_enabled": False,
        "can_start_trading": False,
        "services": [spec.manifest_item() for spec in PHASE3_SERVICE_SPECS],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _upload_text(ssh, remote_path: str, content: str, *, mode: int = 0o644) -> None:
    sftp = ssh.open_sftp()
    try:
        directory = posixpath.dirname(remote_path)
        run_remote_text(ssh, f"mkdir -p {sh(directory)}", timeout=30, check=True)
        with sftp.file(remote_path, "w") as remote:
            remote.write(content)
        sftp.chmod(remote_path, mode)
    finally:
        sftp.close()


def _render_plan() -> dict[str, Any]:
    return {
        "policy_id": POLICY_ID,
        "phase3_root": PHASE3_ROOT,
        "shadow_only": True,
        "live_routing_enabled": False,
        "services": [spec.manifest_item() for spec in PHASE3_SERVICE_SPECS],
    }


def _remote_preflight_command() -> str:
    model_checks = " && ".join(
        f"test -f {sh(spec.model_dir + '/config.json')}" for spec in PHASE3_SERVICE_SPECS
    )
    port_checks = " && ".join(
        f"! ss -ltn 2>/dev/null | grep -q ':{spec.port} '" for spec in PHASE3_SERVICE_SPECS
    )
    return (
        f"test -x {sh(PYTHON_BIN)} && "
        f"test -f {sh(ENV_FILE)} && "
        f"{model_checks} && "
        f"{PYTHON_BIN} - <<'PY'\n"
        "import torch\n"
        "import vllm\n"
        "print('torch', torch.__version__, torch.version.cuda, torch.cuda.device_count())\n"
        "x=torch.ones((2,2), device='cuda')\n"
        "print('cuda_tensor', (x@x).cpu().tolist())\n"
        "print('vllm', getattr(vllm, '__version__', 'unknown'))\n"
        "PY\n"
        f"{port_checks}"
    )


def _stop_phase3_model_services_command(specs: tuple[Phase3ServiceSpec, ...]) -> str:
    services = " ".join(sh(spec.service_name) for spec in specs)
    return f"systemctl stop {services} 2>/dev/null || true"


def _readiness_command(specs: tuple[Phase3ServiceSpec, ...], *, attempts: int = 80) -> str:
    checks: list[str] = []
    for spec in specs:
        url = f"http://127.0.0.1:{spec.port}/v1/models"
        checks.append(
            "ready=0; "
            f"for i in $(seq 1 {attempts}); do "
            f"curl -fsS --max-time 5 {sh(url)} | grep -F {sh(spec.served_model_name)} >/dev/null "
            "&& ready=1 && break; "
            "sleep 5; "
            "done; "
            f"[ \"$ready\" = 1 ] || {{ echo {sh('readiness failed: ' + spec.service_name)}; "
            f"systemctl status {sh(spec.service_name)} --no-pager -l || true; "
            f"tail -n 120 {sh(spec.log_path)} 2>/dev/null || true; exit 3; }}"
        )
    return " && ".join(checks)


def install_services(*, start: bool = False, plan_only: bool = False) -> None:
    safe_print(_render_plan())
    if plan_only:
        return
    raise RuntimeError(
        "Direct Phase 3 vLLM installation is retired; use "
        "migrate_phase3_model_service_identity.py for the verified shared-carrier runtime."
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--start", action="store_true", help="Start/restart services after install.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    install_services(start=bool(args.start), plan_only=bool(args.plan_only))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
