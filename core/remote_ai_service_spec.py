"""Remote vLLM service specifications for server-side model deployment."""

from __future__ import annotations

import json
import re
import shlex
import textwrap
from dataclasses import dataclass
from pathlib import PurePosixPath

REMOTE_AI_ROOT = "/data/trade_ai"
REMOTE_MODEL_ROOT = "/data/trade_models"
REMOTE_RUNTIME_DIRS = (
    f"{REMOTE_AI_ROOT}/scripts",
    f"{REMOTE_AI_ROOT}/systemd",
    f"{REMOTE_AI_ROOT}/logs",
    f"{REMOTE_AI_ROOT}/hf_cache",
)
LEGACY_MAIN_LLM_SERVICES = (
    "deepseek-14b-main.service",
    "deepseek-32b-main.service",
    "qwen3-14b.service",
    "qwen3-14b-main.service",
    "qwen3-32b-main.service",
    "qwen3-32b-review.service",
)
LEGACY_LLM_SCRIPT_PATHS = (
    f"{REMOTE_AI_ROOT}/scripts/start_deepseek_14b_main.sh",
    f"{REMOTE_AI_ROOT}/scripts/start_deepseek_32b_main.sh",
    f"{REMOTE_AI_ROOT}/scripts/start_qwen3_14b.sh",
    f"{REMOTE_AI_ROOT}/scripts/start_qwen3_32b_review.sh",
)
QWEN3_MAIN_REMOTE_MODEL_CLEANUP_PATHS = (
    f"{REMOTE_MODEL_ROOT}/DeepSeek/DeepSeek-R1-Distill-Qwen-32B-AWQ",
)
SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
SAFE_SYSTEMD_SERVICE_RE = re.compile(r"^[A-Za-z0-9_.@-]+\.service$")
SAFE_SINGLE_WORD_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")


def shell_quote(value: str | int | float) -> str:
    return shlex.quote(str(value))


def _reject_control_chars(value: str, *, field_name: str) -> None:
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{field_name} must not contain control characters.")


def _validate_remote_data_path(path: str, *, field_name: str) -> None:
    _reject_control_chars(path, field_name=field_name)
    if "\\" in path:
        raise ValueError(f"{field_name} must not contain backslashes.")
    parsed = PurePosixPath(path)
    if not parsed.is_absolute():
        raise ValueError(f"{field_name} must be an absolute remote path.")
    if parsed.parts[:2] != ("/", "data"):
        raise ValueError(f"{field_name} must stay under /data: {path}")
    if any(part in {"", ".", ".."} for part in parsed.parts[1:]):
        raise ValueError(f"{field_name} must not contain path traversal: {path}")


def _validate_safe_filename(value: str, *, field_name: str) -> None:
    _reject_control_chars(value, field_name=field_name)
    if "/" in value or "\\" in value or value in {".", ".."} or ".." in value:
        raise ValueError(f"{field_name} must be a simple file name.")
    if not SAFE_FILENAME_RE.fullmatch(value):
        raise ValueError(f"{field_name} contains unsupported characters.")


def _validate_single_word(value: str, *, field_name: str) -> None:
    _reject_control_chars(value, field_name=field_name)
    if not SAFE_SINGLE_WORD_RE.fullmatch(value):
        raise ValueError(f"{field_name} contains unsupported characters.")


def data_disk_guard_command() -> str:
    """Return a shell guard that refuses to use /data when it is on the root disk."""
    return (
        "data_src=$(findmnt -no SOURCE /data 2>/dev/null || true); "
        "root_src=$(findmnt -no SOURCE / 2>/dev/null || true); "
        '[ -n "$data_src" ] && [ "$data_src" != "$root_src" ] || '
        "{ echo '/data is not mounted on a separate data disk'; exit 2; }"
    )


@dataclass(frozen=True)
class RemoteVllmServiceSpec:
    """Typed configuration for one remote OpenAI-compatible vLLM service."""

    model_repo: str
    modelscope_model: str
    model_dir: str
    served_model_name: str
    service_name: str
    description: str
    start_script_name: str
    download_script_name: str
    log_name: str
    port: int = 8000
    conda_env: str = "trade_vllm"
    service_user: str = "linux"
    cuda_visible_devices: str = "0"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.88
    dtype: str = "half"
    quantization: str = "awq_marlin"
    enforce_eager: bool = False
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    max_num_seqs: int = 8
    max_num_batched_tokens: int = 8192

    def __post_init__(self) -> None:
        for field_name in ("start_script_name", "download_script_name", "log_name"):
            _validate_safe_filename(str(getattr(self, field_name)), field_name=field_name)

        for field_name in (
            "served_model_name",
            "conda_env",
            "service_user",
            "dtype",
            "quantization",
        ):
            _validate_single_word(str(getattr(self, field_name)), field_name=field_name)

        _reject_control_chars(self.description, field_name="description")
        if not SAFE_SINGLE_WORD_RE.fullmatch(self.cuda_visible_devices):
            raise ValueError("cuda_visible_devices contains unsupported characters.")
        if not SAFE_SYSTEMD_SERVICE_RE.fullmatch(self.service_name):
            raise ValueError("Remote systemd service names must end with .service.")
        if self.max_model_len < 512:
            raise ValueError("max_model_len is too small for trading prompts.")
        if self.max_num_seqs < 1:
            raise ValueError("max_num_seqs must be positive.")
        if self.max_num_batched_tokens < self.max_model_len:
            raise ValueError("max_num_batched_tokens must cover max_model_len.")

        paths = (
            self.model_dir,
            self.start_script_path,
            self.download_script_path,
            self.staged_service_path,
            self.log_path,
        )
        for path in paths:
            _validate_remote_data_path(path, field_name="remote model service path")

    @property
    def start_script_path(self) -> str:
        return f"{REMOTE_AI_ROOT}/scripts/{self.start_script_name}"

    @property
    def download_script_path(self) -> str:
        return f"{REMOTE_AI_ROOT}/scripts/{self.download_script_name}"

    @property
    def staged_service_path(self) -> str:
        return f"{REMOTE_AI_ROOT}/systemd/{self.service_name}"

    @property
    def log_path(self) -> str:
        return f"{REMOTE_AI_ROOT}/logs/{self.log_name}"

    @property
    def hf_cache_dir(self) -> str:
        return f"{REMOTE_AI_ROOT}/hf_cache"

    def runtime_dirs_command(self) -> str:
        return "mkdir -p " + " ".join(shell_quote(path) for path in REMOTE_RUNTIME_DIRS)

    def download_and_run_command(self) -> str:
        return (
            f"chmod +x {shell_quote(self.download_script_path)} && "
            f"{shell_quote(self.download_script_path)}"
        )

    def model_presence_command(self) -> str:
        model_dir = shell_quote(self.model_dir)
        return (
            f"test -f {model_dir}/config.json && "
            f"find {model_dir} -maxdepth 1 -name '*.safetensors' "
            "-type f -size +0c | grep -q ."
        )

    def readiness_command(self, *, attempts: int = 48, sleep_seconds: int = 5) -> str:
        """Return a remote readiness probe for the OpenAI-compatible model endpoint."""
        attempts = max(int(attempts), 1)
        sleep_seconds = max(int(sleep_seconds), 1)
        models_url = shell_quote(f"http://127.0.0.1:{self.port}/v1/models")
        served_model = shell_quote(self.served_model_name)
        return (
            "ready=0; "
            f"for i in $(seq 1 {attempts}); do "
            f"if curl -fsS --max-time 5 {models_url} | grep -F {served_model} >/dev/null; "
            "then ready=1; break; fi; "
            f"sleep {sleep_seconds}; "
            "done; "
            '[ "$ready" = 1 ] || { '
            f"echo 'vLLM readiness failed for {self.served_model_name}'; "
            f"systemctl status {shell_quote(self.service_name)} --no-pager -l || true; "
            f"tail -n 80 {shell_quote(self.log_path)} 2>/dev/null || true; "
            "exit 3; "
            "}; "
            f"echo 'vLLM model ready: {self.served_model_name}'"
        )

    def install_and_restart_command(self, *, sleep_seconds: int = 10, tail_lines: int = 120) -> str:
        return (
            f"{data_disk_guard_command()}; "
            f"chmod +x {shell_quote(self.start_script_path)} && "
            f"sudo install -m 0644 {shell_quote(self.staged_service_path)} "
            f"/etc/systemd/system/{shell_quote(self.service_name)} && "
            "sudo systemctl daemon-reload && "
            f"sudo systemctl enable {shell_quote(self.service_name)} && "
            f"sudo systemctl restart {shell_quote(self.service_name)} && "
            f"sleep {int(sleep_seconds)} && "
            f"systemctl is-active {shell_quote(self.service_name)} && "
            f"{self.readiness_command()} && "
            f"tail -n {int(tail_lines)} {shell_quote(self.log_path)} 2>/dev/null || true"
        )

    def render_start_script(self) -> str:
        enforce_eager = " \\\n              --enforce-eager" if self.enforce_eager else ""
        prefix_caching = (
            " \\\n              --enable-prefix-caching" if self.enable_prefix_caching else ""
        )
        chunked_prefill = (
            " \\\n              --enable-chunked-prefill" if self.enable_chunked_prefill else ""
        )
        return textwrap.dedent(f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            source ~/anaconda3/etc/profile.d/conda.sh
            conda activate {shell_quote(self.conda_env)}
            export CUDA_VISIBLE_DEVICES={shell_quote(self.cuda_visible_devices)}
            export VLLM_WORKER_MULTIPROC_METHOD=spawn
            export VLLM_USE_V1=1
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
            export HF_HOME={shell_quote(self.hf_cache_dir)}
            export HF_HUB_CACHE={shell_quote(self.hf_cache_dir)}/hub
            export TRANSFORMERS_CACHE={shell_quote(self.hf_cache_dir)}/transformers
            LOG={shell_quote(self.log_path)}
            exec python -m vllm.entrypoints.openai.api_server \\
              --host 0.0.0.0 \\
              --port {self.port} \\
              --model {shell_quote(self.model_dir)} \\
              --served-model-name {shell_quote(self.served_model_name)} \\
              --trust-remote-code \\
              --max-model-len {self.max_model_len} \\
              --gpu-memory-utilization {self.gpu_memory_utilization:.2f} \\
              --dtype {shell_quote(self.dtype)} \\
              --quantization {shell_quote(self.quantization)} \\
              --max-num-seqs {self.max_num_seqs} \\
              --max-num-batched-tokens {self.max_num_batched_tokens}{prefix_caching}{chunked_prefill}{enforce_eager} > "$LOG" 2>&1
            """)

    def render_systemd_service(self) -> str:
        return textwrap.dedent(f"""\
            [Unit]
            Description={self.description}
            After=network.target

            [Service]
            Type=simple
            User={self.service_user}
            WorkingDirectory={REMOTE_AI_ROOT}
            ExecStart={self.start_script_path}
            Restart=always
            RestartSec=10
            Environment=CUDA_VISIBLE_DEVICES={self.cuda_visible_devices}
            Environment=VLLM_WORKER_MULTIPROC_METHOD=spawn
            Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
            LimitNOFILE=65535

            [Install]
            WantedBy=multi-user.target
            """)

    def render_download_script(self) -> str:
        model_dir_literal = json.dumps(self.model_dir)
        modelscope_model_literal = json.dumps(self.modelscope_model)
        model_repo_literal = json.dumps(self.model_repo)
        return textwrap.dedent(f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            export HF_HOME={shell_quote(self.hf_cache_dir)}
            export HF_HUB_CACHE={shell_quote(self.hf_cache_dir)}/hub
            export TRANSFORMERS_CACHE={shell_quote(self.hf_cache_dir)}/transformers
            export HF_ENDPOINT=https://hf-mirror.com
            source ~/anaconda3/etc/profile.d/conda.sh
            conda activate {shell_quote(self.conda_env)}
            if command -v modelscope >/dev/null 2>&1; then
              modelscope download \\
                --model {shell_quote(self.modelscope_model)} \\
                --local_dir {shell_quote(self.model_dir)} \\
                --max-workers 8 || \\
                echo "modelscope cli download failed; Python fallback will retry"
            else
              echo "modelscope cli not found; falling back to huggingface_hub"
            fi
            python - <<'PY'
            import json
            import shutil
            import subprocess
            from pathlib import Path

            target = Path({model_dir_literal})

            def model_complete(path: Path) -> bool:
                if not (path / "config.json").exists():
                    return False
                index_path = path / "model.safetensors.index.json"
                if index_path.exists():
                    try:
                        index = json.loads(index_path.read_text(encoding="utf-8"))
                    except Exception:
                        return False
                    required = set((index.get("weight_map") or {{}}).values())
                    if required:
                        return all(
                            (path / name).exists() and (path / name).stat().st_size > 0
                            for name in required
                        )
                shards = list(path.glob("*.safetensors"))
                return bool(shards) and all(shard.stat().st_size > 0 for shard in shards)

            if model_complete(target):
                print("model already present:", target)
            else:
                if target.exists():
                    shutil.rmtree(target)
                target.mkdir(parents=True, exist_ok=True)
                try:
                    subprocess.check_call([
                        "modelscope",
                        "download",
                        "--model",
                        {modelscope_model_literal},
                        "--local_dir",
                        str(target),
                        "--max-workers",
                        "8",
                    ])
                except Exception as exc:
                    print("modelscope download failed, falling back to huggingface_hub:", exc)
                try:
                    from huggingface_hub import snapshot_download
                except Exception:
                    import subprocess, sys
                    subprocess.check_call([
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "-q",
                        "huggingface_hub",
                    ])
                    from huggingface_hub import snapshot_download
                import os
                if not model_complete(target):
                    snapshot_download(
                        repo_id={model_repo_literal},
                        local_dir=str(target),
                        local_dir_use_symlinks=False,
                        resume_download=True,
                        endpoint=os.environ.get("HF_ENDPOINT"),
                    )
                if not model_complete(target):
                    raise RuntimeError(f"incomplete model download: {{target}}")
                print("model downloaded:", target)
            PY
            """)


QWEN3_32B_MAIN_SERVICE = RemoteVllmServiceSpec(
    model_repo="Qwen/Qwen3-32B-AWQ",
    modelscope_model="Qwen/Qwen3-32B-AWQ",
    model_dir=f"{REMOTE_MODEL_ROOT}/Qwen/Qwen3-32B-AWQ",
    served_model_name="qwen3-32b-trade",
    service_name="qwen3-32b-main.service",
    description="Qwen3 32B AWQ vLLM OpenAI API",
    start_script_name="start_qwen3_32b_main.sh",
    download_script_name="download_qwen3_32b_awq.sh",
    log_name="qwen3_32b_main.log",
)


def qwen3_main_cleanup_command() -> str:
    """Stop older LLM services and remove only known obsolete remote model paths."""
    services = " ".join(shell_quote(service) for service in LEGACY_MAIN_LLM_SERVICES)
    service_files = " ".join(
        f"/etc/systemd/system/{shell_quote(service)}" for service in LEGACY_MAIN_LLM_SERVICES
    )
    script_paths = " ".join(shell_quote(path) for path in LEGACY_LLM_SCRIPT_PATHS)
    model_paths = " ".join(shell_quote(path) for path in QWEN3_MAIN_REMOTE_MODEL_CLEANUP_PATHS)
    runtime_dirs = " ".join(
        shell_quote(path)
        for path in (
            f"{REMOTE_MODEL_ROOT}/Qwen",
            *REMOTE_RUNTIME_DIRS,
        )
    )
    return (
        f"{data_disk_guard_command()}; "
        f"sudo systemctl stop {services} local-ai-tools.service 2>/dev/null || true; "
        f"sudo systemctl disable {services} 2>/dev/null || true; "
        f"sudo rm -f {service_files}; "
        "pkill -f '[v]llm.entrypoints.openai.api_server|[d]ownload_deepseek|"
        "[d]ownload_qwen3|[s]napshot_download|[h]uggingface|[m]odelscope' "
        "2>/dev/null || true; "
        "sudo systemctl daemon-reload; "
        f"rm -f {script_paths}; "
        f"rm -rf {model_paths}; "
        f"mkdir -p {runtime_dirs}; "
        "echo data-disk-ready-and-old-llm-services-cleaned"
    )
