"""Fix local AI tools systemd service Python path on the server."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.model_server_bridge import load_model_server_info_from_platform  # noqa: E402
from core.remote_ai_service_spec import shell_quote  # noqa: E402
from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402

REMOTE_SERVICE_DIR = "/data/trade_ai/systemd"
REMOTE_SERVICE_PATH = f"{REMOTE_SERVICE_DIR}/local-ai-tools.service"
SYSTEMD_SERVICE_PATH = "/etc/systemd/system/local-ai-tools.service"


def normalize_remote_python_path(value: str) -> str:
    """Return a validated POSIX path from remote command output."""
    lines = [line.strip() for line in str(value or "").splitlines() if line.strip()]
    if not lines:
        raise ValueError("Remote Python path was not found.")
    path = lines[0]
    if "\x00" in path or any(ch.isspace() for ch in path):
        raise ValueError("Remote Python path contains unsupported characters.")
    posix_path = PurePosixPath(path)
    if not posix_path.is_absolute():
        raise ValueError("Remote Python path must be absolute.")
    return posix_path.as_posix()


def render_local_ai_tools_service(python_bin: str) -> str:
    """Render the local-AI-tools systemd unit without using local OS path rules."""
    clean_python_bin = normalize_remote_python_path(python_bin)
    env_bin = PurePosixPath(clean_python_bin).parent.as_posix()
    return textwrap.dedent(f"""
            [Unit]
            Description=Trade Local AI Tools API
            After=network-online.target qwen3-32b-main.service
            Wants=network-online.target

            [Service]
            User=linux
            WorkingDirectory=/data/trade_ai/tools
            Environment=PATH={env_bin}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
            Environment=LOCAL_AI_TOOLS_ALLOW_UNAUTHENTICATED_LOOPBACK=true
            Environment=LOCAL_AI_TOOLS_CORS_ORIGINS=http://127.0.0.1:8002,http://localhost:8002
            EnvironmentFile=-/data/trade_ai/local_ai_tools.env
            LimitNOFILE=65535
            ExecStart={clean_python_bin} -m uvicorn local_ai_tools_api:app --host 0.0.0.0 --port 8001 --timeout-keep-alive 5
            Restart=always
            RestartSec=5
            StandardOutput=append:/data/trade_ai/logs/local_ai_tools_api.log
            StandardError=append:/data/trade_ai/logs/local_ai_tools_api.err.log

            [Install]
            WantedBy=multi-user.target
            """).strip() + "\n"


def main() -> None:
    info = load_model_server_info_from_platform(ROOT)
    ssh = connect_remote_ssh(ROOT, timeout=15, info=info)
    try:
        found = run_remote_text(
            ssh,
            "find /home /data /opt -path '*/envs/trade_ml/bin/python' 2>/dev/null | head -1",
        ).strip()
        if not found:
            found = run_remote_text(ssh, "command -v python3").strip()
        python_bin = normalize_remote_python_path(found)
        service = render_local_ai_tools_service(python_bin)
        sftp = ssh.open_sftp()
        run_remote_text(ssh, f"mkdir -p {shell_quote(REMOTE_SERVICE_DIR)}")
        with sftp.file(REMOTE_SERVICE_PATH, "w") as remote:
            remote.write(service)
        sftp.close()
        safe_print(f"python={python_bin}")
        safe_print(
            run_remote_text(
                ssh,
                f"sudo install -m 0644 {shell_quote(REMOTE_SERVICE_PATH)} {shell_quote(SYSTEMD_SERVICE_PATH)} && "
                "sudo systemctl daemon-reload && "
                "sudo systemctl restart local-ai-tools.service && "
                "sleep 3 && systemctl is-active local-ai-tools.service && curl -sS http://127.0.0.1:8001/health",
            )
        )
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
