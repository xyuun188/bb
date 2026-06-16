"""Create and start the already-downloaded Qwen3-32B-AWQ main vLLM service."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ai_service_spec import QWEN3_32B_MAIN_SERVICE  # noqa: E402
from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402
from core.model_server_bridge import load_model_server_info_from_platform  # noqa: E402


def main() -> None:
    spec = QWEN3_32B_MAIN_SERVICE
    info = load_model_server_info_from_platform(ROOT)
    ssh = connect_remote_ssh(ROOT, timeout=20, info=info)
    try:
        run_remote_text(ssh, spec.model_presence_command())
        run_remote_text(ssh, spec.runtime_dirs_command())
        sftp = ssh.open_sftp()
        with sftp.file(spec.start_script_path, "w") as remote:
            remote.write(spec.render_start_script())
        with sftp.file(spec.staged_service_path, "w") as remote:
            remote.write(spec.render_systemd_service())
        sftp.close()
        safe_print(run_remote_text(ssh, spec.install_and_restart_command(), timeout=300))
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
