"""Deploy Qwen3-32B-AWQ as the local main LLM service."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ai_service_spec import (  # noqa: E402
    QWEN3_32B_MAIN_SERVICE,
    qwen3_main_cleanup_command,
)
from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402


def main() -> None:
    spec = QWEN3_32B_MAIN_SERVICE
    ssh = connect_remote_ssh(ROOT, timeout=20)
    try:
        safe_print(run_remote_text(ssh, qwen3_main_cleanup_command()))

        sftp = ssh.open_sftp()
        with sftp.file(spec.download_script_path, "w") as remote:
            remote.write(spec.render_download_script())
        sftp.close()
        safe_print(
            run_remote_text(
                ssh,
                spec.download_and_run_command(),
                timeout=7200,
            )
        )

        sftp = ssh.open_sftp()
        with sftp.file(spec.start_script_path, "w") as remote:
            remote.write(spec.render_start_script())
        with sftp.file(spec.staged_service_path, "w") as remote:
            remote.write(spec.render_systemd_service())
        sftp.close()
        safe_print(
            run_remote_text(
                ssh,
                spec.install_and_restart_command(sleep_seconds=8, tail_lines=80),
                timeout=300,
            )
        )
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
