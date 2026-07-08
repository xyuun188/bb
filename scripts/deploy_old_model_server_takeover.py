#!/usr/bin/env python3
"""Deploy one-GPU old-model-server compatibility helpers.

This does not replace the new model-server deployment.  It installs only the
lightweight pieces needed for the old server to temporarily satisfy the current
platform contract while reusing the already-running Qwen3-14B and DeepSeek-14B
services.
"""

from __future__ import annotations

import argparse
import json
import posixpath
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_server_info import DEFAULT_ACCOUNT_INFO_DIR, parse_remote_server_info  # noqa: E402
from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402

OLD_PROFILE_FILENAME = "大模型服务器信息.txt"
REMOTE_ROOT = "/data/BB"
ALIAS_APP_DIR = f"{REMOTE_ROOT}/services/model_alias_proxy"
REMOTE_UPLOAD_DIR = f"{REMOTE_ROOT}/runtime/uploads"
ALIAS_SCRIPT = f"{ALIAS_APP_DIR}/finquant_expert_alias.py"
ALIAS_SERVICE_NAME = "bb-finquant-expert-alias.service"
ALIAS_SERVICE_PATH = f"/etc/systemd/system/{ALIAS_SERVICE_NAME}"
ALIAS_PORT = 8003
UPSTREAM_PORT = 8000


ALIAS_PROXY_CODE = r'''
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = "127.0.0.1"
PORT = 8003
UPSTREAM = "http://127.0.0.1:8000"
ALIAS_MODEL = "BB-FinQuant-Expert-14B"
UPSTREAM_MODEL = "qwen3-14b-trade"


class Handler(BaseHTTPRequestHandler):
    server_version = "BBFinQuantAlias/1.0"

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _write_json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - http.server method name
        if self.path.rstrip("/") == "/v1/models":
            self._write_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": ALIAS_MODEL,
                            "object": "model",
                            "owned_by": "bb-old-server-alias",
                            "root": UPSTREAM_MODEL,
                            "parent": UPSTREAM_MODEL,
                        }
                    ],
                },
            )
            return
        self._proxy()

    def do_POST(self) -> None:  # noqa: N802 - http.server method name
        self._proxy()

    def _proxy(self) -> None:
        body = self.rfile.read(int(self.headers.get("content-length") or 0))
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in {"host", "content-length", "connection"}
        }
        if self.path.startswith("/v1/") and body:
            try:
                payload = json.loads(body.decode("utf-8"))
                if isinstance(payload, dict) and payload.get("model") == ALIAS_MODEL:
                    payload["model"] = UPSTREAM_MODEL
                    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                    headers["content-type"] = "application/json"
            except Exception:
                pass
        request = urllib.request.Request(
            UPSTREAM + self.path,
            data=body if self.command != "GET" else None,
            headers=headers,
            method=self.command,
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
                self.send_response(response.status)
                for key, value in response.headers.items():
                    if key.lower() in {"transfer-encoding", "connection"}:
                        continue
                    self.send_header(key, value)
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except urllib.error.HTTPError as exc:
            data = exc.read()
            self.send_response(exc.code)
            self.send_header("content-type", exc.headers.get("content-type", "application/json"))
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as exc:
            self._write_json(502, {"error": "finquant_alias_upstream_failed", "detail": str(exc)[:200]})


if __name__ == "__main__":
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
'''


def sh(value: str) -> str:
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


def _load_old_server_info(account_dir: Path):
    path = account_dir / OLD_PROFILE_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"old model-server file not found: {path}")
    return parse_remote_server_info(path.read_text(encoding="utf-8", errors="replace"), source_path=path)


def _upload_text(ssh, remote_path: str, content: str, *, mode: int) -> None:
    target_path = remote_path
    upload_path = remote_path
    needs_sudo_install = remote_path.startswith("/etc/")
    if needs_sudo_install:
        upload_path = f"{REMOTE_UPLOAD_DIR}/bb-upload-{posixpath.basename(remote_path)}"
    run_remote_text(ssh, f"mkdir -p {sh(posixpath.dirname(upload_path))}", timeout=30)
    sftp = ssh.open_sftp()
    try:
        with sftp.file(upload_path, "w") as remote:
            remote.write(content)
        sftp.chmod(upload_path, mode)
    finally:
        sftp.close()
    if needs_sudo_install:
        run_remote_text(
            ssh,
            f"sudo install -m {mode:o} {sh(upload_path)} {sh(target_path)} && rm -f {sh(upload_path)}",
            timeout=30,
        )


def _service_text() -> str:
    return textwrap.dedent(
        f"""
        [Unit]
        Description=BB FinQuant Expert alias proxy for old one-GPU model server
        After=network-online.target qwen3-14b-trade.service
        Wants=network-online.target

        [Service]
        Type=simple
        User=linux
        WorkingDirectory={ALIAS_APP_DIR}
        ExecStart=/usr/bin/python3 {ALIAS_SCRIPT}
        Restart=always
        RestartSec=3
        LimitNOFILE=65535

        [Install]
        WantedBy=multi-user.target
        """
    ).lstrip()


def deploy(*, account_dir: Path, apply: bool) -> dict[str, object]:
    info = _load_old_server_info(account_dir)
    if not apply:
        return {
            "apply": False,
            "target": info.redacted(),
            "would_install": {
                "script": ALIAS_SCRIPT,
                "service": ALIAS_SERVICE_PATH,
                "alias_port": ALIAS_PORT,
                "upstream_port": UPSTREAM_PORT,
            },
        }

    ssh = connect_remote_ssh(ROOT, timeout=20, info=info)
    try:
        _upload_text(ssh, ALIAS_SCRIPT, ALIAS_PROXY_CODE, mode=0o755)
        _upload_text(ssh, ALIAS_SERVICE_PATH, _service_text(), mode=0o644)
        run_remote_text(
            ssh,
            "sudo systemctl daemon-reload && "
            f"sudo systemctl enable --now {sh(ALIAS_SERVICE_NAME)} && "
            f"sudo systemctl restart {sh(ALIAS_SERVICE_NAME)}",
            timeout=60,
        )
        probe = run_remote_text(
            ssh,
            f"curl -fsS --max-time 10 http://127.0.0.1:{ALIAS_PORT}/v1/models",
            timeout=20,
        )
        return {
            "apply": True,
            "target": info.redacted(),
            "service": ALIAS_SERVICE_NAME,
            "probe": json.loads(probe),
        }
    finally:
        ssh.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account-dir", type=Path, default=DEFAULT_ACCOUNT_INFO_DIR)
    parser.add_argument("--apply", action="store_true", help="Install and start the alias proxy.")
    args = parser.parse_args(argv)

    result = deploy(account_dir=args.account_dir, apply=args.apply)
    safe_print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
