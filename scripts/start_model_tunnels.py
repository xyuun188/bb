"""Start local SSH tunnels for remote model APIs.

This helper intentionally does not store credentials. It reads the user-provided
server info text file at runtime, opens SSH port forwards, and keeps them alive.
"""

from __future__ import annotations

import argparse
import re
import select
import socketserver
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import paramiko


@dataclass(frozen=True)
class ServerInfo:
    host: str
    port: int
    username: str
    password: str


class ForwardServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class _BaseForwardHandler(socketserver.BaseRequestHandler):
    ssh_client: paramiko.SSHClient
    remote_host: str = "127.0.0.1"
    remote_port: int = 0

    def handle(self) -> None:
        transport = self.ssh_client.get_transport()
        if transport is None or not transport.is_active():
            return
        channel = transport.open_channel(
            "direct-tcpip",
            (self.remote_host, self.remote_port),
            self.request.getpeername(),
        )
        try:
            while True:
                readable, _, _ = select.select([self.request, channel], [], [], 60)
                if self.request in readable:
                    data = self.request.recv(65535)
                    if not data:
                        break
                    channel.sendall(data)
                if channel in readable:
                    data = channel.recv(65535)
                    if not data:
                        break
                    self.request.sendall(data)
        finally:
            channel.close()
            self.request.close()


def _candidate_server_files(project_root: Path) -> list[Path]:
    return [
        path
        for path in project_root.iterdir()
        if path.is_file() and path.suffix.lower() == ".txt" and "资料" in path.name
    ]


def load_server_info(project_root: Path) -> ServerInfo:
    selected: Path | None = None
    raw = b""
    for path in _candidate_server_files(project_root):
        data = path.read_bytes()
        if b"175.155.64.171" in data or b"22184" in data:
            selected = path
            raw = data
            break
    if selected is None:
        raise RuntimeError("server info file was not found")

    text = raw.decode("utf-8", "ignore")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    model_section: list[str] = []
    in_model_section = False
    for line in lines:
        if "大模型" in line or "model" in line.lower():
            in_model_section = True
            continue
        if in_model_section and ("平台" in line or "platform" in line.lower()):
            break
        if in_model_section:
            model_section.append(line)
    if model_section:
        lines = model_section
    section_text = "\n".join(lines)
    host_match = re.search(r"(\d{1,3}(?:\.\d{1,3}){3})", section_text)
    if not host_match:
        raise RuntimeError("server info file does not contain a host")
    port_match = re.search(r"(?:端口|port)\s*[:：]?\s*(\d+)", section_text, re.I)
    numbers = re.findall(r"\b\d{2,5}\b", section_text)
    if not numbers:
        raise RuntimeError("server info file does not contain an SSH port")

    username = "linux"
    password = ""
    if len(lines) >= 3:
        username = re.split(r"[:：]", lines[2], maxsplit=1)[-1].strip() or username
    if len(lines) >= 4:
        password = re.split(r"[:：]", lines[3], maxsplit=1)[-1].strip()
    for line in lines:
        if "账号" in line or "用户" in line or "user" in line.lower():
            username = re.split(r"[:：]", line, maxsplit=1)[-1].strip() or username
        if "密码" in line or "password" in line.lower() or "pwd" in line.lower():
            password = re.split(r"[:：]", line, maxsplit=1)[-1].strip()
    if not password:
        raise RuntimeError("server password was not found")

    return ServerInfo(
        host=host_match.group(1),
        port=int(port_match.group(1) if port_match else numbers[0]),
        username=username,
        password=password,
    )


def connect_ssh(info: ServerInfo, project_root: Path) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    known_hosts = project_root / ".ssh" / "known_hosts"
    if known_hosts.exists():
        client.load_host_keys(str(known_hosts))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(
        info.host,
        port=info.port,
        username=info.username,
        password=info.password,
        timeout=15,
        banner_timeout=15,
        auth_timeout=15,
    )
    transport = client.get_transport()
    if transport is not None:
        transport.set_keepalive(30)
    return client


def start_forward(
    *,
    ssh_client: paramiko.SSHClient,
    local_port: int,
    remote_host: str,
    remote_port: int,
) -> ForwardServer:
    handler = type(
        f"ForwardHandler{local_port}",
        (_BaseForwardHandler,),
        {
            "ssh_client": ssh_client,
            "remote_host": remote_host,
            "remote_port": remote_port,
        },
    )
    server = ForwardServer(("127.0.0.1", local_port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main() -> int:
    parser = argparse.ArgumentParser(description="Start SSH tunnels for model APIs")
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument(
        "--forward",
        action="append",
        default=["8000:127.0.0.1:8000", "8003:127.0.0.1:8003"],
        help="Forward spec local_port:remote_host:remote_port",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    info = load_server_info(project_root)
    client = connect_ssh(info, project_root)
    servers: list[ForwardServer] = []
    for spec in args.forward:
        local_text, remote_host, remote_text = spec.split(":", 2)
        servers.append(
            start_forward(
                ssh_client=client,
                local_port=int(local_text),
                remote_host=remote_host,
                remote_port=int(remote_text),
            )
        )
    ports = ", ".join(str(server.server_address[1]) for server in servers)
    print(f"model tunnels ready on 127.0.0.1: {ports}", flush=True)
    while True:
        transport = client.get_transport()
        if transport is None or not transport.is_active():
            raise RuntimeError("SSH transport is inactive")
        time.sleep(30)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0) from None
    except Exception as exc:
        print(f"model tunnel failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
