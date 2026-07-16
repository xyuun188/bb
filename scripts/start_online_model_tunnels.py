"""Run SSH tunnels from the online platform server to the model server.

The online platform must not call the model server through fragile public port
forwarding for high-volume POST traffic. This process runs on the platform
server and forwards loopback-only ports to the model server's loopback services:

- 127.0.0.1:18000 -> model server 127.0.0.1:8000 (qwen3-14b-trade)
- 127.0.0.1:18001 -> model server 127.0.0.1:8101 (phase3 quant API health)
- 127.0.0.1:18002 -> model server 127.0.0.1:8002 (deepseek-r1-14b-risk)
- 127.0.0.1:18003 -> model server 127.0.0.1:8003 (BB-FinQuant-Expert-14B)

Model-server SSH credentials are loaded from encrypted secure settings on the
platform. Secrets are never printed.
"""

from __future__ import annotations

import argparse
import select
import socketserver
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh  # noqa: E402
from core.safe_output import safe_error_text, safe_print  # noqa: E402
from services.model_server_config import (  # noqa: E402
    load_model_server_info_from_secure_settings_sync,
)

BUFFER_SIZE = 65_535
SELECT_TIMEOUT_SECONDS = 1.0
TRANSPORT_KEEPALIVE_SECONDS = 30


@dataclass(frozen=True, slots=True)
class TunnelSpec:
    """One loopback TCP forwarding rule."""

    name: str
    local_host: str
    local_port: int
    remote_host: str
    remote_port: int


class ForwardServer(socketserver.ThreadingTCPServer):
    """Threaded TCP forwarder bound to one local loopback port."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, spec: TunnelSpec, ssh_transport: Any) -> None:
        self.spec = spec
        self.ssh_transport = ssh_transport
        super().__init__((spec.local_host, spec.local_port), ForwardHandler)


class ForwardHandler(socketserver.BaseRequestHandler):
    """Bidirectionally copy bytes between a local socket and an SSH channel."""

    @staticmethod
    def _recv_or_empty(sock: Any) -> bytes:
        try:
            return sock.recv(BUFFER_SIZE)
        except (ConnectionResetError, BrokenPipeError, OSError):
            return b""

    @staticmethod
    def _sendall_or_closed(sock: Any, data: bytes) -> bool:
        try:
            sock.sendall(data)
            return True
        except (ConnectionResetError, BrokenPipeError, OSError):
            return False

    def handle(self) -> None:
        server = self.server
        assert isinstance(server, ForwardServer)
        try:
            peer = self.request.getpeername()
            channel = server.ssh_transport.open_channel(
                "direct-tcpip",
                (server.spec.remote_host, server.spec.remote_port),
                peer,
            )
        except Exception as exc:  # pragma: no cover - live SSH transport only.
            safe_print(f"{server.spec.name} tunnel open failed: {safe_error_text(exc)}")
            return
        if channel is None:
            safe_print(f"{server.spec.name} tunnel rejected by SSH server")
            return

        try:
            while True:
                readable, _, _ = select.select(
                    [self.request, channel],
                    [],
                    [],
                    SELECT_TIMEOUT_SECONDS,
                )
                if self.request in readable:
                    data = self._recv_or_empty(self.request)
                    if not data:
                        break
                    if not self._sendall_or_closed(channel, data):
                        break
                if channel in readable:
                    data = self._recv_or_empty(channel)
                    if not data:
                        break
                    if not self._sendall_or_closed(self.request, data):
                        break
        finally:
            channel.close()
            self.request.close()


def build_default_tunnels(local_host: str = "127.0.0.1") -> list[TunnelSpec]:
    """Return the approved platform-to-model-server tunnels."""

    return [
        TunnelSpec(
            name="qwen3-14b-trade",
            local_host=local_host,
            local_port=18_000,
            remote_host="127.0.0.1",
            remote_port=8000,
        ),
        TunnelSpec(
            name="phase3-quant-api",
            local_host=local_host,
            local_port=18_001,
            remote_host="127.0.0.1",
            remote_port=8101,
        ),
        TunnelSpec(
            name="deepseek-r1-14b-risk",
            local_host=local_host,
            local_port=18_002,
            remote_host="127.0.0.1",
            remote_port=8002,
        ),
        TunnelSpec(
            name="BB-FinQuant-Expert-14B",
            local_host=local_host,
            local_port=18_003,
            remote_host="127.0.0.1",
            remote_port=8003,
        ),
    ]


def open_dedicated_transports(
    specs: list[TunnelSpec],
    server_info: Any,
) -> tuple[list[Any], list[Any]]:
    """Open one SSH transport per endpoint to prevent cross-model blocking."""

    ssh_clients: list[Any] = []
    transports: list[Any] = []
    try:
        for spec in specs:
            ssh = connect_remote_ssh(ROOT, timeout=20, info=server_info)
            ssh_clients.append(ssh)
            transport = ssh.get_transport()
            if transport is None or not transport.is_active():
                raise RuntimeError(f"{spec.name} SSH transport is not active")
            transport.set_keepalive(TRANSPORT_KEEPALIVE_SECONDS)
            transports.append(transport)
    except Exception:
        for ssh in reversed(ssh_clients):
            ssh.close()
        raise
    return ssh_clients, transports


def start_servers(
    specs: list[TunnelSpec],
    ssh_transports: list[Any],
) -> list[ForwardServer]:
    """Start local forwarders with a dedicated transport for every endpoint."""

    if len(specs) != len(ssh_transports):
        raise ValueError("each tunnel endpoint requires one dedicated SSH transport")

    servers: list[ForwardServer] = []
    for spec, ssh_transport in zip(specs, ssh_transports, strict=True):
        server = ForwardServer(spec, ssh_transport)
        thread = threading.Thread(target=server.serve_forever, name=f"tunnel-{spec.name}")
        thread.daemon = True
        thread.start()
        servers.append(server)
        safe_print(
            f"{spec.name}: http://{spec.local_host}:{spec.local_port} "
            f"-> {spec.remote_host}:{spec.remote_port}"
        )
    return servers


def run_tunnels(specs: list[TunnelSpec]) -> None:
    """Connect isolated SSH transports and keep loopback tunnels alive."""

    info = load_model_server_info_from_secure_settings_sync()
    ssh_clients: list[Any] = []
    transports: list[Any] = []
    servers: list[ForwardServer] = []
    try:
        ssh_clients, transports = open_dedicated_transports(specs, info)
        servers = start_servers(specs, transports)
        safe_print("online model tunnels ready with isolated transports")
        while True:
            inactive_names = [
                spec.name
                for spec, transport in zip(specs, transports, strict=True)
                if not transport.is_active()
            ]
            if inactive_names:
                raise RuntimeError(
                    "SSH transport closed for: " + ", ".join(inactive_names)
                )
            time.sleep(5)
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for ssh in reversed(ssh_clients):
            ssh.close()


def parse_port(value: str) -> int:
    """Parse a positive TCP port for CLI overrides."""

    try:
        port = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("port must be an integer") from None
    if port <= 0 or port > 65_535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-host", default="127.0.0.1")
    parser.add_argument("--qwen-local-port", type=parse_port, default=18_000)
    parser.add_argument("--quant-api-local-port", type=parse_port, default=18_001)
    parser.add_argument("--deepseek-local-port", type=parse_port, default=18_002)
    parser.add_argument("--expert-local-port", type=parse_port, default=18_003)
    args = parser.parse_args(argv)

    specs = build_default_tunnels(local_host=args.local_host)
    specs = [
        TunnelSpec(
            name=spec.name,
            local_host=spec.local_host,
            local_port={
                "qwen3-14b-trade": args.qwen_local_port,
                "phase3-quant-api": args.quant_api_local_port,
                "deepseek-r1-14b-risk": args.deepseek_local_port,
                "BB-FinQuant-Expert-14B": args.expert_local_port,
            }[spec.name],
            remote_host=spec.remote_host,
            remote_port=spec.remote_port,
        )
        for spec in specs
    ]
    run_tunnels(specs)


if __name__ == "__main__":
    main()
