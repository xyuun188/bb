"""Open local SSH tunnels for the dual-14B remote vLLM services.

The trading backend should use local OpenAI-compatible endpoints:
- http://127.0.0.1:8000/v1 -> remote qwen3-14b-trade
- http://127.0.0.1:8002/v1 -> remote deepseek-r1-14b-risk

This keeps remote vLLM ports private while allowing the local paper-trading
process to call them as if they were local services. Server credentials are read
through core.remote_ssh and are never printed.
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

from core.remote_ai_service_spec import (  # noqa: E402
    DEEPSEEK_R1_14B_RISK_SERVICE,
    QWEN3_14B_TRADE_SERVICE,
)
from core.remote_ssh import connect_remote_ssh  # noqa: E402
from core.safe_output import safe_print  # noqa: E402

BUFFER_SIZE = 16384
SELECT_TIMEOUT_SECONDS = 1.0


@dataclass(frozen=True)
class TunnelSpec:
    """One local-to-remote TCP forwarding rule."""

    name: str
    local_host: str
    local_port: int
    remote_host: str
    remote_port: int


class _ForwardServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, spec: TunnelSpec, ssh_transport: Any) -> None:
        self.spec = spec
        self.ssh_transport = ssh_transport
        super().__init__((spec.local_host, spec.local_port), _ForwardHandler)


class _ForwardHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        server = self.server
        assert isinstance(server, _ForwardServer)
        peer = self.request.getpeername()
        try:
            channel = server.ssh_transport.open_channel(
                "direct-tcpip",
                (server.spec.remote_host, server.spec.remote_port),
                peer,
            )
        except Exception as exc:  # pragma: no cover - depends on live SSH transport.
            safe_print(f"{server.spec.name} tunnel open failed: {exc}")
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
                    data = self.request.recv(BUFFER_SIZE)
                    if not data:
                        break
                    channel.sendall(data)
                if channel in readable:
                    data = channel.recv(BUFFER_SIZE)
                    if not data:
                        break
                    self.request.sendall(data)
        finally:
            channel.close()
            self.request.close()


def build_default_tunnels(
    *,
    local_host: str = "127.0.0.1",
    remote_host: str = "127.0.0.1",
    qwen_local_port: int = QWEN3_14B_TRADE_SERVICE.port,
    deepseek_local_port: int = DEEPSEEK_R1_14B_RISK_SERVICE.port,
) -> list[TunnelSpec]:
    """Return the approved local tunnels for the dual-14B deployment."""
    return [
        TunnelSpec(
            name=QWEN3_14B_TRADE_SERVICE.served_model_name,
            local_host=local_host,
            local_port=int(qwen_local_port),
            remote_host=remote_host,
            remote_port=QWEN3_14B_TRADE_SERVICE.port,
        ),
        TunnelSpec(
            name=DEEPSEEK_R1_14B_RISK_SERVICE.served_model_name,
            local_host=local_host,
            local_port=int(deepseek_local_port),
            remote_host=remote_host,
            remote_port=DEEPSEEK_R1_14B_RISK_SERVICE.port,
        ),
    ]


def _start_servers(specs: list[TunnelSpec], ssh_transport: Any) -> list[_ForwardServer]:
    servers: list[_ForwardServer] = []
    for spec in specs:
        server = _ForwardServer(spec, ssh_transport)
        thread = threading.Thread(target=server.serve_forever, name=f"tunnel-{spec.name}")
        thread.daemon = True
        thread.start()
        servers.append(server)
        safe_print(
            f"{spec.name}: http://{spec.local_host}:{spec.local_port}/v1 "
            f"-> {spec.remote_host}:{spec.remote_port}"
        )
    return servers


def run_tunnels(specs: list[TunnelSpec]) -> None:
    """Connect SSH once and keep all local forwarders alive."""
    ssh = connect_remote_ssh(ROOT, timeout=20)
    servers: list[_ForwardServer] = []
    try:
        transport = ssh.get_transport()
        if transport is None or not transport.is_active():
            raise RuntimeError("SSH transport is not active")
        servers = _start_servers(specs, transport)
        safe_print("dual-14B vLLM tunnels ready")
        while transport.is_active():
            time.sleep(5)
        raise RuntimeError("SSH transport closed")
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        ssh.close()


def _positive_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("port must be an integer") from None
    if port <= 0 or port > 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-host", default="127.0.0.1")
    parser.add_argument("--remote-host", default="127.0.0.1")
    parser.add_argument(
        "--qwen-local-port", type=_positive_port, default=QWEN3_14B_TRADE_SERVICE.port
    )
    parser.add_argument(
        "--deepseek-local-port",
        type=_positive_port,
        default=DEEPSEEK_R1_14B_RISK_SERVICE.port,
    )
    args = parser.parse_args(argv)
    specs = build_default_tunnels(
        local_host=args.local_host,
        remote_host=args.remote_host,
        qwen_local_port=args.qwen_local_port,
        deepseek_local_port=args.deepseek_local_port,
    )
    run_tunnels(specs)


if __name__ == "__main__":
    main()
