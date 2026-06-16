#!/usr/bin/env python3
"""Open an SSH tunnel from this workstation to the online PostgreSQL server.

Keep this process running while local tools need to access the online database.
The local .env should point DATABASE_URL at 127.0.0.1:15432.
"""

from __future__ import annotations

import argparse
import select
import socket
import socketserver
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh  # noqa: E402
from core.safe_output import safe_error_text, safe_print  # noqa: E402

DEFAULT_LOCAL_HOST = "127.0.0.1"
DEFAULT_LOCAL_PORT = 15432
DEFAULT_REMOTE_HOST = "127.0.0.1"
DEFAULT_REMOTE_PORT = 5432


@dataclass(frozen=True)
class TunnelTarget:
    remote_host: str
    remote_port: int
    ssh_transport: Any


class ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True
    tunnel_target: TunnelTarget


class ForwardHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        target = self.server.tunnel_target  # type: ignore[attr-defined]
        try:
            channel = target.ssh_transport.open_channel(
                "direct-tcpip",
                (target.remote_host, target.remote_port),
                self.request.getpeername(),
            )
        except Exception as exc:
            safe_print(f"Failed to open tunnel channel: {safe_error_text(exc)}")
            return
        if channel is None:
            safe_print("Failed to open tunnel channel: server refused request")
            return

        try:
            while True:
                readable, _, _ = select.select([self.request, channel], [], [])
                if self.request in readable:
                    data = self.request.recv(16384)
                    if not data:
                        break
                    channel.sendall(data)
                if channel in readable:
                    data = channel.recv(16384)
                    if not data:
                        break
                    self.request.sendall(data)
        except (OSError, EOFError):
            pass
        finally:
            channel.close()
            try:
                self.request.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.request.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-host", default=DEFAULT_LOCAL_HOST)
    parser.add_argument("--local-port", type=int, default=DEFAULT_LOCAL_PORT)
    parser.add_argument("--remote-host", default=DEFAULT_REMOTE_HOST)
    parser.add_argument("--remote-port", type=int, default=DEFAULT_REMOTE_PORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ssh = connect_remote_ssh(ROOT, timeout=20)
    try:
        transport = ssh.get_transport()
        if transport is None or not transport.is_active():
            raise RuntimeError("SSH transport is not active.")
        server = ThreadingTCPServer((args.local_host, args.local_port), ForwardHandler)
        server.tunnel_target = TunnelTarget(
            remote_host=args.remote_host,
            remote_port=args.remote_port,
            ssh_transport=transport,
        )
        stop_event = threading.Event()
        safe_print(
            "Online database tunnel ready: "
            f"{args.local_host}:{args.local_port} -> {args.remote_host}:{args.remote_port}"
        )
        safe_print("Keep this window open while local code reads the online database.")
        try:
            while not stop_event.is_set():
                server.handle_request()
        except KeyboardInterrupt:
            safe_print("Stopping database tunnel...")
        finally:
            server.server_close()
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
