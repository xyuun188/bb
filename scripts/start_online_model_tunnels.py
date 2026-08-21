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
import http.client
import select
import socket
import socketserver
import sys
import threading
import time
from collections.abc import Callable
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
TUNNEL_HEALTH_CHECK_INTERVAL_SECONDS = 15.0
TUNNEL_HEALTH_TIMEOUT_SECONDS = 5.0
# A forwarded request must not leave a handler blocked forever when the
# upstream client gives up or the remote service stops reading.  Without this
# bound, repeated dashboard timeouts accumulate CLOSE-WAIT sockets and
# eventually make the loopback listener look alive while it cannot accept
# useful traffic.
FORWARD_CHANNEL_IO_TIMEOUT_SECONDS = 10.0
FORWARD_CHANNEL_OPEN_TIMEOUT_SECONDS = 15.0
# Spread busy endpoints across more than one SSH session and keep each session
# below the model server's sshd MaxSessions limit.
FORWARD_CHANNELS_PER_TRANSPORT = 5
FORWARD_TRANSPORT_POOL_SIZES = {
    "phase3-quant-api": 2,
    "BB-FinQuant-Expert-14B": 2,
}
FORWARD_DEFAULT_MAX_CONNECTION_SECONDS = 600.0
FORWARD_QUANT_MAX_CONNECTION_SECONDS = 1_800.0
# Training can briefly saturate the quant API while the SSH transport remains healthy.
# Require a sustained semantic outage before rebuilding all four isolated tunnels.
TUNNEL_HEALTH_FAILURE_LIMIT = 6
REQUIRED_HTTP_HEALTH_PATHS = {"phase3-quant-api": "/health/live"}


def _log(message: str) -> None:
    """Write tunnel events immediately so journal timestamps reflect the real failure time."""

    safe_print(message, flush=True)


@dataclass(frozen=True, slots=True)
class TunnelSpec:
    """One loopback TCP forwarding rule."""

    name: str
    local_host: str
    local_port: int
    remote_host: str
    remote_port: int
    max_connection_seconds: float = FORWARD_DEFAULT_MAX_CONNECTION_SECONDS


def _transport_is_active(transport: Any) -> bool:
    """Read Paramiko transport liveness without allowing a stale object to break the loop."""

    try:
        return bool(transport is not None and transport.is_active())
    except Exception:
        return False


def _should_retire_transport(exc: BaseException, transport: Any) -> bool:
    """Retire only a transport failure, never an isolated channel-open refusal."""

    if isinstance(exc, (EOFError, TimeoutError, socket.timeout, ConnectionError, OSError)):
        return True
    message = safe_error_text(exc, limit=240).lower()
    return any(
        marker in message
        for marker in (
            "timeout opening channel",
            "ssh session not active",
            "transport is closed",
            "socket is closed",
            "connection reset by peer",
            "session closed",
            "eof",
        )
    )


def _should_retire_transport_immediately(exc: BaseException, transport: Any) -> bool:
    """Return whether an open failure proves the whole SSH session is already unusable."""

    if not _transport_is_active(transport):
        return True
    if isinstance(exc, EOFError):
        return True
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return False
    if isinstance(exc, (ConnectionError, OSError)):
        return True
    message = safe_error_text(exc, limit=240).lower()
    if "timeout opening channel" in message:
        return False
    return any(marker in message for marker in ("ssh session not active", "eof", "session closed"))


class TransportPool:
    """Bounded, replaceable SSH transport pool for one forwarded endpoint."""

    def __init__(self, transports: list[Any], *, slots_per_transport: int) -> None:
        if not transports:
            raise ValueError("a transport pool requires at least one transport")
        self._condition = threading.Condition()
        self._transports = list(transports)
        self._in_use = {id(transport): 0 for transport in transports}
        self._draining: set[int] = set()
        self._slots_per_transport = max(int(slots_per_transport), 1)
        self._next_index = 0

    @property
    def transports(self) -> tuple[Any, ...]:
        with self._condition:
            return tuple(self._transports)

    def acquire(self, timeout: float) -> Any | None:
        deadline = time.monotonic() + max(float(timeout), 0.0)
        with self._condition:
            while True:
                count = len(self._transports)
                for offset in range(count):
                    index = (self._next_index + offset) % count
                    transport = self._transports[index]
                    if (
                        _transport_is_active(transport)
                        and id(transport) not in self._draining
                        and self._in_use.get(id(transport), 0) < self._slots_per_transport
                    ):
                        self._next_index = (index + 1) % count
                        self._in_use[id(transport)] = self._in_use.get(id(transport), 0) + 1
                        return transport
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(timeout=remaining)

    def release(self, transport: Any) -> None:
        close_transport = False
        with self._condition:
            transport_id = id(transport)
            if transport_id in self._in_use:
                self._in_use[transport_id] = max(self._in_use[transport_id] - 1, 0)
                close_transport = (
                    transport_id in self._draining and self._in_use[transport_id] == 0
                )
            self._condition.notify_all()
        if close_transport:
            self._close_transport(transport)

    @staticmethod
    def _close_transport(transport: Any) -> None:
        try:
            transport.close()
        except (AttributeError, OSError):
            pass

    def request_retirement(self, transport: Any, *, immediate: bool) -> bool:
        """Drain a suspect transport, preserving unrelated active channels when possible."""

        with self._condition:
            transport_id = id(transport)
            if transport_id not in self._in_use:
                return False
            self._draining.add(transport_id)
            active_channels = self._in_use[transport_id]
            close_transport = immediate or active_channels <= 1
            self._condition.notify_all()
        if close_transport:
            self._close_transport(transport)
        return close_transport

    def replace(self, old_transport: Any, new_transport: Any) -> bool:
        """Replace one failed transport while keeping all listeners and other channels alive."""

        with self._condition:
            index = next(
                (
                    current_index
                    for current_index, transport in enumerate(self._transports)
                    if transport is old_transport
                ),
                -1,
            )
            if index < 0:
                return False
            self._transports[index] = new_transport
            self._in_use.pop(id(old_transport), None)
            self._draining.discard(id(old_transport))
            self._in_use[id(new_transport)] = 0
            self._next_index = index % len(self._transports)
            self._condition.notify_all()
            return True

    def inactive_transports(self) -> list[Any]:
        with self._condition:
            return [
                transport for transport in self._transports if not _transport_is_active(transport)
            ]


def probe_tunnel_http_health(spec: TunnelSpec, path: str) -> tuple[bool, str]:
    """Verify that a forwarded endpoint responds, not only that its local port accepts TCP."""

    connection = http.client.HTTPConnection(
        spec.local_host,
        spec.local_port,
        timeout=TUNNEL_HEALTH_TIMEOUT_SECONDS,
    )
    try:
        connection.request("GET", path, headers={"Connection": "close"})
        response = connection.getresponse()
        response.read(1)
        if 200 <= int(response.status) < 300:
            return True, ""
        return False, f"HTTP {response.status}"
    except Exception as exc:
        return False, safe_error_text(exc, limit=120)
    finally:
        connection.close()


def check_required_tunnel_health(
    specs: list[TunnelSpec],
    failure_counts: dict[str, int],
    *,
    probe: Callable[[TunnelSpec, str], tuple[bool, str]] = probe_tunnel_http_health,
    failure_limit: int = TUNNEL_HEALTH_FAILURE_LIMIT,
) -> list[str]:
    """Return endpoints whose semantic health failed for the configured consecutive limit."""

    unhealthy: list[str] = []
    for spec in specs:
        path = REQUIRED_HTTP_HEALTH_PATHS.get(spec.name)
        if not path:
            continue
        healthy, error = probe(spec, path)
        if healthy:
            if failure_counts.get(spec.name, 0):
                _log(f"{spec.name} tunnel health recovered")
            failure_counts[spec.name] = 0
            continue
        count = failure_counts.get(spec.name, 0) + 1
        failure_counts[spec.name] = count
        _log(
            f"{spec.name} tunnel health failed ({count}/{failure_limit}): "
            f"{safe_error_text(error, limit=120)}"
        )
        if count >= max(int(failure_limit or 1), 1):
            unhealthy.append(spec.name)
    return unhealthy


class ForwardServer(socketserver.ThreadingTCPServer):
    """Threaded TCP forwarder bound to one local loopback port."""

    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 128

    def __init__(self, spec: TunnelSpec, ssh_transports: list[Any]) -> None:
        if not ssh_transports:
            raise ValueError(f"{spec.name} requires at least one SSH transport")
        self.spec = spec
        self.transport_pool = TransportPool(
            ssh_transports,
            slots_per_transport=FORWARD_CHANNELS_PER_TRANSPORT,
        )
        self.ssh_transports = self.transport_pool.transports
        self.ssh_transport = self.ssh_transports[0]
        super().__init__((spec.local_host, spec.local_port), ForwardHandler)

    def replace_transport(self, old_transport: Any, new_transport: Any) -> bool:
        replaced = self.transport_pool.replace(old_transport, new_transport)
        if replaced:
            self.ssh_transports = self.transport_pool.transports
            self.ssh_transport = self.ssh_transports[0]
        return replaced


class ForwardHandler(socketserver.BaseRequestHandler):
    """Bidirectionally copy bytes between a local socket and an SSH channel."""

    @staticmethod
    def _recv_or_empty(sock: Any) -> bytes:
        try:
            return sock.recv(BUFFER_SIZE)
        except Exception:
            return b""

    @staticmethod
    def _sendall_or_closed(sock: Any, data: bytes) -> bool:
        try:
            sock.sendall(data)
            return True
        except Exception:
            return False

    def handle(self) -> None:
        server = self.server
        self.request.settimeout(FORWARD_CHANNEL_IO_TIMEOUT_SECONDS)
        assert isinstance(server, ForwardServer)
        transport_pool = getattr(server, "transport_pool", None)
        ssh_transport = (
            transport_pool.acquire(FORWARD_CHANNEL_OPEN_TIMEOUT_SECONDS)
            if transport_pool is not None
            else getattr(server, "ssh_transport", None)
        )
        if ssh_transport is None:
            _log(f"{server.spec.name} tunnel channel capacity exhausted or unavailable")
            return
        try:
            try:
                peer = self.request.getpeername()
                channel = ssh_transport.open_channel(
                    "direct-tcpip",
                    (server.spec.remote_host, server.spec.remote_port),
                    peer,
                    timeout=FORWARD_CHANNEL_OPEN_TIMEOUT_SECONDS,
                )
            except Exception as exc:  # pragma: no cover - live SSH transport only.
                _log(f"{server.spec.name} tunnel open failed: {safe_error_text(exc)}")
                if _should_retire_transport(exc, ssh_transport):
                    immediate = _should_retire_transport_immediately(exc, ssh_transport)
                    if transport_pool is not None:
                        closed = transport_pool.request_retirement(
                            ssh_transport,
                            immediate=immediate,
                        )
                        if not closed:
                            _log(
                                f"{server.spec.name} tunnel transport draining; "
                                "active channels preserved"
                            )
                    elif immediate:
                        TransportPool._close_transport(ssh_transport)
                return
            if channel is None:
                _log(f"{server.spec.name} tunnel rejected by SSH server")
                return
            deadline = time.monotonic() + max(
                float(server.spec.max_connection_seconds),
                FORWARD_CHANNEL_IO_TIMEOUT_SECONDS,
            )
            try:
                channel.settimeout(FORWARD_CHANNEL_IO_TIMEOUT_SECONDS)
            except (AttributeError, OSError):
                pass

            try:
                while time.monotonic() < deadline:
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
            except Exception as exc:  # pragma: no cover - live channel races only.
                _log(
                    f"{server.spec.name} tunnel stream closed: "
                    f"{safe_error_text(exc, limit=120)}"
                )
            finally:
                try:
                    channel.shutdown_write()
                except (AttributeError, OSError):
                    pass
                try:
                    channel.close()
                except (AttributeError, OSError):
                    pass
        finally:
            if transport_pool is not None:
                transport_pool.release(ssh_transport)
            try:
                self.request.shutdown(socket.SHUT_RDWR)
            except (OSError, AttributeError):
                pass
            try:
                self.request.close()
            except (OSError, AttributeError):
                pass


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
            max_connection_seconds=FORWARD_QUANT_MAX_CONNECTION_SECONDS,
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
) -> tuple[list[Any], list[list[Any]]]:
    """Open isolated SSH transport pools so busy endpoints cannot head-of-line block."""

    ssh_clients: list[Any] = []
    transport_pools: list[list[Any]] = []
    try:
        for spec in specs:
            pool: list[Any] = []
            pool_size = max(int(FORWARD_TRANSPORT_POOL_SIZES.get(spec.name, 1)), 1)
            for _index in range(pool_size):
                ssh = connect_remote_ssh(ROOT, timeout=20, info=server_info)
                ssh_clients.append(ssh)
                transport = ssh.get_transport()
                if transport is None or not transport.is_active():
                    raise RuntimeError(f"{spec.name} SSH transport is not active")
                transport.set_keepalive(TRANSPORT_KEEPALIVE_SECONDS)
                pool.append(transport)
            transport_pools.append(pool)
    except Exception:
        for ssh in reversed(ssh_clients):
            ssh.close()
        raise
    return ssh_clients, transport_pools


def start_servers(
    specs: list[TunnelSpec],
    ssh_transport_pools: list[list[Any]],
) -> list[ForwardServer]:
    """Start local forwarders with an isolated transport pool per endpoint."""

    if len(specs) != len(ssh_transport_pools):
        raise ValueError("each tunnel endpoint requires one SSH transport pool")

    servers: list[ForwardServer] = []
    for spec, ssh_transports in zip(specs, ssh_transport_pools, strict=True):
        server = ForwardServer(spec, ssh_transports)
        thread = threading.Thread(target=server.serve_forever, name=f"tunnel-{spec.name}")
        thread.daemon = True
        thread.start()
        servers.append(server)
        _log(
            f"{spec.name}: http://{spec.local_host}:{spec.local_port} "
            f"-> {spec.remote_host}:{spec.remote_port}"
        )
    return servers


def recover_inactive_transports(
    specs: list[TunnelSpec],
    servers: list[ForwardServer],
    ssh_clients: list[Any],
    client_by_transport: dict[int, Any],
    server_info: Any,
) -> list[str]:
    """Replace failed endpoint transports in place and keep listeners serving."""

    failed_names: list[str] = []
    for spec, server in zip(specs, servers, strict=True):
        inactive = server.transport_pool.inactive_transports()
        for old_transport in inactive:
            new_client: Any | None = None
            replacement_installed = False
            try:
                new_client = connect_remote_ssh(ROOT, timeout=20, info=server_info)
                new_transport = new_client.get_transport()
                if new_transport is None or not _transport_is_active(new_transport):
                    raise RuntimeError(f"{spec.name} replacement SSH transport is not active")
                new_transport.set_keepalive(TRANSPORT_KEEPALIVE_SECONDS)
                if not server.replace_transport(old_transport, new_transport):
                    new_client.close()
                    continue
                replacement_installed = True
                old_client = client_by_transport.pop(id(old_transport), None)
                if old_client is not None:
                    try:
                        ssh_clients.remove(old_client)
                    except ValueError:
                        pass
                    try:
                        old_client.close()
                    except Exception as close_exc:
                        _log(
                            f"{spec.name} old SSH client close deferred: "
                            f"{safe_error_text(close_exc, limit=120)}"
                        )
                ssh_clients.append(new_client)
                client_by_transport[id(new_transport)] = new_client
                _log(f"{spec.name} tunnel transport recovered in place")
            except Exception as exc:  # pragma: no cover - live SSH recovery only.
                if new_client is not None and not replacement_installed:
                    new_client.close()
                failed_names.append(spec.name)
                _log(
                    f"{spec.name} tunnel transport recovery deferred: "
                    f"{safe_error_text(exc, limit=180)}"
                )
    return sorted(set(failed_names))


def run_tunnels(specs: list[TunnelSpec]) -> None:
    """Connect isolated SSH transports and keep loopback tunnels alive."""

    info = load_model_server_info_from_secure_settings_sync()
    ssh_clients: list[Any] = []
    transport_pools: list[list[Any]] = []
    servers: list[ForwardServer] = []
    client_by_transport: dict[int, Any] = {}
    try:
        ssh_clients, transport_pools = open_dedicated_transports(specs, info)
        servers = start_servers(specs, transport_pools)
        client_index = 0
        for pool in transport_pools:
            for transport in pool:
                client_by_transport[id(transport)] = ssh_clients[client_index]
                client_index += 1
        _log("online model tunnels ready with isolated transport pools")
        health_failure_counts: dict[str, int] = {}
        next_health_check = time.monotonic() + TUNNEL_HEALTH_CHECK_INTERVAL_SECONDS
        while True:
            inactive_names = [
                server.spec.name
                for server in servers
                if server.transport_pool.inactive_transports()
            ]
            if inactive_names:
                unrecovered_names = recover_inactive_transports(
                    specs,
                    servers,
                    ssh_clients,
                    client_by_transport,
                    info,
                )
                if unrecovered_names:
                    _log("SSH transport recovery pending for: " + ", ".join(unrecovered_names))
            now = time.monotonic()
            if now >= next_health_check:
                unhealthy_names = check_required_tunnel_health(
                    specs,
                    health_failure_counts,
                )
                if unhealthy_names:
                    # A forwarded HTTP health request can legitimately time out while
                    # the model is occupied with inference/training.  The SSH transport
                    # above is the authoritative tunnel liveness signal; rebuilding all
                    # four transports here creates a larger outage than the slow request.
                    _log(
                        "model backend is busy; keeping active SSH tunnels for: "
                        + ", ".join(unhealthy_names)
                    )
                next_health_check = now + TUNNEL_HEALTH_CHECK_INTERVAL_SECONDS
            time.sleep(1)
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
            max_connection_seconds=spec.max_connection_seconds,
        )
        for spec in specs
    ]
    run_tunnels(specs)


if __name__ == "__main__":
    main()
