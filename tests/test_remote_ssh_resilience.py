from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from core.remote_server_info import RemoteServerInfo
from core.remote_ssh import connect_remote_ssh, exec_remote_command


class FakeChannel:
    def __init__(self, status: int = 0) -> None:
        self.status = status

    def recv_exit_status(self) -> int:
        return self.status


class FakeStream:
    def __init__(self, text: str, channel: FakeChannel | None = None) -> None:
        self.text = text
        self.channel = channel or FakeChannel()
        self.offset = 0

    def read(self, size: int | None = None) -> bytes:
        data = self.text.encode("utf-8")
        if size is None:
            chunk = data[self.offset :]
            self.offset = len(data)
            return chunk
        end = min(self.offset + size, len(data))
        chunk = data[self.offset : end]
        self.offset = end
        return chunk


class FakeTransport:
    def __init__(self) -> None:
        self.keepalive_interval: int | None = None

    def set_keepalive(self, interval: int) -> None:
        self.keepalive_interval = interval


class FakeSSHClient:
    def __init__(self) -> None:
        self.system_loaded = False
        self.policy = None
        self.connect_kwargs: dict[str, object] = {}
        self.connect_calls = 0
        self.close_calls = 0
        self.exec_calls = 0
        self.transport = FakeTransport()

    def load_system_host_keys(self) -> None:
        self.system_loaded = True

    def set_missing_host_key_policy(self, policy) -> None:
        self.policy = policy

    def connect(self, **kwargs) -> None:
        self.connect_calls += 1
        self.connect_kwargs = kwargs

    def close(self) -> None:
        self.close_calls += 1

    def get_transport(self) -> FakeTransport:
        return self.transport

    def exec_command(self, command: str, timeout: int):
        self.exec_calls += 1
        if self.exec_calls == 1:
            import paramiko

            raise paramiko.SSHException("Timeout opening channel.")
        channel = FakeChannel(0)
        return None, FakeStream("ok\n", channel), FakeStream("")


def test_connect_remote_ssh_enables_keepalive_and_reconnect_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reject_policy = object()
    fake_client = FakeSSHClient()
    monkeypatch.setitem(
        sys.modules,
        "paramiko",
        SimpleNamespace(RejectPolicy=lambda: reject_policy, SSHClient=lambda: fake_client),
    )
    info = RemoteServerInfo(
        host="203.0.113.17",
        port=31822,
        username="linux",
        password="secret",
        source_path=Path("<test>"),
    )

    ssh = connect_remote_ssh(
        tmp_path,
        timeout=7,
        banner_timeout=8,
        auth_timeout=9,
        keepalive_interval=17,
        info=info,
    )

    assert ssh is fake_client
    assert fake_client.connect_calls == 1
    assert fake_client.connect_kwargs == {
        "hostname": "203.0.113.17",
        "port": 31822,
        "username": "linux",
        "password": "secret",
        "timeout": 7,
        "banner_timeout": 8,
        "auth_timeout": 9,
    }
    assert fake_client.transport.keepalive_interval == 17
    assert fake_client._bb_project_root == tmp_path
    assert fake_client._bb_connect_kwargs == fake_client.connect_kwargs
    assert fake_client._bb_keepalive_interval == 17


def test_connect_remote_ssh_retries_transient_banner_timeout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class LocalSSHException(Exception):
        pass

    reject_policy = object()
    clients: list[FakeSSHClient] = [FakeSSHClient(), FakeSSHClient()]

    def client_factory() -> FakeSSHClient:
        return clients.pop(0)

    def flaky_connect(self: FakeSSHClient, **kwargs) -> None:
        self.connect_calls += 1
        self.connect_kwargs = kwargs
        if len(clients) == 1:
            raise LocalSSHException("Error reading SSH protocol banner")

    monkeypatch.setitem(
        sys.modules,
        "paramiko",
        SimpleNamespace(
            RejectPolicy=lambda: reject_policy,
            SSHClient=client_factory,
            SSHException=LocalSSHException,
        ),
    )
    monkeypatch.setattr(FakeSSHClient, "connect", flaky_connect)
    monkeypatch.setattr("core.remote_ssh.time.sleep", lambda _seconds: None)
    info = RemoteServerInfo(
        host="203.0.113.17",
        port=31822,
        username="linux",
        password="secret",
        source_path=Path("<test>"),
    )

    ssh = connect_remote_ssh(
        tmp_path,
        timeout=7,
        keepalive_interval=17,
        retry_delay_seconds=0.01,
        info=info,
    )

    assert ssh.connect_calls == 1
    assert ssh.transport.keepalive_interval == 17


def test_exec_remote_command_reconnects_before_retrying_channel_timeout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class LocalSSHException(Exception):
        pass

    reject_policy = object()
    fake_client = FakeSSHClient()
    fake_client._bb_project_root = tmp_path
    fake_client._bb_connect_kwargs = {"hostname": "203.0.113.17"}
    fake_client._bb_keepalive_interval = 19
    monkeypatch.setitem(
        sys.modules,
        "paramiko",
        SimpleNamespace(RejectPolicy=lambda: reject_policy, SSHException=LocalSSHException),
    )

    result = exec_remote_command(fake_client, "echo ok", timeout=12)

    assert result.status == 0
    assert result.stdout == "ok\n"
    assert fake_client.exec_calls == 2
    assert fake_client.close_calls == 1
    assert fake_client.connect_calls == 1
    assert fake_client.connect_kwargs == {"hostname": "203.0.113.17"}
    assert fake_client.system_loaded is True
    assert fake_client.policy is reject_policy
    assert fake_client.transport.keepalive_interval == 19
