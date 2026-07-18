import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.remote_server_info import (
    RemoteServerInfo,
    find_model_server_info_file,
    find_server_info_file,
    parse_remote_server_info,
)
from core.remote_ssh import (
    configure_ssh_host_keys,
    connect_remote_ssh,
    exec_remote_command,
    run_remote_text,
)


def test_parse_remote_server_info_chinese_labels() -> None:
    info = parse_remote_server_info(
        "公网IP：203.0.113.17\n端口: 31822\n账号: linux\n密码: super-secret",
        source_path=Path("服务器资料.txt"),
    )

    assert info.host == "203.0.113.17"
    assert info.access_host == ""
    assert info.port == 31822
    assert info.username == "linux"
    assert info.password == "super-secret"
    assert info.redacted()["password"] == "***"


def test_parse_model_server_info_keeps_access_ip_separate_from_ssh_ip() -> None:
    info = parse_remote_server_info(
        "访问公网IP：103.85.84.147\n"
        "SSH公网IP：203.0.113.18\n"
        "端口：22184\n"
        "账号：linux\n"
        "密码：super-secret",
        source_path=Path("大模型服务器信息.txt"),
    )

    assert info.host == "203.0.113.18"
    assert info.access_host == "103.85.84.147"
    assert info.as_dict()["host"] == "203.0.113.18"
    assert info.as_dict()["access_host"] == "103.85.84.147"
    assert info.connection_kwargs()["host"] == "203.0.113.18"


def test_parse_remote_server_info_english_labels() -> None:
    info = parse_remote_server_info(
        "host: 10.0.0.8\nport: 22\nusername: deploy\npassword: example-password"
    )

    assert info.as_dict() == {
        "host": "10.0.0.8",
        "access_host": "10.0.0.8",
        "port": 22,
        "username": "deploy",
        "password": "***",
        "source_path": "<memory>",
    }
    assert info.connection_kwargs() == {
        "host": "10.0.0.8",
        "port": 22,
        "username": "deploy",
        "password": "example-password",
    }


def test_parse_remote_server_info_platform_labels() -> None:
    info = parse_remote_server_info(
        "平台服务器信息：\nIP：45.207.197.48\n用户名：root\n密码：secret\n端口：22",
        source_path=Path("平台服务器信息.txt"),
    )

    assert info.host == "45.207.197.48"
    assert info.port == 22
    assert info.username == "root"
    assert info.password == "secret"


def test_parse_remote_server_info_rejects_missing_fields() -> None:
    with pytest.raises(ValueError, match="missing"):
        parse_remote_server_info("host: 10.0.0.8\nport: 22")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            "host: 999.0.0.1\nport: 22\nusername: deploy\npassword: secret",
            "invalid host",
        ),
        (
            "host: ssh://203.0.113.17\nport: 22\nusername: deploy\npassword: secret",
            "invalid host",
        ),
        (
            "host: example.com/path\nport: 22\nusername: deploy\npassword: secret",
            "invalid host",
        ),
        (
            "host: example.com\nport: 70000\nusername: deploy\npassword: secret",
            "invalid port",
        ),
        (
            "host: example.com\nport: 22\nusername: deploy;rm\npassword: secret",
            "invalid username",
        ),
        (
            "host: example.com\nport: 22\nusername: deploy\npassword: \n",
            "missing",
        ),
    ],
)
def test_parse_remote_server_info_rejects_unsafe_values(
    text: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_remote_server_info(text)


def test_remote_server_info_validates_direct_construction() -> None:
    with pytest.raises(ValueError, match="invalid host"):
        RemoteServerInfo(
            host="999.0.0.1",
            port=22,
            username="linux",
            password="secret",
            source_path=Path("<test>"),
        )


def test_find_server_info_file_prefers_platform_file(tmp_path) -> None:
    (tmp_path / "大模型服务器信息.txt").write_text(
        "IP：10.0.0.2\n用户名：model\n密码：secret\n端口：22",
        encoding="utf-8",
    )
    platform_path = tmp_path / "平台服务器信息.txt"
    platform_path.write_text(
        "IP：10.0.0.1\n用户名：admin\n密码：secret\n端口：22",
        encoding="utf-8",
    )

    assert find_server_info_file(tmp_path) == platform_path


def test_find_server_info_file_uses_account_info_dir(tmp_path, monkeypatch) -> None:
    account_dir = tmp_path / "accounts"
    account_dir.mkdir()
    platform_path = account_dir / "\u5e73\u53f0\u670d\u52a1\u5668\u4fe1\u606f.txt"
    platform_path.write_text(
        "IP\uff1a10.0.0.1\n\u7528\u6237\u540d\uff1aadmin\n\u5bc6\u7801\uff1asecret\n\u7aef\u53e3\uff1a22",
        encoding="utf-8",
    )
    monkeypatch.setenv("BB_ACCOUNT_INFO_DIR", str(account_dir))

    assert find_server_info_file(tmp_path / "project") == platform_path


def test_find_server_info_file_uses_project_account_info_subdirectory(
    tmp_path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    account_dir = project_root / "\u8d26\u6237\u4fe1\u606f"
    account_dir.mkdir(parents=True)
    platform_path = account_dir / "\u5e73\u53f0\u670d\u52a1\u5668\u4fe1\u606f.txt"
    platform_path.write_text(
        "IP\uff1a10.0.0.1\n\u7528\u6237\u540d\uff1aadmin\n\u5bc6\u7801\uff1asecret\n\u7aef\u53e3\uff1a22",
        encoding="utf-8",
    )
    monkeypatch.delenv("BB_ACCOUNT_INFO_DIR", raising=False)

    assert find_server_info_file(project_root) == platform_path


def test_find_model_server_info_file_prefers_model_file(tmp_path) -> None:
    model_path = tmp_path / "\u5927\u6a21\u578b\u670d\u52a1\u5668\u4fe1\u606f.txt"
    model_path.write_text(
        "IP\uff1a10.0.0.2\n\u7528\u6237\u540d\uff1amodel\n\u5bc6\u7801\uff1asecret\n\u7aef\u53e3\uff1a22",
        encoding="utf-8",
    )
    (tmp_path / "\u5e73\u53f0\u670d\u52a1\u5668\u4fe1\u606f.txt").write_text(
        "IP\uff1a10.0.0.1\n\u7528\u6237\u540d\uff1aadmin\n\u5bc6\u7801\uff1asecret\n\u7aef\u53e3\uff1a22",
        encoding="utf-8",
    )

    assert find_model_server_info_file(tmp_path) == model_path


def test_find_model_server_info_file_prefers_new_model_file(tmp_path) -> None:
    old_model_path = tmp_path / "\u5927\u6a21\u578b\u670d\u52a1\u5668\u4fe1\u606f.txt"
    old_model_path.write_text(
        "IP\uff1a10.0.0.2\n\u7528\u6237\u540d\uff1aold\n\u5bc6\u7801\uff1asecret\n\u7aef\u53e3\uff1a22",
        encoding="utf-8",
    )
    new_model_path = tmp_path / "\u65b0\u5927\u6a21\u578b\u670d\u52a1\u5668\u4fe1\u606f.txt"
    new_model_path.write_text(
        "IP\uff1a10.0.0.3\n\u7528\u6237\u540d\uff1anew\n\u5bc6\u7801\uff1asecret\n\u7aef\u53e3\uff1a62001",
        encoding="utf-8",
    )

    assert find_model_server_info_file(tmp_path) == new_model_path


def test_configure_ssh_host_keys_rejects_unknown_hosts(tmp_path, monkeypatch) -> None:
    known_hosts = tmp_path / ".ssh" / "known_hosts"
    known_hosts.parent.mkdir()
    known_hosts.write_text("example.invalid ssh-ed25519 AAAATEST\n", encoding="utf-8")

    reject_policy = object()
    monkeypatch.setitem(
        sys.modules,
        "paramiko",
        SimpleNamespace(RejectPolicy=lambda: reject_policy),
    )

    class FakeSSH:
        def __init__(self) -> None:
            self.system_loaded = False
            self.loaded_host_keys: list[str] = []
            self.policy = None

        def load_system_host_keys(self) -> None:
            self.system_loaded = True

        def load_host_keys(self, path: str) -> None:
            self.loaded_host_keys.append(path)

        def set_missing_host_key_policy(self, policy) -> None:
            self.policy = policy

    ssh = FakeSSH()
    configure_ssh_host_keys(ssh, tmp_path)

    assert ssh.system_loaded is True
    assert ssh.loaded_host_keys == [str(known_hosts)]
    assert ssh.policy is reject_policy


def test_connect_remote_ssh_uses_shared_timeout_policy(tmp_path, monkeypatch) -> None:
    reject_policy = object()

    class FakeSSHClient:
        def __init__(self) -> None:
            self.system_loaded = False
            self.policy = None
            self.connect_kwargs: dict[str, object] = {}
            self.closed = False

        def load_system_host_keys(self) -> None:
            self.system_loaded = True

        def set_missing_host_key_policy(self, policy) -> None:
            self.policy = policy

        def connect(self, **kwargs) -> None:
            self.connect_kwargs = kwargs

        def close(self) -> None:
            self.closed = True

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
        info=info,
    )

    assert ssh is fake_client
    assert fake_client.system_loaded is True
    assert fake_client.policy is reject_policy
    assert fake_client.connect_kwargs == {
        "hostname": "203.0.113.17",
        "port": 31822,
        "username": "linux",
        "password": "secret",
        "timeout": 7,
        "banner_timeout": 8,
        "auth_timeout": 9,
    }
    assert fake_client.closed is False


class FakeChannel:
    def __init__(self, status: int) -> None:
        self.status = status

    def recv_exit_status(self) -> int:
        return self.status


class FakeStream:
    def __init__(self, text: str, channel: FakeChannel | None = None) -> None:
        self.text = text
        self.channel = channel or FakeChannel(0)
        self.offset = 0
        self.read_calls = 0

    def read(self, size: int | None = None) -> bytes:
        self.read_calls += 1
        data = self.text.encode("utf-8")
        if size is None:
            chunk = data[self.offset :]
            self.offset = len(data)
            return chunk
        end = min(self.offset + size, len(data))
        chunk = data[self.offset : end]
        self.offset = end
        return chunk

    @property
    def fully_read(self) -> bool:
        return self.offset >= len(self.text.encode("utf-8"))


class FakeCommandSSH:
    def __init__(self, status: int, stdout: str, stderr: str) -> None:
        self.status = status
        self.stdout = stdout
        self.stderr = stderr
        self.commands: list[tuple[str, int]] = []
        self.stdout_stream: FakeStream | None = None
        self.stderr_stream: FakeStream | None = None

    def exec_command(self, command: str, timeout: int):
        self.commands.append((command, timeout))
        channel = FakeChannel(self.status)
        self.stdout_stream = FakeStream(self.stdout, channel)
        self.stderr_stream = FakeStream(self.stderr)
        return None, self.stdout_stream, self.stderr_stream


def test_exec_remote_command_returns_decoded_result() -> None:
    ssh = FakeCommandSSH(status=0, stdout="ok\n", stderr="warn\n")

    result = exec_remote_command(ssh, "echo ok", timeout=12)

    assert result.status == 0
    assert result.combined_output == "ok\nwarn\n"
    assert ssh.commands == [("echo ok", 12)]


def test_exec_remote_command_bounds_and_drains_output() -> None:
    ssh = FakeCommandSSH(status=0, stdout="a" * 40, stderr="b" * 40)

    result = exec_remote_command(
        ssh,
        "tail -n 10000 /var/log/app.log",
        timeout=12,
        max_output_chars=16,
    )

    assert result.stdout.startswith("a" * 16)
    assert result.stderr.startswith("b" * 16)
    assert "a" * 30 not in result.stdout
    assert "b" * 30 not in result.stderr
    assert result.stdout_truncated is True
    assert result.stderr_truncated is True
    assert "remote stream truncated at 16 characters" in result.stdout
    assert "remote stream truncated at 16 characters" in result.stderr
    assert ssh.stdout_stream is not None and ssh.stdout_stream.fully_read is True
    assert ssh.stderr_stream is not None and ssh.stderr_stream.fully_read is True


def test_run_remote_text_redacts_failure_output() -> None:
    token = "abcdefghi" + "jklmnopqrst" + "uvwxyz123456"
    ssh = FakeCommandSSH(
        status=1,
        stdout='{"token": "stdout-secret-value"}',
        stderr="password=stderr-secret-value",
    )

    with pytest.raises(RuntimeError) as exc_info:
        run_remote_text(ssh, f"curl -H 'Authorization: Bearer {token}'")

    message = str(exc_info.value)
    assert "abcdefghijklmnopqrstuvwxyz" not in message
    assert "stdout-secret-value" not in message
    assert "stderr-secret-value" not in message
    assert "Authorization: ***" in message
    assert '"token": "***"' in message
    assert "password=***" in message


def test_run_remote_text_caps_success_output() -> None:
    ssh = FakeCommandSSH(status=0, stdout="a" * 40, stderr="b" * 40)

    output = run_remote_text(ssh, "tail -n 10000 /var/log/app.log", max_output_chars=16)

    assert output.startswith("a" * 16)
    assert "a" * 30 not in output
    assert "b" * 30 not in output
    assert "remote output truncated at 16 characters" in output


class FakeTransientSSH:
    def __init__(self) -> None:
        self.calls = 0

    def exec_command(self, command: str, timeout: int):
        self.calls += 1
        if self.calls == 1:
            import paramiko

            raise paramiko.SSHException("Timeout opening channel.")
        channel = FakeChannel(0)
        return None, FakeStream("ok\n", channel), FakeStream("")


def test_exec_remote_command_retries_transient_channel_open_error(monkeypatch) -> None:
    class LocalSSHException(Exception):
        pass

    fake_ssh = FakeTransientSSH()
    monkeypatch.setitem(sys.modules, "paramiko", SimpleNamespace(SSHException=LocalSSHException))
    result = exec_remote_command(fake_ssh, "echo ok", timeout=12)

    assert result.status == 0
    assert fake_ssh.calls == 2
    assert result.stdout == "ok\n"


def test_exec_remote_command_does_not_retry_non_transient_errors(monkeypatch) -> None:
    class LocalSSHException(Exception):
        pass

    class FakeFailureSSH:
        def __init__(self) -> None:
            self.calls = 0

        def exec_command(self, command: str, timeout: int):
            self.calls += 1
            raise LocalSSHException("some other ssh error")

    monkeypatch.setitem(sys.modules, "paramiko", SimpleNamespace(SSHException=LocalSSHException))
    fake_ssh = FakeFailureSSH()
    with pytest.raises(LocalSSHException):
        exec_remote_command(fake_ssh, "echo ok", timeout=12)
    assert fake_ssh.calls == 1
