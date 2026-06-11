"""Safe parser for the local remote-server credential file.

The project keeps server connection details in an ignored local text file. This
module centralizes parsing so deployment scripts and monitoring code do not each
carry their own fragile regexes or accidentally print secrets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path

from core.secret_utils import mask_secret

SERVER_INFO_CANDIDATE_NAMES = (
    "服务器资料.txt",
    "server.txt",
    "server_info.txt",
)

SERVER_INFO_GLOBS = (
    "*服务器资料*.txt",
    "*服务器*.txt",
    "*资料*.txt",
    "*server*info*.txt",
    "*server*.txt",
    "*鏈嶅姟鍣*.txt",
)

FIELD_VALUE_RE = r"([^ \t\r\n]+)"
HOST_LABEL_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")
INVALID_IPV4_SHAPE_RE = re.compile(r"^\d+(?:\.\d+){3}$")
USERNAME_RE = re.compile(r"^[A-Za-z0-9._@-]{1,64}$")
MAX_PASSWORD_LENGTH = 4096


def _reject_control_chars(value: str, *, field_name: str) -> None:
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{field_name} must not contain control characters.")


def _normalize_host(value: str) -> str:
    host = str(value or "").strip().rstrip(".")
    if not host:
        raise ValueError("Server info file contains an invalid host.")
    _reject_control_chars(host, field_name="host")
    if any(char.isspace() for char in host) or any(char in host for char in "/\\@:"):
        raise ValueError("Server info file contains an invalid host.")

    try:
        return str(ip_address(host))
    except ValueError:
        if INVALID_IPV4_SHAPE_RE.fullmatch(host):
            raise ValueError("Server info file contains an invalid host.") from None

    if len(host) > 253:
        raise ValueError("Server info file contains an invalid host.")
    labels = host.split(".")
    if not labels or any(not HOST_LABEL_RE.fullmatch(label) for label in labels):
        raise ValueError("Server info file contains an invalid host.")
    return host.lower()


def _normalize_port(value: int | str) -> int:
    try:
        port_int = int(value)
    except (TypeError, ValueError):
        raise ValueError("Server info file contains an invalid port.") from None
    if port_int <= 0 or port_int > 65535:
        raise ValueError("Server info file contains an invalid port.")
    return port_int


def _normalize_username(value: str) -> str:
    username = str(value or "").strip()
    if not username:
        raise ValueError("Server info file contains an invalid username.")
    _reject_control_chars(username, field_name="username")
    if not USERNAME_RE.fullmatch(username):
        raise ValueError("Server info file contains an invalid username.")
    return username


def _normalize_password(value: str) -> str:
    password = str(value or "").strip()
    if not password:
        raise ValueError("Server info file contains an invalid password.")
    _reject_control_chars(password, field_name="password")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise ValueError("Server info file contains an invalid password.")
    return password


@dataclass(frozen=True)
class RemoteServerInfo:
    """Connection details loaded from the local ignored server-info file."""

    host: str
    port: int
    username: str
    password: str
    source_path: Path

    def __post_init__(self) -> None:
        """Normalize and validate connection fields even when built directly."""
        object.__setattr__(self, "host", _normalize_host(self.host))
        object.__setattr__(self, "port", _normalize_port(self.port))
        object.__setattr__(self, "username", _normalize_username(self.username))
        object.__setattr__(self, "password", _normalize_password(self.password))
        if not isinstance(self.source_path, Path):
            object.__setattr__(self, "source_path", Path(str(self.source_path)))

    def as_dict(self) -> dict[str, str | int]:
        """Return a safe public dict shape for diagnostics and API payloads."""
        return {
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "password": mask_secret(self.password),
            "source_path": str(self.source_path.name),
        }

    def connection_kwargs(self) -> dict[str, str | int]:
        """Return plaintext SSH fields for immediate connection only."""
        return {
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "password": self.password,
        }

    def redacted(self) -> dict[str, str | int]:
        """Return safe diagnostics for logs/UI."""
        return self.as_dict()


def _candidate_paths(project_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for name in SERVER_INFO_CANDIDATE_NAMES:
        candidates.append(project_root / name)
    for pattern in SERVER_INFO_GLOBS:
        candidates.extend(project_root.glob(pattern))

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    return deduped


def find_server_info_file(project_root: Path) -> Path:
    """Find the ignored server-info file without inspecting unrelated files."""
    for path in _candidate_paths(project_root):
        if path.exists() and path.is_file():
            return path
    raise FileNotFoundError(
        "Could not find server info file. Expected an ignored local file such as "
        "'服务器资料.txt' in the project root."
    )


def _match_field(text: str, labels: tuple[str, ...], value_pattern: str) -> str | None:
    for label in labels:
        if not label:
            match = re.search(value_pattern, text, re.IGNORECASE)
        else:
            match = re.search(
                rf"(?:^|[\r\n])[ \t]*{label}(?:[ \t]*[:：][ \t]*|[ \t]+)"
                rf"{value_pattern}"
                r"[ \t]*(?:$|[\r\n])",
                text,
                re.IGNORECASE,
            )
        if match:
            return match.group(1).strip()
    return None


def _has_field_label(text: str, labels: tuple[str, ...]) -> bool:
    for label in labels:
        if re.search(rf"(?:^|[\r\n])[ \t]*{label}[ \t]*[:：]?", text, re.IGNORECASE):
            return True
    return False


def parse_remote_server_info(text: str, *, source_path: Path | None = None) -> RemoteServerInfo:
    """Parse server connection details from text.

    Supports the Chinese labels used by the local file plus English aliases, and
    does not expose parsed values in exception messages.
    """
    host_labels = (
        r"公网\s*IP",
        r"公网IP",
        r"主机",
        r"服务器",
        r"host",
        r"ip",
    )
    host = _match_field(text, host_labels, FIELD_VALUE_RE)
    if not host and not _has_field_label(text, host_labels):
        host = _match_field(text, (r"",), r"([0-9]{1,3}(?:\.[0-9]{1,3}){3})")

    port = _match_field(text, (r"端口", r"port"), FIELD_VALUE_RE)
    username = _match_field(
        text,
        (r"账号", r"用户名", r"user(?:name)?", r"login"),
        FIELD_VALUE_RE,
    )
    password = _match_field(text, (r"密码", r"pass(?:word)?"), FIELD_VALUE_RE)

    if not host or not port or not username or not password:
        raise ValueError("Server info file is missing host, port, username, or password.")

    return RemoteServerInfo(
        host=host,
        port=_normalize_port(port),
        username=username,
        password=password,
        source_path=source_path or Path("<memory>"),
    )


def load_remote_server_info(project_root: Path) -> RemoteServerInfo:
    """Load and parse server connection details from the project root."""
    path = find_server_info_file(project_root)
    text = path.read_text(encoding="utf-8", errors="replace")
    return parse_remote_server_info(text, source_path=path)
