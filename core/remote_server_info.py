"""Safe parser for ignored local remote-server credential files.

Deployment and monitoring scripts read server connection details from ignored
text files in the project root. This module keeps parsing strict and makes sure
secrets are never included in diagnostics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path

from core.secret_utils import mask_secret

SERVER_INFO_CANDIDATE_NAMES = (
    "\u5e73\u53f0\u670d\u52a1\u5668\u4fe1\u606f.txt",  # platform server info
    "\u670d\u52a1\u5668\u8d44\u6599.txt",  # server data
    "\u670d\u52a1\u5668\u4fe1\u606f.txt",  # server info
    "server.txt",
    "server_info.txt",
)

SERVER_INFO_GLOBS = (
    "*\u5e73\u53f0\u670d\u52a1\u5668\u4fe1\u606f*.txt",
    "*\u670d\u52a1\u5668\u8d44\u6599*.txt",
    "*\u670d\u52a1\u5668\u4fe1\u606f*.txt",
    "*\u670d\u52a1\u5668*.txt",
    "*\u8d44\u6599*.txt",
    "*server*info*.txt",
    "*server*.txt",
)

FIELD_VALUE_RE = r"([^ \t\r\n]+)"
HOST_LABEL_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")
INVALID_IPV4_SHAPE_RE = re.compile(r"^\d+(?:\.\d+){3}$")
USERNAME_RE = re.compile(r"^[A-Za-z0-9._@-]{1,64}$")
MAX_PASSWORD_LENGTH = 4096

HOST_KEYS = {"host", "ip", "hostname", "server"}
PORT_KEYS = {"port", "ssh_port"}
USERNAME_KEYS = {"user", "username", "login", "account"}
PASSWORD_KEYS = {"pass", "password"}

CHINESE_HOST_LABELS = {
    "\u516c\u7f51ip",
    "\u516c\u7f51 ip",
    "\u4e3b\u673a",
    "\u670d\u52a1\u5668",
    "\u5e73\u53f0\u670d\u52a1\u5668",
}
CHINESE_PORT_LABELS = {"\u7aef\u53e3"}
CHINESE_USERNAME_LABELS = {"\u7528\u6237\u540d", "\u8d26\u53f7", "\u7528\u6237"}
CHINESE_PASSWORD_LABELS = {"\u5bc6\u7801", "\u53e3\u4ee4"}

KEY_VALUE_RE = re.compile(r"^\s*(?P<key>[^:=\uff1a]+?)\s*(?:[:=]|\uff1a)\s*(?P<value>.*?)\s*$")
IPV4_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")


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
    """Connection details loaded from an ignored server-info file."""

    host: str
    port: int
    username: str
    password: str
    source_path: Path

    def __post_init__(self) -> None:
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
    """Find an ignored server-info file without inspecting unrelated files."""
    for path in _candidate_paths(project_root):
        if path.exists() and path.is_file():
            return path
    raise FileNotFoundError(
        "Could not find server info file. Expected an ignored local server info file "
        "in the project root."
    )


def _field_kind(key: str) -> str | None:
    normalized = re.sub(r"\s+", " ", key.strip().lower())
    if normalized in HOST_KEYS or normalized in CHINESE_HOST_LABELS:
        return "host"
    if normalized in PORT_KEYS or normalized in CHINESE_PORT_LABELS:
        return "port"
    if normalized in USERNAME_KEYS or normalized in CHINESE_USERNAME_LABELS:
        return "username"
    if normalized in PASSWORD_KEYS or normalized in CHINESE_PASSWORD_LABELS:
        return "password"
    if "ip" == normalized or normalized.endswith(" ip"):
        return "host"
    return None


def _parse_key_value_lines(text: str) -> tuple[list[tuple[str, str]], set[str]]:
    pairs: list[tuple[str, str]] = []
    explicit_empty_fields: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = KEY_VALUE_RE.match(line)
        if not match:
            continue
        value = match.group("value").strip()
        if not value:
            kind = _field_kind(match.group("key"))
            if kind:
                explicit_empty_fields.add(kind)
            continue
        pairs.append((match.group("key").strip(), value))
    return pairs, explicit_empty_fields


def _fallback_ordered_fields(pairs: list[tuple[str, str]]) -> dict[str, str]:
    """Handle legacy mojibake labels by using value shape and line order."""
    result: dict[str, str] = {}
    remaining_values = [value for _key, value in pairs]

    for value in remaining_values:
        if "host" not in result and IPV4_RE.fullmatch(value):
            result["host"] = value
            continue
        if "port" not in result:
            try:
                port = _normalize_port(value)
            except ValueError:
                pass
            else:
                result["port"] = str(port)
                continue
        if "username" not in result:
            try:
                result["username"] = _normalize_username(value)
            except ValueError:
                pass
            else:
                continue
        if "password" not in result:
            result["password"] = value
    return result


def parse_remote_server_info(text: str, *, source_path: Path | None = None) -> RemoteServerInfo:
    """Parse server connection details from text without leaking values."""
    pairs, explicit_empty_fields = _parse_key_value_lines(text)
    if explicit_empty_fields:
        raise ValueError("Server info file is missing host, port, username, or password.")
    values: dict[str, str] = {}
    for key, value in pairs:
        kind = _field_kind(key)
        if kind and kind not in values:
            values[kind] = value

    if "host" not in values:
        match = IPV4_RE.search(text)
        if match:
            values["host"] = match.group(0)

    fallback = _fallback_ordered_fields(pairs)
    for key, value in fallback.items():
        values.setdefault(key, value)

    if not all(values.get(field) for field in ("host", "port", "username", "password")):
        raise ValueError("Server info file is missing host, port, username, or password.")

    return RemoteServerInfo(
        host=values["host"],
        port=_normalize_port(values["port"]),
        username=values["username"],
        password=values["password"],
        source_path=source_path or Path("<memory>"),
    )


def load_remote_server_info(project_root: Path) -> RemoteServerInfo:
    """Load and parse server connection details from the project root."""
    path = find_server_info_file(project_root)
    text = path.read_text(encoding="utf-8", errors="replace")
    return parse_remote_server_info(text, source_path=path)
