from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from fastapi import Header, HTTPException, Request, Response, status

from config.settings import settings
from core.secret_utils import secret_fingerprint

_LOCAL_CLIENTS = {"127.0.0.1", "::1", "localhost"}
_LOCAL_DASHBOARD_HOSTS = {"127.0.0.1", "::1", "localhost"}
_WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
SESSION_COOKIE_NAME = "bb_dashboard_session"
SESSION_SEPARATOR = "."
PBKDF2_ITERATIONS = 210_000


@dataclass(frozen=True, slots=True)
class DashboardAuthContext:
    username: str
    issued_at: int
    expires_at: int


def _direct_client_host(request: Any) -> str:
    return str(request.client.host if request.client else "").strip().lower()


def _client_host(request: Any) -> str:
    direct_host = _direct_client_host(request)
    if direct_host in _LOCAL_CLIENTS:
        forwarded_for = str(request.headers.get("x-forwarded-for", "") or "").split(",", 1)[0]
        forwarded_host = forwarded_for.strip().lower()
        if forwarded_host:
            return forwarded_host
    return direct_host


def _dashboard_host_exposes_network() -> bool:
    host = str(settings.dashboard_host or "").strip().lower()
    return host not in _LOCAL_DASHBOARD_HOSTS


def _bearer_token(authorization: str | None) -> str:
    value = str(authorization or "").strip()
    if not value.lower().startswith("bearer "):
        return ""
    return value[7:].strip()


def _pad_base64(value: str) -> str:
    return value + "=" * (-len(value) % 4)


def _session_secret() -> bytes:
    value = str(settings.dashboard_session_secret or "").strip()
    if value:
        return value.encode("utf-8")
    fallback = str(settings.dashboard_admin_api_key or settings.dashboard_auth_password_hash or "").strip()
    if fallback:
        return hashlib.sha256(fallback.encode("utf-8")).digest()
    return hashlib.sha256(b"bb-dashboard-session-fallback").digest()


def _sign_payload(payload: str) -> str:
    mac = hmac.new(_session_secret(), payload.encode("utf-8"), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(mac).decode("ascii").rstrip("=")


def _encode_session(context: DashboardAuthContext) -> str:
    payload = f"{context.username}|{context.issued_at}|{context.expires_at}"
    signature = _sign_payload(payload)
    encoded = base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")
    return f"{encoded}{SESSION_SEPARATOR}{signature}"


def _decode_session(token: str) -> DashboardAuthContext | None:
    raw = str(token or "").strip()
    if SESSION_SEPARATOR not in raw:
        return None
    encoded, signature = raw.rsplit(SESSION_SEPARATOR, 1)
    try:
        payload = base64.urlsafe_b64decode(_pad_base64(encoded)).decode("utf-8")
    except Exception:
        return None
    if _sign_payload(payload) != signature:
        return None
    parts = payload.split("|", 2)
    if len(parts) != 3:
        return None
    username, issued_text, expires_text = parts
    try:
        issued_at = int(issued_text)
        expires_at = int(expires_text)
    except ValueError:
        return None
    if expires_at <= int(datetime.now(UTC).timestamp()):
        return None
    return DashboardAuthContext(username=username, issued_at=issued_at, expires_at=expires_at)


def hash_dashboard_password(password: str, *, salt: bytes | None = None) -> str:
    salt_bytes = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        str(password or "").encode("utf-8"),
        salt_bytes,
        PBKDF2_ITERATIONS,
    )
    salt_text = base64.urlsafe_b64encode(salt_bytes).decode("ascii").rstrip("=")
    digest_text = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt_text}${digest_text}"


def verify_dashboard_password(password: str, password_hash: str) -> bool:
    value = str(password_hash or "").strip()
    if not value.startswith("pbkdf2_sha256$"):
        return False
    try:
        _scheme, iterations_text, salt_text, digest_text = value.split("$", 3)
        iterations = int(iterations_text)
        salt = base64.urlsafe_b64decode(_pad_base64(salt_text))
        expected = base64.urlsafe_b64decode(_pad_base64(digest_text))
    except Exception:
        return False
    actual = hashlib.pbkdf2_hmac(
        "sha256",
        str(password or "").encode("utf-8"),
        salt,
        iterations,
    )
    return hmac.compare_digest(actual, expected)


def create_dashboard_session(username: str) -> str:
    now = int(datetime.now(UTC).timestamp())
    expires = now + max(int(settings.dashboard_session_ttl_seconds or 43200), 600)
    return _encode_session(DashboardAuthContext(username=username, issued_at=now, expires_at=expires))


def dashboard_session_from_token(token: str | None) -> DashboardAuthContext | None:
    return _decode_session(token or "")


def read_dashboard_session(request: Request) -> DashboardAuthContext | None:
    token = request.cookies.get(SESSION_COOKIE_NAME) or request.headers.get("x-dashboard-session")
    return dashboard_session_from_token(token)


def dashboard_admin_key_matches(
    authorization: str | None,
    dashboard_admin_key: str | None,
) -> bool:
    configured = str(settings.dashboard_admin_api_key or "").strip()
    if not configured:
        return False
    candidates = [_bearer_token(authorization), str(dashboard_admin_key or "").strip()]
    return any(
        candidate and secrets.compare_digest(candidate, configured) for candidate in candidates
    )


def dashboard_login_required(request: Request) -> bool:
    return bool(
        settings.dashboard_auth_enabled
        or _dashboard_host_exposes_network()
        or _client_host(request) not in _LOCAL_CLIENTS
    )


def ensure_dashboard_login(request: Request) -> DashboardAuthContext | None:
    if not dashboard_login_required(request):
        return None
    session = read_dashboard_session(request)
    if session is not None:
        return session
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Dashboard login required.",
    )


def is_dashboard_write_request(request: Request) -> bool:
    """Return True for dashboard API requests that can change state."""
    return request.method.upper() in _WRITE_METHODS and request.url.path.startswith("/api")


def validate_dashboard_write_access(
    request: Request,
    authorization: str | None,
    dashboard_admin_key: str | None,
) -> None:
    """Protect state-changing dashboard calls with API key, session, or local access."""
    if dashboard_admin_key_matches(authorization, dashboard_admin_key):
        return
    session = ensure_dashboard_login(request)
    if session is not None and settings.dashboard_auth_enabled:
        return
    if settings.dashboard_admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Dashboard admin API key is required for write access.",
        )
    if _client_host(request) in _LOCAL_CLIENTS:
        return
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Remote dashboard write access requires DASHBOARD_ADMIN_API_KEY.",
    )


async def require_dashboard_write_access(
    request: Request,
    authorization: str | None = Header(default=None),
    x_dashboard_admin_key: str | None = Header(default=None),
) -> None:
    """FastAPI dependency for state-changing dashboard endpoints."""
    validate_dashboard_write_access(request, authorization, x_dashboard_admin_key)


async def require_destructive_dashboard_confirmation(
    request: Request,
    x_dashboard_confirm: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
    x_dashboard_admin_key: str | None = Header(default=None),
) -> None:
    """Protect irreversible dashboard actions without breaking local read-only use."""
    if str(x_dashboard_confirm or "").strip().lower() != "delete-records":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Destructive action requires X-Dashboard-Confirm: delete-records.",
        )
    await require_dashboard_write_access(
        request,
        authorization=authorization,
        x_dashboard_admin_key=x_dashboard_admin_key,
    )


def login_response_cookie(response: Response, session_token: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_token,
        max_age=max(int(settings.dashboard_session_ttl_seconds or 43200), 600),
        httponly=True,
        secure=bool(settings.dashboard_auth_cookie_secure),
        samesite="lax",
        path="/",
    )


def logout_response_cookie(response: Response) -> None:
    response.delete_cookie(key=SESSION_COOKIE_NAME, path="/")


def dashboard_password_hash_fingerprint() -> str:
    value = str(settings.dashboard_auth_password_hash or "").strip()
    return secret_fingerprint(value) if value else ""
