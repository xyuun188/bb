"""Security helpers for dashboard API endpoints."""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, Request, status

from config.settings import settings

_LOCAL_CLIENTS = {"127.0.0.1", "::1", "localhost"}
_WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def _client_host(request: Request) -> str:
    return str(request.client.host if request.client else "").strip().lower()


def _bearer_token(authorization: str | None) -> str:
    value = str(authorization or "").strip()
    if not value.lower().startswith("bearer "):
        return ""
    return value[7:].strip()


def _admin_key_matches(
    authorization: str | None,
    dashboard_admin_key: str | None,
) -> bool:
    configured = str(settings.dashboard_admin_api_key or "").strip()
    if not configured:
        return False

    candidates = [
        _bearer_token(authorization),
        str(dashboard_admin_key or "").strip(),
    ]
    return any(
        candidate and secrets.compare_digest(candidate, configured) for candidate in candidates
    )


def is_dashboard_write_request(request: Request) -> bool:
    """Return True for dashboard API requests that can change state."""
    return request.method.upper() in _WRITE_METHODS and request.url.path.startswith("/api")


def validate_dashboard_write_access(
    request: Request,
    authorization: str | None,
    dashboard_admin_key: str | None,
) -> None:
    """Allow local writes by default; require an admin key for remote writes."""
    if settings.dashboard_admin_api_key:
        if _admin_key_matches(authorization, dashboard_admin_key):
            return
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
