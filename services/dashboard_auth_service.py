from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from core.safe_output import safe_error_text
from models.dashboard_auth import DashboardUser
from web_dashboard.api.security import hash_dashboard_password, verify_dashboard_password

USERNAME_RE = re.compile(r"^[a-zA-Z0-9_.-]{3,40}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
DEFAULT_DASHBOARD_USERNAME = "operator"


class DashboardAuthServiceError(RuntimeError):
    """Raised for Dashboard account management failures."""


@dataclass(frozen=True, slots=True)
class PublicDashboardUser:
    username: str
    email: str
    role: str
    is_active: bool
    password_configured: bool
    last_login_at: str | None
    created_at: str | None
    updated_at: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "username": self.username,
            "email": self.email,
            "masked_email": mask_email(self.email),
            "role": self.role,
            "is_active": self.is_active,
            "password_configured": self.password_configured,
            "last_login_at": self.last_login_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True, slots=True)
class AuthenticatedDashboardUser:
    username: str
    role: str
    email: str


def normalize_username(value: str) -> str:
    username = str(value or "").strip().lower()
    if not USERNAME_RE.fullmatch(username):
        raise ValueError(
            "username must be 3-40 characters: letters, numbers, dot, underscore, dash"
        )
    return username


def normalize_email(value: str | None) -> str:
    email = str(value or "").strip().lower()
    if not email:
        return ""
    if not EMAIL_RE.fullmatch(email):
        raise ValueError("email address is invalid")
    return email


def validate_dashboard_password(value: str) -> str:
    password = str(value or "")
    if len(password) < 10:
        raise ValueError("password must be at least 10 characters")
    if len(password) > 256:
        raise ValueError("password is too long")
    return password


def configured_dashboard_username() -> str:
    """Return configured login username without exposing an admin default."""
    configured = str(settings.dashboard_auth_username or "").strip()
    if configured:
        return normalize_username(configured)
    return DEFAULT_DASHBOARD_USERNAME


def mask_email(value: str | None) -> str:
    email = normalize_email(value)
    if not email or "@" not in email:
        return ""
    name, domain = email.split("@", 1)
    if len(name) <= 2:
        masked_name = name[:1] + "*"
    else:
        masked_name = name[:1] + "*" * min(len(name) - 2, 6) + name[-1:]
    return f"{masked_name}@{domain}"


async def ensure_seed_admin_user(session: AsyncSession) -> None:
    result = await session.execute(select(func.count(DashboardUser.id)))
    if int(result.scalar_one() or 0) > 0:
        return
    password_hash = str(settings.dashboard_auth_password_hash or "").strip()
    if not password_hash:
        return
    username = configured_dashboard_username()
    row = DashboardUser(
        username=username,
        email=normalize_email(getattr(settings, "dashboard_auth_email", "")) or None,
        password_hash=password_hash,
        role="admin",
        is_active=True,
    )
    session.add(row)
    await session.flush()


async def has_dashboard_users(session: AsyncSession) -> bool:
    try:
        result = await session.execute(select(func.count(DashboardUser.id)))
        return int(result.scalar_one() or 0) > 0
    except SQLAlchemyError as exc:
        raise DashboardAuthServiceError(safe_error_text(exc)) from exc


async def authenticate_dashboard_user(
    session: AsyncSession,
    username: str,
    password: str,
) -> AuthenticatedDashboardUser | None:
    try:
        normalized = normalize_username(username)
        await ensure_seed_admin_user(session)
        row = await _get_user_row(session, normalized)
        if row is None or not row.is_active:
            return None
        if not verify_dashboard_password(password, row.password_hash):
            return None
        row.last_login_at = datetime.now(UTC)
        await session.flush()
        return AuthenticatedDashboardUser(
            username=row.username,
            role=row.role or "admin",
            email=row.email or "",
        )
    except (SQLAlchemyError, ValueError) as exc:
        raise DashboardAuthServiceError(safe_error_text(exc)) from exc


async def list_dashboard_users(session: AsyncSession) -> list[PublicDashboardUser]:
    await ensure_seed_admin_user(session)
    result = await session.execute(select(DashboardUser).order_by(DashboardUser.username.asc()))
    return [_public_user(row) for row in result.scalars().all()]


async def get_dashboard_user(session: AsyncSession, username: str) -> PublicDashboardUser | None:
    await ensure_seed_admin_user(session)
    row = await _get_user_row(session, normalize_username(username))
    return _public_user(row) if row is not None else None


async def create_dashboard_user(
    session: AsyncSession,
    *,
    username: str,
    email: str = "",
    password: str,
    role: str = "admin",
    is_active: bool = True,
) -> PublicDashboardUser:
    normalized = normalize_username(username)
    password_text = validate_dashboard_password(password)
    email_text = normalize_email(email)
    existing = await _get_user_row(session, normalized)
    if existing is not None:
        raise ValueError("dashboard user already exists")
    await _assert_email_available(session, email_text, exclude_username="")
    row = DashboardUser(
        username=normalized,
        email=email_text or None,
        password_hash=hash_dashboard_password(password_text),
        role=_normalize_role(role),
        is_active=bool(is_active),
    )
    session.add(row)
    await session.flush()
    return _public_user(row)


async def update_dashboard_user(
    session: AsyncSession,
    *,
    username: str,
    email: str | None = None,
    password: str | None = None,
    role: str | None = None,
    is_active: bool | None = None,
) -> PublicDashboardUser:
    normalized = normalize_username(username)
    row = await _get_user_row(session, normalized)
    if row is None:
        raise ValueError("dashboard user not found")
    if email is not None:
        email_text = normalize_email(email)
        await _assert_email_available(session, email_text, exclude_username=normalized)
        row.email = email_text or None
    if password:
        row.password_hash = hash_dashboard_password(validate_dashboard_password(password))
    if role is not None:
        row.role = _normalize_role(role)
    if is_active is not None:
        if not bool(is_active):
            await _assert_other_active_user_exists(session, normalized)
        row.is_active = bool(is_active)
    await session.flush()
    return _public_user(row)


async def update_current_dashboard_account(
    session: AsyncSession,
    *,
    current_username: str,
    username: str,
    email: str,
) -> PublicDashboardUser:
    current = normalize_username(current_username)
    next_username = normalize_username(username)
    email_text = normalize_email(email)
    await ensure_seed_admin_user(session)
    row = await _get_user_row(session, current)
    if row is None:
        password_hash = str(settings.dashboard_auth_password_hash or "").strip()
        if not password_hash:
            raise ValueError("current dashboard user is not initialized")
        row = DashboardUser(
            username=current,
            email=normalize_email(getattr(settings, "dashboard_auth_email", "")) or None,
            password_hash=password_hash,
            role="admin",
            is_active=True,
        )
        session.add(row)
        await session.flush()
    if next_username != current and await _get_user_row(session, next_username) is not None:
        raise ValueError("dashboard username already exists")
    await _assert_email_available(session, email_text, exclude_username=current)
    row.username = next_username
    row.email = email_text or None
    await session.flush()
    return _public_user(row)


async def change_dashboard_user_password(
    session: AsyncSession,
    *,
    username: str,
    new_password: str,
    current_password: str = "",
    require_current_password: bool = True,
) -> PublicDashboardUser:
    normalized = normalize_username(username)
    row = await _get_user_row(session, normalized)
    if row is None:
        raise ValueError("dashboard user not found")
    if require_current_password:
        if not current_password or not verify_dashboard_password(
            current_password, row.password_hash
        ):
            raise ValueError("current password is required")
    row.password_hash = hash_dashboard_password(validate_dashboard_password(new_password))
    await session.flush()
    return _public_user(row)


async def deactivate_dashboard_user(session: AsyncSession, username: str) -> PublicDashboardUser:
    normalized = normalize_username(username)
    row = await _get_user_row(session, normalized)
    if row is None:
        raise ValueError("dashboard user not found")
    await _assert_other_active_user_exists(session, normalized)
    row.is_active = False
    await session.flush()
    return _public_user(row)


async def delete_dashboard_user(session: AsyncSession, username: str) -> PublicDashboardUser:
    normalized = normalize_username(username)
    row = await _get_user_row(session, normalized)
    if row is None:
        raise ValueError("dashboard user not found")
    if row.is_active:
        await _assert_other_active_user_exists(session, normalized)
    public = _public_user(row)
    await session.delete(row)
    await session.flush()
    return public


async def _get_user_row(session: AsyncSession, username: str) -> DashboardUser | None:
    result = await session.execute(select(DashboardUser).where(DashboardUser.username == username))
    return result.scalar_one_or_none()


async def _assert_other_active_user_exists(session: AsyncSession, username: str) -> None:
    result = await session.execute(
        select(func.count(DashboardUser.id)).where(
            DashboardUser.is_active.is_(True),
            DashboardUser.username != username,
        )
    )
    if int(result.scalar_one() or 0) <= 0:
        raise ValueError("cannot disable or delete the last active dashboard user")


async def _assert_email_available(
    session: AsyncSession,
    email: str,
    *,
    exclude_username: str,
) -> None:
    if not email:
        return
    stmt = select(DashboardUser).where(DashboardUser.email == email)
    if exclude_username:
        stmt = stmt.where(DashboardUser.username != exclude_username)
    result = await session.execute(stmt)
    if result.scalar_one_or_none() is not None:
        raise ValueError("dashboard email already exists")


def _iso_from_loaded(row: DashboardUser, name: str) -> str | None:
    value = row.__dict__.get(name)
    return value.isoformat() if value else None


def _public_user(row: DashboardUser) -> PublicDashboardUser:
    return PublicDashboardUser(
        username=row.username,
        email=row.email or "",
        role=row.role or "admin",
        is_active=bool(row.is_active),
        password_configured=bool(row.password_hash),
        last_login_at=_iso_from_loaded(row, "last_login_at"),
        created_at=_iso_from_loaded(row, "created_at"),
        updated_at=_iso_from_loaded(row, "updated_at"),
    )


def _normalize_role(value: str) -> str:
    role = str(value or "admin").strip().lower()
    if role != "admin":
        raise ValueError("only admin dashboard accounts are supported currently")
    return role
