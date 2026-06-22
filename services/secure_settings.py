from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.secret_utils import mask_secret, secret_fingerprint
from models.secure_config import SecureSetting, SecureSettingAudit

MASTER_KEY_ENV = "BB_SECURE_SETTINGS_KEY"
SECURE_KEY_RE = re.compile(r"^[a-z][a-z0-9_.:-]{1,119}$")


class SecureSettingsError(RuntimeError):
    """Raised when secure settings cannot be encrypted or decrypted."""


@dataclass(frozen=True, slots=True)
class PublicSecureSetting:
    key: str
    configured: bool
    masked_value: str
    fingerprint: str
    updated_by: str
    updated_at: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "configured": self.configured,
            "masked_value": self.masked_value,
            "fingerprint": self.fingerprint,
            "updated_by": self.updated_by,
            "updated_at": self.updated_at,
        }


def normalize_secure_key(value: str) -> str:
    key = str(value or "").strip().lower()
    if not SECURE_KEY_RE.fullmatch(key):
        raise ValueError("secure setting key must match ^[a-z][a-z0-9_.:-]{1,119}$")
    return key


class SecureSettingsCipher:
    """AES-GCM cipher using a master key supplied outside the database."""

    def __init__(self, master_key: bytes | None = None) -> None:
        self._master_key = master_key

    @classmethod
    def from_environment(cls) -> SecureSettingsCipher:
        return cls(_load_master_key_from_env())

    def _aesgcm(self) -> AESGCM:
        key = self._master_key
        if not key:
            key = _load_master_key_from_env()
        return AESGCM(key)

    def encrypt(self, *, key: str, plaintext: str) -> tuple[str, str, str]:
        aad = normalize_secure_key(key)
        nonce = os.urandom(12)
        ciphertext = self._aesgcm().encrypt(nonce, plaintext.encode("utf-8"), aad.encode("utf-8"))
        return (
            base64.urlsafe_b64encode(ciphertext).decode("ascii"),
            base64.urlsafe_b64encode(nonce).decode("ascii"),
            aad,
        )

    def decrypt(self, *, key: str, ciphertext: str, nonce: str, aad: str) -> str:
        normalized = normalize_secure_key(key)
        aad_value = aad or normalized
        if aad_value != normalized:
            raise SecureSettingsError("secure setting aad does not match key")
        try:
            raw_ciphertext = base64.urlsafe_b64decode(ciphertext.encode("ascii"))
            raw_nonce = base64.urlsafe_b64decode(nonce.encode("ascii"))
            plaintext = self._aesgcm().decrypt(raw_nonce, raw_ciphertext, aad_value.encode("utf-8"))
        except Exception as exc:
            raise SecureSettingsError("secure setting decrypt failed") from exc
        return plaintext.decode("utf-8")


def _load_master_key_from_env() -> bytes:
    value = os.environ.get(MASTER_KEY_ENV, "").strip()
    if not value:
        raise SecureSettingsError(f"{MASTER_KEY_ENV} is required for encrypted settings")
    try:
        if value.startswith("base64:"):
            key = base64.b64decode(value.split(":", 1)[1], validate=True)
        else:
            key = bytes.fromhex(value)
    except ValueError as exc:
        raise SecureSettingsError(
            f"{MASTER_KEY_ENV} must be hex or base64:<base64> encoded"
        ) from exc
    if len(key) not in {16, 24, 32}:
        raise SecureSettingsError(f"{MASTER_KEY_ENV} must decode to 16, 24, or 32 bytes")
    return key


class SecureSettingsService:
    """Store and retrieve encrypted runtime settings."""

    def __init__(self, session: AsyncSession, cipher: SecureSettingsCipher | None = None) -> None:
        self.session = session
        self.cipher = cipher or SecureSettingsCipher.from_environment()

    async def set_secret(
        self, key: str, value: str, *, actor: str = "system"
    ) -> PublicSecureSetting:
        normalized = normalize_secure_key(key)
        text = str(value or "")
        ciphertext, nonce, aad = self.cipher.encrypt(key=normalized, plaintext=text)
        result = await self.session.execute(
            select(SecureSetting).where(SecureSetting.key == normalized)
        )
        row = result.scalar_one_or_none()
        if row is None:
            row = SecureSetting(key=normalized)
            self.session.add(row)
        row.ciphertext = ciphertext
        row.nonce = nonce
        row.aad = aad
        row.is_secret = True
        row.updated_by = actor or "system"
        self.session.add(
            SecureSettingAudit(
                key=normalized,
                action="update",
                actor=actor or "system",
                success=True,
                reason="encrypted update",
                occurred_at=datetime.now(UTC),
            )
        )
        await self.session.flush()
        await self.session.refresh(row)
        return self.public_view(row, plaintext=text)

    async def get_secret(self, key: str) -> str | None:
        row = await self._get_row(key)
        if row is None:
            return None
        return self.cipher.decrypt(
            key=row.key,
            ciphertext=row.ciphertext,
            nonce=row.nonce,
            aad=row.aad,
        )

    async def public(self, key: str) -> PublicSecureSetting | None:
        row = await self._get_row(key)
        if row is None:
            return None
        return self.public_view(row)

    async def list_public(self, prefix: str | None = None) -> list[PublicSecureSetting]:
        stmt = select(SecureSetting).order_by(SecureSetting.key.asc())
        if prefix:
            stmt = stmt.where(SecureSetting.key.like(f"{normalize_secure_key(prefix)}%"))
        result = await self.session.execute(stmt)
        return [self.public_view(row) for row in result.scalars().all()]

    async def _get_row(self, key: str) -> SecureSetting | None:
        normalized = normalize_secure_key(key)
        result = await self.session.execute(
            select(SecureSetting).where(SecureSetting.key == normalized)
        )
        return result.scalar_one_or_none()

    def public_view(
        self, row: SecureSetting, *, plaintext: str | None = None
    ) -> PublicSecureSetting:
        if plaintext is None:
            try:
                plaintext = self.cipher.decrypt(
                    key=row.key,
                    ciphertext=row.ciphertext,
                    nonce=row.nonce,
                    aad=row.aad,
                )
            except SecureSettingsError:
                plaintext = ""
        updated_at = row.updated_at or row.created_at
        return PublicSecureSetting(
            key=row.key,
            configured=bool(row.ciphertext),
            masked_value=mask_secret(plaintext, show_last=4) if plaintext else "***",
            fingerprint=secret_fingerprint(plaintext) if plaintext else "",
            updated_by=row.updated_by,
            updated_at=updated_at.isoformat() if updated_at else None,
        )
