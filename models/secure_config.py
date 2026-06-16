from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from models.base import Base, TimestampMixin


class SecureSetting(Base, TimestampMixin):
    """Encrypted runtime configuration value.

    The encryption master key is intentionally not stored in this table. It must
    come from the process environment or another system credential provider.
    """

    __tablename__ = "secure_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    ciphertext: Mapped[str] = mapped_column(Text, default="")
    nonce: Mapped[str] = mapped_column(String(80), default="")
    aad: Mapped[str] = mapped_column(String(160), default="")
    is_secret: Mapped[bool] = mapped_column(Boolean, default=True)
    updated_by: Mapped[str] = mapped_column(String(80), default="system")


class SecureSettingAudit(Base, TimestampMixin):
    """Audit trail for secure setting mutations without storing plaintext."""

    __tablename__ = "secure_setting_audit"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(120), index=True)
    action: Mapped[str] = mapped_column(String(40), default="update", index=True)
    actor: Mapped[str] = mapped_column(String(80), default="system")
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    reason: Mapped[str] = mapped_column(Text, default="")
    occurred_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
