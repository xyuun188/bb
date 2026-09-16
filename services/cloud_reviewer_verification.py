"""Durable identity verification state for the optional cloud reviewer."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config.settings import settings


def _verification_path() -> Path:
    return Path(settings.data_dir) / "cloud_reviewer_verification.json"


def cloud_reviewer_route_fingerprint(
    api_base: str,
    model: str,
    revision: str,
    api_key: str,
) -> str:
    """Fingerprint an exact route without persisting the credential itself."""

    payload = {
        "api_base": str(api_base or "").strip().rstrip("/"),
        "model": str(model or "").strip(),
        "revision": str(revision or "").strip(),
        "api_key_sha256": hashlib.sha256(str(api_key or "").encode("utf-8")).hexdigest(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_cloud_reviewer_verification(
    *,
    api_base: str,
    model: str,
    revision: str,
    api_key: str,
    latency_ms: float,
    provider: str,
    identity_source: str,
) -> dict[str, Any]:
    """Atomically persist a successful exact-route connectivity verification."""

    payload: dict[str, Any] = {
        "schema_version": "2026-09-16.v1",
        "connection_verified": True,
        "status": "ready",
        "verified_at": datetime.now(UTC).isoformat(),
        "latency_ms": round(float(latency_ms), 1),
        "provider": str(provider or "").strip() or None,
        "model": str(model or "").strip() or None,
        "revision": str(revision or "").strip() or None,
        "identity_source": str(identity_source or "").strip() or "chat_probe",
        "route_fingerprint": cloud_reviewer_route_fingerprint(
            api_base,
            model,
            revision,
            api_key,
        ),
    }
    path = _verification_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(path)
    return payload


def load_cloud_reviewer_verification(
    *,
    api_base: str,
    model: str,
    revision: str,
    api_key: str,
) -> dict[str, Any]:
    """Return verification only when it belongs to the current exact route."""

    try:
        payload = json.loads(_verification_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    expected = cloud_reviewer_route_fingerprint(api_base, model, revision, api_key)
    matches = bool(payload.get("route_fingerprint") == expected)
    verified = bool(matches and payload.get("connection_verified") is True)
    return {
        "connection_verified": verified,
        "status": payload.get("status") if matches else "not_verified",
        "verified_at": payload.get("verified_at") if matches else None,
        "latency_ms": payload.get("latency_ms") if matches else None,
        "provider": payload.get("provider") if matches else None,
        "identity_source": payload.get("identity_source") if matches else None,
    }
