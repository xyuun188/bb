"""Durable, truthful 24/72-hour acceptance observation state.

The observer records only facts collected by a caller.  It never advances a
window by synthetic timestamps and never marks a gate passed before the real
elapsed time and all required checks are present.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

CONTINUOUS_OBSERVATION_VERSION = "2026-08-29.continuous-observation.v1"
ALLOWED_WINDOW_HOURS = (24, 72)
MAX_SAMPLES = 2000

_REQUIRED_24H_METRICS = (
    "service_restart_count",
    "dashboard_timeout_storm_count",
    "max_analysis_interval_seconds",
    "duplicate_analysis_count",
    "unexplained_data_collection_timeout_count",
    "unresolved_trade_contract_count",
    "unassigned_fill_count",
    "duplicate_funding_attribution_count",
)
_REQUIRED_72H_METRICS = _REQUIRED_24H_METRICS + (
    "model_tunnel_unresolved_timeout_count",
    "training_state_clear",
    "attribution_mismatch_count",
)


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _parse(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalise_metrics(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    result: dict[str, Any] = {}
    for key in _REQUIRED_24H_METRICS + _REQUIRED_72H_METRICS:
        if key not in source:
            continue
        if key == "max_analysis_interval_seconds":
            result[key] = _float(source.get(key))
        elif key == "training_state_clear":
            result[key] = source.get(key) is True
        else:
            result[key] = _int(source.get(key))
    for key in ("source", "blocked_reason"):
        if source.get(key):
            result[key] = str(source[key])[:300]
    return result


class ContinuousObservationStore:
    """Persist and evaluate one real-time observation window."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def _default(self) -> dict[str, Any]:
        return {
            "version": CONTINUOUS_OBSERVATION_VERSION,
            "status": "not_started",
            "required_hours": 24,
            "window_started_at": None,
            "last_sample_at": None,
            "samples": [],
        }

    def read(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self._default()
        if not isinstance(payload, dict):
            return self._default()
        result = self._default()
        result.update(payload)
        result["samples"] = [
            row for row in result.get("samples", []) if isinstance(row, dict)
        ][-MAX_SAMPLES:]
        return result

    def _write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.path.parent, delete=False
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            temporary = Path(handle.name)
        temporary.replace(self.path)

    def start(self, *, required_hours: int = 24, now: datetime | None = None) -> dict[str, Any]:
        hours = int(required_hours)
        if hours not in ALLOWED_WINDOW_HOURS:
            raise ValueError("required_hours must be 24 or 72")
        payload = self.read()
        if payload.get("window_started_at") and int(payload.get("required_hours") or 24) == hours:
            return self.snapshot(now=now)
        started = now or _now()
        payload.update(
            {
                "version": CONTINUOUS_OBSERVATION_VERSION,
                "status": "observing",
                "required_hours": hours,
                "window_started_at": _iso(started),
                "last_sample_at": None,
                "samples": [],
                "blocked_reason": None,
            }
        )
        self._write(payload)
        return self.snapshot(now=started)

    def record(self, metrics: dict[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
        payload = self.read()
        if not payload.get("window_started_at"):
            raise RuntimeError("observation_window_not_started")
        observed_at = now or _now()
        row = {"observed_at": _iso(observed_at), "metrics": _normalise_metrics(metrics)}
        samples = [item for item in payload.get("samples", []) if isinstance(item, dict)]
        samples.append(row)
        payload["samples"] = samples[-MAX_SAMPLES:]
        payload["last_sample_at"] = row["observed_at"]
        blocked_reason = str(metrics.get("blocked_reason") or "").strip()
        if blocked_reason:
            payload["blocked_reason"] = blocked_reason[:300]
        self._write(payload)
        return self.snapshot(now=observed_at)

    def snapshot(self, *, now: datetime | None = None) -> dict[str, Any]:
        payload = self.read()
        current = now or _now()
        started = _parse(payload.get("window_started_at"))
        required_hours = int(payload.get("required_hours") or 24)
        samples = [item for item in payload.get("samples", []) if isinstance(item, dict)]
        latest_metrics = samples[-1].get("metrics", {}) if samples else {}
        elapsed_hours = (
            max((current - started).total_seconds(), 0.0) / 3600.0 if started else 0.0
        )
        required = _REQUIRED_72H_METRICS if required_hours == 72 else _REQUIRED_24H_METRICS
        missing = [key for key in required if key not in latest_metrics or latest_metrics.get(key) is None]
        failures: list[str] = []
        for key in required:
            value = latest_metrics.get(key)
            if key == "max_analysis_interval_seconds":
                if value is not None and float(value) > 180.0:
                    failures.append(key)
            elif key == "training_state_clear":
                if value is not True:
                    failures.append(key)
            elif value is not None and int(value) != 0:
                failures.append(key)
        blocked_reason = str(payload.get("blocked_reason") or "").strip() or None
        if not started:
            status = "not_started"
        elif blocked_reason:
            status = "blocked"
        elif missing or elapsed_hours < required_hours:
            status = "observing"
        elif failures:
            status = "blocked"
            blocked_reason = "gate_failed:" + ",".join(failures)
        else:
            status = "passed"
        return {
            "version": CONTINUOUS_OBSERVATION_VERSION,
            "status": status,
            "required_hours": required_hours,
            "window_started_at": _iso(started) if started else None,
            "last_sample_at": payload.get("last_sample_at"),
            "elapsed_hours": round(elapsed_hours, 6),
            "sample_count": len(samples),
            "missing_metrics": missing,
            "failed_metrics": failures,
            "blocked_reason": blocked_reason,
            "latest_metrics": latest_metrics,
            "evidence": {
                "real_elapsed_time_required": True,
                "synthetic_time_allowed": False,
                "source": "persisted_observation_samples",
            },
        }

