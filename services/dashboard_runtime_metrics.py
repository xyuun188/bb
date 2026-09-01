"""Small durable counters used by the dashboard health observer."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


class DashboardRuntimeMetrics:
    """Persist request failures and process identity without touching trading state."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self.instance_id = uuid.uuid4().hex
        self.started_at = datetime.now(UTC).isoformat()
        self._payload = self._load()
        self._payload["process_restart_count"] = int(
            self._payload.get("process_restart_count") or 0
        ) + (1 if self._payload.get("instance_id") else 0)
        self._payload.update(
            {
                "instance_id": self.instance_id,
                "started_at": self.started_at,
                "dashboard_timeout_count": int(
                    self._payload.get("dashboard_timeout_count") or 0
                ),
                "dashboard_5xx_count": int(self._payload.get("dashboard_5xx_count") or 0),
                "max_request_duration_ms": float(
                    self._payload.get("max_request_duration_ms") or 0.0
                ),
            }
        )
        try:
            self._persist()
        except OSError:
            # The counters remain useful in-process even if the data volume is read-only.
            pass

    def _load(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.path.parent, delete=False
        ) as handle:
            json.dump(self._payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            temporary = Path(handle.name)
        temporary.replace(self.path)

    def record_request(
        self,
        *,
        status_code: int | None,
        duration_ms: float | None,
        timed_out: bool = False,
    ) -> None:
        with self._lock:
            if timed_out:
                self._payload["dashboard_timeout_count"] = int(
                    self._payload.get("dashboard_timeout_count") or 0
                ) + 1
            if status_code is not None and int(status_code) >= 500:
                self._payload["dashboard_5xx_count"] = int(
                    self._payload.get("dashboard_5xx_count") or 0
                ) + 1
            if duration_ms is not None:
                self._payload["max_request_duration_ms"] = max(
                    float(self._payload.get("max_request_duration_ms") or 0.0),
                    float(duration_ms),
                )
            try:
                self._persist()
            except OSError:
                pass

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "process_restart_count": int(self._payload.get("process_restart_count") or 0),
                "dashboard_timeout_count": int(
                    self._payload.get("dashboard_timeout_count") or 0
                ),
                "dashboard_5xx_count": int(self._payload.get("dashboard_5xx_count") or 0),
                "max_request_duration_ms": float(
                    self._payload.get("max_request_duration_ms") or 0.0
                ),
                "instance_id": self.instance_id,
                "started_at": self.started_at,
            }


_stores: dict[str, DashboardRuntimeMetrics] = {}


def get_dashboard_runtime_metrics(path: Path) -> DashboardRuntimeMetrics:
    key = str(Path(path).resolve())
    store = _stores.get(key)
    if store is None:
        store = DashboardRuntimeMetrics(Path(path))
        _stores[key] = store
    return store
