"""Durable, truthful 24/72-hour acceptance observation state.

The observer records only facts collected by a caller.  It never advances a
window by synthetic timestamps and never marks a gate passed before the real
elapsed time and all required checks are present.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

CONTINUOUS_OBSERVATION_VERSION = "2026-09-16.continuous-observation.v3"
ALLOWED_WINDOW_HOURS = (24, 72)
MAX_SAMPLES = 2000
EXPECTED_SAMPLE_INTERVAL_SECONDS = 300.0
MAX_ALLOWED_SAMPLE_GAP_SECONDS = EXPECTED_SAMPLE_INTERVAL_SECONDS * 3
MAX_LATEST_SAMPLE_AGE_SECONDS = EXPECTED_SAMPLE_INTERVAL_SECONDS * 3
WORKER_STATE_VERSION = "2026-09-15.continuous-observation-worker.v1"

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

ObservationMetricsCollector = Callable[[], Awaitable[dict[str, Any]]]


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
    for key in ("source", "blocked_reason", "collection_errors"):
        if source.get(key):
            result[key] = str(source[key])[:300]
    return result


class ContinuousObservationWorkerState:
    """Persist worker liveness separately from the acceptance window facts."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def _default(self) -> dict[str, Any]:
        return {
            "version": WORKER_STATE_VERSION,
            "status": "stopped",
            "worker_pid": None,
            "started_at": None,
            "last_heartbeat_at": None,
            "last_sample_at": None,
            "sample_count": 0,
            "error_count": 0,
            "last_error": None,
            "stopped_at": None,
        }

    def read(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self._default()
        if not isinstance(payload, dict) or payload.get("version") != WORKER_STATE_VERSION:
            return self._default()
        result = self._default()
        result.update(payload)
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

    def _update(self, **fields: Any) -> dict[str, Any]:
        payload = self.read()
        payload.update(fields)
        self._write(payload)
        return payload

    def mark_started(self, *, now: datetime | None = None) -> dict[str, Any]:
        timestamp = _iso(now or _now())
        return self._update(
            status="running",
            worker_pid=os.getpid(),
            started_at=timestamp,
            last_heartbeat_at=timestamp,
            stopped_at=None,
            last_error=None,
        )

    def mark_heartbeat(self, *, now: datetime | None = None) -> dict[str, Any]:
        return self._update(
            status="running",
            last_heartbeat_at=_iso(now or _now()),
        )

    def mark_sample(self, *, now: datetime | None = None) -> dict[str, Any]:
        payload = self.read()
        return self._update(
            status="running",
            last_heartbeat_at=_iso(now or _now()),
            last_sample_at=_iso(now or _now()),
            sample_count=int(payload.get("sample_count") or 0) + 1,
        )

    def mark_error(self, error: str, *, now: datetime | None = None) -> dict[str, Any]:
        payload = self.read()
        return self._update(
            status="degraded",
            last_heartbeat_at=_iso(now or _now()),
            error_count=int(payload.get("error_count") or 0) + 1,
            last_error=str(error)[:300],
        )

    def mark_stopped(self, *, now: datetime | None = None) -> dict[str, Any]:
        return self._update(
            status="stopped",
            last_heartbeat_at=_iso(now or _now()),
            stopped_at=_iso(now or _now()),
        )


def observation_metrics_ready(metrics: Any) -> bool:
    """Return whether a collected sample can start or recover a window."""

    normalized = _normalise_metrics(metrics)
    if normalized.get("blocked_reason") or normalized.get("collection_errors"):
        return False
    analysis_interval = normalized.get("max_analysis_interval_seconds")
    if analysis_interval is None or float(analysis_interval) > 180.0:
        return False
    if normalized.get("training_state_clear") is not True:
        return False
    return all(
        normalized.get(key) is not None
        for key in _REQUIRED_72H_METRICS
        if key not in {"max_analysis_interval_seconds", "training_state_clear"}
    )


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
            "baseline_metrics": {},
            "samples": [],
            "reset_count": 0,
            "last_reset_at": None,
            "last_reset_reason": None,
        }

    def read(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self._default()
        if not isinstance(payload, dict):
            return self._default()
        if payload.get("version") != CONTINUOUS_OBSERVATION_VERSION:
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

    def start(
        self,
        *,
        required_hours: int = 24,
        now: datetime | None = None,
        baseline_metrics: dict[str, Any] | None = None,
        restart: bool = False,
        reset_reason: str | None = None,
    ) -> dict[str, Any]:
        hours = int(required_hours)
        if hours not in ALLOWED_WINDOW_HOURS:
            raise ValueError("required_hours must be 24 or 72")
        payload = self.read()
        if (
            not restart
            and payload.get("window_started_at")
            and int(payload.get("required_hours") or 24) == hours
        ):
            return self.snapshot(now=now)
        started = now or _now()
        normalized_baseline = _normalise_metrics(baseline_metrics or {})
        normalized_baseline.pop("source", None)
        normalized_baseline.pop("blocked_reason", None)
        previous_started = payload.get("window_started_at")
        reset_count = int(payload.get("reset_count") or 0)
        if restart and previous_started:
            reset_count += 1
        payload.update(
            {
                "version": CONTINUOUS_OBSERVATION_VERSION,
                "status": "observing",
                "required_hours": hours,
                "window_started_at": _iso(started),
                "last_sample_at": None,
                "baseline_metrics": normalized_baseline,
                "samples": [],
                "blocked_reason": None,
                "reset_count": reset_count,
                "last_reset_at": _iso(started) if restart and previous_started else payload.get("last_reset_at"),
                "last_reset_reason": (
                    str(reset_reason or "explicit_restart")[:300]
                    if restart and previous_started
                    else payload.get("last_reset_reason")
                ),
            }
        )
        self._write(payload)
        return self.snapshot(now=started)

    def record(self, metrics: dict[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
        payload = self.read()
        if not payload.get("window_started_at"):
            raise RuntimeError("observation_window_not_started")
        observed_at = now or _now()
        normalized = _normalise_metrics(metrics)
        baseline = payload.get("baseline_metrics")
        baseline = baseline if isinstance(baseline, dict) else {}
        for key in dict.fromkeys(_REQUIRED_24H_METRICS + _REQUIRED_72H_METRICS):
            if key in {"max_analysis_interval_seconds", "training_state_clear"}:
                continue
            current = normalized.get(key)
            initial = baseline.get(key)
            if isinstance(current, (int, float)) and isinstance(initial, (int, float)):
                normalized[key] = max(int(current) - int(initial), 0)
        row = {"observed_at": _iso(observed_at), "metrics": normalized}
        samples = [item for item in payload.get("samples", []) if isinstance(item, dict)]
        samples.append(row)
        payload["samples"] = samples[-MAX_SAMPLES:]
        payload["last_sample_at"] = row["observed_at"]
        blocked_reason = str(metrics.get("blocked_reason") or "").strip()
        collection_errors = str(metrics.get("collection_errors") or "").strip()
        if not blocked_reason and collection_errors:
            blocked_reason = f"collection_error:{collection_errors}"
        if blocked_reason:
            payload["blocked_reason"] = blocked_reason[:300]
        self._write(payload)
        return self.snapshot(now=observed_at)

    def snapshot(self, *, now: datetime | None = None) -> dict[str, Any]:
        payload = self.read()
        current = now or _now()
        started = _parse(payload.get("window_started_at"))
        required_hours = int(payload.get("required_hours") or 24)
        baseline = payload.get("baseline_metrics")
        baseline = baseline if isinstance(baseline, dict) else {}
        samples = [item for item in payload.get("samples", []) if isinstance(item, dict)]
        latest_metrics = samples[-1].get("metrics", {}) if samples else {}
        sample_times = [
            parsed
            for parsed in (_parse(item.get("observed_at")) for item in samples)
            if parsed is not None
        ]
        first_sample_delay_seconds = (
            max((sample_times[0] - started).total_seconds(), 0.0)
            if started and sample_times
            else None
        )
        max_sample_gap_seconds = None
        if started and sample_times:
            continuity_points = [started, *sample_times]
            max_sample_gap_seconds = max(
                max((right - left).total_seconds(), 0.0)
                for left, right in zip(continuity_points, continuity_points[1:], strict=False)
            )
        latest_sample_age_seconds = (
            max((current - sample_times[-1]).total_seconds(), 0.0) if sample_times else None
        )
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
        continuity_failures: list[str] = []
        if (
            first_sample_delay_seconds is not None
            and first_sample_delay_seconds > MAX_ALLOWED_SAMPLE_GAP_SECONDS
        ):
            continuity_failures.append("first_sample_delayed")
        if (
            max_sample_gap_seconds is not None
            and max_sample_gap_seconds > MAX_ALLOWED_SAMPLE_GAP_SECONDS
        ):
            continuity_failures.append("sample_gap_exceeded")
        if (
            latest_sample_age_seconds is not None
            and latest_sample_age_seconds > MAX_LATEST_SAMPLE_AGE_SECONDS
        ):
            continuity_failures.append("latest_sample_stale")
        blocked_reason = str(payload.get("blocked_reason") or "").strip() or None
        if not started:
            status = "not_started"
        elif blocked_reason:
            status = "blocked"
        elif continuity_failures:
            status = "blocked"
            blocked_reason = "continuity_failed:" + ",".join(continuity_failures)
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
            "continuity_failures": continuity_failures,
            "blocked_reason": blocked_reason,
            "latest_metrics": latest_metrics,
            "baseline_metrics": baseline if isinstance(baseline, dict) else {},
            "reset_count": int(payload.get("reset_count") or 0),
            "last_reset_at": payload.get("last_reset_at"),
            "last_reset_reason": payload.get("last_reset_reason"),
            "evidence": {
                "real_elapsed_time_required": True,
                "synthetic_time_allowed": False,
                "source": "persisted_observation_samples",
                "expected_sample_interval_seconds": EXPECTED_SAMPLE_INTERVAL_SECONDS,
                "max_allowed_sample_gap_seconds": MAX_ALLOWED_SAMPLE_GAP_SECONDS,
                "max_latest_sample_age_seconds": MAX_LATEST_SAMPLE_AGE_SECONDS,
                "first_sample_delay_seconds": first_sample_delay_seconds,
                "max_sample_gap_seconds": max_sample_gap_seconds,
                "latest_sample_age_seconds": latest_sample_age_seconds,
            },
        }


class ContinuousObservationScheduler:
    """Start and sample both real 24/72-hour observation windows."""

    def __init__(
        self,
        stores: dict[int, ContinuousObservationStore],
        collector: ObservationMetricsCollector,
        *,
        interval_seconds: float = 300.0,
        startup_delay_seconds: float = 5.0,
        worker_state_path: Path | None = None,
    ) -> None:
        self.stores = stores
        self.collector = collector
        self.interval_seconds = max(float(interval_seconds), 1.0)
        self.startup_delay_seconds = max(float(startup_delay_seconds), 0.0)
        self.worker_state = (
            ContinuousObservationWorkerState(worker_state_path)
            if worker_state_path is not None
            else None
        )
        self._task: asyncio.Task[Any] | None = None
        self._stop_event = asyncio.Event()

    @property
    def task(self) -> asyncio.Task[Any] | None:
        return self._task

    async def start(self) -> None:
        """Ensure windows exist, then launch one sampler task."""

        if self._task is not None and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        if self.worker_state is not None:
            self.worker_state.mark_started()
        now = _now()
        pending_stores: list[tuple[int, ContinuousObservationStore]] = []
        for hours in ALLOWED_WINDOW_HOURS:
            store = self.stores.get(hours)
            if store is None:
                continue
            snapshot = store.snapshot(now=now)
            if snapshot.get("status") == "not_started":
                pending_stores.append((hours, store))
        baseline: dict[str, Any] = {}
        if pending_stores:
            try:
                collected = await self.collector()
                baseline = collected if isinstance(collected, dict) else {}
            except Exception:
                baseline = {}
        if observation_metrics_ready(baseline):
            for hours, store in pending_stores:
                store.start(
                    required_hours=hours,
                    now=now,
                    baseline_metrics=baseline,
                )
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        task = self._task
        self._task = None
        self._stop_event.set()
        if task is None:
            if self.worker_state is not None:
                self.worker_state.mark_stopped()
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        if self.worker_state is not None:
            self.worker_state.mark_stopped()

    async def sample_once(self) -> dict[str, Any]:
        """Collect and persist one sample for every active window."""

        if self.worker_state is not None:
            self.worker_state.mark_heartbeat()
        try:
            metrics = await self.collector()
            if not isinstance(metrics, dict):
                raise TypeError("observation_collector_must_return_mapping")
        except Exception as exc:
            if self.worker_state is not None:
                self.worker_state.mark_error(f"collector_error:{type(exc).__name__}")
            metrics = {
                "source": "continuous_observation_scheduler",
                "collection_errors": f"collector_error:{type(exc).__name__}",
            }
        observed_at = _now()
        metrics_ready = observation_metrics_ready(metrics)
        snapshots: dict[str, Any] = {}
        for hours in ALLOWED_WINDOW_HOURS:
            store = self.stores.get(hours)
            if store is None:
                continue
            current = store.snapshot(now=observed_at)
            if current.get("status") == "not_started":
                if not metrics_ready:
                    snapshots[str(hours)] = current
                    continue
                store.start(
                    required_hours=hours,
                    now=observed_at,
                    baseline_metrics=metrics,
                )
            elif current.get("status") == "blocked" and metrics_ready:
                store.start(
                    required_hours=hours,
                    now=observed_at,
                    baseline_metrics=metrics,
                    restart=True,
                    reset_reason=str(current.get("blocked_reason") or "blocked_window"),
                )
            snapshots[str(hours)] = store.record(metrics, now=observed_at)
        if self.worker_state is not None:
            self.worker_state.mark_sample(now=observed_at)
        return snapshots

    async def _run(self) -> None:
        if self.startup_delay_seconds:
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self.startup_delay_seconds
                )
                return
            except TimeoutError:
                pass
        while True:
            await self.sample_once()
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self.interval_seconds
                )
                return
            except TimeoutError:
                continue
