"""Persistent cross-process state and leases for model training schedulers."""

from __future__ import annotations

import json
import os
import socket
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

MODEL_TRAINING_STATE_VERSION = "2026-07-12.v1"
LOCAL_ML_MODEL_IDS = ("local_ml_profit_quality",)
LOCAL_AI_TOOL_MODEL_IDS = (
    "local_ai_profit_prediction",
    "local_ai_loss_filter",
    "local_ai_timeseries",
    "local_ai_sequence",
    "local_ai_sentiment_calibration",
    "local_ai_exit_profile",
)
ALL_TRAINABLE_MODEL_IDS = LOCAL_ML_MODEL_IDS + LOCAL_AI_TOOL_MODEL_IDS
MAX_HISTORY_EVENTS = 30
WRITE_LOCK_STALE_SECONDS = 30.0
WRITE_LOCK_WAIT_SECONDS = 3.0
INTERRUPTED_RETRY_INTERVAL_SECONDS = 5 * 60
RESOURCE_ERROR_RETRY_DELAYS_SECONDS = (5 * 60, 15 * 60, 60 * 60, 3 * 60 * 60, 6 * 60 * 60)
RESOURCE_ERROR_CIRCUIT_THRESHOLD = 3
RESOURCE_ERROR_CIRCUIT_OPEN_SECONDS = 6 * 60 * 60
_RESOURCE_ERROR_MARKERS = (
    "memoryerror",
    "out of memory",
    "oom",
    "cannot allocate memory",
    "resource exhausted",
    "training_process_interrupted",
    "training process interrupted",
)


def classify_training_failure(
    result: dict[str, Any] | None = None,
    *,
    error: str | None = None,
) -> str:
    """Classify failures that should consume the resource circuit budget."""

    payload = result if isinstance(result, dict) else {}
    reason = str(payload.get("reason") or "").lower()
    text = " ".join(
        str(value or "")
        for value in (
            reason,
            payload.get("error"),
            payload.get("message"),
            error,
        )
    ).lower()
    if any(marker in text for marker in _RESOURCE_ERROR_MARKERS):
        return "resource"
    return "functional"


def _resource_retry_delay(failure_count: int) -> float:
    index = max(int(failure_count or 1), 1) - 1
    return float(
        RESOURCE_ERROR_RETRY_DELAYS_SECONDS[
            min(index, len(RESOURCE_ERROR_RETRY_DELAYS_SECONDS) - 1)
        ]
    )


def _row_resource_failure_count(row: dict[str, Any]) -> int:
    """Read resource failure history from both current and legacy state rows."""

    persisted = max(int(row.get("resource_failure_count") or 0), 0)
    if persisted:
        return persisted
    legacy_result = row.get("last_result")
    legacy_error = row.get("last_error")
    if classify_training_failure(
        legacy_result if isinstance(legacy_result, dict) else None,
        error=str(legacy_error or ""),
    ) != "resource":
        return 0
    return max(int(row.get("retry_count") or 0), 0)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime) -> str:
    normalized = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return normalized.astimezone(UTC).isoformat()


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
            process_query_limited_information,
            False,
            pid,
        )
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _fresh_state(now: datetime) -> dict[str, Any]:
    return {
        "version": MODEL_TRAINING_STATE_VERSION,
        "status": "unavailable",
        "state_file_available": False,
        "updated_at": _iso(now),
        "schedulers": {},
        "models": {},
    }


def _result_summary(result: dict[str, Any] | None) -> dict[str, Any]:
    payload = result if isinstance(result, dict) else {}
    keys = (
        "trained",
        "reason",
        "message",
        "error",
        "trained_at",
        "sample_count",
        "shadow_sample_count",
        "trade_sample_count",
        "completed_shadow_sample_count",
        "last_trained_completed_shadow_sample_count",
        "last_trained_completed_sample_count",
        "completed_trade_sample_count",
        "last_trained_completed_trade_sample_count",
        "new_sample_count",
        "new_shadow_sample_count",
        "new_trade_sample_count",
        "completed_training_decision_group_count",
        "last_trained_completed_training_decision_group_count",
        "completed_shadow_raw_decision_group_count",
        "last_trained_completed_shadow_raw_decision_group_count",
        "full_training_probe_at",
        "full_training_probe_performed",
        "new_decision_group_count",
        "cost_complete_sample_count",
        "decision_group_count",
        "train_sample_count",
        "train_decision_group_count",
        "holdout_sample_count",
        "holdout_decision_group_count",
        "purged_training_decision_group_count",
        "purged_training_sample_count",
        "minimum_train_sample_count",
        "minimum_train_decision_group_count",
        "artifact_persisted",
        "readiness_state",
        "live_ml_ready",
        "training_input_fingerprint",
        "input_fingerprint",
        "data_fingerprint",
        "resource_error_class",
    )
    summary: dict[str, Any] = {}
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            summary[key] = value[:1000]
        elif isinstance(value, (bool, int, float)):
            summary[key] = value
    return summary


@dataclass
class TrainingLease:
    path: Path
    token: str
    scheduler_id: str
    run_id: str

    def release(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return
        if isinstance(payload, dict) and payload.get("token") == self.token:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass


@dataclass(frozen=True)
class LeaseAttempt:
    acquired: bool
    reason: str
    lease: TrainingLease | None = None
    recovered_stale_lease: bool = False


class ModelTrainingStateStore:
    """Atomic JSON state shared by trading, Dashboard, and audit processes."""

    def __init__(
        self,
        path: Path,
        *,
        now_provider: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.path = Path(path)
        self.lock_dir = self.path.with_name(f"{self.path.stem}.locks")
        self.write_lock_path = self.lock_dir / "state-write.lock"
        self.now_provider = now_provider
        self.hostname = socket.gethostname()

    def _load(self, *, strict: bool) -> dict[str, Any]:
        now = self.now_provider()
        if not self.path.exists():
            return _fresh_state(now)
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            if strict:
                raise RuntimeError(f"model training state is unreadable: {exc}") from exc
            return {
                **_fresh_state(now),
                "status": "error",
                "error": f"state_read_failed:{type(exc).__name__}",
            }
        if not isinstance(payload, dict):
            if strict:
                raise RuntimeError("model training state must be a JSON object")
            return {
                **_fresh_state(now),
                "status": "error",
                "error": "state_root_not_object",
            }
        if payload.get("version") != MODEL_TRAINING_STATE_VERSION:
            if strict:
                raise RuntimeError("unsupported model training state version")
            payload["status"] = "error"
            payload["error"] = "state_version_unsupported"
        payload.setdefault("schedulers", {})
        payload.setdefault("models", {})
        return payload

    def _acquire_write_lock(self) -> str:
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        deadline = time.monotonic() + WRITE_LOCK_WAIT_SECONDS
        while True:
            try:
                descriptor = os.open(
                    self.write_lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
            except FileExistsError:
                try:
                    age = max(time.time() - self.write_lock_path.stat().st_mtime, 0.0)
                except FileNotFoundError:
                    continue
                if age > WRITE_LOCK_STALE_SECONDS:
                    try:
                        self.write_lock_path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError("model training state write lock timed out") from None
                time.sleep(0.02)
                continue
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(token)
            return token

    def _release_write_lock(self, token: str) -> None:
        try:
            current = self.write_lock_path.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            return
        if current == token:
            try:
                self.write_lock_path.unlink()
            except FileNotFoundError:
                pass

    def _write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)

    def _mutate(self, mutator: Callable[[dict[str, Any], datetime], None]) -> dict[str, Any]:
        token = self._acquire_write_lock()
        try:
            now = self.now_provider()
            payload = self._load(strict=True)
            mutator(payload, now)
            payload["version"] = MODEL_TRAINING_STATE_VERSION
            payload["status"] = "ok"
            payload["state_file_available"] = True
            payload["updated_at"] = _iso(now)
            self._write(payload)
            return payload
        finally:
            self._release_write_lock(token)

    @staticmethod
    def _model_row(payload: dict[str, Any], model_id: str) -> dict[str, Any]:
        models = payload.setdefault("models", {})
        row = models.setdefault(
            model_id,
            {
                "model_id": model_id,
                "state": "never_checked",
                "retry_count": 0,
                "history": [],
            },
        )
        row.setdefault("history", [])
        return row

    @staticmethod
    def _append_history(row: dict[str, Any], event: dict[str, Any]) -> None:
        history = row.setdefault("history", [])
        history.append(event)
        del history[:-MAX_HISTORY_EVENTS]

    def read(self) -> dict[str, Any]:
        payload = self._load(strict=False)
        persisted_status = str(payload.get("status") or "unavailable")
        now = self.now_provider()
        schedulers = payload.get("schedulers")
        schedulers = schedulers if isinstance(schedulers, dict) else {}
        scheduler_model_ids: dict[str, set[str]] = {}
        for scheduler_id, raw in schedulers.items():
            if not isinstance(raw, dict):
                continue
            scheduler_model_ids[str(scheduler_id)] = {
                str(model_id) for model_id in raw.get("model_ids") or [] if str(model_id)
            }
            heartbeat = _parse_datetime(raw.get("heartbeat_at"))
            interval = max(float(raw.get("interval_seconds") or 0.0), 1.0)
            age = (now - heartbeat).total_seconds() if heartbeat is not None else None
            stale_after = max(interval * 1.25, interval + 60.0)
            raw["heartbeat_age_seconds"] = round(max(age, 0.0), 3) if age is not None else None
            raw["heartbeat_stale_after_seconds"] = round(stale_after, 3)
            raw["heartbeat_stale"] = age is None or age > stale_after
        fresh_scheduler_models = {
            str(scheduler_id): scheduler_model_ids.get(str(scheduler_id), set())
            for scheduler_id, raw in schedulers.items()
            if isinstance(raw, dict) and raw.get("heartbeat_stale") is False
        }
        stale_ids: list[str] = []
        superseded_ids: list[str] = []
        for scheduler_id, raw in schedulers.items():
            if not isinstance(raw, dict):
                continue
            normalized_id = str(scheduler_id)
            model_ids = scheduler_model_ids.get(normalized_id, set())
            covered_by = sorted(
                fresh_id
                for fresh_id, fresh_models in fresh_scheduler_models.items()
                if fresh_id != normalized_id and model_ids and model_ids.issubset(fresh_models)
            )
            superseded = bool(raw.get("heartbeat_stale") and covered_by)
            raw["heartbeat_superseded"] = superseded
            raw["heartbeat_superseded_by"] = covered_by
            raw["heartbeat_effective_stale"] = bool(raw.get("heartbeat_stale") and not superseded)
            if superseded:
                superseded_ids.append(normalized_id)
            elif raw["heartbeat_effective_stale"]:
                stale_ids.append(normalized_id)
        payload["status"] = "warning" if stale_ids else persisted_status
        payload["stale_scheduler_ids"] = stale_ids
        payload["superseded_scheduler_ids"] = superseded_ids
        payload["heartbeat_stale"] = bool(stale_ids)
        timed_out_models: list[str] = []
        models = payload.get("models") if isinstance(payload.get("models"), dict) else {}
        failed_model_ids: list[str] = []
        interrupted_model_ids: list[str] = []
        model_state_counts: dict[str, int] = {}
        for model_id, raw in models.items():
            if not isinstance(raw, dict):
                continue
            state = str(raw.get("state") or "unknown")
            model_state_counts[state] = model_state_counts.get(state, 0) + 1
            if state == "failed":
                failed_model_ids.append(str(model_id))
            elif state == "interrupted":
                interrupted_model_ids.append(str(model_id))
            if state != "running":
                continue
            started_at = _parse_datetime(raw.get("last_started_at"))
            timeout_seconds = max(float(raw.get("timeout_seconds") or 0.0), 0.0)
            age = (now - started_at).total_seconds() if started_at is not None else None
            raw["running_age_seconds"] = round(max(age, 0.0), 3) if age is not None else None
            raw["training_timeout_exceeded"] = bool(
                timeout_seconds > 0 and (age is None or age > timeout_seconds)
            )
            if raw["training_timeout_exceeded"]:
                timed_out_models.append(str(model_id))
        payload["timed_out_model_ids"] = timed_out_models
        payload["training_timeout_exceeded"] = bool(timed_out_models)
        payload["failed_model_ids"] = sorted(failed_model_ids)
        payload["interrupted_model_ids"] = sorted(interrupted_model_ids)
        payload["unhealthy_model_ids"] = sorted(
            set(failed_model_ids) | set(interrupted_model_ids) | set(timed_out_models)
        )
        payload["model_state_counts"] = dict(sorted(model_state_counts.items()))
        payload["model_state_healthy"] = not payload["unhealthy_model_ids"]
        if failed_model_ids:
            payload["status"] = "error"
        elif interrupted_model_ids or timed_out_models:
            payload["status"] = "warning"
        return payload

    def heartbeat(
        self,
        *,
        scheduler_id: str,
        model_ids: Iterable[str],
        interval_seconds: float,
    ) -> None:
        model_ids = tuple(dict.fromkeys(str(item) for item in model_ids if str(item)))

        def mutate(payload: dict[str, Any], now: datetime) -> None:
            scheduler = payload.setdefault("schedulers", {}).setdefault(scheduler_id, {})
            scheduler.update(
                {
                    "scheduler_id": scheduler_id,
                    "heartbeat_at": _iso(now),
                    "interval_seconds": max(float(interval_seconds), 1.0),
                    "model_ids": list(model_ids),
                    "owner_pid": os.getpid(),
                    "owner_host": self.hostname,
                }
            )
            for model_id in model_ids:
                row = self._model_row(payload, model_id)
                row["scheduler_heartbeat_id"] = scheduler_id
                row["scheduler_heartbeat_at"] = _iso(now)

        self._mutate(mutate)

    def record_check(
        self,
        *,
        scheduler_id: str,
        model_ids: Iterable[str],
        run_id: str,
        force: bool,
    ) -> None:
        def mutate(payload: dict[str, Any], now: datetime) -> None:
            for model_id in model_ids:
                row = self._model_row(payload, model_id)
                row.update(
                    {
                        "scheduler_id": scheduler_id,
                        "state": "checking",
                        "last_check_at": _iso(now),
                        "last_error": None,
                        "next_check_at": None,
                        "last_force": bool(force),
                        "active_run_id": run_id,
                        "owner_pid": os.getpid(),
                        "owner_host": self.hostname,
                    }
                )

        self._mutate(mutate)

    def start_run(
        self,
        *,
        scheduler_id: str,
        model_ids: Iterable[str],
        run_id: str,
        trigger_reason: str,
        sample_cursor: dict[str, int] | None = None,
        timeout_seconds: float = 0.0,
    ) -> None:
        def mutate(payload: dict[str, Any], now: datetime) -> None:
            for model_id in model_ids:
                row = self._model_row(payload, model_id)
                row.update(
                    {
                        "scheduler_id": scheduler_id,
                        "state": "running",
                        "triggered": True,
                        "trigger_reason": trigger_reason,
                        "last_started_at": _iso(now),
                        "last_error": None,
                        "next_check_at": None,
                        "active_run_id": run_id,
                        "active_sample_cursor": dict(sample_cursor or {}),
                        "timeout_seconds": max(float(timeout_seconds), 0.0),
                        "owner_pid": os.getpid(),
                        "owner_host": self.hostname,
                    }
                )
                self._append_history(
                    row,
                    {
                        "at": _iso(now),
                        "event": "started",
                        "run_id": run_id,
                        "trigger_reason": trigger_reason,
                    },
                )

        self._mutate(mutate)

    def finish_check(
        self,
        *,
        scheduler_id: str,
        model_ids: Iterable[str],
        run_id: str,
        result: dict[str, Any],
        next_check_at: datetime,
    ) -> None:
        summary = _result_summary(result)
        reason = str(summary.get("reason") or "unknown")
        trained = bool(summary.get("trained"))
        error = str(summary.get("error") or "")
        failed = bool(
            error
            or reason
            in {
                "error",
                "load_samples_error",
                "timeout",
                "resource_blocked",
            }
        )
        failure_class = classify_training_failure(summary)
        input_fingerprint = next(
            (
                str(summary.get(key)).strip()
                for key in (
                    "training_input_fingerprint",
                    "input_fingerprint",
                    "data_fingerprint",
                )
                if str(summary.get(key) or "").strip()
            ),
            None,
        )
        state = "succeeded" if trained else "failed" if failed else "skipped"

        def mutate(payload: dict[str, Any], now: datetime) -> None:
            for model_id in model_ids:
                row = self._model_row(payload, model_id)
                started = row.get("state") == "running" and row.get("active_run_id") == run_id
                retry_count = int(row.get("retry_count") or 0)
                effective_state = state
                resource_failure_count = int(row.get("resource_failure_count") or 0)
                previous_fingerprint = str(
                    row.get("resource_failure_fingerprint") or ""
                ).strip()
                if failed and failure_class == "resource":
                    previous_circuit_until = _parse_datetime(
                        row.get("resource_circuit_open_until")
                    )
                    if previous_circuit_until is not None and previous_circuit_until <= now:
                        resource_failure_count = 0
                    if (
                        input_fingerprint
                        and previous_fingerprint
                        and input_fingerprint != previous_fingerprint
                    ):
                        resource_failure_count = 0
                    resource_failure_count += 1
                    effective_next_check_at = now + timedelta(
                        seconds=_resource_retry_delay(resource_failure_count)
                    )
                    if resource_failure_count >= RESOURCE_ERROR_CIRCUIT_THRESHOLD:
                        effective_state = "resource_blocked"
                        effective_next_check_at = now + timedelta(
                            seconds=RESOURCE_ERROR_CIRCUIT_OPEN_SECONDS
                        )
                else:
                    resource_failure_count = 0
                    effective_next_check_at = next_check_at
                row.update(
                    {
                        "scheduler_id": scheduler_id,
                        "state": effective_state,
                        "triggered": bool(started or trained),
                        "trigger_reason": row.get("trigger_reason") if started else reason,
                        "last_finished_at": _iso(now)
                        if started or failed
                        else row.get("last_finished_at"),
                        "last_result": summary,
                        "last_error": error or None,
                        "next_check_at": _iso(effective_next_check_at),
                        "active_run_id": None,
                        "active_sample_cursor": None,
                        "retry_count": retry_count + 1 if failed else 0,
                        "resource_error_class": failure_class if failed else None,
                        "resource_failure_count": resource_failure_count,
                        "resource_failure_fingerprint": input_fingerprint
                        or previous_fingerprint
                        if failed and failure_class == "resource"
                        else None,
                        "resource_circuit_open_until": _iso(effective_next_check_at)
                        if effective_state == "resource_blocked"
                        else None,
                    }
                )
                if trained:
                    row["last_successful_training_at"] = _iso(now)
                cursor = {
                    "shadow": summary.get("last_trained_completed_shadow_sample_count")
                    or summary.get("last_trained_completed_sample_count")
                    or summary.get("completed_shadow_sample_count"),
                    "trade": summary.get("last_trained_completed_trade_sample_count")
                    or summary.get("completed_trade_sample_count"),
                    "decision_group": summary.get(
                        "last_trained_completed_training_decision_group_count"
                    )
                    or summary.get("completed_training_decision_group_count"),
                }
                if trained:
                    row["sample_cursor"] = {
                        key: int(value) for key, value in cursor.items() if value is not None
                    }
                self._append_history(
                    row,
                    {
                        "at": _iso(now),
                        "event": effective_state,
                        "run_id": run_id,
                        "reason": reason,
                        "error": error or None,
                    },
                )

        self._mutate(mutate)

    def record_exception(
        self,
        *,
        scheduler_id: str,
        model_ids: Iterable[str],
        run_id: str,
        error: str,
        next_check_at: datetime,
    ) -> None:
        self.finish_check(
            scheduler_id=scheduler_id,
            model_ids=model_ids,
            run_id=run_id,
            result={"trained": False, "reason": "error", "error": error[:1000]},
            next_check_at=next_check_at,
        )

    def record_timeout(
        self,
        *,
        scheduler_id: str,
        model_ids: Iterable[str],
        run_id: str,
        error: str,
        next_check_at: datetime,
    ) -> None:
        """Close a lease timeout as a failed run instead of leaving it running."""

        self.finish_check(
            scheduler_id=scheduler_id,
            model_ids=model_ids,
            run_id=run_id,
            result={
                "trained": False,
                "reason": "timeout",
                "error": error[:1000],
            },
            next_check_at=next_check_at,
        )

    def training_gate(
        self,
        *,
        scheduler_id: str,
        model_ids: Iterable[str],
        force: bool = False,
        input_fingerprint: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Return the persisted run gate before a trainer acquires a lease."""

        checked_at = now or self.now_provider()
        model_set = {str(model_id) for model_id in model_ids if str(model_id)}
        payload = self.read()
        models = payload.get("models") if isinstance(payload.get("models"), dict) else {}
        rows = [
            row
            for model_id, row in models.items()
            if str(model_id) in model_set and isinstance(row, dict)
        ]
        if not rows:
            return {
                "allowed": True,
                "reason": "training_gate_open",
                "scheduler_id": scheduler_id,
                "model_ids": sorted(model_set),
            }

        active = [row for row in rows if row.get("state") in {"checking", "running"}]
        if active:
            return {
                "allowed": False,
                "reason": "training_in_progress",
                "scheduler_id": scheduler_id,
                "active_run_ids": sorted(
                    {
                        str(row.get("active_run_id"))
                        for row in active
                        if row.get("active_run_id")
                    }
                ),
            }

        now_ts = checked_at
        circuit_rows = [
            row
            for row in rows
            if _row_resource_failure_count(row) >= RESOURCE_ERROR_CIRCUIT_THRESHOLD
            or row.get("state") == "resource_blocked"
        ]
        legacy_rows = [
            row
            for row in circuit_rows
            if not int(row.get("resource_failure_count") or 0)
            and row.get("state") != "resource_blocked"
        ]
        if legacy_rows:
            circuit_until = checked_at + timedelta(seconds=RESOURCE_ERROR_CIRCUIT_OPEN_SECONDS)

            def migrate(payload_to_update: dict[str, Any], _mutate_now: datetime) -> None:
                models_to_update = payload_to_update.get("models")
                if not isinstance(models_to_update, dict):
                    return
                for model_id in model_set:
                    row_to_update = models_to_update.get(model_id)
                    if not isinstance(row_to_update, dict):
                        continue
                    if int(row_to_update.get("resource_failure_count") or 0):
                        continue
                    count = _row_resource_failure_count(row_to_update)
                    if count < RESOURCE_ERROR_CIRCUIT_THRESHOLD:
                        continue
                    row_to_update.update(
                        {
                            "state": "resource_blocked",
                            "resource_error_class": "resource",
                            "resource_failure_count": count,
                            "resource_circuit_open_until": _iso(circuit_until),
                            "next_check_at": _iso(circuit_until),
                            "resource_failure_fingerprint": str(
                                _result_summary(row_to_update.get("last_result")).get(
                                    "training_input_fingerprint"
                                )
                                or _result_summary(row_to_update.get("last_result")).get(
                                    "input_fingerprint"
                                )
                                or ""
                            ).strip()
                            or None,
                        }
                    )
                    self._append_history(
                        row_to_update,
                        {
                            "at": _iso(_mutate_now),
                            "event": "resource_blocked",
                            "reason": "legacy_resource_failure_migrated",
                            "scheduler_id": scheduler_id,
                        },
                    )

            self._mutate(migrate)
        candidate_fingerprint = str(input_fingerprint or "").strip()
        if circuit_rows:
            stored_fingerprints = {
                str(row.get("resource_failure_fingerprint") or "").strip()
                for row in circuit_rows
                if str(row.get("resource_failure_fingerprint") or "").strip()
            }
            until_values = [
                _parse_datetime(row.get("resource_circuit_open_until"))
                for row in circuit_rows
            ]
            until = max((value for value in until_values if value), default=None)
            if until is not None and until <= now_ts:
                return {
                    "allowed": True,
                    "reason": "resource_circuit_half_open",
                    "state": "circuit_half_open",
                    "scheduler_id": scheduler_id,
                    "input_fingerprint": candidate_fingerprint or None,
                    "circuit_open_until": _iso(until),
                }
            # A force flag alone is intentionally insufficient to reopen a
            # resource circuit. The input must change and be explicitly named.
            if not candidate_fingerprint or candidate_fingerprint in stored_fingerprints:
                return {
                    "allowed": False,
                    "reason": "resource_blocked",
                    "state": "circuit_open",
                    "scheduler_id": scheduler_id,
                    "circuit_open_until": _iso(until) if until else None,
                    "resource_failure_count": max(
                        _row_resource_failure_count(row) for row in circuit_rows
                    ),
                    "legacy_state_compatibility": any(
                        not int(row.get("resource_failure_count") or 0)
                        and _row_resource_failure_count(row) > 0
                        for row in circuit_rows
                    ),
                    "force_ignored": bool(force),
                }
            return {
                "allowed": True,
                "reason": "resource_circuit_reset_new_input",
                "state": "circuit_half_open",
                "scheduler_id": scheduler_id,
                "input_fingerprint": candidate_fingerprint,
                "force": bool(force),
            }

        future_checks = [
            parsed
            for parsed in (_parse_datetime(row.get("next_check_at")) for row in rows)
            if parsed is not None and parsed > now_ts
        ]
        if future_checks and not force:
            next_check = max(future_checks)
            return {
                "allowed": False,
                "reason": "retry_backoff_active",
                "state": "cooldown",
                "scheduler_id": scheduler_id,
                "next_check_at": _iso(next_check),
                "remaining_seconds": round(max((next_check - now_ts).total_seconds(), 0.0), 3),
            }
        return {
            "allowed": True,
            "reason": "manual_force_override" if force and future_checks else "training_gate_open",
            "scheduler_id": scheduler_id,
            "force": bool(force),
        }

    def recover_interrupted_runs(
        self,
        *,
        retry_after_seconds: float = INTERRUPTED_RETRY_INTERVAL_SECONDS,
    ) -> list[str]:
        """Mark dead runs interrupted and ensure every interrupted run is retryable.

        Deployments can terminate a training subprocess between its state write and
        the scheduler's next check.  The historical ``interrupted`` event remains
        authoritative, but it must carry a retry timestamp so a fresh scheduler can
        take over instead of leaving the model in a permanently unscheduled state.
        """

        recovered: list[str] = []
        retry_delay = max(float(retry_after_seconds), 1.0)

        def mutate(payload: dict[str, Any], now: datetime) -> None:
            models = payload.get("models")
            if not isinstance(models, dict):
                return
            for model_id, row in models.items():
                if not isinstance(row, dict):
                    continue
                state = row.get("state")
                if state == "interrupted" and not row.get("next_check_at"):
                    recovered.append(str(model_id))
                    resource_failure_count = int(row.get("resource_failure_count") or 0) + 1
                    next_check_at = now + timedelta(seconds=retry_delay)
                    if resource_failure_count >= RESOURCE_ERROR_CIRCUIT_THRESHOLD:
                        row["state"] = "resource_blocked"
                        next_check_at = now + timedelta(seconds=RESOURCE_ERROR_CIRCUIT_OPEN_SECONDS)
                    row["next_check_at"] = _iso(next_check_at)
                    row["resource_failure_count"] = resource_failure_count
                    row["resource_error_class"] = "resource"
                    row["resource_circuit_open_until"] = (
                        _iso(next_check_at) if row.get("state") == "resource_blocked" else None
                    )
                    self._append_history(
                        row,
                        {
                            "at": _iso(now),
                            "event": "retry_scheduled",
                            "reason": "interrupted_training_retry",
                            "next_check_at": _iso(next_check_at),
                        },
                    )
                    continue
                if state not in {"checking", "running"}:
                    continue
                owner_host = str(row.get("owner_host") or "")
                owner_pid = int(row.get("owner_pid") or 0)
                if owner_host == self.hostname and _pid_alive(owner_pid):
                    continue
                recovered.append(str(model_id))
                run_id = str(row.get("active_run_id") or "unknown")
                resource_failure_count = int(row.get("resource_failure_count") or 0) + 1
                next_check_at = now + timedelta(seconds=_resource_retry_delay(resource_failure_count))
                state = "resource_blocked" if resource_failure_count >= RESOURCE_ERROR_CIRCUIT_THRESHOLD else "interrupted"
                if state == "resource_blocked":
                    next_check_at = now + timedelta(seconds=RESOURCE_ERROR_CIRCUIT_OPEN_SECONDS)
                row.update(
                    {
                        "state": state,
                        "last_finished_at": _iso(now),
                        "last_error": "training_process_interrupted",
                        "active_run_id": None,
                        "active_sample_cursor": None,
                        "next_check_at": _iso(next_check_at),
                        "retry_count": int(row.get("retry_count") or 0) + 1,
                        "resource_error_class": "resource",
                        "resource_failure_count": resource_failure_count,
                        "resource_circuit_open_until": (
                            _iso(next_check_at) if state == "resource_blocked" else None
                        ),
                    }
                )
                self._append_history(
                    row,
                    {
                        "at": _iso(now),
                        "event": "interrupted",
                        "run_id": run_id,
                        "error": "training_process_interrupted",
                    },
                )

        self._mutate(mutate)
        return recovered

    def try_acquire_lease(
        self,
        *,
        scheduler_id: str,
        stale_after_seconds: float,
    ) -> LeaseAttempt:
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        lease_path = self.lock_dir / f"{scheduler_id}.lease"
        recovered = False
        for _attempt in range(2):
            token = uuid.uuid4().hex
            run_id = uuid.uuid4().hex
            now = self.now_provider()
            payload = {
                "token": token,
                "run_id": run_id,
                "scheduler_id": scheduler_id,
                "owner_pid": os.getpid(),
                "owner_host": self.hostname,
                "acquired_at": _iso(now),
            }
            try:
                descriptor = os.open(lease_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                try:
                    existing = json.loads(lease_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    existing = {}
                acquired_at = _parse_datetime(existing.get("acquired_at"))
                age = (
                    (now - acquired_at).total_seconds() if acquired_at is not None else float("inf")
                )
                owner_host = str(existing.get("owner_host") or "")
                owner_pid = int(existing.get("owner_pid") or 0)
                owner_alive = owner_host == self.hostname and _pid_alive(owner_pid)
                if owner_alive:
                    return LeaseAttempt(False, "training_in_progress")
                if owner_host != self.hostname and age <= max(float(stale_after_seconds), 1.0):
                    return LeaseAttempt(False, "training_in_progress")
                try:
                    lease_path.unlink()
                except FileNotFoundError:
                    pass
                self.recover_interrupted_runs()
                recovered = True
                continue
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=True, sort_keys=True)
            return LeaseAttempt(
                True,
                "acquired",
                TrainingLease(lease_path, token, scheduler_id, run_id),
                recovered_stale_lease=recovered,
            )
        return LeaseAttempt(False, "lease_acquire_failed", recovered_stale_lease=recovered)
