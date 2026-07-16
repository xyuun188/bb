"""Persistent cross-process state and leases for model training schedulers."""

from __future__ import annotations

import json
import os
import socket
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
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
        "artifact_persisted",
        "readiness_state",
        "allow_live_position_influence",
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
        now = self.now_provider()
        schedulers = payload.get("schedulers")
        schedulers = schedulers if isinstance(schedulers, dict) else {}
        scheduler_model_ids: dict[str, set[str]] = {}
        for scheduler_id, raw in schedulers.items():
            if not isinstance(raw, dict):
                continue
            scheduler_model_ids[str(scheduler_id)] = {
                str(model_id)
                for model_id in raw.get("model_ids") or []
                if str(model_id)
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
            raw["heartbeat_effective_stale"] = bool(
                raw.get("heartbeat_stale") and not superseded
            )
            if superseded:
                superseded_ids.append(normalized_id)
            elif raw["heartbeat_effective_stale"]:
                stale_ids.append(normalized_id)
        payload["status"] = "warning" if stale_ids else payload.get("status", "ok")
        payload["stale_scheduler_ids"] = stale_ids
        payload["superseded_scheduler_ids"] = superseded_ids
        payload["heartbeat_stale"] = bool(stale_ids)
        timed_out_models: list[str] = []
        models = payload.get("models") if isinstance(payload.get("models"), dict) else {}
        for model_id, raw in models.items():
            if not isinstance(raw, dict) or raw.get("state") != "running":
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
        if timed_out_models:
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
                        "active_run_id": run_id,
                        "sample_cursor": dict(sample_cursor or {}),
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
        failed = bool(error or reason in {"error", "load_samples_error", "timeout"})
        state = "succeeded" if trained else "failed" if failed else "skipped"

        def mutate(payload: dict[str, Any], now: datetime) -> None:
            for model_id in model_ids:
                row = self._model_row(payload, model_id)
                started = row.get("state") == "running" and row.get("active_run_id") == run_id
                retry_count = int(row.get("retry_count") or 0)
                row.update(
                    {
                        "scheduler_id": scheduler_id,
                        "state": state,
                        "triggered": bool(started or trained),
                        "trigger_reason": row.get("trigger_reason") if started else reason,
                        "last_finished_at": _iso(now) if started or failed else row.get("last_finished_at"),
                        "last_result": summary,
                        "last_error": error or None,
                        "next_check_at": _iso(next_check_at),
                        "active_run_id": None,
                        "retry_count": retry_count + 1 if failed else 0,
                    }
                )
                cursor = {
                    "shadow": summary.get("last_trained_completed_shadow_sample_count")
                    or summary.get("last_trained_completed_sample_count")
                    or summary.get("completed_shadow_sample_count"),
                    "trade": summary.get("last_trained_completed_trade_sample_count")
                    or summary.get("completed_trade_sample_count"),
                }
                row["sample_cursor"] = {key: int(value) for key, value in cursor.items() if value is not None}
                self._append_history(
                    row,
                    {
                        "at": _iso(now),
                        "event": state,
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

    def recover_interrupted_runs(self) -> list[str]:
        recovered: list[str] = []

        def mutate(payload: dict[str, Any], now: datetime) -> None:
            models = payload.get("models")
            if not isinstance(models, dict):
                return
            for model_id, row in models.items():
                if not isinstance(row, dict) or row.get("state") not in {"checking", "running"}:
                    continue
                owner_host = str(row.get("owner_host") or "")
                owner_pid = int(row.get("owner_pid") or 0)
                if owner_host == self.hostname and _pid_alive(owner_pid):
                    continue
                recovered.append(str(model_id))
                run_id = str(row.get("active_run_id") or "unknown")
                row.update(
                    {
                        "state": "interrupted",
                        "last_finished_at": _iso(now),
                        "last_error": "training_process_interrupted",
                        "active_run_id": None,
                        "retry_count": int(row.get("retry_count") or 0) + 1,
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
                age = (now - acquired_at).total_seconds() if acquired_at is not None else float("inf")
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
