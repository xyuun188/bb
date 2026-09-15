"""Model-training scheduling and isolated worker ownership.

The realtime trading service supplies market-busy state and shared runtime
services. This module owns training leases, subprocesses, trigger policy,
heartbeats, and cursors so training cannot grow inside the trading orchestrator.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import structlog
from sqlalchemy import func, select

from core.safe_output import safe_error_tail, safe_error_text
from db.session import get_session_ctx
from models.learning import ShadowBacktest
from services.local_ai_training_contract import (
    LOCAL_AI_TOOLS_TRAIN_RESULT_PREFIX,
    decision_group_training_trigger,
    training_distribution_drift,
)
from services.ml_signal_service import (
    AUTO_TRAIN_CHECK_INTERVAL_SECONDS,
    AUTO_TRAIN_LEASE_STALE_SECONDS,
    AUTO_TRAIN_RETRY_INTERVAL_SECONDS,
    LOCAL_ML_TRAINING_SCHEDULER_ID,
    MODEL_TRAINING_STATE_STORE,
)
from services.ml_training_contract import LOCAL_ML_AUTO_TRAIN_RESULT_PREFIX
from services.model_training_state import (
    ALL_TRAINABLE_MODEL_IDS,
    LOCAL_AI_TOOL_MODEL_IDS,
    LOCAL_ML_MODEL_IDS,
    ModelTrainingStateStore,
    training_input_fingerprint,
)
from services.trading_params import DEFAULT_TRADING_PARAMS
from services.training_epoch import CURRENT_TRAINING_EPOCH_POLICY

PROJECT_ROOT = Path(__file__).resolve().parents[1]
logger = structlog.get_logger(__name__)

TRAINING_PROCESS_NICE_LEVEL = 10
TRAINING_PROCESS_MAX_WORKERS = 1
TRAINING_STARTUP_GRACE_SECONDS = 180.0
TRAINING_BUSY_RETRY_SECONDS = 30.0
TRAINING_MAX_BUSY_DEFERRAL_SECONDS = 15 * 60.0
AUTO_TRAIN_FAILURE_BACKOFF_MIN_SECONDS = 30 * 60
AUTO_TRAIN_FAILURE_BACKOFF_MAX_SECONDS = 4 * 60 * 60
LOCAL_ML_TRAINING_PARAMS = DEFAULT_TRADING_PARAMS.local_ml_training


def low_priority_training_command(
    command: list[str],
    *,
    platform_name: str | None = None,
    nice_path: str | None = None,
) -> list[str]:
    """Prefer trading latency while allowing training to use idle CPU."""

    selected_platform = platform_name or os.name
    resolved_nice = nice_path if nice_path is not None else shutil.which("nice")
    if selected_platform == "posix" and resolved_nice:
        return [resolved_nice, "-n", str(TRAINING_PROCESS_NICE_LEVEL), *command]
    return list(command)


def training_process_env(base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Bound native thread pools so training cannot starve live inference."""

    env = dict(base_env or os.environ)
    worker_count = str(TRAINING_PROCESS_MAX_WORKERS)
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "TORCH_NUM_THREADS",
    ):
        env[key] = worker_count
    env["BB_TRAINING_MAX_WORKERS"] = worker_count
    env["MALLOC_ARENA_MAX"] = "2"
    return env


class ModelTrainingCoordinatorMixin:
    """Own model-training lifecycle for the trading orchestrator."""

    def initialize_model_training(
        self,
        model_training_state_store: ModelTrainingStateStore | None,
    ) -> None:
        self.model_training_state_store = (
            model_training_state_store or MODEL_TRAINING_STATE_STORE
        )
        self._ml_auto_train_task: asyncio.Task | None = None
        self._model_training_heartbeat_task: asyncio.Task | None = None
        self._active_training_processes: set[asyncio.subprocess.Process] = set()
        self._local_tools_active_training_run_id: str | None = None
        self._local_tools_last_train_started_at: datetime | None = None
        self._local_tools_last_completed_shadow_count = 0
        self._auto_train_busy_since: datetime | None = None
        self._auto_train_failure_count = 0

    def _model_training_state(self) -> ModelTrainingStateStore:
        store = getattr(self, "model_training_state_store", None)
        if store is None:
            store = MODEL_TRAINING_STATE_STORE
            self.model_training_state_store = store
        return store

    def _record_training_subprocess_timeout(
        self,
        *,
        scheduler_id: str,
        model_ids: tuple[str, ...],
        error: str,
        run_id: str | None = None,
    ) -> None:
        """Close the persisted run when its isolated worker is killed by a lease timeout."""

        store = self._model_training_state()
        selected_run_id = run_id
        if not selected_run_id:
            try:
                payload = store.read()
                rows = payload.get("models") if isinstance(payload, dict) else {}
                for model_id in model_ids:
                    row = rows.get(model_id) if isinstance(rows, dict) else None
                    if isinstance(row, dict) and row.get("state") == "running":
                        selected_run_id = str(row.get("active_run_id") or "") or None
                        if selected_run_id:
                            break
            except Exception as exc:
                logger.warning(
                    "failed to read training state before timeout closure",
                    scheduler_id=scheduler_id,
                    error=safe_error_text(exc, limit=180),
                )
        if not selected_run_id:
            return
        try:
            store.record_timeout(
                scheduler_id=scheduler_id,
                model_ids=model_ids,
                run_id=selected_run_id,
                error=error,
                next_check_at=datetime.now(UTC)
                + timedelta(seconds=AUTO_TRAIN_RETRY_INTERVAL_SECONDS),
            )
        except Exception as exc:
            logger.warning(
                "failed to persist training subprocess timeout state",
                scheduler_id=scheduler_id,
                error=safe_error_text(exc, limit=180),
            )
        self._reclaim_training_subprocess_lease(
            store=store,
            scheduler_id=scheduler_id,
        )

    @staticmethod
    def _reclaim_training_subprocess_lease(
        *,
        store: ModelTrainingStateStore,
        scheduler_id: str,
    ) -> None:
        cleanup = store.try_acquire_lease(
            scheduler_id=scheduler_id,
            stale_after_seconds=AUTO_TRAIN_LEASE_STALE_SECONDS,
        )
        if cleanup.acquired and cleanup.lease is not None:
            cleanup.lease.release()

    def _record_training_subprocess_failure(
        self,
        *,
        scheduler_id: str,
        model_ids: tuple[str, ...],
        error: str,
    ) -> None:
        """Close a dead isolated worker's run and reclaim its process-owned lease."""

        store = self._model_training_state()
        run_id: str | None = None
        try:
            payload = store.read()
            rows = payload.get("models") if isinstance(payload, dict) else {}
            for model_id in model_ids:
                row = rows.get(model_id) if isinstance(rows, dict) else None
                if isinstance(row, dict) and row.get("state") in {"checking", "running"}:
                    run_id = str(row.get("active_run_id") or "") or None
                    if run_id:
                        break
            if run_id:
                store.record_exception(
                    scheduler_id=scheduler_id,
                    model_ids=model_ids,
                    run_id=run_id,
                    error=error,
                    next_check_at=datetime.now(UTC)
                    + timedelta(seconds=AUTO_TRAIN_RETRY_INTERVAL_SECONDS),
                )
            self._reclaim_training_subprocess_lease(
                store=store,
                scheduler_id=scheduler_id,
            )
        except Exception as exc:
            logger.warning(
                "failed to close dead training subprocess state",
                scheduler_id=scheduler_id,
                error=safe_error_text(exc, limit=180),
            )

    def _auto_train_failure_delay(self, results: list[dict[str, Any]]) -> float:
        """Back off repeated failures while keeping normal checks on their cadence."""
        failed = any(
            str(result.get("reason") or "")
            in {"error", "invalid_training_response", "load_samples_error", "timeout"}
            for result in results
            if isinstance(result, dict)
        )
        if not failed:
            self._auto_train_failure_count = 0
            return float(AUTO_TRAIN_CHECK_INTERVAL_SECONDS)
        retry_count = int(getattr(self, "_auto_train_failure_count", 0) or 0)
        try:
            payload = self._model_training_state().read()
            models = payload.get("models") if isinstance(payload, dict) else {}
            persisted_retry = max(
                [
                    int(row.get("retry_count") or 0)
                    for row in (models.values() if isinstance(models, dict) else [])
                    if isinstance(row, dict)
                ]
                or [0]
            )
            retry_count = max(retry_count, persisted_retry)
        except Exception as exc:
            logger.debug(
                "persistent auto-train retry count unavailable; using process-local count",
                error=safe_error_text(exc, limit=120),
            )
        retry_count += 1
        self._auto_train_failure_count = retry_count
        return float(
            min(
                AUTO_TRAIN_FAILURE_BACKOFF_MIN_SECONDS * (2 ** max(retry_count - 1, 0)),
                AUTO_TRAIN_FAILURE_BACKOFF_MAX_SECONDS,
            )
        )

    def _auto_training_should_defer(self, *, now: datetime | None = None) -> bool:
        runtime = getattr(self, "_analysis_runtime", {})
        busy = any(
            bool(getattr(runtime.get(scope), "active", False))
            for scope in ("market", "position")
        ) or bool(
            getattr(self, "_active_analysis_symbols", set())
            or (
                getattr(self, "_market_entry_pipeline_semaphore", None)
                and self._market_entry_pipeline_semaphore.locked()
            )
        )
        if not busy:
            self._auto_train_busy_since = None
            return False

        checked_at = now or datetime.now(UTC)
        busy_since = getattr(self, "_auto_train_busy_since", None)
        if not isinstance(busy_since, datetime):
            self._auto_train_busy_since = checked_at
            return True
        if (checked_at - busy_since).total_seconds() < TRAINING_MAX_BUSY_DEFERRAL_SECONDS:
            return True

        self._auto_train_busy_since = None
        logger.warning(
            "auto-training busy deferral limit reached; running isolated trainers",
            deferred_seconds=round((checked_at - busy_since).total_seconds(), 3),
        )
        return False

    def _training_process_set(self) -> set[asyncio.subprocess.Process]:
        processes = getattr(self, "_active_training_processes", None)
        if not isinstance(processes, set):
            processes = set()
            self._active_training_processes = processes
        return processes

    def start_model_training(self) -> None:
        if self._ml_auto_train_task and not self._ml_auto_train_task.done():
            return
        state_store = self._model_training_state()
        for scheduler_id in ("local_ai_tools_auto_train", "local_ml_auto_train"):
            try:
                self._reclaim_training_subprocess_lease(
                    store=state_store,
                    scheduler_id=scheduler_id,
                )
            except Exception as exc:
                logger.warning(
                    "training lease recovery failed at scheduler startup",
                    scheduler_id=scheduler_id,
                    error=safe_error_text(exc, limit=180),
                )
        try:
            recovered = state_store.recover_interrupted_runs()
            if recovered:
                logger.warning("recovered interrupted model training runs", model_ids=recovered)
        except Exception as exc:
            logger.warning(
                "model training state recovery failed; trading continues with training blocked",
                error=safe_error_text(exc, limit=180),
            )
        self._ml_auto_train_task = asyncio.create_task(self._ml_auto_train_loop())
        self._model_training_heartbeat_task = asyncio.create_task(
            self._model_training_heartbeat_loop()
        )

    async def stop_model_training(self) -> None:
        tasks = (
            getattr(self, "_ml_auto_train_task", None),
            getattr(self, "_model_training_heartbeat_task", None),
        )
        self._ml_auto_train_task = None
        self._model_training_heartbeat_task = None
        for task in tasks:
            if not task or task.done():
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _model_training_heartbeat_loop(self) -> None:
        heartbeat_interval = min(max(AUTO_TRAIN_CHECK_INTERVAL_SECONDS / 6, 30.0), 60.0)
        while self._running:
            try:
                self._model_training_state().heartbeat(
                    scheduler_id="platform_model_training_loop",
                    model_ids=ALL_TRAINABLE_MODEL_IDS,
                    interval_seconds=AUTO_TRAIN_CHECK_INTERVAL_SECONDS,
                )
            except Exception as exc:
                logger.warning(
                    "model training scheduler heartbeat write failed",
                    error=safe_error_text(exc, limit=180),
                )
            await asyncio.sleep(heartbeat_interval)

    def _heartbeat_deferred_training_schedulers(self) -> None:
        """Keep deferred child schedulers observable while trading stays busy."""

        state_store = self._model_training_state()
        for scheduler_id, model_ids in (
            ("local_ai_tools_auto_train", LOCAL_AI_TOOL_MODEL_IDS),
            ("local_ml_auto_train", LOCAL_ML_MODEL_IDS),
        ):
            try:
                state_store.heartbeat(
                    scheduler_id=scheduler_id,
                    model_ids=model_ids,
                    interval_seconds=AUTO_TRAIN_CHECK_INTERVAL_SECONDS,
                )
            except Exception as exc:
                logger.warning(
                    "deferred model training scheduler heartbeat write failed",
                    scheduler_id=scheduler_id,
                    error=safe_error_text(exc, limit=180),
                )

    async def _ml_auto_train_loop(self) -> None:
        """Retrain local ML and server-side quant tools without blocking trading."""
        # A service restart commonly follows deployment or host maintenance.
        # Let market data and the first analysis rounds establish a stable CPU
        # baseline before launching a full historical training subprocess.
        started_at = getattr(self, "_start_time", None)
        if isinstance(started_at, datetime):
            startup_delay = max(
                TRAINING_STARTUP_GRACE_SECONDS
                - (datetime.now(UTC) - started_at).total_seconds(),
                0.0,
            )
            if startup_delay > 0.0:
                await asyncio.sleep(startup_delay)
        while self._running:
            async def run_ml_step() -> dict[str, Any]:
                try:
                    result = await self._run_local_ml_training_subprocess()
                    if result.get("trained"):
                        logger.info(
                            "local ML signal model auto-trained",
                            sample_count=result.get("sample_count"),
                            new_sample_count=result.get("new_sample_count"),
                        )
                    elif str(result.get("reason") or "") in {
                        "error",
                        "load_samples_error",
                        "timeout",
                    }:
                        logger.warning(
                            "local ML signal auto-train failed",
                            reason=result.get("reason"),
                            error=result.get("error"),
                        )
                    elif result.get("reason") not in {"not_due", "training_in_progress"}:
                        logger.info(
                            "local ML signal auto-train waiting for mature data",
                            reason=result.get("reason"),
                            train_sample_count=result.get("train_sample_count"),
                            train_decision_group_count=result.get("train_decision_group_count"),
                            purged_training_sample_count=result.get(
                                "purged_training_sample_count"
                            ),
                        )
                    return result
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "local ML signal auto-train loop error",
                        error=safe_error_text(exc),
                    )
                    return {
                        "trained": False,
                        "reason": "error",
                        "error": safe_error_text(exc),
                    }

            async def run_local_tools_step() -> dict[str, Any]:
                try:
                    return await self.train_local_ai_tools()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "local AI tools auto-train loop error",
                        error=safe_error_text(exc),
                    )
                    return {
                        "trained": False,
                        "reason": "error",
                        "error": safe_error_text(exc),
                    }

            # Both trainers read the same large history and use the same model
            # tunnel.  Serial execution prevents CPU, DB and 18001 contention.
            if self._auto_training_should_defer():
                self._heartbeat_deferred_training_schedulers()
                await asyncio.sleep(TRAINING_BUSY_RETRY_SECONDS)
                continue
            ml_result = await run_ml_step()
            local_tools_result = await run_local_tools_step()
            await asyncio.sleep(self._auto_train_failure_delay([ml_result, local_tools_result]))

    async def _run_local_ml_training_subprocess(
        self,
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_local_ml_auto_train.py"),
        ]
        if force:
            command.append("--force")
        process = await asyncio.create_subprocess_exec(
            *low_priority_training_command(command),
            cwd=str(PROJECT_ROOT),
            env=training_process_env(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        active_processes = self._training_process_set()
        active_processes.add(process)
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=AUTO_TRAIN_LEASE_STALE_SECONDS,
            )
        except asyncio.CancelledError:
            if process.returncode is None:
                process.kill()
                await process.wait()
            raise
        except TimeoutError:
            process.kill()
            await process.wait()
            self._record_training_subprocess_timeout(
                scheduler_id="local_ml_auto_train",
                model_ids=("local_ml_profit_quality",),
                error="isolated local ML training exceeded its scheduler lease",
            )
            return {
                "trained": False,
                "reason": "timeout",
                "error": "isolated local ML training exceeded its scheduler lease",
                "training_process_isolated": True,
            }
        finally:
            active_processes.discard(process)

        try:
            stdout_text = stdout.decode("utf-8")
            result_frame = next(
                (
                    line.removeprefix(LOCAL_ML_AUTO_TRAIN_RESULT_PREFIX)
                    for line in reversed(stdout_text.splitlines())
                    if line.startswith(LOCAL_ML_AUTO_TRAIN_RESULT_PREFIX)
                ),
                None,
            )
            if result_frame is None:
                raise ValueError("local ML auto-train result frame missing")
            payload = json.loads(result_frame)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            stderr_text = stderr.decode("utf-8", errors="replace").strip()
            result = {
                "trained": False,
                "reason": "error",
                "error": (
                    safe_error_tail(stderr_text, limit=500, fallback="")
                    or safe_error_tail(exc, limit=500)
                ),
                "training_process_isolated": True,
            }
            self._record_training_subprocess_failure(
                scheduler_id=LOCAL_ML_TRAINING_SCHEDULER_ID,
                model_ids=LOCAL_ML_MODEL_IDS,
                error=str(result["error"]),
            )
            return result
        result = (
            dict(payload)
            if isinstance(payload, dict)
            else {
                "trained": False,
                "reason": "error",
                "error": "isolated local ML training response was not an object",
            }
        )
        if process.returncode != 0 and str(result.get("reason") or "") != "error":
            result.update(
                {
                    "trained": False,
                    "reason": "error",
                    "error": safe_error_tail(
                        stderr.decode("utf-8", errors="replace"),
                        limit=500,
                        fallback=(
                            "isolated local ML training exited with code "
                            f"{process.returncode} without stderr"
                        ),
                    ),
                }
            )
        if process.returncode != 0:
            self._record_training_subprocess_failure(
                scheduler_id=LOCAL_ML_TRAINING_SCHEDULER_ID,
                model_ids=LOCAL_ML_MODEL_IDS,
                error=str(result.get("error") or "isolated local ML training failed"),
            )
        result["training_process_isolated"] = True
        return result

    async def train_local_ai_tools(self, *, force: bool = False) -> dict[str, Any]:
        state_store = self._model_training_state()
        gate = getattr(state_store, "training_gate", None)
        if callable(gate):
            gate_result = gate(
                scheduler_id="local_ai_tools_auto_train",
                model_ids=LOCAL_AI_TOOL_MODEL_IDS,
                force=force,
            )
            if (
                not bool(gate_result.get("allowed"))
                and gate_result.get("reason") == "resource_blocked"
            ):
                # A resource circuit must be reopened automatically when the
                # canonical training cursor has moved. Older callers did not
                # persist a fingerprint, so this probe also migrates those
                # rows without weakening the circuit for an unchanged input.
                cursor_probe = await self._run_local_ai_tools_training_cursor_subprocess()
                probe_fingerprint = (
                    training_input_fingerprint(cursor_probe)
                    if cursor_probe.get("reason") == "cursor_probe_complete"
                    else ""
                )
                if probe_fingerprint:
                    gate_result = gate(
                        scheduler_id="local_ai_tools_auto_train",
                        model_ids=LOCAL_AI_TOOL_MODEL_IDS,
                        force=force,
                        input_fingerprint=probe_fingerprint,
                    )
                    gate_result["input_fingerprint"] = probe_fingerprint
                    gate_result["cursor_probe"] = cursor_probe
            if not bool(gate_result.get("allowed")):
                return {
                    "trained": False,
                    "reason": gate_result.get("reason") or "training_gate_blocked",
                    "training_gate": gate_result,
                }
        lease_attempt = state_store.try_acquire_lease(
            scheduler_id="local_ai_tools_auto_train",
            stale_after_seconds=AUTO_TRAIN_LEASE_STALE_SECONDS,
        )
        if not lease_attempt.acquired or lease_attempt.lease is None:
            return {
                "trained": False,
                "reason": lease_attempt.reason,
                "recovered_stale_lease": lease_attempt.recovered_stale_lease,
            }
        lease = lease_attempt.lease
        self._local_tools_active_training_run_id = lease.run_id
        now = datetime.now(UTC)
        try:
            state_store.heartbeat(
                scheduler_id="local_ai_tools_auto_train",
                model_ids=LOCAL_AI_TOOL_MODEL_IDS,
                interval_seconds=AUTO_TRAIN_CHECK_INTERVAL_SECONDS,
            )
            state_store.record_check(
                scheduler_id="local_ai_tools_auto_train",
                model_ids=LOCAL_AI_TOOL_MODEL_IDS,
                run_id=lease.run_id,
                force=force,
            )
        except Exception:
            self._local_tools_active_training_run_id = None
            lease.release()
            raise
        try:
            result = await self._maybe_train_local_ai_tools_process(force=force)
            result.setdefault("training_input_fingerprint", training_input_fingerprint(result))
            failed = str(result.get("reason") or "") in {
                "error",
                "load_samples_error",
                "timeout",
            }
            delay = (
                AUTO_TRAIN_RETRY_INTERVAL_SECONDS if failed else AUTO_TRAIN_CHECK_INTERVAL_SECONDS
            )
            state_store.finish_check(
                scheduler_id="local_ai_tools_auto_train",
                model_ids=LOCAL_AI_TOOL_MODEL_IDS,
                run_id=lease.run_id,
                result=result,
                next_check_at=datetime.now(UTC) + timedelta(seconds=delay),
            )
            return result
        except asyncio.CancelledError:
            state_store.record_exception(
                scheduler_id="local_ai_tools_auto_train",
                model_ids=LOCAL_AI_TOOL_MODEL_IDS,
                run_id=lease.run_id,
                error="training_cancelled",
                next_check_at=now + timedelta(seconds=AUTO_TRAIN_RETRY_INTERVAL_SECONDS),
            )
            raise
        except Exception as exc:
            error = safe_error_text(exc, limit=180)
            state_store.record_exception(
                scheduler_id="local_ai_tools_auto_train",
                model_ids=LOCAL_AI_TOOL_MODEL_IDS,
                run_id=lease.run_id,
                error=error,
                next_check_at=now + timedelta(seconds=AUTO_TRAIN_RETRY_INTERVAL_SECONDS),
            )
            raise
        finally:
            self._local_tools_active_training_run_id = None
            lease.release()

    async def _maybe_train_local_ai_tools_process(self, *, force: bool = False) -> dict[str, Any]:
        """Push fresh history to the server-side profit/time-series/exit models."""
        if not self.local_ai_tools.enabled():
            return {"trained": False, "reason": "disabled"}

        from services.okx_training_gate import okx_training_refresh_gate

        okx_gate = okx_training_refresh_gate()
        if not bool(okx_gate.get("allowed")):
            return {
                "trained": False,
                "reason": okx_gate.get("reason") or "okx_training_refresh_blocked",
                "okx_daily_reconciliation_gate": okx_gate,
                "trade_sample_cursor_policy": CURRENT_TRAINING_EPOCH_POLICY,
            }

        status_probe_error = ""
        try:
            status = await self.local_ai_tools.status()
        except Exception as exc:
            status_probe_error = safe_error_text(exc, limit=180)
            status = {
                "available": False,
                "service_available": False,
                "model_bundle_available": False,
                "status": "status_probe_error",
                "error": status_probe_error,
            }
            logger.warning(
                "local AI tools status probe failed before training; continuing with local cursors",
                error=status_probe_error,
            )
        server_shadow_count = int((status or {}).get("shadow_sample_count") or 0)
        server_trade_count = int((status or {}).get("trade_sample_count") or 0)
        previous_completed_shadow_total = int(
            (status or {}).get("last_trained_completed_shadow_sample_count")
            or (status or {}).get("completed_shadow_sample_count")
            or self._local_tools_last_completed_shadow_count
            or server_shadow_count
            or 0
        )

        if not force:
            cursor_result = await self._run_local_ai_tools_training_cursor_subprocess()
            if str(cursor_result.get("reason") or "") != "cursor_probe_complete":
                return cursor_result
            completed_shadow_total = self._safe_int(
                cursor_result.get("completed_shadow_sample_count"),
                0,
            )
            completed_trade_total = self._safe_int(
                cursor_result.get("completed_trade_sample_count"),
                0,
            )
            completed_group_total = self._safe_int(
                cursor_result.get("completed_training_decision_group_count"),
                0,
            )
            previous_completed_trade_total = int(
                (status or {}).get("last_trained_completed_trade_sample_count")
                or (status or {}).get("completed_trade_sample_count")
                or server_trade_count
                or 0
            )
            new_shadow = max(completed_shadow_total - previous_completed_shadow_total, 0)
            new_trade = max(completed_trade_total - previous_completed_trade_total, 0)
            shadow_training_view_rebased = completed_shadow_total < previous_completed_shadow_total
            learning_only = not bool(
                (status or {}).get("model_bundle_available", (status or {}).get("available"))
            )
            previous_group_total = self._safe_int(
                (status or {}).get("last_trained_completed_training_decision_group_count"),
                0,
            )
            if previous_group_total <= 0:
                previous_group_total = sum(
                    self._safe_int((status or {}).get(field), 0)
                    for field in (
                        "train_decision_group_count",
                        "holdout_decision_group_count",
                        "purged_holdout_decision_group_count",
                        "train_cost_decision_group_count",
                        "holdout_cost_decision_group_count",
                        "purged_cost_holdout_decision_group_count",
                    )
                )
            training_state_cursor = 0
            training_state_shadow_cursor = 0
            training_state_trade_cursor = 0
            training_state_trained_at = None
            state_store = self._model_training_state()
            read_training_state = getattr(state_store, "read", None)
            if callable(read_training_state):
                state_payload = read_training_state()
                model_rows = state_payload.get("models") or {}
                for model_id in LOCAL_AI_TOOL_MODEL_IDS:
                    model_row = model_rows.get(model_id) or {}
                    training_state_cursor = max(
                        training_state_cursor,
                        self._safe_int(
                            (model_row.get("sample_cursor") or {}).get("decision_group"),
                            0,
                        ),
                    )
                    training_state_shadow_cursor = max(
                        training_state_shadow_cursor,
                        self._safe_int(
                            (model_row.get("sample_cursor") or {}).get("shadow"),
                            0,
                        ),
                    )
                    training_state_trade_cursor = max(
                        training_state_trade_cursor,
                        self._safe_int(
                            (model_row.get("sample_cursor") or {}).get("trade"),
                            0,
                        ),
                    )
                    if model_row.get("sample_cursor"):
                        succeeded_at = model_row.get("last_successful_training_at")
                        if not succeeded_at:
                            succeeded_at = next(
                                (
                                    event.get("at")
                                    for event in reversed(model_row.get("history") or [])
                                    if isinstance(event, dict)
                                    and event.get("event") == "succeeded"
                                    and event.get("at")
                                ),
                                None,
                            )
                        if succeeded_at and (
                            training_state_trained_at is None
                            or str(succeeded_at) > str(training_state_trained_at)
                        ):
                            training_state_trained_at = succeeded_at
            previous_completed_shadow_total = max(
                previous_completed_shadow_total,
                training_state_shadow_cursor,
            )
            previous_completed_trade_total = max(
                previous_completed_trade_total,
                training_state_trade_cursor,
            )
            new_shadow = max(completed_shadow_total - previous_completed_shadow_total, 0)
            new_trade = max(completed_trade_total - previous_completed_trade_total, 0)
            shadow_training_view_rebased = (
                completed_shadow_total < previous_completed_shadow_total
            )
            previous_group_total = max(
                previous_group_total,
                training_state_cursor,
            )
            distribution_drift = training_distribution_drift(
                cursor_result.get("training_distribution_profile") or {},
                (status or {}).get("training_distribution_profile") or {},
                threshold=LOCAL_ML_TRAINING_PARAMS.distribution_drift_threshold,
            )
            trigger = decision_group_training_trigger(
                force=False,
                has_artifact=not learning_only,
                completed_group_count=completed_group_total,
                previous_group_count=previous_group_total,
                trained_at=(training_state_trained_at or (status or {}).get("trained_at")),
                now=datetime.now(UTC),
                distribution_drift=distribution_drift,
                batch_threshold=(LOCAL_ML_TRAINING_PARAMS.batch_decision_group_threshold),
                minimum_increment=(LOCAL_ML_TRAINING_PARAMS.minimum_decision_group_increment),
                drift_minimum_increment=(
                    LOCAL_ML_TRAINING_PARAMS.drift_minimum_decision_group_increment
                ),
                maximum_interval_seconds=(
                    LOCAL_ML_TRAINING_PARAMS.maximum_training_interval_seconds
                ),
                batch_growth_fraction=(
                    LOCAL_ML_TRAINING_PARAMS.batch_decision_group_growth_fraction
                ),
                minimum_retraining_interval_seconds=(
                    LOCAL_ML_TRAINING_PARAMS.minimum_retraining_interval_seconds
                ),
            )
            training_policy = {
                "learning_only": learning_only,
                "trigger": trigger["reason"],
                "trigger_contract": trigger,
                "distribution_requirement": "non_empty_train_and_holdout",
                "training_window_policy": CURRENT_TRAINING_EPOCH_POLICY,
                "cursor_source": "completed_training_decision_group_count",
                "trade_cursor_policy": CURRENT_TRAINING_EPOCH_POLICY,
                "process_boundary": "dedicated_training_subprocess",
                "cursor_process_isolated": True,
                "shadow_training_view_rebased": shadow_training_view_rebased,
            }
            if status_probe_error:
                training_policy["status_probe_error"] = status_probe_error
                training_policy["status_probe_fallback"] = "train_when_due_from_local_counts"
            if not trigger["due"]:
                return {
                    "trained": False,
                    "reason": "not_due",
                    "server_shadow_sample_count": server_shadow_count,
                    "completed_shadow_sample_count": completed_shadow_total,
                    "last_trained_completed_shadow_sample_count": (previous_completed_shadow_total),
                    "completed_trade_sample_count": completed_trade_total,
                    "last_trained_completed_trade_sample_count": previous_completed_trade_total,
                    "new_shadow_sample_count": new_shadow,
                    "new_trade_sample_count": new_trade,
                    "completed_training_decision_group_count": completed_group_total,
                    "last_trained_completed_training_decision_group_count": (previous_group_total),
                    "new_decision_group_count": trigger["new_mature_decision_group_count"],
                    "training_policy": training_policy,
                    "training_process_isolated": True,
                }

            active_run_id = getattr(self, "_local_tools_active_training_run_id", None)
            if active_run_id:
                self._model_training_state().start_run(
                    scheduler_id="local_ai_tools_auto_train",
                    model_ids=LOCAL_AI_TOOL_MODEL_IDS,
                    run_id=active_run_id,
                    trigger_reason="training_due",
                    sample_cursor={
                        "shadow": completed_shadow_total,
                        "trade": completed_trade_total,
                        "decision_group": completed_group_total,
                    },
                    timeout_seconds=AUTO_TRAIN_LEASE_STALE_SECONDS,
                )

            result = await self._run_local_ai_tools_training_subprocess()
            reported_shadow_total = result.get("last_trained_completed_shadow_sample_count")
            if reported_shadow_total is None:
                reported_shadow_total = result.get("completed_shadow_sample_count")
            reported_trade_total = result.get("last_trained_completed_trade_sample_count")
            if reported_trade_total is None:
                reported_trade_total = result.get("completed_trade_sample_count")
            reported_group_total = result.get(
                "last_trained_completed_training_decision_group_count"
            )
            if reported_group_total is None:
                reported_group_total = result.get("completed_training_decision_group_count")
            result["training_window_completed_shadow_sample_count"] = self._safe_int(
                reported_shadow_total,
                0,
            )
            result["training_window_completed_trade_sample_count"] = self._safe_int(
                reported_trade_total,
                0,
            )
            result["training_window_completed_decision_group_count"] = self._safe_int(
                reported_group_total,
                0,
            )
            # The training subprocess receives a bounded fitting window. Its row/group
            # totals are diagnostics, not cumulative scheduler cursors. Advancing with
            # the lightweight probe prevents a rejected challenger from retraining the
            # same historical window every check.
            authoritative_shadow_total = completed_shadow_total
            authoritative_trade_total = completed_trade_total
            authoritative_group_total = completed_group_total
            result["completed_shadow_sample_count"] = completed_shadow_total
            result["completed_trade_sample_count"] = completed_trade_total
            result["completed_training_decision_group_count"] = completed_group_total
            result["new_shadow_sample_count"] = new_shadow
            result["new_trade_sample_count"] = new_trade
            result["new_decision_group_count"] = trigger["new_mature_decision_group_count"]
            result["training_policy"] = training_policy
            result["training_process_isolated"] = True
            if result.get("trained"):
                result["last_trained_completed_shadow_sample_count"] = authoritative_shadow_total
                result["last_trained_completed_trade_sample_count"] = authoritative_trade_total
                result["last_trained_completed_training_decision_group_count"] = (
                    authoritative_group_total
                )
                self._local_tools_last_completed_shadow_count = authoritative_shadow_total
            return result

        result = await self._run_local_ai_tools_training_subprocess()
        result["training_process_isolated"] = True
        result["training_policy"] = {
            "trigger": "forced",
            "process_boundary": "dedicated_training_subprocess",
            "concurrency_policy": "exclusive_local_ai_tools_training_process_lock",
            "training_window_policy": "all_current_clean_samples",
        }
        if status_probe_error:
            result["training_policy"]["status_probe_error"] = status_probe_error
            result["training_policy"]["status_probe_fallback"] = "train_in_isolated_process"
        return result

    async def _run_local_ai_tools_training_subprocess(self) -> dict[str, Any]:
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_local_ai_tools_models.py"),
            "--training-mode",
            "walk_forward",
            "--persist-artifact",
            "--confirm-phase3-rebuild",
        ]
        process = await asyncio.create_subprocess_exec(
            *low_priority_training_command(command),
            cwd=str(PROJECT_ROOT),
            env=training_process_env(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        active_processes = self._training_process_set()
        active_processes.add(process)
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=AUTO_TRAIN_LEASE_STALE_SECONDS,
            )
        except asyncio.CancelledError:
            if process.returncode is None:
                process.kill()
                await process.wait()
            raise
        except TimeoutError:
            process.kill()
            await process.wait()
            self._record_training_subprocess_timeout(
                scheduler_id="local_ai_tools_auto_train",
                model_ids=LOCAL_AI_TOOL_MODEL_IDS,
                error="isolated local AI tools training exceeded its scheduler lease",
                run_id=getattr(self, "_local_tools_active_training_run_id", None),
            )
            return {
                "trained": False,
                "reason": "timeout",
                "error": "isolated local AI tools training exceeded its scheduler lease",
            }
        finally:
            active_processes.discard(process)
        if process.returncode != 0:
            return {
                "trained": False,
                "reason": "error",
                "error": safe_error_tail(
                    stderr.decode("utf-8", errors="replace"),
                    limit=500,
                    fallback=(
                        "isolated local AI tools training exited with code "
                        f"{process.returncode} without stderr"
                    ),
                ),
            }
        try:
            stdout_text = stdout.decode("utf-8")
            result_frame = next(
                (
                    line.removeprefix(LOCAL_AI_TOOLS_TRAIN_RESULT_PREFIX)
                    for line in reversed(stdout_text.splitlines())
                    if line.startswith(LOCAL_AI_TOOLS_TRAIN_RESULT_PREFIX)
                ),
                None,
            )
            if result_frame is None:
                raise ValueError("local AI tools training result frame missing")
            payload = json.loads(result_frame)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            return {
                "trained": False,
                "reason": "invalid_training_response",
                "error": safe_error_text(exc, limit=180),
            }
        return (
            dict(payload)
            if isinstance(payload, dict)
            else {
                "trained": False,
                "reason": "invalid_training_response",
            }
        )

    async def _run_local_ai_tools_training_cursor_subprocess(self) -> dict[str, Any]:
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_local_ai_tools_training_cursors.py"),
        ]
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(PROJECT_ROOT),
            env=training_process_env(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        active_processes = self._training_process_set()
        active_processes.add(process)
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=AUTO_TRAIN_LEASE_STALE_SECONDS,
            )
        except asyncio.CancelledError:
            if process.returncode is None:
                process.kill()
                await process.wait()
            raise
        except TimeoutError:
            process.kill()
            await process.wait()
            return {
                "trained": False,
                "reason": "timeout",
                "error": "isolated Local AI cursor probe exceeded its scheduler lease",
                "training_process_isolated": True,
            }
        finally:
            active_processes.discard(process)
        if process.returncode != 0:
            return {
                "trained": False,
                "reason": "error",
                "error": safe_error_tail(
                    stderr.decode("utf-8", errors="replace"),
                    limit=500,
                    fallback=(
                        "isolated Local AI cursor probe exited with code "
                        f"{process.returncode} without stderr"
                    ),
                ),
                "training_process_isolated": True,
            }
        try:
            payload = json.loads(stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return {
                "trained": False,
                "reason": "error",
                "error": safe_error_text(exc, limit=180),
                "training_process_isolated": True,
            }
        result = (
            dict(payload)
            if isinstance(payload, dict)
            else {
                "trained": False,
                "reason": "error",
                "error": "isolated Local AI cursor response was not an object",
            }
        )
        result["training_process_isolated"] = True
        return result

    async def _completed_shadow_backtest_total(self) -> int:
        async with get_session_ctx() as session:
            result = await session.execute(
                select(func.count(ShadowBacktest.id)).where(
                    ShadowBacktest.status == "completed",
                    ShadowBacktest.long_return_pct.is_not(None),
                    ShadowBacktest.short_return_pct.is_not(None),
                )
            )
            return int(result.scalar() or 0)
