from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from services.ml_signal_service import MLSignalService
from services.model_artifact_registry import ModelArtifactRegistry
from services.model_training_state import (
    LOCAL_AI_TOOL_MODEL_IDS,
    LOCAL_ML_MODEL_IDS,
    MODEL_TRAINING_STATE_VERSION,
    ModelTrainingStateStore,
)


def test_pytest_scheduler_state_isolated_from_runtime_data() -> None:
    from config.settings import settings
    from services import ml_signal_service, trading_service
    from web_dashboard.api import data_collection

    runtime_path = (settings.data_dir / "model_training_scheduler_state.json").resolve()
    test_store = ml_signal_service.MODEL_TRAINING_STATE_STORE

    assert test_store.path.resolve() != runtime_path
    assert trading_service.MODEL_TRAINING_STATE_STORE is test_store
    assert data_collection.MODEL_TRAINING_STATE_STORE is test_store


def test_state_persists_auditable_timeline_for_each_model(tmp_path) -> None:
    now = [datetime(2026, 7, 12, 1, 0, tzinfo=UTC)]
    store = ModelTrainingStateStore(
        tmp_path / "model_training_state.json",
        now_provider=lambda: now[0],
    )
    attempt = store.try_acquire_lease(
        scheduler_id="local_ai_tools_auto_train",
        stale_after_seconds=3600,
    )
    assert attempt.acquired is True
    assert attempt.lease is not None

    run_id = attempt.lease.run_id
    store.heartbeat(
        scheduler_id="platform_model_training_loop",
        model_ids=LOCAL_AI_TOOL_MODEL_IDS,
        interval_seconds=1800,
    )
    store.record_check(
        scheduler_id="local_ai_tools_auto_train",
        model_ids=LOCAL_AI_TOOL_MODEL_IDS,
        run_id=run_id,
        force=False,
    )
    store.start_run(
        scheduler_id="local_ai_tools_auto_train",
        model_ids=LOCAL_AI_TOOL_MODEL_IDS,
        run_id=run_id,
        trigger_reason="training_due",
        sample_cursor={"shadow": 20000, "trade": 341},
    )
    now[0] += timedelta(seconds=12)
    store.finish_check(
        scheduler_id="local_ai_tools_auto_train",
        model_ids=LOCAL_AI_TOOL_MODEL_IDS,
        run_id=run_id,
        result={
            "trained": True,
            "reason": "trained",
            "last_trained_completed_sample_count": 37655,
            "completed_trade_sample_count": 341,
            "artifact_persisted": True,
        },
        next_check_at=now[0] + timedelta(minutes=30),
    )
    attempt.lease.release()

    persisted = json.loads(store.path.read_text(encoding="utf-8"))
    assert persisted["version"] == MODEL_TRAINING_STATE_VERSION
    assert set(persisted["models"]) == set(LOCAL_AI_TOOL_MODEL_IDS)
    for model_id in LOCAL_AI_TOOL_MODEL_IDS:
        row = persisted["models"][model_id]
        assert row["state"] == "succeeded"
        assert row["last_successful_training_at"] == now[0].isoformat()
        assert row["sample_cursor"] == {"shadow": 37655, "trade": 341}
        assert row["last_run_id"] == run_id
        assert row["last_result"]["artifact_persisted"] is True
        assert [event["event"] for event in row["history"]] == ["started", "succeeded"]


def test_resource_failures_backoff_and_open_circuit_until_input_changes(tmp_path) -> None:
    now = [datetime(2026, 8, 28, 1, 0, tzinfo=UTC)]
    store = ModelTrainingStateStore(
        tmp_path / "model_training_state.json",
        now_provider=lambda: now[0],
    )

    def failed_run() -> None:
        attempt = store.try_acquire_lease(
            scheduler_id="local_ai_tools_auto_train",
            stale_after_seconds=3600,
        )
        assert attempt.lease is not None
        run_id = attempt.lease.run_id
        store.record_check(
            scheduler_id="local_ai_tools_auto_train",
            model_ids=LOCAL_AI_TOOL_MODEL_IDS,
            run_id=run_id,
            force=False,
        )
        store.start_run(
            scheduler_id="local_ai_tools_auto_train",
            model_ids=LOCAL_AI_TOOL_MODEL_IDS,
            run_id=run_id,
            trigger_reason="training_due",
        )
        store.finish_check(
            scheduler_id="local_ai_tools_auto_train",
            model_ids=LOCAL_AI_TOOL_MODEL_IDS,
            run_id=run_id,
            result={
                "trained": False,
                "reason": "error",
                "error": "MemoryError: unable to allocate tensor",
                "training_input_fingerprint": "same-input",
            },
            next_check_at=now[0] + timedelta(seconds=5),
        )
        attempt.lease.release()

    failed_run()
    row = store.read()["models"][LOCAL_AI_TOOL_MODEL_IDS[0]]
    assert row["resource_failure_count"] == 1
    assert row["resource_error_class"] == "resource"
    assert row["next_check_at"] == "2026-08-28T01:05:00+00:00"
    assert store.training_gate(
        scheduler_id="local_ai_tools_auto_train",
        model_ids=LOCAL_AI_TOOL_MODEL_IDS,
    )["reason"] == "retry_backoff_active"

    now[0] += timedelta(minutes=5)
    failed_run()
    now[0] += timedelta(minutes=15)
    failed_run()
    row = store.read()["models"][LOCAL_AI_TOOL_MODEL_IDS[0]]
    assert row["state"] == "resource_blocked"
    assert row["resource_failure_count"] == 3
    blocked = store.training_gate(
        scheduler_id="local_ai_tools_auto_train",
        model_ids=LOCAL_AI_TOOL_MODEL_IDS,
        force=True,
        input_fingerprint="same-input",
    )
    assert blocked["allowed"] is False
    assert blocked["reason"] == "resource_blocked"
    reopened = store.training_gate(
        scheduler_id="local_ai_tools_auto_train",
        model_ids=LOCAL_AI_TOOL_MODEL_IDS,
        force=True,
        input_fingerprint="new-input",
    )
    assert reopened["allowed"] is True
    assert reopened["reason"] == "resource_circuit_reset_new_input"


def test_skipped_check_does_not_advance_last_successful_training_cursor(tmp_path) -> None:
    now = [datetime(2026, 7, 27, 1, 0, tzinfo=UTC)]
    store = ModelTrainingStateStore(
        tmp_path / "model_training_state.json",
        now_provider=lambda: now[0],
    )
    first = store.try_acquire_lease(
        scheduler_id="local_ai_tools_auto_train",
        stale_after_seconds=3600,
    )
    assert first.lease is not None
    store.record_check(
        scheduler_id="local_ai_tools_auto_train",
        model_ids=LOCAL_AI_TOOL_MODEL_IDS,
        run_id=first.lease.run_id,
        force=False,
    )
    store.start_run(
        scheduler_id="local_ai_tools_auto_train",
        model_ids=LOCAL_AI_TOOL_MODEL_IDS,
        run_id=first.lease.run_id,
        trigger_reason="mature_decision_group_batch",
        sample_cursor={"shadow": 100, "trade": 50, "decision_group": 150},
    )
    store.finish_check(
        scheduler_id="local_ai_tools_auto_train",
        model_ids=LOCAL_AI_TOOL_MODEL_IDS,
        run_id=first.lease.run_id,
        result={
            "trained": True,
            "last_trained_completed_shadow_sample_count": 100,
            "last_trained_completed_trade_sample_count": 50,
            "last_trained_completed_training_decision_group_count": 150,
        },
        next_check_at=now[0] + timedelta(minutes=30),
    )
    first.lease.release()

    second = store.try_acquire_lease(
        scheduler_id="local_ai_tools_auto_train",
        stale_after_seconds=3600,
    )
    assert second.lease is not None
    store.record_check(
        scheduler_id="local_ai_tools_auto_train",
        model_ids=LOCAL_AI_TOOL_MODEL_IDS,
        run_id=second.lease.run_id,
        force=False,
    )
    store.finish_check(
        scheduler_id="local_ai_tools_auto_train",
        model_ids=LOCAL_AI_TOOL_MODEL_IDS,
        run_id=second.lease.run_id,
        result={
            "trained": False,
            "reason": "not_due",
            "completed_shadow_sample_count": 101,
            "completed_trade_sample_count": 50,
            "completed_training_decision_group_count": 151,
        },
        next_check_at=now[0] + timedelta(minutes=30),
    )

    for row in store.read()["models"].values():
        assert row["sample_cursor"] == {
            "shadow": 100,
            "trade": 50,
            "decision_group": 150,
        }


def test_missing_state_is_unavailable_until_first_persistent_heartbeat(tmp_path) -> None:
    store = ModelTrainingStateStore(tmp_path / "model_training_state.json")

    missing = store.read()
    assert missing["status"] == "unavailable"
    assert missing["state_file_available"] is False

    store.heartbeat(
        scheduler_id="platform_model_training_loop",
        model_ids=LOCAL_ML_MODEL_IDS,
        interval_seconds=1800,
    )
    available = store.read()
    assert available["status"] == "ok"
    assert available["state_file_available"] is True


def test_cross_process_lease_blocks_duplicate_training_and_recovers_dead_owner(tmp_path) -> None:
    now = [datetime(2026, 7, 12, 1, 0, tzinfo=UTC)]
    store = ModelTrainingStateStore(
        tmp_path / "model_training_state.json",
        now_provider=lambda: now[0],
    )
    first = store.try_acquire_lease(
        scheduler_id="local_ml_auto_train",
        stale_after_seconds=3600,
    )
    second = store.try_acquire_lease(
        scheduler_id="local_ml_auto_train",
        stale_after_seconds=3600,
    )

    assert first.acquired is True
    assert second.acquired is False
    assert second.reason == "training_in_progress"
    assert first.lease is not None
    now[0] += timedelta(hours=2)
    still_running = store.try_acquire_lease(
        scheduler_id="local_ml_auto_train",
        stale_after_seconds=60,
    )
    assert still_running.acquired is False
    assert still_running.reason == "training_in_progress"
    first.lease.release()

    lease_path = store.lock_dir / "local_ml_auto_train.lease"
    lease_path.write_text(
        json.dumps(
            {
                "token": "dead",
                "run_id": "dead-run",
                "scheduler_id": "local_ml_auto_train",
                "owner_pid": 2147483647,
                "owner_host": store.hostname,
                "acquired_at": datetime.now(UTC).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    recovered = store.try_acquire_lease(
        scheduler_id="local_ml_auto_train",
        stale_after_seconds=3600,
    )
    assert recovered.acquired is True
    assert recovered.recovered_stale_lease is True
    assert recovered.lease is not None
    recovered.lease.release()


def test_recovery_marks_interrupted_training_and_preserves_history(tmp_path) -> None:
    store = ModelTrainingStateStore(tmp_path / "model_training_state.json")
    run_id = "interrupted-run"
    store.record_check(
        scheduler_id="local_ml_auto_train",
        model_ids=LOCAL_ML_MODEL_IDS,
        run_id=run_id,
        force=False,
    )
    store.start_run(
        scheduler_id="local_ml_auto_train",
        model_ids=LOCAL_ML_MODEL_IDS,
        run_id=run_id,
        trigger_reason="training_due",
    )
    payload = json.loads(store.path.read_text(encoding="utf-8"))
    payload["models"][LOCAL_ML_MODEL_IDS[0]]["owner_pid"] = 2147483647
    store.path.write_text(json.dumps(payload), encoding="utf-8")

    recovered = store.recover_interrupted_runs()
    row = store.read()["models"][LOCAL_ML_MODEL_IDS[0]]

    assert recovered == [LOCAL_ML_MODEL_IDS[0]]
    assert row["state"] == "interrupted"
    assert row["last_error"] == "training_process_interrupted"
    assert row["retry_count"] == 1
    assert row["next_check_at"] is not None
    assert row["history"][-1]["event"] == "interrupted"
    assert row.get("last_successful_training_at") is None


def test_recovery_schedules_existing_interrupted_run_without_duplicate_event(tmp_path) -> None:
    now = [datetime(2026, 8, 22, 1, 0, tzinfo=UTC)]
    store = ModelTrainingStateStore(tmp_path / "model_training_state.json", now_provider=lambda: now[0])
    payload = store.read()
    payload["models"][LOCAL_ML_MODEL_IDS[0]] = {
        "model_id": LOCAL_ML_MODEL_IDS[0],
        "state": "interrupted",
        "last_error": "training_process_interrupted",
        "next_check_at": None,
        "retry_count": 1,
        "history": [{"event": "interrupted", "run_id": "old-run"}],
    }
    store.path.write_text(json.dumps(payload), encoding="utf-8")

    scheduled = store.recover_interrupted_runs(retry_after_seconds=120)
    row = store.read()["models"][LOCAL_ML_MODEL_IDS[0]]

    assert scheduled == [LOCAL_ML_MODEL_IDS[0]]
    assert row["next_check_at"] == "2026-08-22T01:02:00+00:00"
    assert [event["event"] for event in row["history"]] == ["interrupted", "retry_scheduled"]

    assert store.recover_interrupted_runs(retry_after_seconds=120) == []


def test_failed_model_makes_fresh_scheduler_state_error(tmp_path) -> None:
    store = ModelTrainingStateStore(tmp_path / "model_training_state.json")
    run_id = "failed-run"
    store.heartbeat(
        scheduler_id="platform_model_training_loop",
        model_ids=LOCAL_AI_TOOL_MODEL_IDS,
        interval_seconds=1800,
    )
    store.record_check(
        scheduler_id="local_ai_tools_auto_train",
        model_ids=LOCAL_AI_TOOL_MODEL_IDS,
        run_id=run_id,
        force=False,
    )
    store.record_exception(
        scheduler_id="local_ai_tools_auto_train",
        model_ids=LOCAL_AI_TOOL_MODEL_IDS,
        run_id=run_id,
        error="remote training failed",
        next_check_at=datetime.now(UTC) + timedelta(minutes=5),
    )

    status = store.read()

    assert status["heartbeat_stale"] is False
    assert status["status"] == "error"
    assert status["model_state_healthy"] is False
    assert status["failed_model_ids"] == sorted(LOCAL_AI_TOOL_MODEL_IDS)
    assert status["unhealthy_model_ids"] == sorted(LOCAL_AI_TOOL_MODEL_IDS)
    assert status["model_state_counts"] == {"failed": len(LOCAL_AI_TOOL_MODEL_IDS)}


def test_interrupted_model_makes_fresh_scheduler_state_warning(tmp_path) -> None:
    store = ModelTrainingStateStore(tmp_path / "model_training_state.json")
    run_id = "interrupted-run"
    store.heartbeat(
        scheduler_id="platform_model_training_loop",
        model_ids=LOCAL_ML_MODEL_IDS,
        interval_seconds=1800,
    )
    store.record_check(
        scheduler_id="local_ml_auto_train",
        model_ids=LOCAL_ML_MODEL_IDS,
        run_id=run_id,
        force=False,
    )
    store.start_run(
        scheduler_id="local_ml_auto_train",
        model_ids=LOCAL_ML_MODEL_IDS,
        run_id=run_id,
        trigger_reason="training_due",
    )
    payload = json.loads(store.path.read_text(encoding="utf-8"))
    payload["models"][LOCAL_ML_MODEL_IDS[0]]["owner_pid"] = 2147483647
    store.path.write_text(json.dumps(payload), encoding="utf-8")
    store.recover_interrupted_runs()

    status = store.read()

    assert status["heartbeat_stale"] is False
    assert status["status"] == "warning"
    assert status["interrupted_model_ids"] == [LOCAL_ML_MODEL_IDS[0]]
    assert status["model_state_healthy"] is False


def test_scheduler_heartbeat_becomes_warning_after_cycle_is_missed(tmp_path) -> None:
    now = [datetime(2026, 7, 12, 1, 0, tzinfo=UTC)]
    store = ModelTrainingStateStore(
        tmp_path / "model_training_state.json",
        now_provider=lambda: now[0],
    )
    store.heartbeat(
        scheduler_id="platform_model_training_loop",
        model_ids=LOCAL_ML_MODEL_IDS,
        interval_seconds=120,
    )
    assert store.read()["heartbeat_stale"] is False

    now[0] += timedelta(seconds=181)
    status = store.read()
    assert status["status"] == "warning"
    assert status["heartbeat_stale"] is True
    assert status["stale_scheduler_ids"] == ["platform_model_training_loop"]


def test_stale_sub_scheduler_is_superseded_by_fresh_covering_scheduler(tmp_path) -> None:
    now = [datetime(2026, 7, 12, 1, 0, tzinfo=UTC)]
    store = ModelTrainingStateStore(
        tmp_path / "model_training_state.json",
        now_provider=lambda: now[0],
    )
    store.heartbeat(
        scheduler_id="local_ai_tools_auto_train",
        model_ids=LOCAL_AI_TOOL_MODEL_IDS,
        interval_seconds=120,
    )
    now[0] += timedelta(seconds=181)
    store.heartbeat(
        scheduler_id="platform_model_training_loop",
        model_ids=(*LOCAL_ML_MODEL_IDS, *LOCAL_AI_TOOL_MODEL_IDS),
        interval_seconds=120,
    )

    status = store.read()
    legacy = status["schedulers"]["local_ai_tools_auto_train"]

    assert legacy["heartbeat_stale"] is True
    assert legacy["heartbeat_superseded"] is True
    assert legacy["heartbeat_effective_stale"] is False
    assert legacy["heartbeat_superseded_by"] == ["platform_model_training_loop"]
    assert status["heartbeat_stale"] is False
    assert status["stale_scheduler_ids"] == []
    assert status["superseded_scheduler_ids"] == ["local_ai_tools_auto_train"]
    assert status["status"] == "ok"


def test_running_model_timeout_is_observable_without_stealing_live_lease(tmp_path) -> None:
    now = [datetime(2026, 7, 12, 1, 0, tzinfo=UTC)]
    store = ModelTrainingStateStore(
        tmp_path / "model_training_state.json",
        now_provider=lambda: now[0],
    )
    attempt = store.try_acquire_lease(
        scheduler_id="local_ml_auto_train",
        stale_after_seconds=60,
    )
    assert attempt.lease is not None
    store.record_check(
        scheduler_id="local_ml_auto_train",
        model_ids=LOCAL_ML_MODEL_IDS,
        run_id=attempt.lease.run_id,
        force=False,
    )
    store.start_run(
        scheduler_id="local_ml_auto_train",
        model_ids=LOCAL_ML_MODEL_IDS,
        run_id=attempt.lease.run_id,
        trigger_reason="training_due",
        timeout_seconds=60,
    )
    now[0] += timedelta(seconds=61)

    status = store.read()
    duplicate = store.try_acquire_lease(
        scheduler_id="local_ml_auto_train",
        stale_after_seconds=60,
    )

    assert status["status"] == "warning"
    assert status["training_timeout_exceeded"] is True
    assert status["timed_out_model_ids"] == [LOCAL_ML_MODEL_IDS[0]]
    assert duplicate.acquired is False
    attempt.lease.release()


def test_unreadable_state_is_observable_and_blocks_mutation(tmp_path) -> None:
    path = tmp_path / "model_training_state.json"
    path.write_text("not-json", encoding="utf-8")
    store = ModelTrainingStateStore(path)

    status = store.read()
    assert status["status"] == "error"
    assert status["error"] == "state_read_failed:JSONDecodeError"
    with pytest.raises(RuntimeError, match="state is unreadable"):
        store.heartbeat(
            scheduler_id="platform_model_training_loop",
            model_ids=LOCAL_ML_MODEL_IDS,
            interval_seconds=1800,
        )


def test_ml_status_reads_timeline_written_by_another_service_instance(tmp_path) -> None:
    store = ModelTrainingStateStore(tmp_path / "model_training_state.json")
    run_id = "persisted-check"
    store.record_check(
        scheduler_id="local_ml_auto_train",
        model_ids=LOCAL_ML_MODEL_IDS,
        run_id=run_id,
        force=False,
    )
    store.finish_check(
        scheduler_id="local_ml_auto_train",
        model_ids=LOCAL_ML_MODEL_IDS,
        run_id=run_id,
        result={"trained": False, "reason": "not_due", "new_sample_count": 17},
        next_check_at=datetime.now(UTC) + timedelta(minutes=30),
    )
    registry = ModelArtifactRegistry(
        root=tmp_path / "model_artifacts",
        model_id=LOCAL_ML_MODEL_IDS[0],
    )

    status = MLSignalService(
        artifact_registry=registry,
        training_state_store=store,
    ).status()

    assert status["auto_training"] is False
    assert status["auto_train_last_result"]["reason"] == "not_due"
    assert status["auto_train_last_result"]["new_sample_count"] == 17
    assert status["auto_train_persistent_state"]["state"] == "skipped"


@pytest.mark.asyncio
async def test_ml_auto_train_respects_cross_process_lease(tmp_path) -> None:
    store = ModelTrainingStateStore(tmp_path / "model_training_state.json")
    lease = store.try_acquire_lease(
        scheduler_id="local_ml_auto_train",
        stale_after_seconds=3600,
    )
    assert lease.lease is not None
    service = MLSignalService(
        artifact_registry=ModelArtifactRegistry(
            root=tmp_path / "model_artifacts",
            model_id=LOCAL_ML_MODEL_IDS[0],
        ),
        training_state_store=store,
    )

    result = await service.maybe_auto_train(force=True)

    assert result["trained"] is False
    assert result["reason"] == "training_in_progress"
    lease.lease.release()


@pytest.mark.asyncio
async def test_ml_auto_train_releases_lease_when_state_write_fails(tmp_path) -> None:
    state_path = tmp_path / "model_training_state.json"
    state_path.write_text("not-json", encoding="utf-8")
    store = ModelTrainingStateStore(state_path)
    service = MLSignalService(
        artifact_registry=ModelArtifactRegistry(
            root=tmp_path / "model_artifacts",
            model_id=LOCAL_ML_MODEL_IDS[0],
        ),
        training_state_store=store,
    )

    with pytest.raises(RuntimeError, match="state is unreadable"):
        await service.maybe_auto_train(force=True)

    assert not (store.lock_dir / "local_ml_auto_train.lease").exists()
