from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from scripts import run_system_audit_snapshot
from web_dashboard.api import system_audit


@pytest.fixture(autouse=True)
def _reset_system_audit_process_state() -> None:
    system_audit._system_audit_status_cache = None
    system_audit._system_audit_refresh_task = None
    system_audit._system_audit_subprocess_task = None


def test_system_audit_runner_bootstraps_online_runtime_before_settings_imports() -> None:
    source = Path(run_system_audit_snapshot.__file__).read_text(encoding="utf-8")

    bootstrap_index = source.index("load_runtime_env_files(project_root=ROOT)")
    settings_heavy_import_index = source.index("from web_dashboard.api.system_audit import")
    assert bootstrap_index < settings_heavy_import_index
    assert "drop_privileges_to_runtime_user_if_needed(project_root=ROOT)" in source


@pytest.mark.asyncio
async def test_cold_system_audit_returns_warming_payload_and_schedules_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduled = False

    def fake_schedule() -> None:
        nonlocal scheduled
        scheduled = True

    async def unexpected_blocking_refresh(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("cold API request must not await the full audit")

    monkeypatch.setattr(system_audit, "_cached_system_audit_status", lambda: None)
    monkeypatch.setattr(system_audit, "_schedule_system_audit_refresh", fake_schedule)
    monkeypatch.setattr(
        system_audit,
        "refresh_system_audit_snapshot",
        unexpected_blocking_refresh,
    )

    payload = await system_audit.system_audit_status()

    assert scheduled is True
    assert payload["status"] == "warming"
    assert payload["cards"] == []
    assert payload["cache"]["hit"] is False
    assert payload["cache"]["refresh_in_background"] is True


@pytest.mark.asyncio
async def test_system_audit_force_refresh_schedules_even_when_snapshot_is_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked_at = datetime.now(UTC)
    scheduled = False

    def fake_schedule() -> None:
        nonlocal scheduled
        scheduled = True

    monkeypatch.setattr(
        system_audit,
        "_cached_system_audit_status",
        lambda: (checked_at, {"checked_at": checked_at.isoformat(), "status": "warning"}),
    )
    monkeypatch.setattr(system_audit, "_schedule_system_audit_refresh", fake_schedule)
    monkeypatch.setattr(system_audit.settings, "system_audit_history_interval_seconds", 900)

    payload = await system_audit.system_audit_status(refresh=True)

    assert scheduled is True
    assert payload["cache"]["refresh_requested"] is True
    assert payload["cache"]["refresh_in_background"] is True


@pytest.mark.asyncio
async def test_audit_reads_persisted_data_collection_snapshot_before_warming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked_at = datetime.now(UTC)
    persisted = {
        "checked_at": checked_at.isoformat(),
        "status": "ok",
        "source": "dashboard.data_collection",
        "training": {"local_ai_tools": {"status": "ready"}},
    }

    monkeypatch.setattr(
        system_audit.data_collection_api,
        "load_persisted_data_collection_status",
        lambda include_feature_coverage=False: (checked_at, persisted),
    )

    async def unexpected_warming_read(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("persisted snapshot should avoid the warming placeholder")

    monkeypatch.setattr(
        system_audit.data_collection_api,
        "get_data_collection_status",
        unexpected_warming_read,
    )

    payload = await system_audit._data_collection_status_for_audit()

    assert payload["status"] == "ok"
    assert payload["cache"]["persisted_snapshot"] is True
    assert payload["training"]["local_ai_tools"]["status"] == "ready"


@pytest.mark.asyncio
async def test_audit_cold_start_waits_for_one_bounded_data_collection_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        system_audit.data_collection_api,
        "load_persisted_data_collection_status",
        lambda include_feature_coverage=False: None,
    )

    async def fake_get_status(
        include_feature_coverage: bool = True,
        *,
        start_background_refresh: bool = True,
        wait_for_initial_refresh: bool = False,
    ) -> dict[str, Any]:
        calls.append(
            {
                "include_feature_coverage": include_feature_coverage,
                "start_background_refresh": start_background_refresh,
                "wait_for_initial_refresh": wait_for_initial_refresh,
            }
        )
        return {"status": "ok", "cache": {"cold_start": False}}

    monkeypatch.setattr(
        system_audit.data_collection_api,
        "get_data_collection_status",
        fake_get_status,
    )

    payload = await system_audit._data_collection_status_for_audit()

    assert payload["status"] == "ok"
    assert calls == [
        {
            "include_feature_coverage": False,
            "start_background_refresh": True,
            "wait_for_initial_refresh": True,
        }
    ]


def test_warming_snapshot_is_not_reused_as_a_durable_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot_path = tmp_path / "system_audit_latest.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "checked_at": datetime.now(UTC).isoformat(),
                "status": "warming",
                "summary": {},
                "cards": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(system_audit, "_latest_audit_path", lambda: snapshot_path)

    assert system_audit._load_latest_audit_snapshot() is None


class _CompletedProcess:
    def __init__(self, stdout: bytes, stderr: bytes = b"") -> None:
        self.returncode: int | None = 0
        self.stdout = stdout
        self.stderr = stderr
        self.killed = False
        self.waited = False

    async def communicate(self) -> tuple[bytes, bytes]:
        return self.stdout, self.stderr

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        self.waited = True
        return int(self.returncode or 0)


@pytest.mark.asyncio
async def test_system_audit_subprocess_reloads_matching_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checked_at = datetime.now(UTC).isoformat()
    payload = {
        "checked_at": checked_at,
        "status": "warning",
        "summary": {"cards": 27},
    }
    snapshot_path = tmp_path / "system_audit_latest.json"
    snapshot_path.write_text(json.dumps(payload), encoding="utf-8")
    frame = {
        "ok": True,
        "checked_at": checked_at,
        "status": "warning",
    }
    process = _CompletedProcess(
        (
            "audit log\n"
            + system_audit.SYSTEM_AUDIT_RUNNER_RESULT_PREFIX
            + json.dumps(frame)
            + "\n"
        ).encode()
    )
    captured: dict[str, Any] = {}

    async def fake_create_subprocess_exec(*args: str, **kwargs: Any) -> _CompletedProcess:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return process

    monkeypatch.setattr(system_audit, "_latest_audit_path", lambda: snapshot_path)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    result = await system_audit._run_system_audit_subprocess_once(
        record_history=False,
        source="test_refresh",
    )

    assert result == payload
    assert captured["args"][:2] == (
        system_audit.sys.executable,
        str(system_audit.SYSTEM_AUDIT_RUNNER_PATH),
    )
    assert captured["args"][-3:] == ("--source", "test_refresh", "--no-record-history")
    assert captured["kwargs"]["cwd"] == str(system_audit.PROJECT_ROOT)
    assert captured["kwargs"]["env"]["PYTHONIOENCODING"] == "utf-8"
    assert captured["kwargs"]["env"]["MALLOC_ARENA_MAX"] == "2"
    assert captured["kwargs"]["env"]["OMP_NUM_THREADS"] == "1"
    assert captured["kwargs"]["env"]["OPENBLAS_NUM_THREADS"] == "1"
    assert captured["kwargs"]["start_new_session"] is (system_audit.os.name != "nt")
    assert "preexec_fn" not in captured["kwargs"]
    assert system_audit._cached_system_audit_status() == (
        datetime.fromisoformat(checked_at),
        payload,
    )


@pytest.mark.asyncio
async def test_system_audit_subprocess_timeout_kills_and_reaps_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HangingProcess(_CompletedProcess):
        def __init__(self) -> None:
            super().__init__(b"")
            self.returncode = None

        async def communicate(self) -> tuple[bytes, bytes]:
            await asyncio.sleep(60)
            return b"", b""

    process = HangingProcess()

    async def fake_create_subprocess_exec(*_args: str, **_kwargs: Any) -> HangingProcess:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(system_audit, "SYSTEM_AUDIT_SUBPROCESS_TIMEOUT_SECONDS", 0.001)

    with pytest.raises(TimeoutError, match="isolated system audit exceeded"):
        await system_audit._run_system_audit_subprocess_once(
            record_history=True,
            source="timeout_test",
        )

    assert process.killed is True
    assert process.waited is True


@pytest.mark.asyncio
async def test_system_audit_subprocess_surfaces_startup_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _CompletedProcess(b"", b"runner import failed")
    process.returncode = 1

    async def fake_create_subprocess_exec(*_args: str, **_kwargs: Any) -> _CompletedProcess:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    with pytest.raises(RuntimeError, match="runner import failed"):
        await system_audit._run_system_audit_subprocess_once(
            record_history=True,
            source="startup_failure_test",
        )


@pytest.mark.asyncio
async def test_system_audit_refresh_is_single_flight(monkeypatch: pytest.MonkeyPatch) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    calls: list[tuple[bool, str]] = []

    async def fake_run_once(*, record_history: bool, source: str) -> dict[str, Any]:
        calls.append((record_history, source))
        started.set()
        await release.wait()
        return {"status": "ok", "source": source}

    monkeypatch.setattr(system_audit, "_run_system_audit_subprocess_once", fake_run_once)

    first = asyncio.create_task(
        system_audit.refresh_system_audit_snapshot(record_history=True, source="first")
    )
    await started.wait()
    second = asyncio.create_task(
        system_audit.refresh_system_audit_snapshot(record_history=True, source="second")
    )
    await asyncio.sleep(0)
    release.set()

    first_result, second_result = await asyncio.gather(first, second)

    assert calls == [(True, "first")]
    assert first_result == second_result == {"status": "ok", "source": "first"}
    assert system_audit._system_audit_subprocess_task is None


@pytest.mark.asyncio
async def test_background_api_refresh_consumes_subprocess_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_refresh(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("audit failed")

    monkeypatch.setattr(system_audit, "refresh_system_audit_snapshot", fail_refresh)
    system_audit._system_audit_refresh_task = asyncio.current_task()

    await system_audit._refresh_system_audit_status()

    assert system_audit._system_audit_refresh_task is None


@pytest.mark.asyncio
async def test_system_audit_runner_closes_database_after_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = False

    async def fake_collect(**kwargs: Any) -> dict[str, Any]:
        assert kwargs == {"record_history": True, "source": "runner_test"}
        return {
            "checked_at": "2026-08-01T00:00:00+00:00",
            "status": "ok",
        }

    async def fake_close_db() -> None:
        nonlocal closed
        closed = True

    monkeypatch.setattr(run_system_audit_snapshot, "close_db", fake_close_db)

    result = await run_system_audit_snapshot.run_once(
        record_history=True,
        source="runner_test",
        collector=fake_collect,
    )

    assert result == {
        "ok": True,
        "checked_at": "2026-08-01T00:00:00+00:00",
        "status": "ok",
    }
    assert closed is True
