from __future__ import annotations

import asyncio
import signal
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import scripts
from config import settings as settings_module
from scripts import diagnose_online_training_preflight as diagnostic
from services import strategy_replay_governance


@pytest.mark.parametrize(
    ("mode", "expected_status", "expected_exit"),
    (("complete", "complete", 0), ("skipped", "skipped", 3), ("failed", "failed", 2)),
)
def test_preflight_writes_bounded_non_persisting_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
    expected_status: str,
    expected_exit: int,
) -> None:
    trainer = ModuleType("scripts.train_ml_signal_model")
    reports: list[tuple[dict[str, Any], Path]] = []
    handlers: dict[int, Any] = {}
    calls: list[dict[str, Any]] = []

    async def run_training(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        if mode == "failed":
            handlers[signal.SIGTERM](signal.SIGTERM, None)
        return {
            "metadata": {
                "shadow_selection_policy": "test-policy",
                "metrics": {"net_return": -0.1},
                "walk_forward_report": {"fold_count": 3},
            },
            "frame_sample_count": 100,
            "loaded_row_count": 120,
            "authoritative_trade_sample_count": 4,
        }

    def main() -> int:
        assert sys.argv == ["scripts/train_ml_signal_model.py"]
        if mode == "skipped":
            print('{"reason": "local_ml_training_already_running"}')
            return 3
        asyncio.run(trainer.run_training())
        return 0

    def write_report(report: dict[str, Any], *, path: Path) -> None:
        reports.append((report, path))

    trainer.run_training = run_training
    trainer.main = main
    monkeypatch.setitem(sys.modules, "scripts.train_ml_signal_model", trainer)
    monkeypatch.setattr(scripts, "train_ml_signal_model", trainer, raising=False)
    monkeypatch.setattr(strategy_replay_governance, "write_replay_governance", write_report)
    monkeypatch.setattr(settings_module, "settings", SimpleNamespace(data_dir=tmp_path))
    monkeypatch.setattr(
        signal, "signal", lambda number, handler: handlers.update({number: handler})
    )
    monkeypatch.setattr(sys, "argv", ["preflight"])

    with pytest.raises(SystemExit) as result:
        exec(diagnostic.REMOTE_CODE, {})  # noqa: S102 - Exercise the exact SSH payload.

    assert result.value.code == expected_exit
    assert len(reports) == 1
    report, path = reports[0]
    assert path == tmp_path / "strategy_replay/training_preflight.json"
    assert report["status"] == expected_status
    assert report["exit_code"] == expected_exit
    assert report["diagnostic_only"] is True
    assert report["artifact_persisted"] is False
    assert report["promotion_authorized"] is False
    assert report["submits_orders"] is False
    assert all(not kwargs.get("persist_artifact") for kwargs in calls)
    if mode == "complete":
        assert report["frame_sample_count"] == 100
        assert report["metadata"]["shadow_selection_policy"] == "test-policy"
        assert report["metadata"]["metrics"] == {"net_return": -0.1}
    elif mode == "failed":
        assert "terminated" in report["error"]
    else:
        assert "local_ml_training_already_running" in report["result"]


def test_preflight_worker_limits_resources_and_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[tuple[str, dict[str, Any]]] = []

    class SSH:
        closed = False

        def close(self) -> None:
            self.closed = True

    ssh = SSH()
    monkeypatch.setattr(diagnostic, "connect_remote_ssh", lambda *_args, **_kwargs: ssh)

    def run_remote(_ssh: SSH, command: str, **kwargs: Any) -> str:
        assert _ssh is ssh
        commands.append((command, kwargs))
        return '{"status": "complete"}'

    monkeypatch.setattr(diagnostic, "run_remote_text", run_remote)

    assert diagnostic.main() == 0
    assert ssh.closed
    command, kwargs = commands[0]
    assert "--property=CPUQuota=200%" in command
    assert "--property=MemoryMax=2G" in command
    assert "--property=RuntimeMaxSec=3600" in command
    assert "--property=TimeoutStopSec=30" in command
    assert "--setenv=DATABASE_POOL_SIZE=2" in command
    assert kwargs["timeout"] > 3600
