from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from config import settings as settings_module
from scripts import diagnose_online_development_baselines as diagnostic
from services import (
    ml_development_snapshot,
    ml_horizon_development,
    ml_training_dataset,
    strategy_replay_governance,
)


def test_horizon_worker_limits_resources_and_preserves_fixed_cutoff(monkeypatch):
    commands = []
    ssh = SimpleNamespace(close=lambda: commands.append("closed"))
    monkeypatch.setattr(sys, "argv", ["diagnostic", "--horizons"])
    monkeypatch.setattr(diagnostic, "connect_remote_ssh", lambda *_args, **_kwargs: ssh)
    monkeypatch.setattr(
        diagnostic, "run_remote_text",
        lambda _ssh, command, **_kwargs: commands.append(command) or "{}",
    )
    assert diagnostic.main() == 0
    assert commands[-1] == "closed"
    command = commands[0]
    assert "--property=CPUQuota=100%" in command
    assert "--property=MemoryMax=2G" in command
    assert "--property=RuntimeMaxSec=900" in command
    assert "--setenv=DATABASE_POOL_SIZE=2" in command
    assert diagnostic.EARLIEST_EXAM_START in command
    assert "HORIZONS = True" in command
    assert "__CUTOFF__" not in command


def test_existing_snapshot_skips_database_reload(monkeypatch, tmp_path: Path):
    calls = []
    directory = tmp_path / "strategy_replay"
    directory.mkdir()
    (directory / "frozen_development_v1.json").touch()
    monkeypatch.setattr(settings_module, "settings", SimpleNamespace(data_dir=tmp_path))
    monkeypatch.setattr(
        ml_development_snapshot, "load_development_snapshot",
        lambda path, **kwargs: calls.append((path, kwargs)) or ("frame", {"frozen": True}),
    )

    async def forbidden_reload():
        raise AssertionError("an existing snapshot must not reload current labels")

    monkeypatch.setattr(ml_training_dataset, "load_shadow_training_rows", forbidden_reload)
    monkeypatch.setattr(
        ml_horizon_development, "build_horizon_development_report",
        lambda frame, **kwargs: {"status": "complete", "folds": []},
    )
    monkeypatch.setattr(
        strategy_replay_governance, "write_replay_governance",
        lambda report, **kwargs: calls.append((report, kwargs)),
    )
    code = diagnostic.REMOTE_CODE.replace("__NESTED__", "False").replace(
        "__HORIZONS__", "True"
    ).replace("__CUTOFF__", repr(diagnostic.EARLIEST_EXAM_START))
    exec(code, {})  # noqa: S102 - Exercise the exact SSH payload.
    assert calls[0][1] == {"cutoff": diagnostic.EARLIEST_EXAM_START}
    assert calls[1][1]["path"] == directory / "horizon_development.json"
