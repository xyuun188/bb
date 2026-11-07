from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from config import settings as settings_module
from scripts import diagnose_online_development_baselines as diagnostic
from services import (
    ml_development_snapshot,
    ml_horizon_development,
    ml_signal_service,
    ml_training_dataset,
    strategy_replay_governance,
)
from services.ml_development_snapshot import SNAPSHOT_FILENAME


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
    (directory / SNAPSHOT_FILENAME).touch()
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


def test_legacy_snapshot_is_preserved_and_new_version_is_created(monkeypatch, tmp_path: Path):
    calls = []
    directory = tmp_path / "strategy_replay"
    directory.mkdir()
    legacy_path = directory / "frozen_development_v1.json"
    legacy_bytes = b'{"legacy": true}'
    legacy_path.write_bytes(legacy_bytes)
    monkeypatch.setattr(settings_module, "settings", SimpleNamespace(data_dir=tmp_path))

    async def load_rows():
        return ["rows"]

    monkeypatch.setattr(ml_training_dataset, "load_shadow_training_rows", load_rows)
    monkeypatch.setattr(ml_signal_service, "build_training_frame", lambda rows: rows)
    monkeypatch.setattr(
        ml_development_snapshot, "make_development_snapshot",
        lambda frame, **kwargs: {"version": "new"},
    )
    monkeypatch.setattr(
        ml_development_snapshot, "save_development_snapshot",
        lambda snapshot, *, path: calls.append((snapshot, path)) or ("development", snapshot),
    )
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

    assert calls[0] == ({"version": "new"}, directory / SNAPSHOT_FILENAME)
    assert legacy_path.read_bytes() == legacy_bytes


def _source_frame(*, after_cutoff=False, weight=1.0):
    cutoff = pd.Timestamp(diagnostic.EARLIEST_EXAM_START)
    decision = cutoff + pd.Timedelta(minutes=1 if after_cutoff else -10)
    return pd.DataFrame([{
        **{key: 0.0 for key in ml_signal_service.FEATURE_KEYS},
        "id": 1,
        "decision_group": "group-1",
        "symbol": "BTC/USDT",
        "decision_timestamp": decision,
        "label_timestamp": decision + pd.Timedelta(minutes=5),
        "horizon_minutes": 5,
        "long_return_pct": 0.1,
        "short_return_pct": -0.1,
        "long_execution_cost_pct": 0.1,
        "short_execution_cost_pct": 0.1,
        "sample_weight": weight,
    }])


@pytest.mark.parametrize("source_case", ("empty", "after_cutoff", "zero_weight"))
def test_unavailable_frozen_inputs_report_insufficiency_without_research(
    monkeypatch, tmp_path: Path, capsys, source_case,
):
    frame = (
        pd.DataFrame() if source_case == "empty"
        else _source_frame(
            after_cutoff=source_case == "after_cutoff",
            weight=0.0 if source_case == "zero_weight" else 1.0,
        )
    )
    reports = []
    monkeypatch.setattr(settings_module, "settings", SimpleNamespace(data_dir=tmp_path))

    async def load_rows():
        return ["source"]

    monkeypatch.setattr(ml_training_dataset, "load_shadow_training_rows", load_rows)
    monkeypatch.setattr(ml_signal_service, "build_training_frame", lambda rows: frame)

    def forbidden_research(*args, **kwargs):
        raise AssertionError("unavailable frozen inputs cannot be saved or evaluated")

    monkeypatch.setattr(ml_development_snapshot, "save_development_snapshot", forbidden_research)
    monkeypatch.setattr(
        ml_horizon_development, "build_horizon_development_report", forbidden_research,
    )
    monkeypatch.setattr(
        strategy_replay_governance, "write_replay_governance",
        lambda report, **kwargs: reports.append(report),
    )
    code = diagnostic.REMOTE_CODE.replace("__NESTED__", "False").replace(
        "__HORIZONS__", "True"
    ).replace("__CUTOFF__", repr(diagnostic.EARLIEST_EXAM_START))

    exec(code, {})  # noqa: S102 - Exercise the real snapshot validation.

    report = json.loads(capsys.readouterr().out)
    assert report == reports[0]
    assert report["status"] == "insufficient_development_data"
    assert report["reason_code"] == "frozen_development_data_unavailable"
    assert report["development_cutoff"] == diagnostic.EARLIEST_EXAM_START
    assert report["selected_policy_fold_count"] == 0
    assert report["folds"] == []
    for field in (
        "artifact_persisted", "promotion_authorized", "submits_orders",
        "exam_labels_evaluated",
    ):
        assert report[field] is False
    assert not (tmp_path / "strategy_replay" / SNAPSHOT_FILENAME).exists()


@pytest.mark.parametrize("existing_snapshot", (False, True))
def test_invalid_frozen_inputs_are_not_reported_as_insufficient(
    monkeypatch, tmp_path: Path, existing_snapshot,
):
    monkeypatch.setattr(settings_module, "settings", SimpleNamespace(data_dir=tmp_path))

    async def load_rows():
        return ["source"]

    monkeypatch.setattr(ml_training_dataset, "load_shadow_training_rows", load_rows)
    monkeypatch.setattr(
        ml_signal_service, "build_training_frame", lambda rows: _source_frame(weight=-1.0),
    )
    message = "invalid snapshot sample weights"
    if existing_snapshot:
        directory = tmp_path / "strategy_replay"
        directory.mkdir()
        (directory / SNAPSHOT_FILENAME).write_text("{}", encoding="utf-8")
        message = "development snapshot integrity failure"

    code = diagnostic.REMOTE_CODE.replace("__NESTED__", "False").replace(
        "__HORIZONS__", "True"
    ).replace("__CUTOFF__", repr(diagnostic.EARLIEST_EXAM_START))
    with pytest.raises(ValueError, match=message):
        exec(code, {})  # noqa: S102 - Integrity failures must remain errors.
