from __future__ import annotations

import ast
import json
from collections import Counter
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects import postgresql

from models.learning import ShadowBacktest
from scripts import audit_online_horizon_current_development as audit


def test_horizon_development_command_is_bounded_and_cutoff_bound() -> None:
    cutoff = datetime(2026, 10, 24, 5, 57, tzinfo=UTC)
    command = audit._build_remote_command(
        limit=1000,
        hours=24,
        cutoff=cutoff,
        unit="bb-horizon-dev-v1-20261024T055700Z",
        revision="v1",
    )
    assert "--property=CPUQuota=25%" in command
    assert "--property=MemoryMax=1G" in command
    assert "--property=RuntimeMaxSec=300s" in command
    assert audit._remote_runtime_limit_seconds(1000) == 300
    assert "--setenv=DATABASE_POOL_SIZE=1" in command
    rendered = audit._render_remote_script(
        limit=1000,
        hours=24,
        cutoff=cutoff,
        revision="v1",
    )
    assert "2026-10-24T05:57:00Z" in rendered
    assert "__CUTOFF__" not in rendered
    assert '"runtime_limit_seconds": 300' in rendered
    statement = audit._build_development_statement(
        limit=1000, since=cutoff - timedelta(hours=24), cutoff=cutoff, revision="v1",
    )
    sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "max(shadow_backtests.updated_at) DESC" in sql
    assert "exam_labels_evaluated" in rendered
    assert "complete_group_count =" in rendered
    assert '"inner_folds": selection.get("inner_folds", [])' in rendered
    assert '"horizon_coverage": selection.get("horizon_coverage", {})' in rendered
    assert '"purge_policy": research.get("purge_policy")' in rendered
    compile(rendered, "<horizon-audit>", "exec")


def test_compact_report_preserves_actual_selection_coverage() -> None:
    payload = ast.parse(audit.REMOTE_SCRIPT)
    helpers = ast.Module(
        body=[
            node for node in payload.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"_as_dict", "_compact_evidence", "_compact_fold"}
        ],
        type_ignores=[],
    )
    namespace = {"Counter": Counter}
    exec(compile(helpers, "<horizon-audit-helpers>", "exec"), namespace)  # noqa: S102
    coverage = {
        "5": {"ready_fold_count": 2, "exclusion_reasons": []},
        "240": {
            "ready_fold_count": 0,
            "exclusion_reasons": ["insufficient_inner_folds"],
        },
    }
    compact = namespace["_compact_fold"]({
        "training_sample_count": 200,
        "training_positive_weight_sample_count": 190,
        "validation_sample_count": 100,
        "training_sample_count_by_horizon": {"5": 100, "15": 100, "60": 0, "240": 0},
        "label_purged_training_sample_count": 20,
        "selection": {"horizon_coverage": coverage},
    })

    assert compact["horizon_coverage"] == coverage
    assert compact["training_sample_count"] == 200
    assert compact["training_positive_weight_sample_count"] == 190
    assert compact["validation_sample_count"] == 100
    assert compact["training_sample_count_by_horizon"]["5"] == 100
    assert compact["label_purged_training_sample_count"] == 20


def test_large_horizon_development_batch_gets_a_bounded_runtime_budget() -> None:
    cutoff = datetime(2026, 10, 24, 5, 57, tzinfo=UTC)
    command = audit._build_remote_command(
        limit=2500,
        hours=24,
        cutoff=cutoff,
        unit="bb-horizon-dev-v1-20261024T055700Z",
        revision="v1",
    )

    assert audit._remote_runtime_limit_seconds(2500) == 900
    assert "--property=RuntimeMaxSec=900s" in command
    assert "--property=CPUQuota=25%" in command
    assert "--property=MemoryMax=1G" in command
    rendered = audit._render_remote_script(
        limit=2500,
        hours=24,
        cutoff=cutoff,
        revision="v2",
    )
    assert '"runtime_limit_seconds": 900' in rendered
    assert "__RUNTIME_LIMIT_SECONDS__" not in rendered


def test_same_window_revision_requires_matching_input_fingerprint() -> None:
    report = {"input": {"development_input_sha256": "abc123"}}

    audit._validate_expected_input_hash(report, "abc123")
    try:
        audit._validate_expected_input_hash(report, "different")
    except RuntimeError as exc:
        assert "fingerprint changed" in str(exc)
    else:
        raise AssertionError("changed samples must stop the comparison")


@pytest.mark.parametrize("value", ("a" * 63, "g" * 64, "A" * 64, ""))
def test_malformed_expected_hash_is_rejected_before_protocol_registration(value):
    with pytest.raises(ValueError, match="exactly 64"):
        audit._validate_hash_argument(value)
    audit._validate_hash_argument("a" * 64)
    audit._validate_hash_argument(None)


def test_time_coverage_sampling_preserves_old_groups_and_all_horizons() -> None:
    cutoff = datetime(2026, 11, 5, tzinfo=UTC)
    since = cutoff - timedelta(hours=24)
    engine = create_engine("sqlite://")
    table = ShadowBacktest.__table__
    table.create(engine)
    rows = []
    group_times = {1: 23, 2: 17, 3: 11}
    group_times.update({group: 5 + group / 1000 for group in range(100, 200)})
    # A recent repair must not bring an out-of-window decision into v3.
    group_times[4] = 25
    for group, hours_ago in group_times.items():
        for horizon in (5, 15, 60, 240):
            created = cutoff - timedelta(hours=hours_ago)
            rows.append({
                "id": len(rows) + 1, "decision_id": group,
                "created_at": created, "updated_at": cutoff - timedelta(minutes=1),
                "model_name": "ensemble", "symbol": "TEST/USDT",
                "execution_mode": "paper", "status": "completed",
                "horizon_minutes": horizon,
                "due_at": created + timedelta(minutes=horizon),
                "training_feature_snapshot": {
                    "training_market_fact_contract": {"source": audit.NATIVE_SOURCE},
                    "training_label_contract": {"fee_after_complete": True},
                },
            })
    statement = audit._build_development_statement(
        limit=4, since=since, cutoff=cutoff, revision="v3",
    )
    sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "research_ids AS MATERIALIZED" in sql
    with engine.begin() as connection:
        connection.execute(table.insert(), rows)
        selected = list(connection.execute(statement).mappings())
        assert {row["decision_id"] for row in selected} == {1, 2, 3, 100}
        assert len(selected) == 16
        for group in (1, 2, 3, 100):
            assert {row["horizon_minutes"] for row in selected if row["decision_id"] == group} == {
                5, 15, 60, 240,
            }
        connection.execute(table.update().values(
            long_return_pct=1000, short_return_pct=-1000,
            updated_at=cutoff - timedelta(minutes=1),
        ))
        changed = list(connection.execute(statement).mappings())
        assert [row["id"] for row in changed] == [row["id"] for row in selected]
        # A rejected bucket stays empty; older outcomes cannot replace it.
        connection.execute(
            table.update().where(table.c.decision_id == 100).values(
                training_feature_snapshot={},
                updated_at=cutoff - timedelta(minutes=1),
            )
        )
        rejected = list(connection.execute(statement).mappings())
        assert {row["decision_id"] for row in rejected} == {1, 2, 3}
    engine.dispose()


def test_liquidity_regime_remote_payload_selects_fixed_family():
    rendered = audit._render_remote_script(
        limit=2500, hours=672,
        cutoff=datetime(2026, 11, 7, 15, 30, tzinfo=UTC),
        revision="v3", family="liquidity_regime",
    )
    assert "FAMILY = 'liquidity_regime'" in rendered
    assert "build_liquidity_regime_development_report" in rendered
    assert "services/liquidity_regime_development.py" in audit.DEPENDENCIES
    compile(rendered, "<liquidity-regime-audit>", "exec")


def test_registration_is_immutable_and_preserves_original_timestamp(tmp_path):
    path = tmp_path / "protocol.json"
    registration = {"cutoff": "2026-11-07T15:30:00Z", "candidate_count": 8}
    sealed = audit._register_protocol(path, registration)
    assert sealed["registered_at"]
    original = path.read_bytes()
    assert audit._register_protocol(path, registration) == sealed
    with pytest.raises(ValueError, match="protocol cannot change"):
        audit._register_protocol(path, {**registration, "candidate_count": 9})
    assert path.read_bytes() == original


@pytest.mark.parametrize("family", ("ridge", "liquidity_regime", "replay_weights"))
def test_remote_empty_source_is_reported_without_evaluation(monkeypatch, capsys, family):
    from db import session as db_session
    from services import liquidity_regime_development, ml_horizon_development, ml_signal_service

    async def execute(_statement):
        return SimpleNamespace(mappings=lambda: [])

    @asynccontextmanager
    async def read_session(**_kwargs):
        yield SimpleNamespace(execute=execute)

    async def close():
        pass

    def forbidden_research(*_args, **_kwargs):
        raise AssertionError("empty development must not be evaluated")

    monkeypatch.setattr(db_session, "get_read_session_ctx", read_session)
    monkeypatch.setattr(db_session, "close_db", close)
    monkeypatch.setattr(ml_signal_service, "build_training_frame", lambda _rows: pd.DataFrame())
    monkeypatch.setattr(
        liquidity_regime_development, "build_liquidity_regime_development_report", forbidden_research,
    )
    monkeypatch.setattr(
        ml_horizon_development, "build_horizon_development_report", forbidden_research,
    )
    rendered = audit._render_remote_script(
        limit=2500, hours=672,
        cutoff=datetime(2026, 11, 7, 15, 30, tzinfo=UTC),
        revision="v3", family=family,
    )
    exec(rendered, {})  # noqa: S102 - Run the actual bounded remote payload.
    report = json.loads(capsys.readouterr().out)
    assert report["research"]["status"] == "insufficient_development_data"
    assert report["input"]["source_row_count"] == 0
    assert report["input"]["decision_time_start"] is None
    assert report["safety"]["orders_submitted"] is False


@pytest.mark.parametrize("matching_hash", (True, False))
def test_replay_weight_audit_reuses_sealed_inputs_and_checks_identity_before_fitting(
    monkeypatch, tmp_path, capsys, matching_hash,
):
    from config import settings as settings_module
    from db import session as db_session
    from services import ml_replay_weight_development as research
    from services.ml_development_baselines import _development_input_sha256
    from services.ml_development_snapshot import (
        REPLAY_METADATA_COLUMNS,
        save_development_snapshot,
    )
    from tests.test_ml_replay_weight_development import CUTOFF, _snapshot, _source

    development, snapshot = _snapshot(_source())
    cutoff = datetime.fromisoformat(CUTOFF)
    path = tmp_path / "strategy_replay" / "replay-weight-input-v3-20260810T000000Z-2500-672.json"
    save_development_snapshot(snapshot, path=path)
    original_bytes = path.read_bytes()
    monkeypatch.setattr(settings_module, "settings", SimpleNamespace(data_dir=tmp_path))

    def forbidden_reload(**_kwargs):
        raise AssertionError("sealed replay comparison must not reload current labels")

    async def close():
        pass

    calls = []

    def builder(frame, *, snapshot):
        pd.testing.assert_frame_equal(frame, development)
        calls.append(snapshot)
        return {"status": "rejected", "folds": []}

    monkeypatch.setattr(db_session, "get_read_session_ctx", forbidden_reload)
    monkeypatch.setattr(db_session, "close_db", close)
    monkeypatch.setattr(research, "build_replay_weight_development_report", builder)
    expected = _development_input_sha256(development.drop(columns=list(REPLAY_METADATA_COLUMNS)))
    rendered = audit._render_remote_script(
        limit=2500, hours=672, cutoff=cutoff, revision="v3", family="replay_weights",
        expected_base_hash=expected if matching_hash else "wrong",
    )
    if matching_hash:
        exec(rendered, {})  # noqa: S102 - Exercise sealed-input reuse in the actual payload.
        report = json.loads(capsys.readouterr().out)
        assert report["input"]["source_rows_reloaded"] is False
        assert calls == [snapshot]
    else:
        with pytest.raises(RuntimeError, match="fingerprint changed"):
            exec(rendered, {})  # noqa: S102
        assert not calls
    assert path.read_bytes() == original_bytes
