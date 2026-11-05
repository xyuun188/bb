from __future__ import annotations

import ast
from collections import Counter
from datetime import UTC, datetime, timedelta

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
            and node.name in {"_compact_evidence", "_compact_fold"}
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
