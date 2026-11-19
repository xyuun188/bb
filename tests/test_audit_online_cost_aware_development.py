from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy.dialects import postgresql

from scripts import audit_online_cost_aware_development as audit


def test_remote_cost_ledger_is_bounded_and_development_only() -> None:
    source = (
        audit.REMOTE_SCRIPT.replace("__LIMIT__", "1")
        .replace("__HOURS__", "1")
    )
    compile(source, "<online-cost-ledger>", "exec")

    assert "exam_opened" in audit.REMOTE_SCRIPT
    assert "native executable 1m path + fee-after-complete labels" in audit.REMOTE_SCRIPT
    assert ".limit(LIMIT)" in audit.REMOTE_SCRIPT
    assert 'snapshot.get("original_decision_timestamp")' in audit.REMOTE_SCRIPT
    assert 'label.get("label_timestamp")' in audit.REMOTE_SCRIPT
    assert "split_shadow_development_rows(rows)" in audit.REMOTE_SCRIPT
    assert '"selection_policy": "one_completed_row_per_horizon_and_decision_time_bucket"' in audit.REMOTE_SCRIPT
    assert "hindsight_best_action_upper_bound" in audit.REMOTE_SCRIPT


def test_cost_ledger_samples_by_horizon_and_creation_time() -> None:
    scope = {}
    source = audit.REMOTE_SCRIPT.replace("__LIMIT__", "1000").replace("__HOURS__", "24")
    exec(source.removesuffix("asyncio.run(main())\n"), scope)  # noqa: S102

    cutoff = datetime(2026, 11, 20, tzinfo=UTC)
    query = scope["_sample_statement"](cutoff - timedelta(hours=24), cutoff)
    sql = str(query.compile(dialect=postgresql.dialect()))

    assert "PARTITION BY shadow_backtests.horizon_minutes" in sql
    assert "EXTRACT(epoch FROM shadow_backtests.created_at)" in sql
    assert "shadow_backtests.created_at >=" in sql
    assert "shadow_backtests.updated_at <" in sql
    assert "ORDER BY shadow_backtests.updated_at" not in sql
