from datetime import UTC, datetime

from scripts import audit_online_rules_canary_development as audit


def test_remote_rules_canary_audit_is_read_only_and_fixed() -> None:
    assert "fee_after_complete" in audit.REMOTE_SCRIPT
    assert "RULES_CANARY_PRIMARY_HORIZON_MINUTES" in audit.REMOTE_SCRIPT
    assert 'ShadowBacktest.analysis_type == "market"' in audit.REMOTE_SCRIPT
    assert ".limit(LIMIT + 1)" in audit.REMOTE_SCRIPT
    assert "decision_group_limit_saturated" in audit.REMOTE_SCRIPT
    assert "build_rules_canary_development_report" in audit.REMOTE_SCRIPT
    assert "filter_rules_canary_decision_window" in audit.REMOTE_SCRIPT
    assert "ShadowBacktest.updated_at < MATURITY_CUTOFF" in audit.REMOTE_SCRIPT
    assert "cutoff=DECISION_CUTOFF" in audit.REMOTE_SCRIPT
    assert "decision_cutoff_exclusive" in audit.REMOTE_SCRIPT
    assert "maturity_cutoff_exclusive" in audit.REMOTE_SCRIPT
    assert "maturity_guard" in audit.REMOTE_SCRIPT
    assert "database_mutated" in audit.REMOTE_SCRIPT
    assert "orders_submitted" in audit.REMOTE_SCRIPT


def test_rendered_remote_audit_keeps_decision_and_maturity_cutoffs_separate() -> None:
    decision_cutoff = datetime(2026, 10, 25, 19, 6, 53, tzinfo=UTC)
    maturity_cutoff = datetime(2026, 10, 25, 23, 6, 53, tzinfo=UTC)

    rendered = audit._render_remote_script(
        limit=1200,
        hours=6,
        cutoff=decision_cutoff,
        maturity_cutoff=maturity_cutoff,
    )

    assert "'2026-10-25T19:06:53Z'" in rendered
    assert "'2026-10-25T23:06:53Z'" in rendered
    assert "ShadowBacktest.updated_at < MATURITY_CUTOFF" in rendered
    assert "cutoff=DECISION_CUTOFF" in rendered
    assert "__DECISION_CUTOFF__" not in rendered
    assert "__MATURITY_CUTOFF__" not in rendered


def test_maturity_cutoff_includes_max_horizon_and_source_lag() -> None:
    cutoff = datetime(2026, 10, 25, 19, 6, 53, tzinfo=UTC)
    assert audit._minimum_maturity_cutoff(cutoff).isoformat() == (
        "2026-10-25T23:07:53+00:00"
    )


def test_rendered_remote_audit_preserves_legacy_cutoff_default() -> None:
    cutoff = datetime(2026, 10, 25, 19, 6, 53, tzinfo=UTC)

    rendered = audit._render_remote_script(
        limit=1200,
        hours=6,
        cutoff=cutoff,
    )

    assert rendered.count("'2026-10-25T19:06:53Z'") == 2
