from datetime import UTC, datetime

from scripts import audit_online_mature_shadow_cohort as audit


def test_remote_cohort_audit_is_bounded_and_read_only() -> None:
    assert "REPEATABLE READ READ ONLY" in audit.REMOTE_SCRIPT
    assert "assess_shadow_sample" in audit.REMOTE_SCRIPT
    assert "NATIVE_SOURCE" in audit.REMOTE_SCRIPT
    assert 'AIDecision.feature_snapshot["original_decision_timestamp"].as_string()' in audit.REMOTE_SCRIPT
    assert "AIDecision.created_at >= DECISION_SINCE - timedelta(hours=2)" in audit.REMOTE_SCRIPT
    assert "decision_candidate_limit = ROW_LIMIT * 4" in audit.REMOTE_SCRIPT
    assert "decision_candidates_in_window" in audit.REMOTE_SCRIPT
    assert "cast(" not in audit.REMOTE_SCRIPT
    assert "len(decision_rows) == EXPECTED_DECISION_COUNT" in audit.REMOTE_SCRIPT
    assert "if decision_cohort_complete:" in audit.REMOTE_SCRIPT
    assert "database_mutated" in audit.REMOTE_SCRIPT
    assert "orders_submitted" in audit.REMOTE_SCRIPT
    assert "status\": \"maturity_not_reached\"" in audit.REMOTE_SCRIPT
    assert '"label_rows_queried": False' in audit.REMOTE_SCRIPT


def test_rendered_cohort_audit_keeps_all_cutoffs() -> None:
    since = datetime(2026, 10, 25, 12, 36, 52, tzinfo=UTC)
    cutoff = datetime(2026, 10, 25, 13, 6, 52, tzinfo=UTC)
    maturity = datetime(2026, 10, 25, 17, 7, 52, tzinfo=UTC)

    rendered = audit._render_remote_script(
        decision_since=since,
        decision_cutoff=cutoff,
        maturity_cutoff=maturity,
        row_limit=1200,
        expected_decision_count=42,
    )

    assert "'2026-10-25T12:36:52Z'" in rendered
    assert "'2026-10-25T13:06:52Z'" in rendered
    assert "'2026-10-25T17:07:52Z'" in rendered
    assert "EXPECTED_DECISION_COUNT = 42" in rendered
    assert "DECISION_SINCE" in rendered
    assert "DECISION_CUTOFF" in rendered
    assert "MATURITY_CUTOFF" in rendered
    assert "__DECISION_SINCE__" not in rendered
    assert "__DECISION_CUTOFF__" not in rendered
    assert "__MATURITY_CUTOFF__" not in rendered
    assert "__EXPECTED_DECISION_COUNT__" not in rendered


def test_zero_decision_seal_is_explicitly_insufficient() -> None:
    rendered = audit._render_remote_script(
        decision_since=datetime(2026, 10, 26, 23, 10, tzinfo=UTC),
        decision_cutoff=datetime(2026, 10, 27, 5, 10, tzinfo=UTC),
        maturity_cutoff=datetime(2026, 10, 27, 15, 11, tzinfo=UTC),
        row_limit=1200,
        expected_decision_count=0,
    )

    assert "EXPECTED_DECISION_COUNT = 0" in rendered
    assert '"insufficient_decision_samples"' in rendered
    assert '"decision_count_matches_seal"' in rendered


def test_cli_supports_distinct_evidence_output_path() -> None:
    resolved = audit._resolve_evidence_path(
        "data/profitability_remediation/final.json",
        "default.json",
    )
    assert resolved.name == "final.json"

    try:
        audit._resolve_evidence_path("../outside.json", "default.json")
    except ValueError as exc:
        assert "must stay under" in str(exc)
    else:
        raise AssertionError("unsafe evidence output path was accepted")
