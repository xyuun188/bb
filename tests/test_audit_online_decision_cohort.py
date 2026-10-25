from datetime import UTC, datetime

from scripts import audit_online_decision_cohort as audit


def test_decision_cohort_seal_is_read_only_and_does_not_read_labels() -> None:
    assert "REPEATABLE READ READ ONLY" in audit.REMOTE_SCRIPT
    assert 'AIDecision.analysis_type == "market"' in audit.REMOTE_SCRIPT
    assert "ShadowBacktest" not in audit.REMOTE_SCRIPT
    assert '"shadow_labels_read": False' in audit.REMOTE_SCRIPT
    assert ".limit(candidate_limit + 1)" in audit.REMOTE_SCRIPT
    assert "decision_candidate_limit_saturated" in audit.REMOTE_SCRIPT


def test_rendered_decision_cohort_seal_keeps_exact_window() -> None:
    since = datetime(2026, 10, 25, 13, 6, 53, tzinfo=UTC)
    cutoff = datetime(2026, 10, 25, 19, 6, 53, tzinfo=UTC)

    rendered = audit._render_remote_script(
        decision_since=since,
        decision_cutoff=cutoff,
        row_limit=1200,
    )

    assert "'2026-10-25T13:06:53Z'" in rendered
    assert "'2026-10-25T19:06:53Z'" in rendered
    assert "__DECISION_SINCE__" not in rendered
    assert "__DECISION_CUTOFF__" not in rendered
