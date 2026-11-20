from __future__ import annotations

from datetime import UTC, datetime

from scripts import audit_online_fast_exit_development as audit


def test_remote_script_renders_all_bounded_query_parameters() -> None:
    rendered = audit._render_remote_script(
        hours=6,
        limit=250,
        cutoff=datetime(2026, 11, 6, 12, 0, tzinfo=UTC),
    )

    compile(rendered, "<fast-exit-audit>", "exec")
    assert "__HOURS__" not in rendered
    assert "__LIMIT__" not in rendered
    assert "__SCAN_BATCH_SIZE__" not in rendered
    assert "__MAX_SCAN_MULTIPLIER__" not in rendered
    assert "__MAX_SCAN_ROWS__" not in rendered
    assert "__CUTOFF__" not in rendered
    assert "__DEPENDENCIES__" not in rendered
    assert "HOURS = 6" in rendered
    assert "LIMIT = 250" in rendered
    assert f"SCAN_BATCH_SIZE = {audit.SCAN_BATCH_SIZE}" in rendered
    assert f"MAX_SCAN_MULTIPLIER = {audit.MAX_SCAN_MULTIPLIER}" in rendered
    assert f"MAX_SCAN_ROWS = {audit.MAX_SCAN_ROWS}" in rendered


def test_remote_query_keeps_update_window_and_projects_decision_and_label_times() -> None:
    assert "ShadowBacktest.updated_at >= since" in audit.REMOTE_SCRIPT
    assert "ShadowBacktest.updated_at < CUTOFF" in audit.REMOTE_SCRIPT
    assert '"selected"].as_boolean()' in audit.REMOTE_SCRIPT
    assert "source_row_count = 0" in audit.REMOTE_SCRIPT
    assert "source_row_count += len(batch)" in audit.REMOTE_SCRIPT
    assert '"source_row_count": source_row_count' in audit.REMOTE_SCRIPT
    assert "len(source_rows)" not in audit.REMOTE_SCRIPT
    assert "query_scan_limit_reached" in audit.REMOTE_SCRIPT
    assert '"original_decision_timestamp"' in audit.REMOTE_SCRIPT
    assert '["label_timestamp"].as_string().label("label_at")' in audit.REMOTE_SCRIPT
    assert '"outcome_updated_at within [since, cutoff); updated_at is pagination key"' in (
        audit.REMOTE_SCRIPT
    )


def test_remote_query_is_keyset_paginated_and_bounded() -> None:
    assert 'cursor_updated_at = batch[-1]["updated_at"]' in audit.REMOTE_SCRIPT
    assert 'cursor_id = batch[-1]["id"]' in audit.REMOTE_SCRIPT
    assert ".order_by(ShadowBacktest.updated_at.desc(), ShadowBacktest.id.desc())" in (
        audit.REMOTE_SCRIPT
    )
    assert ".limit(min(SCAN_BATCH_SIZE, scan_limit - scanned))" in audit.REMOTE_SCRIPT
    assert "updated_at_index_keyset_pages_with_sql_contract_filters" in audit.REMOTE_SCRIPT


def test_dependency_check_ignores_only_platform_line_endings(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "DEPENDENCIES", ("model.py",))
    path = tmp_path / "model.py"
    path.write_bytes(b"x = 1\n")
    expected = audit._local_dependency_hashes()
    path.write_bytes(b"x = 1\r\n")
    assert audit._local_dependency_hashes() == expected
    path.write_bytes(b"x = 2\r\n")
    assert audit._local_dependency_hashes() != expected
