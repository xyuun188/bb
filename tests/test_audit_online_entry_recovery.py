from __future__ import annotations

from scripts import audit_online_entry_recovery as audit


def test_entry_recovery_audit_skips_large_historical_dry_run_by_default() -> None:
    rendered = audit._render_remote_script(
        include_historical_requalification=False,
    )

    compile(rendered, "<entry-recovery-audit>", "exec")
    assert "INCLUDE_HISTORICAL_REQUALIFICATION = False" in rendered
    assert "max_batches=300" in rendered
    assert "enable with --include-historical-requalification" in rendered
    assert "if INCLUDE_HISTORICAL_REQUALIFICATION:" in rendered


def test_entry_recovery_audit_can_explicitly_include_historical_dry_run() -> None:
    rendered = audit._render_remote_script(
        include_historical_requalification=True,
    )

    compile(rendered, "<entry-recovery-audit>", "exec")
    assert "INCLUDE_HISTORICAL_REQUALIFICATION = True" in rendered
    assert "dry_run=True" in rendered
