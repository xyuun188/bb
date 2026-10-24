from __future__ import annotations

from scripts import audit_online_frozen_feature_budget as audit


def test_remote_audit_compiles_and_validates_availability_contracts() -> None:
    source = (
        audit.REMOTE_SCRIPT.replace("__LIMIT__", "1")
        .replace("__SINCE_MINUTES__", "1")
        .replace("__SOURCE__", "'decision'")
    )
    compile(source, "<online-feature-availability-audit>", "exec")

    assert "feature_availability_reasons(" in audit.REMOTE_SCRIPT
    assert "availability_valid_count" in audit.REMOTE_SCRIPT
    assert "availability_reason_counts" in audit.REMOTE_SCRIPT
    assert "newest_valid_streak_count" in audit.REMOTE_SCRIPT
    assert "newest_valid_streak_distinct_symbols" in audit.REMOTE_SCRIPT
    assert "frozen_size_bytes" in audit.REMOTE_SCRIPT
    assert "derivatives_route_status" in audit.REMOTE_SCRIPT
    assert "derivatives_unavailable_reason" in audit.REMOTE_SCRIPT
    assert "fact_sources" in audit.REMOTE_SCRIPT

