from scripts import audit_online_rules_canary_development as audit


def test_remote_rules_canary_audit_is_read_only_and_fixed() -> None:
    assert "fee_after_complete" in audit.REMOTE_SCRIPT
    assert "RULES_CANARY_PRIMARY_HORIZON_MINUTES" in audit.REMOTE_SCRIPT
    assert "build_rules_canary_development_report" in audit.REMOTE_SCRIPT
    assert "database_mutated" in audit.REMOTE_SCRIPT
    assert "orders_submitted" in audit.REMOTE_SCRIPT
