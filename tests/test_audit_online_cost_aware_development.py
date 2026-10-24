from __future__ import annotations

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
    assert "hindsight_best_action_upper_bound" in audit.REMOTE_SCRIPT
