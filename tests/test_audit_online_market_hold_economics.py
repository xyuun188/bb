from __future__ import annotations

from scripts import audit_online_market_hold_economics as audit


def test_remote_audit_compiles_and_reads_current_consultation_contract() -> None:
    compile(audit.REMOTE_SCRIPT, "<online-market-hold-economics>", "exec")

    assert 'raw.get("consultation")' in audit.REMOTE_SCRIPT
    assert 'raw.get("conflict_consultation")' in audit.REMOTE_SCRIPT
    assert 'consultation.get("consultation_attempts")' in audit.REMOTE_SCRIPT
    assert 'consultation.get("attempts")' in audit.REMOTE_SCRIPT
    assert "consultation_by_analysis_type" in audit.REMOTE_SCRIPT
    assert "production_permission" in audit.REMOTE_SCRIPT


def test_remote_audit_queries_market_economics_and_all_analysis_consultations() -> None:
    assert 'AIDecision.analysis_type == "market"' in audit.REMOTE_SCRIPT
    assert "consultation_rows" in audit.REMOTE_SCRIPT
    assert "AIDecision.raw_llm_response" in audit.REMOTE_SCRIPT
    assert "AIDecision.analysis_type," in audit.REMOTE_SCRIPT
