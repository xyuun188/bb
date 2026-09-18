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
    assert "non_market_consultation_rows" in audit.REMOTE_SCRIPT
    assert 'AIDecision.analysis_type != "market"' in audit.REMOTE_SCRIPT
    assert "AIDecision.raw_llm_response" in audit.REMOTE_SCRIPT
    assert "AIDecision.analysis_type," in audit.REMOTE_SCRIPT


def test_remote_audit_projects_only_required_columns_and_bounds_duplicate_scan() -> None:
    assert "select(AIDecision)" not in audit.REMOTE_SCRIPT
    assert "AIDecision.model_health_opinions," in audit.REMOTE_SCRIPT
    assert "AIDecision.model_health_timings," in audit.REMOTE_SCRIPT
    duplicate_scan = audit.REMOTE_SCRIPT.split("duplicate_rows =", 1)[1]
    assert ".where(\n                    *filters," in duplicate_scan


def test_remote_audit_reports_observation_starvation_economics() -> None:
    compile(audit.REMOTE_SCRIPT, "<online-market-hold-economics>", "exec")

    assert "preferred_paper_observation_side" in audit.REMOTE_SCRIPT
    assert "positive_mean_non_positive_lcb" in audit.REMOTE_SCRIPT
    assert "quality_permission_blockers" in audit.REMOTE_SCRIPT
    assert "gross_expected_return_pct" in audit.REMOTE_SCRIPT
    assert "execution_cost_pct" in audit.REMOTE_SCRIPT
    assert "lcb_penalty_pct" in audit.REMOTE_SCRIPT
    assert "directional_decision_examples" in audit.REMOTE_SCRIPT
    assert "profit_risk_sizing" in audit.REMOTE_SCRIPT
    assert "high_risk_review" in audit.REMOTE_SCRIPT
    assert "hard_review_required" in audit.REMOTE_SCRIPT
    assert "execution_result" in audit.REMOTE_SCRIPT
    assert "exchange_order_id" in audit.REMOTE_SCRIPT
