from services.entry_funnel_diagnostics import (
    build_direction_symmetry_report,
    classify_entry_funnel_reason,
)


def test_funnel_classifies_structured_funding_block() -> None:
    assert classify_entry_funnel_reason(
        raw={
            "opportunity_score": {
                "funding_cost": {
                    "production_eligible": True,
                    "blocked": True,
                }
            }
        },
        action="long",
        was_executed=False,
        has_order=False,
        reason="funding_cost_blocked",
    ) == "funding_cost_blocked"


def test_funnel_does_not_treat_normal_funding_diagnostics_as_a_block() -> None:
    assert classify_entry_funnel_reason(
        raw={
            "opportunity_score": {
                "expected_net_return_pct": 0.4,
                "funding_cost": {
                    "production_eligible": True,
                    "signed_cashflow_pct": -0.02,
                    "reason": "complete",
                },
            },
            "production_trade_gate": {
                "allowed": True,
                "risk": {"blocked": False},
            },
        },
        action="hold",
        was_executed=False,
        has_order=False,
        reason="AI 选择观望",
    ) == "no_candidate"


def test_funnel_classifies_only_a_funding_driven_net_zero_crossing() -> None:
    assert classify_entry_funnel_reason(
        raw={
            "opportunity_score": {
                "expected_net_return_pct": -0.01,
                "funding_cost": {
                    "production_eligible": True,
                    "signed_cashflow_pct": -0.03,
                    "reason": "complete",
                },
            }
        },
        action="long",
        was_executed=False,
        has_order=False,
        reason="预期净收益不足",
    ) == "funding_cost_blocked"


def test_funnel_treats_missing_funding_evidence_as_insufficient_evidence() -> None:
    assert classify_entry_funnel_reason(
        raw={
            "analysis_quality_contract": {
                "analysis_complete": False,
                "decision_eligible": False,
                "reason_code": "funding_evidence_unavailable",
                "reason": "资金费率或下一结算时间证据不完整。",
            },
            "opportunity_score": {
                "funding_cost": {
                    "production_eligible": False,
                    "reason": "next_funding_time_missing",
                }
            },
        },
        action="hold",
        was_executed=False,
        has_order=False,
        reason="证据不足",
    ) == "insufficient_evidence"


def test_funnel_classifies_missing_evidence_before_generic_hold() -> None:
    assert classify_entry_funnel_reason(
        raw={
            "analysis_quality_contract": {
                "analysis_complete": False,
                "decision_eligible": False,
                "reason_code": "insufficient_evidence",
            }
        },
        action="hold",
        was_executed=False,
        has_order=False,
        reason="专家不足",
    ) == "insufficient_evidence"


def test_funnel_ignores_optional_tool_timeout_after_complete_analysis() -> None:
    assert classify_entry_funnel_reason(
        raw={
            "analysis_quality_contract": {
                "analysis_complete": True,
                "decision_eligible": True,
                "reason_code": "analysis_complete",
            },
            "local_ai_tools": {
                "status": "degraded",
                "reason": "local_ai_tools_context_timeout",
            },
            "model_timings": [
                {"name": "local_ai_tools", "status": "timeout"},
            ],
        },
        action="hold",
        was_executed=False,
        has_order=False,
        reason="AI completed expert analysis; optional tools unavailable",
    ) == "no_candidate"


def test_funnel_keeps_explicit_market_model_timeout_as_service_error() -> None:
    assert classify_entry_funnel_reason(
        raw={
            "analysis_quality_contract": {
                "analysis_complete": True,
                "decision_eligible": True,
            },
            "market_model_timeout": {
                "isolated_to_symbol": True,
                "production_permission": False,
            },
        },
        action="hold",
        was_executed=False,
        has_order=False,
    ) == "service_error"


def test_funnel_classifies_reconciliation_before_execution() -> None:
    assert classify_entry_funnel_reason(
        raw={},
        action="short",
        was_executed=False,
        has_order=True,
        reason="okx_authoritative_sync_unhealthy: reconciliation mismatch",
    ) == "account_reconciliation_blocked"


def test_direction_symmetry_report_is_read_only_and_exposes_side_rates() -> None:
    report = build_direction_symmetry_report(
        [
            {"action": "short", "is_entry": True, "was_executed": True},
            {
                "action": "short",
                "is_entry": True,
                "was_executed": False,
                "funnel_reason": "risk_blocked",
            },
            {
                "action": "long",
                "is_entry": True,
                "was_executed": False,
                "funnel_reason": "insufficient_evidence",
            },
            {
                "action": "long",
                "is_entry": True,
                "was_executed": True,
            },
            {"action": "hold", "is_entry": False, "was_executed": False},
        ]
    )

    assert report["read_only"] is True
    assert report["is_entry_permission"] is False
    assert report["status"] == "balanced"
    assert report["short"]["executed_count"] == 1
    assert report["short"]["block_reasons"] == {"risk_blocked": 1}
    assert report["long"]["block_reasons"] == {"insufficient_evidence": 1}
