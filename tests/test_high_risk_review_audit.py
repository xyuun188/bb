from services.high_risk_review_audit import summarize_high_risk_review


def test_high_risk_review_audit_summarizes_independent_gate() -> None:
    report = summarize_high_risk_review(
        [
            {
                "id": 1,
                "symbol": "BTC/USDT",
                "action": "long",
                "was_executed": False,
                "raw_llm_response": {
                    "high_risk_review": {
                        "triggered": True,
                        "status": "completed",
                        "approved": False,
                        "hard_review_required": True,
                        "reasons": ["large_position"],
                        "reason": "risk asymmetric",
                    }
                },
            },
            {
                "id": 2,
                "symbol": "ETH/USDT",
                "action": "short",
                "was_executed": True,
                "raw_llm_response": {
                    "high_risk_review": {
                        "triggered": True,
                        "status": "pending",
                        "approved": None,
                        "hard_review_required": True,
                        "reasons": ["ml_ai_direction_conflict"],
                    }
                },
            },
            {
                "id": 3,
                "symbol": "SOL/USDT",
                "action": "hold",
                "was_executed": False,
                "raw_llm_response": {"high_risk_review": {"triggered": True}},
            },
            {
                "id": 4,
                "symbol": "XRP/USDT",
                "action": "long",
                "was_executed": False,
                "raw_llm_response": {
                    "high_risk_review": {
                        "triggered": False,
                        "status": "skipped_advisory_only",
                        "approved": True,
                        "advisory_reasons": ["today_recovery_after_loss"],
                    }
                },
            },
        ],
        hours=24,
    )

    assert report["audit_only"] is True
    assert report["entry_decision_count"] == 3
    assert report["review_payload_count"] == 3
    assert report["hard_review_required_count"] == 2
    assert report["blocked_count"] == 1
    assert report["executed_without_required_review_count"] == 1
    assert report["status_counts"]["completed"] == 1
    assert report["status_counts"]["pending"] == 1
    assert report["trigger_counts"] == {"triggered": 2, "not_triggered": 1}
    assert report["approved_counts"]["approved_false"] == 1
    assert report["reason_counts"]["large_position"] == 1
    assert report["policy"]["hard_review_must_approve_before_execution"] is True
    assert report["policy"]["ordinary_entries_must_not_call_high_risk_review"] is True
    assert report["policy"]["failed_required_review_blocks_entry"] is True
    assert report["policy"]["audit_can_bypass_risk_controls"] is False
