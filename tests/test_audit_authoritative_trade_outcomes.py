from scripts.audit_authoritative_trade_outcomes import _gap_summary


def test_gap_summary_classifies_complete_and_recovery_candidates() -> None:
    summary = _gap_summary(
        [
            {"outcome_id": "complete", "outcome_evidence_gaps": []},
            {
                "outcome_id": "linkage",
                "outcome_evidence_gaps": [
                    "missing_position_history_entry_orders",
                    "missing_exact_entry_order_decision_link",
                ],
            },
            {
                "outcome_id": "spec",
                "outcome_evidence_gaps": [
                    "missing_contract_ct_val",
                    "missing_local_position_strategy_lineage",
                ],
            },
            {
                "outcome_id": "official",
                "outcome_evidence_gaps": ["missing_official_fee"],
            },
        ]
    )

    assert summary["recovery_class_counts"] == {
        "complete": 1,
        "linkage_only_candidate": 1,
        "linkage_and_exchange_spec_candidate": 1,
        "official_fact_or_settlement_gap": 1,
    }
    assert summary["gap_counts"]["missing_position_history_entry_orders"] == 1
    assert len(summary["incomplete_samples"]) == 3


def test_gap_summary_deduplicates_repeated_gaps_per_outcome() -> None:
    summary = _gap_summary(
        [
            {
                "outcome_id": "duplicate-gap",
                "outcome_evidence_gaps": [
                    "missing_position_history_entry_orders",
                    "missing_position_history_entry_orders",
                ],
            }
        ]
    )

    assert summary["gap_counts"] == {"missing_position_history_entry_orders": 1}
    assert summary["gap_set_counts"] == [
        {"evidence_gaps": ["missing_position_history_entry_orders"], "count": 1}
    ]
