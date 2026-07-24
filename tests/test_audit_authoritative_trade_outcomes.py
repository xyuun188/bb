from types import SimpleNamespace

from scripts.audit_authoritative_trade_outcomes import (
    _compact_gap_summary,
    _gap_summary,
    _slippage_integrity_summary,
    _slippage_storage_summary,
    _strategy_entry_kind_counts,
)
from services.okx_execution_slippage import OKX_FILL_MARK_SLIPPAGE_VERSION


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


def test_compact_gap_summary_removes_large_samples_and_gap_combinations() -> None:
    compact = _compact_gap_summary(
        {
            "gap_counts": {"missing_authoritative_slippage": 10},
            "gap_set_counts": [{"evidence_gaps": ["gap"], "count": 10}],
            "recovery_class_counts": {"official_fact_or_settlement_gap": 10},
            "incomplete_samples": [{"outcome_id": "large"}],
        }
    )

    assert compact == {
        "gap_counts": {"missing_authoritative_slippage": 10},
        "recovery_class_counts": {"official_fact_or_settlement_gap": 10},
    }


def test_trade_audit_distinguishes_normal_exploration_and_fast_training() -> None:
    assert _strategy_entry_kind_counts(
        [
            {"strategy_entry_kind": "normal_strategy_trade"},
            {"strategy_entry_kind": "bounded_risk_paper_exploration"},
            {"strategy_entry_kind": "loss_tolerant_paper_training"},
            {"strategy_entry_kind": "loss_tolerant_paper_training"},
        ]
    ) == {
        "loss_tolerant_paper_training": 2,
        "normal_strategy_trade": 1,
        "bounded_risk_paper_exploration": 1,
    }


def test_slippage_integrity_summary_reports_exact_side_and_order_failures() -> None:
    summary = _slippage_integrity_summary(
        [
            {
                "outcome_id": "ato:1",
                "position_ids": [7],
                "entry_order_ids": ["entry-1"],
                "close_order_ids": ["close-1"],
                "entry_execution_slippage_complete": True,
                "close_execution_slippage_complete": False,
                "execution_slippage_failures": {
                    "entry": {},
                    "close": {
                        "close-1": ["stored_slippage:fill_row_mark_price_invalid"]
                    },
                },
                "outcome_evidence_gaps": ["missing_authoritative_slippage"],
            },
            {
                "outcome_id": "ato:2",
                "entry_execution_slippage_complete": False,
                "close_execution_slippage_complete": False,
                "execution_slippage_failures": {
                    "entry": {
                        "entry-2": ["authoritative_fill_fact_missing"],
                    },
                    "close": {
                        "close-1": ["stored_slippage:fill_row_mark_price_invalid"]
                    },
                },
                "outcome_evidence_gaps": ["missing_authoritative_slippage"],
            },
            {
                "outcome_id": "ato:complete",
                "outcome_evidence_gaps": [],
            },
        ]
    )

    assert summary["missing_outcome_count"] == 2
    assert summary["entry_incomplete_outcome_count"] == 1
    assert summary["close_incomplete_outcome_count"] == 2
    assert summary["unique_failed_order_count"] == 2
    assert summary["failed_order_reason_counts"] == {
        "close:stored_slippage:fill_row_mark_price_invalid": 1,
        "entry:authoritative_fill_fact_missing": 1,
    }
    assert summary["samples"][0]["position_ids"] == [7]


def test_slippage_storage_summary_classifies_version_upgrade_blockers() -> None:
    summary = _slippage_storage_summary(
        [
            SimpleNamespace(
                exchange_order_id="fills-1",
                okx_raw_fills={
                    "fills_history_confirmed": True,
                    "contract_size_verified": True,
                    "contract_size_source": "okx_public_instruments",
                    "rows": [{"tradeId": "trade-1"}],
                    "execution_slippage": {"version": "old", "complete": True},
                },
            ),
            SimpleNamespace(
                exchange_order_id="detail-1",
                okx_raw_fills={
                    "order_detail_confirmed": True,
                    "execution_slippage": {"version": "old", "complete": True},
                },
            ),
            SimpleNamespace(
                exchange_order_id="current-1",
                okx_raw_fills={
                    "execution_slippage": {
                        "version": OKX_FILL_MARK_SLIPPAGE_VERSION,
                        "complete": True,
                    }
                },
            ),
        ]
    )

    assert summary["invalid_version_order_count"] == 2
    assert summary["classification_counts"] == {
        "fills_history:rows_available:public_spec": 1,
        "order_detail:rows_missing:public_spec_missing": 1,
    }
