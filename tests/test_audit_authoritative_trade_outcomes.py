import base64
import gzip
import json
from datetime import UTC, datetime
from types import SimpleNamespace

from scripts.audit_authoritative_trade_outcomes import (
    _compact_gap_summary,
    _gap_summary,
    _online_report,
    _promotion_gate,
    _realized_pnl_sign_counts,
    _slippage_integrity_summary,
    _slippage_storage_summary,
    _strategy_entry_kind_counts,
    _training_exclusion_summary,
    _training_outcome_query_kwargs,
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


def test_trade_audit_counts_profit_and_loss_without_filtering_either_side() -> None:
    assert _realized_pnl_sign_counts(
        [
            {"realized_pnl": 1.25},
            {"realized_pnl": -0.5},
            {"realized_pnl": 0.0},
            {"realized_pnl": None},
        ]
    ) == {"profit": 1, "loss": 1, "flat": 2}


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


def test_authoritative_audit_gate_contract_requires_complete_trainable_outcomes() -> None:
    result = _promotion_gate(
        [{"outcome_complete": False}],
        trainable_count=0,
    )

    assert result["status"] == "blocked"
    assert result["blocked_reasons"] == [
        "authoritative_outcomes_all_incomplete",
        "authoritative_training_samples_missing",
    ]


def test_training_exclusion_summary_counts_each_reason_once_per_outcome() -> None:
    summary = _training_exclusion_summary(
        {
            "record_count": 2,
            "included_count": 1,
            "excluded_count": 1,
            "records": [
                {
                    "exclusion_reasons": [
                        "missing_feature_availability",
                        "missing_feature_availability",
                    ]
                },
                {"exclusion_reasons": ["missing_authoritative_fee"]},
                "invalid-record",
            ],
        }
    )

    assert summary == {
        "record_count": 3,
        "included_count": 1,
        "excluded_count": 1,
        "excluded_reason_counts": {
            "missing_authoritative_fee": 1,
            "missing_feature_availability": 1,
        },
    }


def test_training_outcome_audit_matches_current_training_projection(monkeypatch) -> None:
    start = datetime(2026, 10, 1, tzinfo=UTC)
    monkeypatch.setattr(
        "scripts.audit_authoritative_trade_outcomes.load_training_data_start",
        lambda: start,
    )

    kwargs = _training_outcome_query_kwargs(mode="paper", limit=500)

    assert kwargs == {
        "mode": "paper",
        "since": start,
        "limit": 500,
        "compact": True,
        "include_training_features": True,
        "include_decision_evidence": True,
    }


def test_online_audit_decodes_compressed_json_without_ssh_truncation(
    monkeypatch,
) -> None:
    report = {"status": "ok", "summary_only": True}
    envelope = {
        "returncode": 0,
        "stdout_gzip_b64": base64.b64encode(
            gzip.compress(json.dumps(report).encode("utf-8"))
        ).decode("ascii"),
        "stderr": "",
    }
    calls = {}

    class FakeSSH:
        def close(self) -> None:
            calls["closed"] = True

    def fake_run_remote_text(_ssh, command, **kwargs):
        calls["command"] = command
        calls["kwargs"] = kwargs
        return json.dumps(envelope)

    monkeypatch.setattr(
        "scripts.audit_authoritative_trade_outcomes.connect_remote_ssh",
        lambda *_args, **_kwargs: FakeSSH(),
    )
    monkeypatch.setattr(
        "scripts.audit_authoritative_trade_outcomes.run_remote_text",
        fake_run_remote_text,
    )
    monkeypatch.setattr(
        "scripts.audit_authoritative_trade_outcomes.safe_print",
        lambda value: calls.setdefault("printed", value),
    )

    result = _online_report(mode="paper", position_id=None, summary_only=False)

    assert result == report
    assert calls["printed"] == json.dumps(report)
    assert calls["closed"] is True
    assert calls["kwargs"]["max_output_chars"] == 200_000
    assert "stdout_gzip_b64" in calls["command"]
    assert "--summary-only" in calls["command"]
