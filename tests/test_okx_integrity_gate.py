from __future__ import annotations

from services.okx_integrity_gate import partition_okx_integrity_issues


def test_okx_integrity_gate_quarantines_only_registered_warning_kinds() -> None:
    blocking, quarantined = partition_okx_integrity_issues(
        {
            "issue_count": 2,
            "warning_count": 2,
            "severity_counts": {"warning": 2},
            "issues": [
                {"kind": "okx_fill_not_linked_to_position", "severity": "warning"},
                {"kind": "new_unregistered_warning", "severity": "warning"},
            ],
        }
    )

    assert [item["kind"] for item in quarantined] == ["okx_fill_not_linked_to_position"]
    assert [item["kind"] for item in blocking] == ["new_unregistered_warning"]


def test_okx_integrity_gate_quarantines_native_full_close_identity_only_evidence() -> None:
    blocking, quarantined = partition_okx_integrity_issues(
        {
            "issue_count": 1,
            "warning_count": 1,
            "severity_counts": {"warning": 1},
            "issues": [
                {
                    "kind": "native_full_close_identity_quarantined",
                    "severity": "warning",
                }
            ],
        }
    )

    assert blocking == []
    assert [item["kind"] for item in quarantined] == [
        "native_full_close_identity_quarantined"
    ]


def test_okx_integrity_gate_fails_closed_for_missing_warning_details() -> None:
    blocking, quarantined = partition_okx_integrity_issues(
        {
            "issue_count": 3,
            "manual_review_count": 3,
            "severity_counts": {"warning": 3},
            "issues": [],
        }
    )

    assert quarantined == []
    assert blocking == [{"kind": "unclassified_trade_fact_warning"}]


def test_okx_integrity_gate_blocks_critical_and_repairable_issues() -> None:
    blocking, quarantined = partition_okx_integrity_issues(
        {
            "issue_count": 1,
            "critical_count": 1,
            "repairable_count": 1,
            "issues": [{"kind": "position_quantity_mismatch", "severity": "critical"}],
        }
    )

    assert quarantined == []
    assert {item["kind"] for item in blocking} == {
        "position_quantity_mismatch",
        "trade_fact_repairable_issue",
    }
