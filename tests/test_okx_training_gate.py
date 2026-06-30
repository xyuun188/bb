from __future__ import annotations

import json
from datetime import UTC, datetime

from services.okx_training_gate import okx_training_refresh_gate


def _write_report(tmp_path, payload: dict) -> None:
    path = tmp_path / "okx_daily_reconciliation_reports" / "latest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_okx_training_gate_allows_clean_training_refresh(tmp_path) -> None:
    _write_report(
        tmp_path,
        {
            "status": "warning",
            "generated_at": "2026-06-27T01:00:00+00:00",
            "requires_attention": False,
            "can_open_new_entries": False,
            "can_refresh_training": True,
            "operational_gates": {
                "entry_blocked": True,
                "training_blocked": False,
                "attention_buckets": {"entry": 1, "training": 0, "manual_review": 0},
                "entry_blockers": [{"code": "trading_runtime_heartbeat_stale"}],
                "training_blockers": [],
            },
            "issue_ledger": {
                "summary": {"fixed": 3, "observing": 1, "unresolved": 0, "total": 4}
            },
        },
    )

    gate = okx_training_refresh_gate(
        data_dir=tmp_path,
        now=datetime(2026, 6, 27, 2, 0, tzinfo=UTC),
    )

    assert gate["allowed"] is True
    assert gate["reason"] == "okx_daily_reconciliation_allows_training_refresh"
    assert gate["can_open_new_entries"] is False
    assert gate["can_refresh_training"] is True
    assert gate["issue_ledger_summary"]["unresolved"] == 0


def test_okx_training_gate_blocks_stale_or_attention_reports(tmp_path) -> None:
    _write_report(
        tmp_path,
        {
            "status": "warning",
            "generated_at": "2026-06-20T01:00:00+00:00",
            "requires_attention": True,
            "can_refresh_training": False,
            "operational_gates": {"training_blocked": True},
        },
    )

    gate = okx_training_refresh_gate(
        data_dir=tmp_path,
        now=datetime(2026, 6, 27, 2, 0, tzinfo=UTC),
    )

    assert gate["allowed"] is False
    assert gate["reason"] == "okx_daily_reconciliation_report_stale"
    assert gate["training_blocked"] is True
