from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from scripts import run_okx_daily_reconciliation_report as report_script


def test_okx_daily_reconciliation_report_uses_online_runtime_bootstrap() -> None:
    source = report_script.Path(report_script.__file__).read_text(encoding="utf-8")

    assert "from scripts.runtime_env_bootstrap import" in source
    assert "load_runtime_env_files(project_root=ROOT)" in source
    assert "drop_privileges_to_runtime_user_if_needed(project_root=ROOT)" in source


def test_online_report_command_forces_read_only_online_runtime() -> None:
    args = report_script._parse_args(["--online", "--json-indent", "0"])

    command = report_script._online_report_command(args)

    assert "cd /data/bb/app" in command
    assert "postgresql+asyncpg://bb@/bb_trading?host=/var/run/postgresql" in command
    assert "--stdout-only" in command
    assert "--online" not in command
    assert "--output-dir" not in command


def test_run_online_report_relays_status_without_writing_or_mutating(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[dict[str, object]] = []

    class FakeSsh:
        closed = False

        def close(self) -> None:
            self.closed = True

    ssh = FakeSsh()

    def fake_run_remote_text(_ssh, command, **kwargs):
        calls.append({"command": command, **kwargs})
        return json.dumps(
            {
                "status": "ok",
                "issue_ledger": {"summary": {"unresolved": 0}},
                "summary": {"critical": 0},
                "mutates_database": False,
                "live_order_mutation": False,
            }
        )

    monkeypatch.setattr(report_script, "connect_remote_ssh", lambda *_args, **_kwargs: ssh)
    monkeypatch.setattr(report_script, "run_remote_text", fake_run_remote_text)
    args = report_script._parse_args(["--online", "--stdout-only"])

    assert report_script.run_online_report(args) == 0
    assert ssh.closed is True
    assert calls[0]["check"] is False
    assert "--stdout-only" in str(calls[0]["command"])
    assert json.loads(capsys.readouterr().out)["mutates_database"] is False


def test_online_report_parser_ignores_runtime_logs_before_json() -> None:
    report = report_script._last_json_object(
        "2026-07-22 [info] executor initialized\n"
        "2026-07-22 [info] executor closed\n"
        '{"status":"warning","mutates_database":false}'
    )

    assert report == {"status": "warning", "mutates_database": False}


@pytest.mark.asyncio
async def test_daily_reconciliation_uses_unbounded_read_only_close_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int | None] = []

    async def fake_audit(*, max_close_orders: int | None = 300) -> dict[str, object]:
        calls.append(max_close_orders)
        return {"key": "okx_reconciliation", "status": "ok"}

    monkeypatch.setattr(
        report_script.system_audit,
        "_okx_reconciliation_audit",
        fake_audit,
    )

    result = await report_script._full_okx_reconciliation_audit()

    assert result["status"] == "ok"
    assert calls == [None]


def _card(key: str, status: str) -> dict[str, object]:
    return {
        "key": key,
        "title": key,
        "status": status,
        "summary": f"{key} {status}",
        "details": {"dry_run": True},
        "evidence": [],
        "next_actions": [],
    }


def test_okx_daily_reconciliation_report_builds_issue_ledger() -> None:
    report = report_script.build_report(
        [
            _card("okx_reconciliation", "ok"),
            _card("okx_trade_fact_integrity", "warning"),
            _card("position_price_integrity", "ok"),
            _card("trade_execution_contract", "ok"),
        ],
        generated_at=datetime(2026, 6, 26, 21, 55, tzinfo=UTC),
    )

    assert report["report_type"] == "okx_daily_reconciliation"
    assert report["status"] == "warning"
    assert report["dry_run"] is True
    assert report["mutates_database"] is False
    assert report["live_order_mutation"] is False
    assert report["repair_apply_enabled"] is False
    assert report["requires_attention"] is True
    assert report["can_open_new_entries"] is False
    assert report["can_refresh_training"] is False
    assert report["summary"] == {"cards": 4, "critical": 0, "warning": 1, "ok": 3}
    assert report["issue_ledger"]["summary"]["total"] == 4
    assert report["operational_gates"]["training_blocked"] is True
    assert report["cards"][1]["key"] == "okx_trade_fact_integrity"


def test_okx_daily_reconciliation_report_writes_dated_and_latest_files(tmp_path) -> None:
    report = report_script.build_report(
        [_card("okx_reconciliation", "ok")],
        generated_at=datetime(2026, 6, 26, 21, 56, tzinfo=UTC),
    )

    artifacts = report_script.write_report(report, tmp_path, indent=2)

    report_path = tmp_path / artifacts["report_path"].split("\\")[-1]
    latest_path = tmp_path / "latest.json"
    assert report_path.exists()
    assert latest_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["report_type"] == "okx_daily_reconciliation"
    assert payload["completed"] is True
    assert payload["artifacts"] == artifacts
    assert latest_path.read_text(encoding="utf-8") == report_path.read_text(encoding="utf-8")
    assert not list(tmp_path.glob("*.tmp"))


def test_okx_daily_reconciliation_report_serializes_nested_datetimes(tmp_path) -> None:
    card = _card("okx_trade_fact_integrity", "ok")
    card["details"] = {
        "checked_at": datetime(2026, 6, 27, 1, 2, 3, tzinfo=UTC),
        "samples": [{"seen_at": datetime(2026, 6, 27, 1, 2, 4, tzinfo=UTC)}],
    }

    report = report_script.build_report(
        [card],
        generated_at=datetime(2026, 6, 27, 1, 2, tzinfo=UTC),
    )
    artifacts = report_script.write_report(report, tmp_path, indent=0)

    payload = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert payload["cards"][0]["details"]["checked_at"] == "2026-06-27T01:02:03+00:00"
    assert payload["cards"][0]["details"]["samples"][0]["seen_at"] == (
        "2026-06-27T01:02:04+00:00"
    )
    assert artifacts["latest_path"].endswith("latest.json")


def test_okx_daily_reconciliation_observing_only_warning_exits_success() -> None:
    card = _card("okx_trade_fact_integrity", "warning")
    card["details"] = {
        "issue_count": 0,
        "critical_count": 0,
        "position_fact_link_repair": {"candidate_link_count": 0},
        "okx_authoritative_sync": {"issue_count": 0},
        "runtime_okx_entry_gate": {
            "entry_blocked": True,
            "blocker": "trading_runtime_heartbeat_stale",
            "status": "runtime_heartbeat_stale",
            "sync_status": "ok",
        },
    }
    report = report_script.build_report(
        [
            _card("okx_reconciliation", "ok"),
            card,
            _card("position_price_integrity", "ok"),
            _card("trade_execution_contract", "ok"),
        ],
        generated_at=datetime(2026, 6, 27, 0, 35, tzinfo=UTC),
    )

    assert report["status"] == "warning"
    assert report["issue_ledger"]["summary"] == {
        "fixed": 3,
        "unresolved": 0,
        "observing": 1,
        "total": 4,
    }
    assert report["can_open_new_entries"] is False
    assert report["can_refresh_training"] is True
    assert report["requires_attention"] is False
    assert report["operational_gates"]["entry_blocked"] is True
    assert report["operational_gates"]["training_blocked"] is False
    assert report["operational_gates"]["entry_blockers"][0]["code"] == (
        "trading_runtime_heartbeat_stale"
    )
    assert report_script.exit_code_for_report(report) == 0


def test_okx_daily_reconciliation_info_only_integrity_issue_does_not_block_training() -> None:
    card = _card("okx_trade_fact_integrity", "ok")
    card["details"] = {
        "issue_count": 1,
        "critical_count": 0,
        "warning_count": 0,
        "severity_counts": {"info": 1},
        "kind_counts": {"orphan_position_quarantine_not_exchange_backed": 1},
        "position_fact_link_repair": {"candidate_link_count": 0},
        "okx_authoritative_sync": {
            "okx_pull_available": True,
            "issue_count": 0,
            "manual_review_count": 0,
            "repairable_count": 0,
            "severity_counts": {},
        },
        "runtime_okx_entry_gate": {
            "entry_blocked": False,
            "status": "ok",
            "sync_status": "ok",
        },
    }

    report = report_script.build_report(
        [
            _card("okx_reconciliation", "ok"),
            card,
            _card("position_price_integrity", "ok"),
            _card("trade_execution_contract", "ok"),
        ],
        generated_at=datetime(2026, 6, 27, 0, 36, tzinfo=UTC),
    )

    assert report["issue_ledger"]["summary"]["unresolved"] == 0
    assert report["can_open_new_entries"] is True
    assert report["can_refresh_training"] is True
    assert report["requires_attention"] is False


def test_okx_daily_reconciliation_authoritative_timeout_does_not_block_when_runtime_sync_healthy() -> None:
    card = _card("okx_trade_fact_integrity", "warning")
    card["details"] = {
        "issue_count": 0,
        "critical_count": 0,
        "warning_count": 0,
        "severity_counts": {},
        "kind_counts": {},
        "position_fact_link_repair": {"candidate_link_count": 0},
        "okx_authoritative_sync": {
            "okx_pull_available": False,
            "issue_count": 0,
            "manual_review_count": 0,
            "repairable_count": 0,
            "severity_counts": {},
        },
        "runtime_okx_entry_gate": {
            "entry_blocked": False,
            "sync_status": "ok",
            "status": "ok",
            "last_requires_attention_count": 0,
        },
    }

    report = report_script.build_report(
        [
            _card("okx_reconciliation", "ok"),
            card,
            _card("position_price_integrity", "ok"),
            _card("trade_execution_contract", "ok"),
        ],
        generated_at=datetime(2026, 6, 29, 13, 10, tzinfo=UTC),
    )

    assert report["issue_ledger"]["summary"]["unresolved"] == 0
    assert report["issue_ledger"]["summary"]["observing"] == 1
    assert report["can_open_new_entries"] is True
    assert report["can_refresh_training"] is True
    assert report["requires_attention"] is False


def test_okx_daily_reconciliation_superseded_link_candidates_do_not_block_training() -> None:
    card = _card("okx_trade_fact_integrity", "ok")
    card["details"] = {
        "issue_count": 2,
        "critical_count": 0,
        "warning_count": 0,
        "severity_counts": {"info": 2},
        "kind_counts": {"superseded_position_residual": 2},
        "issues": [
            {
                "kind": "superseded_position_residual",
                "severity": "info",
                "position_id": 846,
            },
            {
                "kind": "superseded_position_residual",
                "severity": "info",
                "position_id": 848,
            },
        ],
        "position_fact_link_repair": {
            "candidate_link_count": 2,
            "diagnostics": [
                {"position_id": 846, "status": "manual_review"},
                {"position_id": 848, "status": "manual_review"},
            ],
        },
        "okx_authoritative_sync": {
            "okx_pull_available": True,
            "issue_count": 0,
            "manual_review_count": 0,
            "repairable_count": 0,
            "severity_counts": {},
        },
        "runtime_okx_entry_gate": {
            "entry_blocked": False,
            "status": "ok",
            "sync_status": "ok",
        },
    }

    report = report_script.build_report(
        [
            _card("okx_reconciliation", "ok"),
            card,
            _card("position_price_integrity", "ok"),
            _card("trade_execution_contract", "ok"),
        ],
        generated_at=datetime(2026, 6, 27, 0, 37, tzinfo=UTC),
    )

    assert report["issue_ledger"]["summary"]["unresolved"] == 0
    assert report["can_open_new_entries"] is True
    assert report["can_refresh_training"] is True
    assert report["requires_attention"] is False


def test_okx_daily_reconciliation_warning_info_only_residuals_do_not_block_entry() -> None:
    card = _card("okx_trade_fact_integrity", "warning")
    card["details"] = {
        "issue_count": 4,
        "critical_count": 0,
        "warning_count": 0,
        "severity_counts": {"info": 4},
        "issues": [
            {
                "kind": "superseded_position_residual",
                "severity": "info",
                "position_id": 987,
                "symbol": "GRASS/USDT",
            },
            {
                "kind": "superseded_position_residual",
                "severity": "info",
                "position_id": 984,
                "symbol": "LIT/USDT",
            },
            {
                "kind": "superseded_position_residual",
                "severity": "info",
                "position_id": 973,
                "symbol": "ETHFI/USDT",
            },
            {
                "kind": "superseded_position_residual",
                "severity": "info",
                "position_id": 985,
                "symbol": "LAB/USDT",
            },
        ],
        "position_fact_link_repair": {
            "candidate_link_count": 4,
            "diagnostics": [
                {"position_id": 987, "status": "manual_review"},
                {"position_id": 984, "status": "manual_review"},
                {"position_id": 973, "status": "manual_review"},
                {"position_id": 985, "status": "manual_review"},
            ],
        },
        "okx_authoritative_sync": {
            "okx_pull_available": True,
            "status": "ok",
            "issue_count": 0,
            "manual_review_count": 0,
            "repairable_count": 0,
            "severity_counts": {},
        },
        "runtime_okx_entry_gate": {
            "entry_blocked": False,
            "status": "ok",
            "sync_status": "ok",
            "last_requires_attention_count": 0,
        },
    }

    report = report_script.build_report(
        [
            _card("okx_reconciliation", "ok"),
            card,
            _card("position_price_integrity", "ok"),
            _card("trade_execution_contract", "ok"),
        ],
        generated_at=datetime(2026, 6, 30, 18, 35, tzinfo=UTC),
    )

    assert report["issue_ledger"]["summary"]["unresolved"] == 0
    assert report["issue_ledger"]["summary"]["observing"] == 1
    assert report["can_open_new_entries"] is True
    assert report["can_refresh_training"] is True
    assert report["requires_attention"] is False


def test_okx_daily_reconciliation_quarantined_integrity_warnings_do_not_block_training() -> None:
    card = _card("okx_trade_fact_integrity", "warning")
    card["details"] = {
        "issue_count": 2,
        "critical_count": 0,
        "warning_count": 2,
        "severity_counts": {"warning": 2},
        "issues": [
            {
                "kind": "order_position_missing",
                "severity": "warning",
                "order_id": 3078,
                "symbol": "FIL/USDT",
            },
            {
                "kind": "position_order_link_missing_local_order",
                "severity": "warning",
                "position_id": 928,
                "symbol": "ENA/USDT",
            },
        ],
        "position_fact_link_repair": {
            "candidate_link_count": 2,
            "diagnostics": [
                {"position_id": 928, "status": "manual_review"},
                {"position_id": 973, "status": "manual_review"},
            ],
        },
        "okx_authoritative_sync": {
            "okx_pull_available": True,
            "status": "ok",
            "issue_count": 0,
            "manual_review_count": 0,
            "repairable_count": 0,
            "severity_counts": {},
        },
        "runtime_okx_entry_gate": {
            "entry_blocked": False,
            "status": "ok",
            "sync_status": "ok",
            "last_requires_attention_count": 0,
        },
    }

    report = report_script.build_report(
        [
            _card("okx_reconciliation", "ok"),
            card,
            _card("position_price_integrity", "ok"),
            _card("trade_execution_contract", "ok"),
        ],
        generated_at=datetime(2026, 7, 3, 10, 30, tzinfo=UTC),
    )

    assert report["status"] == "warning"
    assert report["issue_ledger"]["summary"]["unresolved"] == 1
    assert report["can_open_new_entries"] is True
    assert report["can_refresh_training"] is True
    assert report["requires_attention"] is False
    assert report["operational_gates"]["entry_blockers"] == []
    assert report["operational_gates"]["training_blockers"] == []


def test_okx_daily_reconciliation_unresolved_warning_exits_warning() -> None:
    report = report_script.build_report(
        [
            _card("okx_reconciliation", "warning"),
            _card("okx_trade_fact_integrity", "ok"),
            _card("position_price_integrity", "ok"),
            _card("trade_execution_contract", "ok"),
        ],
        generated_at=datetime(2026, 6, 27, 0, 36, tzinfo=UTC),
    )

    assert report["status"] == "warning"
    assert report["issue_ledger"]["summary"]["unresolved"] == 1
    assert report["can_open_new_entries"] is False
    assert report["can_refresh_training"] is False
    assert report["requires_attention"] is True
    assert report["operational_gates"]["attention_buckets"] == {
        "entry": 1,
        "training": 1,
        "manual_review": 1,
    }
    assert report_script.exit_code_for_report(report) == 1


def test_okx_daily_reconciliation_trade_execution_contract_blocks_entry_not_training() -> None:
    report = report_script.build_report(
        [
            _card("okx_reconciliation", "ok"),
            _card("okx_trade_fact_integrity", "ok"),
            _card("position_price_integrity", "ok"),
            _card("trade_execution_contract", "warning"),
        ],
        generated_at=datetime(2026, 6, 27, 0, 40, tzinfo=UTC),
    )

    assert report["status"] == "warning"
    assert report["issue_ledger"]["summary"]["unresolved"] == 1
    assert report["can_open_new_entries"] is False
    assert report["can_refresh_training"] is True
    assert report["requires_attention"] is True
    assert report["operational_gates"]["entry_blockers"][0]["code"] == (
        "unresolved_trade_execution_contract"
    )
    assert report["operational_gates"]["training_blockers"] == []
    assert report["operational_gates"]["attention_buckets"] == {
        "entry": 1,
        "training": 0,
        "manual_review": 1,
    }


def test_okx_daily_reconciliation_data_integrity_issue_blocks_entry_and_training() -> None:
    card = _card("okx_trade_fact_integrity", "warning")
    card["details"] = {
        "issue_count": 1,
        "critical_count": 0,
        "position_fact_link_repair": {"candidate_link_count": 1},
        "okx_authoritative_sync": {
            "okx_pull_available": True,
            "issue_count": 1,
            "manual_review_count": 1,
        },
        "runtime_okx_entry_gate": {
            "entry_blocked": False,
            "status": "ok",
            "sync_status": "ok",
        },
    }

    report = report_script.build_report(
        [
            _card("okx_reconciliation", "ok"),
            card,
            _card("position_price_integrity", "ok"),
            _card("trade_execution_contract", "ok"),
        ],
        generated_at=datetime(2026, 6, 27, 1, 5, tzinfo=UTC),
    )

    assert report["can_open_new_entries"] is False
    assert report["can_refresh_training"] is False
    assert report["requires_attention"] is True
    assert report["operational_gates"]["training_policy"] == "current_training_epoch_only"
    assert report["operational_gates"]["training_blockers"][0]["card_key"] == (
        "okx_trade_fact_integrity"
    )


@pytest.mark.asyncio
async def test_okx_daily_reconciliation_collect_report_clears_short_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def fake_factory() -> dict[str, object]:
        calls.append("factory")
        return _card("okx_reconciliation", "ok")

    monkeypatch.setattr(
        report_script,
        "OKX_REPORT_CARD_FACTORIES",
        (("okx_reconciliation", fake_factory),),
    )
    monkeypatch.setattr(report_script.system_audit, "_okx_reconciliation_cache", ("cached", {}))
    monkeypatch.setattr(report_script.system_audit, "_okx_authoritative_sync_cache", ("cached", {}))

    report = await report_script.collect_report(allow_cache=False)

    assert calls == ["factory"]
    assert report["status"] == "ok"
    assert report_script.system_audit._okx_reconciliation_cache is None
    assert report_script.system_audit._okx_authoritative_sync_cache is None


@pytest.mark.asyncio
async def test_okx_daily_reconciliation_collect_report_skips_self_latest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[dict[str, object]] = []

    async def fake_factory() -> dict[str, object]:
        observed.append(report_script.system_audit._load_okx_daily_reconciliation_report_summary())
        return _card("okx_trade_fact_integrity", "ok")

    monkeypatch.setattr(
        report_script,
        "OKX_REPORT_CARD_FACTORIES",
        (("okx_trade_fact_integrity", fake_factory),),
    )

    report = await report_script.collect_report(allow_cache=False)

    assert report["status"] == "ok"
    assert observed[0]["status"] == "skipped"
    assert observed[0]["requires_attention"] is False
    assert observed[0]["skip_reason"] == "daily_report_generation_avoids_self_referential_latest"


@pytest.mark.asyncio
async def test_okx_daily_reconciliation_reports_artifact_write_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def fake_collect_report(*, allow_cache: bool = False) -> dict[str, object]:
        assert allow_cache is False
        return report_script.build_report([_card("okx_reconciliation", "ok")])

    def fake_write_report(*_args, **_kwargs):
        raise PermissionError("permission denied")

    monkeypatch.setattr(report_script, "collect_report", fake_collect_report)
    monkeypatch.setattr(report_script, "write_report", fake_write_report)

    exit_code = await report_script.async_main(["--json-indent", "0"])

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert exit_code == 2
    assert payload["status"] == "critical"
    assert payload["artifact_error"]["code"] == "artifact_write_failed"
    assert "permission denied" in payload["artifact_error"]["message"]
