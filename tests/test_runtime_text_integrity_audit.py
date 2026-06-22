from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

import scripts.audit_runtime_text_integrity as audit_module
from scripts.audit_runtime_text_integrity import build_runtime_text_integrity_report


def _u(escaped: str) -> str:
    return escaped.encode("ascii").decode("unicode_escape")


def test_build_runtime_text_integrity_report_counts_suspected_fields() -> None:
    now = datetime(2026, 6, 23, tzinfo=UTC)
    rows = [
        (
            "ai_decisions",
            SimpleNamespace(
                id=1,
                created_at=now - timedelta(minutes=3),
                reasoning="clean",
                execution_reason=_u("\\u93c8\\u8f70\\u7d30\\u7487\\u52eb\\u578e"),
                raw_llm_response={"note": "clean"},
            ),
        ),
        (
            "expert_memories",
            SimpleNamespace(
                id=2,
                created_at=now - timedelta(minutes=2),
                updated_at=now - timedelta(minutes=1),
                lesson="clean",
                market_pattern="clean",
                recommended_action="hold",
                extra={"note": _u("\\u93c8\\u8f70\\u7d30\\u7487\\u52eb\\u578e")},
            ),
        ),
        (
            "strategy_learning_events",
            SimpleNamespace(
                id=3,
                created_at=now - timedelta(minutes=1),
                reason="clean",
                scheduler_reason="clean",
                strategy_snapshot={"note": "clean"},
                market_state={},
                side_weights={},
                expert_integrity={},
                attribution={},
            ),
        ),
    ]

    report = build_runtime_text_integrity_report(rows, generated_at=now)

    assert report["scanned_records"] == 3
    assert report["suspected_records"] == 2
    assert report["suspected_fields"] == 2
    assert report["repairable_count"] >= 0
    assert report["by_table"]["ai_decisions"]["suspected_records"] == 1
    assert report["by_table"]["expert_memories"]["suspected_fields"] == 1
    assert report["examples"][0]["field"] == "execution_reason"
    assert report["policy"]["dry_run"] is True


def test_build_runtime_text_integrity_report_returns_ok_for_clean_rows() -> None:
    now = datetime(2026, 6, 23, tzinfo=UTC)
    report = build_runtime_text_integrity_report(
        [
            (
                "ai_decisions",
                SimpleNamespace(
                    id=1,
                    created_at=now,
                    reasoning="机会评分为正",
                    execution_reason="未执行：风险拒绝",
                    raw_llm_response={"note": "正常"},
                ),
            )
        ],
        generated_at=now,
    )

    assert report["scanned_records"] == 1
    assert report["suspected_records"] == 0
    assert report["by_table"]["ai_decisions"]["status"] == "ok"
    assert report["examples"] == []


@pytest.mark.asyncio
async def test_async_main_reports_structured_warning_when_scan_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def failed_collect(**_kwargs: object) -> dict:
        raise ConnectionRefusedError("database unavailable")

    monkeypatch.setattr(audit_module, "collect_runtime_text_integrity_report", failed_collect)

    exit_code = await audit_module.async_main(["--json-indent", "0"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["status"] == "warning"
    assert payload["scanned_records"] == 0
    assert payload["suspected_records"] == 0
    assert payload["error"]["type"] == "ConnectionRefusedError"
    assert payload["policy"]["dry_run"] is True
    assert payload["policy"]["mutates_database"] is False
