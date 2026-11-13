from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy.dialects import postgresql

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


def test_runtime_text_scan_uses_compact_decision_and_shadow_projections() -> None:
    assert "decision_learning_snapshot" in audit_module.TEXT_AUDIT_FIELDS["ai_decisions"]
    assert "raw_llm_response" not in audit_module.TEXT_AUDIT_FIELDS["ai_decisions"]
    assert audit_module.TEXT_AUDIT_FIELDS["shadow_backtests"] == (
        "training_feature_snapshot",
        "note",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("ensure_ascii", [True, False])
async def test_runtime_text_scan_bounds_projection_reads_and_preserves_full_report(
    monkeypatch: pytest.MonkeyPatch,
    ensure_ascii: bool,
) -> None:
    now = datetime.now(UTC)
    record_ids = list(range(47, 0, -1))
    rows = {
        record_id: {
            "id": record_id,
            "created_at": now - timedelta(seconds=47 - record_id),
            "updated_at": now,
            "reason": "clean",
            "scheduler_reason": "clean",
            "strategy_snapshot": {
                "note": (
                    _u("\\u93c8\\u8f70\\u7d30\\u7487\\u52eb\\u578e")
                    if record_id % 3 == 0
                    else "clean"
                )
            },
            "market_state": {},
            "side_weights": {},
            "expert_integrity": {},
            "attribution": {},
        }
        for record_id in record_ids
    }
    read_sizes: list[int] = []
    open_sessions = 0
    original_builder = audit_module.build_runtime_text_integrity_report

    class Result:
        def __init__(self, values: list) -> None:
            self.values = values

        def scalars(self):
            return self

        def mappings(self):
            return self

        def all(self):
            return self.values

    class Session:
        async def execute(self, statement):
            if len(statement.selected_columns) == 1:
                return Result(record_ids)
            batch_ids = next(
                value for value in statement.compile().params.values() if isinstance(value, list)
            )
            read_sizes.append(len(batch_ids))
            assert len(batch_ids) <= 5
            serialized_rows = []
            for record_id in reversed(batch_ids):
                row = dict(rows[record_id])
                for field in ("strategy_snapshot", "market_state", "side_weights",
                              "expert_integrity", "attribution"):
                    row[field] = json.dumps(row[field], ensure_ascii=ensure_ascii)
                serialized_rows.append(row)
            return Result(serialized_rows)

    @asynccontextmanager
    async def session_context():
        nonlocal open_sessions
        open_sessions += 1
        try:
            yield Session()
        finally:
            open_sessions -= 1

    def build_outside_transaction(*args, **kwargs):
        assert open_sessions == 0
        return original_builder(*args, **kwargs)

    monkeypatch.setattr(
        audit_module,
        "MODEL_BY_TABLE",
        {"strategy_learning_events": audit_module.StrategyLearningEvent},
    )
    monkeypatch.setattr(audit_module, "get_read_session_ctx", session_context)
    monkeypatch.setattr(audit_module, "build_runtime_text_integrity_report", build_outside_transaction)

    report = await audit_module.collect_runtime_text_integrity_report(
        limit_per_table=47, example_limit=4,
    )
    expected = original_builder(
        [("strategy_learning_events", SimpleNamespace(**rows[record_id])) for record_id in record_ids],
        generated_at=datetime.fromisoformat(report["generated_at"]),
        example_limit=4,
    )

    assert sum(read_sizes) == 47
    assert report == expected


def test_runtime_text_projection_reads_json_as_text_without_decoding_other_columns() -> None:
    statement = audit_module._runtime_text_projection_statement(
        "strategy_learning_events", audit_module.StrategyLearningEvent, record_ids=[7],
    )
    sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "CAST(strategy_learning_events.strategy_snapshot AS TEXT)" in sql
    assert "CAST(strategy_learning_events.reason" not in sql


def test_clean_serialized_json_skips_full_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime.now(UTC)
    serialized = json.dumps({"nested": ["机会评分为正", {"value": 1.5}]})

    def forbidden_decode(_value):
        pytest.fail("clean snapshots must not build a decoded object graph")

    monkeypatch.setattr(audit_module.json, "loads", forbidden_decode)
    report = build_runtime_text_integrity_report([
        ("strategy_learning_events", SimpleNamespace(
            id=1, created_at=now,
            strategy_snapshot=audit_module._SerializedJSON(serialized),
        )),
    ])
    assert report["scanned_records"] == 1
    assert report["suspected_records"] == 0


def test_serialized_json_marker_prefilter_does_not_turn_clean_keys_into_bad_fields() -> None:
    now = datetime.now(UTC)
    value = {_u("\\u93c8\\u93c8"): "clean"}
    report = build_runtime_text_integrity_report([
        ("strategy_learning_events", SimpleNamespace(
            id=1, created_at=now,
            strategy_snapshot=audit_module._SerializedJSON(json.dumps(value)),
        )),
    ])
    assert report["scanned_records"] == 1
    assert report["suspected_records"] == 0


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
