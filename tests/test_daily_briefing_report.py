from __future__ import annotations

import importlib.util
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "services" / "daily_briefing.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("bb_daily_briefing", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_blocking_reasons_reads_permission_and_selection() -> None:
    module = _load_module()

    reasons = module._blocking_reasons(
        {
            "entry_permission": {
                "blocking_reasons": [
                    "direction_support_expected_net_not_positive",
                    "direction_support_quant_evidence_missing",
                ]
            },
            "paper_trade_selection": {
                "selected": False,
                "selection_reason": "no_direction",
            },
        }
    )

    assert reasons == [
        "direction_support_expected_net_not_positive",
        "direction_support_quant_evidence_missing",
        "paper_trade_selection:no_direction",
    ]


def test_blocking_reasons_is_empty_for_non_mapping_snapshot() -> None:
    module = _load_module()

    assert module._blocking_reasons(None) == []
    assert module._blocking_reasons("not-a-snapshot") == []


def test_render_text_reports_blockers_and_model_state() -> None:
    module = _load_module()
    report = {
        "window_hours": 24,
        "decisions": {
            "in_window": 300,
            "executed": 0,
            "actions": {"hold": 300},
            "top_reasons": [
                ("direction_support_expected_net_not_positive", 300),
                ("paper_trade_selection:no_direction", 300),
            ],
        },
        "shadow": {"completed_in_window": 12},
        "model": {
            "lifecycle": "canary",
            "artifact_version": "20261108T233138005216Z-95ecdd20",
            "paper_model_eligible": True,
            "live_ml_ready": False,
            "shadow_samples": 176522,
            "trade_samples": 107,
            "message": "",
        },
    }

    text = module.render_text(report)

    assert "决策 300 次" in text
    assert "实际开仓 0 次" in text
    assert "direction_support_expected_net_not_positive" in text
    assert "canary" in text
    assert "模型纸盘调用资格：True" in text
    assert "纸面执行权限" not in text
    assert "实盘权限：False" in text
    assert "窗口内完成 12 条" in text


def test_render_text_survives_model_error() -> None:
    module = _load_module()
    text = module.render_text(
        {
            "window_hours": 6,
            "decisions": {"in_window": 0, "executed": 0},
            "shadow": {},
            "model": {"error": "RuntimeError: boom"},
        }
    )

    assert "读取失败" in text
    assert "无阻塞记录" in text


@pytest.mark.asyncio
@pytest.mark.parametrize("decision_limit", [1, 100])
async def test_build_briefing_counts_full_paper_window_and_only_entry_executions(
    monkeypatch: pytest.MonkeyPatch, decision_limit: int,
) -> None:
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

    from models.base import Base
    from models.decision import AIDecision
    from models.learning import ShadowBacktest

    module = _load_module()
    now = datetime.now(UTC)
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    @asynccontextmanager
    async def read_session(**_kwargs):
        async with AsyncSession(engine) as session:
            yield session

    monkeypatch.setattr("db.session.get_read_session_ctx", read_session)
    monkeypatch.setattr(module, "_model_status", lambda: {})

    def decision(row_id, action, *, minutes=5, paper=True, executed=False,
                 executed_at=None, analysis_type="market", reason=None):
        return {
            "id": row_id, "model_name": "test", "symbol": "BTC/USDT",
            "confidence": 0.5, "action": action, "is_paper": paper,
            "was_executed": executed, "executed_at": executed_at,
            "analysis_type": analysis_type, "execution_reason": reason,
            "created_at": now - timedelta(minutes=minutes),
        }

    try:
        async with engine.begin() as connection:
            await connection.run_sync(
                lambda sync_connection: Base.metadata.create_all(
                    sync_connection,
                    tables=[AIDecision.__table__, ShadowBacktest.__table__],
                )
            )
            await connection.execute(AIDecision.__table__.insert(), [
                decision(1, "short", executed=True, reason="entry filled"),
                decision(2, "long", executed_at=now, reason="recovered fill"),
                decision(3, "close_short", executed=True, analysis_type="position"),
                decision(4, "hold", minutes=4, reason="negative edge"),
                decision(5, "short", minutes=3, reason="risk budget"),
                decision(6, "hold", minutes=2, analysis_type="position", reason="position review"),
                decision(7, "short", paper=False, executed=True),
                decision(8, "short", minutes=120, executed=True),
                decision(9, "short", minutes=-60, executed=True),
            ])
            await connection.execute(ShadowBacktest.__table__.insert(), [
                {
                    "model_name": "test", "symbol": "BTC/USDT",
                    "execution_mode": mode, "status": status,
                    "due_at": now, "created_at": created_at,
                }
                for mode, status, created_at in [
                    ("paper", "completed", now - timedelta(minutes=5)),
                    ("live", "completed", now - timedelta(minutes=5)),
                    ("paper", "completed", now - timedelta(hours=2)),
                    ("paper", "completed", now + timedelta(hours=1)),
                    ("paper", "pending", now - timedelta(minutes=5)),
                ]
            ])

        report = await module.build_briefing(window_hours=1, decision_limit=decision_limit)
        counts = report["decisions"]
        assert counts["in_window"] == 6
        assert counts["executed"] == 2
        assert counts["actions"] == {"short": 2, "long": 1, "close_short": 1, "hold": 2}
        assert counts["reason_sample_count"] == min(decision_limit, 2)
        assert counts["reason_total_count"] == 2
        assert counts["reasons_sampled"] is (decision_limit < 2)
        assert counts["top_reasons"] == (
            [("risk budget", 1)] if decision_limit == 1
            else [("risk budget", 1), ("negative edge", 1)]
        )
        assert report["shadow"]["completed_in_window"] == 1
        text = module.render_text(report)
        assert "决策 6 次，实际开仓 2 次" in text
        assert ("最近 1 条未开仓决策" in text) is (decision_limit == 1)
    finally:
        await engine.dispose()


def test_model_status_distinguishes_running_artifact_from_rejected_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from services.ml_signal_service import MLSignalService

    module = _load_module()
    monkeypatch.setattr(
        MLSignalService,
        "status_metadata",
        lambda _self: {
            "artifact_version": "running-champion",
            "latest_training": {
                "artifact_version": "rejected-candidate",
                "champion_retained": True,
                "sample_cursor": {"shadow": 1300, "trade": 107},
            },
        },
    )

    model = module._model_status()
    text = module.render_text({"model": model})

    assert model["artifact_version"] == "running-champion"
    assert model["latest_training_version"] == "rejected-candidate"
    assert "运行版本：running-champion" in text
    assert "最近候选：rejected-candidate（未替换运行模型）" in text
