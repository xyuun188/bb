from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.learning import ShadowBacktest
from services.ml_signal_service import (
    MLSignalService,
    load_shadow_training_rows,
    select_shadow_training_rows,
    shadow_training_quality_report,
)
from services.training_data_quality import DATA_QUALITY_VERSION


def _service_with_metadata(metadata: dict) -> MLSignalService:
    service = MLSignalService()
    service._bundle = {"metadata": metadata}
    service._ensure_loaded = lambda: None  # type: ignore[method-assign]
    return service


async def _use_temp_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    await close_db()
    db_path = tmp_path / "ml-signal-training.db"
    monkeypatch.setattr(settings, "database_url", f"sqlite+aiosqlite:///{db_path.as_posix()}")
    await init_db()


def _db_shadow_row(
    row_id: int,
    created_at: datetime,
    *,
    action: str = "hold",
    best_action: str = "hold",
    status: str = "completed",
    long_return_pct: float | None = 0.1,
    short_return_pct: float | None = -0.1,
) -> ShadowBacktest:
    return ShadowBacktest(
        id=row_id,
        model_name="ensemble",
        execution_mode="paper",
        symbol=f"TEST{row_id}/USDT",
        analysis_type="market",
        decision_action=action,
        decision_confidence=0.7,
        entry_price=100.0,
        feature_snapshot={"current_price": 100.0},
        status=status,
        due_at=created_at + timedelta(minutes=30),
        horizon_minutes=30,
        actual_price=101.0,
        long_return_pct=long_return_pct,
        short_return_pct=short_return_pct,
        best_action=best_action,
        missed_opportunity=best_action in {"long", "short"},
        created_at=created_at,
    )


class _Classifier:
    named_steps = {"model": SimpleNamespace(classes_=[0, 1])}

    def __init__(self, positive_probability: float) -> None:
        self.positive_probability = positive_probability

    def predict_proba(self, values: object) -> np.ndarray:
        return np.array([[1.0 - self.positive_probability, self.positive_probability]])


class _Regressor:
    def __init__(self, prediction: float) -> None:
        self.prediction = prediction

    def predict(self, values: object) -> np.ndarray:
        return np.array([self.prediction])


def _shadow_row(
    row_id: int,
    *,
    action: str = "hold",
    best_action: str = "hold",
    missed: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=row_id,
        created_at=datetime(2026, 6, 23, 3, row_id % 60, tzinfo=UTC),
        decision_action=action,
        best_action=best_action,
        missed_opportunity=missed,
    )


def test_shadow_training_selection_preserves_trade_and_best_action_samples() -> None:
    recent_hold_rows = [_shadow_row(10_000 - idx) for idx in range(20)]
    trade_rows = [
        _shadow_row(1_000 - idx, action="long" if idx % 2 == 0 else "short") for idx in range(8)
    ]
    missed_rows = [
        _shadow_row(500 - idx, best_action="long" if idx % 2 == 0 else "short", missed=True)
        for idx in range(10)
    ]

    selected = select_shadow_training_rows(
        [*recent_hold_rows, *trade_rows, *missed_rows],
        limit=20,
    )

    selected_ids = [row.id for row in selected]
    non_hold_count = sum(row.decision_action in {"long", "short"} for row in selected)
    best_trade_count = sum(row.best_action in {"long", "short"} for row in selected)
    assert len(selected) == 20
    assert len(set(selected_ids)) == len(selected_ids)
    assert non_hold_count >= 5
    assert best_trade_count >= 10
    assert any(row.id in {item.id for item in recent_hold_rows} for row in selected)


@pytest.mark.asyncio
async def test_load_shadow_training_rows_combines_recent_trade_and_best_action_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    await _use_temp_db(monkeypatch, tmp_path)
    base_time = datetime(2026, 6, 23, 3, 0, tzinfo=UTC)
    recent_holds = [
        _db_shadow_row(10_000 + idx, base_time - timedelta(minutes=idx)) for idx in range(40)
    ]
    decision_trade_rows = [
        _db_shadow_row(
            1_000 + idx,
            base_time - timedelta(hours=2, minutes=idx),
            action="long" if idx % 2 == 0 else "short",
        )
        for idx in range(8)
    ]
    best_trade_rows = [
        _db_shadow_row(
            500 + idx,
            base_time - timedelta(hours=3, minutes=idx),
            best_action="long" if idx % 2 == 0 else "short",
        )
        for idx in range(14)
    ]
    excluded_rows = [
        _db_shadow_row(90, base_time, action="long", status="pending"),
        _db_shadow_row(91, base_time, best_action="short", short_return_pct=None),
    ]
    async with get_session_ctx() as session:
        session.add_all([*recent_holds, *decision_trade_rows, *best_trade_rows, *excluded_rows])

    try:
        selected = await load_shadow_training_rows(limit=20)
    finally:
        await close_db()

    selected_ids = {row.id for row in selected}
    assert len(selected) == 20
    assert all(not isinstance(row, ShadowBacktest) for row in selected)
    assert 90 not in selected_ids
    assert 91 not in selected_ids
    assert sum(row.decision_action in {"long", "short"} for row in selected) >= 5
    assert sum(row.best_action in {"long", "short"} for row in selected) >= 10
    assert any(row.id >= 10_000 for row in selected)


def test_ml_signal_quality_report_excludes_shadow_future_leakage() -> None:
    row = SimpleNamespace(
        symbol="BTC/USDT",
        analysis_type="market",
        decision_action="long",
        decision_confidence=0.72,
        horizon_minutes=30,
        feature_snapshot={
            "current_price": 100.0,
            "spread_pct": 0.01,
            "feature_timestamp": datetime(2026, 6, 23, 1, 5, tzinfo=UTC).isoformat(),
        },
        long_return_pct=0.2,
        short_return_pct=-0.1,
        best_action="long",
        missed_opportunity=False,
        due_at=datetime(2026, 6, 23, 1, 0, tzinfo=UTC),
    )

    report = shadow_training_quality_report([row])["quality_report"]

    assert report["totals"]["excluded"] == 1
    assert report["top_reasons"][0]["reason"] == "shadow:future_leakage"


def test_ml_signal_status_exposes_learning_only_readiness_reasons() -> None:
    service = _service_with_metadata(
        {
            "version": "2026-06-18T00:00:00+00:00",
            "trained_at": "2026-06-18T00:00:00+00:00",
            "sample_count": 260,
            "test_count": 65,
            "quality_report": {
                "data_quality_version": "2026-06-19.v1",
                "totals": {"total": 260, "included": 260, "downweighted": 0, "excluded": 0},
            },
            "metrics": {
                "long_auc": 1.0,
                "short_auc": 1.0,
                "long_accuracy": 1.0,
                "short_accuracy": 1.0,
                "top_long_avg_return_pct": 0.08,
                "bottom_long_avg_return_pct": -0.27,
                "top_short_avg_return_pct": 0.06,
                "bottom_short_avg_return_pct": -0.24,
                "top_long_win_rate": 1.0,
                "bottom_long_win_rate": 0.0,
                "top_short_win_rate": 1.0,
                "bottom_short_win_rate": 0.0,
            },
        }
    )

    status = service.status()

    reason_codes = {item["code"] for item in status["readiness"]["blocking_reasons"]}
    assert status["readiness_state"] == "learning_only"
    assert status["allow_live_position_influence"] is False
    assert status["readiness"]["metrics"]["dirty_sample_ratio"] == 0.0
    assert status["readiness"]["metrics"]["training_data_version"] == "2026-06-19.v1"
    assert status["readiness"]["metrics"]["required_training_data_version"] == DATA_QUALITY_VERSION
    assert "sample_count_below_threshold" in reason_codes
    assert "test_count_below_threshold" in reason_codes
    assert "long_pr_auc_missing" in reason_codes
    assert "short_pr_auc_missing" in reason_codes
    assert "training_data_version_stale" in reason_codes
    assert "model_stale" in reason_codes
    assert status["readiness"]["next_training_conditions"]["min_new_samples"] > 0


def test_ml_signal_status_marks_ready_only_when_all_readiness_metrics_pass() -> None:
    service = _service_with_metadata(
        {
            "version": datetime.now(UTC).isoformat(),
            "trained_at": datetime.now(UTC).isoformat(),
            "sample_count": 1200,
            "test_count": 240,
            "quality_report": {
                "data_quality_version": DATA_QUALITY_VERSION,
                "totals": {"total": 1200, "included": 1200, "downweighted": 0, "excluded": 0},
            },
            "metrics": {
                "long_auc": 0.61,
                "short_auc": 0.62,
                "long_pr_auc": 0.58,
                "short_pr_auc": 0.57,
                "long_accuracy": 0.58,
                "short_accuracy": 0.59,
                "top_long_avg_return_pct": 0.16,
                "bottom_long_avg_return_pct": -0.03,
                "top_short_avg_return_pct": 0.15,
                "bottom_short_avg_return_pct": -0.02,
                "top_long_win_rate": 0.72,
                "bottom_long_win_rate": 0.41,
                "top_short_win_rate": 0.70,
                "bottom_short_win_rate": 0.42,
            },
        }
    )

    status = service.status()

    assert status["status"] == "ready"
    assert status["readiness_state"] == "ready"
    assert status["allow_live_position_influence"] is True
    assert status["readiness"]["blocking_reasons"] == []
    assert status["readiness"]["metrics"]["long_pr_auc"] == 0.58


def test_ml_signal_predict_blocks_profit_signal_until_readiness_allows_live_influence() -> None:
    stale_quality_metadata = {
        "version": datetime.now(UTC).isoformat(),
        "trained_at": datetime.now(UTC).isoformat(),
        "sample_count": 1200,
        "test_count": 240,
        "quality_report": {
            "data_quality_version": "2026-06-19.v1",
            "totals": {"total": 1200, "included": 1200, "downweighted": 0, "excluded": 0},
        },
        "metrics": {
            "long_auc": 0.64,
            "short_auc": 0.62,
            "long_pr_auc": 0.61,
            "short_pr_auc": 0.59,
            "long_accuracy": 0.62,
            "short_accuracy": 0.60,
            "top_long_avg_return_pct": 0.22,
            "bottom_long_avg_return_pct": -0.03,
            "top_short_avg_return_pct": 0.18,
            "bottom_short_avg_return_pct": -0.02,
            "top_long_win_rate": 0.75,
            "bottom_long_win_rate": 0.38,
            "top_short_win_rate": 0.72,
            "bottom_short_win_rate": 0.40,
        },
    }
    service = _service_with_metadata(stale_quality_metadata)
    service._bundle.update(
        {
            "long_classifier": _Classifier(0.82),
            "short_classifier": _Classifier(0.24),
            "long_regressor": _Regressor(0.24),
            "short_regressor": _Regressor(0.02),
        }
    )

    prediction = service.predict({"current_price": 100.0, "atr_14": 1.0}, horizons=(10,))

    assert prediction["readiness_state"] == "degraded"
    assert prediction["allow_live_position_influence"] is False
    assert prediction["influence_policy"]["enabled"] is True
    assert prediction["influence_enabled"] is False
    assert prediction["profit_signal"] is False
    assert prediction["predictions"][0]["ml_influence_enabled"] is False
    assert prediction["predictions"][0]["profit_signal"] is False
