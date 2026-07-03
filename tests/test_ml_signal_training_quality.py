from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.learning import ShadowBacktest
from scripts import evaluate_ml_training_windows as ml_window_eval
from scripts import train_ml_signal_model as train_ml_signal_script
from services import ml_signal_service as ml_signal_module
from services.artifact_retirement_audit import (
    PHASE3_ARTIFACT_POLICY_ID,
    PHASE3_REQUIRED_PROMOTION_FLOW,
    PHASE3_REQUIRED_TRAINING_POLICY,
)
from services.ml_readiness import build_ml_readiness_report
from services.ml_signal_service import (
    FEATURE_KEYS,
    MLSignalService,
    build_training_frame,
    count_shadow_training_rows,
    load_shadow_training_rows,
    select_shadow_training_rows,
    shadow_training_quality_report,
    train_from_frame,
)
from services.phase3_boundary import PHASE3_CLEAN_START_UTC
from services.training_data_quality import DATA_QUALITY_VERSION, quality_report


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


@pytest.mark.asyncio
async def test_local_ml_training_counts_only_phase3_clean_shadow_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    await _use_temp_db(monkeypatch, tmp_path)
    async with get_session_ctx() as session:
        session.add_all(
            [
                _db_shadow_row(1, PHASE3_CLEAN_START_UTC - timedelta(minutes=5)),
                _db_shadow_row(2, PHASE3_CLEAN_START_UTC + timedelta(minutes=5)),
                _db_shadow_row(
                    3,
                    PHASE3_CLEAN_START_UTC + timedelta(minutes=10),
                    action="long",
                    best_action="long",
                ),
            ]
        )
        await session.flush()

    try:
        selected = await load_shadow_training_rows(limit=10)
        count = await count_shadow_training_rows()
    finally:
        await close_db()

    assert count == 2
    assert {row.id for row in selected}.issubset({2, 3})
    assert {row.id for row in selected}
    assert 1 not in {row.id for row in selected}


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
    confidence: float = 0.7,
    feature_snapshot: dict[str, object] | None = None,
    created_at: datetime | None = None,
) -> SimpleNamespace:
    row_created_at = created_at or datetime(2026, 6, 23, 3, row_id % 60, tzinfo=UTC)
    return SimpleNamespace(
        id=row_id,
        created_at=row_created_at,
        symbol=f"TEST{row_id}/USDT",
        analysis_type="market",
        decision_action=action,
        decision_confidence=confidence,
        horizon_minutes=30,
        feature_snapshot=feature_snapshot or {"current_price": 100.0, "spread_pct": 0.01},
        long_return_pct=0.16 if best_action == "long" else -0.06,
        short_return_pct=0.14 if best_action == "short" else -0.05,
        best_action=best_action,
        missed_opportunity=missed,
        due_at=row_created_at + timedelta(minutes=30),
    )


def _training_frame(row_count: int = 80) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx in range(row_count):
        row = {key: 0.0 for key in FEATURE_KEYS}
        row.update(
            {
                "id": idx + 1,
                "symbol": "BTC/USDT" if idx % 2 == 0 else "ETH/USDT",
                "long_return_pct": 0.2 if idx % 4 == 0 else -0.05,
                "short_return_pct": 0.18 if idx % 4 == 1 else -0.04,
                "long_win": int(idx % 4 == 0),
                "short_win": int(idx % 4 == 1),
                "sample_weight": 1.0,
                "data_quality_status": "included",
                "data_quality_score": 1.0,
                "quality_reasons": [],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _ml_training_metadata(
    *,
    artifact_persisted: bool,
    ready: bool,
    completed_sample_count: int = 1300,
) -> dict[str, object]:
    top_return = 0.16 if ready else -0.08
    bottom_return = -0.03 if ready else -0.12
    now = datetime.now(UTC).isoformat()
    return {
        "version": now,
        "trained_at": now,
        "sample_count": 1200,
        "test_count": 240,
        "last_trained_completed_shadow_sample_count": completed_sample_count,
        "training_run_mode": "persist" if artifact_persisted else "dry_run",
        "artifact_persisted": artifact_persisted,
        "quality_report": {
            "data_quality_version": DATA_QUALITY_VERSION,
            "totals": {"total": 1200, "included": 1200, "downweighted": 0, "excluded": 0},
        },
        "training_window_composition": {
            "sample_count": 1200,
            "decision_action_counts": {"long": 600, "short": 600},
            "best_action_counts": {"long": 600, "short": 600},
        },
        "metrics": {
            "long_auc": 0.64,
            "short_auc": 0.63,
            "long_pr_auc": 0.60,
            "short_pr_auc": 0.59,
            "long_accuracy": 0.61,
            "short_accuracy": 0.60,
            "top_long_avg_return_pct": top_return,
            "bottom_long_avg_return_pct": bottom_return,
            "top_short_avg_return_pct": top_return,
            "bottom_short_avg_return_pct": bottom_return,
            "top_long_win_rate": 0.72 if ready else 0.48,
            "bottom_long_win_rate": 0.41 if ready else 0.52,
            "top_short_win_rate": 0.71 if ready else 0.47,
            "bottom_short_win_rate": 0.40 if ready else 0.51,
        },
    }


def test_train_from_frame_can_evaluate_without_persisting_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "winrate_model.joblib"
    metadata_path = tmp_path / "winrate_model_metadata.json"
    monkeypatch.setattr(ml_signal_module, "MODEL_PATH", model_path)
    monkeypatch.setattr(ml_signal_module, "METADATA_PATH", metadata_path)

    metadata = train_from_frame(
        _training_frame(),
        min_samples=10,
        completed_sample_count=80,
        persist_artifact=False,
    )

    assert metadata["artifact_persisted"] is False
    assert metadata["training_run_mode"] == "dry_run"
    assert metadata["artifact_policy_id"] == PHASE3_ARTIFACT_POLICY_ID
    assert metadata["phase"] == "phase3_model_factory"
    assert metadata["training_policy"] == PHASE3_REQUIRED_TRAINING_POLICY
    assert metadata["trade_sample_cursor_policy"] == PHASE3_REQUIRED_TRAINING_POLICY
    assert metadata["training_mode"] == "walk_forward"
    assert metadata["model_stage"] == "shadow"
    assert metadata["evaluation_policy"]["promotion_flow"] == PHASE3_REQUIRED_PROMOTION_FLOW
    assert metadata["evaluation_policy"]["live_mutation"] is False
    assert not model_path.exists()
    assert not metadata_path.exists()


def test_train_from_frame_reports_score_bucket_diagnostic_segments() -> None:
    frame = _training_frame(120)
    frame["decision_action"] = ["hold", "long", "short"] * 40
    frame["best_action"] = ["short", "hold", "long"] * 40
    frame["horizon_minutes"] = [10, 30, 60] * 40
    frame["data_quality_status"] = ["included", "downweighted", "included"] * 40
    frame["sample_weight"] = [1.0, 0.35, 0.8] * 40
    frame["quality_reasons"] = [
        [],
        ["hold_observation_downweighted"],
        ["wide_spread_feature"],
    ] * 40

    metadata = train_from_frame(
        frame,
        min_samples=10,
        completed_sample_count=120,
        persist_artifact=False,
    )

    diagnostics = metadata["score_bucket_diagnostics"]
    for side in ("long", "short"):
        assert set(diagnostics[side]) == {"top", "bottom"}
        for bucket in ("top", "bottom"):
            summary = diagnostics[side][bucket]
            assert summary["count"] > 0
            assert "avg_return_pct" in summary
            assert "win_rate" in summary
            assert "avg_sample_weight" in summary
            assert summary["action_counts"]
            assert summary["best_action_counts"]
            assert summary["horizon_counts"]
            assert summary["data_quality_status_counts"]
            assert isinstance(summary["top_quality_reasons"], list)


def test_build_training_frame_preserves_diagnostic_sample_context() -> None:
    row = SimpleNamespace(
        id=7,
        symbol="BTC/USDT",
        analysis_type="market",
        decision_action="short",
        decision_confidence=0.72,
        horizon_minutes=30,
        feature_snapshot={
            "current_price": 100.0,
            "spread_pct": 0.01,
            "abnormal_wick_count_72h": 2,
            "entry_activity_volume_ratio": 1.8,
            "notional_24h_usdt": 9999.0,
            "liquidation_risk_score": 0.42,
            "direct_sentiment_data_available": True,
            "direct_news_item_count": 3,
        },
        long_return_pct=-0.12,
        short_return_pct=0.18,
        best_action="short",
        missed_opportunity=False,
        due_at=datetime(2026, 6, 23, 1, 0, tzinfo=UTC),
    )

    frame = build_training_frame([row])

    assert frame.loc[0, "decision_action"] == "short"
    assert frame.loc[0, "best_action"] == "short"
    assert bool(frame.loc[0, "missed_opportunity"]) is False
    assert frame.loc[0, "abnormal_wick_count_72h"] == 2.0
    assert frame.loc[0, "entry_activity_volume_ratio"] == 1.8
    assert frame.loc[0, "log_notional_24h_usdt"] > 3.0
    assert frame.loc[0, "liquidation_risk_score"] == 0.42
    assert frame.loc[0, "direct_sentiment_data_available"] == 1.0
    assert frame.loc[0, "direct_news_item_count"] == 3.0


@pytest.mark.asyncio
async def test_train_ml_signal_script_defaults_to_preflight_without_persist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    async def forbidden_quarantine(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("dry-run must not quarantine or mutate training rows")

    async def load_rows(*, limit: int) -> list[object]:
        assert limit == 20
        return [object()]

    def quality_report(_rows: list[object]) -> dict[str, object]:
        return {"quality_report": {"totals": {"total": 1}}}

    def build_frame(_rows: list[object]) -> pd.DataFrame:
        return _training_frame()

    async def count_rows() -> int:
        return 80

    def train_frame(_frame: pd.DataFrame, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"artifact_persisted": kwargs["persist_artifact"]}

    monkeypatch.setattr(
        train_ml_signal_script, "quarantine_dirty_shadow_samples", forbidden_quarantine
    )
    monkeypatch.setattr(train_ml_signal_script, "load_shadow_training_rows", load_rows)
    monkeypatch.setattr(train_ml_signal_script, "shadow_training_quality_report", quality_report)
    monkeypatch.setattr(train_ml_signal_script, "build_training_frame", build_frame)
    monkeypatch.setattr(train_ml_signal_script, "count_shadow_training_rows", count_rows)
    monkeypatch.setattr(train_ml_signal_script, "train_from_frame", train_frame)

    result = await train_ml_signal_script.run_training(
        limit=20,
        min_samples=10,
        skip_quarantine=False,
    )

    assert result["training_quarantine"] == {
        "skipped": True,
        "reason": "phase3_preflight_no_quarantine_writes",
    }
    assert result["dry_run"] is True
    assert result["preflight_only"] is True
    assert result["persist_artifact_requested"] is False
    assert calls[0]["persist_artifact"] is False
    assert result["metadata"] == {"artifact_persisted": False}


@pytest.mark.asyncio
async def test_train_ml_signal_script_requires_confirmation_to_persist() -> None:
    with pytest.raises(ValueError, match="confirm_phase3_rebuild"):
        await train_ml_signal_script.run_training(
            limit=20,
            min_samples=10,
            persist_artifact=True,
            confirm_phase3_rebuild=False,
        )


@pytest.mark.asyncio
async def test_train_ml_signal_script_blocks_persist_when_okx_gate_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        train_ml_signal_script,
        "okx_training_refresh_gate",
        lambda: {
            "allowed": False,
            "reason": "okx_daily_reconciliation_training_blocked",
            "can_refresh_training": False,
        },
    )

    with pytest.raises(ValueError, match="OKX daily reconciliation blocks"):
        await train_ml_signal_script.run_training(
            limit=100,
            min_samples=10,
            persist_artifact=True,
            confirm_phase3_rebuild=True,
        )


@pytest.mark.asyncio
async def test_train_ml_signal_script_confirmed_rebuild_can_persist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    quarantine_calls: list[dict[str, object]] = []

    async def quarantine(**kwargs: object) -> dict[str, object]:
        quarantine_calls.append(kwargs)
        return {"skipped": False, "quarantined": 0}

    async def load_rows(*, limit: int) -> list[object]:
        assert limit == 20
        return [object()]

    def quality_report(_rows: list[object]) -> dict[str, object]:
        return {"quality_report": {"totals": {"total": 1}}}

    def build_frame(_rows: list[object]) -> pd.DataFrame:
        return _training_frame()

    async def count_rows() -> int:
        return 80

    def train_frame(_frame: pd.DataFrame, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"artifact_persisted": kwargs["persist_artifact"]}

    monkeypatch.setattr(
        train_ml_signal_script,
        "okx_training_refresh_gate",
        lambda: {
            "allowed": True,
            "reason": "okx_daily_reconciliation_allows_training_refresh",
            "can_refresh_training": True,
        },
    )
    monkeypatch.setattr(train_ml_signal_script, "quarantine_dirty_shadow_samples", quarantine)
    monkeypatch.setattr(train_ml_signal_script, "load_shadow_training_rows", load_rows)
    monkeypatch.setattr(train_ml_signal_script, "shadow_training_quality_report", quality_report)
    monkeypatch.setattr(train_ml_signal_script, "build_training_frame", build_frame)
    monkeypatch.setattr(train_ml_signal_script, "count_shadow_training_rows", count_rows)
    monkeypatch.setattr(train_ml_signal_script, "train_from_frame", train_frame)

    result = await train_ml_signal_script.run_training(
        limit=20,
        min_samples=10,
        persist_artifact=True,
        confirm_phase3_rebuild=True,
    )

    assert quarantine_calls == [{"batch_size": 20, "max_batches": 1}]
    assert calls[0]["persist_artifact"] is True
    assert result["dry_run"] is False
    assert result["preflight_only"] is False
    assert result["persist_artifact_requested"] is True
    assert result["confirm_phase3_rebuild"] is True
    assert result["metadata"] == {"artifact_persisted": True}


@pytest.mark.asyncio
async def test_ml_signal_auto_train_persists_latest_artifact_even_when_candidate_is_degraded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = MLSignalService()
    calls: list[bool] = []
    ensure_load_calls: list[str] = []

    async def completed_shadow_sample_count() -> int:
        return 1300

    async def quarantine_dirty_training_samples(**_kwargs: object) -> dict[str, object]:
        return {"scanned": 1300, "quarantined": 0}

    async def load_rows(*, limit: int) -> list[object]:
        assert limit > 0
        return [object()]

    def quality_report(_rows: list[object]) -> dict[str, object]:
        return {"quality_report": {"totals": {"total": 1}}}

    def build_frame(_rows: list[object]) -> pd.DataFrame:
        return _training_frame()

    def train_frame(_frame: pd.DataFrame, **kwargs: object) -> dict[str, object]:
        calls.append(bool(kwargs.get("persist_artifact")))
        return _ml_training_metadata(
            artifact_persisted=bool(kwargs["persist_artifact"]),
            ready=False,
        )

    service._completed_shadow_sample_count = completed_shadow_sample_count  # type: ignore[method-assign]
    service._current_metadata = lambda: {  # type: ignore[method-assign]
        "sample_count": 1000,
        "last_trained_completed_shadow_sample_count": 1000,
        "trained_at": datetime.now(UTC).isoformat(),
    }
    service._quarantine_dirty_training_samples = quarantine_dirty_training_samples  # type: ignore[method-assign]
    service._ensure_loaded = lambda: ensure_load_calls.append("load")  # type: ignore[method-assign]
    monkeypatch.setattr("services.ml_signal_service.load_shadow_training_rows", load_rows)
    monkeypatch.setattr("services.ml_signal_service.shadow_training_quality_report", quality_report)
    monkeypatch.setattr("services.ml_signal_service.build_training_frame", build_frame)
    monkeypatch.setattr("services.ml_signal_service.train_from_frame", train_frame)

    result = await service.maybe_auto_train(force=True)

    assert calls == [False, True]
    assert ensure_load_calls == ["load"]
    assert result["trained"] is True
    assert result["reason"] == "trained_learning_only"
    assert result["artifact_persisted"] is True
    assert result["candidate"]["artifact_persisted"] is False
    assert result["candidate_readiness"]["allow_live_position_influence"] is False
    assert result["allow_live_position_influence"] is False
    assert result["readiness_state"] == "degraded"
    reason_codes = {item["code"] for item in result["candidate_readiness"]["blocking_reasons"]}
    assert "long_top_return_below_threshold" in reason_codes
    assert "short_top_return_below_threshold" in reason_codes


@pytest.mark.asyncio
async def test_ml_signal_auto_train_promotes_ready_candidate_only_after_dry_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = MLSignalService()
    calls: list[bool] = []
    ensure_load_calls: list[str] = []

    async def completed_shadow_sample_count() -> int:
        return 1300

    async def quarantine_dirty_training_samples(**_kwargs: object) -> dict[str, object]:
        return {"scanned": 1300, "quarantined": 0}

    async def load_rows(*, limit: int) -> list[object]:
        assert limit > 0
        return [object()]

    def quality_report(_rows: list[object]) -> dict[str, object]:
        return {"quality_report": {"totals": {"total": 1}}}

    def build_frame(_rows: list[object]) -> pd.DataFrame:
        return _training_frame()

    def train_frame(_frame: pd.DataFrame, **kwargs: object) -> dict[str, object]:
        persist_artifact = bool(kwargs["persist_artifact"])
        calls.append(persist_artifact)
        return _ml_training_metadata(
            artifact_persisted=persist_artifact,
            ready=True,
        )

    service._completed_shadow_sample_count = completed_shadow_sample_count  # type: ignore[method-assign]
    service._current_metadata = lambda: {  # type: ignore[method-assign]
        "sample_count": 1000,
        "last_trained_completed_shadow_sample_count": 1000,
        "trained_at": datetime.now(UTC).isoformat(),
    }
    service._quarantine_dirty_training_samples = quarantine_dirty_training_samples  # type: ignore[method-assign]
    service._ensure_loaded = lambda: ensure_load_calls.append("load")  # type: ignore[method-assign]
    monkeypatch.setattr("services.ml_signal_service.load_shadow_training_rows", load_rows)
    monkeypatch.setattr("services.ml_signal_service.shadow_training_quality_report", quality_report)
    monkeypatch.setattr("services.ml_signal_service.build_training_frame", build_frame)
    monkeypatch.setattr("services.ml_signal_service.train_from_frame", train_frame)

    result = await service.maybe_auto_train(force=True)

    assert calls == [False, True]
    assert ensure_load_calls == ["load"]
    assert result["trained"] is True
    assert result["reason"] == "trained"
    assert result["artifact_persisted"] is True
    assert result["candidate"]["artifact_persisted"] is False
    assert result["candidate_readiness"]["allow_live_position_influence"] is True


def test_shadow_training_selection_includes_clean_missed_trade_opportunities() -> None:
    recent_hold_rows = [_shadow_row(10_000 - idx) for idx in range(20)]
    trade_rows = [
        _shadow_row(
            1_000 - idx,
            action="long" if idx % 2 == 0 else "short",
            best_action="long" if idx % 2 == 0 else "short",
        )
        for idx in range(8)
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
    missed_count = sum(bool(row.missed_opportunity) for row in selected)
    best_trade_count = sum(row.best_action in {"long", "short"} for row in selected)
    assert len(selected) == 12
    assert len(set(selected_ids)) == len(selected_ids)
    assert non_hold_count == 8
    assert missed_count == 4
    assert best_trade_count == len(selected)
    assert not any(row.id in {item.id for item in recent_hold_rows} for row in selected)


def test_ml_training_window_evaluator_exposes_extended_diagnostic_variants() -> None:
    names = [variant.name for variant in ml_window_eval.extended_variants()]
    assert "diagnostic_decision_equals_best" in names
    assert "diagnostic_decision_not_equals_best" in names
    assert "diagnostic_horizon_60" in names
    assert "diagnostic_decision_equals_best_short" in names

    rows = [
        _shadow_row(1, action="long", best_action="long"),
        _shadow_row(2, action="short", best_action="long"),
        _shadow_row(3, action="short", best_action="short"),
    ]
    selectors = {variant.name: variant.selector for variant in ml_window_eval.extended_variants()}

    matched = selectors["diagnostic_decision_equals_best"](rows, 10)
    mismatched = selectors["diagnostic_decision_not_equals_best"](rows, 10)
    matched_short = selectors["diagnostic_decision_equals_best_short"](rows, 10)

    assert {row.id for row in matched} == {1, 3}
    assert {row.id for row in mismatched} == {2}
    assert [row.id for row in matched_short] == [3]


def test_shadow_training_selection_prioritizes_trainable_signal_over_low_quality_hold() -> None:
    noisy_holds = [
        _shadow_row(20_000 - idx, action="hold", best_action="hold", confidence=0.01)
        for idx in range(30)
    ]
    clean_trade_rows = [
        _shadow_row(
            1_000 - idx,
            action="long" if idx % 2 == 0 else "short",
            best_action="long" if idx % 2 == 0 else "short",
            confidence=0.78,
        )
        for idx in range(12)
    ]
    clean_missed_rows = [
        _shadow_row(
            500 - idx,
            action="hold",
            best_action="long" if idx % 2 == 0 else "short",
            missed=True,
            confidence=0.66,
        )
        for idx in range(12)
    ]

    selected = select_shadow_training_rows(
        [*noisy_holds, *clean_trade_rows, *clean_missed_rows],
        limit=20,
    )

    noisy_selected = [row for row in selected if row.decision_confidence < 0.05]
    non_hold_count = sum(row.decision_action in {"long", "short"} for row in selected)
    best_trade_count = sum(row.best_action in {"long", "short"} for row in selected)
    assert len(selected) == 18
    assert len(noisy_selected) == 0
    assert non_hold_count == 12
    assert best_trade_count == len(selected)


def test_shadow_training_selection_includes_low_confidence_missed_hold_opportunities() -> None:
    base_time = datetime(2026, 6, 23, 12, 0, tzinfo=UTC)
    noisy_missed_holds = [
        _shadow_row(
            30_000 - idx,
            action="hold",
            best_action="long" if idx % 2 == 0 else "short",
            missed=True,
            confidence=0.01,
            created_at=base_time - timedelta(hours=2, minutes=idx),
        )
        for idx in range(40)
    ]
    clean_trade_rows = [
        _shadow_row(
            2_000 - idx,
            action="long" if idx % 2 == 0 else "short",
            best_action="long" if idx % 2 == 0 else "short",
            confidence=0.82,
            created_at=base_time - timedelta(minutes=idx),
        )
        for idx in range(8)
    ]

    selected = select_shadow_training_rows(
        [*noisy_missed_holds, *clean_trade_rows],
        limit=20,
    )

    assert len(selected) == 12
    assert sum(row.decision_action in {"long", "short"} for row in selected) == 8
    assert sum(row.decision_action == "hold" and row.missed_opportunity for row in selected) == 4
    assert sum(row.best_action in {"long", "short"} for row in selected) == len(selected)


def test_shadow_training_selection_excludes_low_confidence_non_opportunity_holds() -> None:
    noisy_holds = [
        _shadow_row(40_000 - idx, action="hold", best_action="hold", confidence=0.01)
        for idx in range(40)
    ]
    clean_missed_rows = [
        _shadow_row(
            3_000 - idx,
            action="hold",
            best_action="long" if idx % 2 == 0 else "short",
            missed=True,
            confidence=0.72,
        )
        for idx in range(12)
    ]

    selected = select_shadow_training_rows([*noisy_holds, *clean_missed_rows], limit=20)

    assert selected == []
    assert not any(row.id in {item.id for item in noisy_holds} for row in selected)


def test_shadow_training_selection_caps_recent_missed_hold_bursts() -> None:
    base_time = datetime(2026, 6, 23, 12, 0, tzinfo=UTC)
    recent_missed_rows = [
        _shadow_row(
            30_000 - idx,
            action="hold",
            best_action="long" if idx % 2 == 0 else "short",
            missed=True,
            confidence=0.01,
            created_at=base_time - timedelta(minutes=idx),
        )
        for idx in range(40)
    ]
    directional_rows = [
        _shadow_row(
            20_000 - idx,
            action="long" if idx % 2 == 0 else "short",
            best_action="long" if idx % 2 == 0 else "short",
            confidence=0.82,
            created_at=base_time - timedelta(hours=1, minutes=idx),
        )
        for idx in range(20)
    ]
    older_missed_rows = [
        _shadow_row(
            10_000 - idx,
            action="hold",
            best_action="long" if idx % 2 == 0 else "short",
            missed=True,
            confidence=0.01,
            created_at=base_time - timedelta(hours=2, minutes=idx),
        )
        for idx in range(40)
    ]

    selected = select_shadow_training_rows(
        [*recent_missed_rows, *directional_rows, *older_missed_rows],
        limit=60,
    )

    newest_quartile = selected[: max(int(len(selected) * 0.25), 1)]
    assert len(selected) == 30
    assert sum(row.decision_action in {"long", "short"} for row in selected) == 20
    assert sum(row.decision_action == "hold" and row.missed_opportunity for row in selected) == 10
    assert all(row.decision_action in {"long", "short"} for row in newest_quartile)


def test_train_from_frame_reports_training_window_composition() -> None:
    frame = _training_frame(120)
    frame["decision_action"] = ["hold", "long", "short"] * 40
    frame["best_action"] = ["short", "long", "short"] * 40
    frame["data_quality_status"] = ["downweighted", "included", "included"] * 40
    frame["sample_weight"] = [0.25, 1.0, 1.0] * 40

    metadata = train_from_frame(
        frame,
        min_samples=10,
        completed_sample_count=120,
        persist_artifact=False,
    )

    composition = metadata["training_window_composition"]
    assert composition["sample_count"] == 120
    assert composition["decision_action_counts"] == {"hold": 40, "long": 40, "short": 40}
    assert composition["best_action_counts"] == {"short": 80, "long": 40}
    assert composition["data_quality_status_counts"] == {"downweighted": 40, "included": 80}
    assert composition["effective_weight_ratio"] == pytest.approx((40 * 0.25 + 80) / 120)
    assert "top_long_tail_loss_rate" in metadata["metrics"]
    assert "top_short_tail_loss_rate" in metadata["metrics"]
    assert (
        metadata["expected_return_calibration"]["long"]["policy"]
        == "classifier_probability_times_empirical_payoff_minus_excess_tail_loss"
    )
    assert "tail_loss_rate" in metadata["expected_return_calibration"]["short"]


def test_quality_report_separates_missed_opportunity_downweight_from_contamination() -> None:
    report = quality_report(
        {
            "shadow": [
                {
                    "data_quality_status": "downweighted",
                    "sample_weight": 0.25,
                    "quality_reasons": [
                        "hold_missed_opportunity_downweighted",
                        "very_low_decision_confidence",
                    ],
                },
                {
                    "data_quality_status": "downweighted",
                    "sample_weight": 0.25,
                    "quality_reasons": ["wide_spread_feature"],
                },
            ]
        }
    )

    totals = report["totals"]
    assert totals["downweighted"] == 2
    assert totals["benign_downweighted"] == 1
    assert totals["contamination_downweighted"] == 1


def test_ml_readiness_dirty_ratio_ignores_benign_missed_opportunity_downweights() -> None:
    metadata = {
        "version": "2026-07-03T00:00:00+00:00",
        "trained_at": "2026-07-03T00:00:00+00:00",
        "sample_count": 1000,
        "test_count": 250,
        "quality_report": {
            "data_quality_version": DATA_QUALITY_VERSION,
            "totals": {
                "total": 1000,
                "included": 600,
                "downweighted": 400,
                "benign_downweighted": 395,
                "contamination_downweighted": 5,
                "excluded": 0,
            },
        },
        "metrics": {
            "long_auc": 0.7,
            "short_auc": 0.7,
            "long_pr_auc": 0.7,
            "short_pr_auc": 0.7,
            "long_accuracy": 0.7,
            "short_accuracy": 0.7,
            "top_long_avg_return_pct": 0.2,
            "bottom_long_avg_return_pct": -0.1,
            "top_long_tail_loss_rate": 0.22,
            "bottom_long_tail_loss_rate": 0.31,
            "top_short_avg_return_pct": 0.2,
            "bottom_short_avg_return_pct": -0.1,
            "top_short_tail_loss_rate": 0.28,
            "bottom_short_tail_loss_rate": 0.35,
            "top_long_win_rate": 0.7,
            "bottom_long_win_rate": 0.3,
            "top_short_win_rate": 0.7,
            "bottom_short_win_rate": 0.3,
        },
    }

    readiness = build_ml_readiness_report(metadata, {"enabled": True})

    assert readiness["metrics"]["dirty_sample_ratio"] == 0.005
    assert readiness["metrics"]["benign_downweighted_sample_count"] == 395
    assert readiness["metrics"]["contamination_downweighted_sample_count"] == 5
    assert readiness["metrics"]["top_short_tail_loss_rate"] == 0.28
    assert "dirty_sample_ratio_high" not in {
        item["code"] for item in readiness["blocking_reasons"]
    }


def test_ml_signal_predict_uses_calibrated_expected_return_before_raw_regressor() -> None:
    metadata = {
        "version": datetime.now(UTC).isoformat(),
        "trained_at": datetime.now(UTC).isoformat(),
        "sample_count": 1200,
        "test_count": 240,
        "quality_report": {
            "data_quality_version": DATA_QUALITY_VERSION,
            "totals": {"total": 1200, "included": 1200, "downweighted": 0, "excluded": 0},
        },
        "metrics": {
            "long_auc": 0.7,
            "short_auc": 0.7,
            "long_pr_auc": 0.7,
            "short_pr_auc": 0.7,
            "long_accuracy": 0.7,
            "short_accuracy": 0.7,
            "top_long_avg_return_pct": 0.2,
            "bottom_long_avg_return_pct": -0.1,
            "top_short_avg_return_pct": 0.2,
            "bottom_short_avg_return_pct": -0.1,
            "top_long_win_rate": 0.7,
            "bottom_long_win_rate": 0.3,
            "top_short_win_rate": 0.7,
            "bottom_short_win_rate": 0.3,
        },
        "expected_return_calibration": {
            "long": {"win_avg_return_pct": 1.0, "non_win_avg_return_pct": -0.5},
            "short": {"win_avg_return_pct": 0.8, "non_win_avg_return_pct": -0.4},
        },
    }
    service = MLSignalService()
    service._bundle = {
        "metadata": metadata,
        "long_classifier": _Classifier(0.8),
        "short_classifier": _Classifier(0.2),
        "long_regressor": _Regressor(-9.0),
        "short_regressor": _Regressor(9.0),
        "feature_keys": FEATURE_KEYS,
    }
    service._ensure_loaded = lambda: None  # type: ignore[method-assign]

    prediction = service.predict({"current_price": 100.0, "spread_pct": 0.01}, horizons=(30,))
    primary = prediction["predictions"][0]

    assert primary["best_side"] == "long"
    assert primary["long_expected_return_pct"] == pytest.approx(0.7)
    assert primary["short_expected_return_pct"] == pytest.approx(-0.16)


def test_ml_signal_predict_penalizes_excess_tail_loss_probability() -> None:
    metadata = {
        "version": datetime.now(UTC).isoformat(),
        "trained_at": datetime.now(UTC).isoformat(),
        "sample_count": 1200,
        "test_count": 240,
        "quality_report": {
            "data_quality_version": DATA_QUALITY_VERSION,
            "totals": {"total": 1200, "included": 1200, "downweighted": 0, "excluded": 0},
        },
        "metrics": {
            "long_auc": 0.7,
            "short_auc": 0.7,
            "long_pr_auc": 0.7,
            "short_pr_auc": 0.7,
            "long_accuracy": 0.7,
            "short_accuracy": 0.7,
            "top_long_avg_return_pct": 0.2,
            "bottom_long_avg_return_pct": -0.1,
            "top_short_avg_return_pct": 0.2,
            "bottom_short_avg_return_pct": -0.1,
            "top_long_win_rate": 0.7,
            "bottom_long_win_rate": 0.3,
            "top_short_win_rate": 0.7,
            "bottom_short_win_rate": 0.3,
        },
        "expected_return_calibration": {
            "long": {
                "win_avg_return_pct": 0.7,
                "non_win_avg_return_pct": -0.3,
                "tail_loss_rate": 0.10,
                "tail_loss_avg_return_pct": -2.0,
            },
            "short": {
                "win_avg_return_pct": 0.7,
                "non_win_avg_return_pct": -0.3,
                "tail_loss_rate": 0.10,
                "tail_loss_avg_return_pct": -2.0,
            },
        },
    }
    service = MLSignalService()
    service._bundle = {
        "metadata": metadata,
        "long_classifier": _Classifier(0.6),
        "short_classifier": _Classifier(0.6),
        "long_tail_classifier": _Classifier(0.10),
        "short_tail_classifier": _Classifier(0.55),
        "long_regressor": _Regressor(0.3),
        "short_regressor": _Regressor(0.3),
        "feature_keys": FEATURE_KEYS,
    }
    service._ensure_loaded = lambda: None  # type: ignore[method-assign]

    prediction = service.predict({"current_price": 100.0, "spread_pct": 0.01}, horizons=(30,))
    primary = prediction["predictions"][0]

    assert primary["best_side"] == "long"
    assert primary["long_expected_return_pct"] == pytest.approx(0.3)
    assert primary["short_expected_return_pct"] == pytest.approx(-0.915)
    assert primary["short_tail_loss_probability"] == pytest.approx(0.55)
    assert primary["best_tail_loss_probability"] == pytest.approx(0.10)


@pytest.mark.asyncio
async def test_load_shadow_training_rows_combines_recent_trade_and_best_action_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    await _use_temp_db(monkeypatch, tmp_path)
    base_time = datetime(2026, 6, 28, 3, 0, tzinfo=UTC)
    recent_holds = [
        _db_shadow_row(10_000 + idx, base_time - timedelta(minutes=idx)) for idx in range(40)
    ]
    decision_trade_rows = [
        _db_shadow_row(
            1_000 + idx,
            base_time - timedelta(hours=2, minutes=idx),
            action="long" if idx % 2 == 0 else "short",
            best_action="long" if idx % 2 == 0 else "short",
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
    assert len(selected) == 12
    assert all(not isinstance(row, ShadowBacktest) for row in selected)
    assert 90 not in selected_ids
    assert 91 not in selected_ids
    assert sum(row.decision_action in {"long", "short"} for row in selected) == 8
    assert sum(row.decision_action == "hold" and row.missed_opportunity for row in selected) == 4
    assert sum(row.best_action in {"long", "short"} for row in selected) == len(selected)
    assert not any(row.id >= 10_000 for row in selected)


@pytest.mark.asyncio
async def test_load_shadow_training_rows_pulls_deeper_best_trade_pool(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    await _use_temp_db(monkeypatch, tmp_path)
    base_time = datetime(2026, 6, 28, 3, 0, tzinfo=UTC)
    recent_holds = [
        _db_shadow_row(20_000 + idx, base_time - timedelta(minutes=idx)) for idx in range(80)
    ]
    deeper_best_trade_rows = [
        _db_shadow_row(
            2_000 + idx,
            base_time - timedelta(hours=2, minutes=idx),
            action="long" if idx % 2 == 0 else "short",
            best_action="long" if idx % 2 == 0 else "short",
        )
        for idx in range(25)
    ]
    async with get_session_ctx() as session:
        session.add_all([*recent_holds, *deeper_best_trade_rows])

    try:
        selected = await load_shadow_training_rows(limit=20)
    finally:
        await close_db()

    assert len(selected) == 20
    assert {row.decision_action for row in selected} <= {"long", "short"}
    assert {row.best_action for row in selected} <= {"long", "short"}
    assert not any(row.id >= 20_000 for row in selected)


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
