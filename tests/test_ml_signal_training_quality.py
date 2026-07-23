from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from core.training_contracts import (
    SHADOW_LABEL_VERSION,
    build_shadow_label_contract,
    compact_shadow_label_contract,
)
from db.repositories.memory_repo import MemoryRepository
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
    _configure_single_row_inference,
    _leave_one_symbol_out_stability,
    _training_data_sha256,
    build_training_frame,
    count_shadow_training_rows,
    load_shadow_training_rows,
    select_shadow_training_rows,
    shadow_training_quality_report,
    train_from_frame,
)
from services.model_artifact_registry import ARTIFACT_REGISTRY_VERSION, ModelArtifactRegistry
from services.phase3_boundary import PHASE3_CLEAN_START_UTC
from services.profit_supervision import (
    AUTHORITATIVE_REALIZED_RETURN_TASK,
    COUNTERFACTUAL_EXECUTION_COST_TASK,
    PROFIT_SUPERVISION_VERSION,
)
from services.return_objective import (
    RETURN_LABEL_NAME,
    RETURN_LABEL_VERSION,
    RETURN_OBJECTIVE_NAME,
    RETURN_OBJECTIVE_VERSION,
)
from services.training_data_quality import (
    DATA_QUALITY_VERSION,
    MARKET_FACT_CONTRACT_VERSION,
    quality_report,
)


def test_loaded_local_ml_estimators_disable_parallel_single_row_inference() -> None:
    estimator = SimpleNamespace(n_jobs=-1)
    bundle = {"long_regressor": SimpleNamespace(named_steps={"model": estimator})}

    _configure_single_row_inference(bundle)

    assert estimator.n_jobs == 1


def _with_return_objective(metadata: dict) -> dict:
    metadata = dict(metadata)
    metadata.setdefault("objective_name", RETURN_OBJECTIVE_NAME)
    metadata.setdefault("objective_version", RETURN_OBJECTIVE_VERSION)
    metadata.setdefault("label_name", RETURN_LABEL_NAME)
    metadata.setdefault("label_version", RETURN_LABEL_VERSION)
    metadata.setdefault(
        "training_cost_policy",
        "separated_market_opportunity_and_execution_cost_tasks",
    )
    metadata.setdefault("profit_supervision_version", PROFIT_SUPERVISION_VERSION)
    metadata.setdefault(
        "profit_supervision_report",
        {
            "version": PROFIT_SUPERVISION_VERSION,
            "shadow_market_sample_count": int(metadata.get("sample_count") or 1),
            "shadow_counterfactual_cost_sample_count": int(
                metadata.get("sample_count") or 1
            ),
            "actual_realized_return_sample_count": 2,
        },
    )
    metadata.setdefault(
        "actual_trade_calibration",
        {
            "version": PROFIT_SUPERVISION_VERSION,
            "profiles": {
                f"*|{side}": {
                    "source_authority": "okx_position_history",
                    "symbol": "*",
                    "side": side,
                    "net_return_after_cost_pct": {
                        "count": 2,
                        "expected": 0.4,
                        "lower_hinge": 0.2,
                    },
                    "slippage_pct": {
                        "count": 2,
                        "expected": 0.02,
                        "upper_hinge": 0.04,
                    },
                }
                for side in ("long", "short")
            },
        },
    )
    metadata.setdefault("legacy_fixed_training_thresholds_enabled", False)
    metadata.setdefault(
        "market_fact_contract",
        {
            "version": MARKET_FACT_CONTRACT_VERSION,
            "status": "clean",
            "violation_count": 0,
            "assertions": {
                "native_instrument_identity_verified": True,
                "same_contract_price_path_verified": True,
                "executable_market_fact_verified": True,
            },
            "provenance": {
                "source": "test_native_market_facts",
                "observation_window": "test_fixture_window",
                "sample_count": int(metadata.get("sample_count") or 1),
                "generated_at": "2026-07-14T00:00:00+00:00",
                "strategy_version": "test.native-market-fact.v1",
                "fallback_reason": "",
                "data_fingerprint": "test-market-fact-fingerprint",
            },
        },
    )
    metadata.setdefault(
        "tail_loss_policy",
        {
            side: {
                "source": "artifact_holdout_fee_after_return_distribution",
                "observation_window": "test_fixture_window",
                "sample_count": int(metadata.get("sample_count") or 1),
                "generated_at": "2026-07-12T00:00:00+00:00",
                "strategy_version": "test.dynamic-tail.v1",
                "fallback_reason": "",
            }
            for side in ("long", "short")
        },
    )
    metadata.setdefault("tail_loss_scale_pct", {"long": 0.18, "short": 0.18})
    metadata.setdefault("training_data_sha256", "a" * 64)
    metadata.setdefault("source_code_sha256", "b" * 64)
    metadata.setdefault(
        "evaluation_group_policy",
        "chronological_disjoint_decision_groups",
    )
    metadata.setdefault("train_decision_group_count", 2)
    metadata.setdefault("test_decision_group_count", 2)
    metadata.setdefault(
        "governance_report",
        {
            "quality_fingerprint": "test-quality-fingerprint",
            "artifact_quality_fingerprint": "test-quality-fingerprint",
            "artifact_matches_quality": True,
        },
    )
    ready_return_evidence = {
        "count": 4,
        "avg_return_pct": 0.4,
        "return_lcb_pct": 0.2,
        "profit_factor": 4.0,
        "cvar_10_pct": -0.05,
        "max_drawdown_pct": 0.05,
        "promotion_math_ready": True,
    }
    metadata.setdefault(
        "walk_forward_report",
        {
            "status": "complete",
            "decision_group_disjoint": True,
            "chronological_label_disjoint": True,
            "model_refit_per_fold": True,
            "folds": [
                {
                    "fold": 1,
                    "decision_group_overlap_count": 0,
                    "sides": {
                        side: dict(ready_return_evidence)
                        for side in ("long", "short")
                    },
                },
                {
                    "fold": 2,
                    "decision_group_overlap_count": 0,
                    "sides": {
                        side: dict(ready_return_evidence)
                        for side in ("long", "short")
                    },
                },
            ],
            "sides": {
                side: {
                    **dict(ready_return_evidence),
                    "market_regime_stability": {"stable": True},
                }
                for side in ("long", "short")
            },
        },
    )
    metadata.setdefault(
        "leave_one_symbol_out_report",
        {
            side: {"stable": True, "rows": []}
            for side in ("long", "short")
        },
    )
    metadata.setdefault(
        "oos_return_evaluation",
        {
            side: dict(ready_return_evidence) for side in ("long", "short")
        },
    )
    metadata.setdefault(
        "authoritative_trade_return_evidence",
        {
            "version": "2026-07-15.authoritative-trade-return-evidence.v1",
            "data_fingerprint": "c" * 64,
            "sides": {
                side: dict(ready_return_evidence) for side in ("long", "short")
            },
        },
    )
    metrics = dict(metadata.get("metrics") or {})
    for side in ("long", "short"):
        top_return = float(metrics.get(f"top_{side}_avg_return_pct") or 0.0)
        metrics.setdefault(f"top_{side}_return_lcb_pct", top_return - 0.01)
        metrics.setdefault(f"top_{side}_profit_factor", 1.8 if top_return > 0 else 0.8)
        metrics.setdefault(f"top_{side}_tail_loss_rate", 0.05)
        metrics.setdefault(f"bottom_{side}_tail_loss_rate", 0.10)
    metadata["metrics"] = metrics
    return metadata


def _clean_training_market_fact_contract() -> dict[str, object]:
    return {
        "version": MARKET_FACT_CONTRACT_VERSION,
        "status": "clean",
        "violation_count": 0,
        "violation_reason_codes": "",
        "native_instrument_identity_verified": True,
        "same_contract_price_path_verified": True,
        "executable_market_fact_verified": True,
        "data_fingerprint": "test-shadow-market-facts",
    }


def _clean_training_label_contract(
    row_id: int,
    due_at: datetime,
    *,
    decision_id: int | None = None,
    horizon_minutes: int = 30,
    long_return_pct: float = 0.1,
    short_return_pct: float = -0.1,
    best_action: str = "hold",
) -> dict[str, object]:
    return compact_shadow_label_contract(
        build_shadow_label_contract(
            shadow_backtest_id=row_id,
            decision_id=decision_id or row_id + 10_000,
            horizon_minutes=horizon_minutes,
            long_return_pct=long_return_pct,
            short_return_pct=short_return_pct,
            best_action=best_action,
            market_fact_contract=_clean_training_market_fact_contract(),
            cost_facts={"round_trip_fee_pct": 0.08},
            label_timestamp=due_at,
        )
    )


def _service_with_metadata(metadata: dict) -> MLSignalService:
    service = MLSignalService()
    service._bundle = {"metadata": _with_return_objective(metadata)}
    service._resolved_artifact = SimpleNamespace(
        activation_manifest={
            "activation_stage": "live",
            "readiness_state": "ready",
            "production_influence_authorized": True,
            "blocking_reasons": [],
        }
    )
    service._artifact_registry_status = lambda: {  # type: ignore[method-assign]
        "available": True,
        "activation_manifest": service._resolved_artifact.activation_manifest,
    }
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
        selected = await load_shadow_training_rows()
        count = await count_shadow_training_rows()
    finally:
        await close_db()

    assert count == 2
    assert {row.id for row in selected}.issubset({2, 3})
    assert {row.id for row in selected}
    assert 1 not in {row.id for row in selected}


@pytest.mark.asyncio
async def test_shadow_label_identity_is_idempotent_and_new_version_is_append_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    await _use_temp_db(monkeypatch, tmp_path)
    due_at = datetime(2026, 7, 14, 1, 0, tzinfo=UTC)
    payload = {
        "decision_id": 7001,
        "model_name": "ensemble_trader",
        "execution_mode": "paper",
        "symbol": "BTC/USDT",
        "analysis_type": "market",
        "decision_action": "hold",
        "decision_confidence": 0.7,
        "entry_price": 100.0,
        "feature_snapshot": {},
        "raw_llm_response": {},
        "status": "pending",
        "due_at": due_at,
        "horizon_minutes": 30,
    }
    try:
        async with get_session_ctx() as session:
            repo = MemoryRepository(session)
            first = await repo.create_shadow_backtest(dict(payload))
            duplicate = await repo.create_shadow_backtest(dict(payload))
            next_version = await repo.create_shadow_backtest(
                {**payload, "label_version": "future-shadow-label.v2"}
            )

            assert duplicate.id == first.id
            assert duplicate.label_version == SHADOW_LABEL_VERSION
            assert next_version.id != first.id
            assert next_version.label_version == "future-shadow-label.v2"
    finally:
        await close_db()


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
    due_at = created_at + timedelta(minutes=30)
    decision_id = row_id + 10_000
    return ShadowBacktest(
        id=row_id,
        decision_id=decision_id,
        label_version=SHADOW_LABEL_VERSION,
        model_name="ensemble",
        execution_mode="paper",
        symbol=f"TEST{row_id}/USDT",
        analysis_type="market",
        decision_action=action,
        decision_confidence=0.7,
        entry_price=100.0,
        feature_snapshot={
            "current_price": 100.0,
            "spread_pct": 0.01,
            "round_trip_fee_pct": 0.08,
            "funding_rate": 0.0,
            "funding_interval_minutes": 480.0,
            "training_market_fact_contract": _clean_training_market_fact_contract(),
            "training_label_contract": _clean_training_label_contract(
                row_id,
                due_at,
                decision_id=decision_id,
                long_return_pct=float(long_return_pct or 0.0),
                short_return_pct=float(short_return_pct or 0.0),
                best_action=best_action,
            ),
        },
        status=status,
        due_at=due_at,
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
    def __init__(
        self,
        prediction: float,
        *,
        tree_predictions: tuple[float, ...] | None = None,
    ) -> None:
        self.prediction = prediction
        spread = max(abs(prediction) * 0.01, 0.001)
        member_predictions = tree_predictions or (
            prediction - spread,
            prediction + spread,
        )
        self.named_steps = {
            "imputer": SimpleNamespace(transform=lambda values: values),
            "model": SimpleNamespace(
                estimators_=[
                    SimpleNamespace(
                        predict=lambda _values, value=value: np.array([value])
                    )
                    for value in member_predictions
                ]
            ),
        }

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
    cost_complete_features: dict[str, object] = {
        "current_price": 100.0,
        "spread_pct": 0.01,
        "round_trip_fee_pct": 0.08,
        "funding_rate": 0.0,
        "funding_interval_minutes": 480.0,
        "training_market_fact_contract": _clean_training_market_fact_contract(),
    }
    cost_complete_features.update(feature_snapshot or {})
    due_at = row_created_at + timedelta(minutes=30)
    decision_id = row_id + 10_000
    cost_complete_features.setdefault(
        "training_label_contract",
        _clean_training_label_contract(
            row_id,
            due_at,
            decision_id=decision_id,
            long_return_pct=0.16 if best_action == "long" else -0.06,
            short_return_pct=0.14 if best_action == "short" else -0.05,
            best_action=best_action,
        ),
    )
    return SimpleNamespace(
        id=row_id,
        decision_id=decision_id,
        label_version=SHADOW_LABEL_VERSION,
        created_at=row_created_at,
        symbol=f"TEST{row_id}/USDT",
        analysis_type="market",
        decision_action=action,
        decision_confidence=confidence,
        horizon_minutes=30,
        feature_snapshot=cost_complete_features,
        long_return_pct=0.16 if best_action == "long" else -0.06,
        short_return_pct=0.14 if best_action == "short" else -0.05,
        best_action=best_action,
        missed_opportunity=missed,
        due_at=due_at,
    )


def _training_frame(row_count: int = 80) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx in range(row_count):
        row = {key: 0.0 for key in FEATURE_KEYS}
        row.update(
            {
                "id": idx + 1,
                "decision_group": f"shadow_decision:{idx + 1}",
                "label_timestamp": (
                    datetime(2026, 7, 14, tzinfo=UTC)
                    + timedelta(minutes=idx * 61)
                ).isoformat(),
                "horizon_minutes": 30,
                "symbol": "BTC/USDT" if idx % 2 == 0 else "ETH/USDT",
                "long_return_pct": 0.2 if idx % 4 == 0 else -0.05,
                "short_return_pct": 0.18 if idx % 4 == 1 else -0.04,
                "long_win": int(idx % 4 == 0),
                "short_win": int(idx % 4 == 1),
                "long_execution_cost_pct": 0.08,
                "short_execution_cost_pct": 0.08,
                "sample_weight": 1.0,
                "data_quality_status": "included",
                "data_quality_score": 1.0,
                "quality_reasons": [],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _authoritative_trade_sample() -> dict[str, object]:
    return {
        "symbol": "BTC/USDT",
        "side": "long",
        "sample_weight": 1.0,
        "profit_supervision": {
            "version": PROFIT_SUPERVISION_VERSION,
            "tasks": {
                COUNTERFACTUAL_EXECUTION_COST_TASK: {
                    "eligible": True,
                    "total_cost_pct": 0.08,
                    "slippage_pct": 0.03,
                },
                AUTHORITATIVE_REALIZED_RETURN_TASK: {
                    "eligible": True,
                    "side": "long",
                    "realized_net_return_pct": 0.4,
                    "hold_minutes": 30.0,
                },
            },
        },
    }


def _ml_training_metadata(
    *,
    artifact_persisted: bool,
    ready: bool,
    completed_sample_count: int = 1300,
) -> dict[str, object]:
    top_return = 0.16 if ready else -0.08
    bottom_return = -0.03 if ready else -0.12
    now = datetime.now(UTC).isoformat()
    return _with_return_objective({
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
    })


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
        completed_sample_count=80,
        trade_samples=[_authoritative_trade_sample()],
        persist_artifact=False,
    )

    assert metadata["artifact_persisted"] is False
    assert metadata["training_run_mode"] == "dry_run"
    assert metadata["artifact_policy_id"] == PHASE3_ARTIFACT_POLICY_ID
    assert metadata["phase"] == "phase3_model_factory"
    assert metadata["training_policy"] == PHASE3_REQUIRED_TRAINING_POLICY
    assert metadata["trade_sample_cursor_policy"] == PHASE3_REQUIRED_TRAINING_POLICY
    assert metadata["training_mode"] == "walk_forward"
    assert metadata["model_stage"] == "candidate"
    assert len(metadata["training_data_sha256"]) == 64
    assert len(metadata["source_code_sha256"]) == 64
    assert metadata["walk_forward_report"]["status"] == "complete"
    assert metadata["walk_forward_report"]["model_refit_per_fold"] is True
    assert metadata["walk_forward_report"]["decision_group_disjoint"] is True
    assert metadata["walk_forward_report"]["chronological_label_disjoint"] is True
    assert all(
        fold["decision_group_overlap_count"] == 0
        for fold in metadata["walk_forward_report"]["folds"]
    )
    assert all(
        fold["training_label_end"] < fold["validation_decision_start"]
        for fold in metadata["walk_forward_report"]["folds"]
    )
    assert metadata["artifact_activation_manifest"][
        "production_influence_authorized"
    ] is False
    assert metadata["live_promotion_manifest"]["status"] == "not_issued"
    assert metadata["profit_supervision_report"][
        "actual_execution_cost_sample_count"
    ] == 1
    assert metadata["profit_supervision_report"][
        "actual_realized_return_sample_count"
    ] == 1
    assert metadata["profit_supervision_report"]["authoritative_realized_trade"][
        "net_return_after_cost_pct"
    ]["count"] == 1
    assert metadata["quality_report"]["profit_supervision"][
        "actual_realized_return_sample_count"
    ] == 1
    assert metadata["evaluation_policy"]["promotion_flow"] == PHASE3_REQUIRED_PROMOTION_FLOW
    assert metadata["evaluation_policy"]["live_mutation"] is False
    assert not model_path.exists()
    assert not metadata_path.exists()


def test_train_from_frame_persists_and_loads_registry_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry = ModelArtifactRegistry(
        root=tmp_path / "model_artifacts",
        model_id="local_ml_profit_quality",
    )
    monkeypatch.setattr(ml_signal_module, "ML_SIGNAL_ARTIFACT_REGISTRY", registry)

    metadata = train_from_frame(
        _training_frame(),
        completed_sample_count=80,
        persist_artifact=True,
    )
    assert metadata["artifact_registry_version"] == ARTIFACT_REGISTRY_VERSION
    assert metadata["artifact_sha256"]
    assert metadata["metrics"]["long_accuracy"] is not None
    assert metadata["metrics"]["short_accuracy"] is not None
    assert registry.candidate_path.exists()
    assert not registry.current_path.exists()

    registry.promote_candidate(
        {
            "activation_stage": "shadow",
            "readiness_state": "degraded",
            "production_influence_authorized": False,
            "blocking_reasons": ["test_shadow_activation"],
        }
    )
    service = MLSignalService(artifact_registry=registry)
    status = service.status()

    assert registry.current_path.exists()
    assert status["available"] is True
    assert status["allow_live_position_influence"] is False
    assert status["artifact_registry"]["available"] is True
    assert status["artifact_registry"]["sha256"] == metadata["artifact_sha256"]
    governance = status["governance_report"]
    assert governance["artifact_quality_fingerprint"] == governance["quality_fingerprint"]
    assert governance["artifact_matches_quality"] is True
    assert governance["requires_artifact_refresh"] is False


def test_train_from_frame_binds_exact_market_fact_training_view_contract() -> None:
    report = quality_report(
        {
            "shadow": [
                {
                    "data_quality_status": "included",
                    "sample_weight": 1.0,
                    "quality_reasons": [],
                    "features": {
                        "training_market_fact_contract": (
                            _clean_training_market_fact_contract()
                        )
                    },
                }
            ]
        }
    )

    metadata = train_from_frame(
        _training_frame(),
        completed_sample_count=80,
        training_quality_report=report,
        persist_artifact=False,
    )

    assert metadata["market_fact_contract"] == report["market_fact_contract"]
    assert metadata["market_fact_contract"]["status"] == "clean"
    assert metadata["market_fact_contract"]["provenance"]["data_fingerprint"]


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


def test_training_data_fingerprint_is_order_independent_and_content_bound() -> None:
    frame = _training_frame(12)
    shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)

    original = _training_data_sha256(frame)
    assert _training_data_sha256(shuffled) == original

    changed = frame.copy()
    changed.loc[0, "long_return_pct"] = 9.0
    assert _training_data_sha256(changed) != original


def test_tail_policy_is_derived_only_from_chronological_training_partition() -> None:
    baseline = _training_frame(20)
    changed_holdout = baseline.copy()
    changed_holdout.loc[10:, "long_return_pct"] = -999.0
    changed_holdout.loc[10:, "short_return_pct"] = -777.0

    baseline_metadata = train_from_frame(baseline, persist_artifact=False)
    changed_metadata = train_from_frame(changed_holdout, persist_artifact=False)

    for side in ("long", "short"):
        assert {
            key: value
            for key, value in changed_metadata["tail_loss_policy"][side].items()
            if key != "generated_at"
        } == {
            key: value
            for key, value in baseline_metadata["tail_loss_policy"][side].items()
            if key != "generated_at"
        }
    assert all(
        side_report["training_tail_loss_policy"]["observation_window"]
        == "walk_forward_training_groups_only"
        for fold in changed_metadata["walk_forward_report"]["folds"]
        for side_report in fold["sides"].values()
    )


def test_walk_forward_purges_unavailable_multi_horizon_decision_groups() -> None:
    rows: list[dict[str, object]] = []
    decision_start = datetime(2026, 7, 14, tzinfo=UTC)
    row_id = 0
    for decision_index in range(30):
        decision_at = decision_start + timedelta(minutes=decision_index * 5)
        for horizon in (10, 60):
            row_id += 1
            row: dict[str, object] = {key: 0.0 for key in FEATURE_KEYS}
            row.update(
                {
                    "id": row_id,
                    "decision_group": f"shadow_decision:{decision_index + 1}",
                    "decision_timestamp": decision_at.isoformat(),
                    "label_timestamp": (
                        decision_at + timedelta(minutes=horizon)
                    ).isoformat(),
                    "horizon_minutes": horizon,
                    "symbol": "BTC/USDT" if decision_index % 2 == 0 else "ETH/USDT",
                    "long_return_pct": 0.3 if decision_index % 3 == 0 else -0.1,
                    "short_return_pct": 0.25 if decision_index % 3 == 1 else -0.08,
                    "long_execution_cost_pct": 0.05,
                    "short_execution_cost_pct": 0.05,
                    "sample_weight": 1.0,
                }
            )
            rows.append(row)

    report = ml_signal_module._walk_forward_return_report(pd.DataFrame(rows))

    assert report["status"] == "complete"
    assert report["chronological_label_disjoint"] is True
    assert any(
        fold["purged_training_decision_group_count"] > 0
        for fold in report["folds"]
    )
    assert all(
        fold["training_label_end"] < fold["validation_decision_start"]
        and fold["decision_group_overlap_count"] == 0
        for fold in report["folds"]
    )


def test_leave_one_symbol_out_detects_single_symbol_profit_support() -> None:
    rows = [
        {
            "symbol": "ROBO/USDT",
            "decision_group": f"robo:{index}",
            "label_timestamp": f"2026-07-14T00:{index:02d}:00+00:00",
            "return_pct": -0.1 if index == 0 else 2.0,
            "score": 100.0 + index,
        }
        for index in range(10)
    ] + [
        {
            "symbol": "BTC/USDT",
            "decision_group": f"btc:{index}",
            "label_timestamp": f"2026-07-14T01:{index % 60:02d}:00+00:00",
            "return_pct": -1.0,
            "score": float(index),
        }
        for index in range(90)
    ]

    report = _leave_one_symbol_out_stability(rows)
    robo_removed = next(
        row for row in report["rows"] if row["excluded_symbol"] == "ROBO/USDT"
    )

    assert report["stable"] is False
    assert robo_removed["evidence"]["promotion_math_ready"] is False
    assert robo_removed["evidence"]["return_lcb_pct"] < 0.0


def test_build_training_frame_preserves_diagnostic_sample_context() -> None:
    due_at = datetime(2026, 6, 23, 1, 0, tzinfo=UTC)
    row = SimpleNamespace(
        id=7,
        decision_id=10007,
        label_version=SHADOW_LABEL_VERSION,
        symbol="BTC/USDT",
        analysis_type="market",
        decision_action="short",
        decision_confidence=0.72,
        horizon_minutes=30,
        feature_snapshot={
            "current_price": 100.0,
                "spread_pct": 0.01,
                "taker_fee_rate": 0.0004,
            "funding_rate": 0.0001,
            "funding_interval_hours": 8,
            "abnormal_wick_count_72h": 2,
            "entry_activity_volume_ratio": 1.8,
            "notional_24h_usdt": 9999.0,
            "liquidation_risk_score": 0.42,
            "direct_sentiment_data_available": True,
            "direct_news_item_count": 3,
            "training_market_fact_contract": _clean_training_market_fact_contract(),
            "training_label_contract": _clean_training_label_contract(
                7,
                due_at,
                decision_id=10007,
                long_return_pct=-0.12,
                short_return_pct=0.18,
                best_action="short",
            ),
        },
        long_return_pct=-0.12,
        short_return_pct=0.18,
        best_action="short",
        missed_opportunity=False,
        due_at=due_at,
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

    async def load_rows() -> list[object]:
        return [object()]

    def quality_report(_rows: list[object]) -> dict[str, object]:
        return {"quality_report": {"totals": {"total": 1}}}

    def build_frame(_rows: list[object]) -> pd.DataFrame:
        return _training_frame()

    async def count_rows() -> int:
        return 80

    async def load_trade_samples() -> list[dict[str, object]]:
        return []

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
    monkeypatch.setattr(
        train_ml_signal_script,
        "load_authoritative_trade_training_samples",
        load_trade_samples,
    )
    monkeypatch.setattr(train_ml_signal_script, "train_from_frame", train_frame)

    result = await train_ml_signal_script.run_training(skip_quarantine=False)

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

    async def load_rows() -> list[object]:
        return [object()]

    def quality_report(_rows: list[object]) -> dict[str, object]:
        return {"quality_report": {"totals": {"total": 1}}}

    def build_frame(_rows: list[object]) -> pd.DataFrame:
        return _training_frame()

    async def count_rows() -> int:
        return 80

    async def load_trade_samples() -> list[dict[str, object]]:
        return []

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
    monkeypatch.setattr(
        train_ml_signal_script,
        "load_authoritative_trade_training_samples",
        load_trade_samples,
    )
    monkeypatch.setattr(train_ml_signal_script, "train_from_frame", train_frame)

    result = await train_ml_signal_script.run_training(
        persist_artifact=True,
        confirm_phase3_rebuild=True,
    )

    assert quarantine_calls == [{}]
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

    async def load_rows() -> list[object]:
        return [object()]

    async def load_trade_samples() -> list[dict[str, object]]:
        return []

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
    promotion_evidence: list[dict[str, object]] = []

    def promote_candidate(evidence: dict[str, object]) -> SimpleNamespace:
        promotion_evidence.append(evidence)
        return SimpleNamespace(version="candidate-v1")

    service.artifact_registry = SimpleNamespace(
        promote_candidate=promote_candidate,
        transition_current=promote_candidate,
    )
    monkeypatch.setattr("services.ml_signal_service.load_shadow_training_rows", load_rows)
    monkeypatch.setattr("services.ml_signal_service.shadow_training_quality_report", quality_report)
    monkeypatch.setattr("services.ml_signal_service.build_training_frame", build_frame)
    monkeypatch.setattr("services.ml_signal_service.train_from_frame", train_frame)
    monkeypatch.setattr(
        "services.ml_signal_service.load_authoritative_trade_training_samples",
        load_trade_samples,
    )

    result = await service.maybe_auto_train(force=True)

    assert calls == [False, True]
    assert ensure_load_calls == ["load"]
    assert result["trained"] is True
    assert result["reason"] == "trained_paper_bootstrap_canary_activated"
    assert result["artifact_persisted"] is True
    assert result["candidate"]["artifact_persisted"] is False
    assert result["candidate_readiness"]["allow_live_position_influence"] is False
    assert result["allow_live_position_influence"] is False
    assert result["artifact_activation_stage"] == "canary"
    assert result["paper_canary_authorized"] is True
    assert [item["activation_stage"] for item in promotion_evidence] == [
        "shadow",
        "canary",
    ]
    assert promotion_evidence[1]["paper_canary_authorized"] is True
    assert promotion_evidence[1]["production_influence_authorized"] is False
    assert promotion_evidence[1]["strategy_blueprint"][
        "paper_execution_eligible"
    ] is True
    assert promotion_evidence[1]["strategy_blueprint"][
        "live_execution_permission"
    ] is False
    assert result["readiness_state"] == "degraded"
    reason_codes = {item["code"] for item in result["candidate_readiness"]["blocking_reasons"]}
    assert "long_top_return_lcb_not_positive" in reason_codes
    assert "short_top_return_lcb_not_positive" in reason_codes


@pytest.mark.asyncio
async def test_ml_signal_retrains_when_data_quality_contract_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = MLSignalService()
    calls: list[bool] = []

    async def completed_shadow_sample_count() -> int:
        return 1300

    async def quarantine_dirty_training_samples(**_kwargs: object) -> dict[str, object]:
        return {"scanned": 1300, "quarantined": 0}

    async def load_rows() -> list[object]:
        return [object()]

    async def load_trade_samples() -> list[dict[str, object]]:
        return []

    def quality_report(_rows: list[object]) -> dict[str, object]:
        return {"quality_report": {"totals": {"total": 1}}}

    def build_frame(_rows: list[object]) -> pd.DataFrame:
        return _training_frame()

    def train_frame(_frame: pd.DataFrame, **kwargs: object) -> dict[str, object]:
        persist_artifact = bool(kwargs["persist_artifact"])
        calls.append(persist_artifact)
        return _ml_training_metadata(
            artifact_persisted=persist_artifact,
            ready=False,
        )

    stale_metadata = _ml_training_metadata(
        artifact_persisted=True,
        ready=False,
    )
    stale_metadata["quality_report"] = {
        **stale_metadata["quality_report"],
        "data_quality_version": "2026-07-14.separated-profit-supervision.v4",
    }
    service._completed_shadow_sample_count = completed_shadow_sample_count  # type: ignore[method-assign]
    service._current_metadata = lambda: stale_metadata  # type: ignore[method-assign]
    service._quarantine_dirty_training_samples = quarantine_dirty_training_samples  # type: ignore[method-assign]
    service._ensure_loaded = lambda: None  # type: ignore[method-assign]
    promotion_evidence: list[dict[str, object]] = []

    def promote_candidate(evidence: dict[str, object]) -> SimpleNamespace:
        promotion_evidence.append(evidence)
        return SimpleNamespace(version="candidate-v1")

    service.artifact_registry = SimpleNamespace(
        promote_candidate=promote_candidate,
        transition_current=promote_candidate,
    )
    monkeypatch.setattr("services.ml_signal_service.load_shadow_training_rows", load_rows)
    monkeypatch.setattr("services.ml_signal_service.shadow_training_quality_report", quality_report)
    monkeypatch.setattr("services.ml_signal_service.build_training_frame", build_frame)
    monkeypatch.setattr("services.ml_signal_service.train_from_frame", train_frame)
    monkeypatch.setattr(
        "services.ml_signal_service.load_authoritative_trade_training_samples",
        load_trade_samples,
    )

    result = await service.maybe_auto_train()

    assert calls == [False, True]
    assert result["trained"] is True
    assert result["training_policy"]["trigger"] == "training_data_contract_changed"
    assert result["training_policy"]["training_data_contract_stale"] is True
    assert len(promotion_evidence) == 2


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

    async def load_rows() -> list[object]:
        return [object()]

    async def load_trade_samples() -> list[dict[str, object]]:
        return []

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
    promotion_evidence: list[dict[str, object]] = []

    def promote_candidate(evidence: dict[str, object]) -> SimpleNamespace:
        promotion_evidence.append(evidence)
        return SimpleNamespace(version="candidate-v1")

    service.artifact_registry = SimpleNamespace(
        promote_candidate=promote_candidate,
        transition_current=promote_candidate,
    )
    monkeypatch.setattr("services.ml_signal_service.load_shadow_training_rows", load_rows)
    monkeypatch.setattr("services.ml_signal_service.shadow_training_quality_report", quality_report)
    monkeypatch.setattr("services.ml_signal_service.build_training_frame", build_frame)
    monkeypatch.setattr("services.ml_signal_service.train_from_frame", train_frame)
    monkeypatch.setattr(
        "services.ml_signal_service.load_authoritative_trade_training_samples",
        load_trade_samples,
    )

    result = await service.maybe_auto_train(force=True)

    assert calls == [False, True]
    assert ensure_load_calls == ["load"]
    assert result["trained"] is True
    assert result["reason"] == "trained_active_activated"
    assert result["artifact_persisted"] is True
    assert result["candidate"]["artifact_persisted"] is False
    assert result["candidate_readiness"]["allow_live_position_influence"] is True
    assert result["allow_live_position_influence"] is True
    assert result["artifact_activation_stage"] == "active"
    assert result["live_enabled_sides"] == ["long", "short"]
    assert [item["activation_stage"] for item in promotion_evidence] == [
        "shadow",
        "canary",
        "active",
    ]
    assert promotion_evidence[2]["production_influence_authorized"] is True
    assert promotion_evidence[2]["live_enabled_sides"] == ["long", "short"]


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
    )

    selected_ids = [row.id for row in selected]
    non_hold_count = sum(row.decision_action in {"long", "short"} for row in selected)
    missed_count = sum(bool(row.missed_opportunity) for row in selected)
    best_trade_count = sum(row.best_action in {"long", "short"} for row in selected)
    assert len(selected) == 18
    assert len(set(selected_ids)) == len(selected_ids)
    assert non_hold_count == 8
    assert missed_count == 10
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
    )

    noisy_selected = [row for row in selected if row.decision_confidence < 0.05]
    non_hold_count = sum(row.decision_action in {"long", "short"} for row in selected)
    best_trade_count = sum(row.best_action in {"long", "short"} for row in selected)
    assert len(selected) == 24
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
    )

    assert len(selected) == 48
    assert sum(row.decision_action in {"long", "short"} for row in selected) == 8
    assert sum(row.decision_action == "hold" and row.missed_opportunity for row in selected) == 40
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

    selected = select_shadow_training_rows([*noisy_holds, *clean_missed_rows])

    assert len(selected) == 12
    assert all(row.missed_opportunity for row in selected)
    assert not any(row.id in {item.id for item in noisy_holds} for row in selected)


def test_shadow_training_selection_keeps_complete_clean_opportunity_history() -> None:
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
    )

    assert len(selected) == 100
    assert sum(row.decision_action in {"long", "short"} for row in selected) == 20
    assert sum(row.decision_action == "hold" and row.missed_opportunity for row in selected) == 80
    assert all(row.missed_opportunity for row in selected[:40])


def test_train_from_frame_reports_training_window_composition() -> None:
    frame = _training_frame(120)
    frame["decision_action"] = ["hold", "long", "short"] * 40
    frame["best_action"] = ["short", "long", "short"] * 40
    frame["data_quality_status"] = ["downweighted", "included", "included"] * 40
    frame["sample_weight"] = [0.25, 1.0, 1.0] * 40

    metadata = train_from_frame(
        frame,
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
    assert metadata["objective_name"] == RETURN_OBJECTIVE_NAME
    assert metadata["objective_version"] == RETURN_OBJECTIVE_VERSION
    assert metadata["label_version"] == RETURN_LABEL_VERSION
    assert metadata["prediction_distribution"]["lower_bound"] == (
        "tree_prediction_lower_hinge"
    )
    assert metadata["prediction_distribution"]["uncertainty_source"] == (
        "random_forest_tree_empirical_order_statistics"
    )
    assert "expected_return_calibration" not in metadata
    assert "top_long_return_lcb_pct" in metadata["metrics"]
    assert "top_short_profit_factor" in metadata["metrics"]
    replay_holdout = metadata["strategy_replay_holdout"]
    assert replay_holdout["sample_count"] == metadata["test_count"]
    assert replay_holdout["decision_group_count"] == metadata[
        "test_decision_group_count"
    ]
    assert replay_holdout["shadow_source_id_ranges"]


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
    metadata = _with_return_objective({
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
    })

    readiness = build_ml_readiness_report(metadata, {"enabled": True})

    assert readiness["metrics"]["dirty_sample_ratio"] == 0.005
    assert readiness["metrics"]["benign_downweighted_sample_count"] == 395
    assert readiness["metrics"]["contamination_downweighted_sample_count"] == 5
    assert readiness["metrics"]["top_short_tail_loss_rate"] == 0.28
    assert "dirty_sample_ratio_high" not in {
        item["code"] for item in readiness["blocking_reasons"]
    }


def test_ml_readiness_allows_low_win_rate_high_fee_after_return() -> None:
    metadata = _ml_training_metadata(artifact_persisted=True, ready=True)
    metrics = metadata["metrics"]
    assert isinstance(metrics, dict)
    for side in ("long", "short"):
        metrics[f"{side}_auc"] = 0.20
        metrics[f"{side}_pr_auc"] = 0.20
        metrics[f"{side}_accuracy"] = 0.35
        metrics[f"top_{side}_win_rate"] = 0.35
        metrics[f"bottom_{side}_win_rate"] = 0.70
        metrics[f"top_{side}_avg_return_pct"] = 0.75
        metrics[f"bottom_{side}_avg_return_pct"] = -0.10
        metrics[f"top_{side}_return_lcb_pct"] = 0.30
        metrics[f"top_{side}_profit_factor"] = 2.15
        metrics[f"top_{side}_tail_loss_rate"] = 0.05
        metrics[f"bottom_{side}_tail_loss_rate"] = 0.10

    readiness = build_ml_readiness_report(metadata, {"enabled": True})

    assert readiness["state"] == "ready"
    assert readiness["allow_live_position_influence"] is True
    assert readiness["blocking_reasons"] == []


def test_ml_readiness_blocks_artifact_without_native_market_fact_contract() -> None:
    metadata = _ml_training_metadata(artifact_persisted=True, ready=True)
    metadata.pop("market_fact_contract")

    readiness = build_ml_readiness_report(metadata, {"enabled": True})
    codes = {item["code"] for item in readiness["blocking_reasons"]}

    assert readiness["state"] == "degraded"
    assert readiness["allow_live_position_influence"] is False
    assert "artifact_market_fact_contract_missing_or_stale" in codes


def test_ml_readiness_blocks_artifact_with_market_fact_contract_violation() -> None:
    metadata = _ml_training_metadata(artifact_persisted=True, ready=True)
    contract = metadata["market_fact_contract"]
    assert isinstance(contract, dict)
    contract["status"] = "quarantined"
    contract["violation_count"] = 71

    readiness = build_ml_readiness_report(metadata, {"enabled": True})
    codes = {item["code"] for item in readiness["blocking_reasons"]}

    assert readiness["state"] == "degraded"
    assert readiness["allow_live_position_influence"] is False
    assert "artifact_market_fact_contract_violated" in codes


def test_ml_readiness_blocks_high_win_rate_negative_fee_after_return() -> None:
    metadata = _ml_training_metadata(artifact_persisted=True, ready=True)
    metrics = metadata["metrics"]
    assert isinstance(metrics, dict)
    for side in ("long", "short"):
        metrics[f"{side}_auc"] = 0.95
        metrics[f"{side}_pr_auc"] = 0.95
        metrics[f"{side}_accuracy"] = 0.80
        metrics[f"top_{side}_win_rate"] = 0.80
        metrics[f"bottom_{side}_win_rate"] = 0.40
        metrics[f"top_{side}_avg_return_pct"] = -0.32
        metrics[f"bottom_{side}_avg_return_pct"] = -0.10
        metrics[f"top_{side}_return_lcb_pct"] = -0.60
        metrics[f"top_{side}_profit_factor"] = 0.20
        metrics[f"top_{side}_tail_loss_rate"] = 0.20
        metrics[f"bottom_{side}_tail_loss_rate"] = 0.10

    readiness = build_ml_readiness_report(metadata, {"enabled": True})
    codes = {item["code"] for item in readiness["blocking_reasons"]}

    assert readiness["state"] == "degraded"
    assert readiness["allow_live_position_influence"] is False
    assert "long_top_return_lcb_not_positive" in codes
    assert "short_top_profit_factor_not_above_one" in codes


def test_ml_readiness_separates_paper_bootstrap_from_live_profit_gate() -> None:
    metadata = _ml_training_metadata(artifact_persisted=True, ready=False)

    readiness = build_ml_readiness_report(metadata, {"enabled": False})

    assert readiness["allow_live_position_influence"] is False
    assert readiness["paper_canary"]["authorized"] is True
    assert readiness["paper_canary"]["execution_scope"] == "paper_only"
    assert readiness["paper_canary"]["production_permission"] is False
    assert set(readiness["paper_canary"]["eligible_sides"]) == {"long", "short"}


def test_ml_signal_predict_uses_direct_regressor_not_win_probability_calibration() -> None:
    metadata = _with_return_objective({
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
    })
    service = MLSignalService()
    service._bundle = {
        "metadata": metadata,
        "long_classifier": _Classifier(0.8),
        "short_classifier": _Classifier(0.2),
        "long_regressor": _Regressor(-9.0),
        "short_regressor": _Regressor(9.0),
        "long_cost_regressor": _Regressor(0.08),
        "short_cost_regressor": _Regressor(0.08),
        "feature_keys": FEATURE_KEYS,
    }
    service._ensure_loaded = lambda: None  # type: ignore[method-assign]

    prediction = service.predict({"current_price": 100.0, "spread_pct": 0.01}, horizons=(30,))
    primary = prediction["predictions"][0]
    distributions = primary["return_distribution_contract"]

    assert primary["best_side"] == "short"
    assert distributions["long"]["raw_expected_return_pct"] == pytest.approx(-9.0)
    assert distributions["short"]["raw_expected_return_pct"] == pytest.approx(9.0)


def test_ml_signal_predict_penalizes_excess_tail_loss_probability() -> None:
    metadata = _with_return_objective({
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
        "tail_loss_scale_pct": {"long": 0.18, "short": 0.18},
    })
    service = MLSignalService()
    service._bundle = {
        "metadata": metadata,
        "long_classifier": _Classifier(0.6),
        "short_classifier": _Classifier(0.6),
        "long_tail_classifier": _Classifier(0.10),
        "short_tail_classifier": _Classifier(0.55),
        "long_regressor": _Regressor(0.3),
        "short_regressor": _Regressor(0.3),
        "long_cost_regressor": _Regressor(0.08),
        "short_cost_regressor": _Regressor(0.08),
        "feature_keys": FEATURE_KEYS,
    }
    service._ensure_loaded = lambda: None  # type: ignore[method-assign]

    prediction = service.predict({"current_price": 100.0, "spread_pct": 0.01}, horizons=(30,))
    primary = prediction["predictions"][0]
    distributions = primary["return_distribution_contract"]

    assert primary["best_side"] == "long"
    assert distributions["long"]["raw_expected_return_pct"] == pytest.approx(0.3)
    assert distributions["short"]["raw_expected_return_pct"] == pytest.approx(0.3)
    assert distributions["long"]["objective_expected_return_pct"] == pytest.approx(
        0.279
    )
    assert distributions["short"]["objective_expected_return_pct"] == pytest.approx(
        0.198
    )
    assert distributions["short"]["tail_loss_probability"] == pytest.approx(0.55)
    assert distributions["long"]["tail_loss_probability"] == pytest.approx(0.10)


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
        selected = await load_shadow_training_rows()
    finally:
        await close_db()

    selected_ids = {row.id for row in selected}
    assert len(selected) == 22
    assert all(not isinstance(row, ShadowBacktest) for row in selected)
    assert 90 not in selected_ids
    assert 91 not in selected_ids
    assert sum(row.decision_action in {"long", "short"} for row in selected) == 8
    assert sum(row.decision_action == "hold" and row.missed_opportunity for row in selected) == 14
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
        selected = await load_shadow_training_rows()
    finally:
        await close_db()

    assert len(selected) == 25
    assert {row.decision_action for row in selected} <= {"long", "short"}
    assert {row.best_action for row in selected} <= {"long", "short"}
    assert not any(row.id >= 20_000 for row in selected)


@pytest.mark.asyncio
async def test_load_shadow_training_rows_projects_only_training_and_quality_features(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    await _use_temp_db(monkeypatch, tmp_path)
    created_at = datetime(2026, 6, 28, 3, 0, tzinfo=UTC)
    row = _db_shadow_row(
        77,
        created_at,
        action="long",
        best_action="long",
    )
    row.feature_snapshot = {
        "current_price": 100.0,
        "spread_pct": 0.01,
        "round_trip_fee_pct": 0.08,
        "funding_rate": 0.0,
        "funding_interval_minutes": 480.0,
        "feature_timestamp": created_at.isoformat(),
        "market_data_quality": {"code": ""},
        "training_market_fact_contract": _clean_training_market_fact_contract(),
        "training_label_contract": _clean_training_label_contract(
            77,
            row.due_at,
            decision_id=row.decision_id,
            long_return_pct=float(row.long_return_pct or 0.0),
            short_return_pct=float(row.short_return_pct or 0.0),
            best_action=str(row.best_action or "hold"),
        ),
        "unused_llm_context": {"transcript": "x" * 100_000},
    }
    async with get_session_ctx() as session:
        session.add(row)

    try:
        selected = await load_shadow_training_rows()
        async with get_session_ctx() as session:
            compact = await session.get(ShadowBacktest, 77)
            assert compact is not None
            assert compact.training_feature_snapshot_version == 1
            assert compact.training_feature_snapshot["market_data_quality"] == {"code": ""}
            assert "unused_llm_context" not in compact.training_feature_snapshot
    finally:
        await close_db()

    assert len(selected) == 1
    snapshot = selected[0].feature_snapshot
    assert snapshot["current_price"] == 100.0
    assert snapshot["spread_pct"] == 0.01
    assert snapshot["feature_timestamp"] == created_at.isoformat()
    assert snapshot["market_data_quality"] == {"code": ""}
    assert "unused_llm_context" not in snapshot


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
    assert status["readiness_state"] == "degraded"
    assert status["allow_live_position_influence"] is False
    assert status["readiness"]["metrics"]["dirty_sample_ratio"] == 0.0
    assert status["readiness"]["metrics"]["training_data_version"] == "2026-06-19.v1"
    assert status["readiness"]["metrics"]["required_training_data_version"] == DATA_QUALITY_VERSION
    assert "sample_count_below_threshold" not in reason_codes
    assert "test_count_below_threshold" not in reason_codes
    assert "long_pr_auc_missing" not in reason_codes
    assert "short_pr_auc_missing" not in reason_codes
    assert "training_data_version_stale" in reason_codes
    assert "model_stale" not in reason_codes
    next_conditions = status["readiness"]["next_training_conditions"]
    assert next_conditions["trigger"] == (
        "new_authoritative_cost_complete_sample_or_data_contract_change"
    )
    assert next_conditions["trigger"].startswith("new_authoritative_cost_complete_sample")


def test_ml_signal_readiness_surfaces_fee_after_profit_bucket_diagnostics() -> None:
    metadata = _with_return_objective({
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
            "short_auc": 0.61,
            "long_pr_auc": 0.60,
            "short_pr_auc": 0.60,
            "long_accuracy": 0.60,
            "short_accuracy": 0.60,
            "top_long_avg_return_pct": -0.14,
            "bottom_long_avg_return_pct": -0.02,
            "top_long_win_rate": 0.42,
            "bottom_long_win_rate": 0.51,
            "top_long_tail_loss_rate": 0.18,
            "bottom_long_tail_loss_rate": 0.07,
            "top_short_avg_return_pct": 0.18,
            "bottom_short_avg_return_pct": -0.03,
            "top_short_win_rate": 0.70,
            "bottom_short_win_rate": 0.42,
            "top_short_tail_loss_rate": 0.04,
            "bottom_short_tail_loss_rate": 0.09,
        },
        "score_bucket_diagnostics": {
            "long": {
                "top": {
                    "count": 48,
                    "tail_loss_rate": 0.18,
                    "action_counts": {"long": 48},
                    "top_quality_reasons": [{"reason": "fee_drag", "count": 9}],
                },
                "bottom": {
                    "count": 48,
                    "tail_loss_rate": 0.07,
                    "action_counts": {"short": 48},
                },
            }
        },
    })

    readiness = build_ml_readiness_report(metadata, {"enabled": True})
    long_diag = readiness["profit_quality_diagnostics"]["long"]

    assert readiness["allow_live_position_influence"] is True
    assert long_diag["training_target"] == "fee_after_realized_return_quality"
    assert long_diag["top_avg_return_pct"] == -0.14
    assert long_diag["top_bottom_return_spread_pct"] == -0.12
    assert "top_score_bucket_not_better_than_bottom" in long_diag["diagnosis"]
    assert "top_score_tail_loss_worse_than_bottom" in long_diag["diagnosis"]
    assert long_diag["top_bucket"]["top_quality_reasons"][0]["reason"] == "fee_drag"
    assert readiness["metrics"]["top_long_bottom_return_spread_pct"] == -0.12


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


def test_live_activation_auto_degrades_when_training_fingerprint_is_invalid() -> None:
    metadata = _ml_training_metadata(artifact_persisted=True, ready=True)
    metadata["training_data_sha256"] = "invalid"
    service = _service_with_metadata(metadata)

    status = service.status()
    codes = {item["code"] for item in status["readiness"]["blocking_reasons"]}

    assert status["allow_live_position_influence"] is False
    assert status["influence_policy"]["enabled"] is False
    assert "artifact_training_data_sha256_missing_or_invalid" in codes
    assert "artifact_current_readiness_revalidation_failed" in codes


def test_actual_trade_profit_factor_must_be_defined_for_each_live_side() -> None:
    metadata = _ml_training_metadata(artifact_persisted=True, ready=True)
    long_evidence = metadata["authoritative_trade_return_evidence"]["sides"]["long"]
    long_evidence["profit_factor"] = None
    long_evidence["promotion_math_ready"] = False
    service = _service_with_metadata(metadata)

    status = service.status()
    long_codes = {
        item["code"]
        for item in status["readiness"]["side_blocking_reasons"]["long"]
    }

    assert status["readiness_state"] == "partial_ready"
    assert status["readiness"]["live_enabled_sides"] == ["short"]
    assert status["influence_policy"]["long"]["enabled"] is False
    assert status["influence_policy"]["short"]["enabled"] is True
    assert "long_authoritative_profit_factor_undefined" in long_codes


def test_symbol_removal_instability_blocks_that_prediction_side() -> None:
    metadata = _ml_training_metadata(artifact_persisted=True, ready=True)
    metadata["leave_one_symbol_out_report"]["long"]["stable"] = False
    service = _service_with_metadata(metadata)
    service._bundle.update(
        {
            "long_classifier": _Classifier(0.84),
            "short_classifier": _Classifier(0.20),
            "long_tail_classifier": _Classifier(0.10),
            "short_tail_classifier": _Classifier(0.10),
            "long_regressor": _Regressor(0.24),
            "short_regressor": _Regressor(0.02),
            "long_cost_regressor": _Regressor(0.08),
            "short_cost_regressor": _Regressor(0.08),
        }
    )

    prediction = service.predict(
        {"current_price": 100.0, "atr_14": 1.0},
        horizons=(10,),
    )
    long_codes = {
        item["code"]
        for item in prediction["readiness"]["side_blocking_reasons"]["long"]
    }

    assert prediction["readiness"]["live_enabled_sides"] == ["short"]
    assert prediction["predictions"][0]["best_side"] == "long"
    assert prediction["allow_live_position_influence"] is False
    assert prediction["influence_policy"]["long"]["enabled"] is False
    assert "long_leave_one_symbol_out_stability_failed" in long_codes


def test_ml_signal_status_allows_directional_partial_live_influence() -> None:
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
                "long_auc": 0.64,
                "short_auc": 0.55,
                "long_pr_auc": 0.61,
                "short_pr_auc": 0.42,
                "long_accuracy": 0.62,
                "short_accuracy": 0.51,
                "top_long_avg_return_pct": 0.22,
                "bottom_long_avg_return_pct": -0.03,
                "top_short_avg_return_pct": -0.08,
                "bottom_short_avg_return_pct": -0.02,
                "top_long_win_rate": 0.75,
                "bottom_long_win_rate": 0.38,
                "top_short_win_rate": 0.42,
                "bottom_short_win_rate": 0.51,
            },
        }
    )

    status = service.status()

    assert status["readiness_state"] == "partial_ready"
    assert status["status"] == "ready"
    assert status["allow_live_position_influence"] is True
    assert status["readiness"]["live_enabled_sides"] == ["long"]
    assert status["readiness"]["blocking_reasons"] == []
    short_codes = {
        item["code"] for item in status["readiness"]["side_blocking_reasons"]["short"]
    }
    assert "short_pr_auc_below_threshold" not in short_codes
    assert "short_top_return_not_above_bottom" in short_codes


def test_ml_signal_predict_uses_enabled_side_when_other_side_is_degraded() -> None:
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
            "long_auc": 0.64,
            "short_auc": 0.55,
            "long_pr_auc": 0.61,
            "short_pr_auc": 0.42,
            "long_accuracy": 0.62,
            "short_accuracy": 0.51,
            "top_long_avg_return_pct": 0.22,
            "bottom_long_avg_return_pct": -0.03,
            "top_short_avg_return_pct": -0.08,
            "bottom_short_avg_return_pct": -0.02,
            "top_long_win_rate": 0.75,
            "bottom_long_win_rate": 0.38,
            "top_short_win_rate": 0.42,
            "bottom_short_win_rate": 0.51,
        },
    }
    service = _service_with_metadata(metadata)
    service._bundle.update(
        {
            "long_classifier": _Classifier(0.84),
            "short_classifier": _Classifier(0.20),
            "long_tail_classifier": _Classifier(0.10),
            "short_tail_classifier": _Classifier(0.10),
            "long_regressor": _Regressor(0.24),
            "short_regressor": _Regressor(0.02),
            "long_cost_regressor": _Regressor(0.08),
            "short_cost_regressor": _Regressor(0.08),
        }
    )

    prediction = service.predict({"current_price": 100.0, "atr_14": 1.0}, horizons=(10,))

    assert prediction["readiness_state"] == "partial_ready"
    assert prediction["allow_live_position_influence"] is True
    assert prediction["predictions"][0]["best_side"] == "long"
    assert prediction["predictions"][0]["ml_influence_enabled"] is True
    assert prediction["predictions"][0]["profit_signal"] is True


def test_ml_signal_predict_blocks_lower_quantile_above_point_without_clamping() -> None:
    metadata = _ml_training_metadata(artifact_persisted=True, ready=True)
    service = _service_with_metadata(metadata)
    service._bundle.update(
        {
            "long_classifier": _Classifier(0.8),
            "short_classifier": _Classifier(0.2),
            "long_tail_classifier": _Classifier(0.1),
            "short_tail_classifier": _Classifier(0.1),
            "long_regressor": _Regressor(
                0.46,
                tree_predictions=(0.496, 0.51),
            ),
            "short_regressor": _Regressor(
                0.2,
                tree_predictions=(0.22, 0.23),
            ),
            "long_cost_regressor": _Regressor(0.08),
            "short_cost_regressor": _Regressor(0.08),
        }
    )

    prediction = service.predict(
        {"symbol": "ICP/USDT", "current_price": 5.0, "atr_14": 0.2},
        horizons=(30,),
    )
    primary = prediction["predictions"][0]
    long_distribution = primary["return_distribution_contract"]["long"]

    assert long_distribution["raw_expected_return_pct"] == pytest.approx(0.46)
    assert long_distribution["lower_quantile_return_pct"] == pytest.approx(0.496)
    assert long_distribution["production_eligible"] is False
    assert "lower_quantile_above_raw_expected" in long_distribution["blockers"]
    assert prediction["allow_live_position_influence"] is False
    assert prediction["prediction_quality"]["production_eligible"] is False


def test_strategy_replay_batch_matches_single_prediction_contract() -> None:
    metadata = _ml_training_metadata(artifact_persisted=True, ready=True)
    service = _service_with_metadata(metadata)
    service._bundle.update(
        {
            "long_classifier": _Classifier(0.8),
            "short_classifier": _Classifier(0.2),
            "long_tail_classifier": _Classifier(0.1),
            "short_tail_classifier": _Classifier(0.1),
            "long_regressor": _Regressor(0.46, tree_predictions=(0.31, 0.61)),
            "short_regressor": _Regressor(0.2, tree_predictions=(0.1, 0.3)),
            "long_cost_regressor": _Regressor(0.08, tree_predictions=(0.04, 0.12)),
            "short_cost_regressor": _Regressor(0.09, tree_predictions=(0.05, 0.13)),
        }
    )
    features = {"symbol": "BTC/USDT", "current_price": 100.0, "atr_14": 1.0}

    single = service.predict(features, horizons=(10,))
    batch = service.predict_strategy_replay_batch(
        [features],
        horizon_minutes=10,
    )[0]

    assert batch["model_version"] == single["model_version"]
    assert batch["predictions"][0]["best_side"] == single["predictions"][0][
        "best_side"
    ]
    assert batch["predictions"][0]["return_distribution_contract"] == single[
        "predictions"
    ][0]["return_distribution_contract"]
    assert batch["predictions"][0]["actual_trade_calibration_ready"] == single[
        "predictions"
    ][0]["actual_trade_calibration_ready"]


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
            "long_cost_regressor": _Regressor(0.08),
            "short_cost_regressor": _Regressor(0.08),
        }
    )

    prediction = service.predict({"current_price": 100.0, "atr_14": 1.0}, horizons=(10,))

    assert prediction["readiness_state"] == "degraded"
    assert prediction["allow_live_position_influence"] is False
    assert prediction["influence_policy"]["enabled"] is False
    assert prediction["influence_enabled"] is False
    assert prediction["profit_signal"] is False
    assert prediction["predictions"][0]["ml_influence_enabled"] is False
    assert prediction["predictions"][0]["profit_signal"] is False
