from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import event

from config.settings import settings
from db.session import close_db, get_engine, get_session_ctx, init_db
from models.learning import ShadowBacktest
from scripts import run_specialist_shadow_evaluation as runner
from services import specialist_shadow_evaluation as specialist_evaluation
from services.continuous_model_weight import ContinuousModelWeightPolicy
from services.entry_direction_competition import _paper_quant_quality_permissions
from services.ml_prediction_contract import ENTRY_COHORT_VERSION, PREDICTION_SCORE_POLICY_VERSION
from services.profit_training_contract import PROFIT_TRAINING_TARGET
from services.specialist_shadow_evaluation import (
    SpecialistShadowEvaluationAccumulator,
    SpecialistShadowEvaluationService,
    _empty_metric,
    _mean_lcb,
    _mean_lcb_from_moments,
    _regime_stability_report,
    _remember_shadow_event,
    _rolling_distribution_report,
    _walk_forward_report,
    summarize_specialist_shadow_evaluation,
)


@pytest.fixture(autouse=True)
def _current_training_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "services.specialist_shadow_evaluation.load_training_epoch_start",
        lambda: datetime(2026, 7, 23, tzinfo=UTC),
    )


async def _use_temp_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    await close_db()
    db_path = tmp_path / "specialist-shadow.db"
    monkeypatch.setattr(settings, "database_url", f"sqlite+aiosqlite:///{db_path.as_posix()}")
    await init_db()


def test_specialist_shadow_evaluation_script_imports_online_runtime_bootstrap() -> None:
    source = runner.ROOT.joinpath("scripts", "run_specialist_shadow_evaluation.py").read_text(
        encoding="utf-8"
    )

    assert "from scripts.runtime_env_bootstrap import" in source
    assert "load_runtime_env_files(project_root=ROOT)" in source
    assert "drop_privileges_to_runtime_user_if_needed(project_root=ROOT)" in source
    assert "_load_evaluation_trade_samples" in source
    assert "authoritative_trade_samples=authoritative_trade_samples" in source


def test_specialist_shadow_evaluation_default_report_dir_matches_phase3_readers() -> None:
    assert runner.DEFAULT_REPORT_DIR == "phase3"


def test_specialist_reports_reject_legacy_return_label_fallback() -> None:
    legacy_event = {
        "return_after_all_cost_pct": 9.0,
        "label_timestamp": "2026-07-12T00:00:00+00:00",
        "market_regime": "trend",
    }

    walk_forward = _walk_forward_report([legacy_event])
    regime = _regime_stability_report([legacy_event])
    rolling = _rolling_distribution_report([legacy_event])

    assert walk_forward["sample_count"] == 0
    assert walk_forward["missing_canonical_return_count"] == 1
    assert regime["regimes"] == []
    assert regime["missing_canonical_return_count"] == 1
    assert rolling["sample_count"] == 0
    assert rolling["missing_canonical_return_count"] == 1


def test_entry_decision_persists_market_regime_for_authoritative_evaluation() -> None:
    source = Path("ai_brain/ensemble_coordinator.py").read_text(encoding="utf-8")

    assert 'raw["market_regime"] = dict(context["market_regime"])' in source


def _row(
    *,
    symbol: str = "BTC/USDT",
    best_action: str = "long",
    predicted_side: str = "long",
    realized: float = 0.42,
    expected: float = 0.31,
    tool: str = "time_series_prediction",
    actual_inference: bool = True,
    index: int = 1,
) -> SimpleNamespace:
    now = datetime.now(UTC) - timedelta(minutes=index)
    long_return = realized if predicted_side == "long" else -realized
    short_return = realized if predicted_side == "short" else -realized
    local_shadow = {
        tool: {
            "model": "local-timeseries-ensemble-v1",
            "best_side": predicted_side,
            "expected_return_pct": expected,
            "specialist_inference_active": actual_inference,
            "professional_model_shadow": {
                "actual_inference": actual_inference,
                "primary_model": "google/timesfm-2.5-200m-pytorch",
                "challenger_model": "amazon/chronos-2",
                "shadow_result": {
                    "model": "timesfm-2.5-shadow-challenger",
                    "actual_inference": actual_inference,
                    "expected_return_pct": expected,
                    "best_side": predicted_side,
                    "sequence_length": 60,
                },
            },
        }
    }
    if tool == "time_series_prediction":
        local_shadow[tool]["timesfm_shadow_side"] = predicted_side
        local_shadow[tool]["timesfm_shadow_expected_return_pct"] = expected
        local_shadow[tool]["professional_model_shadow"] = {
            "actual_inference": actual_inference,
            "primary_model": "google/timesfm-2.5-200m-pytorch",
            "challenger_model": "amazon/chronos-2",
            "primary_shadow_result": {
                "model": "chronos-2-shadow-primary",
                "actual_inference": actual_inference,
                "expected_return_pct": expected / 2,
                "best_side": predicted_side,
                "sequence_length": 60,
            },
            "challenger_shadow_result": {
                "model": "timesfm-2.5-shadow-challenger",
                "actual_inference": actual_inference,
                "expected_return_pct": expected,
                "best_side": predicted_side,
                "sequence_length": 60,
            },
        }
    return SimpleNamespace(
        id=index,
        decision_id=10_000 + index,
        status="completed",
        symbol=symbol,
        best_action=best_action,
        long_return_pct=long_return,
        short_return_pct=short_return,
        horizon_minutes=10,
        feature_snapshot={
            "symbol": symbol,
            "market_regime": "trend" if index % 2 else "range",
            "spread_pct": 0.02,
            "taker_fee_rate": 0.0004,
            "funding_rate": 0.0001,
            "funding_data_available": True,
            "funding_interval_hours": 8.0,
            "local_ai_tools_shadow": local_shadow,
        },
        due_at=now,
        created_at=now,
    )


@pytest.mark.parametrize(("side", "cashflow"), (("long", -0.01), ("short", 0.01)))
@pytest.mark.parametrize("crosses_settlement", (False, True))
def test_specialist_funding_matches_training_settlement_schedule(side, cashflow, crosses_settlement):
    row = _row(predicted_side=side, realized=0.42)
    decision_at = datetime(2026, 11, 10, 7, tzinfo=UTC)
    row.horizon_minutes = 240
    row.feature_snapshot.update({
        "feature_timestamp": (decision_at + timedelta(hours=1)).isoformat(),
        "original_decision_timestamp": decision_at.isoformat(),
        "funding_rate_observed_at": decision_at.isoformat(),
        "next_funding_time": (decision_at + timedelta(hours=1 if crosses_settlement else 5)).isoformat(),
    })
    result = specialist_evaluation._cost_complete_shadow_return(
        row, side=side, gross_return_pct=0.42,
    )
    assert result is not None
    net, costs = result
    expected_funding = cashflow if crosses_settlement else 0.0
    assert costs["funding_drag_pct"] == pytest.approx(-expected_funding)
    assert net == pytest.approx(0.42 - costs["fee_pct"] - costs["slippage_pct"] + expected_funding)
    labels = specialist_evaluation.shadow_fee_after_return_labels(
        specialist_evaluation._shadow_quality_payload(row)
    )
    assert net == pytest.approx(labels[f"{side}_net_return_after_all_cost_pct"])


@pytest.mark.parametrize("invalid", ("unavailable", "future_observation", "missing_schedule"))
def test_specialist_rejects_incomplete_funding_evidence(invalid):
    row = _row()
    decision_at = datetime(2026, 11, 10, 7, tzinfo=UTC)
    row.feature_snapshot.update({
        "timestamp": decision_at.isoformat(),
        "funding_rate_observed_at": decision_at.isoformat(),
        "next_funding_time": (decision_at + timedelta(hours=1)).isoformat(),
    })
    if invalid == "unavailable":
        row.feature_snapshot["funding_data_available"] = False
    elif invalid == "future_observation":
        row.feature_snapshot["funding_rate_observed_at"] = (
            decision_at + timedelta(minutes=1)
        ).isoformat()
    else:
        row.feature_snapshot.pop("next_funding_time")
    assert specialist_evaluation._cost_complete_shadow_return(
        row, side="long", gross_return_pct=0.42,
    ) is None


def test_specialist_best_side_recomputes_current_costs_without_mutating_audit_label():
    row = _row(best_action="short", predicted_side="long", realized=0.42)
    assert specialist_evaluation._actual_best_side(row) == "long"
    assert row.best_action == "short"
    row.feature_snapshot["funding_data_available"] = False
    assert specialist_evaluation._actual_best_side(row) == ""


def _authoritative_sample(
    index: int,
    *,
    pnl_ratio_pct: float = 0.25,
    predicted_side: str = "long",
    position_side: str = "long",
    actual_inference: bool = True,
) -> dict:
    professional = {
        "actual_inference": actual_inference,
        "primary_model": "google/timesfm-2.5-200m-pytorch",
        "challenger_model": "amazon/chronos-2",
        "primary_shadow_result": {
            "actual_inference": actual_inference,
            "best_side": predicted_side,
            "sequence_length": 60,
        },
        "challenger_shadow_result": {
            "actual_inference": actual_inference,
            "best_side": predicted_side,
            "sequence_length": 60,
        },
    }
    return {
        "source": "okx_position_history",
        "id": index,
        "lifecycle_key": f"okx-lifecycle-{index}",
        "decision_id": 20_000 + index,
        "trade_fact_trusted": True,
        "side": position_side,
        "symbol": "BTC/USDT",
        PROFIT_TRAINING_TARGET: pnl_ratio_pct,
        "label_timestamp": (datetime(2026, 7, 1, tzinfo=UTC) + timedelta(hours=index)).isoformat(),
        "raw_llm_response": {
            "market_regime": "trend" if index % 2 else "range",
            "local_ai_tools": {
                "time_series_prediction": {
                    "specialist_inference_active": actual_inference,
                    "professional_model_shadow": professional,
                }
            },
        },
    }


def test_specialist_shadow_evaluation_skips_baseline_only_profit_shadow() -> None:
    row = _row(tool="profit_prediction", actual_inference=False)
    row.feature_snapshot["local_ai_tools_shadow"]["profit_prediction"][
        "professional_model_shadow"
    ] = {
        "kind": "profit",
        "actual_inference": False,
        "baseline_response": True,
        "activation_blocker": "profit_specialist_pending_phase3_clean_rebuild",
    }

    report = summarize_specialist_shadow_evaluation([row])

    assert report["completed_count"] == 1
    assert report["eligible_shadow_count"] == 0
    assert report["model_count"] == 0
    assert report["skipped_reasons"]["profit_prediction_baseline_only_shadow"] == 1


def test_specialist_shadow_evaluation_skips_non_specialist_heuristic_shadow() -> None:
    row = _row(tool="profit_prediction", actual_inference=False)
    row.feature_snapshot["local_ai_tools_shadow"]["profit_prediction"][
        "professional_model_shadow"
    ] = {
        "kind": "profit",
        "actual_inference": False,
        "baseline_response": False,
    }

    report = summarize_specialist_shadow_evaluation([row])

    assert report["completed_count"] == 1
    assert report["eligible_shadow_count"] == 0
    assert report["model_count"] == 0
    assert report["skipped_reasons"]["profit_prediction_non_specialist_shadow"] == 1


def test_local_ml_shadow_settles_its_own_decision_time_prediction() -> None:
    row = _row(predicted_side="long", realized=0.42)
    row.feature_snapshot["local_ml_shadow"] = {
        "version": "2026-11-09.local-ml-shadow.v1",
        "available": True,
        "actual_inference": True,
        "model_version": "local-ml-v7",
        "prediction_score_policy_version": PREDICTION_SCORE_POLICY_VERSION,
        "primary_horizon_minutes": 10,
        "status": "learning_only",
        "readiness_state": "learning_only",
        "predictions": [
            {
                "horizon_minutes": 10,
                "best_side": "long",
                "long": {
                    "objective_net_return_pct": 0.25,
                },
                "short": {
                    "objective_net_return_pct": -0.20,
                },
            }
        ],
    }

    report = summarize_specialist_shadow_evaluation([row])

    local = next(item for item in report["models"] if item["tool"] == "local_ml")
    assert local["model"] == "local-ml-v7"
    assert local["shadow_training_only"] is True
    assert local["direction_count"] == 1
    assert local["actual_inference_count"] == 1
    assert local["avg_shadow_return_after_all_cost_pct"] > 0.0
    assert local["promotion_ready"] is False
    entry = local["entry_return_evaluation"]
    assert entry["version"] == ENTRY_COHORT_VERSION
    assert entry["direction_count"] == 1
    assert entry["avg_shadow_return_after_all_cost_pct"] == local["avg_shadow_return_after_all_cost_pct"]


def test_local_ml_shadow_does_not_settle_other_model_shadow_sides() -> None:
    row = _row(predicted_side="short", realized=0.42)
    row.feature_snapshot["local_ml_shadow"] = {
        "version": "2026-11-09.local-ml-shadow.v1",
        "available": True,
        "actual_inference": True,
        "model_version": "local-ml-v7",
        "prediction_score_policy_version": PREDICTION_SCORE_POLICY_VERSION,
        "primary_horizon_minutes": 10,
        "predictions": [
            {
                "horizon_minutes": 10,
                "best_side": "long",
                "long": {"objective_net_return_pct": 0.25},
                "short": {"objective_net_return_pct": -0.20},
            }
        ],
    }

    report = summarize_specialist_shadow_evaluation([row])

    local = next(item for item in report["models"] if item["tool"] == "local_ml")
    timeseries = next(
        item for item in report["models"] if item["tool"] == "time_series_prediction"
    )
    assert local["direction_count"] == 1
    assert timeseries["direction_count"] == 1
    assert local["avg_shadow_return_after_all_cost_pct"] != timeseries[
        "avg_shadow_return_after_all_cost_pct"
    ]


def _local_ml_row(*, index=1, realized=0.42, version="local-ml-v7"):
    row = _row(index=index, realized=realized)
    row.feature_snapshot["local_ml_shadow"] = {
        "available": True,
        "actual_inference": True,
        "model_version": version,
        "prediction_score_policy_version": PREDICTION_SCORE_POLICY_VERSION,
        "primary_horizon_minutes": 10,
        "predictions": [{
            "horizon_minutes": 10,
            "best_side": "long",
            "long": {"objective_net_return_pct": 0.25},
            "short": {"objective_net_return_pct": -0.20},
        }],
    }
    return row


@pytest.mark.parametrize("realized", [-0.5, 0.5])
def test_local_ml_entry_cohort_excludes_hold_predictions_regardless_of_outcome(realized):
    row = _local_ml_row(realized=realized)
    prediction = row.feature_snapshot["local_ml_shadow"]["predictions"][0]
    prediction["long"]["objective_net_return_pct"] = -0.1
    prediction["short"]["objective_net_return_pct"] = -0.2

    local = next(item for item in summarize_specialist_shadow_evaluation([row])["models"]
                 if item["tool"] == "local_ml")

    assert local["direction_count"] == 1
    assert local["entry_return_evaluation"]["direction_count"] == 0
    assert local["entry_return_evaluation"]["shadow_return_lcb_pct"] is None


def test_local_ml_positive_entry_keeps_losses_and_separates_model_versions():
    loss = _local_ml_row(realized=-0.5)
    win = _local_ml_row(index=2, realized=0.5, version="other-version")
    report = summarize_specialist_shadow_evaluation([loss, win])
    models = {item["model"]: item for item in report["models"] if item["tool"] == "local_ml"}

    entry = models["local-ml-v7"]["entry_return_evaluation"]
    assert entry["direction_count"] == 1
    assert entry["gross_loss_return_pct"] > 0.5
    assert entry["gross_profit_return_pct"] == 0.0
    assert entry["shadow_return_lcb_pct"] < 0.0
    assert models["other-version"]["entry_return_evaluation"]["shadow_return_lcb_pct"] > 0.0


@pytest.mark.parametrize("invalid", [
    "missing_other_side", "expected_only", "gross_only", "nan", "infinity",
    "wrong_best_side", "zero", "unavailable", "no_inference", "no_primary", "other_primary",
    "incomplete_cost",
    "legacy_score_policy", "missing_score_policy",
])
def test_local_ml_entry_cohort_requires_complete_frozen_primary_prediction(invalid):
    row = _local_ml_row()
    shadow = row.feature_snapshot["local_ml_shadow"]
    prediction = shadow["predictions"][0]
    if invalid == "missing_other_side":
        prediction.pop("short")
    elif invalid in {"expected_only", "gross_only"}:
        key = "expected_net_return_pct" if invalid == "expected_only" else "objective_expected_return_pct"
        prediction["long"] = {key: 10.0}
    elif invalid in {"nan", "infinity"}:
        prediction["long"]["objective_net_return_pct"] = float("nan" if invalid == "nan" else "inf")
    elif invalid == "wrong_best_side":
        prediction["best_side"] = "short"
    elif invalid == "zero":
        prediction["long"]["objective_net_return_pct"] = 0.0
    elif invalid == "unavailable":
        shadow["available"] = False
    elif invalid == "no_inference":
        shadow["actual_inference"] = False
    elif invalid == "no_primary":
        shadow.pop("primary_horizon_minutes")
    elif invalid == "other_primary":
        shadow["primary_horizon_minutes"] = 240
    elif invalid == "legacy_score_policy":
        shadow["prediction_score_policy_version"] = "tree-mean-only-legacy"
    elif invalid == "missing_score_policy":
        shadow.pop("prediction_score_policy_version")
    else:
        row.feature_snapshot["funding_data_available"] = False

    report = summarize_specialist_shadow_evaluation([row])
    local = next(item for item in report["models"] if item["tool"] == "local_ml")
    assert local["entry_return_evaluation"]["direction_count"] == 0


def test_local_ml_positive_entry_selects_short_and_uses_long_on_ties():
    short_row = _local_ml_row()
    prediction = short_row.feature_snapshot["local_ml_shadow"]["predictions"][0]
    prediction["short"]["objective_net_return_pct"] = 0.5
    prediction["best_side"] = "short"
    tie_row = _local_ml_row(index=2)
    tie_row.feature_snapshot["local_ml_shadow"]["predictions"][0]["short"]["objective_net_return_pct"] = 0.25

    report = summarize_specialist_shadow_evaluation([short_row, tie_row])
    local = next(item for item in report["models"] if item["tool"] == "local_ml")
    entry = local["entry_return_evaluation"]
    assert entry["direction_count"] == 2
    assert [event["predicted_side"] for event in entry["shadow_events"]] == ["short", "long"]
    assert entry["gross_loss_return_pct"] > 0
    assert entry["gross_profit_return_pct"] > 0


@pytest.mark.parametrize("legacy_return", [-100.0, 100.0])
def test_local_ml_same_artifact_cannot_mix_results_from_different_scoring_policies(legacy_return):
    current = _local_ml_row(realized=-0.5)
    legacy = _local_ml_row(index=2, realized=legacy_return)
    legacy.feature_snapshot["local_ml_shadow"].pop("prediction_score_policy_version")

    report = summarize_specialist_shadow_evaluation([current, legacy])
    local = next(row for row in report["models"] if row["tool"] == "local_ml")
    entry = local["entry_return_evaluation"]
    assert local["direction_count"] == 2
    assert entry["direction_count"] == 1
    assert entry["prediction_score_policy_version"] == PREDICTION_SCORE_POLICY_VERSION
    assert entry["avg_shadow_return_after_all_cost_pct"] < -0.5
    assert entry["gross_profit_return_pct"] == 0.0


def test_local_ml_other_horizon_cannot_duplicate_primary_entry_evidence():
    primary = _local_ml_row()
    other = _local_ml_row(index=2)
    other.horizon_minutes = 30
    other.feature_snapshot["local_ml_shadow"]["predictions"].append({
        "horizon_minutes": 30,
        "best_side": "long",
        "long": {"objective_net_return_pct": 0.5},
        "short": {"objective_net_return_pct": 0.25},
    })
    report = summarize_specialist_shadow_evaluation([primary, other])
    local = next(item for item in report["models"] if item["tool"] == "local_ml")
    assert local["direction_count"] == 2
    assert local["entry_return_evaluation"]["direction_count"] == 1


@pytest.mark.parametrize("profitable", [False, True])
def test_frozen_local_ml_entry_outcomes_reach_only_the_matching_model_permission(profitable):
    sign = 1 if profitable else -1
    rows = [_local_ml_row(realized=sign * 0.5)]
    for index in range(2, 12):
        hold = _local_ml_row(index=index, realized=-sign * 5.0)
        hold.feature_snapshot["local_ml_shadow"]["predictions"][0]["long"]["objective_net_return_pct"] = -0.1
        rows.append(hold)
    specialist = summarize_specialist_shadow_evaluation(rows)
    weights = ContinuousModelWeightPolicy().build(
        execution_mode="paper", market_regime={}, health_report={},
        specialist_report=specialist, contribution_performance={},
    )
    strategy = {"execution_mode": "paper", "continuous_model_weights": weights}
    permission = _paper_quant_quality_permissions(strategy, {"model_version": "local-ml-v7"})["local_ml"]
    assert permission["paper_execution_permission"] is profitable
    assert permission["paper_execution_evidence"]["sample_count"] == 1
    assert _paper_quant_quality_permissions(strategy, {"model_version": "other-version"})["local_ml"][
        "paper_execution_permission"
    ] is False


def test_specialist_shadow_evaluation_reports_timesfm_challenger_metrics() -> None:
    report = summarize_specialist_shadow_evaluation(
        [
            _row(index=index, realized=0.20 + index / 100)
            for index in range(1, 35)
        ],
        authoritative_trade_samples=[
            _authoritative_sample(
                index,
                pnl_ratio_pct=-0.01 if index == 17 else 0.25,
            )
            for index in range(1, 35)
        ],
    )

    assert report["ok"] is True
    assert report["completed_count"] == 34
    assert report["eligible_shadow_count"] == 34
    models = {row["model"]: row for row in report["models"]}
    assert set(models) == {"google/timesfm-2.5-200m-pytorch", "amazon/chronos-2"}
    model = models["google/timesfm-2.5-200m-pytorch"]
    assert model["actual_inference_count"] == 34
    assert model["direction_count"] == 34
    assert model["direction_hit_rate"] == 1.0
    assert model["avg_realized_return_pct"] > 0.2
    assert model["profit_factor"] is None
    assert model["authoritative_direction_aligned_count"] == 34
    assert model["authoritative_profit_factor"] > 1.0
    assert model["walk_forward"]["status"] == "stable"
    assert model["market_regime_stability"]["status"] == "stable"
    assert model["rolling_distribution"]["status"] == "stable"
    assert model["promotion_ready"] is True
    assert model["promotion_blockers"] == []
    assert model["blockers"] == []
    assert model["blocked_reasons"] == []
    assert model["authoritative_return_lcb_pct"] > 0.0
    assert report["summary"]["promotion_ready_count"] == 2
    assert report["authoritative_eligible_count"] == 34
    assert report["promotion_gate"]["requires_at_least_one_promotion_ready_model"] is True


def test_shadow_return_lcb_running_moments_match_batch_calculation() -> None:
    values = [0.31, -0.08, 0.12, -0.22, 0.07, 0.19]
    metric = _empty_metric("time_series_prediction", "timesfm")
    for value in values:
        _remember_shadow_event(metric, {"net_return_after_all_cost_pct": value})

    lcb = _mean_lcb_from_moments(
        metric["shadow_return_count"],
        metric["shadow_return_mean_pct"],
        metric["shadow_return_m2_pct"],
    )

    assert lcb == pytest.approx(_mean_lcb(values))
    assert metric["shadow_event_count"] == len(values)
    assert len(metric["shadow_event_samples"]) == len(values)


def test_specialist_shadow_report_supports_incremental_row_accumulation() -> None:
    rows = [
        _row(index=index, realized=0.2 + index / 100)
        for index in range(1, 35)
    ]
    authoritative_samples = [
        _authoritative_sample(index, pnl_ratio_pct=-0.01 if index == 17 else 0.25)
        for index in range(1, 35)
    ]
    expected = summarize_specialist_shadow_evaluation(
        rows,
        authoritative_trade_samples=authoritative_samples,
    )
    accumulator = SpecialistShadowEvaluationAccumulator()
    for row in rows:
        accumulator.add_shadow_row(row)
    actual = summarize_specialist_shadow_evaluation(
        (),
        authoritative_trade_samples=authoritative_samples,
        accumulator=accumulator,
    )

    for report in (expected, actual):
        report.pop("generated_at")
        for model in report["models"]:
            for shadow_event in model["shadow_events"]:
                shadow_event["execution_cost"]["policy_provenance"].pop("generated_at")
                shadow_event["execution_cost"]["funding_cost"]["policy_provenance"].pop(
                    "generated_at"
                )
    assert actual == expected


def test_specialist_shadow_evaluation_blocks_undefined_authoritative_profit_factor() -> None:
    report = summarize_specialist_shadow_evaluation(
        [_row(index=index, realized=0.3) for index in range(1, 35)],
        authoritative_trade_samples=[
            _authoritative_sample(index, pnl_ratio_pct=0.25)
            for index in range(1, 35)
        ],
    )
    model = report["models"][0]

    assert model["authoritative_profit_factor"] is None
    assert model["promotion_ready"] is False
    assert "authoritative_profit_factor_undefined" in model["promotion_blockers"]


def test_specialist_shadow_evaluation_blocks_false_signal_losses() -> None:
    rows = [
        _row(
            best_action="short",
            predicted_side="long",
            realized=-0.25,
            expected=0.35,
            index=index,
        )
        for index in range(1, 35)
    ]

    report = summarize_specialist_shadow_evaluation(rows)
    model = report["models"][0]

    assert model["promotion_ready"] is False
    assert model["false_signal_count"] == 34
    assert model["tail_loss_count"] == 0
    assert model["tail_loss_symbols"] == []
    assert model["worst_samples"][0]["symbol"] == "BTC/USDT"
    assert model["worst_samples"][0]["predicted_side"] == "long"
    assert model["worst_samples"][0]["actual_best_side"] == "short"
    assert model["worst_samples"][0]["actual_return_pct"] < -0.25
    assert model["shadow_return_lcb_pct"] < 0.0
    assert "authoritative_return_distribution_missing" in model["promotion_blockers"]
    assert "authoritative_fee_after_return_lcb_not_positive" in model["promotion_blockers"]
    assert model["blocked_reasons"] == model["promotion_blockers"]
    assert model["blocked_reason_counts"]["authoritative_return_distribution_missing"] == 1
    assert report["summary"]["blocked_count"] == 2
    assert {
        item["reason"] for item in report["summary"]["top_blocked_reasons"]
    } >= {
        "authoritative_return_distribution_missing",
        "authoritative_fee_after_return_lcb_not_positive",
    }


def test_specialist_shadow_evaluation_quarantines_legacy_mixed_timeseries() -> None:
    row = _row(index=1)
    tool = row.feature_snapshot["local_ai_tools_shadow"]["time_series_prediction"]
    tool["professional_model_shadow"] = {
        "actual_inference": True,
        "shadow_result": {
            "model": "chronos-2-shadow-primary",
            "actual_inference": True,
            "expected_return_pct": 0.22,
            "best_side": "long",
            "sequence_length": 4,
        },
    }

    report = summarize_specialist_shadow_evaluation([row])
    model = report["models"][0]

    assert model["model"] == "chronos-2-shadow-primary"
    assert model["legacy_mixed_shadow_count"] == 1
    assert model["legacy_quarantined_count"] == 1
    assert model["actual_inference_count"] == 0
    assert model["direction_count"] == 0
    assert model["tail_loss_count"] == 0
    assert "authoritative_return_distribution_missing" in model["promotion_blockers"]
    assert "authoritative_fee_after_return_lcb_not_positive" in model["promotion_blockers"]
    assert "legacy_mixed_shadow_result_not_promotable" not in model["promotion_blockers"]


def test_timeseries_sequence_length_is_not_a_fixed_promotion_gate() -> None:
    row = _row(index=2)
    tool = row.feature_snapshot["local_ai_tools_shadow"]["time_series_prediction"]
    tool["professional_model_shadow"] = {
        "actual_inference": True,
        "primary_shadow_result": {
            "model": "chronos-2-shadow-primary",
            "actual_inference": True,
            "expected_return_pct": 0.22,
            "best_side": "long",
            "sequence_length": 4,
        },
    }

    model = summarize_specialist_shadow_evaluation([row])["models"][0]

    assert model["actual_inference_count"] == 1
    assert model["legacy_quarantined_count"] == 0
    assert all("sequence" not in reason for reason in model["promotion_blockers"])


def test_specialist_shadow_evaluation_skips_rows_without_shadow_evidence() -> None:
    now = datetime.now(UTC)
    report = summarize_specialist_shadow_evaluation(
        [
            SimpleNamespace(
                id=1,
                status="completed",
                symbol="BTC/USDT",
                best_action="long",
                long_return_pct=0.1,
                short_return_pct=-0.1,
                feature_snapshot={},
                due_at=now,
                created_at=now,
            ),
            SimpleNamespace(
                id=2,
                status="pending",
                symbol="ETH/USDT",
                feature_snapshot={},
                due_at=now,
                created_at=now,
            ),
        ]
    )

    assert report["completed_count"] == 1
    assert report["eligible_shadow_count"] == 0
    assert report["model_count"] == 0
    assert report["skipped_reasons"]["missing_local_ai_tools_shadow"] == 1
    assert report["skipped_reasons"]["not_completed"] == 1


def test_specialist_shadow_evaluation_uses_per_event_execution_cost() -> None:
    report = summarize_specialist_shadow_evaluation([_row(realized=0.10)])
    model = next(
        row
        for row in report["models"]
        if row["model"] == "google/timesfm-2.5-200m-pytorch"
    )

    event = model["shadow_events"][0]
    cost = event["execution_cost"]
    expected = (
        0.10
        - cost["fee_pct"]
        - cost["slippage_pct"]
        - cost["funding_drag_pct"]
    )
    assert model["avg_shadow_return_after_all_cost_pct"] == round(expected, 6)
    assert event["gross_return_pct"] == 0.10
    assert cost["production_eligible"] is True


def test_specialist_shadow_evaluation_records_per_model_fallbacks() -> None:
    report = summarize_specialist_shadow_evaluation([_row(actual_inference=False)])
    models = {row["model"]: row for row in report["models"]}

    assert report["eligible_shadow_count"] == 1
    assert models["google/timesfm-2.5-200m-pytorch"]["fallback_count"] == 1
    assert models["google/timesfm-2.5-200m-pytorch"]["shadow_fallback_count"] == 1
    assert models["amazon/chronos-2"]["fallback_count"] == 1
    assert models["google/timesfm-2.5-200m-pytorch"]["actual_inference_count"] == 0


def test_sentiment_without_per_model_predictions_never_uses_calibrator_identity() -> None:
    row = _row(tool="sentiment_analysis", actual_inference=True)
    tool = row.feature_snapshot["local_ai_tools_shadow"]["sentiment_analysis"]
    tool["model"] = "local-sentiment-trained-v2"
    tool["professional_model_shadow"] = {
        "actual_inference": True,
        "primary_model": "ProsusAI/finbert",
        "challenger_model": "yiyanghkust/finbert-tone",
    }

    report = summarize_specialist_shadow_evaluation([row])
    models = {item["model"]: item for item in report["models"]}

    assert set(models) == {"ProsusAI/finbert", "yiyanghkust/finbert-tone"}
    assert all(item["fallback_count"] == 1 for item in models.values())
    assert "local-sentiment-trained-v2" not in models


def test_specialist_shadow_only_evidence_can_never_promote() -> None:
    report = summarize_specialist_shadow_evaluation(
        [_row(index=index, realized=0.4) for index in range(1, 35)]
    )
    model = report["models"][0]

    assert model["promotion_ready"] is False
    assert "authoritative_return_distribution_missing" in model["promotion_blockers"]
    assert "authoritative_fee_after_return_lcb_not_positive" in model["promotion_blockers"]


def test_specialist_report_bounds_event_rows_without_changing_full_window_metrics() -> None:
    report = summarize_specialist_shadow_evaluation(
        [_row(index=index, realized=0.4) for index in range(1, 35)]
    )
    model = next(
        row
        for row in report["models"]
        if row["model"] == "google/timesfm-2.5-200m-pytorch"
    )

    assert model["shadow_event_count"] == 34
    assert len(model["shadow_events"]) == 8
    assert model["shadow_events_truncated"] is True
    assert model["direction_count"] == 34


def test_authoritative_return_is_not_assigned_to_opposite_prediction() -> None:
    report = summarize_specialist_shadow_evaluation(
        [_row(index=index, realized=0.4) for index in range(1, 35)],
        authoritative_trade_samples=[
            _authoritative_sample(index, predicted_side="short", position_side="long")
            for index in range(1, 35)
        ],
    )
    model = next(
        row
        for row in report["models"]
        if row["model"] == "google/timesfm-2.5-200m-pytorch"
    )

    assert model["authoritative_actual_inference_count"] == 34
    assert model["authoritative_direction_aligned_count"] == 0
    assert model["authoritative_direction_mismatch_count"] == 34
    assert model["authoritative_events"] == []
    assert all(
        evidence["label_reason"] == "prediction_not_aligned_with_observed_position"
        for evidence in model["authoritative_evidence"]
    )
    assert model["promotion_ready"] is False


@pytest.mark.asyncio
async def test_specialist_report_mode_filter_uses_only_selected_execution_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    await _use_temp_db(monkeypatch, tmp_path)
    now = datetime.now(UTC)
    async with get_session_ctx() as session:
        session.add_all(
            [
                ShadowBacktest(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="BTC/USDT",
                    status="completed",
                    due_at=now,
                    long_return_pct=0.2,
                    short_return_pct=-0.2,
                    created_at=now,
                ),
                ShadowBacktest(
                    model_name="ensemble_trader",
                    execution_mode="live",
                    symbol="ETH/USDT",
                    status="completed",
                    due_at=now,
                    long_return_pct=-0.3,
                    short_return_pct=0.3,
                    created_at=now,
                ),
            ]
        )

    try:
        paper = await SpecialistShadowEvaluationService().report(
            hours=24,
            mode="paper",
        )
        live = await SpecialistShadowEvaluationService().report(
            hours=24,
            mode="live",
        )
    finally:
        await close_db()

    assert paper["execution_mode"] == "paper"
    assert live["execution_mode"] == "live"
    assert paper["completed_count"] == 1
    assert live["completed_count"] == 1
    assert paper["query_policy"]["ordered_by_training_index"] is True
    assert "ordered_by_primary_key" not in paper["query_policy"]
    assert paper["query_policy"]["training_feature_snapshot_column"] is True
    assert paper["query_policy"]["keyset_pagination"] is True
    assert paper["query_policy"]["row_batch_size"] == 1000


@pytest.mark.asyncio
async def test_specialist_report_uses_composite_index_range_for_keyset_pages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    await _use_temp_db(monkeypatch, tmp_path)
    monkeypatch.setattr(specialist_evaluation, "SHADOW_QUERY_BATCH_SIZE", 1)
    original_summarize = specialist_evaluation.summarize_specialist_shadow_evaluation
    summary_calls = 0

    def assert_streamed_summary(rows, **kwargs):
        nonlocal summary_calls
        summary_calls += 1
        assert list(rows) == []
        assert isinstance(kwargs.get("accumulator"), SpecialistShadowEvaluationAccumulator)
        return original_summarize((), **kwargs)

    monkeypatch.setattr(
        specialist_evaluation,
        "summarize_specialist_shadow_evaluation",
        assert_streamed_summary,
    )
    now = datetime.now(UTC)
    async with get_session_ctx() as session:
        session.add_all(
            [
                ShadowBacktest(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol=f"BTC/USDT-{index}",
                    status="completed",
                    due_at=now,
                    long_return_pct=0.2,
                    short_return_pct=-0.2,
                    created_at=now - timedelta(minutes=index),
                )
                for index in range(2)
            ]
        )

    statements: list[str] = []
    engine = await get_engine()

    def capture_statement(
        _conn: object,
        _cursor: object,
        statement: str,
        _params: object,
        _context: object,
        _many: bool,
    ) -> None:
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture_statement)
    try:
        report = await SpecialistShadowEvaluationService().report(hours=24, mode="paper")
    finally:
        event.remove(
            engine.sync_engine,
            "before_cursor_execute",
            capture_statement,
        )
        await close_db()

    select_statements = [
        statement
        for statement in statements
        if "FROM shadow_backtests" in statement
    ]
    assert report["completed_count"] == 2
    assert summary_calls == 1
    assert report["query_policy"]["query_count"] == 3
    assert len(select_statements) == 3
    assert "(shadow_backtests.created_at, shadow_backtests.id) < (?, ?)" in select_statements[1]
