from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.learning import ShadowBacktest
from scripts import run_specialist_shadow_evaluation as runner
from services.specialist_shadow_evaluation import (
    SpecialistShadowEvaluationService,
    _regime_stability_report,
    _rolling_distribution_report,
    _walk_forward_report,
    summarize_specialist_shadow_evaluation,
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
    assert "_load_authoritative_trade_samples" in source
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
            "funding_interval_hours": 8.0,
            "local_ai_tools_shadow": local_shadow,
        },
        due_at=now,
        created_at=now,
    )


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
        "authoritative_pnl_ratio_pct": pnl_ratio_pct,
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
