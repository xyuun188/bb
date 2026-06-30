from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from scripts import run_specialist_shadow_evaluation as runner
from services.specialist_shadow_evaluation import summarize_specialist_shadow_evaluation


def test_specialist_shadow_evaluation_script_imports_online_runtime_bootstrap() -> None:
    source = runner.ROOT.joinpath("scripts", "run_specialist_shadow_evaluation.py").read_text(
        encoding="utf-8"
    )

    assert "from scripts.runtime_env_bootstrap import" in source
    assert "load_runtime_env_files(project_root=ROOT)" in source
    assert "drop_privileges_to_runtime_user_if_needed(project_root=ROOT)" in source


def test_specialist_shadow_evaluation_default_report_dir_matches_phase3_readers() -> None:
    assert runner.DEFAULT_REPORT_DIR == "phase3"


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
        status="completed",
        symbol=symbol,
        best_action=best_action,
        long_return_pct=long_return,
        short_return_pct=short_return,
        feature_snapshot={
            "symbol": symbol,
            "local_ai_tools_shadow": local_shadow,
        },
        due_at=now,
        created_at=now,
    )


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
        ]
    )

    assert report["ok"] is True
    assert report["completed_count"] == 34
    assert report["eligible_shadow_count"] == 34
    models = {row["model"]: row for row in report["models"]}
    assert set(models) == {"chronos-2-shadow-primary", "timesfm-2.5-shadow-challenger"}
    model = models["timesfm-2.5-shadow-challenger"]
    assert model["actual_inference_count"] == 34
    assert model["direction_count"] == 34
    assert model["direction_hit_rate"] == 1.0
    assert model["avg_realized_return_pct"] > 0.2
    assert model["promotion_ready"] is True
    assert model["promotion_blockers"] == []
    assert model["blockers"] == []
    assert model["blocked_reasons"] == []
    assert model["promotion_gate"]["minimum_actual_inference_samples"] == 30
    assert report["summary"]["promotion_ready_count"] == 2
    assert report["promotion_gate"]["requires_at_least_one_promotion_ready_model"] is True


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
    assert model["tail_loss_count"] == 34
    assert model["tail_loss_symbols"] == [{"symbol": "BTC/USDT", "count": 34}]
    assert model["worst_samples"][0]["symbol"] == "BTC/USDT"
    assert model["worst_samples"][0]["predicted_side"] == "long"
    assert model["worst_samples"][0]["actual_best_side"] == "short"
    assert model["worst_samples"][0]["actual_return_pct"] == -0.25
    assert model["promotion_gate"]["tail_loss_count"] == 34
    assert "direction_hit_rate_below_floor" in model["promotion_blockers"]
    assert "avg_realized_return_below_floor" in model["promotion_blockers"]
    assert "false_signal_loss_exceeds_floor" in model["promotion_blockers"]
    assert model["blocked_reasons"] == model["promotion_blockers"]
    assert model["blocked_reason_counts"]["false_signal_loss_exceeds_floor"] == 1
    assert report["summary"]["blocked_count"] == 2
    assert {
        item["reason"] for item in report["summary"]["top_blocked_reasons"]
    } >= {
        "direction_hit_rate_below_floor",
        "avg_realized_return_below_floor",
        "false_signal_loss_exceeds_floor",
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
    assert model["legacy_sequence_too_short_count"] == 1
    assert model["sequence_too_short_count"] == 0
    assert model["actual_inference_count"] == 0
    assert model["direction_count"] == 0
    assert model["tail_loss_count"] == 0
    assert model["promotion_blockers"] == ["specialist_shadow_sample_floor_not_met"]
    assert "legacy_mixed_shadow_result_not_promotable" not in model["promotion_blockers"]
    assert "timeseries_sequence_too_short_for_promotion" not in model["promotion_blockers"]


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
