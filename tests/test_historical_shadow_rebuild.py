from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from core.training_contracts import SHADOW_LABEL_VERSION, shadow_label_contract_reasons
from services.historical_shadow_rebuild import (
    HISTORICAL_SHADOW_REBUILD_VERSION,
    HISTORICAL_SHADOW_SOURCE,
    build_historical_shadow_sample,
)
from services.training_data_quality import annotate_sample
from tests.feature_availability_fixtures import seal_synthetic_feature_availability


def _decision() -> SimpleNamespace:
    decision = SimpleNamespace(
        id=17,
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action="hold",
        confidence=0.7,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        is_paper=True,
        created_at=datetime(2026, 9, 1, 0, 0, 25, tzinfo=UTC),
        raw_llm_response={"large": "not copied"},
        feature_snapshot={
            "symbol": "BTC/USDT",
            "current_price": 100.0,
            "rsi_14": 55.0,
            "market_data_quality": {
                "code": "orderbook_depth_invalid",
                "exclude_from_training": True,
            },
            "training_quality_reason": (
                "market_data_quality:orderbook_depth_invalid"
            ),
        },
    )
    seal_synthetic_feature_availability(
        decision.feature_snapshot,
        decision_at=decision.created_at,
    )
    return decision


def _bars(count: int) -> list[dict[str, object]]:
    start = datetime(2026, 9, 1, 0, 0, tzinfo=UTC)
    return [
        {
            "open_time": start + timedelta(minutes=index),
            "open": 100.0 + index,
            "high": 101.0 + index,
            "low": 99.0 + index,
            "close": 100.0 + index,
            "volume": 1000.0 + index,
        }
        for index in range(count)
    ]


def test_historical_shadow_sample_requires_complete_contiguous_window() -> None:
    assert build_historical_shadow_sample(
        _decision(), horizon_minutes=5, bars=_bars(5)
    ) is None


def test_historical_shadow_sample_is_market_only_and_has_real_identity() -> None:
    sample = build_historical_shadow_sample(
        _decision(),
        horizon_minutes=5,
        bars=_bars(6),
        shadow_backtest_id=91,
    )

    assert sample is not None
    assert sample["label_version"] == SHADOW_LABEL_VERSION
    assert sample["raw_llm_response"] == {}
    features = sample["feature_snapshot"]
    assert features["historical_market_path_only"] is True
    assert features["historical_shadow_rebuild_version"] == (
        HISTORICAL_SHADOW_REBUILD_VERSION
    )
    assert features["historical_market_path_entry_fact"]["role"] == "entry"
    assert features["market_fact"]["received_at"] == "2026-09-01T00:00:17+00:00"
    assert features["training_market_fact_contract"]["source"] == (
        HISTORICAL_SHADOW_SOURCE
    )
    availability = features["training_market_fact_contract"]
    assert availability["scheduled_label_at"] == "2026-09-01T00:05:25+00:00"
    assert availability["result_source_available_at"] == "2026-09-01T00:06:00+00:00"
    assert availability["label_maturity_lag_seconds"] == 35.0
    assert availability["delivery_latency_observed"] is False
    assert sample["due_at"] == datetime(2026, 9, 1, 0, 5, 25, tzinfo=UTC)
    assert features["training_label_contract"]["label_timestamp"] == (
        "2026-09-01T00:06:00+00:00"
    )
    label = features["training_label_contract"]
    assert label["shadow_backtest_id"] == 91
    assert not shadow_label_contract_reasons(
        label,
        decision_id=17,
        horizon_minutes=5,
        label_version=SHADOW_LABEL_VERSION,
    )

    annotated = annotate_sample(
        {
            "id": 91,
            "decision_id": 17,
            "label_version": SHADOW_LABEL_VERSION,
            "symbol": sample["symbol"],
            "decision_action": sample["decision_action"],
            "decision_confidence": sample["decision_confidence"],
            "horizon_minutes": sample["horizon_minutes"],
            "features": features,
            "long_return_pct": sample["long_return_pct"],
            "short_return_pct": sample["short_return_pct"],
            "best_action": sample["best_action"],
            "label_timestamp": sample["due_at"].isoformat(),
        },
        "shadow",
    )
    assert annotated["exclude_from_training"] is False
    assert annotated["data_quality_status"] == "downweighted"
    assert "historical_market_path_only" in annotated["quality_reasons"]
    tasks = annotated["profit_supervision"]["tasks"]
    assert tasks["market_opportunity_distribution"]["eligible"] is True
    assert tasks["execution_cost_and_slippage_distribution"]["eligible"] is False


def test_historical_shadow_sample_does_not_invent_missing_feature_contract() -> None:
    decision = _decision()
    decision.feature_snapshot.pop("feature_availability_contract")
    sample = build_historical_shadow_sample(
        decision,
        horizon_minutes=5,
        bars=_bars(6),
        shadow_backtest_id=95,
    )

    assert sample is not None
    annotated = annotate_sample(
        {
            "id": 95,
            "decision_id": sample["decision_id"],
            "label_version": sample["label_version"],
            "symbol": sample["symbol"],
            "decision_action": sample["decision_action"],
            "decision_confidence": sample["decision_confidence"],
            "horizon_minutes": sample["horizon_minutes"],
            "features": sample["feature_snapshot"],
            "long_return_pct": sample["long_return_pct"],
            "short_return_pct": sample["short_return_pct"],
            "best_action": sample["best_action"],
            "label_timestamp": sample["due_at"].isoformat(),
        },
        "shadow",
    )

    assert annotated["exclude_from_training"] is True
    assert "feature_availability:feature_availability_contract_missing" in (
        annotated["quality_reasons"]
    )


def test_historical_shadow_invalid_path_contract_remains_excluded() -> None:
    sample = build_historical_shadow_sample(
        _decision(),
        horizon_minutes=5,
        bars=_bars(6),
        shadow_backtest_id=92,
    )
    assert sample is not None
    features = sample["feature_snapshot"]
    features["historical_shadow_rebuild_version"] = "stale"

    annotated = annotate_sample(
        {
            "id": 92,
            "decision_id": sample["decision_id"],
            "label_version": sample["label_version"],
            "symbol": sample["symbol"],
            "decision_action": sample["decision_action"],
            "decision_confidence": sample["decision_confidence"],
            "horizon_minutes": sample["horizon_minutes"],
            "features": features,
            "long_return_pct": sample["long_return_pct"],
            "short_return_pct": sample["short_return_pct"],
            "best_action": sample["best_action"],
            "label_timestamp": sample["due_at"].isoformat(),
        },
        "shadow",
    )

    assert annotated["exclude_from_training"] is True
    assert (
        "market_fact_contract:historical_shadow_rebuild_version_missing_or_stale"
        in annotated["quality_reasons"]
    )


def test_historical_shadow_keeps_one_second_close_maturity_lag() -> None:
    decision = _decision()
    decision.created_at = datetime(2026, 9, 1, 0, 0, 59, tzinfo=UTC)
    bars = _bars(6)
    sample = build_historical_shadow_sample(
        decision, horizon_minutes=5, bars=bars, shadow_backtest_id=93
    )

    assert sample is not None
    availability = sample["feature_snapshot"]["training_market_fact_contract"]
    assert availability["scheduled_label_at"] == "2026-09-01T00:05:59+00:00"
    assert availability["result_source_available_at"] == "2026-09-01T00:06:00+00:00"
    assert availability["label_maturity_lag_seconds"] == 1.0


@pytest.mark.parametrize("price", [None, 0, -1, float("nan")])
def test_historical_shadow_never_borrows_a_future_close_as_entry(price) -> None:
    decision = _decision()
    decision.feature_snapshot["current_price"] = price
    assert build_historical_shadow_sample(
        decision, horizon_minutes=5, bars=_bars(6)
    ) is None


def test_historical_shadow_keeps_original_entry_fact_in_decision_features() -> None:
    decision = _decision()
    decision.feature_snapshot["market_fact"] = {
        "fact_id": "original", "prices": {"last": 100.0},
    }
    sample = build_historical_shadow_sample(
        decision, horizon_minutes=5, bars=_bars(6), shadow_backtest_id=94,
    )
    assert sample is not None
    assert sample["feature_snapshot"]["market_fact"] == decision.feature_snapshot["market_fact"]


@pytest.mark.parametrize("lag", [-1, 61])
def test_historical_shadow_quality_rejects_early_and_late_labels(lag) -> None:
    sample = build_historical_shadow_sample(
        _decision(), horizon_minutes=5, bars=_bars(6), shadow_backtest_id=94,
    )
    assert sample is not None
    features = sample["feature_snapshot"]
    market = features["training_market_fact_contract"]
    market["result_source_available_at"] = (
        sample["due_at"] + timedelta(seconds=lag)
    ).isoformat()
    assessment = annotate_sample({
        "id": 94, "decision_id": 17, "horizon_minutes": 5,
        "label_version": SHADOW_LABEL_VERSION, "features": features,
        "long_return_pct": sample["long_return_pct"],
        "short_return_pct": sample["short_return_pct"],
        "label_timestamp": sample["due_at"],
    }, "shadow")
    assert assessment["exclude_from_training"] is True
    assert any("historical_label_availab" in reason
               for reason in assessment["quality_reasons"])
