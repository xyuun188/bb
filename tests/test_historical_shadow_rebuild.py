from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from core.training_contracts import SHADOW_LABEL_VERSION, shadow_label_contract_reasons
from services.historical_shadow_rebuild import (
    HISTORICAL_SHADOW_REBUILD_VERSION,
    HISTORICAL_SHADOW_SOURCE,
    build_historical_shadow_sample,
)
from services.training_data_quality import annotate_sample


def _decision() -> SimpleNamespace:
    return SimpleNamespace(
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
        feature_snapshot={"current_price": 100.0, "rsi_14": 55.0},
    )


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
    assert features["training_market_fact_contract"]["source"] == (
        HISTORICAL_SHADOW_SOURCE
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
