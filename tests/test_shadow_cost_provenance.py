from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from services.execution_cost_model import execution_cost_estimate
from services.profit_supervision import (
    COUNTERFACTUAL_EXECUTION_COST_TASK,
    MARKET_OPPORTUNITY_TASK,
    build_profit_supervision_contract,
    shadow_fee_after_return_labels,
)
from services.shadow_backtest_service import shadow_fee_after_outcome


@pytest.fixture
def features():
    return {
        "bid": 99.99,
        "ask": 100.01,
        "orderbook_bid_depth": 100_000.0,
        "orderbook_ask_depth": 100_000.0,
        "taker_fee_rate": 0.0004,
        "funding_rate": 0.0001,
        "funding_data_available": True,
        "funding_interval_minutes": 480,
    }


@pytest.mark.parametrize("provenance", [
    {"historical_market_path_only": True},
    {"training_market_fact_contract": {"status": "historical_ohlcv_only"}},
    {"training_market_fact_contract": {"executable_market_fact_verified": False}},
    {"market_fact_contract": {"assertions": {
        "executable_market_fact_verified": False,
    }}},
    {
        "training_market_fact_contract": {"executable_market_fact_verified": True},
        "market_fact_contract": {"assertions": {
            "executable_market_fact_verified": False,
        }},
    },
])
def test_market_only_evidence_cannot_supervise_cost_despite_copied_numbers(
    features, provenance,
):
    features.update(provenance)
    sample = {
        "features": features, "horizon_minutes": 5, "sample_weight": 1.0,
        "long_return_pct": 0.8, "short_return_pct": -0.8,
    }
    estimate = execution_cost_estimate(features)
    assert estimate.fee_pct > 0
    assert estimate.spread_pct > 0
    assert estimate.production_eligible is False
    assert estimate.policy_provenance["fallback_reason"] in {
        "historical_market_path_has_no_executable_cost_evidence",
        "market_fact_not_executable",
    }
    tasks = build_profit_supervision_contract(sample, kind="shadow")["tasks"]
    assert tasks[MARKET_OPPORTUNITY_TASK]["eligible"] is True
    assert tasks[COUNTERFACTUAL_EXECUTION_COST_TASK]["eligible"] is False
    assert tasks[COUNTERFACTUAL_EXECUTION_COST_TASK]["long_total_cost_pct"] is None
    assert shadow_fee_after_return_labels(sample)["complete"] is False
    outcome = shadow_fee_after_outcome(
        SimpleNamespace(feature_snapshot=features, horizon_minutes=5),
        long_return=0.008, short_return=-0.008,
    )
    assert outcome["cost_complete"] is False
    assert outcome["long_net_return_after_all_cost_pct"] is None
    assert outcome["short_net_return_after_all_cost_pct"] is None
    assert all(
        row["long_fee_after_margin_return_pct"] is None
        and row["short_fee_after_margin_return_pct"] is None
        for row in outcome["leverage_counterfactuals"]
    )


def test_uncontradicted_executable_snapshot_still_has_cost_supervision(features):
    features["training_market_fact_contract"] = {
        "executable_market_fact_verified": True,
    }
    sample = {
        "features": features, "horizon_minutes": 5, "sample_weight": 1.0,
        "long_return_pct": 0.8, "short_return_pct": -0.8,
    }
    assert execution_cost_estimate(features).production_eligible is True
    tasks = build_profit_supervision_contract(sample, kind="shadow")["tasks"]
    assert tasks[COUNTERFACTUAL_EXECUTION_COST_TASK]["eligible"] is True
    assert shadow_fee_after_return_labels(sample)["complete"] is True


def test_rebuilt_historical_row_without_frozen_features_is_excluded_from_net_training(
    features,
):
    from services.historical_shadow_rebuild import build_historical_shadow_sample
    from services.ml_signal_service import build_training_frame
    from services.training_data_quality import annotate_sample

    start = datetime(2026, 9, 1, tzinfo=UTC)
    decision = SimpleNamespace(
        id=7, symbol="BTC/USDT", created_at=start + timedelta(seconds=25),
        feature_snapshot={**features, "current_price": 100.0},
        action="hold", confidence=0.7, stop_loss_pct=0.02, take_profit_pct=0.03,
    )
    bars = [{
        "open_time": start + timedelta(minutes=index),
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.1, "volume": 1000.0,
    } for index in range(6)]
    payload = build_historical_shadow_sample(
        decision, horizon_minutes=5, bars=bars, shadow_backtest_id=91,
    )
    assert payload is not None
    sample = {
        **payload, "id": 91, "features": payload["feature_snapshot"],
        "label_timestamp": payload["due_at"],
    }
    annotated = annotate_sample(sample, "shadow")
    assert annotated["exclude_from_training"] is True
    assert "feature_availability:feature_availability_contract_missing" in (
        annotated["quality_reasons"]
    )
    tasks = annotated["profit_supervision"]["tasks"]
    assert tasks[MARKET_OPPORTUNITY_TASK]["eligible"] is False
    assert tasks[COUNTERFACTUAL_EXECUTION_COST_TASK]["eligible"] is False
    assert annotated["fee_after_label_complete"] is False
    assert build_training_frame([SimpleNamespace(id=91, **payload)]).empty


def test_old_supervision_version_cannot_authorize_model_influence():
    from services.ml_signal_service import _side_influence_status

    result = _side_influence_status(
        {"profit_supervision_version": "2026-07-24.separated-profit-supervision.v2"},
        "long",
    )
    assert "artifact separated profit supervision contract is missing" in result[
        "reasons"
    ]
    assert result["enabled"] is False
    assert result["influence_weight"] == 0
