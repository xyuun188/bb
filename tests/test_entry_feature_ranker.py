from types import SimpleNamespace

import pytest

from services.entry_feature_ranker import EntryFeatureRankerPolicy


def _ranker() -> EntryFeatureRankerPolicy:
    return EntryFeatureRankerPolicy(
        suspicious_symbol_reason=lambda _symbol: None,
        major_symbols=frozenset({"BTC/USDT", "ETH/USDT"}),
    )


def _feature(symbol: str, **overrides):
    values = {
        "symbol": symbol,
        "current_price": 100.0,
        "volume_24h": 20_000.0,
        "volume_ratio": 0.40,
        "adx_14": 18.0,
        "returns_1": 0.002,
        "returns_5": 0.004,
        "returns_20": 0.006,
        "volatility_20": 0.02,
        "change_24h_pct": 2.0,
        "bb_pct": 0.5,
        "price_vs_sma20": 0.01,
        "price_vs_sma50": 0.02,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _rank(features: dict[str, object], limit: int):
    return _ranker().rank(features, limit)


def test_entry_feature_policy_moves_with_market_cross_section() -> None:
    def policy_for(volume_ratios: list[float]) -> dict:
        result = _rank(
            {
                f"S{index}/USDT": _feature(f"S{index}/USDT", volume_ratio=value)
                for index, value in enumerate(volume_ratios)
            },
            len(volume_ratios),
        )
        return result.diagnostics["dynamic_policy"]["values"]

    calm = policy_for([0.1, 0.2, 0.3, 0.4])
    active = policy_for([1.1, 1.2, 1.3, 1.4])

    assert calm["analysis_volume_floor"]["value"] != active["analysis_volume_floor"]["value"]
    assert calm["tradable_volume_floor"]["value"] != active["tradable_volume_floor"]["value"]
    for name, value in active.items():
        assert value["source"].startswith("empirical_order_statistics")
        assert value["observation_window"] == "current_market_feature_cross_section"
        assert value["generated_at"]
        assert value["strategy_version"]
        if name.endswith("volume_floor"):
            assert value["sample_count"] == 4


def test_ranker_has_no_hold_or_rotation_penalty_contract() -> None:
    result = _rank(
        {
            "SOL/USDT": _feature("SOL/USDT", returns_5=0.010),
            "LINK/USDT": _feature("LINK/USDT", returns_5=0.006),
        },
        1,
    )

    assert list(result.selected) == ["SOL/USDT"]
    diagnostic = result.diagnostics["symbols"][0]
    assert "recent_hold_penalty" not in diagnostic
    assert "rotation_penalty" not in diagnostic


def test_ranker_uses_explicit_okx_swap_notional() -> None:
    feature = _feature(
        "PEPE/USDT",
        current_price=0.000002355,
        volume_24h=5_357_584.8,
        volume_24h_base=53_575_848_000_000,
        notional_24h_usdt=126_171_122.04,
        volume_24h_source="quote",
    )
    result = _rank({"PEPE/USDT": feature}, 1)

    metrics = result.diagnostics["symbols"][0]["filter_metrics"]
    assert metrics["notional_24h"] == pytest.approx(126_171_122.04)
    assert metrics["notional_24h_source"] == "quote"


def test_ranker_uses_entry_activity_volume_for_dynamic_policy() -> None:
    feature = _feature(
        "SOL/USDT",
        volume_ratio=0.01,
        volume_ratio_timeframe="1h",
        entry_activity_volume_ratio=0.40,
        entry_activity_volume_timeframe="1m",
        indicator_snapshot_available=True,
    )
    result = _rank({"SOL/USDT": feature}, 1)

    metrics = result.diagnostics["symbols"][0]["filter_metrics"]
    assert metrics["volume_ratio"] == pytest.approx(0.40)
    assert metrics["volume_ratio_source"] == "entry_activity_volume_ratio"
    assert metrics["threshold_source"] == "current_market_feature_cross_section"


def test_missing_market_anchor_fails_closed_without_fixed_fallback() -> None:
    missing = _feature(
        "SOL/USDT",
        current_price=0.0,
        close=0.0,
        volume_24h=0.0,
        indicator_snapshot_available=False,
        technical_indicator_timeframe="",
        short_returns_timeframe="",
    )
    result = _rank({"SOL/USDT": missing}, 1)

    assert result.selected == {}
    assert result.diagnostics["rank_underfilled"] is True
    assert result.diagnostics["rank_underfill_reason"] == "missing_indicator_snapshot"


def test_fallback_market_anchor_score_is_bounded_when_volatility_is_zero() -> None:
    feature = _feature(
        "KAITO/USDT",
        indicator_snapshot_available=False,
        returns_1=0.0,
        returns_5=0.0,
        returns_20=0.0,
        volatility_20=0.0,
        change_24h_pct=12.0,
    )

    score = _ranker().feature_opportunity_score(feature)

    assert score < 20.0
    assert score > 0.0
