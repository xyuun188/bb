from types import SimpleNamespace

import pytest

from services.entry_feature_ranker import EntryFeatureRankerPolicy


def _ranker() -> EntryFeatureRankerPolicy:
    return EntryFeatureRankerPolicy(
        suspicious_symbol_reason=lambda _symbol: None,
        min_entry_volume_ratio_provider=lambda: 0.30,
        min_entry_adx_provider=lambda: 14.0,
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


def test_entry_feature_ranker_classifies_hard_and_soft_candidates() -> None:
    ranker = _ranker()
    hard = _feature("SOL/USDT")
    soft = _feature("DOGE/USDT", volume_ratio=0.07, volume_24h=10_000.0, adx_14=9.0)

    assert ranker.is_auto_tradeable_feature(hard) is True
    assert ranker.is_auto_analysis_candidate_feature(hard) is True
    assert ranker.is_auto_tradeable_feature(soft) is False
    assert ranker.is_auto_analysis_candidate_feature(soft) is True


def test_entry_feature_ranker_uses_explicit_okx_swap_notional() -> None:
    ranker = _ranker()
    pepe_like = _feature(
        "PEPE/USDT",
        current_price=0.000002355,
        volume_24h=5_357_584.8,
        volume_24h_base=53_575_848_000_000,
        notional_24h_usdt=126_171_122.04,
        volume_24h_source="quote",
        volume_ratio=0.40,
        adx_14=18.0,
    )
    legacy_contract_only = _feature(
        "PEPE/USDT",
        current_price=0.000002355,
        volume_24h=5_357_584.8,
        volume_ratio=0.40,
        adx_14=18.0,
    )

    assert ranker.is_auto_tradeable_feature(pepe_like) is True
    assert ranker.is_auto_analysis_candidate_feature(pepe_like) is True
    assert ranker.is_auto_analysis_candidate_feature(legacy_contract_only) is False

    result = ranker.rank(
        {"PEPE/USDT": pepe_like},
        1,
        recent_hold_penalty=lambda _symbol: 0.0,
        recent_analysis_penalty=lambda _symbol: 0.0,
        no_opportunity_rotation_penalty=lambda _symbol, _feature: 0.0,
    )
    metrics = result.diagnostics["symbols"][0]["filter_metrics"]
    assert metrics["notional_24h"] == pytest.approx(126_171_122.04)
    assert metrics["notional_24h_source"] == "quote"


def test_entry_feature_ranker_uses_entry_activity_volume_for_candidate_filter() -> None:
    ranker = _ranker()
    active = _feature(
        "SOL/USDT",
        volume_ratio=0.01,
        volume_ratio_timeframe="1h",
        entry_activity_volume_ratio=0.40,
        entry_activity_volume_timeframe="1m",
        indicator_snapshot_available=True,
    )

    assert ranker.is_auto_analysis_candidate_feature(active) is True

    result = ranker.rank(
        {"SOL/USDT": active},
        1,
        recent_hold_penalty=lambda _symbol: 0.0,
        recent_analysis_penalty=lambda _symbol: 0.0,
        no_opportunity_rotation_penalty=lambda _symbol, _feature: 0.0,
    )
    metrics = result.diagnostics["symbols"][0]["filter_metrics"]
    assert metrics["volume_ratio"] == pytest.approx(0.40)
    assert metrics["volume_ratio_source"] == "entry_activity_volume_ratio"
    assert metrics["trend_volume_ratio"] == pytest.approx(0.01)
    assert metrics["entry_activity_volume_timeframe"] == "1m"
    assert metrics["runtime_entry_volume_ratio_advisory"] == pytest.approx(0.30)
    assert metrics["runtime_entry_adx_advisory"] == pytest.approx(14.0)
    preview = result.diagnostics["ranked_symbol_sample"][0]
    assert preview["volume_ratio"] == pytest.approx(0.40)
    assert preview["volume_ratio_source"] == "entry_activity_volume_ratio"
    assert preview["trend_volume_ratio"] == pytest.approx(0.01)
    assert preview["trend_volume_ratio_timeframe"] == "1h"
    assert preview["entry_activity_volume_timeframe"] == "1m"


def test_entry_feature_ranker_rejects_missing_indicator_snapshot_without_fallback() -> None:
    ranker = _ranker()
    missing = _feature(
        "SOL/USDT",
        volume_ratio=1.0,
        adx_14=20.0,
        indicator_snapshot_available=False,
        technical_indicator_timeframe="",
        short_returns_timeframe="",
    )

    result = ranker.rank(
        {"SOL/USDT": missing},
        1,
        recent_hold_penalty=lambda _symbol: 0.0,
        recent_analysis_penalty=lambda _symbol: 0.0,
        no_opportunity_rotation_penalty=lambda _symbol, _feature: 0.0,
    )

    assert result.selected == {}
    assert result.diagnostics["rank_underfilled"] is True
    assert result.diagnostics["rank_underfill_reason"] == "missing_indicator_snapshot"
    reason_counts = {
        item["reason"]: item["count"] for item in result.diagnostics["filtered_out_reason_counts"]
    }
    assert reason_counts["missing_indicator_snapshot"] == 1
    filtered = result.diagnostics["filtered_symbol_sample"][0]
    assert filtered["filter_reasons"] == ["missing_indicator_snapshot"]


def test_entry_feature_ranker_uses_secondary_fill_when_hard_candidates_are_not_enough() -> None:
    ranker = _ranker()
    result = ranker.rank(
        {
            "SOL/USDT": _feature("SOL/USDT"),
            "DOGE/USDT": _feature(
                "DOGE/USDT",
                volume_ratio=0.07,
                volume_24h=10_000.0,
                adx_14=9.0,
            ),
        },
        2,
        recent_hold_penalty=lambda _symbol: 0.0,
        recent_analysis_penalty=lambda _symbol: 0.0,
        no_opportunity_rotation_penalty=lambda _symbol, _feature: 0.0,
    )

    assert list(result.selected) == ["SOL/USDT", "DOGE/USDT"]
    assert result.diagnostics["tradable_candidates"] == 1
    assert result.diagnostics["secondary_candidates"] == 1
    assert result.diagnostics["symbols"][1]["selection_tier"] == "secondary_fill"
    assert result.diagnostics["ranked_symbol_sample"][0]["selected"] is True
    assert (
        result.diagnostics["ranked_symbol_sample"][0]["non_selected_reason"]
        == "selected_for_market_analysis"
    )


def test_entry_feature_ranker_penalizes_recent_hold_symbols() -> None:
    ranker = _ranker()
    result = ranker.rank(
        {
            "SOL/USDT": _feature("SOL/USDT", returns_5=0.010),
            "LINK/USDT": _feature("LINK/USDT", returns_5=0.006),
        },
        1,
        recent_hold_penalty=lambda symbol: 300.0 if symbol == "SOL/USDT" else 0.0,
        recent_analysis_penalty=lambda _symbol: 0.0,
        no_opportunity_rotation_penalty=lambda _symbol, _feature: 0.0,
    )

    assert list(result.selected) == ["LINK/USDT"]


def test_entry_feature_ranker_defers_recent_analysis_when_fresh_candidates_exist() -> None:
    ranker = _ranker()
    result = ranker.rank(
        {
            "SOL/USDT": _feature("SOL/USDT", returns_5=0.060),
            "LINK/USDT": _feature("LINK/USDT", returns_5=0.050),
            "XRP/USDT": _feature("XRP/USDT", returns_5=0.006),
            "ADA/USDT": _feature("ADA/USDT", returns_5=0.005),
        },
        2,
        recent_hold_penalty=lambda _symbol: 0.0,
        recent_analysis_penalty=lambda symbol: 1.0 if symbol in {"SOL/USDT", "LINK/USDT"} else 0.0,
        no_opportunity_rotation_penalty=lambda _symbol, _feature: 0.0,
    )

    assert list(result.selected) == ["XRP/USDT", "ADA/USDT"]
    diagnostics = {item["symbol"]: item for item in result.diagnostics["ranked_symbol_sample"]}
    assert diagnostics["SOL/USDT"]["non_selected_reason"] == ("recent_analysis_diversity_deferred")
    assert result.diagnostics["recent_analysis_diversity"]["applied"] is True
    assert result.diagnostics["recent_analysis_diversity"]["recent_deferred_count"] == 2


def test_entry_feature_ranker_explains_symbols_outside_market_budget() -> None:
    ranker = _ranker()
    result = ranker.rank(
        {
            "SOL/USDT": _feature("SOL/USDT", returns_5=0.020),
            "LINK/USDT": _feature("LINK/USDT", returns_5=0.015),
            "DOGE/USDT": _feature(
                "DOGE/USDT",
                volume_ratio=0.10,
                volume_24h=10_000.0,
                adx_14=9.0,
            ),
            "THIN/USDT": _feature(
                "THIN/USDT",
                volume_ratio=0.01,
                volume_24h=1.0,
                adx_14=1.0,
            ),
        },
        1,
        recent_hold_penalty=lambda _symbol: 0.0,
        recent_analysis_penalty=lambda _symbol: 0.0,
        no_opportunity_rotation_penalty=lambda _symbol, _feature: 0.0,
    )

    diagnostics = {item["symbol"]: item for item in result.diagnostics["ranked_symbol_sample"]}

    assert result.diagnostics["market_symbol_limit"] == 1
    assert result.diagnostics["filtered_out_candidates"] == 1
    assert diagnostics["SOL/USDT"]["selected"] is True
    assert diagnostics["LINK/USDT"]["selected"] is False
    assert diagnostics["LINK/USDT"]["non_selected_reason"] == "outside_market_symbol_budget"
    assert diagnostics["DOGE/USDT"]["selection_tier"] == "secondary_fill"
    assert "THIN/USDT" not in diagnostics


def test_entry_feature_ranker_fills_underused_analysis_budget_with_near_miss() -> None:
    ranker = _ranker()
    result = ranker.rank(
        {
            "SOL/USDT": _feature("SOL/USDT"),
            "NEAR/USDT": _feature(
                "NEAR/USDT",
                volume_ratio=0.04,
                volume_24h=10_000.0,
                adx_14=12.0,
            ),
        },
        2,
        recent_hold_penalty=lambda _symbol: 0.0,
        recent_analysis_penalty=lambda _symbol: 0.0,
        no_opportunity_rotation_penalty=lambda _symbol, _feature: 0.0,
    )

    assert list(result.selected) == ["SOL/USDT", "NEAR/USDT"]
    assert result.diagnostics["rank_underfilled"] is False
    assert result.diagnostics["fallback_filtered_fill_count"] == 1
    assert result.diagnostics["fallback_filtered_fill_policy"]["applied"] is True
    assert result.diagnostics["fallback_filtered_fill_policy"]["symbols"] == ["NEAR/USDT"]
    selected = {item["symbol"]: item for item in result.diagnostics["symbols"]}
    assert selected["NEAR/USDT"]["selection_tier"] == "fallback_score"
    assert "analysis_volume_ratio_below_floor" in selected["NEAR/USDT"]["filter_reasons"]
    filtered = {item["symbol"]: item for item in result.diagnostics["filtered_symbol_sample"]}
    assert filtered["NEAR/USDT"]["selected"] is True


def test_entry_feature_ranker_explains_filtered_symbols_when_rank_underfills() -> None:
    ranker = _ranker()
    result = ranker.rank(
        {
            "SOL/USDT": _feature("SOL/USDT"),
            "THIN/USDT": _feature(
                "THIN/USDT",
                volume_ratio=0.01,
                volume_24h=1.0,
                adx_14=1.0,
            ),
            "WILD/USDT": _feature(
                "WILD/USDT",
                volatility_20=0.30,
                change_24h_pct=45.0,
            ),
        },
        2,
        recent_hold_penalty=lambda _symbol: 0.0,
        recent_analysis_penalty=lambda _symbol: 0.0,
        no_opportunity_rotation_penalty=lambda _symbol, _feature: 0.0,
    )

    filtered = {item["symbol"]: item for item in result.diagnostics["filtered_symbol_sample"]}
    reason_counts = {
        item["reason"]: item["count"] for item in result.diagnostics["filtered_out_reason_counts"]
    }

    assert list(result.selected) == ["SOL/USDT"]
    assert result.diagnostics["rank_underfilled"] is True
    assert result.diagnostics["rank_underfill_reason"] == (
        "insufficient_tradeable_or_secondary_candidates"
    )
    assert result.diagnostics["filtered_out_candidates"] == 2
    assert result.diagnostics["fallback_filtered_fill_count"] == 0
    assert result.diagnostics["fallback_filtered_fill_policy"]["applied"] is False
    assert reason_counts["analysis_volume_ratio_below_floor"] == 1
    assert reason_counts["analysis_notional_below_floor"] == 1
    assert reason_counts["analysis_volatility_above_cap"] == 1
    assert reason_counts["analysis_day_change_above_cap"] == 1
    assert filtered["THIN/USDT"]["non_selected_reason"] == "feature_filter_rejected"
    assert "analysis_volume_ratio_below_floor" in filtered["THIN/USDT"]["filter_reasons"]
    assert filtered["WILD/USDT"]["filter_metrics"]["change_24h"] == 45.0
