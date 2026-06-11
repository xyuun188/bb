from types import SimpleNamespace

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
    soft = _feature("DOGE/USDT", volume_ratio=0.10, volume_24h=10_000.0, adx_14=9.0)

    assert ranker.is_auto_tradeable_feature(hard) is True
    assert ranker.is_auto_analysis_candidate_feature(hard) is True
    assert ranker.is_auto_tradeable_feature(soft) is False
    assert ranker.is_auto_analysis_candidate_feature(soft) is True


def test_entry_feature_ranker_uses_secondary_fill_when_hard_candidates_are_not_enough() -> None:
    ranker = _ranker()
    result = ranker.rank(
        {
            "SOL/USDT": _feature("SOL/USDT"),
            "DOGE/USDT": _feature(
                "DOGE/USDT",
                volume_ratio=0.10,
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
