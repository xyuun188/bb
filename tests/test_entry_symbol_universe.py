from __future__ import annotations

from services.entry_symbol_universe import EntrySymbolUniversePolicy


def _normalize(symbol: object) -> str | None:
    if not symbol:
        return None
    return str(symbol).replace("-", "/").upper()


def test_entry_symbol_universe_dedupes_normalized_symbols() -> None:
    policy = EntrySymbolUniversePolicy(_normalize)

    assert policy.dedupe_symbols(["btc-usdt", "BTC/USDT", "", "eth-usdt"]) == [
        "BTC/USDT",
        "ETH/USDT",
    ]


def test_entry_symbol_universe_counts_open_position_groups() -> None:
    policy = EntrySymbolUniversePolicy(_normalize)

    assert (
        policy.open_position_group_count(
            [
                {"model_name": "ensemble_trader", "symbol": "btc-usdt", "side": "long"},
                {"model_name": "ensemble_trader", "symbol": "BTC/USDT", "side": "long"},
                {"model_name": "ensemble_trader", "symbol": "BTC/USDT", "side": "short"},
                {"model_name": "other", "symbol": "BTC/USDT", "side": "long"},
                {"model_name": "other", "symbol": "ETH/USDT", "side": "flat"},
                {"model_name": "other", "symbol": "SOL/USDT", "side": "long", "is_open": False},
            ]
        )
        == 3
    )


def test_entry_symbol_universe_filters_open_position_market_symbols() -> None:
    policy = EntrySymbolUniversePolicy(_normalize)

    result = policy.filter_open_position_market_symbols(
        ["BTC/USDT", "ETH/USDT"],
        [{"symbol": "btc-usdt"}],
    )

    assert result.symbols == ["ETH/USDT"]
    assert result.skipped == ["BTC/USDT"]


def test_entry_symbol_universe_filters_active_analysis_symbols() -> None:
    policy = EntrySymbolUniversePolicy(_normalize)

    result = policy.filter_unclaimed_market_symbols(
        ["BTC/USDT", "ETH/USDT"],
        {"BTC/USDT"},
    )

    assert result.symbols == ["ETH/USDT"]
    assert result.skipped == ["BTC/USDT"]


def test_entry_symbol_universe_filters_blocked_new_symbols_but_keeps_open_position() -> None:
    policy = EntrySymbolUniversePolicy(_normalize)

    result = policy.filter_blocked_new_symbols(
        ["BTC/USDT", "ETH/USDT"],
        [{"symbol": "BTC/USDT"}],
        suspicious_reason=lambda symbol: "suspicious" if symbol == "ETH/USDT" else None,
        blocked_reason=lambda _symbol: None,
    )

    assert result.symbols == ["BTC/USDT"]
    assert len(result.skipped) == 1
    assert result.skipped[0].symbol == "ETH/USDT"
    assert result.skipped[0].reason == "suspicious"
