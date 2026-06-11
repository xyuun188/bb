from types import SimpleNamespace

from services.entry_market_regime import EntryMarketRegimeContextPolicy
from services.trading_service import TradingService


def _feature(
    symbol: str,
    *,
    returns_5: float = 0.0,
    returns_20: float = 0.0,
    price_vs_sma20: float = 0.0,
    price_vs_sma50: float = 0.0,
    adx_14: float = 12.0,
    current_price: float = 100.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol,
        returns_5=returns_5,
        returns_20=returns_20,
        price_vs_sma20=price_vs_sma20,
        price_vs_sma50=price_vs_sma50,
        adx_14=adx_14,
        current_price=current_price,
    )


def _policy() -> EntryMarketRegimeContextPolicy:
    return EntryMarketRegimeContextPolicy(lambda feature: feature.current_price > 0)


def test_market_regime_context_returns_unknown_without_valid_rows() -> None:
    context = _policy().context({"bad": _feature("BAD/USDT", current_price=0.0)})

    assert context == {
        "mode": "unknown",
        "confidence": 0.0,
        "avoid_long": False,
        "avoid_short": False,
    }


def test_market_regime_context_detects_rebound_squeeze_up() -> None:
    context = _policy().context(
        {
            "BTC/USDT": _feature("BTC/USDT", returns_5=0.004, returns_20=0.002),
            "ETH/USDT": _feature("ETH/USDT", returns_5=0.003, returns_20=0.001),
            "SOL/USDT": _feature("SOL/USDT", returns_5=0.004, returns_20=0.007),
            "BNB/USDT": _feature("BNB/USDT", returns_5=0.005, returns_20=0.006),
        }
    )

    assert context["mode"] == "rebound_squeeze_up"
    assert context["avoid_short"] is True
    assert context["avoid_long"] is False
    assert context["up_5_ratio"] == 1.0
    assert context["btc_eth_filter"]["allow_alt_long"] is True


def test_market_regime_context_detects_selloff_and_btc_eth_alt_long_block() -> None:
    context = _policy().context(
        {
            "BTC/USDT": _feature("BTC/USDT", returns_5=-0.003, returns_20=-0.006, adx_14=22.0),
            "ETH/USDT": _feature("ETH/USDT", returns_5=-0.004, returns_20=-0.007, adx_14=20.0),
            "SOL/USDT": _feature("SOL/USDT", returns_5=-0.003, returns_20=-0.008),
            "BNB/USDT": _feature("BNB/USDT", returns_5=-0.004, returns_20=-0.008),
        }
    )

    assert context["mode"] == "selloff_squeeze_down"
    assert context["avoid_long"] is True
    assert context["btc_eth_filter"]["allow_alt_long"] is False
    assert "Broad market is falling" in context["btc_eth_filter"]["reason"]


def test_btc_eth_alt_long_filter_allows_when_context_missing() -> None:
    result = _policy().btc_eth_alt_long_filter([])

    assert result["allow_alt_long"] is True
    assert "unavailable" in result["reason"]


def test_trading_service_market_regime_delegates_to_policy() -> None:
    service = object.__new__(TradingService)

    context = service._market_regime_context(
        {
            "BTC/USDT": _feature("BTC/USDT", returns_5=0.004),
            "ETH/USDT": _feature("ETH/USDT", returns_5=0.004),
        }
    )
    filter_result = service._btc_eth_alt_long_filter(
        [
            _feature("BTC/USDT", returns_5=-0.003, returns_20=-0.006, adx_14=18.0),
            _feature("ETH/USDT", returns_5=-0.004, returns_20=-0.007, adx_14=18.0),
        ]
    )

    assert context["mode"] == "rebound_squeeze_up"
    assert filter_result["allow_alt_long"] is False
