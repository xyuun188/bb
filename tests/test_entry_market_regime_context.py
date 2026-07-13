from types import SimpleNamespace

import pytest

from services.entry_market_regime import EntryMarketRegimeContextPolicy
from services.trading_service import TradingService


def _feature(symbol: str, **overrides) -> SimpleNamespace:
    values = {
        "symbol": symbol,
        "returns_5": 0.0,
        "returns_20": 0.0,
        "price_vs_sma20": 0.0,
        "price_vs_sma50": 0.0,
        "adx_14": 12.0,
        "current_price": 100.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _policy() -> EntryMarketRegimeContextPolicy:
    return EntryMarketRegimeContextPolicy(lambda feature: feature.current_price > 0)


def test_market_regime_context_fails_closed_without_valid_rows() -> None:
    context = _policy().context({"bad": _feature("BAD/USDT", current_price=0.0)})

    assert context["mode"] == "observation_unavailable"
    assert context["sample_count"] == 0
    assert context["production_permission"] is False
    assert context["policy_provenance"]["fallback_reason"]
    assert "avoid_long" not in context
    assert "avoid_short" not in context


def test_market_regime_context_reports_cross_section_without_fixed_direction_rules() -> None:
    context = _policy().context(
        {
            "BTC/USDT": _feature("BTC/USDT", returns_5=0.004, returns_20=0.002),
            "ETH/USDT": _feature("ETH/USDT", returns_5=-0.002, returns_20=-0.004),
        }
    )

    assert context["mode"] == "return_distribution_observation"
    assert context["sample_count"] == 2
    assert context["avg_returns_5"] == pytest.approx(0.001)
    assert context["avg_returns_20"] == pytest.approx(-0.001)
    assert context["production_permission"] is False
    assert "btc_eth_filter" not in context
    assert "avoid_long" not in context
    assert "avoid_short" not in context


def test_trading_service_market_regime_delegates_to_observation_policy() -> None:
    service = object.__new__(TradingService)
    context = service._market_regime_context(
        {"BTC/USDT": _feature("BTC/USDT", returns_5=0.004)}
    )

    assert context["mode"] == "return_distribution_observation"
    assert context["production_permission"] is False
    assert not hasattr(TradingService, "_btc_eth_alt_long_filter")
