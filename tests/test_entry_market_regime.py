from ai_brain.base_model import Action, DecisionOutput
from services.entry_market_regime import EntryMarketRegimePolicy


def _decision(action: Action, symbol: str = "ARB/USDT") -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol=symbol,
        action=action,
        confidence=0.8,
        reasoning="entry",
        position_size_pct=0.03,
        raw_response={},
    )


def _normalize(symbol) -> str | None:
    return str(symbol).upper() if symbol is not None else None


def test_market_regime_adds_alt_long_advisory_without_blocking() -> None:
    decision = _decision(Action.LONG)
    policy = EntryMarketRegimePolicy(_normalize, {"BTC/USDT", "ETH/USDT"})

    reason = policy.reason(
        decision,
        {"btc_eth_filter": {"allow_alt_long": False, "avg_adx_14": 22.0}},
    )

    assert reason is None
    advisory = decision.raw_response["alt_long_style_filter"]
    assert advisory["blocked"] is False
    assert advisory["soft_warning"] is True
    assert advisory["btc_eth_filter"]["allow_alt_long"] is False


def test_market_regime_skips_allowed_major_and_short() -> None:
    policy = EntryMarketRegimePolicy(_normalize, {"BTC/USDT", "ETH/USDT"})
    btc_long = _decision(Action.LONG, "BTC/USDT")
    alt_short = _decision(Action.SHORT, "ARB/USDT")

    assert policy.reason(btc_long, {"btc_eth_filter": {"allow_alt_long": False}}) is None
    assert policy.reason(alt_short, {"btc_eth_filter": {"allow_alt_long": False}}) is None
    assert "alt_long_style_filter" not in btc_long.raw_response
    assert "alt_long_style_filter" not in alt_short.raw_response
