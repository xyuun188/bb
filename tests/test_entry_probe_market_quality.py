from types import SimpleNamespace

from services.entry_probe_market_quality import EntryProbeMarketQualityPolicy


def _fv(**values):
    defaults = {
        "current_price": 100.0,
        "close": 100.0,
        "returns_20": 0.0,
        "volume_ratio": 1.0,
        "price_vs_sma20": 0.0,
        "price_vs_sma50": 0.0,
    }
    defaults.update(values)
    return SimpleNamespace(**defaults)


def test_probe_market_quality_blocks_price_field_gap() -> None:
    reason = EntryProbeMarketQualityPolicy().block_reason(
        _fv(current_price=104.0, close=100.0), "long"
    )

    assert reason is not None
    assert "current_price" in reason


def test_probe_market_quality_blocks_contra_long_and_short() -> None:
    policy = EntryProbeMarketQualityPolicy()

    long_reason = policy.block_reason(
        _fv(returns_20=-0.06, price_vs_sma20=-0.1, price_vs_sma50=-0.2),
        "long",
    )
    short_reason = policy.block_reason(
        _fv(returns_20=0.06, price_vs_sma20=0.1, price_vs_sma50=0.2),
        "short",
    )

    assert long_reason is not None
    assert short_reason is not None


def test_probe_market_quality_blocks_low_volume_only_when_positive() -> None:
    policy = EntryProbeMarketQualityPolicy()

    assert policy.block_reason(_fv(volume_ratio=0.01), "long") is not None
    assert policy.block_reason(_fv(volume_ratio=0.0), "long") is None
    assert policy.block_reason(_fv(volume_ratio=0.2), "long") is None
