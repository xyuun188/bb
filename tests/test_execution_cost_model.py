from config.settings import settings
from services.execution_cost_model import execution_cost_estimate


def test_execution_cost_uses_dynamic_spread_not_configured_max_slippage(monkeypatch) -> None:
    monkeypatch.setattr(settings, "max_slippage_pct", 0.005)

    estimate = execution_cost_estimate(
        {
            "spread_pct": 0.015,
            "orderbook_bid_depth": 120_000.0,
            "orderbook_ask_depth": 110_000.0,
            "orderbook_imbalance": 0.10,
        }
    )

    assert estimate.configured_max_slippage_pct == 0.5
    assert estimate.slippage_pct == 0.05
    assert estimate.slippage_pct < estimate.configured_max_slippage_pct
    assert estimate.slippage_source == "dynamic_microstructure"
    assert estimate.spread_source == "spread_pct"
    assert estimate.total_pct == estimate.fee_pct + estimate.slippage_pct


def test_execution_cost_caps_extreme_microstructure_by_configured_max(monkeypatch) -> None:
    monkeypatch.setattr(settings, "max_slippage_pct", 0.003)

    estimate = execution_cost_estimate(
        {
            "spread_pct": 2.4,
            "orderbook_bid_depth": 0.0,
            "orderbook_ask_depth": 0.0,
            "orderbook_imbalance": 0.95,
        }
    )

    assert estimate.configured_max_slippage_pct == 0.3
    assert estimate.slippage_pct == 0.3
    assert estimate.slippage_source == "capped_by_configured_max"


def test_execution_cost_can_derive_spread_from_bid_ask(monkeypatch) -> None:
    monkeypatch.setattr(settings, "max_slippage_pct", 0.005)

    estimate = execution_cost_estimate({"bid": 99.95, "ask": 100.05})

    assert estimate.spread_source == "bid_ask"
    assert estimate.spread_pct > 0
    assert estimate.slippage_pct >= 0.05
