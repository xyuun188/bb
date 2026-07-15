from services.entry_position_exposure import EntryPositionExposurePolicy


def test_entry_position_exposure_summarizes_notional_and_staged_counts() -> None:
    policy = EntryPositionExposurePolicy()

    context = policy.context(
        [
            {
                "symbol": "BTC/USDT",
                "side": "long",
                "quantity": 2,
                "current_price": 100.0,
                "unrealized_pnl": 3.5,
            },
            {
                "symbol": "ETH/USDT",
                "side": "short",
                "notional": 50.0,
                "unrealizedPnl": -1.0,
            },
            {
                "symbol": "XRP/USDT",
                "side": "long",
                "quantity": 999,
                "current_price": 1.0,
                "is_open": False,
            },
        ],
        {"side_totals": {"long": 1, "short": 0}},
    )

    assert context["long_notional"] == 200.0
    assert context["short_notional"] == 50.0
    assert context["total_unrealized_pnl"] == 2.5
    assert context["long_count"] == 2
    assert context["short_count"] == 1
    assert context["staged_long_count"] == 1
    assert context["dominant_side"] == "long"


def test_entry_position_exposure_uses_count_dominance_when_notional_is_balanced() -> None:
    policy = EntryPositionExposurePolicy()

    context = policy.context(
        [
            {"side": "long", "notional": 100.0},
            {"side": "long", "notional": 10.0},
            {"side": "short", "notional": 100.0},
        ],
        {"side_totals": {"long": 1}},
    )

    assert context["net_ratio"] < 0.1
    assert context["long_count_share"] == 0.75
    assert context["dominant_side"] == "long"
