from services.portfolio_profit_protection import PortfolioProfitProtectionPolicy
from services.trading_service import TradingService


def _normalize(value) -> str:
    return str(value or "").split(":")[0]


def _policy() -> PortfolioProfitProtectionPolicy:
    return PortfolioProfitProtectionPolicy(
        normalize_symbol=_normalize,
        default_model_name="ensemble_trader",
    )


def test_portfolio_profit_context_groups_and_focuses_winners() -> None:
    context = _policy().context(
        [
            {
                "model_name": "ensemble_trader",
                "symbol": "BTC/USDT:USDT",
                "side": "long",
                "entry_price": 100.0,
                "quantity": 1.0,
                "unrealized_pnl": 2.4,
                "created_at": "2026-06-10T01:00:00Z",
            },
            {
                "model_name": "ensemble_trader",
                "symbol": "ETH/USDT",
                "side": "short",
                "entry_price": 50.0,
                "quantity": 1.0,
                "unrealized_pnl": 0.8,
            },
            {
                "model_name": "ensemble_trader",
                "symbol": "XRP/USDT",
                "side": "long",
                "entry_price": 10.0,
                "quantity": 1.0,
                "unrealized_pnl": -0.2,
            },
        ]
    )

    assert context["active"] is True
    assert context["threshold_usdt"] == 3.0
    assert context["total_unrealized_pnl"] == 3.0
    assert context["total_positive_unrealized_pnl"] == 3.2
    assert context["top_groups"][0]["symbol"] == "BTC/USDT"
    assert context["top_groups"][0]["profit_share"] == 0.75
    assert [item["symbol"] for item in context["focus_groups"]] == ["BTC/USDT", "ETH/USDT"]
    assert context["instruction"]


def test_portfolio_profit_context_ignores_closed_and_invalid_positions() -> None:
    context = _policy().context(
        [
            {
                "symbol": "BTC/USDT",
                "side": "long",
                "entry_price": 100.0,
                "quantity": 1.0,
                "unrealized_pnl": 10.0,
                "is_open": False,
            },
            {
                "symbol": "ETH/USDT",
                "side": "flat",
                "entry_price": 100.0,
                "quantity": 1.0,
                "unrealized_pnl": 10.0,
            },
        ]
    )

    assert context["active"] is False
    assert context["total_unrealized_pnl"] == 0.0
    assert context["focus_groups"] == []


def test_symbol_context_uses_focus_group_when_available() -> None:
    context = _policy().context(
        [
            {
                "model_name": "ensemble_trader",
                "symbol": "BTC/USDT",
                "side": "long",
                "entry_price": 100.0,
                "quantity": 1.0,
                "unrealized_pnl": 3.2,
            }
        ]
    )

    symbol_context = _policy().symbol_context(context, "ensemble_trader", "BTC/USDT:USDT")

    assert symbol_context["active"] is True
    assert symbol_context["is_focus"] is True
    assert symbol_context["current_group"]["symbol"] == "BTC/USDT"
    assert symbol_context["required_choice"] == [
        "continue_hold_with_reason",
        "add_to_winner_if_trend_continues",
        "partial_lock_profit",
        "full_close",
    ]


def test_symbol_context_falls_back_to_passed_positions_for_non_top_group() -> None:
    symbol_context = _policy().symbol_context(
        {
            "active": True,
            "threshold_usdt": 3.0,
            "total_unrealized_pnl": 5.0,
            "total_positive_unrealized_pnl": 5.0,
            "focus_groups": [],
            "top_groups": [],
        },
        "ensemble_trader",
        "SOL/USDT",
        [
            {
                "side": "long",
                "entry_price": 20.0,
                "quantity": 2.0,
                "unrealized_pnl": 1.4,
            }
        ],
    )

    assert symbol_context["active"] is True
    assert symbol_context["is_focus"] is False
    assert symbol_context["current_group"]["symbol"] == "SOL/USDT"
    assert symbol_context["current_group"]["profit_pct"] == 0.035


def test_trading_service_portfolio_profit_delegates_use_policy() -> None:
    service = object.__new__(TradingService)

    context = service._portfolio_profit_protection_context(
        [
            {
                "model_name": "ensemble_trader",
                "symbol": "BTC/USDT:USDT",
                "side": "long",
                "entry_price": 100.0,
                "quantity": 1.0,
                "unrealized_pnl": 3.5,
            }
        ]
    )
    symbol_context = service._portfolio_profit_protection_symbol_context(
        context,
        "ensemble_trader",
        "BTC/USDT",
    )

    assert context["active"] is True
    assert symbol_context["is_focus"] is True
