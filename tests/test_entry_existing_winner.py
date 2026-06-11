from __future__ import annotations

from typing import Any

from ai_brain.base_model import Action, DecisionOutput
from services.entry_existing_winner import EntryExistingWinnerContextPolicy


def _normalize_symbol(value: Any) -> str:
    return str(value or "").replace("/", "-").replace("-SWAP", "")


def _decision(action: Action = Action.LONG) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=action,
        confidence=0.8,
        reasoning="测试开仓",
        position_size_pct=0.05,
        suggested_leverage=3.0,
        raw_response={},
        feature_snapshot={"current_price": 100.0},
    )


def test_existing_winner_context_aggregates_same_symbol_and_side_winners() -> None:
    policy = EntryExistingWinnerContextPolicy(_normalize_symbol)

    context = policy.context(
        _decision(Action.LONG),
        [
            {
                "symbol": "BTC-USDT-SWAP",
                "side": "long",
                "quantity": 2.0,
                "entry_price": 100.0,
                "current_price": 110.0,
                "unrealized_pnl": 12.5,
            },
            {
                "symbol": "BTC/USDT",
                "side": "long",
                "quantity": 1.0,
                "entry_price": 100.0,
                "current_price": 105.0,
                "unrealized_pnl": 2.5,
            },
        ],
    )

    assert context == {
        "has_winner": True,
        "symbol": "BTC-USDT",
        "side": "long",
        "positions": 2,
        "quantity": 3.0,
        "notional_usdt": 325.0,
        "unrealized_pnl": 15.0,
        "pnl_ratio": 0.046154,
    }


def test_existing_winner_context_ignores_other_side_and_closed_positions() -> None:
    policy = EntryExistingWinnerContextPolicy(_normalize_symbol)

    context = policy.context(
        _decision(Action.LONG),
        [
            {"symbol": "BTC/USDT", "side": "short", "unrealized_pnl": 20.0},
            {"symbol": "BTC/USDT", "side": "long", "is_open": False, "unrealized_pnl": 20.0},
        ],
    )

    assert context == {"has_winner": False}


def test_existing_winner_context_prefers_direct_notional_fields() -> None:
    policy = EntryExistingWinnerContextPolicy(_normalize_symbol)

    context = policy.context(
        _decision(Action.SHORT),
        [
            {
                "symbol": "BTC/USDT",
                "side": "short",
                "quantity": 10.0,
                "entry_price": 100.0,
                "current_price": 90.0,
                "notionalUsd": 250.0,
                "unrealized_pnl": 5.0,
            }
        ],
    )

    assert context["has_winner"] is True
    assert context["notional_usdt"] == 250.0
    assert context["quantity"] == 10.0
    assert context["pnl_ratio"] == 0.02
