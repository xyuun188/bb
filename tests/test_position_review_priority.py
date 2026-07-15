from typing import Any

import pytest

from services.current_position_management import build_current_position_management_contract
from services.position_review_priority import PositionReviewPriorityPolicy


def _policy(peaks: dict[Any, dict[str, Any]] | None = None) -> PositionReviewPriorityPolicy:
    states = peaks or {}
    return PositionReviewPriorityPolicy(
        normalize_symbol=lambda value: str(value or "").split(":")[0],
        position_peak_key=lambda model, symbol, side: (model, symbol, side),
        position_peaks_provider=lambda: states,
    )


def _aggregate(rows, model_name, symbol, side):
    return {**rows[0], "model_name": model_name, "symbol": symbol, "side": side}


def _management_contract() -> dict[str, Any]:
    return build_current_position_management_contract(
        {
            "symbol": "BTC/USDT",
            "side": "long",
            "quantity": 10.0,
            "contracts": 10.0,
            "entry_price": 100.0,
            "current_price": 99.0,
            "entry_fee_usdt": 0.05,
            "full_entry_fee_usdt": 0.05,
            "full_entry_notional_usdt": 1_000.0,
            "entry_fee_evidence_complete": True,
            "entry_fee_source": "okx_fills_history",
            "stop_loss_price": 98.0,
            "take_profit_price": 110.0,
            "protection_evidence_complete": True,
            "protection_orders": [
                {
                    "algo_id": "oco-btc",
                    "state": "live",
                    "contracts": 10.0,
                    "reduce_only": True,
                    "stop_loss_price": 98.0,
                    "take_profit_price": 110.0,
                }
            ],
            "position_stressed_loss_usdt": 20.0,
            "portfolio_stressed_loss_usdt": 20.0,
            "portfolio_gross_notional_usdt": 990.0,
            "account_equity_usdt": 10_000.0,
            "open_position_count": 1,
            "entry_order_ids": ["entry-btc"],
            "entry_decision_ids": [],
            "original_entry_contract_complete": False,
            "original_entry_contract_gaps": ["historical_contract_missing"],
        }
    )


def test_priority_is_continuous_dynamic_exit_fraction() -> None:
    scans = _policy().scan_groups(
        [
            (
                ("ensemble_trader", "BTC/USDT"),
                [
                    {
                        "side": "long",
                        "entry_price": 100.0,
                        "current_price": 99.0,
                        "quantity": 10.0,
                        "notional_usdt": 990.0,
                        "unrealized_pnl": -10.0,
                        "entry_fee_usdt": 0.05,
                        "stop_loss_pct": 0.02,
                        "stop_loss": 98.0,
                        "take_profit": 110.0,
                        "current_management_contract": _management_contract(),
                    }
                ],
            )
        ],
        {},
        {},
        aggregate_position_group=_aggregate,
    )

    scan = scans[("ensemble_trader", "BTC/USDT")]
    assert 0.0 < scan["priority_score"] < 100.0
    assert scan["priority_score"] == pytest.approx(scan["exit_score"])
    assert scan["add_score"] == 0.0
    assert scan["force_exit_candidate"] is False


def test_profitable_priority_requires_cost_complete_dynamic_exit() -> None:
    scans = _policy(
        {
            ("ensemble_trader", "BTC/USDT", "long"): {
                "peak_unrealized_pnl": 20.0,
            }
        }
    ).scan_groups(
        [
            (
                ("ensemble_trader", "BTC/USDT"),
                [
                    {
                        "side": "long",
                        "entry_price": 100.0,
                        "current_price": 101.0,
                        "quantity": 10.0,
                        "notional_usdt": 1010.0,
                        "unrealized_pnl": 10.0,
                        "stop_loss_pct": 0.02,
                    }
                ],
            )
        ],
        {},
        {},
        aggregate_position_group=_aggregate,
    )

    scan = scans[("ensemble_trader", "BTC/USDT")]
    assert scan["priority_score"] == 0.0
    assert scan["dynamic_exit_eligible"] is False


def test_dynamic_exit_eligibility_is_the_only_urgent_signal() -> None:
    policy = _policy()

    assert policy.is_urgent_exit_scan({"dynamic_exit_eligible": True}) is True
    assert policy.is_urgent_exit_scan({"exit_score": 100.0}) is False
