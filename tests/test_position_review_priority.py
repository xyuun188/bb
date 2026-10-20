from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from data_feed.feature_vector import FeatureVector
from services.current_position_management import build_current_position_management_contract
from services.position_group_aggregator import PositionGroupAggregator
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
                        "created_at": datetime.now(UTC) - timedelta(minutes=20),
                        "current_management_contract": _management_contract(),
                    }
                ],
            )
        ],
        {
            "BTC/USDT": FeatureVector(
                symbol="BTC/USDT",
                current_price=99.0,
                returns_1=-0.01,
                returns_5=-0.008,
                returns_20=-0.005,
            )
        },
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
    assert scan["dynamic_exit_policy"]["eligible"] is False
    assert scan["reason"] == scan["dynamic_exit_policy"]["reason"]
    assert scan["reason"] != "dynamic_exit_pressure_zero"


@pytest.mark.parametrize("missing_fragment_evidence", [False, True])
def test_priority_preserves_management_and_cost_evidence_through_aggregation(
    missing_fragment_evidence: bool,
) -> None:
    row = {
        "model_name": "ensemble_trader",
        "symbol": "BTC/USDT",
        "side": "long",
        "entry_price": 100.0,
        "current_price": 99.0,
        "quantity": 10.0,
        "notional_usdt": 990.0,
        "unrealized_pnl": -10.0,
        "entry_fee_usdt": 0.05,
        "entry_fee": 0.05,
        "stop_loss_pct": 0.02,
        "stop_loss": 98.0,
        "take_profit": 110.0,
        "created_at": datetime.now(UTC) - timedelta(minutes=20),
        "current_management_contract": _management_contract(),
    }
    aggregator = PositionGroupAggregator(
        normalize_symbol=lambda value: str(value or "").split(":")[0]
    )
    feature = FeatureVector(
        symbol="BTC/USDT",
        current_price=99.0,
        returns_1=-0.01,
        returns_5=-0.008,
        returns_20=-0.005,
    )
    second = {
        **row,
        "current_management_contract": (
            {} if missing_fragment_evidence else row["current_management_contract"]
        ),
    }

    scan = _policy().scan_groups(
        [(("ensemble_trader", "BTC/USDT"), [row, second])],
        {"BTC/USDT": feature},
        {},
        aggregate_position_group=aggregator.aggregate,
    )[("ensemble_trader", "BTC/USDT")]

    assert (
        scan["dynamic_exit_policy"]["current_management_contract_complete"]
        is not missing_fragment_evidence
    )
    assert scan["dynamic_exit_policy"]["execution_cost_complete"] is True
    assert "exit_execution_cost_missing" not in scan["reason"]
    assert scan["dynamic_exit_eligible"] is not missing_fragment_evidence
    if missing_fragment_evidence:
        assert "current_position_management_contract_incomplete" in scan["reason"]
        assert scan["priority_score"] == 0.0
    else:
        assert "current_position_management_contract_incomplete" not in scan["reason"]


def test_priority_keeps_confirmed_lifecycle_and_profit_lock_quantities() -> None:
    management = _management_contract()
    management["lifecycle_entry_quantity"] = 100.0
    management["profit_lock_exit_ledger"] = {"realized_quantity": 50.0}
    row = {
        "model_name": "ensemble_trader",
        "symbol": "BTC/USDT",
        "side": "long",
        "quantity": 10.0,
        "entry_price": 100.0,
        "current_price": 101.0,
        "unrealized_pnl": 10.0,
        "entry_fee_usdt": 0.05,
        "stop_loss": 98.0,
        "take_profit": 110.0,
        "created_at": datetime.now(UTC) - timedelta(minutes=20),
        "current_management_contract": management,
    }
    scan = _policy().scan_groups(
        [(("ensemble_trader", "BTC/USDT"), [row])],
        {},
        {},
        aggregate_position_group=PositionGroupAggregator(lambda symbol: symbol).aggregate,
    )[("ensemble_trader", "BTC/USDT")]

    assert scan["dynamic_exit_policy"]["current_management_contract_complete"] is True
    assert scan["dynamic_exit_policy"]["execution_cost_complete"] is True
    assert scan["dynamic_exit_policy"]["lifecycle_closed_fraction"] == pytest.approx(0.9)
    assert scan["dynamic_exit_policy"]["profit_lock_realized_fraction"] == pytest.approx(0.5)
    assert scan["reason"] == "profit_lock_target_already_filled"
    assert scan["dynamic_exit_eligible"] is False
    assert scan["priority_score"] == 0.0


@pytest.mark.parametrize("eligible_side", [None, "short"])
def test_blocked_assessment_preserves_reason_without_granting_exit(
    monkeypatch: pytest.MonkeyPatch, eligible_side: str | None
) -> None:
    def assess(_policy, position, _feature, **_kwargs):
        eligible = position["side"] == eligible_side
        return {
            "eligible": eligible,
            "close_fraction": 0.1 if eligible else 0.8,
            "reason": "dynamic_exit_policy_passed" if eligible else "exit_execution_cost_incomplete",
        }

    monkeypatch.setattr(PositionReviewPriorityPolicy, "_assessment", assess)
    key = ("ensemble_trader", "BTC/USDT")
    scan = _policy().scan_groups(
        [(key, [{"side": "long"}, {"side": "short"}])],
        {},
        {},
        aggregate_position_group=_aggregate,
    )[key]

    if eligible_side:
        assert scan["dynamic_exit_eligible"] is True
        assert scan["priority_score"] == pytest.approx(10.0)
        assert scan["reason"] == "dynamic_exit_policy_passed"
        assert _policy().is_urgent_exit_scan(scan) is True
    else:
        assert scan["dynamic_exit_policy"]["close_fraction"] == 0.8
        assert scan["reason"] == "exit_execution_cost_incomplete"
        assert scan["dynamic_exit_eligible"] is False
        assert scan["priority_score"] == scan["exit_score"] == 0.0
        assert _policy().is_urgent_exit_scan(scan) is False


def test_dynamic_exit_eligibility_is_the_only_urgent_signal() -> None:
    policy = _policy()

    assert policy.is_urgent_exit_scan({"dynamic_exit_eligible": True}) is True
    assert policy.is_urgent_exit_scan({"exit_score": 100.0}) is False
