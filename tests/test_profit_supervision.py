import pytest

from services.okx_execution_slippage import (
    OKX_ROUND_TRIP_SLIPPAGE_SOURCE,
    PAPER_ROUND_TRIP_SLIPPAGE_VERSION,
)
from services.okx_native_facts import OKX_STOP_LOSS_SLIPPAGE_SOURCE
from services.profit_supervision import (
    AUTHORITATIVE_REALIZED_RETURN_TASK,
    COUNTERFACTUAL_EXECUTION_COST_TASK,
    MARKET_OPPORTUNITY_TASK,
    PROFIT_SUPERVISION_VERSION,
    apply_correlation_group_weights,
    authoritative_execution_cost_calibration,
    authoritative_trade_calibration,
    build_profit_supervision_contract,
    profit_supervision_report,
    select_trade_calibration,
)
from services.profit_training_contract import PROFIT_TRAINING_TARGET


def _shadow_sample(*, sample_id: int, decision_id: int, horizon: int) -> dict:
    sample = {
        "id": sample_id,
        "decision_id": decision_id,
        "symbol": "BTC/USDT",
        "horizon_minutes": horizon,
        "long_return_pct": 0.8,
        "short_return_pct": -0.8,
        "sample_weight": 1.0,
        "exclude_from_training": False,
        "features": {
            "symbol": "BTC/USDT",
            "current_price": 100.0,
            "bid": 99.99,
            "ask": 100.01,
            "orderbook_bid_depth": 10_000.0,
            "orderbook_ask_depth": 9_000.0,
            "taker_fee_rate": 0.0004,
            "funding_rate": 0.0001,
            "funding_data_available": True,
            "funding_interval_minutes": 480,
        },
    }
    sample["profit_supervision"] = build_profit_supervision_contract(
        sample,
        kind="shadow",
    )
    return sample


def _trade_sample() -> dict:
    sample = {
        "id": 91,
        "position_id": 44,
        "lifecycle_key": "okx-position:44",
        "source": "okx_position_history",
        "trade_fact_trusted": True,
        "symbol": "BTC/USDT",
        "side": "long",
        "holding_minutes": 45.0,
        "slippage": 0.03,
        "slippage_source": OKX_ROUND_TRIP_SLIPPAGE_SOURCE,
        "protection_execution_supervision_ready": True,
        "sample_weight": 1.0,
        "exclude_from_training": False,
        "profit_learning_labels": {
            "net_return_after_all_cost_pct": 0.6,
            "realized_net_pnl_usdt": 12.0,
            "gross_return_on_notional_pct": 0.72,
            "fee_return_pct": 0.08,
            "slippage_return_pct": 0.04,
            "funding_return_pct": 0.0,
            "exit_timing_label": "timely",
            "payoff_profile_label": "positive_payoff",
            "losing_exit_attribution": "",
        },
        "profit_training_contract": {
            "eligible": True,
            "target": PROFIT_TRAINING_TARGET,
            "target_value": 0.6,
            "outcome": "profit",
            "decision_authority": "system",
            "evidence_fingerprint": "profit-contract-test",
            "reason": "profit_training_sample_ready",
            "blockers": [],
        },
    }
    sample["profit_supervision"] = build_profit_supervision_contract(
        sample,
        kind="trade",
    )
    return sample


def test_shadow_and_okx_trade_authorities_are_mutually_exclusive() -> None:
    shadow = _shadow_sample(sample_id=1, decision_id=7, horizon=10)
    trade = _trade_sample()
    shadow_tasks = shadow["profit_supervision"]["tasks"]
    trade_tasks = trade["profit_supervision"]["tasks"]

    assert shadow["profit_supervision"]["version"] == PROFIT_SUPERVISION_VERSION
    assert shadow_tasks[MARKET_OPPORTUNITY_TASK]["eligible"] is True
    assert shadow_tasks[COUNTERFACTUAL_EXECUTION_COST_TASK]["eligible"] is True
    assert shadow_tasks[AUTHORITATIVE_REALIZED_RETURN_TASK]["eligible"] is False
    assert trade_tasks[MARKET_OPPORTUNITY_TASK]["eligible"] is False
    assert trade_tasks[COUNTERFACTUAL_EXECUTION_COST_TASK]["eligible"] is True
    assert trade_tasks[AUTHORITATIVE_REALIZED_RETURN_TASK]["eligible"] is True


def test_verified_execution_pair_is_authoritative_realized_return() -> None:
    trade = _trade_sample()
    trade["source"] = "okx_verified_execution_pair"
    trade["profit_supervision"] = build_profit_supervision_contract(
        trade,
        kind="trade",
    )

    task = trade["profit_supervision"]["tasks"][AUTHORITATIVE_REALIZED_RETURN_TASK]

    assert task["eligible"] is True
    assert task["source_authority"] == "okx_verified_execution_pair"


def test_authoritative_return_prefers_profit_training_contract_target() -> None:
    trade = _trade_sample()
    trade["profit_learning_labels"]["net_return_after_all_cost_pct"] = 99.0
    trade["profit_training_contract"]["target_value"] = -2.4
    trade["profit_training_contract"]["outcome"] = "loss"

    contract = build_profit_supervision_contract(trade, kind="trade")
    task = contract["tasks"][AUTHORITATIVE_REALIZED_RETURN_TASK]

    assert task["return_target"] == PROFIT_TRAINING_TARGET
    assert task[PROFIT_TRAINING_TARGET] == -2.4
    assert "realized_net_return_pct" not in task


def test_multi_horizon_rows_share_one_decision_weight_budget() -> None:
    rows = [
        _shadow_sample(sample_id=index, decision_id=7, horizon=horizon)
        for index, horizon in enumerate((5, 15, 60), start=1)
    ]

    apply_correlation_group_weights(rows, kind="shadow")

    assert sum(row["sample_weight"] for row in rows) == pytest.approx(1.0)
    assert {
        row["correlation_weight"]["correlation_group"] for row in rows
    } == {"shadow_decision:7"}
    assert all(
        row["correlation_weight"]["fixed_sampling_ratio"] is False for row in rows
    )


def test_reports_keep_shadow_market_cost_and_actual_returns_separate() -> None:
    shadow = _shadow_sample(sample_id=1, decision_id=7, horizon=10)
    trade = _trade_sample()

    report = profit_supervision_report([shadow], [trade])

    assert report["shadow_market_sample_count"] == 1
    assert report["shadow_counterfactual_cost_sample_count"] == 1
    assert report["actual_execution_cost_sample_count"] == 1
    assert report["actual_realized_return_sample_count"] == 1
    assert report["shadow_cost_basis_counts"] == {"entry_snapshot_round_trip_projection": 1}
    assert report["shadow_samples_are_actual_returns"] is False
    assert report["authoritative_realized_trade"][PROFIT_TRAINING_TARGET][
        "expected"
    ] == pytest.approx(0.6)


def test_trade_calibration_selects_symbol_side_then_global_side() -> None:
    trade = _trade_sample()
    calibration = authoritative_trade_calibration([trade])

    exact = select_trade_calibration(
        calibration,
        symbol="BTC-USDT-SWAP",
        side="long",
    )
    global_side = select_trade_calibration(
        calibration,
        symbol="ETH/USDT",
        side="long",
    )

    assert exact["profile_source"] == "symbol_side"
    assert global_side["profile_source"] == "global_side"
    assert exact[PROFIT_TRAINING_TARGET]["count"] == 1
    assert "net_return_after_cost_pct" not in exact
    assert exact["slippage_pct"]["count"] == 1


def test_stop_slippage_requires_a_complete_okx_stop_fill_lifecycle() -> None:
    verified = _trade_sample()
    verified.update(
        {
            "execution_mode": "paper",
            "inst_id": "BTC-USDT-SWAP",
            "outcome_complete": True,
            "verified_stop_loss_fills": [
                {"order_id": "stop-order", "slippage_pct": 0.12, "credential_scope": "current"},
            ],
            "training_label_contract": {
                "execution_mode": "paper",
                "complete": True,
                PROFIT_TRAINING_TARGET: -1.0,
                "slippage_pct": 0.15,
                "slippage_source": OKX_ROUND_TRIP_SLIPPAGE_SOURCE,
            },
        }
    )
    calibration = authoritative_execution_cost_calibration(
        [verified, dict(verified)], execution_mode="paper", credential_scope="current"
    )
    exact = calibration["stop_loss_slippage_profiles"]["BTC/USDT|long"]
    assert exact["stop_loss_slippage_pct"]["count"] == 1
    assert exact["stop_loss_slippage_pct"]["expected"] == pytest.approx(0.12)
    assert exact["stop_loss_slippage_source"] == OKX_STOP_LOSS_SLIPPAGE_SOURCE
    assert exact["stop_loss_slippage_credential_scope"] == "current"
    assert set(calibration["stop_loss_slippage_profiles"]) == {"BTC/USDT|long"}
    stop_only = authoritative_execution_cost_calibration(
        [{**verified, "outcome_complete": False, "training_label_contract": {}}],
        execution_mode="paper",
        credential_scope="current",
    )
    assert stop_only["profiles"] == {}
    assert stop_only["stop_loss_slippage_profiles"]["BTC/USDT|long"] == exact

    for field, invalid_value in (
        ("verified_stop_loss_fills", []),
        ("verified_stop_loss_fills", [{"slippage_pct": 0.12, "credential_scope": "current"}]),
        ("verified_stop_loss_fills", [
            {"order_id": "stop-order", "slippage_pct": 0.12, "credential_scope": "old"},
        ]),
        ("inst_id", "ETH-USDT-SWAP"),
        ("execution_mode", "live"),
    ):
        calibration = authoritative_execution_cost_calibration(
            [{**verified, field: invalid_value}],
            execution_mode="paper",
            credential_scope="current",
        )
        assert calibration["stop_loss_slippage_profiles"] == {}


def test_execution_calibration_learns_costs_of_excluded_losing_trades_only_in_own_mode() -> None:
    outcome = {
        "symbol": "BTC/USDT",
        "side": "short",
        "source": "okx_position_history",
        "execution_mode": "paper",
        "lifecycle_key": "paper|BTC|one",
        "outcome_complete": True,
        "trade_fact_trusted": True,
        "exclude_from_training": True,
        "training_label_contract": {
            "complete": True,
            "execution_mode": "paper",
            PROFIT_TRAINING_TARGET: -2.0,
            "slippage_pct": 0.3,
            "slippage_source": OKX_ROUND_TRIP_SLIPPAGE_SOURCE,
        },
    }
    calibration = authoritative_execution_cost_calibration(
        [outcome, dict(outcome), {**outcome, "execution_mode": "live"}],
        execution_mode="paper",
    )
    profile = select_trade_calibration(calibration, symbol="BTC/USDT", side="short")
    assert calibration["actual_realized_return_sample_count"] == 1
    assert profile["slippage_pct"]["expected"] == pytest.approx(0.3)
    assert profile["execution_mode"] == "paper"
    incomplete = {**outcome, "outcome_complete": False}
    assert authoritative_execution_cost_calibration(
        [incomplete], execution_mode="paper"
    )["profiles"] == {}


def test_authoritative_return_rejects_old_net_return_label_without_contract() -> None:
    trade = _trade_sample()
    trade["profit_training_contract"] = {
        "eligible": False,
        "target": PROFIT_TRAINING_TARGET,
        "target_value": None,
        "reason": "missing_contract",
    }
    trade["profit_learning_labels"]["net_return_after_all_cost_pct"] = 99.0

    contract = build_profit_supervision_contract(trade, kind="trade")
    task = contract["tasks"][AUTHORITATIVE_REALIZED_RETURN_TASK]

    assert task["eligible"] is False
    assert task[PROFIT_TRAINING_TARGET] is None
    assert task["return_target"] == PROFIT_TRAINING_TARGET


@pytest.mark.parametrize("mode,complete,signed_pct,expected", [
    ("paper", True, 0.23, 0.23),
    ("paper", True, -0.23, 0.0),
    ("paper", True, 10.13, 10.13),
    ("paper", False, 0.23, 163.9),
    ("live", True, 0.23, 163.9),
])
def test_round_trip_calibration_cancels_basis_only_for_complete_paper_fills(
    mode: str, complete: bool, signed_pct: float, expected: float,
) -> None:
    sample = _trade_sample()
    sample.update({
        "execution_mode": mode,
        "notional": 3.045,
        "slippage": 163.9,
        "outcome_complete": True,
        "paper_round_trip_cost": {
            "version": PAPER_ROUND_TRIP_SLIPPAGE_VERSION,
            "source": OKX_ROUND_TRIP_SLIPPAGE_SOURCE,
            "execution_mode": mode,
            "complete": complete,
            "basis": "net_signed_fill_mark_round_trip",
            "signed_slippage_usdt": 3.045 * signed_pct / 100.0,
            "notional": 3.045,
        },
        "training_label_contract": {
            "complete": True, "execution_mode": mode,
            "slippage_source": OKX_ROUND_TRIP_SLIPPAGE_SOURCE,
            "slippage_pct": 163.9,
            PROFIT_TRAINING_TARGET: -0.33,
        },
    })
    supervision = build_profit_supervision_contract(sample, kind="trade")
    costs = supervision["tasks"][COUNTERFACTUAL_EXECUTION_COST_TASK]
    assert costs["slippage_pct"] == pytest.approx(expected)
    assert costs["adverse_fill_slippage_pct"] == 163.9
    realized = supervision["tasks"][AUTHORITATIVE_REALIZED_RETURN_TASK]
    assert realized[PROFIT_TRAINING_TARGET] == 0.6
    calibration = authoritative_execution_cost_calibration([sample], execution_mode=mode)
    profile = select_trade_calibration(calibration, symbol="BTC/USDT", side="long")
    assert profile["slippage_pct"]["expected"] == pytest.approx(expected)
    assert profile["adverse_fill_slippage_pct"]["expected"] == 163.9
    assert profile[PROFIT_TRAINING_TARGET]["expected"] == -0.33


def test_confirmed_round_trip_slippage_can_supervise_actual_execution_cost() -> None:
    trade = _trade_sample()
    trade["profit_learning_labels"]["slippage_return_pct"] = None
    trade["profit_supervision"] = build_profit_supervision_contract(
        trade,
        kind="trade",
    )

    cost = trade["profit_supervision"]["tasks"][
        COUNTERFACTUAL_EXECUTION_COST_TASK
    ]

    assert cost["eligible"] is True
    assert cost["slippage_pct"] == pytest.approx(0.03)
    assert cost["slippage_source"] == OKX_ROUND_TRIP_SLIPPAGE_SOURCE
