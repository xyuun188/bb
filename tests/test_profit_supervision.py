import pytest

from services.profit_supervision import (
    AUTHORITATIVE_REALIZED_RETURN_TASK,
    COUNTERFACTUAL_EXECUTION_COST_TASK,
    MARKET_OPPORTUNITY_TASK,
    PROFIT_SUPERVISION_VERSION,
    apply_correlation_group_weights,
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
        "hold_minutes": 45.0,
        "stop_loss_slippage_pct": 0.03,
        "stop_loss_slippage_source": "okx_configured_stop_trigger_to_fills_vwap",
        "protection_execution_supervision_ready": True,
        "sample_weight": 1.0,
        "exclude_from_training": False,
        "profit_learning_labels": {
            "net_return_after_cost_pct": 0.6,
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
    trade["profit_learning_labels"]["net_return_after_cost_pct"] = 99.0
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


def test_authoritative_return_rejects_old_net_return_label_without_contract() -> None:
    trade = _trade_sample()
    trade["profit_training_contract"] = {
        "eligible": False,
        "target": PROFIT_TRAINING_TARGET,
        "target_value": None,
        "reason": "missing_contract",
    }
    trade["profit_learning_labels"]["net_return_after_cost_pct"] = 99.0

    contract = build_profit_supervision_contract(trade, kind="trade")
    task = contract["tasks"][AUTHORITATIVE_REALIZED_RETURN_TASK]

    assert task["eligible"] is False
    assert task[PROFIT_TRAINING_TARGET] is None
    assert task["return_target"] == PROFIT_TRAINING_TARGET


def test_confirmed_stop_fill_slippage_can_supervise_actual_execution_cost() -> None:
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
    assert cost["slippage_source"] == "okx_stop_trigger_to_fill_slippage"
