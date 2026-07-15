from services.dynamic_leverage_allocator import DynamicLeverageAllocator, DynamicLeverageInput


def _input(**overrides):
    base = {
        "symbol": "BTC/USDT",
        "requested_leverage": 3.0,
        "system_max_leverage": 20.0,
        "target_notional_usdt": 240.0,
        "available_margin_usdt": 100.0,
        "stressed_loss_fraction": 0.02,
        "expected_net_return_pct": 1.2,
        "return_lcb_pct": 0.8,
        "expected_loss_pct": 0.3,
        "profit_quality_ratio": 1.2,
        "loss_probability": 0.38,
        "tail_risk_score": 0.30,
        "aligned_source_count": 3,
        "atr_pct": 0.006,
        "execution_cost": {
            "production_eligible": True,
            "slippage_pct": 0.02,
            "spread_pct": 0.02,
        },
        "portfolio_capacity_fraction": 0.96,
    }
    base.update(overrides)
    return DynamicLeverageInput(**base)


def test_dynamic_leverage_lifts_exploration_signal_without_fixed_three_x_cap():
    decision = DynamicLeverageAllocator().allocate(_input())

    assert decision.final_integer_leverage > 3
    assert decision.final_integer_leverage == int(decision.final_integer_leverage)
    assert decision.limiting_factor in {
        "volatility",
        "signal_quality",
        "risk_budget",
        "liquidity",
        "history",
        "portfolio",
        "system_max",
    }


def test_dynamic_leverage_continuously_tempers_weak_return_quality():
    decision = DynamicLeverageAllocator().allocate(
        _input(
            requested_leverage=10.0,
            expected_net_return_pct=0.20,
            profit_quality_ratio=0.40,
            loss_probability=0.55,
            tail_risk_score=0.78,
            aligned_source_count=3,
        )
    )

    assert decision.final_integer_leverage < 10
    assert decision.rounding_policy == "floor_to_exchange_integer"
    assert decision.policy_provenance["production_eligible"] is True
    assert decision.policy_provenance["fallback_reason"] == ""


def test_dynamic_leverage_observes_required_margin_without_using_size_to_set_budget():
    decision = DynamicLeverageAllocator().allocate(
        _input(
            requested_leverage=9.7,
            target_notional_usdt=375.0,
            available_margin_usdt=100.0,
        )
    )

    assert decision.required_margin_leverage == 3.75
    assert decision.final_integer_leverage <= decision.system_max_leverage


def test_requested_leverage_is_not_the_exchange_system_maximum() -> None:
    low_request = DynamicLeverageAllocator().allocate(_input(requested_leverage=2.0))
    high_request = DynamicLeverageAllocator().allocate(_input(requested_leverage=12.0))

    assert low_request.system_max_leverage == high_request.system_max_leverage == 20.0
    assert low_request.final_integer_leverage == high_request.final_integer_leverage


def test_dynamic_leverage_missing_cost_distribution_falls_back_to_one_x() -> None:
    decision = DynamicLeverageAllocator().allocate(_input(execution_cost={}))

    assert decision.final_integer_leverage == 1
    assert decision.liquidity_leverage == 1.0
    assert decision.policy_provenance["production_eligible"] is False
    assert decision.policy_provenance["fallback_reason"] == "live_execution_cost_incomplete"
