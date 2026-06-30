from services.dynamic_leverage_allocator import DynamicLeverageAllocator, DynamicLeverageInput


def _input(**overrides):
    base = {
        "symbol": "BTC/USDT",
        "requested_leverage": 3.0,
        "system_max_leverage": 20.0,
        "balance": 1000.0,
        "position_size_pct": 0.02,
        "stress_stop_loss_pct": 0.02,
        "max_loss_usdt": 16.0,
        "expected_net_return_pct": 1.2,
        "profit_quality_ratio": 1.2,
        "loss_probability": 0.38,
        "tail_risk_score": 0.30,
        "score": 3.0,
        "min_score_required": 0.95,
        "confidence": 0.82,
        "aligned_source_count": 3,
        "evidence_tier": "exploration",
        "evidence_effective_score": 48.0,
        "low_payoff_quality": False,
        "weak_history": False,
        "negative_local_expected": False,
        "symbol_profit_tier": "neutral",
        "quality_tier": "base",
        "high_quality_entry": True,
        "atr_pct": 0.006,
        "execution_cost": {},
        "open_positions_count": 1,
        "portfolio_exposure_pct": 0.04,
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


def test_dynamic_leverage_temper_risk_flags_instead_of_preserving_requested():
    decision = DynamicLeverageAllocator().allocate(
        _input(
            requested_leverage=10.0,
            expected_net_return_pct=0.20,
            profit_quality_ratio=0.40,
            loss_probability=0.55,
            tail_risk_score=0.78,
            score=0.80,
            aligned_source_count=0,
            low_payoff_quality=True,
            high_quality_entry=False,
            quality_tier="probe",
        )
    )

    assert decision.final_integer_leverage < 10
    assert decision.rounding_policy == "floor_for_risk"
    assert "tempered_by_risk_flags" in decision.reasons


def test_dynamic_leverage_clamps_to_integer_risk_budget():
    decision = DynamicLeverageAllocator().allocate(
        _input(
            requested_leverage=9.7,
            balance=1000.0,
            position_size_pct=0.08,
            stress_stop_loss_pct=0.03,
            max_loss_usdt=9.0,
        )
    )

    assert decision.risk_budget_leverage == 3.75
    assert decision.final_integer_leverage == 3
