from services.entry_strategy_mode import EntryStrategyModeContextPolicy


def _base_kwargs(**overrides):
    values = {
        "market_regime": {"mode": "range", "confidence": 0.4},
        "daily_state": {"today_total_pnl": 5.0, "today_high_water_pnl": 6.0},
        "side_performance": {},
        "side_performance_multiday": {},
        "symbol_side_performance": {},
        "model_contribution_performance": {},
        "position_exposure": {"dominant_side": "long", "net_ratio": 0.8},
        "position_group_count": 3,
        "account_equity": 1000.0,
        "account_config": {},
    }
    values.update(overrides)
    return values


def test_strategy_context_is_observation_only() -> None:
    result = EntryStrategyModeContextPolicy().build(**_base_kwargs())

    assert result["strategy"] == "authoritative_return_capture"
    assert result["goal"] == "maximize_realized_fee_after_return"
    assert result["policy_provenance"]["source"] == (
        "account_pnl_side_returns_and_current_portfolio_state"
    )
    assert "allow_long" not in result
    assert "allow_short" not in result
    assert "blocked_directions" not in result
    assert "strategy_profile_id" not in result
    assert "min_opportunity_score" not in result


def test_drawdown_changes_observation_pressure_without_switching_strategy() -> None:
    low = EntryStrategyModeContextPolicy().build(
        **_base_kwargs(daily_state={"today_total_pnl": 0.0, "today_high_water_pnl": 1.0})
    )
    high = EntryStrategyModeContextPolicy().build(
        **_base_kwargs(daily_state={"today_total_pnl": -40.0, "today_high_water_pnl": 10.0})
    )

    assert high["drawdown_pressure"] > low["drawdown_pressure"]
    assert high["strategy"] == low["strategy"] == "authoritative_return_capture"


def test_side_history_cannot_create_direction_or_size_permissions() -> None:
    result = EntryStrategyModeContextPolicy().build(
        **_base_kwargs(
            side_performance={
                "long": {"count": 4, "pnl": -18.0, "profit_factor": 0.2},
                "short": {"count": 4, "pnl": 9.0, "profit_factor": 2.0},
            },
            side_performance_multiday={
                "long": {"count": 10, "pnl": -30.0, "return_lcb_pct": -2.0},
                "short": {"count": 10, "pnl": 25.0, "return_lcb_pct": 1.0},
            },
        )
    )

    for side in ("long", "short"):
        observation = result["side_quality"][side]
        assert observation["production_permission"] is False
        assert "position_size_multiplier" not in observation
        assert "score_adjustment" not in observation
    assert "preferred_direction" not in result
