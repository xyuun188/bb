from services.analysis_budget import AnalysisBudgetPolicy


def _policy() -> AnalysisBudgetPolicy:
    return AnalysisBudgetPolicy(
        normalize_symbol=lambda value: str(value),
        open_position_group_counter=lambda rows: len(rows or []),
        portfolio_profit_context_provider=lambda _rows: {},
        position_review_scanner=lambda *_args: {},
        urgent_exit_checker=lambda _scan: False,
    )


def test_analysis_budget_is_operational_not_entry_permission() -> None:
    result = _policy().context(
        [],
        {},
        base_market_limit=20,
        run_position_analysis=False,
        run_market_analysis=True,
    )
    assert 0 < result["market_symbol_limit"] <= 20
    assert result["market_symbol_limit_is_entry_gate"] is False
    assert result["market_limit_diagnostics"]["read_only"] is True


def test_strategy_learning_cannot_restore_fixed_roster_targets() -> None:
    result = _policy().context(
        [{"symbol": "BTC/USDT", "model_name": "ensemble_trader"}],
        {},
        base_market_limit=8,
        run_position_analysis=False,
        run_market_analysis=True,
        strategy_context={
            "target_position_groups": 999,
            "strategy_learning": {
                "runtime": {
                    "target_position_groups": 999,
                    "roster_fill_market_symbol_min": 999,
                }
            },
        },
    )
    assert result["target_position_groups"] == 0
    assert result["roster_underfilled"] is False
    assert result["market_symbol_limit"] <= 8


def test_account_pause_still_zeroes_analysis_budget() -> None:
    result = _policy().context(
        [],
        {},
        base_market_limit=8,
        run_position_analysis=False,
        run_market_analysis=True,
        new_pair_pause_reason="account hard risk",
    )
    assert result["market_symbol_limit"] == 0
    assert result["market_limit_policy"] == "new_pair_pause"
