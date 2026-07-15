from risk_manager.position_limits import PositionLimitChecker


def _contract(**overrides):
    contract = {
        "account_equity_usdt": 1000.0,
        "available_margin_usdt": 250.0,
        "final_margin_usdt": 10.0,
        "final_notional_usdt": 250.0,
        "planned_stressed_loss_usdt": 5.0,
        "stressed_loss_fraction": 0.02,
        "portfolio_risk_budget_usdt": 20.0,
        "exchange_contract_specs": {},
    }
    contract.update(overrides)
    return contract


def test_position_limit_checker_uses_dynamic_margin_and_stress_budgets():
    checker = PositionLimitChecker()
    result = checker.check_contract_entry_limits(
        risk_contract=_contract(),
        current_positions=[],
        account_balance=1000.0,
        symbol="BTC/USDT",
    )

    assert result.passed is True


def test_missing_dynamic_stress_does_not_restore_fixed_notional_capacity():
    checker = PositionLimitChecker()

    snapshot = checker.contract_exposure_snapshot(
        risk_contract=_contract(
            stressed_loss_fraction=0.0,
            planned_stressed_loss_usdt=0.0,
        ),
        current_positions=[],
        account_balance=1000,
    )

    assert snapshot.proposed_stop_risk_pct == 0.0
    assert snapshot.notional_limit_pct == 0.0


def test_contract_exposure_uses_okx_contract_size_for_swap_positions():
    checker = PositionLimitChecker()
    position = {
        "symbol": "BCH/USDT",
        "side": "short",
        "quantity": 1000,
        "contracts": 1000,
        "contract_size": 0.005,
        "info": {"instId": "BCH-USDT-SWAP"},
        "entry_price": 400,
        "leverage": 10,
        "is_open": True,
    }

    snapshot = checker.contract_exposure_snapshot(
        risk_contract=_contract(
            account_equity_usdt=5000.0,
            available_margin_usdt=5000.0,
            final_margin_usdt=0.0,
            final_notional_usdt=0.0,
            planned_stressed_loss_usdt=0.0,
            exchange_contract_specs={
                "BCH-USDT-SWAP": {"ctVal": "0.005", "ctMult": "2"}
            },
        ),
        current_positions=[position],
        account_balance=5000,
    )

    assert snapshot.current_notional_pct == 0.8
    assert snapshot.current_margin_usdt == 400
    assert snapshot.current_margin_pct == 0.08


def test_contract_exposure_prefers_exchange_margin_when_available():
    checker = PositionLimitChecker()
    position = {
        "symbol": "BCH/USDT",
        "side": "short",
        "quantity": 1000,
        "entry_price": 400,
        "leverage": 10,
        "notional": 4000,
        "margin": 286.5,
        "is_open": True,
    }

    snapshot = checker.contract_exposure_snapshot(
        risk_contract=_contract(
            account_equity_usdt=5000.0,
            available_margin_usdt=5000.0,
            final_margin_usdt=0.0,
            final_notional_usdt=0.0,
            planned_stressed_loss_usdt=0.0,
        ),
        current_positions=[position],
        account_balance=5000,
    )

    assert snapshot.current_notional_pct == 0.8
    assert snapshot.current_margin_usdt == 286.5
    assert snapshot.current_margin_pct == 0.0573
