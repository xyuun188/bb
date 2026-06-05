from risk_manager.position_limits import PositionLimitChecker


def test_contract_exposure_uses_okx_contract_size_for_swap_positions():
    checker = PositionLimitChecker()
    position = {
        "symbol": "BCH/USDT",
        "side": "short",
        "quantity": 1000,
        "contracts": 1000,
        "contract_size": 0.01,
        "entry_price": 400,
        "leverage": 10,
        "is_open": True,
    }

    snapshot = checker.contract_exposure_snapshot(
        proposed_margin_pct=0.0,
        proposed_leverage=5,
        proposed_stop_loss_pct=0.05,
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
        proposed_margin_pct=0.0,
        proposed_leverage=5,
        proposed_stop_loss_pct=0.05,
        current_positions=[position],
        account_balance=5000,
    )

    assert snapshot.current_notional_pct == 0.8
    assert snapshot.current_margin_usdt == 286.5
    assert snapshot.current_margin_pct == 0.0573
