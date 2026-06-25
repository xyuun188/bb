from risk_manager.position_limits import PositionLimitChecker


def test_position_limit_checker_reads_runtime_settings(monkeypatch):
    checker = PositionLimitChecker()
    monkeypatch.setattr("risk_manager.position_limits.settings.max_position_pct", 0.10)
    monkeypatch.setattr("risk_manager.position_limits.settings.max_leverage", 4.0)

    assert checker.max_position_pct == 0.10
    assert checker.max_leverage == 4.0
    result = checker.check_leverage(6.0)

    assert result.adjusted_size_pct == 4.0


def test_contract_exposure_snapshot_reads_runtime_hard_stop(monkeypatch):
    checker = PositionLimitChecker()
    monkeypatch.setattr("risk_manager.position_limits.settings.hard_stop_loss_pct", 0.02)

    snapshot = checker.contract_exposure_snapshot(
        proposed_margin_pct=0.10,
        proposed_leverage=5.0,
        proposed_stop_loss_pct=0.0,
        current_positions=[],
        account_balance=1000,
    )

    assert snapshot.proposed_stop_risk_pct == 0.01


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
