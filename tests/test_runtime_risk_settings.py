from risk_manager.circuit_breaker import BreakerState, CircuitBreaker


def test_trade_losses_are_diagnostic_not_a_fixed_pnl_gate() -> None:
    breaker = CircuitBreaker()

    for _ in range(200):
        breaker.record_trade(-100.0)

    assert breaker.state == BreakerState.CLOSED
    assert breaker.get_state()["consecutive_losses"] == 200
    assert breaker.get_state()["daily_pnl"] == -20000.0


def test_profit_resets_diagnostic_consecutive_loss_counter() -> None:
    breaker = CircuitBreaker()
    breaker.record_trade(-1.0)
    breaker.record_trade(0.5)

    assert breaker.state == BreakerState.CLOSED
    assert breaker.get_state()["consecutive_losses"] == 0
