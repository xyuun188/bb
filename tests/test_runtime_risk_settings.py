from risk_manager.circuit_breaker import BreakerState, CircuitBreaker
from risk_manager.stop_loss import StopLossManager, StopLossType


def test_circuit_breaker_reads_runtime_daily_loss_setting(monkeypatch):
    breaker = CircuitBreaker()
    breaker.record_trade(-4.0)

    monkeypatch.setattr("risk_manager.circuit_breaker.settings.max_daily_loss_pct", 0.03)
    breaker.evaluate_daily_loss(account_balance=100.0)

    assert breaker.state == BreakerState.OPEN


def test_stop_loss_manager_reads_runtime_hard_stop_setting(monkeypatch):
    manager = StopLossManager()

    monkeypatch.setattr("risk_manager.stop_loss.settings.hard_stop_loss_pct", 0.02)
    result = manager.evaluate(
        symbol="BTC/USDT",
        side="long",
        entry_price=100.0,
        current_price=97.9,
    )

    assert result.triggered is True
    assert result.stop_type == StopLossType.HARD
