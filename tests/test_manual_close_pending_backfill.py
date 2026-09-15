from datetime import UTC, datetime

from executor.base_executor import ExecutionResult, OrderStatus
from services.trading_service import TradingService


def _result(*, remaining: float) -> ExecutionResult:
    return ExecutionResult(
        order_id="okx_native_full_close_fill_pending",
        symbol="ADA/USDT",
        side="sell",
        order_type="market",
        quantity=719.0,
        price=0.2035,
        status=OrderStatus.PARTIAL,
        timestamp=datetime.now(UTC),
        raw_response={
            "exit_tracking": True,
            "okx_native_close_position": True,
            "requires_okx_fill_backfill": True,
            "position_contracts_after": remaining,
            "remaining_contracts": remaining,
        },
    )


def test_manual_close_accepts_native_full_close_when_exchange_is_flat() -> None:
    assert TradingService._manual_close_exchange_flat_pending_backfill(_result(remaining=0.0))


def test_manual_close_rejects_pending_backfill_when_exchange_position_remains() -> None:
    assert not TradingService._manual_close_exchange_flat_pending_backfill(_result(remaining=1.0))


def test_manual_close_does_not_persist_synthetic_exchange_order_id() -> None:
    result = _result(remaining=0.0)
    assert TradingService._manual_close_exchange_order_id(result) == ""
