from __future__ import annotations

from typing import Any

import pytest

from ai_brain.base_model import DecisionOutput
from executor.base_executor import AbstractExecutor, ExecutionResult, OrderStatus
from executor.position_tracker import PositionTracker


class _SoftOnlyExecutor(AbstractExecutor):
    def __init__(self) -> None:
        self.position_calls = 0
        self.open_order_calls = 0

    async def place_order(
        self,
        decision: DecisionOutput,
        account_id: str | None = None,
    ) -> ExecutionResult:
        return ExecutionResult(
            order_id="noop",
            symbol=decision.symbol,
            side="hold",
            order_type="market",
            quantity=0.0,
            price=0.0,
            status=OrderStatus.REJECTED,
        )

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        return False

    async def get_balance(self, asset: str = "USDT") -> float:
        return 0.0

    async def get_positions(self, symbol: str | None = None) -> list[dict]:
        self.position_calls += 1
        return [{"symbol": symbol or "BTC/USDT"}]

    async def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        self.open_order_calls += 1
        return [{"symbol": symbol or "BTC/USDT"}]

    async def initialize(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None


class _StrictPositionExecutor:
    def __init__(self, rows: list[dict[str, Any]] | None = None, error: Exception | None = None):
        self.rows = rows or []
        self.error = error
        self.strict_calls = 0
        self.soft_calls = 0

    async def get_positions(self) -> list[dict[str, Any]]:
        self.soft_calls += 1
        raise AssertionError("position tracker must not use soft get_positions")

    async def get_positions_strict(self) -> list[dict[str, Any]]:
        self.strict_calls += 1
        if self.error is not None:
            raise self.error
        return self.rows


@pytest.mark.asyncio
async def test_base_executor_strict_methods_do_not_delegate_to_soft_reads() -> None:
    executor = _SoftOnlyExecutor()

    with pytest.raises(NotImplementedError, match="strict authoritative positions"):
        await executor.get_positions_strict("BTC/USDT")
    with pytest.raises(NotImplementedError, match="strict authoritative open orders"):
        await executor.get_open_orders_strict("BTC/USDT")

    assert executor.position_calls == 0
    assert executor.open_order_calls == 0


@pytest.mark.asyncio
async def test_position_tracker_sync_uses_strict_positions_only() -> None:
    tracker = PositionTracker()
    executor = _StrictPositionExecutor(
        [
            {
                "symbol": "BTC/USDT",
                "side": "long",
                "contracts": 2.0,
                "entryPrice": 100.0,
                "leverage": 5.0,
                "unrealizedPnl": 3.5,
            }
        ]
    )

    await tracker.sync_from_executor(executor, model_name="ensemble_trader")

    assert executor.strict_calls == 1
    assert executor.soft_calls == 0
    assert tracker.get_positions("ensemble_trader") == [
        {
            "symbol": "BTC/USDT",
            "side": "long",
            "quantity": 2.0,
            "entry_price": 100.0,
            "leverage": 5.0,
            "unrealized_pnl": 3.5,
            "is_open": True,
        }
    ]


@pytest.mark.asyncio
async def test_position_tracker_clears_stale_positions_when_strict_sync_fails() -> None:
    tracker = PositionTracker()
    tracker.add_position(
        "ensemble_trader",
        {"id": "old", "symbol": "OLD/USDT", "side": "long", "is_open": True},
    )
    executor = _StrictPositionExecutor(error=RuntimeError("OKX native snapshot unavailable"))

    await tracker.sync_from_executor(executor, model_name="ensemble_trader")

    assert executor.strict_calls == 1
    assert executor.soft_calls == 0
    assert tracker.get_positions("ensemble_trader") == []
