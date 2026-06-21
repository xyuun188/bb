from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from ai_brain.base_model import Action, DecisionOutput
from executor.base_executor import ExecutionResult, OrderStatus
from services.trade_order_log_service import TradeOrderLogService


class FakeSessionContext:
    def __init__(self, session: Any) -> None:
        self.session = session

    async def __aenter__(self) -> Any:
        return self.session

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False


class FakeTradeRepo:
    def __init__(self) -> None:
        self.orders: list[dict[str, Any]] = []

    async def create_order(self, data: dict[str, Any]) -> None:
        self.orders.append(data)


def _decision() -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.LONG,
        confidence=0.7,
        reasoning="test",
        position_size_pct=0.03,
        suggested_leverage=2.0,
    )


@pytest.mark.asyncio
async def test_trade_order_log_service_persists_order_payload() -> None:
    repo = FakeTradeRepo()
    filled_at = datetime(2026, 6, 10, 1, 2, tzinfo=UTC)
    result = ExecutionResult(
        order_id="local-1",
        exchange_order_id="okx-1",
        symbol="BTC/USDT",
        side="buy",
        order_type="market",
        quantity=2.5,
        price=101.2,
        status=OrderStatus.FILLED,
        fee=0.12,
        timestamp=filled_at,
    )
    service = TradeOrderLogService(
        execution_mode_provider=lambda model_name: f"mode:{model_name}",
        session_context_factory=lambda: FakeSessionContext(object()),
        trade_repo_factory=lambda _session: repo,
    )

    await service.log_trade(result, "ensemble_trader", _decision(), decision_id=77)

    assert repo.orders == [
        {
            "model_name": "ensemble_trader",
            "execution_mode": "mode:ensemble_trader",
            "symbol": "BTC/USDT",
            "side": "buy",
            "order_type": "market",
            "quantity": 2.5,
            "price": 101.2,
            "status": "filled",
            "fee": 0.12,
            "decision_id": 77,
            "exchange_order_id": "okx-1",
            "filled_at": filled_at,
        }
    ]


@pytest.mark.asyncio
async def test_trade_order_log_service_skips_zero_quantity_tracking_order() -> None:
    repo = FakeTradeRepo()
    result = ExecutionResult(
        order_id="exit_tracking",
        exchange_order_id="okx-exit-1",
        symbol="PROS/USDT",
        side="sell",
        order_type="market",
        quantity=0.0,
        price=0.5666,
        status=OrderStatus.OPEN,
        raw_response={"exit_tracking": True, "existing_exit_order": True},
    )
    service = TradeOrderLogService(
        execution_mode_provider=lambda model_name: f"mode:{model_name}",
        session_context_factory=lambda: FakeSessionContext(object()),
        trade_repo_factory=lambda _session: repo,
    )

    await service.log_trade(result, "ensemble_trader", _decision(), decision_id=78)

    assert repo.orders == []


@pytest.mark.asyncio
async def test_trade_order_log_service_keeps_rejected_zero_quantity_diagnostics() -> None:
    repo = FakeTradeRepo()
    result = ExecutionResult(
        order_id="rejected",
        symbol="PROS/USDT",
        side="buy",
        order_type="market",
        quantity=0.0,
        price=0.5666,
        status=OrderStatus.REJECTED,
        raw_response={"error": "pre-submit rejected"},
    )
    service = TradeOrderLogService(
        execution_mode_provider=lambda model_name: f"mode:{model_name}",
        session_context_factory=lambda: FakeSessionContext(object()),
        trade_repo_factory=lambda _session: repo,
    )

    await service.log_trade(result, "ensemble_trader", _decision(), decision_id=79)

    assert len(repo.orders) == 1
    assert repo.orders[0]["status"] == "rejected"
    assert repo.orders[0]["quantity"] == 0.0
