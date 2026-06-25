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
async def test_trade_order_log_service_uses_okx_inst_id_symbol_over_ccxt_alias() -> None:
    repo = FakeTradeRepo()
    result = ExecutionResult(
        order_id="okx-h-1",
        exchange_order_id="okx-h-1",
        symbol="WLFI/USDT",
        side="sell",
        order_type="market",
        quantity=1176.0,
        price=0.0817,
        status=OrderStatus.FILLED,
        raw_response={
            "symbol": "WLFI/USDT:USDT",
            "info": {"instId": "H-USDT-SWAP"},
            "canonical_exchange_symbol": "H/USDT",
        },
    )
    service = TradeOrderLogService(
        execution_mode_provider=lambda model_name: f"mode:{model_name}",
        session_context_factory=lambda: FakeSessionContext(object()),
        trade_repo_factory=lambda _session: repo,
    )

    await service.log_trade(result, "ensemble_trader", _decision(), decision_id=82)

    assert repo.orders[0]["symbol"] == "H/USDT"


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
async def test_trade_order_log_service_skips_unconfirmed_filled_order() -> None:
    repo = FakeTradeRepo()
    result = ExecutionResult(
        order_id="local-paper-fill",
        exchange_order_id=None,
        symbol="USAR/USDT",
        side="sell",
        order_type="market",
        quantity=10.0,
        price=3.85,
        status=OrderStatus.FILLED,
        raw_response={},
    )
    service = TradeOrderLogService(
        execution_mode_provider=lambda model_name: f"mode:{model_name}",
        session_context_factory=lambda: FakeSessionContext(object()),
        trade_repo_factory=lambda _session: repo,
    )

    await service.log_trade(result, "ensemble_trader", _decision(), decision_id=80)

    assert repo.orders == []


@pytest.mark.asyncio
async def test_trade_order_log_service_skips_unconfirmed_partial_order() -> None:
    repo = FakeTradeRepo()
    result = ExecutionResult(
        order_id="local-partial-fill",
        exchange_order_id=None,
        symbol="BTC/USDT",
        side="sell",
        order_type="market",
        quantity=0.0046,
        price=104000.0,
        status=OrderStatus.PARTIAL,
        raw_response={},
    )
    service = TradeOrderLogService(
        execution_mode_provider=lambda model_name: f"mode:{model_name}",
        session_context_factory=lambda: FakeSessionContext(object()),
        trade_repo_factory=lambda _session: repo,
    )

    await service.log_trade(result, "ensemble_trader", _decision(), decision_id=81)

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


@pytest.mark.asyncio
async def test_trade_order_log_service_uses_decision_symbol_for_unconfirmed_rejection() -> None:
    repo = FakeTradeRepo()
    decision = _decision()
    decision.symbol = "SPK/USDT"
    result = ExecutionResult(
        order_id="rejected",
        symbol="SAHARA/USDT",
        side="sell",
        order_type="market",
        quantity=0.0,
        price=0.0182,
        status=OrderStatus.REJECTED,
        raw_response={
            "okx_symbol": "SAHARA/USDT:USDT",
            "canonical_exchange_symbol": "SAHARA/USDT",
            "execution_blocker": "system_pre_submit_market_order_max",
        },
    )
    service = TradeOrderLogService(
        execution_mode_provider=lambda model_name: f"mode:{model_name}",
        session_context_factory=lambda: FakeSessionContext(object()),
        trade_repo_factory=lambda _session: repo,
    )

    await service.log_trade(result, "ensemble_trader", decision, decision_id=132210)

    assert repo.orders[0]["symbol"] == "SPK/USDT"
