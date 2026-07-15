from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from ai_brain.base_model import Action, DecisionOutput
from executor.base_executor import ExecutionResult, OrderStatus
from services.trade_order_log_service import TradeOrderLogService, TradeOrderPersistenceError


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


class FailingTradeRepo(FakeTradeRepo):
    async def create_order(self, data: dict[str, Any]) -> None:
        raise RuntimeError("database write failed")


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
async def test_trade_order_log_service_surfaces_persistence_failure() -> None:
    service = TradeOrderLogService(
        execution_mode_provider=lambda _model_name: "paper",
        session_context_factory=lambda: FakeSessionContext(object()),
        trade_repo_factory=lambda _session: FailingTradeRepo(),
    )
    result = ExecutionResult(
        order_id="local-1",
        exchange_order_id="okx-1",
        symbol="BTC/USDT",
        side="buy",
        order_type="market",
        quantity=2.5,
        price=101.2,
        status=OrderStatus.FILLED,
    )

    with pytest.raises(TradeOrderPersistenceError):
        await service.log_trade(result, "ensemble_trader", _decision(), decision_id=77)


@pytest.mark.asyncio
async def test_trade_order_log_service_persists_structured_okx_rejection_fact() -> None:
    repo = FakeTradeRepo()
    rejected_at = datetime(2026, 7, 11, 1, 2, tzinfo=UTC)
    result = ExecutionResult(
        order_id="okx_rejected",
        symbol="BTC/USDT",
        side="buy",
        order_type="market",
        quantity=0.0,
        price=101.2,
        status=OrderStatus.REJECTED,
        timestamp=rejected_at,
        raw_response={
            "okx_rejection": True,
            "okx_symbol": "BTC-USDT-SWAP",
            "raw_error": "OKX API error [59247]: Operation failed",
            "okx_error_code": "59247",
            "okx_error_payload": {
                "code": "0",
                "data": [{"sCode": "59247", "sMsg": "Operation failed"}],
            },
            "request_params": {"tdMode": "cross", "posSide": "long"},
            "okx_order_rules": {"pre_submit_valid": True},
            "leverage_check": {"actual_leverage": 2},
        },
    )
    service = TradeOrderLogService(
        execution_mode_provider=lambda _model_name: "paper",
        session_context_factory=lambda: FakeSessionContext(object()),
        trade_repo_factory=lambda _session: repo,
    )

    await service.log_trade(result, "ensemble_trader", _decision(), decision_id=78)

    order = repo.orders[0]
    assert order["status"] == "rejected"
    assert order["okx_inst_id"] == "BTC-USDT-SWAP"
    assert order["okx_state"] == "rejected_no_exchange_fill"
    assert order["okx_sync_status"] == "okx_rejected_no_fill"
    assert order["okx_synced_at"] == rejected_at
    assert "59247" in order["okx_last_error"]
    assert order["okx_raw_fills"]["error_code"] == "59247"
    assert order["okx_raw_fills"]["error_payload"] == result.raw_response[
        "okx_error_payload"
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
async def test_trade_order_log_service_native_inst_id_overrides_wrong_canonical_alias() -> None:
    repo = FakeTradeRepo()
    decision = _decision()
    decision.symbol = "SPK/USDT"
    result = ExecutionResult(
        order_id="spk-entry",
        exchange_order_id="spk-entry",
        symbol="SAHARA/USDT",
        side="buy",
        order_type="market",
        quantity=10.0,
        price=0.012,
        status=OrderStatus.FILLED,
        raw_response={
            "symbol": "SAHARA/USDT:USDT",
            "canonical_exchange_symbol": "SAHARA/USDT",
            "info": {"instId": "SPK-USDT-SWAP", "ordId": "spk-entry"},
        },
    )
    service = TradeOrderLogService(
        execution_mode_provider=lambda model_name: f"mode:{model_name}",
        session_context_factory=lambda: FakeSessionContext(object()),
        trade_repo_factory=lambda _session: repo,
    )

    await service.log_trade(result, "ensemble_trader", decision, decision_id=83)

    assert repo.orders[0]["symbol"] == "SPK/USDT"


@pytest.mark.asyncio
async def test_trade_order_log_service_persists_okx_execution_result_facts() -> None:
    repo = FakeTradeRepo()
    filled_at = datetime(2026, 7, 1, 6, 46, tzinfo=UTC)
    result = ExecutionResult(
        order_id="3703940352525967360",
        exchange_order_id="3703940352525967360",
        symbol="ACT-USDT-SWAP",
        side="buy",
        order_type="market",
        quantity=1580.0,
        price=0.0097,
        status=OrderStatus.FILLED,
        fee=0.007663,
        pnl=0.4582,
        timestamp=filled_at,
        raw_response={
            "id": "3703940352525967360",
            "filled_contracts": 158.0,
            "filled": 158.0,
            "average": 0.0097,
            "protection_submission": {
                "source_authority": "local_submit_plus_okx_create_order_response",
                "exchange_confirmation_recorded": True,
                "algo_ids": ["algo-act-1"],
            },
            "info": {
                "ordId": "3703940352525967360",
                "instId": "ACT-USDT-SWAP",
                "side": "buy",
                "state": "filled",
                "accFillSz": "158",
                "fillSz": "9",
                "avgPx": "0.0097",
                "fee": "-0.007663",
                "pnl": "0.4582",
                "tradeId": "535631715",
            },
        },
    )
    decision = _decision()
    decision.symbol = "ACT/USDT"
    service = TradeOrderLogService(
        execution_mode_provider=lambda model_name: f"mode:{model_name}",
        session_context_factory=lambda: FakeSessionContext(object()),
        trade_repo_factory=lambda _session: repo,
    )

    await service.log_trade(result, "ensemble_trader", decision, decision_id=18794)

    order = repo.orders[0]
    assert order["symbol"] == "ACT/USDT"
    assert order["okx_inst_id"] == "ACT-USDT-SWAP"
    assert order["okx_sync_status"] == "okx_execution_result_confirmed"
    assert order["okx_state"] == "filled"
    assert order["okx_trade_ids"] == "535631715"
    assert order["okx_fill_contracts"] == pytest.approx(158.0)
    assert order["okx_fill_pnl"] == pytest.approx(0.4582)
    assert order["okx_raw_fills"]["execution_result_confirmed"] is True
    assert order["okx_raw_fills"]["order_id"] == "3703940352525967360"
    assert order["okx_raw_fills"]["inst_id"] == "ACT-USDT-SWAP"
    assert order["okx_raw_fills"]["protection_submission"]["algo_ids"] == [
        "algo-act-1"
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
async def test_trade_order_log_service_skips_native_full_close_without_order_id() -> (
    None
):
    repo = FakeTradeRepo()
    filled_at = datetime(2026, 6, 25, 20, 54, tzinfo=UTC)
    result = ExecutionResult(
        order_id="okx_native_full_close",
        exchange_order_id=None,
        symbol="AI16Z/USDT",
        side="sell",
        order_type="market",
        quantity=366.0,
        price=0.0513,
        status=OrderStatus.FILLED,
        timestamp=filled_at,
        raw_response={
            "okx_native_close_position": True,
            "position_contracts_before": 36.6,
            "position_contracts_after": 0.0,
            "remaining_contracts": 0.0,
            "filled_contracts": 36.6,
        },
    )
    decision = _decision()
    decision.symbol = "AI16Z/USDT"
    service = TradeOrderLogService(
        execution_mode_provider=lambda model_name: f"mode:{model_name}",
        session_context_factory=lambda: FakeSessionContext(object()),
        trade_repo_factory=lambda _session: repo,
    )

    await service.log_trade(result, "ensemble_trader", decision, decision_id=132611)

    assert repo.orders == []


@pytest.mark.asyncio
async def test_trade_order_log_service_persists_native_full_close_pending_backfill() -> None:
    repo = FakeTradeRepo()
    filled_at = datetime(2026, 6, 25, 20, 54, tzinfo=UTC)
    result = ExecutionResult(
        order_id="okx_native_full_close_fill_pending",
        exchange_order_id=None,
        symbol="AI16Z/USDT",
        side="sell",
        order_type="market",
        quantity=366.0,
        price=0.0513,
        status=OrderStatus.PARTIAL,
        timestamp=filled_at,
        raw_response={
            "exit_tracking": True,
            "okx_native_close_position": True,
            "requires_okx_fill_backfill": True,
            "request_params": {"instId": "AI16Z-USDT-SWAP"},
            "position_contracts_before": 36.6,
            "position_contracts_after": 0.0,
            "remaining_contracts": 0.0,
            "filled_contracts": 36.6,
            "contract_size": 10.0,
            "base_quantity": 366.0,
        },
    )
    decision = _decision()
    decision.symbol = "AI16Z/USDT"
    service = TradeOrderLogService(
        execution_mode_provider=lambda model_name: f"mode:{model_name}",
        session_context_factory=lambda: FakeSessionContext(object()),
        trade_repo_factory=lambda _session: repo,
    )

    await service.log_trade(result, "ensemble_trader", decision, decision_id=132611)

    assert repo.orders[0]["exchange_order_id"] is None
    assert repo.orders[0]["status"] == "partial"
    assert repo.orders[0]["okx_inst_id"] == "AI16Z-USDT-SWAP"
    assert repo.orders[0]["okx_sync_status"] == "okx_native_full_close_pending_backfill"
    assert repo.orders[0]["okx_raw_fills"]["requires_okx_fill_backfill"] is True


@pytest.mark.asyncio
async def test_trade_order_log_service_persists_native_full_close_with_real_fill_order_id() -> (
    None
):
    repo = FakeTradeRepo()
    filled_at = datetime(2026, 6, 25, 20, 54, tzinfo=UTC)
    result = ExecutionResult(
        order_id="native-close-order",
        exchange_order_id="native-close-order",
        symbol="AI16Z/USDT",
        side="sell",
        order_type="market",
        quantity=366.0,
        price=0.0513,
        status=OrderStatus.FILLED,
        timestamp=filled_at,
        raw_response={
            "okx_native_close_position": True,
            "position_contracts_before": 36.6,
            "position_contracts_after": 0.0,
            "remaining_contracts": 0.0,
            "filled_contracts": 36.6,
            "native_close_fill": {"order_id": "native-close-order"},
            "info": {"instId": "AI16Z-USDT-SWAP"},
        },
    )
    decision = _decision()
    decision.symbol = "AI16Z/USDT"
    service = TradeOrderLogService(
        execution_mode_provider=lambda model_name: f"mode:{model_name}",
        session_context_factory=lambda: FakeSessionContext(object()),
        trade_repo_factory=lambda _session: repo,
    )

    await service.log_trade(result, "ensemble_trader", decision, decision_id=132611)

    assert repo.orders[0]["exchange_order_id"] == "native-close-order"
    assert repo.orders[0]["symbol"] == "AI16Z/USDT"


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
