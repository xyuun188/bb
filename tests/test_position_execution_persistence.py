from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from ai_brain.base_model import Action, DecisionOutput
from executor.base_executor import OrderStatus
from services.entry_fee_provider import proportional_fee
from services.position_execution_persistence import PositionExecutionPersistenceService


class FakeSession:
    def __init__(self) -> None:
        self.flush_count = 0

    async def flush(self) -> None:
        self.flush_count += 1


class FakeSessionContext:
    def __init__(self, session: FakeSession) -> None:
        self.session = session

    async def __aenter__(self) -> FakeSession:
        return self.session

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False


class FakeTradeRepo:
    def __init__(self, positions: list[Any] | None = None) -> None:
        self.positions = positions or []
        self.opened: list[dict[str, Any]] = []

    async def open_position(self, data: dict[str, Any]) -> Any:
        self.opened.append(data)
        return SimpleNamespace(id=100 + len(self.opened), **data)

    async def get_matching_open_positions(
        self,
        *,
        model_name: str,
        symbol: str,
        side: str,
        execution_mode: str,
    ) -> list[Any]:
        return [
            position
            for position in self.positions
            if position.model_name == model_name
            and position.symbol == symbol
            and position.side == side
            and position.execution_mode == execution_mode
            and position.is_open
        ]


def _decision(action: Action) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=action,
        confidence=0.7,
        reasoning="test",
        position_size_pct=0.03,
        suggested_leverage=3.0,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
    )


def _result(**overrides: Any) -> SimpleNamespace:
    data = {
        "symbol": "BTC/USDT",
        "quantity": 2.0,
        "price": 110.0,
        "fee": 1.0,
        "timestamp": datetime(2026, 6, 10, 12, 0),
        "pnl": 0.0,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def _position(**overrides: Any) -> SimpleNamespace:
    data = {
        "id": 1,
        "model_name": "ensemble_trader",
        "execution_mode": "paper",
        "symbol": "BTC/USDT",
        "side": "long",
        "quantity": 2.0,
        "entry_price": 100.0,
        "current_price": 100.0,
        "leverage": 3.0,
        "unrealized_pnl": 0.0,
        "realized_pnl": 0.0,
        "stop_loss_price": 98.0,
        "take_profit_price": 104.0,
        "is_open": True,
        "created_at": datetime(2026, 6, 10, 11, 0),
        "closed_at": None,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def _service(
    *,
    session: FakeSession,
    repo: FakeTradeRepo,
    confirmed: bool = True,
    exit_progress: bool = False,
    exchange_backed_ids: set[int] | None = None,
    entry_fee: float = 0.0,
    reflections: list[dict[str, Any]] | None = None,
    removed_peaks: list[tuple[str, str, str]] | None = None,
) -> PositionExecutionPersistenceService:
    reflections = reflections if reflections is not None else []
    removed_peaks = removed_peaks if removed_peaks is not None else []

    async def exchange_backed_id_provider(_session: Any, _positions: list[Any]) -> set[int]:
        return exchange_backed_ids or set()

    async def entry_fee_provider(_session: Any, _position: Any, _close_qty: float) -> float:
        return entry_fee

    async def reflection_recorder(session_arg: Any, position: Any, **kwargs: Any) -> None:
        reflections.append({"session": session_arg, "position": position, **kwargs})

    return PositionExecutionPersistenceService(
        exchange_confirmed_checker=lambda _result: confirmed,
        exit_progress_checker=lambda _result: exit_progress,
        exchange_backed_id_provider=exchange_backed_id_provider,
        entry_fee_provider=entry_fee_provider,
        proportional_fee=proportional_fee,
        trade_reflection_recorder=reflection_recorder,
        position_peak_remover=lambda model, symbol, side: removed_peaks.append(
            (model, symbol, side)
        ),
        session_context_factory=lambda: FakeSessionContext(session),
        trade_repo_factory=lambda _session: repo,
    )


@pytest.mark.asyncio
async def test_persist_entry_opens_position_with_protection_prices() -> None:
    session = FakeSession()
    repo = FakeTradeRepo()

    await _service(session=session, repo=repo).persist(
        model_name="ensemble_trader",
        decision=_decision(Action.LONG),
        result=_result(price=100.0, quantity=1.5),
        execution_mode="paper",
    )

    assert repo.opened == [
        {
            "model_name": "ensemble_trader",
            "execution_mode": "paper",
            "symbol": "BTC/USDT",
            "side": "long",
            "quantity": 1.5,
            "entry_price": 100.0,
            "current_price": 100.0,
            "leverage": 3.0,
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0,
            "stop_loss_price": 98.0,
            "take_profit_price": 104.0,
        }
    ]


@pytest.mark.asyncio
async def test_persist_entry_uses_okx_inst_id_symbol_over_ccxt_alias() -> None:
    session = FakeSession()
    repo = FakeTradeRepo()

    await _service(session=session, repo=repo).persist(
        model_name="ensemble_trader",
        decision=_decision(Action.SHORT),
        result=_result(
            symbol="WLFI/USDT",
            price=0.0817,
            quantity=1176.0,
            exchange_order_id="h-entry",
            raw_response={
                "symbol": "WLFI/USDT:USDT",
                "info": {"instId": "H-USDT-SWAP", "posId": "h-pos"},
                "canonical_exchange_symbol": "H/USDT",
            },
        ),
        execution_mode="paper",
    )

    assert repo.opened[0]["symbol"] == "H/USDT"
    assert repo.opened[0]["side"] == "short"
    assert repo.opened[0]["okx_inst_id"] == "H-USDT-SWAP"
    assert repo.opened[0]["okx_pos_id"] == "h-pos"
    assert repo.opened[0]["entry_exchange_order_id"] == "h-entry"


@pytest.mark.asyncio
async def test_persist_entry_keeps_same_symbol_side_add_entry_as_new_position() -> None:
    session = FakeSession()
    existing = _position(
        id=1,
        quantity=1.0,
        entry_price=90.0,
        current_price=95.0,
        entry_exchange_order_id="old-entry",
    )
    repo = FakeTradeRepo([existing])

    await _service(session=session, repo=repo).persist(
        model_name="ensemble_trader",
        decision=_decision(Action.LONG),
        result=_result(
            price=100.0,
            quantity=3.0,
            exchange_order_id="new-entry",
            raw_response={"info": {"instId": "BTC-USDT-SWAP", "posId": "btc-net"}},
        ),
        execution_mode="paper",
    )

    assert existing.quantity == 1.0
    assert existing.entry_price == 90.0
    assert repo.opened == [
        {
            "model_name": "ensemble_trader",
            "execution_mode": "paper",
            "symbol": "BTC/USDT",
            "side": "long",
            "quantity": 3.0,
            "entry_price": 100.0,
            "current_price": 100.0,
            "leverage": 3.0,
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0,
            "stop_loss_price": 98.0,
            "take_profit_price": 104.0,
            "okx_inst_id": "BTC-USDT-SWAP",
            "okx_pos_id": "btc-net",
            "entry_exchange_order_id": "new-entry",
        }
    ]


@pytest.mark.asyncio
async def test_persist_entry_ignores_duplicate_exchange_order_id_callback() -> None:
    session = FakeSession()
    existing = _position(
        id=1,
        quantity=1.0,
        entry_price=90.0,
        current_price=95.0,
        entry_exchange_order_id="dup-entry",
    )
    repo = FakeTradeRepo([existing])

    await _service(session=session, repo=repo).persist(
        model_name="ensemble_trader",
        decision=_decision(Action.LONG),
        result=_result(
            price=100.0,
            quantity=3.0,
            exchange_order_id="dup-entry",
            raw_response={"info": {"instId": "BTC-USDT-SWAP", "posId": "btc-net"}},
        ),
        execution_mode="paper",
    )

    assert repo.opened == []
    assert existing.quantity == 1.0
    assert existing.current_price == 100.0
    assert existing.okx_inst_id == "BTC-USDT-SWAP"
    assert existing.okx_pos_id == "btc-net"
    assert existing.entry_exchange_order_id == "dup-entry"


@pytest.mark.asyncio
async def test_persist_entry_native_inst_id_overrides_wrong_canonical_alias() -> None:
    session = FakeSession()
    repo = FakeTradeRepo()
    decision = _decision(Action.LONG)
    decision.symbol = "SPK/USDT"

    await _service(session=session, repo=repo).persist(
        model_name="ensemble_trader",
        decision=decision,
        result=_result(
            symbol="SAHARA/USDT",
            price=0.012,
            quantity=10.0,
            exchange_order_id="spk-entry",
            raw_response={
                "symbol": "SAHARA/USDT:USDT",
                "canonical_exchange_symbol": "SAHARA/USDT",
                "info": {"instId": "SPK-USDT-SWAP", "ordId": "spk-entry"},
            },
        ),
        execution_mode="paper",
    )

    assert repo.opened[0]["symbol"] == "SPK/USDT"
    assert repo.opened[0]["okx_inst_id"] == "SPK-USDT-SWAP"
    assert repo.opened[0]["entry_exchange_order_id"] == "spk-entry"


@pytest.mark.asyncio
async def test_persist_partial_exit_splits_closed_position_and_records_reflection() -> None:
    session = FakeSession()
    open_position = _position(quantity=5.0)
    repo = FakeTradeRepo([open_position])
    reflections: list[dict[str, Any]] = []
    result = _result(quantity=2.0, price=110.0, fee=1.0)

    await _service(
        session=session,
        repo=repo,
        entry_fee=0.4,
        reflections=reflections,
    ).persist(
        model_name="ensemble_trader",
        decision=_decision(Action.CLOSE_LONG),
        result=result,
        execution_mode="paper",
    )

    assert open_position.quantity == 3.0
    assert repo.opened[0]["is_open"] is False
    assert repo.opened[0]["quantity"] == 2.0
    assert repo.opened[0]["realized_pnl"] == pytest.approx(18.6)
    assert repo.opened[0]["close_fill_pnl"] == pytest.approx(20.0)
    assert repo.opened[0]["entry_fee"] == pytest.approx(0.4)
    assert repo.opened[0]["close_fee"] == pytest.approx(1.0)
    assert repo.opened[0]["funding_fee"] == pytest.approx(0.0)
    assert repo.opened[0]["settlement_status"] == "provisional"
    assert repo.opened[0]["settlement_source"] == "system_execution"
    assert result.pnl == pytest.approx(18.6)
    assert session.flush_count == 1
    assert reflections[0]["position"].quantity == 2.0
    assert reflections[0]["entry_fee"] == 0.4
    assert reflections[0]["close_fee"] == 1.0
    assert reflections[0]["gross_pnl"] == 20.0


@pytest.mark.asyncio
async def test_persist_full_exit_closes_positions_and_removes_profit_peaks() -> None:
    session = FakeSession()
    base = datetime(2026, 6, 10, 11, 0)
    first = _position(id=1, quantity=1.0, created_at=base + timedelta(minutes=1))
    second = _position(id=2, quantity=1.0, created_at=base)
    repo = FakeTradeRepo([first, second])
    reflections: list[dict[str, Any]] = []
    removed_peaks: list[tuple[str, str, str]] = []
    result = _result(
        quantity=2.0,
        price=110.0,
        fee=2.0,
        exchange_order_id="close-order-1",
        raw_response={"info": {"instId": "BTC-USDT-SWAP", "posId": "btc-pos"}},
    )

    await _service(
        session=session,
        repo=repo,
        exchange_backed_ids={1},
        entry_fee=0.5,
        reflections=reflections,
        removed_peaks=removed_peaks,
    ).persist(
        model_name="ensemble_trader",
        decision=_decision(Action.CLOSE_LONG),
        result=result,
        execution_mode="paper",
    )

    assert first.is_open is False
    assert second.is_open is False
    assert first.realized_pnl == pytest.approx(8.5)
    assert second.realized_pnl == pytest.approx(8.5)
    assert first.close_fill_pnl == pytest.approx(10.0)
    assert first.entry_fee == pytest.approx(0.5)
    assert first.close_fee == pytest.approx(1.0)
    assert first.funding_fee == pytest.approx(0.0)
    assert first.settlement_status == "provisional"
    assert first.settlement_source == "system_execution"
    assert second.close_fill_pnl == pytest.approx(10.0)
    assert second.entry_fee == pytest.approx(0.5)
    assert second.close_fee == pytest.approx(1.0)
    assert second.funding_fee == pytest.approx(0.0)
    assert second.settlement_status == "provisional"
    assert second.settlement_source == "system_execution"
    assert first.okx_inst_id == "BTC-USDT-SWAP"
    assert second.okx_inst_id == "BTC-USDT-SWAP"
    assert first.okx_pos_id == "btc-pos"
    assert second.okx_pos_id == "btc-pos"
    assert first.close_exchange_order_id == "close-order-1"
    assert second.close_exchange_order_id == "close-order-1"
    assert result.pnl == pytest.approx(17.0)
    assert removed_peaks == [
        ("ensemble_trader", "BTC/USDT", "long"),
        ("ensemble_trader", "BTC/USDT", "long"),
    ]
    assert [item["position"].id for item in reflections] == [1, 2]


@pytest.mark.asyncio
async def test_persist_exit_uses_okx_native_fill_pnl_over_local_formula() -> None:
    session = FakeSession()
    position = _position(id=1, quantity=2.0, entry_price=100.0)
    repo = FakeTradeRepo([position])
    reflections: list[dict[str, Any]] = []
    result = _result(
        quantity=2.0,
        price=110.0,
        fee=1.0,
        raw_response={
            "native_close_fill": {
                "pnl": 6.0,
                "quantity": 2.0,
                "order_id": "okx-close-1",
            }
        },
    )

    await _service(
        session=session,
        repo=repo,
        entry_fee=0.5,
        reflections=reflections,
    ).persist(
        model_name="ensemble_trader",
        decision=_decision(Action.CLOSE_LONG),
        result=result,
        execution_mode="paper",
    )

    assert position.is_open is False
    assert position.realized_pnl == pytest.approx(4.5)
    assert result.pnl == pytest.approx(4.5)
    assert reflections[0]["gross_pnl"] == pytest.approx(6.0)


@pytest.mark.asyncio
async def test_persist_full_exit_does_not_close_on_synthetic_native_close_order_id() -> None:
    session = FakeSession()
    position = _position(id=1, quantity=2.0)
    repo = FakeTradeRepo([position])
    result = _result(
        quantity=2.0,
        price=110.0,
        fee=1.0,
        order_id="okx_native_full_close",
        exchange_order_id=None,
        raw_response={
            "okx_native_close_position": True,
            "info": {"instId": "BTC-USDT-SWAP"},
        },
    )

    await _service(session=session, repo=repo, confirmed=False).persist(
        model_name="ensemble_trader",
        decision=_decision(Action.CLOSE_LONG),
        result=result,
        execution_mode="paper",
    )

    assert position.is_open is True
    assert getattr(position, "close_exchange_order_id", None) in {None, ""}
    assert getattr(position, "okx_inst_id", None) is None
    assert session.flush_count == 0


@pytest.mark.asyncio
async def test_persist_exit_closes_native_full_close_pending_backfill_without_synthetic_id() -> None:
    session = FakeSession()
    position = _position(id=1, quantity=2.0)
    repo = FakeTradeRepo([position])
    reflections: list[dict[str, Any]] = []
    result = _result(
        quantity=2.0,
        price=110.0,
        fee=0.0,
        status=OrderStatus.PARTIAL,
        order_id="okx_native_full_close_fill_pending",
        exchange_order_id=None,
        raw_response={
            "exit_tracking": True,
            "okx_native_close_position": True,
            "requires_okx_fill_backfill": True,
            "request_params": {"instId": "BTC-USDT-SWAP"},
            "position_contracts_before": 2.0,
            "position_contracts_after": 0.0,
            "remaining_contracts": 0.0,
            "filled_contracts": 2.0,
            "base_quantity": 2.0,
        },
    )

    await _service(
        session=session,
        repo=repo,
        confirmed=False,
        exit_progress=True,
        entry_fee=0.5,
        reflections=reflections,
    ).persist(
        model_name="ensemble_trader",
        decision=_decision(Action.CLOSE_LONG),
        result=result,
        execution_mode="paper",
    )

    assert position.is_open is False
    assert getattr(position, "close_exchange_order_id", None) in {None, ""}
    assert position.settlement_status == "pending_okx_fill_backfill"
    assert position.settlement_source == "okx_native_full_close_pending_backfill"
    assert position.settlement_raw["requires_okx_fill_backfill"] is True
    assert reflections
    assert session.flush_count == 1


@pytest.mark.asyncio
async def test_persist_exit_does_not_leave_or_apply_floating_point_dust() -> None:
    session = FakeSession()
    first = _position(id=1, quantity=0.92)
    second = _position(id=2, quantity=0.5, created_at=datetime(2026, 6, 10, 11, 5))
    repo = FakeTradeRepo([first, second])
    reflections: list[dict[str, Any]] = []
    result = _result(quantity=0.9200000000000001, price=110.0, fee=0.1)

    await _service(session=session, repo=repo, reflections=reflections).persist(
        model_name="ensemble_trader",
        decision=_decision(Action.CLOSE_LONG),
        result=result,
        execution_mode="paper",
    )

    assert first.is_open is False
    assert second.is_open is True
    assert second.quantity == 0.5
    assert repo.opened == []
    assert [item["position"].id for item in reflections] == [1]


@pytest.mark.asyncio
async def test_persist_skips_unconfirmed_or_invalid_execution() -> None:
    session = FakeSession()
    repo = FakeTradeRepo()

    await _service(session=session, repo=repo, confirmed=False).persist(
        model_name="ensemble_trader",
        decision=_decision(Action.LONG),
        result=_result(quantity=1.0),
        execution_mode="paper",
    )
    await _service(session=session, repo=repo).persist(
        model_name="ensemble_trader",
        decision=_decision(Action.LONG),
        result=_result(quantity=0.0),
        execution_mode="paper",
    )

    assert repo.opened == []
