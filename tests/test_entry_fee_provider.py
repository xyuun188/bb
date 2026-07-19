from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from services.current_position_management import build_current_position_management_contract
from services.entry_fee_provider import EntryFeeProvider, proportional_fee


class _FakeResult:
    def __init__(self, value: Any) -> None:
        self.value = value

    def scalar_one_or_none(self) -> Any:
        return self.value


class _FakeSession:
    def __init__(self, *values: Any) -> None:
        self.values = list(values)
        self.statements: list[Any] = []

    async def execute(self, statement: Any) -> _FakeResult:
        self.statements.append(statement)
        if not self.values:
            return _FakeResult(None)
        return _FakeResult(self.values.pop(0))


def _position(**kwargs: Any) -> SimpleNamespace:
    defaults = {
        "model_name": "ensemble_trader",
        "execution_mode": "paper",
        "symbol": "BTC/USDT",
        "side": "long",
        "created_at": datetime(2026, 6, 8, 12, 0, tzinfo=UTC),
        "quantity": 4.0,
        "entry_price": 100.0,
        "entry_fee": 2.0,
        "stop_loss_price": 98.0,
        "take_profit_price": 110.0,
        "entry_exchange_order_id": "entry-1",
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _order(**kwargs: Any) -> SimpleNamespace:
    defaults = {
        "fee": 2.0,
        "quantity": 4.0,
        "created_at": datetime(2026, 6, 8, 12, 0, tzinfo=UTC),
        "execution_mode": "paper",
        "status": "filled",
        "exchange_order_id": "entry-1",
        "okx_raw_fills": {
            "fills_history_confirmed": True,
            "order_id": "entry-1",
            "fee_abs": 2.0,
        },
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_proportional_fee_prorates_and_handles_bad_values():
    assert proportional_fee(2.0, close_qty=1.0, total_qty=4.0) == 0.5
    assert proportional_fee(-2.0, close_qty=2.0, total_qty=4.0) == 1.0
    assert proportional_fee(2.0, close_qty=10.0, total_qty=4.0) == 2.0
    assert proportional_fee(2.0, close_qty=1.0, total_qty=0.0) == 2.0
    assert proportional_fee("bad", close_qty=1.0, total_qty=4.0) == 0.0


@pytest.mark.asyncio
async def test_entry_fee_provider_uses_exact_linked_okx_fill():
    session = _FakeSession(
        _order(
            fee=3.0,
            quantity=6.0,
            okx_raw_fills={
                "fills_history_confirmed": True,
                "order_id": "entry-1",
                "fee_abs": 3.0,
            },
        )
    )

    fee = await EntryFeeProvider().entry_fee_for_position(session, _position(), close_qty=2.0)

    assert fee == 1.0
    assert len(session.statements) == 1


@pytest.mark.asyncio
async def test_entry_fee_provider_accepts_verified_okx_execution_result():
    session = _FakeSession(
        _order(
            fee=3.0,
            quantity=6.0,
            okx_raw_fills={
                "fills_history_confirmed": False,
                "execution_result_confirmed": True,
                "contract_size_verified": True,
                "contract_size_source": "okx_public_instruments",
                "order_id": "entry-1",
                "fee_abs": 3.0,
            },
        )
    )

    fee = await EntryFeeProvider().entry_fee_for_position(
        session,
        _position(),
        close_qty=2.0,
    )

    assert fee == 1.0


@pytest.mark.asyncio
async def test_entry_fee_provider_rejects_unverified_execution_result():
    session = _FakeSession(
        _order(
            okx_raw_fills={
                "fills_history_confirmed": False,
                "execution_result_confirmed": True,
                "contract_size_verified": False,
                "order_id": "entry-1",
                "fee_abs": 3.0,
            },
        )
    )

    fee = await EntryFeeProvider().entry_fee_for_position(
        session,
        _position(),
        close_qty=2.0,
    )

    assert fee == 0.0


@pytest.mark.asyncio
async def test_entry_fee_provider_does_not_guess_from_time_window():
    session = _FakeSession(
        _order(
            exchange_order_id="other-entry",
            okx_raw_fills={
                "fills_history_confirmed": True,
                "order_id": "other-entry",
                "fee_abs": 4.0,
            },
        )
    )

    fee = await EntryFeeProvider().entry_fee_for_position(session, _position(), close_qty=2.0)

    assert fee == 0.0
    assert len(session.statements) == 1


@pytest.mark.asyncio
async def test_entry_fee_provider_uses_valid_current_management_fee_when_fill_reload_missing():
    position = _position(entry_exchange_order_id=None)
    position.current_management_contract = build_current_position_management_contract(
        {
            "symbol": position.symbol,
            "side": position.side,
            "quantity": position.quantity,
            "contracts": position.quantity,
            "entry_price": position.entry_price,
            "current_price": 101.0,
            "entry_fee_usdt": position.entry_fee,
            "full_entry_fee_usdt": position.entry_fee,
            "full_entry_notional_usdt": position.entry_price * position.quantity,
            "entry_fee_evidence_complete": True,
            "entry_fee_source": "okx_fills_history",
            "stop_loss_price": 98.0,
            "take_profit_price": 110.0,
            "protection_evidence_complete": True,
            "protection_orders": [
                {
                    "algo_id": "oco-1",
                    "state": "live",
                    "contracts": 4.0,
                    "reduce_only": True,
                    "stop_loss_price": 98.0,
                    "take_profit_price": 110.0,
                }
            ],
            "position_stressed_loss_usdt": 8.0,
            "portfolio_stressed_loss_usdt": 8.0,
            "portfolio_gross_notional_usdt": 404.0,
            "account_equity_usdt": 1_000.0,
            "open_position_count": 1,
            "entry_order_ids": ["entry-1"],
            "entry_decision_ids": [],
            "original_entry_contract_complete": False,
            "original_entry_contract_gaps": ["historical_contract_missing"],
        }
    )
    session = _FakeSession()

    fee = await EntryFeeProvider().entry_fee_for_position(
        session,
        position,
        close_qty=1.0,
    )

    assert fee == 0.5
    assert len(session.statements) == 0


@pytest.mark.asyncio
async def test_entry_fee_provider_returns_zero_without_matching_order():
    session = _FakeSession(None)

    fee = await EntryFeeProvider().entry_fee_for_position(
        session,
        _position(),
        close_qty=1.0,
    )

    assert fee == 0.0
    assert len(session.statements) == 1
