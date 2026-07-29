from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from core.symbols import normalize_trading_symbol
from services.current_position_management import (
    CURRENT_POSITION_MANAGEMENT_KIND,
    CURRENT_POSITION_MANAGEMENT_VERSION,
    current_position_management_contract_complete,
)
from services.sync_service import (
    OkxSyncService,
    _confirmed_local_close_fill_for_position,
    _merge_local_position_candidates,
    _okx_close_fill_order_payload,
    normalized_open_position_context,
)


def _float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@pytest.mark.asyncio
async def test_partial_entry_recovery_runs_single_flight_and_requests_fact_sync() -> None:
    recovery_requests: list[str] = []
    release = asyncio.Event()
    calls = 0

    class Executor:
        async def finalize_active_partial_entries(self) -> list[dict[str, object]]:
            nonlocal calls
            calls += 1
            await release.wait()
            return [{"order_id": "partial-entry", "terminal": True}]

    service = OkxSyncService(
        order_fact_recovery_trigger=lambda mode: recovery_requests.append(mode)
    )

    assert service._start_partial_entry_recovery(Executor()) is True
    assert service._start_partial_entry_recovery(Executor()) is False
    release.set()
    task = service._partial_entry_recovery_task
    assert task is not None
    await task
    await asyncio.sleep(0)

    assert calls == 1
    assert recovery_requests == ["paper"]
    assert service._partial_entry_recovery_task is None


def test_normalized_open_position_context_uses_contract_value_and_stable_open_time() -> None:
    opened_at = datetime.now(UTC) - timedelta(hours=7)
    updated_at = datetime.now(UTC)
    context = normalized_open_position_context(
        {
            "model_name": "ensemble_trader",
            "symbol": "AUCTION/USDT:USDT",
            "side": "short",
            "quantity": 99.5,
            "contracts": 99.5,
            "contractSize": 0.1,
            "entryPrice": 3.532,
            "markPrice": 3.528,
            "unrealizedPnl": 0.0398,
            "leverage": 10,
            "info": {
                "instId": "AUCTION-USDT-SWAP",
                "posId": "auction-pos",
                "pos": "99.5",
                "ctVal": "0.1",
                "avgPx": "3.532",
                "markPx": "3.528",
                "upl": "0.0398",
                "cTime": str(int(opened_at.timestamp() * 1000)),
                "uTime": str(int(updated_at.timestamp() * 1000)),
            },
        },
        symbol_normalizer=normalize_trading_symbol,
        float_parser=_float,
    )

    assert context["symbol"] == "AUCTION/USDT"
    assert context["quantity"] == pytest.approx(9.95)
    assert context["contracts"] == pytest.approx(99.5)
    assert context["contract_size"] == pytest.approx(0.1)
    assert context["notional"] == pytest.approx(35.1434)
    assert str(context["created_at"]).startswith(opened_at.isoformat(timespec="seconds")[:19])
    assert not str(context["created_at"]).startswith(updated_at.isoformat(timespec="seconds")[:19])


def test_normalized_open_position_context_does_not_copy_obsolete_policy_metadata() -> None:
    context = normalized_open_position_context(
        {
            "model_name": "ensemble_trader",
            "symbol": "BTC/USDT:USDT",
            "side": "long",
            "contracts": "10",
            "entry_price": "100",
            "current_price": "101",
            "entry_exchange_order_id": "entry-a,entry-b",
            "entry_legs": [{"exchange_order_id": "entry-a"}, {"exchange_order_id": "entry-b"}],
            "entry_fee": 0.12,
            "current_management_contract": {
                "contract_version": "2026-07-15.current-position-management.v1",
                "management_eligible": True,
                "exit_fee_rate_proxy": 0.0006,
            },
            "profit_first_trade_plan": {"decision_lane": "old"},
            "profit_first_exit_plan": {"max_hold_minutes": 360},
            "info": {"ctVal": "0.1", "instId": "BTC-USDT-SWAP", "posId": "btc-pos"},
        },
        symbol_normalizer=normalize_trading_symbol,
        float_parser=_float,
    )

    assert context["entry_exchange_order_id"] == "entry-a,entry-b"
    assert context["entry_legs"] == [
        {"exchange_order_id": "entry-a"},
        {"exchange_order_id": "entry-b"},
    ]
    assert context["entry_fee_usdt"] == pytest.approx(0.12)
    assert context["exit_fee_rate"] == pytest.approx(0.0006)
    assert context["current_management_contract"]["management_eligible"] is True
    assert "profit_first_trade_plan" not in context
    assert "profit_first_exit_plan" not in context


def test_normalized_context_preserves_paper_training_horizon_lifecycle() -> None:
    lifecycle = {
        "version": "2026-07-22.paper-training-position-lifecycle.v1",
        "kind": "normal_paper_training_position",
        "symbol": "PEPE/USDT",
        "side": "short",
        "horizon_minutes": 10.0,
    }
    context = normalized_open_position_context(
        {
            "symbol": "PEPE/USDT:USDT",
            "side": "short",
            "contracts": 171.7,
            "entry_price": 0.00000284,
            "current_price": 0.00000285,
            "execution_mode": "paper",
            "paper_training_lifecycle": lifecycle,
            "info": {
                "ctVal": "10000000",
                "instId": "PEPE-USDT-SWAP",
                "posId": "pepe-pos",
            },
        },
        symbol_normalizer=normalize_trading_symbol,
        float_parser=_float,
    )

    assert context["paper_training_lifecycle"] == lifecycle


def _management_contract(*, contracts: float = 109.0, quantity: float = 109.0) -> dict:
    return {
        "contract_version": CURRENT_POSITION_MANAGEMENT_VERSION,
        "kind": CURRENT_POSITION_MANAGEMENT_KIND,
        "management_eligible": True,
        "blockers": [],
        "symbol": "KAITO/USDT",
        "side": "long",
        "contracts": contracts,
        "quantity": quantity,
    }


def test_normalized_context_restores_contract_size_from_matching_management_contract() -> None:
    context = normalized_open_position_context(
        {
            "symbol": "KAITO/USDT:USDT",
            "side": "long",
            "contracts": 109.0,
            "entryPrice": 0.80,
            "markPrice": 0.81,
            "notional": 88.225557,
            "execution_mode": "paper",
            "current_management_contract": _management_contract(),
            "info": {
                "instId": "KAITO-USDT-SWAP",
                "pos": "109",
                "avgPx": "0.80",
                "markPx": "0.81",
                "notionalUsd": "88.225557",
            },
        },
        symbol_normalizer=normalize_trading_symbol,
        float_parser=_float,
    )

    assert context["quantity"] == pytest.approx(109.0)
    assert context["contract_size"] == pytest.approx(1.0)
    assert context["contract_size_source"] == ("current_management_contract_okx_contract_spec")


def test_normalized_context_does_not_restore_contract_size_after_contracts_change() -> None:
    context = normalized_open_position_context(
        {
            "symbol": "KAITO/USDT:USDT",
            "side": "long",
            "contracts": 108.0,
            "entryPrice": 0.80,
            "markPrice": 0.81,
            "notional": 87.415557,
            "execution_mode": "paper",
            "current_management_contract": _management_contract(),
            "info": {
                "instId": "KAITO-USDT-SWAP",
                "pos": "108",
                "avgPx": "0.80",
                "markPx": "0.81",
                "notionalUsd": "87.415557",
            },
        },
        symbol_normalizer=normalize_trading_symbol,
        float_parser=_float,
    )

    assert context["quantity"] != pytest.approx(109.0)
    assert context["contract_size_source"] == "exchange_position_snapshot"


def test_merge_local_position_candidates_keeps_order_identity_without_old_plans() -> None:
    merged = _merge_local_position_candidates(
        [
            {
                "symbol": "BTC/USDT",
                "side": "long",
                "created_at": "2026-07-03T01:00:00+00:00",
                "entry_exchange_order_id": "entry-a",
                "entry_legs": [{"exchange_order_id": "entry-a"}],
            },
            {
                "symbol": "BTC/USDT",
                "side": "long",
                "created_at": "2026-07-03T02:00:00+00:00",
                "entry_exchange_order_id": "entry-b",
                "entry_legs": [{"exchange_order_id": "entry-b"}],
            },
        ],
        exchange_position={"symbol": "BTC-USDT-SWAP", "side": "long", "info": {"posId": "btc-pos"}},
    )

    assert merged["entry_exchange_order_id"] == "entry-a,entry-b"
    assert merged["created_at"] == "2026-07-03T01:00:00+00:00"
    assert {leg["exchange_order_id"] for leg in merged["entry_legs"]} == {"entry-a", "entry-b"}
    assert "profit_first_trade_plan" not in merged
    assert "profit_first_exit_plan" not in merged


def test_okx_close_fill_order_payload_persists_native_fill_fact() -> None:
    filled_at = datetime(2026, 7, 2, 1, 20, 53, tzinfo=UTC)
    payload = _okx_close_fill_order_payload(
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="SAND/USDT",
        side="buy",
        quantity=1455.0,
        price=0.0498,
        fee=0.0289836,
        decision_id=123,
        close_order_id="3706182741831417856",
        filled_at=filled_at,
        close_fill={
            "source": "okx_fills_history",
            "pnl": -3.58808175,
            "contracts": 145.5,
            "contract_size": 10.0,
            "contract_size_source": "okx_public_instruments",
            "quantity": 1455.0,
            "order_info": {
                "instId": "SAND-USDT-SWAP",
                "tradeId": "sand-close-trade",
                "fillSz": "145.5",
                "fillPnl": "-3.58808175",
                "ts": str(int(filled_at.timestamp() * 1000)),
            },
        },
        okx_inst_id="SAND-USDT-SWAP",
    )

    assert payload["okx_sync_status"] == "okx_confirmed"
    assert payload["okx_fill_contracts"] == pytest.approx(145.5)
    assert payload["okx_fill_pnl"] == pytest.approx(-3.58808175)
    assert payload["okx_raw_fills"]["fills_history_confirmed"] is True
    assert payload["okx_raw_fills"]["contract_size_verified"] is True
    assert payload["okx_raw_fills"]["contract_size_source"] == "okx_public_instruments"


def test_okx_close_fill_order_payload_does_not_invent_contract_size_authority() -> None:
    payload = _okx_close_fill_order_payload(
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="HBAR/USDT",
        side="sell",
        quantity=190.0,
        price=0.0738,
        fee=0.007011,
        decision_id=112901,
        close_order_id="3771703227805569024",
        filled_at=datetime(2026, 7, 24, 15, 45, 15, tzinfo=UTC),
        close_fill={
            "source": "okx_fills_history",
            "contracts": 1.9,
            "contract_size": 100.0,
            "order_info": {
                "instId": "HBAR-USDT-SWAP",
                "tradeId": "40613588",
                "fillSz": "1.9",
            },
        },
        okx_inst_id="HBAR-USDT-SWAP",
    )

    assert payload["okx_raw_fills"]["contract_size"] == 100.0
    assert payload["okx_raw_fills"]["contract_size_verified"] is False
    assert payload["okx_raw_fills"]["contract_size_source"] == "missing"


@pytest.mark.asyncio
async def test_management_refresh_builds_one_contract_for_fragmented_okx_net_position() -> None:
    first = SimpleNamespace(
        id=1,
        is_open=True,
        execution_mode="paper",
        symbol="BTC/USDT",
        side="long",
        quantity=2.0,
        entry_price=100.0,
        entry_fee=0.2,
        entry_exchange_order_id="entry-a",
        current_management_contract={},
        updated_at=None,
    )
    second = SimpleNamespace(
        id=2,
        is_open=True,
        execution_mode="paper",
        symbol="BTC/USDT",
        side="long",
        quantity=1.0,
        entry_price=100.0,
        entry_fee=0.1,
        entry_exchange_order_id="entry-b",
        current_management_contract={},
        updated_at=None,
    )
    orders = [
        SimpleNamespace(
            exchange_order_id="entry-a",
            status="filled",
            okx_raw_fills={
                "fee_abs": 0.2,
                "fills_history_confirmed": True,
                "order_id": "entry-a",
            },
            price=100.0,
            quantity=2.0,
            decision_id=11,
        ),
        SimpleNamespace(
            exchange_order_id="entry-b",
            status="filled",
            okx_raw_fills={
                "fee_abs": 0.1,
                "fills_history_confirmed": True,
                "order_id": "entry-b",
            },
            price=100.0,
            quantity=1.0,
            decision_id=12,
        ),
    ]

    class Result:
        class Scalars:
            @staticmethod
            def all():
                return orders

        @staticmethod
        def scalars():
            return Result.Scalars()

    class Session:
        @staticmethod
        async def execute(_statement):
            return Result()

    async def balance(_mode):
        return {"equity": 1_000.0}

    service = OkxSyncService(account_equity_provider=balance)
    refreshed = await service._refresh_current_position_management_contracts(
        session=Session(),
        positions=[first, second],
        exchange_positions=[
            {
                "symbol": "BTC/USDT:USDT",
                "side": "long",
                "contracts": 3.0,
                "contractSize": 1.0,
                "entryPrice": 100.0,
                "markPrice": 101.0,
                "info": {"instId": "BTC-USDT-SWAP", "pos": "3", "posSide": "long"},
            }
        ],
        protection_by_key={
            ("BTC/USDT", "long"): {
                "stop_loss_price": 95.0,
                "take_profit_price": 110.0,
                "protection_orders": [
                    {
                        "algo_id": "oco-a",
                        "state": "live",
                        "contracts": 3.0,
                        "reduce_only": True,
                        "stop_loss_price": 95.0,
                        "take_profit_price": 110.0,
                    }
                ],
            }
        },
        symbol_normalizer=normalize_trading_symbol,
        float_parser=_float,
    )

    contract = first.current_management_contract
    assert refreshed == 2
    assert second.current_management_contract == contract
    assert contract["position_scope"] == "exchange_net_position_group"
    assert contract["position_fragment_ids"] == [1, 2]
    assert contract["entry_fee_usdt"] == pytest.approx(0.3)
    assert contract["contracts"] == pytest.approx(3.0)
    assert current_position_management_contract_complete(
        SimpleNamespace(
            symbol="BTC/USDT",
            side="long",
            quantity=3.0,
            entry_price=100.0,
            entry_fee=0.3,
            stop_loss_price=95.0,
            take_profit_price=110.0,
        ),
        contract,
    )


@pytest.mark.asyncio
async def test_confirmed_local_close_order_is_reused_before_remote_fill_lookup() -> None:
    filled_at = datetime.now(UTC)
    order = SimpleNamespace(
        quantity=109.0,
        price=0.81,
        fee=0.04,
        exchange_order_id="kaito-close-1",
        filled_at=filled_at,
        created_at=filled_at,
        okx_fill_contracts=109.0,
        okx_fill_pnl=0.42,
        okx_raw_fills={},
    )

    class ScalarRows:
        def __init__(self, rows):
            self.rows = rows

        def all(self):
            return self.rows

    class Result:
        def __init__(self, rows):
            self.rows = rows

        def scalars(self):
            return ScalarRows(self.rows)

    class Session:
        calls = 0

        @classmethod
        async def execute(cls, _statement):
            cls.calls += 1
            return Result([order] if cls.calls == 1 else [])

    fill = await _confirmed_local_close_fill_for_position(
        Session(),
        SimpleNamespace(
            symbol="KAITO/USDT",
            side="long",
            execution_mode="paper",
            quantity=109.0,
            created_at=filled_at - timedelta(minutes=10),
        ),
    )

    assert fill["source"] == "local_okx_confirmed_close_order"
    assert fill["order_id"] == "kaito-close-1"
    assert fill["quantity"] == pytest.approx(109.0)


@pytest.mark.asyncio
async def test_confirmed_local_close_order_is_not_reused_after_another_position_consumed_it(
) -> None:
    filled_at = datetime.now(UTC)
    order = SimpleNamespace(
        quantity=1_000_000.0,
        price=0.000002748,
        fee=0.001374,
        exchange_order_id="pepe-partial-close",
        filled_at=filled_at,
        created_at=filled_at,
        okx_fill_contracts=0.1,
        okx_fill_pnl=-0.003,
        okx_raw_fills={},
    )
    consumed_position = SimpleNamespace(
        id=41,
        close_exchange_order_id="pepe-partial-close",
    )

    class ScalarRows:
        def __init__(self, rows):
            self.rows = rows

        def all(self):
            return self.rows

    class Result:
        def __init__(self, rows):
            self.rows = rows

        def scalars(self):
            return ScalarRows(self.rows)

    class Session:
        calls = 0

        @classmethod
        async def execute(cls, _statement):
            cls.calls += 1
            return Result([order] if cls.calls == 1 else [consumed_position])

    fill = await _confirmed_local_close_fill_for_position(
        Session(),
        SimpleNamespace(
            id=42,
            symbol="PEPE/USDT",
            side="long",
            execution_mode="paper",
            quantity=1_000_000.0,
            created_at=filled_at - timedelta(hours=5),
        ),
    )

    assert fill == {}
