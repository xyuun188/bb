from __future__ import annotations

import inspect
from datetime import UTC, datetime
from typing import Any

import pytest

from config.settings import settings
from db.repositories.trade_repo import TradeRepository
from db.session import close_db, get_session_ctx, init_db
from models.trade import Order, Position
from services.okx_order_fact_sync import OKX_SYNC_CONFIRMED
from services.okx_position_history_store import upsert_okx_position_history_row
from services.okx_position_settlement_sync import OkxPositionSettlementSyncService
from web_dashboard.api.dashboard import get_positions as get_dashboard_positions


class _FakeCcxt:
    def __init__(
        self,
        *,
        history_rows: list[dict[str, Any]],
        bills_error: Exception | None = None,
    ) -> None:
        self.history_rows = history_rows
        self.bills_error = bills_error
        self.history_calls: list[dict[str, Any]] = []
        self.bill_calls: list[dict[str, Any]] = []

    async def privateGetAccountPositionsHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        self.history_calls.append(dict(params))
        return {"data": list(self.history_rows)}

    async def privateGetAccountBills(self, params: dict[str, Any]) -> dict[str, Any]:
        self.bill_calls.append(dict(params))
        if self.bills_error is not None:
            raise self.bills_error
        return {"data": []}


async def _seed_okx_position_history_rows(rows: list[dict[str, Any]]) -> None:
    async with get_session_ctx() as session:
        for row in rows:
            await upsert_okx_position_history_row(
                session,
                row,
                mode="paper",
                source="test_okx_position_history_mirror",
                match_status="test_seed",
            )


class _FakeExecutor:
    def __init__(self, ccxt: _FakeCcxt) -> None:
        self.ccxt = ccxt
        self.initialized = False
        self.closed = False

    async def initialize(self) -> None:
        self.initialized = True

    async def shutdown(self) -> None:
        self.closed = True

    async def _get_ccxt(self) -> _FakeCcxt:
        return self.ccxt

    async def _with_retry(self, fn, *args, **kwargs):
        result = fn(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result


def _executor_factory(ccxt: _FakeCcxt):
    def factory(*_args, **_kwargs) -> _FakeExecutor:
        return _FakeExecutor(ccxt)

    return factory


def _ms(value: datetime) -> str:
    return str(int(value.timestamp() * 1000))


async def _create_closed_position(
    *,
    symbol: str = "AI16Z/USDT",
    side: str = "long",
    status: str = "settling",
    closed_at: datetime,
    settlement_raw: dict[str, Any] | None = None,
) -> int:
    opened_at = datetime(2026, 7, 5, 1, 0, tzinfo=UTC)
    inst_id = symbol.replace("/", "-") + "-SWAP"
    async with get_session_ctx() as session:
        repo = TradeRepository(session)
        position = await repo.open_position(
            {
                "model_name": "ensemble_trader",
                "execution_mode": "paper",
                "symbol": symbol,
                "side": side,
                "quantity": 10.0,
                "entry_price": 1.0,
                "current_price": 1.8,
                "leverage": 2.0,
                "unrealized_pnl": 0.0,
                "realized_pnl": 8.3,
                "close_fill_pnl": 8.5,
                "entry_fee": 0.08,
                "close_fee": 0.12,
                "funding_fee": 0.0,
                "settlement_status": status,
                "settlement_source": "system_execution",
                "settlement_raw": settlement_raw or {},
                "is_open": False,
                "okx_inst_id": inst_id,
                "okx_pos_id": f"pos-{symbol.lower().replace('/', '-')}-1",
                "entry_exchange_order_id": f"entry-{symbol.lower().replace('/', '-')}",
                "close_exchange_order_id": f"close-{symbol.lower().replace('/', '-')}",
                "closed_at": closed_at,
                "created_at": opened_at,
            }
        )
        return int(position.id)


@pytest.mark.asyncio
async def test_okx_position_settlement_sync_reconciles_official_funding_fee(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'settlement-success.db').as_posix()}",
    )
    await init_db()
    closed_at = datetime(2026, 7, 5, 1, 20, tzinfo=UTC)
    try:
        position_id = await _create_closed_position(closed_at=closed_at)
        ccxt = _FakeCcxt(
            history_rows=[
                {
                    "instId": "AI16Z-USDT-SWAP",
                    "posId": "pos-ai16z-usdt-1",
                    "posSide": "long",
                    "cTime": _ms(datetime(2026, 7, 5, 1, 0, tzinfo=UTC)),
                    "uTime": _ms(closed_at),
                    "realizedPnl": "7.93",
                    "pnl": "8.5",
                    "fee": "-0.2",
                    "fundingFee": "-0.37",
                    "openAvgPx": "1.0",
                    "closeAvgPx": "1.8",
                }
            ]
        )

        summary = await OkxPositionSettlementSyncService(
            mode="paper",
            lookback_hours=24 * 14,
            executor_factory=_executor_factory(ccxt),
        ).sync_once()

        assert summary["reconciled_count"] == 1
        async with get_session_ctx() as session:
            position = await session.get(Position, position_id)
            assert position is not None
            assert position.settlement_status == "reconciled"
            assert position.settlement_source == "okx_position_history_settlement"
            assert position.realized_pnl == pytest.approx(7.93)
            assert position.close_fill_pnl == pytest.approx(8.5)
            assert position.entry_fee + position.close_fee == pytest.approx(0.2)
            assert position.funding_fee == pytest.approx(-0.37)
            assert position.settlement_raw["funding_fee_source"] == (
                "okx_positions_history.fundingFee"
            )
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_okx_position_settlement_sync_records_funding_fee_failure_reason(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'settlement-funding-error.db').as_posix()}",
    )
    await init_db()
    closed_at = datetime(2026, 7, 5, 1, 20, tzinfo=UTC)
    try:
        position_id = await _create_closed_position(closed_at=closed_at)
        ccxt = _FakeCcxt(
            history_rows=[
                {
                    "instId": "AI16Z-USDT-SWAP",
                    "posId": "pos-ai16z-usdt-1",
                    "posSide": "long",
                    "cTime": _ms(datetime(2026, 7, 5, 1, 0, tzinfo=UTC)),
                    "uTime": _ms(closed_at),
                    "realizedPnl": "7.93",
                    "pnl": "8.5",
                    "fee": "-0.2",
                }
            ],
            bills_error=TimeoutError("account bills timeout"),
        )

        summary = await OkxPositionSettlementSyncService(
            mode="paper",
            lookback_hours=24 * 14,
            retry_seconds=10.0,
            executor_factory=_executor_factory(ccxt),
        ).sync_once()

        assert summary["exception_count"] == 1
        async with get_session_ctx() as session:
            position = await session.get(Position, position_id)
            assert position is not None
            assert position.settlement_status == "settlement_exception"
            assert position.settlement_raw["last_error_code"] == "funding_fee_api_error"
            assert "account bills timeout" in position.settlement_raw["last_error_message"]
            assert position.settlement_raw["next_settlement_retry_at"]
            assert position.settlement_raw["settlement_attempt_count"] == 1
            assert position.settlement_raw["funding_fee_status"] == (
                "unknown_until_official_settlement"
            )
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_dashboard_closed_history_hides_unsettled_positions(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from web_dashboard.api import dashboard as dashboard_api

    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'settlement-dashboard.db').as_posix()}",
    )
    await init_db()
    closed_at = datetime(2026, 7, 5, 1, 20, tzinfo=UTC)

    async def official_position_history_rows(*_args, **_kwargs):
        return [
            {
                "instId": "VISIBLE-USDT-SWAP",
                "posId": "visible-pos",
                "posSide": "net",
                "openAvgPx": "1.0",
                "closeAvgPx": "1.2",
                "openMaxPos": "3",
                "closeTotalPos": "3",
                "realizedPnl": "0.58",
                "pnl": "0.6",
                "fundingFee": "0",
                "type": "2",
                "cTime": str(int(closed_at.timestamp() * 1000)),
                "uTime": str(int(closed_at.timestamp() * 1000)),
            }
        ]

    monkeypatch.setattr(
        dashboard_api,
        "_dashboard_okx_position_history_rows",
        official_position_history_rows,
    )
    await _seed_okx_position_history_rows(await official_position_history_rows())
    try:
        await _create_closed_position(
            symbol="HIDDEN/USDT",
            status="settling",
            closed_at=closed_at,
        )
        await _create_closed_position(
            symbol="VISIBLE/USDT",
            status="reconciled",
            closed_at=closed_at,
        )
        async with get_session_ctx() as session:
            session.add_all(
                [
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="VISIBLE/USDT",
                        side="buy",
                        order_type="market",
                        quantity=3.0,
                        price=1.0,
                        status="filled",
                        fee=0.01,
                        exchange_order_id="unsettled-entry-visible",
                        filled_at=closed_at,
                        created_at=closed_at,
                        okx_inst_id="VISIBLE-USDT-SWAP",
                        okx_sync_status=OKX_SYNC_CONFIRMED,
                        okx_raw_fills={
                            "fills_history_confirmed": True,
                            "fill_pnl": 0.0,
                            "base_quantity": 3.0,
                            "avg_price": 1.0,
                            "fee_abs": 0.01,
                            "timestamp": closed_at.isoformat(),
                        },
                    ),
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="VISIBLE/USDT",
                        side="sell",
                        order_type="market",
                        quantity=3.0,
                        price=1.2,
                        status="filled",
                        fee=0.01,
                        exchange_order_id="unsettled-close-visible",
                        filled_at=closed_at,
                        created_at=closed_at,
                        okx_inst_id="VISIBLE-USDT-SWAP",
                        okx_sync_status=OKX_SYNC_CONFIRMED,
                        okx_fill_pnl=0.6,
                        okx_raw_fills={
                            "fills_history_confirmed": True,
                            "fill_pnl": 0.6,
                            "base_quantity": 3.0,
                            "avg_price": 1.2,
                            "fee_abs": 0.01,
                            "timestamp": closed_at.isoformat(),
                        },
                    ),
                ]
            )

        payload = await get_dashboard_positions(mode="paper", closed_only=True)

        symbols = {row["symbol"] for row in payload["positions"]}
        assert "VISIBLE/USDT" in symbols
        assert "HIDDEN/USDT" not in symbols
        assert [
            row for row in payload["positions"] if row.get("settlement_status") in {"", None}
        ] == []
    finally:
        await close_db()
