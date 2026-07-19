from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from config.settings import settings
from db.repositories.trade_repo import TradeRepository
from db.session import close_db, get_session_ctx, init_db
from models.decision import AIDecision
from models.trade import OkxPositionHistory, Order, Position
from services.authoritative_trade_outcome import load_authoritative_trade_outcomes
from services.okx_order_fact_sync import (
    OKX_SYNC_CONFIRMED,
    OKX_SYNC_EXECUTION_RESULT_CONFIRMED,
)
from services.okx_position_history_store import upsert_okx_position_history_row
from services.okx_position_settlement_sync import OkxPositionSettlementSyncService
from tests.paper_canary_fixtures import complete_paper_canary_raw
from web_dashboard.api.dashboard import get_positions as get_dashboard_positions


class _FakeCcxt:
    def __init__(
        self,
        *,
        history_rows: list[dict[str, Any]],
        bills_error: Exception | None = None,
        position_rows: list[dict[str, Any]] | None = None,
        positions_error: Exception | None = None,
    ) -> None:
        self.history_rows = history_rows
        self.bills_error = bills_error
        self.position_rows = list(position_rows or [])
        self.positions_error = positions_error
        self.history_calls: list[dict[str, Any]] = []
        self.bill_calls: list[dict[str, Any]] = []
        self.position_calls: list[dict[str, Any]] = []

    async def privateGetAccountPositionsHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        self.history_calls.append(dict(params))
        return {"data": list(self.history_rows)}

    async def privateGetAccountBills(self, params: dict[str, Any]) -> dict[str, Any]:
        self.bill_calls.append(dict(params))
        if self.bills_error is not None:
            raise self.bills_error
        return {"data": []}

    async def privateGetAccountPositions(self, params: dict[str, Any]) -> dict[str, Any]:
        self.position_calls.append(dict(params))
        if self.positions_error is not None:
            raise self.positions_error
        return {"data": list(self.position_rows)}


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


async def _attach_entry_decision(
    *,
    symbol: str,
    side: str,
    exchange_order_id: str,
    executed_at: datetime,
) -> int:
    async with get_session_ctx() as session:
        decision = AIDecision(
            model_name="ensemble_trader",
            symbol=symbol,
            action=side,
            confidence=0.7,
            position_size_pct=0.01,
            suggested_leverage=1.0,
            stop_loss_pct=0.01,
            take_profit_pct=0.02,
            raw_llm_response={
                "paper_bootstrap_canary": {
                    "authorized": True,
                    "requested": True,
                }
            },
            is_paper=True,
            was_executed=True,
            executed_at=executed_at,
            execution_price=1.0,
        )
        session.add(decision)
        await session.flush()
        session.add(
            Order(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol=symbol,
                side="buy" if side == "long" else "sell",
                order_type="market",
                quantity=10.0,
                price=1.0,
                status="filled",
                fee=0.08,
                decision_id=int(decision.id),
                exchange_order_id=exchange_order_id,
                filled_at=executed_at,
            )
        )
        await session.flush()
        return int(decision.id)


async def _attach_verified_execution_pair(
    *,
    position_id: int,
    symbol: str,
    side: str,
    closed_at: datetime,
) -> tuple[str, str]:
    inst_id = symbol.replace("/", "-") + "-SWAP"
    entry_order_id = f"entry-{symbol.lower().replace('/', '-')}"
    close_order_id = f"close-{symbol.lower().replace('/', '-')}"
    opened_at = datetime(2026, 7, 5, 1, 0, tzinfo=UTC)
    raw = complete_paper_canary_raw()
    raw["pre_order_execution_facts"] = {
        "contract_spec": {
            "instId": inst_id,
            "ctVal": "1",
            "ctMult": "1",
            "lotSz": "1",
            "source": "okx_public_instruments",
        }
    }
    async with get_session_ctx() as session:
        position = await session.get(Position, position_id)
        assert position is not None
        position.entry_fee = 0.0
        position.close_fee = 0.12
        position.stop_loss_price = 0.9
        position.take_profit_price = 1.2
        decision = AIDecision(
            model_name="ensemble_trader",
            symbol=symbol,
            action=side,
            confidence=0.7,
            position_size_pct=0.01,
            suggested_leverage=2.0,
            stop_loss_pct=0.1,
            take_profit_pct=0.2,
            raw_llm_response=raw,
            is_paper=True,
            was_executed=True,
            executed_at=opened_at,
            execution_price=1.0,
        )
        session.add(decision)
        await session.flush()

        def order_payload(
            *,
            order_id: str,
            order_side: str,
            price: float,
            fee: float,
            fill_pnl: float,
            filled_at: datetime,
            decision_id: int | None,
        ) -> Order:
            order = Order(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol=symbol,
                side=order_side,
                order_type="market",
                quantity=10.0,
                price=price,
                status="filled",
                fee=fee,
                decision_id=decision_id,
                exchange_order_id=order_id,
                filled_at=filled_at,
                created_at=filled_at,
            )
            order.okx_inst_id = inst_id
            order.okx_fill_contracts = 10.0
            order.okx_fill_pnl = fill_pnl
            order.okx_sync_status = OKX_SYNC_EXECUTION_RESULT_CONFIRMED
            order.okx_raw_fills = {
                "source": "okx_execution_result",
                "fills_history_confirmed": False,
                "execution_result_confirmed": True,
                "order_id": order_id,
                "inst_id": inst_id,
                "contracts": 10.0,
                "contract_size": 1.0,
                "contract_size_verified": True,
                "contract_size_source": "okx_public_instruments",
                "base_quantity": 10.0,
                "avg_price": price,
                "fee_abs": fee,
                "fill_pnl": fill_pnl,
            }
            return order

        session.add_all(
            [
                order_payload(
                    order_id=entry_order_id,
                    order_side="buy" if side == "long" else "sell",
                    price=1.0,
                    fee=0.08,
                    fill_pnl=0.0,
                    filled_at=opened_at,
                    decision_id=int(decision.id),
                ),
                order_payload(
                    order_id=close_order_id,
                    order_side="sell" if side == "long" else "buy",
                    price=1.8,
                    fee=0.12,
                    fill_pnl=8.5,
                    filled_at=closed_at,
                    decision_id=None,
                ),
            ]
        )
    return entry_order_id, close_order_id


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
    closed_at = datetime.now(UTC) - timedelta(minutes=20)
    try:
        position_id = await _create_closed_position(closed_at=closed_at)
        ccxt = _FakeCcxt(
            history_rows=[
                {
                    "instId": "AI16Z-USDT-SWAP",
                    "posId": "pos-ai16z-usdt-1",
                    "posSide": "long",
                    "cTime": _ms(closed_at - timedelta(minutes=20)),
                    "uTime": _ms(closed_at),
                    "realizedPnl": "7.93",
                    "pnl": "8.5",
                    "fee": "-0.2",
                    "fundingFee": "-0.37",
                    "pnlRatio": "0.793",
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
async def test_settlement_uses_verified_execution_pair_when_position_history_lags(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'settlement-execution-pair.db').as_posix()}",
    )
    await init_db()
    closed_at = datetime.now(UTC) - timedelta(minutes=20)
    try:
        position_id = await _create_closed_position(
            symbol="EXACT/USDT",
            status="settlement_exception",
            closed_at=closed_at,
        )
        await _attach_verified_execution_pair(
            position_id=position_id,
            symbol="EXACT/USDT",
            side="long",
            closed_at=closed_at,
        )
        ccxt = _FakeCcxt(history_rows=[])

        summary = await OkxPositionSettlementSyncService(
            mode="paper",
            lookback_hours=24 * 14,
            executor_factory=_executor_factory(ccxt),
        ).sync_once()

        assert summary["reconciled_count"] == 1
        assert ccxt.bill_calls
        async with get_session_ctx() as session:
            position = await session.get(Position, position_id)
            histories = list(
                (
                    await session.execute(
                        OkxPositionHistory.__table__.select().where(
                            OkxPositionHistory.__table__.c.position_ids.is_not(None)
                        )
                    )
                ).all()
            )
            assert position is not None
            assert position.settlement_status == "reconciled"
            assert position.settlement_source == "okx_verified_execution_pair_settlement"
            assert position.close_fill_pnl == pytest.approx(8.5)
            assert position.entry_fee == pytest.approx(0.08)
            assert position.close_fee == pytest.approx(0.12)
            assert position.funding_fee == pytest.approx(0.0)
            assert position.realized_pnl == pytest.approx(8.3)
            assert histories[0]._mapping["source"] == (
                "okx_verified_execution_pair_settlement"
            )

        outcomes = await load_authoritative_trade_outcomes(mode="paper")
        outcome = next(
            item for item in outcomes if position_id in item.get("position_ids", [])
        )
        assert outcome["source"] == "okx_verified_execution_pair"
        assert outcome["outcome_complete"] is True
        assert outcome["realized_pnl"] == pytest.approx(8.3)
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_execution_pair_uses_fill_times_and_prorates_shared_entry_contracts(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'settlement-pair-allocation.db').as_posix()}",
    )
    await init_db()
    closed_at = datetime.now(UTC) - timedelta(minutes=20)
    opened_at = datetime(2026, 7, 5, 1, 0, tzinfo=UTC)
    try:
        position_id = await _create_closed_position(
            symbol="ALLOC/USDT",
            status="settlement_exception",
            closed_at=closed_at,
        )
        entry_order_id, _ = await _attach_verified_execution_pair(
            position_id=position_id,
            symbol="ALLOC/USDT",
            side="long",
            closed_at=closed_at,
        )
        async with get_session_ctx() as session:
            position = await session.get(Position, position_id)
            entry_order = (
                await session.execute(
                    Order.__table__.select().where(
                        Order.__table__.c.exchange_order_id == entry_order_id
                    )
                )
            ).first()
            assert position is not None
            assert entry_order is not None
            position.created_at = opened_at + timedelta(hours=2)
            order = await session.get(Order, int(entry_order._mapping["id"]))
            assert order is not None
            order.quantity = 20.0
            order.fee = 0.16
            order.okx_fill_contracts = 20.0
            order.okx_raw_fills = {
                **dict(order.okx_raw_fills or {}),
                "base_quantity": 20.0,
                "contracts": 20.0,
                "fee_abs": 0.16,
            }

        ccxt = _FakeCcxt(history_rows=[])
        summary = await OkxPositionSettlementSyncService(
            mode="paper",
            lookback_hours=24 * 14,
            executor_factory=_executor_factory(ccxt),
        ).sync_once()

        assert summary["reconciled_count"] == 1
        async with get_session_ctx() as session:
            history = (
                await session.execute(OkxPositionHistory.__table__.select())
            ).first()
            assert history is not None
            assert history._mapping["opened_at"] == opened_at.replace(tzinfo=None)
            assert history._mapping["open_max_pos"] == pytest.approx(10.0)
            assert history._mapping["close_total_pos"] == pytest.approx(10.0)
            assert history._mapping["fee"] == pytest.approx(-0.2)

            official_row = {
                **dict(history._mapping["raw_row"]),
                "realizedPnl": "8.3",
            }
            await upsert_okx_position_history_row(
                session,
                official_row,
                mode="paper",
                source="okx_position_history_sync",
                match_status="official_pos_id",
            )

        async with get_session_ctx() as session:
            histories = list(
                (await session.execute(OkxPositionHistory.__table__.select())).all()
            )
            assert len(histories) == 1
            assert histories[0]._mapping["source"] == "okx_position_history_sync"
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_execution_pair_fallback_rejects_position_lifecycle_still_open(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'settlement-pair-still-open.db').as_posix()}",
    )
    await init_db()
    closed_at = datetime.now(UTC) - timedelta(minutes=20)
    try:
        position_id = await _create_closed_position(
            symbol="STILLOPEN/USDT",
            status="settlement_exception",
            closed_at=closed_at,
        )
        await _attach_verified_execution_pair(
            position_id=position_id,
            symbol="STILLOPEN/USDT",
            side="long",
            closed_at=closed_at,
        )
        ccxt = _FakeCcxt(
            history_rows=[],
            position_rows=[
                {
                    "instId": "STILLOPEN-USDT-SWAP",
                    "posId": "pos-stillopen-usdt-1",
                    "posSide": "net",
                    "pos": "10",
                    "avgPx": "1",
                    "lever": "2",
                }
            ],
        )

        summary = await OkxPositionSettlementSyncService(
            mode="paper",
            lookback_hours=24 * 14,
            executor_factory=_executor_factory(ccxt),
        ).sync_once()

        assert summary["reconciled_count"] == 0
        assert summary["exception_count"] == 1
        assert ccxt.position_calls
        assert not ccxt.bill_calls
        async with get_session_ctx() as session:
            position = await session.get(Position, position_id)
            assert position is not None
            assert position.settlement_status == "settlement_exception"
            assert position.settlement_raw["last_error_code"] == "positions_history_no_rows"
            assert position.settlement_raw["last_error_context"][
                "execution_pair_fallback_code"
            ] == "execution_pair_lifecycle_still_open"
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_execution_pair_fallback_requires_successful_funding_bill_audit(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'settlement-pair-funding-error.db').as_posix()}",
    )
    await init_db()
    closed_at = datetime.now(UTC) - timedelta(minutes=20)
    try:
        position_id = await _create_closed_position(
            symbol="PAIRFAIL/USDT",
            status="settlement_exception",
            closed_at=closed_at,
        )
        await _attach_verified_execution_pair(
            position_id=position_id,
            symbol="PAIRFAIL/USDT",
            side="long",
            closed_at=closed_at,
        )
        ccxt = _FakeCcxt(
            history_rows=[],
            bills_error=TimeoutError("funding audit unavailable"),
        )

        summary = await OkxPositionSettlementSyncService(
            mode="paper",
            lookback_hours=24 * 14,
            executor_factory=_executor_factory(ccxt),
        ).sync_once()

        assert summary["reconciled_count"] == 0
        assert summary["exception_count"] == 1
        async with get_session_ctx() as session:
            position = await session.get(Position, position_id)
            assert position is not None
            assert position.settlement_status == "settlement_exception"
            assert position.settlement_raw["last_error_code"] == "funding_fee_api_error"
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_position_history_gap_without_verified_order_pair_remains_unsettled(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'settlement-no-pair.db').as_posix()}",
    )
    await init_db()
    closed_at = datetime.now(UTC) - timedelta(minutes=20)
    try:
        position_id = await _create_closed_position(
            symbol="NOPAIR/USDT",
            status="settlement_exception",
            closed_at=closed_at,
        )
        ccxt = _FakeCcxt(history_rows=[])

        summary = await OkxPositionSettlementSyncService(
            mode="paper",
            lookback_hours=24 * 14,
            executor_factory=_executor_factory(ccxt),
        ).sync_once()

        assert summary["reconciled_count"] == 0
        assert summary["exception_count"] == 1
        async with get_session_ctx() as session:
            position = await session.get(Position, position_id)
            assert position is not None
            assert position.settlement_status == "settlement_exception"
            assert position.settlement_raw["last_error_code"] == "positions_history_no_rows"
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_settlement_retires_duplicate_closed_lifecycle_before_reconciliation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'settlement-duplicate.db').as_posix()}",
    )
    await init_db()
    closed_at = datetime.now(UTC) - timedelta(minutes=20)
    try:
        canonical_id = await _create_closed_position(
            symbol="DUP/USDT",
            status="settlement_exception",
            closed_at=closed_at,
        )
        duplicate_id = await _create_closed_position(
            symbol="DUP/USDT",
            status="settlement_exception",
            closed_at=closed_at,
        )
        await _attach_verified_execution_pair(
            position_id=canonical_id,
            symbol="DUP/USDT",
            side="long",
            closed_at=closed_at,
        )
        ccxt = _FakeCcxt(history_rows=[])

        summary = await OkxPositionSettlementSyncService(
            mode="paper",
            lookback_hours=24 * 14,
            executor_factory=_executor_factory(ccxt),
        ).sync_once()

        assert summary["checked_count"] == 1
        assert summary["reconciled_count"] == 1
        async with get_session_ctx() as session:
            canonical = await session.get(Position, canonical_id)
            duplicate = await session.get(Position, duplicate_id)
            assert canonical is not None
            assert duplicate is not None
            assert canonical.settlement_status == "reconciled"
            assert duplicate.settlement_status == "superseded_position_residual"
            assert duplicate.settlement_raw["canonical_position_id"] == canonical_id
            assert duplicate.settlement_raw["reason"] == (
                "duplicate_local_closed_position_for_same_okx_lifecycle"
            )
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_final_settlement_backfills_missing_entry_decision_outcome(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'settlement-outcome-backfill.db').as_posix()}",
    )
    await init_db()
    closed_at = datetime.now(UTC) - timedelta(minutes=20)
    try:
        position_id = await _create_closed_position(
            status="okx_position_history",
            closed_at=closed_at,
        )
        decision_id = await _attach_entry_decision(
            symbol="AI16Z/USDT",
            side="long",
            exchange_order_id="entry-ai16z-usdt",
            executed_at=closed_at - timedelta(minutes=19),
        )
        await _seed_okx_position_history_rows(
            [
                {
                    "instId": "AI16Z-USDT-SWAP",
                    "posId": "pos-ai16z-usdt-1",
                    "posSide": "long",
                    "type": "2",
                    "cTime": _ms(closed_at - timedelta(minutes=20)),
                    "uTime": _ms(closed_at),
                    "realizedPnl": "7.93",
                    "pnl": "8.5",
                    "fee": "-0.2",
                    "fundingFee": "-0.37",
                    "pnlRatio": "0.793",
                    "openAvgPx": "1.0",
                    "closeAvgPx": "1.8",
                    "openMaxPos": "10",
                    "closeTotalPos": "10",
                }
            ]
        )
        ccxt = _FakeCcxt(history_rows=[])

        summary = await OkxPositionSettlementSyncService(
            mode="paper",
            lookback_hours=24 * 14,
            executor_factory=_executor_factory(ccxt),
        ).sync_once()
        repeated = await OkxPositionSettlementSyncService(
            mode="paper",
            lookback_hours=24 * 14,
            executor_factory=_executor_factory(ccxt),
        ).sync_once()

        assert summary["reconciled_count"] == 0
        assert summary["decision_outcome_count"] == 1
        assert repeated["decision_outcome_count"] == 0
        assert ccxt.history_calls == []
        async with get_session_ctx() as session:
            position = await session.get(Position, position_id)
            decision = await session.get(AIDecision, decision_id)
            assert position is not None
            assert decision is not None
            assert decision.outcome == "profit"
            assert decision.outcome_pnl_pct == pytest.approx(79.3)
            settlement = decision.raw_llm_response["authoritative_settlement_outcome"]
            assert settlement["position_id"] == position_id
            assert settlement["authority"] == "okx_position_history"
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
    closed_at = datetime.now(UTC) - timedelta(minutes=20)
    try:
        position_id = await _create_closed_position(closed_at=closed_at)
        ccxt = _FakeCcxt(
            history_rows=[
                {
                    "instId": "AI16Z-USDT-SWAP",
                    "posId": "pos-ai16z-usdt-1",
                    "posSide": "long",
                    "cTime": _ms(closed_at - timedelta(minutes=20)),
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
async def test_settlement_sync_restores_superseded_residual_before_api_call(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'settlement-superseded.db').as_posix()}",
    )
    await init_db()
    closed_at = datetime.now(UTC) - timedelta(minutes=20)
    try:
        position_id = await _create_closed_position(
            status="settlement_exception",
            closed_at=closed_at,
            settlement_raw={
                "reason": "duplicate_local_open_position_for_same_okx_pos_id",
                "canonical_position_id": 99,
                "last_error_code": "positions_history_no_rows",
            },
        )
        ccxt = _FakeCcxt(history_rows=[])

        summary = await OkxPositionSettlementSyncService(
            mode="paper",
            lookback_hours=24 * 14,
            executor_factory=_executor_factory(ccxt),
        ).sync_once()

        assert summary["checked_count"] == 0
        assert ccxt.history_calls == []
        async with get_session_ctx() as session:
            position = await session.get(Position, position_id)
            assert position is not None
            assert position.settlement_status == "superseded_position_residual"
            assert position.settlement_source == "okx_current_position_deduplication"
            assert position.settlement_raw["canonical_position_id"] == 99
            assert position.settlement_raw["last_error_code"] == ("positions_history_no_rows")
            assert position.settlement_raw["restored_from_status"] == ("settlement_exception")
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
