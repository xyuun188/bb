from __future__ import annotations

import asyncio
import inspect
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import select

from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.account import OkxAccountBill
from models.trade import OkxPositionHistory
from services.okx_settlement_fact_sync import OkxSettlementFactSyncService


class _FakeCcxt:
    def __init__(
        self,
        *,
        history_rows: list[dict[str, Any]],
        bills: list[dict[str, Any]],
        delay_seconds: float = 0.0,
    ) -> None:
        self.history_rows = history_rows
        self.bills = bills
        self.delay_seconds = delay_seconds
        self.calls: list[str] = []

    async def privateGetAccountPositionsHistory(self, _params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("position_history")
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        return {"data": self.history_rows}

    async def privateGetAccountBills(self, _params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("account_bills")
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        return {"data": self.bills}

    async def publicGetPublicInstruments(self, _params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("contract_specs")
        return {
            "data": [
                {
                    "instId": "ADA-USDT-SWAP",
                    "instType": "SWAP",
                    "ctVal": "1",
                    "ctMult": "1",
                    "lotSz": "1",
                    "minSz": "1",
                    "settleCcy": "USDT",
                }
            ]
        }


class _FakeExecutor:
    def __init__(self, ccxt: _FakeCcxt) -> None:
        self.ccxt = ccxt

    async def initialize(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    async def _get_ccxt(self) -> _FakeCcxt:
        return self.ccxt

    async def _with_retry(self, fn, *args, **kwargs):
        result = fn(*args, **kwargs)
        return await result if inspect.isawaitable(result) else result


def _executor_factory(ccxt: _FakeCcxt):
    return lambda *_args, **_kwargs: _FakeExecutor(ccxt)


def _ms(value: datetime) -> str:
    return str(int(value.timestamp() * 1000))


async def _init_test_db(tmp_path, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / name).as_posix()}",
    )
    await init_db()


@pytest.mark.asyncio
async def test_settlement_fact_sync_mirrors_history_and_funding_bills(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _init_test_db(tmp_path, monkeypatch, "settlement-facts.db")
    now = datetime.now(UTC)
    history_row = {
        "instId": "ADA-USDT-SWAP",
        "posId": "ada-pos-1",
        "posSide": "long",
        "type": "2",
        "cTime": _ms(now - timedelta(minutes=20)),
        "uTime": _ms(now - timedelta(minutes=5)),
        "openAvgPx": "0.6",
        "closeAvgPx": "0.64",
        "openMaxPos": "100",
        "closeTotalPos": "100",
        "realizedPnl": "4.2",
        "fundingFee": "-0.03",
        "fee": "-0.11",
    }
    bill = {
        "billId": "funding-1",
        "instId": "ADA-USDT-SWAP",
        "posSide": "long",
        "type": "8",
        "subType": "173",
        "balChg": "-0.03",
        "ts": _ms(now - timedelta(minutes=10)),
    }
    ccxt = _FakeCcxt(history_rows=[history_row], bills=[bill])
    try:
        report = await OkxSettlementFactSyncService(
            mode="paper",
            executor_factory=_executor_factory(ccxt),
        ).sync_once()

        assert report["status"] == "ok"
        assert report["position_history_count"] == 1
        assert report["position_history_inserted_count"] == 1
        assert report["account_bill_count"] == 1
        assert report["account_bill_inserted_count"] == 1
        assert {"position_history", "account_bills"} <= set(report["completed_stages"])
        async with get_session_ctx() as session:
            history = (await session.execute(select(OkxPositionHistory))).scalar_one()
            stored_bill = (await session.execute(select(OkxAccountBill))).scalar_one()
        assert history.source == "okx_settlement_fact_mirror"
        assert history.realized_pnl == pytest.approx(4.2)
        assert history.raw_row["_bb_contract_spec"]["ctVal"] == "1"
        assert stored_bill.source == "okx_settlement_fact_mirror"
        assert stored_bill.funding_fee == pytest.approx(-0.03)
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_settlement_fact_sync_defers_slow_private_pulls_under_one_budget(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _init_test_db(tmp_path, monkeypatch, "settlement-facts-timeout.db")
    ccxt = _FakeCcxt(history_rows=[], bills=[], delay_seconds=2.0)
    try:
        report = await OkxSettlementFactSyncService(
            mode="paper",
            timeout_seconds=0.5,
            executor_factory=_executor_factory(ccxt),
        ).sync_once()

        assert report["status"] == "deferred"
        assert {"position_history", "account_bills"} <= set(report["deferred_stages"])
        assert report["stage_errors"] == []
    finally:
        await close_db()
