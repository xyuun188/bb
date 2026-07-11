from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import select

from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.trade import OkxPositionHistory
from services.okx_position_history_sync import OkxPositionHistoryMirrorSyncService


class _FakeCcxt:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        history_error: Exception | None = None,
    ) -> None:
        self.rows = rows
        self.history_error = history_error
        self.history_calls: list[dict[str, Any]] = []
        self.instrument_calls: list[dict[str, Any]] = []

    async def privateGetAccountPositionsHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        self.history_calls.append(dict(params))
        if self.history_error:
            raise self.history_error
        return {"data": list(self.rows)}

    async def publicGetPublicInstruments(self, params: dict[str, Any]) -> dict[str, Any]:
        self.instrument_calls.append(dict(params))
        return {
            "data": [
                {
                    "instId": row["instId"],
                    "instType": "SWAP",
                    "ctVal": "0.1",
                    "ctMult": "1",
                    "lotSz": "1",
                    "minSz": "1",
                    "settleCcy": "USDT",
                }
                for row in self.rows
            ]
        }


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


@pytest.mark.asyncio
async def test_position_history_mirror_sync_persists_account_rows_without_local_positions(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'history-sync.db').as_posix()}",
    )
    await init_db()
    now = datetime.now(UTC)
    rows = [
        {
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
        },
        {
            "instId": "AXS-USDT-SWAP",
            "posId": "axs-pos-1",
            "posSide": "short",
            "type": "2",
            "cTime": _ms(now - timedelta(minutes=18)),
            "uTime": _ms(now - timedelta(minutes=4)),
            "openAvgPx": "2.9",
            "closeAvgPx": "2.8",
            "openMaxPos": "10",
            "closeTotalPos": "10",
            "realizedPnl": "1.0",
        },
    ]
    ccxt = _FakeCcxt(rows)
    try:
        report = await OkxPositionHistoryMirrorSyncService(
            mode="paper",
            lookback_hours=24,
            executor_factory=_executor_factory(ccxt),
        ).sync_once()

        assert report["status"] == "ok"
        assert report["live_count"] == 2
        assert report["inserted_count"] == 2
        assert report["updated_count"] == 0
        assert ccxt.history_calls
        assert ccxt.instrument_calls == [{"instType": "SWAP"}]
        assert ccxt.history_calls[0]["instType"] == "SWAP"
        assert "instId" not in ccxt.history_calls[0]
        assert "posId" not in ccxt.history_calls[0]

        async with get_session_ctx() as session:
            result = await session.execute(
                select(OkxPositionHistory).order_by(OkxPositionHistory.updated_at_okx.desc())
            )
            records = list(result.scalars().all())
        assert [record.inst_id for record in records] == ["AXS-USDT-SWAP", "ADA-USDT-SWAP"]
        assert records[0].source == "okx_position_history_account_sync"
        assert records[0].match_status == "okx_account_position_history"
        assert records[1].realized_pnl == pytest.approx(4.2)
        assert records[1].funding_fee == pytest.approx(-0.03)
        assert records[1].raw_row["_bb_contract_spec"] == {
            "instId": "ADA-USDT-SWAP",
            "instType": "SWAP",
            "ctVal": "0.1",
            "ctMult": "1",
            "ctValCcy": "",
            "lotSz": "1",
            "minSz": "1",
            "settleCcy": "USDT",
            "source": "okx_public_instruments",
        }
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_position_history_mirror_sync_updates_existing_rows(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'history-sync-update.db').as_posix()}",
    )
    await init_db()
    now = datetime.now(UTC)
    row = {
        "instId": "JUP-USDT-SWAP",
        "posId": "jup-pos-1",
        "posSide": "long",
        "type": "2",
        "cTime": _ms(now - timedelta(minutes=30)),
        "uTime": _ms(now - timedelta(minutes=2)),
        "openAvgPx": "0.4",
        "closeAvgPx": "0.42",
        "openMaxPos": "100",
        "closeTotalPos": "100",
        "realizedPnl": "2.0",
    }
    ccxt = _FakeCcxt([dict(row)])
    try:
        service = OkxPositionHistoryMirrorSyncService(
            mode="paper",
            lookback_hours=24,
            executor_factory=_executor_factory(ccxt),
        )
        first = await service.sync_once()
        ccxt.rows[0]["realizedPnl"] = "2.3"
        ccxt.rows[0]["uTime"] = _ms(now - timedelta(minutes=1))
        ccxt.rows[0]["closeTotalPos"] = "90"
        second = await service.sync_once()

        assert first["inserted_count"] == 1
        assert second["inserted_count"] == 0
        assert second["updated_count"] == 1
        async with get_session_ctx() as session:
            result = await session.execute(select(OkxPositionHistory))
            records = list(result.scalars().all())
        assert len(records) == 1
        assert records[0].realized_pnl == pytest.approx(2.3)
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_contract_specs_backfill_even_when_private_history_temporarily_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'history-spec-fallback.db').as_posix()}",
    )
    await init_db()
    now = datetime.now(UTC)
    async with get_session_ctx() as session:
        session.add(
            OkxPositionHistory(
                mode="paper",
                row_identity="paper|ADA-USDT-SWAP|ada-pos-2|long|2",
                inst_id="ADA-USDT-SWAP",
                symbol="ADA/USDT",
                pos_id="ada-pos-2",
                pos_side="long",
                side="long",
                close_status="full",
                raw_row={"instId": "ADA-USDT-SWAP", "posId": "ada-pos-2"},
                sync_status="synced",
                synced_at=now,
            )
        )
    ccxt = _FakeCcxt(
        [{"instId": "ADA-USDT-SWAP"}],
        history_error=RuntimeError("temporary private history failure"),
    )
    try:
        report = await OkxPositionHistoryMirrorSyncService(
            mode="paper",
            executor_factory=_executor_factory(ccxt),
        ).sync_once()

        assert report["status"] == "degraded"
        assert ccxt.instrument_calls == [{"instType": "SWAP"}]
        async with get_session_ctx() as session:
            record = (await session.execute(select(OkxPositionHistory))).scalars().one()
        assert record.raw_row["_bb_contract_spec"]["ctVal"] == "0.1"
        assert record.raw_row["_bb_contract_spec"]["ctMult"] == "1"
    finally:
        await close_db()
