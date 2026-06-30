from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.trade import Order, Position
from services.okx_order_fact_sync import (
    OKX_SYNC_CONFIRMED,
    OKX_SYNC_NO_FILL_REJECTED,
    OKX_SYNC_ORDER_ONLY,
    OKX_SYNC_OKX_ONLY,
    OKX_SYNC_POSITION_CONFIRMED,
    OKX_SYNC_UNVERIFIED,
    PHASE3_DEFAULT_ORDER_SYNC_START,
    OkxOrderFactSyncService,
    _db_naive_since,
)
from web_dashboard.api.trades import get_trade_detail, get_trades


class _FillCcxt:
    def __init__(self) -> None:
        start_ms = int((PHASE3_DEFAULT_ORDER_SYNC_START + timedelta(minutes=10)).timestamp() * 1000)
        old_ms = int((PHASE3_DEFAULT_ORDER_SYNC_START - timedelta(hours=2)).timestamp() * 1000)
        self.rows = [
            {
                "ordId": "phase3-order",
                "tradeId": "phase3-trade",
                "instId": "SPK-USDT-SWAP",
                "side": "buy",
                "fillSz": "12",
                "fillPx": "0.0345",
                "fee": "-0.02",
                "fillPnl": "1.23",
                "ts": str(start_ms),
            },
            {
                "ordId": "old-order",
                "tradeId": "old-trade",
                "instId": "HOME-USDT-SWAP",
                "side": "buy",
                "fillSz": "99",
                "fillPx": "0.01",
                "fee": "-0.01",
                "fillPnl": "9.99",
                "ts": str(old_ms),
            },
            {
                "ordId": "btc-ctval-order",
                "tradeId": "btc-trade",
                "instId": "BTC-USDT-SWAP",
                "side": "buy",
                "fillSz": "3",
                "fillPx": "60000",
                "fee": "-0.12",
                "fillPnl": "0",
                "ts": str(start_ms + 1),
            },
        ]
        self.order_rows = [
            {
                "ordId": "phase3-order",
                "instId": "SPK-USDT-SWAP",
                "side": "buy",
                "ordType": "market",
                "state": "filled",
                "sz": "12",
                "accFillSz": "12",
                "avgPx": "0.0345",
                "cTime": str(start_ms),
                "uTime": str(start_ms),
            },
            {
                "ordId": "local-only",
                "instId": "BTC-USDT-SWAP",
                "side": "buy",
                "ordType": "market",
                "state": "filled",
                "sz": "1",
                "accFillSz": "1",
                "avgPx": "100",
                "cTime": str(start_ms + 2),
                "uTime": str(start_ms + 2),
            },
            {
                "ordId": "okx-canceled",
                "instId": "DOGE-USDT-SWAP",
                "side": "buy",
                "ordType": "limit",
                "state": "canceled",
                "sz": "100",
                "accFillSz": "0",
                "px": "0.2",
                "cTime": str(start_ms + 3),
                "uTime": str(start_ms + 4),
            },
            {
                "ordId": "old-order",
                "instId": "HOME-USDT-SWAP",
                "side": "buy",
                "ordType": "market",
                "state": "filled",
                "sz": "99",
                "accFillSz": "99",
                "avgPx": "0.01",
                "cTime": str(old_ms),
                "uTime": str(old_ms),
            },
        ]
        self.position_history_rows = [
            {
                "instId": "SPK-USDT-SWAP",
                "posId": "spk-phase3-pos",
                "posSide": "short",
                "openAvgPx": "0.039",
                "closeAvgPx": "0.0345",
                "openMaxPos": "12",
                "closeTotalPos": "12",
                "realizedPnl": "1.23",
                "lever": "3",
                "cTime": str(start_ms - 60_000),
                "uTime": str(start_ms),
            }
        ]
        self.order_history_params: list[dict[str, Any]] = []

    async def privateGetTradeFillsHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        order_id = str(params.get("ordId") or "")
        if order_id:
            return {"data": [row for row in self.rows if row["ordId"] == order_id]}
        since = int(params.get("begin") or 0)
        return {
            "data": [
                row for row in self.rows if int(row.get("ts") or 0) >= since
            ]
        }

    async def privateGetTradeOrdersHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        self.order_history_params.append(dict(params))
        order_id = str(params.get("ordId") or "")
        if order_id:
            return {"data": [row for row in self.order_rows if row["ordId"] == order_id]}
        since = int(params.get("begin") or 0)
        return {
            "data": [
                row for row in self.order_rows if int(row.get("cTime") or 0) >= since
            ]
        }

    async def privateGetAccountPositionsHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        since = int(params.get("begin") or 0)
        return {
            "data": [
                row
                for row in self.position_history_rows
                if int(row.get("uTime") or 0) >= since
            ]
        }

    async def privateGetAccountPositions(self, params: dict[str, Any]) -> dict[str, Any]:
        assert params["instType"] == "SWAP"
        return {"data": []}

    async def publicGetPublicInstruments(self, params: dict[str, Any]) -> dict[str, Any]:
        assert params["instType"] == "SWAP"
        return {
            "data": [
                {"instId": "SPK-USDT-SWAP", "ctVal": "1"},
                {"instId": "HOME-USDT-SWAP", "ctVal": "1"},
                {"instId": "BTC-USDT-SWAP", "ctVal": "0.01"},
            ]
        }


class _Executor:
    ccxt_instances: list[_FillCcxt] = []

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.ccxt = _FillCcxt()
        self.__class__.ccxt_instances.append(self.ccxt)

    async def initialize(self) -> None:
        return None

    async def _get_ccxt(self) -> _FillCcxt:
        return self.ccxt

    async def _with_retry(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        return await fn(*args, **kwargs)

    async def shutdown(self) -> None:
        return None


class _CurrentPositionOnlyCcxt:
    def __init__(self) -> None:
        self.fill_params: list[dict[str, Any]] = []
        self.order_history_params: list[dict[str, Any]] = []
        self.position_params: list[dict[str, Any]] = []
        self.position_ts = int(
            (PHASE3_DEFAULT_ORDER_SYNC_START + timedelta(minutes=40)).timestamp() * 1000
        )
        self.entry_order_id = "3695537280216961024"

    async def privateGetTradeFillsHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        self.fill_params.append(dict(params))
        return {
            "data": [
                {
                    "ordId": self.entry_order_id,
                    "tradeId": "196207763",
                    "instId": "FLOKI-USDT-SWAP",
                    "side": "sell",
                    "posSide": "net",
                    "fillSz": "6",
                    "fillPx": "0.00002174",
                    "fee": "-0.006522",
                    "fillPnl": "0",
                    "ts": str(self.position_ts),
                }
            ]
        }

    async def privateGetTradeOrdersHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        self.order_history_params.append(dict(params))
        order_id = str(params.get("ordId") or "")
        rows = [
            {
                "ordId": self.entry_order_id,
                "instId": "FLOKI-USDT-SWAP",
                "side": "sell",
                "ordType": "market",
                "state": "filled",
                "sz": "6",
                "accFillSz": "6",
                "avgPx": "0.00002174",
                "cTime": str(self.position_ts),
                "uTime": str(self.position_ts),
            }
        ]
        if order_id:
            rows = [row for row in rows if row["ordId"] == order_id]
        return {"data": rows}

    async def privateGetAccountPositionsHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"data": []}

    async def privateGetAccountPositions(self, params: dict[str, Any]) -> dict[str, Any]:
        self.position_params.append(dict(params))
        return {
            "data": [
                {
                    "instId": "FLOKI-USDT-SWAP",
                    "posId": "3695537280250515456",
                    "tradeId": "196207763",
                    "posSide": "net",
                    "pos": "-6",
                    "ctVal": "100000",
                    "avgPx": "0.00002174",
                    "markPx": "0.00002156",
                    "upl": "0.108",
                    "fee": "-0.006522",
                    "cTime": str(self.position_ts),
                    "uTime": str(self.position_ts),
                }
            ]
        }

    async def publicGetPublicInstruments(self, params: dict[str, Any]) -> dict[str, Any]:
        assert params["instType"] == "SWAP"
        return {"data": [{"instId": "FLOKI-USDT-SWAP", "ctVal": "100000"}]}


class _CurrentPositionOnlyExecutor:
    ccxt_instances: list[_CurrentPositionOnlyCcxt] = []

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.ccxt = _CurrentPositionOnlyCcxt()
        self.__class__.ccxt_instances.append(self.ccxt)

    async def initialize(self) -> None:
        return None

    async def _get_ccxt(self) -> _CurrentPositionOnlyCcxt:
        return self.ccxt

    async def _with_retry(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        return await fn(*args, **kwargs)

    async def shutdown(self) -> None:
        return None


class _CurrentPositionNoFillCcxt(_CurrentPositionOnlyCcxt):
    async def privateGetTradeFillsHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        self.fill_params.append(dict(params))
        return {"data": []}

    async def privateGetTradeOrdersHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        self.order_history_params.append(dict(params))
        return {"data": []}


class _CurrentPositionNoFillExecutor:
    ccxt_instances: list[_CurrentPositionNoFillCcxt] = []

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.ccxt = _CurrentPositionNoFillCcxt()
        self.__class__.ccxt_instances.append(self.ccxt)

    async def initialize(self) -> None:
        return None

    async def _get_ccxt(self) -> _CurrentPositionNoFillCcxt:
        return self.ccxt

    async def _with_retry(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        return await fn(*args, **kwargs)

    async def shutdown(self) -> None:
        return None


class _FillPairOnlyCcxt:
    def __init__(self) -> None:
        start_ms = int((PHASE3_DEFAULT_ORDER_SYNC_START + timedelta(hours=7)).timestamp() * 1000)
        close_ms = int((PHASE3_DEFAULT_ORDER_SYNC_START + timedelta(hours=11, minutes=40)).timestamp() * 1000)
        self.entry_order_id = "3695269432500391936"
        self.close_order_id = "3695833143904538624"
        self.rows = [
            {
                "ordId": self.entry_order_id,
                "tradeId": "557741",
                "instId": "MET-USDT-SWAP",
                "side": "sell",
                "posSide": "net",
                "fillSz": "10",
                "fillPx": "0.1601",
                "fee": "-0.008005",
                "fillPnl": "0",
                "ts": str(start_ms),
            },
            {
                "ordId": self.close_order_id,
                "tradeId": "558702",
                "instId": "MET-USDT-SWAP",
                "side": "buy",
                "posSide": "net",
                "fillSz": "10",
                "fillPx": "0.16437",
                "fee": "-0.0082185",
                "fillPnl": "-0.427",
                "ts": str(close_ms),
            },
        ]
        self.order_rows = [
            {
                "ordId": self.entry_order_id,
                "instId": "MET-USDT-SWAP",
                "side": "sell",
                "ordType": "market",
                "state": "filled",
                "sz": "10",
                "accFillSz": "10",
                "avgPx": "0.1601",
                "cTime": str(start_ms),
                "uTime": str(start_ms),
            },
            {
                "ordId": self.close_order_id,
                "instId": "MET-USDT-SWAP",
                "side": "buy",
                "ordType": "market",
                "state": "filled",
                "sz": "10",
                "accFillSz": "10",
                "avgPx": "0.16437",
                "cTime": str(close_ms),
                "uTime": str(close_ms),
            },
        ]

    async def privateGetTradeFillsHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        order_id = str(params.get("ordId") or "")
        rows = [row for row in self.rows if not order_id or row["ordId"] == order_id]
        return {"data": rows}

    async def privateGetTradeOrdersHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        order_id = str(params.get("ordId") or "")
        rows = [row for row in self.order_rows if not order_id or row["ordId"] == order_id]
        return {"data": rows}

    async def privateGetAccountPositionsHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"data": []}

    async def privateGetAccountPositions(self, params: dict[str, Any]) -> dict[str, Any]:
        assert params["instType"] == "SWAP"
        return {"data": []}

    async def publicGetPublicInstruments(self, params: dict[str, Any]) -> dict[str, Any]:
        assert params["instType"] == "SWAP"
        return {"data": [{"instId": "MET-USDT-SWAP", "ctVal": "10"}]}


class _FillPairOnlyExecutor:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.ccxt = _FillPairOnlyCcxt()

    async def initialize(self) -> None:
        return None

    async def _get_ccxt(self) -> _FillPairOnlyCcxt:
        return self.ccxt

    async def _with_retry(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        return await fn(*args, **kwargs)

    async def shutdown(self) -> None:
        return None


class _RepeatedPosIdLifecycleCcxt:
    pos_id = "3693581910187675649"
    old_entry_order_id = "3693581910187675648"
    old_close_order_id = "3694631550098051072"
    new_entry_order_id = "3695029485428248576"
    new_close_order_id = "3695363208212353024"

    def __init__(self) -> None:
        old_open_ms = int(datetime(2026, 6, 27, 17, 1, 58, 852000, tzinfo=UTC).timestamp() * 1000)
        old_close_ms = int(datetime(2026, 6, 28, 1, 43, 20, 560000, tzinfo=UTC).timestamp() * 1000)
        new_open_ms = int(datetime(2026, 6, 28, 5, 0, 59, 957000, tzinfo=UTC).timestamp() * 1000)
        new_close_ms = int(datetime(2026, 6, 28, 7, 46, 45, 670000, tzinfo=UTC).timestamp() * 1000)
        self.rows = [
            {
                "ordId": self.old_entry_order_id,
                "tradeId": "act-old-entry-trade",
                "instId": "ACT-USDT-SWAP",
                "side": "sell",
                "posSide": "net",
                "fillSz": "376",
                "fillPx": "0.0079327925531915",
                "fee": "-0.0149",
                "fillPnl": "0",
                "ts": str(old_open_ms),
            },
            {
                "ordId": self.old_close_order_id,
                "tradeId": "act-old-close-trade",
                "instId": "ACT-USDT-SWAP",
                "side": "buy",
                "posSide": "net",
                "fillSz": "376",
                "fillPx": "0.00836",
                "fee": "-0.0157",
                "fillPnl": "-1.6033",
                "ts": str(old_close_ms),
            },
            {
                "ordId": self.new_entry_order_id,
                "tradeId": "act-new-entry-trade",
                "instId": "ACT-USDT-SWAP",
                "side": "sell",
                "posSide": "net",
                "fillSz": "127",
                "fillPx": "0.01049",
                "fee": "-0.00666",
                "fillPnl": "0",
                "ts": str(new_open_ms),
            },
            {
                "ordId": self.new_close_order_id,
                "tradeId": "act-new-close-trade",
                "instId": "ACT-USDT-SWAP",
                "side": "buy",
                "posSide": "net",
                "fillSz": "127",
                "fillPx": "0.00908",
                "fee": "-0.00577",
                "fillPnl": "1.7907",
                "ts": str(new_close_ms),
            },
        ]
        self.order_rows = [
            {
                "ordId": row["ordId"],
                "instId": row["instId"],
                "side": row["side"],
                "ordType": "market",
                "state": "filled",
                "sz": row["fillSz"],
                "accFillSz": row["fillSz"],
                "avgPx": row["fillPx"],
                "cTime": row["ts"],
                "uTime": row["ts"],
            }
            for row in self.rows
        ]
        self.position_history_rows = [
            {
                "instId": "ACT-USDT-SWAP",
                "posId": self.pos_id,
                "posSide": "net",
                "openAvgPx": "0.0079327925531915",
                "closeAvgPx": "0.00836",
                "openMaxPos": "376",
                "closeTotalPos": "376",
                "realizedPnl": "-1.63397697",
                "pnl": "-1.6033",
                "pnlRatio": "-0.05476",
                "lever": "3",
                "cTime": str(old_open_ms),
                "uTime": str(old_close_ms),
            },
            {
                "instId": "ACT-USDT-SWAP",
                "posId": self.pos_id,
                "posSide": "net",
                "openAvgPx": "0.01049",
                "closeAvgPx": "0.00908",
                "openMaxPos": "127",
                "closeTotalPos": "127",
                "realizedPnl": "1.77827305",
                "pnl": "1.7907",
                "pnlRatio": "0.4004428026692088",
                "lever": "3",
                "cTime": str(new_open_ms),
                "uTime": str(new_close_ms),
            },
        ]

    async def privateGetTradeFillsHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        order_id = str(params.get("ordId") or "")
        rows = [row for row in self.rows if not order_id or row["ordId"] == order_id]
        since = int(params.get("begin") or 0)
        return {"data": [row for row in rows if int(row.get("ts") or 0) >= since]}

    async def privateGetTradeOrdersHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        order_id = str(params.get("ordId") or "")
        rows = [row for row in self.order_rows if not order_id or row["ordId"] == order_id]
        since = int(params.get("begin") or 0)
        return {"data": [row for row in rows if int(row.get("cTime") or 0) >= since]}

    async def privateGetAccountPositionsHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        since = int(params.get("begin") or 0)
        return {
            "data": [
                row
                for row in self.position_history_rows
                if int(row.get("uTime") or 0) >= since
            ]
        }

    async def privateGetAccountPositions(self, params: dict[str, Any]) -> dict[str, Any]:
        assert params["instType"] == "SWAP"
        return {"data": []}

    async def publicGetPublicInstruments(self, params: dict[str, Any]) -> dict[str, Any]:
        assert params["instType"] == "SWAP"
        return {"data": [{"instId": "ACT-USDT-SWAP", "ctVal": "10"}]}


class _RepeatedPosIdLifecycleExecutor:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.ccxt = _RepeatedPosIdLifecycleCcxt()

    async def initialize(self) -> None:
        return None

    async def _get_ccxt(self) -> _RepeatedPosIdLifecycleCcxt:
        return self.ccxt

    async def _with_retry(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        return await fn(*args, **kwargs)

    async def shutdown(self) -> None:
        return None


class _MultiFillLifecycleCcxt:
    def __init__(self) -> None:
        open_ms = int(datetime(2026, 6, 27, 17, 0, tzinfo=UTC).timestamp() * 1000)
        add_ms = int(datetime(2026, 6, 27, 17, 30, tzinfo=UTC).timestamp() * 1000)
        reduce_ms = int(datetime(2026, 6, 27, 18, 30, tzinfo=UTC).timestamp() * 1000)
        close_ms = int(datetime(2026, 6, 27, 19, 0, tzinfo=UTC).timestamp() * 1000)
        self.rows = [
            {
                "ordId": "inj-entry-1",
                "tradeId": "inj-entry-trade-1",
                "instId": "INJ-USDT-SWAP",
                "side": "sell",
                "posSide": "net",
                "fillSz": "10",
                "fillPx": "4.80",
                "fee": "-0.01",
                "fillPnl": "0",
                "ts": str(open_ms),
            },
            {
                "ordId": "inj-entry-2",
                "tradeId": "inj-entry-trade-2",
                "instId": "INJ-USDT-SWAP",
                "side": "sell",
                "posSide": "net",
                "fillSz": "20",
                "fillPx": "4.70",
                "fee": "-0.02",
                "fillPnl": "0",
                "ts": str(add_ms),
            },
            {
                "ordId": "inj-close-1",
                "tradeId": "inj-close-trade-1",
                "instId": "INJ-USDT-SWAP",
                "side": "buy",
                "posSide": "net",
                "fillSz": "15",
                "fillPx": "4.60",
                "fee": "-0.015",
                "fillPnl": "1.5",
                "ts": str(reduce_ms),
            },
            {
                "ordId": "inj-close-2",
                "tradeId": "inj-close-trade-2",
                "instId": "INJ-USDT-SWAP",
                "side": "buy",
                "posSide": "net",
                "fillSz": "15",
                "fillPx": "4.55",
                "fee": "-0.015",
                "fillPnl": "2.0",
                "ts": str(close_ms),
            },
        ]
        self.order_rows = [
            {
                "ordId": row["ordId"],
                "instId": row["instId"],
                "side": row["side"],
                "ordType": "market",
                "state": "filled",
                "sz": row["fillSz"],
                "accFillSz": row["fillSz"],
                "avgPx": row["fillPx"],
                "cTime": row["ts"],
                "uTime": row["ts"],
            }
            for row in self.rows
        ]
        self.position_history_rows = [
            {
                "instId": "INJ-USDT-SWAP",
                "posId": "inj-net-lifecycle",
                "posSide": "net",
                "openAvgPx": "4.733333333333333",
                "closeAvgPx": "4.575",
                "openMaxPos": "30",
                "closeTotalPos": "30",
                "realizedPnl": "3.5",
                "lever": "3",
                "cTime": str(open_ms),
                "uTime": str(close_ms),
            }
        ]

    async def privateGetTradeFillsHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        order_id = str(params.get("ordId") or "")
        rows = [row for row in self.rows if not order_id or row["ordId"] == order_id]
        since = int(params.get("begin") or 0)
        return {"data": [row for row in rows if int(row.get("ts") or 0) >= since]}

    async def privateGetTradeOrdersHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        order_id = str(params.get("ordId") or "")
        rows = [row for row in self.order_rows if not order_id or row["ordId"] == order_id]
        since = int(params.get("begin") or 0)
        return {"data": [row for row in rows if int(row.get("cTime") or 0) >= since]}

    async def privateGetAccountPositionsHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        since = int(params.get("begin") or 0)
        return {
            "data": [
                row
                for row in self.position_history_rows
                if int(row.get("uTime") or 0) >= since
            ]
        }

    async def privateGetAccountPositions(self, params: dict[str, Any]) -> dict[str, Any]:
        assert params["instType"] == "SWAP"
        return {"data": []}

    async def publicGetPublicInstruments(self, params: dict[str, Any]) -> dict[str, Any]:
        assert params["instType"] == "SWAP"
        return {"data": [{"instId": "INJ-USDT-SWAP", "ctVal": "0.1"}]}


class _MultiFillLifecycleExecutor:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.ccxt = _MultiFillLifecycleCcxt()

    async def initialize(self) -> None:
        return None

    async def _get_ccxt(self) -> _MultiFillLifecycleCcxt:
        return self.ccxt

    async def _with_retry(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        return await fn(*args, **kwargs)

    async def shutdown(self) -> None:
        return None


def test_order_fact_sync_effective_since_is_phase3_start_not_rolling_lookback() -> None:
    service = OkxOrderFactSyncService(
        mode="paper",
        lookback_hours=1,
        cold_start_marker_path=None,
    )
    future_now = PHASE3_DEFAULT_ORDER_SYNC_START + timedelta(days=10)

    assert service._effective_since(future_now) == PHASE3_DEFAULT_ORDER_SYNC_START.astimezone(UTC)


def test_order_fact_sync_db_since_uses_phase3_beijing_midnight_as_utc_instant() -> None:
    assert _db_naive_since(PHASE3_DEFAULT_ORDER_SYNC_START) == datetime(2026, 6, 27, 16, 0)


@pytest.mark.asyncio
async def test_order_fact_sync_confirms_only_phase3_orders_and_backfills_okx_facts(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-order-facts.db').as_posix()}",
    )
    await init_db()
    _Executor.ccxt_instances.clear()
    phase3_time = (PHASE3_DEFAULT_ORDER_SYNC_START + timedelta(minutes=10)).astimezone(UTC).replace(tzinfo=None)
    old_time = (PHASE3_DEFAULT_ORDER_SYNC_START - timedelta(hours=2)).astimezone(UTC).replace(tzinfo=None)
    try:
        async with get_session_ctx() as session:
            session.add_all(
                [
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="SPK/USDT",
                        side="sell",
                        order_type="market",
                        quantity=1.0,
                        price=0.03,
                        status="filled",
                        fee=0.0,
                        exchange_order_id="phase3-order",
                        filled_at=phase3_time,
                        created_at=phase3_time,
                    ),
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="HOME/USDT",
                        side="buy",
                        order_type="market",
                        quantity=99.0,
                        price=0.01,
                        status="filled",
                        fee=0.0,
                        exchange_order_id="old-order",
                        filled_at=old_time,
                        created_at=old_time,
                    ),
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="BTC/USDT",
                        side="buy",
                        order_type="market",
                        quantity=1.0,
                        price=100.0,
                        status="filled",
                        fee=0.0,
                        exchange_order_id="local-only",
                        filled_at=phase3_time,
                        created_at=phase3_time,
                    ),
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="COAI/USDT",
                        side="buy",
                        order_type="market",
                        quantity=0.0,
                        price=0.0,
                        status="rejected",
                        fee=0.0,
                        exchange_order_id=None,
                        filled_at=None,
                        created_at=phase3_time,
                    ),
                ]
            )

        report = await OkxOrderFactSyncService(
            mode="paper",
            lookback_hours=72,
            executor_factory=_Executor,
            cold_start_marker_path=None,
        ).sync()

        async with get_session_ctx() as session:
            rows = (
                await session.execute(
                    Order.__table__.select().order_by(Order.__table__.c.created_at.asc())
                )
            ).all()
            orders = {row._mapping["exchange_order_id"]: row._mapping for row in rows}

        assert report["okx_pull_available"] is True
        assert report["phase3_order_sync_start"] == "2026-06-27T16:00:00+00:00"
        assert report["phase3_order_sync_start_local"] == "2026-06-28T00:00:00+08:00"
        assert report["confirmed_count"] == 1
        assert report["unverified_count"] == 1
        assert report["backfilled_count"] == 1
        assert report["position_history_checked_count"] == 1
        assert report["position_history_backfilled_count"] == 1
        assert "old-order" in orders
        assert orders["old-order"]["okx_sync_status"] is None
        assert orders["phase3-order"]["okx_sync_status"] == OKX_SYNC_CONFIRMED
        assert orders["phase3-order"]["okx_inst_id"] == "SPK-USDT-SWAP"
        assert orders["phase3-order"]["okx_trade_ids"] == "phase3-trade"
        assert orders["phase3-order"]["quantity"] == pytest.approx(12.0)
        assert orders["phase3-order"]["price"] == pytest.approx(0.0345)
        assert orders["phase3-order"]["fee"] == pytest.approx(0.02)
        assert orders["phase3-order"]["okx_fill_pnl"] == pytest.approx(1.23)
        assert orders["btc-ctval-order"]["quantity"] == pytest.approx(0.03)
        assert orders["btc-ctval-order"]["okx_fill_contracts"] == pytest.approx(3.0)
        assert orders["local-only"]["okx_sync_status"] == OKX_SYNC_UNVERIFIED
        assert orders["local-only"]["okx_state"] == "filled"
        assert orders["local-only"]["okx_trade_ids"] is None
        assert orders["local-only"]["okx_fill_contracts"] is None
        assert orders["local-only"]["okx_fill_pnl"] is None
        assert orders["local-only"]["okx_raw_fills"]["fills_history_confirmed"] is False
        assert "rows" not in orders["local-only"]["okx_raw_fills"]
        assert orders["okx-canceled"]["status"] == "canceled"
        assert orders["okx-canceled"]["okx_sync_status"] == OKX_SYNC_ORDER_ONLY
        assert orders["okx-canceled"]["okx_state"] == "canceled"
        rejected = next(row._mapping for row in rows if row._mapping["symbol"] == "COAI/USDT")
        assert rejected["okx_sync_status"] == OKX_SYNC_NO_FILL_REJECTED
        assert rejected["okx_state"] == "rejected_no_exchange_fill"
        position_rows = (
            await session.execute(
                Position.__table__.select().where(
                    Position.__table__.c.okx_pos_id == "spk-phase3-pos"
                )
            )
        ).all()
        assert len(position_rows) == 1
        position = position_rows[0]._mapping
        assert position["model_name"] == "okx_authoritative_sync"
        assert position["symbol"] == "SPK/USDT"
        assert position["side"] == "short"
        assert position["quantity"] == pytest.approx(12.0)
        assert position["entry_price"] == pytest.approx(0.039)
        assert position["current_price"] == pytest.approx(0.0345)
        assert position["realized_pnl"] == pytest.approx(1.23)
        assert position["close_exchange_order_id"] == "phase3-order"
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_order_fact_sync_does_not_target_query_orders_already_seen_account_wide(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-order-gap-query.db').as_posix()}",
    )
    await init_db()
    _Executor.ccxt_instances.clear()
    phase3_time = (PHASE3_DEFAULT_ORDER_SYNC_START + timedelta(minutes=10)).astimezone(UTC).replace(tzinfo=None)
    try:
        async with get_session_ctx() as session:
            session.add_all(
                [
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="SPK/USDT",
                        side="sell",
                        order_type="market",
                        quantity=1.0,
                        price=0.03,
                        status="filled",
                        fee=0.0,
                        exchange_order_id="phase3-order",
                        filled_at=phase3_time,
                        created_at=phase3_time,
                    ),
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="BTC/USDT",
                        side="buy",
                        order_type="market",
                        quantity=1.0,
                        price=100.0,
                        status="filled",
                        fee=0.0,
                        exchange_order_id="local-only",
                        okx_trade_ids="stale-trade",
                        okx_fill_contracts=1.0,
                        okx_fill_pnl=9.99,
                        okx_raw_fills={
                            "trade_ids": ["stale-trade"],
                            "contracts": 1.0,
                            "avg_price": 100.0,
                            "fill_pnl": 9.99,
                            "rows": [{"tradeId": "stale-trade"}],
                        },
                        filled_at=phase3_time,
                        created_at=phase3_time,
                    ),
                ]
            )

        await OkxOrderFactSyncService(
            mode="paper",
            executor_factory=_Executor,
            cold_start_marker_path=None,
        ).sync()

        ccxt = _Executor.ccxt_instances[-1]
        target_queries = [
            params for params in ccxt.order_history_params if params.get("ordId")
        ]

        assert target_queries == []
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_order_fact_sync_does_not_duplicate_beijing_midnight_orders(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-order-utc-boundary.db').as_posix()}",
    )
    await init_db()
    # 2026-06-28 00:10 Asia/Shanghai is 2026-06-27 16:10 UTC. Online
    # PostgreSQL stores this as a UTC instant, so the local DB boundary must not
    # compare against naive Beijing midnight (2026-06-28 00:00).
    utc_db_time = datetime(2026, 6, 27, 16, 10)
    try:
        async with get_session_ctx() as session:
            session.add(
                Order(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="SPK/USDT",
                    side="sell",
                    order_type="market",
                    quantity=1.0,
                    price=0.03,
                    status="filled",
                    fee=0.0,
                    exchange_order_id="phase3-order",
                    filled_at=utc_db_time,
                    created_at=utc_db_time,
                )
            )

        report = await OkxOrderFactSyncService(
            mode="paper",
            lookback_hours=1,
            executor_factory=_Executor,
            cold_start_marker_path=None,
        ).sync()

        async with get_session_ctx() as session:
            rows = (
                await session.execute(
                    Order.__table__.select().where(
                        Order.__table__.c.exchange_order_id == "phase3-order"
                    )
                )
            ).all()
            order = rows[0]._mapping

        assert len(rows) == 1
        assert report["confirmed_count"] == 1
        assert report["backfilled_count"] == 0
        assert order["okx_sync_status"] == OKX_SYNC_CONFIRMED
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_trade_api_success_requires_okx_order_confirmation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-order-api.db').as_posix()}",
    )
    await init_db()
    now = datetime.now(UTC)
    try:
        async with get_session_ctx() as session:
            session.add_all(
                [
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="SPK/USDT",
                        side="sell",
                        order_type="market",
                        quantity=12.0,
                        price=0.0345,
                        status="filled",
                        fee=0.02,
                        exchange_order_id="confirmed",
                        okx_sync_status=OKX_SYNC_CONFIRMED,
                        okx_fill_pnl=1.23,
                        okx_trade_ids="trade-1",
                        okx_synced_at=now,
                        filled_at=now,
                        created_at=now,
                    ),
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="BTC/USDT",
                        side="buy",
                        order_type="market",
                        quantity=1.0,
                        price=100.0,
                        status="filled",
                        fee=0.0,
                        exchange_order_id="unverified",
                        okx_sync_status=OKX_SYNC_UNVERIFIED,
                        filled_at=now - timedelta(seconds=1),
                        created_at=now - timedelta(seconds=1),
                    ),
                ]
            )

        trades = await get_trades(model_name=None, symbol=None, mode="paper", limit=10, page=1)
        unverified = next(item for item in trades["trades"] if item["exchange_order_id"] == "unverified")
        confirmed = next(item for item in trades["trades"] if item["exchange_order_id"] == "confirmed")
        detail = await get_trade_detail(confirmed["id"])

        assert confirmed["success"] is True
        assert confirmed["okx_confirmed"] is True
        assert confirmed["okx_fill_pnl"] == pytest.approx(1.23)
        assert unverified["success"] is False
        assert unverified["okx_confirmed"] is False
        assert detail["success"] is True
        assert detail["okx_sync_status"] == OKX_SYNC_CONFIRMED
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_order_fact_sync_marks_current_position_confirmed_without_fill_confirmation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-position-confirmed.db').as_posix()}",
    )
    await init_db()
    phase3_time = (
        PHASE3_DEFAULT_ORDER_SYNC_START + timedelta(minutes=20)
    ).astimezone(UTC).replace(tzinfo=None)
    try:
        async with get_session_ctx() as session:
            order = Order(
                model_name="ensemble_trader",
                execution_mode="paper",
                symbol="FLOKI/USDT",
                side="sell",
                order_type="market",
                quantity=600000.0,
                price=0.00002174,
                status="filled",
                fee=0.0,
                exchange_order_id="3695537280216961024",
                okx_trade_ids="stale-trade",
                okx_fill_contracts=6.0,
                okx_fill_pnl=9.99,
                okx_raw_fills={
                    "trade_ids": ["stale-trade"],
                    "contracts": 6.0,
                    "avg_price": 0.00002174,
                    "fill_pnl": 9.99,
                    "rows": [{"tradeId": "stale-trade"}],
                },
                filled_at=phase3_time,
                created_at=phase3_time,
            )
            session.add(order)
            await session.flush()
            from models.trade import Position

            session.add(
                Position(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="FLOKI/USDT",
                    side="short",
                    quantity=600000.0,
                    entry_price=0.00002174,
                    current_price=0.00002156,
                    is_open=True,
                    okx_inst_id="FLOKI-USDT-SWAP",
                    okx_pos_id="3695537280250515456",
                    entry_exchange_order_id="3695537280216961024",
                    created_at=phase3_time,
                )
            )

        report = await OkxOrderFactSyncService(
            mode="paper",
            executor_factory=_CurrentPositionNoFillExecutor,
            cold_start_marker_path=None,
        ).sync()

        async with get_session_ctx() as session:
            row = (
                await session.execute(
                    Order.__table__.select().where(
                        Order.__table__.c.exchange_order_id == "3695537280216961024"
                    )
                )
            ).one()._mapping

        assert report["confirmed_count"] == 0
        assert report["position_confirmed_count"] == 1
        assert report["unverified_count"] == 0
        assert row["okx_sync_status"] == OKX_SYNC_POSITION_CONFIRMED
        assert row["okx_state"] == "open_position_confirmed"
        assert row["okx_trade_ids"] is None
        assert row["okx_fill_contracts"] is None
        assert row["okx_fill_pnl"] is None
        assert row["fee"] == pytest.approx(0.006522)
        assert row["quantity"] == pytest.approx(600000.0)
        assert row["okx_raw_fills"]["position_snapshot_confirmed"] is True
        assert row["okx_raw_fills"]["fills_history_confirmed"] is False
        assert row["okx_raw_fills"]["pos_id"] == "3695537280250515456"
        assert row["okx_raw_fills"]["position_trade_id"] == "196207763"

        trades = await get_trades(model_name=None, symbol=None, mode="paper", limit=10, page=1)
        item = next(
            trade
            for trade in trades["trades"]
            if trade["exchange_order_id"] == "3695537280216961024"
        )
        assert item["okx_confirmed"] is False
        assert item["success"] is False
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_order_fact_sync_backfills_open_position_cache_from_okx_current_positions(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-current-position-backfill.db').as_posix()}",
    )
    await init_db()
    try:
        report = await OkxOrderFactSyncService(
            mode="paper",
            executor_factory=_CurrentPositionOnlyExecutor,
            cold_start_marker_path=None,
        ).sync()

        async with get_session_ctx() as session:
            rows = (
                await session.execute(
                    Position.__table__.select().where(
                        Position.__table__.c.okx_pos_id == "3695537280250515456"
                    )
                )
            ).all()
            order_rows = (
                await session.execute(
                    Order.__table__.select().where(
                        Order.__table__.c.exchange_order_id == "3695537280216961024"
                    )
                )
            ).all()

        assert report["current_position_checked_count"] == 1
        assert report["current_position_backfilled_count"] == 1
        assert report["current_position_updated_count"] == 0
        assert report["current_position_skipped_count"] == 0
        assert len(rows) == 1
        position = rows[0]._mapping
        assert position["model_name"] == "okx_authoritative_sync"
        assert position["execution_mode"] == "paper"
        assert position["symbol"] == "FLOKI/USDT"
        assert position["side"] == "short"
        assert position["quantity"] == pytest.approx(600000.0)
        assert position["entry_price"] == pytest.approx(0.00002174)
        assert position["current_price"] == pytest.approx(0.00002156)
        assert position["unrealized_pnl"] == pytest.approx(0.108)
        assert position["realized_pnl"] == pytest.approx(0.0)
        assert position["is_open"] is True
        assert position["closed_at"] is None
        assert position["okx_inst_id"] == "FLOKI-USDT-SWAP"
        assert position["entry_exchange_order_id"] == "3695537280216961024"
        assert position["close_exchange_order_id"] is None
        assert len(order_rows) == 1
        assert order_rows[0]._mapping["okx_sync_status"] == OKX_SYNC_OKX_ONLY
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_order_fact_sync_updates_existing_open_position_cache_from_okx_current_positions(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-current-position-update.db').as_posix()}",
    )
    await init_db()
    try:
        async with get_session_ctx() as session:
            session.add(
                Position(
                    model_name="old_local_cache",
                    execution_mode="paper",
                    symbol="FLOKI/USDT",
                    side="short",
                    quantity=100.0,
                    entry_price=0.1,
                    current_price=0.2,
                    unrealized_pnl=99.0,
                    realized_pnl=88.0,
                    is_open=True,
                    okx_inst_id="FLOKI-USDT-SWAP",
                    okx_pos_id="3695537280250515456",
                    entry_exchange_order_id="stale-local-order",
                    close_exchange_order_id="stale-close-order",
                )
            )

        report = await OkxOrderFactSyncService(
            mode="paper",
            executor_factory=_CurrentPositionOnlyExecutor,
            cold_start_marker_path=None,
        ).sync()

        async with get_session_ctx() as session:
            rows = (
                await session.execute(
                    Position.__table__.select().where(
                        Position.__table__.c.okx_pos_id == "3695537280250515456"
                    )
                )
            ).all()

        assert report["current_position_checked_count"] == 1
        assert report["current_position_backfilled_count"] == 0
        assert report["current_position_updated_count"] == 1
        assert len(rows) == 1
        position = rows[0]._mapping
        assert position["model_name"] == "okx_authoritative_sync"
        assert position["quantity"] == pytest.approx(600000.0)
        assert position["entry_price"] == pytest.approx(0.00002174)
        assert position["current_price"] == pytest.approx(0.00002156)
        assert position["unrealized_pnl"] == pytest.approx(0.108)
        assert position["realized_pnl"] == pytest.approx(0.0)
        assert position["is_open"] is True
        assert position["entry_exchange_order_id"] == "stale-local-order,3695537280216961024"
        assert position["close_exchange_order_id"] is None
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_order_fact_sync_preserves_canonical_open_position_entry_links_when_duplicates_exist(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-current-position-duplicate-links.db').as_posix()}",
    )
    await init_db()
    phase3_time = (
        PHASE3_DEFAULT_ORDER_SYNC_START + timedelta(minutes=40)
    ).astimezone(UTC).replace(tzinfo=None)
    try:
        async with get_session_ctx() as session:
            session.add_all(
                [
                    Position(
                        model_name="okx_authoritative_sync",
                        execution_mode="paper",
                        symbol="FLOKI/USDT",
                        side="short",
                        quantity=0.000000001,
                        entry_price=0.00002174,
                        current_price=0.00002156,
                        unrealized_pnl=0.0,
                        realized_pnl=0.0,
                        is_open=True,
                        okx_inst_id="FLOKI-USDT-SWAP",
                        okx_pos_id="3695537280250515456",
                        entry_exchange_order_id=None,
                        created_at=phase3_time,
                    ),
                    Position(
                        model_name="okx_authoritative_sync",
                        execution_mode="paper",
                        symbol="FLOKI/USDT",
                        side="short",
                        quantity=600000.0,
                        entry_price=0.00002174,
                        current_price=0.00002156,
                        unrealized_pnl=0.108,
                        realized_pnl=0.0,
                        is_open=True,
                        okx_inst_id="FLOKI-USDT-SWAP",
                        okx_pos_id="3695537280250515456",
                        entry_exchange_order_id="existing-entry-a,existing-entry-b",
                        created_at=phase3_time,
                    ),
                ]
            )

        report = await OkxOrderFactSyncService(
            mode="paper",
            executor_factory=_CurrentPositionOnlyExecutor,
            cold_start_marker_path=None,
        ).sync()

        async with get_session_ctx() as session:
            rows = (
                await session.execute(
                    Position.__table__.select()
                    .where(Position.__table__.c.okx_pos_id == "3695537280250515456")
                    .order_by(Position.__table__.c.quantity.desc())
                )
            ).all()

        assert report["current_position_checked_count"] == 1
        assert report["current_position_updated_count"] == 1
        canonical = rows[0]._mapping
        dust = rows[1]._mapping
        assert canonical["quantity"] == pytest.approx(600000.0)
        assert canonical["entry_exchange_order_id"] == (
            "existing-entry-a,existing-entry-b,3695537280216961024"
        )
        assert dust["quantity"] == pytest.approx(0.000000001)
        assert dust["entry_exchange_order_id"] is None
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_order_fact_sync_backfills_closed_position_from_okx_fill_pair_when_history_missing(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-fill-pair-position.db').as_posix()}",
    )
    await init_db()
    try:
        report = await OkxOrderFactSyncService(
            mode="paper",
            executor_factory=_FillPairOnlyExecutor,
            cold_start_marker_path=None,
        ).sync()

        async with get_session_ctx() as session:
            position_rows = (
                await session.execute(
                    Position.__table__.select().where(
                        Position.__table__.c.symbol == "MET/USDT"
                    )
                )
            ).all()
            order_rows = (
                await session.execute(
                    Order.__table__.select().where(Order.__table__.c.symbol == "MET/USDT")
                )
            ).all()

        assert report["backfilled_count"] == 2
        assert report["position_history_checked_count"] == 0
        assert report["fill_pair_position_checked_count"] == 1
        assert report["fill_pair_position_backfilled_count"] == 1
        assert report["fill_pair_position_skipped_count"] == 0
        assert len(order_rows) == 2
        assert len(position_rows) == 1
        position = position_rows[0]._mapping
        assert position["model_name"] == "okx_authoritative_sync"
        assert position["side"] == "short"
        assert position["quantity"] == pytest.approx(100.0)
        assert position["entry_price"] == pytest.approx(0.1601)
        assert position["current_price"] == pytest.approx(0.16437)
        assert position["realized_pnl"] == pytest.approx(-0.4432235)
        assert position["is_open"] is False
        assert position["okx_inst_id"] == "MET-USDT-SWAP"
        assert position["entry_exchange_order_id"] == "3695269432500391936"
        assert position["close_exchange_order_id"] == "3695833143904538624"
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_order_fact_sync_splits_reused_okx_pos_id_into_distinct_lifecycles(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-reused-pos-id.db').as_posix()}",
    )
    await init_db()
    old_open = datetime(2026, 6, 27, 17, 1, 58, 852000)
    old_close = datetime(2026, 6, 28, 1, 43, 20, 560000)
    try:
        async with get_session_ctx() as session:
            session.add(
                Position(
                    model_name="old_polluted_cache",
                    execution_mode="paper",
                    symbol="ACT/USDT",
                    side="short",
                    quantity=3760.0,
                    entry_price=0.0079327925531915,
                    current_price=0.00836,
                    leverage=3.0,
                    unrealized_pnl=0.0,
                    realized_pnl=-1.63397697,
                    is_open=False,
                    okx_inst_id="ACT-USDT-SWAP",
                    okx_pos_id=_RepeatedPosIdLifecycleCcxt.pos_id,
                    entry_exchange_order_id=(
                        f"{_RepeatedPosIdLifecycleCcxt.old_entry_order_id},"
                        f"{_RepeatedPosIdLifecycleCcxt.new_entry_order_id}"
                    ),
                    close_exchange_order_id=(
                        f"{_RepeatedPosIdLifecycleCcxt.old_close_order_id},"
                        f"{_RepeatedPosIdLifecycleCcxt.new_close_order_id}"
                    ),
                    closed_at=old_close,
                    created_at=old_open,
                )
            )

        report = await OkxOrderFactSyncService(
            mode="paper",
            executor_factory=_RepeatedPosIdLifecycleExecutor,
            cold_start_marker_path=None,
        ).sync()

        async with get_session_ctx() as session:
            position_rows = (
                await session.execute(
                    Position.__table__
                    .select()
                    .where(Position.__table__.c.symbol == "ACT/USDT")
                    .order_by(Position.__table__.c.created_at.asc())
                )
            ).all()
            order_rows = (
                await session.execute(
                    Order.__table__
                    .select()
                    .where(Order.__table__.c.symbol == "ACT/USDT")
                    .order_by(Order.__table__.c.filled_at.asc())
                )
            ).all()

        positions = [row._mapping for row in position_rows]
        orders = [row._mapping for row in order_rows]
        profitable = next(
            row
            for row in positions
            if row["entry_exchange_order_id"] == _RepeatedPosIdLifecycleCcxt.new_entry_order_id
        )

        assert report["position_history_checked_count"] == 2
        assert report["position_history_backfilled_count"] == 1
        assert report["position_history_updated_count"] == 1
        assert len(positions) == 2
        assert sum(
            1
            for row in positions
            if row["okx_pos_id"] == _RepeatedPosIdLifecycleCcxt.pos_id
            and row["okx_inst_id"] == "ACT-USDT-SWAP"
            and row["is_open"] is False
        ) == 2
        assert len(orders) == 4
        assert positions[0]["entry_exchange_order_id"] == _RepeatedPosIdLifecycleCcxt.old_entry_order_id
        assert positions[0]["close_exchange_order_id"] == _RepeatedPosIdLifecycleCcxt.old_close_order_id
        assert _RepeatedPosIdLifecycleCcxt.new_entry_order_id not in positions[0]["entry_exchange_order_id"]
        assert _RepeatedPosIdLifecycleCcxt.new_close_order_id not in positions[0]["close_exchange_order_id"]
        assert profitable["side"] == "short"
        assert profitable["quantity"] == pytest.approx(1270.0)
        assert profitable["entry_price"] == pytest.approx(0.01049)
        assert profitable["current_price"] == pytest.approx(0.00908)
        assert profitable["realized_pnl"] == pytest.approx(1.77827305)
        assert profitable["leverage"] == pytest.approx(3.0)
        assert profitable["close_exchange_order_id"] == _RepeatedPosIdLifecycleCcxt.new_close_order_id
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_order_fact_sync_links_all_orders_inside_position_history_lifecycle(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-multi-fill-lifecycle.db').as_posix()}",
    )
    await init_db()
    try:
        report = await OkxOrderFactSyncService(
            mode="paper",
            executor_factory=_MultiFillLifecycleExecutor,
            cold_start_marker_path=None,
        ).sync()

        async with get_session_ctx() as session:
            position_rows = (
                await session.execute(
                    Position.__table__.select().where(
                        Position.__table__.c.okx_pos_id == "inj-net-lifecycle"
                    )
                )
            ).all()

        assert report["position_history_checked_count"] == 1
        assert report["position_history_backfilled_count"] == 1
        assert len(position_rows) == 1
        position = position_rows[0]._mapping
        assert position["side"] == "short"
        assert position["quantity"] == pytest.approx(3.0)
        assert position["entry_exchange_order_id"] == "inj-entry-1,inj-entry-2"
        assert position["close_exchange_order_id"] == "inj-close-1,inj-close-2"
        assert position["realized_pnl"] == pytest.approx(3.5)
    finally:
        await close_db()
