from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from services.okx_native_facts import OkxNativeFactsClient, group_okx_native_fill_rows


class _FakeCcxt:
    def __init__(self, rows: list[dict[str, Any]], *, fail_instrument: bool = False) -> None:
        self.rows = rows
        self.fail_instrument = fail_instrument
        self.params: list[dict[str, Any]] = []
        self.instrument_params: list[dict[str, Any]] = []

    async def privateGetTradeFillsHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        self.params.append(dict(params))
        if self.fail_instrument and params.get("instId"):
            raise RuntimeError("51001 instrument does not exist")
        return {"data": self.rows}

    async def publicGetPublicInstruments(self, params: dict[str, Any]) -> dict[str, Any]:
        self.instrument_params.append(dict(params))
        return {
            "data": [
                {
                    "instType": "SWAP",
                    "instId": "NEAR-USDT-SWAP",
                    "ctVal": "10",
                    "settleCcy": "USDT",
                    "state": "live",
                },
                {
                    "instType": "SWAP",
                    "instId": "BTC-USDT-SWAP",
                    "ctVal": "0.01",
                    "settleCcy": "USDT",
                    "state": "live",
                },
            ]
        }


class _NativeStateCcxt:
    def __init__(self) -> None:
        self.position_params: list[dict[str, Any]] = []
        self.pending_order_params: list[dict[str, Any]] = []
        self.algo_pending_params: list[dict[str, Any]] = []

    async def privateGetAccountPositions(self, params: dict[str, Any]) -> dict[str, Any]:
        self.position_params.append(dict(params))
        return {
            "data": [
                {
                    "instId": "SPK-USDT-SWAP",
                    "posId": "spk-pos-1",
                    "tradeId": "spk-trade-1",
                    "posSide": "net",
                    "pos": "-200",
                    "ctVal": "1",
                    "avgPx": "0.012",
                    "markPx": "0.011",
                    "upl": "0.2",
                    "fee": "-0.006",
                    "lever": "3",
                    "uTime": "1780000004000",
                },
                {
                    "instId": "HOME-USDT-SWAP",
                    "posSide": "long",
                    "pos": "0",
                    "ctVal": "1",
                    "markPx": "0.02",
                },
            ]
        }

    async def privateGetTradeOrdersPending(self, params: dict[str, Any]) -> dict[str, Any]:
        self.pending_order_params.append(dict(params))
        return {
            "data": [
                {
                    "instId": "SPK-USDT-SWAP",
                    "ordId": "spk-close-1",
                    "clOrdId": "local-close-1",
                    "side": "buy",
                    "ordType": "market",
                    "state": "live",
                    "sz": "120",
                    "accFillSz": "20",
                    "reduceOnly": "true",
                    "cTime": "1780000000000",
                    "uTime": "1780000001000",
                }
            ]
        }

    async def privateGetTradeOrdersAlgoPending(self, params: dict[str, Any]) -> dict[str, Any]:
        self.algo_pending_params.append(dict(params))
        if params.get("ordType") != "conditional":
            return {"data": []}
        return {
            "data": [
                {
                    "instId": "SPK-USDT-SWAP",
                    "algoId": "spk-tpsl-1",
                    "algoClOrdId": "local-tpsl-1",
                    "side": "buy",
                    "posSide": "short",
                    "ordType": "conditional",
                    "state": "live",
                    "tpTriggerPx": "0.0105",
                    "tpOrdPx": "-1",
                    "slTriggerPx": "0.0125",
                    "slOrdPx": "-1",
                    "cTime": "1780000000000",
                    "uTime": "1780000002000",
                },
                {
                    "instId": "OTHER-USDT-SWAP",
                    "algoId": "other-tpsl",
                    "side": "sell",
                    "posSide": "long",
                    "ordType": "conditional",
                    "state": "live",
                    "tpTriggerPx": "1",
                    "uTime": "1780000003000",
                },
            ]
        }


class _NoNativeStateCcxt:
    async def fetch_positions(self, _symbols: list[str] | None = None) -> list[dict[str, Any]]:
        raise AssertionError("native current-state reads must not use CCXT fetch_positions")

    async def fetch_open_orders(self, _symbol: str | None = None) -> list[dict[str, Any]]:
        raise AssertionError("native current-state reads must not use CCXT fetch_open_orders")


class _FakeExecutor:
    def __init__(self, ccxt: _FakeCcxt) -> None:
        self.ccxt = ccxt

    async def _get_ccxt(self) -> _FakeCcxt:
        return self.ccxt

    async def _with_retry(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        return await fn(*args, **kwargs)


class _PagedFillCcxt:
    def __init__(self, pages: dict[str, list[dict[str, Any]]]) -> None:
        self.pages = pages
        self.params: list[dict[str, Any]] = []

    async def privateGetTradeFillsHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        self.params.append(dict(params))
        if params.get("ordId"):
            return {"data": self.pages.get(str(params.get("ordId")), [])}
        cursor = str(params.get("after") or "")
        return {"data": self.pages.get(cursor, [])}


class _OrderHistoryCcxt(_FakeCcxt):
    def __init__(self) -> None:
        super().__init__([])
        self.order_history_params: list[dict[str, Any]] = []

    async def privateGetTradeOrdersHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        self.order_history_params.append(dict(params))
        return {
            "data": [
                {
                    "instId": params.get("instId", "AAVE-USDT-SWAP"),
                    "ordId": params.get("ordId"),
                    "side": "buy",
                    "reduceOnly": "true",
                    "algoId": "algo-close-1",
                    "source": "7",
                }
            ]
        }


class _PagedOrderHistoryCcxt:
    def __init__(self) -> None:
        self.params: list[dict[str, Any]] = []
        self.archive_params: list[dict[str, Any]] = []

    async def privateGetTradeOrdersHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        self.params.append(dict(params))
        timestamp = int(datetime(2026, 6, 28, 1, 0, tzinfo=UTC).timestamp() * 1000)
        if params.get("ordId"):
            return {
                "data": [
                    {
                        "instId": "SPK-USDT-SWAP",
                        "ordId": str(params["ordId"]),
                        "side": "sell",
                        "state": "filled",
                        "cTime": str(timestamp),
                        "uTime": str(timestamp),
                    }
                ]
            }
        return {
            "data": [
                {
                    "instId": "DOGE-USDT-SWAP",
                    "ordId": "account-order",
                    "side": "buy",
                    "state": "canceled",
                    "cTime": str(timestamp),
                    "uTime": str(timestamp),
                }
            ]
        }

    async def privateGetTradeOrdersHistoryArchive(self, params: dict[str, Any]) -> dict[str, Any]:
        self.archive_params.append(dict(params))
        return {"data": []}


class _ArchiveOnlyOrderHistoryCcxt:
    def __init__(self) -> None:
        self.params: list[dict[str, Any]] = []
        self.archive_params: list[dict[str, Any]] = []

    async def privateGetTradeOrdersHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        self.params.append(dict(params))
        return {"data": []}

    async def privateGetTradeOrdersHistoryArchive(self, params: dict[str, Any]) -> dict[str, Any]:
        self.archive_params.append(dict(params))
        timestamp = int(datetime(2026, 6, 28, 2, 0, tzinfo=UTC).timestamp() * 1000)
        return {
            "data": [
                {
                    "instId": "FLOKI-USDT-SWAP",
                    "ordId": str(params.get("ordId") or "archive-order"),
                    "side": "sell",
                    "state": "filled",
                    "cTime": str(timestamp),
                    "uTime": str(timestamp),
                }
            ]
        }


class _PositionHistoryCcxt:
    def __init__(self) -> None:
        self.params: list[dict[str, Any]] = []
        first_ts = int(datetime(2026, 6, 28, 1, 0, tzinfo=UTC).timestamp() * 1000)
        second_ts = int(datetime(2026, 6, 28, 2, 0, tzinfo=UTC).timestamp() * 1000)
        old_ts = int(datetime(2026, 6, 27, 1, 0, tzinfo=UTC).timestamp() * 1000)
        self.pages = {
            "": [
                {
                    "instId": "SPK-USDT-SWAP",
                    "posId": "spk-pos-1",
                    "posSide": "short",
                    "openAvgPx": "0.034",
                    "closeAvgPx": "0.031",
                    "realizedPnl": "1.23",
                    "uTime": str(first_ts),
                },
                {
                    "instId": "ARB-USDT-SWAP",
                    "posId": "arb-pos-1",
                    "posSide": "long",
                    "openAvgPx": "0.01",
                    "closeAvgPx": "0.02",
                    "realizedPnl": "0.23",
                    "uTime": str(first_ts + 1),
                },
            ],
            "spk-pos-1": [
                {
                    "instId": "FLOKI-USDT-SWAP",
                    "posId": "floki-pos-1",
                    "posSide": "short",
                    "openAvgPx": "0.00002174",
                    "closeAvgPx": "0.000021",
                    "realizedPnl": "0.108",
                    "uTime": str(second_ts),
                }
            ],
            "old": [
                {
                    "instId": "HOME-USDT-SWAP",
                    "posId": "home-old",
                    "posSide": "long",
                    "openAvgPx": "0.01",
                    "closeAvgPx": "0.02",
                    "realizedPnl": "9.99",
                    "uTime": str(old_ts),
                }
            ],
        }

    async def privateGetAccountPositionsHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        self.params.append(dict(params))
        if params.get("posId"):
            wanted = {item.strip() for item in str(params["posId"]).split(",") if item.strip()}
            rows = [
                row
                for page in self.pages.values()
                for row in page
                if row.get("posId") in wanted
            ]
            return {"data": rows}
        cursor = str(params.get("after") or "")
        return {"data": self.pages.get(cursor, [])}


def test_group_okx_native_fill_rows_uses_inst_id_not_ccxt_alias() -> None:
    timestamp = int(datetime(2026, 6, 26, 12, 0, tzinfo=UTC).timestamp() * 1000)

    groups = group_okx_native_fill_rows(
        [
            {
                "instId": "SPK-USDT-SWAP",
                "ordId": "close-spk",
                "tradeId": "trade-1",
                "side": "buy",
                "posSide": "net",
                "fillSz": "120",
                "fillPx": "0.017",
                "fee": "-0.01",
                "fillPnl": "1.2",
                "ts": str(timestamp),
            },
            {
                "instId": "SPK-USDT-SWAP",
                "ordId": "close-spk",
                "tradeId": "trade-2",
                "side": "buy",
                "fillSz": "80",
                "fillPx": "0.018",
                "fee": "-0.02",
                "fillPnl": "0.8",
                "ts": str(timestamp + 1),
            },
        ]
    )

    assert len(groups) == 1
    group = groups[0]
    assert group.inst_id == "SPK-USDT-SWAP"
    assert group.symbol == "SPK/USDT"
    assert group.order_id == "close-spk"
    assert group.trade_ids == ("trade-1", "trade-2")
    assert group.side == "buy"
    assert group.pos_side == "net"
    assert group.contracts == pytest.approx(200.0)
    assert group.avg_price == pytest.approx(((120 * 0.017) + (80 * 0.018)) / 200)
    assert group.fee_abs == pytest.approx(0.03)
    assert group.fill_pnl == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_native_facts_client_filters_by_inst_id_side_and_since() -> None:
    now = datetime.now(UTC)
    current_ts = int(now.timestamp() * 1000)
    old_ts = int((now - timedelta(hours=2)).timestamp() * 1000)
    ccxt = _FakeCcxt(
        [
            {
                "instId": "LAB-USDT-SWAP",
                "ordId": "target",
                "tradeId": "trade-1",
                "side": "sell",
                "fillSz": "9",
                "fillPx": "17.44",
                "ts": str(current_ts),
            },
            {
                "instId": "LAB-USDT-SWAP",
                "ordId": "old",
                "tradeId": "trade-old",
                "side": "sell",
                "fillSz": "9",
                "fillPx": "16",
                "ts": str(old_ts),
            },
            {
                "instId": "LAB-USDT-SWAP",
                "ordId": "wrong-side",
                "tradeId": "trade-2",
                "side": "buy",
                "fillSz": "9",
                "fillPx": "18",
                "ts": str(current_ts),
            },
            {
                "instId": "OTHER-USDT-SWAP",
                "ordId": "wrong-symbol",
                "tradeId": "trade-3",
                "side": "sell",
                "fillSz": "1",
                "fillPx": "99",
                "ts": str(current_ts),
            },
        ]
    )

    groups = await OkxNativeFactsClient(_FakeExecutor(ccxt)).fetch_fill_groups(
        inst_ids=["LAB-USDT-SWAP"],
        since=now - timedelta(minutes=5),
        side="sell",
    )

    assert [group.order_id for group in groups] == ["target"]
    since_ms = str(int((now - timedelta(minutes=5)).timestamp() * 1000))
    assert ccxt.params == [
        {
            "instType": "SWAP",
            "instId": "LAB-USDT-SWAP",
            "limit": "100",
            "begin": since_ms,
        }
    ]


@pytest.mark.asyncio
async def test_native_facts_client_paginates_fills_history_with_after_cursor() -> None:
    now = datetime.now(UTC)
    current_ts = int(now.timestamp() * 1000)
    page_1_old_cursor = "bill-page-1-old"
    ccxt = _PagedFillCcxt(
        {
            "": [
                {
                    "billId": "bill-page-1-new",
                    "instId": "LAB-USDT-SWAP",
                    "ordId": "newer",
                    "tradeId": "trade-newer",
                    "side": "buy",
                    "fillSz": "1",
                    "fillPx": "10",
                    "ts": str(current_ts),
                },
                {
                    "billId": page_1_old_cursor,
                    "instId": "LAB-USDT-SWAP",
                    "ordId": "page-1-old",
                    "tradeId": "trade-page-1-old",
                    "side": "buy",
                    "fillSz": "1",
                    "fillPx": "11",
                    "ts": str(current_ts - 1),
                },
            ],
            page_1_old_cursor: [
                {
                    "billId": "bill-page-2-target",
                    "instId": "LAB-USDT-SWAP",
                    "ordId": "older-target",
                    "tradeId": "trade-older-target",
                    "side": "buy",
                    "fillSz": "2",
                    "fillPx": "12",
                    "ts": str(current_ts - 2),
                }
            ],
        }
    )

    groups = await OkxNativeFactsClient(_FakeExecutor(ccxt)).fetch_fill_groups(
        inst_ids=["LAB-USDT-SWAP"],
        since=now - timedelta(minutes=5),
        limit=2,
        max_pages=2,
    )

    assert {group.order_id for group in groups} == {"newer", "page-1-old", "older-target"}
    since_ms = str(int((now - timedelta(minutes=5)).timestamp() * 1000))
    assert ccxt.params == [
        {"instType": "SWAP", "instId": "LAB-USDT-SWAP", "limit": "2", "begin": since_ms},
        {
            "instType": "SWAP",
            "instId": "LAB-USDT-SWAP",
            "limit": "2",
            "begin": since_ms,
            "after": page_1_old_cursor,
        },
    ]


@pytest.mark.asyncio
async def test_native_facts_client_targets_missing_order_ids_after_bounded_pull() -> None:
    timestamp = int(datetime.now(UTC).timestamp() * 1000)
    ccxt = _PagedFillCcxt(
        {
            "": [
                {
                    "billId": "bill-visible",
                    "instId": "LAB-USDT-SWAP",
                    "ordId": "visible",
                    "tradeId": "trade-visible",
                    "side": "buy",
                    "fillSz": "1",
                    "fillPx": "10",
                    "ts": str(timestamp),
                }
            ],
            "missing-target": [
                {
                    "billId": "bill-target",
                    "instId": "LAB-USDT-SWAP",
                    "ordId": "missing-target",
                    "tradeId": "trade-target",
                    "side": "buy",
                    "fillSz": "3",
                    "fillPx": "12",
                    "ts": str(timestamp),
                }
            ],
        }
    )

    groups = await OkxNativeFactsClient(_FakeExecutor(ccxt)).fetch_fill_groups(
        account_wide_only=True,
        order_ids=["missing-target"],
    )

    assert {group.order_id for group in groups} == {"visible", "missing-target"}
    assert ccxt.params == [
        {"instType": "SWAP", "limit": "100"},
        {"instType": "SWAP", "ordId": "missing-target", "limit": "100"},
    ]


@pytest.mark.asyncio
async def test_native_facts_client_fetches_order_history_context_by_order_id() -> None:
    timestamp = int(datetime.now(UTC).timestamp() * 1000)
    fill = group_okx_native_fill_rows(
        [
            {
                "instId": "AAVE-USDT-SWAP",
                "ordId": "protection-close-1",
                "tradeId": "trade-close",
                "side": "buy",
                "fillSz": "6.1",
                "fillPx": "97.54",
                "ts": str(timestamp),
            }
        ]
    )[0]
    ccxt = _OrderHistoryCcxt()

    contexts = await OkxNativeFactsClient(_FakeExecutor(ccxt)).fetch_order_history_contexts(
        fills=[fill],
        limit=5,
    )

    assert set(contexts) == {"protection-close-1"}
    assert contexts["protection-close-1"][0]["algoId"] == "algo-close-1"
    assert ccxt.order_history_params == [
        {
            "instType": "SWAP",
            "ordId": "protection-close-1",
            "limit": "5",
            "instId": "AAVE-USDT-SWAP",
        }
    ]


@pytest.mark.asyncio
async def test_native_facts_client_fetches_order_history_rows_with_phase3_begin() -> None:
    ccxt = _PagedOrderHistoryCcxt()
    since = datetime(2026, 6, 28, 0, 0, tzinfo=UTC)

    rows = await OkxNativeFactsClient(_FakeExecutor(ccxt)).fetch_order_history_rows(
        order_ids=["target-order"],
        since=since,
        limit=5,
    )

    assert [row["ordId"] for row in rows] == ["target-order"]
    assert ccxt.params == [
        {
            "instType": "SWAP",
            "ordId": "target-order",
            "limit": "5",
            "begin": str(int(since.timestamp() * 1000)),
        }
    ]


@pytest.mark.asyncio
async def test_native_facts_client_fetches_account_order_history_rows() -> None:
    ccxt = _PagedOrderHistoryCcxt()
    since = datetime(2026, 6, 28, 0, 0, tzinfo=UTC)

    rows = await OkxNativeFactsClient(_FakeExecutor(ccxt)).fetch_order_history_rows(
        since=since,
        limit=5,
    )

    assert [row["ordId"] for row in rows] == ["account-order"]
    assert ccxt.params == [
        {
            "instType": "SWAP",
            "limit": "5",
            "begin": str(int(since.timestamp() * 1000)),
        }
    ]


@pytest.mark.asyncio
async def test_native_facts_client_falls_back_to_archive_order_history_rows() -> None:
    ccxt = _ArchiveOnlyOrderHistoryCcxt()
    since = datetime(2026, 6, 28, 0, 0, tzinfo=UTC)

    rows = await OkxNativeFactsClient(_FakeExecutor(ccxt)).fetch_order_history_rows(
        order_ids=["floki-entry"],
        since=since,
        limit=5,
    )

    assert [row["ordId"] for row in rows] == ["floki-entry"]
    assert ccxt.params == [
        {
            "instType": "SWAP",
            "ordId": "floki-entry",
            "limit": "5",
            "begin": str(int(since.timestamp() * 1000)),
        }
    ]
    assert ccxt.archive_params == ccxt.params


@pytest.mark.asyncio
async def test_native_facts_client_fetches_position_history_rows_with_phase3_begin() -> None:
    ccxt = _PositionHistoryCcxt()
    since = datetime(2026, 6, 28, 0, 0, tzinfo=UTC)

    rows = await OkxNativeFactsClient(_FakeExecutor(ccxt)).fetch_position_history_rows(
        since=since,
        limit=2,
        max_pages=2,
    )

    assert [row["posId"] for row in rows] == ["floki-pos-1", "arb-pos-1", "spk-pos-1"]
    assert ccxt.params == [
        {
            "instType": "SWAP",
            "limit": "2",
            "begin": str(int(since.timestamp() * 1000)),
        },
        {
            "instType": "SWAP",
            "limit": "2",
            "begin": str(int(since.timestamp() * 1000)),
            "after": "spk-pos-1",
        },
    ]


@pytest.mark.asyncio
async def test_native_facts_client_fetches_position_history_by_pos_ids() -> None:
    ccxt = _PositionHistoryCcxt()
    since = datetime(2026, 6, 28, 0, 0, tzinfo=UTC)

    rows = await OkxNativeFactsClient(_FakeExecutor(ccxt)).fetch_position_history_rows(
        pos_ids=["floki-pos-1"],
        since=since,
        limit=5,
    )

    assert [row["posId"] for row in rows] == ["floki-pos-1"]
    assert ccxt.params == [
        {
            "instType": "SWAP",
            "posId": "floki-pos-1",
            "limit": "5",
            "begin": str(int(since.timestamp() * 1000)),
        }
    ]


@pytest.mark.asyncio
async def test_native_facts_client_prioritizes_explicit_order_history_ids() -> None:
    timestamp = int(datetime.now(UTC).timestamp() * 1000)
    fills = group_okx_native_fill_rows(
        [
            {
                "instId": "AAVE-USDT-SWAP",
                "ordId": f"noise-{index}",
                "tradeId": f"noise-trade-{index}",
                "side": "buy",
                "fillSz": "1",
                "fillPx": "97",
                "ts": str(timestamp + index),
            }
            for index in range(3)
        ]
        + [
            {
                "instId": "AAVE-USDT-SWAP",
                "ordId": "priority-close",
                "tradeId": "priority-trade",
                "side": "buy",
                "fillSz": "6.1",
                "fillPx": "97.54",
                "ts": str(timestamp - 1),
            }
        ]
    )
    ccxt = _OrderHistoryCcxt()

    contexts = await OkxNativeFactsClient(_FakeExecutor(ccxt)).fetch_order_history_contexts(
        fills=fills,
        order_ids=["priority-close"],
        limit=5,
        max_queries=2,
    )

    assert "priority-close" in contexts
    assert ccxt.order_history_params[0]["ordId"] == "priority-close"
    assert ccxt.order_history_params[0]["instId"] == "AAVE-USDT-SWAP"
    assert len(ccxt.order_history_params) == 2


@pytest.mark.asyncio
async def test_native_facts_client_account_wide_only_uses_single_fill_pull() -> None:
    timestamp = int(datetime.now(UTC).timestamp() * 1000)
    ccxt = _FakeCcxt(
        [
            {
                "instId": "LAB-USDT-SWAP",
                "ordId": "target-lab",
                "tradeId": "trade-1",
                "side": "sell",
                "fillSz": "9",
                "fillPx": "17.44",
                "ts": str(timestamp),
            },
            {
                "instId": "OTHER-USDT-SWAP",
                "ordId": "other",
                "tradeId": "trade-2",
                "side": "sell",
                "fillSz": "1",
                "fillPx": "99",
                "ts": str(timestamp),
            },
        ]
    )

    groups = await OkxNativeFactsClient(_FakeExecutor(ccxt)).fetch_fill_groups(
        inst_ids=["LAB-USDT-SWAP", "NEAR-USDT-SWAP"],
        account_wide_only=True,
    )

    assert [group.order_id for group in groups] == ["target-lab"]
    assert ccxt.params == [{"instType": "SWAP", "limit": "100"}]


@pytest.mark.asyncio
async def test_native_facts_client_uses_account_wide_fallback_for_missing_instrument() -> None:
    timestamp = int(datetime.now(UTC).timestamp() * 1000)
    ccxt = _FakeCcxt(
        [
            {
                "instId": "LAB-USDT-SWAP",
                "ordId": "close-lab",
                "tradeId": "trade-1",
                "side": "sell",
                "fillSz": "9",
                "fillPx": "17.44",
                "ts": str(timestamp),
            },
            {
                "instId": "OTHER-USDT-SWAP",
                "ordId": "other",
                "tradeId": "trade-2",
                "side": "sell",
                "fillSz": "9",
                "fillPx": "99",
                "ts": str(timestamp),
            },
        ],
        fail_instrument=True,
    )

    groups = await OkxNativeFactsClient(_FakeExecutor(ccxt)).fetch_fill_groups(
        symbols=["LAB/USDT"],
        side="sell",
    )

    assert [group.order_id for group in groups] == ["close-lab"]
    assert ccxt.params == [
        {"instType": "SWAP", "instId": "LAB-USDT-SWAP", "limit": "100"},
        {"instType": "SWAP", "limit": "100"},
    ]


@pytest.mark.asyncio
async def test_native_facts_client_strict_fill_groups_raise_when_all_reads_fail() -> None:
    ccxt = _FakeCcxt([], fail_instrument=True)

    async def fail_account_wide(params: dict[str, Any]) -> dict[str, Any]:
        ccxt.params.append(dict(params))
        raise RuntimeError("OKX fills history unavailable")

    ccxt.privateGetTradeFillsHistory = fail_account_wide  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="OKX fills history unavailable"):
        await OkxNativeFactsClient(_FakeExecutor(ccxt)).fetch_fill_groups(
            symbols=["LAB/USDT"],
            side="sell",
            strict=True,
        )

    assert ccxt.params == [
        {"instType": "SWAP", "instId": "LAB-USDT-SWAP", "limit": "100"},
        {"instType": "SWAP", "limit": "100"},
    ]


@pytest.mark.asyncio
async def test_native_facts_client_fetch_positions_uses_signed_okx_net_position() -> None:
    ccxt = _NativeStateCcxt()

    positions = await OkxNativeFactsClient(_FakeExecutor(ccxt)).fetch_positions(
        symbols=["SPK/USDT"]
    )

    assert len(positions) == 1
    assert positions[0]["symbol"] == "SPK-USDT-SWAP"
    assert positions[0]["side"] == "short"
    assert positions[0]["contracts"] == pytest.approx(200.0)
    assert positions[0]["contractSize"] == pytest.approx(1.0)
    assert positions[0]["info"]["instId"] == "SPK-USDT-SWAP"
    assert positions[0]["info"]["posId"] == "spk-pos-1"
    assert positions[0]["info"]["tradeId"] == "spk-trade-1"
    assert positions[0]["info"]["fee"] == "-0.006"
    assert ccxt.position_params == [{"instType": "SWAP", "instId": "SPK-USDT-SWAP"}]


@pytest.mark.asyncio
async def test_native_facts_client_fetch_contract_sizes_uses_okx_public_instruments() -> None:
    ccxt = _FakeCcxt([])

    sizes = await OkxNativeFactsClient(_FakeExecutor(ccxt)).fetch_contract_sizes(
        inst_ids=["NEAR-USDT-SWAP"]
    )

    assert sizes == {"NEAR-USDT-SWAP": pytest.approx(10.0)}
    assert ccxt.instrument_params == [{"instType": "SWAP"}]


@pytest.mark.asyncio
async def test_native_facts_client_fetch_open_orders_uses_okx_pending_orders() -> None:
    ccxt = _NativeStateCcxt()

    orders = await OkxNativeFactsClient(_FakeExecutor(ccxt)).fetch_open_orders(
        symbols=["SPK/USDT"]
    )

    assert len(orders) == 1
    assert orders[0]["id"] == "spk-close-1"
    assert orders[0]["symbol"] == "SPK-USDT-SWAP"
    assert orders[0]["side"] == "buy"
    assert orders[0]["reduceOnly"] is True
    assert orders[0]["remaining"] == pytest.approx(100.0)
    assert ccxt.pending_order_params == [
        {"instType": "SWAP", "instId": "SPK-USDT-SWAP", "limit": "100"}
    ]


@pytest.mark.asyncio
async def test_native_facts_client_fetch_position_protection_orders_uses_okx_algo_pending() -> None:
    ccxt = _NativeStateCcxt()

    orders = await OkxNativeFactsClient(_FakeExecutor(ccxt)).fetch_position_protection_orders(
        symbols=["SPK/USDT"]
    )

    assert len(orders) == 1
    assert orders[0]["symbol"] == "SPK/USDT"
    assert orders[0]["position_side"] == "short"
    assert orders[0]["close_side"] == "buy"
    assert orders[0]["order_type"] == "conditional"
    assert orders[0]["take_profit_price"] == pytest.approx(0.0105)
    assert orders[0]["stop_loss_price"] == pytest.approx(0.0125)
    assert orders[0]["algo_id"] == "spk-tpsl-1"
    assert ccxt.algo_pending_params == [
        {
            "instType": "SWAP",
            "instId": "SPK-USDT-SWAP",
            "ordType": "conditional",
            "limit": "100",
        },
        {
            "instType": "SWAP",
            "instId": "SPK-USDT-SWAP",
            "ordType": "oco",
            "limit": "100",
        },
        {
            "instType": "SWAP",
            "instId": "SPK-USDT-SWAP",
            "ordType": "trigger",
            "limit": "100",
        },
        {
            "instType": "SWAP",
            "instId": "SPK-USDT-SWAP",
            "ordType": "move_order_stop",
            "limit": "100",
        },
    ]


@pytest.mark.asyncio
async def test_native_current_state_raises_when_okx_native_api_missing() -> None:
    client = OkxNativeFactsClient(_FakeExecutor(_NoNativeStateCcxt()))

    with pytest.raises(RuntimeError, match="native positions API is unavailable"):
        await client.fetch_positions(symbols=["SPK/USDT"])

    with pytest.raises(RuntimeError, match="native pending-orders API is unavailable"):
        await client.fetch_open_orders(symbols=["SPK/USDT"])

    with pytest.raises(RuntimeError, match="native algo pending-orders API is unavailable"):
        await client.fetch_position_protection_orders(symbols=["SPK/USDT"])
