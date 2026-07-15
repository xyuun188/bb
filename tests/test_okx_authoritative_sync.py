from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.decision import AIDecision
from models.trade import Order, Position
from services.okx_authoritative_sync import (
    MAX_AUTHORITATIVE_FILL_PAGES,
    OkxAuthoritativeSyncService,
    OkxFillGroup,
)


class _FakeCcxt:
    def __init__(self, *, timestamp_ms: int | None = None) -> None:
        self.timestamp_ms = timestamp_ms or int(datetime.now(UTC).timestamp() * 1000)

    async def privateGetTradeFillsHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        assert params["instType"] == "SWAP"
        return {
            "data": [
                {
                    "ordId": "okx-only-fill",
                    "tradeId": "trade-1",
                    "instId": "BTC-USDT-SWAP",
                    "side": "sell",
                    "fillSz": "2",
                    "fillPx": "90",
                    "fee": "-0.01",
                    "fillPnl": "0",
                    "ts": str(self.timestamp_ms),
                }
            ]
        }

    async def publicGetPublicInstruments(self, params: dict[str, Any]) -> dict[str, Any]:
        assert params["instType"] == "SWAP"
        return {
            "data": [
                {"instId": "BTC-USDT-SWAP", "ctVal": "0.01"},
                {"instId": "ETH-USDT-SWAP", "ctVal": "1"},
                {"instId": "SPK-USDT-SWAP", "ctVal": "1"},
            ]
        }


class _FakeExecutor:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.shutdown_called = False

    async def initialize(self) -> None:
        return None

    async def get_positions_strict(self) -> list[dict[str, Any]]:
        return [
            {
                "symbol": "ETH/USDT:USDT",
                "side": "short",
                "contracts": 2,
                "markPrice": 90,
                "entryPrice": 100,
                "info": {
                    "instId": "ETH-USDT-SWAP",
                    "posSide": "short",
                    "pos": "-2",
                    "ctVal": "1",
                    "avgPx": "100",
                    "markPx": "90",
                    "upl": "20",
                },
            }
        ]

    async def _get_ccxt(self) -> _FakeCcxt:
        return _FakeCcxt()

    async def _with_retry(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        return await fn(*args, **kwargs)

    async def shutdown(self) -> None:
        self.shutdown_called = True


class _AgedUnlinkedFillExecutor(_FakeExecutor):
    async def _get_ccxt(self) -> _FakeCcxt:
        timestamp_ms = int((datetime.now(UTC) - timedelta(minutes=10)).timestamp() * 1000)
        return _FakeCcxt(timestamp_ms=timestamp_ms)


class _FreshUnlinkedFillExecutor(_FakeExecutor):
    async def get_positions_strict(self) -> list[dict[str, Any]]:
        return []


class _LimitBackfillFillCcxt(_FakeCcxt):
    async def privateGetTradeFillsHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        assert params["instType"] == "SWAP"
        timestamp_ms = int((datetime.now(UTC) - timedelta(minutes=10)).timestamp() * 1000)
        return {
            "data": [
                {
                    "ordId": "older-matched-order",
                    "tradeId": "older-matched-trade",
                    "instId": "BTC-USDT-SWAP",
                    "side": "buy",
                    "posSide": "net",
                    "fillSz": "2",
                    "fillPx": "100",
                    "fee": "-0.01",
                    "fillPnl": "0",
                    "ts": str(timestamp_ms),
                }
            ]
        }


class _LimitBackfillFillExecutor(_FakeExecutor):
    async def get_positions_strict(self) -> list[dict[str, Any]]:
        return []

    async def _get_ccxt(self) -> _LimitBackfillFillCcxt:
        return _LimitBackfillFillCcxt()


class _RetryOnceExecutor(_FakeExecutor):
    attempts = 0

    async def initialize(self) -> None:
        type(self).attempts += 1
        if type(self).attempts == 1:
            raise TimeoutError("temporary okx timeout")
        return None


class _AlwaysTimeoutExecutor(_FakeExecutor):
    async def initialize(self) -> None:
        raise TimeoutError("persistent okx timeout")


class _SlowFillCcxt(_FakeCcxt):
    async def privateGetTradeFillsHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        import asyncio

        await asyncio.sleep(0.75)
        return await super().privateGetTradeFillsHistory(params)


class _FillTimeoutExecutor(_FakeExecutor):
    async def get_positions_strict(self) -> list[dict[str, Any]]:
        return []

    async def _get_ccxt(self) -> _SlowFillCcxt:
        return _SlowFillCcxt()


class _LifecycleCoveredFillCcxt(_FakeCcxt):
    def __init__(self, *, timestamp_ms: int) -> None:
        super().__init__(timestamp_ms=timestamp_ms)

    async def privateGetTradeFillsHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        assert params["instType"] == "SWAP"
        return {
            "data": [
                {
                    "ordId": "lab-partial-close",
                    "tradeId": "lab-partial-close-trade",
                    "instId": "LAB-USDT-SWAP",
                    "side": "buy",
                    "posSide": "net",
                    "fillSz": "197",
                    "fillPx": "8.03",
                    "fee": "-0.0790955",
                    "fillPnl": "32.46256117",
                    "ts": str(self.timestamp_ms),
                }
            ]
        }

    async def publicGetPublicInstruments(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "data": [
                {"instId": "LAB-USDT-SWAP", "ctVal": "0.1"},
            ]
        }


class _LifecycleCoveredFillExecutor(_FakeExecutor):
    timestamp_ms = int(datetime(2026, 7, 4, 9, 33, 57, tzinfo=UTC).timestamp() * 1000)

    async def get_positions_strict(self) -> list[dict[str, Any]]:
        return []

    async def _get_ccxt(self) -> _LifecycleCoveredFillCcxt:
        return _LifecycleCoveredFillCcxt(timestamp_ms=self.timestamp_ms)


class _SlowOptionalStageService(OkxAuthoritativeSyncService):
    async def _fetch_order_history_contexts(
        self,
        executor: Any,
        *,
        exchange_fills: list[OkxFillGroup],
        local_exchange_order_ids: set[str],
        priority_order_ids: set[str] | None = None,
    ) -> dict[str, tuple[dict[str, Any], ...]]:
        import asyncio

        await asyncio.sleep(0.75)
        return await super()._fetch_order_history_contexts(
            executor,
            exchange_fills=exchange_fills,
            local_exchange_order_ids=local_exchange_order_ids,
            priority_order_ids=priority_order_ids,
        )

    async def _fetch_contract_sizes(
        self,
        executor: Any,
        *,
        symbols: set[str],
        inst_ids: set[str],
    ) -> dict[str, float]:
        import asyncio

        await asyncio.sleep(0.75)
        return await super()._fetch_contract_sizes(
            executor,
            symbols=symbols,
            inst_ids=inst_ids,
        )


@pytest.mark.asyncio
async def test_okx_authoritative_sync_reports_exchange_and_local_mismatches(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-authoritative.db').as_posix()}",
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
                        symbol="BTC/USDT",
                        side="buy",
                        order_type="market",
                        quantity=1.0,
                        price=100.0,
                        status="filled",
                        exchange_order_id="local-only-order",
                        filled_at=now - timedelta(minutes=5),
                        created_at=now - timedelta(minutes=5),
                    ),
                    Position(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="BTC/USDT",
                        side="long",
                        quantity=1.0,
                        entry_price=100.0,
                        current_price=101.0,
                        is_open=True,
                        entry_exchange_order_id="local-only-order",
                        created_at=now - timedelta(minutes=5),
                    ),
                ]
            )

        report = await OkxAuthoritativeSyncService(
            mode="paper",
            lookback_hours=24,
            executor_factory=_AgedUnlinkedFillExecutor,
        ).collect()

        kinds = {issue["kind"] for issue in report["issues"]}
        assert report["read_only"] is True
        assert report["audit_only"] is True
        assert report["okx_pull_available"] is True
        assert report["status"] == "critical"
        assert report["apply_policy"]["can_write_database"] is False
        assert report["apply_policy"]["requires_backup"] is True
        assert "okx_open_position_missing_locally" in kinds
        assert "local_open_position_missing_on_okx" in kinds
        assert "okx_fill_missing_local_order" in kinds
        assert "local_order_not_found_in_recent_okx_fills" in kinds
        assert report["classification_counts"]["manual_review"] >= 3
        assert report["classification_counts"]["skipped"] >= 1
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_okx_authoritative_sync_retries_transient_pull_timeout(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    _RetryOnceExecutor.attempts = 0
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-authoritative-retry.db').as_posix()}",
    )
    await init_db()
    try:
        report = await OkxAuthoritativeSyncService(
            mode="paper",
            executor_factory=_RetryOnceExecutor,
            max_pull_attempts=2,
        ).collect()

        assert report["okx_pull_available"] is True
        assert report["fetch_errors"] == []
        assert report["pull_attempts"] == 2
        assert report["pull_success_attempt"] == 2
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_okx_authoritative_sync_does_not_diff_empty_okx_when_pull_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-authoritative-timeout.db').as_posix()}",
    )
    await init_db()
    now = datetime.now(UTC)
    try:
        async with get_session_ctx() as session:
            session.add(
                Position(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="BTC/USDT",
                    side="long",
                    quantity=1.0,
                    entry_price=100.0,
                    current_price=101.0,
                    is_open=True,
                    entry_exchange_order_id="local-only-order",
                    created_at=now - timedelta(minutes=5),
                )
            )

        report = await OkxAuthoritativeSyncService(
            mode="paper",
            executor_factory=_AlwaysTimeoutExecutor,
            max_pull_attempts=2,
        ).collect()

        assert report["okx_pull_available"] is False
        assert report["status"] == "warning"
        assert report["issues"] == []
        assert report["issue_count"] == 0
        assert report["pull_attempts"] == 2
        assert report["pull_success_attempt"] is None
        assert report["fetch_errors"][0]["stage"] == "okx_initialize"
        assert report["pull_stages"][0]["stage"] == "okx_initialize"
        assert report["pull_stages"][0]["status"] == "error"
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_okx_authoritative_sync_reports_exact_timeout_stage(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-authoritative-stage.db').as_posix()}",
    )
    await init_db()
    try:
        report = await OkxAuthoritativeSyncService(
            mode="paper",
            executor_factory=_FillTimeoutExecutor,
            max_pull_attempts=1,
            timeout_seconds=0.001,
        ).collect()

        assert report["okx_pull_available"] is False
        assert report["issues"] == []
        assert report["fetch_errors"][0]["stage"] == "okx_fills"
        stages = report["pull_stages"]
        assert [item["stage"] for item in stages] == [
            "okx_initialize",
            "okx_positions",
            "okx_fills",
        ]
        assert stages[-1]["status"] == "error"
        assert stages[-1]["error_type"] == "TimeoutError"
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_okx_authoritative_sync_observes_fresh_fill_until_local_order_sync_window_expires(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-authoritative-fresh-fill.db').as_posix()}",
    )
    await init_db()
    try:
        report = await OkxAuthoritativeSyncService(
            mode="paper",
            lookback_hours=24,
            executor_factory=_FreshUnlinkedFillExecutor,
        ).collect()

        assert report["status"] == "ok"
        assert report["issue_count"] == 0
        assert report["pending_local_order_sync_count"] == 1
        assert report["issues"] == []
        assert report["observations"][0]["kind"] == "okx_fill_pending_local_order_sync"
        assert report["observations"][0]["severity"] == "info"
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_okx_authoritative_sync_soft_fails_optional_enrichment_stages(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-authoritative-optional-stage.db').as_posix()}",
    )
    await init_db()
    now = datetime.now(UTC)
    try:
        async with get_session_ctx() as session:
            session.add(
                Order(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="BTC/USDT",
                    side="buy",
                    order_type="market",
                    quantity=1.0,
                    price=100.0,
                    status="filled",
                    exchange_order_id="okx-only-fill",
                    filled_at=now - timedelta(minutes=5),
                    created_at=now - timedelta(minutes=5),
                )
            )

        report = await _SlowOptionalStageService(
            mode="paper",
            executor_factory=_FakeExecutor,
            timeout_seconds=0.001,
            max_pull_attempts=1,
        ).collect()

        assert report["okx_pull_available"] is True
        assert report["pull_success_attempt"] == 1
        warning_stages = [
            item for item in report["pull_stages"] if item["stage"] in {"okx_order_history_contexts", "okx_contract_sizes"}
        ]
        assert all(item["status"] == "warning" for item in warning_stages)
    finally:
        await close_db()


class _OldFillExecutor(_FakeExecutor):
    async def _get_ccxt(self) -> _FakeCcxt:
        old_timestamp_ms = int((datetime.now(UTC) - timedelta(hours=25)).timestamp() * 1000)
        return _FakeCcxt(timestamp_ms=old_timestamp_ms)


class _PreResetFillExecutor(_FakeExecutor):
    async def get_positions_strict(self) -> list[dict[str, Any]]:
        return []

    async def _get_ccxt(self) -> _FakeCcxt:
        pre_reset_timestamp_ms = int(
            (datetime.now(UTC) - timedelta(hours=2)).timestamp() * 1000
        )
        return _FakeCcxt(timestamp_ms=pre_reset_timestamp_ms)


class _SpkPositionExecutor(_FakeExecutor):
    async def get_positions_strict(self) -> list[dict[str, Any]]:
        return [
            {
                "symbol": "SPK/USDT:USDT",
                "side": "long",
                "contracts": 12,
                "markPrice": 0.034,
                "entryPrice": 0.031,
                "info": {
                    "instId": "SPK-USDT-SWAP",
                    "posSide": "long",
                    "pos": "12",
                    "ctVal": "1",
                    "avgPx": "0.031",
                    "markPx": "0.034",
                    "upl": "0.036",
                },
            }
        ]

    async def _get_ccxt(self) -> _FakeCcxt:
        old_timestamp_ms = int((datetime.now(UTC) - timedelta(hours=25)).timestamp() * 1000)
        return _FakeCcxt(timestamp_ms=old_timestamp_ms)


class _SpkFillCcxt:
    def __init__(self) -> None:
        self.params: list[dict[str, Any]] = []

    async def privateGetTradeFillsHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        self.params.append(dict(params))
        if params.get("instId") and params.get("instId") != "SPK-USDT-SWAP":
            return {"data": []}
        return {
            "data": [
                {
                    "ordId": "spk-order-1",
                    "tradeId": "spk-trade-1",
                    "instId": "SPK-USDT-SWAP",
                    "side": "buy",
                    "fillSz": "12",
                    "fillPx": "0.031",
                    "fee": "-0.01",
                    "fillPnl": "0",
                    "ts": str(int(datetime.now(UTC).timestamp() * 1000)),
                }
            ]
        }

    async def publicGetPublicInstruments(self, params: dict[str, Any]) -> dict[str, Any]:
        assert params["instType"] == "SWAP"
        return {"data": [{"instId": "SPK-USDT-SWAP", "ctVal": "1"}]}


class _SpkFillExecutor(_FakeExecutor):
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        super().__init__(*_args, **_kwargs)
        self.ccxt = _SpkFillCcxt()

    async def get_positions_strict(self) -> list[dict[str, Any]]:
        return []

    async def _get_ccxt(self) -> _SpkFillCcxt:
        return self.ccxt


class _NearFillCcxt:
    def __init__(self) -> None:
        self.params: list[dict[str, Any]] = []
        self.instrument_params: list[dict[str, Any]] = []

    async def privateGetTradeFillsHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        self.params.append(dict(params))
        if params.get("instId") and params.get("instId") != "NEAR-USDT-SWAP":
            return {"data": []}
        return {
            "data": [
                {
                    "ordId": "near-order-1",
                    "tradeId": "near-trade-1",
                    "instId": "NEAR-USDT-SWAP",
                    "side": "buy",
                    "fillSz": "3.7",
                    "fillPx": "2.31",
                    "fee": "-0.01",
                    "fillPnl": "0",
                    "ts": str(int(datetime.now(UTC).timestamp() * 1000)),
                }
            ]
        }

    async def publicGetPublicInstruments(self, params: dict[str, Any]) -> dict[str, Any]:
        self.instrument_params.append(dict(params))
        assert params["instType"] == "SWAP"
        return {"data": [{"instId": "NEAR-USDT-SWAP", "ctVal": "10"}]}


class _NearFillExecutor(_FakeExecutor):
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        super().__init__(*_args, **_kwargs)
        self.ccxt = _NearFillCcxt()

    async def get_positions_strict(self) -> list[dict[str, Any]]:
        return []

    async def _get_ccxt(self) -> _NearFillCcxt:
        return self.ccxt


class _LabFillNoInstrumentCcxt:
    async def privateGetTradeFillsHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        if params.get("instId") and params.get("instId") != "LAB-USDT-SWAP":
            return {"data": []}
        return {
            "data": [
                {
                    "ordId": "lab-order-1",
                    "tradeId": "lab-trade-1",
                    "instId": "LAB-USDT-SWAP",
                    "side": "sell",
                    "fillSz": "11",
                    "fillPx": "16.436",
                    "fee": "-0.00904",
                    "fillPnl": "-0.1217",
                    "ts": str(int(datetime.now(UTC).timestamp() * 1000)),
                }
            ]
        }

    async def publicGetPublicInstruments(self, params: dict[str, Any]) -> dict[str, Any]:
        assert params["instType"] == "SWAP"
        return {"data": []}


class _LabFillNoInstrumentExecutor(_FakeExecutor):
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        super().__init__(*_args, **_kwargs)
        self.ccxt = _LabFillNoInstrumentCcxt()

    async def get_positions_strict(self) -> list[dict[str, Any]]:
        return []

    async def _get_ccxt(self) -> _LabFillNoInstrumentCcxt:
        return self.ccxt


class _FlokiPositionOnlyCcxt:
    async def privateGetTradeFillsHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        assert params["instType"] == "SWAP"
        return {"data": []}

    async def publicGetPublicInstruments(self, params: dict[str, Any]) -> dict[str, Any]:
        assert params["instType"] == "SWAP"
        return {"data": [{"instId": "FLOKI-USDT-SWAP", "ctVal": "100000"}]}


class _FlokiPositionOnlyExecutor(_FakeExecutor):
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        super().__init__(*_args, **_kwargs)
        self.ccxt = _FlokiPositionOnlyCcxt()

    async def get_positions_strict(self) -> list[dict[str, Any]]:
        return [
            {
                "symbol": "FLOKI/USDT:USDT",
                "side": "short",
                "contracts": 6,
                "markPrice": 0.00002156,
                "entryPrice": 0.00002174,
                "info": {
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
                    "uTime": "1782637993421",
                },
            }
        ]

    async def _get_ccxt(self) -> _FlokiPositionOnlyCcxt:
        return self.ccxt


class _TargetOrderFillCcxt:
    def __init__(self) -> None:
        self.params: list[dict[str, Any]] = []
        self.instrument_params: list[dict[str, Any]] = []

    async def privateGetTradeFillsHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        self.params.append(dict(params))
        if params.get("ordId") == "target-order-1":
            return {
                "data": [
                    {
                        "ordId": "target-order-1",
                        "tradeId": "target-trade-1",
                        "instId": "OP-USDT-SWAP",
                        "side": "buy",
                        "fillSz": "242",
                        "fillPx": "0.91",
                        "fee": "-0.01",
                        "fillPnl": "0",
                        "ts": str(int(datetime.now(UTC).timestamp() * 1000)),
                    }
                ]
            }
        return {
            "data": [
                {
                    "ordId": "recent-unrelated",
                    "tradeId": "recent-trade-1",
                    "instId": "BTC-USDT-SWAP",
                    "side": "sell",
                    "fillSz": "1",
                    "fillPx": "90000",
                    "fee": "-0.01",
                    "fillPnl": "0",
                    "ts": str(int(datetime.now(UTC).timestamp() * 1000)),
                }
            ]
        }

    async def publicGetPublicInstruments(self, params: dict[str, Any]) -> dict[str, Any]:
        self.instrument_params.append(dict(params))
        assert params["instType"] == "SWAP"
        return {"data": [{"instId": "OP-USDT-SWAP", "ctVal": "1"}]}


class _TargetOrderFillExecutor(_FakeExecutor):
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        super().__init__(*_args, **_kwargs)
        self.ccxt = _TargetOrderFillCcxt()

    async def get_positions_strict(self) -> list[dict[str, Any]]:
        return []

    async def _get_ccxt(self) -> _TargetOrderFillCcxt:
        return self.ccxt


class _LinkedProtectionFillCcxt:
    def __init__(self) -> None:
        self.fill_params: list[dict[str, Any]] = []
        self.order_history_params: list[dict[str, Any]] = []
        self.instrument_params: list[dict[str, Any]] = []
        self.timestamp_ms = int(datetime.now(UTC).timestamp() * 1000)

    async def privateGetTradeFillsHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        self.fill_params.append(dict(params))
        order_id = str(params.get("ordId") or "")
        if order_id == "aave-entry-order":
            return {
                "data": [
                    {
                        "ordId": "aave-entry-order",
                        "tradeId": "entry-trade",
                        "instId": "AAVE-USDT-SWAP",
                        "side": "sell",
                        "fillSz": "6.1",
                        "fillPx": "93.58",
                        "fee": "-0.028",
                        "fillPnl": "0",
                        "ts": str(self.timestamp_ms - 60000),
                    }
                ]
            }
        return {
            "data": [
                {
                    "ordId": "aave-protection-close",
                    "tradeId": "close-trade",
                    "instId": "AAVE-USDT-SWAP",
                    "side": "buy",
                    "fillSz": "6.1",
                    "fillPx": "97.54",
                    "fee": "-0.029",
                    "fillPnl": "-1.84",
                    "ts": str(self.timestamp_ms),
                    "clOrdId": "Oaave-protection-close",
                    "tag": "6b9ad766b55dBCDE",
                }
            ]
        }

    async def privateGetTradeOrdersHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        self.order_history_params.append(dict(params))
        if params.get("ordId") == "aave-protection-close":
            return {
                "data": [
                    {
                        "ordId": "aave-protection-close",
                        "instId": "AAVE-USDT-SWAP",
                        "side": "buy",
                        "reduceOnly": "true",
                        "algoId": "aave-oco-triggered",
                        "source": "7",
                        "clOrdId": "Oaave-protection-close",
                    },
                    {
                        "ordId": "aave-entry-order",
                        "instId": "AAVE-USDT-SWAP",
                        "side": "sell",
                        "reduceOnly": "false",
                        "attachAlgoOrds": [
                            {
                                "attachAlgoId": "aave-oco-attached",
                                "tpTriggerPx": "81.88",
                                "slTriggerPx": "97.5",
                                "tpOrdPx": "-1",
                                "slOrdPx": "-1",
                            }
                        ],
                    },
                ]
            }
        return {"data": []}

    async def privateGetTradeOrderAlgoDetails(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        assert params == {"algoId": "aave-oco-triggered"}
        return {
            "data": [
                {
                    "algoId": "aave-oco-triggered",
                    "instId": "AAVE-USDT-SWAP",
                    "ordId": "aave-protection-close",
                    "ordType": "oco",
                    "side": "buy",
                    "posSide": "net",
                    "actualSide": "sl",
                    "state": "effective",
                    "actualSz": "6.1",
                    "slTriggerPx": "97.5",
                    "tpTriggerPx": "81.88",
                    "triggerTime": str(self.timestamp_ms - 25),
                    "cTime": str(self.timestamp_ms - 60000),
                    "uTime": str(self.timestamp_ms - 25),
                }
            ]
        }

    async def publicGetPublicInstruments(self, params: dict[str, Any]) -> dict[str, Any]:
        self.instrument_params.append(dict(params))
        return {"data": [{"instId": "AAVE-USDT-SWAP", "ctVal": "0.1"}]}


class _LinkedProtectionFillExecutor(_FakeExecutor):
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        super().__init__(*_args, **_kwargs)
        self.ccxt = _LinkedProtectionFillCcxt()

    async def get_positions_strict(self) -> list[dict[str, Any]]:
        return [
            {
                "symbol": "AAVE-USDT-SWAP",
                "side": "short",
                "contracts": 2.2,
                "markPrice": 93.83,
                "entryPrice": 94.51,
                "info": {
                    "instId": "AAVE-USDT-SWAP",
                    "posSide": "net",
                    "pos": "-2.2",
                    "ctVal": "0.1",
                    "avgPx": "94.51",
                    "markPx": "93.83",
                },
            }
        ]

    async def _get_ccxt(self) -> _LinkedProtectionFillCcxt:
        return self.ccxt


@pytest.mark.asyncio
async def test_okx_authoritative_sync_ignores_okx_fills_outside_window(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-authoritative-window.db').as_posix()}",
    )
    await init_db()
    try:
        report = await OkxAuthoritativeSyncService(
            mode="paper",
            lookback_hours=24,
            executor_factory=_OldFillExecutor,
        ).collect()

        kinds = {issue["kind"] for issue in report["issues"]}
        assert report["okx_fill_order_count"] == 0
        assert "okx_fill_missing_local_order" not in kinds
        assert report["status"] == "critical"
        assert report["okx_position_count"] == 1
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_okx_authoritative_sync_targets_local_exchange_order_ids_not_in_first_pull(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-authoritative-target-order.db').as_posix()}",
    )
    await init_db()
    now = datetime.now(UTC)
    try:
        async with get_session_ctx() as session:
            decision = AIDecision(
                model_name="ensemble_trader",
                symbol="OP/USDT",
                action="long",
                confidence=0.8,
                reasoning="targeted order-id fill lookup",
                is_paper=True,
                was_executed=True,
                raw_llm_response={
                    "execution_result": {
                        "raw_response": {
                            "info": {"instId": "OP-USDT-SWAP", "ctVal": "1"},
                        }
                    }
                },
            )
            session.add(decision)
            await session.flush()
            session.add(
                Order(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="OP/USDT",
                    side="buy",
                    order_type="market",
                    quantity=242.0,
                    price=0.91,
                    status="filled",
                    decision_id=decision.id,
                    exchange_order_id="target-order-1",
                    filled_at=now - timedelta(minutes=5),
                    created_at=now - timedelta(minutes=5),
                )
            )

        report = await OkxAuthoritativeSyncService(
            mode="paper",
            lookback_hours=24,
            executor_factory=_TargetOrderFillExecutor,
        ).collect()

        kinds = {issue["kind"] for issue in report["issues"]}
        assert "local_order_not_found_in_recent_okx_fills" not in kinds
        assert "local_order_quantity_differs_from_okx_fill" not in kinds
        assert "target-order-1" in {sample["order_id"] for sample in report["okx_fill_samples"]}
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_okx_authoritative_sync_classifies_linked_protection_fill_missing_order(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-authoritative-linked-protection.db').as_posix()}",
    )
    await init_db()
    now = datetime.now(UTC)
    try:
        async with get_session_ctx() as session:
            decision = AIDecision(
                model_name="ensemble_trader",
                symbol="AAVE/USDT",
                action="short",
                confidence=0.85,
                reasoning="entry with attached OKX protection",
                is_paper=True,
                was_executed=True,
                raw_llm_response={
                    "execution_result": {
                        "raw_response": {
                            "info": {
                                "instId": "AAVE-USDT-SWAP",
                                "attachAlgoOrds": [
                                    {
                                        "attachAlgoId": "aave-oco-attached",
                                        "tpTriggerPx": "81.88",
                                        "slTriggerPx": "97.5",
                                    }
                                ],
                            },
                            "contract_size": 0.1,
                        },
                    }
                },
            )
            session.add(decision)
            await session.flush()
            session.add_all(
                [
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="AAVE/USDT",
                        side="sell",
                        order_type="market",
                        quantity=0.61,
                        price=93.58,
                        status="filled",
                        decision_id=decision.id,
                        exchange_order_id="aave-entry-order",
                        filled_at=now - timedelta(minutes=10),
                        created_at=now - timedelta(minutes=10),
                    ),
                    Position(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="AAVE/USDT",
                        side="short",
                        quantity=0.22,
                        entry_price=94.51,
                        current_price=93.83,
                        is_open=True,
                        okx_inst_id="AAVE-USDT-SWAP",
                        entry_exchange_order_id="aave-entry-order",
                        created_at=now - timedelta(minutes=10),
                    ),
                ]
            )

        report = await OkxAuthoritativeSyncService(
            mode="paper",
            lookback_hours=24,
            executor_factory=_LinkedProtectionFillExecutor,
        ).collect()

        kinds = {issue["kind"] for issue in report["issues"]}
        linked_issue = next(
            issue
            for issue in report["issues"]
            if issue["kind"] == "okx_linked_protection_fill_missing_local_order"
        )
        assert "okx_fill_missing_local_order" not in kinds
        assert linked_issue["classification"] == "repairable"
        assert linked_issue["severity"] == "warning"
        assert linked_issue["exchange_order_id"] == "aave-protection-close"
        assert linked_issue["linked_exchange_order_id"] == "aave-entry-order"
        assert linked_issue["linked_local_order_id"] is not None
        assert linked_issue["okx_algo_id"] == "aave-oco-triggered"
        execution = linked_issue["protection_execution"]
        assert execution["lifecycle_complete"] is True
        assert execution["actual_side"] == "sl"
        assert execution["configured_trigger_price"] == pytest.approx(97.5)
        assert execution["actual_trigger_market_price"] is None
        assert execution["trigger_to_first_fill_ms"] == pytest.approx(25.0)
        assert execution["fill_price"] == pytest.approx(97.54)
        assert "--create-linked-protection-fill-orders" in linked_issue["repair_entrypoint"]
        assert report["okx_order_history_context_count"] >= 1
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_okx_authoritative_sync_accepts_local_linked_protection_order_row(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-authoritative-linked-protection-local-order.db').as_posix()}",
    )
    await init_db()
    now = datetime.now(UTC)
    try:
        async with get_session_ctx() as session:
            decision = AIDecision(
                model_name="ensemble_trader",
                symbol="AAVE/USDT",
                action="short",
                confidence=0.85,
                reasoning="entry with attached OKX protection",
                is_paper=True,
                was_executed=True,
            )
            session.add(decision)
            await session.flush()
            session.add_all(
                [
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="AAVE/USDT",
                        side="sell",
                        order_type="market",
                        quantity=0.61,
                        price=93.58,
                        status="filled",
                        decision_id=decision.id,
                        exchange_order_id="aave-entry-order",
                        filled_at=now - timedelta(minutes=10),
                        created_at=now - timedelta(minutes=10),
                    ),
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="AAVE/USDT",
                        side="buy",
                        order_type="market",
                        quantity=0.61,
                        price=97.54,
                        status="filled",
                        decision_id=decision.id,
                        exchange_order_id="aave-protection-close",
                        filled_at=now - timedelta(minutes=1),
                        created_at=now - timedelta(minutes=1),
                    ),
                    Position(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="AAVE/USDT",
                        side="short",
                        quantity=0.22,
                        entry_price=94.51,
                        current_price=93.83,
                        is_open=True,
                        okx_inst_id="AAVE-USDT-SWAP",
                        entry_exchange_order_id="aave-entry-order",
                        created_at=now - timedelta(minutes=10),
                    ),
                ]
            )

        report = await OkxAuthoritativeSyncService(
            mode="paper",
            lookback_hours=24,
            executor_factory=_LinkedProtectionFillExecutor,
        ).collect()

        kinds = {issue["kind"] for issue in report["issues"]}
        assert "okx_fill_not_linked_to_position" not in kinds
        assert "okx_linked_protection_fill_not_linked_to_position" not in kinds
        assert "okx_fill_missing_local_order" not in kinds
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_okx_authoritative_sync_uses_context_entry_order_outside_window(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-authoritative-context-entry.db').as_posix()}",
    )
    await init_db()
    now = datetime.now(UTC)
    try:
        async with get_session_ctx() as session:
            decision = AIDecision(
                model_name="ensemble_trader",
                symbol="AAVE/USDT",
                action="short",
                confidence=0.85,
                reasoning="entry with attached OKX protection outside observation window",
                is_paper=True,
                was_executed=True,
            )
            session.add(decision)
            await session.flush()
            session.add_all(
                [
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="AAVE/USDT",
                        side="sell",
                        order_type="market",
                        quantity=0.61,
                        price=93.58,
                        status="filled",
                        decision_id=decision.id,
                        exchange_order_id="aave-entry-order",
                        filled_at=now - timedelta(hours=5),
                        created_at=now - timedelta(hours=5),
                    ),
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="AAVE/USDT",
                        side="buy",
                        order_type="market",
                        quantity=0.61,
                        price=97.54,
                        status="filled",
                        decision_id=decision.id,
                        exchange_order_id="aave-protection-close",
                        filled_at=now - timedelta(minutes=1),
                        created_at=now - timedelta(minutes=1),
                    ),
                    Position(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="AAVE/USDT",
                        side="short",
                        quantity=0.22,
                        entry_price=94.51,
                        current_price=93.83,
                        is_open=True,
                        okx_inst_id="AAVE-USDT-SWAP",
                        entry_exchange_order_id="aave-entry-order",
                        created_at=now - timedelta(hours=5),
                    ),
                ]
            )

        report = await OkxAuthoritativeSyncService(
            mode="paper",
            lookback_hours=2,
            executor_factory=_LinkedProtectionFillExecutor,
        ).collect()

        kinds = {issue["kind"] for issue in report["issues"]}
        assert report["context_local_order_count"] >= 1
        assert "okx_fill_not_linked_to_position" not in kinds
        assert "local_order_not_found_in_recent_okx_fills" not in kinds
        assert report["issue_count"] == 0
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_okx_authoritative_sync_does_not_block_lifecycle_covered_fill(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-authoritative-lifecycle-covered.db').as_posix()}",
    )
    await init_db()
    opened_at = datetime(2026, 7, 4, 9, 18, 29, tzinfo=UTC)
    partial_close_at = datetime(2026, 7, 4, 9, 33, 57, tzinfo=UTC)
    closed_at = datetime(2026, 7, 4, 9, 39, 37, tzinfo=UTC)
    try:
        async with get_session_ctx() as session:
            session.add_all(
                [
                    Order(
                        model_name="okx_authoritative_sync",
                        execution_mode="paper",
                        symbol="LAB/USDT",
                        side="buy",
                        order_type="market",
                        quantity=19.7,
                        price=8.03,
                        status="filled",
                        exchange_order_id="lab-partial-close",
                        filled_at=partial_close_at,
                        created_at=partial_close_at,
                        okx_inst_id="LAB-USDT-SWAP",
                        okx_fill_contracts=197.0,
                        okx_sync_status="okx_confirmed",
                        okx_raw_fills={
                            "fills_history_confirmed": True,
                            "order_id": "lab-partial-close",
                            "inst_id": "LAB-USDT-SWAP",
                            "contracts": 197.0,
                            "contract_size": 0.1,
                            "base_quantity": 19.7,
                            "avg_price": 8.03,
                            "fee_abs": 0.0790955,
                            "fill_pnl": 32.46256117,
                            "timestamp": partial_close_at.isoformat(),
                        },
                    ),
                    Position(
                        model_name="okx_authoritative_sync",
                        execution_mode="paper",
                        symbol="LAB/USDT",
                        side="short",
                        quantity=40.1,
                        entry_price=8.48,
                        current_price=8.71,
                        realized_pnl=-9.4656451,
                        is_open=False,
                        okx_inst_id="LAB-USDT-SWAP",
                        okx_pos_id="3712750691686252544",
                        entry_exchange_order_id="lab-entry",
                        close_exchange_order_id="lab-final-close",
                        created_at=opened_at,
                        closed_at=closed_at,
                    ),
                ]
            )

        report = await OkxAuthoritativeSyncService(
            mode="paper",
            lookback_hours=24,
            executor_factory=_LifecycleCoveredFillExecutor,
        ).collect()

        kinds = {issue["kind"] for issue in report["issues"]}
        assert "okx_fill_not_linked_to_position" not in kinds
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_okx_authoritative_sync_prioritizes_unlinked_local_order_contexts() -> None:
    timestamp_ms = int(datetime.now(UTC).timestamp() * 1000)
    noise_fills = [
        OkxFillGroup(
            order_id=f"noise-linked-{index}",
            trade_ids=(f"noise-trade-{index}",),
            inst_id="AAVE-USDT-SWAP",
            symbol="AAVE/USDT",
            side="buy",
            pos_side="net",
            contracts=1.0,
            avg_price=97.0,
            fee_abs=0.0,
            fill_pnl=0.0,
            timestamp_ms=float(timestamp_ms + index),
            timestamp=datetime.fromtimestamp((timestamp_ms + index) / 1000, UTC),
            raw_count=1,
        )
        for index in range(35)
    ]
    protection_fill = OkxFillGroup(
        order_id="aave-protection-close",
        trade_ids=("close-trade",),
        inst_id="AAVE-USDT-SWAP",
        symbol="AAVE/USDT",
        side="buy",
        pos_side="net",
        contracts=6.1,
        avg_price=97.54,
        fee_abs=0.0,
        fill_pnl=-1.84,
        timestamp_ms=float(timestamp_ms - 1),
        timestamp=datetime.fromtimestamp((timestamp_ms - 1) / 1000, UTC),
        raw_count=1,
    )
    executor = _LinkedProtectionFillExecutor()

    contexts = await OkxAuthoritativeSyncService()._fetch_order_history_contexts(
        executor,
        exchange_fills=[*noise_fills, protection_fill],
        local_exchange_order_ids={
            *(fill.order_id for fill in noise_fills),
            "aave-entry-order",
            "aave-protection-close",
        },
        priority_order_ids={"aave-protection-close"},
    )

    assert "aave-protection-close" in contexts
    assert executor.ccxt.order_history_params[0]["ordId"] == "aave-protection-close"


@pytest.mark.asyncio
async def test_okx_authoritative_sync_ignores_pre_cold_start_fills(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-authoritative-watermark.db').as_posix()}",
    )
    marker_path = tmp_path / "phase3_cold_start_reset_marker.json"
    marker_path.write_text(
        json.dumps(
            {
                "mode": "paper",
                "policy_id": "PHASE3_COLD_START_RESET",
                "reset_at": (datetime.now(UTC) - timedelta(hours=1)).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    await init_db()
    try:
        report = await OkxAuthoritativeSyncService(
            mode="paper",
            lookback_hours=24,
            executor_factory=_PreResetFillExecutor,
            cold_start_marker_path=marker_path,
        ).collect()

        kinds = {issue["kind"] for issue in report["issues"]}
        assert report["cold_start_watermark_applied"] is True
        assert report["cold_start_reset_at"] is not None
        assert report["okx_fill_order_count"] == 0
        assert "okx_fill_missing_local_order" not in kinds
        assert report["status"] == "ok"
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_okx_authoritative_sync_matches_local_position_by_okx_inst_id(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-authoritative-inst-id.db').as_posix()}",
    )
    await init_db()
    now = datetime.now(UTC)
    try:
        async with get_session_ctx() as session:
            session.add(
                Position(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="SAHARA/USDT",
                    side="long",
                    quantity=12.0,
                    entry_price=0.031,
                    current_price=0.034,
                    is_open=True,
                    okx_inst_id="SPK-USDT-SWAP",
                    created_at=now - timedelta(minutes=5),
                )
            )

        report = await OkxAuthoritativeSyncService(
            mode="paper",
            lookback_hours=24,
            executor_factory=_SpkPositionExecutor,
        ).collect()

        kinds = {issue["kind"] for issue in report["issues"]}
        assert "okx_open_position_missing_locally" not in kinds
        assert "local_open_position_missing_on_okx" not in kinds
        assert report["okx_position_count"] == 1
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_okx_authoritative_sync_pulls_order_fills_by_execution_inst_id(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-authoritative-order-inst-id.db').as_posix()}",
    )
    await init_db()
    now = datetime.now(UTC)
    try:
        async with get_session_ctx() as session:
            decision = AIDecision(
                model_name="ensemble_trader",
                symbol="SAHARA/USDT",
                action="long",
                confidence=0.8,
                reasoning="legacy alias",
                is_paper=True,
                was_executed=True,
                raw_llm_response={
                    "execution_result": {
                        "raw_response": {
                            "info": {"instId": "SPK-USDT-SWAP", "ctVal": "1"},
                        }
                    }
                },
            )
            session.add(decision)
            await session.flush()
            session.add(
                Order(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="SAHARA/USDT",
                    side="buy",
                    order_type="market",
                    quantity=12.0,
                    price=0.031,
                    status="filled",
                    decision_id=decision.id,
                    exchange_order_id="spk-order-1",
                    filled_at=now - timedelta(minutes=5),
                    created_at=now - timedelta(minutes=5),
                )
            )

        report = await OkxAuthoritativeSyncService(
            mode="paper",
            lookback_hours=24,
            executor_factory=_SpkFillExecutor,
        ).collect()

        kinds = {issue["kind"] for issue in report["issues"]}
        assert "local_order_not_found_in_recent_okx_fills" not in kinds
        assert "okx_fill_missing_local_order" not in kinds
        assert report["okx_fill_order_count"] == 1
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_okx_authoritative_sync_uses_execution_contract_size_for_fill_quantity(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-authoritative-contract-size.db').as_posix()}",
    )
    await init_db()
    now = datetime.now(UTC)
    try:
        async with get_session_ctx() as session:
            decision = AIDecision(
                model_name="ensemble_trader",
                symbol="NEAR/USDT",
                action="long",
                confidence=0.8,
                reasoning="contract-size regression",
                is_paper=True,
                was_executed=True,
                raw_llm_response={
                    "execution_result": {
                        "quantity": 37.0,
                        "raw_response": {
                            "info": {"instId": "NEAR-USDT-SWAP"},
                            "contract_size": 10.0,
                            "order_contracts": 3.7,
                            "filled_contracts": 3.7,
                            "base_quantity": 37.0,
                        },
                    }
                },
            )
            session.add(decision)
            await session.flush()
            session.add(
                Order(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="NEAR/USDT",
                    side="buy",
                    order_type="market",
                    quantity=37.0,
                    price=2.31,
                    status="filled",
                    decision_id=decision.id,
                    exchange_order_id="near-order-1",
                    filled_at=now - timedelta(minutes=5),
                    created_at=now - timedelta(minutes=5),
                )
            )

        report = await OkxAuthoritativeSyncService(
            mode="paper",
            lookback_hours=24,
            executor_factory=_NearFillExecutor,
        ).collect()

        kinds = {issue["kind"] for issue in report["issues"]}
        assert report["okx_fill_order_count"] == 1
        assert "local_order_not_found_in_recent_okx_fills" not in kinds
        assert "okx_fill_missing_local_order" not in kinds
        assert "local_order_quantity_differs_from_okx_fill" not in kinds
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_okx_authoritative_sync_prefers_confirmed_order_raw_fill_quantity_over_legacy_decision_payload(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-authoritative-order-raw-cache.db').as_posix()}",
    )
    await init_db()
    now = datetime.now(UTC)
    try:
        async with get_session_ctx() as session:
            decision = AIDecision(
                model_name="ensemble_trader",
                symbol="LAB/USDT",
                action="close_long",
                confidence=0.98,
                reasoning="legacy execution payload used base quantity",
                is_paper=True,
                was_executed=True,
                raw_llm_response={
                    "execution_result": {
                        "quantity": 1.1,
                        "raw_response": {
                            "contract_size": 0.1,
                            "filled_contracts": 11.0,
                            "base_quantity": 1.1,
                        },
                    }
                },
            )
            session.add(decision)
            await session.flush()
            session.add(
                Order(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="LAB/USDT",
                    side="sell",
                    order_type="market",
                    quantity=11.0,
                    price=16.436,
                    status="filled",
                    decision_id=decision.id,
                    exchange_order_id="lab-order-1",
                    okx_inst_id="LAB-USDT-SWAP",
                    okx_fill_contracts=11.0,
                    okx_sync_status="okx_confirmed",
                    okx_raw_fills={
                        "order_id": "lab-order-1",
                        "inst_id": "LAB-USDT-SWAP",
                        "contracts": 11.0,
                        "contract_size": 1.0,
                        "base_quantity": 11.0,
                        "fills_history_confirmed": True,
                    },
                    filled_at=now - timedelta(minutes=5),
                    created_at=now - timedelta(minutes=5),
                )
            )

        report = await OkxAuthoritativeSyncService(
            mode="paper",
            lookback_hours=24,
            executor_factory=_LabFillNoInstrumentExecutor,
        ).collect()

        kinds = {issue["kind"] for issue in report["issues"]}
        assert report["okx_fill_order_count"] == 1
        assert "local_order_quantity_differs_from_okx_fill" not in kinds
        assert "local_order_not_found_in_recent_okx_fills" not in kinds
        assert "okx_fill_missing_local_order" not in kinds
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_okx_authoritative_sync_accepts_current_position_confirmed_open_entry(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-authoritative-position-confirmed.db').as_posix()}",
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
                        symbol="FLOKI/USDT",
                        side="sell",
                        order_type="market",
                        quantity=600000.0,
                        price=0.00002174,
                        status="filled",
                        exchange_order_id="3695537280216961024",
                        filled_at=now - timedelta(minutes=5),
                        created_at=now - timedelta(minutes=5),
                    ),
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
                        created_at=now - timedelta(minutes=5),
                    ),
                ]
            )

        report = await OkxAuthoritativeSyncService(
            mode="paper",
            lookback_hours=24,
            executor_factory=_FlokiPositionOnlyExecutor,
        ).collect()

        kinds = {issue["kind"] for issue in report["issues"]}
        assert "local_order_not_found_in_recent_okx_fills" not in kinds
        assert "local_open_position_missing_on_okx" not in kinds
        assert "okx_open_position_missing_locally" not in kinds
        assert report["status"] == "ok"
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_okx_authoritative_sync_accepts_persisted_position_confirmation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-authoritative-persisted-position-confirmed.db').as_posix()}",
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
                        symbol="FLOKI/USDT",
                        side="sell",
                        order_type="market",
                        quantity=600000.0,
                        price=0.00002174,
                        status="filled",
                        exchange_order_id="3695537280216961024",
                        okx_inst_id="FLOKI-USDT-SWAP",
                        okx_sync_status="okx_position_confirmed",
                        okx_raw_fills={
                            "order_id": "3695537280216961024",
                            "inst_id": "FLOKI-USDT-SWAP",
                            "pos_id": "3695537280250515456",
                            "position_snapshot_confirmed": True,
                            "fills_history_confirmed": False,
                            "rows": [],
                        },
                        filled_at=now - timedelta(minutes=5),
                        created_at=now - timedelta(minutes=5),
                    ),
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
                        created_at=now - timedelta(minutes=5),
                    ),
                ]
            )

        report = await OkxAuthoritativeSyncService(
            mode="paper",
            lookback_hours=24,
            executor_factory=_FlokiPositionOnlyExecutor,
        ).collect()

        kinds = {issue["kind"] for issue in report["issues"]}
        assert "local_order_not_found_in_recent_okx_fills" not in kinds
        assert report["status"] == "ok"
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_okx_authoritative_sync_loads_old_open_position_for_recent_entry_order(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-authoritative-old-open-position.db').as_posix()}",
    )
    await init_db()
    now = datetime.now(UTC)
    try:
        async with get_session_ctx() as session:
            old_positions = [
                Position(
                    model_name="old",
                    execution_mode="paper",
                    symbol="FLOKI/USDT",
                    side="short",
                    quantity=1.0,
                    entry_price=0.00002174,
                    current_price=0.00002156,
                    is_open=True,
                    okx_inst_id="FLOKI-USDT-SWAP",
                    created_at=now - timedelta(days=2, minutes=idx),
                )
                for idx in range(8)
            ]
            session.add_all(
                [
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="FLOKI/USDT",
                        side="sell",
                        order_type="market",
                        quantity=600000.0,
                        price=0.00002174,
                        status="filled",
                        exchange_order_id="3695537280216961024",
                        filled_at=now - timedelta(minutes=5),
                        created_at=now - timedelta(minutes=5),
                    ),
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
                        created_at=now - timedelta(days=3),
                    ),
                    *old_positions,
                ]
            )

        report = await OkxAuthoritativeSyncService(
            mode="paper",
            lookback_hours=24,
            limit=3,
            executor_factory=_FlokiPositionOnlyExecutor,
        ).collect()

        kinds = {issue["kind"] for issue in report["issues"]}
        assert "okx_fill_not_linked_to_position" not in kinds
        assert report["local_position_count"] > 3
        assert report["status"] == "ok"
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_okx_authoritative_sync_backfills_local_facts_beyond_primary_limit(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-authoritative-limit-backfill.db').as_posix()}",
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
                        symbol="ETH/USDT",
                        side="sell",
                        order_type="market",
                        quantity=1.0,
                        price=90.0,
                        status="filled",
                        exchange_order_id="newer-local-order",
                        filled_at=now - timedelta(minutes=1),
                        created_at=now - timedelta(minutes=1),
                    ),
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="BTC/USDT",
                        side="buy",
                        order_type="market",
                        quantity=0.02,
                        price=100.0,
                        status="filled",
                        exchange_order_id="older-matched-order",
                        filled_at=now - timedelta(minutes=10),
                        created_at=now - timedelta(minutes=10),
                    ),
                    Position(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="ETH/USDT",
                        side="short",
                        quantity=1.0,
                        entry_price=90.0,
                        current_price=90.0,
                        is_open=False,
                        entry_exchange_order_id="newer-local-order",
                        created_at=now - timedelta(minutes=2),
                        closed_at=now - timedelta(minutes=1),
                    ),
                    Position(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="BTC/USDT",
                        side="long",
                        quantity=0.02,
                        entry_price=100.0,
                        current_price=100.0,
                        is_open=False,
                        okx_inst_id="BTC-USDT-SWAP",
                        entry_exchange_order_id="older-matched-order",
                        created_at=now - timedelta(minutes=20),
                        closed_at=now - timedelta(minutes=5),
                    ),
                ]
            )

        report = await OkxAuthoritativeSyncService(
            mode="paper",
            lookback_hours=24,
            limit=1,
            executor_factory=_LimitBackfillFillExecutor,
        ).collect()

        kinds = {issue["kind"] for issue in report["issues"]}
        assert "okx_fill_missing_local_order" not in kinds
        assert "okx_fill_not_linked_to_position" not in kinds
        assert report["local_order_count"] == 2
        assert report["local_position_count"] == 2
        assert report["supplemental_local_order_count"] == 1
        assert report["supplemental_local_position_count"] == 1
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_okx_authoritative_sync_fill_page_cap_does_not_follow_row_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _CapturingNativeFactsClient:
        def __init__(self, _executor: Any) -> None:
            pass

        async def fetch_fill_groups(self, **kwargs: Any) -> list[Any]:
            captured.update(kwargs)
            return []

    monkeypatch.setattr(
        "services.okx_authoritative_sync.OkxNativeFactsClient",
        _CapturingNativeFactsClient,
    )
    service = OkxAuthoritativeSyncService(limit=1)

    groups = await service._fetch_fills(
        object(),
        symbols=set(),
        since=datetime.now(UTC) - timedelta(hours=24),
        target_order_ids=set(),
    )

    assert groups == []
    assert captured["limit"] == 100
    assert captured["max_pages"] == MAX_AUTHORITATIVE_FILL_PAGES == 10
    assert captured["account_wide_only"] is True
