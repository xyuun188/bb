from __future__ import annotations

from typing import Any

import pytest

from ai_brain.base_model import Action, DecisionOutput
from executor.okx_executor import OKXExecutor


class _CancelCcxt:
    def __init__(self) -> None:
        self.cancel_calls: list[dict[str, Any]] = []

    async def privatePostTradeCancelOrder(self, params: dict[str, Any]) -> dict[str, Any]:
        self.cancel_calls.append(params)
        return {
            "code": "0",
            "data": [{"ordId": params["ordId"], "sCode": "0"}],
        }


class _CloseCcxt:
    def __init__(self) -> None:
        self.close_calls: list[dict[str, Any]] = []

    async def privatePostTradeClosePosition(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        self.close_calls.append(params)
        return {"code": "0", "data": [{"sCode": "0"}]}


class _PendingSettlementCancelCcxt(_CloseCcxt):
    def __init__(self) -> None:
        super().__init__()
        self.cancel_calls: list[dict[str, Any]] = []

    async def privatePostTradeCancelOrder(self, params: dict[str, Any]) -> dict[str, Any]:
        self.cancel_calls.append(params)
        raise RuntimeError(
            "OKX API error [51410]: Cancellation failed as the order is already "
            "in canceling status or pending settlement."
        )


@pytest.mark.asyncio
async def test_partial_entry_cancels_residual_and_proves_terminal_fill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = OKXExecutor(mode="paper", load_markets_on_initialize=False)
    ccxt = _CancelCcxt()

    async def fetch_detail(_ccxt: Any, order_id: str, _symbol: str) -> dict[str, Any]:
        return {
            "id": order_id,
            "symbol": "YB/USDT:USDT",
            "side": "sell",
            "amount": 3125.0,
            "filled": 760.0,
            "average": 0.0464,
            "status": "canceled",
            "info": {
                "ordId": order_id,
                "instId": "YB-USDT-SWAP",
                "state": "canceled",
                "sz": "3125",
                "accFillSz": "760",
            },
        }

    monkeypatch.setattr(executor, "_fetch_native_order_detail", fetch_detail)
    result = await executor._finalize_partial_entry_order(
        ccxt,
        {
            "id": "3783775068963442688",
            "symbol": "YB/USDT:USDT",
            "side": "sell",
            "amount": 3125.0,
            "filled": 760.0,
            "average": 0.0464,
            "status": "partially_filled",
            "info": {
                "ordId": "3783775068963442688",
                "instId": "YB-USDT-SWAP",
                "state": "partially_filled",
            },
        },
        "YB-USDT-SWAP",
    )

    assert ccxt.cancel_calls == [
        {"instId": "YB-USDT-SWAP", "ordId": "3783775068963442688"}
    ]
    assert result["status"] == "partially_filled"
    assert result["entry_residual_terminal"] is True
    assert result["remaining_contracts"] == 0.0
    assert result["entry_partial_fill_finalization"]["terminal"] is True
    assert result["entry_partial_fill_finalization"]["cancelled_residual_contracts"] == 2365.0


@pytest.mark.asyncio
async def test_partial_entry_records_cancel_pending_settlement() -> None:
    executor = OKXExecutor(mode="paper", load_markets_on_initialize=False)
    ccxt = _PendingSettlementCancelCcxt()

    result = await executor._finalize_partial_entry_order(
        ccxt,
        {
            "id": "stale-entry",
            "symbol": "YB/USDT:USDT",
            "side": "sell",
            "amount": 3125.0,
            "filled": 760.0,
            "status": "partially_filled",
            "info": {
                "ordId": "stale-entry",
                "instId": "YB-USDT-SWAP",
                "state": "partially_filled",
            },
        },
        "YB-USDT-SWAP",
    )

    finalization = result["entry_partial_fill_finalization"]
    assert ccxt.cancel_calls == [
        {"instId": "YB-USDT-SWAP", "ordId": "stale-entry"}
    ]
    assert finalization["terminal"] is False
    assert finalization["cancel_acknowledged"] is False
    assert finalization["cancel_pending_settlement"] is True
    assert "51410" in finalization["cancel_error"]


@pytest.mark.asyncio
async def test_terminal_partial_entry_rebuilds_exact_uncovered_oco(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = OKXExecutor(mode="paper", load_markets_on_initialize=False)
    protection_reads = 0
    create_calls: list[dict[str, Any]] = []

    async def no_sleep(_seconds: float) -> None:
        return None

    async def positions(_symbol: str | None = None) -> list[dict[str, Any]]:
        return [
            {
                "symbol": "YB/USDT",
                "side": "short",
                "contracts": 760.0,
                "info": {"instId": "YB-USDT-SWAP", "posSide": "short"},
            }
        ]

    async def protection(_symbol: str | None = None) -> list[dict[str, Any]]:
        nonlocal protection_reads
        protection_reads += 1
        if create_calls:
            return [
                {
                    "symbol": "YB/USDT",
                    "position_side": "short",
                    "contracts": 760.0,
                }
            ]
        return []

    async def create_protection(**kwargs: Any) -> dict[str, Any]:
        create_calls.append(kwargs)
        return {"code": "0", "data": [{"algoId": "yb-recovery-oco", "sCode": "0"}]}

    monkeypatch.setattr("executor.okx_executor.asyncio.sleep", no_sleep)
    monkeypatch.setattr(executor, "get_positions_strict", positions)
    monkeypatch.setattr(executor, "get_position_protection_orders", protection)
    monkeypatch.setattr(executor, "create_position_protection_order", create_protection)
    decision = DecisionOutput(
        model_name="ensemble_trader",
        symbol="YB/USDT",
        action=Action.SHORT,
        confidence=0.7,
        reasoning="test",
        position_size_pct=0.1,
        stop_loss_pct=0.01,
        take_profit_pct=0.02,
    )

    report = await executor._ensure_partial_entry_protection(
        decision=decision,
        params={
            "attachAlgoOrds": [
                {
                    "slTriggerPx": "0.047",
                    "tpTriggerPx": "0.043",
                }
            ]
        },
        filled_contracts=760.0,
    )

    assert protection_reads == 4
    assert create_calls == [
        {
            "inst_id": "YB-USDT-SWAP",
            "position_side": "short",
            "okx_position_side": "short",
            "contracts": 760.0,
            "stop_loss_price": 0.047,
            "take_profit_price": 0.043,
        }
    ]
    assert report["verified"] is True
    assert report["status"] == "created_and_verified"


@pytest.mark.asyncio
async def test_stale_partial_entry_force_close_flattens_and_cancels_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = OKXExecutor(mode="paper", load_markets_on_initialize=False)
    ccxt = _CloseCcxt()
    position_reads = 0

    async def no_sleep(_seconds: float) -> None:
        return None

    async def positions(_symbol: str | None = None) -> list[dict[str, Any]]:
        nonlocal position_reads
        position_reads += 1
        if position_reads > 1:
            return []
        return [
            {
                "symbol": "YB/USDT",
                "side": "short",
                "contracts": 760.0,
                "info": {"instId": "YB-USDT-SWAP", "posSide": "net"},
            }
        ]

    async def no_open_orders(_symbol: str | None = None) -> list[dict[str, Any]]:
        return []

    monkeypatch.setattr("executor.okx_executor.asyncio.sleep", no_sleep)
    monkeypatch.setattr(executor, "_order_age_seconds", lambda _order: 120.0)
    monkeypatch.setattr(executor, "get_positions_strict", positions)
    monkeypatch.setattr(executor, "get_open_orders_strict", no_open_orders)

    report = await executor._force_close_stale_partial_entry(
        ccxt,
        {
            "id": "3783775068963442688",
            "symbol": "YB/USDT:USDT",
            "side": "sell",
            "status": "partially_filled",
            "amount": 3125.0,
            "filled": 760.0,
            "info": {
                "ordId": "3783775068963442688",
                "instId": "YB-USDT-SWAP",
                "side": "sell",
                "posSide": "net",
                "tdMode": "cross",
            },
        },
        "YB-USDT-SWAP",
    )

    assert ccxt.close_calls == [
        {
            "instId": "YB-USDT-SWAP",
            "mgnMode": "cross",
            "autoCxl": True,
        }
    ]
    assert report["attempted"] is True
    assert report["acknowledged"] is True
    assert report["position_contracts_before"] == 760.0
    assert report["position_contracts_after"] == 0.0
    assert report["residual_order_active_after"] is False
    assert report["recovered"] is True
    assert report["status"] == "flattened_and_cancelled"


@pytest.mark.asyncio
async def test_recent_partial_entry_waits_before_force_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = OKXExecutor(mode="paper", load_markets_on_initialize=False)
    ccxt = _CloseCcxt()
    monkeypatch.setattr(executor, "_order_age_seconds", lambda _order: 5.0)

    report = await executor._force_close_stale_partial_entry(
        ccxt,
        {
            "id": "recent-partial",
            "side": "buy",
            "filled": 1.0,
            "info": {"instId": "BTC-USDT-SWAP"},
        },
        "BTC-USDT-SWAP",
    )

    assert ccxt.close_calls == []
    assert report["attempted"] is False
    assert report["status"] == "waiting_for_cancel_terminalization"


@pytest.mark.asyncio
async def test_stale_partial_entry_reuses_recent_active_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = OKXExecutor(mode="paper", load_markets_on_initialize=False)
    ccxt = _CloseCcxt()

    async def positions(_symbol: str | None = None) -> list[dict[str, Any]]:
        return [
            {
                "symbol": "YB/USDT",
                "side": "short",
                "contracts": 760.0,
                "info": {"instId": "YB-USDT-SWAP", "posSide": "net"},
            }
        ]

    async def existing_exit(
        _ccxt: Any,
        _symbol: str,
        _side: str,
    ) -> dict[str, Any]:
        return {
            "id": "existing-close",
            "symbol": "YB-USDT-SWAP",
            "side": "buy",
            "status": "open",
            "amount": 760.0,
            "filled": 0.0,
            "reduceOnly": True,
        }

    ages = iter((120.0, 5.0))
    monkeypatch.setattr(executor, "_order_age_seconds", lambda _order: next(ages))
    monkeypatch.setattr(executor, "get_positions_strict", positions)
    monkeypatch.setattr(executor, "_find_active_exit_order", existing_exit)

    report = await executor._force_close_stale_partial_entry(
        ccxt,
        {
            "id": "stale-entry",
            "side": "sell",
            "filled": 760.0,
            "info": {"instId": "YB-USDT-SWAP", "tdMode": "cross"},
        },
        "YB-USDT-SWAP",
    )

    assert ccxt.close_calls == []
    assert report["attempted"] is False
    assert report["existing_exit_order_id"] == "existing-close"
    assert report["status"] == "existing_exit_order_tracking"


@pytest.mark.asyncio
async def test_stale_partial_entry_tracks_exit_pending_settlement_without_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = OKXExecutor(mode="paper", load_markets_on_initialize=False)
    ccxt = _PendingSettlementCancelCcxt()

    async def positions(_symbol: str | None = None) -> list[dict[str, Any]]:
        return [
            {
                "symbol": "YB/USDT",
                "side": "short",
                "contracts": 760.0,
                "info": {"instId": "YB-USDT-SWAP", "posSide": "net"},
            }
        ]

    async def existing_exit(
        _ccxt: Any,
        _symbol: str,
        _side: str,
    ) -> dict[str, Any]:
        return {
            "id": "existing-close",
            "symbol": "YB-USDT-SWAP",
            "side": "buy",
            "status": "open",
            "amount": 760.0,
            "filled": 0.0,
            "reduceOnly": True,
        }

    monkeypatch.setattr(executor, "_order_age_seconds", lambda _order: 120.0)
    monkeypatch.setattr(executor, "get_positions_strict", positions)
    monkeypatch.setattr(executor, "_find_active_exit_order", existing_exit)

    report = await executor._force_close_stale_partial_entry(
        ccxt,
        {
            "id": "stale-entry",
            "side": "sell",
            "filled": 760.0,
            "info": {"instId": "YB-USDT-SWAP", "tdMode": "cross"},
        },
        "YB-USDT-SWAP",
    )

    assert ccxt.cancel_calls == [
        {"instId": "YB-USDT-SWAP", "ordId": "existing-close"}
    ]
    assert ccxt.close_calls == []
    assert report["attempted"] is False
    assert report["existing_exit_cancel"]["cancel_pending_settlement"] is True
    assert report["status"] == "existing_exit_pending_settlement"
