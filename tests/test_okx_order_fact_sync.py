from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from ai_brain.base_model import Action, DecisionOutput
from config.settings import settings
from db.session import close_db, get_session_ctx, init_db
from models.account import OkxAccountBill
from models.decision import AIDecision
from models.learning import StrategyLearningEvent
from models.trade import OkxPositionHistory, Order, Position
from services.okx_native_facts import OkxNativeFillGroup
from services.okx_order_fact_sync import (
    OKX_POSITION_SYNC_SUPPRESSION_EVENT_TYPE,
    OKX_SYNC_CONFIRMED,
    OKX_SYNC_EXECUTION_RESULT_CONFIRMED,
    OKX_SYNC_NATIVE_CLOSE_BACKFILL_PENDING,
    OKX_SYNC_NO_FILL_REJECTED,
    OKX_SYNC_OKX_ONLY,
    OKX_SYNC_ORDER_ONLY,
    OKX_SYNC_POSITION_CONFIRMED,
    OKX_SYNC_UNVERIFIED,
    PHASE3_DEFAULT_ORDER_SYNC_START,
    OkxContractSizeCatalog,
    OkxOrderFactSyncService,
    _apply_paper_training_exchange_recovery_to_decision,
    _apply_position_history_payload,
    _build_contract_size_catalog,
    _current_position_entry_fee_evidence,
    _db_naive_since,
    _matching_current_position_entry_orders,
    _matching_native_full_close_pending_fill,
    _order_needs_account_contract_size_repair,
    _order_needs_okx_fact_refresh,
    _order_needs_okx_pull,
    _paper_training_decision_for_order_fact,
    _repair_stored_fill_contract_size_from_instruments,
    _stored_fill_base_quantity,
)
from services.paper_training import (
    attach_paper_training_order_identity,
    build_paper_training_contract,
)
from web_dashboard.api.trades import get_trade_detail, get_trades


def test_paper_catalog_overrides_public_contract_size_with_account_evidence() -> None:
    catalog = _build_contract_size_catalog(
        mode="paper",
        public_sizes={"ZAMA-USDT-SWAP": 10.0},
        exchange_positions=[
            {
                "instId": "ZAMA-USDT-SWAP",
                "pos": "24",
                "avgPx": "0.04122",
                "markPx": "0.04112",
                "lever": "1",
                "imr": "98.688",
                "notionalUsd": "98.688",
            }
        ],
    )

    assert isinstance(catalog, OkxContractSizeCatalog)
    assert catalog["ZAMA-USDT-SWAP"] == pytest.approx(100.0, rel=0.001)
    assert catalog.source_for("ZAMA-USDT-SWAP").startswith("okx_account_position_")


def test_live_catalog_keeps_public_contract_size_unchanged() -> None:
    catalog = _build_contract_size_catalog(
        mode="live",
        public_sizes={"ZAMA-USDT-SWAP": 10.0},
        exchange_positions=[
            {
                "instId": "ZAMA-USDT-SWAP",
                "pos": "24",
                "avgPx": "0.04122",
                "markPx": "0.04112",
                "lever": "1",
                "imr": "98.688",
                "notionalUsd": "98.688",
            }
        ],
    )

    assert catalog["ZAMA-USDT-SWAP"] == 10.0
    assert catalog.source_for("ZAMA-USDT-SWAP") == "okx_public_instruments"


def test_paper_catalog_snaps_account_rounding_noise_to_public_tick_size() -> None:
    catalog = _build_contract_size_catalog(
        mode="paper",
        public_sizes={"CFX-USDT-SWAP": 10.0},
        exchange_positions=[
            {
                "instId": "CFX-USDT-SWAP",
                "pos": "100",
                "avgPx": "0.10",
                "markPx": "0.10",
                "lever": "1",
                "imr": "99.918",
                "notionalUsd": "100",
            }
        ],
    )

    assert catalog["CFX-USDT-SWAP"] == 10.0
    assert catalog.source_for("CFX-USDT-SWAP").startswith("okx_account_position_")


def test_paper_catalog_recovers_closed_demo_multiplier_from_history_and_fills() -> None:
    opened = datetime(2026, 7, 22, 9, 48, 44, tzinfo=UTC)
    fills = [
        OkxNativeFillGroup(
            order_id="close-1",
            trade_ids=("trade-1",),
            inst_id="ZAMA-USDT-SWAP",
            symbol="ZAMA/USDT",
            side="sell",
            pos_side="net",
            contracts=38.0,
            avg_price=0.04115,
            fee_abs=0.078185,
            fill_pnl=-0.266,
            timestamp_ms=(opened + timedelta(minutes=2)).timestamp() * 1000,
            timestamp=opened + timedelta(minutes=2),
            raw_count=1,
        ),
        OkxNativeFillGroup(
            order_id="close-2",
            trade_ids=("trade-2",),
            inst_id="ZAMA-USDT-SWAP",
            symbol="ZAMA/USDT",
            side="sell",
            pos_side="net",
            contracts=24.0,
            avg_price=0.04119,
            fee_abs=0.049428,
            fill_pnl=-0.072,
            timestamp_ms=(opened + timedelta(minutes=11)).timestamp() * 1000,
            timestamp=opened + timedelta(minutes=11),
            raw_count=1,
        ),
    ]
    catalog = _build_contract_size_catalog(
        mode="paper",
        public_sizes={"ZAMA-USDT-SWAP": 10.0},
        exchange_positions=[],
        position_history_rows=[
            {
                "instId": "ZAMA-USDT-SWAP",
                "direction": "long",
                "openMaxPos": "62",
                "closeTotalPos": "62",
                "openAvgPx": "0.04122",
                "closeAvgPx": "0.0411654838709677",
                "pnl": "-0.338",
                "cTime": str(int(opened.timestamp() * 1000)),
                "uTime": str(int((opened + timedelta(minutes=11)).timestamp() * 1000)),
            }
        ],
        fills=fills,
    )

    assert catalog["ZAMA-USDT-SWAP"] == pytest.approx(100.0)
    assert catalog.source_for("ZAMA-USDT-SWAP") == (
        "okx_account_position_history_pnl_fill_crosscheck"
    )


def test_paper_catalog_quarantines_conflicting_account_evidence() -> None:
    catalog = _build_contract_size_catalog(
        mode="paper",
        public_sizes={"ZAMA-USDT-SWAP": 10.0},
        exchange_positions=[
            {
                "instId": "ZAMA-USDT-SWAP",
                "pos": "24",
                "avgPx": "0.04122",
                "markPx": "0.04112",
                "lever": "1",
                "imr": "98.688",
                "notionalUsd": "9.8688",
            }
        ],
    )

    assert "ZAMA-USDT-SWAP" not in catalog
    assert catalog.is_quarantined("ZAMA-USDT-SWAP") is True


def test_confirmed_order_is_repaired_when_account_contract_size_disagrees() -> None:
    catalog = OkxContractSizeCatalog({"ZAMA-USDT-SWAP": 10.0})
    catalog.set_verified(
        "ZAMA-USDT-SWAP",
        100.0,
        source="okx_account_position_history_pnl_fill_crosscheck",
    )
    order = Order(
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="ZAMA/USDT",
        side="buy",
        order_type="market",
        quantity=620.0,
        price=0.04122,
        status="filled",
        fee=0.127782,
        exchange_order_id="3765187269268054016",
    )
    order.okx_inst_id = "ZAMA-USDT-SWAP"
    order.okx_fill_contracts = 62.0
    order.okx_sync_status = OKX_SYNC_CONFIRMED
    order.okx_raw_fills = {
        "fills_history_confirmed": True,
        "order_id": "3765187269268054016",
        "inst_id": "ZAMA-USDT-SWAP",
        "contracts": 62.0,
        "contract_size": 10.0,
        "contract_size_verified": True,
        "contract_size_source": "okx_public_instruments",
        "base_quantity": 620.0,
    }

    assert _order_needs_account_contract_size_repair(order, catalog) is True
    assert _repair_stored_fill_contract_size_from_instruments(
        order,
        contract_sizes=catalog,
        now=datetime.now(UTC),
    ) is True
    assert order.quantity == pytest.approx(6200.0)
    assert order.okx_raw_fills["contract_size"] == pytest.approx(100.0)
    assert order.okx_raw_fills["contract_size_source"] == (
        "okx_account_position_history_pnl_fill_crosscheck"
    )
    assert _order_needs_account_contract_size_repair(order, catalog) is False


def test_stored_fill_base_quantity_prefers_okx_contract_size_over_stale_base_quantity() -> None:
    assert _stored_fill_base_quantity(
        {
            "contracts": 160.0,
            "contract_size": 0.1,
            "base_quantity": 15.265700483091791,
        }
    ) == pytest.approx(16.0)


def test_okx_fill_client_identity_restores_exact_paper_training_decision() -> None:
    decision_output = DecisionOutput(
        model_name="ensemble_trader",
        symbol="ENA/USDT",
        action=Action.SHORT,
        confidence=0.2,
        reasoning="paper training",
        raw_response={
            "paper_training": build_paper_training_contract(
                symbol="ENA/USDT",
                selected_side="short",
                signal_source="direction_competition_observation",
                expected_net_return_pct=-0.2,
                return_lcb_pct=-0.3,
                feature_opportunity_score=6.2,
                horizon_minutes=10.0,
            ),
            "paper_training_mode": "bootstrap",
        },
    )
    attach_paper_training_order_identity(decision_output, 104208, "paper")
    decision = AIDecision(
        id=104208,
        model_name="ensemble_trader",
        symbol="ENA/USDT",
        action="short",
        confidence=0.2,
        is_paper=True,
        was_executed=False,
        raw_llm_response=decision_output.raw_response,
    )
    filled_at = datetime(2026, 7, 22, 8, 0, 57, tzinfo=UTC)
    fill = OkxNativeFillGroup(
        order_id="okx-entry-1",
        trade_ids=("trade-1",),
        inst_id="ENA-USDT-SWAP",
        symbol="ENA/USDT",
        side="sell",
        pos_side="net",
        contracts=1858.0,
        avg_price=0.0865,
        fee_abs=0.8,
        fill_pnl=0.0,
        timestamp_ms=filled_at.timestamp() * 1000,
        timestamp=filled_at,
        raw_count=1,
        rows=(
            {
                "ordId": "okx-entry-1",
                "clOrdId": "BBPT104208",
                "instId": "ENA-USDT-SWAP",
                "side": "sell",
            },
        ),
    )

    recovered = _paper_training_decision_for_order_fact(
        fill=fill,
        order_row=None,
        decisions_by_id={104208: decision},
    )
    assert recovered is decision

    _apply_paper_training_exchange_recovery_to_decision(
        decision,
        fill=fill,
        client_order_id="BBPT104208",
        now=filled_at + timedelta(seconds=1),
    )

    assert decision.was_executed is True
    assert decision.execution_price == pytest.approx(0.0865)
    recovery = decision.raw_llm_response["paper_training_exchange_recovery"]
    assert recovery["source_authority"] == "okx_native_fills_and_client_order_identity"
    assert recovery["exchange_order_id"] == "okx-entry-1"


def test_current_position_fee_accepts_verified_okx_execution_result() -> None:
    order = Order(
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="FIL/USDT",
        side="buy",
        order_type="market",
        quantity=537.5,
        price=0.73219888,
        status="filled",
        fee=0.19677845,
        exchange_order_id="entry-verified",
    )
    order.okx_inst_id = "FIL-USDT-SWAP"
    order.okx_fill_contracts = 5375.0
    order.okx_raw_fills = {
        "source": "okx_execution_result",
        "fills_history_confirmed": False,
        "execution_result_confirmed": True,
        "order_id": "entry-verified",
        "inst_id": "FIL-USDT-SWAP",
        "contracts": 5375.0,
        "contract_size": 0.1,
        "contract_size_verified": True,
        "contract_size_source": "okx_public_instruments",
        "base_quantity": 537.5,
        "avg_price": 0.73219888,
        "fee_abs": 0.19677845,
    }

    evidence = _current_position_entry_fee_evidence(
        ["entry-verified"],
        [order],
        current_quantity=537.5,
    )

    assert evidence["complete"] is True
    assert evidence["source"] == "okx_execution_result"
    assert evidence["allocated_entry_fee_usdt"] == pytest.approx(0.19677845)


def test_current_position_fee_rejects_unverified_execution_result_contract_size() -> None:
    order = Order(
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="FIL/USDT",
        side="buy",
        order_type="market",
        quantity=537.5,
        price=0.73219888,
        status="filled",
        fee=0.19677845,
        exchange_order_id="entry-unverified",
    )
    order.okx_inst_id = "FIL-USDT-SWAP"
    order.okx_fill_contracts = 5375.0
    order.okx_raw_fills = {
        "source": "okx_execution_result",
        "fills_history_confirmed": False,
        "execution_result_confirmed": True,
        "order_id": "entry-unverified",
        "inst_id": "FIL-USDT-SWAP",
        "contracts": 5375.0,
        "contract_size": 0.1,
        "contract_size_verified": False,
        "base_quantity": 537.5,
        "avg_price": 0.73219888,
        "fee_abs": 0.19677845,
    }

    evidence = _current_position_entry_fee_evidence(
        ["entry-unverified"],
        [order],
        current_quantity=537.5,
    )

    assert evidence["complete"] is False
    assert evidence["fee_fact_missing_order_ids"] == ["entry-unverified"]


def test_current_position_entry_links_reconstruct_multi_entry_net_lifecycle() -> None:
    opened_at = datetime.now(UTC) - timedelta(minutes=5)

    def order(
        exchange_order_id: str,
        side: str,
        quantity: float,
        price: float,
        offset_seconds: int,
    ) -> Order:
        row = Order(
            model_name="okx_authoritative_sync",
            execution_mode="paper",
            symbol="CELO/USDT",
            side=side,
            order_type="market",
            quantity=quantity,
            price=price,
            status="filled",
            fee=0.0,
            exchange_order_id=exchange_order_id,
            filled_at=opened_at + timedelta(seconds=offset_seconds),
            created_at=opened_at + timedelta(seconds=offset_seconds),
        )
        row.okx_inst_id = "CELO-USDT-SWAP"
        row.okx_sync_status = OKX_SYNC_OKX_ONLY
        return row

    orders = [
        order("entry-1", "buy", 5.0, 10.0, 1),
        order("entry-2", "buy", 5.0, 12.0, 2),
        order("partial-close", "sell", 2.0, 13.0, 3),
    ]

    matched = _matching_current_position_entry_orders(
        {
            "side": "long",
            "symbol": "CELO/USDT",
            "okx_inst_id": "CELO-USDT-SWAP",
            "created_at": opened_at,
            "quantity": 8.0,
            "entry_price": 11.0,
        },
        orders,
    )

    assert [row.exchange_order_id for row in matched] == ["entry-1", "entry-2"]


def test_position_entry_order_link_column_supports_multi_order_lifecycle() -> None:
    assert Position.__table__.c.entry_exchange_order_id.type.length == 500


def test_native_full_close_pending_order_targets_and_matches_real_fill() -> None:
    filled_at = datetime(2026, 6, 25, 20, 54, tzinfo=UTC)
    order = Order(
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="AI16Z/USDT",
        side="sell",
        order_type="market",
        quantity=366.0,
        price=0.0513,
        status="partial",
        fee=0.0,
        decision_id=132611,
        exchange_order_id=None,
        filled_at=filled_at,
    )
    order.okx_inst_id = "AI16Z-USDT-SWAP"
    order.okx_sync_status = OKX_SYNC_NATIVE_CLOSE_BACKFILL_PENDING
    order.okx_raw_fills = {
        "source": OKX_SYNC_NATIVE_CLOSE_BACKFILL_PENDING,
        "requires_okx_fill_backfill": True,
        "inst_id": "AI16Z-USDT-SWAP",
        "contracts": 36.6,
        "contract_size": 10.0,
        "base_quantity": 366.0,
        "timestamp": filled_at.isoformat(),
    }
    fill = OkxNativeFillGroup(
        order_id="real-okx-close",
        trade_ids=("trade-1",),
        inst_id="AI16Z-USDT-SWAP",
        symbol="AI16Z/USDT",
        side="sell",
        pos_side="long",
        contracts=36.6,
        avg_price=0.0513,
        fee_abs=0.03,
        fill_pnl=1.23,
        timestamp_ms=filled_at.timestamp() * 1000,
        timestamp=filled_at + timedelta(seconds=8),
        raw_count=1,
        rows=(),
    )

    assert _order_needs_okx_pull(order) is True
    assert _order_needs_okx_fact_refresh(order) is True
    assert (
        _matching_native_full_close_pending_fill(
            order,
            fills=[fill],
            contract_sizes={"AI16Z-USDT-SWAP": 10.0},
        )
        is fill
    )


def test_fill_refresh_preserves_submission_and_persists_protection_execution() -> None:
    filled_at = datetime(2026, 7, 15, 1, 5, tzinfo=UTC)
    order = Order(
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="CHZ/USDT",
        side="buy",
        order_type="market",
        quantity=388.0,
        price=0.01659,
        status="filled",
        fee=0.0,
        exchange_order_id="close-stop-1",
        filled_at=filled_at,
    )
    order.okx_raw_fills = {
        "protection_submission": {
            "state": "confirmed",
            "algo_ids": ["algo-stop-1"],
        }
    }
    fill = OkxNativeFillGroup(
        order_id="close-stop-1",
        trade_ids=("trade-stop-1",),
        inst_id="CHZ-USDT-SWAP",
        symbol="CHZ/USDT",
        side="buy",
        pos_side="net",
        contracts=388.0,
        avg_price=0.01659,
        fee_abs=0.01,
        fill_pnl=-1.0,
        timestamp_ms=filled_at.timestamp() * 1000,
        timestamp=filled_at,
        raw_count=1,
        rows=(),
    )
    lifecycle = {
        "lifecycle_complete": True,
        "actual_side": "sl",
        "algo_id": "algo-stop-1",
    }

    OkxOrderFactSyncService._apply_fill_to_order(
        order,
        fill,
        now=filled_at,
        sync_status=OKX_SYNC_CONFIRMED,
        contract_size=1.0,
        contract_size_source="okx_public_instruments",
        protection_execution=lifecycle,
    )

    assert order.okx_raw_fills["protection_submission"]["state"] == "confirmed"
    assert order.okx_raw_fills["protection_execution"] == lifecycle
    assert order.okx_raw_fills["fills_history_confirmed"] is True


def test_confirmed_algo_fill_refreshes_only_until_protection_execution_is_complete() -> None:
    order = Order(
        model_name="okx_authoritative_sync",
        execution_mode="paper",
        symbol="CHZ/USDT",
        side="buy",
        order_type="market",
        quantity=3880.0,
        price=0.01659,
        status="filled",
        fee=0.01,
        exchange_order_id="close-stop-1",
    )
    order.okx_sync_status = OKX_SYNC_CONFIRMED
    order.okx_inst_id = "CHZ-USDT-SWAP"
    order.okx_fill_contracts = 388.0
    order.okx_raw_fills = {
        "fills_history_confirmed": True,
        "order_id": "close-stop-1",
        "inst_id": "CHZ-USDT-SWAP",
        "contract_size_verified": True,
        "contract_size_source": "okx_public_instruments",
        "contract_size": 10.0,
        "contracts": 388.0,
        "base_quantity": 3880.0,
        "avg_price": 0.01659,
        "rows": [{"clOrdId": "Oclose-stop-1", "ordId": "close-stop-1"}],
    }

    assert _order_needs_okx_pull(order) is True
    assert _order_needs_okx_fact_refresh(order) is True

    order.okx_raw_fills["protection_execution"] = {
        "lifecycle_complete": True,
        "source_authority": "okx_algo_history_plus_fills_history",
    }

    assert _order_needs_okx_pull(order) is False
    assert _order_needs_okx_fact_refresh(order) is False


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
        self.account_bill_rows: list[dict[str, str]] = []
        self.fill_history_params: list[dict[str, Any]] = []
        self.order_history_params: list[dict[str, Any]] = []

    async def privateGetTradeFillsHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        self.fill_history_params.append(dict(params))
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

    async def privateGetAccountBills(self, params: dict[str, Any]) -> dict[str, Any]:
        since = int(params.get("begin") or 0)
        return {
            "data": [
                row
                for row in self.account_bill_rows
                if int(row.get("ts") or 0) >= since
            ]
        }

    async def privateGetAccountBillsArchive(self, params: dict[str, Any]) -> dict[str, Any]:
        return await self.privateGetAccountBills(params)

    async def publicGetPublicInstruments(self, params: dict[str, Any]) -> dict[str, Any]:
        assert params["instType"] == "SWAP"
        return {
            "data": [
                {"instId": "SPK-USDT-SWAP", "ctVal": "1"},
                {"instId": "HOME-USDT-SWAP", "ctVal": "1"},
                {"instId": "BTC-USDT-SWAP", "ctVal": "0.01"},
                {"instId": "XPL-USDT-SWAP", "ctVal": "10"},
            ]
        }


class _PositionHistoryBusyCcxt(_FillCcxt):
    async def privateGetAccountPositionsHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        raise TimeoutError("positions-history busy")


class _OrderHistoryBusyCcxt(_FillCcxt):
    async def privateGetTradeOrdersHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        raise TimeoutError("orders-history busy")


class _AccountWideFillBusyCcxt(_FillCcxt):
    async def privateGetTradeFillsHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        if params.get("ordId"):
            return await super().privateGetTradeFillsHistory(params)
        self.fill_history_params.append(dict(params))
        raise TimeoutError("account-wide fills-history busy")


class _FundingBillCcxt(_FillCcxt):
    def __init__(self) -> None:
        super().__init__()
        funding_ms = int((PHASE3_DEFAULT_ORDER_SYNC_START + timedelta(hours=8)).timestamp() * 1000)
        self.account_bill_rows = [
            {
                "billId": "funding-bill-1",
                "instId": "SPK-USDT-SWAP",
                "posSide": "short",
                "ccy": "USDT",
                "type": "8",
                "subType": "173",
                "balChg": "-0.17",
                "pnl": "-0.17",
                "fee": "0",
                "ts": str(funding_ms),
            }
        ]


class _SlowFundingBillCcxt(_FillCcxt):
    async def privateGetAccountBills(self, params: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(0.8)
        return await super().privateGetAccountBills(params)


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


class _StaleVerifiedContractSizeCcxt(_FillCcxt):
    def __init__(self) -> None:
        super().__init__()
        start_ms = int((PHASE3_DEFAULT_ORDER_SYNC_START + timedelta(minutes=11)).timestamp() * 1000)
        self.rows.append(
            {
                "ordId": "xpl-stale-ctval-order",
                "tradeId": "xpl-stale-trade",
                "instId": "XPL-USDT-SWAP",
                "side": "buy",
                "fillSz": "178",
                "fillPx": "0.01685",
                "fee": "-0.0149965",
                "fillPnl": "0",
                "ts": str(start_ms),
            }
        )


class _StaleVerifiedContractSizeExecutor(_Executor):
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.ccxt = _StaleVerifiedContractSizeCcxt()

    async def _get_ccxt(self) -> _StaleVerifiedContractSizeCcxt:
        return self.ccxt


class _FundingBillExecutor:
    ccxt_instances: list[_FundingBillCcxt] = []

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.ccxt = _FundingBillCcxt()
        self.__class__.ccxt_instances.append(self.ccxt)

    async def initialize(self) -> None:
        return None

    async def _get_ccxt(self) -> _FundingBillCcxt:
        return self.ccxt

    async def _with_retry(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        return await fn(*args, **kwargs)

    async def shutdown(self) -> None:
        return None


class _SlowFundingBillExecutor:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.ccxt = _SlowFundingBillCcxt()

    async def initialize(self) -> None:
        return None

    async def _get_ccxt(self) -> _SlowFundingBillCcxt:
        return self.ccxt

    async def _with_retry(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        return await fn(*args, **kwargs)

    async def shutdown(self) -> None:
        return None


class _PositionHistoryBusyExecutor:
    ccxt_instances: list[_PositionHistoryBusyCcxt] = []

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.ccxt = _PositionHistoryBusyCcxt()
        self.__class__.ccxt_instances.append(self.ccxt)

    async def initialize(self) -> None:
        return None

    async def _get_ccxt(self) -> _PositionHistoryBusyCcxt:
        return self.ccxt

    async def _with_retry(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        return await fn(*args, **kwargs)

    async def shutdown(self) -> None:
        return None


class _OrderHistoryBusyExecutor:
    ccxt_instances: list[_OrderHistoryBusyCcxt] = []

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.ccxt = _OrderHistoryBusyCcxt()
        self.__class__.ccxt_instances.append(self.ccxt)

    async def initialize(self) -> None:
        return None

    async def _get_ccxt(self) -> _OrderHistoryBusyCcxt:
        return self.ccxt

    async def _with_retry(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        return await fn(*args, **kwargs)

    async def shutdown(self) -> None:
        return None


class _AccountWideFillBusyExecutor:
    ccxt_instances: list[_AccountWideFillBusyCcxt] = []

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.ccxt = _AccountWideFillBusyCcxt()
        self.__class__.ccxt_instances.append(self.ccxt)

    async def initialize(self) -> None:
        return None

    async def _get_ccxt(self) -> _AccountWideFillBusyCcxt:
        return self.ccxt

    async def _with_retry(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        return await fn(*args, **kwargs)

    async def shutdown(self) -> None:
        return None


class _NoHistoryExecutor:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.ccxt = self

    async def initialize(self) -> None:
        return None

    async def _get_ccxt(self) -> _NoHistoryExecutor:
        return self

    async def _with_retry(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        return await fn(*args, **kwargs)

    async def shutdown(self) -> None:
        return None

    async def privateGetTradeFillsHistory(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {"data": []}

    async def privateGetTradeOrdersHistory(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {"data": []}

    async def privateGetTradeOrdersHistoryArchive(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {"data": []}

    async def privateGetAccountPositionsHistory(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {"data": []}

    async def privateGetAccountPositions(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {"data": []}

    async def publicGetPublicInstruments(self, _params: dict[str, Any]) -> dict[str, Any]:
        return {"data": [{"instId": "ACT-USDT-SWAP", "ctVal": "10"}]}


class _UnavailableExecutor:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def initialize(self) -> None:
        raise TimeoutError("OKX pull timeout")

    async def shutdown(self) -> None:
        return None


class _PullTimeoutExecutor(_NoHistoryExecutor):
    async def privateGetAccountPositions(self, _params: dict[str, Any]) -> dict[str, Any]:
        raise TimeoutError("OKX pull timeout")


class _FundingBillThenPullTimeoutCcxt(_FundingBillCcxt):
    async def privateGetAccountPositions(self, _params: dict[str, Any]) -> dict[str, Any]:
        raise TimeoutError("OKX pull timeout")


class _FundingBillThenPullTimeoutExecutor:
    ccxt_instances: list[_FundingBillThenPullTimeoutCcxt] = []

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.ccxt = _FundingBillThenPullTimeoutCcxt()
        self.__class__.ccxt_instances.append(self.ccxt)

    async def initialize(self) -> None:
        return None

    async def _get_ccxt(self) -> _FundingBillThenPullTimeoutCcxt:
        return self.ccxt

    async def _with_retry(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        return await fn(*args, **kwargs)

    async def shutdown(self) -> None:
        return None


def test_position_history_payload_preserves_existing_order_links() -> None:
    position = Position(
        model_name="manual_repair",
        execution_mode="paper",
        symbol="LAB/USDT",
        side="long",
        quantity=100.0,
        entry_price=0.1,
        current_price=0.11,
        leverage=1.0,
        unrealized_pnl=0.0,
        realized_pnl=-0.12,
        is_open=False,
        okx_inst_id="LAB-USDT-SWAP-OFF20260630-5",
        okx_pos_id="lab-pos",
        entry_exchange_order_id="entry-repaired",
        close_exchange_order_id="close-repaired",
        closed_at=datetime(2026, 6, 28, 22, 17, tzinfo=UTC),
        created_at=datetime(2026, 6, 28, 21, 45, tzinfo=UTC),
    )
    now = datetime(2026, 6, 30, 18, 55, tzinfo=UTC)

    _apply_position_history_payload(
        position,
        {
            "model_name": "okx_authoritative_sync",
            "execution_mode": "paper",
            "symbol": "LAB/USDT",
            "side": "long",
            "quantity": 100.0,
            "entry_price": 0.1,
            "current_price": 0.11,
            "leverage": 1.0,
            "unrealized_pnl": 0.0,
            "realized_pnl": -0.12,
            "closed_at": datetime(2026, 6, 28, 22, 17, tzinfo=UTC),
            "created_at": datetime(2026, 6, 28, 21, 45, tzinfo=UTC),
            "okx_inst_id": "LAB-USDT-SWAP-OFF20260630-5",
            "okx_pos_id": "lab-pos",
            "entry_exchange_order_id": None,
            "close_exchange_order_id": None,
        },
        now=now,
    )

    assert position.entry_exchange_order_id == "entry-repaired"
    assert position.close_exchange_order_id == "close-repaired"
    assert position.updated_at == now


def test_position_history_payload_merges_existing_and_authoritative_order_links() -> None:
    position = Position(
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="PEPE/USDT",
        side="long",
        quantity=1000000.0,
        entry_price=0.0000026,
        current_price=0.0000027,
        leverage=1.0,
        unrealized_pnl=0.0,
        realized_pnl=0.12,
        is_open=False,
        okx_inst_id="PEPE-USDT-SWAP",
        okx_pos_id="pepe-pos",
        entry_exchange_order_id="entry-a",
        close_exchange_order_id="partial-close-a,partial-close-b",
        closed_at=datetime(2026, 7, 4, 15, 13, tzinfo=UTC),
        created_at=datetime(2026, 7, 3, 19, 19, tzinfo=UTC),
    )
    now = datetime(2026, 7, 8, 20, 22, tzinfo=UTC)

    _apply_position_history_payload(
        position,
        {
            "model_name": "okx_authoritative_sync",
            "execution_mode": "paper",
            "symbol": "PEPE/USDT",
            "side": "long",
            "quantity": 999999.9,
            "entry_price": 0.000002599,
            "current_price": 0.000002735,
            "leverage": 1.0,
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.1333329865,
            "closed_at": datetime(2026, 7, 4, 15, 13, tzinfo=UTC),
            "created_at": datetime(2026, 7, 3, 19, 19, tzinfo=UTC),
            "okx_inst_id": "PEPE-USDT-SWAP",
            "okx_pos_id": "pepe-pos",
            "entry_exchange_order_id": "entry-a",
            "close_exchange_order_id": "final-close",
        },
        now=now,
    )

    assert position.entry_exchange_order_id == "entry-a"
    assert position.close_exchange_order_id == "partial-close-a,partial-close-b,final-close"
    assert position.updated_at == now


def test_position_history_payload_keeps_long_close_order_link_chains() -> None:
    position = Position(
        model_name="ensemble_trader",
        execution_mode="paper",
        symbol="PEPE/USDT",
        side="long",
        quantity=1000000.0,
        entry_price=0.0000026,
        current_price=0.0000027,
        leverage=1.0,
        unrealized_pnl=0.0,
        realized_pnl=0.12,
        is_open=False,
        okx_inst_id="PEPE-USDT-SWAP",
        okx_pos_id="pepe-pos",
        entry_exchange_order_id="entry-a",
        close_exchange_order_id=(
            "3712904697435881472,3712930282153410560,"
            "3713658183647723520,3713658183647723521"
        ),
        closed_at=datetime(2026, 7, 4, 15, 13, tzinfo=UTC),
        created_at=datetime(2026, 7, 3, 19, 19, tzinfo=UTC),
    )
    now = datetime(2026, 7, 8, 20, 22, tzinfo=UTC)

    _apply_position_history_payload(
        position,
        {
            "model_name": "okx_authoritative_sync",
            "execution_mode": "paper",
            "symbol": "PEPE/USDT",
            "side": "long",
            "quantity": 999999.9,
            "entry_price": 0.000002599,
            "current_price": 0.000002735,
            "leverage": 1.0,
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.1333329865,
            "closed_at": datetime(2026, 7, 4, 15, 13, tzinfo=UTC),
            "created_at": datetime(2026, 7, 3, 19, 19, tzinfo=UTC),
            "okx_inst_id": "PEPE-USDT-SWAP",
            "okx_pos_id": "pepe-pos",
            "entry_exchange_order_id": "entry-a",
            "close_exchange_order_id": "3713658183647723522,3713658183647723523",
        },
        now=now,
    )

    close_ids = set(position.close_exchange_order_id.split(","))
    assert close_ids == {
        "3712904697435881472",
        "3712930282153410560",
        "3713658183647723520",
        "3713658183647723521",
        "3713658183647723522",
        "3713658183647723523",
    }


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

    async def privateGetTradeOrdersAlgoPending(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        if str(params.get("ordType") or "") != "oco":
            return {"data": []}
        return {
            "data": [
                {
                    "instId": "FLOKI-USDT-SWAP",
                    "algoId": "floki-oco-1",
                    "ordType": "oco",
                    "state": "live",
                    "side": "buy",
                    "posSide": "short",
                    "sz": "6",
                    "reduceOnly": "true",
                    "slTriggerPx": "0.000023",
                    "tpTriggerPx": "0.000019",
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

    async def get_balance_snapshot(self) -> dict[str, Any]:
        return {"equity": 1_000.0, "free": 900.0, "total": 1_000.0}

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


class _CurrentPositionProtectionTimeoutCcxt(_CurrentPositionOnlyCcxt):
    async def privateGetTradeOrdersAlgoPending(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        raise TimeoutError("protection snapshot unavailable")


class _CurrentPositionProtectionTimeoutExecutor(_CurrentPositionOnlyExecutor):
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.ccxt = _CurrentPositionProtectionTimeoutCcxt()


class _PartialCloseCurrentPositionCcxt:
    def __init__(self) -> None:
        self.position_ts = int(
            (PHASE3_DEFAULT_ORDER_SYNC_START + timedelta(hours=3)).timestamp() * 1000
        )
        self.close_ts = self.position_ts + 60_000
        self.entry_order_id = "3711253572185980928"
        self.close_order_id = "3712882834810830848"

    async def privateGetTradeFillsHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        order_id = str(params.get("ordId") or "")
        rows = [
            {
                "ordId": self.close_order_id,
                "tradeId": "pepe-close-trade",
                "instId": "PEPE-USDT-SWAP",
                "side": "sell",
                "posSide": "net",
                "fillSz": "0.3",
                "fillPx": "0.000002728",
                "fee": "-0.004092",
                "fillPnl": "0.387",
                "ts": str(self.close_ts),
            }
        ]
        if order_id:
            rows = [row for row in rows if row["ordId"] == order_id]
        return {"data": rows}

    async def privateGetTradeOrdersHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"data": []}

    async def privateGetAccountPositionsHistory(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"data": []}

    async def privateGetAccountPositions(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "data": [
                {
                    "instId": "PEPE-USDT-SWAP",
                    "posId": "pepe-position",
                    "tradeId": "pepe-entry-trade",
                    "posSide": "net",
                    "pos": "0.1",
                    "ctVal": "10000000",
                    "avgPx": "0.000002599",
                    "markPx": "0.00000272",
                    "upl": "0.12",
                    "cTime": str(self.position_ts),
                    "uTime": str(self.close_ts),
                }
            ]
        }

    async def privateGetAccountBills(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"data": []}

    async def privateGetAccountBillsArchive(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"data": []}

    async def publicGetPublicInstruments(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"data": [{"instId": "PEPE-USDT-SWAP", "ctVal": "10000000"}]}


class _PartialCloseCurrentPositionExecutor:
    ccxt_instances: list[_PartialCloseCurrentPositionCcxt] = []

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.ccxt = _PartialCloseCurrentPositionCcxt()
        self.__class__.ccxt_instances.append(self.ccxt)

    async def initialize(self) -> None:
        return None

    async def _get_ccxt(self) -> _PartialCloseCurrentPositionCcxt:
        return self.ccxt

    async def _with_retry(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        return await fn(*args, **kwargs)

    async def get_balance_snapshot(self) -> dict[str, Any]:
        return {"equity": 1_000.0, "free": 900.0, "total": 1_000.0}

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
async def test_order_fact_sync_repairs_confirmed_order_contract_size_from_instruments(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-contract-size-repair.db').as_posix()}",
    )
    await init_db()
    phase3_time = (
        PHASE3_DEFAULT_ORDER_SYNC_START + timedelta(minutes=10)
    ).astimezone(UTC).replace(tzinfo=None)
    try:
        async with get_session_ctx() as session:
            session.add(
                Order(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="BTC/USDT",
                    side="buy",
                    order_type="market",
                    quantity=3.0,
                    price=60000.0,
                    status="filled",
                    fee=0.12,
                    exchange_order_id="stale-btc-order",
                    filled_at=phase3_time,
                    created_at=phase3_time,
                    okx_inst_id="BTC-USDT-SWAP",
                    okx_trade_ids="btc-trade",
                    okx_fill_contracts=3.0,
                    okx_fill_pnl=0.0,
                    okx_sync_status=OKX_SYNC_CONFIRMED,
                    okx_raw_fills={
                        "fills_history_confirmed": True,
                        "order_id": "stale-btc-order",
                        "trade_ids": ["btc-trade"],
                        "inst_id": "BTC-USDT-SWAP",
                        "contracts": 3.0,
                        "contract_size": 1.0,
                        "base_quantity": 3.0,
                        "avg_price": 60000.0,
                        "fee_abs": 0.12,
                        "fill_pnl": 0.0,
                        "timestamp": phase3_time.isoformat(),
                    },
                )
            )

        report = await OkxOrderFactSyncService(
            mode="paper",
            lookback_hours=72,
            executor_factory=_Executor,
            cold_start_marker_path=None,
        ).sync()

        async with get_session_ctx() as session:
            row = (
                await session.execute(
                    Order.__table__.select().where(
                        Order.__table__.c.exchange_order_id == "stale-btc-order"
                    )
                )
            ).one()._mapping

        raw = row["okx_raw_fills"]
        assert report["okx_pull_available"] is True
        assert row["quantity"] == pytest.approx(0.03)
        assert row["okx_fill_contracts"] == pytest.approx(3.0)
        assert raw["contract_size"] == pytest.approx(0.01)
        assert raw["base_quantity"] == pytest.approx(0.03)
        assert raw["contract_size_verified"] is True
        assert raw["contract_size_source"] == "okx_public_instruments"
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_order_fact_sync_rechecks_ambiguous_verified_contract_size_from_instruments(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-ambiguous-verified-contract-size.db').as_posix()}",
    )
    await init_db()
    phase3_time = (
        PHASE3_DEFAULT_ORDER_SYNC_START + timedelta(minutes=11)
    ).astimezone(UTC).replace(tzinfo=None)
    try:
        async with get_session_ctx() as session:
            session.add(
                Order(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="XPL/USDT",
                    side="buy",
                    order_type="market",
                    quantity=178.0,
                    price=0.01685,
                    status="filled",
                    fee=0.0149965,
                    exchange_order_id="xpl-stale-ctval-order",
                    filled_at=phase3_time,
                    created_at=phase3_time,
                    okx_inst_id="XPL-USDT-SWAP",
                    okx_trade_ids="xpl-stale-trade",
                    okx_fill_contracts=178.0,
                    okx_fill_pnl=0.0,
                    okx_sync_status=OKX_SYNC_CONFIRMED,
                    okx_raw_fills={
                        "fills_history_confirmed": True,
                        "order_id": "xpl-stale-ctval-order",
                        "trade_ids": ["xpl-stale-trade"],
                        "inst_id": "XPL-USDT-SWAP",
                        "contracts": 178.0,
                        "contract_size": 1.0,
                        "contract_size_verified": True,
                        "contract_size_source": "okx_public_instruments_or_fill_row",
                        "base_quantity": 178.0,
                        "avg_price": 0.01685,
                        "fee_abs": 0.0149965,
                        "fill_pnl": 0.0,
                        "timestamp": phase3_time.isoformat(),
                    },
                )
            )

        report = await OkxOrderFactSyncService(
            mode="paper",
            lookback_hours=72,
            executor_factory=_StaleVerifiedContractSizeExecutor,
            cold_start_marker_path=None,
        ).sync()

        async with get_session_ctx() as session:
            row = (
                await session.execute(
                    Order.__table__.select().where(
                        Order.__table__.c.exchange_order_id == "xpl-stale-ctval-order"
                    )
                )
            ).one()._mapping

        raw = row["okx_raw_fills"]
        assert report["okx_pull_available"] is True
        assert row["quantity"] == pytest.approx(1780.0)
        assert row["okx_fill_contracts"] == pytest.approx(178.0)
        assert raw["contract_size"] == pytest.approx(10.0)
        assert raw["base_quantity"] == pytest.approx(1780.0)
        assert raw["contract_size_verified"] is True
        assert raw["contract_size_source"] == "okx_public_instruments"
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_order_fact_sync_repairs_execution_result_contract_size_from_instruments(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-execution-result-contract-size.db').as_posix()}",
    )
    await init_db()
    phase3_time = (
        PHASE3_DEFAULT_ORDER_SYNC_START + timedelta(minutes=12)
    ).astimezone(UTC).replace(tzinfo=None)
    try:
        async with get_session_ctx() as session:
            session.add(
                Order(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="XPL/USDT",
                    side="buy",
                    order_type="market",
                    quantity=146.0,
                    price=0.0921,
                    status="filled",
                    fee=0.067233,
                    exchange_order_id="stale-xpl-execution-result",
                    filled_at=phase3_time,
                    created_at=phase3_time,
                    okx_inst_id="XPL-USDT-SWAP",
                    okx_fill_contracts=146.0,
                    okx_fill_pnl=0.0,
                    okx_sync_status=OKX_SYNC_EXECUTION_RESULT_CONFIRMED,
                    okx_raw_fills={
                        "source": "okx_execution_result",
                        "fills_history_confirmed": False,
                        "execution_result_confirmed": True,
                        "order_id": "stale-xpl-execution-result",
                        "trade_ids": [],
                        "inst_id": "XPL-USDT-SWAP",
                        "contracts": 146.0,
                        "base_quantity": 146.0,
                        "avg_price": 0.0921,
                        "fee_abs": 0.067233,
                        "fill_pnl": 0.0,
                        "timestamp": phase3_time.isoformat(),
                        "rows": [],
                    },
                )
            )

        report = await OkxOrderFactSyncService(
            mode="paper",
            lookback_hours=72,
            executor_factory=_Executor,
            cold_start_marker_path=None,
        ).sync()

        async with get_session_ctx() as session:
            row = (
                await session.execute(
                    Order.__table__.select().where(
                        Order.__table__.c.exchange_order_id == "stale-xpl-execution-result"
                    )
                )
            ).one()._mapping

        raw = row["okx_raw_fills"]
        assert report["okx_pull_available"] is True
        assert row["quantity"] == pytest.approx(1460.0)
        assert row["okx_fill_contracts"] == pytest.approx(146.0)
        assert raw["contract_size"] == pytest.approx(10.0)
        assert raw["base_quantity"] == pytest.approx(1460.0)
        assert raw["contract_size_verified"] is True
        assert raw["contract_size_source"] == "okx_public_instruments"
        assert raw["execution_result_confirmed"] is True
        assert raw["fills_history_confirmed"] is False
    finally:
        await close_db()


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
            position_rows = (
                await session.execute(
                    Position.__table__.select().where(
                        Position.__table__.c.okx_pos_id == "spk-phase3-pos"
                    )
                )
            ).all()

        assert report["okx_pull_available"] is True
        assert report["phase3_order_sync_start"] == "2026-06-27T16:00:00+00:00"
        assert report["phase3_order_sync_start_local"] == "2026-06-28T00:00:00+08:00"
        assert report["confirmed_count"] == 1
        assert report["unverified_count"] == 1
        assert report["backfilled_count"] == 1
        assert report["position_history_checked_count"] == 1
        assert report["position_history_backfilled_count"] == 0
        assert report["position_history_skipped_count"] == 1
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
        assert position_rows == []
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_order_fact_sync_closes_matching_open_position_from_okx_position_history(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-history-closes-open-position.db').as_posix()}",
    )
    await init_db()
    phase3_time = (
        PHASE3_DEFAULT_ORDER_SYNC_START + timedelta(minutes=9)
    ).astimezone(UTC).replace(tzinfo=None)
    try:
        async with get_session_ctx() as session:
            session.add(
                Position(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="SPK/USDT",
                    side="short",
                    quantity=12.0,
                    entry_price=0.039,
                    current_price=0.0345,
                    leverage=3.0,
                    unrealized_pnl=0.0,
                    realized_pnl=1.21,
                    is_open=True,
                    closed_at=None,
                    created_at=phase3_time,
                    okx_inst_id="SPK-USDT-SWAP",
                    okx_pos_id="spk-phase3-pos",
                    close_exchange_order_id="phase3-order",
                )
            )

        report = await OkxOrderFactSyncService(
            mode="paper",
            lookback_hours=72,
            executor_factory=_Executor,
            cold_start_marker_path=None,
        ).sync()

        async with get_session_ctx() as session:
            position_rows = (
                await session.execute(
                    Position.__table__.select().where(
                        Position.__table__.c.okx_pos_id == "spk-phase3-pos"
                    )
                )
            ).all()
            history_rows = list(
                (
                    await session.execute(
                        OkxPositionHistory.__table__.select().where(
                            OkxPositionHistory.__table__.c.pos_id == "spk-phase3-pos"
                        )
                    )
                ).all()
            )

        assert report["position_history_checked_count"] == 1
        assert report["position_history_backfilled_count"] == 0
        assert report["position_history_updated_count"] == 1
        assert len(position_rows) == 1
        position = position_rows[0]._mapping
        assert position["is_open"] is False
        assert position["closed_at"] is not None
        assert position["model_name"] == "ensemble_trader"
        assert position["settlement_status"] == "okx_position_history"
        assert position["settlement_source"] == "okx_position_history"
        assert position["realized_pnl"] == pytest.approx(1.23)
        assert position["close_exchange_order_id"] == "phase3-order"
        assert len(history_rows) == 1
        assert history_rows[0]._mapping["position_ids"] == [str(position["id"])]
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_order_fact_sync_closes_open_position_from_stored_linked_close_orders_when_okx_unavailable(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'stored-linked-close-orders.db').as_posix()}",
    )
    await init_db()
    opened_at = (
        PHASE3_DEFAULT_ORDER_SYNC_START + timedelta(days=7, minutes=4)
    ).astimezone(UTC).replace(tzinfo=None)
    closed_at = opened_at + timedelta(hours=18)
    try:
        async with get_session_ctx() as session:
            session.add(
                Position(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="PROS/USDT",
                    side="long",
                    quantity=77.0,
                    entry_price=0.4294363636363637,
                    current_price=0.461,
                    leverage=1.0,
                    unrealized_pnl=0.0,
                    realized_pnl=4.4136505,
                    is_open=True,
                    closed_at=None,
                    created_at=opened_at,
                    okx_inst_id="PROS-USDT-SWAP",
                    okx_pos_id="3713396586719182848",
                    entry_exchange_order_id="pros-entry",
                    close_exchange_order_id="pros-close",
                )
            )
            for order_id, side, price, fee, fill_pnl, filled_at in (
                ("pros-entry", "buy", 0.4294363636363637, 0.0165333, 0.0, opened_at),
                ("pros-close", "sell", 0.487, 0.0187495, 4.4324, closed_at),
            ):
                session.add(
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="PROS/USDT",
                        side=side,
                        order_type="market",
                        quantity=77.0,
                        price=price,
                        status="filled",
                        fee=fee,
                        exchange_order_id=order_id,
                        filled_at=filled_at,
                        created_at=filled_at,
                        okx_inst_id="PROS-USDT-SWAP",
                        okx_trade_ids=f"trade-{order_id}",
                        okx_fill_contracts=77.0,
                        okx_fill_pnl=fill_pnl,
                        okx_sync_status=OKX_SYNC_CONFIRMED,
                        okx_raw_fills={
                            "order_id": order_id,
                            "trade_ids": [f"trade-{order_id}"],
                            "inst_id": "PROS-USDT-SWAP",
                            "contracts": 77.0,
                            "contract_size": 1.0,
                            "base_quantity": 77.0,
                            "avg_price": price,
                            "fee_abs": fee,
                            "fill_pnl": fill_pnl,
                            "timestamp": filled_at.isoformat(),
                            "fills_history_confirmed": True,
                        },
                    )
                )

        report = await OkxOrderFactSyncService(
            mode="paper",
            lookback_hours=72,
            executor_factory=_UnavailableExecutor,
            cold_start_marker_path=None,
        ).sync()

        async with get_session_ctx() as session:
            row = (
                await session.execute(
                    Position.__table__.select().where(Position.__table__.c.symbol == "PROS/USDT")
                )
            ).one()._mapping

        assert report["okx_pull_available"] is False
        assert report["current_position_updated_count"] == 1
        assert row["is_open"] is False
        assert row["closed_at"] is not None
        assert row["settlement_status"] == "reconciled"
        assert row["settlement_source"] == "okx_stored_linked_close_orders"
        assert row["realized_pnl"] == pytest.approx(4.3971172)
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_order_fact_sync_slow_account_bills_do_not_starve_core_order_facts(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-order-budget.db').as_posix()}",
    )
    await init_db()
    phase3_time = (PHASE3_DEFAULT_ORDER_SYNC_START + timedelta(minutes=10)).astimezone(UTC).replace(tzinfo=None)
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
                    filled_at=phase3_time,
                    created_at=phase3_time,
                )
            )

        report = await OkxOrderFactSyncService(
            mode="paper",
            lookback_hours=72,
            timeout_seconds=2.0,
            executor_factory=_SlowFundingBillExecutor,
            cold_start_marker_path=None,
        ).sync()

        assert report["okx_pull_available"] is True
        assert report["status"] == "warning"
        assert report["account_bill_error"]
        assert report["confirmed_count"] == 1
        assert report["unverified_count"] == 0
        assert report["position_history_backfilled_count"] == 0
        assert report["position_history_skipped_count"] == 1
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_order_fact_sync_order_history_timeout_keeps_core_fill_confirmation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-order-history-timeout.db').as_posix()}",
    )
    await init_db()
    _OrderHistoryBusyExecutor.ccxt_instances.clear()
    phase3_time = (
        PHASE3_DEFAULT_ORDER_SYNC_START + timedelta(minutes=10)
    ).astimezone(UTC).replace(tzinfo=None)
    try:
        async with get_session_ctx() as session:
            session.add(
                Order(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="SPK/USDT",
                    side="buy",
                    order_type="market",
                    quantity=1.0,
                    price=0.03,
                    status="filled",
                    fee=0.0,
                    exchange_order_id="phase3-order",
                    filled_at=phase3_time,
                    created_at=phase3_time,
                )
            )

        report = await OkxOrderFactSyncService(
            mode="paper",
            lookback_hours=72,
            timeout_seconds=2.0,
            executor_factory=_OrderHistoryBusyExecutor,
            cold_start_marker_path=None,
        ).sync()

        async with get_session_ctx() as session:
            row = (
                await session.execute(
                    Order.__table__.select().where(
                        Order.__table__.c.exchange_order_id == "phase3-order"
                    )
                )
            ).first()._mapping

        assert report["okx_pull_available"] is True
        assert report["status"] == "warning"
        assert any("orders_history" in item for item in report["optional_stage_errors"])
        assert report["confirmed_count"] == 1
        assert row["okx_sync_status"] == OKX_SYNC_CONFIRMED
        assert row["quantity"] == pytest.approx(12.0)
        assert row["price"] == pytest.approx(0.0345)
        assert row["fee"] == pytest.approx(0.02)
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_order_fact_sync_account_wide_fill_timeout_keeps_target_order_confirmation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-account-wide-fill-timeout.db').as_posix()}",
    )
    await init_db()
    _AccountWideFillBusyExecutor.ccxt_instances.clear()
    phase3_time = (
        PHASE3_DEFAULT_ORDER_SYNC_START + timedelta(minutes=10)
    ).astimezone(UTC).replace(tzinfo=None)
    try:
        async with get_session_ctx() as session:
            session.add(
                Order(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="SPK/USDT",
                    side="buy",
                    order_type="market",
                    quantity=1.0,
                    price=0.03,
                    status="filled",
                    fee=0.0,
                    exchange_order_id="phase3-order",
                    filled_at=phase3_time,
                    created_at=phase3_time,
                )
            )

        report = await OkxOrderFactSyncService(
            mode="paper",
            lookback_hours=72,
            timeout_seconds=2.0,
            executor_factory=_AccountWideFillBusyExecutor,
            cold_start_marker_path=None,
        ).sync()

        async with get_session_ctx() as session:
            row = (
                await session.execute(
                    Order.__table__.select().where(
                        Order.__table__.c.exchange_order_id == "phase3-order"
                    )
                )
            ).first()._mapping

        ccxt = _AccountWideFillBusyExecutor.ccxt_instances[-1]
        assert report["okx_pull_available"] is True
        assert report["status"] == "warning"
        assert any(
            "fills_history_account_context" in item
            for item in report["optional_stage_errors"]
        )
        assert report["confirmed_count"] == 1
        assert report["unverified_count"] == 0
        assert row["okx_sync_status"] == OKX_SYNC_CONFIRMED
        assert row["okx_raw_fills"]["fills_history_confirmed"] is True
        assert ccxt.fill_history_params[0].get("ordId") == "phase3-order"
        assert any("ordId" not in params for params in ccxt.fill_history_params)
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_order_fact_sync_prioritizes_target_order_fill_lookup_over_inst_scan(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-target-order-first.db').as_posix()}",
    )
    await init_db()
    _Executor.ccxt_instances.clear()
    phase3_time = (
        PHASE3_DEFAULT_ORDER_SYNC_START + timedelta(minutes=10)
    ).astimezone(UTC).replace(tzinfo=None)
    try:
        async with get_session_ctx() as session:
            session.add(
                Order(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="SPK/USDT",
                    side="buy",
                    order_type="market",
                    quantity=1.0,
                    price=0.03,
                    status="filled",
                    fee=0.0,
                    exchange_order_id="phase3-order",
                    filled_at=phase3_time,
                    created_at=phase3_time,
                )
            )

        report = await OkxOrderFactSyncService(
            mode="paper",
            lookback_hours=72,
            timeout_seconds=2.0,
            executor_factory=_Executor,
            cold_start_marker_path=None,
        ).sync()

        ccxt = _Executor.ccxt_instances[-1]
        assert report["okx_pull_available"] is True
        assert report["confirmed_count"] == 1
        assert ccxt.fill_history_params[0] == {
            "instType": "SWAP",
            "ordId": "phase3-order",
            "limit": "100",
            "begin": str(int(PHASE3_DEFAULT_ORDER_SYNC_START.timestamp() * 1000)),
        }
        assert any(
            "ordId" not in params and "instId" not in params
            for params in ccxt.fill_history_params
        )
        assert not any(
            "instId" in params and "ordId" not in params
            for params in ccxt.fill_history_params
        )
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_order_fact_sync_persists_okx_funding_account_bills(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-account-bills.db').as_posix()}",
    )
    await init_db()
    _FundingBillExecutor.ccxt_instances.clear()
    phase3_time = (
        PHASE3_DEFAULT_ORDER_SYNC_START + timedelta(minutes=10)
    ).astimezone(UTC).replace(tzinfo=None)
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
                    filled_at=phase3_time,
                    created_at=phase3_time,
                )
            )

        report = await OkxOrderFactSyncService(
            mode="paper",
            lookback_hours=72,
            executor_factory=_FundingBillExecutor,
            cold_start_marker_path=None,
        ).sync()

        async with get_session_ctx() as session:
            bill_rows = list(
                (
                    await session.execute(
                        OkxAccountBill.__table__.select().order_by(
                            OkxAccountBill.__table__.c.bill_id.asc()
                        )
                    )
                ).all()
            )
    finally:
        await close_db()

    assert report["account_bill_error"] is None
    assert report["account_bill_checked_count"] == 1
    assert report["account_bill_backfilled_count"] == 1
    assert len(bill_rows) == 1
    row = bill_rows[0]._mapping
    assert row["bill_id"] == "funding-bill-1"
    assert row["inst_id"] == "SPK-USDT-SWAP"
    assert row["pos_side"] == "short"
    assert row["bill_sub_type"] == "173"
    assert row["funding_fee"] == pytest.approx(-0.17)


@pytest.mark.asyncio
async def test_order_fact_sync_keeps_order_confirmation_when_position_history_is_busy(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-position-history-busy.db').as_posix()}",
    )
    await init_db()
    _PositionHistoryBusyExecutor.ccxt_instances.clear()
    phase3_time = (
        PHASE3_DEFAULT_ORDER_SYNC_START + timedelta(minutes=10)
    ).astimezone(UTC).replace(tzinfo=None)
    try:
        async with get_session_ctx() as session:
            session.add_all(
                [
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="SPK/USDT",
                        side="buy",
                        order_type="market",
                        quantity=12.0,
                        price=0.0345,
                        status="filled",
                        fee=0.0,
                        exchange_order_id="phase3-order",
                        filled_at=phase3_time,
                        created_at=phase3_time,
                    ),
                    Position(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="SPK/USDT",
                        side="short",
                        quantity=12.0,
                        entry_price=0.039,
                        current_price=0.0345,
                        leverage=3.0,
                        unrealized_pnl=0.0,
                        realized_pnl=-99.0,
                        is_open=False,
                        okx_inst_id="SPK-USDT-SWAP",
                        okx_pos_id="spk-phase3-pos",
                        entry_exchange_order_id="entry-order",
                        close_exchange_order_id="phase3-order",
                        closed_at=phase3_time,
                        created_at=phase3_time - timedelta(minutes=5),
                    ),
                ]
            )

        report = await OkxOrderFactSyncService(
            mode="paper",
            executor_factory=_PositionHistoryBusyExecutor,
            cold_start_marker_path=None,
        ).sync()

        async with get_session_ctx() as session:
            row = (
                await session.execute(
                    Order.__table__.select().where(
                        Order.__table__.c.exchange_order_id == "phase3-order"
                    )
                )
            ).first()._mapping

        assert report["okx_pull_available"] is True
        assert report["status"] == "warning"
        assert "positions-history busy" in report["position_history_error"]
        assert report["confirmed_count"] == 1
        assert row["okx_sync_status"] == OKX_SYNC_CONFIRMED
        assert row["okx_fill_pnl"] == pytest.approx(1.23)
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_order_fact_sync_persists_funding_bills_even_when_okx_pull_times_out(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-funding-bills-timeout.db').as_posix()}",
    )
    await init_db()
    phase3_time = (
        PHASE3_DEFAULT_ORDER_SYNC_START + timedelta(minutes=10)
    ).astimezone(UTC).replace(tzinfo=None)
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
                    filled_at=phase3_time,
                    created_at=phase3_time,
                )
            )

        report = await OkxOrderFactSyncService(
            mode="paper",
            lookback_hours=72,
            executor_factory=_FundingBillThenPullTimeoutExecutor,
            cold_start_marker_path=None,
        ).sync()

        async with get_session_ctx() as session:
            bill_rows = list(
                (
                    await session.execute(
                        OkxAccountBill.__table__.select().order_by(
                            OkxAccountBill.__table__.c.bill_id.asc()
                        )
                    )
                ).all()
            )
    finally:
        await close_db()

    assert report["okx_pull_available"] is True
    assert report["error"] is None
    assert any("positions: OKX pull timeout" in item for item in report["optional_stage_errors"])
    assert report["status"] == "warning"
    assert report["account_bill_error"] is None
    assert report["account_bill_checked_count"] == 1
    assert report["account_bill_backfilled_count"] == 1
    assert len(bill_rows) == 1
    row = bill_rows[0]._mapping
    assert row["bill_id"] == "funding-bill-1"
    assert row["inst_id"] == "SPK-USDT-SWAP"
    assert row["funding_fee"] == pytest.approx(-0.17)


@pytest.mark.asyncio
async def test_order_fact_sync_repairs_closed_position_pnl_from_okx_close_fill(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-close-fill-pnl-repair.db').as_posix()}",
    )
    await init_db()
    opened_at = (
        PHASE3_DEFAULT_ORDER_SYNC_START + timedelta(hours=7)
    ).astimezone(UTC).replace(tzinfo=None)
    closed_at = (
        PHASE3_DEFAULT_ORDER_SYNC_START + timedelta(hours=11, minutes=40)
    ).astimezone(UTC).replace(tzinfo=None)
    try:
        async with get_session_ctx() as session:
            session.add_all(
                [
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="MET/USDT",
                        side="sell",
                        order_type="market",
                        quantity=100.0,
                        price=0.1601,
                        status="filled",
                        fee=0.0,
                        exchange_order_id="3695269432500391936",
                        filled_at=opened_at,
                        created_at=opened_at,
                    ),
                    Position(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="MET/USDT",
                        side="short",
                        quantity=100.0,
                        entry_price=0.1601,
                        current_price=0.16437,
                        leverage=3.0,
                        unrealized_pnl=0.0,
                        realized_pnl=12.34,
                        settlement_status="reconciled",
                        is_open=False,
                        okx_inst_id="MET-USDT-SWAP",
                        okx_pos_id="met-pos",
                        entry_exchange_order_id="3695269432500391936",
                        close_exchange_order_id="3695833143904538624",
                        closed_at=closed_at,
                        created_at=opened_at,
                    ),
                ]
            )

        report = await OkxOrderFactSyncService(
            mode="paper",
            executor_factory=_FillPairOnlyExecutor,
            cold_start_marker_path=None,
        ).sync()

        async with get_session_ctx() as session:
            row = (
                await session.execute(
                    Position.__table__.select().where(
                        Position.__table__.c.close_exchange_order_id == "3695833143904538624"
                    )
                )
            ).first()._mapping

        assert report["closed_position_pnl_repaired_count"] == 1
        assert row["realized_pnl"] == pytest.approx(-0.4432235)
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_order_fact_sync_repairs_closed_position_pnl_from_stored_okx_facts_with_funding_fee(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-stored-close-fill-pnl-repair.db').as_posix()}",
    )
    await init_db()
    opened_at = (
        PHASE3_DEFAULT_ORDER_SYNC_START + timedelta(hours=12)
    ).astimezone(UTC).replace(tzinfo=None)
    closed_at = opened_at + timedelta(minutes=12)
    try:
        async with get_session_ctx() as session:
            session.add_all(
                [
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="AI16Z/USDT",
                        side="sell",
                        order_type="market",
                        quantity=100.0,
                        price=0.08,
                        status="filled",
                        fee=0.01,
                        exchange_order_id="ai16z-entry-okx",
                        filled_at=opened_at,
                        created_at=opened_at,
                        okx_inst_id="AI16Z-USDT-SWAP",
                        okx_trade_ids="trade-ai16z-entry-okx",
                        okx_fill_contracts=10.0,
                        okx_fill_pnl=0.0,
                        okx_sync_status=OKX_SYNC_CONFIRMED,
                        okx_raw_fills={
                            "fills_history_confirmed": True,
                            "order_id": "ai16z-entry-okx",
                            "trade_ids": ["trade-ai16z-entry-okx"],
                            "inst_id": "AI16Z-USDT-SWAP",
                            "contracts": 10.0,
                            "contract_size": 10.0,
                            "base_quantity": 100.0,
                            "avg_price": 0.08,
                            "fee_abs": 0.01,
                            "fill_pnl": 0.0,
                            "timestamp": opened_at.isoformat(),
                        },
                    ),
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="AI16Z/USDT",
                        side="buy",
                        order_type="market",
                        quantity=100.0,
                        price=0.072,
                        status="filled",
                        fee=0.01,
                        exchange_order_id="ai16z-close-okx",
                        filled_at=closed_at,
                        created_at=closed_at,
                        okx_inst_id="AI16Z-USDT-SWAP",
                        okx_trade_ids="trade-ai16z-close-okx",
                        okx_fill_contracts=10.0,
                        okx_fill_pnl=7.95,
                        okx_sync_status=OKX_SYNC_CONFIRMED,
                        okx_raw_fills={
                            "fills_history_confirmed": True,
                            "order_id": "ai16z-close-okx",
                            "trade_ids": ["trade-ai16z-close-okx"],
                            "inst_id": "AI16Z-USDT-SWAP",
                            "contracts": 10.0,
                            "contract_size": 10.0,
                            "base_quantity": 100.0,
                            "avg_price": 0.072,
                            "fee_abs": 0.01,
                            "fill_pnl": 7.95,
                            "timestamp": closed_at.isoformat(),
                        },
                    ),
                    Position(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="AI16Z/USDT",
                        side="short",
                        quantity=100.0,
                        entry_price=0.08,
                        current_price=0.072,
                        leverage=1.0,
                        unrealized_pnl=0.0,
                        realized_pnl=15.1136,
                        settlement_status="reconciled",
                        is_open=False,
                        okx_inst_id="AI16Z-USDT-SWAP",
                        entry_exchange_order_id="ai16z-entry-okx",
                        close_exchange_order_id="ai16z-close-okx",
                        closed_at=closed_at,
                        created_at=opened_at,
                    ),
                    OkxAccountBill(
                        mode="paper",
                        bill_id="ai16z-funding",
                        inst_id="AI16Z-USDT-SWAP",
                        pos_side="short",
                        ccy="USDT",
                        bill_type="8",
                        bill_sub_type="173",
                        bill_ts=opened_at + timedelta(minutes=5),
                        balance_change=-0.2,
                        pnl=-0.2,
                        fee=0.0,
                        funding_fee=-0.2,
                        source="okx_account_bills",
                        raw_bill={"subType": "173", "pnl": "-0.2"},
                    ),
                ]
            )

        report = await OkxOrderFactSyncService(
            mode="paper",
            executor_factory=_NoHistoryExecutor,
            cold_start_marker_path=None,
        ).sync()

        async with get_session_ctx() as session:
            row = (
                await session.execute(
                    Position.__table__.select().where(
                        Position.__table__.c.close_exchange_order_id == "ai16z-close-okx"
                    )
                )
            ).first()._mapping

        assert report["confirmed_count"] == 0
        assert report["closed_position_pnl_repaired_count"] == 1
        assert row["realized_pnl"] == pytest.approx(7.73)
        assert row["close_fill_pnl"] == pytest.approx(7.95)
        assert row["entry_fee"] == pytest.approx(0.01)
        assert row["close_fee"] == pytest.approx(0.01)
        assert row["funding_fee"] == pytest.approx(-0.2)
        assert row["settlement_status"] == "reconciled"
        assert row["settlement_source"] == "okx_order_fact_sync"
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_order_fact_sync_repairs_confirmed_order_columns_from_stored_okx_fill(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-stored-fill-order-repair.db').as_posix()}",
    )
    await init_db()
    filled_at = (
        PHASE3_DEFAULT_ORDER_SYNC_START + timedelta(hours=12)
    ).astimezone(UTC).replace(tzinfo=None)
    try:
        async with get_session_ctx() as session:
            session.add(
                Order(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="SAND/USDT",
                    side="buy",
                    order_type="market",
                    quantity=2910.0,
                    price=0.0498,
                    status="filled",
                    fee=0.0289836,
                    exchange_order_id="sand-close-okx",
                    filled_at=filled_at,
                    created_at=filled_at,
                    okx_inst_id="SAND-USDT-SWAP",
                    okx_trade_ids="trade-sand-close-okx",
                    okx_fill_contracts=291.0,
                    okx_fill_pnl=-7.245,
                    okx_sync_status=OKX_SYNC_CONFIRMED,
                    okx_raw_fills={
                        "fills_history_confirmed": True,
                        "order_id": "sand-close-okx",
                        "trade_ids": ["trade-sand-close-okx"],
                        "inst_id": "SAND-USDT-SWAP",
                        "contracts": 291.0,
                        "contract_size": 10.0,
                        "base_quantity": 1455.0,
                        "avg_price": 0.0498,
                        "fee_abs": 0.0289836,
                        "fill_pnl": -7.245,
                        "timestamp": filled_at.isoformat(),
                    },
                )
            )

        report = await OkxOrderFactSyncService(
            mode="paper",
            executor_factory=_NoHistoryExecutor,
            cold_start_marker_path=None,
        ).sync()

        async with get_session_ctx() as session:
            row = (
                await session.execute(
                    Order.__table__.select().where(
                        Order.__table__.c.exchange_order_id == "sand-close-okx"
                    )
                )
            ).first()._mapping

        assert report["confirmed_count"] == 1
        assert row["quantity"] == pytest.approx(2910.0)
        assert row["price"] == pytest.approx(0.0498)
        assert row["fee"] == pytest.approx(0.0289836)
        assert row["okx_fill_contracts"] == pytest.approx(291.0)
        assert row["okx_fill_pnl"] == pytest.approx(-7.245)
        assert row["okx_sync_status"] == OKX_SYNC_CONFIRMED
        assert row["okx_raw_fills"]["fills_history_confirmed"] is True
        assert row["okx_raw_fills"]["base_quantity"] == pytest.approx(2910.0)
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_order_fact_sync_repairs_stored_okx_facts_when_okx_pull_times_out(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-timeout-stored-pnl-repair.db').as_posix()}",
    )
    await init_db()
    opened_at = (
        PHASE3_DEFAULT_ORDER_SYNC_START + timedelta(hours=12)
    ).astimezone(UTC).replace(tzinfo=None)
    closed_at = opened_at + timedelta(minutes=12)
    try:
        async with get_session_ctx() as session:
            session.add_all(
                [
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="AI16Z/USDT",
                        side="sell",
                        order_type="market",
                        quantity=100.0,
                        price=0.08,
                        status="filled",
                        fee=0.01,
                        exchange_order_id="ai16z-entry-timeout",
                        filled_at=opened_at,
                        created_at=opened_at,
                        okx_inst_id="AI16Z-USDT-SWAP",
                        okx_trade_ids="trade-ai16z-entry-timeout",
                        okx_fill_contracts=10.0,
                        okx_fill_pnl=0.0,
                        okx_sync_status=OKX_SYNC_CONFIRMED,
                        okx_raw_fills={
                            "fills_history_confirmed": True,
                            "order_id": "ai16z-entry-timeout",
                            "trade_ids": ["trade-ai16z-entry-timeout"],
                            "inst_id": "AI16Z-USDT-SWAP",
                            "contracts": 10.0,
                            "contract_size": 10.0,
                            "base_quantity": 100.0,
                            "avg_price": 0.08,
                            "fee_abs": 0.01,
                            "fill_pnl": 0.0,
                            "timestamp": opened_at.isoformat(),
                        },
                    ),
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="AI16Z/USDT",
                        side="buy",
                        order_type="market",
                        quantity=100.0,
                        price=0.072,
                        status="filled",
                        fee=0.01,
                        exchange_order_id="ai16z-close-timeout",
                        filled_at=closed_at,
                        created_at=closed_at,
                        okx_inst_id="AI16Z-USDT-SWAP",
                        okx_trade_ids="trade-ai16z-close-timeout",
                        okx_fill_contracts=10.0,
                        okx_fill_pnl=7.95,
                        okx_sync_status=OKX_SYNC_CONFIRMED,
                        okx_raw_fills={
                            "fills_history_confirmed": True,
                            "order_id": "ai16z-close-timeout",
                            "trade_ids": ["trade-ai16z-close-timeout"],
                            "inst_id": "AI16Z-USDT-SWAP",
                            "contracts": 10.0,
                            "contract_size": 10.0,
                            "base_quantity": 100.0,
                            "avg_price": 0.072,
                            "fee_abs": 0.01,
                            "fill_pnl": 7.95,
                            "timestamp": closed_at.isoformat(),
                        },
                    ),
                    Position(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="AI16Z/USDT",
                        side="short",
                        quantity=100.0,
                        entry_price=0.08,
                        current_price=0.072,
                        leverage=1.0,
                        unrealized_pnl=0.0,
                        realized_pnl=15.1136,
                        settlement_status="reconciled",
                        is_open=False,
                        okx_inst_id="AI16Z-USDT-SWAP",
                        entry_exchange_order_id="ai16z-entry-timeout",
                        close_exchange_order_id="ai16z-close-timeout",
                        closed_at=closed_at,
                        created_at=opened_at,
                    ),
                ]
            )

        report = await OkxOrderFactSyncService(
            mode="paper",
            executor_factory=_PullTimeoutExecutor,
            cold_start_marker_path=None,
        ).sync()

        async with get_session_ctx() as session:
            row = (
                await session.execute(
                    Position.__table__.select().where(
                        Position.__table__.c.close_exchange_order_id == "ai16z-close-timeout"
                    )
                )
            ).first()._mapping

        assert report["okx_pull_available"] is True
        assert report["status"] == "warning"
        assert report["error"] is None
        assert any("positions: OKX pull timeout" in item for item in report["optional_stage_errors"])
        assert report["closed_position_pnl_repaired_count"] == 1
        assert row["realized_pnl"] == pytest.approx(7.93)
        assert row["close_fill_pnl"] == pytest.approx(7.95)
        assert row["entry_fee"] == pytest.approx(0.01)
        assert row["close_fee"] == pytest.approx(0.01)
        assert row["funding_fee"] == pytest.approx(0.0)
        assert row["settlement_status"] == "reconciled"
        assert row["settlement_source"] == "okx_order_fact_sync"
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_order_fact_sync_does_not_recount_already_recovered_execution_facts_on_timeout(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-timeout-no-recount.db').as_posix()}",
    )
    await init_db()
    filled_at = (
        PHASE3_DEFAULT_ORDER_SYNC_START + timedelta(days=4)
    ).astimezone(UTC).replace(tzinfo=None)
    try:
        async with get_session_ctx() as session:
            session.add(
                Order(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="SKY/USDT",
                    side="sell",
                    order_type="market",
                    quantity=3600.0,
                    price=0.0527,
                    status="filled",
                    fee=0.09486,
                    exchange_order_id="3705061053542670336",
                    filled_at=filled_at,
                    created_at=filled_at,
                    okx_inst_id="SKY-USDT-SWAP",
                    okx_trade_ids="16350886",
                    okx_fill_contracts=36.0,
                    okx_fill_pnl=0.0,
                    okx_sync_status=OKX_SYNC_EXECUTION_RESULT_CONFIRMED,
                    okx_raw_fills={
                        "source": "okx_execution_result",
                        "fills_history_confirmed": False,
                        "execution_result_confirmed": True,
                        "order_id": "3705061053542670336",
                        "trade_ids": ["16350886"],
                        "inst_id": "SKY-USDT-SWAP",
                        "contracts": 36.0,
                        "base_quantity": 3600.0,
                        "avg_price": 0.0527,
                        "fee_abs": 0.09486,
                        "fill_pnl": 0.0,
                        "timestamp": filled_at.isoformat(),
                    },
                )
            )

        report = await OkxOrderFactSyncService(
            mode="paper",
            executor_factory=_PullTimeoutExecutor,
            cold_start_marker_path=None,
        ).sync()

        assert report["okx_pull_available"] is True
        assert report["local_checked"] == 1
        assert report["error"] is None
        assert any("positions: OKX pull timeout" in item for item in report["optional_stage_errors"])
        assert report["confirmed_count"] == 0
        assert report["unverified_count"] == 0
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_order_fact_sync_recovers_close_fill_decision_fact_when_okx_pull_times_out(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-timeout-close-fill-decision.db').as_posix()}",
    )
    await init_db()
    opened_at = (
        PHASE3_DEFAULT_ORDER_SYNC_START + timedelta(days=4)
    ).astimezone(UTC).replace(tzinfo=None)
    closed_at = opened_at + timedelta(minutes=59)
    try:
        async with get_session_ctx() as session:
            close_decision = AIDecision(
                model_name="ensemble_trader",
                symbol="SKY/USDT",
                action="close_short",
                confidence=1.0,
                reasoning="exchange reconcile",
                position_size_pct=1.0,
                suggested_leverage=4.0,
                raw_llm_response={
                    "system_sync": True,
                    "source": "okx_position_reconcile",
                    "close_fill": {
                        "price": 0.05292,
                        "fee": 0.095256,
                        "order_id": "3705179573634957312",
                        "timestamp_ms": 1782925356210.0,
                        "timestamp": "2026-07-01T17:02:36.210000+00:00",
                        "quantity": 3600.0,
                        "contracts": 36.0,
                        "contract_size": 100.0,
                        "pnl": -0.792,
                        "source": "okx_fills_history",
                        "order_info": {
                            "ordId": "3705179573634957312",
                            "tradeId": "16354195",
                            "instId": "SKY-USDT-SWAP",
                            "posSide": "net",
                            "side": "buy",
                            "fillSz": "36",
                            "fillPx": "0.05292",
                            "fee": "-0.095256",
                            "fillPnl": "-0.792",
                            "ts": "1782925356210",
                        },
                    },
                },
                is_paper=True,
                was_executed=True,
                created_at=closed_at,
            )
            session.add(close_decision)
            await session.flush()
            session.add_all(
                [
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="SKY/USDT",
                        side="sell",
                        order_type="market",
                        quantity=3600.0,
                        price=0.0527,
                        status="filled",
                        fee=0.09486,
                        exchange_order_id="3705061053542670336",
                        filled_at=opened_at,
                        created_at=opened_at,
                        okx_inst_id="SKY-USDT-SWAP",
                        okx_trade_ids="16350886",
                        okx_fill_contracts=36.0,
                        okx_fill_pnl=0.0,
                        okx_sync_status=OKX_SYNC_EXECUTION_RESULT_CONFIRMED,
                        okx_raw_fills={
                            "source": "okx_execution_result",
                            "fills_history_confirmed": False,
                            "execution_result_confirmed": True,
                            "order_id": "3705061053542670336",
                            "trade_ids": ["16350886"],
                            "inst_id": "SKY-USDT-SWAP",
                            "contracts": 36.0,
                            "base_quantity": 3600.0,
                            "avg_price": 0.0527,
                            "fee_abs": 0.09486,
                            "fill_pnl": 0.0,
                            "timestamp": opened_at.isoformat(),
                        },
                    ),
                    Order(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="SKY/USDT",
                        side="buy",
                        order_type="market",
                        quantity=3600.0,
                        price=0.05292,
                        status="filled",
                        fee=0.095256,
                        exchange_order_id="3705179573634957312",
                        decision_id=close_decision.id,
                        filled_at=closed_at,
                        created_at=closed_at,
                    ),
                    Position(
                        model_name="ensemble_trader",
                        execution_mode="paper",
                        symbol="SKY/USDT",
                        side="short",
                        quantity=3600.0,
                        entry_price=0.0527,
                        current_price=0.05292,
                        leverage=4.0,
                        unrealized_pnl=0.0,
                        realized_pnl=12.34,
                        settlement_status="reconciled",
                        is_open=False,
                        okx_inst_id="SKY-USDT-SWAP",
                        okx_pos_id="3705061055321055232",
                        entry_exchange_order_id="3705061053542670336",
                        close_exchange_order_id="3705179573634957312",
                        closed_at=closed_at,
                        created_at=opened_at,
                    ),
                ]
            )

        report = await OkxOrderFactSyncService(
            mode="paper",
            executor_factory=_PullTimeoutExecutor,
            cold_start_marker_path=None,
        ).sync()

        async with get_session_ctx() as session:
            close_order = (
                await session.execute(
                    Order.__table__.select().where(
                        Order.__table__.c.exchange_order_id == "3705179573634957312"
                    )
                )
            ).first()._mapping
            position = (
                await session.execute(
                    Position.__table__.select().where(
                        Position.__table__.c.close_exchange_order_id == "3705179573634957312"
                    )
                )
            ).first()._mapping

        assert report["okx_pull_available"] is True
        assert report["error"] is None
        assert any("positions: OKX pull timeout" in item for item in report["optional_stage_errors"])
        assert report["confirmed_count"] == 1
        assert report["closed_position_pnl_repaired_count"] == 1
        assert close_order["okx_sync_status"] == OKX_SYNC_CONFIRMED
        assert close_order["okx_inst_id"] == "SKY-USDT-SWAP"
        assert close_order["okx_trade_ids"] == "16354195"
        assert close_order["okx_fill_contracts"] == pytest.approx(36.0)
        assert close_order["okx_fill_pnl"] == pytest.approx(-0.792)
        assert close_order["okx_raw_fills"]["fills_history_confirmed"] is True
        assert position["realized_pnl"] == pytest.approx(-0.982116)
        assert position["close_fill_pnl"] == pytest.approx(-0.792)
        assert position["entry_fee"] == pytest.approx(0.09486)
        assert position["close_fee"] == pytest.approx(0.095256)
        assert position["funding_fee"] == pytest.approx(0.0)
        assert position["settlement_status"] == "reconciled"
        assert position["settlement_source"] == "okx_order_fact_sync"
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
async def test_order_fact_sync_uses_full_account_history_page_budget(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-account-history-pages.db').as_posix()}",
    )
    await init_db()
    calls: dict[str, list[int]] = {"fills": [], "orders": [], "positions": []}

    class FakeNativeFacts:
        def __init__(self, _executor: Any) -> None:
            pass

        async def fetch_positions(self) -> list[dict[str, Any]]:
            return []

        async def fetch_fill_groups(self, **kwargs: Any) -> list[Any]:
            if kwargs.get("account_wide_only"):
                calls["fills"].append(int(kwargs["max_pages"]))
            return []

        async def fetch_order_history_rows(self, **kwargs: Any) -> list[dict[str, Any]]:
            if not kwargs.get("order_ids"):
                calls["orders"].append(int(kwargs["max_pages"]))
            return []

        async def fetch_contract_sizes(self, **_kwargs: Any) -> dict[str, float]:
            return {}

        async def fetch_account_bills(self, **_kwargs: Any) -> list[Any]:
            return []

        async def fetch_position_history_rows(self, **kwargs: Any) -> list[dict[str, Any]]:
            if not kwargs.get("pos_ids"):
                calls["positions"].append(int(kwargs["max_pages"]))
            return []

    class FakeExecutor:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        async def initialize(self) -> None:
            return None

        async def shutdown(self) -> None:
            return None

    monkeypatch.setattr("services.okx_order_fact_sync.OkxNativeFactsClient", FakeNativeFacts)
    try:
        report = await OkxOrderFactSyncService(
            mode="paper",
            limit=500,
            executor_factory=FakeExecutor,
            cold_start_marker_path=None,
        ).sync()
    finally:
        await close_db()

    assert report["okx_pull_available"] is True
    assert calls == {"fills": [7], "orders": [7], "positions": [7]}


@pytest.mark.asyncio
async def test_order_fact_sync_keeps_okx_execution_result_confirmed_when_history_lags(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-execution-result-fact.db').as_posix()}",
    )
    await init_db()
    phase3_time = (
        PHASE3_DEFAULT_ORDER_SYNC_START + timedelta(days=3, minutes=10)
    ).astimezone(UTC).replace(tzinfo=None)
    try:
        async with get_session_ctx() as session:
            session.add(
                Order(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="ACT/USDT",
                    side="buy",
                    order_type="market",
                    quantity=1580.0,
                    price=0.0097,
                    status="filled",
                    fee=0.007663,
                    exchange_order_id="3703940352525967360",
                    filled_at=phase3_time,
                    created_at=phase3_time,
                    okx_inst_id="ACT-USDT-SWAP",
                    okx_trade_ids="535631715",
                    okx_fill_contracts=158.0,
                    okx_fill_pnl=0.4582,
                    okx_state="filled",
                    okx_sync_status=OKX_SYNC_EXECUTION_RESULT_CONFIRMED,
                    okx_raw_fills={
                        "source": "okx_execution_result",
                        "fills_history_confirmed": False,
                        "execution_result_confirmed": True,
                        "order_id": "3703940352525967360",
                        "trade_ids": ["535631715"],
                        "inst_id": "ACT-USDT-SWAP",
                        "contracts": 158.0,
                        "base_quantity": 1580.0,
                        "avg_price": 0.0097,
                        "fee_abs": 0.007663,
                        "fill_pnl": 0.4582,
                        "rows": [{"ordId": "3703940352525967360", "accFillSz": "158"}],
                    },
                )
            )

        report = await OkxOrderFactSyncService(
            mode="paper",
            executor_factory=_NoHistoryExecutor,
            cold_start_marker_path=None,
        ).sync()

        async with get_session_ctx() as session:
            row = (
                await session.execute(
                    Order.__table__.select().where(
                        Order.__table__.c.exchange_order_id == "3703940352525967360"
                    )
                )
            ).one()._mapping

        assert report["confirmed_count"] == 0
        assert report["unverified_count"] == 0
        assert row["okx_sync_status"] == OKX_SYNC_EXECUTION_RESULT_CONFIRMED
        assert row["okx_state"] == "execution_result_confirmed"
        assert row["okx_fill_contracts"] == pytest.approx(158.0)
        assert row["okx_fill_pnl"] == pytest.approx(0.4582)
        assert row["okx_last_error"] is None
        assert row["okx_raw_fills"]["execution_result_confirmed"] is True
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_order_fact_sync_recovers_okx_execution_result_fact_from_decision(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-execution-result-decision-recovery.db').as_posix()}",
    )
    await init_db()
    phase3_time = (
        PHASE3_DEFAULT_ORDER_SYNC_START + timedelta(days=3, minutes=10)
    ).astimezone(UTC).replace(tzinfo=None)
    try:
        async with get_session_ctx() as session:
            decision = AIDecision(
                model_name="ensemble_trader",
                symbol="ACT/USDT",
                action="close_short",
                confidence=0.88,
                reasoning="close",
                position_size_pct=1.0,
                suggested_leverage=1.0,
                raw_llm_response={
                    "execution_result": {
                        "order_id": "3703940352525967360",
                        "exchange_order_id": "3703940352525967360",
                        "status": "filled",
                        "quantity": 1580.0,
                        "price": 0.0097,
                        "fee": 0.007663,
                        "pnl": 0.4582,
                        "timestamp": "2026-07-01T06:46:00+00:00",
                        "raw_response": {
                            "id": "3703940352525967360",
                            "filled_contracts": 158.0,
                            "average": 0.0097,
                            "info": {
                                "ordId": "3703940352525967360",
                                "instId": "ACT-USDT-SWAP",
                                "state": "filled",
                                "accFillSz": "158",
                                "avgPx": "0.0097",
                                "fee": "-0.007663",
                                "pnl": "0.4582",
                                "tradeId": "535631715",
                            },
                        },
                    }
                },
                is_paper=True,
                was_executed=True,
                created_at=phase3_time,
            )
            session.add(decision)
            await session.flush()
            session.add(
                Order(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="ACT/USDT",
                    side="buy",
                    order_type="market",
                    quantity=1580.0,
                    price=0.0097,
                    status="filled",
                    fee=0.007663,
                    decision_id=decision.id,
                    exchange_order_id="3703940352525967360",
                    filled_at=phase3_time,
                    created_at=phase3_time,
                    okx_inst_id="ACT-USDT-SWAP",
                    okx_state="missing_okx_fill",
                    okx_sync_status=OKX_SYNC_UNVERIFIED,
                    okx_last_error="OKX orders-history/fills-history did not confirm this local filled order",
                )
            )

        report = await OkxOrderFactSyncService(
            mode="paper",
            executor_factory=_NoHistoryExecutor,
            cold_start_marker_path=None,
        ).sync()

        async with get_session_ctx() as session:
            row = (
                await session.execute(
                    Order.__table__.select().where(
                        Order.__table__.c.exchange_order_id == "3703940352525967360"
                    )
                )
            ).one()._mapping

        assert report["confirmed_count"] == 1
        assert report["unverified_count"] == 0
        assert row["okx_sync_status"] == OKX_SYNC_EXECUTION_RESULT_CONFIRMED
        assert row["okx_state"] == "execution_result_confirmed"
        assert row["okx_trade_ids"] == "535631715"
        assert row["okx_fill_contracts"] == pytest.approx(158.0)
        assert row["okx_fill_pnl"] == pytest.approx(0.4582)
        assert row["okx_last_error"] is None
        assert row["okx_raw_fills"]["recovered_from_decision"] == decision.id
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
        assert position["entry_fee"] == pytest.approx(0.006522)
        assert position["stop_loss_price"] == pytest.approx(0.000023)
        assert position["take_profit_price"] == pytest.approx(0.000019)
        assert position["current_management_contract"]["management_eligible"] is True, position[
            "current_management_contract"
        ]
        assert position["current_management_contract"]["can_expand_position"] is False
        assert position["current_management_contract"]["original_entry_contract_status"] == (
            "historical_entry_contract_incomplete_preserved"
        )
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
        assert position["model_name"] == "old_local_cache"
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
async def test_order_fact_sync_preserves_complete_management_contract_on_protection_timeout(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-current-position-protection-timeout.db').as_posix()}",
    )
    await init_db()
    try:
        await OkxOrderFactSyncService(
            mode="paper",
            executor_factory=_CurrentPositionOnlyExecutor,
            cold_start_marker_path=None,
        ).sync()
        async with get_session_ctx() as session:
            before = (
                await session.execute(
                    Position.__table__.select().where(
                        Position.__table__.c.okx_pos_id == "3695537280250515456"
                    )
                )
            ).one()._mapping
            before_contract = dict(before["current_management_contract"])
            assert before_contract["management_eligible"] is True

        report = await OkxOrderFactSyncService(
            mode="paper",
            executor_factory=_CurrentPositionProtectionTimeoutExecutor,
            cold_start_marker_path=None,
        ).sync()

        async with get_session_ctx() as session:
            after = (
                await session.execute(
                    Position.__table__.select().where(
                        Position.__table__.c.okx_pos_id == "3695537280250515456"
                    )
                )
            ).one()._mapping

        assert any(
            item.startswith("active_position_protection:")
            for item in report["optional_stage_errors"]
        )
        assert after["current_management_contract"] == before_contract
        assert after["current_management_contract"]["management_eligible"] is True
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_order_fact_sync_links_reduce_only_partial_close_to_open_position(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-partial-close-link.db').as_posix()}",
    )
    await init_db()
    position_time = (
        PHASE3_DEFAULT_ORDER_SYNC_START + timedelta(hours=2)
    ).astimezone(UTC).replace(tzinfo=None)
    try:
        async with get_session_ctx() as session:
            session.add(
                Position(
                    model_name="ensemble_trader",
                    execution_mode="paper",
                    symbol="PEPE/USDT",
                    side="long",
                    quantity=4000000.0,
                    entry_price=0.000002599,
                    current_price=0.0000026,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    is_open=True,
                    created_at=position_time,
                    okx_inst_id="PEPE-USDT-SWAP",
                    okx_pos_id="pepe-position",
                    entry_exchange_order_id="3711253572185980928",
                )
            )

        report = await OkxOrderFactSyncService(
            mode="paper",
            executor_factory=_PartialCloseCurrentPositionExecutor,
            cold_start_marker_path=None,
        ).sync()

        async with get_session_ctx() as session:
            position = (
                await session.execute(
                    Position.__table__.select().where(
                        Position.__table__.c.okx_pos_id == "pepe-position"
                    )
                )
            ).one()._mapping
            close_order = (
                await session.execute(
                    Order.__table__.select().where(
                        Order.__table__.c.exchange_order_id == "3712882834810830848"
                    )
                )
            ).one()._mapping

        assert report["okx_pull_available"] is True
        assert report["current_position_updated_count"] == 1
        assert position["is_open"] is True
        assert position["quantity"] == pytest.approx(1000000.0)
        assert position["close_exchange_order_id"] == "3712882834810830848"
        assert position["realized_pnl"] == pytest.approx(0.382908)
        assert close_order["okx_sync_status"] == OKX_SYNC_OKX_ONLY
        assert close_order["okx_raw_fills"].get("order_rows", []) == []
    finally:
        await close_db()


@pytest.mark.asyncio
async def test_order_fact_sync_links_unlinked_partial_close_to_closed_lifecycle(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-partial-close-closed-lifecycle.db').as_posix()}",
    )
    await init_db()
    position_time = (
        PHASE3_DEFAULT_ORDER_SYNC_START + timedelta(hours=2)
    ).astimezone(UTC).replace(tzinfo=None)
    closed_time = (
        PHASE3_DEFAULT_ORDER_SYNC_START + timedelta(hours=5)
    ).astimezone(UTC).replace(tzinfo=None)
    try:
        async with get_session_ctx() as session:
            session.add(
                Position(
                    model_name="okx_authoritative_sync",
                    execution_mode="paper",
                    symbol="PEPE/USDT",
                    side="long",
                    quantity=1000000.0,
                    entry_price=0.000002599,
                    current_price=0.000002735,
                    leverage=1.0,
                    unrealized_pnl=0.0,
                    realized_pnl=1.80904525,
                    is_open=False,
                    created_at=position_time,
                    closed_at=closed_time,
                    okx_inst_id="PEPE-USDT-SWAP",
                    okx_pos_id="pepe-position",
                    entry_exchange_order_id="3711253572185980928",
                    close_exchange_order_id="3713658183647723520",
                )
            )

        report = await OkxOrderFactSyncService(
            mode="paper",
            executor_factory=_PartialCloseCurrentPositionExecutor,
            cold_start_marker_path=None,
        ).sync()

        async with get_session_ctx() as session:
            position = (
                await session.execute(
                    Position.__table__.select().where(
                        Position.__table__.c.okx_pos_id == "pepe-position",
                        Position.__table__.c.is_open.is_(False),
                    )
                )
            ).one()._mapping

        assert report["okx_pull_available"] is True
        assert position["realized_pnl"] == pytest.approx(1.80904525)
        assert set(position["close_exchange_order_id"].split(",")) == {
            "3712882834810830848",
            "3713658183647723520",
        }
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
async def test_order_fact_sync_suppresses_manual_deleted_fill_pair_position(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-suppressed-fill-pair.db').as_posix()}",
    )
    await init_db()
    try:
        async with get_session_ctx() as session:
            session.add(
                StrategyLearningEvent(
                    model_name="okx_authoritative_sync",
                    execution_mode="paper",
                    symbol="MET/USDT",
                    side="short",
                    action="suppress_sync",
                    event_type=OKX_POSITION_SYNC_SUPPRESSION_EVENT_TYPE,
                    event_status="active",
                    severity="info",
                    reason="manual user deletion of invalid historical position row",
                    attribution={
                        "okx_inst_id": "MET-USDT-SWAP",
                        "entry_exchange_order_ids": ["3695269432500391936"],
                        "close_exchange_order_ids": ["3695833143904538624"],
                    },
                    exclude_from_training=True,
                )
            )

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
        assert report["fill_pair_position_checked_count"] == 1
        assert report["fill_pair_position_backfilled_count"] == 0
        assert report["fill_pair_position_skipped_count"] == 1
        assert len(order_rows) == 2
        assert len(position_rows) == 0
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
async def test_order_fact_sync_suppresses_manual_deleted_position_history_lifecycle(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await close_db()
    monkeypatch.setattr(
        settings,
        "database_url",
        f"sqlite+aiosqlite:///{(tmp_path / 'okx-suppressed-history-lifecycle.db').as_posix()}",
    )
    await init_db()
    try:
        async with get_session_ctx() as session:
            session.add(
                StrategyLearningEvent(
                    model_name="okx_authoritative_sync",
                    execution_mode="paper",
                    symbol="ACT/USDT",
                    side="short",
                    action="suppress_sync",
                    event_type=OKX_POSITION_SYNC_SUPPRESSION_EVENT_TYPE,
                    event_status="active",
                    severity="info",
                    reason="manual user deletion of invalid OKX historical lifecycle",
                    attribution={
                        "okx_inst_id": "ACT-USDT-SWAP",
                        "okx_pos_id": _RepeatedPosIdLifecycleCcxt.pos_id,
                        "entry_exchange_order_ids": [
                            _RepeatedPosIdLifecycleCcxt.new_entry_order_id
                        ],
                        "close_exchange_order_ids": [
                            _RepeatedPosIdLifecycleCcxt.new_close_order_id
                        ],
                        "closed_at": "2026-06-28T07:46:45.670000+00:00",
                    },
                    exclude_from_training=True,
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
                    Order.__table__.select().where(Order.__table__.c.symbol == "ACT/USDT")
                )
            ).all()

        positions = [row._mapping for row in position_rows]
        assert report["position_history_checked_count"] == 2
        assert report["position_history_backfilled_count"] == 1
        assert report["position_history_skipped_count"] == 1
        assert report["fill_pair_position_backfilled_count"] == 0
        assert len(order_rows) == 4
        assert len(positions) == 1
        assert positions[0]["entry_exchange_order_id"] == _RepeatedPosIdLifecycleCcxt.old_entry_order_id
        assert positions[0]["close_exchange_order_id"] == _RepeatedPosIdLifecycleCcxt.old_close_order_id
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
