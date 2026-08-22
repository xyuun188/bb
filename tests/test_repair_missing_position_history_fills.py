from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from scripts.repair_missing_position_history_fills import (
    RebuildPlan,
    _apply_reversal_boundary_allocations,
    _order_fill_fact_needs_refresh,
    _plan_needs_history_rebuild,
    _raw_fill_fact,
)
from services.okx_native_facts import (
    OKX_ACCOUNT_BILLS_TRADE_SOURCE,
    OkxNativeFillGroup,
)


def _plan(
    *,
    history_id: int,
    entry_target: float,
    close_target: float,
    entry_matched: bool,
    close_matched: bool,
) -> RebuildPlan:
    return RebuildPlan(
        history_id=history_id,
        inst_id="TRX-USDT-SWAP",
        symbol="TRX/USDT",
        entry_target_contracts=entry_target,
        close_target_contracts=close_target,
        entry_order_ids=(f"entry-{history_id}",) if entry_matched else (),
        close_order_ids=(f"close-{history_id}",) if close_matched else (),
        old_entry_order_ids=(),
        old_close_order_ids=(),
        entry_matched=entry_matched,
        close_matched=close_matched,
        entry_allocations=(),
        close_allocations=(),
        old_allocation_document={},
    )


def test_raw_fill_fact_preserves_verified_account_bill_trade_authority() -> None:
    timestamp = datetime(2026, 7, 4, 9, 18, tzinfo=UTC)
    row = {
        "instId": "LAB-USDT-SWAP",
        "ordId": "entry-lab",
        "tradeId": "trade-lab",
        "side": "sell",
        "posSide": "net",
        "fillSz": "5",
        "fillPx": "9.7",
        "fillMarkPx": "9.71",
        "fee": "-0.01",
        "fillPnl": "0",
        "ts": str(int(timestamp.timestamp() * 1000)),
        "_bb_fill_fact_source": OKX_ACCOUNT_BILLS_TRADE_SOURCE,
    }
    fill = OkxNativeFillGroup(
        order_id="entry-lab",
        trade_ids=("trade-lab",),
        inst_id="LAB-USDT-SWAP",
        symbol="LAB/USDT",
        side="sell",
        pos_side="net",
        contracts=5.0,
        avg_price=9.7,
        fee_abs=0.01,
        fill_pnl=0.0,
        timestamp_ms=timestamp.timestamp() * 1000,
        timestamp=timestamp,
        raw_count=1,
        rows=(row,),
    )

    raw = _raw_fill_fact(fill, 10.0)

    assert raw["source"] == OKX_ACCOUNT_BILLS_TRADE_SOURCE
    assert raw["account_bills_trade_confirmed"] is True
    assert raw["fills_history_confirmed"] is False
    assert raw["contract_size_verified"] is True
    assert raw["base_quantity"] == pytest.approx(50.0)
    assert raw["execution_slippage"]["complete"] is True


def test_order_fill_refresh_ignores_non_contract_recovery_metadata() -> None:
    timestamp = datetime(2026, 7, 4, 9, 18, tzinfo=UTC)
    fill = OkxNativeFillGroup(
        order_id="entry-history",
        trade_ids=("trade-history",),
        inst_id="LAB-USDT-SWAP",
        symbol="LAB/USDT",
        side="sell",
        pos_side="net",
        contracts=5.0,
        avg_price=9.7,
        fee_abs=0.01,
        fill_pnl=0.0,
        timestamp_ms=timestamp.timestamp() * 1000,
        timestamp=timestamp,
        raw_count=1,
        rows=({},),
    )
    raw = _raw_fill_fact(fill, 10.0)
    raw.update(
        {
            "source": "okx_fills_history_targeted",
            "timestamp": "different-recovery-format",
            "trade_ids": ["older-trade-shape"],
            "rows": [],
            "execution_slippage": {"complete": False},
        }
    )
    order = SimpleNamespace(
        exchange_order_id=fill.order_id,
        quantity=50.0,
        okx_fill_contracts=5.0,
        price=9.7,
        fee=0.01,
        okx_raw_fills=raw,
    )

    assert _order_fill_fact_needs_refresh(order, fill, 10.0) is False


def test_order_fill_refresh_requires_verified_contract_size() -> None:
    timestamp = datetime(2026, 7, 4, 9, 18, tzinfo=UTC)
    fill = OkxNativeFillGroup(
        order_id="entry-history",
        trade_ids=("trade-history",),
        inst_id="LAB-USDT-SWAP",
        symbol="LAB/USDT",
        side="sell",
        pos_side="net",
        contracts=5.0,
        avg_price=9.7,
        fee_abs=0.01,
        fill_pnl=0.0,
        timestamp_ms=timestamp.timestamp() * 1000,
        timestamp=timestamp,
        raw_count=1,
        rows=({},),
    )
    raw = _raw_fill_fact(fill, 10.0)
    raw["contract_size_verified"] = False
    order = SimpleNamespace(
        exchange_order_id=fill.order_id,
        quantity=50.0,
        okx_fill_contracts=5.0,
        price=9.7,
        fee=0.01,
        okx_raw_fills=raw,
    )

    assert _order_fill_fact_needs_refresh(order, fill, 10.0) is True


def test_account_bill_fill_refresh_requires_archive_source() -> None:
    timestamp = datetime(2026, 7, 4, 9, 18, tzinfo=UTC)
    row = {"_bb_fill_fact_source": OKX_ACCOUNT_BILLS_TRADE_SOURCE}
    fill = OkxNativeFillGroup(
        order_id="entry-lab",
        trade_ids=("trade-lab",),
        inst_id="LAB-USDT-SWAP",
        symbol="LAB/USDT",
        side="sell",
        pos_side="net",
        contracts=5.0,
        avg_price=9.7,
        fee_abs=0.01,
        fill_pnl=0.0,
        timestamp_ms=timestamp.timestamp() * 1000,
        timestamp=timestamp,
        raw_count=1,
        rows=(row,),
    )
    raw = _raw_fill_fact(fill, 10.0)
    raw["source"] = "okx_fills_history"
    order = SimpleNamespace(
        exchange_order_id=fill.order_id,
        quantity=50.0,
        okx_fill_contracts=5.0,
        price=9.7,
        fee=0.01,
        okx_raw_fills=raw,
    )

    assert _order_fill_fact_needs_refresh(order, fill, 10.0) is True


def test_history_rebuild_replays_complete_changed_links_deterministically() -> None:
    plan = replace(
        _plan(
            history_id=99,
            entry_target=5.0,
            close_target=5.0,
            entry_matched=True,
            close_matched=True,
        ),
        old_entry_order_ids=("old-entry",),
        old_close_order_ids=("old-close",),
    )

    assert plan.changed is True
    assert _plan_needs_history_rebuild(plan, SimpleNamespace(evidence_gaps=[])) is True
    assert (
        _plan_needs_history_rebuild(
            plan,
            SimpleNamespace(
                evidence_gaps=["position_history_close_quantity_not_matched_to_orders"]
            ),
        )
        is True
    )


def test_exact_reversal_boundary_order_is_split_across_two_lifecycles() -> None:
    boundary_at = datetime(2026, 8, 12, 7, 35, tzinfo=UTC)
    previous = SimpleNamespace(
        id=101,
        inst_id="TRX-USDT-SWAP",
        side="long",
        opened_at=boundary_at - timedelta(hours=1),
        updated_at_okx=boundary_at,
        raw_row={"posSide": "long"},
    )
    following = SimpleNamespace(
        id=102,
        inst_id="TRX-USDT-SWAP",
        side="short",
        opened_at=boundary_at,
        updated_at_okx=boundary_at + timedelta(hours=1),
        raw_row={"posSide": "short"},
    )
    fill = OkxNativeFillGroup(
        order_id="3713173120610959360",
        trade_ids=("trade-1",),
        inst_id="TRX-USDT-SWAP",
        symbol="TRX/USDT",
        side="sell",
        pos_side="net",
        contracts=0.38,
        avg_price=0.3,
        fee_abs=0.01,
        fill_pnl=0.02,
        timestamp_ms=boundary_at.timestamp() * 1000,
        timestamp=boundary_at,
        raw_count=1,
        rows=(),
    )

    plans = _apply_reversal_boundary_allocations(
        [previous, following],
        [
            _plan(
                history_id=101,
                entry_target=0.09,
                close_target=0.09,
                entry_matched=True,
                close_matched=False,
            ),
            _plan(
                history_id=102,
                entry_target=0.29,
                close_target=0.29,
                entry_matched=False,
                close_matched=True,
            ),
        ],
        {"TRX-USDT-SWAP": [fill]},
    )

    previous_plan, following_plan = plans
    assert previous_plan.close_order_ids == (fill.order_id,)
    assert following_plan.entry_order_ids == (fill.order_id,)
    assert previous_plan.close_matched is True
    assert following_plan.entry_matched is True
    assert previous_plan.close_allocations[0]["allocated_contracts"] == pytest.approx(0.09)
    assert following_plan.entry_allocations[0]["allocated_contracts"] == pytest.approx(0.29)
    assert previous_plan.close_allocations[0]["order_contracts"] == pytest.approx(0.38)
    assert following_plan.entry_allocations[0]["order_contracts"] == pytest.approx(0.38)
    assert previous_plan.evidence_gaps == ()
    assert following_plan.evidence_gaps == ()


def test_reversal_allocation_rejects_non_exact_order_quantity() -> None:
    boundary_at = datetime(2026, 8, 12, 7, 35, tzinfo=UTC)
    histories = [
        SimpleNamespace(
            id=101,
            inst_id="TRX-USDT-SWAP",
            side="long",
            opened_at=boundary_at - timedelta(hours=1),
            updated_at_okx=boundary_at,
            raw_row={"posSide": "long"},
        ),
        SimpleNamespace(
            id=102,
            inst_id="TRX-USDT-SWAP",
            side="short",
            opened_at=boundary_at,
            updated_at_okx=boundary_at + timedelta(hours=1),
            raw_row={"posSide": "short"},
        ),
    ]
    fill = OkxNativeFillGroup(
        order_id="wrong-size",
        trade_ids=("trade-1",),
        inst_id="TRX-USDT-SWAP",
        symbol="TRX/USDT",
        side="sell",
        pos_side="net",
        contracts=0.37,
        avg_price=0.3,
        fee_abs=0.01,
        fill_pnl=0.02,
        timestamp_ms=boundary_at.timestamp() * 1000,
        timestamp=boundary_at,
        raw_count=1,
        rows=(),
    )
    original = [
        _plan(
            history_id=101,
            entry_target=0.09,
            close_target=0.09,
            entry_matched=True,
            close_matched=False,
        ),
        _plan(
            history_id=102,
            entry_target=0.29,
            close_target=0.29,
            entry_matched=False,
            close_matched=True,
        ),
    ]

    rebuilt = _apply_reversal_boundary_allocations(
        histories,
        original,
        {"TRX-USDT-SWAP": [fill]},
    )

    assert rebuilt == original


def test_reversal_allocation_accounts_for_later_lifecycle_additions() -> None:
    boundary_at = datetime(2026, 7, 9, 2, 39, 10, 381000, tzinfo=UTC)
    previous = SimpleNamespace(
        id=201,
        inst_id="JUP-USDT-SWAP",
        side="short",
        opened_at=boundary_at - timedelta(hours=2),
        updated_at_okx=boundary_at,
        raw_row={"posSide": "short"},
    )
    following = SimpleNamespace(
        id=202,
        inst_id="JUP-USDT-SWAP",
        side="long",
        opened_at=boundary_at,
        updated_at_okx=boundary_at + timedelta(hours=8),
        raw_row={"posSide": "long"},
    )
    boundary_fill = OkxNativeFillGroup(
        order_id="reversal-88",
        trade_ids=("trade-boundary",),
        inst_id="JUP-USDT-SWAP",
        symbol="JUP/USDT",
        side="buy",
        pos_side="net",
        contracts=88.0,
        avg_price=0.2096,
        fee_abs=0.092224,
        fill_pnl=0.1829,
        timestamp_ms=boundary_at.timestamp() * 1000,
        timestamp=boundary_at,
        raw_count=1,
        rows=(),
    )
    later_addition = OkxNativeFillGroup(
        order_id="later-entry-131",
        trade_ids=("trade-add",),
        inst_id="JUP-USDT-SWAP",
        symbol="JUP/USDT",
        side="buy",
        pos_side="net",
        contracts=131.0,
        avg_price=0.21,
        fee_abs=0.13,
        fill_pnl=0.0,
        timestamp_ms=(boundary_at + timedelta(hours=1)).timestamp() * 1000,
        timestamp=boundary_at + timedelta(hours=1),
        raw_count=1,
        rows=(),
    )
    plans = _apply_reversal_boundary_allocations(
        [previous, following],
        [
            _plan(
                history_id=201,
                entry_target=62.0,
                close_target=62.0,
                entry_matched=True,
                close_matched=False,
            ),
            _plan(
                history_id=202,
                entry_target=157.0,
                close_target=157.0,
                entry_matched=False,
                close_matched=True,
            ),
        ],
        {"JUP-USDT-SWAP": [boundary_fill, later_addition]},
    )

    previous_plan, following_plan = plans
    assert previous_plan.close_order_ids == ("reversal-88",)
    assert following_plan.entry_order_ids == ("reversal-88", "later-entry-131")
    assert previous_plan.close_allocations[0]["allocated_contracts"] == pytest.approx(62.0)
    assert following_plan.entry_allocations[0]["allocated_contracts"] == pytest.approx(26.0)
    assert previous_plan.evidence_gaps == ()
    assert following_plan.evidence_gaps == ()
