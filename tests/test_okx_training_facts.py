from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from services.authoritative_trade_outcome import build_authoritative_trade_outcome
from services.entry_direction_support import assess_directional_entry_support
from services.normal_paper_trade import (
    build_normal_paper_trade_contract,
    normal_paper_trade_contract_reasons,
)
from services.okx_execution_slippage import (
    OKX_ROUND_TRIP_SLIPPAGE_SOURCE,
    build_okx_fill_mark_slippage,
)
from services.okx_lifecycle_order_allocations import (
    LIFECYCLE_ORDER_ALLOCATIONS_KEY,
    build_lifecycle_order_allocation,
    build_lifecycle_order_allocation_document,
)
from services.okx_training_facts import (
    _funding_training_evidence,
    build_funding_bill_lifecycle_facts,
    build_okx_history_training_sample,
)
from services.production_trade_gate import PRODUCTION_TRADE_GATE_VERSION
from services.profit_training_contract import PROFIT_TRAINING_TARGET
from services.training_data_quality import annotate_training_payload
from tests.legacy_paper_contract_fixtures import (
    build_legacy_normal_paper_trade_contract,
    build_legacy_normal_paper_v4_trade_contract,
)
from tests.legacy_paper_contract_fixtures import (
    build_legacy_paper_exploration_contract as build_paper_exploration_contract,
)
from tests.legacy_paper_contract_fixtures import (
    build_legacy_paper_training_contract as build_paper_training_contract,
)


def _history(**overrides):
    opened = datetime(2026, 7, 11, 1, tzinfo=UTC)
    raw = {
        "instId": "BTC-USDT-SWAP",
        "posId": "pos-1",
        "posSide": "long",
        "realizedPnl": "8.5",
        "pnl": "10",
        "fee": "-1",
        "fundingFee": "-0.5",
        "pnlRatio": "0.0085",
        "_bb_contract_spec": {
            "ctVal": "0.01",
            "ctMult": "1",
            "lotSz": "1",
            "source": "okx_public_instruments",
        },
        "_bb_contract_spec_source": "okx_public_instruments",
    }
    values = {
        "id": 1,
        "mode": "paper",
        "row_identity": "paper|BTC-USDT-SWAP|pos-1|long|1",
        "inst_id": "BTC-USDT-SWAP",
        "symbol": "BTC/USDT",
        "pos_id": "pos-1",
        "side": "long",
        "close_status": "full",
        "opened_at": opened,
        "updated_at_okx": opened + timedelta(hours=1),
        "open_avg_px": 100_000.0,
        "close_avg_px": 100_500.0,
        "open_max_pos": 2.0,
        "leverage": 2.0,
        "realized_pnl": 8.5,
        "pnl": 10.0,
        "pnl_ratio": 0.0085,
        "funding_fee": -0.5,
        "fee": -1.0,
        "entry_order_ids": ["entry-1"],
        "close_order_ids": ["close-1"],
        "linked_order_ids": ["entry-1", "close-1"],
        "position_ids": [7],
        "evidence_gaps": [],
        "raw_row": raw,
        "sync_status": "synced",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _slippage_fact(
    *,
    order_id: str,
    trade_id: str,
    side: str,
    average_price: float,
    mark_price: float,
    contracts: float = 2.0,
    contract_size: float = 0.01,
    inst_id: str = "BTC-USDT-SWAP",
) -> dict:
    return build_okx_fill_mark_slippage(
        order_id=order_id,
        inst_id=inst_id,
        side=side,
        contracts=contracts,
        average_price=average_price,
        contract_size=contract_size,
        rows=[
            {
                "ordId": order_id,
                "instId": inst_id,
                "tradeId": trade_id,
                "side": side,
                "fillSz": str(contracts),
                "fillPx": str(average_price),
                "fillMarkPx": str(mark_price),
            }
        ],
    )


def _complete_lineage() -> dict:
    return {
        "positions_by_id": {
            7: SimpleNamespace(
                model_name="ensemble_trader",
                stop_loss_price=98_000.0,
                take_profit_price=104_000.0,
            )
        },
        "orders_by_exchange_id": {
            "entry-1": SimpleNamespace(
                exchange_order_id="entry-1",
                okx_inst_id="BTC-USDT-SWAP",
                side="buy",
                quantity=0.02,
                price=100_000.0,
                fee=0.4,
                okx_fill_contracts=2.0,
                okx_trade_ids="trade-entry",
                decision_id=91,
                order_type="market",
                okx_raw_fills={
                    "fills_history_confirmed": True,
                    "order_id": "entry-1",
                    "trade_ids": ["trade-entry"],
                    "inst_id": "BTC-USDT-SWAP",
                    "contracts": 2.0,
                    "base_quantity": 0.02,
                    "avg_price": 100_000.0,
                    "fee_abs": 0.4,
                    "contract_size": 0.01,
                    "contract_size_verified": True,
                    "contract_size_source": "okx_public_instruments",
                    "execution_slippage": _slippage_fact(
                        order_id="entry-1",
                        trade_id="trade-entry",
                        side="buy",
                        average_price=100_000.0,
                        mark_price=99_980.0,
                    ),
                    "protection_submission": {
                        "exchange_confirmation_recorded": True,
                        "source_authority": "local_submit_plus_okx_create_order_response",
                        "exchange_confirmed_at": "2026-07-11T01:00:00+00:00",
                    },
                },
            ),
            "close-1": SimpleNamespace(
                exchange_order_id="close-1",
                okx_inst_id="BTC-USDT-SWAP",
                side="sell",
                quantity=0.02,
                price=100_500.0,
                fee=0.6,
                okx_fill_contracts=2.0,
                okx_trade_ids="trade-close",
                decision_id=92,
                order_type="market",
                okx_raw_fills={
                    "fills_history_confirmed": True,
                    "order_id": "close-1",
                    "trade_ids": ["trade-close"],
                    "inst_id": "BTC-USDT-SWAP",
                    "contracts": 2.0,
                    "base_quantity": 0.02,
                    "avg_price": 100_500.0,
                    "fee_abs": 0.6,
                    "contract_size": 0.01,
                    "contract_size_verified": True,
                    "contract_size_source": "okx_public_instruments",
                    "execution_slippage": _slippage_fact(
                        order_id="close-1",
                        trade_id="trade-close",
                        side="sell",
                        average_price=100_500.0,
                        mark_price=100_580.0,
                    ),
                    "protection_execution": {
                        "lifecycle_complete": True,
                        "source_authority": "okx_algo_history_plus_fills_history",
                        "actual_side": "sl",
                        "stop_loss_slippage_pct": 0.1,
                        "stop_loss_slippage_source": ("okx_configured_stop_trigger_to_fills_vwap"),
                    },
                },
            ),
        },
        "decision_raw_by_order_id": {
            "entry-1": {
                "production_trade_gate": {
                    "version": PRODUCTION_TRADE_GATE_VERSION,
                    "can_trade": True,
                    "mode": "live_ml",
                    "decision_authority": "model",
                    "model_can_influence": True,
                },
                "opportunity_score": {"expected_net_return_pct": 0.8},
            }
        },
        "decision_feature_by_order_id": {
            "entry-1": {
                "symbol": "BTC/USDT",
                "returns_1": 0.01,
                "spread_pct": 0.02,
                "horizon_minutes": 60,
            }
        },
    }


def _set_close_fill_price(lineage: dict, price: float) -> None:
    order = lineage["orders_by_exchange_id"]["close-1"]
    order.price = price
    order.okx_raw_fills["avg_price"] = price
    order.okx_raw_fills["execution_slippage"] = _slippage_fact(
        order_id="close-1",
        trade_id="trade-close",
        side="sell",
        average_price=price,
        mark_price=price + 80.0,
    )


def _outcome(sample: dict) -> dict:
    reflection = SimpleNamespace(
        id=501,
        position_id=7,
        source="authoritative_trade_outcome",
        outcome=sample.get("outcome"),
        mistake_summary="fact",
        improvement_summary="recalibrate distribution",
        created_at=datetime(2026, 7, 11, 2, tzinfo=UTC),
    )
    return build_authoritative_trade_outcome(sample, reflection=reflection)


def test_authoritative_okx_lifecycle_builds_one_contract_aware_sample() -> None:
    sample = build_okx_history_training_sample(
        _history(),
        **_complete_lineage(),
    )

    assert sample["source"] == "okx_position_history"
    assert sample["quantity"] == 2.0
    assert sample["quantity_unit"] == "contracts"
    assert sample["notional"] == 2000.0
    assert sample["notional_source"] == ("okx_entry_fill_base_quantity_and_average_price")
    assert sample["gross_return_price_consistent"] is True
    assert sample[PROFIT_TRAINING_TARGET] == pytest.approx(8.5 / 2000.0 * 100.0)
    assert sample["okx_trade_ids"] == ["trade-entry", "trade-close"]
    assert sample["trade_fact_trusted"] is True
    assert sample["features"]["spread_pct"] == pytest.approx(0.02)
    assert sample["decision_timestamp"] == "2026-07-11T01:00:00+00:00"
    assert sample["training_evidence_gaps"] == []
    assert sample["strategy_lineage_complete"] is True
    assert sample["profit_training_contract"]["eligible"] is True
    assert sample["profit_training_contract"]["outcome"] == "profit"
    assert sample["profit_training_contract"]["target_value"] == pytest.approx(8.5 / 2000.0 * 100.0)

    outcome = _outcome(sample)
    label = outcome["training_label_contract"]
    assert label["execution_mode"] == "paper"
    assert label["net_return_after_all_cost_pct"] == pytest.approx(8.5 / 2000.0 * 100.0)
    assert label["realized_net_pnl_usdt"] == 8.5


def test_training_allocates_reversal_entry_order_to_one_lifecycle() -> None:
    boundary_at = datetime(2026, 7, 11, 1, tzinfo=UTC)
    reversal_order_id = "reversal-entry"
    lineage = _complete_lineage()
    entry = lineage["orders_by_exchange_id"].pop("entry-1")
    entry.exchange_order_id = reversal_order_id
    entry.okx_raw_fills["order_id"] = reversal_order_id
    entry.fee = 0.400002
    entry.okx_raw_fills["fee_abs"] = 0.400002
    lineage["orders_by_exchange_id"][reversal_order_id] = entry
    lineage["decision_raw_by_order_id"][reversal_order_id] = lineage[
        "decision_raw_by_order_id"
    ].pop("entry-1")
    lineage["decision_feature_by_order_id"][reversal_order_id] = lineage[
        "decision_feature_by_order_id"
    ].pop("entry-1")
    close = lineage["orders_by_exchange_id"]["close-1"]
    close.quantity = 0.01
    close.fee = 0.3
    close.okx_fill_contracts = 1.0
    close.okx_raw_fills.update(
        {
            "contracts": 1.0,
            "base_quantity": 0.01,
            "fee_abs": 0.3,
            "execution_slippage": _slippage_fact(
                order_id="close-1",
                trade_id="trade-close",
                side="sell",
                average_price=100_500.0,
                mark_price=100_580.0,
                contracts=1.0,
            ),
        }
    )
    history = _history(
        open_max_pos=1.0,
        close_total_pos=1.0,
        fee=-0.5,
        entry_order_ids=[reversal_order_id],
        linked_order_ids=[reversal_order_id, "close-1"],
    )
    history.raw_row = {
        **history.raw_row,
        "openMaxPos": "1",
        "closeTotalPos": "1",
        "fee": "-0.5",
        LIFECYCLE_ORDER_ALLOCATIONS_KEY: build_lifecycle_order_allocation_document(
            entry=[
                build_lifecycle_order_allocation(
                    order_id=reversal_order_id,
                    allocated_contracts=1.0,
                    order_contracts=2.0,
                    boundary_at=boundary_at.isoformat(),
                    peer_history_id=99,
                    peer_role="close",
                )
            ]
        ),
    }

    allocated = build_okx_history_training_sample(history, **lineage)
    unallocated_history = _history(
        open_max_pos=1.0,
        close_total_pos=1.0,
        fee=-0.5,
        entry_order_ids=[reversal_order_id],
        linked_order_ids=[reversal_order_id, "close-1"],
    )
    unallocated_history.raw_row = {
        **history.raw_row,
        "openMaxPos": "1",
        "closeTotalPos": "1",
        "fee": "-0.5",
    }
    unallocated_history.raw_row.pop(LIFECYCLE_ORDER_ALLOCATIONS_KEY, None)
    unallocated = build_okx_history_training_sample(unallocated_history, **lineage)

    assert allocated["fill_contracts"] == pytest.approx(1.0)
    assert allocated["entry_fee"] == pytest.approx(0.2)
    assert allocated["lifecycle_fee_reconciliation"]["applied"] is True
    assert "order_fee_total_mismatch" not in allocated["training_evidence_gaps"]
    assert allocated["lifecycle_order_allocation_failures"] == {
        "entry": {},
        "close": {},
    }
    assert "entry_fill_contracts_history_mismatch" not in allocated["training_evidence_gaps"]
    assert "entry_fill_contracts_history_mismatch" in unallocated["training_evidence_gaps"]


def test_verified_fill_contract_size_overrides_stale_history_spec() -> None:
    history = _history()
    history.raw_row = {
        **history.raw_row,
        "_bb_contract_spec": {
            **history.raw_row["_bb_contract_spec"],
            "ctVal": "1",
        },
    }

    sample = build_okx_history_training_sample(history, **_complete_lineage())

    assert sample["public_or_stored_contract_ct_val"] == pytest.approx(1.0)
    assert sample["contract_ct_val"] == pytest.approx(0.01)
    assert sample["contract_ct_val_source"] == "okx_public_instruments_verified_order_fills"
    assert "entry_fill_contract_quantity_mismatch" not in sample["training_evidence_gaps"]
    assert "close_fill_contract_quantity_mismatch" not in sample["training_evidence_gaps"]


def test_complete_lifecycle_aggregates_multiple_close_orders() -> None:
    lineage = _complete_lineage()
    first_close = lineage["orders_by_exchange_id"]["close-1"]
    first_close.quantity = 0.01
    first_close.fee = 0.3
    first_close.okx_fill_contracts = 1.0
    first_close.okx_trade_ids = "trade-close-1"
    first_close.okx_raw_fills.update(
        {
            "trade_ids": ["trade-close-1"],
            "contracts": 1.0,
            "base_quantity": 0.01,
            "fee_abs": 0.3,
            "execution_slippage": _slippage_fact(
                order_id="close-1",
                trade_id="trade-close-1",
                side="sell",
                average_price=100_500.0,
                mark_price=100_580.0,
                contracts=1.0,
            ),
        }
    )
    lineage["orders_by_exchange_id"]["close-2"] = SimpleNamespace(
        exchange_order_id="close-2",
        okx_inst_id="BTC-USDT-SWAP",
        side="sell",
        quantity=0.01,
        price=100_500.0,
        fee=0.3,
        okx_fill_contracts=1.0,
        okx_fill_pnl=5.0,
        okx_trade_ids="trade-close-2",
        decision_id=93,
        order_type="market",
        okx_raw_fills={
            "fills_history_confirmed": True,
            "order_id": "close-2",
            "trade_ids": ["trade-close-2"],
            "inst_id": "BTC-USDT-SWAP",
            "contracts": 1.0,
            "base_quantity": 0.01,
            "avg_price": 100_500.0,
            "fee_abs": 0.3,
            "fill_pnl": 5.0,
            "contract_size": 0.01,
            "contract_size_verified": True,
            "contract_size_source": "okx_public_instruments",
            "execution_slippage": _slippage_fact(
                order_id="close-2",
                trade_id="trade-close-2",
                side="sell",
                average_price=100_500.0,
                mark_price=100_580.0,
                contracts=1.0,
            ),
        },
    )
    history = _history(
        close_order_ids=["close-1", "close-2"],
        linked_order_ids=["entry-1", "close-1", "close-2"],
    )

    sample = build_okx_history_training_sample(history, **lineage)

    assert sample["close_order_ids"] == ["close-1", "close-2"]
    assert sample["close_order_id"] == "close-2"
    assert sample["close_fee"] == pytest.approx(0.6)
    assert sample["trade_fact_trusted"] is True
    assert sample["profit_training_contract"]["eligible"] is True
    outcome = _outcome(sample)
    assert outcome["training_label_contract"]["close_order_ids"] == [
        "close-1",
        "close-2",
    ]


def test_historical_price_and_fill_pnl_recover_changed_contract_value() -> None:
    history = _history(
        open_avg_px=100.0,
        close_avg_px=101.0,
        open_max_pos=100.0,
        realized_pnl=0.9,
        pnl=1.0,
        pnl_ratio=0.018,
        fee=-0.1,
        funding_fee=0.0,
    )
    history.raw_row = {
        **history.raw_row,
        "openAvgPx": "100",
        "closeAvgPx": "101",
        "realizedPnl": "0.9",
        "pnl": "1",
        "fee": "-0.1",
        "fundingFee": "0",
        "_bb_contract_spec": {
            "ctVal": "1",
            "ctMult": "1",
            "lotSz": "1",
            "source": "okx_public_instruments",
        },
    }
    lineage = _complete_lineage()
    entry = lineage["orders_by_exchange_id"]["entry-1"]
    entry.quantity = 100.0
    entry.price = 100.0
    entry.fee = 0.04
    entry.okx_fill_contracts = 100.0
    entry.okx_raw_fills.update(
        {
            "contracts": 100.0,
            "base_quantity": 100.0,
            "avg_price": 100.0,
            "fee_abs": 0.04,
            "contract_size": 1.0,
            "execution_slippage": _slippage_fact(
                order_id="entry-1",
                trade_id="trade-entry",
                side="buy",
                average_price=100.0,
                mark_price=99.9,
                contracts=100.0,
                contract_size=1.0,
            ),
        }
    )
    close = lineage["orders_by_exchange_id"]["close-1"]
    close.quantity = 100.0
    close.price = 101.0
    close.fee = 0.06
    close.okx_fill_contracts = 100.0
    close.okx_fill_pnl = 1.0
    close.okx_raw_fills.update(
        {
            "contracts": 100.0,
            "base_quantity": 100.0,
            "avg_price": 101.0,
            "fee_abs": 0.06,
            "fill_pnl": 1.0,
            "contract_size": 1.0,
            "execution_slippage": _slippage_fact(
                order_id="close-1",
                trade_id="trade-close",
                side="sell",
                average_price=101.0,
                mark_price=101.1,
                contracts=100.0,
                contract_size=1.0,
            ),
        }
    )

    sample = build_okx_history_training_sample(history, **lineage)

    assert sample["historical_contract_reconciliation"]["applied"] is True
    assert sample["contract_ct_val"] == pytest.approx(0.01)
    assert sample["notional"] == pytest.approx(100.0)
    assert sample["notional_source"] == ("okx_fills_history_pnl_and_position_history_price_path")
    assert sample["gross_return_price_consistent"] is True
    assert sample[PROFIT_TRAINING_TARGET] == pytest.approx(0.9)
    assert sample["trade_fact_trusted"] is True
    assert sample["profit_training_contract"]["eligible"] is True


def test_historical_margin_return_recovers_funding_only_changed_contract_value() -> None:
    history = _history(
        row_identity="paper|YB-USDT-SWAP|yb-pos|net|1785428549448",
        inst_id="YB-USDT-SWAP",
        symbol="YB/USDT",
        pos_id="yb-pos",
        side="short",
        open_avg_px=0.0456,
        close_avg_px=0.0456,
        open_max_pos=1797.0,
        leverage=7.0,
        realized_pnl=-245.1902274,
        pnl=0.0,
        pnl_ratio=-2.094538157894737,
        funding_fee=-244.616625,
        fee=-0.5736024,
    )
    history.raw_row = {
        **history.raw_row,
        "instId": "YB-USDT-SWAP",
        "posId": "yb-pos",
        "posSide": "net",
        "direction": "short",
        "openAvgPx": "0.0456",
        "closeAvgPx": "0.0456",
        "openMaxPos": "1797",
        "closeTotalPos": "1797",
        "lever": "7",
        "realizedPnl": "-245.1902274",
        "pnl": "0",
        "pnlRatio": "-2.094538157894737",
        "fee": "-0.5736024",
        "fundingFee": "-244.616625",
        "_bb_contract_spec": {
            "ctVal": "1",
            "ctMult": "1",
            "lotSz": "1",
            "source": "okx_public_instruments",
        },
    }
    lineage = _complete_lineage()
    entry = lineage["orders_by_exchange_id"]["entry-1"]
    entry.okx_inst_id = "YB-USDT-SWAP"
    entry.side = "sell"
    entry.quantity = 1797.0
    entry.price = 0.0456
    entry.fee = 0.2868
    entry.okx_fill_contracts = 1797.0
    entry.okx_raw_fills.update(
        {
            "inst_id": "YB-USDT-SWAP",
            "contracts": 1797.0,
            "base_quantity": 1797.0,
            "avg_price": 0.0456,
            "fee_abs": 0.2868,
            "contract_size": 1.0,
            "execution_slippage": _slippage_fact(
                order_id="entry-1",
                trade_id="trade-entry",
                side="sell",
                average_price=0.0456,
                mark_price=0.04561,
                contracts=1797.0,
                contract_size=1.0,
                inst_id="YB-USDT-SWAP",
            ),
        }
    )
    close = lineage["orders_by_exchange_id"]["close-1"]
    close.okx_inst_id = "YB-USDT-SWAP"
    close.side = "buy"
    close.quantity = 1797.0
    close.price = 0.0456
    close.fee = 0.2868024
    close.okx_fill_contracts = 1797.0
    close.okx_fill_pnl = 0.0
    close.okx_raw_fills.update(
        {
            "inst_id": "YB-USDT-SWAP",
            "contracts": 1797.0,
            "base_quantity": 1797.0,
            "avg_price": 0.0456,
            "fee_abs": 0.2868024,
            "fill_pnl": 0.0,
            "contract_size": 1.0,
            "execution_slippage": _slippage_fact(
                order_id="close-1",
                trade_id="trade-close",
                side="buy",
                average_price=0.0456,
                mark_price=0.04559,
                contracts=1797.0,
                contract_size=1.0,
                inst_id="YB-USDT-SWAP",
            ),
        }
    )

    sample = build_okx_history_training_sample(
        history,
        **lineage,
        funding_bill_lifecycle_facts={
            "mirror_available": True,
            "bill_count": 2,
            "bill_ids": ["yb-funding-1", "yb-funding-2"],
            "signed_funding_fee_usdt": -244.616625,
            "shared_bill_ids": [],
            "attribution_complete": True,
            "source": "okx_account_bills",
        },
    )

    assert sample["historical_contract_reconciliation"]["applied"] is True
    assert sample["historical_contract_reconciliation"]["price_path_notional"] is None
    assert sample["contract_ct_val"] == pytest.approx(10.0)
    assert sample["notional"] == pytest.approx(819.432)
    assert sample["notional_source"] == ("okx_position_history_realized_pnl_pnl_ratio_and_leverage")
    assert sample[PROFIT_TRAINING_TARGET] == pytest.approx(-29.921973684210524)
    assert sample["gross_return_price_consistent"] is True
    assert sample["funding_evidence_status"] == "verified_extreme_account_bills"
    assert sample["funding_bill_count"] == 2
    assert sample["funding_attribution_complete"] is True
    assert sample["trade_fact_trusted"] is True, sample["training_evidence_gaps"]
    assert sample["profit_training_contract"]["eligible"] is True


def test_extreme_funding_without_bill_reconciliation_is_pending_review() -> None:
    evidence = _funding_training_evidence(
        funding_fee=-244.616625,
        notional=819.432,
        official_funding_present=True,
        bill_facts={
            "mirror_available": True,
            "bill_count": 0,
            "bill_ids": [],
            "signed_funding_fee_usdt": 0.0,
            "shared_bill_ids": [],
            "attribution_complete": True,
        },
    )

    assert evidence["status"] == "pending_review_extreme_missing_account_bills"
    assert evidence["eligible"] is False
    assert evidence["attribution_complete"] is False
    assert evidence["gaps"] == ["extreme_funding_missing_account_bill_reconciliation"]


def test_nonzero_funding_also_requires_account_bill_reconciliation() -> None:
    evidence = _funding_training_evidence(
        funding_fee=-0.5,
        notional=100.0,
        official_funding_present=True,
        bill_facts={
            "mirror_available": True,
            "bill_count": 0,
            "bill_ids": [],
            "signed_funding_fee_usdt": 0.0,
            "shared_bill_ids": [],
            "attribution_complete": True,
        },
    )

    assert evidence["status"] == "pending_review_missing_account_bills"
    assert evidence["eligible"] is False
    assert evidence["attribution_complete"] is False
    assert evidence["gaps"] == ["funding_missing_account_bill_reconciliation"]


def test_non_extreme_funding_with_matching_account_bill_is_verified() -> None:
    evidence = _funding_training_evidence(
        funding_fee=0.5,
        notional=100.0,
        official_funding_present=True,
        bill_facts={
            "mirror_available": True,
            "bill_count": 1,
            "bill_ids": ["funding-1"],
            "signed_funding_fee_usdt": 0.5,
            "shared_bill_ids": [],
            "attribution_complete": True,
        },
    )

    assert evidence["status"] == "verified_account_bills"
    assert evidence["eligible"] is True
    assert evidence["attribution_complete"] is True
    assert evidence["account_bill_ids"] == ["funding-1"]


def test_funding_bill_shared_by_overlapping_lifecycles_is_not_attributed() -> None:
    opened = datetime(2026, 7, 11, 1, tzinfo=UTC)
    histories = [
        _history(
            row_identity=f"paper|BTC-USDT-SWAP|pos-{index}|long|{index}",
            pos_id=f"pos-{index}",
            opened_at=opened + timedelta(minutes=index * 5),
            updated_at_okx=opened + timedelta(hours=1),
        )
        for index in (1, 2)
    ]
    bill = SimpleNamespace(
        id=1,
        mode="paper",
        bill_id="shared-funding",
        inst_id="BTC-USDT-SWAP",
        pos_side="long",
        bill_ts=opened + timedelta(minutes=30),
        funding_fee=-0.5,
    )

    facts = build_funding_bill_lifecycle_facts(histories, [bill])

    assert all(
        row["attribution_complete"] is False and row["shared_bill_ids"] == ["shared-funding"]
        for row in facts.values()
    )


def test_conflicting_price_and_margin_notional_authorities_are_quarantined() -> None:
    history = _history(
        realized_pnl=8.5,
        pnl_ratio=0.017,
    )
    lineage = _complete_lineage()
    lineage["orders_by_exchange_id"]["close-1"].okx_fill_pnl = 10.0

    sample = build_okx_history_training_sample(history, **lineage)

    reconciliation = sample["historical_contract_reconciliation"]
    assert reconciliation["applied"] is False, reconciliation
    assert reconciliation["authority_conflict"] is True
    assert reconciliation["reason"] == "authoritative_historical_notional_sources_conflict"
    assert "historical_contract_notional_authorities_conflict" in sample["training_evidence_gaps"]
    assert sample["trade_fact_trusted"] is False


def test_rules_canary_loss_keeps_rule_authority_and_model_shadow_lesson() -> None:
    history = _history(
        mode="live",
        row_identity="live|BTC-USDT-SWAP|pos-1|long|1",
        close_avg_px=99_650.0,
        realized_pnl=-8.5,
        pnl=-7.0,
        pnl_ratio=-0.0085,
    )
    history.raw_row = {
        **history.raw_row,
        "realizedPnl": "-8.5",
        "pnl": "-7",
        "pnlRatio": "-0.0085",
    }
    lineage = _complete_lineage()
    _set_close_fill_price(lineage, 99_650.0)
    lineage["decision_raw_by_order_id"]["entry-1"] = {
        "production_trade_gate": {
            "version": PRODUCTION_TRADE_GATE_VERSION,
            "can_trade": True,
            "mode": "live_rules_canary",
            "decision_authority": "rules",
            "model_can_influence": False,
        },
        "live_rules_canary_signal": {
            "version": "test-rules-canary-signal",
            "production_eligible": True,
            "decision_authority": "rules",
            "model_can_influence": False,
            "action": "long",
        },
        "model_shadow_decision": {
            "action": "short",
            "confidence": 0.8,
            "observation_only": True,
            "can_authorize_entry": False,
            "can_change_size_or_leverage": False,
        },
    }

    sample = build_okx_history_training_sample(history, **lineage)

    assert sample["decision_authority"] == "rules"
    assert sample["model_shadow_prediction"]["action"] == "short"
    assert sample["model_shadow_prediction"]["rules_execution_action"] == "long"
    contract = sample["profit_training_contract"]
    assert contract["eligible"] is True
    assert contract["outcome"] == "loss"
    assert contract["model_shadow_alignment"] == "avoided_losing_side"


def test_training_rejects_account_derived_contract_size_override() -> None:
    history = _history(pnl=100.0, realized_pnl=98.5, pnl_ratio=None)
    lineage = _complete_lineage()
    entry_order = lineage["orders_by_exchange_id"]["entry-1"]
    entry_order.quantity = 0.2
    entry_order.okx_raw_fills.update(
        {
            "base_quantity": 0.2,
            "contract_size": 0.1,
            "contract_size_verified": True,
            "contract_size_source": ("okx_account_position_margin_notional_crosscheck"),
        }
    )
    sample = build_okx_history_training_sample(history, **lineage)

    assert sample["contract_ct_val"] == pytest.approx(0.01)
    assert sample["contract_ct_val_source"] == "okx_public_instruments"
    assert sample["notional"] == pytest.approx(20_000.0)
    assert "entry_fill_contract_quantity_mismatch" in sample["training_evidence_gaps"]
    assert sample["trade_fact_trusted"] is False


def test_valid_paper_exploration_is_a_normal_trainable_trade_with_selection_reason() -> None:
    provenance = {
        "source": "test_cost_complete_return_distribution",
        "observation_window": "current_test_candidate",
        "sample_count": 3,
        "generated_at": "2026-07-21T00:00:00+00:00",
        "strategy_version": "test.v1",
        "fallback_reason": "",
    }
    selected = {
        "eligible": True,
        "side": "long",
        "expected_net_return_pct": 0.3,
        "return_lcb_pct": -0.1,
        "lcb_gap_ratio": 1.0 / 3.0,
        "loss_probability": 0.3,
        "tail_risk_score": 0.2,
        "return_source_count": 3,
        "historical_evidence_count": 0,
        "validated_route_evidence_count": 0,
        "reliable_evidence_count": 0,
        "exploration_maturity_source": "cold_start",
        "exploration_maturity_evidence": {
            "available": False,
            "source": "validated_continuous_strategy_route",
            "evidence_count": 0,
            "can_authorize_entry": False,
            "can_change_size_or_leverage": False,
        },
        "exploration_allocation_multiplier": 1.0,
        "prediction_horizon_minutes": 30.0,
        "valid_for_seconds": 1800.0,
        "feature_opportunity_score": 8.0,
        "information_value_score": 0.04,
        "policy_provenance": provenance,
    }
    evidence = {
        "preferred_exploration_side": "long",
        "paper_exploration": {
            "preferred_side": "long",
            "selected": selected,
            "reason": "bounded_paper_exploration_side_selected",
        },
    }
    direction_support = assess_directional_entry_support(
        {
            "long": {
                "evidence": [
                    {
                        "source": "local_ml",
                        "decision_eligible": True,
                        "raw_expected_return_pct": 0.3,
                        "objective_expected_return_pct": 0.1,
                        "horizon_minutes": 30,
                    }
                ]
            }
        },
        [
            {
                "model_name": "trend_expert",
                "action": "long",
                "reasoning": "trend supports long",
                "effective_weight": 0.2,
                "source_group": "llm:expert",
            },
            {
                "model_name": "momentum_expert",
                "action": "long",
                "reasoning": "momentum supports long",
                "effective_weight": 0.2,
                "source_group": "llm:expert",
            },
            {
                "model_name": "risk_expert",
                "action": "hold",
                "reasoning": "no hard risk",
                "effective_weight": 0.1,
                "source_group": "llm:expert",
            },
        ],
        "long",
    )
    contract = build_paper_exploration_contract(
        evidence,
        symbol="BTC/USDT",
        independent_direction_support=direction_support,
    )
    lineage = _complete_lineage()
    lineage["decision_raw_by_order_id"]["entry-1"] = {
        "entry_candidate_evidence": evidence,
        "paper_exploration": contract,
    }

    sample = build_okx_history_training_sample(_history(), **lineage)

    assert sample["strategy_entry_supervision_eligible"] is True
    assert sample["strategy_training_role"] == "entry_strategy"
    assert sample["strategy_entry_kind"] == "normal_strategy_trade"
    assert sample["historical_entry_contract_kind"] == "paper_exploration"
    assert sample["strategy_selection_reason"] == ("bounded_paper_exploration_side_selected")
    assert sample["paper_exploration_evidence"]["sample_target"] is None
    assert sample["paper_exploration_evidence"]["daily_sample_quota"] is None


def test_paper_training_loss_is_a_normal_authoritative_training_sample() -> None:
    lineage = _complete_lineage()
    _set_close_fill_price(lineage, 99_650.0)
    lineage["decision_raw_by_order_id"]["entry-1"] = {
        "paper_training": build_paper_training_contract(
            symbol="BTC/USDT",
            selected_side="long",
            signal_source="local_ml_observation",
            expected_net_return_pct=-0.5,
            return_lcb_pct=-0.8,
            horizon_minutes=10.0,
        ),
        "paper_training_mode": "bootstrap",
    }
    history = _history(
        close_avg_px=99_650.0,
        realized_pnl=-8.5,
        pnl=-7.0,
        pnl_ratio=-0.0085,
    )
    history.raw_row = {
        **history.raw_row,
        "realizedPnl": "-8.5",
        "pnl": "-7",
        "pnlRatio": "-0.0085",
    }

    raw_sample = build_okx_history_training_sample(history, **lineage)
    sample = _outcome(raw_sample)
    payload = annotate_training_payload(
        shadow_samples=[],
        trade_samples=[sample],
        sequence_samples=[],
        text_sentiment_samples=[],
    )

    assert sample["strategy_entry_supervision_eligible"] is True
    assert sample["strategy_training_role"] == "entry_strategy"
    assert sample["strategy_entry_kind"] == "normal_strategy_trade"
    assert sample["historical_entry_contract_kind"] == "paper_training"
    assert sample["paper_training_evidence"]["loss_tolerant_for_training"] is True
    assert raw_sample["profit_training_contract"]["eligible"] is True
    assert raw_sample["profit_training_contract"]["outcome"] == "loss"
    assert sample["paper_training_evidence"]["sample_target"] is None
    assert sample["paper_training_evidence"]["daily_sample_quota"] is None
    assert len(payload["trade_samples"]) == 1
    assert payload["trade_samples"][0]["profit_learning_labels"]["realized_net_pnl_usdt"] == -8.5


@pytest.mark.parametrize(
    ("close_price", "realized_pnl", "expected_outcome"),
    [
        (100_500.0, 8.5, "profit"),
        (99_650.0, -8.5, "loss"),
    ],
)
def test_normal_paper_profit_and_loss_are_authoritative_training_samples(
    close_price: float,
    realized_pnl: float,
    expected_outcome: str,
) -> None:
    lineage = _complete_lineage()
    _set_close_fill_price(lineage, close_price)
    contract = build_normal_paper_trade_contract(
        symbol="BTC/USDT",
        side="long",
        selection_reason="strategy_edge_selected",
        direction_support={
            "eligible": True,
            "selected_side": "long",
            "prediction_horizon_minutes": 30.0,
            "expected_net_return_pct": 0.2,
            "objective_net_return_pct": 0.1,
            "loss_probability": 0.3,
            "quant_evidence_families": ["local_ml"],
            "strong_expert_opposition": False,
        },
    )
    lineage["decision_raw_by_order_id"]["entry-1"] = {"normal_paper_trade": contract}
    history = _history(
        close_avg_px=close_price,
        realized_pnl=realized_pnl,
        pnl=10.0 if realized_pnl > 0 else -7.0,
        pnl_ratio=0.0085 if realized_pnl > 0 else -0.0085,
    )
    history.raw_row = {
        **history.raw_row,
        "realizedPnl": str(realized_pnl),
        "pnl": "10" if realized_pnl > 0 else "-7",
        "pnlRatio": "0.0085" if realized_pnl > 0 else "-0.0085",
    }

    sample = build_okx_history_training_sample(history, **lineage)
    outcome = _outcome(sample)
    payload = annotate_training_payload(
        shadow_samples=[],
        trade_samples=[outcome],
        sequence_samples=[],
        text_sentiment_samples=[],
    )

    assert sample["strategy_entry_kind"] == "normal_strategy_trade"
    assert sample["strategy_selection_reason"] == "strategy_edge_selected"
    assert sample["decision_authority"] == "ensemble"
    assert sample["strategy_entry_supervision_eligible"] is True
    assert sample["profit_training_contract"]["eligible"] is True
    assert sample["profit_training_contract"]["outcome"] == expected_outcome
    assert sample["normal_paper_trade_evidence"]["production_permission"] is False
    assert len(payload["trade_samples"]) == 1


def test_historical_normal_paper_v1_is_recovered_without_runtime_authority() -> None:
    lineage = _complete_lineage()
    historical_contract = build_legacy_normal_paper_trade_contract(
        symbol="BTC/USDT",
        side="long",
    )
    lineage["decision_raw_by_order_id"]["entry-1"] = {"normal_paper_trade": historical_contract}

    sample = build_okx_history_training_sample(_history(), **lineage)

    assert normal_paper_trade_contract_reasons(historical_contract)
    assert "invalid_normal_paper_trade_contract" not in sample["training_evidence_gaps"]
    assert sample["decision_authority"] == "ensemble"
    assert sample["strategy_entry_kind"] == "normal_strategy_trade"
    assert sample["historical_entry_contract_kind"] == "normal_paper_v1"
    assert sample["strategy_entry_supervision_eligible"] is True
    assert sample["profit_training_contract"]["eligible"] is True
    assert sample["normal_paper_trade_evidence"]["contract_generation"] == "historical_normal_v1"


def test_v4_negative_objective_contract_remains_historical_training_eligible() -> None:
    lineage = _complete_lineage()
    contract = build_legacy_normal_paper_v4_trade_contract(
        symbol="BTC/USDT",
        side="long",
        objective_net_return_pct=-0.2,
    )
    lineage["decision_raw_by_order_id"]["entry-1"] = {"normal_paper_trade": contract}

    sample = build_okx_history_training_sample(_history(), **lineage)

    assert normal_paper_trade_contract_reasons(contract)
    assert "invalid_normal_paper_trade_contract" not in sample["training_evidence_gaps"]
    assert sample["historical_entry_contract_kind"] == "normal_paper_v4"
    assert sample["strategy_selection_reason"] == "strategy_edge_selected"
    assert sample["normal_paper_trade_evidence"]["contract_generation"] == (
        "historical_expected_net_v4"
    )
    assert sample["profit_training_contract"]["eligible"] is True


def test_historical_normal_wrapper_preserves_legacy_training_identity() -> None:
    lineage = _complete_lineage()
    lineage["decision_raw_by_order_id"]["entry-1"] = {
        "normal_paper_trade": build_legacy_normal_paper_trade_contract(
            symbol="BTC/USDT",
            side="long",
            route_kind="cold_start_exploration",
        ),
        "paper_training": build_paper_training_contract(
            symbol="BTC/USDT",
            selected_side="long",
            signal_source="local_ml_observation",
            horizon_minutes=30.0,
        ),
    }

    sample = build_okx_history_training_sample(_history(), **lineage)

    assert "invalid_normal_paper_trade_contract" not in sample["training_evidence_gaps"]
    assert sample["strategy_entry_kind"] == "normal_strategy_trade"
    assert sample["historical_entry_contract_kind"] == "paper_training"
    assert sample["decision_authority"] == "ensemble"
    assert sample["profit_training_contract"]["eligible"] is True


def test_paper_training_contract_is_never_trainable_as_a_live_trade() -> None:
    lineage = _complete_lineage()
    lineage["decision_raw_by_order_id"]["entry-1"] = {
        "paper_training": build_paper_training_contract(
            symbol="BTC/USDT",
            selected_side="long",
            signal_source="local_ml_observation",
            horizon_minutes=10.0,
        )
    }

    sample = build_okx_history_training_sample(
        _history(mode="live"),
        **lineage,
    )

    assert sample["strategy_entry_supervision_eligible"] is False
    assert sample["strategy_training_role"] == "invalid_paper_training_research_only"
    assert "invalid_paper_training_contract" in sample["training_evidence_gaps"]


def test_stale_contract_multiplier_is_quarantined_without_pnl_notional_fallback() -> None:
    history = _history(
        inst_id="LIT-USDT-SWAP",
        symbol="LIT/USDT",
        open_avg_px=2.5,
        close_avg_px=2.35,
        realized_pnl=-4.55,
        pnl=-4.5,
        fee=-0.05,
        funding_fee=0.0,
        pnl_ratio=-0.0606666667,
    )
    history.raw_row = {
        **history.raw_row,
        "instId": "LIT-USDT-SWAP",
        "realizedPnl": "-4.55",
        "pnl": "-4.5",
        "fee": "-0.05",
        "fundingFee": "0",
        "pnlRatio": "-0.0606666667",
        "_bb_contract_spec": {"ctVal": "1", "ctMult": "1", "lotSz": "1"},
    }
    lineage = _complete_lineage()
    entry_order = lineage["orders_by_exchange_id"]["entry-1"]
    entry_order.okx_fill_contracts = 3.0
    entry_order.quantity = 30.0
    entry_order.price = 2.5
    entry_order.fee = 0.02
    entry_order.okx_raw_fills.update(
        {
            "contracts": 3.0,
            "base_quantity": 30.0,
            "avg_price": 2.5,
            "fee_abs": 0.02,
        }
    )
    close_order = lineage["orders_by_exchange_id"]["close-1"]
    close_order.okx_fill_contracts = 3.0
    close_order.quantity = 30.0
    close_order.price = 2.35
    close_order.fee = 0.03
    close_order.okx_raw_fills.update(
        {
            "contracts": 3.0,
            "base_quantity": 30.0,
            "avg_price": 2.35,
            "fee_abs": 0.03,
        }
    )

    sample = build_okx_history_training_sample(history, **lineage)

    assert sample["notional"] is None
    assert sample["notional_source"] == ""
    assert sample["gross_price_return_pct"] == pytest.approx(-6.0)
    assert sample["gross_return_on_notional_pct"] is None
    assert sample["gross_return_price_consistent"] is False
    assert "missing_authoritative_entry_fill_facts" in sample["training_evidence_gaps"]
    assert PROFIT_TRAINING_TARGET not in sample
    assert sample["profit_training_contract"]["eligible"] is False


def test_multiple_entry_decisions_are_quarantined_from_strategy_training() -> None:
    lineage = _complete_lineage()
    lineage["orders_by_exchange_id"]["entry-2"] = SimpleNamespace(
        okx_fill_contracts=1.0,
        okx_trade_ids="trade-entry-2",
        decision_id=93,
    )
    history = _history(entry_order_ids=["entry-1", "entry-2"])

    sample = _outcome(build_okx_history_training_sample(history, **lineage))
    payload = annotate_training_payload(
        shadow_samples=[],
        trade_samples=[sample],
        sequence_samples=[],
        text_sentiment_samples=[],
    )

    assert sample["entry_decision_ids"] == [91, 93]
    assert sample["decision_id"] == 0
    assert sample["strategy_training_role"] == "aggregate_position_research_only"
    assert "multiple_entry_decision_lineage" in sample["training_evidence_gaps"]
    assert payload["trade_samples"] == []


def test_obsolete_sampling_entry_is_research_only() -> None:
    lineage = _complete_lineage()
    lineage["decision_raw_by_order_id"]["entry-1"] = {
        "paper_bootstrap_canary": {
            "trade_kind": "observation_only_probe",
            "continuous_training_after_settlement": False,
        }
    }

    sample = _outcome(build_okx_history_training_sample(_history(), **lineage))
    payload = annotate_training_payload(
        shadow_samples=[],
        trade_samples=[sample],
        sequence_samples=[],
        text_sentiment_samples=[],
    )

    assert sample["strategy_training_role"] == "obsolete_sampling_research_only"
    assert "obsolete_sampling_entry_not_strategy_trainable" in sample["training_evidence_gaps"]
    assert payload["trade_samples"] == []


def test_okx_demo_alias_normalizes_to_paper_and_invalid_mode_is_quarantined() -> None:
    demo = build_okx_history_training_sample(
        _history(mode="demo"),
        **_complete_lineage(),
    )
    invalid = build_okx_history_training_sample(
        _history(mode="unknown"),
        **_complete_lineage(),
    )

    assert demo["execution_mode"] == "paper"
    assert demo["source_execution_mode"] == "demo"
    assert "missing_or_invalid_execution_mode" not in demo["training_evidence_gaps"]
    assert invalid["execution_mode"] == ""
    assert "missing_or_invalid_execution_mode" in invalid["training_evidence_gaps"]


def test_authoritative_sample_uses_exact_entry_order_decision_evidence() -> None:
    entry = SimpleNamespace(
        okx_fill_contracts=2.0,
        okx_trade_ids="trade-entry",
        decision_id=91,
    )
    raw = {"local_ai_tools": {"time_series_prediction": {"model": "timesfm"}}}

    sample = build_okx_history_training_sample(
        _history(position_ids=[7]),
        orders_by_exchange_id={"entry-1": entry},
        decision_raw_by_position_id={7: {"local_ai_tools": {"wrong": True}}},
        decision_raw_by_order_id={"entry-1": raw},
    )

    assert sample["decision_id"] == 91
    assert sample["raw_llm_response"] == raw


def test_exact_entry_decision_recovers_missing_planned_protection_prices() -> None:
    lineage = _complete_lineage()
    lineage["positions_by_id"][7].stop_loss_price = None
    lineage["positions_by_id"][7].take_profit_price = None
    lineage["decision_execution_by_order_id"] = {
        "entry-1": {
            "decision_id": 91,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
        }
    }

    sample = build_okx_history_training_sample(_history(), **lineage)

    assert sample["planned_stop_loss_price"] == pytest.approx(98_000.0)
    assert sample["planned_take_profit_price"] == pytest.approx(104_000.0)
    assert "missing_planned_stop_loss_lineage" not in sample["strategy_lineage_gaps"]
    assert "missing_planned_take_profit_lineage" not in sample["strategy_lineage_gaps"]


def test_missing_official_funding_and_contract_spec_are_quarantined_with_reasons() -> None:
    history = _history()
    history.raw_row = {
        key: value
        for key, value in history.raw_row.items()
        if key not in {"fundingFee", "_bb_contract_spec"}
    }

    sample = _outcome(build_okx_history_training_sample(history))
    payload = annotate_training_payload(
        shadow_samples=[],
        trade_samples=[sample],
        sequence_samples=[],
        text_sentiment_samples=[],
    )

    assert "missing_official_funding_fee" in sample["training_evidence_gaps"]
    assert "missing_contract_ct_val" in sample["training_evidence_gaps"]
    assert payload["trade_samples"] == []
    reasons = {item["reason"] for item in payload["quality_report"]["top_reasons"]}
    assert "trade:incomplete_okx_lifecycle:missing_official_funding_fee" in reasons


def test_training_report_blocks_pnl_return_sign_mismatch() -> None:
    sample = _outcome(build_okx_history_training_sample(_history(), **_complete_lineage()))
    sample[PROFIT_TRAINING_TARGET] = -8.5
    payload = annotate_training_payload(
        shadow_samples=[],
        trade_samples=[sample],
        sequence_samples=[],
        text_sentiment_samples=[],
    )

    consistency = payload["quality_report"]["training_label_consistency"]
    assert consistency["status"] == "blocked"
    assert consistency["promotion_blocked"] is True
    assert consistency["errors"][0]["reason"] == "pnl_return_sign_mismatch"


def test_authoritative_loss_with_exact_entry_lineage_remains_supervision_ready() -> None:
    history = _history(
        close_avg_px=99_650.0,
        realized_pnl=-8.5,
        pnl=-7.0,
        pnl_ratio=-0.0085,
    )
    history.raw_row = {
        **history.raw_row,
        "realizedPnl": "-8.5",
        "pnl": "-7",
        "pnlRatio": "-0.0085",
    }
    lineage = _complete_lineage()
    _set_close_fill_price(lineage, 99_650.0)
    sample = _outcome(build_okx_history_training_sample(history, **lineage))

    payload = annotate_training_payload(
        shadow_samples=[],
        trade_samples=[sample],
        sequence_samples=[],
        text_sentiment_samples=[],
    )

    assert len(payload["trade_samples"]) == 1
    trade = payload["trade_samples"][0]
    assert trade["data_quality_status"] == "included"
    labels = trade["profit_learning_labels"]
    assert labels["training_supervision_ready"] is True
    assert labels["exit_attribution_supervision_ready"] is True
    assert labels["losing_exit_attribution"] == "authoritative_multi_factor_outcome"
    assert labels["realized_net_pnl_usdt"] == -8.5


def test_entry_order_decision_id_is_preserved_when_raw_payload_is_empty() -> None:
    sample = build_okx_history_training_sample(
        _history(),
        orders_by_exchange_id={
            "entry-1": SimpleNamespace(decision_id=91, okx_fill_contracts=2.0),
            "close-1": SimpleNamespace(decision_id=92),
        },
    )

    assert sample["decision_id"] == 91
    assert sample["decision_lineage_source"] == "exact_entry_order_decision_id"
    assert "missing_exact_entry_order_decision_link" not in sample["strategy_lineage_gaps"]
    assert "missing_exact_entry_order_decision_payload" in sample["strategy_lineage_gaps"]


def test_position_fallback_payload_is_not_misreported_as_exact_entry_lineage() -> None:
    sample = build_okx_history_training_sample(
        _history(),
        decision_raw_by_position_id={7: {"opportunity_score": {"score": 1.0}}},
    )

    assert sample["decision_id"] == 0
    assert sample["decision_lineage_source"] == "position_time_fallback_payload"
    assert "missing_exact_entry_order_decision_link" in sample["strategy_lineage_gaps"]


def test_round_trip_slippage_uses_fill_mark_facts_not_protection_trigger() -> None:
    lineage = _complete_lineage()
    _set_close_fill_price(lineage, 97_000.0)
    lineage["orders_by_exchange_id"]["entry-1"].okx_raw_fills["protection_submission"] = {
        "source_authority": "local_submit_plus_okx_create_order_response",
        "exchange_confirmation_recorded": True,
        "exchange_confirmed_at": "2026-07-11T01:00:01+00:00",
        "algo_ids": ["algo-stop-1"],
    }
    lineage["orders_by_exchange_id"]["close-1"].okx_raw_fills["protection_execution"] = {
        "source_authority": "okx_algo_history_plus_fills_history",
        "lifecycle_complete": True,
        "algo_id": "algo-stop-1",
        "generated_order_id": "close-1",
        "actual_side": "sl",
        "configured_trigger_price": 97_500.0,
        "actual_trigger_market_price": None,
        "actual_trigger_market_price_available": False,
        "exchange_confirmed_at_ms": 1783731601000,
        "triggered_at_ms": 1783735200000,
        "fill_started_at_ms": 1783735200025,
        "fill_completed_at_ms": 1783735200030,
        "trigger_to_first_fill_ms": 25.0,
        "fill_mark_price": 97_450.0,
        "fill_index_price": 97_460.0,
        "fill_path_min_price": 96_950.0,
        "fill_path_max_price": 97_100.0,
        "fill_mark_slippage_pct": 0.461775,
        "trigger_path_extrema_available": False,
        "trigger_orderbook_snapshot_available": False,
        "stop_loss_slippage_pct": (97_500.0 - 97_000.0) / 97_500.0 * 100.0,
        "stop_loss_slippage_source": "okx_configured_stop_trigger_to_fills_vwap",
    }
    lineage["decision_raw_by_order_id"]["entry-1"] = {
        "profit_risk_sizing": {
            "risk_budget_usdt": 5.0,
            "planned_stressed_loss_usdt": 4.5,
        }
    }
    history = _history(
        close_avg_px=97_000.0,
        realized_pnl=-8.5,
        pnl=-7.0,
        pnl_ratio=-0.0085,
    )
    history.raw_row = {
        **history.raw_row,
        "realizedPnl": "-8.5",
        "pnl": "-7",
        "pnlRatio": "-0.0085",
    }

    sample = build_okx_history_training_sample(history, **lineage)

    assert sample["stop_loss_fill_confirmed"] is True
    assert sample["slippage"] == pytest.approx((0.4 + 1.6) / 2_000.0 * 100.0)
    assert sample["slippage"] != pytest.approx((97_500.0 - 97_000.0) / 97_500.0 * 100.0)
    assert sample["slippage_source"] == OKX_ROUND_TRIP_SLIPPAGE_SOURCE
    assert sample["execution_slippage_usdt"] == pytest.approx(2.0)
    assert sample["actual_trigger_market_price"] is None
    assert sample["protection_lifecycle_complete"] is True
    assert sample["trigger_to_first_fill_ms"] == pytest.approx(25.0)
    assert sample["execution_actual_over_budget_loss_usdt"] == pytest.approx(3.5)
    assert "actual_trigger_market_price_unavailable" in sample["protection_execution_gaps"]


def test_protection_execution_is_not_required_for_round_trip_slippage() -> None:
    lineage = _complete_lineage()
    _set_close_fill_price(lineage, 97_000.0)
    lineage["orders_by_exchange_id"]["close-1"].okx_raw_fills.pop("protection_execution")

    sample = build_okx_history_training_sample(
        _history(close_avg_px=97_000.0),
        **lineage,
    )

    assert sample["stop_loss_fill_confirmed"] is False
    assert sample["slippage"] == pytest.approx(0.1)
    assert sample["slippage_source"] == OKX_ROUND_TRIP_SLIPPAGE_SOURCE
    assert sample["profit_training_contract"]["eligible"] is True


def test_missing_fill_mark_fact_cannot_create_slippage_label() -> None:
    lineage = _complete_lineage()
    lineage["orders_by_exchange_id"]["close-1"].okx_raw_fills.pop("execution_slippage")

    sample = build_okx_history_training_sample(_history(), **lineage)

    assert sample["slippage"] is None
    assert sample["slippage_source"] == ""
    assert sample["close_execution_slippage_complete"] is False
    assert sample["execution_slippage_failures"] == {
        "entry": {},
        "close": {"close-1": ["stored_slippage:fact_missing:fills_history"]},
    }
    assert sample["profit_training_contract"]["eligible"] is False
    assert "slippage_missing_or_invalid" in sample["profit_training_contract"]["blockers"]


def test_legacy_fill_aliases_and_local_contract_count_cannot_authorize_training() -> None:
    lineage = _complete_lineage()
    entry = lineage["orders_by_exchange_id"]["entry-1"]
    raw = entry.okx_raw_fills
    raw["filled_base_quantity"] = raw.pop("base_quantity")
    raw["average"] = raw.pop("avg_price")
    raw["filled_contracts"] = raw.pop("contracts")
    entry.okx_fill_contracts = 2.0

    sample = build_okx_history_training_sample(_history(), **lineage)

    assert sample["entry_fee"] is None
    assert sample["notional"] is None
    assert sample["fill_contracts"] is None
    assert "missing_authoritative_entry_fill_facts" in sample["training_evidence_gaps"]
    assert sample["execution_slippage_failures"]["entry"] == {
        "entry-1": ["authoritative_fill_fact_missing"]
    }


def test_execution_result_fill_fact_cannot_authorize_training() -> None:
    lineage = _complete_lineage()
    entry = lineage["orders_by_exchange_id"]["entry-1"]
    entry.okx_raw_fills.update(
        {
            "fills_history_confirmed": False,
            "execution_result_confirmed": True,
            "source": "okx_execution_result",
        }
    )

    sample = build_okx_history_training_sample(_history(), **lineage)

    assert sample["entry_fee"] is None
    assert sample["notional"] is None
    assert "missing_authoritative_entry_fill_facts" in sample["training_evidence_gaps"]
