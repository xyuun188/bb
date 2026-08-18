from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from services.current_position_management import (
    CURRENT_POSITION_MANAGEMENT_VERSION,
    build_current_position_management_contract,
    current_position_management_contract_complete,
)


def _facts(**overrides):
    values = {
        "symbol": "BTC/USDT",
        "side": "long",
        "quantity": 2.0,
        "contracts": 20.0,
        "entry_price": 100.0,
        "current_price": 103.0,
        "unrealized_pnl_usdt": 6.0,
        "entry_fee_usdt": 0.12,
        "full_entry_fee_usdt": 0.24,
        "full_entry_notional_usdt": 400.0,
        "entry_fee_evidence_complete": True,
        "entry_fee_source": "okx_fills_history",
        "stop_loss_price": 98.0,
        "take_profit_price": 110.0,
        "protection_evidence_complete": True,
        "protection_orders": [
            {
                "algo_id": "oco-1",
                "state": "live",
                "contracts": 20.0,
                "reduce_only": True,
                "stop_loss_price": 98.0,
                "take_profit_price": 110.0,
            }
        ],
        "position_stressed_loss_usdt": 4.0,
        "portfolio_stressed_loss_usdt": 10.0,
        "portfolio_gross_notional_usdt": 500.0,
        "account_equity_usdt": 1_000.0,
        "open_position_count": 2,
        "entry_order_ids": ["entry-1"],
        "entry_decision_ids": [99],
        "original_entry_contract_complete": False,
        "original_entry_contract_gaps": ["decision_99:risk_contract_missing"],
    }
    values.update(overrides)
    return values


def test_takeover_contract_is_reduce_only_and_preserves_historical_gap() -> None:
    contract = build_current_position_management_contract(_facts())

    assert contract["management_eligible"] is True
    assert contract["can_expand_position"] is False
    assert contract["can_increase_leverage"] is False
    assert contract["replaces_original_entry_contract"] is False
    assert contract["original_entry_contract_status"] == (
        "historical_entry_contract_incomplete_preserved"
    )
    assert contract["original_entry_contract_gaps"] == ["decision_99:risk_contract_missing"]
    assert contract["entry_fee_usdt"] == pytest.approx(0.12)
    assert contract["funding_fee_usdt"] == 0.0
    assert contract["funding_bill_count"] == 0
    assert contract["funding_evidence_complete"] is True
    assert contract["funding_evidence_eligible"] is True
    assert contract["funding_fee_included"] is True
    assert contract["settled_funding_fee"] == 0.0
    assert contract["expected_future_funding_cashflow"] == 0.0
    assert contract["current_lifecycle_net_pnl"] == pytest.approx(
        6.0 - 0.12 - (206.0 * 0.0006)
    )
    assert contract["projected_hold_net_pnl"] == pytest.approx(
        contract["current_lifecycle_net_pnl"]
    )
    assert contract["funding_evidence_status"] == (
        "settled_complete_future_unavailable"
    )
    assert contract["exit_fee_rate_proxy"] == pytest.approx(0.0006)
    assert contract["policy_provenance"]["contract_fingerprint"]


def test_takeover_timestamp_is_stable_while_current_facts_refresh() -> None:
    first_at = datetime(2026, 7, 15, 1, 0, tzinfo=UTC)
    first = build_current_position_management_contract(_facts(), now=first_at)
    second = build_current_position_management_contract(
        _facts(current_price=104.0),
        previous_contract=first,
        now=first_at + timedelta(minutes=5),
    )

    assert second["takeover_at"] == first["takeover_at"]
    assert second["refreshed_at"] != first["refreshed_at"]
    assert second["policy_provenance"]["input_fingerprint"] != (
        first["policy_provenance"]["input_fingerprint"]
    )


def test_takeover_refresh_preserves_persisted_paper_canary_lifecycle() -> None:
    lifecycle = {
        "version": "2026-07-19.paper-bootstrap-position-lifecycle.v1",
        "kind": "paper_bootstrap_canary_position",
        "authorized": True,
        "execution_scope": "paper_only",
        "production_permission": False,
        "symbol": "BTC/USDT",
        "side": "long",
        "horizon_minutes": 10,
        "expires_at": "2026-07-15T01:10:00+00:00",
    }

    contract = build_current_position_management_contract(
        _facts(current_price=104.0),
        previous_contract={"paper_canary_lifecycle": lifecycle},
    )

    assert contract["paper_canary_lifecycle"] == lifecycle


def test_takeover_refresh_preserves_paper_training_lifecycle() -> None:
    lifecycle = {
        "version": "2026-07-22.paper-training-position-lifecycle.v1",
        "kind": "normal_paper_training_position",
        "decision_id": 104168,
    }

    contract = build_current_position_management_contract(
        _facts(current_price=104.0),
        previous_contract={"paper_training_lifecycle": lifecycle},
    )

    assert contract["paper_training_lifecycle"] == lifecycle


def test_zero_actual_fee_is_complete_when_okx_fee_evidence_is_explicit() -> None:
    contract = build_current_position_management_contract(
        _facts(entry_fee_usdt=0.0, full_entry_fee_usdt=0.0)
    )
    position = SimpleNamespace(
        symbol="BTC/USDT",
        side="long",
        quantity=2.0,
        entry_price=100.0,
        entry_fee=0.0,
        stop_loss_price=98.0,
        take_profit_price=110.0,
        current_management_contract=contract,
    )

    assert contract["contract_version"] == CURRENT_POSITION_MANAGEMENT_VERSION
    assert contract["management_eligible"] is True
    assert current_position_management_contract_complete(position) is True


def test_takeover_contract_fails_closed_without_current_oco_or_account_equity() -> None:
    contract = build_current_position_management_contract(
        _facts(
            protection_evidence_complete=False,
            account_equity_usdt=0.0,
        )
    )

    assert contract["management_eligible"] is False
    assert "okx_protection_evidence_incomplete" in contract["blockers"]
    assert "current_account_equity_missing" in contract["blockers"]


def test_takeover_contract_uses_authoritative_notional_and_blocks_identity_gap() -> None:
    contract = build_current_position_management_contract(
        _facts(
            quantity=3949.0,
            current_price=376.6,
            contracts=3949.0,
            contract_size=1.0,
            position_notional_usdt=0.02309 * 3949.0,
            position_notional_source="okx_notional_usd",
            exchange_identity_gaps=["okx_private_position_underlying_differs_from_public_instrument"],
            funding_fee_usdt=2030.93,
            funding_bill_count=4,
            funding_fee_source="okx_account_bills",
            funding_evidence_complete=True,
            funding_evidence_eligible=True,
        )
    )

    assert contract["position_notional_usdt"] == pytest.approx(0.02309 * 3949.0)
    assert contract["position_notional_source"] == "okx_notional_usd"
    assert contract["management_eligible"] is False
    assert contract["funding_fee_usdt"] == pytest.approx(2030.93)
    assert contract["funding_evidence_eligible"] is False
    assert "okx_private_position_underlying_differs_from_public_instrument" in contract["blockers"]


def test_takeover_contract_keeps_settled_and_future_funding_separate() -> None:
    contract = build_current_position_management_contract(
        _facts(
            position_notional_usdt=206.0,
            funding_fee_usdt=10.0,
            funding_bill_count=2,
            funding_fee_source="okx_account_bills",
            funding_evidence_complete=True,
            funding_evidence_eligible=True,
            expected_future_funding_cashflow=-1.5,
            expected_future_price_pnl_usdt=2.0,
            future_funding_evidence_status="complete",
            next_funding_time="2026-08-17T08:00:00+00:00",
            estimated_exit_slippage_usdt=0.2,
        )
    )

    expected_current = 6.0 + 10.0 - 0.12 - (206.0 * 0.0006) - 0.2
    assert contract["settled_funding_fee"] == pytest.approx(10.0)
    assert contract["expected_future_funding_cashflow"] == pytest.approx(-1.5)
    assert contract["current_lifecycle_net_pnl"] == pytest.approx(expected_current)
    assert contract["projected_hold_net_pnl"] == pytest.approx(
        expected_current + 2.0 - 1.5
    )
    assert contract["funding_fee_included"] is True
    assert contract["funding_evidence_status"] == "complete"


def test_split_oco_validates_by_exact_inventory_not_single_position_price() -> None:
    contract = build_current_position_management_contract(
        _facts(
            contracts=13.0,
            side="short",
            quantity=1_300.0,
            entry_price=0.0133,
            current_price=0.0127,
            entry_fee_usdt=0.01,
            full_entry_fee_usdt=0.01,
            full_entry_notional_usdt=17.29,
            stop_loss_price=0.01395,
            take_profit_price=0.0118,
            protection_orders=[
                {
                    "algo_id": "oco-a",
                    "state": "live",
                    "contracts": 7.0,
                    "reduce_only": True,
                    "stop_loss_price": 0.0139,
                    "take_profit_price": 0.0116,
                },
                {
                    "algo_id": "oco-b",
                    "state": "live",
                    "contracts": 6.0,
                    "reduce_only": True,
                    "stop_loss_price": 0.0140,
                    "take_profit_price": 0.0120,
                },
            ],
            position_stressed_loss_usdt=0.845,
        )
    )
    position = SimpleNamespace(
        symbol="BTC/USDT",
        side="short",
        quantity=1_300.0,
        entry_price=0.0133,
        entry_fee=0.01,
        stop_loss_price=0.0140,
        take_profit_price=0.0120,
        current_management_contract=contract,
    )

    assert current_position_management_contract_complete(position) is True
