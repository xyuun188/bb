from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest

from scripts.audit_profitability_economics import (
    fetch_bills,
    summarize_bills,
    summarize_lifecycles,
)

SINCE = datetime(2026, 10, 21, tzinfo=UTC)


def outcome(key="one", **changes):
    row = {
        "event_type": "AuthoritativeTradeOutcome",
        "lifecycle_key": key,
        "outcome_fingerprint": key,
        "close_status": "full",
        "decision_timestamp": (SINCE - timedelta(hours=2)).isoformat(),
        "holding_minutes": 120,
        "normal_paper_trade_evidence": {"prediction_horizon_minutes": 5},
        "realized_net_pnl_components": {
            "gross_pnl_usdt": 1,
            "official_fee_signed_usdt": -2,
            "funding_fee_usdt": 0.25,
            "liquidation_penalty_usdt": 0,
            "reported_realized_net_pnl_usdt": -0.75,
            "formula_consistent": True,
        },
        "execution_slippage_usdt": 0.4,
        "outcome_complete": True,
        "outcome_evidence_gaps": [],
    }
    return row | changes


def bill(key="one", **changes):
    return {
        "billId": key, "ts": str(int(SINCE.timestamp() * 1000)),
        "type": "2", "subType": "1", "ccy": "USDT", "pnl": "1",
        "fee": "-0.2", "balChg": "0.8", "instId": "BTC-USDT-SWAP",
    } | changes


def test_partial_updates_do_not_enter_full_round_trip_totals():
    result = summarize_lifecycles(
        [outcome(), outcome("partial", close_status="partial")], since=SINCE
    )
    assert result["fully_closed_count"] == 1
    assert result["partial_or_unknown_count"] == 1
    assert result["full_lifecycle_totals"]["realized_net_pnl_usdt"] == -0.75
    assert result["full_lifecycles_opened_before_window_count"] == 1


def test_slippage_is_measured_but_never_deducted_twice_and_horizon_is_diagnostic():
    result = summarize_lifecycles([outcome()], since=SINCE)
    assert result["full_lifecycle_totals"]["realized_net_pnl_usdt"] == -0.75
    assert result["positive_gross_but_negative_net_count"] == 1
    assert result["measured_slippage"]["signed_execution_slippage_usdt"] == 0.4
    assert result["measured_slippage"]["subtracted_again_from_realized_pnl"] is False
    assert result["holding_vs_forecast_horizon"]["maximum_ratio"] == 24
    assert result["holding_vs_forecast_horizon"]["causal_exit_loss_inferred"] is False


def test_missing_cost_or_broken_settlement_is_excluded_instead_of_imputed():
    missing = outcome("missing")
    missing["realized_net_pnl_components"]["official_fee_signed_usdt"] = None
    broken = outcome("broken")
    broken["realized_net_pnl_components"]["formula_consistent"] = False
    result = summarize_lifecycles([outcome(), missing, broken], since=SINCE)
    assert result["full_excluded_from_totals_count"] == 2
    assert result["full_lifecycle_totals"]["realized_net_pnl_usdt"] == -0.75


def test_duplicate_lifecycles_are_deduplicated_but_conflicts_fail_closed():
    one = outcome()
    result = summarize_lifecycles([one, deepcopy(one)], since=SINCE)
    assert result["identical_duplicate_count"] == 1
    assert result["unique_lifecycle_count"] == 1
    with pytest.raises(ValueError, match="conflicting"):
        summarize_lifecycles([one, outcome(holding_minutes=6)], since=SINCE)


def test_bill_funding_is_not_counted_again_as_price_pnl():
    result = summarize_bills([
        bill(),
        bill("funding", type="8", subType="173", pnl="0.3", fee="0", balChg="0.3"),
        bill("rebate", pnl="0", fee="0.1", balChg="0.1"),
    ], outcomes=[])
    assert result["trade_and_funding_cash_change_usdt"] == pytest.approx(1.2)
    assert result["by_kind"]["funding"]["price_pnl_usdt"] == 0
    assert result["by_kind"]["funding"]["funding_signed_usdt"] == 0.3
    assert result["by_kind"]["trade"]["fees_signed_usdt"] == pytest.approx(-0.1)
    assert result["by_kind"]["trade"]["unexplained_cash_residual_usdt"] == pytest.approx(0)


def test_bill_duplicates_and_unrelated_transfers_cannot_inflate_trading_pnl():
    one = bill(ordId="entry")
    result = summarize_bills(
        [one, deepcopy(one), bill("transfer", type="1", pnl="0", fee="0", balChg="100")],
        outcomes=[outcome(entry_order_ids=["entry"])],
    )
    assert result["unique_bill_count"] == 2
    assert result["identical_duplicate_count"] == 1
    assert result["trade_and_funding_cash_change_usdt"] == 0.8
    assert result["bills_with_order_in_sampled_lifecycles"] == 1
    assert result["by_kind"]["other"]["cash_change_usdt"] == 100
    with pytest.raises(ValueError, match="conflicting"):
        summarize_bills([one, one | {"fee": "-10"}], outcomes=[])


def test_unexplained_bill_cash_change_remains_visible():
    result = summarize_bills([bill(balChg="2")], outcomes=[])
    assert result["by_kind"]["trade"]["unexplained_cash_residual_usdt"] == pytest.approx(1.2)


class Exchange:
    def __init__(self, pages):
        self.pages = iter(pages)
        self.calls = []

    async def privateGetAccountBills(self, params):
        self.calls.append(params)
        return next(self.pages)


def page(rows):
    return {"code": "0", "data": rows}


@pytest.mark.asyncio
async def test_bills_use_exclusive_cutoff_and_prove_history_boundary():
    end = SINCE + timedelta(hours=24)
    exchange = Exchange([page([
        bill("at-cutoff", ts=str(int(end.timestamp() * 1000))),
        bill("at-start"),
        bill("before", ts=str(int((SINCE - timedelta(seconds=1)).timestamp() * 1000))),
    ])])
    rows, coverage = await fetch_bills(exchange, since=SINCE, until=end, max_pages=2)
    assert [row["billId"] for row in rows] == ["at-start"]
    assert coverage == {"complete": True, "pages": 1, "reason": "window_boundary_crossed"}


@pytest.mark.asyncio
async def test_full_page_exhaustion_never_claims_complete_window():
    exchange = Exchange([page([bill(str(index)) for index in range(100)])])
    rows, coverage = await fetch_bills(
        exchange, since=SINCE, until=SINCE + timedelta(days=1), max_pages=1
    )
    assert len(rows) == 100
    assert coverage["complete"] is False
    assert coverage["reason"] == "page_budget_exhausted"


@pytest.mark.asyncio
async def test_failed_official_response_is_not_an_empty_success():
    exchange = Exchange([{"code": "50011", "data": []}])
    with pytest.raises(ValueError, match="request failed"):
        await fetch_bills(exchange, since=SINCE, until=SINCE + timedelta(days=1), max_pages=2)


@pytest.mark.asyncio
async def test_repeated_cursor_stops_with_incomplete_status():
    repeated = page([bill(str(index)) for index in range(100)])
    exchange = Exchange([repeated, repeated])
    _, coverage = await fetch_bills(
        exchange, since=SINCE, until=SINCE + timedelta(days=1), max_pages=3
    )
    assert coverage["reason"] == "cursor_stalled"
    assert coverage["complete"] is False
