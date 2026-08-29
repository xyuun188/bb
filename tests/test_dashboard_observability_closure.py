import pytest

from web_dashboard.api import dashboard


@pytest.mark.asyncio
async def test_authoritative_profit_observability_excludes_incomplete_and_shadow(monkeypatch):
    async def fake_load(**_kwargs):
        return [
            {
                "outcome_complete": True,
                "trade_fact_trusted": True,
                "gross_pnl_usdt": 10.0,
                "entry_fee_usdt": 1.0,
                "close_fee_usdt": 1.0,
                "execution_slippage_usdt": 0.5,
                "funding_fee_usdt": 0.2,
                "liquidation_penalty_usdt": 0.0,
                "realized_net_pnl_usdt": 8.2,
                "realized_net_pnl_components": {
                    "components_total_usdt": 8.2,
                    "reported_realized_net_pnl_usdt": 8.2,
                },
                "counterfactual_evidence": [{"production_weight": 0.0}],
            },
            {
                "outcome_complete": False,
                "trade_fact_trusted": False,
                "realized_net_pnl_usdt": 999.0,
            },
        ]

    monkeypatch.setattr(
        "services.authoritative_trade_outcome.load_authoritative_trade_outcomes",
        fake_load,
    )
    result = await dashboard._build_authoritative_profit_observability(mode="paper")
    assert result["status"] == "ok"
    assert result["sample_count"] == 1
    assert result["excluded_incomplete_count"] == 1
    assert result["shadow_excluded"] is True
    assert result["attribution_mismatch_count"] == 0
    assert result["totals"]["fee_after_net_pnl"] == pytest.approx(7.7)


@pytest.mark.asyncio
async def test_authoritative_profit_observability_marks_reconciliation_mismatch(monkeypatch):
    async def fake_load(**_kwargs):
        return [
            {
                "outcome_complete": True,
                "trade_fact_trusted": True,
                "gross_pnl_usdt": 10.0,
                "entry_fee_usdt": 1.0,
                "close_fee_usdt": 1.0,
                "execution_slippage_usdt": 0.5,
                "funding_fee_usdt": 0.0,
                "liquidation_penalty_usdt": 0.0,
                "realized_net_pnl_usdt": 8.0,
                "realized_net_pnl_components": {
                    "components_total_usdt": 9.0,
                    "reported_realized_net_pnl_usdt": 8.0,
                },
            }
        ]

    monkeypatch.setattr(
        "services.authoritative_trade_outcome.load_authoritative_trade_outcomes",
        fake_load,
    )
    result = await dashboard._build_authoritative_profit_observability(mode="paper")
    assert result["status"] == "partial"
    assert result["attribution_mismatch_count"] == 1

