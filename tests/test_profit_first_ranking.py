from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from services.okx_order_fact_sync import OKX_SYNC_CONFIRMED
from services.profit_first_ranking import (
    ProfitFirstRankingService,
    _filter_trusted_closed_positions_with_orders,
)
from services.trade_fact_trust import orders_by_exchange_id


def _raw(
    *,
    strategy: str = "balanced_probe",
    lane: str = "validated_probe",
    sources: list[str] | None = None,
    side: str = "long",
) -> dict:
    model_sources = sources or ["decision_llm", "server_profit"]
    return {
        "profit_first_trade_plan": {
            "plan_version": "profit-first-v3.1",
            "symbol": "BTC/USDT",
            "side": side,
            "strategy_profile_id": strategy,
            "decision_lane": lane,
            "expected_net_return_pct": 0.8,
            "loss_probability": 0.35,
            "tail_loss_probability": 0.5,
            "position_size_pct": 0.04,
            "exit_plan_id": "pfep-test",
            "model_sources": model_sources,
            "model_contributions": [
                {"source": source, "field_path": f"profit_first.{source}"}
                for source in model_sources
            ],
        }
    }


def _position(
    idx: int,
    *,
    pnl: float,
    strategy: str = "balanced_probe",
    lane: str = "validated_probe",
    source: str = "server_profit",
    closed_offset_minutes: int = 0,
    side: str = "long",
) -> SimpleNamespace:
    closed_at = datetime(2026, 6, 29, 8, 0, tzinfo=UTC) + timedelta(minutes=closed_offset_minutes)
    return SimpleNamespace(
        id=idx,
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        side=side,
        realized_pnl=pnl,
        fee=0.01,
        created_at=closed_at - timedelta(minutes=45),
        closed_at=closed_at,
        entry_raw=_raw(
            strategy=strategy,
            lane=lane,
            sources=["decision_llm", source],
            side=side,
        ),
    )


def _linked_position(idx: int, *, pnl: float) -> SimpleNamespace:
    row = _position(idx, pnl=pnl)
    row.entry_exchange_order_id = f"entry-{idx}"
    row.close_exchange_order_id = f"close-{idx}"
    return row


def _order(
    order_id: str,
    *,
    okx_sync_status: str | None = OKX_SYNC_CONFIRMED,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=100,
        exchange_order_id=order_id,
        okx_sync_status=okx_sync_status,
    )


def _trusted_for_ranking(
    positions: list[SimpleNamespace],
    orders: list[SimpleNamespace],
) -> tuple[list[SimpleNamespace], dict]:
    return _filter_trusted_closed_positions_with_orders(
        positions,
        orders_by_exchange_id(orders),
    )


def test_profit_first_ranking_promotes_profitable_profile_after_sample_floor() -> None:
    positions = [_position(idx, pnl=1.0, closed_offset_minutes=idx) for idx in range(1, 22)]
    report = ProfitFirstRankingService(min_canary_samples=20).build_report(
        decisions=[],
        closed_positions=positions,
    )

    assert report["audit_only"] is True
    assert report["live_mutation"] is False
    assert report["ranking_ready"] is True
    top = report["strategy_rankings"][0]
    assert top["recommended_stage"] == "canary"
    assert top["can_increase_budget"] is True
    assert "positive_realized_net_pnl" in top["ranking_reasons"]


def test_profit_first_ranking_disables_repeated_losing_profile() -> None:
    positions = [
        _position(idx, pnl=-2.5, strategy="loss_loop", closed_offset_minutes=idx)
        for idx in range(1, 5)
    ]
    report = ProfitFirstRankingService(disable_consecutive_losses=3).build_report(
        decisions=[],
        closed_positions=positions,
    )

    top = report["strategy_rankings"][0]
    assert top["recommended_stage"] == "disable"
    assert top["can_keep_live_size"] is False
    assert "consecutive_losses" in top["ranking_reasons"]
    assert report["summary"]["disable_count"] == 1
    assert report["blockers"][0]["severity"] == "blocking"


def test_profit_first_ranking_demotes_single_tail_loss_without_disabling_resume() -> None:
    positions = [
        _position(1, pnl=-9.5, strategy="single_tail_loss", closed_offset_minutes=1)
    ]
    report = ProfitFirstRankingService(
        disable_consecutive_losses=3,
        max_tail_loss_usdt=8.0,
    ).build_report(
        decisions=[],
        closed_positions=positions,
    )

    top = report["strategy_rankings"][0]
    assert top["recommended_stage"] == "demote"
    assert top["can_increase_budget"] is False
    assert top["can_keep_live_size"] is False
    assert "tail_loss" in top["ranking_reasons"]
    assert report["summary"]["disable_count"] == 0
    assert report["summary"]["demote_count"] == 1
    assert report["blockers"][0]["severity"] == "warning"


def test_profit_first_ranking_keeps_blocking_disable_details_before_warning_truncation() -> None:
    disabled = [
        _position(idx, pnl=-2.5, strategy="loss_loop", closed_offset_minutes=idx)
        for idx in range(1, 5)
    ]
    demoted = [
        _position(
            idx + 100,
            pnl=-0.1,
            strategy=f"demote_{idx}",
            closed_offset_minutes=idx + 100,
        )
        for idx in range(1, 45)
    ]
    report = ProfitFirstRankingService(disable_consecutive_losses=3).build_report(
        decisions=[],
        closed_positions=[*demoted, *disabled],
    )

    assert report["summary"]["disable_count"] == 1
    assert report["summary"]["demote_count"] >= 40
    assert report["blockers"][0]["severity"] == "blocking"
    assert report["blockers"][0]["code"] == "strategy_disable"


def test_profit_first_ranking_demotes_negative_model_source_weight() -> None:
    positions = [
        _position(
            idx,
            pnl=-0.6,
            strategy="weak_source",
            source="timeseries",
            closed_offset_minutes=idx,
        )
        for idx in range(1, 7)
    ]
    report = ProfitFirstRankingService().build_report(
        decisions=[],
        closed_positions=positions,
    )

    source = next(row for row in report["source_rankings"] if row["source"] == "timeseries")
    assert source["recommended_stage"] == "demote"
    assert source["weight_multiplier"] < 1.0
    assert "negative_realized_net_pnl" in source["ranking_reasons"]


def test_profit_first_ranking_filters_untrusted_closed_position_facts() -> None:
    positions = [_linked_position(1, pnl=1.0)]
    trusted, fact_report = _trusted_for_ranking(
        positions,
        orders=[],
    )
    report = ProfitFirstRankingService(min_canary_samples=1).build_report(
        decisions=[],
        closed_positions=trusted,
        trade_fact_report=fact_report,
    )

    assert report["ranking_ready"] is False
    assert report["strategy_rankings"] == []
    assert report["summary"]["checked_closed_position_count"] == 1
    assert report["summary"]["trusted_closed_position_count"] == 0
    assert report["summary"]["quarantined_closed_position_count"] == 1
    assert report["trade_fact_report"]["reason_counts"] == {
        "entry_order_not_okx_confirmed": 1
    }


def test_profit_first_ranking_uses_okx_confirmed_linked_closed_position_facts() -> None:
    positions = [_linked_position(1, pnl=1.0)]
    trusted, fact_report = _trusted_for_ranking(
        positions,
        orders=[_order("entry-1"), _order("close-1")],
    )
    report = ProfitFirstRankingService(min_canary_samples=1).build_report(
        decisions=[],
        closed_positions=trusted,
        trade_fact_report=fact_report,
    )

    assert report["ranking_ready"] is True
    assert report["strategy_rankings"][0]["recommended_stage"] == "canary"
    assert report["summary"]["checked_closed_position_count"] == 1
    assert report["summary"]["trusted_closed_position_count"] == 1
    assert report["summary"]["quarantined_closed_position_count"] == 0
    assert report["trade_fact_report"]["policy"] == "okx_confirmed_closed_positions_only"


def test_profit_first_runtime_feedback_demotes_losing_short_without_hard_ban() -> None:
    positions = [
        *[
            _position(idx, pnl=1.0, side="long", closed_offset_minutes=idx)
            for idx in range(1, 5)
        ],
        *[
            _position(
                idx + 100,
                pnl=-3.0,
                side="short",
                strategy="short_loss_loop",
                closed_offset_minutes=idx + 100,
            )
            for idx in range(1, 5)
        ],
    ]

    report = ProfitFirstRankingService(min_canary_samples=2).build_report(
        decisions=[],
        closed_positions=positions,
    )

    feedback = report["runtime_feedback"]
    short = feedback["side_feedback"]["short"]
    long = feedback["side_feedback"]["long"]

    assert feedback["audit_only"] is True
    assert feedback["live_mutation"] is False
    assert feedback["live_weight_mutation"] is False
    assert feedback["can_influence_strategy_context"] is True
    assert short["recommended_stage"] == "demote"
    assert short["weight_multiplier"] < 1.0
    assert short["hard_ban"] is False
    assert long["weight_multiplier"] >= 1.0
    assert feedback["policy"]["side_weight_policy"] == (
        "relative_window_realized_pnl_not_fixed_usdt_thresholds"
    )


def test_profit_first_runtime_feedback_reports_missing_exit_plan_reference() -> None:
    clean = _position(1, pnl=1.0)
    dirty = _position(2, pnl=-1.0)
    dirty.entry_raw = {"profit_first_trade_plan": {"strategy_profile_id": "legacy"}}

    report = ProfitFirstRankingService(min_canary_samples=1).build_report(
        decisions=[],
        closed_positions=[clean, dirty],
    )

    exit_reference = report["runtime_feedback"]["exit_plan_reference"]

    assert report["summary"]["exit_plan_reference_missing_count"] == 1
    assert exit_reference["checked_count"] == 2
    assert exit_reference["missing_count"] == 1
    assert exit_reference["training_attribution_blocker"] is True
    assert dirty.id in exit_reference["missing_position_ids"]


def test_profit_first_runtime_feedback_blocks_negative_local_ml_live_influence() -> None:
    positions = [
        _position(
            idx,
            pnl=-0.8,
            source="local_ml",
            closed_offset_minutes=idx,
        )
        for idx in range(1, 7)
    ]

    report = ProfitFirstRankingService().build_report(
        decisions=[],
        closed_positions=positions,
    )

    local_ml = report["runtime_feedback"]["local_ml_live_influence"]

    assert local_ml["allow_live_entry_influence"] is False
    assert local_ml["recommended_stage"] == "demote"
    assert local_ml["realized_net_pnl"] < 0
    assert local_ml["can_change_model_routing"] is False
    assert local_ml["reason"] == "degraded_or_negative_realized_net_pnl_source"
