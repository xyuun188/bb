from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from services.model_contribution_performance import ModelContributionPerformanceService


def _decision(decision_id: int, raw: dict) -> SimpleNamespace:
    return SimpleNamespace(id=decision_id, action="long", raw_llm_response=raw)


def _position(index: int, pnl: float, *, created_at: datetime) -> SimpleNamespace:
    return SimpleNamespace(
        symbol="BTC/USDT",
        side="long",
        realized_pnl=pnl,
        created_at=created_at + timedelta(seconds=index),
        entry_exchange_order_id=f"entry-{index}",
        close_exchange_order_id=f"close-{index}",
    )


def _order(index: int, decision_id: int, *, created_at: datetime) -> SimpleNamespace:
    return SimpleNamespace(
        symbol="BTC/USDT",
        decision_id=decision_id,
        exchange_order_id=f"entry-{index}",
        filled_at=created_at + timedelta(seconds=index + 10),
        created_at=created_at + timedelta(seconds=index + 10),
    )


def _current_raw() -> dict:
    return {
        "opportunity_score": {
            "expected_net_breakdown": {
                "components": [
                    {"key": "local_ml", "production_eligible": True},
                    {"key": "server_profit", "production_eligible": True},
                    {"key": "timeseries", "production_eligible": True},
                    {"key": "sentiment", "production_eligible": False},
                    {"key": "shadow_memory", "production_eligible": False},
                ]
            }
        }
    }


def test_model_contribution_is_read_only_for_current_return_sources() -> None:
    now = datetime(2026, 6, 10, tzinfo=UTC)
    positions = [_position(i, 3.0, created_at=now) for i in range(5)]
    orders = [_order(i, i + 1, created_at=now) for i in range(5)]
    decisions = {i + 1: _decision(i + 1, _current_raw()) for i in range(5)}

    stats = ModelContributionPerformanceService().build_stats(positions, orders, decisions)

    for source in ("ml_profit_model", "server_profit_model", "timeseries_model"):
        bucket = stats[source]
        assert bucket["count"] == 5
        assert bucket["pnl"] == pytest.approx(15.0)
        assert bucket["pnl_lcb_usdt"] > 0
        assert bucket["production_permission"] is False
        assert "score_multiplier" not in bucket
        assert "size_multiplier" not in bucket
    assert stats["sentiment_model"]["count"] == 0
    assert stats["shadow_memory"]["count"] == 0
    assert stats["expert_alignment"]["count"] == 0


def test_legacy_profit_first_and_alignment_flags_are_ignored() -> None:
    now = datetime(2026, 6, 10, tzinfo=UTC)
    raw = {
        "opportunity_score": {
            "ml_aligned": True,
            "local_profit_aligned": True,
            "evidence_score": {"components": [{"source": "sentiment", "status": "aligned"}]},
        },
        "profit_first_trade_plan": {
            "model_sources": ["decision_llm", "server_profit", "high_risk_review"]
        },
    }

    stats = ModelContributionPerformanceService().build_stats(
        [_position(1, 4.0, created_at=now)],
        [_order(1, 1, created_at=now)],
        {1: _decision(1, raw)},
    )

    assert all(bucket["count"] == 0 for bucket in stats.values())


def test_model_contribution_excludes_untrusted_trade_facts() -> None:
    now = datetime(2026, 6, 10, tzinfo=UTC)
    trusted = _position(1, 3.0, created_at=now)
    dirty = _position(2, 50.0, created_at=now)
    dirty.close_exchange_order_id = ""

    stats = ModelContributionPerformanceService().build_stats(
        [trusted, dirty],
        [_order(1, 1, created_at=now), _order(2, 2, created_at=now)],
        {1: _decision(1, _current_raw()), 2: _decision(2, _current_raw())},
    )

    assert stats["ml_profit_model"]["count"] == 1
    assert stats["ml_profit_model"]["pnl"] == pytest.approx(3.0)


def test_model_contribution_matches_okx_symbol_variants() -> None:
    now = datetime(2026, 6, 10, tzinfo=UTC)
    position = _position(1, 3.0, created_at=now)
    order = _order(1, 7, created_at=now)
    order.symbol = "BTC-USDT-SWAP"

    service = ModelContributionPerformanceService()
    diagnostics = service.build_lineage_diagnostics(
        [position],
        [order],
        {7: _decision(7, _current_raw())},
    )
    stats = service.build_stats([position], [order], {7: _decision(7, _current_raw())})

    assert diagnostics["matched_position_count"] == 1
    assert diagnostics["reason"] == "ok"
    assert stats["ml_profit_model"]["count"] == 1


def test_missing_quant_sources_do_not_create_ai_fallback_attribution() -> None:
    service = ModelContributionPerformanceService()
    assert service.contribution_sources({}, {}, "short") == []


def test_model_contribution_cannot_adjust_production_score_or_size() -> None:
    service = ModelContributionPerformanceService()
    performance = {
        "ml_profit_model": {
            "count": 100,
            "pnl": -1000.0,
            "profit_factor": 0.1,
            "pnl_lcb_usdt": -50.0,
        }
    }

    adjustment = service.score_adjustment(["ml_profit_model"], performance)

    assert adjustment["active"] is False
    assert adjustment["score_adjustment"] == 0.0
    assert "score_multiplier" not in adjustment
    assert "size_multiplier" not in adjustment
    assert adjustment["production_permission"] is False
