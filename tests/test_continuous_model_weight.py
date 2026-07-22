from __future__ import annotations

from datetime import UTC, datetime

import pytest

from services.continuous_model_weight import (
    COLD_START_MULTIPLIER,
    ContinuousModelWeightEvidenceService,
    ContinuousModelWeightPolicy,
)


def _bucket(pnls: list[float]) -> dict:
    profit = sum(value for value in pnls if value > 0)
    loss = abs(sum(value for value in pnls if value < 0))
    average = sum(pnls) / len(pnls)
    return {
        "count": len(pnls),
        "avg_pnl": average,
        "pnl_lcb_usdt": min(average, sorted(pnls)[max(len(pnls) // 4 - 1, 0)]),
        "profit": profit,
        "loss": loss,
        "profit_factor": profit / loss if loss > 0 else None,
    }


def _health(*, errors: float = 0.0, no_return: float = 0.0) -> dict:
    return {
        "components": {
            name: {
                "windows": {
                    "24h": {
                        "participation_count": 20,
                        "json_error_rate": errors,
                        "no_return_rate": no_return,
                        "wrong_recommendation_rate": 0.2,
                    }
                }
            }
            for name in (
                "trend_expert",
                "momentum_expert",
                "sentiment_expert",
                "position_expert",
                "risk_expert",
            )
        }
    }


def _build(
    policy: ContinuousModelWeightPolicy,
    *,
    contribution: dict | None = None,
    health: dict | None = None,
    specialist: dict | None = None,
    regime: str = "trend",
    mode: str = "paper",
) -> dict:
    return policy.build(
        execution_mode=mode,
        market_regime={"regime": regime},
        health_report=health or _health(),
        specialist_report=specialist or {},
        contribution_performance=contribution or {},
        generated_at=datetime(2026, 7, 22, tzinfo=UTC),
    )


def test_total_return_outweighs_low_win_rate() -> None:
    policy = ContinuousModelWeightPolicy()
    high_payoff = _bucket([10000.0, *([-100.0] * 99)])
    frequent_small_wins = _bucket([2.0] * 60 + [-5.0] * 40)

    report = _build(
        policy,
        contribution={
            "expert:trend_expert": high_payoff,
            "expert:momentum_expert": frequent_small_wins,
        },
    )

    trend = report["expert_weights"]["trend_expert"]
    momentum = report["expert_weights"]["momentum_expert"]
    assert high_payoff["count"] == frequent_small_wins["count"] == 100
    assert trend["effective_multiplier"] > momentum["effective_multiplier"]
    assert report["objective"].startswith("fee_after_total_return")


def test_negative_fee_after_return_smoothly_downweights_model() -> None:
    policy = ContinuousModelWeightPolicy()
    first = _build(
        policy,
        contribution={"expert:trend_expert": _bucket([10.0, 8.0, 5.0] * 10)},
    )
    second = _build(
        policy,
        contribution={"expert:trend_expert": _bucket([-10.0, -8.0, -5.0] * 10)},
    )

    before = first["expert_weights"]["trend_expert"]["effective_multiplier"]
    after = second["expert_weights"]["trend_expert"]["effective_multiplier"]
    target = second["expert_weights"]["trend_expert"]["target_multiplier"]
    assert target < after < before
    assert second["weight_changes"]


def test_new_model_starts_low_but_remains_observable() -> None:
    report = _build(ContinuousModelWeightPolicy(), contribution={}, health={"components": {}})

    item = report["expert_weights"]["trend_expert"]
    assert item["cold_start"] is True
    assert item["effective_multiplier"] == COLD_START_MULTIPLIER
    assert item["effective_multiplier"] > 0
    assert report["failed_models_remain_observable"] is True


def test_unstable_model_is_penalized_without_being_removed() -> None:
    report = _build(
        ContinuousModelWeightPolicy(),
        health=_health(errors=0.6, no_return=0.5),
        contribution={"expert:risk_expert": _bucket([5.0] * 30)},
    )

    risk = report["expert_weights"]["risk_expert"]
    assert 0 < risk["effective_multiplier"] < 0.35
    assert risk["shadow_health"]["stability_multiplier"] == 0.2


def test_tail_loss_and_drawdown_reduce_weight_with_same_return_summary() -> None:
    safer = _bucket([4.0, -1.0, 3.0, -1.0] * 10)
    dangerous = dict(safer)
    safer.update({"worst_pnl_usdt": -1.0, "max_drawdown_usdt": 2.0})
    dangerous.update({"worst_pnl_usdt": -50.0, "max_drawdown_usdt": 80.0})

    safer_report = _build(
        ContinuousModelWeightPolicy(),
        contribution={"expert:trend_expert": safer},
    )
    dangerous_report = _build(
        ContinuousModelWeightPolicy(),
        contribution={"expert:trend_expert": dangerous},
    )

    safer_weight = safer_report["expert_weights"]["trend_expert"]
    dangerous_weight = dangerous_report["expert_weights"]["trend_expert"]
    assert safer_weight["effective_multiplier"] > dangerous_weight["effective_multiplier"]
    assert dangerous_weight["actual_fee_after_return"]["tail_loss_signal"] < 0.0
    assert dangerous_weight["actual_fee_after_return"]["drawdown_signal"] < 0.0


def test_smoothing_state_is_separate_by_market_regime() -> None:
    policy = ContinuousModelWeightPolicy()
    trend = _build(
        policy,
        regime="trend",
        contribution={"expert:trend_expert": _bucket([5.0] * 30)},
    )
    volatile = _build(
        policy,
        regime="volatile",
        contribution={"expert:trend_expert": _bucket([-5.0] * 30)},
    )

    assert trend["scenario"] == "paper:trend"
    assert volatile["scenario"] == "paper:volatile"
    assert volatile["expert_weights"]["trend_expert"]["previous_multiplier"] is None


def test_cross_section_metrics_produce_recomputable_market_regime() -> None:
    policy = ContinuousModelWeightPolicy()
    trend = policy.build(
        execution_mode="paper",
        market_regime={
            "mode": "return_distribution_observation",
            "avg_adx_14": 31.0,
            "avg_returns_20": 0.006,
            "avg_price_vs_sma20": 0.004,
            "avg_price_vs_sma50": 0.003,
        },
        health_report=_health(),
        specialist_report={},
        contribution_performance={},
    )
    ranging = policy.build(
        execution_mode="paper",
        market_regime={
            "mode": "return_distribution_observation",
            "avg_adx_14": 19.0,
            "avg_returns_20": -0.002,
            "avg_price_vs_sma20": 0.001,
            "avg_price_vs_sma50": 0.0,
        },
        health_report=_health(),
        specialist_report={},
        contribution_performance={},
    )

    assert trend["scenario"] == "paper:trend_up"
    assert ranging["scenario"] == "paper:range_bound"


def test_live_mode_always_keeps_original_weights() -> None:
    report = _build(
        ContinuousModelWeightPolicy(),
        mode="live",
        contribution={"expert:trend_expert": _bucket([-100.0] * 50)},
    )

    assert report["applied"] is False
    assert report["live_weights_unchanged"] is True
    assert report["expert_weights"]["trend_expert"]["effective_multiplier"] == 1.0
    assert report["quant_source_weights"]["local_ml"]["effective_multiplier"] == 1.0


@pytest.mark.asyncio
async def test_evidence_service_reads_paper_health_and_shadow_only() -> None:
    calls: list[tuple[str, dict]] = []

    class Health:
        async def report(self, **kwargs):
            calls.append(("health", kwargs))
            return {"components": {}}

    class Specialist:
        async def report(self, **kwargs):
            calls.append(("specialist", kwargs))
            return {"models": []}

    service = ContinuousModelWeightEvidenceService(
        health_service=Health(),
        specialist_service=Specialist(),
    )

    report = await service.report("paper")

    assert report["available"] is True
    assert calls == [
        ("health", {"hours": 72, "limit": 1200, "mode": "paper"}),
        ("specialist", {"hours": 72, "mode": "paper"}),
    ]
