from __future__ import annotations

from services.model_promotion_policy import (
    build_phase3_promotion_recommendation,
    build_return_objective_report,
)


def _cost_complete_sample(net_return: float) -> dict[str, object]:
    return {
        "net_return_after_cost_pct": net_return,
        "cost_complete": True,
        "fee_return_pct": 0.04,
        "slippage_return_pct": 0.03,
        "funding_return_pct": 0.01,
    }


def _healthy_paper_observation() -> dict[str, object]:
    return {
        "status": "healthy",
        "can_use_for_promotion": True,
        "starts_trading_service": False,
        "submits_orders": False,
        "changes_model_routing": False,
    }


def _recommendation(return_report: dict[str, object]) -> dict[str, object]:
    return build_phase3_promotion_recommendation(
        training_mode="walk_forward",
        model_stage="live",
        quality_report={"totals": {"total": 4, "effective_weight_ratio": 1.0}},
        governance_report={"trainable_sample_count": 4, "contamination_risk": "low"},
        evaluation_policy={"live_mutation": True, "requires_paper_observation": True},
        paper_observation_report=_healthy_paper_observation(),
        return_objective_report=return_report,
    )


def test_return_objective_promotes_positive_fee_after_distribution() -> None:
    report = build_return_objective_report(
        trade_samples=[_cost_complete_sample(value) for value in (0.8, 0.7, 0.6, 0.5)]
    )

    assert report["promotion_ready"] is True
    assert report["optimization_target"] == "realized_fee_after_return"
    assert report["empirical_return_lower_hinge_pct"] > 0
    assert report["policy_provenance"]["sample_count"] == 4


def test_high_win_rate_negative_expectancy_cannot_promote() -> None:
    report = build_return_objective_report(
        trade_samples=[
            *[_cost_complete_sample(0.1) for _ in range(9)],
            _cost_complete_sample(-2.0),
        ]
    )

    assert report["promotion_ready"] is False
    assert report["average_net_return_after_cost_pct"] < 0
    assert "average_fee_after_return_not_positive" in report["blocking_reasons"]


def test_low_win_rate_positive_payoff_still_requires_positive_lower_half() -> None:
    report = build_return_objective_report(
        trade_samples=[
            _cost_complete_sample(4.0),
            _cost_complete_sample(-0.2),
            _cost_complete_sample(-0.2),
        ]
    )

    assert report["average_net_return_after_cost_pct"] > 0
    assert report["promotion_ready"] is False
    assert "empirical_return_lower_hinge_not_positive" in report["blocking_reasons"]


def test_cost_incomplete_returns_are_excluded_and_fail_closed() -> None:
    report = build_return_objective_report(
        trade_samples=[{"net_return_after_cost_pct": 9.0, "cost_complete": False}]
    )

    assert report["available"] is False
    assert report["excluded_cost_incomplete_count"] == 1
    assert report["policy_provenance"]["fallback_reason"]


def test_return_objective_reads_unified_profit_learning_labels() -> None:
    report = build_return_objective_report(
        trade_samples=[
            {
                "data_quality_status": "included",
                "exclude_from_training": False,
                "profit_learning_labels": {
                    "sample_kind": "trade",
                    "cost_basis_label": "fee_plus_funding",
                    "net_return_after_cost_pct": -1.25,
                    "training_supervision_ready": True,
                    "realized_net_pnl_usdt": -1.25,
                    "fee_estimate_usdt": 0.08,
                    "funding_fee_usdt": 0.0,
                    "notional_usdt": 100.0,
                },
            }
        ]
    )

    assert report["available"] is True
    assert report["sample_count"] == 1
    assert report["average_net_return_after_cost_pct"] == -1.25
    assert "cost_complete_return_distribution_missing" not in report["blocking_reasons"]
    assert "average_fee_after_return_not_positive" in report["blocking_reasons"]


def test_unified_profit_learning_labels_fail_closed_without_funding_cost() -> None:
    report = build_return_objective_report(
        trade_samples=[
            {
                "data_quality_status": "included",
                "exclude_from_training": False,
                "profit_learning_labels": {
                    "sample_kind": "trade",
                    "cost_basis_label": "fee_only",
                    "net_return_after_cost_pct": 1.25,
                    "training_supervision_ready": True,
                    "realized_net_pnl_usdt": 1.25,
                    "fee_estimate_usdt": 0.08,
                    "funding_fee_usdt": None,
                    "notional_usdt": 100.0,
                },
            }
        ]
    )

    assert report["available"] is False
    assert report["sample_count"] == 0
    assert "cost_complete_return_distribution_missing" in report["blocking_reasons"]


def test_phase3_promotion_uses_return_report_not_sample_count_or_win_rate() -> None:
    report = build_return_objective_report(
        trade_samples=[_cost_complete_sample(value) for value in (0.8, 0.7, 0.6, 0.5)]
    )
    recommendation = _recommendation(report)

    assert recommendation["live_ready"] is True
    assert recommendation["recommended_stage"] == "live"
    assert recommendation["observed_sample_counts"]["counts_are_diagnostic_only"] is True
    assert recommendation["live_mutation"] is False


def test_phase3_promotion_fails_closed_without_return_report() -> None:
    recommendation = _recommendation({})

    assert recommendation["canary_ready"] is False
    assert recommendation["live_ready"] is False
    assert "return_objective_report_missing" in recommendation["canary_blocking_reasons"]
