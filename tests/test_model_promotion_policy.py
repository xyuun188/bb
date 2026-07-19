from __future__ import annotations

from services.model_promotion_policy import (
    build_phase3_promotion_recommendation,
    build_return_objective_report,
)
from services.profit_supervision import (
    AUTHORITATIVE_REALIZED_RETURN_TASK,
    COUNTERFACTUAL_EXECUTION_COST_TASK,
    MARKET_OPPORTUNITY_TASK,
    PROFIT_SUPERVISION_VERSION,
)


def _cost_complete_sample(net_return: float) -> dict[str, object]:
    return {
        "net_return_after_cost_pct": net_return,
        "cost_complete": True,
        "fee_return_pct": 0.04,
        "slippage_return_pct": 0.03,
        "funding_return_pct": 0.01,
        "sample_weight": 1.0,
        "profit_supervision": {
            "version": PROFIT_SUPERVISION_VERSION,
            "tasks": {
                MARKET_OPPORTUNITY_TASK: {"eligible": False},
                COUNTERFACTUAL_EXECUTION_COST_TASK: {
                    "eligible": True,
                    "total_cost_pct": 0.08,
                    "slippage_pct": 0.03,
                },
                AUTHORITATIVE_REALIZED_RETURN_TASK: {
                    "eligible": True,
                    "realized_net_return_pct": net_return,
                },
            },
        },
    }


def _shadow_sample() -> dict[str, object]:
    return {
        "sample_weight": 1.0,
        "profit_supervision": {
            "version": PROFIT_SUPERVISION_VERSION,
            "tasks": {
                MARKET_OPPORTUNITY_TASK: {
                    "eligible": True,
                    "long_gross_market_return_pct": 0.5,
                    "short_gross_market_return_pct": -0.5,
                },
                COUNTERFACTUAL_EXECUTION_COST_TASK: {
                    "eligible": True,
                    "long_total_cost_pct": 0.08,
                    "short_total_cost_pct": 0.08,
                },
                AUTHORITATIVE_REALIZED_RETURN_TASK: {"eligible": False},
            },
        },
    }


def _return_report(values: tuple[float, ...]) -> dict[str, object]:
    return build_return_objective_report(
        shadow_samples=[_shadow_sample()],
        trade_samples=[_cost_complete_sample(value) for value in values],
    )


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


def test_promotion_fails_closed_when_contamination_classification_is_unknown() -> None:
    report = build_phase3_promotion_recommendation(
        training_mode="walk_forward",
        model_stage="live",
        quality_report={"totals": {"total": 4, "effective_weight_ratio": 1.0}},
        governance_report={"trainable_sample_count": 4, "contamination_risk": "unknown"},
        evaluation_policy={"live_mutation": True, "requires_paper_observation": True},
        paper_observation_report=_healthy_paper_observation(),
        return_objective_report=_return_report((0.8, 0.7, 0.6, -0.1)),
    )

    assert report["recommended_stage"] == "degraded"
    assert report["canary_ready"] is False
    assert "contamination_risk_unverified" in report["canary_blocking_reasons"]


def test_return_objective_promotes_positive_fee_after_distribution() -> None:
    report = _return_report((0.8, 0.7, 0.6, -0.1))

    assert report["promotion_ready"] is True
    assert report["optimization_target"] == "realized_fee_after_return"
    assert report["empirical_return_lower_hinge_pct"] > 0
    assert report["policy_provenance"]["sample_count"] == 4


def test_return_objective_blocks_undefined_profit_factor() -> None:
    report = _return_report((0.8, 0.7, 0.6, 0.5))

    assert report["profit_factor"] is None
    assert report["promotion_ready"] is False
    assert "profit_factor_undefined" in report["blocking_reasons"]


def test_high_win_rate_negative_expectancy_cannot_promote() -> None:
    report = build_return_objective_report(
        shadow_samples=[_shadow_sample()],
        trade_samples=[
            *[_cost_complete_sample(0.1) for _ in range(9)],
            _cost_complete_sample(-2.0),
        ],
    )

    assert report["promotion_ready"] is False
    assert report["average_net_return_after_cost_pct"] < 0
    assert "average_fee_after_return_not_positive" in report["blocking_reasons"]


def test_low_win_rate_positive_payoff_still_requires_positive_lower_half() -> None:
    report = build_return_objective_report(
        shadow_samples=[_shadow_sample()],
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


def test_legacy_profit_learning_labels_cannot_bypass_supervision_contract() -> None:
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

    assert report["available"] is False
    assert report["sample_count"] == 0
    assert "authoritative_realized_return_distribution_missing" in report[
        "blocking_reasons"
    ]


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
    assert "authoritative_execution_cost_distribution_missing" in report[
        "blocking_reasons"
    ]


def test_phase3_promotion_uses_return_report_not_sample_count_or_win_rate() -> None:
    report = _return_report((0.8, 0.7, 0.6, -0.1))
    recommendation = _recommendation(report)

    assert recommendation["live_ready"] is True
    assert recommendation["active_ready"] is True
    assert recommendation["recommended_stage"] == "active"
    assert recommendation["observed_sample_counts"]["counts_are_diagnostic_only"] is True
    assert recommendation["live_mutation"] is True


def test_phase3_promotion_allows_paper_canary_but_blocks_live_without_return_report() -> None:
    recommendation = _recommendation({})

    assert recommendation["canary_ready"] is True
    assert recommendation["canary_execution_scope"] == "paper_only"
    assert recommendation["canary_production_permission"] is False
    assert recommendation["live_ready"] is False
    assert "return_objective_report_missing" in recommendation["live_blocking_reasons"]
