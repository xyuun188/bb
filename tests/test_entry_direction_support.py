from __future__ import annotations

import pytest

from services.entry_direction_support import (
    assess_directional_entry_support,
    assess_paper_model_trade_support,
    directional_entry_support_reasons,
    summarize_paper_quantitative_evidence,
)


def _row(source: str, *, raw: float = 0.4, objective: float = 0.2) -> dict:
    return {
        "source": source,
        "decision_eligible": True,
        "raw_expected_return_pct": raw,
        "objective_expected_return_pct": objective,
        "horizon_minutes": 30,
        "return_distribution_contract": {"tail_loss_probability": 0.3},
    }


def _opinion(name: str, action: str, *, source_group: str) -> dict:
    return {
        "model_name": name,
        "action": action,
        "reasoning": "auditable",
        "effective_weight": 0.2,
        "source_group": source_group,
        "trace_only_fallback": False,
    }


def test_governed_candidate_still_requires_positive_objective_evidence() -> None:
    support = assess_directional_entry_support(
        {"long": {"evidence": [_row("local_ml")] }},
        [
            _opinion("trend", "long", source_group="llm:trend"),
            _opinion("momentum", "long", source_group="llm:momentum"),
            _opinion("risk", "hold", source_group="llm:risk"),
        ],
        "long",
    )

    assert support["eligible"] is True
    assert support["quant_evidence_families"] == ["local_ml"]
    assert directional_entry_support_reasons(support, "long") == []


def test_single_complete_quant_family_can_authorize_paper_direction() -> None:
    support = assess_paper_model_trade_support(
        {
            "long": {
                "evidence": [
                    _row("server_profit"),
                    _row("timeseries"),
                    _row("sentiment"),
                ]
            },
            "short": {
                "evidence": [
                    _row("server_profit", raw=-0.2, objective=-0.3),
                    _row("timeseries", raw=-0.1, objective=-0.2),
                    _row("sentiment", raw=-0.1, objective=-0.2),
                ]
            },
        },
        [],
        "long",
        execution_cost_pct=0.1,
    )

    assert support["eligible"] is True
    assert support["quant_evidence_families"] == ["local_ai_tools"]
    assert support["quant_family_count"] == 1
    assert support["single_quant_family"] is True
    assert support["quantitative_sources"] == [
        "sentiment",
        "server_profit",
        "timeseries",
    ]
    assert support["blocking_reasons"] == []


def test_all_hold_experts_do_not_block_auditable_paper_model_direction() -> None:
    support = assess_paper_model_trade_support(
        {
            "long": {"evidence": [_row("local_ml"), _row("server_profit")]},
            "short": {
                "evidence": [
                    _row("local_ml", raw=-0.2, objective=-0.3),
                    _row("server_profit", raw=-0.1, objective=-0.2),
                ]
            },
        },
        [
            _opinion("trend", "hold", source_group="llm:trend"),
            _opinion("momentum", "hold", source_group="llm:momentum"),
            _opinion("risk", "hold", source_group="llm:risk"),
        ],
        "long",
        execution_cost_pct=0.1,
    )

    assert support["eligible"] is True
    assert support["hold_expert_count"] == 3
    assert support["aligned_expert_count"] == 0
    assert support["blocking_reasons"] == []


def test_negative_net_paper_direction_is_analysis_only() -> None:
    support = assess_paper_model_trade_support(
        {"long": {"evidence": [_row("local_ml", raw=0.05, objective=-0.2)]}},
        [],
        "long",
        execution_cost_pct=0.1,
    )

    assert support["eligible"] is False
    assert support["expected_net_return_pct"] == pytest.approx(-0.05)
    assert support["objective_net_return_pct"] == pytest.approx(-0.3)
    assert support["quant_evidence_families"] == []
    assert "direction_support_expected_net_not_positive" in support["blocking_reasons"]


def test_two_independent_expert_groups_can_block_strong_opposition() -> None:
    support = assess_paper_model_trade_support(
        {
            "long": {"evidence": [_row("local_ml"), _row("server_profit")]},
            "short": {
                "evidence": [
                    _row("local_ml", raw=-0.2, objective=-0.3),
                    _row("server_profit", raw=-0.1, objective=-0.2),
                ]
            },
        },
        [
            _opinion("trend", "long", source_group="llm:trend"),
            _opinion("sentiment", "short", source_group="llm:sentiment"),
            _opinion("position", "short", source_group="llm:position"),
        ],
        "long",
        execution_cost_pct=0.1,
    )

    assert support["eligible"] is False
    assert support["strong_expert_opposition"] is True
    assert "direction_support_strong_expert_opposition" in support[
        "blocking_reasons"
    ]


@pytest.mark.parametrize(
    ("competition", "cost", "expected_reason"),
    [
        ({"long": {"evidence": []}}, 0.1, "direction_support_quant_evidence_missing"),
        (
            {"long": {"evidence": [_row("local_ml")] }},
            None,
            "direction_support_execution_cost_incomplete",
        ),
        (
            {
                "long": {
                    "evidence": [
                        {
                            **_row("local_ml"),
                            "horizon_minutes": None,
                        }
                    ]
                }
            },
            0.1,
            "direction_support_quant_evidence_missing",
        ),
    ],
)
def test_paper_support_rejects_incomplete_market_or_cost_facts(
    competition: dict,
    cost: float | None,
    expected_reason: str,
) -> None:
    support = assess_paper_model_trade_support(
        competition,
        [],
        "long",
        execution_cost_pct=cost,
    )

    assert support["eligible"] is False
    assert expected_reason in support["blocking_reasons"]


def test_support_fingerprint_detects_tampering() -> None:
    support = assess_paper_model_trade_support(
        {"short": {"evidence": [_row("local_ml")] }},
        [],
        "short",
        execution_cost_pct=0.1,
    )
    support["aligned_expert_count"] = 99

    assert "direction_support_fingerprint_mismatch" in (
        directional_entry_support_reasons(support, "short")
    )


def test_paper_quantitative_summary_is_available_before_expert_support() -> None:
    local_ml = _row("local_ml", raw=0.05, objective=-0.2)
    server_profit = _row("server_profit", raw=0.35, objective=-0.1)
    timeseries = _row("timeseries", raw=0.25, objective=0.05)
    for row, probability in ((local_ml, 0.5), (server_profit, 0.3), (timeseries, 0.2)):
        row["return_distribution_contract"] = {
            "tail_loss_probability": probability
        }

    summary = summarize_paper_quantitative_evidence(
        {"long": {"evidence": [local_ml, server_profit, timeseries]}},
        "long",
        execution_cost_pct=0.1,
    )

    assert summary["diagnostic_only"] is True
    assert summary["production_permission"] is False
    assert summary["execution_cost_complete"] is True
    assert summary["expected_net_return_pct"] == pytest.approx(0.075)
    assert summary["quant_evidence_families"] == ["local_ai_tools"]


def test_paper_quantitative_summary_never_averages_different_horizons() -> None:
    server_profit = _row("server_profit", raw=0.2, objective=0.1)
    server_profit["horizon_minutes"] = 10
    timeseries = _row("timeseries", raw=0.8, objective=0.4)
    timeseries["horizon_minutes"] = 60

    summary = summarize_paper_quantitative_evidence(
        {"long": {"evidence": [server_profit, timeseries]}},
        "long",
        execution_cost_pct=0.1,
    )

    assert summary["available_prediction_horizons"] == [10.0, 60.0]
    assert summary["prediction_horizon_minutes"] == 60.0
    assert summary["expected_net_return_pct"] == pytest.approx(0.7)
    assert summary["objective_net_return_pct"] == pytest.approx(0.3)
    assert summary["quant_family_summaries"][0]["sources"] == ["timeseries"]
    assert summary["horizon_selection_policy"] == (
        "best_fee_after_return_coherent_horizon"
    )

    support = assess_paper_model_trade_support(
        {"long": {"evidence": [server_profit, timeseries]}},
        [],
        "long",
        execution_cost_pct=0.1,
    )
    assert support["prediction_horizon_minutes"] == 60.0
    assert support["available_prediction_horizons"] == [10.0, 60.0]
    assert support["eligible"] is False
    assert "direction_support_quant_family_conflict" in (
        directional_entry_support_reasons(support, "long")
    )


def test_paper_quantitative_summary_inherits_direction_cohort_selection() -> None:
    server_profit = _row("server_profit", raw=0.2, objective=0.1)
    server_profit["horizon_minutes"] = 10
    timeseries = _row("timeseries", raw=0.8, objective=0.4)
    timeseries["horizon_minutes"] = 60
    competition = {
        "selected_horizon_minutes": 10.0,
        "horizon_cohort_selection": {
            "selected_horizon_minutes": 10.0,
            "selection_reason": "highest_continuous_weight_then_shortest_horizon",
            "available_horizon_groups": [
                {"horizon_minutes": 10.0},
                {"horizon_minutes": 60.0},
            ],
        },
        "long": {"evidence": [server_profit, timeseries]},
    }

    summary = summarize_paper_quantitative_evidence(
        competition,
        "long",
        execution_cost_pct=0.1,
    )

    assert summary["prediction_horizon_minutes"] == 10.0
    assert summary["available_prediction_horizons"] == [10.0, 60.0]
    assert summary["expected_net_return_pct"] == pytest.approx(0.1)
    assert summary["quant_family_summaries"][0]["sources"] == ["server_profit"]
    assert summary["horizon_selection_policy"] == (
        "highest_continuous_weight_then_shortest_horizon"
    )


def test_two_independent_quant_families_must_prefer_the_same_side() -> None:
    competition = {
        "selected_horizon_minutes": 30.0,
        "horizon_cohort_selection": {"selected_horizon_minutes": 30.0},
        "long": {"evidence": [_row("local_ml"), _row("timeseries")]},
        "short": {
            "evidence": [
                _row("local_ml", raw=-0.2, objective=-0.3),
                _row("timeseries", raw=-0.1, objective=-0.2),
            ]
        },
    }

    support = assess_paper_model_trade_support(
        competition,
        [],
        "long",
        execution_cost_pct=0.1,
    )

    assert support["eligible"] is True
    assert support["quant_evidence_families"] == ["local_ai_tools", "local_ml"]

    competition["short"]["evidence"][1] = _row(
        "timeseries", raw=0.8, objective=0.6
    )
    conflicted = assess_paper_model_trade_support(
        competition,
        [],
        "long",
        execution_cost_pct=0.1,
    )

    assert conflicted["eligible"] is False
    assert conflicted["quant_evidence_families"] == ["local_ml"]
    assert conflicted["conflicting_quant_families"] == ["local_ai_tools"]
    assert "direction_support_quant_family_conflict" in conflicted[
        "blocking_reasons"
    ]


def test_v7_single_aligned_family_still_requires_positive_objective_net() -> None:
    competition = {
        "selected_horizon_minutes": 5.0,
        "horizon_cohort_selection": {"selected_horizon_minutes": 5.0},
        "short": {
            "evidence": [
                {
                    **_row("local_ml", raw=0.37, objective=-0.61),
                    "horizon_minutes": 5.0,
                }
            ]
        },
        "long": {
            "evidence": [
                {
                    **_row("local_ml", raw=-0.37, objective=-2.25),
                    "horizon_minutes": 5.0,
                }
            ]
        },
    }

    support = assess_paper_model_trade_support(
        competition,
        [
            _opinion("trend", "hold", source_group="llm:shared"),
            _opinion("risk", "hold", source_group="llm:shared"),
        ],
        "short",
        execution_cost_pct=0.1,
    )

    assert support["eligible"] is False
    assert support["expected_net_return_pct"] == pytest.approx(0.27)
    assert support["objective_net_return_pct"] == pytest.approx(-0.71)
    assert support["quant_evidence_families"] == []
    assert "direction_support_objective_net_not_positive" in support[
        "blocking_reasons"
    ]


def test_quality_observation_allows_positive_mean_with_negative_lcb() -> None:
    long_row = _row("local_ml", raw=0.45, objective=-0.20)
    long_row["paper_return_quality_governance"] = {
        "paper_execution_permission": False,
        "paper_execution_reason": "fee_after_return_lcb_not_positive",
        "paper_execution_blockers": ["fee_after_return_lcb_not_positive"],
        "paper_execution_evidence_source": "trusted_settlement",
        "paper_execution_evidence": {"sample_count": 0},
        "break_even_contract": {"return_lcb_above_zero": False},
    }
    support = assess_paper_model_trade_support(
        {
            "selected_horizon_minutes": 30.0,
            "horizon_cohort_selection": {"selected_horizon_minutes": 30.0},
            "long": {"evidence": [long_row]},
            "short": {
                "evidence": [_row("local_ml", raw=-0.2, objective=-0.4)]
            },
        },
        [],
        "long",
        execution_cost_pct=0.1,
    )

    assert support["eligible"] is True
    assert support["expected_net_return_pct"] == pytest.approx(0.35)
    assert support["objective_net_return_pct"] == pytest.approx(-0.30)
    assert support["paper_quality_observation_only"] is True
    assert directional_entry_support_reasons(support, "long") == []


def test_negative_lcb_quality_observation_does_not_leak_into_production_support() -> None:
    long_row = _row("local_ml", raw=0.45, objective=-0.20)
    long_row["paper_return_quality_governance"] = {
        "paper_execution_permission": False,
        "paper_execution_reason": "fee_after_return_lcb_not_positive",
        "paper_execution_blockers": ["fee_after_return_lcb_not_positive"],
    }
    support = assess_directional_entry_support(
        {
            "long": {"evidence": [long_row]},
            "short": {"evidence": [_row("local_ml", raw=-0.2, objective=-0.4)]},
        },
        [
            _opinion("trend", "long", source_group="llm:trend"),
            _opinion("momentum", "long", source_group="llm:momentum"),
            _opinion("risk", "hold", source_group="llm:risk"),
        ],
        "long",
        support_scope="governed_return_candidate",
        execution_cost_pct=0.1,
    )

    assert support["eligible"] is False
    assert support["quant_evidence_families"] == []
    assert "direction_support_quant_evidence_missing" in support["blocking_reasons"]
    assert support["production_permission"] is False


def test_quality_observation_still_rejects_high_loss_probability() -> None:
    long_row = _row("local_ml", raw=0.45, objective=-0.20)
    long_row["return_distribution_contract"] = {"tail_loss_probability": 0.61}
    long_row["paper_return_quality_governance"] = {
        "paper_execution_permission": False,
        "paper_execution_reason": "fee_after_return_lcb_not_positive",
        "paper_execution_blockers": ["fee_after_return_lcb_not_positive"],
    }
    support = assess_paper_model_trade_support(
        {
            "long": {"evidence": [long_row]},
            "short": {
                "evidence": [_row("local_ml", raw=-0.2, objective=-0.4)]
            },
        },
        [],
        "long",
        execution_cost_pct=0.1,
    )

    assert support["eligible"] is False
    assert (
        "direction_support_quality_observation_loss_probability_too_high"
        in support["blocking_reasons"]
    )
