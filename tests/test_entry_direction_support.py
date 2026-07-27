import pytest

from services.entry_direction_support import (
    assess_directional_entry_support,
    assess_unpromoted_model_intervention_support,
    directional_entry_support_reasons,
    summarize_unpromoted_quantitative_evidence,
)


def _row(source: str, *, raw: float = 0.4, objective: float = 0.2) -> dict:
    return {
        "source": source,
        "raw_expected_return_pct": raw,
        "objective_expected_return_pct": objective,
        "horizon_minutes": 30,
    }


def _opinion(name: str, action: str, *, source_group: str = "llm:expert") -> dict:
    return {
        "model_name": name,
        "action": action,
        "reasoning": "auditable",
        "effective_weight": 0.2,
        "source_group": source_group,
        "trace_only_fallback": False,
    }


def test_positive_quant_family_and_two_aligned_experts_are_independent_support() -> None:
    support = assess_directional_entry_support(
        {"long": {"evidence": [_row("local_ml")]}},
        [
            _opinion("trend", "long"),
            _opinion("momentum", "long"),
            _opinion("risk", "hold"),
        ],
        "long",
    )

    assert support["eligible"] is True
    assert support["quant_evidence_families"] == ["local_ml"]
    assert support["independent_support_group_count"] == 2
    assert directional_entry_support_reasons(support, "long") == []


def test_correlated_local_ai_tools_outputs_count_as_one_family() -> None:
    support = assess_directional_entry_support(
        {
            "long": {
                "evidence": [
                    _row("server_profit"),
                    _row("timeseries"),
                    _row("sentiment"),
                ]
            }
        },
        [
            _opinion("trend", "long"),
            _opinion("momentum", "long"),
            _opinion("risk", "hold"),
        ],
        "long",
    )

    assert support["eligible"] is True
    assert support["quant_evidence_families"] == ["local_ai_tools"]
    assert support["positive_quant_sources"] == [
        "sentiment",
        "server_profit",
        "timeseries",
    ]


def test_all_hold_experts_cannot_authorize_model_intervention() -> None:
    support = assess_directional_entry_support(
        {"long": {"evidence": [_row("local_ml"), _row("server_profit")]}},
        [
            _opinion("trend", "hold"),
            _opinion("momentum", "hold"),
            _opinion("risk", "hold"),
        ],
        "long",
    )

    assert support["eligible"] is False
    assert "direction_support_experts_all_hold" in support["blocking_reasons"]
    assert "direction_support_aligned_experts_insufficient" in support[
        "blocking_reasons"
    ]


def test_negative_quant_observations_are_not_support() -> None:
    support = assess_directional_entry_support(
        {
            "long": {
                "evidence": [
                    _row("local_ml", raw=0.2, objective=-0.1),
                    _row("server_profit", raw=-0.2, objective=-0.3),
                ]
            }
        },
        [
            _opinion("trend", "long"),
            _opinion("momentum", "long"),
            _opinion("risk", "hold"),
        ],
        "long",
    )

    assert support["eligible"] is False
    assert support["quant_evidence_families"] == []
    assert "direction_support_positive_quant_evidence_missing" in support[
        "blocking_reasons"
    ]


def test_expert_opposition_must_be_resolved() -> None:
    support = assess_directional_entry_support(
        {"long": {"evidence": [_row("local_ml")]}},
        [
            _opinion("trend", "long"),
            _opinion("momentum", "long"),
            _opinion("sentiment", "short"),
            _opinion("position", "short"),
        ],
        "long",
    )

    assert support["eligible"] is False
    assert "direction_support_expert_opposition_not_resolved" in support[
        "blocking_reasons"
    ]


def test_support_fingerprint_detects_tampering() -> None:
    support = assess_directional_entry_support(
        {"short": {"evidence": [_row("local_ml")]}},
        [
            _opinion("trend", "short"),
            _opinion("momentum", "short"),
            _opinion("risk", "hold"),
        ],
        "short",
    )
    support["aligned_expert_count"] = 99

    assert "direction_support_fingerprint_mismatch" in (
        directional_entry_support_reasons(support, "short")
    )


def test_unpromoted_intervention_uses_positive_mean_after_execution_cost() -> None:
    row = _row("local_ml", raw=0.4, objective=-0.2)
    row["return_distribution_contract"] = {"tail_loss_probability": 0.3}
    support = assess_unpromoted_model_intervention_support(
        {"long": {"evidence": [row]}},
        [
            _opinion("trend", "long"),
            _opinion("momentum", "long"),
            _opinion("risk", "hold"),
        ],
        "long",
        execution_cost_pct=0.1,
    )

    assert support["eligible"] is True
    assert support["expected_net_return_pct"] == pytest.approx(0.3)
    assert support["objective_net_return_pct"] == pytest.approx(-0.3)
    assert support["loss_probability"] == pytest.approx(0.3)
    assert directional_entry_support_reasons(support, "long") == []


def test_unpromoted_intervention_rejects_mean_below_execution_cost() -> None:
    row = _row("local_ml", raw=0.05, objective=-0.2)
    row["return_distribution_contract"] = {"tail_loss_probability": 0.3}
    support = assess_unpromoted_model_intervention_support(
        {"long": {"evidence": [row]}},
        [
            _opinion("trend", "long"),
            _opinion("momentum", "long"),
            _opinion("risk", "hold"),
        ],
        "long",
        execution_cost_pct=0.1,
    )

    assert support["eligible"] is False
    assert "direction_support_expected_net_return_not_positive" in support[
        "blocking_reasons"
    ]


def test_unpromoted_quantitative_summary_is_available_before_expert_support() -> None:
    local_ml = _row("local_ml", raw=0.05, objective=-0.2)
    server_profit = _row("server_profit", raw=0.35, objective=-0.1)
    timeseries = _row("timeseries", raw=0.25, objective=0.05)
    for row, probability in ((local_ml, 0.5), (server_profit, 0.3), (timeseries, 0.2)):
        row["return_distribution_contract"] = {
            "tail_loss_probability": probability
        }

    summary = summarize_unpromoted_quantitative_evidence(
        {"long": {"evidence": [local_ml, server_profit, timeseries]}},
        "long",
        execution_cost_pct=0.1,
    )

    assert summary["diagnostic_only"] is True
    assert summary["production_permission"] is False
    assert summary["execution_cost_complete"] is True
    assert summary["expected_net_return_pct"] == pytest.approx(0.075)
    assert summary["quant_evidence_families"] == ["local_ai_tools"]
