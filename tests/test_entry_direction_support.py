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


def test_correlated_local_ai_tools_outputs_count_as_one_family() -> None:
    support = assess_paper_model_trade_support(
        {
            "long": {
                "evidence": [
                    _row("server_profit"),
                    _row("timeseries"),
                    _row("sentiment"),
                ]
            }
        },
        [],
        "long",
        execution_cost_pct=0.1,
    )

    assert support["eligible"] is True
    assert support["quant_evidence_families"] == ["local_ai_tools"]
    assert support["quantitative_sources"] == [
        "sentiment",
        "server_profit",
        "timeseries",
    ]


def test_all_hold_experts_do_not_block_auditable_paper_model_direction() -> None:
    support = assess_paper_model_trade_support(
        {"long": {"evidence": [_row("local_ml")] }},
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


def test_negative_net_paper_direction_remains_eligible_for_coverage_sampling() -> None:
    support = assess_paper_model_trade_support(
        {"long": {"evidence": [_row("local_ml", raw=0.05, objective=-0.2)]}},
        [],
        "long",
        execution_cost_pct=0.1,
    )

    assert support["eligible"] is True
    assert support["expected_net_return_pct"] == pytest.approx(-0.05)
    assert support["objective_net_return_pct"] == pytest.approx(-0.3)
    assert support["quant_evidence_families"] == ["local_ml"]


def test_two_independent_expert_groups_can_block_strong_opposition() -> None:
    support = assess_paper_model_trade_support(
        {"long": {"evidence": [_row("local_ml")] }},
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
