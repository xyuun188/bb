from services.entry_direction_support import (
    assess_directional_entry_support,
    directional_entry_support_reasons,
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
