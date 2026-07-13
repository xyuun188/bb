from __future__ import annotations

from dataclasses import fields

from services.strategy_learning import (
    StrategyCandidateGenerator,
    StrategyFeedback,
    StrategyLearningEngine,
    StrategyProfile,
)


def _feedback() -> StrategyFeedback:
    return StrategyFeedback(
        mode="paper",
        window_hours=168,
        generated_at="2026-07-12T00:00:00+00:00",
        totals={"sample_count": 4, "realized_net_pnl_usdt": 12.5},
        side_performance={},
        open_position_pressure={},
        decision_quality={},
        shadow_feedback={},
        expert_memory={},
        manual_intervention={},
        trade_fact_quarantine={},
        reflection_feedback={},
        event_feedback={},
        authoritative_return_observation={
            "sample_count": 4,
            "realized_net_pnl_usdt": 12.5,
        },
        problems=[],
        root_causes=[],
        training_policy={},
    )


def test_strategy_learning_exposes_only_authoritative_return_observation() -> None:
    profiles = StrategyCandidateGenerator().generate(_feedback())
    assert len(profiles) == 1
    assert profiles[0].profile_id == "authoritative_return_observation"
    assert profiles[0].status == "observation_only"
    assert profiles[0].params == {}
    assert profiles[0].promotion["production_permission"] is False


def test_external_profile_cannot_reenter_production_scheduler() -> None:
    engine = StrategyLearningEngine()
    candidate = StrategyProfile(
        profile_id="external_execution_override",
        version=1,
        label="external override",
        status="candidate",
        source="external",
        description="must be ignored",
        params={
            "entry_threshold": -1,
            "position_fraction": 1,
            "leverage": 99,
            "production_permission": True,
        },
    )
    payload = engine.build_from_feedback(_feedback(), extra_profiles=[candidate])
    assert payload["schedule"]["scheduler_mode"] == "observation_only"
    assert payload["schedule"]["runtime"]["production_permission"] is False
    assert [row["id"] for row in payload["schedule"]["candidates"]] == [
        "authoritative_return_observation"
    ]


def test_strategy_learning_context_cannot_mutate_execution_fields() -> None:
    engine = StrategyLearningEngine()
    original = {
        "entry_threshold": "sentinel",
        "position_fraction": "sentinel",
        "leverage": "sentinel",
        "exit_fraction": "sentinel",
        "production_permission": "sentinel",
    }
    result = engine.apply_to_context(dict(original), engine.build_from_feedback(_feedback()))
    for key, value in original.items():
        assert result[key] == value
    learning = result["strategy_learning"]
    assert learning["read_only"] is True
    assert learning["production_permission"] is False
    assert learning["optimization_target"] == "realized_fee_after_return"
    provenance = learning["policy_provenance"]
    assert provenance["sample_count"] == 4
    assert provenance["source"] == "authoritative_closed_position_return_attribution"


def test_strategy_schedule_keeps_execution_fields_out_of_runtime() -> None:
    payload = StrategyLearningEngine().build_from_feedback(_feedback())
    runtime = payload["schedule"]["runtime"]
    assert "entry_threshold" not in runtime
    assert "position_fraction" not in runtime
    assert "leverage" not in runtime
    assert "exit_fraction" not in runtime


def test_feedback_contract_still_carries_authoritative_audit_sections() -> None:
    names = {item.name for item in fields(StrategyFeedback)}
    assert {"totals", "trade_fact_quarantine", "reflection_feedback", "training_policy"} <= names
