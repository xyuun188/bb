import inspect
from pathlib import Path

from ai_brain import ensemble_coordinator, llm_agent, prompts
from services import (
    entry_candidate_evidence,
    expert_memory_service,
    memory_feedback,
    shadow_backtest_service,
)


def test_fixed_confidence_leverage_floors_cannot_return() -> None:
    source = inspect.getsource(llm_agent) + inspect.getsource(ensemble_coordinator)

    for forbidden in (
        "_entry_min_leverage",
        "_entry_leverage_cap",
        "_leverage_cap_for_entry",
        "5-10x",
        "强信号最低 10x",
    ):
        assert forbidden not in source


def test_shadow_memory_cannot_return_as_production_probe_authority() -> None:
    assert not Path("services/entry_evidence_probe.py").exists()
    assert not Path("services/entry_quant_profit_probe.py").exists()
    source = "\n".join(
        inspect.getsource(module)
        for module in (
            memory_feedback,
            entry_candidate_evidence,
            prompts,
        )
    )

    for forbidden in (
        "memory_supported_probe_candidate",
        "missed_opportunity_memory",
        "probe_when_ev_ok",
        "prefer_small_probe_when_current_ev_positive",
    ):
        assert forbidden not in source


def test_shadow_outcome_and_expert_weight_have_no_fixed_success_threshold_path() -> None:
    shadow_source = inspect.getsource(shadow_backtest_service)
    expert_source = inspect.getsource(expert_memory_service)

    assert "shadow_memory_min_return_pct" not in shadow_source
    assert "SHADOW_MISSED_OPPORTUNITY_THRESHOLD" not in shadow_source
    assert "performance_edge" not in expert_source
    assert "success >= failure" not in expert_source
    assert "failure >= success" not in expert_source
