import inspect

from ai_brain import llm_agent
from ai_brain.ensemble_coordinator import EnsembleCoordinator


def test_fixed_technical_hold_rewrite_is_removed() -> None:
    assert not hasattr(llm_agent, "_apply_aggressive_hold_policy")
    assert not hasattr(llm_agent, "_directional_edge")


def test_entry_coordinator_has_no_confidence_based_leverage_floor() -> None:
    coordinator_source = inspect.getsource(EnsembleCoordinator)

    assert "_entry_min_leverage" not in coordinator_source
    assert "_entry_leverage_cap" not in coordinator_source
    assert "最低杠杆" not in coordinator_source
    assert "5-10x" not in coordinator_source
