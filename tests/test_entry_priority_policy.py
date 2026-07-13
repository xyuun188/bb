from ai_brain.base_model import Action, DecisionOutput
from services.entry_priority import EntryExecutionPriorityPolicy


def _decision(raw_response: dict) -> DecisionOutput:
    return DecisionOutput(
        model_name="ensemble_trader",
        symbol="BTC/USDT",
        action=Action.LONG,
        confidence=0.9,
        reasoning="return candidate",
        raw_response=raw_response,
    )


def test_no_candidate_can_bypass_same_round_return_ranking() -> None:
    decision = _decision({"opportunity_score": {"return_lcb_pct": 9.0}})
    assert EntryExecutionPriorityPolicy().immediate_execution_reason(decision) is None


def test_wait_reason_reports_return_lcb_and_downside() -> None:
    decision = _decision(
        {"opportunity_score": {"return_lcb_pct": 0.42, "expected_loss_pct": 0.11}}
    )
    reason = EntryExecutionPriorityPolicy().wait_sort_reason(
        decision, rank=2, candidate_count=5
    )
    assert "2/5" in reason
    assert "0.4200%" in reason
    assert "0.1100%" in reason
