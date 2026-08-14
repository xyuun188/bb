from datetime import UTC, datetime, timedelta

from core.reason_codes import ReasonCode, reason_code_for_blocker
from services.decision_state import DecisionStage, DecisionStageStatus, append_decision_stage
from services.entry_candidate_filter import RejectedEntryCandidate


def test_decision_stage_automatically_adds_structured_reason_evidence() -> None:
    release = datetime(2026, 8, 14, 1, tzinfo=UTC) + timedelta(hours=1)
    raw = append_decision_stage(
        {},
        DecisionStage.RISK_CHECK,
        DecisionStageStatus.BLOCKED,
        "cooldown is active",
        reason_code=ReasonCode.RISK_COOLDOWN,
        observed_value=12,
        threshold=60,
        release_at=release,
        source_event="trade_closed",
    )

    event = raw["decision_state_machine"]["stages"][0]
    summary = raw["decision_state_machine"]["summary"]
    assert event["reason_code"] == ReasonCode.RISK_COOLDOWN
    assert event["reason_evidence"]["observed_value"] == 12
    assert event["reason_evidence"]["threshold"] == 60
    assert event["reason_evidence"]["release_at"] == release.isoformat()
    assert summary["final_reason_code"] == ReasonCode.RISK_COOLDOWN


def test_legacy_stage_and_candidate_blockers_receive_stable_codes() -> None:
    raw = append_decision_stage(
        {},
        DecisionStage.EXCHANGE_SUBMIT,
        DecisionStageStatus.FAILED,
        "exchange rejected order",
    )
    assert raw["decision_state_machine"]["stages"][0]["reason_code"] == ReasonCode.EXCHANGE_REJECTED
    assert reason_code_for_blocker("entry_capacity") == ReasonCode.RISK_CAPACITY
    assert RejectedEntryCandidate.__dataclass_fields__["reason_code"].default == "SYSTEM_POLICY_BLOCKED"
