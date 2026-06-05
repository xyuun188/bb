"""Decision state tracking helpers.

The trading code is still being split apart.  This module gives every decision
one visible state trail first, so the dashboard can show where a decision was
accepted, blocked, submitted, confirmed, or synced.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class DecisionStage:
    AI_ANALYSIS = "ai_analysis"
    STRATEGY_ARBITRATION = "strategy_arbitration"
    RISK_CHECK = "risk_check"
    EXCHANGE_SUBMIT = "exchange_submit"
    EXCHANGE_CONFIRM = "exchange_confirm"
    LOCAL_SYNC = "local_sync"


class DecisionStageStatus:
    PENDING = "pending"
    PASSED = "passed"
    BLOCKED = "blocked"
    FAILED = "failed"
    SKIPPED = "skipped"
    COMPLETED = "completed"


STAGE_LABELS = {
    DecisionStage.AI_ANALYSIS: "AI分析",
    DecisionStage.STRATEGY_ARBITRATION: "策略仲裁",
    DecisionStage.RISK_CHECK: "风控检查",
    DecisionStage.EXCHANGE_SUBMIT: "OKX提交",
    DecisionStage.EXCHANGE_CONFIRM: "成交确认",
    DecisionStage.LOCAL_SYNC: "本地同步",
}

STATUS_LABELS = {
    DecisionStageStatus.PENDING: "处理中",
    DecisionStageStatus.PASSED: "通过",
    DecisionStageStatus.BLOCKED: "拦截",
    DecisionStageStatus.FAILED: "失败",
    DecisionStageStatus.SKIPPED: "跳过",
    DecisionStageStatus.COMPLETED: "完成",
}


TERMINAL_STATUSES = {
    DecisionStageStatus.BLOCKED,
    DecisionStageStatus.FAILED,
    DecisionStageStatus.SKIPPED,
    DecisionStageStatus.COMPLETED,
}


ORDERED_STAGES = (
    DecisionStage.AI_ANALYSIS,
    DecisionStage.STRATEGY_ARBITRATION,
    DecisionStage.RISK_CHECK,
    DecisionStage.EXCHANGE_SUBMIT,
    DecisionStage.EXCHANGE_CONFIRM,
    DecisionStage.LOCAL_SYNC,
)


def append_decision_stage(
    raw_response: dict[str, Any] | None,
    stage: str,
    status: str,
    reason: str | None = None,
    data: dict[str, Any] | None = None,
    *,
    at: datetime | None = None,
) -> dict[str, Any]:
    """Append a state-machine event to a raw LLM response payload."""

    raw: dict[str, Any] = dict(raw_response or {})
    machine = raw.get("decision_state_machine")
    if not isinstance(machine, dict):
        machine = {}
    stages = machine.get("stages")
    if not isinstance(stages, list):
        stages = []

    event = {
        "stage": stage,
        "stage_label": STAGE_LABELS.get(stage, stage),
        "status": status,
        "status_label": STATUS_LABELS.get(status, status),
        "reason": str(reason or "").strip(),
        "at": (at or datetime.now(timezone.utc)).isoformat(),
    }
    if data:
        event["data"] = data
    stages.append(event)

    machine["stages"] = stages
    machine["current_stage"] = stage
    machine["current_stage_label"] = event["stage_label"]
    machine["current_status"] = status
    machine["current_status_label"] = event["status_label"]
    machine["last_reason"] = event["reason"]
    machine["updated_at"] = event["at"]
    machine["summary"] = summarize_decision_stages(stages)
    raw["decision_state_machine"] = machine
    return raw


def summarize_decision_stages(stages: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Build a compact per-stage summary for API/dashboard display."""

    summary: dict[str, Any] = {
        "completed_stage_count": 0,
        "blocked": False,
        "failed": False,
        "final_stage": None,
        "final_status": None,
        "final_reason": "",
        "by_stage": [],
    }
    if not stages:
        return summary

    latest_by_stage: dict[str, dict[str, Any]] = {}
    for event in stages:
        if not isinstance(event, dict):
            continue
        stage = str(event.get("stage") or "")
        if not stage:
            continue
        latest_by_stage[stage] = event

    ordered = [stage for stage in ORDERED_STAGES if stage in latest_by_stage]
    ordered.extend(stage for stage in latest_by_stage if stage not in ordered)

    by_stage = []
    for stage in ordered:
        event = latest_by_stage[stage]
        status = str(event.get("status") or "")
        by_stage.append({
            "stage": stage,
            "stage_label": event.get("stage_label") or STAGE_LABELS.get(stage, stage),
            "status": status,
            "status_label": event.get("status_label") or STATUS_LABELS.get(status, status),
            "reason": event.get("reason") or "",
            "at": event.get("at"),
        })
        if status in {DecisionStageStatus.PASSED, DecisionStageStatus.COMPLETED}:
            summary["completed_stage_count"] += 1
        if status == DecisionStageStatus.BLOCKED:
            summary["blocked"] = True
        if status == DecisionStageStatus.FAILED:
            summary["failed"] = True

    final = by_stage[-1] if by_stage else {}
    summary["final_stage"] = final.get("stage")
    summary["final_status"] = final.get("status")
    summary["final_reason"] = final.get("reason") or ""
    summary["by_stage"] = by_stage
    return summary


def decision_state_from_raw(raw_response: dict[str, Any] | None) -> dict[str, Any]:
    """Return a normalized decision state machine object."""

    raw = raw_response if isinstance(raw_response, dict) else {}
    machine = raw.get("decision_state_machine")
    if not isinstance(machine, dict):
        return {"stages": [], "summary": summarize_decision_stages([])}
    stages = machine.get("stages")
    if not isinstance(stages, list):
        stages = []
    machine = dict(machine)
    machine["stages"] = stages
    machine["summary"] = summarize_decision_stages(stages)
    return machine
