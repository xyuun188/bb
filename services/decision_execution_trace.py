"""Build user-visible execution traces from decision state events."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from services.okx_error_classifier import is_okx_temporary_service_error
from services.decision_state import (
    ORDERED_STAGES,
    STAGE_LABELS,
    STATUS_LABELS,
    DecisionStage,
    DecisionStageStatus,
    decision_state_from_raw,
)

PROBLEM_STATUSES = {DecisionStageStatus.BLOCKED, DecisionStageStatus.FAILED}
SUCCESS_STATUSES = {DecisionStageStatus.PASSED, DecisionStageStatus.COMPLETED}


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    text = str(value or "").strip()
    return text or None


def _duration_between(start: Any, end: Any) -> float | None:
    start_text = _iso(start)
    end_text = _iso(end)
    if not start_text or not end_text:
        return None
    try:
        start_dt = datetime.fromisoformat(start_text.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end_text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return round(max((end_dt - start_dt).total_seconds(), 0.0), 3)


def _status_label(status: str) -> str:
    return STATUS_LABELS.get(status, status or "未知")


def _stage_label(stage: str) -> str:
    return STAGE_LABELS.get(stage, stage or "未知步骤")


def _normal_event(event: dict[str, Any], index: int) -> dict[str, Any]:
    stage = str(event.get("stage") or "")
    status = str(event.get("status") or "")
    duration_sec = _safe_float_or_none(event.get("duration_sec"))
    data = _safe_dict(event.get("data"))
    return {
        "index": index,
        "stage": stage,
        "stage_label": event.get("stage_label") or _stage_label(stage),
        "status": status,
        "status_label": event.get("status_label") or _status_label(status),
        "reason": str(event.get("reason") or "").strip(),
        "at": _iso(event.get("at")),
        "duration_sec": round(duration_sec, 3) if duration_sec is not None else None,
        "duration_missing": duration_sec is None,
        "data": data,
    }


def _synthesized_order_steps(
    *,
    order_status: str | None,
    order_created_at: Any = None,
    order_filled_at: Any = None,
    reason: str = "",
) -> list[dict[str, Any]]:
    status = str(order_status or "").lower()
    success = status == "filled"
    duration_sec = _duration_between(order_created_at, order_filled_at)
    submit_status = DecisionStageStatus.PASSED if success else DecisionStageStatus.FAILED
    local_status = DecisionStageStatus.COMPLETED if success else DecisionStageStatus.SKIPPED
    return [
        {
            "index": 1,
            "stage": DecisionStage.EXCHANGE_SUBMIT,
            "stage_label": _stage_label(DecisionStage.EXCHANGE_SUBMIT),
            "status": submit_status,
            "status_label": _status_label(submit_status),
            "reason": "历史订单缺少 AI 状态链，按订单状态还原交易所提交结果。",
            "at": _iso(order_created_at),
            "duration_sec": duration_sec,
            "duration_missing": duration_sec is None,
            "data": {"source": "order_snapshot"},
        },
        {
            "index": 2,
            "stage": DecisionStage.LOCAL_SYNC,
            "stage_label": _stage_label(DecisionStage.LOCAL_SYNC),
            "status": local_status,
            "status_label": _status_label(local_status),
            "reason": reason
            or ("订单已成交并同步。" if success else "订单未成交，本地仓位未改动。"),
            "at": _iso(order_filled_at or order_created_at),
            "duration_sec": 0.0 if success else None,
            "duration_missing": not success,
            "data": {"source": "order_snapshot", "order_status": status},
        },
    ]


def build_execution_trace(
    raw_response: dict[str, Any] | None,
    *,
    order_status: str | None = None,
    order_created_at: Any = None,
    order_filled_at: Any = None,
    fallback_reason: str = "",
) -> dict[str, Any]:
    """Return a stable API payload for execution detail timelines."""

    machine = decision_state_from_raw(raw_response)
    raw_events = machine.get("stages")
    events = [
        _normal_event(event, index)
        for index, event in enumerate(raw_events or [], start=1)
        if isinstance(event, dict)
    ]

    if not events:
        events = _synthesized_order_steps(
            order_status=order_status,
            order_created_at=order_created_at,
            order_filled_at=order_filled_at,
            reason=fallback_reason,
        )

    latest_by_stage: dict[str, dict[str, Any]] = {}
    duration_by_stage: dict[str, float] = {}
    missing_duration_by_stage: dict[str, bool] = {}
    for event in events:
        stage = str(event.get("stage") or "")
        if not stage:
            continue
        latest_by_stage[stage] = event
        duration = _safe_float_or_none(event.get("duration_sec"))
        if duration is None:
            missing_duration_by_stage[stage] = True
        else:
            duration_by_stage[stage] = duration_by_stage.get(stage, 0.0) + duration

    ordered = [stage for stage in ORDERED_STAGES if stage in latest_by_stage]
    ordered.extend(stage for stage in latest_by_stage if stage not in ordered)

    steps: list[dict[str, Any]] = []
    for index, stage in enumerate(ordered, start=1):
        event = dict(latest_by_stage[stage])
        duration = duration_by_stage.get(stage)
        event["step_no"] = index
        event["duration_sec"] = round(duration, 3) if duration is not None else None
        event["duration_missing"] = bool(
            event["duration_sec"] is None or missing_duration_by_stage.get(stage)
        )
        steps.append(event)

    failed_step = next(
        (step for step in steps if str(step.get("status") or "") in PROBLEM_STATUSES),
        None,
    )
    final_step = failed_step or (steps[-1] if steps else None)
    success = bool(
        not failed_step
        and final_step
        and (
            str(final_step.get("status") or "") in SUCCESS_STATUSES
            or str(order_status or "").lower() == "filled"
        )
    )
    final_reason = " ".join(
        str(part or "")
        for part in (
            final_step.get("reason") if final_step else "",
            fallback_reason,
        )
    )
    transient_exchange_error = is_okx_temporary_service_error(final_reason)
    total_duration = sum(
        float(step["duration_sec"]) for step in steps if step.get("duration_sec") is not None
    )
    repair_suggestions = repair_suggestions_for_trace(
        failed_step=failed_step,
        order_status=order_status,
        fallback_reason=fallback_reason,
    )
    return {
        "execution_steps": steps,
        "stage_events": events,
        "final_result": {
            "success": success,
            "status": (
                "success"
                if success
                else ("transient_exchange_error" if transient_exchange_error else "failed")
            ),
            "stage": final_step.get("stage") if final_step else None,
            "stage_label": final_step.get("stage_label") if final_step else "",
            "status_label": (
                "交易所临时不可用"
                if transient_exchange_error
                else (final_step.get("status_label") if final_step else "")
            ),
            "reason": final_step.get("reason") if final_step else fallback_reason,
            "total_duration_sec": round(total_duration, 3),
        },
        "failed_step": failed_step,
        "repair_suggestions": repair_suggestions,
    }


def repair_suggestions_for_trace(
    *,
    failed_step: dict[str, Any] | None,
    order_status: str | None,
    fallback_reason: str,
) -> list[str]:
    """Return safe, concrete remediation hints without performing risky actions."""

    reason = " ".join(
        str(part or "")
        for part in (
            failed_step.get("reason") if failed_step else "",
            fallback_reason,
            order_status,
        )
    ).lower()
    suggestions: list[str] = []
    if is_okx_temporary_service_error(reason):
        return [
            "OKX 交易所服务临时不可用：系统未拿到成交确认，本地仓位未改动；已临时跳过该币种，稍后自动重试。"
        ]
    if failed_step:
        stage = str(failed_step.get("stage") or "")
        if stage == DecisionStage.RISK_CHECK:
            suggestions.append(
                "检查该步骤的 blocker/data 字段，确认是风控拦截、重复订单还是账户配置问题。"
            )
        elif stage in {DecisionStage.EXCHANGE_SUBMIT, DecisionStage.EXCHANGE_CONFIRM}:
            suggestions.append(
                "检查 OKX 返回码、交易对最小张数/最小名义价值、杠杆模式和账户可用保证金。"
            )
        elif stage == DecisionStage.LOCAL_SYNC:
            suggestions.append("刷新持仓同步任务并核对 OKX 实际仓位，避免本地记录与交易所不一致。")

    if any(code in reason for code in ("51008", "51004", "insufficient")):
        suggestions.append("OKX 保证金不足：降低订单名义价值或释放占用保证金后再执行。")
    if any(code in reason for code in ("51155", "51169", "minimum", "min size")):
        suggestions.append(
            "下单数量不符合交易规则：按 OKX 合约最小张数、精度和最小名义价值重新计算。"
        )
    if "open interest" in reason or "platform" in reason and "limit" in reason:
        suggestions.append("OKX 合约总持仓上限触发：暂时跳过该交易对，等待交易所限制解除。")
    if "timeout" in reason or "超时" in reason:
        suggestions.append("交易所响应超时：先同步 OKX 订单/仓位，再决定是否重试，避免重复下单。")

    if not suggestions:
        suggestions.append("未发现可自动安全修复项，请查看步骤 data 和原始交易所返回信息定位。")
    return suggestions
