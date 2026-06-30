"""Profit-First v3 entry/exit binding helpers."""

from __future__ import annotations

from typing import Any

from ai_brain.base_model import Action, DecisionOutput


def attach_profit_first_exit_reference(
    decision: DecisionOutput,
    open_positions: list[dict[str, Any]] | None,
    *,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Attach the original entry exit-plan reference to an exit decision."""

    raw = decision.raw_response if isinstance(decision.raw_response, dict) else {}
    raw = dict(raw)
    if not decision.is_exit:
        decision.raw_response = raw
        return raw

    side = "long" if decision.action == Action.CLOSE_LONG else "short"
    match = _matching_position(
        open_positions or [],
        model_name=model_name,
        symbol=decision.symbol,
        side=side,
    )
    reference = _reference_from_position(match)
    close_evidence = raw.get("close_evidence") if isinstance(raw.get("close_evidence"), dict) else {}
    close_evidence = dict(close_evidence)
    if reference.get("exit_plan_id"):
        close_evidence["profit_first_exit_plan_id"] = reference["exit_plan_id"]
        raw["profit_first_exit_reference"] = {
            **reference,
            "source": "matched_open_position",
            "missing_original_exit_plan_reference": False,
        }
    else:
        plan_failure_reason = _plan_failure_reason(raw, close_evidence)
        raw["profit_first_exit_reference"] = {
            "exit_plan_id": "",
            "source": "missing_matched_open_position_exit_plan",
            "missing_original_exit_plan_reference": True,
            "plan_failure_reason": plan_failure_reason,
        }
        if plan_failure_reason:
            close_evidence["profit_first_plan_failure_reason"] = plan_failure_reason
    raw["close_evidence"] = close_evidence
    decision.raw_response = raw
    return raw


def _matching_position(
    open_positions: list[dict[str, Any]],
    *,
    model_name: str | None,
    symbol: str,
    side: str,
) -> dict[str, Any]:
    normalized_symbol = _normalize_symbol(symbol)
    for position in open_positions:
        if not isinstance(position, dict):
            continue
        if model_name and str(position.get("model_name") or "") != str(model_name):
            continue
        if _normalize_symbol(position.get("symbol")) != normalized_symbol:
            continue
        if str(position.get("side") or "").lower() != side:
            continue
        return position
    return {}


def _reference_from_position(position: dict[str, Any]) -> dict[str, Any]:
    if not position:
        return {}
    exit_plan = position.get("profit_first_exit_plan")
    exit_plan = exit_plan if isinstance(exit_plan, dict) else {}
    trade_plan = position.get("profit_first_trade_plan")
    trade_plan = trade_plan if isinstance(trade_plan, dict) else {}
    exit_plan_id = (
        position.get("profit_first_exit_plan_id")
        or exit_plan.get("exit_plan_id")
        or trade_plan.get("exit_plan_id")
        or ""
    )
    if not str(exit_plan_id or "").strip():
        return {}
    return {
        "exit_plan_id": str(exit_plan_id),
        "entry_symbol": position.get("symbol") or trade_plan.get("symbol") or "",
        "entry_side": position.get("side") or trade_plan.get("side") or "",
        "entry_plan_version": trade_plan.get("plan_version") or "",
        "entry_decision_lane": trade_plan.get("decision_lane") or "",
        "max_hold_minutes": exit_plan.get("max_hold_minutes") or trade_plan.get("max_hold_minutes"),
        "profit_drawdown_exit_pct": exit_plan.get("profit_drawdown_exit_pct")
        or trade_plan.get("profit_drawdown_exit_pct"),
    }


def _plan_failure_reason(raw: dict[str, Any], close_evidence: dict[str, Any]) -> str:
    for value in (
        raw.get("profit_first_plan_failure_reason"),
        raw.get("plan_failure_reason"),
        close_evidence.get("profit_first_plan_failure_reason"),
        close_evidence.get("plan_failure_reason"),
    ):
        text = str(value or "").strip()
        if text:
            return text[:240]
    return ""


def _normalize_symbol(value: Any) -> str:
    text = str(value or "").upper().strip()
    if "-" in text and "/" not in text:
        parts = text.split("-")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
    return text
