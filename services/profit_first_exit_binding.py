"""Profit-First v3 entry/exit binding helpers."""

from __future__ import annotations

from datetime import UTC, datetime
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

    close_evidence = raw.get("close_evidence") if isinstance(raw.get("close_evidence"), dict) else {}
    close_evidence = dict(close_evidence)
    target_exit_plan_id = _first_text(
        close_evidence.get("profit_first_exit_plan_id"),
        raw.get("profit_first_exit_plan_id"),
        raw.get("exit_plan_id"),
    )
    target_entry_order_id = _first_text(
        close_evidence.get("target_entry_exchange_order_id"),
        close_evidence.get("entry_exchange_order_id"),
        raw.get("target_entry_exchange_order_id"),
        raw.get("entry_exchange_order_id"),
    )
    target_okx_pos_id = _first_text(
        close_evidence.get("okx_pos_id"),
        raw.get("okx_pos_id"),
        raw.get("position_id"),
    )
    side = "long" if decision.action == Action.CLOSE_LONG else "short"
    match = _matching_position(
        open_positions or [],
        model_name=model_name,
        symbol=decision.symbol,
        side=side,
        target_exit_plan_id=target_exit_plan_id,
        target_entry_order_id=target_entry_order_id,
        target_okx_pos_id=target_okx_pos_id,
    )
    reference = _reference_from_position(
        match,
        target_exit_plan_id=target_exit_plan_id,
        target_entry_order_id=target_entry_order_id,
    )
    if not reference.get("exit_plan_id"):
        reference = _fallback_reference_from_decision(
            symbol=decision.symbol,
            side=side,
            target_exit_plan_id=target_exit_plan_id,
            target_entry_order_id=target_entry_order_id,
            target_okx_pos_id=target_okx_pos_id,
        )
    if reference.get("exit_plan_id"):
        close_evidence["profit_first_exit_plan_id"] = reference["exit_plan_id"]
        raw["profit_first_exit_reference"] = {
            **reference,
            "source": reference.get("source") or "matched_open_position",
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
    target_exit_plan_id: str = "",
    target_entry_order_id: str = "",
    target_okx_pos_id: str = "",
) -> dict[str, Any]:
    normalized_symbol = _normalize_symbol(symbol)
    candidates: list[tuple[int, float, dict[str, Any]]] = []
    for position in open_positions:
        if not isinstance(position, dict):
            continue
        if model_name and str(position.get("model_name") or "") != str(model_name):
            continue
        if _normalize_symbol(position.get("symbol")) != normalized_symbol:
            continue
        if str(position.get("side") or "").lower() != side:
            continue
        score = 0
        reference = _reference_from_position(position)
        if reference.get("exit_plan_id"):
            score += 5
        if target_exit_plan_id and str(reference.get("exit_plan_id") or "").strip() == target_exit_plan_id:
            score += 100
        if target_entry_order_id and _position_has_entry_order_id(position, target_entry_order_id):
            score += 80
        if target_okx_pos_id and str(position.get("okx_pos_id") or "").strip() == target_okx_pos_id:
            score += 60
        candidates.append((score, _created_sort_value(position), position))
    if candidates:
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return candidates[0][2]
    return {}


def _reference_from_position(
    position: dict[str, Any],
    *,
    target_exit_plan_id: str = "",
    target_entry_order_id: str = "",
) -> dict[str, Any]:
    if not position:
        return {}
    exit_plan = position.get("profit_first_exit_plan")
    exit_plan = exit_plan if isinstance(exit_plan, dict) else {}
    trade_plan = position.get("profit_first_trade_plan")
    trade_plan = trade_plan if isinstance(trade_plan, dict) else {}
    leg_reference = _entry_leg_reference(
        position,
        target_exit_plan_id=target_exit_plan_id,
        target_entry_order_id=target_entry_order_id,
    )
    exit_plan_id = (
        leg_reference.get("exit_plan_id")
        or position.get("profit_first_exit_plan_id")
        or exit_plan.get("exit_plan_id")
        or trade_plan.get("exit_plan_id")
        or ""
    )
    if not str(exit_plan_id or "").strip():
        return {}
    reference = {
        "exit_plan_id": str(exit_plan_id),
        "entry_symbol": position.get("symbol") or trade_plan.get("symbol") or "",
        "entry_side": position.get("side") or trade_plan.get("side") or "",
        "entry_plan_version": trade_plan.get("plan_version") or "",
        "entry_decision_lane": trade_plan.get("decision_lane") or "",
        "entry_exchange_order_id": _first_text(
            leg_reference.get("entry_exchange_order_id"),
            position.get("entry_exchange_order_id"),
            _first_text(*(leg.get("exchange_order_id") for leg in _entry_legs(position))),
        ),
        "max_hold_minutes": exit_plan.get("max_hold_minutes") or trade_plan.get("max_hold_minutes"),
        "profit_drawdown_exit_pct": exit_plan.get("profit_drawdown_exit_pct")
        or trade_plan.get("profit_drawdown_exit_pct"),
    }
    if leg_reference.get("source"):
        reference["source"] = leg_reference["source"]
    return reference


def _entry_leg_reference(
    position: dict[str, Any],
    *,
    target_exit_plan_id: str = "",
    target_entry_order_id: str = "",
) -> dict[str, Any]:
    legs = _entry_legs(position)
    if not legs:
        return {}

    if target_entry_order_id:
        for leg in legs:
            leg_order_id = str(leg.get("exchange_order_id") or "").strip()
            if leg_order_id != target_entry_order_id:
                continue
            exit_plan_id = _first_text(
                leg.get("profit_first_exit_plan_id"),
                leg.get("exit_plan_id"),
                target_exit_plan_id,
            )
            if exit_plan_id:
                return {
                    "exit_plan_id": exit_plan_id,
                    "entry_exchange_order_id": leg_order_id,
                    "source": "matched_open_position_entry_leg",
                }

    if target_exit_plan_id:
        for leg in legs:
            leg_exit_plan_id = _first_text(
                leg.get("profit_first_exit_plan_id"),
                leg.get("exit_plan_id"),
            )
            if leg_exit_plan_id == target_exit_plan_id:
                return {
                    "exit_plan_id": leg_exit_plan_id,
                    "entry_exchange_order_id": str(leg.get("exchange_order_id") or "").strip(),
                    "source": "matched_open_position_entry_leg",
                }

    unique_leg_plan_ids = {
        _first_text(leg.get("profit_first_exit_plan_id"), leg.get("exit_plan_id"))
        for leg in legs
        if _first_text(leg.get("profit_first_exit_plan_id"), leg.get("exit_plan_id"))
    }
    if len(unique_leg_plan_ids) == 1:
        only_exit_plan_id = next(iter(unique_leg_plan_ids))
        matching_leg = next(
            (
                leg
                for leg in legs
                if _first_text(leg.get("profit_first_exit_plan_id"), leg.get("exit_plan_id"))
                == only_exit_plan_id
            ),
            {},
        )
        return {
            "exit_plan_id": only_exit_plan_id,
            "entry_exchange_order_id": str(matching_leg.get("exchange_order_id") or "").strip(),
            "source": "matched_open_position_entry_leg",
        }
    return {}


def _fallback_reference_from_decision(
    *,
    symbol: str,
    side: str,
    target_exit_plan_id: str,
    target_entry_order_id: str,
    target_okx_pos_id: str,
) -> dict[str, Any]:
    if not target_exit_plan_id:
        return {}
    return {
        "exit_plan_id": target_exit_plan_id,
        "entry_symbol": symbol,
        "entry_side": side,
        "entry_exchange_order_id": target_entry_order_id,
        "okx_pos_id": target_okx_pos_id,
        "source": "decision_payload_exit_plan_id",
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


def _entry_legs(position: dict[str, Any]) -> list[dict[str, Any]]:
    legs = position.get("entry_legs")
    return [leg for leg in legs if isinstance(leg, dict)] if isinstance(legs, list) else []


def _position_has_entry_order_id(position: dict[str, Any], target_order_id: str) -> bool:
    order_id = str(target_order_id or "").strip()
    if not order_id:
        return False
    if str(position.get("entry_exchange_order_id") or "").strip() == order_id:
        return True
    for leg in _entry_legs(position):
        if str(leg.get("exchange_order_id") or "").strip() == order_id:
            return True
    return False


def _created_sort_value(position: dict[str, Any]) -> float:
    raw_value = position.get("created_at")
    if isinstance(raw_value, datetime):
        value = raw_value
    elif isinstance(raw_value, str):
        try:
            value = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
        except ValueError:
            return 0.0
    else:
        return 0.0
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.timestamp()


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""
