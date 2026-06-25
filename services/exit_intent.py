"""Structured exit-intent classification shared by exit policies."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from ai_brain.base_model import DecisionOutput


class ExitIntent(StrEnum):
    """Normalized reasons that explain why an exit is being attempted."""

    HARD_RISK = "hard_risk"
    TREND_FAILURE = "trend_failure"
    PREDICTIVE_DOWNSIDE = "predictive_downside"
    PROFIT_DRAWDOWN = "profit_drawdown"
    PROFIT_PROTECTION = "profit_protection"
    CAPITAL_ROTATION = "capital_rotation"
    LOSS_REPAIR = "loss_repair"
    ORDINARY = "ordinary"
    HOLD = "hold"


PROTECTIVE_DOWNSIDE_EXIT_TEXT_TERMS = (
    "可能会跌",
    "可能下跌",
    "后续下跌",
    "继续下跌",
    "下行风险",
    "趋势转弱",
    "趋势走弱",
    "趋势失效",
    "趋势破坏",
    "反向压力",
    "反转风险",
    "预防性",
    "保护本金",
    "避免回撤",
    "避免回吐",
    "先撤退",
    "先离场",
    "主动撤退",
    "风险扩大",
    "downside",
    "trend failure",
    "trend invalid",
    "reversal risk",
    "capital protection",
)

PROTECTIVE_DOWNSIDE_INTENTS = frozenset(
    {
        ExitIntent.HARD_RISK,
        ExitIntent.TREND_FAILURE,
        ExitIntent.PREDICTIVE_DOWNSIDE,
    }
)

COOLDOWN_BYPASS_INTENTS = frozenset(
    {
        ExitIntent.HARD_RISK,
        ExitIntent.TREND_FAILURE,
        ExitIntent.PREDICTIVE_DOWNSIDE,
        ExitIntent.PROFIT_DRAWDOWN,
    }
)

PROFIT_EXIT_INTENTS = frozenset(
    {
        ExitIntent.PROFIT_DRAWDOWN,
        ExitIntent.PROFIT_PROTECTION,
        ExitIntent.CAPITAL_ROTATION,
    }
)


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalized_existing_intent(value: Any) -> ExitIntent | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    aliases = {
        "hard_stop": ExitIntent.HARD_RISK,
        "forced_exit": ExitIntent.HARD_RISK,
        "risk_engine": ExitIntent.HARD_RISK,
        "predictive_reversal": ExitIntent.PREDICTIVE_DOWNSIDE,
        "predictive_exit": ExitIntent.PREDICTIVE_DOWNSIDE,
        "downside": ExitIntent.PREDICTIVE_DOWNSIDE,
        "profit_lock": ExitIntent.PROFIT_PROTECTION,
        "take_profit": ExitIntent.PROFIT_PROTECTION,
        "lock_profit": ExitIntent.PROFIT_PROTECTION,
        "retrace": ExitIntent.PROFIT_DRAWDOWN,
        "drawdown": ExitIntent.PROFIT_DRAWDOWN,
        "rotation": ExitIntent.CAPITAL_ROTATION,
        "loss_reduce": ExitIntent.LOSS_REPAIR,
        "loss_compress": ExitIntent.LOSS_REPAIR,
    }
    if text in aliases:
        return aliases[text]
    try:
        return ExitIntent(text)
    except ValueError:
        return None


def _contains_protective_downside_text(text: str) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in PROTECTIVE_DOWNSIDE_EXIT_TEXT_TERMS)


def is_low_quality_release_without_hard_risk(raw: dict[str, Any]) -> bool:
    policy = _safe_dict(raw.get("position_release_policy"))
    close_evidence = _safe_dict(raw.get("close_evidence"))
    exit_quality = _safe_dict(raw.get("exit_quality"))
    invalidation = _safe_dict(exit_quality.get("invalidation"))
    source = str(policy.get("source") or close_evidence.get("source") or "").lower()
    if source not in {"position_quality_capacity_release", "low_quality_position_release"}:
        return False
    fast_trigger = str(raw.get("fast_risk_trigger") or "").strip().lower()
    hard_fast_trigger = fast_trigger in {
        "stop_loss",
        "hard_adverse_move",
        "near_stop_progress",
        "fast_adverse_move",
    }
    return not bool(
        close_evidence.get("hard_risk")
        or close_evidence.get("raw_hard_risk")
        or hard_fast_trigger
        or invalidation.get("severe")
        or invalidation.get("key_break")
        or invalidation.get("trend_reversal")
    )


def classify_exit_intent(
    decision: DecisionOutput,
    *,
    update_raw: bool = True,
) -> ExitIntent:
    """Infer and optionally persist the structured intent for an exit decision."""

    if not decision.is_exit:
        return ExitIntent.HOLD

    raw = _safe_dict(decision.raw_response)
    close_evidence = _safe_dict(raw.get("close_evidence"))
    low_quality_release = is_low_quality_release_without_hard_risk(raw)
    existing = None
    if low_quality_release:
        intent = ExitIntent.CAPITAL_ROTATION
    else:
        existing = (
            _normalized_existing_intent(raw.get("exit_intent"))
            or _normalized_existing_intent(close_evidence.get("exit_intent"))
            or _normalized_existing_intent(close_evidence.get("exit_reason"))
        )
    if existing is not None:
        intent = existing
    elif not low_quality_release:
        execution_profit = _safe_dict(raw.get("execution_profit_protection"))
        exit_quality = _safe_dict(raw.get("exit_quality"))
        invalidation = _safe_dict(exit_quality.get("invalidation"))
        fast_trigger = str(raw.get("fast_risk_trigger") or "").strip().lower()
        reasoning_text = str(decision.reasoning or "")

        hard_risk = bool(
            raw.get("forced_exit")
            or close_evidence.get("hard_risk")
            or close_evidence.get("raw_hard_risk")
            or close_evidence.get("forced_exit")
            or fast_trigger
            in {"stop_loss", "hard_adverse_move", "near_stop_progress", "fast_adverse_move"}
            or (
                raw.get("fast_risk_exit")
                and fast_trigger
                and not (
                    fast_trigger.startswith("profit_drawdown") or fast_trigger == "take_profit"
                )
            )
            or decision.model_name == "risk_engine"
        )
        trend_failure = bool(
            close_evidence.get("trend_failure")
            or close_evidence.get("trend_invalidation")
            or close_evidence.get("thesis_invalidated")
            or invalidation.get("severe")
            or invalidation.get("key_break")
            or invalidation.get("trend_reversal")
        )
        predictive_downside = bool(
            close_evidence.get("predictive_reversal_exit")
            or close_evidence.get("predictive_full_exit")
            or close_evidence.get("predictive_exit")
            or close_evidence.get("strong_opposite_pressure")
            or close_evidence.get("moderate_opposite_pressure")
            or close_evidence.get("capital_protection")
            or close_evidence.get("preventive_exit")
            or _contains_protective_downside_text(reasoning_text)
        )
        loss_repair = bool(
            close_evidence.get("loss_repair")
            or close_evidence.get("loss_repair_evidence")
            or close_evidence.get("position_loss")
        )
        profit_drawdown = bool(
            fast_trigger.startswith("profit_drawdown")
            or close_evidence.get("profit_retrace_protection")
            or close_evidence.get("predictive_reversal_exit")
            and close_evidence.get("position_profit")
        )
        capital_rotation = bool(close_evidence.get("capital_rotation_profit"))
        profit_protection = bool(
            close_evidence.get("profit_protection")
            or close_evidence.get("portfolio_focus_profit_lock")
            or close_evidence.get("quick_profit")
            or execution_profit.get("allow")
            or fast_trigger == "take_profit"
        )

        if hard_risk:
            intent = ExitIntent.HARD_RISK
        elif trend_failure:
            intent = ExitIntent.TREND_FAILURE
        elif predictive_downside and not close_evidence.get("loss_repair"):
            intent = ExitIntent.PREDICTIVE_DOWNSIDE
        elif profit_drawdown:
            intent = ExitIntent.PROFIT_DRAWDOWN
        elif capital_rotation:
            intent = ExitIntent.CAPITAL_ROTATION
        elif profit_protection:
            intent = ExitIntent.PROFIT_PROTECTION
        elif loss_repair:
            intent = ExitIntent.LOSS_REPAIR
        else:
            intent = ExitIntent.ORDINARY

    if update_raw:
        raw["exit_intent"] = intent.value
        if close_evidence:
            close_evidence["exit_intent"] = intent.value
            raw["close_evidence"] = close_evidence
        raw["exit_intent_policy"] = {
            "intent": intent.value,
            "structured": True,
            "source": "exit_intent_classifier",
        }
        decision.raw_response = raw
    return intent
