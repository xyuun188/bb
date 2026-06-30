from __future__ import annotations

import argparse
import secrets
import sys
from collections import Counter
from pathlib import Path

# ruff: noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text
from core.safe_output import safe_print

REMOTE_SCRIPT_TEMPLATE = r"""
import asyncio
import json
import math
import sys
from collections import Counter
from datetime import UTC, datetime, timedelta

sys.path.insert(0, "/data/bb/app")

from sqlalchemy import select
from db.session import get_session_ctx
from models.decision import AIDecision
from models.trade import Order, Position
from models.learning import ShadowBacktest, ExpertMemory, StrategyLearningEvent
from services.decision_state import decision_state_from_raw
from services.execution_reason_localizer import localize_execution_reason
from services.ml_signal_service import MLSignalService
from services.trade_fact_trust import closed_position_trade_fact_untrusted_reason
from services.trade_execution_contract import TradeExecutionContractService
from services.entry_evidence import (
    ENTRY_EVIDENCE_SCORE_MEDIUM,
    ENTRY_EVIDENCE_SCORE_NORMAL,
    ENTRY_EVIDENCE_SCORE_PROBE,
    ENTRY_EVIDENCE_SCORE_SMALL,
    ENTRY_EVIDENCE_SCORE_WEAK_PROBE,
)

WINDOW_MINUTES = __WINDOW_MINUTES__
SUMMARY_ONLY = __SUMMARY_ONLY__
MARKET_SYMBOL_ONLY = __MARKET_SYMBOL_ONLY__
ENTRY_ONLY = __ENTRY_ONLY__
FAST_CLOSE_MINUTES = 15
now = datetime.now(UTC)
since = now - timedelta(minutes=WINDOW_MINUTES)


def aware(dt):
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def safe_dict(value):
    return value if isinstance(value, dict) else {}


def safe_list(value):
    return value if isinstance(value, list) else []


def json_safe(value):
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return str(value)


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default


def maybe_float(value):
    try:
        if value is None:
            return None
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    except Exception:
        return None


def roundv(value, digits=6):
    return round(safe_float(value), digits)


def round_optional(value, digits=6):
    number = maybe_float(value)
    if number is None:
        return None
    return round(number, digits)


def selected_side(decision):
    action = str(decision.action or "").lower().strip()
    if action in {"long", "open_long", "buy"}:
        return "long"
    if action in {"short", "open_short", "sell"}:
        return "short"
    opp = safe_dict(safe_dict(decision.raw_llm_response).get("opportunity_score"))
    return str(opp.get("side") or "").lower()


def analysis_type(decision):
    raw = safe_dict(decision.raw_llm_response)
    value = str(
        getattr(decision, "analysis_type", "") or raw.get("analysis_type") or "unknown"
    ).lower().strip()
    if value in {"position", "holding", "holdings"}:
        return "position_review"
    if value in {"market_scan", "symbol_scan"}:
        return "market"
    return value or "unknown"


def selected_side_evidence(raw, side):
    evidence = safe_dict(raw.get("entry_candidate_evidence"))
    side_evidence = safe_dict(evidence.get(side))
    if side_evidence:
        return side_evidence
    if str(evidence.get("side") or "").lower() == side:
        return evidence
    return {}


def opportunity(decision):
    return safe_dict(safe_dict(decision.raw_llm_response).get("opportunity_score"))


def expected_net(decision):
    raw = safe_dict(decision.raw_llm_response)
    opp = opportunity(decision)
    side_ev = selected_side_evidence(raw, selected_side(decision))
    if side_ev:
        return safe_float(
            side_ev.get("expected_net_return_pct"),
            safe_float(opp.get("expected_net_return_pct")),
        )
    return safe_float(opp.get("expected_net_return_pct"))


def profit_quality(decision):
    raw = safe_dict(decision.raw_llm_response)
    opp = opportunity(decision)
    side_ev = selected_side_evidence(raw, selected_side(decision))
    if side_ev:
        return safe_float(
            side_ev.get("profit_quality_ratio"),
            safe_float(opp.get("profit_quality_ratio")),
        )
    return safe_float(opp.get("profit_quality_ratio"))


def probe_diagnostics(decision):
    raw = safe_dict(decision.raw_llm_response)
    side = selected_side(decision)
    side_ev = selected_side_evidence(raw, side)
    blocked = safe_dict(raw.get("evidence_profit_probe_blocked"))
    block_reasons = safe_list(side_ev.get("probe_conversion_block_reasons"))
    return {
        "side": side or str(side_ev.get("side") or blocked.get("side") or "").lower(),
        "recommendation": side_ev.get("recommendation") or blocked.get("recommendation") or "",
        "probe_conversion_ready": side_ev.get("probe_conversion_ready"),
        "probe_conversion_block_reasons": block_reasons,
        "probe_conversion_thresholds": safe_dict(
            side_ev.get("probe_conversion_thresholds")
        )
        or safe_dict(blocked.get("thresholds")),
        "evidence_profit_probe_blocked": {
            "blocked": bool(blocked.get("blocked")),
            "block_kind": blocked.get("block_kind") or "",
            "block_reasons": safe_list(blocked.get("block_reasons")),
                "reason": short_text(blocked.get("reason"), 260, localize=True),
            "side": blocked.get("side") or "",
            "expected_net_return_pct": roundv(blocked.get("expected_net_return_pct")),
            "profit_quality_ratio": roundv(blocked.get("profit_quality_ratio")),
            "loss_probability": roundv(blocked.get("loss_probability")),
            "tail_risk_score": roundv(blocked.get("tail_risk_score")),
            "thresholds": safe_dict(blocked.get("thresholds")),
        }
        if blocked
        else {},
    }


def normalize_relief_for_final_contract(relief, final_shadow_only, final_tier, final_score):
    relief = safe_dict(relief)
    if not relief.get("applied") or not relief.get("shadow_only") or final_shadow_only:
        return relief
    normalized = dict(relief)
    normalized["tradeable_probe"] = True
    normalized["shadow_only"] = False
    normalized["final_tier"] = final_tier
    normalized["final_effective_score"] = roundv(final_score)
    normalized["state_override_reason"] = (
        "Final evidence tier is tradeable; earlier shadow-only relief no longer blocks execution."
    )
    return normalized


def evidence(decision):
    opp = opportunity(decision)
    ev = safe_dict(opp.get("evidence_score"))
    breakdown = safe_dict(opp.get("expected_net_breakdown"))
    components = {
        str(item.get("key") or ""): item
        for item in safe_list(breakdown.get("components"))
        if isinstance(item, dict)
    }
    shadow_memory = safe_dict(components.get("shadow_memory"))
    final_tier = ev.get("tier") or opp.get("evidence_tier") or ""
    final_score = safe_float(ev.get("effective_score", opp.get("score")))
    final_shadow_only = bool(ev.get("shadow_only"))
    return {
        "tier": final_tier,
        "effective_score": roundv(final_score),
        "entry_evidence_score": roundv(ev.get("score")),
        "entry_evidence_score_offset": roundv(safe_float(ev.get("score")) - final_score),
        "score": roundv(opp.get("score")),
        "min_score_required": roundv(opp.get("min_score_required")),
        "expected_net_return_pct": roundv(expected_net(decision)),
        "aggregate_expected_net_return_pct": roundv(opp.get("expected_net_return_pct")),
        "profit_quality_ratio": roundv(profit_quality(decision)),
        "loss_probability": roundv(
            opp.get("server_profit_loss_probability", opp.get("loss_probability"))
        ),
        "tail_risk_score": roundv(opp.get("tail_risk_score")),
        "hard_block": bool(ev.get("hard_block")),
        "hard_block_reasons": safe_list(ev.get("hard_block_reasons")),
        "advisory_wait_reasons": safe_list(ev.get("advisory_wait_reasons")),
        "aligned_support_sources": safe_list(ev.get("aligned_support_sources")),
        "major_opposites": safe_list(ev.get("major_opposites")),
        "weak_opposites": safe_list(ev.get("weak_opposites")),
        "strong_opposites": safe_list(ev.get("strong_opposites")),
          "strong_positive_net_relief": safe_dict(
              safe_dict(opp.get("evidence_score")).get("strong_positive_net_relief")
          ),
        "missing_key_degraded_relief": safe_dict(
            safe_dict(opp.get("evidence_score")).get("missing_key_degraded_relief")
        ),
          "positive_net_probe_relief": normalize_relief_for_final_contract(
              safe_dict(safe_dict(opp.get("evidence_score")).get("positive_net_probe_relief")),
              final_shadow_only,
            final_tier,
            final_score,
        ),
        "memory_missed_opportunity_relief": safe_dict(
            safe_dict(opp.get("evidence_score")).get("memory_missed_opportunity_relief")
        ),
        "short_probe_relief": normalize_relief_for_final_contract(
            safe_dict(safe_dict(opp.get("evidence_score")).get("short_probe_relief")),
            final_shadow_only,
            final_tier,
            final_score,
        ),
        "short_evidence_adjustment": safe_dict(
            safe_dict(opp.get("evidence_score")).get("short_evidence_adjustment")
        ),
        "memory_habit_adjustment": safe_dict(opp.get("memory_habit_adjustment")),
        "tradeable_probe": bool(ev.get("tradeable_probe")),
        "shadow_only": final_shadow_only,
        "vector_memory_adjustment": safe_dict(opp.get("vector_memory_adjustment")),
        "side_quality_adjustment": safe_dict(opp.get("side_quality_adjustment")),
        "expected_net_formula": breakdown.get("formula") or "",
        "ai_expected_return_policy": opp.get("ai_expected_return_policy") or "",
        "ai_expected_return_weight": roundv(opp.get("ai_expected_return_weight")),
        "ai_expected_return_independent_probe_support": safe_list(
            opp.get("ai_expected_return_independent_probe_support")
        ),
        "shadow_memory_component": shadow_memory,
    }


def expected_net_components(decision):
    opp = opportunity(decision)
    breakdown = safe_dict(opp.get("expected_net_breakdown"))
    result = {}
    for item in safe_list(breakdown.get("components")):
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "unknown")
        result[key] = safe_float(item.get("contribution_pct"))
    return result


def evidence_components(decision):
    ev = safe_dict(opportunity(decision).get("evidence_score"))
    result = []
    for item in safe_list(ev.get("components")):
        if isinstance(item, dict):
            result.append(item)
    return result


ENTRY_EVIDENCE_RELIEF_KEYS = (
    "missing_key_degraded_relief",
    "positive_net_probe_relief",
    "memory_missed_opportunity_relief",
    "strong_positive_net_relief",
    "short_probe_relief",
)


def entry_skip_kind(decision):
    raw = safe_dict(decision.raw_llm_response)
    shadow = safe_dict(raw.get("entry_evidence_shadow_only"))
    if shadow.get("skip_kind"):
        return str(shadow.get("skip_kind"))
    machine = safe_dict(raw.get("decision_state_machine"))
    for event in reversed(safe_list(machine.get("stages"))):
        if not isinstance(event, dict):
            continue
        data = safe_dict(event.get("data"))
        if data.get("skip_kind"):
            return str(data.get("skip_kind"))
    summary = safe_dict(machine.get("summary"))
    final_stage = str(summary.get("final_stage") or "").lower()
    final_status = str(summary.get("final_status") or "").lower()
    if bool(getattr(decision, "was_executed", False)) or (
        final_stage == "local_sync" and final_status == "completed"
    ):
        return "executed"
    if final_stage == "local_sync" and final_status in {"skipped", "failed"}:
        return "exchange_not_confirmed"
    return "unknown"


def local_ml_readiness_summary():
    try:
        status = MLSignalService().status()
    except Exception as exc:
        return {"available": False, "status": "error", "error": str(exc)[:180]}
    readiness = safe_dict(status.get("readiness"))
    quality = safe_dict(status.get("quality_report"))
    composition = safe_dict(status.get("training_window_composition"))
    metrics = safe_dict(readiness.get("metrics"))
    blocking = safe_list(readiness.get("blocking_reasons"))
    return {
        "available": bool(status.get("available")),
        "status": status.get("status"),
        "readiness_state": status.get("readiness_state") or readiness.get("state"),
        "allow_live_position_influence": bool(
            status.get("allow_live_position_influence")
            or readiness.get("allow_live_position_influence")
        ),
        "advisory_enabled": bool(status.get("advisory_enabled")),
        "blocking_reason_codes": [
            item.get("code") for item in blocking if isinstance(item, dict)
        ],
        "metrics": {
            key: metrics.get(key)
            for key in (
                "sample_count",
                "test_count",
                "dirty_sample_ratio",
                "long_pr_auc",
                "short_pr_auc",
                "top_long_avg_return_pct",
                "top_short_avg_return_pct",
                "model_age_seconds",
                "training_data_version",
                "required_training_data_version",
            )
        },
        "quality_totals": safe_dict(quality.get("totals")),
        "quality_top_reasons": safe_list(quality.get("top_reasons"))[:8],
        "quality_by_kind": safe_dict(quality.get("by_kind")),
        "quality_top_actions": safe_list(quality.get("top_actions"))[:8],
        "quality_top_timeframes": safe_list(quality.get("top_timeframes"))[:8],
        "training_window_composition": pick(
            composition,
            (
                "sample_count",
                "decision_action_counts",
                "best_action_counts",
                "data_quality_status_counts",
                "effective_weight",
                "effective_weight_ratio",
            ),
        ),
    }


async def trade_execution_contract_summary():
    try:
        report = await TradeExecutionContractService().report(since=since, limit=600)
    except Exception as exc:
        return {"status": "error", "error": short_text(str(exc), 300)}
    summary = safe_dict(report.get("summary"))
    violation_count = int(safe_float(summary.get("contract_violation_count"), 0))
    return {
        "status": "ok" if violation_count == 0 else "violation",
        "audit_only": bool(report.get("audit_only")),
        "can_bypass_risk_controls": bool(report.get("can_bypass_risk_controls")),
        "summary": {
            "decision_count": summary.get("decision_count"),
            "executed_entry_count": summary.get("executed_entry_count"),
            "contract_violation_count": summary.get("contract_violation_count"),
            "weak_evidence_executed_count": summary.get("weak_evidence_executed_count"),
            "negative_expected_executed_count": summary.get(
                "negative_expected_executed_count"
            ),
            "fast_loss_count": summary.get("fast_loss_count"),
            "fast_loss_without_strong_exit_count": summary.get(
                "fast_loss_without_strong_exit_count"
            ),
            "reentry_without_strong_unlock_count": summary.get(
                "reentry_without_strong_unlock_count"
            ),
        },
        "violation_reason_counts": safe_dict(report.get("violation_reason_counts")),
        "query_policy": safe_dict(report.get("query_policy")),
        "violations": json_safe(safe_list(report.get("violations"))[:10]),
        "fast_loss_samples": json_safe(safe_list(report.get("fast_loss_samples"))[:10]),
    }


def is_shadow_only_entry_decision(decision):
    return bool(evidence(decision).get("shadow_only"))


def sizing(decision):
    raw = safe_dict(decision.raw_llm_response)
    profit = safe_dict(raw.get("profit_risk_sizing"))
    strategy = safe_dict(profit.get("strategy_learning_sizing"))
    return {
        "position_size_pct": roundv(decision.position_size_pct),
        "leverage": roundv(decision.suggested_leverage),
        "final_notional_usdt": roundv(profit.get("final_notional_usdt")),
        "quality_tier": profit.get("quality_tier") or "",
        "low_payoff_quality": bool(profit.get("low_payoff_quality")),
        "low_payoff_reasons": safe_list(profit.get("low_payoff_reasons")),
        "notional_floor_applied": bool(profit.get("notional_floor_applied")),
        "notional_floor_blocked": profit.get("notional_floor_blocked") or "",
        "meaningful_size_reason": profit.get("meaningful_size_reason") or "",
        "strategy_sizing_applied": bool(strategy.get("applied")),
        "strategy_probe_cap_applied": bool(strategy.get("probe_cap_applied")),
        "strategy_max_probe_size_pct": roundv(strategy.get("max_probe_size_pct")),
        "strategy_reason": strategy.get("reason") or "",
        "pnl_structure_guard": safe_dict(profit.get("pnl_structure_guard")),
        "risk_budget_boost": safe_dict(profit.get("risk_budget_boost")),
    }


def state(decision):
    machine = decision_state_from_raw(safe_dict(decision.raw_llm_response))
    summary = safe_dict(machine.get("summary"))
    return {
        "final_stage": summary.get("final_stage"),
        "final_status": summary.get("final_status"),
        "final_reason": localize_execution_reason(
            summary.get("final_reason") or decision.execution_reason or ""
        )
        or "",
        "completed_stage_count": summary.get("completed_stage_count"),
        "blocked": bool(summary.get("blocked")),
        "failed": bool(summary.get("failed")),
    }


def short_text(text, limit=260, *, localize=False):
    if localize:
        text = localize_execution_reason(str(text or ""))
    text = str(text or "").replace("\n", " ").strip()
    return text[:limit] + ("…" if len(text) > limit else "")


def order_notional(order):
    return safe_float(order.quantity) * safe_float(order.price)


def order_execution_result(decision):
    if decision is None:
        return {}
    raw = safe_dict(decision.raw_llm_response)
    result = safe_dict(raw.get("execution_result"))
    if not result:
        return {}
    raw_result = safe_dict(result.get("raw_response"))
    return {
        "source": result.get("source") or "",
        "order_id": result.get("order_id"),
        "exchange_order_id": result.get("exchange_order_id"),
        "status": result.get("status") or "",
        "quantity": roundv(result.get("quantity")),
        "price": roundv(result.get("price")),
        "fee": roundv(result.get("fee")),
        "pnl": roundv(result.get("pnl")),
        "exchange_confirmed": bool(result.get("exchange_confirmed")),
        "exit_progress": bool(result.get("exit_progress")),
        "error": short_text(raw_result.get("error"), 500),
        "raw_error": short_text(raw_result.get("raw_error"), 900),
        "execution_blocker": raw_result.get("execution_blocker") or "",
        "okx_rejection": bool(raw_result.get("okx_rejection")),
        "system_pre_submit_rejection": bool(
            raw_result.get("system_pre_submit_rejection")
        ),
        "okx_symbol": raw_result.get("okx_symbol") or "",
        "planned_order_contracts": round_optional(raw_result.get("planned_order_contracts")),
        "planned_base_quantity": round_optional(raw_result.get("planned_base_quantity")),
        "okx_order_rules": safe_dict(raw_result.get("okx_order_rules")),
        "request_params": safe_dict(raw_result.get("request_params")),
    }


def executed_entry_sizing_reason_tags(ev, sz):
    tags = []
    evidence_tier = str(ev.get("tier") or "").strip()
    if evidence_tier:
        tags.append(f"evidence_tier:{evidence_tier}")
    quality_tier = str(sz.get("quality_tier") or "").strip()
    if quality_tier:
        tags.append(f"sizing_quality:{quality_tier}")
    if sz.get("low_payoff_quality"):
        tags.append("low_payoff_quality")
    for reason in safe_list(sz.get("low_payoff_reasons")):
        if reason:
            tags.append(f"low_payoff:{reason}")
    if sz.get("notional_floor_applied"):
        tags.append("notional_floor_applied")
    if sz.get("notional_floor_blocked"):
        tags.append("notional_floor_blocked")
    if sz.get("strategy_probe_cap_applied"):
        tags.append("strategy_probe_cap_applied")
    if maybe_float(sz.get("strategy_max_probe_size_pct")):
        tags.append("strategy_max_probe_size_pct")
    if str(sz.get("meaningful_size_reason") or "").strip():
        tags.append("meaningful_size_reason")
    if ev.get("shadow_only"):
        tags.append("evidence_shadow_only")
    if ev.get("hard_block"):
        tags.append("evidence_hard_block")
    return tags


def compact_execution_result_for_entry(decision):
    result = order_execution_result(decision)
    return pick(
        result,
        (
            "source",
            "status",
            "exchange_confirmed",
            "okx_symbol",
            "planned_order_contracts",
            "planned_base_quantity",
            "execution_blocker",
            "okx_rejection",
            "system_pre_submit_rejection",
        ),
    )


def _close_action_for_position(pos):
    side = str(getattr(pos, "side", "") or "").lower()
    return "close_long" if side == "long" else "close_short"


def _close_order_side_for_position(pos):
    side = str(getattr(pos, "side", "") or "").lower()
    return "sell" if side == "long" else "buy"


def _decision_has_exit_context(decision):
    raw = safe_dict(getattr(decision, "raw_llm_response", None))
    if not raw:
        return False
    return bool(
        raw.get("close_evidence")
        or raw.get("position_release_policy")
        or raw.get("exit_quality")
        or raw.get("exit_intent")
        or raw.get("fast_risk_trigger")
        or raw.get("forced_exit")
        or raw.get("execution_profit_protection")
    )


def _close_order_candidates_for_position(pos, orders):
    symbol = str(getattr(pos, "symbol", "") or "")
    close_side = _close_order_side_for_position(pos)
    opened = aware(getattr(pos, "created_at", None))
    closed = aware(getattr(pos, "closed_at", None))
    candidates = []
    for order in orders:
        if str(getattr(order, "symbol", "") or "") != symbol:
            continue
        if str(getattr(order, "side", "") or "").lower() != close_side:
            continue
        if str(getattr(order, "status", "") or "").lower() != "filled":
            continue
        order_time = aware(getattr(order, "filled_at", None)) or aware(
            getattr(order, "created_at", None)
        )
        if opened and order_time and order_time < opened - timedelta(minutes=2):
            continue
        if closed and order_time and order_time > closed + timedelta(minutes=5):
            continue
        candidates.append(order)
    candidates.sort(
        key=lambda order: abs(
            (
                (aware(getattr(order, "filled_at", None)) or aware(getattr(order, "created_at", None)) or closed or now)
                - (closed or now)
            ).total_seconds()
        )
    )
    return candidates


def _matching_close_decision_for_position(pos, orders, decision_by_id, decisions):
    fallback_decision = None
    fallback_order = None
    for order in _close_order_candidates_for_position(pos, orders):
        decision = decision_by_id.get(getattr(order, "decision_id", None))
        if decision and _decision_has_exit_context(decision):
            return decision, order, "close_order_decision"
        if decision and fallback_decision is None:
            fallback_decision = decision
            fallback_order = order

    symbol = str(getattr(pos, "symbol", "") or "")
    close_action = _close_action_for_position(pos)
    opened = aware(getattr(pos, "created_at", None))
    closed = aware(getattr(pos, "closed_at", None))
    nearby = []
    for decision in decisions:
        if str(getattr(decision, "symbol", "") or "") != symbol:
            continue
        if str(getattr(decision, "action", "") or "").lower() != close_action:
            continue
        created = aware(getattr(decision, "created_at", None))
        if opened and created and created < opened - timedelta(minutes=2):
            continue
        if closed and created and created > closed + timedelta(minutes=5):
            continue
        nearby.append(decision)
    nearby.sort(
        key=lambda decision: abs(
            ((aware(getattr(decision, "created_at", None)) or closed or now) - (closed or now)).total_seconds()
        )
    )
    for decision in nearby:
        if _decision_has_exit_context(decision):
            return decision, fallback_order, "nearby_exit_decision"
    if fallback_decision:
        return fallback_decision, fallback_order, "close_order_decision_without_raw"
    return None, fallback_order, "position_fields"


def _exit_trigger_from_decision(decision):
    if decision is None:
        return "unknown"
    raw = safe_dict(getattr(decision, "raw_llm_response", None))
    close_evidence = safe_dict(raw.get("close_evidence"))
    release_policy = safe_dict(raw.get("position_release_policy"))
    fast_trigger = str(raw.get("fast_risk_trigger") or "").strip().lower()
    if fast_trigger:
        return f"fast_risk:{fast_trigger}"
    release_reason = str(release_policy.get("release_reason") or "").strip()
    release_source = str(
        release_policy.get("source") or close_evidence.get("source") or ""
    ).strip()
    if release_reason:
        return f"position_release:{release_reason}"
    if release_source:
        return f"position_release:{release_source}"
    if close_evidence.get("profit_retrace_protection"):
        return "profit_lock:retrace"
    if (
        close_evidence.get("profit_protection")
        or close_evidence.get("profit_lock_ready_for_exit")
        or close_evidence.get("portfolio_focus_profit_lock")
        or safe_dict(raw.get("execution_profit_protection")).get("allow")
    ):
        return "profit_lock"
    if raw.get("forced_exit") or close_evidence.get("hard_risk") or close_evidence.get("raw_hard_risk"):
        return "hard_risk"
    if (
        close_evidence.get("predictive_exit")
        or close_evidence.get("predictive_reversal_exit")
        or close_evidence.get("strong_opposite_pressure")
        or close_evidence.get("moderate_opposite_pressure")
    ):
        return "predictive_downside"
    exit_intent = str(raw.get("exit_intent") or close_evidence.get("exit_intent") or "").strip()
    if exit_intent:
        return f"exit_intent:{exit_intent}"
    action = str(getattr(decision, "action", "") or "").lower()
    if action in {"close_long", "close_short"}:
        return "system_exit"
    return "unknown"


def closed_position_pnl_diagnostics(closed_rows, orders, decision_by_id, decisions):
    raw_closed_count = len(closed_rows)
    trade_fact_quarantine_reasons = Counter()
    trusted_closed_rows = []
    for pos in closed_rows:
        reason = closed_position_trade_fact_untrusted_reason(pos)
        if reason:
            trade_fact_quarantine_reasons[reason] += 1
            continue
        trusted_closed_rows.append(pos)
    closed_rows = trusted_closed_rows
    pnl_values = []
    hold_minutes = []
    winning_pnls = []
    losing_pnls = []
    symbol_counts = Counter()
    symbol_pnl = Counter()
    side_counts = Counter()
    side_pnl = Counter()
    symbol_loss_counts = Counter()
    side_loss_counts = Counter()
    trigger_counts = Counter()
    fast_loss_count = 0
    samples = []

    for pos in closed_rows:
        created = aware(getattr(pos, "created_at", None))
        closed_at = aware(getattr(pos, "closed_at", None))
        hold_min = None
        if created and closed_at:
            hold_min = max((closed_at - created).total_seconds() / 60.0, 0.0)
            hold_minutes.append(hold_min)
        realized = safe_float(getattr(pos, "realized_pnl", None))
        pnl_values.append(realized)
        symbol = str(getattr(pos, "symbol", "") or "unknown")
        side = str(getattr(pos, "side", "") or "unknown")
        symbol_counts[symbol] += 1
        symbol_pnl[symbol] += realized
        side_counts[side] += 1
        side_pnl[side] += realized
        close_decision, close_order, trigger_source = _matching_close_decision_for_position(
            pos,
            orders,
            decision_by_id,
            decisions,
        )
        trigger = _exit_trigger_from_decision(close_decision)
        if trigger == "unknown":
            trigger = str(
            getattr(pos, "close_reason", None)
            or getattr(pos, "exit_reason", None)
            or getattr(pos, "status", None)
            or "unknown"
            )
        trigger_counts[trigger] += 1
        if realized > 0:
            winning_pnls.append(realized)
        elif realized < 0:
            losing_pnls.append(realized)
            symbol_loss_counts[symbol] += 1
            side_loss_counts[side] += 1
            if hold_min is not None and hold_min <= FAST_CLOSE_MINUTES:
                fast_loss_count += 1
        if len(samples) < 12:
            notional = safe_float(getattr(pos, "quantity", None)) * safe_float(
                getattr(pos, "entry_price", None)
            )
            samples.append(
                {
                    "id": getattr(pos, "id", None),
                    "symbol": symbol,
                    "side": side,
                    "hold_minutes": roundv(hold_min),
                    "quantity": roundv(getattr(pos, "quantity", None)),
                    "entry_price": roundv(getattr(pos, "entry_price", None)),
                    "close_price": roundv(
                        getattr(pos, "close_price", None)
                        or getattr(pos, "current_price", None)
                    ),
                    "notional_usdt": roundv(notional),
                    "realized_pnl": roundv(realized),
                    "created_at": created.isoformat() if created else "",
                    "closed_at": closed_at.isoformat() if closed_at else "",
                    "trigger": trigger,
                    "trigger_source": trigger_source,
                    "close_decision_id": getattr(close_decision, "id", None),
                    "close_order_id": getattr(close_order, "id", None),
                }
            )

    total_pnl = sum(pnl_values)
    closed_count = len(closed_rows)
    win_count = sum(1 for value in pnl_values if value > 0)
    loss_count = sum(1 for value in pnl_values if value < 0)
    flat_count = closed_count - win_count - loss_count
    gross_profit = sum(winning_pnls)
    gross_loss_abs = abs(sum(losing_pnls))
    profit_factor = None if gross_loss_abs <= 0 else gross_profit / gross_loss_abs

    return {
        "read_only": True,
        "closed_count": closed_count,
        "raw_closed_count": raw_closed_count,
        "trade_fact_quarantined_count": raw_closed_count - closed_count,
        "trade_fact_quarantine_reasons": dict(trade_fact_quarantine_reasons),
        "win_count": win_count,
        "loss_count": loss_count,
        "flat_count": flat_count,
        "win_rate": roundv(win_count / closed_count) if closed_count else 0.0,
        "total_realized_pnl": roundv(total_pnl),
        "avg_realized_pnl": roundv(total_pnl / closed_count) if closed_count else 0.0,
        "gross_profit": roundv(gross_profit),
        "gross_loss_abs": roundv(gross_loss_abs),
        "profit_factor": round_optional(profit_factor),
        "realized_pnl_stats": stats(pnl_values),
        "hold_minutes_stats": stats(hold_minutes),
        "fast_loss_close_under_15m": fast_loss_count,
        "symbol_counts": symbol_counter_rows(symbol_counts, 10),
        "symbol_loss_counts": symbol_counter_rows(symbol_loss_counts, 10),
        "symbol_pnl": [
            {"symbol": symbol, "realized_pnl": roundv(pnl)}
            for symbol, pnl in symbol_pnl.most_common(10)
        ],
        "worst_symbol_pnl": [
            {"symbol": symbol, "realized_pnl": roundv(pnl)}
            for symbol, pnl in sorted(symbol_pnl.items(), key=lambda item: item[1])[:10]
        ],
        "side_counts": counter_rows(side_counts, 10),
        "side_loss_counts": counter_rows(side_loss_counts, 10),
        "side_pnl": [
            {"side": side, "realized_pnl": roundv(pnl)}
            for side, pnl in side_pnl.most_common(10)
        ],
        "worst_side_pnl": [
            {"side": side, "realized_pnl": roundv(pnl)}
            for side, pnl in sorted(side_pnl.items(), key=lambda item: item[1])[:10]
        ],
        "trigger_counts": counter_rows(trigger_counts, 10),
        "samples": samples,
        "diagnostic_boundary": (
            "Read-only closed-position realized PnL diagnostics. Use this to decide "
            "whether the system is actually making money after closes before changing "
            "entry evidence, sizing, leverage, ML readiness, or exit rules."
        ),
    }


def executed_entry_sizing_diagnostics(entry_rows, order_by_decision):
    rows = [row for row in entry_rows if bool(getattr(row, "was_executed", False))]
    order_status_counts = Counter()
    evidence_tier_counts = Counter()
    quality_tier_counts = Counter()
    reason_tag_counts = Counter()
    order_notionals = []
    sizing_notionals = []
    fill_ratios = []
    size_values = []
    leverage_values = []
    expected_nets = []
    profit_qualities = []
    loss_probabilities = []
    tail_risks = []
    filled_order_count = 0
    missing_order_count = 0
    samples = []

    for decision in rows:
        ev = evidence(decision)
        sz = sizing(decision)
        order = order_by_decision.get(decision.id)
        order_status = str(getattr(order, "status", "") or "missing_order").lower()
        order_status_counts[order_status] += 1
        if order is None:
            missing_order_count += 1
        elif order_status == "filled":
            filled_order_count += 1

        evidence_tier = str(ev.get("tier") or "unknown")
        quality_tier = str(sz.get("quality_tier") or "unknown")
        evidence_tier_counts[evidence_tier] += 1
        quality_tier_counts[quality_tier] += 1

        tags = executed_entry_sizing_reason_tags(ev, sz)
        for tag in tags:
            reason_tag_counts[tag] += 1

        order_notional_value = order_notional(order) if order is not None else None
        sizing_notional_value = maybe_float(sz.get("final_notional_usdt"))
        size_value = maybe_float(sz.get("position_size_pct"))
        leverage_value = maybe_float(sz.get("leverage"))
        expected_net_value = maybe_float(ev.get("expected_net_return_pct"))
        profit_quality_value = maybe_float(ev.get("profit_quality_ratio"))
        loss_probability_value = maybe_float(ev.get("loss_probability"))
        tail_risk_value = maybe_float(ev.get("tail_risk_score"))

        if order_notional_value is not None:
            order_notionals.append(order_notional_value)
        if sizing_notional_value is not None:
            sizing_notionals.append(sizing_notional_value)
        if (
            order_notional_value is not None
            and sizing_notional_value is not None
            and sizing_notional_value > 0
        ):
            fill_ratios.append(order_notional_value / sizing_notional_value)
        if size_value is not None:
            size_values.append(size_value)
        if leverage_value is not None:
            leverage_values.append(leverage_value)
        if expected_net_value is not None:
            expected_nets.append(expected_net_value)
        if profit_quality_value is not None:
            profit_qualities.append(profit_quality_value)
        if loss_probability_value is not None:
            loss_probabilities.append(loss_probability_value)
        if tail_risk_value is not None:
            tail_risks.append(tail_risk_value)

        if len(samples) < 20:
            created_at = aware(decision.created_at)
            executed_at = aware(decision.executed_at)
            filled_at = aware(getattr(order, "filled_at", None)) if order else None
            order_payload = None
            if order is not None:
                order_payload = {
                    "id": order.id,
                    "status": order.status,
                    "side": order.side,
                    "quantity": roundv(order.quantity),
                    "price": roundv(order.price),
                    "notional": roundv(order_notional_value),
                    "filled_at": filled_at.isoformat() if filled_at else "",
                }
            notional_gap = None
            fill_ratio = None
            if order_notional_value is not None and sizing_notional_value is not None:
                notional_gap = order_notional_value - sizing_notional_value
                if sizing_notional_value > 0:
                    fill_ratio = order_notional_value / sizing_notional_value
            samples.append(
                {
                    "id": decision.id,
                    "time": created_at.isoformat() if created_at else "",
                    "symbol": decision.symbol,
                    "action": decision.action,
                    "analysis_type": analysis_type(decision),
                    "decision": {
                        "position_size_pct": roundv(size_value),
                        "suggested_leverage": roundv(leverage_value),
                        "was_executed": bool(decision.was_executed),
                        "executed_at": executed_at.isoformat() if executed_at else "",
                        "execution_price": roundv(decision.execution_price),
                    },
                    "order": order_payload,
                    "evidence": pick(
                        ev,
                        (
                            "tier",
                            "effective_score",
                            "entry_evidence_score",
                            "expected_net_return_pct",
                            "aggregate_expected_net_return_pct",
                            "ai_expected_return_policy",
                            "ai_expected_return_weight",
                            "ai_expected_return_independent_probe_support",
                            "profit_quality_ratio",
                            "loss_probability",
                            "tail_risk_score",
                            "tradeable_probe",
                            "shadow_only",
                        ),
                    ),
                    "sizing": pick(
                        sz,
                        (
                            "position_size_pct",
                            "leverage",
                            "final_notional_usdt",
                            "quality_tier",
                            "low_payoff_quality",
                            "low_payoff_reasons",
                            "notional_floor_applied",
                            "notional_floor_blocked",
                            "meaningful_size_reason",
                            "strategy_probe_cap_applied",
                            "strategy_max_probe_size_pct",
                            "strategy_reason",
                        ),
                    ),
                    "sizing_reason_tags": tags[:10],
                    "notional_gap_usdt": roundv(notional_gap),
                    "notional_fill_ratio": roundv(fill_ratio),
                    "execution_result": compact_execution_result_for_entry(decision),
                }
            )

    return {
        "read_only": True,
        "executed_entry_count": len(rows),
        "market_executed_entry_count": sum(
            1 for row in rows if analysis_type(row) == "market"
        ),
        "filled_order_count": filled_order_count,
        "missing_order_count": missing_order_count,
        "order_status_counts": dict(order_status_counts.most_common(12)),
        "evidence_tier_counts": dict(evidence_tier_counts.most_common(12)),
        "sizing_quality_tier_counts": dict(quality_tier_counts.most_common(12)),
        "sizing_reason_tag_counts": dict(reason_tag_counts.most_common(20)),
        "order_notional_stats": stats(order_notionals),
        "sizing_final_notional_stats": stats(sizing_notionals),
        "notional_fill_ratio_stats": stats(fill_ratios),
        "decision_position_size_pct_stats": stats(size_values),
        "decision_leverage_stats": stats(leverage_values),
        "expected_net_stats": stats(expected_nets),
        "profit_quality_stats": stats(profit_qualities),
        "loss_probability_stats": stats(loss_probabilities),
        "tail_risk_stats": stats(tail_risks),
        "samples": samples,
        "diagnostic_boundary": (
            "Read-only executed-entry sizing/order diagnostics; use this to explain "
            "small filled orders, leverage differences, and weak payoff/risk context "
            "before changing sizing, leverage, evidence, ML, OKX, or risk gates."
        ),
    }


def stats(vals):
    vals = [safe_float(v) for v in vals]
    if not vals:
        return {"count": 0}
    vals_sorted = sorted(vals)
    return {
        "count": len(vals),
        "min": roundv(vals_sorted[0]),
        "p25": roundv(vals_sorted[len(vals_sorted)//4]),
        "median": roundv(vals_sorted[len(vals_sorted)//2]),
        "p75": roundv(vals_sorted[(len(vals_sorted)*3)//4]),
        "max": roundv(vals_sorted[-1]),
        "positive": sum(1 for v in vals if v > 0),
        "zero": sum(1 for v in vals if abs(v) < 1e-12),
        "negative": sum(1 for v in vals if v < 0),
    }


def stats_present(vals):
    vals = [maybe_float(v) for v in vals]
    vals = [v for v in vals if v is not None]
    return stats(vals)


def pick(mapping, keys):
    mapping = safe_dict(mapping)
    return {key: mapping.get(key) for key in keys}


def counter_rows(counter, limit=20):
    return [
        {"value": str(key), "count": int(count)}
        for key, count in counter.most_common(limit)
    ]


def symbol_counter_rows(counter, limit=20):
    return [
        {"symbol": str(symbol), "count": int(count)}
        for symbol, count in counter.most_common(limit)
    ]


def top_share(counter, top_n=3):
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return roundv(sum(count for _key, count in counter.most_common(top_n)) / total)


def compact_text_counter(mapping, limit=6, text_limit=90):
    counter = Counter()
    for key, count in safe_dict(mapping).items():
        counter[short_text(key, text_limit)] += int(safe_float(count, 0))
    return counter_rows(counter, limit)


def compact_closed_position_pnl_diagnostics(value):
    value = safe_dict(value)
    compact = pick(
        value,
        (
            "read_only",
            "closed_count",
            "win_count",
            "loss_count",
            "flat_count",
            "win_rate",
            "total_realized_pnl",
            "avg_realized_pnl",
            "gross_profit",
            "gross_loss_abs",
            "profit_factor",
            "realized_pnl_stats",
            "hold_minutes_stats",
            "fast_loss_close_under_15m",
            "symbol_counts",
            "symbol_loss_counts",
            "symbol_pnl",
            "worst_symbol_pnl",
            "side_counts",
            "side_loss_counts",
            "side_pnl",
            "worst_side_pnl",
            "trigger_counts",
            "diagnostic_boundary",
        ),
    )
    compact["samples"] = safe_list(value.get("samples"))[:5]
    return compact


def candidate_funnel_aggregate(funnels):
    funnels = [safe_dict(funnel) for funnel in funnels if isinstance(funnel, dict)]
    if not funnels:
        return {"count": 0}
    numeric_keys = (
        "scan_symbol_count",
        "feature_fetch_requested_count",
        "feature_valid_count",
        "feature_invalid_count",
        "market_feature_before_rank_count",
        "market_symbol_budget",
        "rank_selected_count",
        "rank_tradable_candidates",
        "rank_secondary_candidates",
        "rank_filtered_out_candidates",
        "recent_analysis_dedupe_count",
        "market_feature_after_dedupe_count",
    )
    budget_sources = Counter()
    budget_policies = Counter()
    underfill_reasons = Counter()
    filtered_reasons = Counter()
    selected_symbols = Counter()
    filtered_symbols = Counter()
    outside_budget_symbols = Counter()
    for funnel in funnels:
        budget = safe_dict(funnel.get("analysis_budget"))
        budget_sources[str(budget.get("budget_source") or "unknown")] += 1
        budget_policies[str(budget.get("market_limit_policy") or "unknown")] += 1
        if funnel.get("rank_underfilled"):
            underfill_reasons[str(funnel.get("rank_underfill_reason") or "unknown")] += 1
        for item in safe_list(funnel.get("rank_filtered_out_reason_counts")):
            if isinstance(item, dict):
                filtered_reasons[str(item.get("reason") or "unknown")] += int(
                    safe_float(item.get("count"), 0)
                )
        for item in safe_list(funnel.get("ranked_symbol_sample")):
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol") or "unknown")
            if item.get("selected"):
                selected_symbols[symbol] += 1
            elif item.get("non_selected_reason") == "outside_market_symbol_budget":
                outside_budget_symbols[symbol] += 1
        for item in safe_list(funnel.get("filtered_symbol_sample")):
            if isinstance(item, dict):
                filtered_symbols[str(item.get("symbol") or "unknown")] += 1
    return {
        "count": len(funnels),
        "metric_stats": {key: stats([funnel.get(key) for funnel in funnels]) for key in numeric_keys},
        "rank_underfilled_count": sum(1 for funnel in funnels if funnel.get("rank_underfilled")),
        "budget_source_counts": counter_rows(budget_sources, 10),
        "market_limit_policy_counts": counter_rows(budget_policies, 10),
        "rank_underfill_reason_counts": counter_rows(underfill_reasons, 10),
        "filtered_out_reason_counts": counter_rows(filtered_reasons, 12),
        "selected_symbol_counts": symbol_counter_rows(selected_symbols, 12),
        "outside_budget_symbol_counts": symbol_counter_rows(outside_budget_symbols, 12),
        "filtered_symbol_counts": symbol_counter_rows(filtered_symbols, 12),
        "diagnostic_boundary": (
            "Read-only aggregate over recent market candidate funnels; use it to confirm "
            "whether latest-funnel bottlenecks persist across the window before changing "
            "ranker, budget, evidence, sizing, leverage, or ML gates."
        ),
    }


def market_analysis_progress(decision):
    raw = safe_dict(decision.raw_llm_response)
    return safe_dict(raw.get("market_analysis_progress"))


def market_analysis_progress_aggregate(decisions):
    rows = [market_analysis_progress(decision) for decision in decisions]
    rows = [row for row in rows if row]
    if not rows:
        return {
            "read_only": True,
            "count": 0,
            "diagnostic_boundary": (
                "Read-only market AI throughput aggregate; empty means the window has no "
                "new market decisions carrying throughput diagnostics yet."
            ),
        }
    latest = rows[0]
    return {
        "read_only": True,
        "count": len(rows),
        "symbol_counts": symbol_counter_rows(
            Counter(str(row.get("symbol") or "unknown") for row in rows),
            8,
        ),
        "processed_index_stats": stats(
            [safe_float(row.get("processed_index")) for row in rows]
        ),
        "ranked_market_symbol_count_stats": stats(
            [safe_float(row.get("ranked_market_symbol_count")) for row in rows]
        ),
        "remaining_after_symbol_stats": stats(
            [safe_float(row.get("remaining_after_this_symbol")) for row in rows]
        ),
        "round_elapsed_before_ai_stats": stats(
            [safe_float(row.get("round_elapsed_seconds_before_ai")) for row in rows]
        ),
        "full_round_elapsed_before_ai_stats": stats_present(
            [row.get("full_round_elapsed_seconds_before_ai") for row in rows]
        ),
        "market_ai_elapsed_before_symbol_stats": stats_present(
            [row.get("market_ai_elapsed_seconds_before_symbol") for row in rows]
        ),
        "market_round_time_budget_stats": stats(
            [safe_float(row.get("market_round_time_budget_seconds")) for row in rows]
        ),
        "budget_used_ratio_before_ai_stats": stats(
            [safe_float(row.get("budget_used_ratio_before_ai")) for row in rows]
        ),
        "market_ai_budget_used_ratio_before_symbol_stats": stats_present(
            [row.get("market_ai_budget_used_ratio_before_symbol") for row in rows]
        ),
        "latest": pick(
            latest,
            (
                "symbol",
                "processed_index",
                "ranked_market_symbol_count",
                "remaining_after_this_symbol",
                "round_elapsed_seconds_before_ai",
                "full_round_elapsed_seconds_before_ai",
                "market_ai_elapsed_seconds_before_symbol",
                "market_round_time_budget_seconds",
                "budget_used_ratio_before_ai",
                "market_ai_budget_used_ratio_before_symbol",
                "budget_clock_scope",
                "diagnostic_boundary",
            ),
        ),
        "diagnostic_boundary": (
            "Read-only aggregate showing how much of the ranked market shortlist reached "
            "AI analysis in the window. Use it to explain repeated symbols or narrow AI "
            "coverage before changing ranker, budget, evidence, sizing, leverage, or risk gates."
        ),
    }


def normalized_symbol(value):
    return str(value or "").upper().strip()


def candidate_filter_outcome_diagnostics(funnel_decisions, market_entry_decisions):
    candidate_rows = {}
    category_counts = Counter()
    reason_counts = Counter()

    def upsert(symbol_value, category, item, seen_at):
        symbol = normalized_symbol(symbol_value)
        if not symbol or symbol == "UNKNOWN":
            return
        row = candidate_rows.setdefault(
            symbol,
            {
                "symbol": symbol,
                "sample_count": 0,
                "categories": Counter(),
                "reason_counts": Counter(),
                "first_seen_at": seen_at,
                "last_seen_at": seen_at,
            },
        )
        row["sample_count"] += 1
        row["categories"][category] += 1
        category_counts[category] += 1
        if seen_at is not None:
            first_seen = row.get("first_seen_at")
            last_seen = row.get("last_seen_at")
            if first_seen is None or seen_at < first_seen:
                row["first_seen_at"] = seen_at
            if last_seen is None or seen_at > last_seen:
                row["last_seen_at"] = seen_at
        raw_reasons = list(safe_list(item.get("filter_reasons")))
        non_selected_reason = str(item.get("non_selected_reason") or "").strip()
        if non_selected_reason:
            raw_reasons.append(non_selected_reason)
        for reason in raw_reasons:
            if reason:
                row["reason_counts"][str(reason)] += 1
                reason_counts[str(reason)] += 1

    for decision in funnel_decisions:
        funnel = safe_dict(safe_dict(decision.raw_llm_response).get("market_candidate_funnel"))
        seen_at = aware(decision.created_at)
        for item in safe_list(funnel.get("ranked_symbol_sample")):
            if not isinstance(item, dict) or item.get("selected"):
                continue
            reason = str(item.get("non_selected_reason") or "").strip()
            if reason == "outside_market_symbol_budget":
                category = "outside_market_symbol_budget"
            elif reason == "feature_filter_rejected":
                category = "feature_filter_rejected"
            else:
                category = "non_selected_other"
            upsert(item.get("symbol"), category, item, seen_at)
        for item in safe_list(funnel.get("filtered_symbol_sample")):
            if isinstance(item, dict):
                upsert(item.get("symbol"), "feature_filter_rejected", item, seen_at)

    if not candidate_rows:
        return {
            "read_only": True,
            "sampled_symbol_count": 0,
            "market_entry_after_filter_count": 0,
            "diagnostic_boundary": (
                "Read-only outcome replay for sampled non-selected candidate symbols; "
                "empty means the current window had no ranker non-selected samples."
            ),
        }

    entry_after_filter_count = 0
    positive_expected_net_count = 0
    executed_count = 0
    outcome_symbols = Counter()
    positive_symbols = Counter()
    skip_counts = Counter()
    tier_counts = Counter()
    expected_net_values = []
    examples = []
    for decision in market_entry_decisions:
        symbol = normalized_symbol(decision.symbol)
        row = candidate_rows.get(symbol)
        if not row:
            continue
        created_at = aware(decision.created_at)
        first_seen = row.get("first_seen_at")
        if created_at is not None and first_seen is not None and created_at < first_seen:
            continue
        ev = evidence(decision)
        net = safe_float(ev.get("expected_net_return_pct"))
        tier = str(ev.get("tier") or "unknown")
        skip_kind = entry_skip_kind(decision)
        entry_after_filter_count += 1
        expected_net_values.append(net)
        outcome_symbols[symbol] += 1
        skip_counts[skip_kind] += 1
        tier_counts[tier] += 1
        if net > 0:
            positive_expected_net_count += 1
            positive_symbols[symbol] += 1
        if bool(decision.was_executed):
            executed_count += 1
        if len(examples) < 10:
            examples.append(
                {
                    "id": decision.id,
                    "time": created_at.isoformat() if created_at else "",
                    "symbol": decision.symbol,
                    "action": decision.action,
                    "expected_net_return_pct": roundv(net),
                    "evidence_tier": tier,
                    "skip_kind": skip_kind,
                    "executed": bool(decision.was_executed),
                    "non_selected_categories": counter_rows(row["categories"], 6),
                    "non_selected_reasons": counter_rows(row["reason_counts"], 6),
                }
            )

    sampled_counter = Counter(
        {symbol: int(row.get("sample_count") or 0) for symbol, row in candidate_rows.items()}
    )
    symbol_examples = []
    for symbol, _count in sampled_counter.most_common(8):
        row = candidate_rows[symbol]
        first_seen = row.get("first_seen_at")
        last_seen = row.get("last_seen_at")
        symbol_examples.append(
            {
                "symbol": symbol,
                "sample_count": int(row.get("sample_count") or 0),
                "categories": counter_rows(row["categories"], 6),
                "reason_counts": counter_rows(row["reason_counts"], 6),
                "first_seen_at": first_seen.isoformat() if first_seen else "",
                "last_seen_at": last_seen.isoformat() if last_seen else "",
            }
        )

    return {
        "read_only": True,
        "sampled_symbol_count": len(candidate_rows),
        "sampled_occurrence_count": sum(sampled_counter.values()),
        "category_counts": counter_rows(category_counts, 10),
        "reason_counts": counter_rows(reason_counts, 12),
        "sampled_symbol_counts": symbol_counter_rows(sampled_counter, 12),
        "market_entry_after_filter_count": entry_after_filter_count,
        "market_entry_after_filter_symbol_count": len(outcome_symbols),
        "positive_expected_net_after_filter_count": positive_expected_net_count,
        "executed_after_filter_count": executed_count,
        "outcome_symbol_counts": symbol_counter_rows(outcome_symbols, 12),
        "positive_expected_net_symbol_counts": symbol_counter_rows(positive_symbols, 12),
        "skip_kind_counts": counter_rows(skip_counts, 12),
        "evidence_tier_counts": counter_rows(tier_counts, 12),
        "expected_net_stats": stats(expected_net_values),
        "symbol_examples": symbol_examples,
        "market_entry_examples": examples,
        "diagnostic_boundary": (
            "Read-only replay joining sampled non-selected ranker symbols with later "
            "market entry decisions in the same window. Positive outcomes here only "
            "justify deeper offline review; they do not relax evidence, sizing, leverage, "
            "ML readiness, or risk gates."
        ),
    }


def summary_report(report):
    local_ml = safe_dict(report.get("local_ml_readiness"))
    contract = safe_dict(report.get("trade_execution_contract"))
    return {
        "window_minutes": report.get("window_minutes"),
        "generated_at": report.get("generated_at"),
        "counts": pick(
            report.get("counts"),
            (
                "decisions",
                "orders",
                "filled_orders",
                "failed_orders",
                "rejected_orders",
                "pending_or_open_orders",
                "positions_created",
                "positions_closed",
                "open_positions",
                "fast_loss_close_under_15m",
            ),
        ),
        "order_status_counts": safe_dict(report.get("order_status_counts")),
        "trade_execution_contract": {
            "status": contract.get("status"),
            "audit_only": contract.get("audit_only"),
            "can_bypass_risk_controls": contract.get("can_bypass_risk_controls"),
            "summary": safe_dict(contract.get("summary")),
            "violation_reason_counts": safe_dict(
                contract.get("violation_reason_counts")
            ),
            "fast_loss_samples": safe_list(contract.get("fast_loss_samples"))[:5],
            "violations": safe_list(contract.get("violations"))[:5],
        },
        "local_ml_readiness": {
            "status": local_ml.get("status"),
            "readiness_state": local_ml.get("readiness_state"),
            "allow_live_position_influence": local_ml.get(
                "allow_live_position_influence"
            ),
            "blocking_reason_codes": safe_list(local_ml.get("blocking_reason_codes")),
            "metrics": safe_dict(local_ml.get("metrics")),
            "training_window_composition": safe_dict(
                local_ml.get("training_window_composition")
            ),
        },
        "market_symbol_diagnostics": compact_market_symbol_diagnostics(
            report.get("market_symbol_diagnostics"),
            include_latest=False,
        ),
        "closed_position_pnl_diagnostics": compact_closed_position_pnl_diagnostics(
            report.get("closed_position_pnl_diagnostics")
        ),
        "rejected_order_examples": safe_list(report.get("rejected_order_examples"))[:5],
        "fast_loss_positions": safe_list(report.get("fast_loss_positions"))[:5],
    }


def compact_candidate_funnel(funnel):
    funnel = safe_dict(funnel)

    def compact_market_limit_diagnostics(value):
        value = safe_dict(value)
        return pick(
            value,
            (
                "read_only",
                "is_entry_gate",
                "budget_source",
                "strategy_profile_id",
                "risk_level",
                "market_limit_policy",
                "configured_market_symbol_limit",
                "selected_market_symbol_limit",
                "position_group_count",
                "total_position_groups",
                "target_position_groups",
                "roster_underfilled",
            ),
        )

    def compact_budget(value):
        value = safe_dict(value)
        result = pick(
            value,
            (
                "budget_source",
                "market_limit_policy",
                "market_symbol_limit",
            ),
        )
        result["market_limit_diagnostics"] = compact_market_limit_diagnostics(
            value.get("market_limit_diagnostics")
        )
        return result

    def compact_rank_item(item):
        item = safe_dict(item)
        metrics = safe_dict(item.get("filter_metrics"))
        return {
            "symbol": item.get("symbol"),
            "score": item.get("score"),
            "net_score": item.get("net_score"),
            "selected": bool(item.get("selected")),
            "non_selected_reason": item.get("non_selected_reason"),
            "selection_tier": item.get("selection_tier"),
            "filter_reasons": safe_list(item.get("filter_reasons")),
            "volume_ratio": item.get("volume_ratio", metrics.get("volume_ratio")),
            "volume_ratio_source": item.get(
                "volume_ratio_source",
                metrics.get("volume_ratio_source"),
            ),
            "trend_volume_ratio": item.get(
                "trend_volume_ratio",
                metrics.get("trend_volume_ratio"),
            ),
            "trend_volume_ratio_timeframe": item.get(
                "trend_volume_ratio_timeframe",
                metrics.get("trend_volume_ratio_timeframe"),
            ),
            "entry_activity_volume_ratio": item.get(
                "entry_activity_volume_ratio",
                metrics.get("entry_activity_volume_ratio"),
            ),
            "entry_activity_volume_timeframe": item.get(
                "entry_activity_volume_timeframe",
                metrics.get("entry_activity_volume_timeframe"),
            ),
            "adx": item.get("adx", metrics.get("adx")),
            "change_24h": item.get("change_24h", metrics.get("change_24h")),
            "notional_24h": metrics.get("notional_24h"),
        }

    compact = pick(
        funnel,
        (
            "read_only",
            "is_entry_gate",
            "mode",
            "run_market_analysis",
            "scan_symbol_count",
            "blocked_filter_count",
            "open_position_filtered_count",
            "unclaimed_filtered_count",
            "feature_fetch_requested_count",
            "feature_valid_count",
            "feature_invalid_count",
            "market_feature_before_rank_count",
            "market_feature_after_dedupe_count",
            "recent_analysis_dedupe_count",
            "market_budget_rotation",
            "market_symbol_budget",
            "rank_selected_count",
            "rank_tradable_candidates",
            "rank_secondary_candidates",
            "rank_total_candidates",
            "rank_underfilled",
            "rank_underfill_reason",
            "rank_filtered_out_candidates",
        ),
    )
    compact["market_budget_rotation"] = safe_dict(funnel.get("market_budget_rotation"))
    compact["rank_filtered_out_reason_counts"] = safe_list(
        funnel.get("rank_filtered_out_reason_counts")
    )[:6]
    for key in ("rank_top_symbols", "ranked_symbol_sample", "filtered_symbol_sample"):
        compact[key] = [
            compact_rank_item(item)
            for item in safe_list(funnel.get(key))[:2]
        ]
    budget = safe_dict(funnel.get("analysis_budget"))
    if budget:
        compact["analysis_budget"] = compact_budget(budget)
    return compact


def compact_candidate_funnel_window(window):
    window = safe_dict(window)
    metric_stats = safe_dict(window.get("metric_stats"))
    metric_keys = (
        "scan_symbol_count",
        "feature_fetch_requested_count",
        "feature_valid_count",
        "feature_invalid_count",
        "market_symbol_budget",
        "rank_selected_count",
        "rank_filtered_out_candidates",
        "recent_analysis_dedupe_count",
    )
    return {
        "count": window.get("count", 0),
        "rank_underfilled_count": window.get("rank_underfilled_count", 0),
        "metric_stats": {
            key: pick(
                safe_dict(metric_stats.get(key)),
                ("count", "median", "p75", "max", "positive", "zero"),
            )
            for key in metric_keys
            if isinstance(metric_stats.get(key), dict)
        },
        "budget_source_counts": safe_list(window.get("budget_source_counts"))[:4],
        "market_limit_policy_counts": safe_list(
            window.get("market_limit_policy_counts")
        )[:4],
        "rank_underfill_reason_counts": safe_list(
            window.get("rank_underfill_reason_counts")
        )[:4],
        "filtered_out_reason_counts": safe_list(
            window.get("filtered_out_reason_counts")
        )[:6],
        "selected_symbol_counts": safe_list(window.get("selected_symbol_counts"))[:6],
        "outside_budget_symbol_counts": safe_list(
            window.get("outside_budget_symbol_counts")
        )[:6],
        "filtered_symbol_counts": safe_list(window.get("filtered_symbol_counts"))[:6],
        "diagnostic_boundary": window.get("diagnostic_boundary"),
    }


def compact_candidate_filter_outcomes(outcomes):
    outcomes = safe_dict(outcomes)
    compact = pick(
        outcomes,
        (
            "read_only",
            "sampled_symbol_count",
            "sampled_occurrence_count",
            "market_entry_after_filter_count",
            "market_entry_after_filter_symbol_count",
            "positive_expected_net_after_filter_count",
            "executed_after_filter_count",
            "expected_net_stats",
            "diagnostic_boundary",
        ),
    )
    for key in (
        "category_counts",
        "reason_counts",
        "sampled_symbol_counts",
        "outcome_symbol_counts",
        "positive_expected_net_symbol_counts",
        "skip_kind_counts",
        "evidence_tier_counts",
    ):
        compact[key] = safe_list(outcomes.get(key))[:6]
    for key in ("symbol_examples", "market_entry_examples"):
        compact[key] = safe_list(outcomes.get(key))[:2]
    return compact


def compact_market_symbol_diagnostics(diagnostics, include_latest=True):
    diagnostics = dict(safe_dict(diagnostics))
    for key in ("market_top_symbols", "market_entry_top_symbols"):
        diagnostics[key] = safe_list(diagnostics.get(key))[:8]
    compact = {
        key: diagnostics.get(key)
        for key in (
            "market_decision_count",
            "market_unique_symbol_count",
            "market_top_symbols",
            "market_entry_count",
            "market_entry_unique_symbol_count",
            "market_entry_top_symbols",
            "market_entry_action_counts",
            "market_entry_skip_kind_counts",
            "market_entry_tier_counts",
            "market_top3_share",
            "market_entry_top3_share",
            "entry_unique_to_market_unique_ratio",
            "candidate_funnel_sample_count",
            "candidate_funnel_window",
            "market_analysis_progress",
            "candidate_filter_outcomes",
            "diagnostic_boundary",
        )
    }
    compact["candidate_filter_outcomes"] = compact_candidate_filter_outcomes(
        diagnostics.get("candidate_filter_outcomes")
    )
    compact["candidate_funnel_window"] = compact_candidate_funnel_window(
        diagnostics.get("candidate_funnel_window")
    )
    compact["market_analysis_progress"] = safe_dict(
        diagnostics.get("market_analysis_progress")
    )
    if include_latest:
        compact["latest_candidate_funnel"] = compact_candidate_funnel(
            diagnostics.get("latest_candidate_funnel")
        )
    return compact


def market_symbol_only_report(report):
    report = safe_dict(report)
    counts = safe_dict(report.get("counts"))
    contract = safe_dict(report.get("trade_execution_contract"))
    local_ml = safe_dict(report.get("local_ml_readiness"))
    diagnostics = compact_market_symbol_diagnostics(
        report.get("market_symbol_diagnostics"),
        include_latest=True,
    )
    return {
        "window_minutes": report.get("window_minutes"),
        "generated_at": report.get("generated_at"),
        "counts": pick(
            counts,
            (
                "decisions",
                "market_decisions",
                "market_entry_decisions",
                "orders",
                "failed_orders",
                "rejected_orders",
                "positions_created",
                "positions_closed",
                "open_positions",
                "fast_loss_close_under_15m",
            ),
        ),
        "trade_execution_contract": {
            "status": contract.get("status"),
            "can_bypass_risk_controls": contract.get("can_bypass_risk_controls"),
            "summary": pick(
                safe_dict(contract.get("summary")),
                (
                    "decision_count",
                    "executed_entry_count",
                    "contract_violation_count",
                    "weak_evidence_executed_count",
                    "negative_expected_executed_count",
                    "fast_loss_count",
                    "fast_loss_without_strong_exit_count",
                    "reentry_without_strong_unlock_count",
                ),
            ),
        },
        "closed_position_pnl_diagnostics": compact_closed_position_pnl_diagnostics(
            report.get("closed_position_pnl_diagnostics")
        ),
        "local_ml_readiness": {
            "status": local_ml.get("status"),
            "readiness_state": local_ml.get("readiness_state"),
            "allow_live_position_influence": local_ml.get(
                "allow_live_position_influence"
            ),
            "blocking_reason_codes": safe_list(local_ml.get("blocking_reason_codes")),
            "metrics": pick(
                safe_dict(local_ml.get("metrics")),
                (
                    "sample_count",
                    "dirty_sample_ratio",
                    "long_pr_auc",
                    "short_pr_auc",
                    "top_long_avg_return_pct",
                    "top_short_avg_return_pct",
                    "training_data_version",
                    "required_training_data_version",
                ),
            ),
            "training_window_composition": safe_dict(
                local_ml.get("training_window_composition")
            ),
        },
        "market_symbol_diagnostics": diagnostics,
        "diagnostic_boundary": (
            "Read-only compact market symbol and candidate funnel diagnostics. "
            "Use this output to explain repeated symbols or no-entry windows before "
            "changing ranker, budget, evidence, sizing, leverage, or ML gates."
        ),
    }


def entry_only_report(report):
    report = safe_dict(report)
    counts = safe_dict(report.get("counts"))
    contract = safe_dict(report.get("trade_execution_contract"))
    local_ml = safe_dict(report.get("local_ml_readiness"))
    examples = [
        compact_entry_example(item)
        for item in safe_list(report.get("entry_examples"))
        if safe_dict(item).get("analysis_type") in {"market", "entry_candidate"}
    ]
    ai_policy_counts = Counter()
    for item in examples:
        policy = safe_dict(item.get("evidence")).get("ai_expected_return_policy") or "missing"
        ai_policy_counts[str(policy)] += 1
    return {
        "window_minutes": report.get("window_minutes"),
        "generated_at": report.get("generated_at"),
        "counts": pick(
            counts,
            (
                "decisions",
                "market_decisions",
                "entry_decisions",
                "market_entry_decisions",
                "executed_entries",
                "orders",
                "filled_orders",
                "failed_orders",
                "rejected_orders",
                "positions_created",
                "positions_closed",
                "open_positions",
                "fast_loss_close_under_15m",
            ),
        ),
        "trade_execution_contract": {
            "status": contract.get("status"),
            "can_bypass_risk_controls": contract.get("can_bypass_risk_controls"),
            "summary": pick(
                safe_dict(contract.get("summary")),
                (
                    "decision_count",
                    "executed_entry_count",
                    "contract_violation_count",
                    "weak_evidence_executed_count",
                    "negative_expected_executed_count",
                    "fast_loss_count",
                    "fast_loss_without_strong_exit_count",
                    "reentry_without_strong_unlock_count",
                ),
            ),
        },
        "local_ml_readiness": {
            "status": local_ml.get("status"),
            "readiness_state": local_ml.get("readiness_state"),
            "allow_live_position_influence": local_ml.get(
                "allow_live_position_influence"
            ),
            "blocking_reason_codes": safe_list(local_ml.get("blocking_reason_codes")),
            "metrics": pick(
                safe_dict(local_ml.get("metrics")),
                (
                    "sample_count",
                    "test_count",
                    "dirty_sample_ratio",
                    "long_pr_auc",
                    "short_pr_auc",
                    "top_long_avg_return_pct",
                    "top_short_avg_return_pct",
                    "training_data_version",
                    "required_training_data_version",
                ),
            ),
            "training_window_composition": safe_dict(
                local_ml.get("training_window_composition")
            ),
        },
        "entry_evidence_thresholds": safe_dict(report.get("entry_evidence_thresholds")),
        "market_entry_final_skip_kind_counts": safe_dict(
            report.get("market_entry_final_skip_kind_counts")
        ),
        "market_entry_evidence_tier_counts": safe_dict(
            report.get("market_entry_evidence_tier_counts")
        ),
        "market_entry_evidence_component_status_counts": safe_dict(
            report.get("market_entry_evidence_component_status_counts")
        ),
        "market_entry_evidence_component_point_stats": safe_dict(
            report.get("market_entry_evidence_component_point_stats")
        ),
        "market_entry_evidence_relief_applied_counts": safe_dict(
            report.get("market_entry_evidence_relief_applied_counts")
        ),
        "market_entry_probe_recommendation_counts": safe_dict(
            report.get("market_entry_probe_recommendation_counts")
        ),
        "market_entry_probe_conversion_ready_counts": safe_dict(
            report.get("market_entry_probe_conversion_ready_counts")
        ),
        "market_entry_probe_conversion_block_reason_counts": safe_dict(
            report.get("market_entry_probe_conversion_block_reason_counts")
        ),
        "market_entry_profit_probe_block_kind_counts": safe_dict(
            report.get("market_entry_profit_probe_block_kind_counts")
        ),
        "market_entry_profit_probe_block_reason_counts": safe_dict(
            report.get("market_entry_profit_probe_block_reason_counts")
        ),
        "market_entry_advisory_wait_reason_counts": safe_dict(
            report.get("market_entry_advisory_wait_reason_counts")
        ),
        "market_entry_expected_net_component_stats": safe_dict(
            report.get("market_entry_expected_net_component_stats")
        ),
        "market_entry_score_gap_stats": safe_dict(
            report.get("market_entry_score_gap_stats")
        ),
        "market_entry_evidence_effective_score_stats": safe_dict(
            report.get("market_entry_evidence_effective_score_stats")
        ),
        "market_entry_profit_quality_stats": safe_dict(
            report.get("market_entry_profit_quality_stats")
        ),
        "market_entry_loss_probability_stats": safe_dict(
            report.get("market_entry_loss_probability_stats")
        ),
        "market_entry_tail_risk_stats": safe_dict(
            report.get("market_entry_tail_risk_stats")
        ),
        "position_size_pct_stats": safe_dict(report.get("position_size_pct_stats")),
        "quality_tier_counts": safe_dict(report.get("quality_tier_counts")),
        "low_payoff_entry_count": report.get("low_payoff_entry_count"),
        "low_payoff_reason_counts": safe_dict(report.get("low_payoff_reason_counts")),
        "low_payoff_missing_reason_count": report.get("low_payoff_missing_reason_count"),
        "entry_analysis_type_counts": safe_dict(report.get("entry_analysis_type_counts")),
        "entry_analysis_type_skip_kind_counts": safe_dict(
            report.get("entry_analysis_type_skip_kind_counts")
        ),
        "entry_analysis_type_evidence_tier_counts": safe_dict(
            report.get("entry_analysis_type_evidence_tier_counts")
        ),
        "entry_analysis_type_quality_tier_counts": safe_dict(
            report.get("entry_analysis_type_quality_tier_counts")
        ),
        "entry_analysis_type_low_payoff_counts": safe_dict(
            report.get("entry_analysis_type_low_payoff_counts")
        ),
        "entry_analysis_type_low_payoff_reason_counts": safe_dict(
            report.get("entry_analysis_type_low_payoff_reason_counts")
        ),
        "entry_analysis_type_low_payoff_missing_reason_counts": safe_dict(
            report.get("entry_analysis_type_low_payoff_missing_reason_counts")
        ),
        "entry_analysis_type_notional_floor_blocked_counts": compact_text_counter(
            report.get("entry_analysis_type_notional_floor_blocked_counts")
        ),
        "entry_analysis_type_metric_stats": safe_dict(
            report.get("entry_analysis_type_metric_stats")
        ),
        "high_risk_review_status_counts": safe_dict(
            report.get("high_risk_review_status_counts")
        ),
        "high_risk_review_trigger_counts": safe_dict(
            report.get("high_risk_review_trigger_counts")
        ),
        "high_risk_review_approved_counts": safe_dict(
            report.get("high_risk_review_approved_counts")
        ),
        "high_risk_review_reason_counts": compact_text_counter(
            report.get("high_risk_review_reason_counts")
        ),
        "high_risk_review_trigger_reason_counts": safe_dict(
            report.get("high_risk_review_trigger_reason_counts")
        ),
        "notional_floor_blocked_counts": compact_text_counter(
            report.get("notional_floor_blocked_counts")
        ),
        "entry_ai_expected_return_policy_counts": dict(
            ai_policy_counts.most_common(12)
        ),
        "executed_entry_sizing_diagnostics": compact_executed_entry_diagnostics(
            report.get("executed_entry_sizing_diagnostics")
        ),
        "entry_examples": examples[:6],
        "diagnostic_boundary": (
            "Read-only compact market entry diagnostics. Use it to identify whether "
            "entries are blocked by evidence, expected-net quality, sizing, ML readiness, "
            "or execution contract before changing thresholds, leverage, or position size."
        ),
    }


def compact_executed_entry_sample(item):
    item = safe_dict(item)
    return {
        "id": item.get("id"),
        "time": item.get("time"),
        "symbol": item.get("symbol"),
        "action": item.get("action"),
        "analysis_type": item.get("analysis_type"),
        "decision": pick(
            safe_dict(item.get("decision")),
            (
                "position_size_pct",
                "suggested_leverage",
                "was_executed",
                "executed_at",
                "execution_price",
            ),
        ),
        "order": pick(
            safe_dict(item.get("order")),
            ("id", "status", "side", "quantity", "price", "notional", "filled_at"),
        )
        if safe_dict(item.get("order"))
        else None,
        "evidence": pick(
            safe_dict(item.get("evidence")),
            (
                "tier",
                "effective_score",
                "entry_evidence_score",
                "expected_net_return_pct",
                "aggregate_expected_net_return_pct",
                "ai_expected_return_policy",
                "ai_expected_return_weight",
                "ai_expected_return_independent_probe_support",
                "profit_quality_ratio",
                "loss_probability",
                "tail_risk_score",
                "tradeable_probe",
                "shadow_only",
            ),
        ),
        "sizing": pick(
            safe_dict(item.get("sizing")),
            (
                "position_size_pct",
                "leverage",
                "final_notional_usdt",
                "quality_tier",
                "low_payoff_quality",
                "low_payoff_reasons",
                "notional_floor_applied",
                "notional_floor_blocked",
                "meaningful_size_reason",
                "strategy_probe_cap_applied",
                "strategy_max_probe_size_pct",
                "strategy_reason",
            ),
        ),
        "sizing_reason_tags": safe_list(item.get("sizing_reason_tags"))[:10],
        "notional_gap_usdt": item.get("notional_gap_usdt"),
        "notional_fill_ratio": item.get("notional_fill_ratio"),
        "execution_result": pick(
            safe_dict(item.get("execution_result")),
            (
                "source",
                "status",
                "exchange_confirmed",
                "okx_symbol",
                "planned_order_contracts",
                "planned_base_quantity",
                "execution_blocker",
                "okx_rejection",
                "system_pre_submit_rejection",
            ),
        ),
    }


def compact_executed_entry_diagnostics(value):
    value = safe_dict(value)
    compact = pick(
        value,
        (
            "read_only",
            "executed_entry_count",
            "market_executed_entry_count",
            "filled_order_count",
            "missing_order_count",
            "order_status_counts",
            "evidence_tier_counts",
            "sizing_quality_tier_counts",
            "sizing_reason_tag_counts",
            "order_notional_stats",
            "sizing_final_notional_stats",
            "notional_fill_ratio_stats",
            "decision_position_size_pct_stats",
            "decision_leverage_stats",
            "expected_net_stats",
            "profit_quality_stats",
            "loss_probability_stats",
            "tail_risk_stats",
            "diagnostic_boundary",
        ),
    )
    samples = safe_list(value.get("samples"))
    ai_policy_counts = Counter()
    for item in samples:
        policy = safe_dict(safe_dict(item).get("evidence")).get(
            "ai_expected_return_policy"
        ) or "missing"
        ai_policy_counts[str(policy)] += 1
    compact["ai_expected_return_policy_counts"] = dict(
        ai_policy_counts.most_common(12)
    )
    compact["samples"] = [compact_executed_entry_sample(item) for item in samples[:3]]
    return compact


def compact_entry_example(item):
    item = safe_dict(item)
    ev = safe_dict(item.get("evidence"))
    probe = safe_dict(item.get("probe"))
    sz = safe_dict(item.get("sizing"))
    st = safe_dict(item.get("state"))
    review = safe_dict(item.get("high_risk_review"))
    order = safe_dict(item.get("order"))
    return {
        "id": item.get("id"),
        "time": item.get("time"),
        "symbol": item.get("symbol"),
        "action": item.get("action"),
        "analysis_type": item.get("analysis_type"),
        "executed": bool(item.get("executed")),
        "skip_kind": _compact_entry_skip_kind(item),
        "reason": short_text(item.get("reason"), 320),
        "state": pick(st, ("final_stage", "final_status", "blocked", "failed")),
        "evidence": pick(
            ev,
            (
                "tier",
                "effective_score",
                "entry_evidence_score",
                "score",
                "min_score_required",
                "expected_net_return_pct",
                "aggregate_expected_net_return_pct",
                "ai_expected_return_policy",
                "ai_expected_return_weight",
                "ai_expected_return_independent_probe_support",
                "profit_quality_ratio",
                "loss_probability",
                "tail_risk_score",
                "hard_block",
                "hard_block_reasons",
                "advisory_wait_reasons",
                "aligned_support_sources",
                "major_opposites",
                "weak_opposites",
                "strong_opposites",
                "tradeable_probe",
                "shadow_only",
            ),
        ),
        "probe": {
            "recommendation": probe.get("recommendation") or "",
            "probe_conversion_ready": probe.get("probe_conversion_ready"),
            "probe_conversion_block_reasons": safe_list(
                probe.get("probe_conversion_block_reasons")
            ),
            "probe_conversion_thresholds": safe_dict(
                probe.get("probe_conversion_thresholds")
            ),
            "evidence_profit_probe_blocked": pick(
                safe_dict(probe.get("evidence_profit_probe_blocked")),
                (
                    "blocked",
                    "block_kind",
                    "block_reasons",
                    "reason",
                    "expected_net_return_pct",
                    "profit_quality_ratio",
                    "loss_probability",
                    "tail_risk_score",
                    "thresholds",
                ),
            ),
        },
        "sizing": pick(
            sz,
            (
                "position_size_pct",
                "leverage",
                "final_notional_usdt",
                "quality_tier",
                "low_payoff_quality",
                "low_payoff_reasons",
                "notional_floor_applied",
                "notional_floor_blocked",
                "meaningful_size_reason",
                "strategy_sizing_applied",
                "strategy_probe_cap_applied",
                "strategy_max_probe_size_pct",
                "strategy_reason",
            ),
        ),
        "high_risk_review": (
            pick(
                review,
                (
                    "triggered",
                    "status",
                    "approved",
                    "hard_review_required",
                    "reasons",
                    "advisory_reasons",
                    "reason",
                    "confidence",
                ),
            )
            if review
            else {}
        ),
        "order": pick(order, ("status", "quantity", "price", "notional")) if order else None,
    }


def _compact_entry_skip_kind(item):
    item = safe_dict(item)
    if item.get("executed"):
        return "executed"
    reason = str(item.get("reason") or "")
    state_data = safe_dict(item.get("state"))
    final_stage = str(state_data.get("final_stage") or "").lower()
    final_status = str(state_data.get("final_status") or "").lower()
    if "弱证据学习档" in reason:
        return "entry_evidence_shadow_only"
    if "动态证据不足" in reason or "极小探针" in reason:
        return "entry_evidence_wait"
    if final_stage == "risk_check" and final_status == "blocked":
        high_risk_review = safe_dict(item.get("high_risk_review"))
        if high_risk_review:
            return "high_risk_review_blocked"
        return "risk_check_blocked"
    if final_stage == "risk_check" and final_status == "skipped":
        return "risk_check_skipped"
    return "unknown"


async def main():
    async with get_session_ctx() as session:
        decisions = list((await session.execute(
            select(AIDecision)
            .where(AIDecision.created_at >= since.replace(tzinfo=None))
            .order_by(AIDecision.created_at.desc())
            .limit(1600)
        )).scalars().all())
        orders = list((await session.execute(
            select(Order)
            .where(Order.created_at >= since.replace(tzinfo=None))
            .order_by(Order.created_at.desc())
            .limit(600)
        )).scalars().all())
        positions = list((await session.execute(
            select(Position)
            .where(Position.created_at >= since.replace(tzinfo=None))
            .order_by(Position.created_at.desc())
            .limit(600)
        )).scalars().all())
        closed = list((await session.execute(
            select(Position)
            .where(Position.is_open.is_(False), Position.closed_at >= since.replace(tzinfo=None))
            .order_by(Position.closed_at.desc())
            .limit(500)
        )).scalars().all())
        open_positions = list((await session.execute(
            select(Position)
            .where(Position.is_open.is_(True))
            .order_by(Position.created_at.desc())
            .limit(300)
        )).scalars().all())
        shadow_recent = list((await session.execute(
            select(ShadowBacktest)
            .where(ShadowBacktest.created_at >= since.replace(tzinfo=None))
            .order_by(ShadowBacktest.created_at.desc())
            .limit(1000)
        )).scalars().all())
        shadow_completed = list((await session.execute(
            select(ShadowBacktest)
            .where(ShadowBacktest.status == "completed")
            .order_by(ShadowBacktest.created_at.desc())
            .limit(1000)
        )).scalars().all())
        memories = list((await session.execute(
            select(ExpertMemory)
            .where(ExpertMemory.created_at >= since.replace(tzinfo=None))
            .order_by(ExpertMemory.created_at.desc())
            .limit(300)
        )).scalars().all())
        events = list((await session.execute(
            select(StrategyLearningEvent)
            .where(StrategyLearningEvent.created_at >= since.replace(tzinfo=None))
            .order_by(StrategyLearningEvent.created_at.desc())
            .limit(800)
        )).scalars().all())

    entry_decisions = [d for d in decisions if str(d.action or "").lower() in {"long", "short"}]
    hold_decisions = [d for d in decisions if str(d.action or "").lower() == "hold"]
    market_decisions = [d for d in decisions if analysis_type(d) == "market"]
    position_review_decisions = [
        d for d in decisions if analysis_type(d) == "position_review"
    ]
    market_entry_decisions = [d for d in entry_decisions if analysis_type(d) == "market"]
    market_symbol_counts = Counter(str(d.symbol or "unknown") for d in market_decisions)
    market_entry_symbol_counts = Counter(
        str(d.symbol or "unknown") for d in market_entry_decisions
    )
    market_candidate_funnels = [
        safe_dict(safe_dict(d.raw_llm_response).get("market_candidate_funnel"))
        for d in market_decisions
        if isinstance(safe_dict(d.raw_llm_response).get("market_candidate_funnel"), dict)
    ]
    latest_candidate_funnel = market_candidate_funnels[0] if market_candidate_funnels else {}
    candidate_funnel_window = candidate_funnel_aggregate(market_candidate_funnels)
    candidate_filter_outcomes = candidate_filter_outcome_diagnostics(
        market_decisions,
        market_entry_decisions,
    )
    market_progress = market_analysis_progress_aggregate(market_decisions)
    executed_entries = [d for d in entry_decisions if bool(d.was_executed)]
    decision_by_id = {d.id: d for d in decisions}
    order_by_decision = {}
    for order in orders:
        if order.decision_id and order.decision_id not in order_by_decision:
            order_by_decision[order.decision_id] = order
    trade_contract = await trade_execution_contract_summary()

    reason_counts = Counter()
    state_counts = Counter()
    expected_values = []
    size_values = []
    quality_tiers = Counter()
    low_payoff_count = 0
    low_payoff_reason_counts = Counter()
    low_payoff_missing_reason_count = 0
    notional_floor_blocked = Counter()
    strategy_probe_count = 0
    memory_applied = Counter()
    shadow_memory_component_counts = Counter()
    shadow_memory_contributions = []
    examples = []
    cooldown_examples = []
    shadow_only_examples = []
    market_entry_score_gaps = []
    market_entry_evidence_raw_scores = []
    market_entry_evidence_effective_scores = []
    market_entry_evidence_score_offsets = []
    market_entry_profit_quality_values = []
    market_entry_loss_probabilities = []
    market_entry_tail_risks = []
    market_entry_component_contributions = {}
    market_entry_evidence_component_points = {}
    market_entry_evidence_tier_counts = Counter()
    market_entry_final_skip_kind_counts = Counter()
    market_entry_evidence_component_status_counts = Counter()
    market_entry_evidence_relief_applied_counts = Counter()
    market_entry_probe_recommendation_counts = Counter()
    market_entry_probe_conversion_ready_counts = Counter()
    market_entry_probe_conversion_block_reason_counts = Counter()
    market_entry_profit_probe_block_kind_counts = Counter()
    market_entry_profit_probe_block_reason_counts = Counter()
    market_entry_advisory_wait_reason_counts = Counter()
    market_entry_shadow_only_count = 0
    market_entry_tradeable_probe_count = 0
    market_entry_hard_block_count = 0
    entry_analysis_type_counts = Counter()
    entry_analysis_type_skip_kind_counts = Counter()
    entry_analysis_type_evidence_tier_counts = Counter()
    entry_analysis_type_quality_tier_counts = Counter()
    entry_analysis_type_low_payoff_counts = Counter()
    entry_analysis_type_low_payoff_reason_counts = Counter()
    entry_analysis_type_low_payoff_missing_reason_counts = Counter()
    entry_analysis_type_notional_floor_blocked_counts = Counter()
    entry_analysis_type_metric_values = {}
    high_risk_review_status_counts = Counter()
    high_risk_review_trigger_counts = Counter()
    high_risk_review_approved_counts = Counter()
    high_risk_review_reason_counts = Counter()
    high_risk_review_trigger_reason_counts = Counter()
    analysis_type_counts = Counter(analysis_type(d) for d in decisions)
    analysis_type_action_counts = Counter(
        f"{analysis_type(d)}:{str(d.action or 'unknown').lower()}" for d in decisions
    )
    entry_candidate_evidence_by_type = Counter()
    for d in entry_decisions:
        raw = safe_dict(d.raw_llm_response)
        if isinstance(raw.get("entry_candidate_evidence"), dict):
            entry_candidate_evidence_by_type[analysis_type(d)] += 1

    for d in entry_decisions:
        st = state(d)
        sz = sizing(d)
        ev = evidence(d)
        probe = probe_diagnostics(d)
        atype = analysis_type(d)
        raw = safe_dict(d.raw_llm_response)
        review = safe_dict(raw.get("high_risk_review"))
        if review:
            high_risk_review_status_counts[
                str(review.get("status") or "unknown")
            ] += 1
            high_risk_review_trigger_counts[
                "triggered" if review.get("triggered") else "not_triggered"
            ] += 1
            approved = review.get("approved")
            if approved is True:
                approved_key = "approved_true"
            elif approved is False:
                approved_key = "approved_false"
            else:
                approved_key = "approved_unknown"
            high_risk_review_approved_counts[approved_key] += 1
            reason_text = short_text(review.get("reason"), 140)
            if reason_text:
                high_risk_review_reason_counts[reason_text] += 1
            for review_reason in safe_list(review.get("reasons")):
                if review_reason:
                    high_risk_review_trigger_reason_counts[str(review_reason)] += 1
            for review_reason in safe_list(review.get("advisory_reasons")):
                if review_reason:
                    high_risk_review_trigger_reason_counts[
                        f"advisory:{review_reason}"
                    ] += 1
        skip_kind = entry_skip_kind(d)
        entry_analysis_type_counts[atype] += 1
        entry_analysis_type_skip_kind_counts[f"{atype}:{skip_kind}"] += 1
        entry_analysis_type_evidence_tier_counts[
            f"{atype}:{ev.get('tier') or 'unknown'}"
        ] += 1
        entry_analysis_type_quality_tier_counts[
            f"{atype}:{sz['quality_tier'] or 'unknown'}"
        ] += 1
        metric_bucket = entry_analysis_type_metric_values.setdefault(
            atype,
            {
                "expected_net_return_pct": [],
                "profit_quality_ratio": [],
                "loss_probability": [],
                "tail_risk_score": [],
                "position_size_pct": [],
            },
        )
        metric_bucket["expected_net_return_pct"].append(
            safe_float(ev.get("expected_net_return_pct"))
        )
        metric_bucket["profit_quality_ratio"].append(
            safe_float(ev.get("profit_quality_ratio"))
        )
        metric_bucket["loss_probability"].append(safe_float(ev.get("loss_probability")))
        metric_bucket["tail_risk_score"].append(safe_float(ev.get("tail_risk_score")))
        metric_bucket["position_size_pct"].append(safe_float(d.position_size_pct))
        if atype == "market":
            market_entry_score_gaps.append(
                safe_float(ev.get("score")) - safe_float(ev.get("min_score_required"))
            )
            market_entry_evidence_raw_scores.append(
                safe_float(ev.get("entry_evidence_score"))
            )
            market_entry_evidence_effective_scores.append(
                safe_float(ev.get("effective_score"))
            )
            market_entry_evidence_score_offsets.append(
                safe_float(ev.get("entry_evidence_score_offset"))
            )
            market_entry_evidence_tier_counts[str(ev.get("tier") or "unknown")] += 1
            market_entry_final_skip_kind_counts[entry_skip_kind(d)] += 1
            if ev.get("shadow_only"):
                market_entry_shadow_only_count += 1
            if ev.get("tradeable_probe"):
                market_entry_tradeable_probe_count += 1
            if ev.get("hard_block"):
                market_entry_hard_block_count += 1
            for item in evidence_components(d):
                source = str(item.get("source") or "unknown")
                status = str(item.get("status") or "unknown")
                market_entry_evidence_component_status_counts[f"{source}:{status}"] += 1
                market_entry_evidence_component_points.setdefault(source, []).append(
                    safe_float(item.get("points"))
                )
            for key in ENTRY_EVIDENCE_RELIEF_KEYS:
                relief = safe_dict(ev.get(key))
                state_key = "applied" if relief.get("applied") else "not_applied"
                market_entry_evidence_relief_applied_counts[f"{key}:{state_key}"] += 1
                if relief.get("tradeable_probe"):
                    market_entry_evidence_relief_applied_counts[f"{key}:tradeable_probe"] += 1
                if relief.get("shadow_only"):
                    market_entry_evidence_relief_applied_counts[f"{key}:shadow_only"] += 1
            recommendation = str(probe.get("recommendation") or "unknown")
            market_entry_probe_recommendation_counts[recommendation] += 1
            ready_value = probe.get("probe_conversion_ready")
            if ready_value is True:
                ready_key = "ready"
            elif ready_value is False:
                ready_key = "not_ready"
            else:
                ready_key = "unknown"
            market_entry_probe_conversion_ready_counts[ready_key] += 1
            for block_reason in safe_list(probe.get("probe_conversion_block_reasons")):
                market_entry_probe_conversion_block_reason_counts[
                    str(block_reason)
                ] += 1
            blocked = safe_dict(probe.get("evidence_profit_probe_blocked"))
            if blocked.get("blocked"):
                block_kind = str(blocked.get("block_kind") or "unknown")
                market_entry_profit_probe_block_kind_counts[block_kind] += 1
                for block_reason in safe_list(blocked.get("block_reasons")):
                    market_entry_profit_probe_block_reason_counts[
                        str(block_reason)
                    ] += 1
            for wait_reason in safe_list(ev.get("advisory_wait_reasons")):
                if wait_reason:
                    market_entry_advisory_wait_reason_counts[str(wait_reason)] += 1
            market_entry_profit_quality_values.append(
                safe_float(ev.get("profit_quality_ratio"))
            )
            market_entry_loss_probabilities.append(safe_float(ev.get("loss_probability")))
            market_entry_tail_risks.append(safe_float(ev.get("tail_risk_score")))
            for key, contribution in expected_net_components(d).items():
                market_entry_component_contributions.setdefault(key, []).append(contribution)
        reason = localize_execution_reason(st["final_reason"] or d.execution_reason or "") or ""
        reason_counts[reason[:100] if reason else "无原因"] += 1
        state_counts[f"{st['final_stage']}:{st['final_status']}"] += 1
        expected_values.append(ev["expected_net_return_pct"])
        size_values.append(roundv(d.position_size_pct))
        quality_tiers[sz["quality_tier"] or "unknown"] += 1
        if sz["low_payoff_quality"]:
            low_payoff_count += 1
            entry_analysis_type_low_payoff_counts[atype] += 1
            low_payoff_reasons = safe_list(sz.get("low_payoff_reasons"))
            if low_payoff_reasons:
                for reason_item in low_payoff_reasons:
                    if reason_item:
                        low_payoff_reason_counts[str(reason_item)] += 1
                        entry_analysis_type_low_payoff_reason_counts[
                            f"{atype}:{reason_item}"
                        ] += 1
            else:
                low_payoff_missing_reason_count += 1
                entry_analysis_type_low_payoff_missing_reason_counts[atype] += 1
        if sz["notional_floor_blocked"]:
            notional_floor_blocked[sz["notional_floor_blocked"]] += 1
            entry_analysis_type_notional_floor_blocked_counts[
                f"{atype}:{sz['notional_floor_blocked']}"
            ] += 1
        if sz["strategy_probe_cap_applied"] or sz["strategy_max_probe_size_pct"]:
            strategy_probe_count += 1
        if ev["memory_habit_adjustment"].get("applied"):
            memory_applied[str(ev["memory_habit_adjustment"].get("stance") or "applied")] += 1
        shadow_component = safe_dict(ev.get("shadow_memory_component"))
        if shadow_component:
            shadow_key = "available" if shadow_component.get("available") else "blocked"
            shadow_memory_component_counts[shadow_key] += 1
            contribution = safe_float(shadow_component.get("contribution_pct"), 0.0)
            if contribution > 0:
                shadow_memory_contributions.append(contribution)
        if is_shadow_only_entry_decision(d):
            shadow_only_examples.append(d)
        cooldown = safe_dict(raw.get("loss_cooldown_override")) or safe_dict(
            safe_dict(raw.get("opportunity_score")).get("loss_cooldown_override")
        )
        if cooldown:
            cooldown_examples.append({
                "id": d.id,
                "time": aware(d.created_at).isoformat() if aware(d.created_at) else "",
                "symbol": d.symbol,
                "action": d.action,
                "executed": bool(d.was_executed),
                "cooldown_allowed": cooldown.get("allowed"),
                "cooldown_failed": cooldown.get("failed"),
                "metrics": cooldown.get("metrics"),
                "reason": short_text(reason, 260, localize=True),
            })
        if len(examples) < 50:
            order = order_by_decision.get(d.id)
            examples.append({
                "id": d.id,
                "time": aware(d.created_at).isoformat() if aware(d.created_at) else "",
                "symbol": d.symbol,
                "action": d.action,
                "analysis_type": analysis_type(d),
                "executed": bool(d.was_executed),
                "reason": short_text(reason, 280, localize=True),
                "state": st,
                "evidence": ev,
                "probe": probe,
                "sizing": sz,
                "high_risk_review": safe_dict(raw.get("high_risk_review")),
                "execution_result": order_execution_result(d),
                "order": ({
                    "status": order.status,
                    "quantity": roundv(order.quantity),
                    "price": roundv(order.price),
                    "notional": roundv(order_notional(order)),
                } if order else None),
            })

    fast_loss = []
    for pos in closed:
        created = aware(pos.created_at)
        closed_at = aware(pos.closed_at)
        hold_min = None
        if created and closed_at:
            hold_min = (closed_at - created).total_seconds() / 60.0
        realized = safe_float(pos.realized_pnl)
        notional = safe_float(pos.quantity) * safe_float(pos.entry_price)
        if hold_min is not None and hold_min <= FAST_CLOSE_MINUTES and realized < 0:
            fast_loss.append({
                "id": pos.id,
                "symbol": pos.symbol,
                "side": pos.side,
                "hold_minutes": round(hold_min, 2),
                "quantity": roundv(pos.quantity),
                "entry_price": roundv(pos.entry_price),
                "notional_usdt": roundv(notional),
                "realized_pnl": roundv(realized),
                "created_at": created.isoformat() if created else "",
                "closed_at": closed_at.isoformat() if closed_at else "",
            })

    current_open = []
    for pos in open_positions[:100]:
        notional = safe_float(pos.quantity) * safe_float(pos.entry_price)
        current_open.append({
            "id": pos.id,
            "symbol": pos.symbol,
            "side": pos.side,
            "quantity": roundv(pos.quantity),
            "entry_price": roundv(pos.entry_price),
            "current_price": roundv(pos.current_price),
            "notional_usdt": roundv(notional),
            "unrealized_pnl": roundv(pos.unrealized_pnl),
            "created_at": aware(pos.created_at).isoformat() if aware(pos.created_at) else "",
        })

    shadow_counts = Counter()
    shadow_by_best = Counter()
    missed_by_side = Counter()
    missed_samples = []
    for row in shadow_completed:
        shadow_counts[str(row.status or "unknown")] += 1
        shadow_by_best[str(row.best_action or "unknown")] += 1
        if row.missed_opportunity:
            missed_by_side[str(row.best_action or "unknown")] += 1
            if len(missed_samples) < 20:
                missed_samples.append({
                    "id": row.id,
                    "decision_id": row.decision_id,
                    "symbol": row.symbol,
                    "decision_action": row.decision_action,
                    "best_action": row.best_action,
                    "long_return_pct": roundv(row.long_return_pct),
                    "short_return_pct": roundv(row.short_return_pct),
                    "horizon_minutes": row.horizon_minutes,
                    "created_at": aware(row.created_at).isoformat() if aware(row.created_at) else "",
                    "note": short_text(row.note, 180),
                })

    memory_counts = Counter(str(m.memory_type or "unknown") for m in memories)
    event_counts = Counter(str(e.event_type or "unknown") for e in events)
    order_status_counts = Counter(str(o.status or "unknown").lower() for o in orders)
    filled_orders = [o for o in orders if str(o.status or "").lower() == "filled"]
    non_filled_orders = [o for o in orders if str(o.status or "").lower() != "filled"]
    rejected_orders = [
        o
        for o in orders
        if str(o.status or "").lower()
        in {"rejected", "failed", "error", "cancelled", "canceled"}
    ]
    pending_or_open_orders = [
        o
        for o in orders
        if str(o.status or "").lower() in {"pending", "open", "partial"}
    ]
    failed_orders = non_filled_orders
    rejected_order_examples = []
    for order in rejected_orders[:30]:
        decision = decision_by_id.get(order.decision_id)
        rejected_order_examples.append(
            {
                "id": order.id,
                "decision_id": order.decision_id,
                "time": aware(order.created_at).isoformat() if aware(order.created_at) else "",
                "symbol": order.symbol,
                "side": order.side,
                "status": order.status,
                "quantity": roundv(order.quantity),
                "price": roundv(order.price),
                "notional": roundv(order_notional(order)),
                "exchange_order_id": order.exchange_order_id,
                "execution_reason": short_text(
                    getattr(decision, "execution_reason", "") if decision else "",
                    500,
                    localize=True,
                ),
                "execution_parameters": safe_dict(
                    safe_dict(getattr(decision, "raw_llm_response", None)).get(
                        "execution_parameters"
                    )
                    if decision
                    else {}
                ),
                "execution_result": order_execution_result(decision),
            }
        )
    market_symbol_diagnostics = {
        "market_decision_count": len(market_decisions),
        "market_unique_symbol_count": len(market_symbol_counts),
        "market_top_symbols": symbol_counter_rows(market_symbol_counts),
        "market_entry_count": len(market_entry_decisions),
        "market_entry_unique_symbol_count": len(market_entry_symbol_counts),
        "market_entry_top_symbols": symbol_counter_rows(market_entry_symbol_counts),
        "market_entry_action_counts": counter_rows(
            Counter(str(d.action or "unknown").lower() for d in market_entry_decisions),
            10,
        ),
        "market_entry_skip_kind_counts": counter_rows(
            market_entry_final_skip_kind_counts,
            20,
        ),
        "market_entry_tier_counts": counter_rows(
            market_entry_evidence_tier_counts,
            20,
        ),
        "market_top3_share": top_share(market_symbol_counts),
        "market_entry_top3_share": top_share(market_entry_symbol_counts),
        "entry_unique_to_market_unique_ratio": roundv(
            len(market_entry_symbol_counts) / max(len(market_symbol_counts), 1)
        ),
        "candidate_funnel_sample_count": len(market_candidate_funnels),
        "latest_candidate_funnel": latest_candidate_funnel,
        "candidate_funnel_window": candidate_funnel_window,
        "market_analysis_progress": market_progress,
        "candidate_filter_outcomes": candidate_filter_outcomes,
        "diagnostic_boundary": (
            "Read-only symbol funnel diagnostics; repeated entry symbols require checking "
            "feature fetch budget, ranker selection, recent-analysis dedupe, evidence tier, "
            "and skip_kind before changing any trading gate."
        ),
    }
    report = {
        "window_minutes": WINDOW_MINUTES,
        "generated_at": now.isoformat(),
        "counts": {
            "decisions": len(decisions),
            "market_decisions": len(market_decisions),
            "position_review_decisions": len(position_review_decisions),
            "hold_decisions": len(hold_decisions),
            "entry_decisions": len(entry_decisions),
            "market_entry_decisions": len(market_entry_decisions),
            "executed_entries": len(executed_entries),
            "orders": len(orders),
            "filled_orders": len(filled_orders),
            "failed_orders": len(failed_orders),
            "non_filled_orders": len(non_filled_orders),
            "rejected_orders": len(rejected_orders),
            "pending_or_open_orders": len(pending_or_open_orders),
            "positions_created": len(positions),
            "positions_closed": len(closed),
            "open_positions": len(open_positions),
            "fast_loss_close_under_15m": len(fast_loss),
            "shadow_recent": len(shadow_recent),
            "shadow_completed_sample": len(shadow_completed),
            "missed_opportunity_sample": sum(1 for s in shadow_completed if s.missed_opportunity),
            "expert_memory_recent": len(memories),
            "strategy_learning_events": len(events),
        },
        "action_counts": dict(Counter(str(d.action or "unknown").lower() for d in decisions)),
        "order_status_counts": dict(order_status_counts.most_common(20)),
        "analysis_type_counts": dict(analysis_type_counts.most_common(20)),
        "analysis_type_action_counts": dict(analysis_type_action_counts.most_common(40)),
        "market_symbol_diagnostics": market_symbol_diagnostics,
        "entry_candidate_evidence_by_type": dict(
            entry_candidate_evidence_by_type.most_common(20)
        ),
        "entry_state_counts": dict(state_counts.most_common(20)),
        "entry_reason_counts": [{"reason_prefix": k, "count": v} for k, v in reason_counts.most_common(20)],
        "expected_net_stats": stats(expected_values),
        "position_size_pct_stats": stats(size_values),
        "quality_tier_counts": dict(quality_tiers.most_common(20)),
        "low_payoff_entry_count": low_payoff_count,
        "low_payoff_reason_counts": dict(low_payoff_reason_counts.most_common(20)),
        "low_payoff_missing_reason_count": low_payoff_missing_reason_count,
        "entry_analysis_type_counts": dict(entry_analysis_type_counts.most_common(20)),
        "entry_analysis_type_skip_kind_counts": dict(
            entry_analysis_type_skip_kind_counts.most_common(40)
        ),
        "entry_analysis_type_evidence_tier_counts": dict(
            entry_analysis_type_evidence_tier_counts.most_common(40)
        ),
        "entry_analysis_type_quality_tier_counts": dict(
            entry_analysis_type_quality_tier_counts.most_common(40)
        ),
        "entry_analysis_type_low_payoff_counts": dict(
            entry_analysis_type_low_payoff_counts.most_common(20)
        ),
        "entry_analysis_type_low_payoff_reason_counts": dict(
            entry_analysis_type_low_payoff_reason_counts.most_common(40)
        ),
        "entry_analysis_type_low_payoff_missing_reason_counts": dict(
            entry_analysis_type_low_payoff_missing_reason_counts.most_common(20)
        ),
        "entry_analysis_type_notional_floor_blocked_counts": dict(
            entry_analysis_type_notional_floor_blocked_counts.most_common(40)
        ),
        "entry_analysis_type_metric_stats": {
            analysis_key: {
                metric_key: stats(metric_values)
                for metric_key, metric_values in metrics.items()
            }
            for analysis_key, metrics in sorted(entry_analysis_type_metric_values.items())
        },
        "high_risk_review_status_counts": dict(
            high_risk_review_status_counts.most_common(20)
        ),
        "high_risk_review_trigger_counts": dict(
            high_risk_review_trigger_counts.most_common(10)
        ),
        "high_risk_review_approved_counts": dict(
            high_risk_review_approved_counts.most_common(10)
        ),
        "high_risk_review_reason_counts": dict(
            high_risk_review_reason_counts.most_common(20)
        ),
        "high_risk_review_trigger_reason_counts": dict(
            high_risk_review_trigger_reason_counts.most_common(30)
        ),
        "strategy_probe_cap_count": strategy_probe_count,
        "memory_habit_applied_counts": dict(memory_applied.most_common(10)),
          "shadow_memory_component_counts": dict(shadow_memory_component_counts.most_common(10)),
          "shadow_memory_contribution_stats": stats(shadow_memory_contributions),
        "entry_evidence_thresholds": {
            "weak_probe": ENTRY_EVIDENCE_SCORE_WEAK_PROBE,
            "exploration": ENTRY_EVIDENCE_SCORE_PROBE,
            "small": ENTRY_EVIDENCE_SCORE_SMALL,
            "medium": ENTRY_EVIDENCE_SCORE_MEDIUM,
            "normal": ENTRY_EVIDENCE_SCORE_NORMAL,
        },
          "market_entry_opportunity_score_gap_stats": stats(market_entry_score_gaps),
          "market_entry_score_gap_stats": stats(market_entry_score_gaps),
        "market_entry_evidence_raw_score_stats": stats(
            market_entry_evidence_raw_scores
        ),
          "market_entry_evidence_effective_score_stats": stats(
              market_entry_evidence_effective_scores
          ),
        "market_entry_evidence_score_offset_stats": stats(
            market_entry_evidence_score_offsets
        ),
          "market_entry_evidence_tier_counts": dict(
            market_entry_evidence_tier_counts.most_common(20)
        ),
        "market_entry_final_skip_kind_counts": dict(
            market_entry_final_skip_kind_counts.most_common(20)
        ),
          "market_entry_evidence_component_status_counts": dict(
              market_entry_evidence_component_status_counts.most_common(40)
          ),
        "market_entry_evidence_component_point_stats": {
            key: stats(values)
            for key, values in sorted(market_entry_evidence_component_points.items())
        },
        "market_entry_evidence_relief_applied_counts": dict(
            market_entry_evidence_relief_applied_counts.most_common(30)
        ),
        "market_entry_probe_recommendation_counts": dict(
            market_entry_probe_recommendation_counts.most_common(20)
        ),
        "market_entry_probe_conversion_ready_counts": dict(
            market_entry_probe_conversion_ready_counts.most_common(10)
        ),
        "market_entry_probe_conversion_block_reason_counts": dict(
            market_entry_probe_conversion_block_reason_counts.most_common(20)
        ),
        "market_entry_profit_probe_block_kind_counts": dict(
            market_entry_profit_probe_block_kind_counts.most_common(20)
        ),
        "market_entry_profit_probe_block_reason_counts": dict(
            market_entry_profit_probe_block_reason_counts.most_common(20)
        ),
        "market_entry_advisory_wait_reason_counts": dict(
            market_entry_advisory_wait_reason_counts.most_common(20)
        ),
          "market_entry_evidence_shadow_only_count": market_entry_shadow_only_count,
          "market_entry_evidence_tradeable_probe_count": market_entry_tradeable_probe_count,
          "market_entry_evidence_hard_block_count": market_entry_hard_block_count,
        "market_entry_profit_quality_stats": stats(market_entry_profit_quality_values),
        "market_entry_loss_probability_stats": stats(market_entry_loss_probabilities),
        "market_entry_tail_risk_stats": stats(market_entry_tail_risks),
        "market_entry_expected_net_component_stats": {
            key: stats(values)
            for key, values in sorted(market_entry_component_contributions.items())
        },
        "local_ml_readiness": local_ml_readiness_summary(),
        "trade_execution_contract": trade_contract,
        "closed_position_pnl_diagnostics": closed_position_pnl_diagnostics(
            closed,
            orders,
            decision_by_id,
            decisions,
        ),
        "executed_entry_sizing_diagnostics": executed_entry_sizing_diagnostics(
            entry_decisions,
            order_by_decision,
        ),
        "shadow_only_positive_net_count": len(shadow_only_examples),
        "notional_floor_blocked_counts": dict(notional_floor_blocked.most_common(12)),
        "shadow_completed_best_action_counts": dict(shadow_by_best.most_common(10)),
        "missed_by_best_action": dict(missed_by_side.most_common(10)),
        "expert_memory_type_counts_recent": dict(memory_counts.most_common(20)),
        "strategy_event_type_counts": dict(event_counts.most_common(20)),
        "missed_samples": missed_samples,
        "entry_examples": examples,
        "rejected_order_examples": rejected_order_examples,
        "loss_cooldown_examples": cooldown_examples[:30],
        "fast_loss_positions": fast_loss[:40],
        "current_open_positions": current_open,
    }
    if ENTRY_ONLY:
        output = entry_only_report(report)
    elif MARKET_SYMBOL_ONLY:
        output = market_symbol_only_report(report)
    elif SUMMARY_ONLY:
        output = summary_report(report)
    else:
        output = report
    print(json.dumps(output, ensure_ascii=False, indent=2))

asyncio.run(main())
"""

LAUNCHER_SCRIPT = r"""
import os
import pwd
import subprocess
import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("sample_path")
args = parser.parse_args()

pid = subprocess.check_output(["systemctl", "show", "-p", "MainPID", "--value", "bb-dashboard.service"], text=True).strip()
env = {}
if pid and pid != "0":
    data = Path(f"/proc/{pid}/environ").read_bytes()
    for part in data.split(b"\0"):
        if b"=" not in part:
            continue
        key, value = part.split(b"=", 1)
        try:
            env[key.decode()] = value.decode()
        except UnicodeDecodeError:
            pass
for key in ("PATH", "LANG", "LC_ALL"):
    env.setdefault(key, os.environ.get(key, ""))
env["PYTHONPATH"] = "/data/bb/app"
user = pwd.getpwnam("bb")

def demote():
    os.setgid(user.pw_gid)
    os.setuid(user.pw_uid)

result = subprocess.run(
    ["/data/bb/app/.venv/bin/python", args.sample_path],
    cwd="/data/bb/app",
    env=env,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    preexec_fn=demote,
    timeout=180,
)
sys.stdout.write(result.stdout)
sys.stderr.write(result.stderr)
sys.exit(result.returncode)
"""


def _build_remote_command(
    minutes: int,
    *,
    token: str | None = None,
    summary: bool = False,
    market_symbol_only: bool = False,
    entry_only: bool = False,
    output_path: str | None = None,
) -> str:
    safe_minutes = max(int(minutes or 480), 1)
    safe_token = token or secrets.token_hex(6)
    tmp_dir = "/data/bb/app/tmp/codex-strategy-health"
    sample_path = f"{tmp_dir}/sample_{safe_minutes}_{safe_token}.py"
    launcher_path = f"{tmp_dir}/launcher_{safe_minutes}_{safe_token}.py"
    result_path = output_path or ""
    remote_script = (
        REMOTE_SCRIPT_TEMPLATE.replace("__WINDOW_MINUTES__", str(safe_minutes))
        .replace("__SUMMARY_ONLY__", "True" if summary else "False")
        .replace("__MARKET_SYMBOL_ONLY__", "True" if market_symbol_only else "False")
        .replace("__ENTRY_ONLY__", "True" if entry_only else "False")
    )
    return f"""
set -eo pipefail
cd /data/bb/app
mkdir -p {tmp_dir}
chmod 0750 {tmp_dir}
cat > {sample_path} <<'PY'
{remote_script}
PY
cat > {launcher_path} <<'PY'
{LAUNCHER_SCRIPT}
PY
chmod 0644 {sample_path} {launcher_path}
if [ -n "{result_path}" ]; then
  python3 {launcher_path} {sample_path} > "{result_path}"
  chmod 0640 "{result_path}"
  printf '%s\\n' "{result_path}"
else
  python3 {launcher_path} {sample_path}
fi
rm -f {sample_path} {launcher_path}
"""


def _remote_result_path(minutes: int, token: str) -> str:
    safe_minutes = max(int(minutes or 480), 1)
    return f"/data/bb/app/tmp/codex-strategy-health/result_{safe_minutes}_{token}.json"


def _pick(mapping: dict, keys: tuple[str, ...]) -> dict:
    return {key: mapping.get(key) for key in keys}


def _summarize_report(report: dict) -> dict:
    counts = report.get("counts") if isinstance(report.get("counts"), dict) else {}
    local_ml = (
        report.get("local_ml_readiness")
        if isinstance(report.get("local_ml_readiness"), dict)
        else {}
    )
    contract = (
        report.get("trade_execution_contract")
        if isinstance(report.get("trade_execution_contract"), dict)
        else {}
    )
    return {
        "window_minutes": report.get("window_minutes"),
        "generated_at": report.get("generated_at"),
        "counts": _pick(
            counts,
            (
                "decisions",
                "orders",
                "filled_orders",
                "failed_orders",
                "rejected_orders",
                "pending_or_open_orders",
                "positions_created",
                "positions_closed",
                "open_positions",
                "fast_loss_close_under_15m",
            ),
        ),
        "order_status_counts": report.get("order_status_counts", {}),
        "trade_execution_contract": {
            "status": contract.get("status"),
            "audit_only": contract.get("audit_only"),
            "can_bypass_risk_controls": contract.get("can_bypass_risk_controls"),
            "summary": contract.get("summary", {}),
            "violation_reason_counts": contract.get("violation_reason_counts", {}),
            "fast_loss_samples": (
                contract.get("fast_loss_samples", [])[:5]
                if isinstance(contract.get("fast_loss_samples"), list)
                else []
            ),
            "violations": (
                contract.get("violations", [])[:5]
                if isinstance(contract.get("violations"), list)
                else []
            ),
        },
        "local_ml_readiness": {
            "status": local_ml.get("status"),
            "readiness_state": local_ml.get("readiness_state"),
            "allow_live_position_influence": local_ml.get("allow_live_position_influence"),
            "blocking_reason_codes": local_ml.get("blocking_reason_codes", []),
            "metrics": local_ml.get("metrics", {}),
            "training_window_composition": (
                local_ml.get("training_window_composition")
                if isinstance(local_ml.get("training_window_composition"), dict)
                else {}
            ),
        },
        "market_symbol_diagnostics": _compact_market_symbol_diagnostics(
            report.get("market_symbol_diagnostics"),
            include_latest=False,
        ),
        "closed_position_pnl_diagnostics": _compact_closed_position_pnl_diagnostics(
            report.get("closed_position_pnl_diagnostics")
        ),
        "rejected_order_examples": (
            report.get("rejected_order_examples", [])[:5]
            if isinstance(report.get("rejected_order_examples"), list)
            else []
        ),
        "fast_loss_positions": (
            report.get("fast_loss_positions", [])[:5]
            if isinstance(report.get("fast_loss_positions"), list)
            else []
        ),
    }


def _compact_candidate_funnel(funnel: dict) -> dict:
    if not isinstance(funnel, dict):
        return {}

    def compact_market_limit_diagnostics(value: dict | None) -> dict:
        value = value if isinstance(value, dict) else {}
        return _pick(
            value,
            (
                "read_only",
                "is_entry_gate",
                "budget_source",
                "strategy_profile_id",
                "risk_level",
                "market_limit_policy",
                "configured_market_symbol_limit",
                "selected_market_symbol_limit",
                "position_group_count",
                "total_position_groups",
                "target_position_groups",
                "roster_underfilled",
            ),
        )

    def compact_budget(value: dict | None) -> dict:
        value = value if isinstance(value, dict) else {}
        result = _pick(
            value,
            (
                "budget_source",
                "market_limit_policy",
                "market_symbol_limit",
            ),
        )
        result["market_limit_diagnostics"] = compact_market_limit_diagnostics(
            value.get("market_limit_diagnostics")
        )
        return result

    def compact_feature_fetch_budget(value: dict | None) -> dict:
        value = value if isinstance(value, dict) else {}
        return _pick(
            value,
            (
                "read_only",
                "is_entry_gate",
                "total_candidates",
                "position_symbols",
                "market_candidates",
                "configured_market_symbol_limit",
                "target_market_feature_fetch_count",
                "max_market_feature_fetch_count",
                "selected_market_feature_fetch_count",
                "selected_total_feature_fetch_count",
                "pool_multiplier",
                "pool_min",
                "pool_max",
            ),
        )

    def compact_rank_item(item: dict | None) -> dict:
        item = item if isinstance(item, dict) else {}
        metrics = item.get("filter_metrics") if isinstance(item.get("filter_metrics"), dict) else {}
        return {
            "symbol": item.get("symbol"),
            "score": item.get("score"),
            "net_score": item.get("net_score"),
            "selected": bool(item.get("selected")),
            "non_selected_reason": item.get("non_selected_reason"),
            "selection_tier": item.get("selection_tier"),
            "filter_reasons": item.get("filter_reasons", []),
            "volume_ratio": item.get("volume_ratio", metrics.get("volume_ratio")),
            "volume_ratio_source": item.get(
                "volume_ratio_source",
                metrics.get("volume_ratio_source"),
            ),
            "trend_volume_ratio": item.get(
                "trend_volume_ratio",
                metrics.get("trend_volume_ratio"),
            ),
            "trend_volume_ratio_timeframe": item.get(
                "trend_volume_ratio_timeframe",
                metrics.get("trend_volume_ratio_timeframe"),
            ),
            "entry_activity_volume_ratio": item.get(
                "entry_activity_volume_ratio",
                metrics.get("entry_activity_volume_ratio"),
            ),
            "entry_activity_volume_timeframe": item.get(
                "entry_activity_volume_timeframe",
                metrics.get("entry_activity_volume_timeframe"),
            ),
            "adx": item.get("adx", metrics.get("adx")),
            "change_24h": item.get("change_24h", metrics.get("change_24h")),
            "notional_24h": metrics.get("notional_24h"),
        }

    compact = _pick(
        funnel,
        (
            "read_only",
            "is_entry_gate",
            "mode",
            "run_market_analysis",
            "scan_symbol_count",
            "blocked_filter_count",
            "open_position_filtered_count",
            "unclaimed_filtered_count",
            "feature_fetch_requested_count",
            "feature_fetch_budget",
            "feature_valid_count",
            "feature_invalid_count",
            "market_feature_before_rank_count",
            "market_feature_after_dedupe_count",
            "recent_analysis_dedupe_count",
            "market_budget_rotation",
            "market_symbol_budget",
            "rank_selected_count",
            "rank_tradable_candidates",
            "rank_secondary_candidates",
            "rank_total_candidates",
            "rank_underfilled",
            "rank_underfill_reason",
            "rank_filtered_out_candidates",
        ),
    )
    rotation = funnel.get("market_budget_rotation")
    compact["market_budget_rotation"] = rotation if isinstance(rotation, dict) else {}
    compact["feature_fetch_budget"] = compact_feature_fetch_budget(
        funnel.get("feature_fetch_budget")
    )
    reasons = funnel.get("rank_filtered_out_reason_counts")
    compact["rank_filtered_out_reason_counts"] = reasons[:6] if isinstance(reasons, list) else []
    for key in ("rank_top_symbols", "ranked_symbol_sample", "filtered_symbol_sample"):
        value = funnel.get(key)
        compact[key] = [
            compact_rank_item(item) for item in (value[:2] if isinstance(value, list) else [])
        ]
    budget = funnel.get("analysis_budget")
    if isinstance(budget, dict):
        compact["analysis_budget"] = compact_budget(budget)
    return compact


def _compact_candidate_funnel_window(window: dict) -> dict:
    if not isinstance(window, dict):
        return {}
    metric_stats = (
        window.get("metric_stats") if isinstance(window.get("metric_stats"), dict) else {}
    )
    metric_keys = (
        "scan_symbol_count",
        "feature_fetch_requested_count",
        "feature_valid_count",
        "feature_invalid_count",
        "market_symbol_budget",
        "rank_selected_count",
        "rank_filtered_out_candidates",
        "recent_analysis_dedupe_count",
    )
    return {
        "count": window.get("count", 0),
        "rank_underfilled_count": window.get("rank_underfilled_count", 0),
        "metric_stats": {
            key: _pick(
                metric_stats.get(key) if isinstance(metric_stats.get(key), dict) else {},
                ("count", "median", "p75", "max", "positive", "zero"),
            )
            for key in metric_keys
            if isinstance(metric_stats.get(key), dict)
        },
        "budget_source_counts": (
            window.get("budget_source_counts")[:4]
            if isinstance(window.get("budget_source_counts"), list)
            else []
        ),
        "market_limit_policy_counts": (
            window.get("market_limit_policy_counts")[:4]
            if isinstance(window.get("market_limit_policy_counts"), list)
            else []
        ),
        "rank_underfill_reason_counts": (
            window.get("rank_underfill_reason_counts")[:4]
            if isinstance(window.get("rank_underfill_reason_counts"), list)
            else []
        ),
        "filtered_out_reason_counts": (
            window.get("filtered_out_reason_counts")[:6]
            if isinstance(window.get("filtered_out_reason_counts"), list)
            else []
        ),
        "selected_symbol_counts": (
            window.get("selected_symbol_counts")[:6]
            if isinstance(window.get("selected_symbol_counts"), list)
            else []
        ),
        "outside_budget_symbol_counts": (
            window.get("outside_budget_symbol_counts")[:6]
            if isinstance(window.get("outside_budget_symbol_counts"), list)
            else []
        ),
        "filtered_symbol_counts": (
            window.get("filtered_symbol_counts")[:6]
            if isinstance(window.get("filtered_symbol_counts"), list)
            else []
        ),
        "diagnostic_boundary": window.get("diagnostic_boundary"),
    }


def _compact_candidate_filter_outcomes(outcomes: dict) -> dict:
    if not isinstance(outcomes, dict):
        return {}
    compact = _pick(
        outcomes,
        (
            "read_only",
            "sampled_symbol_count",
            "sampled_occurrence_count",
            "market_entry_after_filter_count",
            "market_entry_after_filter_symbol_count",
            "positive_expected_net_after_filter_count",
            "executed_after_filter_count",
            "expected_net_stats",
            "diagnostic_boundary",
        ),
    )
    for key in (
        "category_counts",
        "reason_counts",
        "sampled_symbol_counts",
        "outcome_symbol_counts",
        "positive_expected_net_symbol_counts",
        "skip_kind_counts",
        "evidence_tier_counts",
    ):
        value = outcomes.get(key)
        compact[key] = value[:6] if isinstance(value, list) else []
    for key in ("symbol_examples", "market_entry_examples"):
        value = outcomes.get(key)
        compact[key] = value[:2] if isinstance(value, list) else []
    return compact


def _compact_market_symbol_diagnostics(
    diagnostics: dict | None,
    *,
    include_latest: bool = True,
) -> dict:
    diagnostics = dict(diagnostics) if isinstance(diagnostics, dict) else {}
    for key in ("market_top_symbols", "market_entry_top_symbols"):
        value = diagnostics.get(key)
        diagnostics[key] = value[:8] if isinstance(value, list) else []
    compact = {
        key: diagnostics.get(key)
        for key in (
            "market_decision_count",
            "market_unique_symbol_count",
            "market_top_symbols",
            "market_entry_count",
            "market_entry_unique_symbol_count",
            "market_entry_top_symbols",
            "market_entry_action_counts",
            "market_entry_skip_kind_counts",
            "market_entry_tier_counts",
            "market_top3_share",
            "market_entry_top3_share",
            "entry_unique_to_market_unique_ratio",
            "candidate_funnel_sample_count",
            "candidate_funnel_window",
            "market_analysis_progress",
            "candidate_filter_outcomes",
            "diagnostic_boundary",
        )
    }
    compact["candidate_filter_outcomes"] = _compact_candidate_filter_outcomes(
        diagnostics.get("candidate_filter_outcomes")
    )
    compact["candidate_funnel_window"] = _compact_candidate_funnel_window(
        diagnostics.get("candidate_funnel_window")
    )
    progress = diagnostics.get("market_analysis_progress")
    compact["market_analysis_progress"] = progress if isinstance(progress, dict) else {}
    if include_latest:
        compact["latest_candidate_funnel"] = _compact_candidate_funnel(
            diagnostics.get("latest_candidate_funnel")
        )
    return compact


def _compact_text_counter(mapping: dict | None, limit: int = 6, text_limit: int = 90) -> list[dict]:
    values = mapping if isinstance(mapping, dict) else {}
    counter = Counter()
    for key, count in values.items():
        text = str(key or "").replace("\n", " ").strip()
        if len(text) > text_limit:
            text = f"{text[:text_limit]}..."
        try:
            numeric_count = int(count)
        except (TypeError, ValueError):
            numeric_count = 0
        counter[text] += numeric_count
    return [{"value": str(key), "count": int(count)} for key, count in counter.most_common(limit)]


def _compact_closed_position_pnl_diagnostics(value: dict | None) -> dict:
    value = value if isinstance(value, dict) else {}
    compact = _pick(
        value,
        (
            "read_only",
            "closed_count",
            "win_count",
            "loss_count",
            "flat_count",
            "win_rate",
            "total_realized_pnl",
            "avg_realized_pnl",
            "gross_profit",
            "gross_loss_abs",
            "profit_factor",
            "realized_pnl_stats",
            "hold_minutes_stats",
            "fast_loss_close_under_15m",
            "symbol_counts",
            "symbol_loss_counts",
            "symbol_pnl",
            "worst_symbol_pnl",
            "side_counts",
            "side_loss_counts",
            "side_pnl",
            "worst_side_pnl",
            "trigger_counts",
            "diagnostic_boundary",
        ),
    )
    samples = value.get("samples") if isinstance(value.get("samples"), list) else []
    compact["samples"] = samples[:5]
    return compact


def _market_symbol_only_report(report: dict) -> dict:
    counts = report.get("counts") if isinstance(report.get("counts"), dict) else {}
    contract = (
        report.get("trade_execution_contract")
        if isinstance(report.get("trade_execution_contract"), dict)
        else {}
    )
    local_ml = (
        report.get("local_ml_readiness")
        if isinstance(report.get("local_ml_readiness"), dict)
        else {}
    )
    diagnostics = _compact_market_symbol_diagnostics(
        report.get("market_symbol_diagnostics"),
        include_latest=True,
    )
    return {
        "window_minutes": report.get("window_minutes"),
        "generated_at": report.get("generated_at"),
        "counts": _pick(
            counts,
            (
                "decisions",
                "market_decisions",
                "market_entry_decisions",
                "orders",
                "failed_orders",
                "rejected_orders",
                "positions_created",
                "positions_closed",
                "open_positions",
                "fast_loss_close_under_15m",
            ),
        ),
        "trade_execution_contract": {
            "status": contract.get("status"),
            "can_bypass_risk_controls": contract.get("can_bypass_risk_controls"),
            "summary": _pick(
                contract.get("summary") if isinstance(contract.get("summary"), dict) else {},
                (
                    "decision_count",
                    "executed_entry_count",
                    "contract_violation_count",
                    "weak_evidence_executed_count",
                    "negative_expected_executed_count",
                    "fast_loss_count",
                    "fast_loss_without_strong_exit_count",
                    "reentry_without_strong_unlock_count",
                ),
            ),
        },
        "closed_position_pnl_diagnostics": _compact_closed_position_pnl_diagnostics(
            report.get("closed_position_pnl_diagnostics")
        ),
        "local_ml_readiness": {
            "status": local_ml.get("status"),
            "readiness_state": local_ml.get("readiness_state"),
            "allow_live_position_influence": local_ml.get("allow_live_position_influence"),
            "blocking_reason_codes": local_ml.get("blocking_reason_codes", []),
            "metrics": _pick(
                local_ml.get("metrics") if isinstance(local_ml.get("metrics"), dict) else {},
                (
                    "sample_count",
                    "dirty_sample_ratio",
                    "long_pr_auc",
                    "short_pr_auc",
                    "top_long_avg_return_pct",
                    "top_short_avg_return_pct",
                    "training_data_version",
                    "required_training_data_version",
                ),
            ),
            "training_window_composition": (
                local_ml.get("training_window_composition")
                if isinstance(local_ml.get("training_window_composition"), dict)
                else {}
            ),
        },
        "market_symbol_diagnostics": diagnostics,
        "diagnostic_boundary": (
            "Read-only compact market symbol and candidate funnel diagnostics. "
            "Use this output to explain repeated symbols or no-entry windows before "
            "changing ranker, budget, evidence, sizing, leverage, or ML gates."
        ),
    }


def _entry_only_report(report: dict) -> dict:
    counts = report.get("counts") if isinstance(report.get("counts"), dict) else {}
    contract = (
        report.get("trade_execution_contract")
        if isinstance(report.get("trade_execution_contract"), dict)
        else {}
    )
    local_ml = (
        report.get("local_ml_readiness")
        if isinstance(report.get("local_ml_readiness"), dict)
        else {}
    )
    examples = [
        _compact_entry_example(item)
        for item in (
            report.get("entry_examples") if isinstance(report.get("entry_examples"), list) else []
        )
        if isinstance(item, dict) and item.get("analysis_type") in {"market", "entry_candidate"}
    ]
    ai_policy_counts = Counter()
    for item in examples:
        policy = (
            item.get("evidence", {}).get("ai_expected_return_policy")
            if isinstance(item.get("evidence"), dict)
            else None
        ) or "missing"
        ai_policy_counts[str(policy)] += 1
    return {
        "window_minutes": report.get("window_minutes"),
        "generated_at": report.get("generated_at"),
        "counts": _pick(
            counts,
            (
                "decisions",
                "market_decisions",
                "entry_decisions",
                "market_entry_decisions",
                "executed_entries",
                "orders",
                "filled_orders",
                "failed_orders",
                "rejected_orders",
                "positions_created",
                "positions_closed",
                "open_positions",
                "fast_loss_close_under_15m",
            ),
        ),
        "trade_execution_contract": {
            "status": contract.get("status"),
            "can_bypass_risk_controls": contract.get("can_bypass_risk_controls"),
            "summary": _pick(
                contract.get("summary") if isinstance(contract.get("summary"), dict) else {},
                (
                    "decision_count",
                    "executed_entry_count",
                    "contract_violation_count",
                    "weak_evidence_executed_count",
                    "negative_expected_executed_count",
                    "fast_loss_count",
                    "fast_loss_without_strong_exit_count",
                    "reentry_without_strong_unlock_count",
                ),
            ),
        },
        "closed_position_pnl_diagnostics": _compact_closed_position_pnl_diagnostics(
            report.get("closed_position_pnl_diagnostics")
        ),
        "local_ml_readiness": {
            "status": local_ml.get("status"),
            "readiness_state": local_ml.get("readiness_state"),
            "allow_live_position_influence": local_ml.get("allow_live_position_influence"),
            "blocking_reason_codes": local_ml.get("blocking_reason_codes", []),
            "metrics": _pick(
                local_ml.get("metrics") if isinstance(local_ml.get("metrics"), dict) else {},
                (
                    "sample_count",
                    "test_count",
                    "dirty_sample_ratio",
                    "long_pr_auc",
                    "short_pr_auc",
                    "top_long_avg_return_pct",
                    "top_short_avg_return_pct",
                    "training_data_version",
                    "required_training_data_version",
                ),
            ),
            "training_window_composition": (
                local_ml.get("training_window_composition")
                if isinstance(local_ml.get("training_window_composition"), dict)
                else {}
            ),
        },
        "entry_evidence_thresholds": report.get("entry_evidence_thresholds", {}),
        "market_entry_final_skip_kind_counts": report.get(
            "market_entry_final_skip_kind_counts", {}
        ),
        "market_entry_evidence_tier_counts": report.get("market_entry_evidence_tier_counts", {}),
        "market_entry_evidence_component_status_counts": report.get(
            "market_entry_evidence_component_status_counts", {}
        ),
        "market_entry_evidence_component_point_stats": report.get(
            "market_entry_evidence_component_point_stats", {}
        ),
        "market_entry_evidence_relief_applied_counts": report.get(
            "market_entry_evidence_relief_applied_counts", {}
        ),
        "market_entry_probe_recommendation_counts": report.get(
            "market_entry_probe_recommendation_counts", {}
        ),
        "market_entry_probe_conversion_ready_counts": report.get(
            "market_entry_probe_conversion_ready_counts", {}
        ),
        "market_entry_probe_conversion_block_reason_counts": report.get(
            "market_entry_probe_conversion_block_reason_counts", {}
        ),
        "market_entry_profit_probe_block_kind_counts": report.get(
            "market_entry_profit_probe_block_kind_counts", {}
        ),
        "market_entry_profit_probe_block_reason_counts": report.get(
            "market_entry_profit_probe_block_reason_counts", {}
        ),
        "market_entry_advisory_wait_reason_counts": report.get(
            "market_entry_advisory_wait_reason_counts", {}
        ),
        "market_entry_expected_net_component_stats": report.get(
            "market_entry_expected_net_component_stats", {}
        ),
        "market_entry_score_gap_stats": report.get("market_entry_score_gap_stats", {}),
        "market_entry_evidence_effective_score_stats": report.get(
            "market_entry_evidence_effective_score_stats", {}
        ),
        "market_entry_profit_quality_stats": report.get("market_entry_profit_quality_stats", {}),
        "market_entry_loss_probability_stats": report.get(
            "market_entry_loss_probability_stats", {}
        ),
        "market_entry_tail_risk_stats": report.get("market_entry_tail_risk_stats", {}),
        "position_size_pct_stats": report.get("position_size_pct_stats", {}),
        "quality_tier_counts": report.get("quality_tier_counts", {}),
        "low_payoff_entry_count": report.get("low_payoff_entry_count"),
        "low_payoff_reason_counts": report.get("low_payoff_reason_counts", {}),
        "low_payoff_missing_reason_count": report.get("low_payoff_missing_reason_count"),
        "entry_analysis_type_counts": report.get("entry_analysis_type_counts", {}),
        "entry_analysis_type_skip_kind_counts": report.get(
            "entry_analysis_type_skip_kind_counts", {}
        ),
        "entry_analysis_type_evidence_tier_counts": report.get(
            "entry_analysis_type_evidence_tier_counts", {}
        ),
        "entry_analysis_type_quality_tier_counts": report.get(
            "entry_analysis_type_quality_tier_counts", {}
        ),
        "entry_analysis_type_low_payoff_counts": report.get(
            "entry_analysis_type_low_payoff_counts", {}
        ),
        "entry_analysis_type_low_payoff_reason_counts": report.get(
            "entry_analysis_type_low_payoff_reason_counts", {}
        ),
        "entry_analysis_type_low_payoff_missing_reason_counts": report.get(
            "entry_analysis_type_low_payoff_missing_reason_counts", {}
        ),
        "entry_analysis_type_notional_floor_blocked_counts": _compact_text_counter(
            report.get("entry_analysis_type_notional_floor_blocked_counts", {})
        ),
        "entry_analysis_type_metric_stats": report.get("entry_analysis_type_metric_stats", {}),
        "high_risk_review_status_counts": report.get("high_risk_review_status_counts", {}),
        "high_risk_review_trigger_counts": report.get("high_risk_review_trigger_counts", {}),
        "high_risk_review_approved_counts": report.get("high_risk_review_approved_counts", {}),
        "high_risk_review_reason_counts": _compact_text_counter(
            report.get("high_risk_review_reason_counts", {})
        ),
        "high_risk_review_trigger_reason_counts": report.get(
            "high_risk_review_trigger_reason_counts", {}
        ),
        "notional_floor_blocked_counts": _compact_text_counter(
            report.get("notional_floor_blocked_counts", {})
        ),
        "entry_ai_expected_return_policy_counts": dict(ai_policy_counts.most_common(12)),
        "executed_entry_sizing_diagnostics": _compact_executed_entry_diagnostics(
            report.get("executed_entry_sizing_diagnostics")
        ),
        "entry_examples": examples[:6],
        "diagnostic_boundary": (
            "Read-only compact market entry diagnostics. Use it to identify whether "
            "entries are blocked by evidence, expected-net quality, sizing, ML readiness, "
            "or execution contract before changing thresholds, leverage, or position size."
        ),
    }


def _compact_executed_entry_sample(item: dict | None) -> dict:
    item = item if isinstance(item, dict) else {}
    decision = item.get("decision") if isinstance(item.get("decision"), dict) else {}
    order = item.get("order") if isinstance(item.get("order"), dict) else {}
    evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
    sizing = item.get("sizing") if isinstance(item.get("sizing"), dict) else {}
    execution_result = (
        item.get("execution_result") if isinstance(item.get("execution_result"), dict) else {}
    )
    return {
        "id": item.get("id"),
        "time": item.get("time"),
        "symbol": item.get("symbol"),
        "action": item.get("action"),
        "analysis_type": item.get("analysis_type"),
        "decision": _pick(
            decision,
            (
                "position_size_pct",
                "suggested_leverage",
                "was_executed",
                "executed_at",
                "execution_price",
            ),
        ),
        "order": (
            _pick(
                order,
                ("id", "status", "side", "quantity", "price", "notional", "filled_at"),
            )
            if order
            else None
        ),
        "evidence": _pick(
            evidence,
            (
                "tier",
                "effective_score",
                "entry_evidence_score",
                "expected_net_return_pct",
                "aggregate_expected_net_return_pct",
                "ai_expected_return_policy",
                "ai_expected_return_weight",
                "ai_expected_return_independent_probe_support",
                "profit_quality_ratio",
                "loss_probability",
                "tail_risk_score",
                "tradeable_probe",
                "shadow_only",
            ),
        ),
        "sizing": _pick(
            sizing,
            (
                "position_size_pct",
                "leverage",
                "final_notional_usdt",
                "quality_tier",
                "low_payoff_quality",
                "low_payoff_reasons",
                "notional_floor_applied",
                "notional_floor_blocked",
                "meaningful_size_reason",
                "strategy_probe_cap_applied",
                "strategy_max_probe_size_pct",
                "strategy_reason",
            ),
        ),
        "sizing_reason_tags": (
            item.get("sizing_reason_tags")
            if isinstance(item.get("sizing_reason_tags"), list)
            else []
        )[:10],
        "notional_gap_usdt": item.get("notional_gap_usdt"),
        "notional_fill_ratio": item.get("notional_fill_ratio"),
        "execution_result": _pick(
            execution_result,
            (
                "source",
                "status",
                "exchange_confirmed",
                "okx_symbol",
                "planned_order_contracts",
                "planned_base_quantity",
                "execution_blocker",
                "okx_rejection",
                "system_pre_submit_rejection",
            ),
        ),
    }


def _compact_executed_entry_diagnostics(value: dict | None) -> dict:
    value = value if isinstance(value, dict) else {}
    compact = _pick(
        value,
        (
            "read_only",
            "executed_entry_count",
            "market_executed_entry_count",
            "filled_order_count",
            "missing_order_count",
            "order_status_counts",
            "evidence_tier_counts",
            "sizing_quality_tier_counts",
            "sizing_reason_tag_counts",
            "order_notional_stats",
            "sizing_final_notional_stats",
            "notional_fill_ratio_stats",
            "decision_position_size_pct_stats",
            "decision_leverage_stats",
            "expected_net_stats",
            "profit_quality_stats",
            "loss_probability_stats",
            "tail_risk_stats",
            "diagnostic_boundary",
        ),
    )
    samples = value.get("samples") if isinstance(value.get("samples"), list) else []
    ai_policy_counts = Counter()
    for item in samples:
        evidence = item.get("evidence") if isinstance(item, dict) else {}
        policy = (
            evidence.get("ai_expected_return_policy") if isinstance(evidence, dict) else None
        ) or "missing"
        ai_policy_counts[str(policy)] += 1
    compact["ai_expected_return_policy_counts"] = dict(ai_policy_counts.most_common(12))
    compact["samples"] = [_compact_executed_entry_sample(item) for item in samples[:3]]
    return compact


def _compact_entry_example(item: dict | None) -> dict:
    item = item if isinstance(item, dict) else {}
    evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
    probe = item.get("probe") if isinstance(item.get("probe"), dict) else {}
    sizing = item.get("sizing") if isinstance(item.get("sizing"), dict) else {}
    state = item.get("state") if isinstance(item.get("state"), dict) else {}
    review = item.get("high_risk_review") if isinstance(item.get("high_risk_review"), dict) else {}
    order = item.get("order") if isinstance(item.get("order"), dict) else {}
    return {
        "id": item.get("id"),
        "time": item.get("time"),
        "symbol": item.get("symbol"),
        "action": item.get("action"),
        "analysis_type": item.get("analysis_type"),
        "executed": bool(item.get("executed")),
        "skip_kind": _compact_entry_skip_kind(item),
        "reason": str(item.get("reason") or "").replace("\n", " ").strip()[:320],
        "state": _pick(state, ("final_stage", "final_status", "blocked", "failed")),
        "evidence": _pick(
            evidence,
            (
                "tier",
                "effective_score",
                "entry_evidence_score",
                "score",
                "min_score_required",
                "expected_net_return_pct",
                "aggregate_expected_net_return_pct",
                "ai_expected_return_policy",
                "ai_expected_return_weight",
                "ai_expected_return_independent_probe_support",
                "profit_quality_ratio",
                "loss_probability",
                "tail_risk_score",
                "hard_block",
                "hard_block_reasons",
                "advisory_wait_reasons",
                "aligned_support_sources",
                "major_opposites",
                "weak_opposites",
                "strong_opposites",
                "tradeable_probe",
                "shadow_only",
            ),
        ),
        "probe": {
            "recommendation": probe.get("recommendation") or "",
            "probe_conversion_ready": probe.get("probe_conversion_ready"),
            "probe_conversion_block_reasons": (
                probe.get("probe_conversion_block_reasons")
                if isinstance(probe.get("probe_conversion_block_reasons"), list)
                else []
            ),
            "probe_conversion_thresholds": (
                probe.get("probe_conversion_thresholds")
                if isinstance(probe.get("probe_conversion_thresholds"), dict)
                else {}
            ),
            "evidence_profit_probe_blocked": _pick(
                (
                    probe.get("evidence_profit_probe_blocked")
                    if isinstance(probe.get("evidence_profit_probe_blocked"), dict)
                    else {}
                ),
                (
                    "blocked",
                    "block_kind",
                    "block_reasons",
                    "reason",
                    "expected_net_return_pct",
                    "profit_quality_ratio",
                    "loss_probability",
                    "tail_risk_score",
                    "thresholds",
                ),
            ),
        },
        "sizing": _pick(
            sizing,
            (
                "position_size_pct",
                "leverage",
                "final_notional_usdt",
                "quality_tier",
                "low_payoff_quality",
                "low_payoff_reasons",
                "notional_floor_applied",
                "notional_floor_blocked",
                "meaningful_size_reason",
                "strategy_sizing_applied",
                "strategy_probe_cap_applied",
                "strategy_max_probe_size_pct",
                "strategy_reason",
            ),
        ),
        "high_risk_review": (
            _pick(
                review,
                (
                    "triggered",
                    "status",
                    "approved",
                    "hard_review_required",
                    "reasons",
                    "advisory_reasons",
                    "reason",
                    "confidence",
                ),
            )
            if review
            else {}
        ),
        "order": _pick(order, ("status", "quantity", "price", "notional")) if order else None,
    }


def _compact_entry_skip_kind(item: dict | None) -> str:
    item = item if isinstance(item, dict) else {}
    if item.get("executed"):
        return "executed"
    reason = str(item.get("reason") or "")
    state = item.get("state") if isinstance(item.get("state"), dict) else {}
    final_stage = str(state.get("final_stage") or "").lower()
    final_status = str(state.get("final_status") or "").lower()
    if "弱证据学习档" in reason:
        return "entry_evidence_shadow_only"
    if "动态证据不足" in reason or "极小探针" in reason:
        return "entry_evidence_wait"
    if final_stage == "risk_check" and final_status == "blocked":
        high_risk_review = (
            item.get("high_risk_review") if isinstance(item.get("high_risk_review"), dict) else {}
        )
        if high_risk_review:
            return "high_risk_review_blocked"
        return "risk_check_blocked"
    if final_stage == "risk_check" and final_status == "skipped":
        return "risk_check_skipped"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect online strategy health.")
    parser.add_argument(
        "--minutes",
        type=int,
        default=480,
        help="Lookback window in minutes. Default: 480 (8 hours).",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print only stop-signal and contract summary fields.",
    )
    parser.add_argument(
        "--market-symbol-only",
        action="store_true",
        help="Print compact market symbol and candidate funnel diagnostics only.",
    )
    parser.add_argument(
        "--entry-only",
        action="store_true",
        help="Print compact entry evidence, sizing, and execution diagnostics only.",
    )
    args = parser.parse_args()
    minutes = max(int(args.minutes or 480), 1)
    token = secrets.token_hex(6)
    result_path = _remote_result_path(minutes, token)
    command = _build_remote_command(
        minutes,
        token=token,
        summary=args.summary,
        market_symbol_only=args.market_symbol_only,
        entry_only=args.entry_only,
        output_path=result_path,
    )

    ssh = connect_remote_ssh(ROOT, timeout=25)
    try:
        run_remote_text(ssh, command, timeout=220, max_output_chars=4000)
        sftp = ssh.open_sftp()
        try:
            with sftp.file(result_path, "r") as remote_file:
                out = remote_file.read().decode("utf-8", errors="replace")
        finally:
            try:
                sftp.remove(result_path)
            except OSError:
                pass
            sftp.close()
        safe_print(out)
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
