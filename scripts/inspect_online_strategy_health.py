from __future__ import annotations

import argparse
import secrets
import sys
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
from services.ml_signal_service import MLSignalService
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


def roundv(value, digits=6):
    return round(safe_float(value), digits)


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
        "final_reason": summary.get("final_reason") or decision.execution_reason or "",
        "completed_stage_count": summary.get("completed_stage_count"),
        "blocked": bool(summary.get("blocked")),
        "failed": bool(summary.get("failed")),
    }


def short_text(text, limit=260):
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
        "planned_order_contracts": roundv(raw_result.get("planned_order_contracts")),
        "planned_base_quantity": roundv(raw_result.get("planned_base_quantity")),
        "okx_order_rules": safe_dict(raw_result.get("okx_order_rules")),
        "request_params": safe_dict(raw_result.get("request_params")),
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


def pick(mapping, keys):
    mapping = safe_dict(mapping)
    return {key: mapping.get(key) for key in keys}


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
            "fast_loss_samples": safe_list(contract.get("fast_loss_samples")),
            "violations": safe_list(contract.get("violations")),
        },
        "local_ml_readiness": {
            "status": local_ml.get("status"),
            "readiness_state": local_ml.get("readiness_state"),
            "allow_live_position_influence": local_ml.get(
                "allow_live_position_influence"
            ),
            "blocking_reason_codes": safe_list(local_ml.get("blocking_reason_codes")),
            "metrics": safe_dict(local_ml.get("metrics")),
        },
        "rejected_order_examples": safe_list(report.get("rejected_order_examples")),
        "fast_loss_positions": safe_list(report.get("fast_loss_positions")),
    }


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
    market_entry_advisory_wait_reason_counts = Counter()
    market_entry_shadow_only_count = 0
    market_entry_tradeable_probe_count = 0
    market_entry_hard_block_count = 0
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
        if analysis_type(d) == "market":
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
        reason = st["final_reason"] or d.execution_reason or ""
        reason_counts[reason[:100] if reason else "无原因"] += 1
        state_counts[f"{st['final_stage']}:{st['final_status']}"] += 1
        expected_values.append(ev["expected_net_return_pct"])
        size_values.append(roundv(d.position_size_pct))
        quality_tiers[sz["quality_tier"] or "unknown"] += 1
        if sz["low_payoff_quality"]:
            low_payoff_count += 1
        if sz["notional_floor_blocked"]:
            notional_floor_blocked[sz["notional_floor_blocked"]] += 1
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
        raw = safe_dict(d.raw_llm_response)
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
                "reason": short_text(reason, 260),
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
                "reason": short_text(reason, 280),
                "state": st,
                "evidence": ev,
                "sizing": sz,
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
        "entry_candidate_evidence_by_type": dict(
            entry_candidate_evidence_by_type.most_common(20)
        ),
        "entry_state_counts": dict(state_counts.most_common(20)),
        "entry_reason_counts": [{"reason_prefix": k, "count": v} for k, v in reason_counts.most_common(20)],
        "expected_net_stats": stats(expected_values),
        "position_size_pct_stats": stats(size_values),
        "quality_tier_counts": dict(quality_tiers.most_common(20)),
        "low_payoff_entry_count": low_payoff_count,
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
    output = summary_report(report) if SUMMARY_ONLY else report
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


def _build_remote_command(minutes: int, *, token: str | None = None, summary: bool = False) -> str:
    safe_minutes = max(int(minutes or 480), 1)
    safe_token = token or secrets.token_hex(6)
    tmp_dir = "/data/bb/app/tmp/codex-strategy-health"
    sample_path = f"{tmp_dir}/sample_{safe_minutes}_{safe_token}.py"
    launcher_path = f"{tmp_dir}/launcher_{safe_minutes}_{safe_token}.py"
    remote_script = REMOTE_SCRIPT_TEMPLATE.replace("__WINDOW_MINUTES__", str(safe_minutes)).replace(
        "__SUMMARY_ONLY__", "True" if summary else "False"
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
python3 {launcher_path} {sample_path}
rm -f {sample_path} {launcher_path}
"""


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
            "fast_loss_samples": contract.get("fast_loss_samples", []),
            "violations": contract.get("violations", []),
        },
        "local_ml_readiness": {
            "status": local_ml.get("status"),
            "readiness_state": local_ml.get("readiness_state"),
            "allow_live_position_influence": local_ml.get("allow_live_position_influence"),
            "blocking_reason_codes": local_ml.get("blocking_reason_codes", []),
            "metrics": local_ml.get("metrics", {}),
        },
        "rejected_order_examples": report.get("rejected_order_examples", []),
        "fast_loss_positions": report.get("fast_loss_positions", []),
    }


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
    args = parser.parse_args()
    minutes = max(int(args.minutes or 480), 1)
    command = _build_remote_command(minutes, summary=args.summary)

    ssh = connect_remote_ssh(ROOT, timeout=25)
    try:
        out = run_remote_text(ssh, command, timeout=220, max_output_chars=100000)
        safe_print(out)
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
