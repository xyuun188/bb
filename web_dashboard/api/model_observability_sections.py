"""Read-only model and trade evidence sections used by Dashboard observability.

The section readers live outside the Dashboard route module so expensive database
reads stay isolated from HTTP orchestration and can be independently bounded.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select

from core.safe_output import safe_error_text
from db.repositories.memory_repo import MemoryRepository
from db.session import get_session_ctx
from models.decision import AIDecision
from models.learning import ExpertMemory, TradeReflection
from services import authoritative_trade_outcome


def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if value is None or value == "":
            return default
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


async def build_authoritative_profit_observability(
    *,
    mode: str | None = None,
    since_hours: float | None = None,
) -> dict[str, Any]:
    """Summarise only complete trusted OKX outcomes; shadow data is excluded."""

    try:
        outcomes = await authoritative_trade_outcome.load_authoritative_trade_outcomes(
            mode=mode if mode in {"paper", "live"} else None,
            since=(
                datetime.now(UTC) - timedelta(hours=float(since_hours))
                if since_hours is not None
                else None
            ),
            limit=500,
            compact=True,
        )
    except Exception as exc:
        return {
            "status": "error",
            "degraded_reason": (
                "authoritative_outcomes_unavailable:"
                + safe_error_text(exc, limit=120)
            ),
            "observed": False,
        }

    complete = [
        row
        for row in outcomes
        if isinstance(row, dict)
        and row.get("outcome_complete") is True
        and row.get("trade_fact_trusted") is True
    ]
    totals = {
        "gross_pnl": 0.0,
        "fee": 0.0,
        "slippage": 0.0,
        "funding_fee": 0.0,
        "liquidation_penalty": 0.0,
        "fee_after_net_pnl": 0.0,
        "realized_net_pnl": 0.0,
    }
    mismatch_count = 0
    equation_observed_count = 0
    for row in complete:
        gross = _safe_float(row.get("gross_pnl_usdt", row.get("gross_pnl")), None)
        entry_fee = _safe_float(row.get("entry_fee_usdt", row.get("entry_fee")), None)
        close_fee = _safe_float(row.get("close_fee_usdt", row.get("close_fee")), None)
        slippage = _safe_float(
            row.get("execution_slippage_usdt", row.get("slippage_cost_usdt")), None
        )
        funding = _safe_float(row.get("funding_fee_usdt", row.get("funding_fee")), None)
        penalty = _safe_float(
            row.get("liquidation_penalty_usdt", row.get("liquidation_penalty")),
            0.0,
        )
        realized = _safe_float(
            row.get("realized_net_pnl_usdt", row.get("realized_pnl")),
            None,
        )
        if gross is not None:
            totals["gross_pnl"] += gross
        if entry_fee is not None:
            totals["fee"] += entry_fee
        if close_fee is not None:
            totals["fee"] += close_fee
        if slippage is not None:
            totals["slippage"] += slippage
        if funding is not None:
            totals["funding_fee"] += funding
        if penalty is not None:
            totals["liquidation_penalty"] += penalty
        if realized is not None:
            totals["realized_net_pnl"] += realized
        if None not in (gross, entry_fee, close_fee, slippage, funding, penalty, realized):
            totals["fee_after_net_pnl"] += (
                gross - entry_fee - close_fee - slippage + funding - penalty
            )
        components = row.get("realized_net_pnl_components")
        if isinstance(components, dict):
            expected = _safe_float(components.get("components_total_usdt"), None)
            reported = _safe_float(
                components.get("reported_realized_net_pnl_usdt", realized),
                None,
            )
            if expected is not None and reported is not None:
                equation_observed_count += 1
                if not math.isclose(expected, reported, rel_tol=1e-5, abs_tol=1e-5):
                    mismatch_count += 1
    status = "ok" if complete and mismatch_count == 0 else "missing" if not outcomes else "partial"
    return {
        "status": status,
        "observed": bool(complete),
        "sample_count": len(complete),
        "excluded_incomplete_count": max(len(outcomes) - len(complete), 0),
        "equation_observed_count": equation_observed_count,
        "attribution_mismatch_count": mismatch_count,
        "formula": "realized_net_pnl = gross_pnl - fee - slippage + funding_fee - liquidation_penalty",
        "totals": {key: round(value, 8) for key, value in totals.items()},
        "shadow_excluded": True,
        "degraded_reason": (
            "authoritative_outcome_not_found" if not outcomes else None
        ),
    }


async def build_expert_memory_observability(
    *,
    trading_service_active: bool,
    mode: str | None = None,
) -> dict[str, Any]:
    """Expose authoritative memory counts without treating pending rows as evidence."""

    if not trading_service_active:
        return {
            "status": "deferred",
            "observed": False,
            "degraded_reason": "trading_service_inactive",
            "source": "dashboard.expert_memory_observability",
        }

    try:
        outcomes = await authoritative_trade_outcome.load_authoritative_trade_outcomes(
            mode=mode if mode in {"paper", "live"} else None,
            limit=500,
            compact=True,
        )
        outcome_positions = {
            int(position_id)
            for outcome in outcomes
            for position_id in (
                outcome.get("position_ids") or [outcome.get("position_id")]
            )
            if str(position_id or "").isdigit() and int(position_id) > 0
        }
        complete_outcomes = [
            outcome
            for outcome in outcomes
            if outcome.get("outcome_complete") is True
            and outcome.get("trade_fact_trusted") is True
        ]
        shadow_count = sum(
            len(outcome.get("counterfactual_evidence") or []) for outcome in outcomes
        )
        eligible_count = sum(
            bool(
                outcome.get("outcome_complete") is True
                and outcome.get("trade_fact_trusted") is True
            )
            for outcome in outcomes
        )
        async with get_session_ctx() as session:
            repo = MemoryRepository(session)
            memory_count = await repo.count_memories()
            reflection_count = await repo.count_reflections()
            reflection_rows = list(
                (await session.execute(select(TradeReflection.position_id).limit(5000)))
                .scalars()
                .all()
            )
            memory_extra_rows = list(
                (await session.execute(select(ExpertMemory.extra).limit(5000)))
                .scalars()
                .all()
            )
            authoritative_memory_count = sum(
                isinstance(extra, dict) and bool(str(extra.get("outcome_id") or "").strip())
                for extra in memory_extra_rows
            )
        orphan_reflection_count = sum(
            int(position_id or 0) > 0 and int(position_id) not in outcome_positions
            for position_id in reflection_rows
        )
        pending_count = max(reflection_count - len(complete_outcomes), 0)
        return {
            "status": (
                "ok"
                if complete_outcomes and len(complete_outcomes) == len(outcomes)
                else "partial"
                if outcomes
                else "missing"
            ),
            "memory_count": memory_count,
            "reflection_count": reflection_count,
            "authoritative_outcome_count": len(outcomes),
            "complete_authoritative_outcome_count": len(complete_outcomes),
            "pending_settlement_count": pending_count,
            "orphan_position_count": orphan_reflection_count,
            "shadow_sample_count": shadow_count,
            "production_evidence_eligible_count": eligible_count,
            "authoritative_memory_count": authoritative_memory_count,
            "shadow_production_weight": 0.0,
        }
    except Exception as exc:
        return {
            "status": "error",
            "degraded_reason": (
                "expert_memory_observability_unavailable:"
                + safe_error_text(exc, limit=120)
            ),
        }


async def latest_analysis_observability(
    *,
    ensemble_trader_name: str,
) -> dict[str, Any]:
    """Read one bounded analysis-quality sample for the model observability card."""

    try:
        from sqlalchemy import select

        async with get_session_ctx() as session:
            result = await session.execute(
                select(
                    AIDecision.raw_llm_response,
                    AIDecision.created_at,
                    AIDecision.id,
                )
                .where(AIDecision.model_name == ensemble_trader_name)
                .order_by(AIDecision.created_at.desc(), AIDecision.id.desc())
                .limit(50)
            )
            for raw, created_at, decision_id in result.all():
                payload = raw if isinstance(raw, dict) else {}
                quality = payload.get("analysis_quality_contract")
                if not isinstance(quality, dict):
                    continue
                counts = quality.get("status_counts")
                counts = counts if isinstance(counts, dict) else {}
                expected = int(quality.get("expected_expert_count") or 0)
                attempted = int(quality.get("attempted_expert_count") or 0)
                returned = int(quality.get("returned_expert_count") or 0)
                successful = int(quality.get("successful_expert_count") or 0)
                quality_status = "ok" if expected > 0 else "partial"
                return {
                    "status": quality_status,
                    "decision_id": decision_id,
                    "round_id": payload.get("round_id") or quality.get("round_id"),
                    "checked_at": created_at.isoformat() if created_at else None,
                    "expected_expert_count": expected,
                    "attempted_expert_count": attempted,
                    "returned_expert_count": returned,
                    "successful_expert_count": successful,
                    "failed_expert_count": sum(
                        int(counts.get(key) or 0)
                        for key in ("timeout", "parse_failed", "empty", "unavailable")
                    ),
                    "skipped_expert_count": int(counts.get("skipped") or 0),
                    "analysis_complete": quality.get("analysis_complete"),
                    "decision_eligible": quality.get("decision_eligible"),
                    "reason_code": quality.get("reason_code")
                    or ("expected_expert_count_zero" if expected == 0 else None),
                }
    except Exception as exc:
        return {
            "status": "error",
            "degraded_reason": (
                "analysis_observability_unavailable:"
                + safe_error_text(exc, limit=120)
            ),
        }
    return {
        "status": "missing",
        "degraded_reason": "analysis_quality_contract_not_found",
    }
