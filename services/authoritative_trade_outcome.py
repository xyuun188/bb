"""Canonical fee-after outcome events shared by OKX demo and live trades."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select

from core.runtime_data_retention_contract import is_ai_decision_retention_payload
from core.training_contracts import (
    AUTHORITATIVE_TRADE_LABEL_VERSION,
    AUTHORITATIVE_TRADE_OUTCOME_AUTHORITY,
    AUTHORITATIVE_TRADE_OUTCOME_SOURCES,
    AUTHORITATIVE_TRADE_OUTCOME_VERSION,
)
from db.session import get_read_session_ctx
from models.decision import AIDecision
from models.learning import ShadowBacktest, TradeReflection
from models.trade import Order, Position
from services.okx_position_history_store import load_okx_position_history_records
from services.okx_training_facts import build_okx_history_training_sample
from services.profit_training_contract import (
    PROFIT_TRAINING_TARGET,
    validate_profit_training_sample,
)

AUTHORITATIVE_TRADE_OUTCOME_CONSUMERS = (
    "local_ml",
    "local_ai_tools",
    "strategy_scheduler",
    "expert_memory",
    "dashboard",
)
_DERIVED_OUTCOME_KEYS = {
    "event_type",
    "outcome_id",
    "outcome_version",
    "outcome_fingerprint",
    "authority_level",
    "authority_rank",
    "settlement_fact_trusted",
    "actual_outcome_precedence",
    "reflection",
    "reflection_id",
    "reflection_status",
    "counterfactual_evidence",
    "counterfactual_production_weight",
    "attribution",
    "loss_attribution",
    "outcome_evidence_gaps",
    "outcome_complete",
    "consumer_provenance",
    "learning_summary",
}


def _safe_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _as_utc(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_execution_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode in {"paper", "demo", "sim", "simulation"}:
        return "paper"
    if mode in {"live", "real", "production"}:
        return "live"
    return ""


def _profit_label_contract(
    sample: dict[str, Any],
    *,
    evidence_gaps: list[str],
) -> dict[str, Any]:
    notional = _safe_float(sample.get("notional"), None)
    realized_pnl = _safe_float(sample.get("realized_pnl"), None)
    net_return = _safe_float(sample.get(PROFIT_TRAINING_TARGET), None)
    payload = {
        "version": AUTHORITATIVE_TRADE_LABEL_VERSION,
        "label_name": PROFIT_TRAINING_TARGET,
        "execution_mode": sample.get("execution_mode"),
        "lifecycle_key": sample.get("lifecycle_key"),
        "decision_id": int(sample.get("decision_id") or 0),
        "entry_order_id": sample.get("entry_order_id"),
        "close_order_id": sample.get("close_order_id"),
        "label_timestamp": sample.get("label_timestamp"),
        PROFIT_TRAINING_TARGET: net_return,
        "realized_net_pnl_usdt": realized_pnl,
        "gross_pnl_usdt": _safe_float(sample.get("gross_pnl"), None),
        "entry_fee_usdt": _safe_float(sample.get("entry_fee"), None),
        "close_fee_usdt": _safe_float(sample.get("close_fee"), None),
        "funding_fee_usdt": _safe_float(sample.get("funding_fee"), None),
        "liquidation_penalty_usdt": _safe_float(
            sample.get("liquidation_penalty"), None
        ),
        "notional": notional,
        "slippage_pct": _safe_float(sample.get("slippage"), None),
        "settlement_source": sample.get("settlement_source"),
        "entry_fee_source": sample.get("entry_fee_source"),
        "close_fee_source": sample.get("close_fee_source"),
        "funding_fee_source": sample.get("funding_fee_source"),
        "slippage_source": sample.get("slippage_source"),
        "complete": not evidence_gaps,
        "evidence_gaps": list(evidence_gaps),
    }
    return {**payload, "fingerprint": _fingerprint(payload)}


def _reflection_payload(reflection: Any | None) -> dict[str, Any]:
    if reflection is None:
        return {}
    return {
        "reflection_id": int(getattr(reflection, "id", 0) or 0),
        "position_id": int(getattr(reflection, "position_id", 0) or 0),
        "source": str(getattr(reflection, "source", "") or ""),
        "outcome": str(getattr(reflection, "outcome", "") or ""),
        "mistake_summary": str(getattr(reflection, "mistake_summary", "") or ""),
        "improvement_summary": str(getattr(reflection, "improvement_summary", "") or ""),
        "created_at": (
            reflection.created_at.isoformat()
            if isinstance(getattr(reflection, "created_at", None), datetime)
            else None
        ),
    }


def _shadow_counterfactuals(
    rows: list[Any],
    *,
    decision_id: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        if int(getattr(row, "decision_id", 0) or 0) != decision_id:
            continue
        if str(getattr(row, "status", "") or "").lower() != "completed":
            continue
        result.append(
            {
                "shadow_backtest_id": int(getattr(row, "id", 0) or 0),
                "horizon_minutes": int(getattr(row, "horizon_minutes", 0) or 0),
                "long_return_pct": _safe_float(getattr(row, "long_return_pct", None), None),
                "short_return_pct": _safe_float(getattr(row, "short_return_pct", None), None),
                "best_action": str(getattr(row, "best_action", "") or ""),
                "evidence_role": "counterfactual_observation_only",
                "authority_below_actual_outcome": True,
                "production_weight": 0.0,
                "may_override_actual_outcome": False,
            }
        )
    return sorted(result, key=lambda item: (item["horizon_minutes"], item["shadow_backtest_id"]))


def _unavailable_attribution(reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "contribution_usdt": None,
        "contribution_return_pct": None,
        "reason": reason,
        "may_be_inferred_from_shadow": False,
    }


def _attribution(sample: dict[str, Any]) -> dict[str, Any]:
    over_budget = _safe_float(sample.get("execution_actual_over_budget_loss_usdt"), None)
    execution_slippage_pct = _safe_float(sample.get("slippage"), None)
    execution_slippage_usdt = _safe_float(
        sample.get("execution_slippage_usdt"),
        None,
    )
    planned_stop = _safe_float(sample.get("planned_stop_loss_price"), None)
    entry_price = _safe_float(sample.get("entry_price"), None)
    planned_stop_distance_pct = (
        abs(planned_stop - entry_price) / entry_price * 100.0
        if planned_stop is not None
        and entry_price is not None
        and planned_stop > 0
        and entry_price > 0
        else None
    )
    return {
        "direction_error": _unavailable_attribution(
            "A causal direction contribution needs a same-lifecycle executable counterfactual."
        ),
        "entry_timing_error": _unavailable_attribution(
            "A causal entry-timing contribution needs an authoritative alternative fill path."
        ),
        "position_size_excess": {
            "status": "measured" if over_budget is not None else "unavailable",
            "contribution_usdt": -abs(over_budget) if over_budget is not None else None,
            "contribution_return_pct": None,
            "actual_over_budget_loss_usdt": over_budget,
            "source": "dynamic_risk_budget_vs_okx_realized_loss",
        },
        "stop_distance": {
            "status": "diagnostic" if planned_stop_distance_pct is not None else "unavailable",
            "contribution_usdt": None,
            "contribution_return_pct": None,
            "planned_stop_distance_pct": planned_stop_distance_pct,
            "reason": "Stop distance is observed but is not assigned a fabricated causal PnL value.",
        },
        "execution_slippage": {
            "status": "measured" if execution_slippage_usdt is not None else "unavailable",
            "contribution_usdt": (
                -execution_slippage_usdt
                if execution_slippage_usdt is not None
                else None
            ),
            "contribution_return_pct": (
                -abs(execution_slippage_pct)
                if execution_slippage_pct is not None
                else None
            ),
            "source": sample.get("slippage_source"),
            "entry_slippage_usdt": sample.get("entry_execution_slippage_usdt"),
            "close_slippage_usdt": sample.get("close_execution_slippage_usdt"),
        },
        "holding_duration_error": _unavailable_attribution(
            "A holding-duration contribution needs an authoritative path after alternative exits."
        ),
        "exit_timing_error": _unavailable_attribution(
            "A causal exit contribution needs executable alternative exit fills."
        ),
        "realized_costs": {
            "status": "measured",
            "entry_fee_usdt": _safe_float(sample.get("entry_fee"), None),
            "close_fee_usdt": _safe_float(sample.get("close_fee"), None),
            "funding_usdt": _safe_float(sample.get("funding_fee"), 0.0),
            "liquidation_penalty_usdt": _safe_float(sample.get("liquidation_penalty"), 0.0),
            "entry_fee_source": sample.get("entry_fee_source"),
            "close_fee_source": sample.get("close_fee_source"),
        },
        "causal_decomposition_complete": False,
        "unknown_components_are_zero": False,
    }


def build_authoritative_trade_outcome(
    sample: dict[str, Any],
    *,
    reflection: Any | None = None,
    shadow_rows: list[Any] | None = None,
) -> dict[str, Any]:
    """Promote one OKX lifecycle sample into the only real-trade outcome contract."""

    sample = dict(sample)
    source = str(sample.get("source") or "").strip()
    lifecycle_key = str(sample.get("lifecycle_key") or "").strip()
    if source not in AUTHORITATIVE_TRADE_OUTCOME_SOURCES or not lifecycle_key:
        raise ValueError("authoritative outcome requires one verified OKX settlement lifecycle")

    source_execution_mode = str(
        sample.get("source_execution_mode") or sample.get("execution_mode") or ""
    ).strip().lower()
    execution_mode = _canonical_execution_mode(sample.get("execution_mode"))
    sample["source_execution_mode"] = source_execution_mode
    sample["execution_mode"] = execution_mode
    decision_id = int(sample.get("decision_id") or 0)
    reflection_fact = _reflection_payload(reflection)
    counterfactuals = _shadow_counterfactuals(
        list(shadow_rows or []),
        decision_id=decision_id,
    )
    gaps = list(
        dict.fromkeys(
            str(value)
            for value in sample.get("training_evidence_gaps", [])
            if value and str(value) != "missing_trade_reflection_link"
        )
    )
    if not execution_mode:
        gaps.append("missing_or_invalid_execution_mode")
    if not decision_id:
        gaps.append("missing_exact_entry_order_decision_link")
    gaps = list(dict.fromkeys(gaps))
    authority = str(sample.get("decision_authority") or "").strip().lower()
    sample["decision_authority"] = authority
    profit_contract = validate_profit_training_sample(sample)
    sample["profit_training_contract"] = profit_contract.to_dict()
    gaps.extend(
        f"profit_training_contract:{blocker}" for blocker in profit_contract.blockers
    )
    gaps = list(dict.fromkeys(gaps))
    label_contract = _profit_label_contract(sample, evidence_gaps=gaps)
    sample["training_label_contract"] = label_contract

    identity = {
        "version": AUTHORITATIVE_TRADE_OUTCOME_VERSION,
        "mode": sample.get("execution_mode"),
        "lifecycle_key": lifecycle_key,
        "okx_pos_id": sample.get("okx_pos_id"),
    }
    outcome_id = f"ato:{_fingerprint(identity)[:24]}"
    immutable_facts = {
        key: value
        for key, value in sample.items()
        if key not in _DERIVED_OUTCOME_KEYS
    }
    immutable_facts.update(
        {
            "outcome_id": outcome_id,
            "outcome_version": AUTHORITATIVE_TRADE_OUTCOME_VERSION,
        }
    )
    outcome_fingerprint = _fingerprint(immutable_facts)
    outcome = dict(sample)
    outcome.update(
        {
            "event_type": "AuthoritativeTradeOutcome",
            "outcome_id": outcome_id,
            "outcome_version": AUTHORITATIVE_TRADE_OUTCOME_VERSION,
            "outcome_fingerprint": outcome_fingerprint,
            "authority_level": AUTHORITATIVE_TRADE_OUTCOME_AUTHORITY,
            "authority_rank": 100,
            "settlement_fact_trusted": bool(sample.get("trade_fact_trusted")),
            "actual_outcome_precedence": "authoritative",
            "reflection": reflection_fact,
            "reflection_id": reflection_fact.get("reflection_id"),
            "reflection_status": "available" if reflection_fact else "pending_optional",
            "counterfactual_evidence": counterfactuals,
            "counterfactual_production_weight": 0.0,
            "attribution": _attribution(sample),
            "loss_attribution": (
                "authoritative_multi_factor_outcome"
                if (_safe_float(sample.get("realized_pnl"), 0.0) or 0.0) < 0
                else ""
            ),
            "outcome_evidence_gaps": gaps,
            "outcome_complete": not gaps,
            "consumer_provenance": {
                "contract_version": AUTHORITATIVE_TRADE_OUTCOME_VERSION,
                "outcome_id": outcome_id,
                "outcome_fingerprint": outcome_fingerprint,
                "training_label_version": AUTHORITATIVE_TRADE_LABEL_VERSION,
                "training_label_fingerprint": label_contract["fingerprint"],
                "consumers": list(AUTHORITATIVE_TRADE_OUTCOME_CONSUMERS),
                "single_owner": "services.authoritative_trade_outcome",
            },
            "learning_summary": {
                "objective": "maximize_expected_realized_net_return_after_cost",
                "realized_net_pnl_usdt": sample.get("realized_pnl"),
                PROFIT_TRAINING_TARGET: label_contract.get(PROFIT_TRAINING_TARGET),
                "execution_slippage_pct": sample.get("slippage"),
                "actual_over_budget_loss_usdt": sample.get(
                    "execution_actual_over_budget_loss_usdt"
                ),
                "uncertainty_rule": (
                    "Actual loss and tail execution facts recalibrate the distribution; "
                    "one loss never hard-codes the opposite direction."
                ),
            },
        }
    )
    outcome["training_evidence_gaps"] = gaps
    outcome["trade_fact_trusted"] = bool(sample.get("trade_fact_trusted")) and not gaps
    outcome["trade_fact_trust_reason"] = gaps[0] if gaps else ""
    return outcome


async def load_authoritative_trade_outcomes(
    *,
    mode: str | None = None,
    since: datetime | None = None,
    limit: int | None = None,
    session_factory: Callable[[], AbstractAsyncContextManager[Any]] = get_read_session_ctx,
) -> list[dict[str, Any]]:
    """Load deterministic outcome events; no local-position PnL fallback is allowed."""

    async with session_factory() as session:
        requested_limit = max(int(limit), 1) if limit is not None else 5000
        histories = await load_okx_position_history_records(
            session,
            mode=mode,
            limit=requested_limit,
        )
        since_utc = _as_utc(since)
        if since_utc is not None:
            histories = [
                history
                for history in histories
                if (_as_utc(history.updated_at_okx) or datetime.min.replace(tzinfo=UTC))
                >= since_utc
            ]
        if limit is not None:
            histories = histories[:requested_limit]

        position_ids = {
            int(value)
            for history in histories
            for value in (history.position_ids or [])
            if str(value or "").isdigit() and int(value) > 0
        }
        exchange_order_ids = {
            str(value or "").strip()
            for history in histories
            for value in [
                *(history.entry_order_ids or []),
                *(history.close_order_ids or []),
                *(history.linked_order_ids or []),
            ]
            if str(value or "").strip()
        }
        positions = (
            list(
                (
                    await session.execute(select(Position).where(Position.id.in_(position_ids)))
                ).scalars().all()
            )
            if position_ids
            else []
        )
        orders = (
            list(
                (
                    await session.execute(
                        select(Order).where(Order.exchange_order_id.in_(exchange_order_ids))
                    )
                ).scalars().all()
            )
            if exchange_order_ids
            else []
        )
        decision_ids = {
            int(order.decision_id or 0) for order in orders if int(order.decision_id or 0) > 0
        }
        decisions = (
            list(
                (
                    await session.execute(
                        select(AIDecision).where(AIDecision.id.in_(decision_ids))
                    )
                ).scalars().all()
            )
            if decision_ids
            else []
        )
        reflections = (
            list(
                (
                    await session.execute(
                        select(TradeReflection).where(TradeReflection.position_id.in_(position_ids))
                    )
                ).scalars().all()
            )
            if position_ids
            else []
        )
        shadows = (
            list(
                (
                    await session.execute(
                        select(ShadowBacktest).where(ShadowBacktest.decision_id.in_(decision_ids))
                    )
                ).scalars().all()
            )
            if decision_ids
            else []
        )

    positions_by_id = {int(row.id): row for row in positions}
    orders_by_exchange_id = {
        str(row.exchange_order_id): row for row in orders if str(row.exchange_order_id or "").strip()
    }
    decisions_by_id = {int(row.id): row for row in decisions}
    decision_raw_by_order_id = {
        str(order.exchange_order_id): _decision_learning_payload(
            decisions_by_id.get(int(order.decision_id or 0))
        )
        for order in orders
        if str(order.exchange_order_id or "").strip() and int(order.decision_id or 0) > 0
    }
    decision_execution_by_order_id = {
        str(order.exchange_order_id): {
            "decision_id": int(decision.id or 0),
            "model_name": str(decision.model_name or ""),
            "stop_loss_pct": _safe_float(decision.stop_loss_pct, None),
            "take_profit_pct": _safe_float(decision.take_profit_pct, None),
        }
        for order in orders
        if str(order.exchange_order_id or "").strip()
        and int(order.decision_id or 0) > 0
        and (decision := decisions_by_id.get(int(order.decision_id or 0))) is not None
    }
    reflections_by_position_id = {
        int(row.position_id): row
        for row in sorted(reflections, key=lambda item: int(item.id or 0), reverse=True)
        if int(row.position_id or 0) > 0
    }
    results: list[dict[str, Any]] = []
    for history in histories:
        sample = build_okx_history_training_sample(
            history,
            positions_by_id=positions_by_id,
            orders_by_exchange_id=orders_by_exchange_id,
            decision_raw_by_order_id=decision_raw_by_order_id,
            decision_execution_by_order_id=decision_execution_by_order_id,
        )
        reflection = next(
            (
                reflections_by_position_id[position_id]
                for position_id in sample.get("position_ids", [])
                if position_id in reflections_by_position_id
            ),
            None,
        )
        results.append(
            build_authoritative_trade_outcome(
                sample,
                reflection=reflection,
                shadow_rows=shadows,
            )
        )
    results.reverse()
    return results


def _decision_learning_payload(decision: Any | None) -> dict[str, Any]:
    if decision is None:
        return {}
    raw = _safe_dict(getattr(decision, "raw_llm_response", None))
    if raw and not is_ai_decision_retention_payload(raw):
        return raw
    return _safe_dict(getattr(decision, "decision_learning_snapshot", None))
