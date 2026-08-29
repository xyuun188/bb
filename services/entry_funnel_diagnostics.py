"""Auditable, read-only diagnostics for the new-entry decision funnel."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

ENTRY_FUNNEL_REASONS = (
    "no_candidate",
    "insufficient_evidence",
    "direction_conflict",
    "risk_blocked",
    "funding_cost_blocked",
    "pre_order_facts_blocked",
    "execution_blocked",
    "account_reconciliation_blocked",
    "service_error",
)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str:
    return str(value or "").strip()


_REASON_FIELDS = frozenset(
    {
        "reason",
        "reason_code",
        "final_reason",
        "final_reason_code",
        "failure_reason",
        "error",
        "error_code",
        "blocker",
        "blockers",
        "blocking_reason",
        "blocking_reasons",
        "block_reasons",
        "execution_blocker",
    }
)


def _reason_fragments(value: Any) -> list[str]:
    fragments: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = str(key or "").strip().lower()
            if normalized_key in _REASON_FIELDS and item not in (None, {}, [], ""):
                fragments.append(str(item))
            if isinstance(item, (dict, list, tuple)):
                fragments.extend(_reason_fragments(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, (dict, list, tuple)):
                fragments.extend(_reason_fragments(item))
    return fragments


def _reason_blob(raw: dict[str, Any], reason: Any) -> str:
    # Diagnostic payloads always contain keys such as ``funding_cost`` and
    # ``risk``. Only reason-bearing values may classify a block; field names
    # alone are not evidence that the corresponding gate rejected an entry.
    return " ".join([_text(reason), *_reason_fragments(raw)]).lower()


def _shallow_reason_fragments(value: Any) -> list[str]:
    fragments: list[str] = []
    if not isinstance(value, dict):
        return fragments
    for key, item in value.items():
        normalized_key = str(key or "").strip().lower()
        if normalized_key not in _REASON_FIELDS or item in (None, {}, [], ""):
            continue
        if isinstance(item, (list, tuple)):
            fragments.extend(str(part) for part in item if not isinstance(part, (dict, list, tuple)))
        elif not isinstance(item, dict):
            fragments.append(str(item))
    return fragments


def _explicit_risk_reason_blob(payload: dict[str, Any], reason: Any) -> str:
    """Read only decision/execution blockers, never expert narrative text."""

    fragments = [_text(reason), *_shallow_reason_fragments(payload)]
    for key in (
        "decision_state_machine",
        "decision_state",
        "execution_trace",
        "policy_gate",
        "trade_gate",
    ):
        container = _dict(payload.get(key))
        fragments.extend(_shallow_reason_fragments(container))
        fragments.extend(_shallow_reason_fragments(_dict(container.get("summary"))))
        fragments.extend(_shallow_reason_fragments(_dict(container.get("failed_step"))))
    return " ".join(fragment for fragment in fragments if fragment).lower()


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _funding_crossed_net_zero(opportunity: dict[str, Any]) -> bool:
    funding = _dict(opportunity.get("funding_cost"))
    if funding.get("production_eligible") is not True:
        return False
    cashflow = _safe_float(funding.get("signed_cashflow_pct"))
    net_return = _safe_float(
        opportunity.get("expected_realized_net_return_pct")
        if opportunity.get("expected_realized_net_return_pct") is not None
        else opportunity.get("expected_net_return_pct")
    )
    return bool(
        cashflow is not None
        and cashflow < 0.0
        and net_return is not None
        and net_return <= 0.0
        and net_return - cashflow > 0.0
    )


def classify_entry_funnel_reason(
    *,
    raw: dict[str, Any] | None,
    action: Any,
    was_executed: bool,
    has_order: bool,
    reason: Any = None,
) -> str | None:
    """Classify one non-executed market decision without changing permission."""

    if was_executed:
        return None
    payload = _dict(raw)
    text = _reason_blob(payload, reason)
    explicit_risk_text = _explicit_risk_reason_blob(payload, reason)
    action_text = _text(action).lower()
    quality = _dict(payload.get("analysis_quality_contract"))
    state = _dict(payload.get("decision_state_machine") or payload.get("decision_state"))
    summary = _dict(state.get("summary"))
    final_code = _text(
        summary.get("final_reason_code")
        or quality.get("reason_code")
        or _dict(payload.get("execution_trace")).get("reason_code")
    ).upper()
    authoritative_sync = _dict(payload.get("okx_authoritative_sync"))
    production_gate = _dict(payload.get("production_trade_gate"))
    opportunity = _dict(payload.get("opportunity_score"))
    funding = _dict(opportunity.get("funding_cost"))
    direction = _dict(payload.get("direction_competition"))
    high_risk_review = _dict(payload.get("high_risk_review"))
    quality_complete = bool(
        quality.get("analysis_complete") is True
        and quality.get("decision_eligible") is True
    )
    explicit_runtime_failure = bool(
        payload.get("market_model_timeout")
        or payload.get("expert_failures")
        or final_code in {"MODEL_UNAVAILABLE", "MODEL_TIMEOUT", "MODEL_INVALID_OUTPUT"}
    )

    # A directional signal may be generated while the execution layer still
    # lacks a fresh, authoritative OKX fact snapshot. This is a deliberate
    # fail-closed pre-submit decision, not an exchange rejection.
    pre_order_fact_tokens = (
        "pre_order_execution_facts_ineligible",
        "pre_order_execution_facts_fingerprint_missing",
        "pre_order_execution_facts_unavailable",
        "authoritative pre-order execution facts are unavailable",
        "authoritative pre-order execution facts are incomplete",
        "execution facts are unavailable",
        "execution facts are incomplete",
        "交易前权威事实不可用",
        "交易前权威事实不完整",
        "交易前执行事实不可用",
        "交易前执行事实不完整",
    )
    pre_order_fact_blocked = bool(
        action_text in {"long", "short"}
        and (
            any(token in text or token in explicit_risk_text for token in pre_order_fact_tokens)
            or final_code in {
                "PRE_ORDER_EXECUTION_FACTS_UNAVAILABLE",
                "PRE_ORDER_EXECUTION_FACTS_INCOMPLETE",
            }
        )
    )
    if pre_order_fact_blocked:
        return "pre_order_facts_blocked"

    if any(
        token in text
        for token in (
            "account_reconciliation",
            "authoritative_sync",
            "reconciliation",
            "对账",
            "仓位不一致",
            "账户状态未确认",
        )
    ) or final_code in {"RECONCILIATION_MISMATCH", "RECONCILIATION_PENDING"} or (
        authoritative_sync
        and (
            authoritative_sync.get("can_open_new_entries") is False
            or (_safe_float(authoritative_sync.get("unresolved_count")) or 0.0) > 0.0
        )
    ):
        return "account_reconciliation_blocked"

    if (not quality_complete or explicit_runtime_failure) and (any(
        token in text
        for token in (
            "service_error",
            "model_timeout",
            "market_model_timeout",
            "local_ai_tools_context_timeout",
            "专家协作整体超过",
            "服务异常",
            "接口异常",
            "provider_error",
            "超时",
        )
    ) or explicit_runtime_failure):
        return "service_error"

    if (
        final_code.startswith("FUNDING_EVIDENCE_")
        or _text(quality.get("reason_code")) == "funding_evidence_unavailable"
        or "funding_evidence_unavailable" in text
        or "资金费证据不可用" in text
    ):
        return "insufficient_evidence"

    if funding.get("blocked") is True or _funding_crossed_net_zero(opportunity) or any(
        token in text
        for token in (
            "funding_cost_blocked",
            "entry_funding_cost",
            "funding cost blocked",
            "资金费成本阻断",
            "资金费支出侵蚀",
        )
    ) or final_code.startswith("FUNDING_"):
        return "funding_cost_blocked"

    if any(
        token in text
        for token in (
            "direction_conflict",
            "direction conflict",
            "ml_ai_direction_conflict",
            "方向冲突",
            "多空冲突",
            "证据冲突",
        )
    ) or final_code in {"PROFIT_GATE_INSUFFICIENT_EVIDENCE"} and direction.get(
        "blocked"
    ) is True or high_risk_review.get("ml_ai_direction_conflict") is True:
        return "direction_conflict"

    if (
        quality.get("analysis_complete") is False
        or quality.get("decision_eligible") is False
        or "insufficient_evidence" in text
        or "证据不足" in text
        or final_code == "SIGNAL_UNAVAILABLE"
    ):
        return "insufficient_evidence"

    if any(
        token in explicit_risk_text
        for token in (
            "risk_blocked",
            "risk blocked",
            "risk_check_blocked",
            "risk_check_failed",
            "risk_limit_exceeded",
            "risk budget exhausted",
            "风控阻断",
            "风险阻断",
            "风险门禁",
            "熔断触发",
            "余额不足",
            "保证金不足",
        )
    ) or final_code.startswith("RISK_") or final_code == "ACCOUNT_BALANCE" or (
        production_gate.get("allowed") is False
        and _dict(production_gate.get("risk")).get("blocked") is True
    ):
        return "risk_blocked"

    if has_order or action_text in {"long", "short"}:
        return "execution_blocked"
    return "no_candidate"


def build_direction_symmetry_report(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize long/short selection without making a trading recommendation."""

    sides: dict[str, dict[str, Any]] = {
        "long": {
            "scan_count": 0,
            "signal_count": 0,
            "observation_count": 0,
            "executed_count": 0,
            "blocked_count": 0,
            "block_reasons": {},
        },
        "short": {
            "scan_count": 0,
            "signal_count": 0,
            "observation_count": 0,
            "executed_count": 0,
            "blocked_count": 0,
            "block_reasons": {},
        },
    }
    for row in rows:
        side = _text(row.get("action") or row.get("side")).lower()
        if side not in sides:
            continue
        item = sides[side]
        item["scan_count"] += 1
        analysis_only = row.get("analysis_only") is True
        if analysis_only:
            item["observation_count"] += 1
        elif row.get("is_entry") is True or side in {"long", "short"}:
            item["signal_count"] += 1
        if not analysis_only and row.get("was_executed", row.get("executed")) is True:
            item["executed_count"] += 1
        elif not analysis_only:
            item["blocked_count"] += 1
            reason = _text(row.get("funnel_reason")) or "unknown"
            counts = item["block_reasons"]
            counts[reason] = int(counts.get(reason, 0)) + 1

    for item in sides.values():
        item["signal_rate"] = (
            item["signal_count"] / item["scan_count"] if item["scan_count"] else 0.0
        )
        item["execution_rate"] = (
            item["executed_count"] / item["signal_count"]
            if item["signal_count"]
            else 0.0
        )
        item["block_reasons"] = dict(
            sorted(item["block_reasons"].items(), key=lambda pair: (-pair[1], pair[0]))
        )

    long_signals = sides["long"]["signal_count"]
    short_signals = sides["short"]["signal_count"]
    total_signals = long_signals + short_signals
    if total_signals < 4:
        status = "insufficient_data"
        dominant = None
    elif not long_signals or not short_signals:
        status = "asymmetric"
        dominant = "long" if long_signals > short_signals else "short"
    else:
        ratio = max(long_signals, short_signals) / min(long_signals, short_signals)
        status = "asymmetric" if ratio >= 3.0 else "balanced"
        dominant = "long" if long_signals > short_signals else "short" if short_signals > long_signals else None
    return {
        "version": "2026-08-30.entry-direction-symmetry.v2",
        "read_only": True,
        "is_entry_permission": False,
        "status": status,
        "dominant_side": dominant,
        "long": sides["long"],
        "short": sides["short"],
        "total_directional_signals": total_signals,
        "total_directional_observations": sum(
            int(item["observation_count"]) for item in sides.values()
        ),
        "policy": "diagnostic_only; never relaxes entry thresholds or risk gates",
    }


def build_entry_funnel_report(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Build the per-round canonical reason counts and directional report."""

    counts = {reason: 0 for reason in ENTRY_FUNNEL_REASONS}
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        reason = _text(row.get("funnel_reason"))
        executed = row.get("was_executed", row.get("executed")) is True
        if not executed and reason in counts:
            counts[reason] += 1
        analysis_only = row.get("analysis_only") is True
        normalized_rows.append(
            {
                "action": row.get("action"),
                "is_entry": (
                    not analysis_only
                    and (
                        row.get("is_entry") is True
                        or _text(row.get("action")).lower() in {"long", "short"}
                    )
                ),
                "analysis_only": analysis_only,
                "was_executed": executed,
                "funnel_reason": reason,
            }
        )
    return {
        "version": "2026-08-29.entry-funnel.v2",
        "read_only": True,
        "is_entry_permission": False,
        "reason_counts": counts,
        "direction_symmetry": build_direction_symmetry_report(normalized_rows),
    }
