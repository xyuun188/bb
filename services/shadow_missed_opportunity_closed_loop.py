from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from statistics import pstdev
from typing import Any

MIN_REPEATED_MISSES = 3
DEFAULT_REPORT_WINDOW_HOURS = 24
DEFAULT_REPORT_LIMIT = 200
MIN_STABLE_AVG_RETURN_PCT = 0.35
MIN_ADOPTED_COUNT = 5
MIN_ADOPTED_AVG_RETURN_PCT = 0.65
MIN_POSITIVE_RATIO = 0.75
MAX_STABILITY_STD_TO_AVG = 0.65
MAX_LOSS_PROBABILITY = 0.58
MAX_TAIL_RISK = 0.88
MAX_ABNORMAL_WICK_PCT = 6.0
MAX_VOLATILITY_20 = 0.085
PROBE_POSITION_SIZE_CAP_PCT = 0.015
ADOPTED_PROBE_POSITION_SIZE_CAP_PCT = 0.012


class ShadowMissedOpportunityClosedLoopService:
    def __init__(self, session_context_factory: Any | None = None) -> None:
        self._session_context_factory = session_context_factory

    async def report(
        self,
        *,
        hours: int = DEFAULT_REPORT_WINDOW_HOURS,
        limit: int = DEFAULT_REPORT_LIMIT,
    ) -> dict[str, Any]:
        from sqlalchemy import select

        from db.session import get_read_session_ctx
        from models.decision import AIDecision
        from models.learning import ShadowBacktest

        capped_hours = max(1, min(int(hours or DEFAULT_REPORT_WINDOW_HOURS), 168))
        capped_limit = max(50, min(int(limit or DEFAULT_REPORT_LIMIT), 1000))
        since = datetime.now(UTC) - timedelta(hours=capped_hours)
        session_factory = self._session_context_factory or get_read_session_ctx
        async with session_factory() as session:
            shadow_result = await session.execute(
                select(ShadowBacktest).order_by(ShadowBacktest.id.desc()).limit(capped_limit)
            )
            decision_result = await session.execute(
                select(AIDecision).order_by(AIDecision.id.desc()).limit(capped_limit)
            )
        report = summarize_shadow_missed_opportunities(
            [row for row in shadow_result.scalars().all() if _row_in_window(row, since)],
            decisions=[
                row for row in decision_result.scalars().all() if _row_in_window(row, since)
            ],
        )
        report["window_hours"] = capped_hours
        report["query_policy"] = {
            "online_safe": True,
            "ordered_by_primary_key": True,
            "db_status_filter": False,
            "db_time_filter": False,
            "row_limit": capped_limit,
        }
        return report


def summarize_shadow_missed_opportunities(
    shadows: Sequence[Any],
    *,
    decisions: Sequence[Any] | None = None,
) -> dict[str, Any]:
    completed = [row for row in shadows if str(_row_get(row, "status") or "") == "completed"]
    missed_rows = [row for row in completed if bool(_row_get(row, "missed_opportunity"))]
    groups: dict[tuple[str, str], list[Any]] = defaultdict(list)
    standalone_blocked: list[dict[str, Any]] = []

    for row in missed_rows:
        symbol = _symbol(row)
        side = _best_side(row)
        if not symbol or side not in {"long", "short"}:
            standalone_blocked.append(
                _blocked_group(
                    symbol=symbol,
                    side=side,
                    count=1,
                    returns=[],
                    reasons=["missing_symbol_or_side"],
                    examples=[_example(row, side)],
                )
            )
            continue
        groups[(symbol, side)].append(row)

    adopted: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = list(standalone_blocked)
    observing: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter(row["primary_block_reason"] for row in blocked)

    for (symbol, side), rows in sorted(groups.items()):
        evaluated = _evaluate_group(symbol, side, rows)
        status = str(evaluated.get("status") or "")
        if status == "adopted_learning":
            adopted.append(evaluated)
        elif status == "probe_ready":
            probes.append(evaluated)
        elif status == "observe_only":
            observing.append(evaluated)
        else:
            blocked.append(evaluated)
            for reason in _safe_list(evaluated.get("blocked_reasons")):
                reason_counts[str(reason)] += 1

    weak_evidence_executed = _weak_evidence_executed_count(decisions or [])
    summary = {
        "completed_count": len(completed),
        "missed_count": len(missed_rows),
        "group_count": len(groups),
        "adopted_count": len(adopted),
        "probe_count": len(probes),
        "blocked_count": len(blocked),
        "observe_only_count": len(observing),
        "weak_evidence_executed_count": weak_evidence_executed,
    }
    return {
        "audit_only": True,
        "live_entry_mutation": False,
        "can_bypass_risk_controls": False,
        "weak_evidence_execution_allowed": False,
        "global_missed_count_can_drive_entries": False,
        "usable_group_count": len(adopted) + len(probes),
        "summary": summary,
        "adopted": adopted[:20],
        "probe_candidates": probes[:20],
        "blocked": blocked[:20],
        "observe_only": observing[:20],
        "blocked_reason_counts": dict(reason_counts),
        "safety_rules": [
            "same_symbol_same_side_repeated_required",
            "stable_positive_return_required",
            "low_risk_evidence_required",
            "model_direction_alignment_required",
            "similar_market_structure_required",
            "missed_opportunity_never_forces_open",
            "net_return_loss_probability_tail_risk_okx_rules_still_apply",
        ],
        "probe_policy": {
            "max_position_size_pct": PROBE_POSITION_SIZE_CAP_PCT,
            "exit_rules": _exit_rules(),
            "hard_risk_controls_required": True,
            "expected_net_required": True,
            "okx_rules_required": True,
        },
    }


def _evaluate_group(symbol: str, side: str, rows: list[Any]) -> dict[str, Any]:
    returns = [_side_return(row, side) for row in rows]
    examples = [_example(row, side) for row in rows[:5]]
    blocked_reasons: list[str] = []
    adoption_reasons: list[str] = []

    if len(rows) < MIN_REPEATED_MISSES:
        blocked_reasons.append("insufficient_repeated_same_symbol_side")
    else:
        adoption_reasons.append("same_symbol_same_side_repeated")

    positive_returns = [value for value in returns if value >= MIN_STABLE_AVG_RETURN_PCT]
    avg_return = sum(returns) / len(returns) if returns else 0.0
    max_return = max(returns) if returns else 0.0
    median_return = sorted(returns)[len(returns) // 2] if returns else 0.0
    positive_ratio = len(positive_returns) / max(len(returns), 1)
    std_return = pstdev(returns) if len(returns) >= 2 else 0.0
    if _one_off_move(returns):
        blocked_reasons.append("one_off_move")
    elif avg_return < MIN_STABLE_AVG_RETURN_PCT or positive_ratio < MIN_POSITIVE_RATIO:
        blocked_reasons.append("unstable_or_weak_positive_return")
    elif std_return > max(abs(avg_return) * MAX_STABILITY_STD_TO_AVG, 0.18):
        blocked_reasons.append("return_not_stable")
    else:
        adoption_reasons.append("stable_positive_return_after_miss")

    risk = _risk_summary(rows)
    if risk["high_risk"]:
        blocked_reasons.append("high_risk_evidence")
    else:
        adoption_reasons.append("low_risk_evidence")

    alignment = _model_alignment(rows, side)
    if not alignment["aligned"]:
        blocked_reasons.append(alignment["reason"])
    else:
        adoption_reasons.append("model_direction_aligned")

    structure = _structure_summary(rows)
    if not structure["similar"]:
        blocked_reasons.append("market_structure_not_similar")
    else:
        adoption_reasons.append("similar_market_structure")

    base = {
        "symbol": symbol,
        "side": side,
        "missed_count": len(rows),
        "avg_return_pct": round(avg_return, 6),
        "median_return_pct": round(median_return, 6),
        "max_return_pct": round(max_return, 6),
        "return_std_pct": round(std_return, 6),
        "positive_ratio": round(positive_ratio, 6),
        "risk": risk,
        "model_alignment": alignment,
        "market_structure": structure,
        "examples": examples,
        "can_force_open": False,
        "can_bypass_risk_controls": False,
    }
    if blocked_reasons:
        return _blocked_group(
            symbol=symbol,
            side=side,
            count=len(rows),
            returns=returns,
            reasons=list(dict.fromkeys(blocked_reasons)),
            examples=examples,
            extra=base,
        )

    strong = len(rows) >= MIN_ADOPTED_COUNT and avg_return >= MIN_ADOPTED_AVG_RETURN_PCT
    probe_rules = {
        "max_position_size_pct": (
            ADOPTED_PROBE_POSITION_SIZE_CAP_PCT if strong else PROBE_POSITION_SIZE_CAP_PCT
        ),
        "exit_rules": _exit_rules(),
        "requires_current_positive_expected_net": True,
        "requires_loss_probability_gate": True,
        "requires_tail_risk_gate": True,
        "requires_okx_order_rules": True,
    }
    if strong:
        return {
            **base,
            "status": "adopted_learning",
            "adoption_reasons": adoption_reasons,
            "allow_controlled_probe": True,
            "can_enter_training_positive": True,
            "training_label_policy": "positive_only_through_clean_training_view",
            "probe_rules": probe_rules,
        }
    return {
        **base,
        "status": "probe_ready",
        "adoption_reasons": adoption_reasons,
        "allow_controlled_probe": True,
        "can_enter_training_positive": False,
        "training_label_policy": "observe_until_more_confirmed_samples",
        "probe_rules": probe_rules,
    }


def _blocked_group(
    *,
    symbol: str,
    side: str,
    count: int,
    returns: list[float],
    reasons: list[str],
    examples: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    primary = reasons[0] if reasons else "blocked"
    base = dict(extra or {})
    base.update(
        {
            "symbol": symbol,
            "side": side,
            "status": "blocked",
            "missed_count": count,
            "blocked_reasons": reasons,
            "primary_block_reason": primary,
            "avg_return_pct": round(sum(returns) / len(returns), 6) if returns else 0.0,
            "examples": examples,
            "allow_controlled_probe": False,
            "can_enter_training_positive": False,
            "can_force_open": False,
            "can_bypass_risk_controls": False,
        }
    )
    return base


def _one_off_move(returns: list[float]) -> bool:
    if len(returns) < MIN_REPEATED_MISSES:
        return False
    ordered = sorted(returns)
    median = ordered[len(ordered) // 2]
    maximum = ordered[-1]
    return bool(maximum >= 0.80 and median < MIN_STABLE_AVG_RETURN_PCT)


def _risk_summary(rows: list[Any]) -> dict[str, Any]:
    loss_values: list[float] = []
    tail_values: list[float] = []
    wick_values: list[float] = []
    volatility_values: list[float] = []
    for row in rows:
        snapshot = _safe_dict(_row_get(row, "feature_snapshot"))
        raw = _safe_dict(_row_get(row, "raw_llm_response"))
        opportunity = _safe_dict(raw.get("opportunity_score"))
        evidence = _safe_dict(opportunity.get("evidence_score"))
        loss_values.append(
            _first_float(
                snapshot,
                opportunity,
                evidence,
                keys=("loss_probability", "server_profit_loss_probability"),
                default=0.50,
            )
        )
        tail_values.append(
            _first_float(snapshot, opportunity, evidence, keys=("tail_risk_score",), default=0.70)
        )
        wick_values.append(_safe_float(snapshot.get("abnormal_wick_max_pct"), 0.0))
        volatility_values.append(_safe_float(snapshot.get("volatility_20"), 0.0))
    avg_loss = sum(loss_values) / len(loss_values) if loss_values else 1.0
    max_loss = max(loss_values) if loss_values else 1.0
    avg_tail = sum(tail_values) / len(tail_values) if tail_values else 1.0
    max_tail = max(tail_values) if tail_values else 1.0
    max_wick = max(wick_values) if wick_values else 0.0
    max_volatility = max(volatility_values) if volatility_values else 0.0
    high_risk = bool(
        avg_loss > MAX_LOSS_PROBABILITY
        or max_loss > 0.68
        or avg_tail > MAX_TAIL_RISK
        or max_tail > 1.0
        or max_wick >= MAX_ABNORMAL_WICK_PCT
        or max_volatility > MAX_VOLATILITY_20
    )
    return {
        "high_risk": high_risk,
        "avg_loss_probability": round(avg_loss, 6),
        "max_loss_probability": round(max_loss, 6),
        "avg_tail_risk_score": round(avg_tail, 6),
        "max_tail_risk_score": round(max_tail, 6),
        "max_abnormal_wick_pct": round(max_wick, 6),
        "max_volatility_20": round(max_volatility, 6),
    }


def _model_alignment(rows: list[Any], side: str) -> dict[str, Any]:
    observed: list[str] = []
    for row in rows:
        raw = _safe_dict(_row_get(row, "raw_llm_response"))
        observed.extend(_extract_model_sides(raw))
    aligned = sum(1 for item in observed if item == side)
    conflicted = sum(1 for item in observed if item in {"long", "short"} and item != side)
    if not observed:
        return {
            "aligned": False,
            "reason": "model_direction_missing",
            "aligned_count": 0,
            "conflicted_count": 0,
            "observed_sides": [],
        }
    total = aligned + conflicted
    is_aligned = aligned >= max(conflicted + 1, math.ceil(max(total, 1) * 0.67))
    return {
        "aligned": is_aligned,
        "reason": "aligned" if is_aligned else "model_direction_not_aligned",
        "aligned_count": aligned,
        "conflicted_count": conflicted,
        "observed_sides": observed[:10],
    }


def _extract_model_sides(raw: dict[str, Any]) -> list[str]:
    sides: list[str] = []
    candidates = [
        _safe_dict(raw.get("ml_signal")),
        _safe_dict(raw.get("local_ai_tools")),
        _safe_dict(raw.get("direction_competition")),
        _safe_dict(raw.get("entry_candidate_evidence")),
    ]
    for item in candidates:
        for key in ("best_side", "side", "predicted_side", "preferred_side"):
            side = _side(item.get(key))
            if side:
                sides.append(side)
        prediction = _safe_dict(item.get("time_series_prediction"))
        side = _side(prediction.get("best_side"))
        if side:
            sides.append(side)
        for row in _safe_list(item.get("predictions")):
            side = _side(_safe_dict(row).get("best_side"))
            if side:
                sides.append(side)
    return sides


def _structure_summary(rows: list[Any]) -> dict[str, Any]:
    buckets = Counter(_market_bucket(_safe_dict(_row_get(row, "feature_snapshot"))) for row in rows)
    buckets.pop("unknown", None)
    if not buckets:
        return {"similar": False, "reason": "market_structure_missing", "buckets": {}}
    top, count = buckets.most_common(1)[0]
    ratio = count / max(len(rows), 1)
    return {
        "similar": ratio >= 0.67,
        "reason": "similar" if ratio >= 0.67 else "market_structure_diverged",
        "dominant_bucket": top,
        "dominant_ratio": round(ratio, 6),
        "buckets": dict(buckets),
    }


def _market_bucket(snapshot: dict[str, Any]) -> str:
    explicit = str(snapshot.get("market_structure") or snapshot.get("structure_bucket") or "")
    if explicit.strip():
        return explicit.strip().lower()[:80]
    price_vs_sma = _safe_float(snapshot.get("price_vs_sma20"), 0.0)
    returns_20 = _safe_float(snapshot.get("returns_20"), 0.0)
    volume = _safe_float(snapshot.get("volume_ratio"), 1.0)
    adx = _safe_float(snapshot.get("adx"), 0.0)
    trend = (
        "up"
        if price_vs_sma > 0.01 and returns_20 > 0
        else "down" if price_vs_sma < -0.01 and returns_20 < 0 else "flat"
    )
    volume_bucket = "high_volume" if volume >= 1.4 else "normal_volume"
    trend_strength = "strong" if adx >= 25 else "weak"
    return f"{trend}_{trend_strength}_{volume_bucket}"


def _example(row: Any, side: str) -> dict[str, Any]:
    return {
        "id": _row_get(row, "id"),
        "symbol": _symbol(row),
        "side": side,
        "best_action": _best_side(row),
        "return_pct": round(_side_return(row, side), 6),
        "horizon_minutes": _row_get(row, "horizon_minutes"),
        "created_at": _iso(_row_get(row, "created_at")),
    }


def _weak_evidence_executed_count(decisions: Sequence[Any]) -> int:
    count = 0
    for row in decisions:
        action = _side(_row_get(row, "action"))
        if action not in {"long", "short"} or not bool(_row_get(row, "was_executed")):
            continue
        raw = _safe_dict(_row_get(row, "raw_llm_response"))
        tier = str(
            _safe_dict(_safe_dict(raw.get("opportunity_score")).get("evidence_score")).get("tier")
            or ""
        )
        if tier in {"weak_conflict_probe", "degraded_missing_probe"}:
            count += 1
    return count


def _best_side(row: Any) -> str:
    side = _side(_row_get(row, "best_action"))
    if side:
        return side
    long_return = _safe_float(_row_get(row, "long_return_pct"), 0.0)
    short_return = _safe_float(_row_get(row, "short_return_pct"), 0.0)
    if long_return > 0 and long_return >= short_return:
        return "long"
    if short_return > 0 and short_return > long_return:
        return "short"
    return ""


def _side_return(row: Any, side: str) -> float:
    if side == "long":
        return _safe_float(_row_get(row, "long_return_pct"), 0.0)
    if side == "short":
        return _safe_float(_row_get(row, "short_return_pct"), 0.0)
    return 0.0


def _symbol(row: Any) -> str:
    return str(_row_get(row, "symbol") or "").strip().upper()


def _side(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"long", "buy", "bullish"}:
        return "long"
    if normalized in {"short", "sell", "bearish"}:
        return "short"
    return ""


def _first_float(*sources: dict[str, Any], keys: tuple[str, ...], default: float = 0.0) -> float:
    for source in sources:
        for key in keys:
            if key in source and source.get(key) is not None:
                return _safe_float(source.get(key), default)
    return default


def _exit_rules() -> list[str]:
    return [
        "stop_loss_required_before_any_probe",
        "exit_on_expected_net_turns_negative",
        "exit_on_loss_probability_or_tail_risk_gate_failure",
        "no_reopen_after_loss_without_new_strong_evidence",
    ]


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _row_in_window(row: Any, since: datetime) -> bool:
    created_at = _row_get(row, "created_at")
    if not isinstance(created_at, datetime):
        return True
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    return created_at.astimezone(UTC) >= since


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _iso(value: Any) -> str | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()
