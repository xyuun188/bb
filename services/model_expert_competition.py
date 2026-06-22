from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from db.session import get_read_session_ctx
from models.decision import AIDecision
from models.learning import ShadowBacktest
from services.model_expert_health import (
    DEFAULT_WINDOWS_HOURS,
    _action,
    _created_at,
    _expert_rows,
    _extract_component_rows,
    _safe_float,
)

MIN_BASELINE_SAMPLES = 3
MIN_COMPETITOR_SAMPLES = 2


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _decision_action(row: Any) -> str:
    return _action(getattr(row, "action", None))


def _executed_outcome_rows(decisions: list[Any], now: datetime, window_hours: int) -> list[Any]:
    since = now - timedelta(hours=window_hours)
    rows: list[Any] = []
    for decision in decisions:
        created = _created_at(decision)
        if created is None or created < since:
            continue
        action = _decision_action(decision)
        if action not in {"long", "short"}:
            continue
        if not bool(getattr(decision, "was_executed", False)):
            continue
        if getattr(decision, "outcome_pnl_pct", None) is None:
            continue
        rows.append(decision)
    return rows


def _metric_block(pnls: list[float]) -> dict[str, Any]:
    sample_count = len(pnls)
    profit = sum(value for value in pnls if value > 0)
    loss = abs(sum(value for value in pnls if value < 0))
    wins = sum(1 for value in pnls if value > 0)
    avg_profit = profit / wins if wins else 0.0
    loss_count = sample_count - wins
    avg_loss = loss / loss_count if loss_count else 0.0
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in pnls:
        equity += value
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    return {
        "sample_count": sample_count,
        "net_pnl_pct": round(sum(pnls), 6),
        "win_rate": round(wins / sample_count, 4) if sample_count else 0.0,
        "profit_factor": round(profit / loss, 6) if loss > 0 else (999.0 if profit > 0 else 0.0),
        "avg_profit_pct": round(avg_profit, 6),
        "avg_loss_pct": round(avg_loss, 6),
        "profit_loss_ratio": round(avg_profit / avg_loss, 6) if avg_loss > 0 else 0.0,
        "max_drawdown_pct": round(max_drawdown, 6),
        "fast_loss_rate": 0.0,
        "small_position_rate": 0.0,
    }


def _baseline_metrics(decisions: list[Any], now: datetime, window_hours: int) -> dict[str, Any]:
    rows = _executed_outcome_rows(decisions, now, window_hours)
    block = _metric_block([_safe_float(getattr(row, "outcome_pnl_pct", None), 0.0) for row in rows])
    if rows:
        small_positions = [
            row for row in rows if _safe_float(getattr(row, "position_size_pct", 0.0)) <= 0.02
        ]
        block["small_position_rate"] = round(len(small_positions) / len(rows), 4)
    return block


def _component_metrics(
    decisions: list[Any], now: datetime, window_hours: int
) -> dict[str, dict[str, Any]]:
    rows = _executed_outcome_rows(decisions, now, window_hours)
    samples: dict[str, list[float]] = {}
    duration: dict[str, list[float]] = {}
    json_errors: dict[str, int] = {}
    no_returns: dict[str, int] = {}
    for decision in rows:
        action = _decision_action(decision)
        pnl = _safe_float(getattr(decision, "outcome_pnl_pct", None), 0.0)
        expert_actions = {
            str(
                item.get("expert_name") or item.get("model_name") or item.get("name") or ""
            ): _action(item.get("action"))
            for item in _expert_rows(decision)
            if isinstance(item, dict)
        }
        for component in _extract_component_rows(decision):
            name = str(component.get("name") or "").strip()
            rec_action = _action(component.get("action"))
            if rec_action == "unknown":
                rec_action = expert_actions.get(name, "unknown")
            if not name:
                continue
            samples.setdefault(name, []).append(pnl if rec_action == action else 0.0)
            duration.setdefault(name, []).append(_safe_float(component.get("duration_sec"), 0.0))
            if bool(component.get("json_error")):
                json_errors[name] = json_errors.get(name, 0) + 1
            if bool(component.get("no_return")):
                no_returns[name] = no_returns.get(name, 0) + 1
    metrics: dict[str, dict[str, Any]] = {}
    for name, pnls in samples.items():
        block = _metric_block(pnls)
        count = max(int(block["sample_count"]), 1)
        block["avg_duration_sec"] = round(sum(duration.get(name, [])) / count, 6)
        block["json_error_rate"] = round(json_errors.get(name, 0) / count, 4)
        block["no_return_rate"] = round(no_returns.get(name, 0) / count, 4)
        metrics[name] = block
    return metrics


def _weight_action(metrics: dict[str, Any], baseline: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if int(metrics.get("sample_count") or 0) < MIN_COMPETITOR_SAMPLES:
        reasons.append("insufficient_competitor_samples")
        return "observe_only", reasons
    delta = _safe_float(metrics.get("net_pnl_pct"), 0.0) - _safe_float(
        baseline.get("net_pnl_pct"), 0.0
    )
    profit_factor = _safe_float(metrics.get("profit_factor"), 0.0)
    if (
        _safe_float(metrics.get("json_error_rate"), 0.0) >= 0.25
        or _safe_float(metrics.get("no_return_rate"), 0.0) >= 0.25
    ):
        reasons.append("unstable_runtime")
        return "pause_shadow", reasons
    if delta > 0 and profit_factor >= 1.15:
        reasons.append("beats_baseline")
        return "increase_shadow_weight", reasons
    if delta < 0 or profit_factor < 0.85:
        reasons.append("lags_baseline")
        return "reduce_shadow_weight", reasons
    reasons.append("near_baseline")
    return "keep_shadow_weight", reasons


def summarize_model_expert_competition(
    decisions: list[Any],
    shadows: list[Any] | None = None,
    *,
    now: datetime | None = None,
    window_hours: int = 72,
) -> dict[str, Any]:
    current = now or datetime.now(UTC)
    baseline = _baseline_metrics(decisions, current, window_hours)
    baseline_available = int(baseline.get("sample_count") or 0) >= MIN_BASELINE_SAMPLES
    competitors: dict[str, Any] = {}
    if baseline_available:
        for name, metrics in sorted(_component_metrics(decisions, current, window_hours).items()):
            action, reasons = _weight_action(metrics, baseline)
            competitors[name] = {
                "name": name,
                "metrics": metrics,
                "baseline_delta": {
                    "net_pnl_pct": round(
                        _safe_float(metrics.get("net_pnl_pct"), 0.0)
                        - _safe_float(baseline.get("net_pnl_pct"), 0.0),
                        6,
                    ),
                    "profit_factor": round(
                        _safe_float(metrics.get("profit_factor"), 0.0)
                        - _safe_float(baseline.get("profit_factor"), 0.0),
                        6,
                    ),
                },
                "recommended_weight_action": action,
                "recommendation_reasons": reasons,
                "can_apply_live_weight": False,
            }

    shadow_rows = [row for row in shadows or [] if str(getattr(row, "status", "")) == "completed"]
    return {
        "audit_only": True,
        "live_weight_mutation": False,
        "can_apply_live_weight": False,
        "window_hours": window_hours,
        "windows_hours": list(DEFAULT_WINDOWS_HOURS),
        "generated_at": current.isoformat(),
        "baseline": baseline,
        "competitors": competitors,
        "layers": {
            "offline_replay": {
                "available": baseline_available,
                "baseline_available": baseline_available,
                "sample_count": baseline.get("sample_count", 0),
            },
            "shadow_competition": {
                "available": bool(shadow_rows),
                "sample_count": len(shadow_rows),
            },
            "sim_ab": {
                "available": False,
                "sample_count": 0,
                "reason": "simulated execution A/B ledger is not available yet",
            },
        },
        "blocking_reasons": [] if baseline_available else ["baseline_missing"],
        "safety_rules": [
            "no_direct_live_weight_change",
            "new_models_shadow_before_live",
            "baseline_required_before_weight_change",
            "separate_shadow_sim_live_statistics",
        ],
    }


class ModelExpertCompetitionService:
    def __init__(self, session_context_factory: Any = get_read_session_ctx) -> None:
        self._session_context_factory = session_context_factory

    async def report(self, *, hours: int = 72, limit: int = 1200) -> dict[str, Any]:
        from sqlalchemy import select

        capped_hours = max(1, min(int(hours or 72), 168))
        capped_limit = max(50, min(int(limit or 1200), 5000))
        since = datetime.now(UTC) - timedelta(hours=capped_hours)
        async with self._session_context_factory() as session:
            decisions_result = await session.execute(
                select(AIDecision)
                .where(AIDecision.created_at >= since)
                .order_by(AIDecision.created_at.desc())
                .limit(capped_limit)
            )
            shadows_result = await session.execute(
                select(ShadowBacktest)
                .where(ShadowBacktest.created_at >= since)
                .order_by(ShadowBacktest.created_at.desc())
                .limit(capped_limit)
            )
        return summarize_model_expert_competition(
            list(decisions_result.scalars().all()),
            list(shadows_result.scalars().all()),
            window_hours=capped_hours,
        )
