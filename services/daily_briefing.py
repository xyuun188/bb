"""Build a short Chinese operating briefing for the BB paper-trading system.

The briefing answers the questions an operator actually asks every day:

1. How many decisions were made, and how many opened a position?
2. When nothing opened, which gate rejected the entry most often?
3. How much compliant shadow evidence finished in the window?
4. What does the current model artifact say about permission and samples?

It is strictly read-only: no permissions, sizing, thresholds or model
pointers are touched, and only projected columns are queried.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any

DEFAULT_WINDOW_HOURS = 24
DEFAULT_DECISION_LIMIT = 1500


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _blocking_reasons(snapshot: Any) -> list[str]:
    """Pull the pre-computed gate reasons out of the learning snapshot."""

    payload = _dict(snapshot)
    permission = _dict(payload.get("entry_permission"))
    reasons = [
        str(item).strip()
        for item in permission.get("blocking_reasons") or []
        if str(item).strip()
    ]
    selection = _dict(payload.get("paper_trade_selection"))
    selection_reason = str(
        selection.get("diagnostic_status") or selection.get("selection_reason") or ""
    ).strip()
    if selection.get("selected") is not True and selection_reason:
        reasons.append(f"paper_trade_selection:{selection_reason}")
    return reasons


def _model_status() -> dict[str, Any]:
    try:
        from services.ml_signal_service import MLSignalService

        metadata = MLSignalService().status_metadata()
        latest = _dict(metadata.get("latest_training"))
        cursor = _dict(latest.get("sample_cursor"))
        return {
            "lifecycle": metadata.get("artifact_lifecycle"),
            "readiness": metadata.get("readiness_state"),
            "trained_at": metadata.get("trained_at"),
            "artifact_version": metadata.get("artifact_version"),
            "latest_training_version": latest.get("artifact_version"),
            "champion_retained": latest.get("champion_retained"),
            "shadow_samples": cursor.get("shadow"),
            "trade_samples": cursor.get("trade"),
            "paper_trading_permission": metadata.get("paper_trading_permission"),
            "paper_model_eligible": metadata.get("paper_model_eligible"),
            "live_ml_ready": metadata.get("live_ml_ready"),
            "message": metadata.get("message"),
        }
    except Exception as exc:  # pragma: no cover - defensive read path
        return {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}


async def build_briefing(
    *,
    window_hours: int = DEFAULT_WINDOW_HOURS,
    decision_limit: int = DEFAULT_DECISION_LIMIT,
) -> dict[str, Any]:
    # Imported lazily so the render helpers stay importable without a
    # database URL (tests, and any offline operator tooling).
    from sqlalchemy import case, func, or_, select

    from db.session import get_read_session_ctx
    from models.decision import AIDecision
    from models.learning import ShadowBacktest

    now = datetime.now(UTC)
    since = now - timedelta(hours=window_hours)
    report: dict[str, Any] = {
        "generated_at": now.isoformat(),
        "window_hours": window_hours,
    }
    window_filters = (
        AIDecision.is_paper.is_(True),
        AIDecision.created_at >= since,
        AIDecision.created_at <= now,
    )
    executed_condition = or_(
        AIDecision.was_executed.is_(True), AIDecision.executed_at.is_not(None)
    )
    blocked_entry_condition = (
        (AIDecision.analysis_type == "market")
        & AIDecision.was_executed.is_not(True)
        & AIDecision.executed_at.is_(None)
    )

    async with get_read_session_ctx(statement_timeout_ms=60000) as session:
        # Count the complete window before limiting the larger reason payloads.
        totals = (
            await session.execute(
                select(
                    AIDecision.action,
                    func.count().label("count"),
                    func.sum(case((executed_condition, 1), else_=0)).label("executed"),
                    func.sum(case((blocked_entry_condition, 1), else_=0)).label("blocked"),
                )
                .where(*window_filters)
                .group_by(AIDecision.action)
            )
        ).all()
        rows = (
            await session.execute(
                select(
                    AIDecision.id,
                    AIDecision.created_at,
                    AIDecision.action,
                    AIDecision.was_executed,
                    AIDecision.executed_at,
                    AIDecision.execution_reason,
                    AIDecision.decision_learning_snapshot.label("learning"),
                )
                .where(*window_filters, blocked_entry_condition)
                .order_by(AIDecision.id.desc())
                .limit(decision_limit)
            )
        ).all()
        shadow_total = (
            await session.execute(
                select(func.count(ShadowBacktest.id)).where(
                    ShadowBacktest.status == "completed",
                    ShadowBacktest.execution_mode == "paper",
                    ShadowBacktest.created_at >= since,
                    ShadowBacktest.created_at <= now,
                )
            )
        ).scalar()

    actions: Counter[str] = Counter()
    executed = 0
    blocked = 0
    for row in totals:
        action = str(row.action or "").lower()
        actions[action] += int(row.count or 0)
        if action in {"long", "short"}:
            executed += int(row.executed or 0)
        blocked += int(row.blocked or 0)
    reasons: Counter[str] = Counter()
    for row in rows:
        direct = str(row.execution_reason or "").strip()
        if direct:
            reasons[direct] += 1
        for reason in _blocking_reasons(row.learning):
            reasons[reason] += 1

    report["decisions"] = {
        "scanned": len(rows),
        "in_window": sum(actions.values()),
        "executed": executed,
        "actions": dict(actions),
        "reason_sample_count": len(rows),
        "reason_total_count": blocked,
        "reasons_sampled": len(rows) < blocked,
        "top_reasons": reasons.most_common(8),
    }
    report["shadow"] = {"completed_in_window": int(shadow_total or 0)}
    report["model"] = _model_status()
    return report


def render_text(report: dict[str, Any]) -> str:
    decisions = report.get("decisions") or {}
    shadow = report.get("shadow") or {}
    model = report.get("model") or {}
    hours = report.get("window_hours")
    lines = [f"=== BB 每日简报（最近 {hours} 小时）===", ""]

    lines.append("【交易】")
    lines.append(
        f"决策 {decisions.get('in_window', 0)} 次，实际开仓 "
        f"{decisions.get('executed', 0)} 次"
    )
    actions = decisions.get("actions") or {}
    if actions:
        lines.append("动作分布：" + "，".join(f"{k or '未知'} {v}" for k, v in actions.items()))
    lines.append("")

    lines.append("【不开仓原因】")
    if decisions.get("reasons_sampled"):
        lines.append(
            f"以下统计最近 {decisions.get('reason_sample_count', 0)} 条未开仓决策"
            f"，窗口内共 {decisions.get('reason_total_count', 0)} 条"
        )
    reasons = decisions.get("top_reasons") or []
    if not reasons:
        lines.append("无阻塞记录")
    for reason, count in reasons:
        lines.append(f"{count:>6} 次  {reason}")
    lines.append("")

    lines.append("【模型】")
    if model.get("error"):
        lines.append(f"读取失败：{model['error']}")
    else:
        lines.append(f"生命周期：{model.get('lifecycle') or '未知'}")
        lines.append(f"运行版本：{model.get('artifact_version') or '未知'}")
        latest_version = model.get("latest_training_version")
        if latest_version and latest_version != model.get("artifact_version"):
            retained = "（未替换运行模型）" if model.get("champion_retained") else ""
            lines.append(f"最近候选：{latest_version}{retained}")
        lines.append(f"模型纸盘调用资格：{model.get('paper_model_eligible')}")
        lines.append(f"实盘权限：{model.get('live_ml_ready')}")
        lines.append(f"最近训练影子样本：{model.get('shadow_samples')}")
        lines.append(f"最近训练成交样本：{model.get('trade_samples')}")
        if model.get("message"):
            lines.append(f"说明：{model['message']}")
    lines.append("")

    lines.append("【影子回测】")
    lines.append(f"窗口内完成 {shadow.get('completed_in_window', 0)} 条")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print the BB daily briefing.")
    parser.add_argument("--hours", type=int, default=DEFAULT_WINDOW_HOURS)
    parser.add_argument("--limit", type=int, default=DEFAULT_DECISION_LIMIT)
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead.")
    args = parser.parse_args(argv)

    report = asyncio.run(
        build_briefing(
            window_hours=max(1, min(args.hours, 720)),
            decision_limit=max(1, args.limit),
        )
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, default=str))
    else:
        print(render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
