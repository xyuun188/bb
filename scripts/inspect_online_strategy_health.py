from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

WINDOW_MINUTES = __WINDOW_MINUTES__
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
        return safe_float(side_ev.get("expected_net_return_pct"), safe_float(opp.get("expected_net_return_pct")))
    return safe_float(opp.get("expected_net_return_pct"))


def profit_quality(decision):
    raw = safe_dict(decision.raw_llm_response)
    opp = opportunity(decision)
    side_ev = selected_side_evidence(raw, selected_side(decision))
    if side_ev:
        return safe_float(side_ev.get("profit_quality_ratio"), safe_float(opp.get("profit_quality_ratio")))
    return safe_float(opp.get("profit_quality_ratio"))


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
    return {
        "tier": ev.get("tier") or opp.get("evidence_tier") or "",
        "effective_score": roundv(ev.get("effective_score", opp.get("score"))),
        "score": roundv(opp.get("score")),
        "min_score_required": roundv(opp.get("min_score_required")),
        "expected_net_return_pct": roundv(expected_net(decision)),
        "aggregate_expected_net_return_pct": roundv(opp.get("expected_net_return_pct")),
        "profit_quality_ratio": roundv(profit_quality(decision)),
        "loss_probability": roundv(opp.get("server_profit_loss_probability", opp.get("loss_probability"))),
        "tail_risk_score": roundv(opp.get("tail_risk_score")),
        "strong_positive_net_relief": safe_dict(safe_dict(opp.get("evidence_score")).get("strong_positive_net_relief")),
        "positive_net_probe_relief": safe_dict(safe_dict(opp.get("evidence_score")).get("positive_net_probe_relief")),
        "memory_missed_opportunity_relief": safe_dict(safe_dict(opp.get("evidence_score")).get("memory_missed_opportunity_relief")),
        "memory_habit_adjustment": safe_dict(opp.get("memory_habit_adjustment")),
        "vector_memory_adjustment": safe_dict(opp.get("vector_memory_adjustment")),
        "side_quality_adjustment": safe_dict(opp.get("side_quality_adjustment")),
        "expected_net_formula": breakdown.get("formula") or "",
        "shadow_memory_component": shadow_memory,
    }


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
    executed_entries = [d for d in entry_decisions if bool(d.was_executed)]
    order_by_decision = {}
    for order in orders:
        if order.decision_id and order.decision_id not in order_by_decision:
            order_by_decision[order.decision_id] = order

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

    for d in entry_decisions:
        st = state(d)
        sz = sizing(d)
        ev = evidence(d)
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
        if ev["positive_net_probe_relief"].get("shadow_only") or safe_dict(safe_dict(opportunity(d).get("evidence_score")).get("positive_net_probe_relief")).get("shadow_only"):
            shadow_only_examples.append(d)
        raw = safe_dict(d.raw_llm_response)
        cooldown = safe_dict(raw.get("loss_cooldown_override")) or safe_dict(safe_dict(raw.get("opportunity_score")).get("loss_cooldown_override"))
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
                "executed": bool(d.was_executed),
                "reason": short_text(reason, 280),
                "state": st,
                "evidence": ev,
                "sizing": sz,
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
    filled_orders = [o for o in orders if str(o.status or "").lower() == "filled"]
    failed_orders = [o for o in orders if str(o.status or "").lower() != "filled"]
    report = {
        "window_minutes": WINDOW_MINUTES,
        "generated_at": now.isoformat(),
        "counts": {
            "decisions": len(decisions),
            "hold_decisions": len(hold_decisions),
            "entry_decisions": len(entry_decisions),
            "executed_entries": len(executed_entries),
            "orders": len(orders),
            "filled_orders": len(filled_orders),
            "failed_orders": len(failed_orders),
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
        "shadow_only_positive_net_count": len(shadow_only_examples),
        "notional_floor_blocked_counts": dict(notional_floor_blocked.most_common(12)),
        "shadow_completed_best_action_counts": dict(shadow_by_best.most_common(10)),
        "missed_by_best_action": dict(missed_by_side.most_common(10)),
        "expert_memory_type_counts_recent": dict(memory_counts.most_common(20)),
        "strategy_event_type_counts": dict(event_counts.most_common(20)),
        "missed_samples": missed_samples,
        "entry_examples": examples,
        "loss_cooldown_examples": cooldown_examples[:30],
        "fast_loss_positions": fast_loss[:40],
        "current_open_positions": current_open,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

asyncio.run(main())
"""

LAUNCHER_SCRIPT = r"""
import os
import pwd
import subprocess
import argparse
import sys
from pathlib import Path

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
    ["/data/bb/app/.venv/bin/python", "/tmp/codex_strategy_sample.py"],
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect online strategy health.")
    parser.add_argument(
        "--minutes",
        type=int,
        default=480,
        help="Lookback window in minutes. Default: 480 (8 hours).",
    )
    args = parser.parse_args()
    minutes = max(int(args.minutes or 480), 1)
    remote_script = REMOTE_SCRIPT_TEMPLATE.replace("__WINDOW_MINUTES__", str(minutes))
    command = f"""
set -eo pipefail
cd /data/bb/app
cat > /tmp/codex_strategy_sample.py <<'PY'
{remote_script}
PY
cat > /tmp/codex_strategy_launcher.py <<'PY'
{LAUNCHER_SCRIPT}
PY
chmod 0644 /tmp/codex_strategy_sample.py /tmp/codex_strategy_launcher.py
python3 /tmp/codex_strategy_launcher.py
"""

    ssh = connect_remote_ssh(ROOT, timeout=25)
    try:
        out = run_remote_text(ssh, command, timeout=220, max_output_chars=100000)
        safe_print(out)
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
