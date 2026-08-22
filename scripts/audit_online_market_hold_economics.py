from __future__ import annotations

import argparse
import base64
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402

REMOTE_SCRIPT = r'''
import asyncio
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select

from db.session import get_read_session_ctx
from models.decision import AIDecision
from services.entry_funnel_diagnostics import classify_entry_funnel_reason

LIMIT = __LIMIT__
SINCE_MINUTES = __SINCE_MINUTES__


def _dict(value):
    return value if isinstance(value, dict) else {}


def _list(value):
    return value if isinstance(value, list) else []


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_summary(values):
    finite = sorted(value for value in values if value is not None)
    if not finite:
        return {"count": 0, "minimum": None, "p50": None, "p95": None, "maximum": None}

    def percentile(ratio):
        index = min(max(int(round((len(finite) - 1) * ratio)), 0), len(finite) - 1)
        return round(finite[index], 6)

    return {
        "count": len(finite),
        "minimum": round(finite[0], 6),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "maximum": round(finite[-1], 6),
    }


def _normalize(value):
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, defaultdict):
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


async def main():
    since = (
        datetime.now(UTC) - timedelta(minutes=SINCE_MINUTES)
        if SINCE_MINUTES > 0
        else None
    )
    async with get_read_session_ctx() as session:
        filters = [
            AIDecision.is_paper.is_(True),
            AIDecision.analysis_type == "market",
        ]
        if since is not None:
            filters.append(AIDecision.created_at >= since)
        rows = list(
            (
                await session.execute(
                    select(AIDecision)
                    .where(*filters)
                    .order_by(AIDecision.id.desc())
                    .limit(LIMIT)
                )
            )
            .scalars()
            .all()
        )
        consultation_rows = list(
            (
                await session.execute(
                    select(
                        AIDecision.id,
                        AIDecision.analysis_type,
                        AIDecision.symbol,
                        AIDecision.created_at,
                        AIDecision.raw_llm_response,
                    )
                    .where(
                        AIDecision.is_paper.is_(True),
                        *([AIDecision.created_at >= since] if since is not None else []),
                    )
                    .order_by(AIDecision.id.desc())
                    .limit(LIMIT)
                )
            )
            .all()
        )
        duplicate_rows = (
            await session.execute(
                select(
                    AIDecision.analysis_idempotency_key,
                    func.count(AIDecision.id),
                )
                .where(AIDecision.analysis_idempotency_key.is_not(None))
                .group_by(AIDecision.analysis_idempotency_key)
                .having(func.count(AIDecision.id) > 1)
            )
        ).all()

    summary = {
        "row_count": len(rows),
        "id_range": [rows[-1].id if rows else None, rows[0].id if rows else None],
        "time_range": [
            str(rows[-1].created_at) if rows else None,
            str(rows[0].created_at) if rows else None,
        ],
        "final_actions": Counter(),
        "quality": Counter(),
        "quality_reason_codes": Counter(),
        "latency_values": {
            "stage_duration_sec": [],
            "model_duration_sec": [],
        },
        "expert_latency_values": defaultdict(list),
        "expert_status": Counter(),
        "consultation_latency_values": [],
        "expert_actions": Counter(),
        "expert_actions_by_role": defaultdict(Counter),
        "expert_count_distribution": Counter(),
        "all_experts_hold": 0,
        "evidence_present": 0,
        "preferred_side": Counter(),
        "side": {
            "long": Counter(),
            "short": Counter(),
        },
        "side_metric_values": {
            "long": {"expected_net_return_pct": [], "return_lcb_pct": []},
            "short": {"expected_net_return_pct": [], "return_lcb_pct": []},
        },
        "authoritative_candidate": Counter(),
        "cross_validation_count_distribution": Counter(),
        "consultation": Counter(),
        "consultation_by_analysis_type": defaultdict(Counter),
        "consultation_examples": [],
        "funnel_reasons": Counter(),
        "incomplete_examples": [],
        "service_error_examples": [],
        "positive_all_hold_examples": [],
        "positive_not_selected_examples": [],
        "positive_production_examples": [],
        "symbol_counts": Counter(),
        "duplicate_idempotency_key_count": len(duplicate_rows),
    }

    for row in rows:
        raw = _dict(row.raw_llm_response)
        quality = _dict(raw.get("analysis_quality_contract"))
        latency = _dict(raw.get("latency_summary"))
        evidence = _dict(raw.get("entry_candidate_evidence"))
        authoritative = _dict(raw.get("authoritative_return_candidate"))
        opinions = _list(raw.get("opinions")) or _list(row.model_health_opinions)
        action = str(row.action or "unknown").lower()

        summary["final_actions"][action] += 1
        summary["symbol_counts"][row.symbol] += 1
        summary["quality"][
            "analysis_complete"
            if quality.get("analysis_complete") is True
            else "analysis_incomplete"
        ] += 1
        summary["quality_reason_codes"][
            str(quality.get("reason_code") or "missing")
        ] += 1
        for metric in ("stage_duration_sec", "model_duration_sec"):
            value = _float(latency.get(metric))
            if value is not None:
                summary["latency_values"][metric].append(value)
        model_timings = _list(raw.get("model_timings")) or _list(row.model_health_timings)
        for timing in model_timings:
            if not isinstance(timing, dict):
                continue
            name = str(timing.get("name") or timing.get("model") or "unknown")
            duration = _float(timing.get("duration_sec"))
            if duration is not None:
                summary["expert_latency_values"][name].append(duration)
            summary["expert_status"][f"{name}:{str(timing.get('status') or 'unknown')}"] += 1
        cross_validations = _list(raw.get("cross_validations"))
        summary["cross_validation_count_distribution"][
            str(len(cross_validations))
        ] += 1
        summary["quality"][
            "decision_eligible"
            if quality.get("decision_eligible") is True
            else "decision_ineligible"
        ] += 1

        expert_actions = []
        for opinion in opinions:
            if not isinstance(opinion, dict):
                continue
            role = str(
                opinion.get("model_name")
                or opinion.get("expert_name")
                or "unknown"
            )
            expert_action = str(opinion.get("action") or "unknown").lower()
            expert_actions.append(expert_action)
            summary["expert_actions"][expert_action] += 1
            summary["expert_actions_by_role"][role][expert_action] += 1

        summary["expert_count_distribution"][str(len(expert_actions))] += 1
        all_hold = bool(expert_actions) and all(
            expert_action == "hold" for expert_action in expert_actions
        )
        if all_hold:
            summary["all_experts_hold"] += 1

        if evidence:
            summary["evidence_present"] += 1
        preferred = str(
            evidence.get("preferred_side_by_evidence") or "missing"
        ).lower()
        summary["preferred_side"][preferred] += 1

        positive_sides = []
        production_sides = []
        for side in ("long", "short"):
            side_evidence = _dict(evidence.get(side))
            side_summary = summary["side"][side]
            if not side_evidence:
                side_summary["missing"] += 1
                continue
            side_summary["present"] += 1
            for flag in (
                "decision_eligible",
                "paper_eligible",
                "production_eligible",
                "positive_fee_after_return_edge",
                "return_distribution_ready",
            ):
                side_summary[
                    flag if side_evidence.get(flag) is True else f"not_{flag}"
                ] += 1

            expected_return = _float(side_evidence.get("expected_net_return_pct"))
            return_lcb = _float(side_evidence.get("return_lcb_pct"))
            if expected_return is not None:
                summary["side_metric_values"][side]["expected_net_return_pct"].append(
                    expected_return
                )
            if return_lcb is not None:
                summary["side_metric_values"][side]["return_lcb_pct"].append(return_lcb)
            positive = bool(
                expected_return is not None
                and return_lcb is not None
                and expected_return > 0.0
                and return_lcb > 0.0
            )
            if not positive:
                side_summary["non_positive_or_missing"] += 1
                continue

            positive_sides.append(side)
            side_summary["positive_expected_and_lcb"] += 1
            if side_evidence.get("decision_eligible") is True:
                side_summary["positive_decision_eligible"] += 1
            if side_evidence.get("paper_eligible") is True:
                side_summary["positive_paper_eligible"] += 1
            if side_evidence.get("production_eligible") is True:
                side_summary["positive_production_eligible"] += 1
                production_sides.append(side)

        if authoritative:
            for flag in ("return_candidate_eligible", "production_eligible"):
                summary["authoritative_candidate"][
                    flag if authoritative.get(flag) is True else f"not_{flag}"
                ] += 1
            support = _dict(authoritative.get("independent_direction_support"))
            support_reason = str(support.get("reason") or "missing")
            summary["authoritative_candidate"][f"support_{support_reason}"] += 1

        funnel_reason = classify_entry_funnel_reason(
            raw=raw,
            action=row.action,
            was_executed=bool(row.was_executed),
            has_order=False,
            reason=row.execution_reason,
        )
        summary["funnel_reasons"][str(funnel_reason or "executed")] += 1

        if (
            quality.get("analysis_complete") is not True
            and len(summary["incomplete_examples"]) < 20
        ):
            summary["incomplete_examples"].append(
                {
                    "id": row.id,
                    "symbol": row.symbol,
                    "created_at": row.created_at,
                    "execution_reason": row.execution_reason,
                    "quality": quality,
                    "market_model_timeout": raw.get("market_model_timeout"),
                    "expert_failures": raw.get("expert_failures"),
                    "decision_state_summary": _dict(
                        _dict(
                            raw.get("decision_state_machine")
                            or raw.get("decision_state")
                        ).get("summary")
                    ),
                    "model_timings": _list(raw.get("model_timings"))
                    or _list(row.model_health_timings),
                }
            )
        if (
            funnel_reason == "service_error"
            and len(summary["service_error_examples"]) < 20
        ):
            summary["service_error_examples"].append(
                {
                    "id": row.id,
                    "symbol": row.symbol,
                    "created_at": row.created_at,
                    "execution_reason": row.execution_reason,
                    "market_model_timeout": raw.get("market_model_timeout"),
                    "expert_failures": raw.get("expert_failures"),
                }
            )

        common_example = {
            "id": row.id,
            "symbol": row.symbol,
            "final_action": row.action,
            "preferred": preferred,
            "positive_sides": positive_sides,
            "production_sides": production_sides,
            "authoritative_return_candidate_eligible": authoritative.get(
                "return_candidate_eligible"
            ),
            "authoritative_production_eligible": authoritative.get(
                "production_eligible"
            ),
            "direction_support": _dict(
                authoritative.get("independent_direction_support")
            ),
        }
        if (
            positive_sides
            and all_hold
            and len(summary["positive_all_hold_examples"]) < 20
        ):
            summary["positive_all_hold_examples"].append(common_example)
        if (
            positive_sides
            and action == "hold"
            and len(summary["positive_not_selected_examples"]) < 20
        ):
            summary["positive_not_selected_examples"].append(common_example)
        if (
            production_sides
            and len(summary["positive_production_examples"]) < 20
        ):
            summary["positive_production_examples"].append(common_example)

    for row in consultation_rows:
        raw = _dict(row.raw_llm_response)
        consultation = _dict(
            raw.get("consultation") or raw.get("conflict_consultation")
        )
        if not consultation:
            continue
        analysis_type = str(row.analysis_type or "unknown")
        consultation_status = str(consultation.get("status") or "unknown")
        resolution_status = str(
            consultation.get("resolution_status") or "unknown"
        )
        summary["consultation"][consultation_status] += 1
        summary["consultation"][f"resolution_{resolution_status}"] += 1
        summary["consultation_by_analysis_type"][analysis_type][
            consultation_status
        ] += 1
        attempts = _list(consultation.get("consultation_attempts")) or _list(
            consultation.get("attempts")
        )
        attempt_statuses = []
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            attempt_status = str(attempt.get("status") or "unknown")
            attempt_statuses.append(attempt_status)
            duration = _float(
                attempt.get("duration_sec")
                or _dict(attempt.get("runtime_metrics")).get("inference_duration_seconds")
            )
            if duration is not None:
                summary["consultation_latency_values"].append(duration)
            summary["consultation"][f"attempt_{attempt_status}"] += 1
            summary["consultation_by_analysis_type"][analysis_type][
                f"attempt_{attempt_status}"
            ] += 1
        if len(summary["consultation_examples"]) < 20:
            summary["consultation_examples"].append(
                {
                    "id": row.id,
                    "analysis_type": analysis_type,
                    "symbol": row.symbol,
                    "created_at": row.created_at,
                    "status": consultation_status,
                    "resolution_status": resolution_status,
                    "resolved_action": consultation.get("resolved_action"),
                    "major_conflict_count": len(
                        _list(consultation.get("major_conflicts"))
                    ),
                    "attempt_statuses": attempt_statuses,
                    "production_permission": consultation.get(
                        "production_permission"
                    ),
                }
            )

    summary["top_symbols"] = summary["symbol_counts"].most_common(25)
    summary["max_symbol_share"] = (
        summary["top_symbols"][0][1] / len(rows)
        if rows and summary["top_symbols"]
        else 0.0
    )
    chronological = sorted(
        (row.created_at for row in rows if row.created_at is not None)
    )
    gaps = [
        (current - previous).total_seconds()
        for previous, current in zip(chronological, chronological[1:], strict=False)
    ]
    summary["analysis_gap_seconds"] = {
        "maximum": round(max(gaps), 3) if gaps else None,
        "average": round(sum(gaps) / len(gaps), 3) if gaps else None,
    }
    summary["latency"] = {
        metric: _metric_summary(values)
        for metric, values in summary.pop("latency_values").items()
    }
    summary["expert_latency"] = {
        name: _metric_summary(values)
        for name, values in summary.pop("expert_latency_values").items()
    }
    summary["expert_status"] = dict(summary["expert_status"])
    summary["consultation_latency"] = _metric_summary(
        summary.pop("consultation_latency_values")
    )
    summary["side_metrics"] = {
        side: {
            metric: _metric_summary(values)
            for metric, values in metrics.items()
        }
        for side, metrics in summary.pop("side_metric_values").items()
    }
    print(json.dumps(_normalize(summary), ensure_ascii=False, default=str))


asyncio.run(main())
'''


def _build_remote_command(limit: int, since_minutes: int) -> str:
    remote_script = (
        REMOTE_SCRIPT.replace("__LIMIT__", str(limit))
        .replace("__SINCE_MINUTES__", str(since_minutes))
    )
    payload = base64.b64encode(remote_script.encode("utf-8")).decode("ascii")
    runner = f"import base64;exec(base64.b64decode({payload!r}))"
    return " ".join(
        (
            "systemd-run --quiet --wait --pipe --collect",
            "--property=WorkingDirectory=/data/bb/app",
            "--property=User=bb",
            "--property=EnvironmentFile=-/data/bb/app/.env",
            "--property=EnvironmentFile=/etc/bb/bb-runtime.env",
            "/data/bb/app/.venv/bin/python -c",
            shlex.quote(runner),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether recent online market holds suppress positive evidence."
    )
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--minutes", type=int, default=0)
    args = parser.parse_args()
    limit = max(1, min(int(args.limit or 500), 5000))
    since_minutes = max(0, min(int(args.minutes or 0), 7 * 24 * 60))
    ssh = connect_remote_ssh(ROOT, timeout=25)
    try:
        output = run_remote_text(
            ssh,
            _build_remote_command(limit, since_minutes),
            timeout=180,
            max_output_chars=100_000,
        )
    finally:
        ssh.close()
    payload = json.loads(output)
    safe_print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
