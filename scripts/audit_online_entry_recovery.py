"""Read bounded entry, shadow-settlement and training recovery evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh  # noqa: E402
from core.safe_output import safe_print  # noqa: E402
from scripts import audit_online_market_hold_economics as transport  # noqa: E402

REMOTE_SCRIPT = r'''
import asyncio
import json
from collections import Counter
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select

from core.feature_availability import feature_availability_reasons
from db.session import get_read_session_ctx
from models.decision import AIDecision
from models.learning import ShadowBacktest
from services.entry_funnel_diagnostics import classify_entry_funnel_reason
from services.ml_signal_service import MLSignalService, MODEL_TRAINING_STATE_STORE
from services.shadow_training_quarantine import (
    requalify_verified_historical_shadow_samples,
)

LIMIT = __LIMIT__
SINCE_MINUTES = __SINCE_MINUTES__


def _dict(value):
    return value if isinstance(value, dict) else {}


def _selection(value):
    item = _dict(value)
    return {
        key: item.get(key) for key in (
            "selected", "selected_side", "selection_reason", "authorized",
            "paper_training_only", "paper_quality_observation_only",
            "current_edge_validated", "expected_net_return_pct",
            "prediction_horizon_minutes", "diagnostic_status",
            "blocking_reasons", "reason", "granted",
        ) if key in item
    }


async def query(statement):
    async with get_read_session_ctx(statement_timeout_ms=20000) as session:
        return (await session.execute(statement)).all()


async def main():
    now = datetime.now(UTC)
    since = now - timedelta(minutes=SINCE_MINUTES)
    report = {"generated_at": now.isoformat(), "since": since.isoformat()}

    async def capture(name, statement, convert):
        try:
            report[name] = convert(await asyncio.wait_for(query(statement), timeout=30))
        except Exception as exc:
            report[name] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}

    def compact_pending(rows):
        result = []
        for row in rows:
            item = dict(row._mapping)
            snapshot = _dict(item.pop("feature_snapshot", {}))
            market_quality = _dict(snapshot.get("market_data_quality"))
            contract = _dict(
                snapshot.get("training_market_fact_contract")
                or snapshot.get("market_fact_contract")
            )
            item["market_data_quality_code"] = market_quality.get("code")
            item["historical_market_path_only"] = (
                snapshot.get("historical_market_path_only") is True
            )
            item["historical_path_status"] = contract.get("path_status")
            result.append(item)
        return result

    await capture(
        "decisions",
        select(
            AIDecision.analysis_type, AIDecision.action, AIDecision.was_executed,
            func.count(), func.min(AIDecision.created_at), func.max(AIDecision.created_at),
        ).where(
            AIDecision.is_paper.is_(True), AIDecision.created_at >= since,
        ).group_by(AIDecision.analysis_type, AIDecision.action, AIDecision.was_executed),
        lambda rows: [dict(row._mapping) for row in rows],
    )
    await capture(
        "shadow_settlements",
        select(
            ShadowBacktest.status, ShadowBacktest.horizon_minutes, func.count(),
            func.max(ShadowBacktest.created_at).label("latest_created_at"),
            func.max(ShadowBacktest.updated_at).label("latest_updated_at"),
            func.min(ShadowBacktest.due_at).label("oldest_due_at"),
        ).where(
            ShadowBacktest.execution_mode == "paper", ShadowBacktest.created_at >= since,
        ).group_by(ShadowBacktest.status, ShadowBacktest.horizon_minutes),
        lambda rows: [dict(row._mapping) for row in rows],
    )
    await capture(
        "overdue",
        select(
            ShadowBacktest.horizon_minutes, func.count().label("count"),
            func.min(ShadowBacktest.due_at).label("oldest_due_at"),
        ).where(
            ShadowBacktest.execution_mode == "paper",
            ShadowBacktest.status == "pending", ShadowBacktest.due_at < now,
        ).group_by(ShadowBacktest.horizon_minutes),
        lambda rows: [dict(row._mapping) for row in rows],
    )
    oldest_ids = select(ShadowBacktest.id).where(
        ShadowBacktest.status == "pending", ShadowBacktest.due_at < now,
    ).order_by(ShadowBacktest.due_at, ShadowBacktest.id).limit(14).scalar_subquery()
    await capture(
        "oldest_pending",
        select(
            ShadowBacktest.id, ShadowBacktest.symbol, ShadowBacktest.horizon_minutes,
            ShadowBacktest.due_at, ShadowBacktest.feature_snapshot,
        ).where(ShadowBacktest.id.in_(oldest_ids)).order_by(ShadowBacktest.due_at),
        compact_pending,
    )
    quarantine_ids = select(ShadowBacktest.id).where(
        ShadowBacktest.execution_mode == "paper",
        ShadowBacktest.status == "quarantined", ShadowBacktest.created_at >= since,
    ).order_by(ShadowBacktest.id.desc()).limit(100).scalar_subquery()
    await capture(
        "quarantine_reasons",
        select(ShadowBacktest.note).where(ShadowBacktest.id.in_(quarantine_ids)),
        lambda rows: dict(Counter(
            str(row[0] or "").split("[training_quarantine]")[-1].strip()
            for row in rows
        )),
    )
    raw = AIDecision.raw_llm_response
    fields = (
        "paper_trade_selection", "normal_paper_trade", "entry_permission",
        "entry_candidate_evidence", "analysis_quality_contract",
        "authoritative_return_candidate", "continuous_model_weights",
        "ml_signal", "local_ml_signal",
    )
    recent_ids = select(AIDecision.id).where(
        AIDecision.is_paper.is_(True), AIDecision.analysis_type == "market",
        AIDecision.created_at >= since,
    ).order_by(AIDecision.id.desc()).limit(LIMIT).scalar_subquery()
    await capture(
        "recent",
        select(
            AIDecision.id, AIDecision.symbol, AIDecision.action,
            AIDecision.created_at, AIDecision.was_executed, AIDecision.execution_reason,
            AIDecision.feature_snapshot.label("feature_snapshot"),
            *(raw[field].label(field) for field in fields),
        ).where(AIDecision.id.in_(recent_ids)).order_by(AIDecision.id.desc()),
        lambda rows: [dict(row._mapping) for row in rows],
    )
    recent = report.get("recent")
    if isinstance(recent, list):
        reasons = Counter()
        identities = Counter()
        permissions = {}
        examples = []
        availability_reasons = Counter()
        availability_clean = 0
        sealed_count = 0
        for row in recent:
            snapshot = _dict(row.get("feature_snapshot"))
            if isinstance(snapshot.get("feature_availability_contract"), dict):
                sealed_count += 1
            reasons_for_snapshot = feature_availability_reasons(
                snapshot,
                decision_at=row.get("created_at"),
            )
            if reasons_for_snapshot:
                availability_reasons.update(reasons_for_snapshot)
            else:
                availability_clean += 1
            reason = classify_entry_funnel_reason(
                raw={field: row.get(field) for field in fields},
                action=row["action"], was_executed=row["was_executed"],
                has_order=False, reason=row["execution_reason"],
            )
            reasons[reason or "executed"] += 1
            weights = _dict(row.get("continuous_model_weights"))
            for source, item in _dict(weights.get("quant_source_weights")).items():
                item = _dict(item)
                permissions.setdefault(source, {
                    key: item.get(key) for key in (
                        "paper_execution_permission", "paper_execution_reason",
                        "paper_execution_blockers", "paper_execution_evidence_source",
                        "paper_execution_evidence", "shadow_fee_after_return",
                    )
                })
            signal = _dict(row.get("local_ml_signal") or row.get("ml_signal"))
            identities[str(signal.get("model_version") or "unavailable")] += 1
            selection = _dict(row.get("paper_trade_selection"))
            if len(examples) < 6 or (selection.get("selected") and len(examples) < 12):
                evidence = _dict(row.get("entry_candidate_evidence"))
                examples.append({
                    key: row.get(key) for key in (
                        "id", "symbol", "action", "created_at", "execution_reason",
                    )
                } | {
                    "funnel_reason": reason,
                    "selection": _selection(selection),
                    "entry_permission": _selection(row.get("entry_permission")),
                    "analysis_complete": _dict(
                        row.get("analysis_quality_contract")
                    ).get("analysis_complete"),
                    "quality_permissions": evidence.get("quality_permissions"),
                    "model_version": signal.get("model_version"),
                    "derivatives_route_status": snapshot.get("derivatives_route_status"),
                    "derivatives_required_routes": snapshot.get(
                        "derivatives_required_routes"
                    ),
                    "derivatives_unavailable_reason": snapshot.get(
                        "derivatives_unavailable_reason"
                    ),
                })
        report["recent"] = {
            "count": len(recent), "funnel_reasons": dict(reasons),
            "model_versions": dict(identities),
            "availability": {
                "sealed_count": sealed_count,
                "clean_count": availability_clean,
                "invalid_count": len(recent) - availability_clean,
                "reason_counts": dict(availability_reasons),
            },
            "quality_permissions": permissions, "examples": examples,
        }
    state = MODEL_TRAINING_STATE_STORE.read()
    models = {}
    for key, item in _dict(state.get("models")).items():
        if not isinstance(item, dict):
            continue
        result = _dict(item.get("last_result"))
        models[key] = {
            field: item.get(field) for field in (
                "state", "last_check_at", "last_successful_training_at", "last_error",
            )
        } | {"last_result": {
            field: result.get(field) for field in (
                "status", "reason", "next_check_at", "retrain_reason",
                "current_sample_count", "last_trained_sample_count",
                "new_sample_count", "sample_count",
            ) if field in result
        }}
    report["training"] = {
        "updated_at": state.get("updated_at"),
        "schedulers": state.get("schedulers"),
        "models": models,
    }
    report["historical_requalification_dry_run"] = (
        await requalify_verified_historical_shadow_samples(
            batch_size=1000,
            max_batches=300,
            dry_run=True,
        )
    )
    metadata = MLSignalService().status_metadata()
    report["model"] = {key: metadata.get(key) for key in (
        "version", "artifact_version", "trained_at", "latest_training",
        "artifact_lifecycle", "readiness_state", "influence_policy",
    )}
    print(json.dumps(report, ensure_ascii=False, default=str))


asyncio.run(main())
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--minutes", type=int, default=840)
    args = parser.parse_args()
    transport.REMOTE_SCRIPT = REMOTE_SCRIPT
    ssh = connect_remote_ssh(ROOT, timeout=25)
    try:
        report = transport._read_remote_report(
            ssh, max(1, min(args.limit, 300)), max(1, min(args.minutes, 10080))
        )
    finally:
        ssh.close()
    safe_print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
