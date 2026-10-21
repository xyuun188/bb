"""Read-only original-clock and component-lineage audit of frozen development IDs."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shlex
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from sqlalchemy import select  # noqa: E402

from db.session import close_db, get_read_session_ctx  # noqa: E402
from models.decision import AIDecision  # noqa: E402
from models.learning import ShadowBacktest  # noqa: E402
from scripts.audit_development_evidence import (  # noqa: E402
    CUTOFF,
    EXPECTED_SNAPSHOT_SHA256,
    SNAPSHOT,
    _timestamp,
)
from services.execution_cost_model import _parse_timestamp  # noqa: E402
from services.ml_development_snapshot import load_development_snapshot  # noqa: E402

DEPENDENCIES = (
    "scripts/audit_development_timing.py",
    "scripts/audit_development_evidence.py",
    "services/ml_development_snapshot.py",
    "models/decision.py",
    "models/learning.py",
    "core/market_facts.py",
    "core/runtime_data_retention_contract.py",
    "services/execution_cost_model.py",
)
COMPONENTS = (
    "market_source", "market_received", "book_source", "book_received",
    "fee_observed", "funding_observed",
)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str,
    ).encode()).hexdigest()


def code_identity() -> dict[str, Any]:
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
              for name in DEPENDENCIES}
    return {"dependency_sha256": hashes, "sha256": _fingerprint(hashes)}


def timing_columns() -> tuple[Any, ...]:
    """Select scalar JSON paths, never the wide feature/decision payloads."""
    shadow = ShadowBacktest.training_feature_snapshot
    original = AIDecision.feature_snapshot
    contract = ShadowBacktest.feature_snapshot["market_fact_contract"]
    return (
        ShadowBacktest.id, ShadowBacktest.decision_id, ShadowBacktest.created_at,
        ShadowBacktest.due_at, ShadowBacktest.horizon_minutes,
        shadow["timestamp"].label("feature_at"),
        shadow["historical_market_path_only"].label("historical"),
        AIDecision.created_at.label("original_created_at"),
        AIDecision.raw_llm_response["timing"].label("original_timing"),
        AIDecision.decision_learning_snapshot["timing"].label("compact_timing"),
        original["timestamp"].label("original_feature_at"),
        original["market_fact"]["source_timestamp_ms"].label("market_source"),
        original["market_fact"]["received_at"].label("market_received"),
        original["orderbook_fact"]["source_timestamp_ms"].label("book_source"),
        original["orderbook_fact"]["received_at"].label("book_received"),
        original["fee_rate_observed_at"].label("fee_observed"),
        original["funding_rate_observed_at"].label("funding_observed"),
        shadow["fee_rate_observed_at"].label("shadow_fee_observed"),
        shadow["funding_rate_observed_at"].label("shadow_funding_observed"),
        shadow["taker_fee_rate"].label("shadow_taker_fee"),
        original["taker_fee_rate"].label("original_taker_fee"),
        shadow["training_market_fact_contract"]["observation_window"].label(
            "contract_observation_window"
        ),
        contract["entry_fact"]["source_timestamp_ms"].label("entry_source"),
        contract["entry_fact"]["received_at"].label("entry_received"),
        contract["result_fact"]["source_timestamp_ms"].label("result_source"),
        contract["result_fact"]["received_at"].label("result_received"),
    )


def _clock(value: Any) -> pd.Timestamp | None:
    if value is None or isinstance(value, bool):
        return None
    parsed = _parse_timestamp(str(value))
    return _timestamp(parsed) if parsed is not None else None


def inspect_timing(row: dict[str, Any]) -> dict[str, Any]:
    """Recover recorded clocks without asserting unrecorded component availability."""
    issues: list[str] = []
    raw = row.get("original_timing")
    compact = row.get("compact_timing")
    raw = raw if isinstance(raw, dict) else {}
    compact = compact if isinstance(compact, dict) else {}
    started = _clock(raw.get("analysis_started_at"))
    completed = _clock(raw.get("decision_completed_at"))
    original_created = _clock(row.get("original_created_at"))
    if original_created is None:
        issues.append("original_decision_missing")
    if started is None or completed is None:
        issues.append("original_decision_clock_missing")
    if raw.get("stage") != "market":
        issues.append("original_decision_stage_unverified")
    if started is not None and completed is not None and started > completed:
        issues.append("original_decision_clock_reversed")
    if completed is not None and original_created is not None and completed > original_created:
        issues.append("completion_after_original_insertion")
    for key in ("analysis_started_at", "decision_completed_at"):
        if compact.get(key) is not None and _clock(compact[key]) != _clock(raw.get(key)):
            issues.append(f"original_compact_timing_conflict:{key}")
    clock_recovered = not issues
    observed = _clock(row.get("feature_at"))
    original_observed = _clock(row.get("original_feature_at"))
    if observed is None:
        issues.append("feature_clock_missing")
    if original_observed is None:
        issues.append("original_feature_clock_missing")
    elif observed != original_observed:
        issues.append("shadow_original_feature_clock_mismatch")
    if observed is not None and completed is not None and observed > completed:
        issues.append("feature_constructed_after_decision_completed")
    if row.get("historical") is True:
        issues.append("historical_bar_close_availability_unverified")
    components = {}
    offsets = {}
    for key in COMPONENTS:
        value = _clock(row.get(key))
        components[key] = (
            "missing" if value is None else
            "decision_clock_unresolved" if not clock_recovered else
            "after_completion" if value > completed else "no_later_than_completion"
        )
        if components[key] == "after_completion":
            issues.append(f"component_after_completion:{key}")
    for key in ("fee_observed", "funding_observed"):
        shadow_clock = _clock(row.get(f"shadow_{key}"))
        original_clock = _clock(row.get(key))
        if shadow_clock is not None and shadow_clock != original_clock:
            issues.append(f"shadow_original_cost_clock_mismatch:{key}")
        if shadow_clock is not None and clock_recovered and shadow_clock > completed:
            issues.append(f"shadow_cost_observation_after_completion:{key}")
    if row.get("shadow_taker_fee") != row.get("original_taker_fee"):
        issues.append("shadow_original_taker_fee_value_mismatch")
    window = row.get("contract_observation_window")
    if isinstance(window, str):
        try:
            window = json.loads(window)
        except (TypeError, ValueError):
            window = None
    if isinstance(window, dict):
        for key, boundary in (
            ("contract_entry_observed", "start"), ("contract_result_observed", "end"),
        ):
            value = _clock(window.get(boundary))
            if value is not None and clock_recovered:
                offsets[f"{key}_minus_completion_seconds"] = (
                    value - completed
                ).total_seconds()
    due = _clock(row.get("due_at"))
    created = _clock(row.get("created_at"))
    if isinstance(window, dict) and due is not None:
        result_observed = _clock(window.get("end"))
        if result_observed is not None:
            offsets["contract_result_observed_minus_due_seconds"] = (
                result_observed - due
            ).total_seconds()
            if result_observed < due:
                issues.append("contract_result_observed_precedes_scheduled_label")
    if clock_recovered:
        for key, value in (
            ("stored_creation_minus_completion_seconds", created),
            ("entry_source_minus_completion_seconds", _clock(row.get("entry_source"))),
            ("entry_received_minus_completion_seconds", _clock(row.get("entry_received"))),
        ):
            if value is not None:
                offsets[key] = (value - completed).total_seconds()
        if due is not None:
            offsets["scheduled_horizon_minus_recorded_horizon_seconds"] = (
                (due - completed).total_seconds() - int(row["horizon_minutes"]) * 60
            )
        if observed is not None:
            offsets["feature_construction_age_at_completion_seconds"] = (
                completed - observed
            ).total_seconds()
    result_source = _clock(row.get("result_source"))
    if result_source is not None and due is not None:
        offsets["result_source_minus_due_seconds"] = (result_source - due).total_seconds()
        if result_source < due:
            issues.append("result_source_precedes_scheduled_label")
    # FeatureVector records construction time, not individual indicator/news input clocks.
    components["technical_and_sentiment_inputs"] = "individual_availability_not_recorded"
    return {
        "recorded_original_clock_recovered": clock_recovered,
        "original_completed_at": completed.isoformat() if completed is not None else None,
        "feature_construction_before_completion": (
            observed <= completed if observed is not None and clock_recovered else None
        ),
        "component_clocks": components, "offset_seconds": offsets, "issues": issues,
        "g2_row_accepted": False,
    }


async def audit(*, max_rows: int) -> dict[str, Any]:
    frame, snapshot = load_development_snapshot(SNAPSHOT, cutoff=CUTOFF)
    if snapshot["snapshot_sha256"] != EXPECTED_SNAPSHOT_SHA256:
        raise ValueError("unexpected frozen snapshot identity")
    ids = frame.iloc[:max_rows]["id"].astype(int).tolist()
    counts: Counter = Counter()
    components: dict[str, Counter] = defaultdict(Counter)
    offsets: dict[str, list[float]] = defaultdict(list)
    identities = []
    evidence = []
    examples = []
    async with get_read_session_ctx(
        statement_timeout_ms=15_000,
        transaction_isolation="REPEATABLE READ READ ONLY",
    ) as session:
        for offset in range(0, len(ids), 200):
            page_ids = ids[offset:offset + 200]
            result = await session.execute(
                select(*timing_columns()).outerjoin(
                    AIDecision, ShadowBacktest.decision_id == AIDecision.id,
                ).where(ShadowBacktest.id.in_(page_ids)).order_by(ShadowBacktest.id)
            )
            rows = [dict(row) for row in result.mappings()]
            if len({row["id"] for row in rows}) != len(rows):
                raise ValueError("duplicate shadow source identity")
            counts["source_row_missing"] += len(page_ids) - len(rows)
            for row in rows:
                inspected = inspect_timing(row)
                identities.append(row["id"])
                evidence.append((row["id"], row["decision_id"], _fingerprint(row)))
                counts["rows_checked"] += 1
                counts["historical_rows"] += row.get("historical") is True
                counts["recorded_original_clock_recovered"] += inspected[
                    "recorded_original_clock_recovered"
                ]
                counts["feature_construction_before_completion"] += (
                    inspected["feature_construction_before_completion"] is True
                )
                counts.update(inspected["issues"])
                for key, status in inspected["component_clocks"].items():
                    components[key][status] += 1
                for key, value in inspected["offset_seconds"].items():
                    offsets[key].append(value)
                if len(examples) < 4:
                    examples.append({"id": row["id"], **inspected})
    return {
        "audit_version": "development-timing.v2", "audit_code_identity": code_identity(),
        "generated_at": datetime.now(UTC).isoformat(), "read_only": True,
        "exam_labels_read": False, "database_label_values_selected": False,
        "production_permission": False, "source_row_budget": max_rows,
        "snapshot_sha256": snapshot["snapshot_sha256"], "development_cutoff": CUTOFF,
        "source_coverage_complete": len(identities) == len(ids) == len(frame),
        "checked_source_ids_sha256": _fingerprint(sorted(identities)),
        "source_clock_evidence_sha256": _fingerprint(sorted(evidence)),
        "counts": dict(sorted(counts.items())),
        "component_clock_counts": {key: dict(value) for key, value in components.items()},
        "offset_seconds": {
            key: {
                "count": len(values), "minimum": min(values), "maximum": max(values),
                "negative_count": sum(value < 0 for value in values),
                "p50": float(pd.Series(values).quantile(0.5)),
                "p95": float(pd.Series(values).quantile(0.95)),
            } for key, values in offsets.items()
        },
        "bounded_examples": examples,
        "g2_acceptance": "blocked_by_component_availability_evidence",
        "clock_recovery_is_not_complete_component_lineage": True,
        "profitability_acceptance": "not_evaluated",
    }


def audit_online(*, max_rows: int) -> dict[str, Any]:
    from core.remote_ssh import connect_remote_ssh, run_remote_text

    identity = code_identity()
    unit = f"bb-frozen-timing-r{max_rows}-{identity['sha256'][:12]}"
    output = f"/data/bb/app/data/profitability_remediation/{unit.removeprefix('bb-')}.json"
    command = [
        "systemd-run", "--quiet", "--wait", "--pipe", "--collect", f"--unit={unit}",
        "--property=WorkingDirectory=/data/bb/app", "--property=User=bb",
        "--property=EnvironmentFile=-/data/bb/app/.env",
        "--property=EnvironmentFile=/etc/bb/bb-runtime.env",
        "--property=CPUQuota=100%", "--property=MemoryMax=1G",
        "--property=RuntimeMaxSec=300", "--property=TimeoutStopSec=10",
        "--property=KillMode=control-group", "--property=Nice=10",
        "--setenv=DATABASE_POOL_SIZE=1", "--setenv=DATABASE_MAX_OVERFLOW=0",
        "--setenv=OMP_NUM_THREADS=1", "--setenv=OPENBLAS_NUM_THREADS=1",
        "/data/bb/app/.venv/bin/python", "scripts/audit_development_timing.py",
        "--max-rows", str(max_rows), "--output", output,
    ]
    runner = """
import pathlib,subprocess,sys
p=pathlib.Path(sys.argv[1])
if not p.exists():
    active=subprocess.run(['systemctl','is-active','--quiet',sys.argv[2]],capture_output=True)
    if active.returncode == 0:
        raise RuntimeError('timing audit already active; inspect named unit')
    result=subprocess.run(sys.argv[3:],capture_output=True)
    if result.returncode:
        raise RuntimeError('timing audit failed; inspect named unit')
print(p.read_text())
"""
    ssh = connect_remote_ssh(ROOT, timeout=25)
    try:
        report = json.loads(run_remote_text(ssh, shlex.join([
            "/data/bb/app/.venv/bin/python", "-c", runner, output, unit, *command,
        ]), timeout=330, max_output_chars=50_000))
    finally:
        ssh.close()
    if report.get("audit_code_identity") != identity or report.get("source_row_budget") != max_rows:
        raise ValueError("online timing audit code or budget mismatch")
    return report | {"remote_unit": unit, "remote_report_path": output}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rows", type=int, choices=range(1, 20_001), default=10_000)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    async def run() -> dict[str, Any]:
        try:
            return await asyncio.wait_for(audit(max_rows=args.max_rows), timeout=270)
        finally:
            await close_db()

    report = audit_online(max_rows=args.max_rows) if args.online else asyncio.run(run())
    encoded = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")  # noqa: ASYNC240
    print(encoded)


if __name__ == "__main__":
    main()
