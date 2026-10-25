"""Audit one sealed shadow cohort for native-label eligibility and provenance."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import secrets
import shlex
import sys
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402

DEPENDENCIES = (
    "core/training_contracts.py",
    "services/ml_training_dataset.py",
    "services/training_data_quality.py",
)
HORIZONS = (5, 15, 60, 240)
NATIVE_SOURCE = "shadow_entry_result_and_okx_native_1m_path"

REMOTE_SCRIPT = r"""
import asyncio
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select

from core.feature_availability import feature_availability_reasons
from db.session import close_db, get_read_session_ctx
from models.decision import AIDecision
from models.learning import ShadowBacktest
from services.training_data_quality import assess_shadow_sample

DECISION_SINCE = datetime.fromisoformat(
    __DECISION_SINCE__.replace("Z", "+00:00")
).astimezone(UTC)
DECISION_CUTOFF = datetime.fromisoformat(
    __DECISION_CUTOFF__.replace("Z", "+00:00")
).astimezone(UTC)
MATURITY_CUTOFF = datetime.fromisoformat(
    __MATURITY_CUTOFF__.replace("Z", "+00:00")
).astimezone(UTC)
ROW_LIMIT = __ROW_LIMIT__
EXPECTED_DECISION_COUNT = __EXPECTED_DECISION_COUNT__
DEPENDENCIES = __DEPENDENCIES__
HORIZONS = (5, 15, 60, 240)
NATIVE_SOURCE = "shadow_entry_result_and_okx_native_1m_path"


def dependency_hashes():
    return {
        name: hashlib.sha256(__import__("pathlib").Path(name).read_bytes()).hexdigest()
        for name in DEPENDENCIES
    }


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif value:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _group_key(row: dict[str, Any]) -> str:
    decision_id = int(row.get("decision_id") or 0)
    return str(decision_id if decision_id > 0 else -int(row.get("id") or 0))


def _decision_availability_reasons(decision: dict[str, Any]) -> list[str]:
    features = _dict(decision.get("feature_snapshot"))
    decision_at = features.get("original_decision_timestamp")
    return feature_availability_reasons(features, decision_at=decision_at)


def _row_classification(row: dict[str, Any]) -> tuple[str, list[str]]:
    features = _dict(row.get("training_feature_snapshot"))
    decision_at = _time(features.get("original_decision_timestamp"))
    if decision_at is None:
        return "missing_decision_timestamp", ["decision_timestamp_missing"]
    if decision_at < DECISION_SINCE or decision_at >= DECISION_CUTOFF:
        return "outside_decision_window", ["decision_outside_registered_window"]

    updated_at = _time(row.get("updated_at"))
    status = str(row.get("status") or "")
    if updated_at is None or updated_at >= MATURITY_CUTOFF:
        return "updated_after_maturity_cutoff", ["label_not_available_at_maturity_cutoff"]
    if status == "pending":
        return "pending_at_cutoff", ["label_pending_at_maturity_cutoff"]
    if status == "quarantined":
        note = str(row.get("note") or "")
        marker = "[training_quarantine]"
        reasons = (
            [item.strip() for item in note.split(marker, 1)[1].split(",") if item.strip()]
            if marker in note
            else ["training_quarantine_reason_missing"]
        )
        return "quarantined_at_cutoff", reasons
    if status != "completed":
        return "not_completed_at_cutoff", [f"label_status:{status or 'missing'}"]

    market = _dict(features.get("training_market_fact_contract"))
    label = _dict(features.get("training_label_contract"))
    labels = _dict(label.get("labels")) or label
    native = market.get("source") == NATIVE_SOURCE
    fee_after = labels.get("fee_after_complete") is True
    reasons = []
    if not native:
        reasons.append("native_market_path_missing_or_mismatched")
    if not fee_after:
        reasons.append("fee_after_label_incomplete")

    sample = {
        "id": row.get("id"),
        "decision_id": row.get("decision_id"),
        "symbol": row.get("symbol"),
        "horizon_minutes": row.get("horizon_minutes"),
        "label_version": row.get("label_version"),
        "features": features,
        "long_return_pct": row.get("long_return_pct"),
        "short_return_pct": row.get("short_return_pct"),
        "label_timestamp": row.get("due_at"),
        "original_decision_timestamp": features.get("original_decision_timestamp"),
    }
    assessment = assess_shadow_sample(sample)
    reasons.extend(str(reason) for reason in assessment.reasons)
    if reasons:
        return "mature_excluded", list(dict.fromkeys(reasons))
    return "mature_eligible", []


async def main():
    now = datetime.now(UTC)
    if now < MATURITY_CUTOFF:
        print(
            json.dumps(
                {
                    "experiment": "bounded-sealed-shadow-cohort-native-label-provenance-audit",
                    "status": "maturity_not_reached",
                    "checked_at": now.isoformat(),
                    "window": {
                        "decision_since": DECISION_SINCE.isoformat(),
                        "decision_cutoff_exclusive": DECISION_CUTOFF.isoformat(),
                        "maturity_cutoff_exclusive": MATURITY_CUTOFF.isoformat(),
                        "expected_decision_snapshot_count": EXPECTED_DECISION_COUNT,
                        "horizons": list(HORIZONS),
                        "native_source": NATIVE_SOURCE,
                        "row_limit": ROW_LIMIT,
                        "label_row_limit": ROW_LIMIT * len(HORIZONS),
                    },
                    "input": {
                        "decision_snapshot_count": 0,
                        "decision_candidate_count": 0,
                        "decision_candidate_in_window_count": 0,
                        "decision_cohort_complete": False,
                        "source_row_count": 0,
                        "cohort_row_count": 0,
                    },
                    "safety": {
                        "read_only": True,
                        "database_mutated": False,
                        "label_rows_queried": False,
                        "exam_labels_evaluated": False,
                        "artifact_persisted": False,
                        "promotion_authorized": False,
                        "orders_submitted": False,
                    },
                },
                ensure_ascii=False,
            )
        )
        return
    decision_candidate_limit = ROW_LIMIT * 4
    decision_candidate_statement = (
        select(
            AIDecision.id,
            AIDecision.symbol,
            AIDecision.created_at,
            AIDecision.feature_snapshot["original_decision_timestamp"].as_string().label(
                "original_decision_timestamp"
            ),
        )
        .where(
            AIDecision.is_paper.is_(True),
            AIDecision.analysis_type == "market",
            AIDecision.created_at >= DECISION_SINCE - timedelta(hours=2),
            AIDecision.created_at < DECISION_CUTOFF + timedelta(hours=2),
        )
        .order_by(AIDecision.created_at.asc(), AIDecision.id.asc())
        .limit(decision_candidate_limit + 1)
    )
    async with get_read_session_ctx(
        statement_timeout_ms=15000,
        transaction_isolation="REPEATABLE READ READ ONLY",
    ) as session:
        decision_candidates = [
            dict(row)
            for row in (
                await session.execute(decision_candidate_statement)
            ).mappings()
        ]
        decision_candidates_in_window = [
            row
            for row in decision_candidates
            if (
                (decision_at := _time(row.get("original_decision_timestamp")))
                is not None
                and DECISION_SINCE <= decision_at < DECISION_CUTOFF
            )
        ]
        decision_ids = [
            int(row["id"]) for row in decision_candidates_in_window
        ]
        if decision_ids:
            decision_statement = select(
                AIDecision.id,
                AIDecision.symbol,
                AIDecision.created_at,
                AIDecision.feature_snapshot,
            ).where(AIDecision.id.in_(decision_ids))
            decision_rows = [
                dict(row)
                for row in (await session.execute(decision_statement)).mappings()
            ]
        else:
            decision_rows = []
        decision_cohort_complete = (
            len(decision_rows) == EXPECTED_DECISION_COUNT
            and len(decision_rows) <= ROW_LIMIT
        )
        label_row_limit = ROW_LIMIT * len(HORIZONS)
        if decision_cohort_complete:
            statement = (
                select(
                    ShadowBacktest.id,
                    ShadowBacktest.decision_id,
                    ShadowBacktest.symbol,
                    ShadowBacktest.horizon_minutes,
                    ShadowBacktest.status,
                    ShadowBacktest.created_at,
                    ShadowBacktest.updated_at,
                    ShadowBacktest.due_at,
                    ShadowBacktest.label_version,
                    ShadowBacktest.long_return_pct,
                    ShadowBacktest.short_return_pct,
                    ShadowBacktest.note,
                    ShadowBacktest.training_feature_snapshot,
                )
                .where(
                    ShadowBacktest.execution_mode == "paper",
                    ShadowBacktest.analysis_type == "market",
                    ShadowBacktest.decision_id.in_(decision_ids),
                    ShadowBacktest.horizon_minutes.in_(HORIZONS),
                )
                .order_by(ShadowBacktest.created_at.asc(), ShadowBacktest.id.asc())
                .limit(label_row_limit + 1)
            )
            result = await session.execute(statement)
            source_rows = [dict(row) for row in result.mappings()]
        else:
            source_rows = []

    try:
        decision_availability = Counter(
            reason
            for decision in decision_rows
            for reason in _decision_availability_reasons(decision)
        )
        decision_ids_sha256 = hashlib.sha256(
            ",".join(str(value) for value in sorted(decision_ids)).encode("ascii")
        ).hexdigest()
        groups = {
            str(decision_id): defaultdict(list) for decision_id in decision_ids
        }
        rows = []
        for row in source_rows:
            state, reasons = _row_classification(row)
            report_row = {
                    "id": int(row.get("id") or 0),
                    "decision_id": int(row.get("decision_id") or 0) or None,
                    "symbol": str(row.get("symbol") or ""),
                    "horizon_minutes": int(row.get("horizon_minutes") or 0),
                    "status": str(row.get("status") or ""),
                    "created_at": row.get("created_at"),
                    "updated_at": row.get("updated_at"),
                    "state": state,
                    "reasons": reasons,
                }
            rows.append(report_row)
            group_key = _group_key(report_row)
            if group_key in groups:
                groups[group_key][report_row["horizon_minutes"]].append(report_row)

        state_counts = Counter(row["state"] for row in rows)
        reason_counts = Counter(
            reason for row in rows for reason in row["reasons"]
        )
        by_horizon = {
            str(horizon): dict(
                Counter(
                    row["state"]
                    for row in rows
                    if row["horizon_minutes"] == horizon
                )
            )
            for horizon in HORIZONS
        }
        patterns = Counter()
        all_four_eligible = 0
        mature_all_four = 0
        mature_rows = 0
        duplicate_group_horizon_count = 0
        duplicate_row_count = 0
        for horizon_rows in groups.values():
            duplicate_group_horizon_count += sum(
                len(horizon_rows.get(horizon, [])) > 1 for horizon in HORIZONS
            )
            duplicate_row_count += sum(
                max(len(horizon_rows.get(horizon, [])) - 1, 0)
                for horizon in HORIZONS
            )
            pattern = []
            for horizon in HORIZONS:
                horizon_rows_for_group = horizon_rows.get(horizon, [])
                if not horizon_rows_for_group:
                    pattern.append("missing")
                elif len(horizon_rows_for_group) > 1:
                    pattern.append("duplicate_rows")
                else:
                    pattern.append(horizon_rows_for_group[0]["state"])
            pattern = tuple(pattern)
            patterns[pattern] += 1
            mature_states = {
                "mature_eligible",
                "mature_excluded",
                "quarantined_at_cutoff",
            }
            if all(
                len(horizon_rows.get(horizon, [])) == 1
                and horizon_rows[horizon][0]["state"] in mature_states
                for horizon in HORIZONS
            ):
                mature_all_four += 1
                mature_rows += 4
                if all(
                    horizon_rows[horizon][0]["state"] == "mature_eligible"
                    for horizon in HORIZONS
                ):
                    all_four_eligible += 1

        report = {
            "experiment": "bounded-sealed-shadow-cohort-native-label-provenance-audit",
            "status": (
                "ready_for_label_audit"
                if decision_cohort_complete
                else "sealed_decision_cohort_mismatch"
            ),
            "checked_at": datetime.now(UTC).isoformat(),
            "window": {
                "decision_since": DECISION_SINCE.isoformat(),
                "decision_cutoff_exclusive": DECISION_CUTOFF.isoformat(),
                "maturity_cutoff_exclusive": MATURITY_CUTOFF.isoformat(),
                "expected_decision_snapshot_count": EXPECTED_DECISION_COUNT,
                "horizons": list(HORIZONS),
                "native_source": NATIVE_SOURCE,
                "row_limit": ROW_LIMIT,
                "label_row_limit": label_row_limit,
            },
            "input": {
                "decision_snapshot_count": len(decision_rows),
                "decision_candidate_count": len(decision_candidates),
                "decision_candidate_in_window_count": len(
                    decision_candidates_in_window
                ),
                "decision_candidate_limit": decision_candidate_limit,
                "decision_candidate_limit_saturated": len(decision_candidates)
                > decision_candidate_limit,
                "decision_cohort_complete": decision_cohort_complete,
                "decision_group_ids_sha256": decision_ids_sha256,
                "decision_feature_availability_reason_counts": dict(
                    decision_availability.most_common(40)
                ),
                "source_row_count": len(source_rows),
                "cohort_row_count": len(rows),
                "decision_row_limit_saturated": len(decision_rows) > ROW_LIMIT,
                "label_row_limit_saturated": len(source_rows) > label_row_limit,
                "dependency_sha256": dependency_hashes(),
            },
            "coverage": {
                "decision_group_count": len(groups),
                "mature_all_four_group_count": mature_all_four,
                "mature_all_four_row_count": mature_rows,
                "mature_all_four_eligible_group_count": all_four_eligible,
                "duplicate_group_horizon_count": duplicate_group_horizon_count,
                "duplicate_row_count": duplicate_row_count,
                "state_counts": dict(state_counts),
                "state_counts_by_horizon": by_horizon,
                "reason_counts": dict(reason_counts.most_common(40)),
                "state_patterns": [
                    {"states": list(pattern), "groups": count}
                    for pattern, count in patterns.most_common(40)
                ],
            },
            "sample_rows": rows[:24],
            "safety": {
                "read_only": True,
                "database_mutated": False,
                "label_rows_queried": decision_cohort_complete,
                "exam_labels_evaluated": False,
                "artifact_persisted": False,
                "promotion_authorized": False,
                "orders_submitted": False,
            },
        }
        print(json.dumps(report, ensure_ascii=False, default=str))
    finally:
        await close_db()


asyncio.run(main())
"""


def _parse_cutoff(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("cutoff must include a timezone")
    return parsed.astimezone(UTC)


def _render_remote_script(
    *,
    decision_since: datetime,
    decision_cutoff: datetime,
    maturity_cutoff: datetime,
    row_limit: int,
    expected_decision_count: int,
) -> str:
    return (
        REMOTE_SCRIPT.replace(
            "__DECISION_SINCE__",
            repr(decision_since.isoformat().replace("+00:00", "Z")),
        )
        .replace(
            "__DECISION_CUTOFF__",
            repr(decision_cutoff.isoformat().replace("+00:00", "Z")),
        )
        .replace(
            "__MATURITY_CUTOFF__",
            repr(maturity_cutoff.isoformat().replace("+00:00", "Z")),
        )
        .replace("__ROW_LIMIT__", str(row_limit))
        .replace("__EXPECTED_DECISION_COUNT__", str(expected_decision_count))
        .replace("__DEPENDENCIES__", repr(DEPENDENCIES))
    )


def _local_dependency_hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in DEPENDENCIES
    }


def _resolve_evidence_path(value: str | None, default_name: str) -> Path:
    """Resolve optional evidence output without allowing writes outside the audit dir."""

    evidence_root = (ROOT / "data" / "profitability_remediation").resolve()
    if not value:
        return evidence_root / default_name
    result_path = Path(value)
    if not result_path.is_absolute():
        result_path = ROOT / result_path
    result_path = result_path.resolve()
    try:
        result_path.relative_to(evidence_root)
    except ValueError as exc:
        raise ValueError(
            "--output must stay under data/profitability_remediation"
        ) from exc
    return result_path


def _build_remote_command(
    *,
    decision_since: datetime,
    decision_cutoff: datetime,
    maturity_cutoff: datetime,
    row_limit: int,
    expected_decision_count: int,
    unit: str,
) -> str:
    script = _render_remote_script(
        decision_since=decision_since,
        decision_cutoff=decision_cutoff,
        maturity_cutoff=maturity_cutoff,
        row_limit=row_limit,
        expected_decision_count=expected_decision_count,
    )
    payload = base64.b64encode(script.encode("utf-8")).decode("ascii")
    runner = f"import base64;exec(base64.b64decode({payload!r}))"
    return " ".join(
        (
            "systemd-run --quiet --wait --pipe --collect",
            f"--unit={shlex.quote(unit)}",
            "--property=WorkingDirectory=/data/bb/app",
            "--property=User=bb",
            "--property=CPUQuota=25%",
            "--property=MemoryMax=1G",
            "--property=RuntimeMaxSec=300s",
            "--property=TimeoutStopSec=15s",
            "--property=KillMode=control-group",
            "--property=EnvironmentFile=-/data/bb/app/.env",
            "--property=EnvironmentFile=/etc/bb/bb-runtime.env",
            "--setenv=DATABASE_POOL_SIZE=1",
            "--setenv=DATABASE_MAX_OVERFLOW=0",
            "/data/bb/app/.venv/bin/python -c",
            shlex.quote(runner),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", required=True, help="exclusive decision-window end")
    parser.add_argument("--minutes", type=int, required=True)
    parser.add_argument("--expected-decision-count", type=int, required=True)
    parser.add_argument("--maturity-cutoff", required=True)
    parser.add_argument("--limit", type=int, default=1200)
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional evidence path. Relative paths are rooted at the project "
            "and must stay under data/profitability_remediation."
        ),
    )
    args = parser.parse_args()

    minutes = int(args.minutes)
    expected_decision_count = int(args.expected_decision_count)
    row_limit = int(args.limit)
    if not 1 <= minutes <= 10080:
        parser.error("--minutes must be between 1 and 10080")
    if not 1 <= row_limit <= 2500:
        parser.error("--limit must be between 1 and 2500")
    if not 1 <= expected_decision_count <= row_limit:
        parser.error("--expected-decision-count must be between 1 and --limit")
    decision_cutoff = _parse_cutoff(args.cutoff)
    maturity_cutoff = _parse_cutoff(args.maturity_cutoff)
    if maturity_cutoff < decision_cutoff:
        parser.error("--maturity-cutoff must be at or after --cutoff")
    minimum_maturity_cutoff = decision_cutoff + timedelta(
        minutes=max(HORIZONS),
        seconds=60,
    )
    if maturity_cutoff < minimum_maturity_cutoff:
        parser.error(
            "--maturity-cutoff must include the maximum horizon and 60-second source lag"
        )
    decision_since = decision_cutoff - timedelta(minutes=minutes)
    decision_stamp = decision_cutoff.strftime("%Y%m%dT%H%M%SZ")
    maturity_stamp = maturity_cutoff.strftime("%Y%m%dT%H%M%SZ")
    unit = f"bb-mature-shadow-cohort-d{decision_stamp}-m{maturity_stamp}"
    try:
        result_path = _resolve_evidence_path(
            args.output,
            f"online-mature-shadow-cohort-d{decision_stamp}-m{maturity_stamp}.json",
        )
    except ValueError as exc:
        parser.error(str(exc))
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {result_path}")

    expected_hashes = _local_dependency_hashes()
    ssh = connect_remote_ssh(ROOT, timeout=25)
    sftp = ssh.open_sftp()
    remote_result = (
        "/data/bb/app/tmp/codex-mature-shadow-cohort/"
        f"result_{secrets.token_hex(12)}.json"
    )
    try:
        command = (
            "set -e\numask 077\n"
            "install -d -m 0700 /data/bb/app/tmp/codex-mature-shadow-cohort\n"
            f"{_build_remote_command(decision_since=decision_since, decision_cutoff=decision_cutoff, maturity_cutoff=maturity_cutoff, row_limit=row_limit, expected_decision_count=expected_decision_count, unit=unit)}"
            f" > {shlex.quote(remote_result)}"
        )
        run_remote_text(ssh, command, timeout=320, max_output_chars=6000)
        with sftp.file(remote_result, "r") as remote_file:
            report = json.loads(remote_file.read().decode("utf-8"))
        if report.get("input", {}).get("dependency_sha256") != expected_hashes:
            raise RuntimeError("remote mature-cohort dependencies differ from local")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    finally:
        with suppress(OSError):
            sftp.remove(remote_result)
        sftp.close()
        ssh.close()
    safe_print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"saved_report={result_path}")


if __name__ == "__main__":
    main()
