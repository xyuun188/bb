"""Seal one prospective paper decision cohort without reading shadow labels."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import secrets
import shlex
import sys
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402

DEPENDENCIES = (
    "core/feature_availability.py",
    "models/decision.py",
)
QUERY_MARGIN_HOURS = 2

REMOTE_SCRIPT = r"""
import asyncio
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select

from core.feature_availability import feature_availability_reasons
from db.session import close_db, get_read_session_ctx
from models.decision import AIDecision

DECISION_SINCE = datetime.fromisoformat(
    __DECISION_SINCE__.replace("Z", "+00:00")
).astimezone(UTC)
DECISION_CUTOFF = datetime.fromisoformat(
    __DECISION_CUTOFF__.replace("Z", "+00:00")
).astimezone(UTC)
ROW_LIMIT = __ROW_LIMIT__
QUERY_MARGIN_HOURS = 2
DEPENDENCIES = __DEPENDENCIES__


def dependency_hashes():
    return {
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
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


def _decision_availability_reasons(decision: dict[str, Any]) -> list[str]:
    features = _dict(decision.get("feature_snapshot"))
    return feature_availability_reasons(
        features,
        decision_at=features.get("original_decision_timestamp"),
    )


async def main():
    candidate_limit = ROW_LIMIT * 4
    candidate_statement = (
        select(
            AIDecision.id,
            AIDecision.symbol,
            AIDecision.created_at,
            AIDecision.feature_snapshot[
                "original_decision_timestamp"
            ].as_string().label("original_decision_timestamp"),
        )
        .where(
            AIDecision.is_paper.is_(True),
            AIDecision.analysis_type == "market",
            AIDecision.created_at >= DECISION_SINCE - timedelta(
                hours=QUERY_MARGIN_HOURS
            ),
            AIDecision.created_at < DECISION_CUTOFF + timedelta(
                hours=QUERY_MARGIN_HOURS
            ),
        )
        .order_by(AIDecision.created_at.asc(), AIDecision.id.asc())
        .limit(candidate_limit + 1)
    )
    async with get_read_session_ctx(
        statement_timeout_ms=15000,
        transaction_isolation="REPEATABLE READ READ ONLY",
    ) as session:
        candidates = [
            dict(row)
            for row in (await session.execute(candidate_statement)).mappings()
        ]
        in_window_candidates = [
            row
            for row in candidates
            if (
                (decision_at := _time(row.get("original_decision_timestamp")))
                is not None
                and DECISION_SINCE <= decision_at < DECISION_CUTOFF
            )
        ]
        decision_ids = [int(row["id"]) for row in in_window_candidates]
        if decision_ids:
            decision_statement = select(
                AIDecision.id,
                AIDecision.symbol,
                AIDecision.created_at,
                AIDecision.feature_snapshot,
            ).where(AIDecision.id.in_(decision_ids))
            decisions = [
                dict(row)
                for row in (await session.execute(decision_statement)).mappings()
            ]
        else:
            decisions = []

    try:
        decisions.sort(
            key=lambda row: (
                _time(_dict(row.get("feature_snapshot")).get(
                    "original_decision_timestamp"
                )) or datetime.min.replace(tzinfo=UTC),
                int(row.get("id") or 0),
            )
        )
        decision_ids = [int(row["id"]) for row in decisions]
        ids_sha256 = hashlib.sha256(
            ",".join(str(value) for value in decision_ids).encode("ascii")
        ).hexdigest()
        availability = Counter(
            reason
            for decision in decisions
            for reason in _decision_availability_reasons(decision)
        )
        saturated = len(candidates) > candidate_limit
        report = {
            "experiment": "bounded-prospective-paper-decision-cohort-seal",
            "status": (
                "decision_candidate_limit_saturated"
                if saturated
                else "sealed_decision_cohort"
            ),
            "checked_at": datetime.now(UTC).isoformat(),
            "window": {
                "decision_since": DECISION_SINCE.isoformat(),
                "decision_cutoff_exclusive": DECISION_CUTOFF.isoformat(),
                "row_limit": ROW_LIMIT,
                "candidate_limit": candidate_limit,
                "query_margin_hours": QUERY_MARGIN_HOURS,
            },
            "input": {
                "decision_candidate_count": len(candidates),
                "decision_candidate_in_window_count": len(
                    in_window_candidates
                ),
                "decision_snapshot_count": len(decisions),
                "decision_ids_sha256": ids_sha256,
                "candidate_limit_saturated": saturated,
                "feature_availability_reason_counts": dict(
                    availability.most_common(40)
                ),
                "dependency_sha256": dependency_hashes(),
            },
            "safety": {
                "read_only": True,
                "database_mutated": False,
                "shadow_labels_read": False,
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
    row_limit: int,
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
        .replace("__ROW_LIMIT__", str(row_limit))
        .replace("__DEPENDENCIES__", repr(DEPENDENCIES))
    )


def _build_remote_command(
    *,
    decision_since: datetime,
    decision_cutoff: datetime,
    row_limit: int,
    unit: str,
) -> str:
    script = _render_remote_script(
        decision_since=decision_since,
        decision_cutoff=decision_cutoff,
        row_limit=row_limit,
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


def _local_dependency_hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in DEPENDENCIES
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", required=True)
    parser.add_argument("--cutoff", required=True)
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

    decision_since = _parse_cutoff(args.since)
    decision_cutoff = _parse_cutoff(args.cutoff)
    if decision_since >= decision_cutoff:
        parser.error("--since must be before --cutoff")
    row_limit = int(args.limit)
    if not 1 <= row_limit <= 2500:
        parser.error("--limit must be between 1 and 2500")
    decision_stamp = decision_cutoff.strftime("%Y%m%dT%H%M%SZ")
    evidence_root = (ROOT / "data" / "profitability_remediation").resolve()
    if args.output:
        result_path = Path(args.output)
        if not result_path.is_absolute():
            result_path = ROOT / result_path
        result_path = result_path.resolve()
        try:
            result_path.relative_to(evidence_root)
        except ValueError as exc:
            parser.error("--output must stay under data/profitability_remediation")
            raise AssertionError("unreachable") from exc
    else:
        result_path = evidence_root / f"online-decision-cohort-d{decision_stamp}.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {result_path}")

    expected_hashes = _local_dependency_hashes()
    ssh = connect_remote_ssh(ROOT, timeout=25)
    sftp = ssh.open_sftp()
    remote_result = (
        "/data/bb/app/tmp/codex-decision-cohort/"
        f"result_{secrets.token_hex(12)}.json"
    )
    try:
        command = (
            "set -e\numask 077\n"
            "install -d -m 0700 /data/bb/app/tmp/codex-decision-cohort\n"
            f"{_build_remote_command(decision_since=decision_since, decision_cutoff=decision_cutoff, row_limit=row_limit, unit=f'bb-decision-cohort-d{decision_stamp}')}"
            f" > {shlex.quote(remote_result)}"
        )
        run_remote_text(ssh, command, timeout=320, max_output_chars=6000)
        with sftp.file(remote_result, "r") as remote_file:
            report = json.loads(remote_file.read().decode("utf-8"))
        if report.get("input", {}).get("dependency_sha256") != expected_hashes:
            raise RuntimeError("remote decision-cohort dependencies differ from local")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
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
