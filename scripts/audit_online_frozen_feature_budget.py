"""Measure bounded frozen-feature payload sizes without reading outcome labels."""

from __future__ import annotations

import argparse
import base64
import json
import secrets
import shlex
import sys
from contextlib import suppress
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402

REMOTE_SCRIPT = r'''
import asyncio
import json
from collections import Counter
from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from core.training_feature_snapshot import (
    FROZEN_TRAINING_FEATURE_MAX_BYTES,
    compact_frozen_feature_payload,
    frozen_feature_payload_size,
    training_feature_storage_bytes,
)
from core.feature_availability import feature_availability_reasons
from db.session import get_read_session_ctx
from models.decision import AIDecision
from models.learning import ShadowBacktest

LIMIT = __LIMIT__
SINCE_MINUTES = __SINCE_MINUTES__


def frozen_payload(features):
    if not isinstance(features, dict):
        return {}
    return compact_frozen_feature_payload(features)


def nested_sizes(value, path):
    result = [(path, training_feature_storage_bytes(value))]
    if isinstance(value, dict):
        for key, child in value.items():
            result.extend(nested_sizes(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(nested_sizes(child, f"{path}[{index}]"))
    return result


def main_summary(rows, checked_at, source):
    sizes = []
    errors = Counter()
    contract_versions = Counter()
    frozen_timestamp_count = 0
    original_decision_timestamp_count = 0
    contract_present_count = 0
    availability_reason_counts = Counter()
    availability_valid_count = 0
    availability_invalid_examples = []
    availability_streak_rows = []
    availability_streak_broken = False
    key_sizes = {}
    oversized_examples = []
    for row in rows:
        features = row.feature_snapshot if isinstance(row.feature_snapshot, dict) else {}
        frozen = frozen_payload(features)
        size = frozen_feature_payload_size(features)
        sizes.append(size)
        contract = features.get("feature_availability_contract")
        seal_error = contract.get("seal_error") if isinstance(contract, dict) else None
        if isinstance(contract, dict) and contract:
            contract_present_count += 1
            contract_versions[str(contract.get("version") or "missing")] += 1
        if features.get("feature_snapshot_frozen_at") is not None:
            frozen_timestamp_count += 1
        if features.get("original_decision_timestamp") is not None:
            original_decision_timestamp_count += 1
        availability_reasons = feature_availability_reasons(
            features,
            decision_at=features.get("original_decision_timestamp"),
        )
        if availability_reasons:
            availability_reason_counts.update(availability_reasons)
            if not availability_streak_broken:
                availability_streak_broken = True
            if len(availability_invalid_examples) < 12:
                route_status = features.get("derivatives_route_status")
                route_status = route_status if isinstance(route_status, dict) else {}
                fact_sources = {}
                for key in (
                    "orderbook_fact",
                    "mark_price_fact",
                    "index_price_fact",
                ):
                    fact = features.get(key)
                    fact = fact if isinstance(fact, dict) else {}
                    fact_sources[key] = {
                        field: fact.get(field)
                        for field in (
                            "source_endpoint",
                            "source_channel",
                            "inst_id",
                            "source_timestamp_ms",
                            "received_at",
                        )
                        if fact.get(field) is not None
                    }
                availability_invalid_examples.append(
                    {
                        "id": row.id,
                        "decision_id": getattr(row, "decision_id", None),
                        "symbol": row.symbol,
                        "reasons": availability_reasons[:12],
                        "derivatives_route_status": {
                            key: list(route_status.get(key) or [])[:8]
                            for key in ("completed", "failed", "timed_out")
                        },
                        "derivatives_unavailable_reason": str(
                            features.get("derivatives_unavailable_reason") or ""
                        )[:120],
                        "fact_sources": fact_sources,
                    }
                )
        else:
            availability_valid_count += 1
            if not availability_streak_broken:
                availability_streak_rows.append(row)
        compaction_error = features.get("training_feature_compaction_error")
        error = compaction_error or seal_error
        errors[str(error or "none")] += 1
        contributions = {}
        for key, value in frozen.items():
            contribution = (
                32 + 6 * len(key.encode("utf-8"))
                + training_feature_storage_bytes(value)
            )
            contributions[key] = contribution
            key_sizes.setdefault(key, []).append(contribution)
            for path, nested_size in nested_sizes(value, key):
                key_sizes.setdefault(path, []).append(nested_size)
        if size > FROZEN_TRAINING_FEATURE_MAX_BYTES or error:
            oversized_examples.append({
                "decision_id": row.id,
                "symbol": row.symbol,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "frozen_bytes": size,
                "seal_error": seal_error,
                "training_feature_compaction_error": compaction_error,
                "largest_fields": sorted(
                    (
                        {"field": name, "bytes": amount}
                        for name, amount in contributions.items()
                    ),
                    key=lambda item: item["bytes"],
                    reverse=True,
                )[:8],
            })

    sorted_sizes = sorted(sizes)
    def percentile(ratio):
        if not sorted_sizes:
            return None
        index = min(round((len(sorted_sizes) - 1) * ratio), len(sorted_sizes) - 1)
        return sorted_sizes[index]

    field_summary = [
        {
            "field": name,
            "present_rows": len(values),
            "max_bytes": max(values),
            "p95_bytes": sorted(values)[min(round((len(values)-1)*0.95), len(values)-1)],
        }
        for name, values in key_sizes.items()
    ]
    field_summary.sort(key=lambda item: item["max_bytes"], reverse=True)
    return {
        "checked_at": checked_at.isoformat(),
        "source": source,
        "window_minutes": SINCE_MINUTES,
        "limit": LIMIT,
        "row_count": len(rows),
        "frozen_budget_bytes": FROZEN_TRAINING_FEATURE_MAX_BYTES,
        "frozen_size_bytes": {
            "minimum": min(sorted_sizes) if sorted_sizes else None,
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "maximum": max(sorted_sizes) if sorted_sizes else None,
            "over_budget_count": sum(
                size > FROZEN_TRAINING_FEATURE_MAX_BYTES for size in sizes
            ),
        },
        "seal_error_counts": dict(errors),
        "availability_contract": {
            "present_count": contract_present_count,
            "missing_count": len(rows) - contract_present_count,
            "version_counts": dict(contract_versions),
            "frozen_timestamp_count": frozen_timestamp_count,
            "original_decision_timestamp_count": original_decision_timestamp_count,
            "valid_count": availability_valid_count,
            "invalid_count": len(rows) - availability_valid_count,
            "newest_valid_streak_count": len(availability_streak_rows),
            "newest_valid_streak_distinct_symbols": len(
                {
                    str(row.symbol or "")
                    for row in availability_streak_rows
                    if str(row.symbol or "")
                }
            ),
            "newest_valid_streak_started_at": (
                availability_streak_rows[-1].created_at.isoformat()
                if availability_streak_rows
                and availability_streak_rows[-1].created_at
                else None
            ),
            "newest_valid_streak_latest_at": (
                availability_streak_rows[0].created_at.isoformat()
                if availability_streak_rows
                and availability_streak_rows[0].created_at
                else None
            ),
            "reason_counts": dict(availability_reason_counts.most_common(20)),
            "invalid_examples": availability_invalid_examples,
        },
        "largest_field_contributions": field_summary[:60],
        "oversized_examples": oversized_examples[:12],
    }


async def main():
    checked_at = datetime.now(UTC)
    since = checked_at - timedelta(minutes=SINCE_MINUTES)
    statements = {}
    if __SOURCE__ in {"decision", "both"}:
        statements["decision_snapshot"] = (
            select(
                AIDecision.id,
                    AIDecision.id.label("decision_id"),
                AIDecision.symbol,
                AIDecision.created_at,
                AIDecision.feature_snapshot.label("feature_snapshot"),
            )
            .where(
                AIDecision.is_paper.is_(True),
                AIDecision.analysis_type == "market",
                AIDecision.created_at >= since,
            )
            .order_by(AIDecision.id.desc())
            .limit(LIMIT)
        )
    if __SOURCE__ in {"training", "both"}:
        statements["training_snapshot"] = (
            select(
                ShadowBacktest.id,
                    ShadowBacktest.decision_id,
                ShadowBacktest.symbol,
                ShadowBacktest.created_at,
                ShadowBacktest.training_feature_snapshot.label("feature_snapshot"),
            )
            .where(
                ShadowBacktest.execution_mode == "paper",
                ShadowBacktest.created_at >= since,
            )
            .order_by(ShadowBacktest.id.desc())
            .limit(LIMIT)
        )
    report = {
        "checked_at": checked_at.isoformat(),
        "window_minutes": SINCE_MINUTES,
        "sources": {},
    }
    async with get_read_session_ctx(statement_timeout_ms=8000) as session:
        for source, statement in statements.items():
            rows = list((await session.execute(statement)).all())
            report["sources"][source] = main_summary(rows, checked_at, source)
    print(json.dumps(report, ensure_ascii=False, default=str))


asyncio.run(main())
'''


def _build_remote_command(limit: int, since_minutes: int, source: str) -> str:
    remote_script = (
        REMOTE_SCRIPT.replace("__LIMIT__", str(limit))
        .replace("__SINCE_MINUTES__", str(since_minutes))
        .replace("__SOURCE__", repr(source))
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--minutes", type=int, default=45)
    parser.add_argument(
        "--source",
        choices=("decision", "training", "both"),
        default="both",
        help="Snapshot column to audit without reading outcome labels.",
    )
    args = parser.parse_args()
    limit = max(1, min(int(args.limit), 64))
    minutes = max(1, min(int(args.minutes), 240))
    source = args.source
    result_path = f"/data/bb/app/tmp/codex-feature-budget/result_{secrets.token_hex(12)}.json"
    ssh = connect_remote_ssh(ROOT, timeout=25)
    sftp = ssh.open_sftp()
    try:
        command = (
            "set -e\numask 077\n"
            "install -d -m 0700 /data/bb/app/tmp/codex-feature-budget\n"
            f"{_build_remote_command(limit, minutes, source)} > {shlex.quote(result_path)}"
        )
        run_remote_text(ssh, command, timeout=150, max_output_chars=4000)
        with sftp.file(result_path, "r") as remote_file:
            report = json.loads(remote_file.read().decode("utf-8"))
        if not isinstance(report, dict):
            raise ValueError("online feature budget audit must return a JSON object")
    finally:
        with suppress(OSError):
            sftp.remove(result_path)
        sftp.close()
        ssh.close()
    safe_print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
