"""Bounded read-only audit of authoritative trade training eligibility."""

from __future__ import annotations

import argparse
import base64
import inspect
import json
import secrets
import shlex
import sys
from collections import Counter
from contextlib import suppress
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.remote_ssh import connect_remote_ssh, run_remote_text  # noqa: E402
from core.safe_output import safe_print  # noqa: E402


def _compact_record(sample: dict) -> dict:
    supervision = sample.get("profit_supervision")
    supervision = supervision if isinstance(supervision, dict) else {}
    tasks = supervision.get("tasks")
    tasks = tasks if isinstance(tasks, dict) else {}
    cost = tasks.get("execution_cost_and_slippage_distribution")
    cost = cost if isinstance(cost, dict) else {}
    features = sample.get("features")
    features = features if isinstance(features, dict) else {}
    availability = features.get("feature_availability_contract")
    availability = availability if isinstance(availability, dict) else {}
    return {
        "outcome_id": sample.get("outcome_id"),
        "lifecycle_key": sample.get("lifecycle_key"),
        "decision_id": sample.get("decision_id"),
        "symbol": sample.get("symbol"),
        "side": sample.get("side"),
        "training_status": sample.get("data_quality_status"),
        "quality_reasons": list(sample.get("quality_reasons") or [])[:12],
        "outcome_evidence_gaps": list(sample.get("outcome_evidence_gaps") or [])[:12],
        "profit_supervision_version": supervision.get("version"),
        "cost_eligible": cost.get("eligible") is True,
        "cost_source_authority": cost.get("source_authority"),
        "cost_reason": cost.get("reason"),
        "fee_pct_present": cost.get("fee_pct") is not None,
        "slippage_pct_present": cost.get("slippage_pct") is not None,
        "funding_pct_present": cost.get("funding_return_pct") is not None,
        "slippage_source": cost.get("slippage_source"),
        "feature_contract_version": availability.get("version"),
        "feature_contract_sealed": bool(availability),
    }


def _summarize_records(records: list[dict]) -> dict:
    cost_reason_counts = Counter()
    cost_source_counts = Counter()
    supervision_version_counts = Counter()
    feature_contract_counts = Counter()
    quality_reason_counts = Counter()
    evidence_gap_counts = Counter()
    for row in records:
        cost_reason_counts[str(row.get("cost_reason") or "none")] += 1
        cost_source_counts[str(row.get("cost_source_authority") or "missing")] += 1
        supervision_version_counts[
            str(row.get("profit_supervision_version") or "missing")
        ] += 1
        feature_contract_counts[
            str(row.get("feature_contract_version") or "missing")
        ] += 1
        quality_reason_counts.update(row.get("quality_reasons") or [])
        evidence_gap_counts.update(row.get("outcome_evidence_gaps") or [])
    return {
        "cost_reason_counts": dict(cost_reason_counts.most_common(20)),
        "cost_source_counts": dict(cost_source_counts.most_common(20)),
        "supervision_version_counts": dict(supervision_version_counts.most_common(20)),
        "feature_contract_version_counts": dict(feature_contract_counts.most_common(20)),
        "quality_reason_counts": dict(quality_reason_counts.most_common(20)),
        "outcome_evidence_gap_counts": dict(evidence_gap_counts.most_common(20)),
    }


REMOTE_SCRIPT = r'''
import asyncio
import json
from collections import Counter
from datetime import UTC, datetime

from services.authoritative_trade_outcome import load_authoritative_trade_outcomes
from services.training_data_quality import (
    _authoritative_outcome_manifest,
    annotate_samples,
    quality_report,
    trainable_samples,
)
from services.training_epoch import load_training_data_start

LIMIT = __LIMIT__
MODE = __MODE__


__HELPERS__


async def main():
    since = load_training_data_start()
    outcomes = await load_authoritative_trade_outcomes(
        mode=MODE,
        since=since,
        limit=LIMIT,
        compact=True,
        include_training_features=True,
        include_decision_evidence=True,
    )
    annotated = annotate_samples(outcomes, "trade")
    report = quality_report({
        "shadow": [],
        "trade": annotated,
        "sequence": [],
        "text_sentiment": [],
    })
    manifest = _authoritative_outcome_manifest(annotated)
    trainable = trainable_samples(annotated)
    records = [_compact_record(sample) for sample in annotated]
    print(json.dumps({
        "checked_at": datetime.now(UTC).isoformat(),
        "training_data_started_at": since.isoformat(),
        "mode": MODE,
        "limit": LIMIT,
        "outcome_count": len(outcomes),
        "trainable_count": len(trainable),
        "manifest": {
            key: value for key, value in manifest.items() if key != "records"
        },
        "quality_totals": report.get("totals"),
        "quality_top_reasons": report.get("top_reasons"),
        **_summarize_records(records),
        "sample_records": records[:12],
    }, ensure_ascii=False, default=str))


asyncio.run(main())
'''


def _build_remote_command(limit: int, mode: str) -> str:
    helpers = "\n".join(
        (
            inspect.getsource(_compact_record),
            inspect.getsource(_summarize_records),
        )
    )
    remote_script = (
        REMOTE_SCRIPT.replace("__LIMIT__", str(limit))
        .replace("__MODE__", repr(mode))
        .replace("__HELPERS__", helpers)
    )
    payload = base64.b64encode(remote_script.encode("utf-8")).decode("ascii")
    runner = f"import base64;exec(base64.b64decode({payload!r}))"
    return " ".join(
        (
            "systemd-run --quiet --wait --pipe --collect",
            "--property=WorkingDirectory=/data/bb/app",
            "--property=User=bb",
            "--property=CPUQuota=100%",
            "--property=MemoryMax=1G",
            "--property=RuntimeMaxSec=240s",
            "--property=EnvironmentFile=-/data/bb/app/.env",
            "--property=EnvironmentFile=/etc/bb/bb-runtime.env",
            "/data/bb/app/.venv/bin/python -c",
            shlex.quote(runner),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--mode", choices=("paper", "live"), default="paper")
    args = parser.parse_args()
    limit = max(1, min(int(args.limit), 500))
    mode = str(args.mode)
    result_path = (
        f"/data/bb/app/tmp/codex-training-eligibility/result_{secrets.token_hex(12)}.json"
    )
    ssh = connect_remote_ssh(ROOT, timeout=25)
    sftp = ssh.open_sftp()
    try:
        command = (
            "set -e\numask 077\n"
            "install -d -m 0700 /data/bb/app/tmp/codex-training-eligibility\n"
            f"{_build_remote_command(limit, mode)} > {shlex.quote(result_path)}"
        )
        run_remote_text(ssh, command, timeout=240, max_output_chars=4000)
        with sftp.file(result_path, "r") as remote_file:
            report = json.loads(remote_file.read().decode("utf-8"))
        if not isinstance(report, dict):
            raise ValueError("online training eligibility audit must return an object")
    finally:
        with suppress(OSError):
            sftp.remove(result_path)
        sftp.close()
        ssh.close()
    safe_print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
