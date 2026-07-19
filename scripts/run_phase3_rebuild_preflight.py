"""Build a read-only Phase 3 model rebuild preflight report.

The script does not train, write artifacts, mutate the database, or start
trading.  It only gathers clean training-view counts, audit gates, and the
commands an operator may run later after the readiness gate is clear.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.runtime_env_bootstrap import (
    drop_privileges_to_runtime_user_if_needed,
    load_runtime_env_files,
)

load_runtime_env_files(project_root=ROOT)
drop_privileges_to_runtime_user_if_needed(project_root=ROOT)

from config.settings import settings
from core.safe_output import safe_error_text
from services.artifact_retirement_audit import ArtifactRetirementAuditService
from services.historical_trade_fact_audit import HistoricalTradeFactAuditService
from services.model_promotion_policy import (
    build_phase3_promotion_recommendation,
    build_return_objective_report,
    load_latest_paper_observation_report,
)
from services.phase3_rebuild_readiness import Phase3RebuildReadinessService
from services.trading_params import DEFAULT_TRADING_PARAMS
from services.training_data_quality import annotate_training_payload

_LOCAL_ML_TRAINING_PARAMS = DEFAULT_TRADING_PARAMS.local_ml_training
DEFAULT_HISTORICAL_AUDIT_DAYS = 180
DEFAULT_HISTORICAL_AUDIT_LIMIT = 5000
DEFAULT_REPORT_DIR = "phase3_rebuild_preflight_reports"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _command_list(command: str) -> list[str]:
    return [part for part in command.split(" ") if part]


def _local_ai_base_arg(base_url: str) -> str:
    value = str(base_url or "").strip()
    return f" --base-url {value}" if value else ""


def _build_rebuild_commands(*, local_ai_tools_base_url: str = "") -> dict[str, Any]:
    local_ai_base = _local_ai_base_arg(local_ai_tools_base_url)
    commands = {
        "ml_signal": {
            "preflight_command": "python scripts/train_ml_signal_model.py",
            "confirmed_rebuild_command": (
                "python scripts/train_ml_signal_model.py "
                "--persist-artifact --confirm-phase3-rebuild"
            ),
        },
        "local_ai_tools": {
            "preflight_command": f"python scripts/train_local_ai_tools_models.py{local_ai_base}",
            "confirmed_rebuild_command": (
                "python scripts/train_local_ai_tools_models.py"
                f"{local_ai_base} --persist-artifact --confirm-phase3-rebuild"
            ),
        },
    }
    for item in commands.values():
        item["preflight_argv"] = _command_list(item["preflight_command"])
        item["confirmed_rebuild_argv"] = _command_list(item["confirmed_rebuild_command"])
    return commands


def _safe_report_name(timestamp: str) -> str:
    return timestamp.replace(":", "").replace("-", "").replace("+", "Z").replace(".", "_")


def _report_output_dir(value: Path | None) -> Path:
    if value is not None:
        return value
    return settings.data_dir / DEFAULT_REPORT_DIR


def write_report(report: dict[str, Any], output_dir: Path, *, indent: int | None) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = str(report.get("checked_at") or _now_iso())
    report_path = output_dir / f"phase3-rebuild-preflight-{_safe_report_name(timestamp)}.json"
    latest_path = output_dir / "latest.json"
    artifacts = {"report_path": str(report_path), "latest_path": str(latest_path)}
    report["report_artifacts"] = artifacts
    text = json.dumps(report, ensure_ascii=False, indent=indent, sort_keys=True)
    report_path.write_text(text + "\n", encoding="utf-8")
    latest_path.write_text(text + "\n", encoding="utf-8")
    return artifacts


def _empty_quality_report() -> dict[str, Any]:
    return {
        "data_quality_version": "unavailable",
        "totals": {
            "total": 0,
            "included": 0,
            "downweighted": 0,
            "excluded": 0,
            "effective_weight_ratio": 0.0,
        },
    }


def _training_payload_unavailable(error: Exception) -> dict[str, Any]:
    quality_report = _empty_quality_report()
    governance_report = {
        "status": "unavailable",
        "contamination_risk": "unknown",
        "trainable_sample_count": 0,
        "excluded_sample_count": 0,
        "cleanup_mode": "quarantine_not_delete",
        "training_policy": "clean_training_view_only",
        "error": safe_error_text(error, limit=180),
    }
    return {
        "payload": {
            "shadow_samples": [],
            "trade_samples": [],
            "sequence_samples": [],
            "text_sentiment_samples": [],
            "quality_report": quality_report,
            "governance_report": governance_report,
        },
        "completed_shadow_sample_count": 0,
        "completed_trade_sample_count": 0,
        "raw_shadow_sample_count": 0,
        "trainable_shadow_sample_count": 0,
        "raw_trade_sample_count": 0,
        "trainable_trade_sample_count": 0,
        "quarantined_trade_sample_count": 0,
        "sequence_sample_count": 0,
        "text_sentiment_sample_count": 0,
        "collection_error": safe_error_text(error, limit=180),
    }


async def _collect_training_payload() -> dict[str, Any]:
    from scripts.train_local_ai_tools_models import (
        _completed_shadow_sample_count,
        _completed_trade_sample_count,
        _load_authoritative_trade_samples,
        _load_sequence_samples,
        _load_shadow_samples,
        _load_text_sentiment_samples,
        _load_trade_reflection_samples,
        _merge_trade_samples,
    )

    shadow_samples = await _load_shadow_samples()
    trade_reflection_samples = await _load_trade_reflection_samples()
    authoritative_samples = await _load_authoritative_trade_samples()
    trade_samples = _merge_trade_samples(trade_reflection_samples, authoritative_samples)
    sequence_samples = await _load_sequence_samples()
    text_sentiment_samples = await _load_text_sentiment_samples()
    payload = annotate_training_payload(
        shadow_samples=shadow_samples,
        trade_samples=trade_samples,
        sequence_samples=sequence_samples,
        text_sentiment_samples=text_sentiment_samples,
    )
    completed_shadow_count = await _completed_shadow_sample_count()
    completed_trade_count = await _completed_trade_sample_count()
    raw_trade_sample_count = len(trade_samples)
    trainable_trade_sample_count = len(payload["trade_samples"])
    return {
        "payload": payload,
        "completed_shadow_sample_count": int(completed_shadow_count),
        "completed_trade_sample_count": int(completed_trade_count),
        "raw_shadow_sample_count": len(shadow_samples),
        "trainable_shadow_sample_count": len(payload["shadow_samples"]),
        "raw_trade_sample_count": raw_trade_sample_count,
        "trainable_trade_sample_count": trainable_trade_sample_count,
        "quarantined_trade_sample_count": max(raw_trade_sample_count - trainable_trade_sample_count, 0),
        "sequence_sample_count": len(payload["sequence_samples"]),
        "text_sentiment_sample_count": len(payload["text_sentiment_samples"]),
    }


async def _historical_trade_fact_report(*, days: int, limit: int) -> dict[str, Any]:
    return await HistoricalTradeFactAuditService(lookback_days=days, limit=limit).report()


async def _artifact_retirement_report() -> dict[str, Any]:
    return await ArtifactRetirementAuditService().report()


async def _runtime_probe_report(*, include_runtime_probe: bool) -> dict[str, Any]:
    if not include_runtime_probe:
        return {
            "status": "skipped",
            "reason": "include_runtime_probe_not_requested",
        }
    from services.server_monitor_status import collect_platform_runtime_status

    try:
        runtime = await collect_platform_runtime_status()
    except Exception as exc:
        return {
            "status": "warning",
            "error": safe_error_text(exc, limit=180),
        }
    if not isinstance(runtime, dict):
        return {"status": "warning", "error": "runtime probe returned non-object payload"}
    model_rows = runtime.get("ai_models") if isinstance(runtime.get("ai_models"), list) else []
    local_tools = (
        runtime.get("local_ai_tools") if isinstance(runtime.get("local_ai_tools"), dict) else {}
    )
    unavailable = [
        row
        for row in model_rows
        if isinstance(row, dict) and not bool(row.get("available"))
    ]
    if local_tools and bool(local_tools.get("configured")) and not bool(local_tools.get("available")):
        unavailable.append({"model": "local_ai_tools", "status": local_tools.get("status")})
    return {
        "status": "critical" if unavailable else "ok",
        "unavailable_count": len(unavailable),
        "unavailable_samples": unavailable[:8],
        "ai_model_count": len(model_rows),
        "local_ai_tools_configured": bool(local_tools.get("configured")) if local_tools else False,
    }


async def collect_phase3_rebuild_preflight(
    *,
    historical_audit_days: int = DEFAULT_HISTORICAL_AUDIT_DAYS,
    historical_audit_limit: int = DEFAULT_HISTORICAL_AUDIT_LIMIT,
    include_runtime_probe: bool = False,
    requested_persist_artifact: bool = False,
    confirm_phase3_rebuild: bool = False,
    local_ai_tools_base_url: str = "",
) -> dict[str, Any]:
    started_at = datetime.now(UTC)
    collection_errors: dict[str, str] = {}
    try:
        training = await _collect_training_payload()
    except Exception as exc:
        training = _training_payload_unavailable(exc)
        collection_errors["training_payload"] = training["collection_error"]
    payload = training["payload"]
    evaluation_policy = {
        "promotion_flow": "candidate_to_shadow_to_canary_to_active",
        "live_mutation": False,
        "requires_walk_forward": True,
        "phase": "phase3_model_factory",
    }
    paper_observation_report = load_latest_paper_observation_report(root=ROOT)
    return_objective_report = build_return_objective_report(
        trade_samples=payload["trade_samples"],
        shadow_samples=payload["shadow_samples"],
    )
    promotion = build_phase3_promotion_recommendation(
        training_mode="shadow",
        model_stage="shadow",
        quality_report=payload["quality_report"],
        governance_report=payload["governance_report"],
        evaluation_policy=evaluation_policy,
        paper_observation_report=paper_observation_report,
        completed_shadow_sample_count=training["completed_shadow_sample_count"],
        completed_trade_sample_count=training["completed_trade_sample_count"],
        return_objective_report=return_objective_report,
    )
    local_ai_tools = {
        "available": True,
        "status": "preflight_ready",
        "shadow_sample_count": training["trainable_shadow_sample_count"],
        "trade_sample_count": training["trainable_trade_sample_count"],
        "trainable_trade_sample_count": training["trainable_trade_sample_count"],
        "raw_trade_sample_count": training["raw_trade_sample_count"],
        "quarantined_trade_sample_count": training["quarantined_trade_sample_count"],
        "sequence_sample_count": training["sequence_sample_count"],
        "text_sentiment_sample_count": training["text_sentiment_sample_count"],
        "quality_report": payload["quality_report"],
        "governance_report": payload["governance_report"],
        "return_objective_report": return_objective_report,
        "training_mode": "shadow",
        "model_stage": "shadow",
        "promotion_flow": "candidate_to_shadow_to_canary_to_active",
        "live_mutation": False,
        "evaluation_policy": evaluation_policy,
        "promotion_recommendation": promotion,
    }
    try:
        historical_report = await _historical_trade_fact_report(
            days=historical_audit_days,
            limit=historical_audit_limit,
        )
    except Exception as exc:
        collection_errors["historical_trade_fact_audit"] = safe_error_text(exc, limit=180)
        historical_report = {
            "status": "unavailable",
            "read_only": True,
            "audit_only": True,
            "training_policy": "clean_training_view_only",
            "cleanup_mode": "quarantine_not_delete",
            "trainable_closed_positions": 0,
            "quarantined_closed_positions": 0,
            "error": collection_errors["historical_trade_fact_audit"],
        }
    try:
        artifact_report = await _artifact_retirement_report()
    except Exception as exc:
        collection_errors["artifact_retirement_audit"] = safe_error_text(exc, limit=180)
        artifact_report = {
            "status": "unavailable",
            "read_only": True,
            "audit_only": True,
            "retired_or_untrusted_count": 0,
            "can_delete_artifacts": False,
            "error": collection_errors["artifact_retirement_audit"],
        }
    runtime_probe = await _runtime_probe_report(include_runtime_probe=include_runtime_probe)
    readiness = Phase3RebuildReadinessService().report(
        local_ai_tools=local_ai_tools,
        governance=payload["governance_report"],
        historical_trade_fact_audit=historical_report,
        artifact_retirement_audit=artifact_report,
        runtime_probe=runtime_probe,
        requested_persist_artifact=requested_persist_artifact,
        confirm_phase3_rebuild=confirm_phase3_rebuild,
    )
    commands = _build_rebuild_commands(
        local_ai_tools_base_url=local_ai_tools_base_url or settings.local_ai_tools_api_base
    )
    return {
        "status": readiness["status"],
        "phase": "phase3_model_factory",
        "read_only": True,
        "mutates_database": False,
        "writes_artifacts": False,
        "starts_trading_service": False,
        "requested_persist_artifact": bool(requested_persist_artifact),
        "confirm_phase3_rebuild": bool(confirm_phase3_rebuild),
        "readiness": readiness,
        "collection_errors": collection_errors,
        "training_summary": {
            key: value
            for key, value in training.items()
            if key != "payload"
        },
        "quality_report": payload["quality_report"],
        "governance_report": payload["governance_report"],
        "return_objective_report": return_objective_report,
        "promotion_recommendation": promotion,
        "paper_observation_report": paper_observation_report,
        "historical_trade_fact_audit": historical_report,
        "artifact_retirement_audit": artifact_report,
        "runtime_probe": runtime_probe,
        "commands": commands,
        "operator_sequence": [
            "Review this report and clear readiness.blockers first.",
            "Run preflight commands and compare the metadata output.",
            "Only after readiness is clear, run confirmed rebuild commands with operator approval.",
            "Keep rebuilt artifacts in shadow; canary/live promotion remains a later gate.",
        ],
        "checked_at": _now_iso(),
        "duration_seconds": round((datetime.now(UTC) - started_at).total_seconds(), 6),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-audit-days", type=int, default=DEFAULT_HISTORICAL_AUDIT_DAYS)
    parser.add_argument("--historical-audit-limit", type=int, default=DEFAULT_HISTORICAL_AUDIT_LIMIT)
    parser.add_argument("--include-runtime-probe", action="store_true")
    parser.add_argument("--local-ai-tools-base-url", default=settings.local_ai_tools_api_base)
    parser.add_argument("--persist-artifact", action="store_true")
    parser.add_argument("--confirm-phase3-rebuild", action="store_true")
    parser.add_argument("--json-indent", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for dated report files. Defaults to data/phase3_rebuild_preflight_reports.",
    )
    parser.add_argument(
        "--stdout-only",
        action="store_true",
        help="Print the report without writing report artifacts.",
    )
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Return exit code 2 when the readiness gate is blocked.",
    )
    return parser.parse_args()


async def _main() -> int:
    args = parse_args()
    report = await collect_phase3_rebuild_preflight(
        historical_audit_days=args.historical_audit_days,
        historical_audit_limit=args.historical_audit_limit,
        include_runtime_probe=bool(args.include_runtime_probe),
        requested_persist_artifact=bool(args.persist_artifact),
        confirm_phase3_rebuild=bool(args.confirm_phase3_rebuild),
        local_ai_tools_base_url=args.local_ai_tools_base_url,
    )
    indent = None if int(args.json_indent or 0) <= 0 else int(args.json_indent)
    if not args.stdout_only:
        try:
            write_report(report, _report_output_dir(args.output_dir), indent=indent)
        except Exception as exc:
            collection_errors = report.setdefault("collection_errors", {})
            if isinstance(collection_errors, dict):
                collection_errors["report_artifact_write"] = safe_error_text(exc, limit=180)
            report["status"] = "blocked"
            report["report_artifact_error"] = {
                "code": "report_artifact_write_failed",
                "message": safe_error_text(exc, limit=240),
                "output_dir": str(_report_output_dir(args.output_dir)),
            }
    print(json.dumps(report, ensure_ascii=False, indent=indent, sort_keys=True))
    if args.fail_on_blocked and report["readiness"]["status"] == "blocked":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
