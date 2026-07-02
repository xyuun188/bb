#!/usr/bin/env python3
"""Generate a read-only daily OKX reconciliation report.

The report is intended for systemd timers or cron. It does not mutate the
database, submit orders, close positions, or apply historical repairs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.runtime_env_bootstrap import (  # noqa: E402
    drop_privileges_to_runtime_user_if_needed,
    load_runtime_env_files,
)

load_runtime_env_files(project_root=ROOT)
drop_privileges_to_runtime_user_if_needed(project_root=ROOT)

from config.settings import settings  # noqa: E402
from core.safe_output import safe_error_text  # noqa: E402
from web_dashboard.api import system_audit  # noqa: E402

DEFAULT_REPORT_DIR = "okx_daily_reconciliation_reports"
OKX_REPORT_CARD_FACTORIES = (
    ("okx_reconciliation", system_audit._okx_reconciliation_audit),
    ("okx_trade_fact_integrity", system_audit._okx_trade_fact_integrity_audit),
    ("position_price_integrity", system_audit._position_price_integrity_audit),
    ("trade_execution_contract", system_audit._trade_execution_contract_audit),
)
EXIT_BY_STATUS = {"ok": 0, "warning": 1, "critical": 2}
RUNTIME_ONLY_ENTRY_BLOCKERS = {
    "runtime_heartbeat_unavailable",
    "trading_runtime_inactive",
    "trading_runtime_heartbeat_stale",
    "okx_authoritative_sync_unhealthy",
}
TRAINING_NON_BLOCKING_UNRESOLVED_KEYS = {
    "trade_execution_contract",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for dated report files. Defaults to data/okx_daily_reconciliation_reports.",
    )
    parser.add_argument(
        "--stdout-only",
        action="store_true",
        help="Print the report without writing files.",
    )
    parser.add_argument(
        "--json-indent",
        type=int,
        default=2,
        help="JSON indentation, use 0 for compact output.",
    )
    parser.add_argument(
        "--allow-cache",
        action="store_true",
        help="Allow the short OKX reconciliation audit cache. Default clears it for a fresh dry-run.",
    )
    return parser.parse_args(argv)


async def _collect_card(key: str, factory: Any) -> dict[str, Any]:
    try:
        result = factory()
        if asyncio.iscoroutine(result):
            result = await result
        if isinstance(result, dict):
            return result
        raise TypeError(f"{key} returned {type(result).__name__}")
    except Exception as exc:
        return system_audit._audit_card(
            key,
            key,
            "warning",
            "OKX daily reconciliation card failed during dry-run report generation.",
            details={
                "error": f"{type(exc).__name__}: {str(exc)[:240]}",
                "dry_run": True,
                "mutates_database": False,
            },
        )


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def _card_details(card: dict[str, Any] | None) -> dict[str, Any]:
    return _safe_dict(_safe_dict(card).get("details"))


def _okx_integrity_has_data_issue(details: dict[str, Any]) -> bool:
    link_repair = _safe_dict(details.get("position_fact_link_repair"))
    authoritative = _safe_dict(details.get("okx_authoritative_sync"))
    runtime_gate = _safe_dict(details.get("runtime_okx_entry_gate"))
    severity_counts = _safe_dict(details.get("severity_counts"))
    authoritative_severity_counts = _safe_dict(authoritative.get("severity_counts"))
    if any(
        _safe_int(value) > 0
        for value in (
            details.get("critical_count"),
            details.get("warning_count"),
            severity_counts.get("critical"),
            severity_counts.get("warning"),
            details.get("unresolved_position_fact_link_candidate_count"),
            _unresolved_link_candidate_count(details, link_repair),
            authoritative_severity_counts.get("critical"),
            authoritative_severity_counts.get("warning"),
            authoritative.get("manual_review_count"),
            authoritative.get("repairable_count"),
        )
    ):
        return True
    authoritative_pull_failed = (
        "okx_pull_available" in authoritative
        and authoritative.get("okx_pull_available") is False
    )
    runtime_sync_healthy = (
        runtime_gate.get("entry_blocked") is False
        and runtime_gate.get("sync_status") == "ok"
        and _safe_int(runtime_gate.get("last_requires_attention_count")) == 0
    )
    return bool(authoritative_pull_failed and not runtime_sync_healthy)


def _unresolved_link_candidate_count(
    details: dict[str, Any],
    link_repair: dict[str, Any],
) -> int:
    candidate_count = _safe_int(link_repair.get("candidate_link_count"))
    if candidate_count <= 0:
        return 0
    explicit_unresolved = details.get("unresolved_position_fact_link_candidate_count")
    if explicit_unresolved is not None:
        return _safe_int(explicit_unresolved)
    covered_residual_positions = {
        _safe_int(issue.get("position_id"))
        for issue in _safe_list(details.get("issues"))
        if isinstance(issue, dict)
        and issue.get("kind") == "superseded_position_residual"
        and issue.get("severity") == "info"
        and issue.get("position_id") is not None
    }
    diagnostics = [
        item
        for item in _safe_list(link_repair.get("diagnostics"))
        if isinstance(item, dict)
    ]
    if diagnostics and all(
        _safe_int(item.get("position_id")) in covered_residual_positions
        for item in diagnostics
    ):
        return 0
    return candidate_count


def _dedupe_gate_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        key = (str(item.get("code") or ""), str(item.get("card_key") or ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _operational_gates_from_cards(
    cards: list[dict[str, Any]],
    issue_ledger: dict[str, Any],
) -> dict[str, Any]:
    cards_by_key = {str(card.get("key") or ""): card for card in cards}
    unresolved_rows = _safe_list(issue_ledger.get("unresolved"))
    entry_blockers: list[dict[str, Any]] = []
    training_blockers: list[dict[str, Any]] = []
    attention_items: list[dict[str, Any]] = []

    for row in unresolved_rows:
        if not isinstance(row, dict):
            continue
        key = str(row.get("key") or "")
        item = {
            "code": f"unresolved_{key or 'audit_card'}",
            "card_key": key,
            "status": row.get("status"),
            "summary": row.get("summary"),
            "owner_path": row.get("owner_path"),
            "requires_attention": True,
        }
        entry_blockers.append(item)
        if key not in TRAINING_NON_BLOCKING_UNRESOLVED_KEYS:
            training_blockers.append(item)
        attention_items.append(item)

    integrity_card = cards_by_key.get("okx_trade_fact_integrity")
    integrity_details = _card_details(integrity_card)
    runtime_gate = _safe_dict(integrity_details.get("runtime_okx_entry_gate"))
    data_issue = _okx_integrity_has_data_issue(integrity_details)
    if runtime_gate.get("entry_blocked") is True:
        blocker = str(runtime_gate.get("blocker") or "okx_runtime_entry_gate_blocked")
        runtime_only = blocker in RUNTIME_ONLY_ENTRY_BLOCKERS and not data_issue
        item = {
            "code": blocker,
            "card_key": "okx_trade_fact_integrity",
            "status": runtime_gate.get("status"),
            "summary": runtime_gate.get("reason"),
            "requires_attention": not runtime_only,
        }
        entry_blockers.append(item)
        if runtime_only:
            pass
        else:
            training_blockers.append(item)
            attention_items.append(item)

    if data_issue and not unresolved_rows:
        item = {
            "code": "okx_trade_fact_data_issue",
            "card_key": "okx_trade_fact_integrity",
            "status": _safe_dict(integrity_card).get("status"),
            "summary": _safe_dict(integrity_card).get("summary"),
            "requires_attention": True,
        }
        entry_blockers.append(item)
        training_blockers.append(item)
        attention_items.append(item)

    entry_blockers = _dedupe_gate_items(entry_blockers)
    training_blockers = _dedupe_gate_items(training_blockers)
    attention_items = _dedupe_gate_items(attention_items)
    can_open_new_entries = not entry_blockers
    can_refresh_training = not training_blockers
    requires_attention = bool(attention_items)
    return {
        "source": "okx_daily_reconciliation",
        "dry_run": True,
        "mutates_database": False,
        "can_open_new_entries": can_open_new_entries,
        "entry_blocked": not can_open_new_entries,
        "can_refresh_training": can_refresh_training,
        "training_blocked": not can_refresh_training,
        "requires_attention": requires_attention,
        "can_apply_repair": False,
        "can_write_database": False,
        "training_policy": "clean_training_view_only",
        "entry_blockers": entry_blockers,
        "training_blockers": training_blockers,
        "attention_items": attention_items,
        "attention_buckets": {
            "entry": len(entry_blockers),
            "training": len(training_blockers),
            "manual_review": len(attention_items),
        },
    }


def build_report(cards: list[dict[str, Any]], *, generated_at: datetime | None = None) -> dict[str, Any]:
    generated = generated_at or datetime.now(UTC)
    status = "ok"
    if any(card.get("status") == "critical" for card in cards):
        status = "critical"
    elif any(card.get("status") == "warning" for card in cards):
        status = "warning"
    issue_ledger = system_audit._issue_ledger_from_cards(cards)
    operational_gates = _operational_gates_from_cards(cards, issue_ledger)
    return _json_safe({
        "report_type": "okx_daily_reconciliation",
        "generated_at": generated.isoformat(),
        "status": status,
        "status_label": {"ok": "正常", "warning": "需关注", "critical": "异常"}.get(
            status, status
        ),
        "dry_run": True,
        "mutates_database": False,
        "live_order_mutation": False,
        "repair_apply_enabled": False,
        "training_policy": "do_not_train_dirty_or_unclassified_okx_facts",
        "requires_attention": operational_gates["requires_attention"],
        "can_open_new_entries": operational_gates["can_open_new_entries"],
        "can_refresh_training": operational_gates["can_refresh_training"],
        "summary": {
            "cards": len(cards),
            "critical": sum(1 for card in cards if card.get("status") == "critical"),
            "warning": sum(1 for card in cards if card.get("status") == "warning"),
            "ok": sum(1 for card in cards if card.get("status") == "ok"),
        },
        "issue_ledger": issue_ledger,
        "operational_gates": operational_gates,
        "cards": cards,
    })


async def collect_report(*, allow_cache: bool = False) -> dict[str, Any]:
    if not allow_cache:
        system_audit._okx_reconciliation_cache = None
        system_audit._okx_authoritative_sync_cache = None
    token = system_audit._skip_okx_daily_reconciliation_latest.set(True)
    try:
        cards = [
            await _collect_card(key, factory) for key, factory in OKX_REPORT_CARD_FACTORIES
        ]
    finally:
        system_audit._skip_okx_daily_reconciliation_latest.reset(token)
    return build_report(cards)


def _report_output_dir(value: Path | None) -> Path:
    if value is not None:
        return value
    return settings.data_dir / DEFAULT_REPORT_DIR


def write_report(report: dict[str, Any], output_dir: Path, *, indent: int | None) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = str(report.get("generated_at") or datetime.now(UTC).isoformat())
    safe_name = (
        timestamp.replace(":", "").replace("-", "").replace("+", "Z").replace(".", "_")
    )
    report_path = output_dir / f"okx-daily-reconciliation-{safe_name}.json"
    latest_path = output_dir / "latest.json"
    artifacts = {"report_path": str(report_path), "latest_path": str(latest_path)}
    report["artifacts"] = artifacts
    text = json.dumps(_json_safe(report), ensure_ascii=False, indent=indent, sort_keys=True)
    report_path.write_text(text + "\n", encoding="utf-8")
    latest_path.write_text(text + "\n", encoding="utf-8")
    return artifacts


def exit_code_for_report(report: dict[str, Any]) -> int:
    """Return a timer-friendly exit code without hiding unresolved issues."""

    status = str(report.get("status") or "warning")
    if status == "critical":
        return 2
    if status == "warning":
        ledger = report.get("issue_ledger") if isinstance(report.get("issue_ledger"), dict) else {}
        summary = ledger.get("summary") if isinstance(ledger.get("summary"), dict) else {}
        unresolved = int(summary.get("unresolved") or 0)
        critical = int(report.get("summary", {}).get("critical") or 0)
        return 1 if unresolved or critical else 0
    return EXIT_BY_STATUS.get(status, 1)


async def async_main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    indent = None if int(args.json_indent or 0) <= 0 else int(args.json_indent)
    report = await collect_report(allow_cache=bool(args.allow_cache))
    if not args.stdout_only:
        try:
            report["artifacts"] = write_report(
                report,
                _report_output_dir(args.output_dir),
                indent=indent,
            )
        except Exception as exc:
            report["status"] = "critical"
            report["status_label"] = "异常"
            report["artifact_error"] = {
                "code": "artifact_write_failed",
                "message": safe_error_text(exc, limit=240),
                "output_dir": str(_report_output_dir(args.output_dir)),
            }
            report["summary"]["critical"] = int(report["summary"].get("critical") or 0) + 1
    print(json.dumps(_json_safe(report), ensure_ascii=False, indent=indent, sort_keys=True))
    return exit_code_for_report(report)


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
