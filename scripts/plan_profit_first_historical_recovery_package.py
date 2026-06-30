"""Build a read-only historical recovery package for Profit-First blockers."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import redirect_stdout
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
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
from db.session import get_read_session_ctx  # noqa: E402
from models.decision import AIDecision  # noqa: E402
from models.trade import Order  # noqa: E402
from services.profit_first_historical_recovery_package import (  # noqa: E402
    HistoricalRecoveryInput,
    build_historical_recovery_package,
    target_ids_from_blocking_actions,
)
from services.profit_first_recovery_repair_plan import (  # noqa: E402
    build_profit_first_recovery_repair_plan,
)
from web_dashboard.api.system_audit import collect_system_audit_status  # noqa: E402

DEFAULT_REPORT_DIR = "profit_first_historical_recovery_packages"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _safe_report_name(timestamp: str) -> str:
    return timestamp.replace(":", "").replace("-", "").replace("+", "Z").replace(".", "_")


def _report_output_dir(value: Path | None) -> Path:
    if value is not None:
        return value
    return settings.data_dir / DEFAULT_REPORT_DIR


def write_report(report: dict[str, Any], output_dir: Path, *, indent: int | None) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = str(report.get("generated_at") or _now_iso())
    report_path = output_dir / f"profit-first-historical-recovery-package-{_safe_report_name(timestamp)}.json"
    latest_path = output_dir / "latest.json"
    artifacts = {"report_path": str(report_path), "latest_path": str(latest_path)}
    report["report_artifacts"] = artifacts
    text = json.dumps(report, ensure_ascii=False, indent=indent, sort_keys=True)
    report_path.write_text(text + "\n", encoding="utf-8")
    latest_path.write_text(text + "\n", encoding="utf-8")
    return artifacts


async def _current_blocking_actions() -> list[dict[str, Any]]:
    audit = await collect_system_audit_status(
        record_history=False,
        source="profit_first_historical_recovery_package",
    )
    cards = [card for card in audit.get("cards") or [] if isinstance(card, dict)]
    recovery_card = next(
        (card for card in cards if str(card.get("key") or "") == "profit_first_recovery_blockers"),
        {},
    )
    recovery_details = (
        recovery_card.get("details") if isinstance(recovery_card.get("details"), dict) else {}
    )
    plan = build_profit_first_recovery_repair_plan(recovery_details)
    return [item for item in plan.get("blocking_actions") or [] if isinstance(item, dict)]


async def collect_historical_recovery_package(
    *,
    entry_decision_ids: list[int] | None = None,
    exit_decision_ids: list[int] | None = None,
    order_ids: list[int] | None = None,
    exchange_order_ids: list[str] | None = None,
    use_current_blockers: bool = True,
) -> dict[str, Any]:
    blocking_actions = await _current_blocking_actions() if use_current_blockers else []
    extracted = target_ids_from_blocking_actions(blocking_actions)
    target_entry_ids = _dedupe_ints([*(entry_decision_ids or []), *extracted["entry_decision_ids"]])
    target_exit_ids = _dedupe_ints([*(exit_decision_ids or []), *extracted["exit_decision_ids"]])
    target_order_ids = _dedupe_ints([*(order_ids or []), *extracted["order_ids"]])
    target_exchange_ids = _dedupe_texts([*(exchange_order_ids or []), *extracted["exchange_order_ids"]])

    async with get_read_session_ctx() as session:
        entry_decisions = await _load_decisions(session, target_entry_ids)
        exit_decisions = await _load_decisions(session, target_exit_ids)
        orders = await _load_orders(
            session,
            order_ids=target_order_ids,
            exchange_order_ids=target_exchange_ids,
        )
    report = build_historical_recovery_package(
        HistoricalRecoveryInput(
            entry_decisions=entry_decisions,
            exit_decisions=exit_decisions,
            orders=orders,
            blocking_actions=blocking_actions,
        )
    )
    report["targets"] = {
        "entry_decision_ids": target_entry_ids,
        "exit_decision_ids": target_exit_ids,
        "order_ids": target_order_ids,
        "exchange_order_ids": target_exchange_ids,
        "use_current_blockers": bool(use_current_blockers),
    }
    report["loaded_counts"] = {
        "entry_decisions": len(entry_decisions),
        "exit_decisions": len(exit_decisions),
        "orders": len(orders),
        "blocking_actions": len(blocking_actions),
    }
    return report


async def _load_decisions(session: Any, ids: list[int]) -> list[AIDecision]:
    if not ids:
        return []
    from sqlalchemy import select

    result = await session.execute(
        select(AIDecision).where(AIDecision.id.in_(ids)).order_by(AIDecision.id.asc())
    )
    by_id = {int(row.id): row for row in result.scalars().all()}
    return [by_id[item] for item in ids if item in by_id]


async def _load_orders(
    session: Any,
    *,
    order_ids: list[int],
    exchange_order_ids: list[str],
) -> list[Order]:
    from sqlalchemy import or_, select

    clauses = []
    if order_ids:
        clauses.append(Order.id.in_(order_ids))
    if exchange_order_ids:
        clauses.append(Order.exchange_order_id.in_(exchange_order_ids))
    if not clauses:
        return []
    result = await session.execute(select(Order).where(or_(*clauses)).order_by(Order.id.asc()))
    return list(result.scalars().all())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-indent", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stdout-only", action="store_true")
    parser.add_argument("--skip-current-blockers", action="store_true")
    parser.add_argument("--entry-decision-id", action="append", type=int, default=[])
    parser.add_argument("--exit-decision-id", action="append", type=int, default=[])
    parser.add_argument("--order-id", action="append", type=int, default=[])
    parser.add_argument("--exchange-order-id", action="append", default=[])
    return parser.parse_args()


async def _main() -> int:
    args = parse_args()
    indent = None if int(args.json_indent or 0) <= 0 else int(args.json_indent)
    with redirect_stdout(sys.stderr):
        try:
            report = await collect_historical_recovery_package(
                entry_decision_ids=[int(item) for item in args.entry_decision_id or []],
                exit_decision_ids=[int(item) for item in args.exit_decision_id or []],
                order_ids=[int(item) for item in args.order_id or []],
                exchange_order_ids=[
                    str(item).strip() for item in args.exchange_order_id or [] if str(item).strip()
                ],
                use_current_blockers=not bool(args.skip_current_blockers),
            )
        except Exception as exc:
            report = {
                "report_type": "profit_first_historical_recovery_package",
                "status": "unavailable",
                "generated_at": _now_iso(),
                "dry_run": True,
                "read_only": True,
                "audit_only": True,
                "mutates_database": False,
                "starts_trading_service": False,
                "submits_orders": False,
                "changes_model_routing": False,
                "changes_live_sizing": False,
                "live_mutation": False,
                "resume_allowed_by_this_package": False,
                "error": safe_error_text(exc, limit=240),
            }
        if not args.stdout_only:
            try:
                write_report(report, _report_output_dir(args.output_dir), indent=indent)
            except Exception as exc:
                report["status"] = "unavailable"
                report["report_artifact_error"] = {
                    "code": "profit_first_historical_recovery_package_write_failed",
                    "message": safe_error_text(exc, limit=240),
                }
    print(json.dumps(_json_safe(report), ensure_ascii=False, indent=indent, sort_keys=True))
    return 0


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def _dedupe_ints(values: list[Any]) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number <= 0 or number in seen:
            continue
        seen.add(number)
        result.append(number)
    return result


def _dedupe_texts(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
