"""Bounded read-only lifecycle economics and independent account-bill audit."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import shlex
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db.session import close_db, get_read_session_ctx  # noqa: E402
from services.authoritative_trade_outcome import load_authoritative_trade_outcomes  # noqa: E402
from services.okx_native_facts import (  # noqa: E402
    _account_bill_from_row,
    _oldest_account_bill_pagination_cursor,
)


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _utc(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("audit cutoff must have an explicit timezone")
    return result.astimezone(UTC)


def summarize_lifecycles(outcomes: list[dict[str, Any]], *, since: datetime) -> dict[str, Any]:
    """Summarize canonical facts; do not invent a second settlement formula."""
    unique: dict[str, dict[str, Any]] = {}
    duplicates = 0
    for row in outcomes:
        key = str(row.get("lifecycle_key") or "")
        if not key or row.get("event_type") != "AuthoritativeTradeOutcome":
            raise ValueError("canonical lifecycle identity is required")
        if key in unique:
            if unique[key] != row:
                raise ValueError("conflicting duplicate lifecycle")
            duplicates += 1
        unique[key] = row
    rows = list(unique.values())
    full = [row for row in rows if row.get("close_status") == "full"]
    partial_rows = [row for row in rows if row.get("close_status") != "full"]
    fields = {
        "gross_pnl_usdt": "gross_pnl_usdt",
        "fees_signed_usdt": "official_fee_signed_usdt",
        "funding_signed_usdt": "funding_fee_usdt",
        "liquidation_penalty_signed_usdt": "liquidation_penalty_usdt",
        "realized_net_pnl_usdt": "reported_realized_net_pnl_usdt",
    }
    valid = []
    for row in full:
        components = row.get("realized_net_pnl_components") or {}
        if components.get("formula_consistent") is True and all(
            _finite(components.get(key)) is not None for key in fields.values()
        ):
            valid.append(row)
    totals = {
        key: math.fsum(row["realized_net_pnl_components"][source] for row in valid)
        for key, source in fields.items()
    }
    gap_counts = Counter(
        str(gap) for row in rows for gap in set(row.get("outcome_evidence_gaps") or [])
    )
    duration_ratios = []
    for row in valid:
        horizon = _finite((row.get("normal_paper_trade_evidence") or {}).get(
            "prediction_horizon_minutes"
        ))
        holding = _finite(row.get("holding_minutes"))
        if horizon is not None and horizon > 0 and holding is not None and holding >= 0:
            duration_ratios.append(holding / horizon)
    slippage = [
        value for row in valid
        if (value := _finite(row.get("execution_slippage_usdt"))) is not None
    ]
    return {
        "source": "services.authoritative_trade_outcome",
        "unique_lifecycle_count": len(rows),
        "identical_duplicate_count": duplicates,
        "fully_closed_count": len(full),
        "partial_or_unknown_count": len(partial_rows),
        "settlement_consistent_full_count": len(valid),
        "full_excluded_from_totals_count": len(full) - len(valid),
        "totals_are_lifecycle_pnl_not_window_or_portfolio_return": True,
        "full_lifecycles_opened_before_window_count": sum(
            bool(row.get("decision_timestamp"))
            and _utc(row["decision_timestamp"]) < since for row in valid
        ),
        "full_lifecycle_totals": totals,
        "profitable_full_count": sum(
            row["realized_net_pnl_components"]["reported_realized_net_pnl_usdt"] > 0
            for row in valid
        ),
        "losing_full_count": sum(
            row["realized_net_pnl_components"]["reported_realized_net_pnl_usdt"] < 0
            for row in valid
        ),
        "positive_gross_but_negative_net_count": sum(
            row["realized_net_pnl_components"]["gross_pnl_usdt"] > 0
            and row["realized_net_pnl_components"]["reported_realized_net_pnl_usdt"] < 0
            for row in valid
        ),
        "complete_training_evidence_count": sum(
            row.get("outcome_complete") is True for row in full
        ),
        "gap_counts": dict(gap_counts.most_common()),
        "measured_slippage": {
            "complete_lifecycle_count": len(slippage),
            "signed_execution_slippage_usdt": math.fsum(slippage),
            "already_embedded_in_fill_price_pnl": True,
            "subtracted_again_from_realized_pnl": False,
        },
        "holding_vs_forecast_horizon": {
            "comparable_count": len(duration_ratios),
            "missing_horizon_or_duration_count": len(valid) - len(duration_ratios),
            "holding_exceeds_forecast_count": sum(value > 1 for value in duration_ratios),
            "maximum_ratio": max(duration_ratios, default=None),
            "causal_exit_loss_inferred": False,
        },
        "source_fingerprint": _fingerprint(sorted(
            (row["lifecycle_key"], row.get("outcome_fingerprint")) for row in rows
        )),
    }


def summarize_bills(
    raw_rows: list[dict[str, Any]], *, outcomes: list[dict[str, Any]]
) -> dict[str, Any]:
    order_ids = {
        str(order_id)
        for row in outcomes
        for order_id in [*(row.get("entry_order_ids") or []), *(row.get("close_order_ids") or [])]
    }
    seen: dict[str, dict[str, Any]] = {}
    for row in raw_rows:
        key = str(row.get("billId") or "")
        if not key:
            raise ValueError("official bill identity is required")
        if key in seen and seen[key] != row:
            raise ValueError("conflicting duplicate bill")
        seen[key] = row
    buckets: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    matched = 0
    excluded_currency = 0
    for row in seen.values():
        if str(row.get("ccy") or "").upper() != "USDT":
            excluded_currency += 1
            continue
        bill = _account_bill_from_row(row)
        if bill is None:
            raise ValueError("invalid native bill")
        funding = bill.bill_sub_type in {"173", "174"}
        kind = "funding" if funding else "trade" if bill.bill_type == "2" else "other"
        item = buckets[kind]
        item["count"] += 1
        item["cash_change_usdt"] += bill.balance_change
        item["price_pnl_usdt"] += 0.0 if funding else bill.pnl
        item["fees_signed_usdt"] += bill.fee
        item["funding_signed_usdt"] += bill.funding_fee
        if str(row.get("ordId") or "") in order_ids:
            matched += 1
    for item in buckets.values():
        item["unexplained_cash_residual_usdt"] = (
            item["cash_change_usdt"] - item["price_pnl_usdt"]
            - item["fees_signed_usdt"] - item["funding_signed_usdt"]
        )
    return {
        "source": "okx_account_bills_with_canonical_native_normalization",
        "unique_bill_count": len(seen),
        "identical_duplicate_count": len(raw_rows) - len(seen),
        "excluded_non_usdt_count": excluded_currency,
        "by_kind": dict(buckets),
        "trade_and_funding_cash_change_usdt": math.fsum(
            buckets[kind].get("cash_change_usdt", 0.0) for kind in ("trade", "funding")
        ),
        "bills_with_order_in_sampled_lifecycles": matched,
        "not_an_equity_or_completed_round_trip_return": True,
        "source_fingerprint": _fingerprint(sorted(seen.items())),
    }


async def fetch_bills(
    exchange: Any, *, since: datetime, until: datetime, max_pages: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read only; make pagination exhaustion explicit instead of assuming coverage."""
    rows = []
    cursor = ""
    cursors: set[str] = set()
    since_ms, until_ms = int(since.timestamp() * 1000), int(until.timestamp() * 1000)
    for page_number in range(1, max_pages + 1):
        params = {"instType": "SWAP", "ccy": "USDT", "limit": "100"}
        if cursor:
            params["after"] = cursor
        response = await exchange.privateGetAccountBills(params)
        if not isinstance(response, dict) or str(response.get("code")) != "0":
            raise ValueError("official bill request failed")
        page = response.get("data")
        if not isinstance(page, list):
            raise ValueError("invalid official bill page")
        timestamps = [int(row["ts"]) for row in page]
        if any(value <= 0 for value in timestamps):
            raise ValueError("invalid official bill timestamp")
        rows.extend(
            row for row, timestamp in zip(page, timestamps, strict=True)
            if since_ms <= timestamp < until_ms
        )
        reason = (
            "window_boundary_crossed" if timestamps and min(timestamps) < since_ms
            else "end_of_history" if len(page) < 100 else ""
        )
        if reason:
            return rows, {"complete": True, "pages": page_number, "reason": reason}
        cursor = _oldest_account_bill_pagination_cursor(page)
        if not cursor or cursor in cursors:
            return rows, {"complete": False, "pages": page_number, "reason": "cursor_stalled"}
        cursors.add(cursor)
    return rows, {"complete": False, "pages": max_pages, "reason": "page_budget_exhausted"}


async def audit(*, until: datetime, hours: int, limit: int, bill_pages: int) -> dict[str, Any]:
    from executor.okx_executor import OKXExecutor

    since = until - timedelta(hours=hours)
    outcomes = await load_authoritative_trade_outcomes(
        mode="paper", since=since, limit=limit + 1, compact=True,
        session_factory=partial(
            get_read_session_ctx, statement_timeout_ms=15_000,
            transaction_isolation="REPEATABLE READ READ ONLY",
        ),
    )
    canonical_rows_loaded = len(outcomes)
    lifecycle_limit_reached = canonical_rows_loaded > limit
    # The canonical loader has a lower boundary only. Enforce our frozen upper boundary.
    outcomes = [
        row for row in outcomes
        if row.get("label_timestamp") and since <= _utc(row["label_timestamp"]) < until
    ]
    outcomes = outcomes[-limit:]
    executor = OKXExecutor(mode="paper", load_markets_on_initialize=False)
    try:
        exchange = await executor._get_ccxt()
        bills, coverage = await fetch_bills(
            exchange, since=since, until=until, max_pages=bill_pages
        )
    finally:
        await executor.shutdown()
    return {
        "audit_version": "profitability-economics.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "window_start_utc": since.isoformat(),
        "window_end_utc_exclusive": until.isoformat(),
        "read_only": True,
        "production_permission": False,
        "profitability_acceptance": "not_evaluated",
        "lifecycle_limit": limit,
        "canonical_rows_loaded_before_cutoff_filter": canonical_rows_loaded,
        "lifecycle_limit_reached": lifecycle_limit_reached,
        "lifecycle_window_complete": not lifecycle_limit_reached,
        "lifecycles": summarize_lifecycles(outcomes, since=since),
        "bill_coverage": coverage,
        "bills": summarize_bills(bills, outcomes=outcomes),
    }


def audit_online(args: argparse.Namespace) -> dict[str, Any]:
    from core.remote_ssh import connect_remote_ssh, run_remote_text

    identity = args.until.strftime("%Y%m%dT%H%M%SZ")
    remote_output = f"/data/bb/app/data/profitability_remediation/economics-{identity}.json"
    command = [
        "systemd-run", "--quiet", "--wait", "--pipe", "--collect",
        f"--unit=bb-profitability-economics-{identity}",
        "--property=WorkingDirectory=/data/bb/app", "--property=User=bb",
        "--property=EnvironmentFile=-/data/bb/app/.env",
        "--property=EnvironmentFile=/etc/bb/bb-runtime.env",
        "--property=CPUQuota=100%", "--property=MemoryMax=1G",
        "--property=RuntimeMaxSec=240", "--property=TimeoutStopSec=10",
        "--property=KillMode=control-group",
        "--setenv=DATABASE_POOL_SIZE=1", "--setenv=DATABASE_MAX_OVERFLOW=0",
        "--setenv=OMP_NUM_THREADS=1", "--setenv=OPENBLAS_NUM_THREADS=1",
        "/data/bb/app/.venv/bin/python", "scripts/audit_profitability_economics.py",
        "--until", args.until.isoformat(), "--hours", str(args.hours),
        "--limit", str(args.limit), "--bill-pages", str(args.bill_pages),
        "--output", remote_output,
    ]
    # Capture diagnostic logs on the host; transport only the aggregate JSON.
    runner = (
        "import pathlib,subprocess,sys; "
        "result=subprocess.run(sys.argv[2:],capture_output=True); "
        "assert result.returncode == 0, 'bounded economics audit failed; inspect its named unit'; "
        "print(pathlib.Path(sys.argv[1]).read_text())"
    )
    ssh = connect_remote_ssh(ROOT, timeout=25)
    try:
        output = run_remote_text(
            ssh, shlex.join(["/data/bb/app/.venv/bin/python", "-c", runner, remote_output, *command]),
            timeout=270, max_output_chars=60_000,
        )
    finally:
        ssh.close()
    report = json.loads(output)
    report["remote_report_path"] = remote_output
    report["remote_unit"] = command[5].removeprefix("--unit=")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--until", required=True, type=_utc)
    parser.add_argument("--hours", type=int, choices=range(1, 169), default=24)
    parser.add_argument("--limit", type=int, choices=range(1, 501), default=120)
    parser.add_argument("--bill-pages", type=int, choices=range(1, 21), default=8)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()
    if not datetime.now(UTC) - timedelta(days=6) <= args.until <= datetime.now(UTC):
        parser.error("cutoff must be within the past six days")
    if args.until - timedelta(hours=args.hours) < datetime.now(UTC) - timedelta(days=7):
        parser.error("window must fit the current-bills seven-day history")

    async def run() -> dict[str, Any]:
        try:
            return await asyncio.wait_for(
                audit(until=args.until, hours=args.hours, limit=args.limit, bill_pages=args.bill_pages),
                timeout=210,
            )
        finally:
            await close_db()

    report = audit_online(args) if args.online else asyncio.run(run())
    encoded = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
